# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Useful functions for running parts of Cluster Health Scanner.

This library is intended to be used by the Cluster Health Scanner CLI. It
emulates what a user would do when running CHS manually.
"""

import json
import subprocess


def _run_command(
    command: str,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
  """Execute a shell command using subprocess.

  Args:
    command: The shell command to be executed.
    check: If True, raises CalledProcessError if the command returns a non-zero
      exit status. Defaults to True.

  Returns:
    The result object containing information about the completed process.
  """
  diag = subprocess.run(
      command,
      shell=True,
      text=True,
      check=check,
      capture_output=True,
  )
  return diag


def _generate_helm_command(
    hc_type: str,
    chart_name: str,
    release_name: str,
    namespace: str | None = None,
    values_file: str | None = None,
    set_values: dict[str, str] | None = None,
) -> str:
  """Generates a Helm command for installing a chart.

  Args:
    hc_type: The type of health check to deploy.
    chart_name: The name of the Helm chart.
    release_name: The name of the Helm release.
    namespace: The namespace to deploy the chart to.
    values_file: The path to a YAML file containing values to override.
    set_values: A dictionary of values to override.

  Returns:
    A list of strings representing the Helm command.
  """

  helm_install_command: str = f'helm install {release_name} {chart_name} '

  if namespace:
    helm_install_command += f'-n {namespace} '
  if values_file:
    helm_install_command += f'-f {values_file} '
  if set_values:
    for k, v in set_values.items():
      helm_install_command += f'--set {k}={v} '
  # Turn off all health checks
  helm_install_command += '--set health_checks.gpu_healthcheck.run_check=false '
  helm_install_command += (
      '--set health_checks.nccl_healthcheck.run_check=false '
  )
  helm_install_command += (
      '--set health_checks.neper_healthcheck.run_check=false '
  )
  helm_install_command += (
      '--set health_checks.straggler_healthcheck.run_check=false '
  )
  helm_install_command += (
      '--set health_checks.tinymax_healthcheck.run_check=false '
  )
  helm_install_command += (
      f'--set health_checks.{hc_type}_healthcheck.run_check=true '
  )

  return helm_install_command


def deploy_health_runner(
    hr_release_name: str,
    hc_type: str,
    wait: int,
    values_file: str | None = None,
    hc_release_name_base: str | None = None,
    additional_helm_env_vars: dict[str, str] | None = None,
    dry_run: bool = False,
) -> str:
  """Deploy health runner.

  Args:
    hr_release_name: The name of the health runner release.
    hc_type: The type of health check to deploy.
    wait: The wait time in minutes to complete.
    values_file: The relative path to a YAML file containing values to override.
    hc_release_name_base: The unique ID to use for health check release names.
      If None, will default to the Health Runner's default.
    additional_helm_env_vars: A dictionary of additional Helm environment
      variables to set.
    dry_run: If True, the install command will be returned but not executed.

  Returns:
    The name of the health runner pod. If dry_run is True, this will be the
    command that would have been run.
  """
  specific_set_values: dict[str, str] = {
      f'health_checks.{hc_type}_healthcheck.env.TIMEOUT_MINUTES': str(wait),
  }
  if additional_helm_env_vars:
    specific_set_values.update(additional_helm_env_vars)
  # If hc_release_name_base is provided, set the HC helm release name to it
  if hc_release_name_base:
    specific_set_values[
        f'health_checks.{hc_type}_healthcheck.env.HELM_RELEASE_NAME_BASE'
    ] = hc_release_name_base
  helm_install_command = _generate_helm_command(
      hc_type=hc_type,
      chart_name='deploy/helm/health_runner',
      release_name=hr_release_name,
      namespace='default',
      values_file=values_file,
      set_values=specific_set_values,
  )

  if dry_run:
    return helm_install_command

  _ = _run_command(helm_install_command)

  helm_resources = json.loads(
      _run_command(
          f'helm status {hr_release_name} --show-resources -o json'
      ).stdout
  )
  hr_job_name = helm_resources['info']['resources']['v1/Job'][0]['metadata'][
      'name'
  ]
  hr_pod = json.loads(
      _run_command(f'kubectl get pod -l job-name={hr_job_name} -o json').stdout
  )
  hr_pod_name = hr_pod['items'][0]['metadata']['name']

  return hr_pod_name


def setup_k8s_cluster(
    launch_label: str,
    launch_label_value: str,
    results_labels: list[str],
    nodes: list[str] | None = None,
) -> None:
  """Set up cluster/nodes as necessary before setting up and running CHS.

  This can include removing labels from previous runs so setup can be done
  correctly.

  Args:
    launch_label: The label a node must have for the health check to run.
    launch_label_value: The value of the launch label.
    results_labels: All labels that the healthcheck writes to the node.
    nodes: The nodes to set up. If None, all nodes will be set up.
  """

  nodes_to_setup = ' '.join(nodes) if nodes else '--all'
  kubectl_label_nodes_base_command = 'kubectl label nodes'

  # Remove past launch labels on all nodes (only current nodes will run)
  remove_launch_labels_command = (
      f'{kubectl_label_nodes_base_command} --all {launch_label}- '
  )
  _run_command(remove_launch_labels_command.strip())

  # Remove past labels on all nodes (can block node affinity)
  remove_labels_command = (
      f'{kubectl_label_nodes_base_command} {nodes_to_setup} '
      + ' '.join(f'{label}-' for label in results_labels)
  )
  _run_command(remove_labels_command.strip())

  # Add labels only for the nodes to be tested
  add_labels_command = (
      f'{kubectl_label_nodes_base_command} {nodes_to_setup} '
      f'{launch_label}={launch_label_value} '
  )
  _run_command(add_labels_command.strip())


def cleanup_k8s_cluster(
    hr_release_name: str,
    launch_label: str,
    nodes: list[str] | None = None,
) -> None:
  """Uninstall helm chart for health check and remove labels from nodes.

  Args:
    hr_release_name: The name of the health runner release.
    launch_label: The label a node must have for the health check to run.
    nodes: The nodes to clean up. If None, all nodes will be cleaned up.
  """

  # Uninistall helm chart
  helm_uninstall_command = f'helm uninstall {hr_release_name}'
  _run_command(helm_uninstall_command)

  nodes_to_cleanup = ' '.join(nodes) if nodes else '--all'
  remove_labels_command = (
      f'kubectl label nodes {nodes_to_cleanup} {launch_label}-'
  )
  _run_command(remove_labels_command)
