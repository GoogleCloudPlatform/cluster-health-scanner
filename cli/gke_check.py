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

"""A GKE implementation of the healthscan check interface."""

import math
import subprocess
import sys
import time
from typing import Any

import click
from kubernetes import client
from kubernetes import config

import check
import gke_node_fetcher
import launch_helm


class GkeCheck(check.Check):
  """A standard implementation of a healthscan check."""

  def sigint_handler(
      self,
      signum: Any,
      frame: Any,
  ) -> None:
    """Handler for SIGINT signal.

    Args:
      signum: The signal number.
      frame: The current stack frame.
    """
    print(f'Received {signum} signal on frame {frame}. Exiting...')
    # Perform any necessary cleanup actions here
    # For example: close file handlers, release resources, etc.
    click.echo(
        click.style(
            '\nCLEANING UP...',
            fg='red',
            bold=True,
        )
    )
    self.clean_up()
    # Stops proceeding anything the CLI is doing
    sys.exit(0)

  def __init__(
      self,
      name: str,
      description: str,
      machine_type: str,
      supported_machine_types: frozenset[str],
      nodes: list[str],
      results_labels: list[str] | None,
      launch_label: str | None,
      launch_label_value: str = 'true',
      run_only_on_available_nodes: bool = False,
      kubectl_core_api: client.CoreV1Api | None = None,
      timeout_sec: int = 15 * 60,
      dry_run: bool = False,
      container_name: str | None = None,
      test_result_label: str | None = None,
  ):
    """Initialize a check to run on a GKE cluster.

    Note: This class's instance takes ownership of the `nodes` list.
    Args:
      name: The name of the check.
      description: The description of the check.
      machine_type: The machine type of the cluster to run the check on.
      supported_machine_types: The machine types supported by the check.
      nodes: The nodes to run the check on.
      results_labels: The labels to use for the results.
      launch_label: The label to use for the launch.
      launch_label_value: The value to use for the launch label.
      run_only_on_available_nodes: Whether to run the check only on available
        nodes.
      kubectl_core_api: The kubectl core api to use for the check.
      timeout_sec: The timeout in seconds for the check.
      dry_run: Whether to run the check in dry run mode.
      container_name: The name of the main container used by the check.
      test_result_label: The label to use for the test results.
    """
    super().__init__(
        name=name,
        description=description,
        machine_type=machine_type,
        supported_machine_types=supported_machine_types,
        dry_run=dry_run,
    )
    self.results_labels = results_labels
    self.nodes = nodes
    self.launch_label = launch_label
    self.launch_label_value = launch_label_value
    self.run_only_on_available_nodes = run_only_on_available_nodes
    self.timeout_sec = timeout_sec
    self.check_logs = None
    self.check_container_name = container_name
    self.test_result_label = test_result_label

    self.hr_release_name: str = f'chs-hr-{self.name}-cli'
    # Generate a unique base name for the HC Helm release
    # Default to HC release w/ no special base name
    # Possible example: chs-hc-gpu-cli-12345678-1723456789
    self.hc_release_name_base: str = f'chs-hc-{self.name}-cli'

    if not dry_run:
      # Used to interface with the GKE cluster
      if kubectl_core_api:
        self._v1 = kubectl_core_api
      else:
        config.load_kube_config()
        self._v1 = client.CoreV1Api()

  def set_up(self):
    """Set up for the check on a GKE cluster."""
    if self.dry_run:
      click.echo(
          click.style(
              'Dry run mode enabled. Skipping set_up.',
              fg='red',
              bold=True,
          )
      )
      return
    try:
      self.nodes, occupied_nodes = gke_node_fetcher.fetch_gke_nodes(
          kubectl_core_api=self._v1,
          machine_type=self.machine_type,
          nodes=self.nodes,
          run_only_on_available_nodes=self.run_only_on_available_nodes,
      )
      if occupied_nodes:
        click.echo(
            click.style(
                'WARNING: Running only on available nodes is not recommended.\n'
                'The following nodes are occupied and will be skipped: '
                f'{occupied_nodes}',
                fg='red',
                bold=True,
            )
        )
    except ValueError as e:
      raise click.Abort(str(e))
    except client.rest.ApiException as e:
      raise click.Abort(f'Failed to list nodes in cluster: {e}')
    launch_helm.setup_k8s_cluster(
        launch_label=self.launch_label,
        launch_label_value=self.launch_label_value,
        results_labels=self.results_labels,
        nodes=self.nodes,
    )

  def _get_helm_releases(
      self,
      release_name_base: str | None,
  ) -> list[str]:
    """Get all Helm releases with the given release name base.

    Args:
      release_name_base: The base name of the Helm release to filter by.

    Returns:
      A list of Helm releases with the given release name base.
    """
    # Note this will use the default helm ls limit of 256
    helm_ls_command = [
        'helm',
        'ls',
        '-a',
        '--no-headers',
    ]
    if release_name_base:
      helm_ls_command.extend([
          '--filter',
          release_name_base,
      ])
    else:
      # If no release name base is provided, assume no helm releases to return
      return []
    try:
      helm_ls_output = subprocess.run(
          helm_ls_command,
          text=True,
          check=True,
          capture_output=True,
      )
      # Keep only the release name (at the beginning of each line)
      helm_releases = [
          release_name.split('\t')[0].strip()  # Separated w/ tabs
          for release_name in helm_ls_output.stdout.strip().split('\n')
          if release_name  # Catch the case where the release name is empty
      ]
    # Can happen when a non-zero exit code is returned
    except subprocess.CalledProcessError as e:
      click.echo(
          click.style(
              text=f'Failed to get Helm releases:\n{e}',
              fg='red',
              bold=True,
          ),
      )
      helm_releases = []
    # Catch if helm is not installed
    except FileNotFoundError as e:
      click.echo(
          click.style(
              text=(
                  'Failed to get Helm releases (`helm` likely not installed):\n'
                  f'{e}'
              ),
              fg='red',
              bold=True,
          ),
      )
      helm_releases = []

    return helm_releases

  def clean_up(self) -> None:
    """Clean up after the check on a GKE cluster."""
    if self.dry_run:
      click.echo(
          click.style(
              'Dry run mode enabled. Skipping clean_up.',
              fg='red',
              bold=True,
          )
      )
      return
    # Attempt to clean up all HC Helm releases not already uninstalled
    helm_releases = self._get_helm_releases(
        release_name_base=(
            getattr(self, 'hc_release_name_base', None)
        ),
    )
    # Iterate over each release and uninstall it
    for release_name in helm_releases:
      helm_uninstall_command = [
          'helm',
          'uninstall',
          release_name,
      ]
      click.echo(f'Uninstalling "{release_name}"')
      uninstall_result = subprocess.run(
          helm_uninstall_command,
          text=True,
          check=False,
          capture_output=True,
      )
      # Check if the overall operation was successful
      if uninstall_result.returncode == 0:
        click.echo(f'Release "{release_name}" uninstalled successfully.')
      else:
        click.echo(f'Release "{release_name}" failed to uninstall.')
        click.echo(f'Uninstall result: {uninstall_result.stdout.strip()}')

    # Other processes to clean up like HR Helm release, labels, etc.
    if getattr(self, 'hc_release_name_base', None):
      launch_helm.cleanup_k8s_cluster(
          hr_release_name=self.hr_release_name,
          launch_label=self.launch_label,
          nodes=self.nodes,
      )

    return

  def _get_values_file(self) -> str:
    """Get the values file for the check."""
    base_path = 'deploy/helm/health_runner/'
    match self.machine_type:
      case 'a3-highgpu-8g':
        return base_path + 'a3high.yaml'
      case 'a3-megagpu-8g':
        # Use the default values for A3 Mega
        return base_path + 'values.yaml'
      case 'a3-ultragpu-8g':
        return base_path + 'a3ultra.yaml'
      case 'a4-highgpu-8g':
        return base_path + 'a4.yaml'
      case 'a4x-highgpu-4g':
        return base_path + 'a4x.yaml'
      case _:
        raise ValueError(f'Unsupported machine type: {self.machine_type}')

  def _get_helm_env_vars(self):
    # Only set N_NODES if nodes are specified (otherwise uses all nodes)
    additional_helm_env_vars: dict[str, str] | None = None
    # If nodes are not specified, then the health runner will use all nodes
    if self.nodes:
      n_nodes = len(self.nodes)
      additional_helm_env_vars: dict[str, str] = {
          f'health_checks.{self.name}_healthcheck.env.N_NODES': str(n_nodes),
      }
    return additional_helm_env_vars

  def _check(self, sleep_sec: int = 300, dry_run: bool = False) -> str:
    """Run the check on a GKE cluster."""
    return launch_helm.deploy_health_runner(
        hr_release_name=self.hr_release_name,
        hc_type=self.name,
        wait=math.floor(sleep_sec / 60),
        values_file=self._get_values_file(),
        hc_release_name_base=self.hc_release_name_base,
        additional_helm_env_vars=self._get_helm_env_vars(),
        dry_run=dry_run,
    )

  def _get_pod_phase(
      self,
      pod_name: str,
      namespace: str = 'default',
  ) -> str:
    """Get the phase of the pod."""

    pod_phase = self._v1.read_namespaced_pod(
        name=pod_name,
        namespace=namespace,
    ).status.phase
    return pod_phase

  def _progress_bar_item_show(
      self,
      pod_name: str | None,
  ) -> str | None:
    """Get the progress bar item to show."""
    if pod_name is None:
      return None
    else:
      pod_status = (
          f'Health Runner Status: {self._get_pod_phase(pod_name=pod_name)}'
      )
      return pod_status

  def get_check_pod(self) -> str | None:
    """Get the name of the canonical pod for the check."""
    return None

  def update_check_logs(self, pod_name: str | None = None):
    """Updates the check logs for the given pod name.

    Args:
      pod_name: The name of the pod to get the logs for.
    """
    if pod_name is None:
      self.check_logs = f'Fetching logs is not defined for check: {self.name}'
      return
    try:
      self.check_logs = self._v1.read_namespaced_pod_log(
          name=pod_name,
          namespace='default',
          container=self.check_container_name,
      )
    except client.rest.ApiException:
      # This error is expected if the pod is not found, and should fail quietly.
      pass

  def run(
      self,
      timeout_sec: int | None = None,
      startup_sec: int = 30,
  ) -> str | None:
    """Run the check.

    Args:
      timeout_sec: The timeout in seconds for the check.
      startup_sec: The time in seconds to wait for the health runner to start.

    Returns:
      The name of the health runner pod.
    """
    click.echo(f'Performing {self.name} check...')

    if not timeout_sec:
      timeout_sec = self.timeout_sec

    if self.dry_run:
      click.echo(
          click.style(
              f'Running {self.name} check in dry run mode...',
              fg='red',
              bold=True,
          )
      )
      dry_run_command = self._check(sleep_sec=timeout_sec, dry_run=self.dry_run)
      click.echo(f'Skipping running command: {dry_run_command}')
      return

    health_runner_pod_name = self._check(
        sleep_sec=timeout_sec,
    )

    start_time = time.time()
    # CLI has extra startup time to allow health runner to complete & clean up
    click.echo('Waiting for Health Runner to start...')
    update_hr_startup_interval_sec = 10
    running_statuses = (
        'Running',
        'Succeeded',
    )
    while self._get_pod_phase(health_runner_pod_name) not in running_statuses:
      # Health Runner has some time to start up but if not started by startup
      # time, then give a warning
      if time.time() - start_time >= startup_sec:
        click.echo(
            click.style(
                text=(
                    f'Health Runner not started after {startup_sec} seconds.\n'
                    'Health Runner may not cleanly exit even if it starts now.'
                ),
                fg='red',
                bold=True,
            )
        )
        break
      else:
        time.sleep(update_hr_startup_interval_sec)

    # Resets the time for progress bar since given HR startup time above
    start_time = time.time()
    update_interval_sec = 5
    check_pod = self.get_check_pod()
    with click.progressbar(
        label=f'{self.name} Health Runner',
        length=timeout_sec,
        item_show_func=self._progress_bar_item_show,
    ) as progress_bar:
      while (
          self._get_pod_phase(health_runner_pod_name)
          not in ['Succeeded', 'Failed', 'Unknown']
          and time.time() - start_time < timeout_sec
      ):
        progress_bar.update(
            n_steps=update_interval_sec,
            current_item=health_runner_pod_name,
        )
        if not check_pod:
          check_pod = self.get_check_pod()
        self.update_check_logs(pod_name=check_pod)
        time.sleep(update_interval_sec)
      progress_bar.update(
          update_interval_sec,
          current_item=health_runner_pod_name,
      )
    return health_runner_pod_name
