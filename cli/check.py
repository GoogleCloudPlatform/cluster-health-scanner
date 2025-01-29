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

"""A base class for defining the interface of healthscan checks."""

import math
import signal
import subprocess
import sys
import time
from typing import Any
import uuid

import click
from kubernetes import client

import common
import launch_helm


class Check:
  """A standard implementation of a healthscan check."""

  name = None

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
      orchestrator: str,
      machine_type: str,
      nodes: list[str],
      results_labels: list[str] | None,
      launch_label: str | None,
      launch_label_value: str = 'true',
      timeout_sec: int = 15 * 60,
  ):
    self.name = name
    self.description = description
    self.orchestrator = orchestrator
    self.machine_type = machine_type
    self.results_labels = results_labels
    self.nodes = nodes
    self.launch_label = launch_label
    self.launch_label_value = launch_label_value
    self.timeout_sec = timeout_sec
    # Used to interface with the GKE cluster
    self._v1 = client.CoreV1Api()

    # Generate a unique base name for the HC Helm release
    guid = str(uuid.uuid4())[:8]
    ts = int(time.time())
    # Ex: chs-hc-gpu-cli-12345678-1723456789
    self.hc_release_name_base: str = f'chs-hc-{self.name}-cli-{guid}-{ts}'

    # Handle SIGINT signal to clean up
    signal.signal(
        signal.SIGINT,
        self.sigint_handler,
    )

  def _gke_set_up(self):
    """Set up for the check on a GKE cluster."""
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
        '--filter',
        release_name_base,
    ]
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

  def _gke_clean_up(self) -> None:
    """Clean up after the check on a GKE cluster."""
    # Attempt to clean up all HC Helm releases not already uninstalled

    helm_releases = self._get_helm_releases(self.hc_release_name_base)
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
        click.echo(
            f'Uninstall result: {uninstall_result.stdout.strip()}'
        )

    # Other processes to clean up like HR Helm release, labels, etc.
    launch_helm.cleanup_k8s_cluster(
        hc_type=self.name,
        launch_label=self.launch_label,
        nodes=self.nodes,
    )

    return

  def _gke_check(self, sleep_sec: int = 300) -> str | None:
    """Run the check on a GKE cluster."""
    n_nodes = len(self.nodes)
    additional_helm_env_vars: dict[str, str] = {
        f'health_checks.{self.name}_healthcheck.env.N_NODES': str(n_nodes),
    }
    pod_name = launch_helm.deploy_health_runner(
        hc_type=self.name,
        wait=math.floor(sleep_sec / 60),
        hc_release_name_base=self.hc_release_name_base,
        additional_helm_env_vars=additional_helm_env_vars,
    )
    return pod_name

  def set_up(self) -> None:
    """Set up for the check."""
    common.run_for_orchestrator(
        orchestrator=self.orchestrator,
        gke_function=self._gke_set_up,
    )

  def clean_up(self) -> None:
    """Clean up after the check."""
    common.run_for_orchestrator(
        orchestrator=self.orchestrator,
        gke_function=self._gke_clean_up,
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

    health_runner_pod_name = common.run_for_orchestrator(
        orchestrator=self.orchestrator,
        gke_function=lambda: self._gke_check(
            sleep_sec=timeout_sec,
        ),
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
    update_interval_sec = 10
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
        time.sleep(update_interval_sec)
      progress_bar.update(
          update_interval_sec,
          current_item=health_runner_pod_name,
      )
    return health_runner_pod_name
