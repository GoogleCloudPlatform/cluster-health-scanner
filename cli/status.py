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

"""Status check for the healthscan command.

This check provides the current healthscan result status of a cluster.
"""

import subprocess

import click

import common
import check


class Status(check.Check):
  """A check to provide the current healthscan result status of a cluster."""

  _description = (
      "A check to provide the current healthscan result status of a cluster."
  )

  _custom_cols = (
      "NODE:.metadata.name,"
      "NEPER_RESULT:.metadata.labels.aiinfra/neper-healthcheck-result,"
      "GPU_RESULT:.metadata.labels.aiinfra/gpu-healthcheck-result,"
      "NCCL_RESULT:.metadata.labels.aiinfra/nccl-healthcheck-result"
  )

  name = "status"

  def __init__(self, orchestrator: str, machine_type: str, nodes: list[str]):
    super().__init__(
        name=self.name,
        description=self._description,
        orchestrator=orchestrator,
        machine_type=machine_type,
        launch_label=None,
        results_labels=None,
        nodes=nodes,
    )

  def _gke_status(self):
    """Get the current healthscan status of a GKE cluster."""
    command = (
        f"kubectl get nodes -o custom-columns={self._custom_cols} "
        f"-l node.kubernetes.io/instance-type={self.machine_type}"
    )
    return subprocess.run(
        command,
        shell=True,
        text=True,
        check=False,
        capture_output=True,
    )

  def set_up(self) -> None:
    """Set up for the status check."""
    # No setup is needed for the status check.

  def clean_up(self) -> None:
    """Clean up after the status check."""
    # No cleanup is needed for the status check.

  def run(
      self,
      timeout_sec: int | None = None,
      startup_sec: int | None = None,
  ) -> str | None:
    """Run the status check.

    Args:
      timeout_sec: The timeout in seconds for the check.
      startup_sec: The time in seconds to wait for the health runner to start.

    Returns:
      The status of the cluster as a string.
    """
    click.echo("Performing status check...")
    return common.run_for_orchestrator(
        orchestrator=self.orchestrator,
        gke_function=self._gke_status,
    ).stdout
