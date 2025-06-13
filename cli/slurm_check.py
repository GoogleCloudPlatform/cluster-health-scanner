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

"""A Slurm implementation of the healthscan check interface."""

import subprocess

import click

import check


class SlurmCheck(check.Check):
  """A check to provide the current healthscan result status of a cluster."""

  def __init__(
      self,
      name: str,
      description: str,
      machine_type: str,
      check_flag: str,
      partition: str,
      nodes: list[str],
      supported_machine_types: frozenset[str],
      dry_run: bool = False,
  ):
    """Initializes a check to run on a Slurm cluster.

    Args:
      name: The name of the check.
      description: The description of the check.
      machine_type: The machine type of the cluster to run the check on.
      check_flag: The flag to pass to the cluster-validation.sh script.
      partition: The partition to run the check on.
      nodes: The nodes to run the check on.
      supported_machine_types: The machine types supported by the check.
      dry_run: Whether to run the check in dry run mode.
    """
    super().__init__(
        name=name,
        description=description,
        machine_type=machine_type,
        supported_machine_types=supported_machine_types,
        dry_run=dry_run,
    )
    self.check_flag = check_flag
    self.partition = partition
    self.nodes = _expand_slurm_nodes(nodes)

  def _get_slurm_run_command(self) -> list[str]:
    """Builds the command to run the slurm check."""
    relative_path = 'deploy/slurm'
    command = [
        f'{relative_path}/cluster-validation.sh',
        '--nodelist',
        ','.join(self.nodes),
        f'--{self.check_flag}',
        '--partition',
        self.partition,
        '--machine-type',
        self.machine_type,
        '--nodes',
        f'{len(self.nodes)}',
        f'--relative-exec-path={relative_path}',
        '--results-dir=results',
    ]
    return command

  def set_up(self) -> None:
    """Slurm set_up is not yet supported."""
    # No setup is needed for the status check.

  def clean_up(self) -> None:
    """Slurm clean_up is not yet supported."""
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
    click.echo(f'Performing {self.name} check...')
    command = self._get_slurm_run_command()
    if self.dry_run:
      click.echo(
          click.style(
              f'Running {self.name} check in dry run mode...',
              fg='red',
              bold=True,
          )
      )
      dry_run_command = ' '.join(command)
      click.echo(f'Skipping running command: {dry_run_command}')
      return None
    result = subprocess.run(
        command, text=True, check=False, capture_output=True
    ).stdout
    click.echo(result)
    return result


def _expand_slurm_nodes(nodes: list[str]) -> list[str]:
  """Expands a list of slurm nodes into a list of nodes."""
  nodelist = []
  for node in nodes:
    nodelist.extend(_expand_slurm_node_pattern(node))
  return nodelist


def _expand_slurm_node_pattern(node_pattern: str) -> list[str]:
  """Expands a slurm node pattern into a list of nodes."""
  slurm_nodelist_expansion_cmd = [
      'scontrol',
      'show',
      'hostname',
      node_pattern,
  ]
  output = subprocess.run(
      slurm_nodelist_expansion_cmd,
      text=True,
      check=True,
      capture_output=True,
  ).stdout.strip()
  nodes = output.split('\n')
  return nodes
