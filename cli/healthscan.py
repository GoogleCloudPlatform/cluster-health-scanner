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

"""CLI for running healthscan on a cluster.

This is part of the larger cluster_diag CLI. To get the full helpstring, run
`cluster_diag healthscan --help`.
"""

import subprocess
import click

import common
import gpu_check
import nccl_check
import neper_check
import status
import straggler_check
import tinymax_check


_SUPPORTED_MACHINE_TYPES = list(common.SUPPORTED_MACHINE_TYPES)
_SUPPORTED_HEALTHCHECKS = [
    status.NAME,
    nccl_check.NAME,
    gpu_check.NAME,
    straggler_check.NAME,
    neper_check.NAME,
    tinymax_check.NAME,
]


def is_helm_installed() -> bool:
  """Checks if Helm is installed and available in the system's PATH."""
  try:
    subprocess.run(
        ['helm', 'version'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    return True
  except (subprocess.CalledProcessError, FileNotFoundError):
    # If the command fails (e.g., helm not found or returns an error),
    # or if the helm executable is not found, it means Helm is not installed.
    return False


@click.command(name='healthscan')
@click.argument(
    'machine_type',
    type=click.Choice(_SUPPORTED_MACHINE_TYPES, case_sensitive=False),
)
@click.option(
    '-c',
    '--check',
    type=click.Choice(_SUPPORTED_HEALTHCHECKS),
    default=_SUPPORTED_HEALTHCHECKS[0],
    help="""
    Check to run. Available checks:

    \b
    - status: (Default) Checks the current healthscan status of the cluster.
    - nccl: Runs a pairwise NCCL bandwidth test on the cluster.
    - gpu: Runs a GPU check on the cluster.
    - straggler: Instruments a straggler check on the cluster.
    - neper: Runs a Network Performand eval on the cluster.
    - tinymax: Runs a ml-framework TinyMax test on the cluster.
""",
)
@click.option(
    '-n',
    '--nodes',
    multiple=True,
    default=[],
    help=(
        'Nodes to run checks on. Defaults to running on all nodes. When using'
        ' slurm, a shortened node format can be used. For example, "node-[0-1]"'
    ),
)
@click.option(
    '--run_only_on_available_nodes',
    default=False,
    is_flag=True,
    help="""
    Force running the healthcheck only on available nodes.
    Unavailable nodes will be skipped.""",
)
@click.option(
    '--dry_run',
    default=False,
    is_flag=True,
    help="""
    Run the healthcheck in dry run mode.
    This will print the commands that would be run, but not run them.""",
)
@click.option(
    '--partition',
    default=None,
    help="""
    Partition to run the healthcheck on.
    This is only used for Slurm clusters.""",
)
@click.pass_context
def cli(
    ctx: click.Context,
    machine_type: str,
    check: str,
    nodes: list[str],
    run_only_on_available_nodes: bool,
    dry_run: bool,
    partition: str,
):
  """Run a healthscan on a cluster."""
  orchestrator = ctx.obj['orchestrator']
  check_runner = None
  if orchestrator == 'slurm':
    if partition is None:
      raise click.MissingParameter(
          'Partition is required for Slurm clusters. Please specify a partition'
          ' using the --partition flag.'
      )

  else:
    if not is_helm_installed():
      print('Helm is not installed. Please install Helm before running '
            'healthscan: https://helm.sh/docs/intro/install/')
      return
  match check:
    case nccl_check.NAME:
      check_runner = nccl_check.get_check_for_orchestrator(
          orchestrator=orchestrator,
          machine_type=machine_type,
          partition=partition,
          nodes=nodes,
          run_only_on_available_nodes=run_only_on_available_nodes,
          dry_run=dry_run,
      )
    case gpu_check.NAME:
      check_runner = gpu_check.get_check_for_orchestrator(
          orchestrator=orchestrator,
          machine_type=machine_type,
          partition=partition,
          nodes=nodes,
          run_only_on_available_nodes=run_only_on_available_nodes,
          dry_run=dry_run,
      )
    case straggler_check.NAME:
      check_runner = straggler_check.get_check_for_orchestrator(
          orchestrator=orchestrator,
          machine_type=machine_type,
          nodes=nodes,
          run_only_on_available_nodes=run_only_on_available_nodes,
          dry_run=dry_run,
      )
    case neper_check.NAME:
      check_runner = neper_check.get_check_for_orchestrator(
          orchestrator=orchestrator,
          machine_type=machine_type,
          nodes=nodes,
          run_only_on_available_nodes=run_only_on_available_nodes,
          dry_run=dry_run,
      )
    case tinymax_check.NAME:
      check_runner = tinymax_check.get_check_for_orchestrator(
          orchestrator=orchestrator,
          machine_type=machine_type,
          nodes=nodes,
          run_only_on_available_nodes=run_only_on_available_nodes,
          dry_run=dry_run,
      )
    case status.NAME:
      check_runner = status.get_check_for_orchestrator(
          orchestrator=orchestrator,
          machine_type=machine_type,
          nodes=nodes,
      )

  if check_runner:
    check_runner.set_up()
    check_runner.run()
    check_runner.clean_up()
