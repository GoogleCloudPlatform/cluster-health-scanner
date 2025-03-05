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


def _get_partition_for_machine(machine_type: str) -> str | None:
  """Returns the partition for the given orchestrator."""
  match machine_type:
    case 'a3-highgpu-8g':
      return 'a3'
    case 'a3-megagpu-8g':
      return 'a3mega'
    case 'a3-ultragpu-8g':
      return 'a3ultra'
    case 'a4-highgpu-8g':
      return 'a4'
    case _:
      raise ValueError(f'Unsupported machine type: {machine_type}')


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
@click.pass_context
def cli(
    ctx: click.Context,
    machine_type: str,
    check: str,
    nodes: list[str],
    run_only_on_available_nodes: bool,
    dry_run: bool,
):
  """Run a healthscan on a cluster."""
  orchestrator = ctx.obj['orchestrator']
  check_runner = None
  partition = None
  if orchestrator == 'slurm':
    partition = _get_partition_for_machine(machine_type)
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
