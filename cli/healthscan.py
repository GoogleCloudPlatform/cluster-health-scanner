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
from kubernetes import client
from kubernetes import config

import common
import gke_node_fetcher
import gpu_check
import label_telemetry
import nccl_check
import neper_check
import slurm_node_fetcher
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
    multiple=True,
    type=click.Choice(_SUPPORTED_HEALTHCHECKS),
    default=[_SUPPORTED_HEALTHCHECKS[0]],
    help="""
    Checks to run. Available checks:

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
@click.option(
    '--disable-usage-analytics',
    default=False,
    is_flag=True,
    help="""
    Disable telemetry tracking.""",
)
@click.option(
    '--gcs-bucket-name',
    'gcs_bucket_name',
    type=str,
    default=None,
    help='GCS bucket name for uploading GPU health check bug reports.',
)


@click.pass_context
def cli(
    ctx: click.Context,
    machine_type: str,
    check: tuple[str, ...],
    nodes: list[str],
    run_only_on_available_nodes: bool,
    dry_run: bool,
    disable_usage_analytics: bool,
    partition: str,
    gcs_bucket_name: str,
):
  """Run a healthscan on a cluster."""
  orchestrator = ctx.obj['orchestrator']
  check_runners = []
  nodes_list = nodes
  kubectl_core_api = None
  if orchestrator == 'slurm':
    if partition is None:
      raise click.MissingParameter(
          'Partition is required for Slurm clusters. Please specify a partition'
          ' using the --partition flag.'
      )
    nodes_list = slurm_node_fetcher.expand_slurm_nodes(nodes)
  else:
    if not is_helm_installed():
      print('Helm is not installed. Please install Helm before running '
            'healthscan: https://helm.sh/docs/intro/install/')
      return
    if orchestrator == 'gke' and not dry_run:
      config.load_kube_config()
      kubectl_core_api = client.CoreV1Api()
      try:
        nodes_list, occupied_nodes = gke_node_fetcher.fetch_gke_nodes(
            kubectl_core_api=kubectl_core_api,
            machine_type=machine_type,
            nodes=nodes,
            run_only_on_available_nodes=run_only_on_available_nodes,
        )
        if occupied_nodes and not run_only_on_available_nodes:
          click.echo(
              click.style(
                  'WARNING: Running only on available nodes is not'
                  ' recommended.\n The following nodes are occupied and will be'
                  f' skipped: {occupied_nodes}',
                  fg='red',
                  bold=True,
              )
          )
      except ValueError as e:
        click.echo(click.style(f'{e}', fg='red', bold=True))
        raise click.Abort()
      except client.rest.ApiException as e:
        click.echo(
            click.style(
                f'Failed to list nodes in cluster: {e}',
                fg='red',
                bold=True,
            )
        )
        raise click.Abort()
  for check_name in check:
    match check_name:
      case nccl_check.NAME:
        check_runners.append(
            nccl_check.get_check_for_orchestrator(
                orchestrator=orchestrator,
                machine_type=machine_type,
                partition=partition,
                nodes=nodes_list,
                run_only_on_available_nodes=run_only_on_available_nodes,
                dry_run=dry_run,
            )
        )
      case gpu_check.NAME:
        check_runners.append(
            gpu_check.get_check_for_orchestrator(
                orchestrator=orchestrator,
                machine_type=machine_type,
                partition=partition,
                nodes=nodes_list,
                run_only_on_available_nodes=run_only_on_available_nodes,
                dry_run=dry_run,
                gcs_bucket_name=gcs_bucket_name
            )
        )
      case straggler_check.NAME:
        check_runners.append(
            straggler_check.get_check_for_orchestrator(
                orchestrator=orchestrator,
                machine_type=machine_type,
                nodes=nodes_list,
                run_only_on_available_nodes=run_only_on_available_nodes,
                dry_run=dry_run,
            )
        )
      case neper_check.NAME:
        check_runners.append(
            neper_check.get_check_for_orchestrator(
                orchestrator=orchestrator,
                machine_type=machine_type,
                nodes=nodes_list,
                run_only_on_available_nodes=run_only_on_available_nodes,
                dry_run=dry_run,
            )
        )
      case tinymax_check.NAME:
        check_runners.append(
            tinymax_check.get_check_for_orchestrator(
                orchestrator=orchestrator,
                machine_type=machine_type,
                nodes=nodes_list,
                run_only_on_available_nodes=run_only_on_available_nodes,
                dry_run=dry_run,
            )
        )
      case status.NAME:
        check_runners.append(
            status.get_check_for_orchestrator(
                orchestrator=orchestrator,
                machine_type=machine_type,
                nodes=nodes_list,
            )
        )

  labels_by_check_name = {}

  for check_runner in check_runners:
    check_runner.set_up()
    check_runner.run()
    check_runner.clean_up()
    if orchestrator == 'gke' and check_runner.test_result_label:
      labels_by_check_name[check_runner.name] = check_runner.test_result_label

  if labels_by_check_name and not disable_usage_analytics:
    # TODO: b/431233627 - update telemetry message
    click.echo(
        click.style(
            'Reporting healthscan telemetry to Google. To disable this, rerun'
            ' with the --disable-usage-analytics flag.',
            fg='yellow',
            bold=True,
        )
    )
    label_telemetry.add_telemetry_labels(
        kubectl_core_api=kubectl_core_api,
        machine_type=machine_type,
        nodes=nodes_list,
        check_name_to_result_label=labels_by_check_name,
        dry_run=dry_run,
    )
