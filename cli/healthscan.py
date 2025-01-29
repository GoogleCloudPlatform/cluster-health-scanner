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
from kubernetes import client
from kubernetes import config

import common
import gpu_check
import nccl_check
import neper_check
import status
import straggler_check


_SUPPORTED_MACHINE_TYPES = common.SUPPORTED_MACHINE_TYPES
_SUPPORTED_HEALTHCHECKS = [
    status.Status.name,
    nccl_check.NcclCheck.name,
    gpu_check.GpuCheck.name,
    straggler_check.StragglerCheck.name,
    neper_check.NeperCheck.name,
]


def _get_occupied_gke_nodes() -> set[str]:
  """Returns all nodes with containers requesting GPU resources."""
  config.load_kube_config()
  v1 = client.CoreV1Api()

  invalid_nodes = set()
  try:
    pods = v1.list_pod_for_all_namespaces(
        watch=False, field_selector='status.phase=Running'
    )
    for pod in pods.items:
      for container in pod.spec.containers:
        if container.resources.requests:
          requested_gpus = set(
              resource_name
              for resource_name, _ in container.resources.requests.items()
              if resource_name == 'nvidia.com/gpu'
          )
          if requested_gpus:
            invalid_nodes.add(pod.spec.node_name)
  except client.rest.ApiException as e:
    click.echo(
        click.style(
            f'Failed to list nodes in cluster: {e}', fg='red', bold=True
        )
    )

  return invalid_nodes


def _find_occupied_nodes_on_cluster(
    orchestrator: str, nodes: list[str]
) -> set[str]:
  """Finds any occupied nodes on the cluster.

  Args:
    orchestrator: The orchestrator type.
    nodes: The nodes to check. If None, all nodes will be checked.

  Returns:
    A list of occupied nodes. Optionally, if nodes is provided, only the
    occupied nodes of those provided will be returned.
  """
  occupied_nodes = common.run_for_orchestrator(
      orchestrator=orchestrator,
      gke_function=_get_occupied_gke_nodes,
  )
  return (
      occupied_nodes if not nodes else set(nodes).intersection(occupied_nodes)
  )


def _validate_gke_cluster_has_machine_type(machine_type: str) -> bool:
  """Returns all nodes with the given machine type."""
  config.load_kube_config()
  v1 = client.CoreV1Api()
  return bool(len(
      v1.list_node(
          label_selector=f'node.kubernetes.io/instance-type={machine_type}'
      ).items
  ))


def _validate_cluster_has_machine_type(
    orchestrator: str, machine_type: str
) -> bool:
  """Validates that the cluster has the given machine type."""
  return common.run_for_orchestrator(
      orchestrator=orchestrator,
      gke_function=lambda: _validate_gke_cluster_has_machine_type(machine_type),
  )


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
""",
)
@click.option(
    '-n',
    '--nodes',
    multiple=True,
    default=[],
    help='Nodes to run checks on. Defaults to running on all nodes.',
)
@click.option(
    '--run_only_on_available_nodes',
    default=False,
    is_flag=True,
    help="""
    Force running the healthcheck only on available nodes.
    Unavailable nodes will be skipped.""",
)
@click.pass_context
def cli(
    ctx: click.Context,
    machine_type: str,
    check: str,
    nodes: list[str],
    run_only_on_available_nodes: bool,
):
  """Run a healthscan on a cluster."""
  orchestrator = ctx.obj['orchestrator']
  if check == status.Status.name:
    click.echo(status.Status(orchestrator, machine_type, nodes).run())
  else:
    if not _validate_cluster_has_machine_type(orchestrator, machine_type):
      click.echo(
          click.style(
              f'Active cluster does not have machine type {machine_type}.',
              fg='red',
              bold=True,
          )
      )
      raise click.Abort()

    occupied_nodes = _find_occupied_nodes_on_cluster(
        orchestrator=orchestrator, nodes=nodes
    )

    if occupied_nodes and not run_only_on_available_nodes:
      click.echo(
          click.style(
              f'The following nodes are occupied: {occupied_nodes}. Please free'
              ' up these nodes before running healthscan.\n'
              ' Alternatively, you can run again with'
              ' --run_only_on_available_nodes to skip these nodes.',
              fg='red',
              bold=True,
          )
      )
      raise click.Abort()
    elif run_only_on_available_nodes:
      click.echo(
          click.style(
              'WARNING: Running only on available nodes is not recommended.\n'
              'The following nodes are occupied and will be skipped: '
              f'{occupied_nodes}',
              fg='red',
              bold=True,
          )
      )

    check_runner = None
    match check:
      case nccl_check.NcclCheck.name:
        check_runner = nccl_check.NcclCheck(orchestrator, machine_type, nodes)
      case gpu_check.GpuCheck.name:
        check_runner = gpu_check.GpuCheck(orchestrator, machine_type, nodes)
      case straggler_check.StragglerCheck.name:
        check_runner = straggler_check.StragglerCheck(
            orchestrator, machine_type, nodes
        )
      case neper_check.NeperCheck.name:
        check_runner = neper_check.NeperCheck(orchestrator, machine_type, nodes)

    if check_runner:
      check_runner.set_up()
      check_runner.run()
      check_runner.clean_up()
