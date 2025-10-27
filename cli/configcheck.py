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

"""CLI for running configcheck on a cluster.

This is part of the larger cluster_diag CLI. To get the full helpstring, run
`cluster_diag configcheck --help`.
"""

import asyncio
import subprocess

import click
from kubernetes import client
from kubernetes import config
import pandas as pd

import common
import config as configcheck_config
import config_differ
import dependencies
import dependency_version_parser
import golden_config_parser
import node_config_fetcher


_SUPPORTED_MACHINE_TYPES = list(common.SUPPORTED_MACHINE_TYPES)
_SUPPORTED_MACHINE_TYPES.remove('a4-highgpu-8g')
_SUPPORTED_MACHINE_TYPES.remove('a4x-highgpu-4g')

GKE_DEPENDENCIES_BY_MACHINE = {
    'a3-ultragpu-8g': dependencies.GKE_DEPENDENCY_PARSERS,
    'a3-megagpu-8g': dependencies.GKE_DEPENDENCY_PARSERS,
    'a3-highgpu-8g': dependencies.GKE_DEPENDENCY_PARSERS,
}

SLURM_DEPENDENCIES_BY_MACHINE = {
    'a3-ultragpu-8g': dependencies.SLURM_DEPENDENCY_PARSERS,
    'a3-megagpu-8g': dependencies.SLURM_DEPENDENCY_PARSERS,
    'a3-highgpu-8g': dependencies.SLURM_DEPENDENCY_PARSERS,
}

_FORMAT_MARKDOWN = 'markdown'
_FORMAT_JSON = 'json'


def _get_dependency_parsers(
    dependency_map: dict[
        str, list[dependency_version_parser.DependencyVersionParser]
    ],
    machine_type: str,
) -> list[dependency_version_parser.DependencyVersionParser]:
  """Returns the dependency parsers for a given machine type and map."""
  if machine_type not in dependency_map:
    raise click.Abort(
        f'Unsupported machine type: {machine_type}. Supported machine'
        f' types: {list(dependency_map.keys())}'
    )
  return dependency_map[machine_type]


def _get_k8s_nodes(machine_type: str) -> list[str]:
  """Returns all nodes with containers requesting GPU resources."""
  config.load_kube_config()
  v1 = client.CoreV1Api()

  return [
      node.metadata.name
      for node in v1.list_node(
          label_selector=f'beta.kubernetes.io/instance-type={machine_type}'
      ).items
  ]


def _get_slurm_nodes(
    partition: str | None = None,
    features: str | None = None,
    state: str | None = None,
) -> list[str]:
  """Returns all nodes from a slurm cluster, filtering by criteria.

  Args:
      partition: The partition to filter by.
      features: The features to filter by.
      state: The state to filter by.

  Returns:
      A list of node names.
  """
  cmd = ['sinfo', '-o', '%N']
  if partition:
    cmd.extend(['-p', partition])
  if features:
    cmd.extend(['-F', features])
  if state:
    cmd.extend(['-T', state])

  all_nodes = []
  try:
    result = subprocess.run(
        cmd, capture_output=True, text=True, check=True
    )
    nodes = result.stdout.strip().split('\n')
    all_nodes.extend(nodes)
  except FileNotFoundError:
    click.echo(
        click.style(
            'Error: `sinfo` command not found. Please run this command from a'
            ' Slurm login node or provide a list of nodes using the `--nodes`'
            ' flag.',
            fg='red',
            bold=True,
        )
    )
    return []
  except subprocess.CalledProcessError as e:
    raise click.Abort(f'Failed to get slurm nodes: {e}')
  return all_nodes


def _get_zone_from_slurm_node(slurm_node: str) -> str:
  """Returns the zone of a given slurm node.

  Args:
      slurm_node: The slurm node to get the zone.

  Returns:
      The zone of the slurm node.
  """
  cmd = [
      'gcloud',
      'compute',
      'instances',
      'describe',
      slurm_node,
      '--format=value(zone.basename())',
  ]
  try:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout.strip()
  except subprocess.CalledProcessError as e:
    raise click.Abort(f'Failed to get zone for {slurm_node}: {e}')


def _get_config_matrix(
    data: list[configcheck_config.NodeConfig],
) -> pd.DataFrame:
  """Returns a DataFrame representing the config data."""
  columns = ['Node Name']
  for attribute_name in sorted(data[0].dependencies.keys()):
    columns.append(attribute_name)
  data_rows = [node.to_csv() for node in data]
  return pd.DataFrame(
      data=data_rows,
      columns=columns,
  )


def _get_diff_matrix(data: list[configcheck_config.NodeDiff]) -> pd.DataFrame:
  """Returns a DataFrame representing the config data."""
  columns = ['Node Name']
  for attribute_name in sorted(data[0].dependency_diffs.keys()):
    columns.append(attribute_name)
  data_rows = [node.to_csv() for node in data]
  return pd.DataFrame(
      data=data_rows,
      columns=columns,
  )


def _get_dynamic_dependency_parsers_for_orchestrator(
    orchestrator: str,
    node_name: str,
    zone: str,
    pod_name: str | None = None,
    workload_container: str | None = None,
) -> list[dependency_version_parser.DependencyVersionParser]:
  """Returns the dynamic dependency parsers for a given orchestrator."""

  dynamic_dependency_parsers = common.run_for_orchestrator(
      orchestrator=orchestrator,
      gke_function=(
          lambda node=node_name, pod_name=pod_name, workload_container=workload_container: dependencies.get_dynamic_dependency_parsers(
              node, zone, pod_name, workload_container=workload_container
          )
      ),
      slurm_function=lambda node=node_name: dependencies.get_slurm_dynamic_dependency_parsers(
          node, zone
      ),
  )
  return dynamic_dependency_parsers


def _get_workload_containers_on_node(node_name: str) -> dict[str, str]:
  """Returns containers actively running on a node which request GPUs."""
  config.load_kube_config()
  v1 = client.CoreV1Api()

  pods = v1.list_pod_for_all_namespaces(
      field_selector=f'spec.nodeName={node_name}'
  ).items

  pod_to_container_names = dict()
  for pod in pods:
    pod_name = pod.metadata.name
    for container_status in pod.status.container_statuses:
      if container_status.state.running:
        # Get the container spec to access the resource requests
        for container_spec in pod.spec.containers:
          if (
              container_spec.name == container_status.name
              and container_spec.resources.requests
              and int(
                  container_spec.resources.requests.get('nvidia.com/gpu', 0)
              )
              > 0
          ):
            pod_to_container_names[pod_name] = container_spec.name
            break

  return pod_to_container_names


def _get_workload_container_on_node(node_name: str) -> tuple[str, str] | None:
  """Returns the workload container actively running on a node which requests GPUs."""
  pod_to_container_names = _get_workload_containers_on_node(node_name)
  if not pod_to_container_names:
    return None
  elif len(pod_to_container_names.items()) > 1:
    raise click.Abort(
        f'Multiple workload containers found on node {node_name}:'
        f' {pod_to_container_names.values()}. Please limit to running a single'
        ' workload container per node.'
    )
  pod_name, workload_container = pod_to_container_names.popitem()
  return (pod_name, workload_container)


def _get_zone_from_k8s_topology_label(node_name: str) -> str:
  """Returns the zone of the node from the K8s topology label."""
  config.load_kube_config()
  v1 = client.CoreV1Api()
  node = v1.read_node(node_name)
  return node.metadata.labels['topology.kubernetes.io/zone']


def _fetch_node_configs(
    project: str,
    node_list: list[str],
    static_dependency_parsers: list[
        dependency_version_parser.DependencyVersionParser
    ],
    zone: str | None,
    workload_container: str | None,
    orchestrator: str,
    sudo: bool = False,
    verbose: bool = False,
) -> list[configcheck_config.NodeConfig]:
  """Fetches the current configuration settings for a given node.

  Args:
    project: The project of the workload to be checked.
    node_list: The list of nodes to fetch configs for.
    static_dependency_parsers: The dependency parsers to use for the static
      dependencies.
    zone: The zone of the workload to be checked.
    workload_container: The name of the workload container to fetch NCCL configs
      from. If not specified, NCCL configs will not be fetched.
    orchestrator: The orchestrator used to run the workload.
    sudo: Whether to run remote commands with sudo. Defaults to False.
    verbose: Whether to enable verbose logging. Defaults to False.

  Returns:
    A list of NodeConfig objects, one for each node.
  """
  with click.progressbar(
      label='Fetching node configs',
      length=len(node_list),
  ) as progress_bar:
    node_data = []
    for node in node_list:
      pod_name = None
      if orchestrator == 'gke':
        if workload_container is None:
          pod_name_to_container_name = _get_workload_container_on_node(node)
          if pod_name_to_container_name:
            pod_name = pod_name_to_container_name[0]
            workload_container = pod_name_to_container_name[1]
        if zone is None:
          zone = _get_zone_from_k8s_topology_label(node)
      elif orchestrator == 'slurm':
        if zone is None:
          zone = _get_zone_from_slurm_node(node)
      dynamic_dependency_parsers = (
          _get_dynamic_dependency_parsers_for_orchestrator(
              orchestrator,
              node,
              zone,
              pod_name=pod_name,
              workload_container=workload_container,
          )
      )
      fetcher = node_config_fetcher.NodeConfigFetcher(
          name=node,
          zone=zone,
          project=project,
          dependency_parsers=static_dependency_parsers
          + dynamic_dependency_parsers,
          sudo=sudo,
          verbose=verbose,
      )
      node_data.append(fetcher.fetch_config())
      progress_bar.update(1)
    return node_data


def _get_gcloud_config_value(key: str) -> str:
  """Returns the value of a gcloud config key."""
  result = subprocess.run(
      ['gcloud', 'config', 'get-value', key],
      check=True,
      capture_output=True,
      text=True,
  )

  if result.returncode != 0 or result.stdout in (None, '(unset)'):
    error_msg = (
        f'Failed to get gcloud config value for {key}. Please run `gcloud'
        f' config set {key} <value>` and try again.'
    )
    click.echo(
        click.style(
            error_msg,
            fg='red',
            bold=True,
        )
    )
    raise click.Abort(error_msg)
  value = result.stdout.strip()
  click.echo(
      click.style(
          f'{key} not set. Inferring {key} from `gcloud config get'
          f' {key}`: {value}',
          fg='yellow',
          bold=True,
      )
  )
  return value


async def _fetch_node_configs_async(
    project: str,
    node_list: list[str],
    static_dependency_parsers: list[
        dependency_version_parser.DependencyVersionParser
    ],
    zone: str | None,
    workload_container: str | None,
    orchestrator: str,
    sudo: bool = False,
    verbose: bool = False,
) -> list[configcheck_config.NodeConfig]:
  """Asynchronously fetches the current configuration settings for a given node.

  Args:
    project: The project of the workload to be checked.
    node_list: The list of nodes to fetch configs for.
    static_dependency_parsers: The dependency parsers to use for the static
      dependencies.
    zone: The zone of the workload to be checked.
    workload_container: The name of the workload container to fetch NCCL configs
      from. If not specified, NCCL configs will not be fetched.
    orchestrator: The orchestrator used to run the workload.
    sudo: Whether to run remote commands with sudo. Defaults to False.
    verbose: Whether to enable verbose logging. Defaults to False.

  Returns:
    A list of NodeConfig objects, one for each node.
  """
  tasks = []
  with click.progressbar(
      label='[Async] Fetching node configs',
      length=len(node_list),
  ) as progress_bar:
    for node in node_list:
      pod_name = None
      if orchestrator == 'gke':
        if workload_container is None:
          pod_name_to_container_name = _get_workload_container_on_node(node)
          if pod_name_to_container_name:
            pod_name = pod_name_to_container_name[0]
            workload_container = pod_name_to_container_name[1]
      dynamic_dependency_parsers = (
          _get_dynamic_dependency_parsers_for_orchestrator(
              orchestrator,
              node,
              zone,
              pod_name=pod_name,
              workload_container=workload_container,
          )
      )
      fetcher = node_config_fetcher.NodeConfigFetcher(
          name=node,
          zone=zone,
          project=project,
          dependency_parsers=static_dependency_parsers
          + dynamic_dependency_parsers,
          sudo=sudo,
          verbose=verbose,
      )
      task = asyncio.create_task(fetcher.fetch_config_async())
      task.add_done_callback(lambda _: progress_bar.update(1))
      tasks.append(task)
  configs = list(await asyncio.gather(*tasks))
  # Reset the terminal to the original state.
  # 
  subprocess.run(['reset'], check=True)
  return configs


@click.command(name='configcheck')
@click.argument(
    'machine_type',
    type=click.Choice(_SUPPORTED_MACHINE_TYPES, case_sensitive=False),
)
@click.option(
    '-n',
    '--nodes',
    default='',
    help=(
        'A comma-separated list of nodes to run checks on. Defaults to running'
        ' on all nodes.'
    ),
)
@click.option(
    '--skip_diff',
    '--nodiff',
    default=False,
    is_flag=True,
    help=(
        'If true, only print the node configs without diffing against the'
        ' golden config.'
    ),
)
@click.option(
    '--run_async',
    '--async',
    default=False,
    is_flag=True,
    help=(
        '[Experimental] If true, run the configcheck in async mode. This will'
        ' reset your terminal as part of the process.'
    ),
)
@click.option(
    '--project',
    default=None,
    help=(
        'The project of the workload to be checked. If not specified, the'
        ' project will be inferred from `gcloud config get project`'
    ),
)
@click.option(
    '--zone',
    default=None,
    help=(
        'The zone of the workload to be checked. If not specified, the zone'
        ' will be inferred per node from the `topology.kubernetes.io/zone`'
        ' label.'
    ),
)
@click.option(
    '--workload_container',
    default=None,
    help=(
        'The name of the workload container to fetch workload configs from. If'
        ' not specified, the workload container will be inferred from the node.'
    ),
)
@click.option(
    '--output_format',
    default=_FORMAT_MARKDOWN,
    type=click.Choice([_FORMAT_MARKDOWN, _FORMAT_JSON]),
    help=(
        'The format to print the output in. Defaults to markdown. Other'
        ' supported formats are `csv` and `json`.'
    ),
)
@click.option(
    '--sudo',
    default=False,
    is_flag=True,
    help=(
        'Run remote commands with sudo. Note: This is sometimes necessary to '
        'fetch configs on nodes/pods/containers with restricted permissions.'
    ),
)
@click.option(
    '--verbose',
    '-v',
    default=False,
    is_flag=True,
    help='Enable verbose logging.',
)
@click.pass_context
def cli(
    ctx: click.Context,
    machine_type: str,
    nodes: str,
    skip_diff: bool,
    run_async: bool,
    project: str | None,
    zone: str | None,
    workload_container: str | None,
    output_format: str,
    sudo: bool,
    verbose: bool,
):
  """Run a configcheck on a cluster."""
  orchestrator = ctx.obj['orchestrator']
  if project is None:
    project = _get_gcloud_config_value('project')
  if nodes:
    node_list = [n.strip() for n in nodes.split(',') if n.strip()]
  else:
    node_list = common.run_for_orchestrator(
        orchestrator=orchestrator,
        gke_function=lambda: _get_k8s_nodes(machine_type),
        slurm_function=_get_slurm_nodes,
    )
  if orchestrator == 'slurm':
    dependency_parsers = _get_dependency_parsers(
        SLURM_DEPENDENCIES_BY_MACHINE, machine_type
    )
  else:
    dependency_parsers = _get_dependency_parsers(
        GKE_DEPENDENCIES_BY_MACHINE, machine_type
    )
  if run_async:
    click.echo(
        click.style(
            'WARNING: Running configcheck in async mode is experimental. Your'
            ' terminal may be reset as part of this process.',
            fg='red',
            bold=True,
        )
    )
    node_data = asyncio.run(
        _fetch_node_configs_async(
            project=project,
            zone=zone,
            node_list=node_list,
            static_dependency_parsers=dependency_parsers,
            workload_container=workload_container,
            orchestrator=orchestrator,
            sudo=sudo,
            verbose=verbose,
        )
    )
  else:
    node_data = _fetch_node_configs(
        project=project,
        zone=zone,
        node_list=node_list,
        static_dependency_parsers=dependency_parsers,
        workload_container=workload_container,
        orchestrator=orchestrator,
        sudo=sudo,
        verbose=verbose,
    )
  if not node_data:
    click.echo(
        click.style(
            'No nodes found to check. Please check your configuration or'
            ' provide a list of nodes using the `--nodes` flag.',
            fg='yellow',
            bold=True,
        )
    )
    return
  if skip_diff:
    df = _get_config_matrix(node_data)
  else:
    golden_config = golden_config_parser.get_golden_configs(
        dependency_parsers=dependency_parsers
        + _get_dynamic_dependency_parsers_for_orchestrator(
            orchestrator,
            node_name='golden',
            zone=zone,
            workload_container='golden',
        ),
        machine_type=machine_type,
    )[0]
    node_diffs = [
        config_differ.diff_configs(
            experiment=node,
            golden=golden_config,
        )
        for node in node_data
    ]
    df = _get_diff_matrix(node_diffs)
  output_data = None
  if output_format == _FORMAT_MARKDOWN:
    output_data = df.to_markdown(tablefmt='pipe')
  elif output_format == _FORMAT_JSON:
    output_data = df.to_json()
  click.echo(output_data)
