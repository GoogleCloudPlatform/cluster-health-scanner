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

"""Module to fetch nodes from a GKE cluster."""

from kubernetes import client


def get_occupied_nodes(
    kubectl_core_api: client.CoreV1Api,
    nodes: list[str] | None = None,
) -> set[str]:
  """Gets all requested nodes that are currently occupied.

  Args:
    kubectl_core_api: The kubectl core api to use for the check.
    nodes: The nodes to check. If None, all nodes will be checked.

  Returns:
    A list of occupied nodes. Optionally, if nodes is provided, only the
    occupied nodes of those provided will be returned.
  """
  occupied_nodes = set()
  try:
    pods = kubectl_core_api.list_pod_for_all_namespaces(
        watch=False, field_selector='status.phase=Running'
    )
    for pod in pods.items:
      for container in pod.spec.containers:
        # Check if container has GPU requests.
        # container.resources.requests is a dict, so we can directly check
        # for the key.
        if (
            container.resources
            and container.resources.requests
            and 'nvidia.com/gpu' in container.resources.requests
        ):
          occupied_nodes.add(pod.spec.node_name)
  except client.rest.ApiException as e:
    raise e

  return (
      occupied_nodes
      if not nodes
      else set(nodes).intersection(occupied_nodes)
  )


def get_nodes_with_machine_type(
    kubectl_core_api: client.CoreV1Api,
    machine_type: str,
) -> list[str]:
  """Gets nodes with the given machine type.

  Args:
    kubectl_core_api: The kubectl core api to use for the check.
    machine_type: The machine type to check.

  Returns:
    A list of all node names with the given machine type.
  """
  return [
      node.metadata.name
      for node in kubectl_core_api.list_node(
          label_selector=(
              f'node.kubernetes.io/instance-type={machine_type}'
          )
      ).items
  ]


def has_machine_type_on_cluster(
    kubectl_core_api: client.CoreV1Api,
    machine_type: str,
) -> bool:
  """Returns if the cluster has nodes with the given machine type.

  Args:
    kubectl_core_api: The kubectl core api to use for the check.
    machine_type: The machine type to check.

  Returns:
    True if the cluster has nodes with the given machine type, False otherwise.
  """
  return bool(
      len(
          get_nodes_with_machine_type(
              kubectl_core_api=kubectl_core_api,
              machine_type=machine_type,
          )
      )
  )


def fetch_gke_nodes(
    kubectl_core_api: client.CoreV1Api,
    machine_type: str,
    nodes: list[str] | None = None,
    run_only_on_available_nodes: bool = False,
) -> tuple[list[str], list[str]]:
  """Gets all available nodes for the given machine type.

  Args:
    kubectl_core_api: The kubectl core api to use for the check.
    machine_type: The machine type to check.
    nodes: The nodes to check. If None, all nodes will be checked.
    run_only_on_available_nodes: If True, returns both available nodes and the
    occupied nodes. If False, and any of the requested nodes are occupied, the
    function will raise an error.

  Returns:
    A tuple containing:
      - A list of available nodes.
      - A list of occupied nodes.

  Raises:
    ValueError: If the cluster does not have the specified machine type or if
    nodes are occupied and run_only_on_available_nodes is False.
  """
  if not has_machine_type_on_cluster(
      kubectl_core_api=kubectl_core_api, machine_type=machine_type
  ):
    raise ValueError(
        f'Active cluster does not have machine type {machine_type}.'
    )
  all_nodes = nodes or get_nodes_with_machine_type(
      kubectl_core_api, machine_type
  )
  occupied_nodes = get_occupied_nodes(
      kubectl_core_api=kubectl_core_api, nodes=nodes
  )

  if occupied_nodes and not run_only_on_available_nodes:
    raise ValueError(
        f'The following nodes are occupied: {occupied_nodes}. Please free'
        ' up these nodes before running healthscan.\n'
        ' Alternatively, you can run again with'
        ' --run_only_on_available_nodes to skip these nodes.'
    )
  elif run_only_on_available_nodes:
    available_nodes = [node for node in all_nodes if node not in occupied_nodes]
    return available_nodes, list(occupied_nodes)
  return all_nodes, list(occupied_nodes)
