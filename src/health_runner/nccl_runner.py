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

"""Runs NCCL health check."""

import itertools
import logging
import os
import random
import time
import uuid

import kubernetes.client
from kubernetes.client.api import batch_v1_api

import checker_common
import common_pb2


_SLEEP_TIME_MINUTES = os.environ.get("SLEEP_TIME_MINUTES", "20")

_KUBECTL = os.environ.get("KUBECTL_PATH", "/app/kubectl")
_HELM = os.environ.get("HELM_PATH", "/usr/local/bin/helm")


def run_nccl_healthcheck():
  """Runs NCCL health check and waits for it to complete."""
  print("cleaning up labels")
  cleanup_labels()

  # TODO - Only select subset of nodes
  kubernetes.config.load_incluster_config()
  v1 = kubernetes.client.CoreV1Api()

  case = os.environ.get("PAIRING_MODE", "random").lower()
  second_pass_enabled = is_second_pass_enabled()
  # Default is to use toplogy awareness when pairing nodes
  require_topology = True
  if case == "intra_rack":
    health_check_func = run_intra_rack_healthcheck
  elif case == "inter_rack":
    health_check_func = run_inter_rack_healthcheck
  elif case == "inter_cluster":
    health_check_func = run_inter_cluster_healthcheck
  elif case == "random":
    require_topology = False
    health_check_func = run_nccl_random_pair_healthcheck
  else:
    print(f"Unknown health check case: {case}")
    case = "random"
    require_topology = False
    health_check_func = run_nccl_random_pair_healthcheck

  node_data = get_nodes_data(
      kube_nodes=v1.list_node().items,
      require_topology=require_topology,
  )
  logging.info("Parsing over %d nodes", len(node_data))
  capacity = get_capacity_topology(node_data)

  logging.info("Running %s w/ pairing mode `%s`", "NCCL", case)
  health_check_func(v1, capacity, second_pass_enabled)


def health_check_with_node_pairs(
    node_pairs: list[tuple[str, str]],
    additional_helm_install_flags: list[str] | None = None,
    job_name_distinctor: str = "unknown-type",
) -> list[str]:
  """Runs NCCL health check with a list of node pairs.

  Args:
    node_pairs: A list of node pairs to run the health check on.
    additional_helm_install_flags: A list of additional flags to pass to helm
      install.
    job_name_distinctor: A string to distinguish the type of job.

  Returns:
    A list of the nodes that were tested.
  """
  # Create a list of job names for to monitor the jobs.
  job_names = []
  tested_nodes = []

  cleanup_functions = []

  # This must be defined in the YAML configuration
  helm_chart_path = os.environ.get("HELM_CHART")
  helm_chart_version = os.environ.get("HELM_CHART_VERSION")
  helm_values = os.environ.get("HELM_VALUES")

  # For the first pass, pair each node
  for node0, node1 in node_pairs:
    # Reset the Helm install flags for each node pair
    helm_install_flags = os.environ.get("HELM_INSTALL_FLAGS")

    short_guid = str(uuid.uuid4())[:8]
    unique_release_name = os.environ.get(
        "HELM_RELEASE_NAME",
        f"internal-health-check-{job_name_distinctor}-{short_guid}",
    )
    unique_job_name = f"hc-{job_name_distinctor}-{short_guid}"
    # Sets nodes and second pass flag for helm chart
    helm_install_flags += (
        f" --set job.name={unique_job_name}"
        f" --set health_check.env.NODE0={node0}"
        f" --set health_check.env.NODE1={node1}"
    )
    if additional_helm_install_flags is not None:
      for flag in additional_helm_install_flags:
        helm_install_flags += f" --set {flag}"

    print(f"Running NCCL test between node {node0} and node {node1}...")
    cleanup_functions.extend(
        checker_common.create_helm_release(
            helm_path=_HELM,
            release_name=unique_release_name,
            chart=helm_chart_path,
            values=helm_values,
            chart_version=helm_chart_version,
            helm_install_flags=helm_install_flags,
        )
    )
    # TODO - Find a better way to avoid this sleep
    # Sleep to allow time for helm releases to create proper ServiceAccount,
    # ClusterRole, etc. Otherwise, errors will occur where resources like the
    # ServiceAccount won't exist to create a SSH connection.
    time.sleep(1)

    job_names.append(unique_job_name)
    tested_nodes.append(node0)
    tested_nodes.append(node1)

  print(f"Waiting for {len(job_names)} jobs to complete...")
  job_api = batch_v1_api.BatchV1Api()
  checker_common.wait_till_jobs_complete(
      job_api,
      job_names,
      timeout_seconds=(int(_SLEEP_TIME_MINUTES) * 60),
      check_interval=20,
  )

  # Cleanup after pods are done (uninstall releases, delete k8s objects, etc.)
  print("All pods are done with first pass")
  for func in cleanup_functions:
    try:
      func()
    except Exception as error:  # pylint: disable=broad-exception-caught
      logging.exception("Cleanup failed (reason: %r).", error)

  return tested_nodes


def run_nccl_random_pair_healthcheck(
    v1: kubernetes.client.CoreV1Api,
    capacity: common_pb2.Capacity,
    second_pass_enabled: bool,
) -> tuple[list[str], list[str]]:
  """Runs NCCL health checks between random nodes and waits for it to complete.

  Args:
    v1: Kubernetes CoreV1Api object.
    capacity: Capacity topology of the cluster.
    second_pass_enabled: Whether second pass is enabled.

  Returns:
    A tuple containing the list of passed nodes and the list of failed nodes.
  """

  nodes = []
  logging.info("Finding all nodes...")
  for cluster in capacity.clusters:
    for rack in cluster.racks:
      for node in rack.nodes:
        nodes.append(node.id)
  logging.info("Found %d nodes", len(nodes))

  # For the first pass, pair each node
  node_pairs = []
  logging.info("Creating node pairs...")
  for i, j in generate_index_pairs(len(nodes)):
    node0 = nodes[i]
    node1 = nodes[j]
    node_pair = (node0, node1)
    node_pairs.append(node_pair)
    logging.info("Paired node %s and node %s", node0, node1)

  tested_nodes = health_check_with_node_pairs(
      node_pairs=node_pairs,
      additional_helm_install_flags=["health_check.env.SECOND_PASS=false"],
      job_name_distinctor="nccl-random-pair",
  )

  # Get the failed and passed nodes from the first pass.
  passed_nodes, failed_nodes = get_nccl_test_results(v1, tested_nodes)

  # If no second pass, no failed nodes, or no passed nodes for second pass,
  # then return results.
  if (
      (not second_pass_enabled)
      or (not failed_nodes)
      or (not passed_nodes)
  ):
    logging.info("Second pass will not run")
    print(f"Found failed nodes: {failed_nodes}")
    print(f"Found passed nodes: {passed_nodes}")
    return list(passed_nodes), list(failed_nodes)

  print(f"Running second pass for {len(failed_nodes)} nodes...")

  second_pass_node_pairs = []
  # For second pass, pair each failed node with a randoomly selected passed node
  # If there are more failed nodes than passed nodes, passed nodes will be
  # cycled through and therefore pair with more than one failed node.
  passed_nodes_list = list(passed_nodes)
  random.shuffle(passed_nodes_list)
  for failed_node, good_node in zip(
      list(failed_nodes), itertools.cycle(passed_nodes_list)
  ):
    node_pair = (failed_node, good_node)
    second_pass_node_pairs.append(node_pair)
    print(
        f"Will run NCCL test between good node {good_node} and failed node"
        f" {failed_node}"
    )

  tested_nodes = health_check_with_node_pairs(
      node_pairs=second_pass_node_pairs,
      additional_helm_install_flags=["health_check.env.SECOND_PASS=true"],
      job_name_distinctor="nccl-2nd-pass",
  )

  # Get the results of the second pass and combine with the first pass results.
  second_passed_nodes, second_failed_nodes = get_nccl_test_results(
      v1=v1,
      nodes=tested_nodes,
  )
  # 
  passed_nodes, failed_nodes = determine_failed_components(
      list(passed_nodes),
      list(failed_nodes),
      list(second_passed_nodes),
      list(second_failed_nodes),
  )

  print(f"found failed nodes: {failed_nodes}")
  print(f"found passed nodes: {passed_nodes}")

  return passed_nodes, failed_nodes


def run_intra_rack_healthcheck(
    v1: kubernetes.client.CoreV1Api,
    capacity: common_pb2.Capacity,
    second_pass_enabled: bool,
) -> tuple[list[str], list[str]]:
  """Checks the racks communication by running nccl tests between each node in the same rack."""
  # Create a dictionary of rack to nodes for easier lookup.
  rack_to_nodes = {}
  logging.info("Finding all racks...")
  for cluster in capacity.clusters:
    for rack in cluster.racks:
      rack_to_nodes[rack.id] = [node.id for node in rack.nodes]
  logging.info("Found %d racks", len(rack_to_nodes))

  # Contains all node pairs created by rack
  node_pairs = []

  # For the first pass, pair each node in the rack
  for rack, nodes_in_rack in rack_to_nodes.items():
    logging.info("%d nodes in rack %s", len(nodes_in_rack), rack)
    if len(nodes_in_rack) < 2:
      print(f"Skipping rack {rack} with less than 2 nodes.")
      continue
    # Add the specific node pairs to the overall list
    for i, j in generate_index_pairs(len(nodes_in_rack)):
      node0 = nodes_in_rack[i]
      node1 = nodes_in_rack[j]
      node_pair = (node0, node1)
      node_pairs.append(node_pair)
      print(
          f"Will run NCCL test between good node {node0} (Rack:"
          f" {rack}) and failed node {node1} (Rack:"
          f" {rack})"
      )

  tested_nodes = health_check_with_node_pairs(
      node_pairs=node_pairs,
      additional_helm_install_flags=["health_check.env.SECOND_PASS=false"],
      job_name_distinctor="nccl-intra-rack",
  )

  # Get the failed and passed nodes from the first pass.
  passed_nodes, failed_nodes = get_nccl_test_results(v1, tested_nodes)

  # If no second pass, no failed nodes, or no passed nodes for second pass,
  # then return results.
  if (
      (not second_pass_enabled)
      or (not failed_nodes)
      or (not passed_nodes)
  ):
    logging.info("Second pass will not run")
    print(f"Found failed nodes: {failed_nodes}")
    print(f"Found passed nodes: {passed_nodes}")
    return list(passed_nodes), list(failed_nodes)

  # For the second pass, pair each failed node with a passed node in the same
  # rack.
  print(f"Running second pass for {len(failed_nodes)} nodes...")
  second_pass_node_pairs = []
  for rack, nodes in rack_to_nodes.items():
    passed_nodes_in_rack = []
    failed_nodes_in_rack = []
    # Determine which nodes in the rack passed and which failed.
    for node in nodes:
      if node in passed_nodes:
        passed_nodes_in_rack.append(node)
      elif node in failed_nodes:
        failed_nodes_in_rack.append(node)

    if not passed_nodes_in_rack:
      print(f"No passed nodes in rack: {rack}")
      continue

    for failed_node in failed_nodes_in_rack:
      # Choose a random healthy node from the same rack
      healthy_node = random.choice(passed_nodes_in_rack)
      node_pair = (failed_node, healthy_node)
      second_pass_node_pairs.append(node_pair)
      print(
          f"Will run NCCL test between good node {healthy_node} (Rack:"
          f" {rack}) and failed node {failed_node} (Rack:"
          f" {rack})"
      )

  tested_nodes = health_check_with_node_pairs(
      node_pairs=second_pass_node_pairs,
      additional_helm_install_flags=["health_check.env.SECOND_PASS=true"],
      job_name_distinctor="nccl-intra-rack-second-pass",
  )

  # Get the results of the second pass and combine with the first pass results.
  second_passed_nodes, second_failed_nodes = get_nccl_test_results(
      v1, tested_nodes
  )
  passed_nodes, failed_nodes = determine_failed_components(
      list(passed_nodes),
      list(failed_nodes),
      list(second_passed_nodes),
      list(second_failed_nodes),
  )

  print(f"found failed nodes: {failed_nodes}")
  print(f"found passed nodes: {passed_nodes}")
  return passed_nodes, failed_nodes


def run_inter_rack_healthcheck(
    v1: kubernetes.client.CoreV1Api,
    capacity: common_pb2.Capacity,
    second_pass_enabled: bool,
) -> tuple[list[str], list[str]]:
  """Checks inter-rack communication by running nccl tests between nodes in different racks (The racks will be in the same cluster)."""
  cluster_to_racks = {}
  rack_to_nodes = {}

  node_pairs = []

  # Create a dictionary of cluster to racks and a dictionary of rack to nodes
  # to make it easier to look up info.
  logging.info("Finding all racks...")
  for cluster in capacity.clusters:
    cluster_to_racks[cluster.id] = []
    for rack in cluster.racks:
      cluster_to_racks[cluster.id].append(rack.id)
      rack_to_nodes[rack.id] = [node.id for node in rack.nodes]
  logging.info("Found %d racks", len(rack_to_nodes))

  for _, racks in cluster_to_racks.items():
    # Pair each rack with another rack in the same cluster
    rack_pairs = generate_index_pairs(len(racks))
    for i, j in rack_pairs:
      rack0 = racks[i]
      rack1 = racks[j]
      rack0_nodes = rack_to_nodes[rack0]
      logging.info("%d nodes in rack %s", len(rack0_nodes), rack0)
      rack1_nodes = rack_to_nodes[rack1]
      logging.info("%d nodes in rack %s", len(rack1_nodes), rack1)
      if not rack0_nodes or not rack1_nodes:
        raise ValueError(f"Rack: {racks[i]} or {racks[j]} has no nodes.")
      # Randomly select a node from each rack
      node0 = random.choice(rack0_nodes)
      node1 = random.choice(rack1_nodes)
      node_pair = (node0, node1)
      node_pairs.append(node_pair)
      print(
          f"Will run NCCL test between node {node0} (Rack:"
          f" {rack0}) and node {node1} (Rack:"
          f" {rack1})"
      )

  tested_nodes = health_check_with_node_pairs(
      node_pairs=node_pairs,
      additional_helm_install_flags=["health_check.env.SECOND_PASS=false"],
      job_name_distinctor="nccl-inter-rack",
  )

  # Check for failures and attempt second pass
  passed_nodes, _ = get_nccl_test_results(v1, tested_nodes)

  passed_racks = []
  failed_racks = []

  # Determine which racks passed and which failed by checking if any of the
  # nodes in the rack passed.
  for rack, nodes in rack_to_nodes.items():
    if any(node in passed_nodes for node in nodes):
      passed_racks.append(rack)
    else:
      failed_racks.append(rack)

  if (
      (not second_pass_enabled)
      or (not failed_racks)
      or (not passed_nodes)
  ):
    logging.info("Second pass will not run")
    print(f"Found failed racks: {failed_racks}")
    print(f"Found passed racks: {passed_racks}")
    return passed_racks, failed_racks

  # If second pass is enabled, we will run the test between the passed and
  # failed racks in the same cluster.
  print(f"Running second pass for {len(failed_racks)} racks...")
  second_pass_node_pairs = []

  for cluster in capacity.clusters:
    for failed_rack in cluster.racks:
      if failed_rack.id not in failed_racks:
        continue

      # Get the list of racks in the same cluster as the failed rack.
      potential_racks = cluster_to_racks[cluster.id]

      # Choose a random healthy rack from the same cluster
      healthy_rack = ""
      for potential_rack in potential_racks:
        if potential_rack in passed_racks:
          healthy_rack = potential_rack
          break

      if not healthy_rack:
        print(f"No healthy rack found for rack: {failed_rack.id}")
        continue

      # Choose a random node from the healthy and failed rack
      failed_nodes = rack_to_nodes[failed_rack.id]
      healthy_nodes = rack_to_nodes[healthy_rack]

      healthy_node = random.choice(healthy_nodes)
      failed_node = random.choice(failed_nodes)
      node_pair = (healthy_node, failed_node)
      second_pass_node_pairs.append(node_pair)

      print(
          f"Will run NCCL test between good node {healthy_node} (Rack:"
          f" {healthy_rack}) and failed node {failed_node} (Rack:"
          f" {failed_rack})"
      )

  tested_nodes = health_check_with_node_pairs(
      node_pairs=second_pass_node_pairs,
      additional_helm_install_flags=["health_check.env.SECOND_PASS=true"],
      job_name_distinctor="nccl-inter-rack-second-pass",
  )

  # Get the results of the second pass and combine with the first pass results
  passed_nodes, _ = get_nccl_test_results(v1, tested_nodes)

  second_passed_racks = []
  second_failed_racks = []

  # Determine which racks passed and which failed by checking if any of the
  # nodes in the rack passed.
  for rack, nodes in rack_to_nodes.items():
    if any(node in passed_nodes for node in nodes):
      second_passed_racks.append(rack)
    else:
      second_failed_racks.append(rack)

  # Combine the results of the first and second pass.
  passed_racks, failed_racks = determine_failed_components(
      passed_racks, failed_racks, second_passed_racks, second_failed_racks
  )

  print(f"After second pass - Failed racks: {failed_racks}")
  print(f"After second pass - Passed racks: {passed_racks}")
  return passed_racks, failed_racks


def run_inter_cluster_healthcheck(
    v1: kubernetes.client.CoreV1Api,
    capacity: common_pb2.Capacity,
    second_pass_enabled: bool,
) -> tuple[list[str], list[str]]:
  """Checks the inter-cluster communication by running nccl tests between nodes in different clusters."""

  cluster_to_nodes = {}
  for cluster in capacity.clusters:
    for rack in cluster.racks:
      if cluster.id not in cluster_to_nodes:
        cluster_to_nodes[cluster.id] = [node.id for node in rack.nodes]
      else:
        cluster_to_nodes[cluster.id].extend([node.id for node in rack.nodes])

  node_pairs = []

  # Pair each cluster with another cluster
  cluster_pairs = generate_index_pairs(len(capacity.clusters))
  for i, j in cluster_pairs:
    cluster0_nodes = cluster_to_nodes[capacity.clusters[i].id]
    cluster1_nodes = cluster_to_nodes[capacity.clusters[j].id]
    if not cluster0_nodes or not cluster1_nodes:
      raise ValueError(
          f"Cluster {capacity.clusters[i].id} or {capacity.clusters[j].id} has"
          " no racks."
      )

    # Randomly select a node from each cluster
    node0 = random.choice(cluster0_nodes)
    node1 = random.choice(cluster1_nodes)
    node_pair = (node0, node1)
    node_pairs.append(node_pair)

    print(
        f"Will run NCCL test between node {node0} (Cluster:"
        f" {capacity.clusters[i].id}) and node"
        f" {node1} (Cluster: {capacity.clusters[j].id})"
    )

  tested_nodes = health_check_with_node_pairs(
      node_pairs=node_pairs,
      additional_helm_install_flags=["health_check.env.SECOND_PASS=false"],
      job_name_distinctor="nccl-inter-cluster",
  )

  # Check for failures and attempt second pass
  passed_nodes, _ = get_nccl_test_results(v1, tested_nodes)

  passed_clusters = []
  failed_clusters = []
  for cluster in capacity.clusters:
    if any(node in passed_nodes for node in cluster_to_nodes[cluster.id]):
      passed_clusters.append(cluster.id)
    else:
      failed_clusters.append(cluster.id)

  if not passed_clusters:
    print("Found no passed clusters.")
    return passed_clusters, failed_clusters

  if not second_pass_enabled or not failed_clusters:
    logging.info("Second pass will not run")
    print(f"Found failed clusters: {failed_clusters}")
    print(f"Found passed clusters: {passed_clusters}")
    return passed_clusters, failed_clusters

  print(f"Running second pass for {len(failed_clusters)} clusters...")
  second_pass_node_pairs = []

  # Loop through the failed clusters and pair it with a random healthy cluster
  for failed_cluster in failed_clusters:
    healthy_cluster = random.choice(passed_clusters)

    failed_nodes = cluster_to_nodes[failed_cluster]
    healthy_nodes = cluster_to_nodes[healthy_cluster]

    # Choose a random healthy node from a healthy cluster
    failed_node = random.choice(failed_nodes)
    healthy_node = random.choice(healthy_nodes)
    node_pair = (failed_node, healthy_node)
    second_pass_node_pairs.append(node_pair)

    print(
        f"Will run NCCL test between good node {healthy_node} (Cluster:"
        f" {healthy_cluster}) and failed node {failed_node} (Cluster:"
        f" {failed_cluster})"
    )

  tested_nodes = health_check_with_node_pairs(
      node_pairs=second_pass_node_pairs,
      additional_helm_install_flags=["health_check.env.SECOND_PASS=true"],
      job_name_distinctor="nccl-inter-cluster-second-pass",
  )

  passed_nodes, _ = get_nccl_test_results(v1, tested_nodes)

  second_passed_clusters = []
  second_failed_clusters = []
  for cluster in capacity.clusters:
    if any(node in passed_nodes for node in cluster_to_nodes[cluster.id]):
      second_passed_clusters.append(cluster.id)
    else:
      second_failed_clusters.append(cluster.id)

  # Combine the results of the first and second pass.
  passed_clusters, failed_clusters = determine_failed_components(
      passed_clusters,
      failed_clusters,
      second_passed_clusters,
      second_failed_clusters,
  )

  print(f"After second pass - Failed racks: {failed_clusters}")
  print(f"After second pass - Passed racks: {passed_clusters}")
  return passed_clusters, failed_clusters


def determine_failed_components(
    first_pass_passed: list[str],
    first_pass_failed: list[str],
    second_pass_passed: list[str],
    second_pass_failed: list[str],
) -> tuple[list[str], list[str]]:
  """Determines the final list of failed and passed nodes."""
  passed_set = set(first_pass_passed)
  failed_set = set(second_pass_failed)

  # Add any node that passed the second pass to the passed set.
  for node in second_pass_passed:
    if node not in passed_set:
      passed_set.add(node)

  # Remove any node that passed the first pass but failed the second pass.
  # Nodes can not move from passed to failed.
  first_pass_failed_set = set(first_pass_failed)
  for node in second_pass_failed:
    if node not in first_pass_failed_set and node in failed_set:
      failed_set.remove(node)

  # Add any node that failed the first pass and is not in the passed set.
  # This can happen if the node was not tested in the second pass.
  for node in first_pass_failed_set:
    if node not in passed_set:
      failed_set.add(node)

  return list(passed_set), list(failed_set)


def get_nccl_test_results(
    v1: kubernetes.client.CoreV1Api,
    nodes: list[str],
) -> tuple[set[str], set[str]]:
  """Gets the results of the nccl tests."""
  passed_nodes = set()
  failed_nodes = set()

  tested_nodes = set(nodes)

  nodes = v1.list_node()
  for node in nodes.items:
    if node.metadata.name not in tested_nodes:
      continue
    test_result = node.metadata.labels.get("aiinfra/nccl-healthcheck-result")
    if test_result == "pass":
      passed_nodes.add(node.metadata.name)
    elif test_result == "fail":
      failed_nodes.add(node.metadata.name)
    elif test_result == "crash":
      failed_nodes.add(node.metadata.name)
    else:
      print(f"Node {node.metadata.name} has unknown result.")
  return passed_nodes, failed_nodes


def cleanup_labels() -> None:
  """Removes any potential labels from previous runs."""
  checker_common.run_command(
      f"{_KUBECTL} label nodes -l aiinfra/nccl-healthcheck-good-pass-needed"
      " aiinfra/nccl-healthcheck-good-pass-needed-"
  )
  checker_common.run_command(
      f"{_KUBECTL} label nodes -l aiinfra/nccl-healthcheck-second-pass-needed"
      " aiinfra/nccl-healthcheck-second-pass-needed-"
  )
  checker_common.run_command(
      f"{_KUBECTL} label nodes -l aiinfra/nccl-healthcheck-runtime-sec"
      " aiinfra/nccl-healthcheck-runtime-sec-"
  )
  checker_common.run_command(
      f"{_KUBECTL} label nodes -l aiinfra/nccl-healthcheck-result"
      " aiinfra/nccl-healthcheck-result-"
  )


def generate_index_pairs(length: int) -> list[tuple[int, int]]:
  """Returns random pairs of indices with no repeated items."""
  if length < 2:
    return []

  indices = list(range(length))
  random.shuffle(indices)
  pairs = []

  while len(indices) > 1:
    index1 = indices.pop()
    index2 = indices.pop()
    pairs.append((index1, index2))

  # If there's an odd number of indices, pair the last one randomly
  if indices:
    last_index = indices.pop()
    random_partner = random.randint(0, length - 1)
    # Ensure the index doesn't pair with itself
    while random_partner == last_index:
      random_partner = random.randint(0, length - 1)
    pairs.append((last_index, random_partner))

  return pairs


def get_nodes_data(
    kube_nodes: list[kubernetes.client.models.V1Node],
    require_topology: bool = False,
) -> list[dict[str, str]]:
  """Lists all nodes in the cluster.

  Args:
    kube_nodes: List of nodes from the cluster
    require_topology: If false, ignore topology labels on nodes
  Returns:
    List of nodes with topology data
  """
  nodes = []
  for kube_node in kube_nodes:
    # Option to only getnodes that have been labelled w/ topology data
    if (
        require_topology
        and ("topology.gke.io/cluster" not in kube_node.metadata.labels)
    ):
      continue
    # TODO - More general check for nodes w/ accelerators
    # Skip if it doesn't have a GPU (e.g. default-pool nodes)
    if "default-pool" in str(kube_node.metadata.name):
      logging.info(
          "Not an accelerator node; skipping node %s",
          kube_node.metadata.name,
      )
      continue

    node = {}
    node["cluster"] = kube_node.metadata.labels.get(
        "topology.gke.io/cluster", "unknown"
    )
    node["rack"] = kube_node.metadata.labels.get(
        "topology.gke.io/rack", "unknown"
    )
    node["host"] = kube_node.metadata.labels.get(
        "topology.gke.io/host", "unknown"
    )
    node["node_id"] = kube_node.metadata.name
    nodes.append(node)
  return nodes


def get_capacity_topology(
    nodes: list[dict[str, str]],
) -> common_pb2.Capacity:
  """Creates a capacity topology from the list of nodes."""

  capacity = common_pb2.Capacity()
  cluster_dict = dict()
  rack_dict = dict()

  for node_data in nodes:
    cluster_id = node_data["cluster"]
    rack_id = node_data["rack"]
    node_id = node_data["node_id"]
    host_id = node_data["host"]

    if cluster_id not in cluster_dict:
      cluster = capacity.clusters.add()
      cluster.id = cluster_id
      cluster_dict[cluster_id] = cluster
    cluster = cluster_dict[cluster_id]

    if rack_id not in rack_dict:
      rack = cluster.racks.add()
      rack.id = rack_id
      rack_dict[rack_id] = rack

    rack = rack_dict[rack_id]
    rack.nodes.append(common_pb2.Node(id=node_id, host=host_id))

  return capacity


def is_second_pass_enabled() -> bool:
  return os.environ.get("SECOND_PASS_ENABLED", "true").lower() == "true"
