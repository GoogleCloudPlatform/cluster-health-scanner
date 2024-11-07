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

from collections.abc import Callable
import logging
import os
import random
import subprocess
import time
import uuid
import kubernetes.client
from kubernetes.client.api import batch_v1_api
import checker_common
import common_pb2

_KUBECTL = os.environ.get("KUBECTL_PATH", "/app/kubectl")
_NCCL_INTER_RACK_YAML_PATH = os.environ.get(
    "NCCL_TOPOLOGY_YAML_PATH", "/app/a3plus/nccl_pairwise_topology.yaml"
)


def run_nccl_healthcheck(num_tests: int):
  """Runs NCCL health check and waits for it to complete."""
  print("cleaning up labels")
  cleanup_labels()

  kubernetes.config.load_incluster_config()
  v1 = kubernetes.client.CoreV1Api()
  node_data = get_nodes_data(v1.list_node().items)
  capacity = get_capacity_topology(node_data)

  case = os.environ.get("HEALTH_CHECK_CASE", "random_pair")
  second_pass_enabled = is_second_pass_enabled()
  if case == "intra_rack":
    run_intra_rack_healthcheck(v1, capacity, second_pass_enabled)
  elif case == "inter_rack":
    run_inter_rack_healthcheck(v1, capacity, second_pass_enabled)
  elif case == "inter_cluster":
    run_inter_cluster_healthcheck(v1, capacity, second_pass_enabled)
  elif case == "random_pair":
    run_nccl_random_pair_healthcheck(num_tests, second_pass_enabled)
  else:
    print(f"Unknown health check case: {case}")
    run_nccl_random_pair_healthcheck(num_tests, second_pass_enabled)


def run_nccl_random_pair_healthcheck(num_tests: int, second_pass_enabled: bool):
  """Runs NCCL health checks between random nodes and waits for it to complete.

  Args:
    num_tests: Number of tests to run.
    second_pass_enabled: Whether second pass is enabled.
  """
  yaml_path = os.path.join("/app", os.environ.get("YAML_FILE"))

  logging.info("Creating %d tests...", num_tests)

  cleanup_functions = _deploy_test_run_and_wait(num_tests, yaml_path, num_tests)

  print("All pods are done with first pass")

  # Step 3: Cleanup cluster
  _cleanup_cluster_and_wait(cleanup_functions)

  second_pass_file = os.environ.get("SECOND_PASS_YAML_FILE")
  if not second_pass_enabled or not second_pass_file:
    print("second pass is not enabled, exiting...")
    return

  print("deploying second pass")
  bad_nodes = _get_int_from_stdout(
      "/app/kubectl get nodes -l aiinfra/nccl-healthcheck-second-pass-needed"
      " --no-headers | wc -l"
  )
  good_nodes = _get_int_from_stdout(
      "/app/kubectl get nodes -l aiinfra/nccl-healthcheck-good-pass-needed"
      " --no-headers | wc -l"
  )

  print(f"found bad nodes: {bad_nodes}")
  print(f"found good nodes: {good_nodes}")

  if not bad_nodes:
    print("Skipping second pass since no nodes failed")
    return

  print("mark all nodes as bad if they didn't complete the first pass")
  checker_common.run_command(
      "/app/kubectl label nodes -l"
      " '!aiinfra/nccl-healthcheck-result,cloud.google.com/gke-accelerator'"
      " aiinfra/nccl-healthcheck-second-pass-needed=true --overwrite"
  )
  print("cleanup timestamp labels so it can be scheduled again")
  checker_common.run_command(
      "/app/kubectl label nodes -l aiinfra/nccl-healthcheck-runtime-sec"
      " aiinfra/nccl-healthcheck-runtime-sec-"
  )

  second_pass_yaml_path = os.path.join("/app", second_pass_file)

  # Look into moving the nodes per test logic fully into the nccl runner.
  nodes_per_test = int(os.environ.get("NODES_CHECKED_PER_TEST", "1"))
  cleanup_functions = _deploy_test_run_and_wait(
      bad_nodes, second_pass_yaml_path, bad_nodes * nodes_per_test
  )

  print("All pods are done with second pass")

  _cleanup_cluster_and_wait(cleanup_functions)
  print("no pods are running after second pass cleanup")

  print("mark all nodes as bad if they didn't complete the second pass in time")

  checker_common.run_command(
      "/app/kubectl label nodes -l"
      " '!aiinfra/nccl-healthcheck-result,cloud.google.com/gke-accelerator'"
      " aiinfra/nccl-healthcheck-result=fail"
  )

  print("nodes with bad result:")
  checker_common.run_command(
      "/app/kubectl get nodes -l aiinfra/nccl-healthcheck-result=fail"
  )
  print("nodes with crash result:")
  checker_common.run_command(
      "/app/kubectl get nodes -l aiinfra/nccl-healthcheck-result=crash"
  )
  print("Health check is done ... bye")


def run_intra_rack_healthcheck(
    v1: kubernetes.client.CoreV1Api,
    capacity: common_pb2.Capacity,
    second_pass_enabled: bool,
) -> tuple[list[str], list[str]]:
  """Checks the racks communication by running nccl tests between each node in the same rack."""
  # Create a dictionary of rack to nodes for easier lookup.
  rack_to_nodes = {}
  for cluster in capacity.clusters:
    for rack in cluster.racks:
      rack_to_nodes[rack.id] = [node.id for node in rack.nodes]

  # Create a list of job names for to monitor the jobs.
  job_names = []
  tested_nodes = []

  # For the first pass, pair each node in the rack
  for rack, nodes in rack_to_nodes.items():
    if len(nodes) < 2:
      print(f"Skipping rack {rack} with less than 2 nodes.")
      continue

    node_pairs = generate_index_pairs(len(nodes))
    for i, j in node_pairs:
      node1 = nodes[i]
      node2 = nodes[j]
      print(
          f"Running NCCL test between node {node1} (Rack:"
          f" {rack}) and node {node2} (Rack:"
          f" {rack})"
      )

      short_guid = str(uuid.uuid4())[:4]
      checker_common.create_k8s_objects(
          _NCCL_INTER_RACK_YAML_PATH,
          _KUBECTL,
          print_output=False,
          mappings={
              "NODE1": node1,
              "NODE2": node2,
              "SECOND_PASS": "false",
              "SHORT_GUID": short_guid,
          },
      )
      job_names.append(f"nccl-healthcheck-{short_guid}")
      tested_nodes.append(node1)
      tested_nodes.append(node2)

  job_api = batch_v1_api.BatchV1Api()
  print(f"Waiting for {len(job_names)} jobs to complete...")
  checker_common.wait_till_jobs_complete(job_api, job_names)

  # Get the failed and passed nodes from the first pass.
  passed_nodes, failed_nodes = get_nccl_test_results(v1, tested_nodes)

  if not second_pass_enabled or not failed_nodes:
    # If no second pass (Or no failed nodes), then return results.
    print(f"Found failed nodes: {failed_nodes}")
    print(f"Found passed nodes: {passed_nodes}")
    return list(passed_nodes), list(failed_nodes)

  print(f"Running second pass for {len(failed_nodes)} nodes...")

  # Reset the jobs name list.
  job_names = []
  tested_nodes = []

  # For the second pass, pair each failed node with a passed node in the same
  # rack.
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
      print(
          f"Running NCCL test between good node {healthy_node} (Rack:"
          f" {rack}) and failed node {failed_node} (Rack:"
          f" {rack})"
      )

      short_guid = str(uuid.uuid4())[:4]
      checker_common.create_k8s_objects(
          _NCCL_INTER_RACK_YAML_PATH,
          _KUBECTL,
          print_output=False,
          mappings={
              "NODE1": healthy_node,
              "NODE2": failed_node,
              "SECOND_PASS": "true",
              "SHORT_GUID": short_guid,
          },
      )
      job_names.append(f"nccl-healthcheck-{short_guid}")
      tested_nodes.append(failed_node)
      # Probably don't need to add healthy node here
      tested_nodes.append(healthy_node)

  checker_common.wait_till_jobs_complete(job_api, job_names)

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

  # Create a dictionary of cluster to racks and a dictionary of rack to nodes
  # to make it easier to look up info.
  for cluster in capacity.clusters:
    cluster_to_racks[cluster.id] = []
    for rack in cluster.racks:
      cluster_to_racks[cluster.id].append(rack.id)
      rack_to_nodes[rack.id] = [node.id for node in rack.nodes]

  # Create a list of job names to monitor the jobs.
  job_names = []
  # Create a list of nodes to hold the nodes that were tested.
  tested_nodes = []

  for _, racks in cluster_to_racks.items():
    # Pair each rack with another rack in the same cluster
    rack_pairs = generate_index_pairs(len(racks))
    for i, j in rack_pairs:
      nodes1 = rack_to_nodes[racks[i]]
      nodes2 = rack_to_nodes[racks[j]]
      if not nodes1 or not nodes2:
        raise ValueError(f"Rack: {racks[i]} or {racks[j]} has no nodes.")
      # Randomly select a node from each rack
      node1 = random.choice(nodes1)
      node2 = random.choice(nodes2)
      print(
          f"Running NCCL test between node {node1} (Rack:"
          f" {racks[i]}) and node {node2} (Rack:"
          f" {racks[j]})"
      )
      short_guid = str(uuid.uuid4())[:4]
      # Run a nccl test between the two racks.
      checker_common.create_k8s_objects(
          _NCCL_INTER_RACK_YAML_PATH,
          _KUBECTL,
          print_output=False,
          mappings={
              "NODE1": node1,
              "NODE2": node2,
              "SECOND_PASS": "false",
              "SHORT_GUID": short_guid,
          },
      )
      job_names.append(f"nccl-healthcheck-{short_guid}")
      tested_nodes.append(node1)
      tested_nodes.append(node2)

  job_api = batch_v1_api.BatchV1Api()
  print(f"Waiting for {len(job_names)} jobs to complete...")
  checker_common.wait_till_jobs_complete(job_api, job_names)

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

  if not second_pass_enabled or not failed_racks:
    print(f"Found failed racks: {failed_racks}")
    print(f"Found passed racks: {passed_racks}")
    return passed_racks, failed_racks

  job_names = []
  tested_nodes = []

  print(f"Running second pass for {len(failed_racks)} racks...")

  # If second pass is enabled, we will run the test between the passed and
  # failed racks in the same cluster.
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

      print(
          f"Running NCCL test between good node {healthy_node} (Rack:"
          f" {healthy_rack}) and failed node {failed_node} (Rack:"
          f" {failed_rack})"
      )

      # Run the second pass between the passed and failed racks in the same
      # cluster.
      short_guid = str(uuid.uuid4())[:4]
      checker_common.create_k8s_objects(
          _NCCL_INTER_RACK_YAML_PATH,
          _KUBECTL,
          print_output=False,
          mappings={
              "NODE1": healthy_node,
              "NODE2": failed_node,
              "SECOND_PASS": "true",
              "SHORT_GUID": short_guid,
          },
      )
      job_names.append(f"nccl-healthcheck-{short_guid}")
      tested_nodes.append(healthy_node)
      tested_nodes.append(failed_node)

  checker_common.wait_till_jobs_complete(job_api, job_names)

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

  # Create a list of job names to monitor the jobs.
  job_names = []
  tested_nodes = []

  # Pair each cluster with another cluster
  cluster_pairs = generate_index_pairs(len(capacity.clusters))
  for i, j in cluster_pairs:
    nodes1 = cluster_to_nodes[capacity.clusters[i].id]
    nodes2 = cluster_to_nodes[capacity.clusters[j].id]
    if not nodes1 or not nodes2:
      raise ValueError(
          f"Cluster {capacity.clusters[i].id} or {capacity.clusters[j].id} has"
          " no racks."
      )

    # Randomly select a node from each cluster
    node1 = random.choice(nodes1)
    node2 = random.choice(nodes2)

    print(
        f"Running NCCL test between node {node1} (Cluster:"
        f" {capacity.clusters[i].id}) and node"
        f" {node2} (Cluster: {capacity.clusters[j].id})"
    )

    # Run a nccl test between the two clusters.
    short_guid = str(uuid.uuid4())[:4]
    checker_common.create_k8s_objects(
        _NCCL_INTER_RACK_YAML_PATH,
        _KUBECTL,
        print_output=False,
        mappings={
            "NODE1": node1,
            "NODE2": node2,
            "SECOND_PASS": "false",
            "SHORT_GUID": short_guid,
        },
    )
    job_names.append(f"nccl-healthcheck-{short_guid}")
    tested_nodes.append(node1)
    tested_nodes.append(node2)

  job_api = batch_v1_api.BatchV1Api()
  checker_common.wait_till_jobs_complete(job_api, job_names)

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
    print(f"Found failed clusters: {passed_clusters}")
    print(f"Found passed clusters: {failed_clusters}")
    return passed_clusters, failed_clusters

  job_names = []
  tested_nodes = []

  print(f"Running second pass for {len(failed_clusters)} clusters...")

  # Loop through the failed clusters and pair it with a random healthy cluster
  for failed_cluster in failed_clusters:
    healthy_cluster = random.choice(passed_clusters)

    failed_nodes = cluster_to_nodes[failed_cluster]
    healthy_nodes = cluster_to_nodes[healthy_cluster]

    failed_node = random.choice(failed_nodes)
    healthy_node = random.choice(healthy_nodes)

    print(
        f"Running NCCL test between good node {healthy_node} (Cluster:"
        f" {healthy_cluster}) and failed node {failed_node} (Cluster:"
        f" {failed_cluster})"
    )

    short_guid = str(uuid.uuid4())[:4]
    checker_common.create_k8s_objects(
        _NCCL_INTER_RACK_YAML_PATH,
        _KUBECTL,
        print_output=False,
        mappings={
            "NODE1": healthy_node,
            "NODE2": failed_node,
            "SECOND_PASS": "true",
            "SHORT_GUID": short_guid,
        },
    )
    job_names.append(f"nccl-healthcheck-{short_guid}")
    tested_nodes.append(healthy_node)
    tested_nodes.append(failed_node)

  checker_common.wait_till_jobs_complete(job_api, job_names)
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


def _get_int_from_stdout(cmd: str, default_value: int = 0) -> int:
  res = checker_common.run_command(cmd, check=True)
  if res.stdout.strip().isdigit():
    return int(res.stdout.strip())
  else:
    print(
        f"Error: Command output is not a digit. Received: {res.stdout.strip()}"
    )
    return default_value


def _wait_for_pods(
    cmd: str,
    target_count: int,
    timeout_seconds: int = 300,
    check_interval: int = 10,
) -> bool:
  """Waits for pods to reach target count.

  Args:
    cmd: Command to run to get the count.
    target_count: Target count to wait for.
    timeout_seconds: Timeout in seconds.
    check_interval: Interval in seconds to check for the count.

  Returns:
    True if the target count is reached.
  """
  start_time = time.time()

  while time.time() - start_time < timeout_seconds:
    count = _get_int_from_stdout(cmd)
    if count == target_count:
      print("done waiting for pods")
      return True

    print(
        f"Waiting for pods to come up... ({int(time.time() - start_time)}s) out"
        f" of {timeout_seconds}s"
    )
    print(f"count: {count} target count: {target_count}")

    time.sleep(check_interval)

  print(
      f"Timeout reached after {timeout_seconds} seconds. Some pods may still"
      " not be in Error or Completed state."
  )
  return False


def _deploy_test_run_and_wait(
    num_tests: int, yaml_path: str, number_of_pods_to_wait: int
) -> list[Callable[[], subprocess.CompletedProcess[str]]]:
  """Deploys tests and waits for them to complete.

  Args:
    num_tests: Number of tests to deploy.
    yaml_path: Path to the yaml file.
    number_of_pods_to_wait: Number of pods to wait for.

  Returns:
    List of cleanup functions.
  """
  cleanup_functions = []
  for i in range(num_tests):
    cleanup_functions.extend(
        checker_common.create_k8s_objects(yaml_path, _KUBECTL)
    )
    # Sleep to force a different timestamp
    time.sleep(1.5)
    logging.info("Deployed test %d / %d.", i + 1, num_tests)
  cmd = (
      "/app/kubectl get po --no-headers -l app-name=nccl-healthcheck | grep -e"
      " Error -e Completed | wc -l"
  )
  _wait_for_pods(
      cmd,
      number_of_pods_to_wait,
      timeout_seconds=10 * 60,  # 10 min
      check_interval=10,
  )
  return cleanup_functions


def _cleanup_cluster_and_wait(
    cleanup_functions: list[Callable[[], subprocess.CompletedProcess[str]]],
) -> None:
  """Runs cluster cleanup and waits for all pods to be gone.

  Args:
    cleanup_functions: List of functions to run for cleanup.
  """
  print("Running cluster cleanup.")
  for func in cleanup_functions:
    try:
      func()
    except Exception:  # pylint: disable=broad-exception-caught
      logging.exception("Cleanup failed.")

  print("wait for all pods to be gone")
  cmd = "/app/kubectl get po --no-headers -l app-name=nccl-healthcheck | wc -l"
  _wait_for_pods(
      cmd,
      0,
      timeout_seconds=5 * 60,  # 15 min
      check_interval=10,
  )
  print("no pods are running after cleanup")


def cleanup_labels() -> None:
  """Removes any potential labels from previous runs."""
  checker_common.run_command(
      "/app/kubectl label nodes -l aiinfra/nccl-healthcheck-good-pass-needed"
      " aiinfra/nccl-healthcheck-good-pass-needed-"
  )
  checker_common.run_command(
      "/app/kubectl label nodes -l aiinfra/nccl-healthcheck-second-pass-needed"
      " aiinfra/nccl-healthcheck-second-pass-needed-"
  )
  checker_common.run_command(
      "/app/kubectl label nodes -l aiinfra/nccl-healthcheck-runtime-sec"
      " aiinfra/nccl-healthcheck-runtime-sec-"
  )
  checker_common.run_command(
      "/app/kubectl label nodes -l aiinfra/nccl-healthcheck-result"
      " aiinfra/nccl-healthcheck-result-"
  )


def generate_index_pairs(length: int) -> list[tuple[int, int]]:
  """Generates pairs of indices from a list of length."""
  if length < 2:
    return []

  indices = list(range(length))
  random.shuffle(indices)
  pairs = []

  while len(indices) > 1:
    index1 = indices.pop(0)
    index2 = indices.pop(0)
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
) -> list[dict[str, str]]:
  """Lists all nodes in the cluster."""
  nodes = []
  for kube_node in kube_nodes:
    if "topology.gke.io/cluster" not in kube_node.metadata.labels:
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

    if cluster_id not in cluster_dict.keys():
      cluster = capacity.clusters.add()
      cluster.id = cluster_id
      cluster_dict[cluster_id] = cluster
    cluster = cluster_dict[cluster_id]

    if rack_id not in rack_dict.keys():
      rack = cluster.racks.add()
      rack.id = rack_id
      rack_dict[rack_id] = rack

    rack = rack_dict[rack_id]
    rack.nodes.append(common_pb2.Node(id=node_id, host=host_id))

  return capacity


def is_second_pass_enabled() -> bool:
  return os.environ.get("SECOND_PASS_ENABLED", "true").lower() == "true"
