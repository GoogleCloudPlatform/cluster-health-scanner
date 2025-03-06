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

"""Performance health check runner runs performacne test on a supercomputer.

A performance test is a test that runs on large blocks in the capacity. The test
will span multiple nodes but the results will be interpreted at a block level.
"""

import copy
import os

import kubernetes.client
from kubernetes.client.api import batch_v1_api

import checker_common
import health_results_pb2
import health_runner_config_pb2

_CHECK_INTERVAL_SECONDS = os.environ.get("CHECK_INTERVAL_SECONDS", "30")


def run_performance_healthcheck(
    health_check: health_runner_config_pb2.HealthCheck,
) -> health_results_pb2.HealthResult:
  """Runs a performance health check."""
  kubernetes.config.load_incluster_config()
  v1 = kubernetes.client.CoreV1Api()
  batch_v1 = kubernetes.client.BatchV1Api()

  health_result = health_results_pb2.HealthResult(
      name=health_runner_config_pb2.HealthCheckName.Name(health_check.name),
      type=health_runner_config_pb2.HealthCheckType.Name(health_check.type),
  )
  # Get the cluster capacity
  node_data = checker_common.get_nodes_data(v1.list_node().items)
  capacity = checker_common.get_capacity_topology(node_data)

  # Group nodes by topology level
  topology_to_nodes = checker_common.create_topology_to_nodes_mapping(
      capacity, health_check.performance_health_check_config.topology_level
  )

  # Filter the topology to testable topology. A topology is testable if it
  # has more than one node.
  topology_to_testable_nodes = {}
  topology_to_results = {}
  for topology, nodes in topology_to_nodes.items():
    if len(nodes) <= 1:
      print(f"Skipping test for {topology} with only {len(nodes)} nodes.")
      topology_to_results[topology] = health_results_pb2.HealthResultList(
          id=topology,
          num_nodes=len(nodes),
          status=health_results_pb2.Status.SKIP,
      )
      continue
    topology_to_testable_nodes[topology] = nodes
    topology_to_results[topology] = health_results_pb2.HealthResultList(
        id=topology,
        num_nodes=len(nodes),
    )

  # Run tests on the topology if there are any testable topologies.
  if topology_to_testable_nodes:
    run_performance_healthchecks(
        batch_v1,
        v1,
        health_check,
        topology_to_testable_nodes,
        topology_to_results,
    )
  else:
    print("No jobs to run. Skipping test.")

  # Add all the results to the health result object
  for result in topology_to_results.values():
    health_result.health_results.append(result)

  return health_result


def run_performance_healthchecks(
    batch_v1: batch_v1_api.BatchV1Api,
    v1: kubernetes.client.CoreV1Api,
    health_check: health_runner_config_pb2.HealthCheck,
    topology_to_testable_nodes: dict[str, list[str]],
    topology_to_results: dict[str, health_results_pb2.HealthResultList],
) -> None:
  """Runs performance health checks on the given topologies.

  Args:
    batch_v1: Kubernetes batch API client.
    v1: Kubernetes core API client.
    health_check: Health check configuration.
    topology_to_testable_nodes: Mapping of topology (Cluster, SBRG, Rack) entity
      to its nodes.
    topology_to_results: Mapping of topology entity to health results.
  """
  health_check_tests = _generate_healthcheck_tests(health_check)
  env_mappings = copy.deepcopy(checker_common.parse_env_mappings(health_check))
  # Run tests on all the topologies for each test configuration
  for test in health_check_tests:
    print(f"Running test: {test}")
    job_names_to_topology = {}
    for topology, nodes in topology_to_testable_nodes.items():
      print(f"Running test for {len(nodes)} nodes in {topology}")
      test_env_mappings = copy.deepcopy(env_mappings)
      # Add topology value so that the job will be scoped to the topology
      test_env_mappings["TOPOLOGY_KEY"] = checker_common.topology_key(
          health_check.performance_health_check_config.topology_level
      )
      test_env_mappings["TOPOLOGY_VALUE"] = topology
      test_env_mappings["NHOSTS"] = len(nodes)
      test_env_mappings.update(test)
      job_name = checker_common.run_healthcheck(health_check, test_env_mappings)
      job_names_to_topology[job_name] = topology

    # Wait for jobs to complete for this test configuration
    print(f"Waiting for {len(job_names_to_topology)} jobs to complete...")
    checker_common.wait_till_jobs_complete(
        batch_v1,
        job_names_to_topology.keys(),
        timeout_seconds=health_check.timeout.seconds,
        check_interval=int(_CHECK_INTERVAL_SECONDS),
    )

    # Update the health results for this test configuration
    _update_health_results(
        v1,
        batch_v1,
        job_names_to_topology,
        topology_to_results,
        health_check,
    )
    # Cleanup all the created jobs for this test configuration
    checker_common.delete_jobs(batch_v1, job_names_to_topology.keys())


def _update_health_results(
    v1: kubernetes.client.CoreV1Api,
    batch_v1: batch_v1_api.BatchV1Api,
    jobs_to_topology: dict[str, str],
    topology_to_results: dict[str, health_results_pb2.HealthResultList],
    health_check: health_runner_config_pb2.HealthCheck,
) -> None:
  """Updates the health results for tests.

  Args:
    v1: Kubernetes client.
    batch_v1: Batch client.
    jobs_to_topology: Dictionary of job names to topology entity.
    topology_to_results: Dictionary of topology to results.
    health_check: Health check configuration.
  """
  # Check jobs for status
  successful_jobs = set()
  for job_name, topology in jobs_to_topology.items():
    if checker_common.job_succeeded(batch_v1, job_name):
      successful_jobs.add(job_name)
    else:
      topology_to_results[topology].status = health_results_pb2.Status.FAIL

  # Collect results from master node for successful jobs
  node_to_topology = []
  for job_name in successful_jobs:
    master_node = _get_master_node(v1, job_name)
    if master_node is None:
      raise ValueError(f"Failed to find master node for job: {job_name}")
    node_to_topology.append((master_node, jobs_to_topology[job_name]))

  # Perform test only support result label for now
  if health_check.result_label is None or not health_check.result_label:
    raise ValueError("No result label specified.")

  # Check the results label on the master node to determine pass/fail
  for node_name, topology in node_to_topology:
    result = topology_to_results[topology]
    node = v1.read_node(node_name)
    result_label = node.metadata.labels.get(health_check.result_label)
    if result_label and result_label == "pass":
      result.status = health_results_pb2.Status.PASS
    else:
      result.status = health_results_pb2.Status.FAIL
    # If a nccl based tests, parse nccl results
    if health_check.name in {
        health_runner_config_pb2.HealthCheckName.HEALTH_CHECK_NCCL_PERFORMANCE,
        health_runner_config_pb2.HealthCheckName.HEALTH_CHECK_NCCL_INTER_RACK,
        health_runner_config_pb2.HealthCheckName.HEALTH_CHECK_NCCL_INTER_CLUSTER,
        health_runner_config_pb2.HealthCheckName.HEALTH_CHECK_NCCL_INTRA_RACK,
    }:
      result.nccl_health_result.append(checker_common.parse_nccl_results(node))


# Generates the tests for the health check. Performance health checks can have
# multiple tests per health check do to parameter variations.
def _generate_healthcheck_tests(
    health_check: health_runner_config_pb2.HealthCheck,
) -> list[dict[str, str]]:
  """Generates the tests for the health check."""
  if health_check.name in {
      health_runner_config_pb2.HealthCheckName.HEALTH_CHECK_NCCL_PERFORMANCE,
      health_runner_config_pb2.HealthCheckName.HEALTH_CHECK_NCCL_INTER_RACK,
      health_runner_config_pb2.HealthCheckName.HEALTH_CHECK_NCCL_INTER_CLUSTER,
  }:
    tests = [
        {"BENCHMARK": benchmark}
        for benchmark in health_check.performance_health_check_config.nccl_performance_health_check_config.benchmarks
    ]
  else:
    # Runs only with the default configuration.
    tests = [{}]

  return tests


def _get_master_node(v1: kubernetes.client.CoreV1Api, job_name: str) -> str:
  """Get the master node for the job."""
  pod_list = v1.list_namespaced_pod(
      namespace="default", label_selector=f"job-name={job_name}"
  )  # Filter for the job.
  master_pod = None
  for pod in pod_list.items:
    # Check the pod name for the index.
    if "-0" in pod.metadata.name:
      print("Found master pod: ", pod.metadata.name)
      master_pod = pod
      break

  if master_pod is None:
    raise ValueError(f"Failed to find master pod for job: {job_name}")

  return master_pod.spec.node_name
