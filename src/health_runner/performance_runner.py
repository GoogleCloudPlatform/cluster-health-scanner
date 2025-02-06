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
import uuid

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
  batch_v1 = batch_v1_api.BatchV1Api()

  health_result = health_results_pb2.HealthResult(
      name=health_runner_config_pb2.HealthCheckName.Name(health_check.name),
      type=health_runner_config_pb2.HealthCheckType.Name(health_check.type),
  )
  # Get the cluster capacity
  node_data = checker_common.get_nodes_data(v1.list_node().items)
  capacity = checker_common.get_capacity_topology(node_data)

  # Group by Superblock Rail Group (SBRG)
  sbrg_to_nodes = {}
  for cluster in capacity.clusters:
    for rack in cluster.racks:
      for node in rack.nodes:
        sbrg_to_nodes.setdefault(cluster.id, []).append(node.id)

  env_mappings = copy.deepcopy(checker_common.parse_env_mappings(health_check))
  # Check for NHOSTS flag in health check params to determine node count
  if env_mappings.get("NHOSTS") is None:
    raise ValueError("NHOSTS not specified in performance health checks.")
  test_node_count = int(env_mappings.get("NHOSTS"))

  # Run test per sbrg if node count is > test node count
  job_names_to_sbrg = {}
  for sbrg, nodes in sbrg_to_nodes.items():
    if len(nodes) < test_node_count:
      print(f"Skipping sbrg {sbrg} with less than {test_node_count} nodes.")
      continue
    job_name = f"diag-performance-{str(uuid.uuid4())[:8]}"
    print(f"Running test for {len(nodes)} nodes in sbrg {sbrg}")
    # Add topology value so that the job will be scoped to the sbrg
    env_mappings["TOPOLOGY_VALUE"] = sbrg
    checker_common.create_job_k8s(
        job_name=job_name,
        yaml_file=health_check.yaml_file,
        env_mappings=env_mappings,
    )
    job_names_to_sbrg[job_name] = sbrg
    # Instead of running a performance test per sbrg, we can run a single
    # performance test for all sbrgs.
    break

  if not job_names_to_sbrg:
    print("Could not find any sbrg with enough nodes. Skipping test.")
    # TODO: Update to skip
    health_result.health_results.append(
        health_results_pb2.HealthResultList(
            id="N/A",
            status=health_results_pb2.Status.FAIL,
        )
    )
    return health_result

  # Wait for jobs to complete
  print(f"Waiting for {len(job_names_to_sbrg)} jobs to complete...")
  checker_common.wait_till_jobs_complete(
      batch_v1,
      job_names_to_sbrg.keys(),
      timeout_seconds=health_check.timeout.seconds,
      check_interval=int(_CHECK_INTERVAL_SECONDS),
  )

  # Create health results
  results = _fetch_results(v1, batch_v1, job_names_to_sbrg, health_check)
  health_result.health_results.extend(results)

  checker_common.delete_jobs(batch_v1, job_names_to_sbrg.keys())
  return health_result


def _fetch_results(
    v1: kubernetes.client.CoreV1Api,
    batch_v1: batch_v1_api.BatchV1Api,
    jobs_to_sbrg: dict[str, str],
    health_check: health_runner_config_pb2.HealthCheck,
) -> list[health_results_pb2.HealthResultList]:
  """Create health results for the failed and successful jobs.

  Args:
    v1: Kubernetes client
    batch_v1: Batch client
    jobs_to_sbrg: Dictionary of job names to sbrg
    health_check: Health check configuration

  Returns:
    List of health results
  """
  results = []

  # Check jobs for status
  successful_jobs = set()
  for job_name, sbrg in jobs_to_sbrg.items():
    if checker_common.job_succeeded(batch_v1, job_name):
      successful_jobs.add(job_name)
    else:
      results.append(
          health_results_pb2.HealthResultList(
              id=sbrg,
              status=health_results_pb2.Status.FAIL,
          )
      )

  # Collect results from master node for successful jobs
  node_to_sbrg = []
  for job_name in successful_jobs:
    master_node = _get_master_node(v1, job_name)
    if master_node is None:
      raise ValueError(f"Failed to find master node for job: {job_name}")
    node_to_sbrg.append((master_node, jobs_to_sbrg[job_name]))

  # Perform test only support result label for now
  if health_check.result_label is None or not health_check.result_label:
    raise ValueError("No result label specified.")

  # Check the results label on the master node to determine pass/fail
  for node_name, sbrg in node_to_sbrg:
    node = v1.read_node(node_name)
    result_label = node.metadata.labels.get(health_check.result_label)
    if result_label is not None and result_label == "pass":
      results.append(
          health_results_pb2.HealthResultList(
              id=sbrg,
              status=health_results_pb2.Status.PASS,
          )
      )
    else:
      results.append(
          health_results_pb2.HealthResultList(
              id=sbrg,
              status=health_results_pb2.Status.FAIL,
          )
      )

  return results


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
