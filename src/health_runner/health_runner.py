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

"""GPU Health Check Daemonset runner.

This application creates a GPU health check daemonset in a Kubernetes cluster,
allows it to run for a specified duration (SLEEP_TIME_MINUTES), and then
terminates it.

Note:
    You must have the necessary permissions to create and delete daemonsets in
    the specified Kubernetes namespace.
"""
from collections.abc import Iterable
import heapq
import logging
import math
import os
import time
import uuid

import checker_common


_SLEEP_TIME_MINUTES = os.environ.get("SLEEP_TIME_MINUTES", "20")
_HELM = os.environ.get("HELM_PATH", "/usr/local/bin/helm")
_KUBECTL = os.environ.get("KUBECTL_PATH", "/app/kubectl")
_K_GPU_NODES_IN_CLUSTER_COMMAND = (
    f"{_KUBECTL} get nodes -l cloud.google.com/gke-accelerator --no-headers"
    " --show-labels=true"
)
_HC_NAME = os.environ.get("hc_name")
_ALLOWLIST_LABEL = f"aiinfra/{_HC_NAME}-healthcheck-test"
_RUNTIME_LABEL_FORMAT = f"aiinfra/{_HC_NAME}-healthcheck-runtime-sec"
_K_ADD_LABEL_FORMAT = "/app/kubectl label nodes %s %s=%s --overwrite"

logging.root.setLevel(logging.INFO)


def ensure_env_variables(required_envs: Iterable[str]) -> None:
  """Ensure necessary environment variables are set."""
  for env in required_envs:
    if env not in os.environ:
      raise ValueError(f"Must set {env}")


def parse_get_nodes_output(get_nodes_output: str) -> dict[str, list[str]]:
  """Parses kubectl get nodes output and returns a dict of node -> labels.

  Args:
    get_nodes_output: kubectl get nodes output.
  Returns:
    A dict of node to labels mapping.
  """
  node_label_map = {}
  for line in get_nodes_output.splitlines():
    cols = line.split()
    if len(cols) != 6:
      err_info = "kubectl get nodes output is unexpected {}".format(line)
      logging.info(err_info)
      raise ValueError(err_info)
    node_label_map[cols[0]] = cols[5].split(",")
  return node_label_map


def label_nodes_for_testing(
    node_label_map: dict[str, list[str]], node_count: int
) -> None:
  """Finds top N nodes based on last runtime and marks same for testing.

  Args:
    node_label_map: node and labels mapping.
    node_count: total nodes to label

  Raises:
    ValueError: If cluster has no applicable nodes or node_count is zero.
  """
  if not node_label_map:
    err_info = "No nodes are selected"
    logging.info(err_info)
    raise ValueError(err_info)
  queue = []
  for node, labels in node_label_map.items():
    for label in labels:
      if label.startswith(_RUNTIME_LABEL_FORMAT):
        queue.append((node, int(label.split("=")[1])))
        break
    queue.append((node, math.inf))

  heapq.heapify(queue)
  nodes_str = ""
  for _ in range(node_count):
    nodes_str = heapq.heappop(queue)[0] + " "

  checker_common.add_label(
      nodes_str, _ALLOWLIST_LABEL, "true", _K_ADD_LABEL_FORMAT
  )


# 
def determine_test_iterations() -> int:
  """Determine the number of tests to run.

  This function will calculate the number of tests to deploy for a given test
  run. The behavior is determined by three environment variables:
  * BLAST_MODE_ENABLED - If set to "true", more than one test can be deployed.
  By default this will fill the cluster and run a test on every compatible node.
  * BLAST_MODE_NUM_TESTS_LIMIT - If set to an integer, places a limit on the
  number of tests that will be deployed.
  * NODES_CHECKED_PER_TEST -  This should be set to the number of discrete nodes
  that will be consumed in the YAML_PATH yaml.
  Used to calculate the number of tests to deploy. (Defaults to 1)

  Returns:
  Integer of the number of tests that will be deployed.
  """
  is_blast_mode = str(os.environ.get("BLAST_MODE_ENABLED")).lower() in [
      "true",
      "1",
  ]

  get_nodes_output = checker_common.run_command(_K_GPU_NODES_IN_CLUSTER_COMMAND)
  if get_nodes_output.returncode:
    err_info = "kubectl get node command failed: {}".format(
        get_nodes_output.stderr)
    logging.info(err_info)
    raise ValueError(err_info)
  node_label_map = parse_get_nodes_output(get_nodes_output.stdout)
  num_nodes = len(node_label_map)
  nodes_per_test = int(os.environ.get("NODES_CHECKED_PER_TEST", "1"))

  # Run on all nodes by default
  nodes_count = (
      num_nodes
      if not os.environ.get("NUM_NODES")
      else min(int(os.environ.get("NUM_NODES", num_nodes)), num_nodes)
  )

  test_count = nodes_count // nodes_per_test
  if is_blast_mode:
    logging.info("Running blast mode")
    if num_nodes % nodes_per_test != 0:
      logging.warning(
          "Not all nodes can be checked. %d are present on the"
          " cluster. %d will be checked per test."
          " %d node(s) will be unchecked.",
          num_nodes,
          nodes_per_test,
          num_nodes % nodes_per_test,
      )
    max_num_tests = num_nodes // nodes_per_test

    manual_limit_str = os.environ.get("BLAST_MODE_NUM_TESTS_LIMIT")
    if manual_limit_str is not None:
      test_count = min(int(manual_limit_str), max_num_tests)
      nodes_count = test_count * nodes_per_test
    else:
      test_count = max_num_tests
  label_nodes_for_testing(node_label_map, nodes_count)
  return test_count


def main() -> None:
  ensure_env_variables(
      required_envs={
          "DRY_RUN",
          "HELM_CHART",  # Must be defined since can't assume health check type
      },
  )

  # Create Helm releases
  # Determine number of tests to run
  num_tests = determine_test_iterations()

  cleanup_functions = []

  logging.info("Creating %d tests...", num_tests)
  for i in range(num_tests):
    # If Helm release name is not unique, it will not install the release
    unique_release_name = os.environ.get(
        "HELM_RELEASE_NAME",
        f"internal-health-check-{i}-{str(uuid.uuid4())[:8]}-{time.time()}"
    )
    # This must be defined in the YAML configuration
    helm_chart_path = os.environ.get("HELM_CHART")
    # 
    helm_chart_version = os.environ.get("HELM_CHART_VERSION")
    # 
    helm_install_flags = os.environ.get("HELM_INSTALL_FLAGS")
    # 
    helm_values = os.environ.get("HELM_VALUES")
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
    # Count of tests deployed should start at 1 to make it clear
    logging.info("Deployed test %d (%d of %d total)", i, i + 1, num_tests)

  logging.info(
      "Health check is running. Waiting for %s minutes before cleaning up...",
      _SLEEP_TIME_MINUTES,
  )
  time.sleep(int(_SLEEP_TIME_MINUTES) * 60)

  # Cleanup cluster (uninstall helm releases, delete k8s objects, etc.)
  for func in cleanup_functions:
    try:
      func()
    except Exception:  # pylint: disable=broad-exception-caught
      logging.exception("Cleanup failed.")

if __name__ == "__main__":
  main()
