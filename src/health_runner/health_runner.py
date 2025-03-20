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
allows it to run for a specified duration (TIMEOUT_MINUTES), and then
terminates it.

Note:
    You must have the necessary permissions to create and delete daemonsets in
    the specified Kubernetes namespace.
"""

from collections.abc import Iterable
import logging
import os
import signal
import socket
import sys
from typing import Any
import uuid

from kubernetes import client

import checker_common
import health_results_pb2
import nccl_runner
from google.protobuf import timestamp_pb2

# Time to wait for HC jobs to complete
_SLEEP_TIME_MINUTES: str = os.environ.get("SLEEP_TIME_MINUTES", "10")
# Time to wait for full HR to complete
# Give enough time for HC + extra time for launch & cleanup
_TIMEOUT_MINUTES: str = os.environ.get(
    "TIMEOUT_MINUTES",
    f"{int(_SLEEP_TIME_MINUTES) + 5}",
)

_YAML_FILE = os.environ.get("YAML_FILE")

_HELM = os.environ.get("HELM_PATH", "/usr/local/bin/helm")
_HELM_CHART = os.environ.get("HELM_CHART")
_HELM_CHART_VERSION = os.environ.get("HELM_CHART_VERSION")
_HELM_INSTALL_FLAGS = os.environ.get("HELM_INSTALL_FLAGS")
_HELM_RELEASE_NAME_BASE = os.environ.get("HELM_RELEASE_NAME_BASE", "chs-hc")
_HC_ENV_PREFIX = "HC_ENV_"

_KUBECTL = os.environ.get("KUBECTL_PATH", "/app/kubectl")
_GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
# 
_K_NAME_GPU_NODES_IN_CLUSTER_COMMAND = (
    f"{_KUBECTL} get nodes -o jsonpath='{{range"
    ' .items[*]}{.metadata.name}{"\\n"}{end}\' | while read node; do'
    f" {_KUBECTL} get node $node -o"
    " jsonpath='{.status.capacity.nvidia\\.com/gpu}' | grep -q '[0-9]' &&"
    " echo $node; done"
)
_K_NUM_GPU_NODES_IN_CLUSTER_COMMAND = (
    _K_NAME_GPU_NODES_IN_CLUSTER_COMMAND + " | wc -l"
)
logging.root.setLevel(logging.INFO)

cleanup_functions = []


def post_run_cleanup() -> None:
  """Clean up after the health check."""
  for func in cleanup_functions:
    try:
      func()
    except Exception:  # pylint: disable=broad-exception-caught
      logging.exception("Cleanup failed.")


def timeout_handler(signum: Any, frame: Any) -> None:
  print(f"Received {signum} signal on frame {frame}. Exiting...")
  # Force a cleanup
  post_run_cleanup()
  # Exit the process
  logging.info("System Exiting...")
  sys.exit(0)

signal.signal(signal.SIGALRM, timeout_handler)


def main() -> None:
  logging.info("%s health runner started", socket.gethostname())
  # Test for NCCL health check
  health_app = os.environ.get("HEALTH_APP", "").lower()
  # Create Helm releases for each health check
  if health_app:
    logging.info("Running NCCL health check via `HEALTH_APP`")
    run_health_app(health_app)
  else:
    signal.alarm(int(_TIMEOUT_MINUTES) * 60)
    # Set timeout
    logging.info("Set timeout to %s minutes", _TIMEOUT_MINUTES)

    logging.info("Running Helm health check")
    run_health_check()


def run_health_app(health_app: str) -> None:
  """Run the health check."""
  health_results = health_results_pb2.HealthResults(
      created_date_time=timestamp_pb2.Timestamp().GetCurrentTime(),
  )
  orchestrator_config = None
  if _HELM_CHART:
    orchestrator_config = checker_common.HelmConfig(
        release_name_base=_HELM_RELEASE_NAME_BASE,
        chart=_HELM_CHART,
        chart_version=_HELM_CHART_VERSION,
        install_flags=_HELM_INSTALL_FLAGS,
    )
  elif _YAML_FILE:
    orchestrator_config = _YAML_FILE

  if health_app == "nccl":
    logging.info("Running NCCL health check via `HEALTH_APP`")
    nccl_result = nccl_runner.run_nccl_healthcheck(
        orchestrator_config=orchestrator_config
    )
    health_results.health_results.append(nccl_result)
  else:
    logging.error("Unsupported health app: %s", health_app)
    return

  print(health_results)

  # If GCS_BUCKET_NAME is set, upload the results to GCS.
  if _GCS_BUCKET_NAME:
    checker_common.upload_results_to_gcs(
        bucket_name=_GCS_BUCKET_NAME,
        health_results=health_results,
    )


def ensure_env_variables(required_envs: Iterable[str]) -> None:
  """Ensure necessary environment variables are set."""
  for env in required_envs:
    if env not in os.environ:
      raise ValueError(f"Must set {env}")


# 
def determine_test_iterations(num_nodes: int | None = None) -> int:
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

  Args:
    num_nodes: The number of nodes in the cluster. If not provided, it will be
      determined automatically.

  Returns:
    Integer of the number of tests that will be deployed.
  """
  is_blast_mode = str(os.environ.get("BLAST_MODE_ENABLED")).lower() in [
      "true",
      "1",
  ]

  if is_blast_mode:
    logging.info("Running blast mode")

    get_nodes_output = checker_common.run_command(
        _K_NUM_GPU_NODES_IN_CLUSTER_COMMAND
    )
    num_nodes = num_nodes if num_nodes else int(get_nodes_output.stdout)
    nodes_per_test = int(os.environ.get("NODES_CHECKED_PER_TEST", "1"))
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
      return min(int(manual_limit_str), max_num_tests)
    else:
      return max_num_tests

  else:
    logging.info("Running single test mode")
    return 1


def run_health_check() -> None:
  """Run the health check."""

  # PREPARATION
  ensure_env_variables(
      required_envs={
          "DRY_RUN",
          "HELM_CHART",  # Must be defined since can't assume health check type
      },
  )

  # This must be defined in the YAML configuration
  helm_chart_path = _HELM_CHART
  # 
  helm_chart_version = _HELM_CHART_VERSION
  # 
  helm_install_flags = _HELM_INSTALL_FLAGS
  # 
  helm_values: dict[str, str] = dict()

  node_names = os.environ.get("HOSTS_CSV", "nil")
  if node_names != "nil":
    node_names = node_names.split(",")
  else:
    node_names = (
        checker_common.run_command(_K_NAME_GPU_NODES_IN_CLUSTER_COMMAND)
        .stdout.strip()
        .split("\n")
    )

  num_nodes = os.environ.get("N_NODES", "nil")
  if num_nodes == "nil":
    num_nodes = len(node_names)
  else:
    num_nodes = int(num_nodes)
  node_names_csv = r"\,".join(node_names)

  # Determine number of tests to run
  num_tests = determine_test_iterations(num_nodes=num_nodes)
  logging.info("Creating %d tests...", num_tests)

  # Pass Node Names & Number of Nodes to all health checks
  helm_values["health_check.env.HOSTS_CSV"] = f'"{node_names_csv}"'
  helm_values["health_check.env.N_NODES"] = str(num_nodes)
  # Pass all other environment variables to health checks
  for key, value in os.environ.items():
    if key.startswith(_HC_ENV_PREFIX):
      # Strip the _HC_ENV_PREFIX prefix and convert to Helm value format
      helm_key = f"health_check.env.{key[len(_HC_ENV_PREFIX):]}"
      helm_values[helm_key] = f'"{value}"'

  # RUN HC
  release_names = []
  for i in range(num_tests):
    # If Helm release name is not unique, it will not install the release
    short_guid = str(uuid.uuid4())[:8]
    hc_release_name_suffix = f"{i}-{short_guid}"
    if _HELM_RELEASE_NAME_BASE:
      unique_release_name = (
          f"{_HELM_RELEASE_NAME_BASE}-{hc_release_name_suffix}"
      )
    else:
      unique_release_name = f"chs-hc-{hc_release_name_suffix}"

    release_names.append(unique_release_name)
    helm_values["job.name"] = f"chs-hc-{i}-{short_guid}"

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
      "Waiting for maximum of %s minutes before cleaning up...",
      _SLEEP_TIME_MINUTES,
  )
  # Helm releases & associated jobs are logged for reference outside of HR
  release_jobs = checker_common.get_created_jobs(release_names)
  jobs_and_releases: list[tuple[str, str]] = list(
      zip(release_jobs, release_names)
  )
  logging.info(
      "Helm charts and associated jobs: %s",
      jobs_and_releases,
  )
  # Sleep until all jobs are complete or timeout is reached
  checker_common.wait_till_jobs_complete(
      job_v1=client.BatchV1Api(),
      jobs_to_monitor=release_jobs,
      timeout_seconds=(int(_SLEEP_TIME_MINUTES) * 60),
      check_interval=10,
  )

  post_run_cleanup()

if __name__ == "__main__":
  signal.signal(signal.SIGTERM, checker_common.sigterm_handler)
  main()
