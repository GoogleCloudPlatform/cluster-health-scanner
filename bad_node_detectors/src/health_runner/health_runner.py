"""GPU Health Check Daemonset runner.

This application creates a GPU health check daemonset in a Kubernetes cluster,
allows it to run for a specified duration (SLEEP_TIME_MINUTES), and then
terminates it.

Note:
    You must have the necessary permissions to create and delete daemonsets in
    the specified Kubernetes namespace.
"""

import logging
import os
import time

import checker_common


_SLEEP_TIME_MINUTES = os.environ.get("SLEEP_TIME_MINUTES", "20")
_KUBECTL = os.environ.get("KUBECTL_PATH", "/app/kubectl")
_IMAGE_VERSION_PATH = os.environ.get("IMAGE_VERSION_PATH", "image_version.txt")
_K_GPU_NODES_IN_CLUSTER_COMMAND = (
    f"{_KUBECTL} get nodes -l cloud.google.com/gke-accelerator"
    " --no-headers | wc -l"
)

logging.root.setLevel(logging.INFO)


def ensure_env_variables() -> None:
  """Ensure necessary environment variables are set."""
  required_envs = ["YAML_FILE", "DRY_RUN"]
  for env in required_envs:
    if env not in os.environ:
      raise ValueError(f"Must set {env}")


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

  if is_blast_mode:
    logging.info("Running blast mode")

    get_nodes_output = checker_common.run_command(
        _K_GPU_NODES_IN_CLUSTER_COMMAND
    )
    num_nodes = int(get_nodes_output.stdout)
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


def main() -> None:
  ensure_env_variables()

  if (
      os.path.exists(_IMAGE_VERSION_PATH)
      and os.environ.get("IMAGE_TAG") is None
  ):
    logging.info("Loading image tag based on %s", _IMAGE_VERSION_PATH)
    with open(_IMAGE_VERSION_PATH, "r") as f:
      tag = f.read().strip()
      os.environ["IMAGE_TAG"] = tag

  # Step 1: Create k8s components
  yaml_path = os.path.join("/app", os.environ.get("YAML_FILE"))

  # Determine number of tests to run
  num_tests = determine_test_iterations()

  cleanup_functions = []

  logging.info("Creating %d tests...", num_tests)
  for i in range(num_tests):
    cleanup_functions.extend(
        checker_common.create_k8s_objects(yaml_path, _KUBECTL)
    )
    # Sleep to force a different timestamp
    time.sleep(1.5)
    logging.info("Deployed test %d / %d.", i, num_tests)

  logging.info(
      "Health check is running. Waiting for %s minutes...", _SLEEP_TIME_MINUTES
  )
  time.sleep(int(_SLEEP_TIME_MINUTES) * 60)

  # Step 3: Cleanup cluster
  for func in cleanup_functions:
    try:
      func()
    except Exception:  # pylint: disable=broad-exception-caught
      logging.exception("Cleanup failed.")

if __name__ == "__main__":
  main()
