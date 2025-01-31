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

"""Runs the tinymax test."""

import os
import re
import time
import checker_common

INSTANCE_TYPE = os.environ.get("INSTANCE_TYPE")
_K_RUN_TINY_MAX_TEST = (
    " /scripts/run-inside-container-enhance.sh"
)
_HEALTHCHECK_TIME_LABEL_KEY = "aiinfra/tinymax-healthcheck-runtime-sec"
_K_ADD_LABEL_FORMAT = "/scripts/kubectl label node %s %s=%s --overwrite"
K_REMOVE_LABEL_FORMAT = "/scripts/kubectl label node %s %s-"
_K_RESULT_LABEL_KEY = "aiinfra/tinymax-healthcheck-result"
_K_TAINT = "aiinfra/tinymax-healthcheck=failed:NoSchedule"
_K_TAINT_NODE_FORMAT = "/scripts/kubectl taint node %s %s"

WORKLOAD_TERMINATE_FILE = "/usr/share/nemo/workload_terminated"


def ensure_env_variables() -> None:
  """Ensure necessary environment variables are set."""
  print("Checking env variables....")
  required_envs = [
      "NODE_NAME",
      "DRY_RUN",
  ]
  for env in required_envs:
    if env not in os.environ:
      raise ValueError(f"Must set {env}")


def run_tinymax_test() -> bool:
  """Runs the tinymax test.

  Returns:
    True if the test passed, False otherwise.
  """
  print("Running tinymax training...")
  result = checker_common.run_command(_K_RUN_TINY_MAX_TEST)
  match = re.search(r"TinyMax TEST PASS!", result.stdout)
  if not match:
    print("Tinymax startup failed. Error: ", result.stderr)
    return False
  print("Tinymax test passed.")
  if INSTANCE_TYPE in (
      "a3-highgpu-8g",
      "a3-megagpu-8g",
  ):
    with open(WORKLOAD_TERMINATE_FILE, "w") as _:
      pass
  return True


def taint_node() -> None:
  if os.environ.get("DRY_RUN") == "true":
    print("Training was set to dry run. Will not taint node.")
    return

  node_name = os.environ.get("NODE_NAME")
  taint_command = _K_TAINT_NODE_FORMAT % (node_name, _K_TAINT)
  print("Applying taint  %s to node %s" % (taint_command, node_name))
  if not checker_common.run_command(taint_command):
    print("Failed to apply taint")


def label_node() -> None:
  """Labels the node with the epoch time that the next test should run."""
  checker_common.add_label(
      os.environ.get("NODE_NAME"),
      _HEALTHCHECK_TIME_LABEL_KEY,
      int(time.time()),
      _K_ADD_LABEL_FORMAT,
  )


def remove_label(node_name: str, label: str) -> None:
  print("removing label %s from node %s" % (label, node_name))
  checker_common.run_command(K_REMOVE_LABEL_FORMAT % (node_name, label))


def main() -> int:
  # Check to make sure all the needed envs variables are set.
  ensure_env_variables()

  # Run a tiny max training. This will train a 7b LLM over a 30 steps.
  success = run_tinymax_test()
  node_name = os.environ.get("NODE_NAME")
  checker_common.add_label(
      node_name,
      _K_RESULT_LABEL_KEY,
      "pass" if success else "fail",
      _K_ADD_LABEL_FORMAT,
  )

  # If the job was not successful and not a dry run, taint the node.
  if not success:
    taint_node()

  label_node()
  print("Finished running. Bye!")


if __name__ == "__main__":
  main()
