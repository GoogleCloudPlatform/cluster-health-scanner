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

"""Runs GPU DCGM Diagnostic on k8s node.

This module runs NVIDIA DCGM (Data Center GPU Manager) diagnostics to assess the
health of GPUs in a given node of a Kubernetes cluster.
It uses the following environment variables for configuration:

Environment Variables:
    - R_LEVEL: The diagnostic level to pass to the DCGM diagnostic tool.
    Defaults to "1" (basic tests).
    - NODE_NAME: The name of the node on which the diagnostics are to be run.
    Defaults to the hostname.

Note:
    Make sure you have the necessary permissions to apply taints to nodes in the
    cluster.
"""

import json
import os
import time

import checker_common
import metrics

_RESULT_LABEL_KEY = "aiinfra/gpu-healthcheck-result"
TAINT_KEY = "aiinfra/gpu-healthcheck"
TAINT_VALUE = "failed"
TAINT_EFFECT = "NoSchedule"
REBOOT_REQUIRED_LABEL_KEY = "aiinfra/reboot-required"
HEALTHCHECK_TIME_LABEL_KEY = "aiinfra/gpu-healthcheck-valid-till-sec"
# for r level see:
# https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/dcgm-diagnostics.html
R_LEVEL = os.environ.get("R_LEVEL") or "1"
DCGM_COMMAND = "dcgmi diag -g 0 -r %s --verbose" % R_LEVEL
NVIDIA_SMI_COMMAND = (
    "/usr/local/nvidia/bin/nvidia-smi"
    " --query-gpu=ecc.errors.uncorrected.volatile.total"
    " --format=csv,noheader,nounits"
)
K_ADD_LABEL_FORMAT = "/app/kubectl label node %s %s=%s --overwrite"
K_REMOVE_LABEL_FORMAT = "/app/kubectl label node %s %s-"
K_TAINT_NODE_FORMAT = "/app/kubectl taint node %s %s=%s:%s"
K_UNTAINT_NODE_FORMAT = "/app/kubectl taint node %s %s-"


def main() -> None:
  """entry point."""
  node_name = os.environ.get("NODE_NAME")

  print("diagnostic on vm '%s' is 'initiated'" % node_name)

  reboot_required = run_reboot_required_check(node_name)
  run_dcgm_diag(node_name, reboot_required)

  # Add timestampt as number of seconds
  # since epoch time January 1, 1970, 00:00:00 (UTC) + 24 (default) hours
  # health validity.
  health_validity = (
      int(time.time())
      + int(os.environ.get("HEALTH_VALIDITY_HOURS", "24")) * 60 * 60
  )
  checker_common.add_label(
      node_name,
      HEALTHCHECK_TIME_LABEL_KEY,
      f"{health_validity}",
      K_ADD_LABEL_FORMAT,
  )


def run_reboot_required_check(node_name: str) -> bool:
  """run reboot required check."""
  smi_output = checker_common.run_command(NVIDIA_SMI_COMMAND)

  total_errors = 0
  reboot_required = False

  smi_lines = smi_output.stdout.splitlines()
  if len(smi_lines) != 8:
    reboot_required = True
    print("vm '%s' has only $s GPUs instead of 8" % node_name, len(smi_lines))

  for l in smi_lines:
    # If any gpu has that flag then we need reboot.
    try:
      total_errors += int(l)
    except ValueError:
      # If l is not a valid integer, handle the exception
      print(f"Error: cannot parse '{l}' to a number.")
      # That will taint the node and let dcgm test to run
      total_errors += 1

  # If any errors then label node as reboot required.
  if reboot_required or total_errors > 0:
    print(
        "adding reboot required label for node %s due to %s errors"
        % (node_name, total_errors),
    )
    checker_common.add_label(
        node_name, REBOOT_REQUIRED_LABEL_KEY, "true", K_ADD_LABEL_FORMAT
    )
    taint_node(node_name, TAINT_KEY, TAINT_VALUE, TAINT_EFFECT)
    return True
  else:
    print("reboot is not required for node %s" % node_name)
    remove_label(node_name, REBOOT_REQUIRED_LABEL_KEY)
    return False


def run_dcgm_diag(node_name: str, reboot_required: bool) -> None:
  """run dcgm diag."""
  diag_output = checker_common.run_command(DCGM_COMMAND)

  failed = is_bad_node(diag_output.stdout)

  print(
      json.dumps(
          metrics.log_dict(
              test_name="dcgm",
              did_pass=not failed,
              node_name=node_name,
              result_data={},
          )
      )
  )
  checker_common.add_label(
      node_name,
      _RESULT_LABEL_KEY,
      "fail" if failed else "pass",
      K_ADD_LABEL_FORMAT,
  )
  if failed:
    print("diagnostic on vm '%s' is 'failed'" % node_name)
    taint_node(node_name, TAINT_KEY, TAINT_VALUE, TAINT_EFFECT)
  else:
    print("diagnostic on vm '%s' is 'passed'" % node_name)
    if not reboot_required:
      un_taint_node(node_name, TAINT_KEY)


def remove_label(node_name: str, label: str) -> None:
  print("removing label %s from node %s" % (label, node_name))
  checker_common.run_command(K_REMOVE_LABEL_FORMAT % (node_name, label))


def taint_node(node_name: str, key: str, value: str, effect: str) -> None:
  print("adding taint %s=%s to node %s" % (key, value, node_name))
  if os.environ.get("DRY_RUN") != "true":
    checker_common.run_command(
        K_TAINT_NODE_FORMAT % (node_name, key, value, effect)
    )


def un_taint_node(node_name: str, key: str) -> None:
  print("removing taint %s from node %s" % (key, node_name))
  checker_common.run_command(K_UNTAINT_NODE_FORMAT % (node_name, key))


def is_bad_node(diag_output: str) -> bool:
  return "error" in diag_output.lower()


if __name__ == "__main__":
  main()
