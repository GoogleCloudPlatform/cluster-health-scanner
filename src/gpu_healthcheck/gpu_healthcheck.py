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
    - STRICT_MODE: If "true", the test will fail on any error. If "false", the
    test will only fail on critical errors. Defaults to "true".

Note:
    Make sure you have the necessary permissions to apply taints to nodes in the
    cluster.
"""

import json
import os
import time

import checker_common
import dcgm_pb2

_RESULT_LABEL_KEY = "aiinfra/gpu-healthcheck-result"
TAINT_KEY = "aiinfra/gpu-healthcheck"
TAINT_VALUE = "failed"
TAINT_EFFECT = "NoSchedule"
REBOOT_REQUIRED_LABEL_KEY = "aiinfra/reboot-required"
HEALTHCHECK_TIME_LABEL_KEY = "aiinfra/gpu-healthcheck-runtime-sec"
# for r level see:
# https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/dcgm-diagnostics.html
R_LEVEL = os.environ.get("R_LEVEL") or "1"
DCGM_COMMAND = "dcgmi diag -g 0 -r %s -j --verbose" % R_LEVEL
NVIDIA_SMI_COMMAND = (
    "/usr/local/nvidia/bin/nvidia-smi"
    " --query-gpu=ecc.errors.uncorrected.volatile.total"
    " --format=csv,noheader,nounits"
)
K_ADD_LABEL_FORMAT = "/app/kubectl label node %s %s=%s --overwrite"
K_REMOVE_LABEL_FORMAT = "/app/kubectl label node %s %s-"
K_TAINT_NODE_FORMAT = "/app/kubectl taint node %s %s=%s:%s"
K_UNTAINT_NODE_FORMAT = "/app/kubectl taint node %s %s-"

# https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/dcgm-diagnostics.html
K_DCGM_ERROR_ISOLATE = 2
K_DCGM_ERROR_RESET = 3


def main() -> None:
  """entry point."""
  node_name = os.environ.get("NODE_NAME")

  print("diagnostic on vm '%s' is 'initiated'" % node_name)

  reboot_required = run_reboot_required_check(node_name)
  run_dcgm_diag(node_name, reboot_required)
  checker_common.add_label(
      node_name,
      HEALTHCHECK_TIME_LABEL_KEY,
      f"{int(time.time())}",
      K_ADD_LABEL_FORMAT,
  )


def convert_output_to_proto(output: str) -> dcgm_pb2.DiagnosticReport:
  """Converts the output of the DCGM diagnostic tool to a proto message."""
  json_data = json.loads(output)
  report = dcgm_pb2.DiagnosticReport()
  report.version = json_data.get("version", "")
  report.driver_version_detected = json_data.get("Driver Version Detected", "")
  report.gpu_device_ids.extend(json_data.get("GPU Device IDs", []))

  for gpu_index, serial in json_data.get("GPU Device Serials", {}).items():
    report.gpu_device_serials[gpu_index] = serial

  for category_data in json_data.get("DCGM GPU Diagnostic", {}).get(
      "test_categories", []
  ):
    category = report.dcgm_gpu_diagnostic.test_categories.add()
    category.category = category_data["category"]

    for test_data in category_data.get("tests", []):
      test = category.tests.add()
      test.name = test_data["name"]
      for result_data in test_data.get("results", []):
        result = test.results.add()
        result.status = result_data["status"]

        if "gpu_id" in result_data:
          result.gpu_id = result_data["gpu_id"]

        if "info" in result_data:
          result.info = result_data["info"]
        if "warnings" in result_data:
          for warning_data in result_data["warnings"]:
            warning = result.warnings.add()
            warning.error_category = warning_data["error_category"]
            warning.error_id = warning_data["error_id"]
            warning.error_severity = warning_data["error_severity"]
            warning.warning = warning_data["warning"]

  return report


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

  try:
    print("Converting from json output to proto")
    output = convert_output_to_proto(diag_output.stdout)
    print(output)
    failed = is_bad_node_from_proto(output)
  except json.JSONDecodeError as e:
    print("Error deserializing JSON: %s", e)
    failed = True

  checker_common.log_results(
      test_name="dcgm",
      passed=not failed,
      node_name=node_name,
      workflow_id=os.environ.get("WORKFLOW_ID"),  # Remove workflow id
  )
  checker_common.add_label(
      node_name,
      _RESULT_LABEL_KEY,
      "fail" if failed else "pass",
      K_ADD_LABEL_FORMAT,
  )
  if failed:
    print(f"Node {node_name} failed dcgm test")
    taint_node(node_name, TAINT_KEY, TAINT_VALUE, TAINT_EFFECT)
  else:
    print(f"Node {node_name} passed dcgm test")
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


def is_bad_node_from_proto(report: dcgm_pb2.DiagnosticReport) -> bool:
  """Returns True if the node is bad based on the DCGM diagnostic report."""
  bad_node = False
  strict_mode = os.environ.get("STRICT_MODE", "true")
  for category in report.dcgm_gpu_diagnostic.test_categories:
    for test in category.tests:
      for result in test.results:
        if result.status.lower() == "fail":
          gpu_info = (
              f" (GPU {result.gpu_id})" if result.gpu_id else ""
          )  # Add GPU ID if available
          print(
              f"Test '{test.name}' in category '{category.category}'"
              f" {gpu_info}: {result.status} - {result.info}",
          )
          # If in strict mode, fail on any error.
          if strict_mode == "true":
            bad_node = True
            continue

          # If not in strict mode check the error severity and only fail on
          # critical errors.
          for warning in result.warnings:
            if warning.error_severity in [
                K_DCGM_ERROR_ISOLATE,
                K_DCGM_ERROR_RESET,
            ]:
              bad_node = True
  return bad_node


if __name__ == "__main__":
  main()
