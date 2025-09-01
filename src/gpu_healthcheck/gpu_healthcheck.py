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

import dataclasses
import glob
import json
import os
import subprocess
import time

import google.cloud.exceptions
import google.cloud.storage
import requests

import checker_common
import dcgm_pb2
import mft_installer


_RESULT_LABEL_KEY = "aiinfra/gpu-healthcheck-result"
TAINT_KEY = "aiinfra/gpu-healthcheck"
TAINT_VALUE = "failed"
TAINT_EFFECT = "NoSchedule"
REBOOT_REQUIRED_LABEL_KEY = "aiinfra/reboot-required"
HEALTHCHECK_TIME_LABEL_KEY = "aiinfra/gpu-healthcheck-runtime-sec"
# for r level see:
# https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/dcgm-diagnostics.html
R_LEVEL = os.environ.get("R_LEVEL")
DCGM_COMMAND = "dcgmi diag -g 0 -r %s -j --verbose"
NVIDIA_SMI_COMMAND = (
    "/usr/local/nvidia/bin/nvidia-smi"
    " --query-gpu=ecc.errors.uncorrected.volatile.total"
    " --format=csv,noheader,nounits"
)
K_ENABLE_PERSISTENCE_MODE = "/usr/local/nvidia/bin/nvidia-smi -pm 1"
K_ADD_LABEL_FORMAT = "/app/kubectl label node %s %s=%s --overwrite"
K_REMOVE_LABEL_FORMAT = "/app/kubectl label node %s %s-"
K_TAINT_NODE_FORMAT = "/app/kubectl taint node %s %s=%s:%s"
NVIDIA_BUG_REPORT_SCRIPT = "/usr/local/nvidia/bin/nvidia-bug-report.sh"
K_UNTAINT_NODE_FORMAT = "/app/kubectl taint node %s %s-"

# https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/dcgm-diagnostics.html
K_DCGM_ERROR_ISOLATE = 2
K_DCGM_ERROR_RESET = 3


def main() -> None:
  """entry point."""
  node_name = os.environ.get("NODE_NAME")

  print("diagnostic on vm '%s' is 'initiated'" % node_name)

  reboot_required = run_reboot_required_check(node_name)
  enable_persistence_mode(node_name)
  run_dcgm_diag(node_name, reboot_required)
  checker_common.add_label(
      node_name,
      HEALTHCHECK_TIME_LABEL_KEY,
      f"{int(time.time())}",
      K_ADD_LABEL_FORMAT,
  )


def enable_persistence_mode(node_name: str) -> None:
  """Enables persistence mode for the node."""
  print(f"enabling persistence mode for node '{node_name}'")
  checker_common.run_command(K_ENABLE_PERSISTENCE_MODE)


def convert_output_to_proto(output: str) -> dcgm_pb2.DiagnosticReport:
  """Converts the output of the DCGM diagnostic tool to a proto message."""
  json_data = json.loads(output)
  report = dcgm_pb2.DiagnosticReport()
  report.version = json_data.get("version", "")
  report.driver_version_detected = json_data.get("Driver Version Detected", "")
  report.gpu_device_ids.extend(json_data.get("GPU Device IDs", []))

  for gpu_index, serial in json_data.get("GPU Device Serials", {}).items():
    report.gpu_device_serials[gpu_index] = serial

  for category_data in json_data.get("DCGM Diagnostic", {}).get(
      "test_categories", []
  ):
    category = report.dcgm_gpu_diagnostic.test_categories.add()
    category.category = category_data["category"]
    print(f"Processing category: {category.category}")

    for test_data in category_data.get("tests", []):
      test = category.tests.add()
      test.name = test_data["name"]
      for result_data in test_data.get("results", []):
        result = test.results.add()
        result.status = result_data["status"]

        if "gpu_id" in result_data:
          result.gpu_id = result_data["gpu_id"]

        if "info" in result_data:
          info = result_data["info"]
          if isinstance(info, str):
            result.infos.append(info)
          elif isinstance(info, list):
            result.infos.extend(info)
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


def generate_dcgm_command() -> str:
  """Generates the command to run DCGM diagnostic tool."""
  r_level = os.environ.get("R_LEVEL", "1")
  command = DCGM_COMMAND % r_level
  dcgm_params = os.environ.get("DCGM_PARAMS")
  if dcgm_params is not None and dcgm_params:
    command += " -p " + dcgm_params
  return command


@dataclasses.dataclass
class Artifact:
  """A simple class to hold information about a file to be uploaded."""
  filepath: str
  content_type: str = "application/gzip"


def upload_report_to_gcs(
    artifact: Artifact, bucket_name: str, node_name: str
) -> None:
  """Uploads a file artifact to GCS using the native Python client library."""
  try:
    # 1. Create a client, which handles authentication automatically.
    storage_client = google.cloud.storage.Client()

    # 2. Get a reference to the GCS bucket.
    bucket = storage_client.bucket(bucket_name)

    # 3. Construct the full destination path for the object.
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S_UTC", time.gmtime())
    destination_path = (
        f"bug_reports/{node_name}_{timestamp}/"
        f"{os.path.basename(artifact.filepath)}"
    )

    print(
        f"Uploading {artifact.filepath} to"
        f" gs://{bucket_name}/{destination_path}"
    )

    # 4. Get a blob reference and upload the file from disk.
    blob = bucket.blob(destination_path)
    blob.upload_from_filename(
        artifact.filepath, content_type=artifact.content_type
    )

    print("Successfully uploaded bug report.")

  except google.cloud.exceptions.GoogleCloudError as e:
    print(f"A Google Cloud Storage error occurred during upload: {e}")


def generate_nvidia_bug_report(node_name: str) -> None:
  """Generates NVIDIA bug report using nvidia-bug-report.sh script.

  Args:
      node_name: The name of the node where the bug report is generated.
  """
  print(f"Generating NVIDIA bug report for node: {node_name}")

  bug_report_output_path = os.environ.get("BUG_REPORT_OUTPUT_PATH", "/tmp")

  report_path = None
  try:
    result = subprocess.run(
        NVIDIA_BUG_REPORT_SCRIPT,
        capture_output=True,
        text=True,
        check=True,
        cwd=bug_report_output_path
    )
    report_files = glob.glob(
        os.path.join(bug_report_output_path, "nvidia-bug-report.log.gz")
    )
    if not report_files:
      print("Could not find the generated bug report file.")
      return  # Exit if the file wasn't created

    report_path = report_files[
        0
    ]  # Get the full path, e.g., /tmp/nvidia-bug-report.log.gz
    print(f"Bug report generated at: {report_path}")

    print(f"NVIDIA bug report generated successfully:\n {result.stdout}")
    print(f"Bug report generated in: {bug_report_output_path}")
  except subprocess.CalledProcessError as e:
    print(f"Error generating NVIDIA bug report: {e}")
    print(f"stderr: {e.stderr}")
  except FileNotFoundError as e:
    print(f"Error generating NVIDIA bug report: {e}")
  except OSError as e:
    print(f"An unexpected error occurred: {e}")

  gcs_bucket_name = os.environ.get("GCS_BUCKET_NAME")
  if gcs_bucket_name:
    print(
        "Found GCS_BUCKET_NAME environment variable. Uploading report..."
    )
    # Use the correct 'report_path' variable which contains the full file path.
    if report_path:
      bug_report_artifact = Artifact(filepath=report_path)
      upload_report_to_gcs(bug_report_artifact, gcs_bucket_name, node_name)
    else:
      print("Skipping GCS upload because no bug report file was found.")
  else:
    print("No GCS_BUCKET_NAME environment variable found.")


def run_dcgm_diag(node_name: str, reboot_required: bool) -> None:
  """run dcgm diag."""
  command = generate_dcgm_command()
  diag_output = checker_common.run_command(command)
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
    try:
      mft_installer.install_mft_if_needed(node_name)
    except (
        FileNotFoundError,
        requests.exceptions.RequestException,
        subprocess.CalledProcessError,
        OSError,
    ) as e:
      print(f"MFT installation failed, bug report may be incomplete: {e}")
    generate_nvidia_bug_report(node_name)
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
  # If no test categories, then something went wrong with the test.
  if not report.dcgm_gpu_diagnostic.test_categories:
    print("No test categories in DCGM diagnostic report")
    return True

  for category in report.dcgm_gpu_diagnostic.test_categories:
    for test in category.tests:
      for result in test.results:
        if result.status.lower() == "fail":
          gpu_info = (
              f" (GPU {result.gpu_id})" if result.gpu_id else ""
          )  # Add GPU ID if available
          print(
              f"Test '{test.name}' in category '{category.category}'"
              f" {gpu_info}: {result.status}",
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
