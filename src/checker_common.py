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

"""Common functions shared between health checkers."""

from collections.abc import Callable, Iterable
import dataclasses
import enum
import json
import logging
import os
import string
import subprocess
import tempfile
import time
from typing import Any
import uuid

from google.cloud import storage
from google.protobuf import json_format
from kubernetes import client
from kubernetes import config
from kubernetes.client.api import batch_v1_api
from urllib3 import exceptions

import common_pb2
import health_results_pb2
import health_runner_config_pb2

ConnectTimeoutError = exceptions.ConnectTimeoutError
_KUBECTL = os.environ.get("KUBECTL_PATH", "/app/kubectl")
HELM = os.environ.get("HELM_PATH", "/usr/local/bin/helm")

K_APPLY_FORMAT = "%s apply -f %s"
K_DELETE_FORMAT = "%s delete -f %s"

K_GSUTIL_COPY_FILE_FORMAT = "gsutil cp %s %s"

K_32GIB_MESSAGE_SIZE = 32 * 1024 * 1024 * 1024
K_16GIB_MESSAGE_SIZE = 16 * 1024 * 1024 * 1024
K_8GIB_MESSAGE_SIZE = 8 * 1024 * 1024 * 1024
K_1GIB_MESSAGE_SIZE = 1024 * 1024 * 1024
K_64MIB_MESSAGE_SIZE = 64 * 1024 * 1024
K_4MIB_MESSAGE_SIZE = 4 * 1024 * 1024

K_SUPPORT_MESSAGE_SIZES = [
    K_32GIB_MESSAGE_SIZE,
    K_16GIB_MESSAGE_SIZE,
    K_8GIB_MESSAGE_SIZE,
    K_1GIB_MESSAGE_SIZE,
    K_64MIB_MESSAGE_SIZE,
    K_4MIB_MESSAGE_SIZE,
]

K_MESSAGE_SIZE_TO_BANDWIDTH_LABEL = {
    K_32GIB_MESSAGE_SIZE: "aiinfra/nccl-healthcheck-32G-bandwidth",
    K_16GIB_MESSAGE_SIZE: "aiinfra/nccl-healthcheck-16G-bandwidth",
    K_8GIB_MESSAGE_SIZE: "aiinfra/nccl-healthcheck-8G-bandwidth",
    K_1GIB_MESSAGE_SIZE: "aiinfra/nccl-healthcheck-1G-bandwidth",
    K_64MIB_MESSAGE_SIZE: "aiinfra/nccl-healthcheck-64MiB-bandwidth",
    K_4MIB_MESSAGE_SIZE: "aiinfra/nccl-healthcheck-4MiB-bandwidth",
}

K_MESSAGE_SIZE_TO_LATENCY_LABEL = {
    K_8GIB_MESSAGE_SIZE: "aiinfra/nccl-healthcheck-8G-latency-ms",
    K_1GIB_MESSAGE_SIZE: "aiinfra/nccl-healthcheck-1G-latency-ms",
    K_64MIB_MESSAGE_SIZE: "aiinfra/nccl-healthcheck-64MiB-latency-ms",
    K_4MIB_MESSAGE_SIZE: "aiinfra/nccl-healthcheck-4MiB-latency-ms",
}


class HelmCommand(enum.Enum):
  INSTALL: str = "install"
  UNINSTALL: str = "uninstall"


def log_results(
    test_name: str,
    passed: bool,
    node_name: str,
    workflow_id: str | None = "",
    result_data: dict[str, Any] | None = None,
) -> None:
  """Logs the results of a test run."""
  log = {
      "name": "chs-test-result",
      "workflow_id": workflow_id,
      "test_name": test_name,
      "passed": passed,
      "node_name": node_name,
      "result_data": result_data,
  }
  print(json.dumps(log))


def remove_label(
    node: str,
    label_key: str,
    kubectl_path: str = _KUBECTL,
) -> None:
  """Removes a label from a node."""
  print(f"removing label {label_key} from node {node}")
  run_command(
      f"{kubectl_path} label node {node} {label_key}-",
      print_output=False,
  )


# 
def label_node(
    node: str,
    label_key: str,
    label_value: str,
):
  """Adds a label to a node.

  Args:
    node: The node to add the label to.
    label_key: The key of the label.
    label_value: The value of the label.
  """
  add_label(
      node,
      label_key,
      label_value,
      f"{_KUBECTL} label node %s %s=%s --overwrite",
  )


def add_label(
    node_name: str,
    label: str,
    value: str,
    label_format: str,
    print_output: bool = True,
) -> None:
  """Adds a label to a node.

  Args:
    node_name (str): Name of the node.
    label (str): label being set.
    value (str): value being set.
    label_format (str): Interpolation format accepting node_name, label, value.
    print_output (bool): If True, prints the output of the command. Defaults to
      True.

  Returns:
    None.
  """
  if print_output:
    print("adding label %s=%s to node %s" % (label, value, node_name))
  run_command(
      label_format % (node_name, label, value), print_output=print_output
  )


def run_command_with_retry(
    command: str,
    print_output: bool = True,
    retry_attempts: int = 3,
    retry_interval_seconds: int = 5,
) -> subprocess.CompletedProcess[str]:
  """Execute a shell command with retries."""
  print(f"running: {command}")
  attempt = 0
  while True:
    try:
      return run_command(command, print_output=print_output, check=True)
    except subprocess.CalledProcessError as e:
      print("command failed: %s" % e)
      if attempt >= retry_attempts:
        # If we've reached the retry limit, raise the error
        raise e
      print("retrying in %s seconds" % retry_interval_seconds)
      time.sleep(retry_interval_seconds)
      attempt += 1


def run_command(
    command: str,
    check: bool = False,
    print_output: bool = True,
) -> subprocess.CompletedProcess[str]:
  """Execute a shell command using subprocess.

  Args:
    command (str): The shell command to be executed.
    check (bool, optional): If True, raises CalledProcessError if the command
      returns a non-zero exit status. Defaults to True.
    print_output (bool, optional): If True, prints the output of the command.
      Defaults to True.

  Returns:
    subprocess.CompletedProcess: The result object containing information about
    the completed process.
  """

  print("running: %s" % command)
  start_time = time.time()
  diag = subprocess.run(
      command,
      shell=True,
      text=True,
      check=check,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
  )
  if print_output:
    print(
        "took: %s seconds\nout: %s, err: %s"
        % (time.time() - start_time, diag.stdout, diag.stderr)
    )

  return diag


def get_created_jobs(release_names: Iterable[str]) -> Iterable[str]:
  """Get jobs created by a given helm release.

  Args:
    release_names: Iterable of helm release names to get jobs for.

  Returns:
    Iterable of job names created by the helm releases.
  """
  config.load_incluster_config()
  batch_v1 = batch_v1_api.BatchV1Api()
  try:
    jobs = batch_v1.list_namespaced_job(namespace="default").items

    release_name_annotation_key = "meta.helm.sh/release-name"
    matching_jobs = [
        job.metadata.name
        for job in jobs
        if job.metadata.annotations
        and release_name_annotation_key in job.metadata.annotations
        and job.metadata.annotations[release_name_annotation_key]
        in release_names
    ]
    return matching_jobs
  except client.ApiException as e:
    print(f"Error getting Jobs: {e}")
    return []


def job_succeeded(
    job_v1: batch_v1_api.BatchV1Api,
    job_name: str,
    namespace: str = "default",
) -> bool:
  """Get the status of a job."""
  attempt = 1
  while attempt <= 3:
    try:
      job = job_v1.read_namespaced_job(name=job_name, namespace=namespace)
      return job.status.succeeded is not None and job.status.succeeded >= 1
    except client.ApiException as e:
      print(f"Unknown error when attempting to list jobs: {e}")
      if attempt >= 3:
        raise e
    print("Sleeping for 1 seconds before retrying")
    time.sleep(1)
    attempt += 1
  return False


def wait_till_jobs_complete(
    job_v1: batch_v1_api.BatchV1Api,
    jobs_to_monitor: Iterable[str],
    namespace: str = "default",
    timeout_seconds: int = 900,
    check_interval: int = 30,
) -> list[str]:
  """Waits for a list of jobs to complete.

  Args:
    job_v1: BatchV1Api object
    jobs_to_monitor: List of job names to monitor.
    namespace: Namespace of the jobs.
    timeout_seconds: Timeout in seconds.
    check_interval: Interval in seconds to check for the jobs.

  Returns:
    list[str]: Any non-completed jobs.
  """
  remaining_jobs = set(jobs_to_monitor)
  start_time = time.time()

  # Poll list jobs API until all jobs are completed or the timeout is reached.
  print(
      f"Polling jobs for {timeout_seconds} seconds and checking every"
      f" {check_interval} seconds"
  )
  while remaining_jobs:
    job_list = []
    try:
      job_list = job_v1.list_namespaced_job(namespace)
    except exceptions.ConnectTimeoutError as e:
      print(f"Connect Timeout Error when attempting to list jobs: {e}")
    except exceptions.MaxRetryError as e:
      print(f"Max retry error when attempting to list jobs: {e}")
    except client.ApiException as e:
      print(f"Unknown error when attempting to list jobs: {e}")

    if job_list:
      for job in job_list.items:
        if job.metadata.name in remaining_jobs:
          if job.status.succeeded is not None and job.status.succeeded >= 1:
            remaining_jobs.remove(job.metadata.name)
            print(f"Job {job.metadata.name} completed successfully.")
          elif job.status.failed is not None and job.status.failed >= 1:
            remaining_jobs.remove(job.metadata.name)
            print(f"Job {job.metadata.name} failed.")

    if not remaining_jobs:
      print("All jobs completed.")
      break

    elapsed_time = time.time() - start_time
    if elapsed_time > timeout_seconds:
      print(f"Timeout ({timeout_seconds} seconds) reached.")
      break

    print(
        f"{int(time.time() - start_time)} secs out of {timeout_seconds} secs",
    )
    print("Remaining jobs: ", remaining_jobs)
    print(f"Sleeping for {check_interval} seconds")
    time.sleep(check_interval)

  return list(remaining_jobs)


@dataclasses.dataclass()
class HelmConfig:
  """Helm configuration for NCCL health check.

  Attributes:
    chart: The name of the Helm chart.
    chart_version: The version of the Helm chart.
    install_flags: The flags to pass to the Helm install command.
    release_name: The name of the Helm release (overrides release_name_base).
    release_name_base: The base name of the Helm release.
  """

  chart: str
  chart_version: str | None = None
  install_flags: str | None = None
  release_name: str | None = None
  release_name_base: str | None = None

  # Runs after init to ensure either a release name or base is specified
  def __post_init__(self):
    # Fine if both are defined
    if self.release_name is None and self.release_name_base is None:
      raise ValueError(
          "Either 'release_name_base' or 'release_name' must be specified."
      )


def create_job_k8s_helm(
    helm_config: HelmConfig,
    env_mappings: dict[str, str] | None = None,
    helm_bin_path: str = HELM,
) -> list[Callable[[], subprocess.CompletedProcess[str]]]:
  """Creates a k8s helm release and returns a function to uninstall it.

  Args:
    helm_config: Helm configuration for the release.
    env_mappings: Environment variables to pass to the helm chart.
    helm_bin_path: Path to the helm binary.

  Returns:
    List of functions to uninstall the helm release.
  """
  if not env_mappings:
    env_mappings = {}

  # Convert the mappings to helm format
  values = {}
  if env_mappings:
    for k, v in env_mappings.items():
      # Assuming the env_mappings are all for the health_check job
      values[f"health_check.env.{k}"] = v

  # If release name is not specified, use the base name after making it unique
  release_name = helm_config.release_name
  if release_name is None:
    ts = time.time()
    identifier = str(uuid.uuid4())[:8]
    release_name = f"{helm_config.release_name_base}-{identifier}-{ts}"

  uninstall_functions = create_helm_release(
      helm_path=helm_bin_path,
      release_name=release_name,
      chart=helm_config.chart,
      values=values,
      chart_version=helm_config.chart_version,
      helm_install_flags=helm_config.install_flags,
      retry=True,
  )
  return uninstall_functions


def create_job_k8s(
    job_name: str,
    yaml_file: str,
    env_mappings: dict[str, str] | None = None,
    print_output: bool = True,
) -> list[Callable[[], subprocess.CompletedProcess[str]]]:
  """Creates a job k8s and returns a function to delete it."""
  if not env_mappings:
    env_mappings = {}

  env_mappings["JOB_NAME"] = job_name
  return create_k8s_objects(
      yaml_path=yaml_file,
      mappings=env_mappings,
      print_output=print_output,
      retry=True,
  )


def run_healthcheck(
    health_check: health_runner_config_pb2.HealthCheck,
    env_mappings: dict[str, str],
    print_output: bool = False,
) -> str:
  """Runs a health check."""
  job_name = f"diag-healthcheck-{str(uuid.uuid4())[:8]}"
  if health_check.yaml_file:
    create_job_k8s(
        job_name=job_name,
        yaml_file=health_check.yaml_file,
        env_mappings=env_mappings,
        print_output=print_output,
    )
  elif health_check.helm_config:
    env_mappings["JOB_NAME"] = job_name
    create_helm_release(
        helm_path=HELM,
        release_name=job_name,
        chart=health_check.helm_config.chart,
        values=env_mappings,
        chart_version=health_check.helm_config.chart_version,
        helm_install_flags=health_check.helm_config.install_flags,
        retry=True,
    )
  else:
    raise ValueError("No health check file specified.")
  return job_name


def generate_helm_command(
    helm_path: str,
    release_name: str,
    chart: str | None = None,
    values: dict[str, str] | None = None,
    chart_version: str | None = None,
    helm_install_flags: str | None = None,
    helm_command_type: HelmCommand = HelmCommand.INSTALL,
) -> str:
  """Generates a helm command."""
  command = f"{helm_path}"
  if helm_command_type == HelmCommand.UNINSTALL:
    command = f"{command} uninstall {release_name}"
  else:  # Default to `helm install` if not specified
    command = f"{command} install {release_name} {chart}"
    # Will default to latest version if not set
    # 
    if chart_version:
      command = f"{command} --version {chart_version}"
    # Allows for custom values to be set in release
    # 
    if values is not None:
      for k, v in values.items():
        command = f"{command} --set {k}={v}"
    # 
    if helm_install_flags:
      command = f"{command} {helm_install_flags}"
  return command


def create_helm_release(
    helm_path: str,
    release_name: str,
    chart: str,
    values: dict[str, str] | None = None,
    chart_version: str | None = None,
    helm_install_flags: str | None = None,
    retry: bool = False,
) -> list[Callable[[], subprocess.CompletedProcess[str]]]:
  """Creates a helm release and returns a function to uninstall it."""

  cleanup_functions = []

  cleanup_functions.append(
      install_helm_release(
          helm_path=helm_path,
          release_name=release_name,
          chart=chart,
          values=values,
          chart_version=chart_version,
          helm_install_flags=helm_install_flags,
          retry=retry,
      )
  )
  return cleanup_functions


def install_helm_release(
    helm_path: str,
    release_name: str,
    chart: str,
    values: dict[str, str] | None = None,
    chart_version: str | None = None,
    helm_install_flags: str | None = None,
    retry: bool = False,
) -> Callable[[], subprocess.CompletedProcess[str]]:
  """Applies a helm chart and returns a function to uninstall it."""

  # generate the helm command
  helm_install_command = generate_helm_command(
      helm_path=helm_path,
      release_name=release_name,
      chart=chart,
      values=values,
      chart_version=chart_version,
      helm_install_flags=helm_install_flags,
      helm_command_type=HelmCommand.INSTALL,
  )
  # Will do the specific release installation
  logging.info("Helm command:\n%s", helm_install_command)
  if retry:
    run_command_with_retry(helm_install_command)
  else:
    run_command(helm_install_command)

  # Will give a function to later uninstall the release
  helm_uninstall_command = generate_helm_command(
      helm_path=helm_path,
      release_name=release_name,
      helm_install_flags=helm_install_flags,
      helm_command_type=HelmCommand.UNINSTALL,
  )
  # 
  uninstall_helm_release = lambda: run_command(helm_uninstall_command)
  return uninstall_helm_release


def create_k8s_objects(
    yaml_path: str,
    mappings: dict[str, str] | None = None,
    print_output: bool = True,
    retry: bool = False,
) -> list[Callable[[], subprocess.CompletedProcess[str]]]:
  """Expands provided yaml file and runs `kubectl apply -f` on the contents."""
  if not _KUBECTL:
    logging.error("No kubectl path specified.")
    return []

  expanded_yaml_content = expand_template(yaml_path, mappings)
  with tempfile.NamedTemporaryFile(delete=False, mode="w") as f:
    file_name = f.name
    f.write(expanded_yaml_content)

  cleanup_functions = []
  try:
    cleanup_functions.append(
        apply_yaml_file(file_name, _KUBECTL, print_output, retry)
    )
  except subprocess.CalledProcessError as e:
    logging.exception("Failed to apply yaml file (reason: %r).", e)

  return cleanup_functions


def apply_yaml_file(
    yaml_path: str,
    kubectl_path: str,
    print_output: bool = True,
    retry: bool = False,
) -> Callable[[], subprocess.CompletedProcess[str]]:
  """Applies YAML file.

  Args:
    yaml_path (str): Relative filesystem path to the yaml file to apply.
    kubectl_path (str): Relative filesystem path to the kubectl binary.
    print_output (bool): If True, prints the output of the command. Defaults to
      True.
    retry (bool): If True, retries the command if it fails. Defaults to False.

  Returns:
    Callable((), subprocess.CompletedProcess(str)): A function that will run
    `kubectl delete -f` on the yaml_path provided for easy cleanup of temporary
    resources.
  """
  command = K_APPLY_FORMAT % (kubectl_path, yaml_path)
  if retry:
    run_command_with_retry(command, print_output)
  else:
    run_command(command, print_output=print_output)

  def delete_yaml_file():
    return run_command(K_DELETE_FORMAT % (kubectl_path, yaml_path))

  return delete_yaml_file


def delete_jobs(
    batch_v1: batch_v1_api.BatchV1Api,
    jobs: Iterable[str],
    namespace: str = "default",
) -> None:
  """Deletes a job."""
  for job_name in jobs:
    try:
      print(f"Deleting job '{job_name}' in namespace '{namespace}'...")
      batch_v1.delete_namespaced_job(
          name=job_name,
          namespace=namespace,
          propagation_policy="Background",
      )
    except client.ApiException as e:
      print(f"Error deleting job '{job_name}': {e}")


def expand_template(
    yaml_template: str,
    mappings: dict[str, str] | None,
) -> str:
  """Expands YAML template."""
  default_mappings = {
      "CHECK_TIME_EPOCH_SEC": int(time.time()),
      "DRY_RUN": os.environ.get("DRY_RUN"),
      "ORIG_CHECK_TIME_EPOCH_SEC": os.environ.get("CHECK_TIME_EPOCH_SEC"),
      "R_LEVEL": os.environ.get("R_LEVEL"),
      "IMAGE_TAG": os.environ.get("IMAGE_TAG", "latest"),
      "SHORT_GUID": os.environ.get("SHORT_GUID", str(uuid.uuid4())[:8]),
      "ITERATIONS": os.environ.get("ITERATIONS", 5),
      "WORKFLOW_ID": os.environ.get("WORKFLOW_ID"),
      "BUG_ID": os.environ.get("BUG_ID"),
      "RUNNER_NAME": os.environ.get("RUNNER_NAME"),
      "RUNNER_UID": os.environ.get("RUNNER_UID"),
      "BANDWIDTH_THRESHOLD": os.environ.get("BANDWIDTH_THRESHOLD"),
      "START_MESSAGE_SIZE": os.environ.get("START_MESSAGE_SIZE"),
      "END_MESSAGE_SIZE": os.environ.get("END_MESSAGE_SIZE"),
      "BENCHMARK": os.environ.get("BENCHMARK"),
  }
  if mappings:
    default_mappings.update(mappings)

  with open(yaml_template, "r") as f:
    t = string.Template(f.read())
    return t.safe_substitute(default_mappings)


def upload_results_to_gcs(
    bucket_name: str,
    health_results: health_results_pb2.HealthResults,
    destination_path: str | None = None,
) -> str:
  """Uploads the health results proto to a GCS bucket."""
  if not bucket_name:
    logging.error("No GCS bucket specified.")
    return ""

  # If workflow_id is set, use it as the file postfix. Otherwise, use a random
  # string.
  if os.environ.get("WORKFLOW_ID"):
    file_name = os.environ.get("WORKFLOW_ID")
  else:
    file_name = str(uuid.uuid4())[:8]
  file_name = f"health_results_{file_name}.json"

  storage_client = storage.Client()
  bucket = storage_client.bucket(bucket_name)

  # If destination_path is set use it in the file path
  if destination_path:
    file_name = destination_path + "/" + file_name

  blob = bucket.blob(file_name)
  json_string = json_format.MessageToJson(
      health_results, preserving_proto_field_name=True
  )

  try:
    blob.upload_from_string(json_string, content_type="application/json")
  except Exception as error:  # pylint: disable=broad-exception-caught
    logging.exception("Failed to upload results to GCS (reason: %r).", error)
    return ""

  result_path = f"gs://{bucket_name}/{file_name}"
  print(f"Uploaded results to {result_path}")
  return result_path


def get_capacity_topology(
    nodes: list[dict[str, str]],
) -> common_pb2.Capacity:
  """Creates a capacity topology from the list of nodes."""

  capacity = common_pb2.Capacity()
  cluster_dict = dict()
  rack_dict = dict()

  # If nodes don't have all topology labels then all nodes will be grouped under
  # a single cluster and rack with the id "unknown"
  for node_data in nodes:
    cluster_id = node_data["cluster"]
    rack_id = node_data["rack"]
    node_id = node_data["node_id"]
    host_id = node_data["host"]

    if cluster_id not in cluster_dict:
      cluster = capacity.clusters.add()
      cluster.id = cluster_id
      cluster_dict[cluster_id] = cluster
    cluster = cluster_dict[cluster_id]

    if rack_id not in rack_dict:
      rack = cluster.racks.add()
      rack.id = rack_id
      rack_dict[rack_id] = rack

    rack = rack_dict[rack_id]
    rack.nodes.append(common_pb2.Node(id=node_id, host=host_id))

  return capacity


def _get_node_data_v1(
    gpu_nodes: Iterable[client.models.V1Node],
) -> list[dict[str, str]]:
  """Returns a list of node data from the given list of nodes.

  Works on topology information provided by the v1 instance API.

  Args:
    gpu_nodes: List of GPU nodes.

  Returns:
    List of node data.
  """
  nodes = []
  for gpu_node in gpu_nodes:
    if "topology.gke.io/cluster" not in gpu_node.metadata.labels:
      continue

    node = {}
    node["cluster"] = gpu_node.metadata.labels.get(
        "topology.gke.io/cluster", "unknown"
    )
    node["rack"] = gpu_node.metadata.labels.get(
        "topology.gke.io/rack", "unknown"
    )
    node["host"] = gpu_node.metadata.labels.get(
        "topology.gke.io/host", "unknown"
    )
    node["node_id"] = gpu_node.metadata.name
    nodes.append(node)
  return nodes


def _get_node_data_v2(
    gpu_nodes: Iterable[client.models.V1Node],
) -> list[dict[str, str]]:
  """Returns a list of node data from the given list of nodes.

  Works on topology information provided by the v2 instance API.

  Args:
    gpu_nodes: List of GPU nodes.

  Returns:
    List of node data.
  """
  nodes = []
  for gpu_node in gpu_nodes:
    node = {}
    node["cluster"] = gpu_node.metadata.labels.get(
        "cloud.google.com/gce-topology-block", "unknown"
    )
    node["rack"] = gpu_node.metadata.labels.get(
        "cloud.google.com/gce-topology-subblock", "unknown"
    )
    node["host"] = gpu_node.metadata.labels.get(
        "cloud.google.com/gce-topology-host", "unknown"
    )
    node["node_id"] = gpu_node.metadata.name
    nodes.append(node)
  return nodes


def get_nodes_data(
    kube_nodes: list[client.models.V1Node],
    filter_label_name: str | None = None,
    filter_label_value: str | None = None,
) -> list[dict[str, str]]:
  """Returns a list of node data from the given list of nodes & conditions.

  Args:
    kube_nodes: List of nodes.
    filter_label_name: Name of the label to filter on.
    filter_label_value: Value of the label to filter on.

  Returns:
    List of node data.
  """
  gpu_nodes = _get_nodes_under_test(
      kube_nodes, filter_label_name, filter_label_value
  )

  # If no nodes are found then return an empty list
  if not gpu_nodes:
    return []

  # Use the first node to determine the topology version.
  if "topology.gke.io/cluster" in gpu_nodes[0].metadata.labels:
    return _get_node_data_v1(gpu_nodes)
  elif "cloud.google.com/gce-topology-host" in gpu_nodes[0].metadata.labels:
    return _get_node_data_v2(gpu_nodes)
  else:
    # If no topology labels are found then all nodes will be grouped under a
    # single cluster and rack with the id "unknown"
    print("No topology labels found.")
    return _get_node_data_v2(gpu_nodes)


def _get_nodes_under_test(
    kube_nodes: list[client.models.V1Node],
    filter_label_name: str | None = None,
    filter_label_value: str | None = None,
) -> list[client.models.V1Node]:
  """Returns a list of nodes under test."""
  # Filter set of nodes before getting data
  gpu_nodes = []
  for node in kube_nodes:
    # Must have a GPU label
    if not has_gpu_resources(node):
      continue

    # If filter label is specified, then the node must have the label and the
    # value must match
    if (filter_label_name and filter_label_value) and (
        filter_label_name not in node.metadata.labels
        or node.metadata.labels[filter_label_name] != filter_label_value
    ):
      continue

    gpu_nodes.append(node)
  return gpu_nodes


def topology_key(
    topology_level: health_runner_config_pb2.TopologyLevel,
) -> str:
  """Get the topology key for the health check."""
  if (
      topology_level
      == health_runner_config_pb2.TopologyLevel.TOPOLOGY_LEVEL_SUBBLOCK
  ):
    return "cloud.google.com/gce-topology-subblock"
  elif (
      topology_level
      == health_runner_config_pb2.TopologyLevel.TOPOLOGY_LEVEL_BLOCK
  ):
    return "cloud.google.com/gce-topology-block"
  elif (
      topology_level
      == health_runner_config_pb2.TopologyLevel.TOPOLOGY_LEVEL_CLUSTER
  ):
    return "cloud.google.com/gce-topology-cluster"

  raise ValueError(f"Unsupported topology level {topology_level}.")


def create_topology_to_nodes_mapping(
    capacity: common_pb2.Capacity,
    topology_level: health_runner_config_pb2.TopologyLevel,
) -> dict[str, list[str]]:
  """Get the topology to nodes mapping for the health check."""
  if (
      topology_level
      == health_runner_config_pb2.TopologyLevel.TOPOLOGY_LEVEL_SUBBLOCK
  ):
    return generate_subblock_topology(capacity)
  elif (
      topology_level
      == health_runner_config_pb2.TopologyLevel.TOPOLOGY_LEVEL_BLOCK
  ):
    return generate_block_topology(capacity)
  raise ValueError(f"Unsupported topology level {topology_level}.")


def generate_block_topology(
    capacity: common_pb2.Capacity,
) -> dict[str, list[str]]:
  """Generates a mapping from block (SBRG) topology to nodes in the block."""
  # TODO: Add sbrg support to the capacity proto.
  block_to_nodes = {}
  for cluster in capacity.clusters:
    for rack in cluster.racks:
      for node in rack.nodes:
        block_to_nodes.setdefault(cluster.id, []).append(node.id)
  return block_to_nodes


def generate_subblock_topology(
    capacity: common_pb2.Capacity,
) -> dict[str, list[str]]:
  """Generates a mapping from subblock (rack) topology to nodes in the subblock."""
  subblock_to_nodes = {}
  for cluster in capacity.clusters:
    for rack in cluster.racks:
      for node in rack.nodes:
        subblock_to_nodes.setdefault(rack.id, []).append(node.id)
  return subblock_to_nodes


def parse_nccl_results(
    node: client.models.V1Node,
) -> health_results_pb2.NCCLHealthResult | None:
  """Parses the NCCL results from the job output."""
  nccl_health_result = health_results_pb2.NCCLHealthResult(
      benchmark=node.metadata.labels.get("aiinfra/nccl-healthcheck-benchmark"),
  )
  average_bandwidth_gbps = node.metadata.labels.get(
      "aiinfra/nccl-healthcheck-bandwidth"
  )
  if average_bandwidth_gbps and average_bandwidth_gbps != "None":
    nccl_health_result.average_bandwidth_gbps = int(average_bandwidth_gbps)
  else:
    return None

  # TODO: Update labels units
  # Attempt to parse the bandwidths and latencies for each message size.
  for msg_size in K_SUPPORT_MESSAGE_SIZES:
    bandwidth_label = K_MESSAGE_SIZE_TO_BANDWIDTH_LABEL.get(msg_size, None)
    latency_label = K_MESSAGE_SIZE_TO_LATENCY_LABEL.get(msg_size, None)
    # If the message size is not supported, skip it.
    if (bandwidth_label is None and latency_label is None) or (
        bandwidth_label not in node.metadata.labels
        and latency_label not in node.metadata.labels
    ):
      continue

    nccl_health_result.bandwidth_measurements.append(
        health_results_pb2.NCCLHealthResult.NCCLBandwidthResult(
            bandwidth_gbps=float(node.metadata.labels.get(bandwidth_label, 0)),
            latency_ms=int(node.metadata.labels.get(latency_label, 0)),
            message_size_bytes=int(msg_size),
        )
    )

  return nccl_health_result


def has_gpu_resources(node: client.models.V1Node) -> bool:
  """Check if the node has GPU resources in capacity or allocatable.

  Args:
      node (kubernetes.client.V1Node): Kubernetes node object

  Returns:
      bool: True if node has GPUs, False otherwise
  """
  # Get the node's status
  node_status = node.status
  # Check both capacity and allocatable for GPU resources
  if node_status and node_status.allocatable:
    gpu_key = "nvidia.com/gpu"  # Standard NVIDIA GPU label

    if gpu_key in node_status.allocatable:
      # Convert to int and check if > 0
      try:
        gpu_count = int(node_status.allocatable.get(gpu_key, 0))
        if gpu_count > 0:
          return True
      except (TypeError, ValueError):
        pass

  return False


def get_node_list() -> list[str]:
  """Returns a list of the nodes names in the cluster."""
  config.load_incluster_config()
  v1 = client.CoreV1Api()
  kube_nodes = v1.list_node().items
  gpu_nodes = _get_nodes_under_test(kube_nodes)
  nodes = []
  for node in gpu_nodes:
    nodes.append(node.metadata.name)
  return nodes


def parse_env_mappings(
    health_check: health_runner_config_pb2.HealthCheck,
) -> dict[str, str]:
  """Parse the env mappings from the health check."""
  env_mappings = {}
  if not health_check.health_check_params:
    return env_mappings

  for param in health_check.health_check_params:
    env_mappings[param.name] = param.value
  print("Env mappings: ", env_mappings)
  return env_mappings


def sigterm_handler(signum: Any, frame: Any) -> None:
  """Handler for SIGTERM signal.

  Args:
    signum (Any): Signal number.
    frame (Any): Current stack frame.
  """
  print(f"Received {signum} signal on frame {frame}. Exiting...")
  exit(0)
