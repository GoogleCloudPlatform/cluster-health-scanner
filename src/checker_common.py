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
import enum
import json
import logging
import os
import string
import subprocess
import tempfile
import time
from typing import Any, Dict
import uuid

from google.cloud import storage
from google.protobuf import json_format
import kubernetes.client
from kubernetes.client.api import batch_v1_api

import common_pb2
import health_results_pb2

_KUBECTL = os.environ.get("KUBECTL_PATH", "/app/kubectl")

# If set, only the nodes that have this label set to true will be used.
_FILTER_LABEL_NAME = os.environ.get("FILTER_LABEL_NAME", "")
_FILTER_LABEL_VALUE = os.environ.get("FILTER_LABEL_VALUE", "true")

# Helm Config
_HELM = os.environ.get("HELM_PATH", "/usr/local/bin/helm")
_HELM_CHART = os.environ.get("HELM_CHART")
_HELM_CHART_VERSION = os.environ.get("HELM_CHART_VERSION")
_HELM_INSTALL_FLAGS = os.environ.get("HELM_INSTALL_FLAGS")
_HELM_RELEASE_NAME = os.environ.get("HELM_RELEASE_NAME")


K_APPLY_FORMAT = "%s apply -f %s"
K_DELETE_FORMAT = "%s delete -f %s"


class HelmCommand(enum.Enum):
  INSTALL: str = "install"
  UNINSTALL: str = "uninstall"


def log_results(
    test_name: str,
    passed: bool,
    node_name: str,
    workflow_id: str | None = "",
    result_data: Dict[str, Any] | None = None,
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


# 
def cleanup_labels(
    label: str,
) -> None:
  """Removes any potential labels from previous runs."""
  logging.info("Removing label: %s", label)
  run_command(f"{_KUBECTL} label nodes -l {label} {label}-")


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
    node_name: str, label: str, value: str, label_format: str
) -> None:
  """Adds a label to a node.

  Args:
    node_name (str): Name of the node.
    label (str): label being set.
    value (str): value being set.
    label_format (str): Interpolation format accepting node_name, label, value.

  Returns:
    None.
  """
  print("adding label %s=%s to node %s" % (label, value, node_name))
  run_command(label_format % (node_name, label, value))


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
  while remaining_jobs:
    job_list = job_v1.list_namespaced_job(namespace)
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

    logging.info(
        "(%.2f sec) out of (%d sec)",
        int(time.time() - start_time),
        timeout_seconds,
    )
    logging.info("Remaining jobs: %s", remaining_jobs)
    logging.info("Sleeping for %d seconds", check_interval)
    time.sleep(check_interval)

  return list(remaining_jobs)


def create_job_k8s(
    job_name: str,
    env_mappings: dict[str, str] | None = None,
) -> list[Callable[[], subprocess.CompletedProcess[str]]]:
  """Creates a job k8s and returns a function to delete it."""
  if not env_mappings:
    env_mappings = {}

  if _HELM_CHART:
    # Convert the mappings to helm format
    values = {}
    if env_mappings:
      for k, v in env_mappings.items():
        # Assuming the env_mappings are all for the health_check job
        values[f"health_check.env.{k}"] = v
    values["job.name"] = job_name

    # Use the helm release name if specified, otherwise generate a default one
    if _HELM_RELEASE_NAME:
      release_name = _HELM_RELEASE_NAME
    else:
      release_name = f"internal-{job_name}"

    return create_helm_release(
        helm_path=_HELM,
        release_name=release_name,
        chart=_HELM_CHART,
        values=values,
        chart_version=_HELM_CHART_VERSION,
        helm_install_flags=_HELM_INSTALL_FLAGS,
    )
  elif os.environ.get("YAML_FILE"):
    env_mappings["JOB_NAME"] = job_name
    return create_k8s_objects(
        yaml_path=os.environ.get("YAML_FILE"),
        mappings=env_mappings,
    )

  logging.error("No k8s object or helm release specified.")
  return []


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
    if chart_version is not None:
      command = f"{command} --version {chart_version}"
    # Allows for custom values to be set in release
    # 
    if values is not None:
      for k, v in values.items():
        command = f"{command} --set {k}={v}"
    # 
    if helm_install_flags is not None:
      command = f"{command} {helm_install_flags}"
  return command


def create_helm_release(
    helm_path: str,
    release_name: str,
    chart: str,
    values: dict[str, str] | None = None,
    chart_version: str | None = None,
    helm_install_flags: str | None = None,
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
) -> list[Callable[[], subprocess.CompletedProcess[str]]]:
  """Expands provided yaml file and runs `kubectl apply -f` on the contents."""
  if not _KUBECTL:
    logging.error("No kubectl path specified.")
    return []

  cleanup_functions = []

  expanded_yaml_content = expand_template(yaml_path, mappings)
  with tempfile.NamedTemporaryFile(delete=False, mode="w") as f:
    file_name = f.name
    f.write(expanded_yaml_content)

  cleanup_functions.append(apply_yaml_file(file_name, _KUBECTL, print_output))
  return cleanup_functions


def apply_yaml_file(
    yaml_path: str,
    kubectl_path: str,
    print_output: bool = True,
) -> Callable[[], subprocess.CompletedProcess[str]]:
  """Applies YAML file.

  Args:
    yaml_path (str): Relative filesystem path to the yaml file to apply.
    kubectl_path (str): Relative filesystem path to the kubectl binary.
    print_output (bool): If True, prints the output of the command. Defaults to
      True.

  Returns:
    Callable((), subprocess.CompletedProcess(str)): A function that will run
    `kubectl delete -f` on the yaml_path provided for easy cleanup of temporary
    resources.
  """
  run_command(
      K_APPLY_FORMAT % (kubectl_path, yaml_path), print_output=print_output
  )

  def delete_yaml_file():
    return run_command(K_DELETE_FORMAT % (kubectl_path, yaml_path))

  return delete_yaml_file


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
      "SHORT_GUID": os.environ.get("SHORT_GUID", str(uuid.uuid4())[:4]),
      "ITERATIONS": os.environ.get("ITERATIONS", 5),
      "WORKFLOW_ID": os.environ.get("WORKFLOW_ID"),
      "BUG_ID": os.environ.get("BUG_ID"),
      "RUNNER_NAME": os.environ.get("RUNNER_NAME"),
      "RUNNER_UID": os.environ.get("RUNNER_UID"),
  }
  if mappings:
    default_mappings.update(mappings)

  with open(yaml_template, "r") as f:
    t = string.Template(f.read())
    return t.safe_substitute(default_mappings)


def upload_results_to_gcs(
    bucket_name: str,
    health_results: health_results_pb2.HealthResults,
) -> None:
  """Uploads the health results proto to a GCS bucket."""
  if not bucket_name:
    logging.error("No GCS bucket specified.")
    return

  # If workflow_id is set, use it as the file postfix. Otherwise, use a random
  # string.
  if os.environ.get("WORKFLOW_ID"):
    file_postfix = os.environ.get("WORKFLOW_ID")
  else:
    file_postfix = str(uuid.uuid4())[:8]

  file_name = f"health_results_{file_postfix}.json"

  storage_client = storage.Client()
  bucket = storage_client.bucket(bucket_name)
  blob = bucket.blob(file_name)

  json_string = json_format.MessageToJson(
      health_results, preserving_proto_field_name=True
  )

  try:
    blob.upload_from_string(json_string, content_type="application/json")
  except Exception as error:  # pylint: disable=broad-exception-caught
    logging.exception("Failed to upload results to GCS (reason: %r).", error)
    return

  print(f"Uploaded results to gs://{bucket_name}/{file_name}")


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
    gpu_nodes: Iterable[kubernetes.client.models.V1Node],
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
    gpu_nodes: Iterable[kubernetes.client.models.V1Node],
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
    kube_nodes: list[kubernetes.client.models.V1Node],
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
    kube_nodes: list[kubernetes.client.models.V1Node],
    filter_label_name: str | None = None,
    filter_label_value: str | None = None,
) -> list[kubernetes.client.models.V1Node]:
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


def has_gpu_resources(node: kubernetes.client.models.V1Node) -> bool:
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
  kubernetes.config.load_incluster_config()
  v1 = kubernetes.client.CoreV1Api()
  kube_nodes = v1.list_node().items
  gpu_nodes = _get_nodes_under_test(kube_nodes)
  nodes = []
  for node in gpu_nodes:
    nodes.append(node.metadata.name)
  return nodes


def sigterm_handler(signum: Any, frame: Any) -> None:
  """Handler for SIGTERM signal.

  Args:
    signum (Any): Signal number.
    frame (Any): Current stack frame.
  """
  print(f"Received {signum} signal on frame {frame}. Exiting...")
  exit(0)
