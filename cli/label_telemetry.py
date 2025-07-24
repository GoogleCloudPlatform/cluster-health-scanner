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

"""Module to add labels to nodes for telemetry for the healthscan command.

This adds labels to nodes for telemetry for the healthscan result status of a
cluster.
"""

import dataclasses
import datetime
import subprocess

import click
from kubernetes import client

_CHS_LAST_SCAN_RESULT = "goog-chs-last-scan-result"
_CHS_LAST_SCAN_TIME = "goog-chs-last-scan-timestamp"


@dataclasses.dataclass
class NodeData:
  """The node zone and check results.

  Attributes:
      zone: The zone of the node.
      check_results: A dictionary of check names to their corresponding results.
  """
  zone: str
  check_results: dict[str, str]


def _get_nodes_with_labels(
    kubectl_core_api: client.CoreV1Api,
    machine_type: str,
) -> list[tuple[str, dict[str, str]]]:
  """Returns the names and labels of all nodes with the given machine type."""
  return [
      (node.metadata.name, node.metadata.labels)
      for node in kubectl_core_api.list_node(
          label_selector=f"node.kubernetes.io/instance-type={machine_type}"
      ).items
  ]


def _fetch_node_data_k8s(
    kubectl_core_api: client.CoreV1Api,
    machine_type: str,
    nodes_to_label: list[str],
    check_name_to_result_label: dict[str, str],
) -> dict[str, NodeData]:
  """Fetches node labels and zones using the Kubernetes API.

  Args:
      kubectl_core_api: The Kubernetes API client.
      machine_type: The machine type of the nodes to fetch.
      nodes_to_label: A list of node names to fetch data for.
      check_name_to_result_label: A dictionary mapping check names to their
        corresponding result labels.

  Returns:
      A dictionary of NodeData, keyed by node name.
  """
  node_data = {}
  for node_name, labels in _get_nodes_with_labels(
      kubectl_core_api, machine_type
  ):
    if node_name in nodes_to_label and (
        zone := labels.get("topology.gke.io/zone", "")
    ):

      check_results = {}
      for check_name in check_name_to_result_label:
        if label_key := check_name_to_result_label.get(check_name):
          check_results[check_name] = labels.get(label_key, "none")
      node_data[node_name] = NodeData(zone=zone, check_results=check_results)

  return node_data


def _update_nodes_with_last_scan_result(
    node_name: str, labels: dict[str, str], zone: str
) -> None:
  """Updates the labels on a GCE instance.

  Args:
    node_name: The name of the node to update.
    labels: A dictionary of labels and their values to update.
    zone: The zone of the node to update.
  """
  cmd = [
      "gcloud",
      "compute",
      "instances",
      "update",
      node_name,
      f"--zone={zone}",
      "--update-labels",
      ",".join([f"{k}={v}" for k, v in labels.items()]),
  ]

  try:
    subprocess.run(cmd, capture_output=True, text=True, check=True)
  except subprocess.CalledProcessError as e:
    click.echo(
        click.style(
            f"Subprocess failed:\n{e.stderr}", fg="red", bold=True
        )
    )
    raise


def add_telemetry_labels(
    kubectl_core_api: client.CoreV1Api,
    machine_type: str,
    nodes: list[str],
    check_name_to_result_label: dict[str, str],
    dry_run: bool = False,
) -> None:
  """Runs the telemetry label addition.

  Args:
    kubectl_core_api: The Kubernetes API client.
    machine_type: The machine type of the nodes to label.
    nodes: A list of node names to label.
    check_name_to_result_label: A dictionary mapping check names to their
      corresponding result labels.
    dry_run: Whether to run in dry run mode.
  """

  if dry_run:
    return

  if not nodes:
    raise click.Abort()

  node_data = _fetch_node_data_k8s(
      kubectl_core_api, machine_type, nodes, check_name_to_result_label
  )

  for node_name, node_data_item in node_data.items():
    check_results = node_data_item.check_results
    labels = "-".join(
        f"{check_name}_{result}"
        for check_name, result in check_results.items()
    )
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    labels_with_timestamp = {
        _CHS_LAST_SCAN_RESULT: labels,
        _CHS_LAST_SCAN_TIME: timestamp,
    }
    _update_nodes_with_last_scan_result(
        node_name, labels_with_timestamp, node_data_item.zone
    )
