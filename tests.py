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

"""File for checking cluster status during CI/CD.

Intended to be run after the health runner has started, it repeatedly pings the
cluster
to check if labels look as expected.
Will error if any nodes do not reach expected state before timeout.

See parse_args() for information about expected calling format.
"""

from __future__ import annotations

import abc
import argparse
from collections.abc import Iterable
import re
import sys
import time
from typing import Any

import kubernetes
from kubernetes.client.models import V1Node

TIMEOUT_SECS: int = 60 * 15
POLL_INTERVAL_SECS: int = 60

V1API: kubernetes.client.CoreV1Api | None = None


def list_nodes() -> list[V1Node]:
  global V1API
  if V1API is None:
    V1API = kubernetes.client.CoreV1Api()
  return V1API.list_node().items


class NodePredicate(abc.ABC):
  """Checks that can be applied to a node."""

  def __init__(self, title: str):
    self.title: str = title

  @abc.abstractmethod
  def __call__(self, node: V1Node) -> bool:
    ...


class LabelRegexMatchPredicate(NodePredicate):
  """Checks if a label is a) present and b) matches the specified regex.

  returns False if either is validated
  """

  title_template: str = "{label} = {regex}"

  def __init__(self, label: str, regex: str) -> None:
    super().__init__(self.title_template.format(label=label, regex=regex))

    self.label: str = label
    self.regex: str = regex

  def __call__(self, node: V1Node) -> bool:
    label_value: str | None = node.metadata.labels.get(self.label)
    if label_value is None:
      return False
    return re.match(self.regex, label_value) is not None


class ClusterRenderer(abc.ABC):
  """Classes for pretty-printing the status of nodes for inspection in logs."""

  @abc.abstractmethod
  def __call__(self, nodes: Iterable[V1Node]) -> str:
    ...


class ReprRenderer(ClusterRenderer):
  """Very raw dumping of each node."""

  def __call__(self, nodes: Iterable[V1Node]) -> str:
    return "\n\n".join(map(repr, nodes))


class PredicateTableRenderer(ClusterRenderer):
  """Pretty prints the status of the checks in a table.

  A cell is True if the node associated with its row passes the check associated
  with its column.
  """

  def __init__(self, predicates: Iterable[NodePredicate]) -> None:
    super().__init__()
    self.predicates: list[NodePredicate] = list(predicates)
    self.header_row = ["Name"] + [pred.title for pred in self.predicates]

  def __call__(self, nodes: Iterable[V1Node]) -> str:
    cells: list[list[bool]] = self._predicate_value_table(nodes)
    return self._render_table([self.header_row] + cells)

  def _predicate_value_table(self, nodes: Iterable[V1Node]) -> list[list[bool]]:
    return [
        [node.metadata.name] + [pred(node) for pred in self.predicates]
        for node in nodes
    ]

  def _render_table(self, cells: list[list[Any]]) -> str:
    # Could just use pandas here, but choosing to avoid unnecessary deps

    # Convert all elements to strings
    str_data: list[list[str]] = [[str(item) for item in row] for row in cells]

    # Find the maximum width for each column
    num_cols: int = len(str_data[0])
    col_widths: list[int] = [
        max(len(row[i]) for row in str_data) for i in range(num_cols)
    ]

    result: str = ""

    # Pad each element in each column
    for row in str_data:
      for i, cell in enumerate(row):
        result += cell.ljust(col_widths[i]) + (" " * 5)
      result += "\n"
    return result


class ClusterStatusChecker:
  """Main workhorse class, applies checks to a filtered set of nodes & prints their status."""

  def __init__(
      self,
      check_predicates: Iterable[NodePredicate],
      filter_predicates: Iterable[NodePredicate] | None,
      renderers: Iterable[ClusterRenderer] | None,
  ):
    self.predicates_to_check: list[NodePredicate] = list(check_predicates)
    self.predicates_to_filter_for: list[NodePredicate] = (
        list(filter_predicates) if filter_predicates else []
    )
    self.renderers: list[ClusterRenderer] = list(renderers) if renderers else []

  def check_cluster_status(self, node_names: set[str]) -> bool:
    nodes: list[V1Node] = self._get_nodes(node_names)
    self._log_nodes(nodes)

    for node in nodes:
      for pred in self.predicates_to_check:
        if not pred(node):
          return False
    return True

  def check_cluster_status_with_retry(
      self,
      timeout_secs: int = TIMEOUT_SECS,
      poll_interval_secs: int = POLL_INTERVAL_SECS,
  ) -> bool:
    start_time: float = time.monotonic()
    nodes_to_check: set[str] = self._node_names_passing_filters()

    while time.monotonic() - start_time < timeout_secs:
      if self.check_cluster_status(nodes_to_check):
        return True

      time.sleep(poll_interval_secs)

    return False

  def _get_nodes(self, node_names: set[str]) -> list[V1Node]:
    nodes: Iterable[V1Node] = list_nodes()
    result: list[V1Node] = [
        node for node in nodes if node.metadata.name in node_names
    ]

    if len(result) != len(node_names):
      missingno = set(node_names) - set([node.metadata.name for node in result])
      print(f"Information missing for {missingno} number of nodes")

    return result

  def _log_nodes(self, nodes: Iterable[V1Node]) -> None:
    nodes = list(nodes)

    for renderer in self.renderers:
      print(renderer(nodes))

  def _node_names_passing_filters(self) -> set[str]:
    nodes: Iterable[V1Node] = list_nodes()

    nodes_passing_filter: list[V1Node] = [
        node
        for node in nodes
        if all([pred(node) for pred in self.predicates_to_filter_for])
    ]

    if not nodes_passing_filter:
      print("No nodes passing filters found")
      sys.exit(1)

    return set([node.metadata.name for node in nodes_passing_filter])


def parse_dict_args_to_predicates(
    arg_strs: Iterable[str],
) -> list[LabelRegexMatchPredicate]:

  result: list[LabelRegexMatchPredicate] = []

  for arg_str in arg_strs:
    label, regex = arg_str.split("=")

    if not label or not regex:
      print(f"Unable to parse filter argument: {arg_str}")
      sys.exit(1)

    result.append(LabelRegexMatchPredicate(label=label, regex=regex))

  return result


def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument(
      "--check",
      required=True,
      type=str,
      nargs="+",
      help=f"""The test will check if these labels match their expected format before {TIMEOUT_SECS} seconds.
Each check should be formatted as label_to_check=regex_to_match
"e.g. aiinfra/nccl-healthcheck-result=^pass$""",
  )

  parser.add_argument(
      "--filter",
      required=False,
      type=str,
      nargs="*",
      default=[],
      help="""Each filter argument should be of the form 'label_name=acceptable_value_regex'.
Nodes will be ignored from the test if they either a) are missing one of the filter labels or b) have values that do not match the corresponding regex.
All filter labels are checked at the BEGINNING of the test.
This allows the test to run on a subset of the nodes""",
  )

  # Filter is added b/c gpu_healthchecks only run on a3-megagpu nodes
  args: argparse.Namespace = parser.parse_args()

  filter_predicates: list[LabelRegexMatchPredicate] = (
      parse_dict_args_to_predicates(args.filter)
  )
  check_predicates: list[LabelRegexMatchPredicate] = (
      parse_dict_args_to_predicates(args.check)
  )

  return ClusterStatusChecker(
      check_predicates=check_predicates,
      filter_predicates=filter_predicates,
      renderers=[ReprRenderer(), PredicateTableRenderer(check_predicates)],
  )


if __name__ == "__main__":
  kubernetes.config.load_kube_config()

  checker: ClusterStatusChecker = parse_args()
  if not checker.check_cluster_status_with_retry():
    print("ERROR: All nodes did not pass before time out.")
    sys.exit(1)
