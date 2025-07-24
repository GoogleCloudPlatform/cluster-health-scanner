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

"""Module to fetch nodes from a slurm cluster."""

import subprocess


def expand_slurm_nodes(nodes: list[str]) -> list[str]:
  """Expands a list of slurm nodes into a list of nodes.

  Args:
    nodes: A list of slurm node patterns to expand (e.g., ['node[1-3]',
      'node4']).

  Returns:
    A list of all hostnames matching the given patterns.
  """
  nodelist = []
  for node in nodes:
    nodelist.extend(_expand_slurm_node_pattern(node))
  return nodelist


def _expand_slurm_node_pattern(node_pattern: str) -> list[str]:
  """Expands a slurm node pattern into a list of hostnames.

  Args:
    node_pattern: The slurm node pattern to expand (e.g., 'node[1-3]').

  Returns:
    A list of hostnames matching the given pattern.
  """
  slurm_nodelist_expansion_cmd = [
      'scontrol',
      'show',
      'hostname',
      node_pattern,
  ]
  output = subprocess.run(
      slurm_nodelist_expansion_cmd,
      text=True,
      check=True,
      capture_output=True,
  ).stdout.strip()
  nodes = output.split('\n')
  return nodes
