# Copyright 2026 Google LLC
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

"""Pairwise NCCL bandwidth check between nodes."""

import json
import logging
import os

from chs.testkit import MultiNodePlugin, NodeGrouping
from chs.utils.topology import build_node_to_location, get_topology

logger = logging.getLogger(__name__)

TOPOLOGY_FILENAME = "topology.json"


class Plugin(MultiNodePlugin):
    """Measures bandwidth between pairs of nodes.

    Pairs are chosen to cover the cluster at each level of its physical
    hierarchy rather than to cover every possible pair, which would grow with
    the square of the cluster size.
    """

    name = "nccl-pairwise"
    category = "Network & Collectives"
    description = "Node-to-node latency check"
    long_description = (
        "Runs NCCL benchmarks between node pairs to measure point-to-point "
        "bandwidth (GB/s) and latency, helping isolate degraded network links."
    )
    application = "nccl-tests"
    default_workloads = ("all-gather", "all-reduce")
    min_nodes_to_run = 2

    def grouping(self, request):
        topology = load_topology(request.workspace_root)
        if topology:
            return NodeGrouping(
                groups=generate_topological_pairs(topology),
                node_locations=build_node_to_location(topology),
            )

        flat = {"default": {"default-subblock": request.nodes}}
        return NodeGrouping(groups=generate_topological_pairs(flat))

    def group_suffix(self, nodes):
        return f"-{nodes[0]}-{nodes[1]}"

    def group_domain_type(self, nodes, grouping):
        if grouping.node_locations and len(nodes) == 2:
            domain = get_topology(nodes[0], nodes[1], grouping.node_locations)
            return domain if domain != "unknown" else "interdomain"
        return "interdomain"


def load_topology(workspace_root: str | None) -> dict | None:
    if not workspace_root:
        return None

    path = os.path.join(workspace_root, TOPOLOGY_FILENAME)
    if not os.path.exists(path):
        logger.debug("No cluster topology file at '%s'.", path)
    else:
        try:
            with open(path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.debug(
                "Failed to parse the cluster topology JSON file at '%s'. "
                "Ensure the file is correctly formatted. Error: %s",
                path,
                e,
            )
        except IOError as e:
            logger.debug(
                "Failed to read the cluster topology file at '%s'. Network "
                "discovery may be incomplete. Error: %s",
                path,
                e,
            )
    logger.warning(
        "Could not load cluster topology data. Pairing nodes arbitrarily, "
        "and judging every pair against the interdomain thresholds."
    )
    return None


def generate_topological_pairs(
    topology: dict[str, dict[str, list[str]]],
) -> list[list[str]]:
    pairs = []
    block_representatives = []

    for block in sorted(topology.keys()):
        subblocks = topology[block]
        subblock_representatives = []

        for subblock in sorted(subblocks.keys()):
            nodes = sorted(subblocks[subblock])
            pairs.extend(_adjacent_pairs(nodes))
            if nodes:
                subblock_representatives.append(nodes[0])

        pairs.extend(_adjacent_pairs(subblock_representatives))
        if subblock_representatives:
            block_representatives.append(subblock_representatives[0])

    pairs.extend(_adjacent_pairs(block_representatives))
    return pairs


def _adjacent_pairs(nodes: list[str]) -> list[list[str]]:
    """Pairs a list two at a time, re-using the last node if it is odd."""
    pairs = [[nodes[i], nodes[i + 1]] for i in range(0, len(nodes) - 1, 2)]
    if len(nodes) % 2 != 0 and len(nodes) > 1:
        pairs.append([nodes[-2], nodes[-1]])
    return pairs
