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

import logging

from chs.testkit import MultiNodePlugin, NodeGrouping
from chs.utils.topology import (
    build_node_to_location,
    get_topology,
    load_topology,
)

logger = logging.getLogger(__name__)


class Plugin(MultiNodePlugin):
    """Measures bandwidth between pairs of nodes.

    Pairs are chosen to cover the cluster at each level of its physical
    hierarchy rather than to cover every possible pair, which would grow with
    the square of the cluster size.

    Set ``allow_interdomain_pairs: false`` to measure only pairs inside one
    NVLink domain. That needs the cluster topology.
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
        allow_interdomain_pairs = request.variables.get(
            "allow_interdomain_pairs", True
        )
        topology = load_topology(request.workspace_root, warn_on_error=True)

        if not topology and not allow_interdomain_pairs:
            logger.warning(
                "The 'nccl-pairwise' diagnostic requires cluster topology "
                "data when allow_interdomain_pairs is False. "
                "Skipping this test."
            )
            return NodeGrouping(groups=[])

        if topology:
            grouping = NodeGrouping(
                groups=generate_topological_pairs(
                    topology, allow_interdomain_pairs
                ),
                node_locations=build_node_to_location(topology),
            )
        else:
            flat = {"default": {"default-subblock": request.nodes}}
            grouping = NodeGrouping(
                groups=generate_topological_pairs(
                    flat, allow_interdomain_pairs
                ),
            )

        if not grouping.groups:
            logger.warning(
                "No node pairs could be formed from %d nodes for "
                "'nccl-pairwise'. Skipping this test.",
                len(request.nodes),
            )
        return grouping

    def group_suffix(self, nodes, grouping):
        return f"-{nodes[0]}-{nodes[1]}"

    def group_domain_type(self, nodes, grouping):
        if grouping.node_locations and len(nodes) == 2:
            domain = get_topology(nodes[0], nodes[1], grouping.node_locations)
            return domain if domain != "unknown" else "interdomain"
        return "interdomain"


def generate_topological_pairs(
    topology: dict[str, dict[str, list[str]]],
    allow_interdomain_pairs: bool = True,
) -> list[list[str]]:
    """Pairs within each subblock, then (if allowed) across subblocks and
    blocks."""
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

        if not allow_interdomain_pairs:
            continue
        pairs.extend(_adjacent_pairs(subblock_representatives))
        if subblock_representatives:
            block_representatives.append(subblock_representatives[0])

    if allow_interdomain_pairs:
        pairs.extend(_adjacent_pairs(block_representatives))
    return pairs


def _adjacent_pairs(nodes: list[str]) -> list[list[str]]:
    """Pairs a list two at a time, re-using the last node if it is odd."""
    pairs = [[nodes[i], nodes[i + 1]] for i in range(0, len(nodes) - 1, 2)]
    if len(nodes) % 2 != 0 and len(nodes) > 1:
        pairs.append([nodes[-2], nodes[-1]])
    return pairs
