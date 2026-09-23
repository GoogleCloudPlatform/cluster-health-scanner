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

"""Cluster network topology utilities for CHS."""

import logging

logger = logging.getLogger(__name__)


def build_node_to_location(
    topology: dict[str, dict[str, list[str]]] | None,
) -> dict[str, dict[str, str]]:
    """Builds a flat node-to-location map from cluster topology."""
    if not topology or not isinstance(topology, dict):
        logger.debug(
            "Topology is missing or not a dictionary; returning empty"
            " node-to-location map."
        )
        return {}
    node_to_location = {}
    for block, subblocks in topology.items():
        if isinstance(subblocks, dict):
            for subblock, nodes in subblocks.items():
                if isinstance(nodes, list):
                    for node in nodes:
                        node_to_location[node] = {
                            "block": block,
                            "subblock": subblock,
                        }
    return node_to_location


def get_topological_context(
    node1: str, node2: str, node_to_location: dict[str, dict[str, str]]
) -> str:
    """Returns the topological relationship string between two nodes."""
    if not node_to_location:
        return "Unknown"

    loc1 = node_to_location.get(node1)
    loc2 = node_to_location.get(node2)

    if loc1 and loc2:
        if loc1["block"] != loc2["block"]:
            return "Inter-Block"
        elif loc1["subblock"] != loc2["subblock"]:
            return "Inter-Subblock"
        else:
            return "Intra-Subblock"

    return "Unknown"


def get_topology(
    node1: str, node2: str, node_to_location: dict[str, dict[str, str]]
) -> str:
    """Returns 'intradomain' or 'interdomain' based on pair topology."""
    topo_ctx = get_topological_context(node1, node2, node_to_location)
    if topo_ctx == "Intra-Subblock":
        return "intradomain"
    elif topo_ctx in ("Inter-Subblock", "Inter-Block"):
        return "interdomain"
    return "unknown"
