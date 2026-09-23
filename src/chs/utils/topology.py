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

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def load_topology(
    workspace_dir: str | os.PathLike[str] | None,
    warn_on_error: bool = False,
) -> dict[str, Any]:
    """Loads topology.json from a workspace directory if present."""
    if not workspace_dir:
        return {}
    topo_path = os.path.join(workspace_dir, "topology.json")
    if not os.path.exists(topo_path):
        return {}
    try:
        with open(topo_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError) as e:
        logger.debug(
            "Failed to load cluster topology file at '%s'. Error: %s",
            topo_path,
            e,
        )
        if warn_on_error:
            logger.warning(
                "Could not load cluster topology data. Some "
                "node-specific diagnostics will be skipped."
            )
        return {}


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


def build_subblock_to_blocks(
    node_to_location: dict[str, dict[str, str]],
) -> dict[str, set[str]]:
    """Builds a mapping from subblock name to set of parent block names."""
    subblock_to_blocks: dict[str, set[str]] = {}
    if not node_to_location:
        return subblock_to_blocks
    for loc in node_to_location.values():
        if isinstance(loc, dict):
            b = loc.get("block")
            sb = loc.get("subblock")
            if b and sb:
                subblock_to_blocks.setdefault(sb, set()).add(b)
    return subblock_to_blocks


def get_domain_label(
    nodes: list[str],
    node_to_location: dict[str, dict[str, str]],
    subblock_to_blocks: dict[str, set[str]] | None = None,
) -> str:
    """Resolves the NVLink domain (topology subblock) shared by nodes."""
    if not nodes or not node_to_location:
        return "Unknown"

    if subblock_to_blocks is None:
        subblock_to_blocks = build_subblock_to_blocks(node_to_location)

    labels = set()
    for node in nodes:
        loc = node_to_location.get(node)
        if not loc or not loc.get("subblock"):
            labels.add("Unknown")
        else:
            b = loc.get("block")
            sb = loc["subblock"]
            if len(subblock_to_blocks.get(sb, set())) > 1:
                labels.add(f"{b}-{sb}")
            else:
                labels.add(sb)

    if len(labels) > 1:
        return f"Mixed ({', '.join(sorted(labels))})"
    return labels.pop()


def generate_domain_groups(
    topology: dict[str, dict[str, list[str]]] | None,
    target_nodes: list[str] | None = None,
    min_nodes_per_domain: int = 3,
) -> list[tuple[str, list[str]]]:
    """Partitions target nodes into NVLink domain (subblock) groups."""
    if not topology:
        return []

    allowed = None if target_nodes is None else set(target_nodes)
    node_to_location = build_node_to_location(topology)
    subblock_to_blocks = build_subblock_to_blocks(node_to_location)

    groups = []
    for block in sorted(topology):
        if not isinstance(topology[block], dict):
            continue
        for subblock in sorted(topology[block]):
            raw_nodes = topology[block][subblock]
            if not isinstance(raw_nodes, list):
                continue
            nodes = sorted(raw_nodes)
            if allowed is not None:
                nodes = [node for node in nodes if node in allowed]

            if not nodes:
                continue

            label = get_domain_label(
                nodes, node_to_location, subblock_to_blocks
            )

            if len(nodes) < min_nodes_per_domain:
                logger.warning(
                    "Skipping domain '%s' because it has %d targeted node(s), "
                    "fewer than the minimum %d required.",
                    label,
                    len(nodes),
                    min_nodes_per_domain,
                )
                continue

            groups.append((label, nodes))

    return groups
