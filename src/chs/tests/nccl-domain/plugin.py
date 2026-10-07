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

"""NCCL collective bandwidth inside each NVLink domain."""

import logging

from chs.testkit import MultiNodePlugin, NodeGrouping
from chs.utils.cluster import sanitize_dns_1123
from chs.utils.topology import (
    build_node_to_location,
    generate_domain_groups,
    get_domain_label,
    load_topology,
)

logger = logging.getLogger(__name__)


class Plugin(MultiNodePlugin):
    """Runs one NCCL job per NVLink domain (IMEX / NVLD rack).

    A slow domain then shows up on its own instead of dragging down one
    cluster-wide number.
    """

    name = "nccl-domain"
    category = "Network & Collectives"
    description = "Per-NVLink-domain ring bandwidth"
    long_description = (
        "Runs NCCL benchmarks separately within each NVLink domain "
        "(IMEX/NVLD racks) to measure intra-domain MNNVL bandwidth (GB/s), "
        "isolating a degraded domain without invalidating the rest of the "
        "cluster."
    )
    application = "nccl-tests"
    default_workloads = ("all-gather", "all-reduce")
    min_nodes_per_domain = 3

    def grouping(self, request):
        topology = load_topology(request.workspace_root, warn_on_error=True)
        if not topology:
            logger.warning(
                "The 'nccl-domain' diagnostic requires cluster topology "
                "data, but none was found. Skipping this test."
            )
            return NodeGrouping(groups=[])

        domains = generate_domain_groups(
            topology, request.nodes, self.min_nodes_per_domain
        )
        if not domains:
            logger.warning(
                "The 'nccl-domain' diagnostic requires domains with at "
                f"least {self.min_nodes_per_domain} nodes, but none met this "
                "criteria. Skipping this test."
            )
        return NodeGrouping(
            groups=[nodes for _, nodes in domains],
            node_locations=build_node_to_location(topology),
        )

    def group_suffix(self, nodes, grouping):
        domain = get_domain_label(nodes, grouping.node_locations)
        return f"-{sanitize_dns_1123(domain)}"

    def group_domain_type(self, nodes, grouping):
        return "intradomain"
