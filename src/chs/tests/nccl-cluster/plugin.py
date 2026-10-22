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

"""Cluster-wide NCCL collective bandwidth check."""

from chs.testkit import MultiNodePlugin
from chs.utils.topology import build_node_to_location, load_topology


class Plugin(MultiNodePlugin):
    name = "nccl-cluster"
    category = "Network & Collectives"
    description = "Full cluster ring bandwidth"
    long_description = (
        "Runs NCCL benchmarks across multiple nodes to measure inter-node "
        "bandwidth (GB/s) and latency under collective workloads."
    )
    application = "nccl-tests"
    default_workloads = ("all-gather", "all-reduce")
    min_nodes_to_run = 3

    def prepare_variables(self, variables, workload, request):
        # Label only, for threshold conditions. Returning it from
        # group_domain_type would also merge the per-domain config blocks,
        # which nccl-cluster must not pick up.
        topology = load_topology(request.workspace_root, warn_on_error=True)
        variables["domain_type"] = cluster_domain_type(request.nodes, topology)


def cluster_domain_type(nodes: list[str], topology: dict) -> str:
    """Returns whether all nodes share one NVLink domain (subblock).

    Returns 'intradomain' if every node is in the same block/subblock,
    'interdomain' if they span more than one, and 'unknown' if any node
    has no topology information.
    """
    node_to_loc = build_node_to_location(topology)
    if not node_to_loc:
        return "unknown"
    domains = set()
    for node in nodes:
        loc = node_to_loc.get(node) or {}
        if not loc.get("block") or not loc.get("subblock"):
            return "unknown"
        domains.add((loc["block"], loc["subblock"]))
    return "intradomain" if len(domains) == 1 else "interdomain"
