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

"""Single-node NCCL collective bandwidth check."""

from chs.testkit import ClusterTestPlugin, NodeGrouping


class Plugin(ClusterTestPlugin):
    name = "nccl-single-node"
    category = "Network & Collectives"
    description = "Single node network testing"
    long_description = (
        "Runs NCCL benchmarks within a single node to measure intra-node "
        "GPU-to-GPU bandwidth (GB/s) and latency."
    )
    application = "nccl-tests"
    default_workloads = ("all-gather", "all-reduce")

    def grouping(self, request):
        return NodeGrouping(groups=[[node] for node in request.nodes])

    def group_suffix(self, nodes):
        return f"-{nodes[0]}"

    def finalize_context(self, context, workload, request):
        """Keeps the collective off the high speed fabric.

        The traffic never leaves the node, so NCCL is pointed at plain sockets
        rather than the cluster interconnect. Without this NCCL may still
        initialise the fabric and the measurement would include it.
        """
        del workload, request
        context.setdefault("env_vars", {}).setdefault("set", {})["NCCL_NET"] = (
            "Socket"
        )
