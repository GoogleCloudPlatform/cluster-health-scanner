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

from chs.testkit import SingleNodePlugin


class Plugin(SingleNodePlugin):
    name = "nccl-single-node"
    category = "Network & Collectives"
    description = "Single node network testing"
    long_description = (
        "Runs NCCL benchmarks within a single node to measure intra-node "
        "GPU-to-GPU bandwidth (GB/s) and latency."
    )
    application = "nccl-tests"
    default_workloads = ("all-gather", "all-reduce")

    def finalize_context(self, context, workload, request):
        context.setdefault("env_vars", {}).setdefault("set", {})["NCCL_NET"] = (
            "Socket"
        )
