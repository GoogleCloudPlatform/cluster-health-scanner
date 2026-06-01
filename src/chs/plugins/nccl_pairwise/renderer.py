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

from chs.protos import bill_of_health_pb2 as pb
from rich.console import RenderableType
from chs.core.default_plugin.renderer import DefaultRenderer
from chs.plugins.utils.nccl_utils import render_nccl_detail


class NcclPairwiseRenderer(DefaultRenderer):
    def __init__(self):
        super().__init__()
        self.topology = None

    def set_topology(self, topology):
        self.topology = topology

    def render_detail(
        self, test_results: list[pb.TestResult]
    ) -> RenderableType:
        return render_nccl_detail(
            "Pairwise Detail (Aggregated)",
            "Pair (Node 1 <-> Node 2)",
            test_results,
            lambda res: " <-> ".join(res.target_resource_ids)
            if res.target_resource_ids
            else "Unknown Pair",
            self.topology,
        )
