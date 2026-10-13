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
from chs.core.utils.rendering import format_nodes
from rich.console import RenderableType
from chs.core.default_plugin.renderer import DefaultRenderer
from chs.core.plugin_base import PluginDetailReport
from chs.core.utils.nccl_utils import (
    render_nccl_detail,
    extract_nccl_detail_data,
)


class NcclSingleNodeRenderer(DefaultRenderer):
    def get_detail_data(
        self, test_results: list[pb.TestResult]
    ) -> PluginDetailReport | None:
        return extract_nccl_detail_data(
            "NCCL Single Node Detail",
            "Node",
            test_results,
            lambda res: (
                format_nodes(res.target_resources)
                if res.target_resources
                else "Unknown Node"
            ),
        )

    def render_detail(
        self, test_results: list[pb.TestResult]
    ) -> RenderableType:
        return render_nccl_detail(
            "NCCL Single Node Detail",
            "Node",
            test_results,
            lambda res: (
                format_nodes(res.target_resources)
                if res.target_resources
                else "Unknown Node"
            ),
        )
