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

from rich.console import RenderableType

from chs.core.default_plugin.renderer import DefaultRenderer
from chs.core.plugin_base import PluginDetailReport
from chs.protos import bill_of_health_pb2 as pb
from chs.plugins.utils.nccl_utils import (
    extract_nccl_detail_data,
    render_nccl_detail,
)
from chs.utils.topology import (
    build_node_to_location,
    build_subblock_to_blocks,
    get_domain_label,
)
from chs.plugins.utils.rendering import extract_node_names, format_nodes

_TITLE = "NCCL Domain Detail"
_RESOURCE_COLUMN = "NVLink Domain"


class NcclDomainRenderer(DefaultRenderer):
    def __init__(self):
        super().__init__()
        self.topology = None
        self._node_to_location = {}
        self._subblock_to_blocks = {}

    def set_topology(self, topology):
        self.topology = topology
        self._node_to_location = (
            build_node_to_location(topology) if topology else {}
        )
        self._subblock_to_blocks = build_subblock_to_blocks(
            self._node_to_location
        )

    def _resource_id(self, res: pb.TestResult) -> str:
        if not res.target_resources:
            return "Unknown Domain"

        nodes = extract_node_names(res.target_resources)
        domain = get_domain_label(
            nodes, self._node_to_location, self._subblock_to_blocks
        )
        if domain == "Unknown":
            return format_nodes(res.target_resources)
        return f"{domain} ({len(nodes)} nodes)"

    def get_detail_data(
        self, test_results: list[pb.TestResult]
    ) -> PluginDetailReport | None:
        return extract_nccl_detail_data(
            _TITLE,
            _RESOURCE_COLUMN,
            test_results,
            self._resource_id,
            self.topology,
        )

    def render_detail(
        self, test_results: list[pb.TestResult]
    ) -> RenderableType:
        return render_nccl_detail(
            _TITLE,
            _RESOURCE_COLUMN,
            test_results,
            self._resource_id,
            self.topology,
        )
