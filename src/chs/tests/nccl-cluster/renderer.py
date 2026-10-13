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
from chs.protos import bill_of_health_pb2 as pb
from chs.core.default_plugin.renderer import DefaultRenderer
from chs.core.plugin_base import PluginDetailReport
from chs.core.utils.nccl_utils import (
    render_nccl_detail,
    extract_nccl_detail_data,
)


class NcclClusterRenderer(DefaultRenderer):
    def get_detail_data(
        self, test_results: list[pb.TestResult]
    ) -> PluginDetailReport | None:
        return extract_nccl_detail_data(
            "NCCL Cluster Detail",
            "Cluster Metric",
            test_results,
            lambda res: "Peak Detected Cluster Bandwidth",
        )

    def render_detail(
        self, test_results: list[pb.TestResult]
    ) -> RenderableType:
        return render_nccl_detail(
            "NCCL Cluster Detail",
            "Cluster Metric",
            test_results,
            lambda res: "Peak Detected Cluster Bandwidth",
        )
