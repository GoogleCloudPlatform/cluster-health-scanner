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

from typing import Any
from chs.core.default_plugin.extractor import DefaultExtractor
from chs.protos import bill_of_health_pb2 as pb


class TpuCollectivesExtractor(DefaultExtractor):
    def get_required_contexts(
        self, contexts: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Preserve all contexts for TPU Collectives results parsing."""
        return contexts

    def extract_test_info(self, exp: dict[str, Any]) -> tuple[str, str, str]:
        """Extracts test name and level for TPU Collectives."""
        name = "tpu-collectives"
        level = exp.get("workload_name", "check")
        return name, level, self.get_description(name)

    def post_process_test_result(self, test_result: pb.TestResult) -> None:
        """Calculates and appends the Peak Bandwidth metric for each node."""
        peak_by_node = {}
        for m in test_result.metrics:
            if m.key == "Bandwidth (median)":
                ctx = m.metadata.fields["context"].string_value
                parts = ctx.split(",")
                node = parts[0]
                val = m.value.number_value
                if node not in peak_by_node or val > peak_by_node[node]["val"]:
                    peak_by_node[node] = {
                        "val": val,
                        "matrix_dim": parts[1]
                        if len(parts) == 3
                        else "unknown",
                        "dtype": parts[2] if len(parts) == 3 else "unknown",
                    }

        for node, info in peak_by_node.items():
            peak_metric = test_result.metrics.add()
            peak_metric.key = "Peak Bandwidth (median)"
            peak_metric.value.number_value = info["val"]
            peak_metric.unit = "GB/s"
            peak_metric.metadata.fields[
                "context"
            ].string_value = f"{node},peak,{info['matrix_dim']},{info['dtype']}"
