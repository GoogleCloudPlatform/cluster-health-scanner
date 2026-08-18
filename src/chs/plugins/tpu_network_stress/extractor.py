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


class TpuNetworkStressExtractor(DefaultExtractor):
    """Overrides DefaultExtractor to preserve test naming and contexts."""

    def extract_test_info(self, exp: dict[str, Any]) -> tuple[str, str, str]:
        return (
            "tpu-network-stress",
            exp.get("workload_name", ""),
            self.get_description(),
        )

    def get_required_contexts(
        self, contexts: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return contexts

    def get_fom_aliases(self) -> dict[str, str]:
        return {
            "bandwidth": "Bandwidth",
        }

    def post_process_test_result(self, test_result: pb.TestResult) -> None:
        """Calculates and appends the Minimum Bandwidth metric for each node."""
        min_bandwidth_by_node: dict[str, float] = {}

        for metric in test_result.metrics:
            if metric.key.lower() == "bandwidth" and metric.value.HasField(
                "number_value"
            ):
                context_name = (
                    metric.metadata.fields["context"].string_value
                    if "context" in metric.metadata.fields
                    else ""
                )
                context_parts = context_name.split(" Axis-")
                if len(context_parts) == 2:
                    node_name = context_parts[0].strip()
                    bandwidth_value = metric.value.number_value
                    if (
                        node_name not in min_bandwidth_by_node
                        or bandwidth_value < min_bandwidth_by_node[node_name]
                    ):
                        min_bandwidth_by_node[node_name] = bandwidth_value
                    metric.key = "Axis Bandwidth"

        for node_name, min_bandwidth_value in min_bandwidth_by_node.items():
            min_metric = test_result.metrics.add()
            min_metric.key = "Bandwidth"
            min_metric.value.number_value = min_bandwidth_value
            min_metric.unit = "GB/s"
            min_metric.metadata.fields["context"].string_value = node_name
