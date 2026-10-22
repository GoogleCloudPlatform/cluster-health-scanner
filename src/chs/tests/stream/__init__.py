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

from chs.core.default_plugin.extractor import DefaultExtractor
from chs.core.plugin_base import TestPlugin
from .renderer import StreamRenderer

from chs.protos import bill_of_health_pb2 as pb


class StreamPlugin(StreamRenderer, DefaultExtractor, TestPlugin):
    def get_relative_fom_keys(self) -> list[tuple[str, str]]:
        return [("Achieved Bandwidth", "Peak Bandwidth")]

    def post_process_test_result(self, test_result: pb.TestResult) -> None:
        for m in test_result.metrics:
            if m.key == "Achieved Bandwidth" and m.unit == "MB/s":
                m.value.number_value /= 1000.0
                m.unit = "GB/s"

    def _filter_validation_results(
        self, test_results: list[pb.TestResult], sorted_nodes: list[str]
    ) -> list[pb.TestResult]:
        """Filters validation results to keep worst Triad metric per node."""
        if not test_results:
            return test_results

        filtered_results = []
        for res in test_results:
            filtered_result = pb.TestResult()
            filtered_result.CopyFrom(res)

            triad_metrics = [
                m
                for m in filtered_result.metrics
                if m.key == "Achieved Bandwidth"
                and m.HasField("metadata")
                and "context" in m.metadata.fields
                and "Triad" in m.metadata.fields["context"].string_value
            ]

            if not triad_metrics:
                filtered_results.append(filtered_result)
                continue

            worst_metric = min(
                triad_metrics, key=lambda m: m.value.number_value
            )

            del filtered_result.metrics[:]
            filtered_result.metrics.append(worst_metric)

            if worst_metric.HasField("validation_criteria"):
                filtered_result.result_status = (
                    worst_metric.validation_criteria.result_status
                )
            else:
                filtered_result.result_status = pb.RESULT_STATUS_UNKNOWN

            filtered_results.append(filtered_result)

        return filtered_results


__chs_plugin__ = {
    "test_name": "stream",
    "plugin": StreamPlugin,
}
