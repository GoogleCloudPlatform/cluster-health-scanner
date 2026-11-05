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

from chs.core.plugin_base import TestPlugin
from .extractor import GemmExtractor
from .renderer import GemmRenderer
from chs.protos import bill_of_health_pb2 as pb


class GemmPlugin(GemmRenderer, GemmExtractor, TestPlugin):
    def get_fom_aliases(self) -> dict[str, str]:
        return {"performance": "Value"}

    def _filter_validation_results(
        self, test_results: list[pb.TestResult], sorted_nodes: list[str]
    ) -> list[pb.TestResult]:
        """Filters validation results to keep only the MEDIAN metric."""
        if not test_results:
            return test_results

        filtered_results = []
        for res in test_results:
            filtered_result = pb.TestResult()
            filtered_result.CopyFrom(res)

            median_metrics = [
                m
                for m in filtered_result.metrics
                if m.key == "Value"
                and m.HasField("metadata")
                and "context" in m.metadata.fields
                and m.metadata.fields["context"].string_value == "MEDIAN"
            ]

            del filtered_result.metrics[:]
            if median_metrics:
                filtered_result.metrics.extend(median_metrics)

            if filtered_result.metrics:
                all_success = all(
                    m.validation_criteria.result_status
                    == pb.RESULT_STATUS_SUCCESS
                    for m in filtered_result.metrics
                    if m.HasField("validation_criteria")
                )
                filtered_result.result_status = (
                    pb.RESULT_STATUS_SUCCESS
                    if all_success
                    else pb.RESULT_STATUS_FAILED
                )
            elif filtered_result.result_status == pb.RESULT_STATUS_SUCCESS:
                filtered_result.result_status = pb.RESULT_STATUS_UNKNOWN

            filtered_results.append(filtered_result)

        return filtered_results


__chs_plugin__ = {
    "test_name": "gemm",
    "plugin": GemmPlugin,
}
