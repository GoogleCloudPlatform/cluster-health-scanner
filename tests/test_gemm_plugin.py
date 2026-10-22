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

from chs.tests.gemm import GemmPlugin
from chs.protos import bill_of_health_pb2 as pb


def test_gemm_plugin_filter_validation_results():
    plugin = GemmPlugin()

    # Case 1: Empty results
    assert plugin._filter_validation_results([], []) == []

    # Case 2: Results with MIN, MAX, MEDIAN metrics
    res = pb.TestResult(test_name="gemm", test_level="standard")
    res.target_resources.add(node_name="node-1")

    m_min = res.metrics.add(key="Value")
    m_min.value.number_value = 90.0
    m_min.metadata.fields["context"].string_value = "MIN"
    m_min.validation_criteria.result_status = pb.RESULT_STATUS_FAILED

    m_median = res.metrics.add(key="Value")
    m_median.value.number_value = 110.0
    m_median.metadata.fields["context"].string_value = "MEDIAN"
    m_median.validation_criteria.result_status = pb.RESULT_STATUS_SUCCESS

    m_max = res.metrics.add(key="Value")
    m_max.value.number_value = 120.0
    m_max.metadata.fields["context"].string_value = "MAX"
    m_max.validation_criteria.result_status = pb.RESULT_STATUS_SUCCESS
    res.result_status = pb.RESULT_STATUS_FAILED

    filtered = plugin._filter_validation_results([res], ["node-1"])

    assert len(filtered) == 1
    filtered_res = filtered[0]
    assert len(filtered_res.metrics) == 1
    assert (
        filtered_res.metrics[0].metadata.fields["context"].string_value
        == "MEDIAN"
    )
    assert filtered_res.result_status == pb.RESULT_STATUS_SUCCESS

    # Case 3: MEDIAN failed
    res2 = pb.TestResult(test_name="gemm", test_level="standard")
    res2.target_resources.add(node_name="node-1")

    m2_min = res2.metrics.add(key="Value")
    m2_min.value.number_value = 80.0
    m2_min.metadata.fields["context"].string_value = "MIN"
    m2_min.validation_criteria.result_status = pb.RESULT_STATUS_FAILED

    m2_median = res2.metrics.add(key="Value")
    m2_median.value.number_value = 95.0
    m2_median.metadata.fields["context"].string_value = "MEDIAN"
    m2_median.validation_criteria.result_status = pb.RESULT_STATUS_FAILED

    filtered2 = plugin._filter_validation_results([res2], ["node-1"])
    assert len(filtered2) == 1
    assert filtered2[0].result_status == pb.RESULT_STATUS_FAILED
