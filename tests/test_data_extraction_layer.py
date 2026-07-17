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

"""Unit tests for UI-decoupled Data Extraction and Analysis layer."""

from chs.core.summary_aggregator import aggregate_boh_summary
from chs.plugins.utils.validation_analyzer import (
    extract_validation_data,
    has_validation_criteria,
)
from chs.protos import bill_of_health_pb2 as pb


def test_validation_analyzer_extraction():
    res = pb.TestResult(
        test_name="tpu-gemm",
        test_level="check",
        result_status=pb.RESULT_STATUS_SUCCESS,
    )
    res.target_resources.add(node_name="node-1")
    m = res.metrics.add(key="Throughput", unit="TFLOP/s/dev")
    m.value.number_value = 120.5
    crit = m.validation_criteria
    crit.name = "throughput_check"
    crit.comparator = pb.Comparator.COMPARATOR_GTE
    crit.expected_value.number_value = 100.0
    crit.result_status = pb.RESULT_STATUS_SUCCESS

    assert has_validation_criteria([res]) is True
    groups = extract_validation_data([res])
    assert len(groups) == 1
    group = groups[0]
    assert group.test_name == "tpu-gemm"
    assert len(group.rows) == 1
    assert group.rows[0].metric == "Throughput"
    assert group.rows[0].status == "MET"
    assert group.rows[0].threshold == ">= 100"


def test_boh_summary_aggregator():
    boh = pb.BillOfHealth(report_id="run-123")
    res = boh.test_results.add(
        test_name="tpu-gemm",
        test_level="check",
        result_status=pb.RESULT_STATUS_SUCCESS,
    )
    res.target_resources.add(node_name="node-1")
    m = res.metrics.add(key="Throughput")
    m.value.number_value = 120.5

    summary = aggregate_boh_summary(boh)
    assert summary is not None
    assert summary.total_nodes == 1
    assert len(summary.test_summaries) == 1
    assert summary.test_summaries[0].met_threshold == 1
