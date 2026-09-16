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
from chs.plugins.stream import StreamPlugin
from chs.plugins.dcgm import DcgmPlugin
from unittest.mock import MagicMock


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
    assert group.target_nodes == ["node-1"]
    assert group.rows[0].target_nodes == ["node-1"]
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


def test_validation_analyzer_no_truncation():
    res = pb.TestResult(
        test_name="tpu-collectives",
        test_level="all_gather",
        result_status=pb.RESULT_STATUS_SUCCESS,
    )
    for i in range(1, 33):
        res.target_resources.add(node_name=f"node-{i}")
    m = res.metrics.add(key="Peak Bandwidth (median)", unit="GB/s")
    m.value.number_value = 160.0
    crit = m.validation_criteria
    crit.name = "peak_bw_check"
    crit.comparator = pb.Comparator.COMPARATOR_GTE
    crit.expected_value.number_value = 150.0
    crit.result_status = pb.RESULT_STATUS_SUCCESS

    groups_default = extract_validation_data([res])
    assert len(groups_default) == 1
    assert len(groups_default[0].target_nodes) == 32
    assert groups_default[0].target_nodes[0] == "node-1"
    assert "node-32" in groups_default[0].nodes_str
    assert "... and" not in groups_default[0].nodes_str

    groups_truncated = extract_validation_data([res], max_display_nodes=5)
    assert len(groups_truncated) == 1
    assert "node-32" not in groups_truncated[0].nodes_str
    assert "... and 27 more" in groups_truncated[0].nodes_str


def test_stream_validation_ratio_extraction_success():
    res = pb.TestResult(
        test_name="stream",
        test_level="standard",
        result_status=pb.RESULT_STATUS_SUCCESS,
    )
    res.target_resources.add(node_name="node-1")

    # Add healthy GPU Achieved Bandwidth metric with relative ratio in metadata
    # (calculated during relative validation as 4300 / 4800 = 0.8958)
    m_rate = res.metrics.add(key="Achieved Bandwidth", unit="GB/s")
    m_rate.value.number_value = 4300.0
    m_rate.metadata.fields["context"].string_value = "Device 0 Triad"
    m_rate.metadata.fields["relative"].number_value = 0.8958
    m_rate.metadata.fields[
        "relative_formula"
    ].string_value = "Achieved Bandwidth / Peak Bandwidth"

    crit = m_rate.validation_criteria
    crit.name = "efficiency"
    crit.comparator = pb.Comparator.COMPARATOR_GTE
    crit.expected_value.number_value = 0.8  # relative threshold
    crit.result_status = pb.RESULT_STATUS_SUCCESS

    mock_plugin = StreamPlugin()
    mock_pm = MagicMock()
    mock_pm.get_plugin.return_value = mock_plugin
    groups = extract_validation_data([res], plugin_manager=mock_pm)
    assert len(groups) == 1
    group = groups[0]

    # Verification: should show the ratio and drop the unit
    assert group.formula == "Achieved Bandwidth / Peak Bandwidth"
    assert group.validation_name == "Efficiency"
    assert len(group.rows) == 1
    assert group.rows[0].target == "node-1"
    assert group.rows[0].metric == "Efficiency"
    assert group.rows[0].value == 0.8958
    assert group.rows[0].formatted_value == "0.90"
    assert group.rows[0].unit == ""
    assert group.rows[0].threshold == ">= 0.8"
    assert group.rows[0].status == "MET"


def test_stream_validation_ratio_extraction_failure():
    res = pb.TestResult(
        test_name="stream",
        test_level="standard",
        result_status=pb.RESULT_STATUS_FAILED,
    )
    res.target_resources.add(node_name="node-1")

    # Add failing GPU Achieved Bandwidth metric with relative ratio in metadata
    # (calculated during relative validation as 3000 / 4800 = 0.625)
    m_rate = res.metrics.add(key="Achieved Bandwidth", unit="GB/s")
    m_rate.value.number_value = 3000.0
    m_rate.metadata.fields["context"].string_value = "Device 0 Triad"
    m_rate.metadata.fields["relative"].number_value = 0.625
    m_rate.metadata.fields[
        "relative_formula"
    ].string_value = "Achieved Bandwidth / Peak Bandwidth"

    crit = m_rate.validation_criteria
    crit.name = "efficiency"
    crit.comparator = pb.Comparator.COMPARATOR_GTE
    crit.expected_value.number_value = 0.8
    crit.result_status = pb.RESULT_STATUS_FAILED

    mock_plugin = StreamPlugin()
    mock_pm = MagicMock()
    mock_pm.get_plugin.return_value = mock_plugin
    groups = extract_validation_data([res], plugin_manager=mock_pm)
    assert len(groups) == 1
    group = groups[0]

    assert group.formula == "Achieved Bandwidth / Peak Bandwidth"
    assert group.validation_name == "Efficiency"
    assert len(group.rows) == 1
    assert group.rows[0].target == "node-1"
    assert group.rows[0].metric == "Efficiency"
    assert group.rows[0].value == 0.625
    assert group.rows[0].formatted_value == "0.62"
    assert group.rows[0].unit == ""
    assert group.rows[0].threshold == ">= 0.8"
    assert group.rows[0].status == "NOT MET"


def test_stream_validation_filtering():
    res = pb.TestResult(
        test_name="stream",
        test_level="standard",
        result_status=pb.RESULT_STATUS_SUCCESS,
    )
    res.target_resources.add(node_name="node-1")

    # Add metrics for multiple devices and operations
    metrics_data = [
        ("Device 0 Copy", 900.0, 0.90, pb.RESULT_STATUS_SUCCESS),
        ("Device 0 Triad", 850.0, 0.85, pb.RESULT_STATUS_SUCCESS),
        ("Device 1 Copy", 910.0, 0.91, pb.RESULT_STATUS_SUCCESS),
        (
            "Device 1 Triad",
            800.0,
            0.80,
            pb.RESULT_STATUS_SUCCESS,
        ),  # Worst Triad
        ("Device 1 Add", 820.0, 0.82, pb.RESULT_STATUS_SUCCESS),
    ]

    for ctx, val, rel, status in metrics_data:
        m = res.metrics.add(key="Achieved Bandwidth", unit="GB/s")
        m.value.number_value = val
        m.metadata.fields["context"].string_value = ctx
        m.metadata.fields["relative"].number_value = rel

        crit = m.validation_criteria
        crit.name = "efficiency"
        crit.comparator = pb.Comparator.COMPARATOR_GTE
        crit.expected_value.number_value = 0.8
        crit.result_status = status

    # Mock plugin manager
    mock_plugin = StreamPlugin()
    mock_pm = MagicMock()
    mock_pm.get_plugin.return_value = mock_plugin

    groups = extract_validation_data([res], plugin_manager=mock_pm)
    assert len(groups) == 1
    group = groups[0]

    # Verify that only the worst Triad is kept
    assert len(group.rows) == 1
    assert group.rows[0].target == "node-1"
    assert group.rows[0].metric == "Efficiency"
    assert group.rows[0].value == 0.80
    assert group.rows[0].formatted_value == "0.80"
    assert group.rows[0].unit == ""
    assert group.rows[0].threshold == ">= 0.8"
    assert group.rows[0].status == "MET"


def test_dcgm_validation_brackets():
    res = pb.TestResult(
        test_name="dcgm",
        test_level="1",
        result_status=pb.RESULT_STATUS_SUCCESS,
    )
    res.target_resources.add(node_name="node-1")

    m = res.metrics.add(key="Memory Bandwidth", unit="GB/s")
    m.value.number_value = 300.0
    m.metadata.fields["context"].string_value = "GPU 0"

    crit = m.validation_criteria
    crit.name = "bandwidth_check"
    crit.comparator = pb.Comparator.COMPARATOR_GTE
    crit.expected_value.number_value = 250.0
    crit.result_status = pb.RESULT_STATUS_SUCCESS

    mock_plugin = DcgmPlugin()
    mock_pm = MagicMock()
    mock_pm.get_plugin.return_value = mock_plugin
    groups = extract_validation_data([res], plugin_manager=mock_pm)
    assert len(groups) == 1
    group = groups[0]

    assert len(group.rows) == 1
    assert group.rows[0].target == "node-1"
    assert group.rows[0].metric == "Memory Bandwidth (GPU 0)"
