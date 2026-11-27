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

"""Unit tests for HtmlReportGenerator."""

import os
import tempfile
from unittest import mock

from chs.core.summary_aggregator import aggregate_boh_summary
from chs.html_generator import HtmlReportGenerator
from chs.pdf_generator import builder
from chs.protos import bill_of_health_pb2 as pb


def test_html_report_generator_dict_structure():
    generator = HtmlReportGenerator()
    boh = pb.BillOfHealth()
    boh.report_id = "test-run-123"
    boh.metadata.fields["system_type"].string_value = "KUBERNETES_CLUSTER"
    boh.metadata.fields["skip_validation"].bool_value = False

    res = boh.test_results.add(
        test_name="tpu-info",
        test_level="local",
        result_status=pb.RESULT_STATUS_SUCCESS,
    )
    t1 = res.target_resources.add()
    t1.node_name = "node-1"
    t2 = res.target_resources.add()
    t2.node_name = "node-2"

    m = res.metrics.add(key="JAX Devices", unit="count")
    m.value.number_value = 8.0

    report_data = generator.generate_report_data(boh)

    assert report_data["metadata"]["report_id"] == "test-run-123"
    assert report_data["metadata"]["system_type"] == "KUBERNETES_CLUSTER"
    assert report_data["summary"]["total_nodes"] == 2
    assert report_data["summary"]["total_test_suites"] == 1
    assert report_data["summary"]["healthy_count"] == 2


def test_html_report_generator_rendering():
    generator = HtmlReportGenerator()
    boh = pb.BillOfHealth()
    boh.report_id = "test-run-456"

    res = boh.test_results.add(
        test_name="tpu-info",
        test_level="local",
        result_status=pb.RESULT_STATUS_SUCCESS,
    )
    t1 = res.target_resources.add()
    t1.node_name = "node-1"
    res.metrics.add(key="JAX Devices").value.number_value = 8.0

    html_str = generator.render_html(boh)
    assert "<!DOCTYPE html>" in html_str
    assert "test-run-456" in html_str
    assert "boh-overview-raw-data" in html_str or "boh-raw-data" in html_str


def test_html_report_generator_save_file():
    generator = HtmlReportGenerator()
    boh = pb.BillOfHealth()
    boh.report_id = "test-run-789"

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "reports", "boh.html")
        generator.save_html(boh, out_path)

        assert os.path.exists(out_path)
        with open(out_path, "r", encoding="utf-8") as f:
            content = f.read()
            assert "test-run-789" in content


def test_overview_report_generation():
    generator = HtmlReportGenerator()
    boh = pb.BillOfHealth()
    boh.report_id = "test-part1-001"

    res = boh.test_results.add(
        test_name="tpu-info",
        test_level="local",
        result_status=pb.RESULT_STATUS_SUCCESS,
    )
    t1 = res.target_resources.add()
    t1.node_name = "node-1"

    overview_data = generator.generate_report_data(boh)
    assert overview_data["metadata"]["report_id"] == "test-part1-001"
    assert overview_data["summary"]["total_nodes"] == 1
    assert overview_data["summary"]["total_test_suites"] == 1
    assert overview_data["summary"]["healthy_count"] == 0

    html_str = generator.render_html(boh)
    assert "boh-overview-raw-data" in html_str
    assert "test-part1-001" in html_str


def test_long_description_in_overview_report():
    from chs.html_generator import HtmlReportGenerator
    from chs.protos import bill_of_health_pb2 as pb

    generator = HtmlReportGenerator()
    boh = pb.BillOfHealth()
    boh.report_id = "test-part1-002"

    res = boh.test_results.add(
        test_name="nvidia-smi",
        test_level="local",
        result_status=pb.RESULT_STATUS_SUCCESS,
    )
    t1 = res.target_resources.add()
    t1.node_name = "node-1"

    overview_data = generator.generate_report_data(boh)
    found_nvidia = False
    for ts in overview_data["test_suite_breakdown"]:
        if "nvidia-smi" in ts["display_name"]:
            assert "long_description" in ts
            assert "Queries GPU status" in ts["long_description"]
            found_nvidia = True
    assert found_nvidia


def test_html_report_generator_validation_formula():
    generator = HtmlReportGenerator()
    boh = pb.BillOfHealth()
    boh.report_id = "test-formula-123"

    res = boh.test_results.add(
        test_name="stream",
        test_level="standard",
        result_status=pb.RESULT_STATUS_SUCCESS,
    )
    t = res.target_resources.add()
    t.node_name = "node-1"

    m = res.metrics.add(key="Achieved Bandwidth", unit="GB/s")
    m.value.number_value = 85.0
    m.metadata.fields["relative"].number_value = 0.85
    m.metadata.fields[
        "relative_formula"
    ].string_value = "Achieved Bandwidth / Peak Bandwidth"

    crit = m.validation_criteria
    crit.name = "efficiency"
    crit.comparator = pb.Comparator.COMPARATOR_GTE
    crit.expected_value.number_value = 0.8
    crit.result_status = pb.RESULT_STATUS_SUCCESS

    report_data = generator.generate_report_data(boh)

    assert len(report_data["validation_tables"]) == 1
    table = report_data["validation_tables"][0]
    assert table["test_name"] == "stream"
    assert table["formula"] == "Achieved Bandwidth / Peak Bandwidth"
    assert table["validation_name"] == "Efficiency"
    assert len(table["rows"]) == 1
    assert table["rows"][0]["metric"] == "Efficiency"


def test_html_report_generator_with_topology():
    topology = {
        "block-0": {
            "subblock-A": ["node-1", "node-2", "node-3"],
        }
    }
    generator = HtmlReportGenerator(topology=topology)
    boh = pb.BillOfHealth()
    boh.report_id = "test-domain-123"

    res = boh.test_results.add(
        test_name="nccl-domain",
        test_level="all-reduce",
        result_status=pb.RESULT_STATUS_SUCCESS,
    )
    for n in ["node-1", "node-2", "node-3"]:
        t = res.target_resources.add()
        t.node_name = n

    m = res.metrics.add(key="Peak Detected Bandwidth", unit="GB/s")
    m.value.number_value = 850.0

    report_data = generator.generate_report_data(boh)
    assert len(report_data["detailed_reports"]) == 1
    tables = report_data["detailed_reports"][0]["tables"]
    assert len(tables) == 1
    rows = tables[0]["rows"]
    assert len(rows) == 1
    assert rows[0]["resource"] == "subblock-A (3 nodes)"
    assert rows[0]["target_nodes"] == ["node-1", "node-2", "node-3"]
    assert (
        report_data["test_suite_breakdown"][0]["ratio_str"]
        == "1 / 1 domains met threshold"
    )


def test_html_report_generator_chs_version_from_boh():
    generator = HtmlReportGenerator()
    boh = pb.BillOfHealth()
    boh.report_id = "test-run"
    boh.chs_version = "1.2.3"
    report_data = generator.generate_report_data(boh)
    assert report_data["metadata"]["chs_version"] == "1.2.3"


def test_html_report_generator_chs_version_from_run_info():
    generator = HtmlReportGenerator()
    boh = pb.BillOfHealth()
    boh.report_id = "test-run"
    report_data = generator.generate_report_data(
        boh, run_info={"chs_version": "2.3.4"}
    )
    assert report_data["metadata"]["chs_version"] == "2.3.4"


def test_html_report_generator_chs_version_fallback():
    generator = HtmlReportGenerator()
    boh = pb.BillOfHealth()
    boh.report_id = "test-run"
    with mock.patch("chs.html_generator.get_version", return_value="9.9.9"):
        report_data = generator.generate_report_data(boh)
        assert report_data["metadata"]["chs_version"] == "9.9.9"


def test_html_report_generator_multinode_metric_bucketing():
    """Multi-node results without custom filter assign per-node metrics."""
    generator = HtmlReportGenerator()
    boh = pb.BillOfHealth()
    boh.report_id = "test-multinode"

    res = boh.test_results.add(
        test_name="dcgm",
        test_level="node",
        result_status=pb.RESULT_STATUS_FAILED,
    )
    res.target_resources.add(node_name="node-a")
    res.target_resources.add(node_name="node-b")

    # node-a passes, node-b fails threshold
    m_a = res.metrics.add(key="Perf")
    m_a.value.number_value = 100.0
    m_a.metadata.fields["context"].string_value = "node-a"
    m_a.validation_criteria.name = "perf"
    m_a.validation_criteria.result_status = pb.RESULT_STATUS_SUCCESS

    m_b = res.metrics.add(key="Perf")
    m_b.value.number_value = 10.0
    m_b.metadata.fields["context"].string_value = "node-b"
    m_b.validation_criteria.name = "perf"
    m_b.validation_criteria.result_status = pb.RESULT_STATUS_FAILED

    report_data = generator.generate_report_data(boh)
    insights = {n["node_name"]: n for n in report_data["node_insights"]}

    assert insights["node-a"]["status"] == "INCOMPLETE"
    assert insights["node-a"]["tests"][0]["status_str"] == "✕ FAILED TO EXECUTE"
    assert insights["node-b"]["status"] == "DEGRADED"
    assert (
        insights["node-b"]["tests"][0]["status_str"]
        == "✕ DID NOT MEET THRESHOLD"
    )


def test_html_report_generator_custom_filter_runs_once_per_result():
    """Custom _filter_validation_results is invoked once per TestResult."""
    generator = HtmlReportGenerator()
    boh = pb.BillOfHealth()
    boh.report_id = "test-custom-filter"

    res = boh.test_results.add(
        test_name="tpu-collectives",
        test_level="slice",
        result_status=pb.RESULT_STATUS_SUCCESS,
    )
    for i in range(4):
        res.target_resources.add(node_name=f"node-{i}")
    m = res.metrics.add(key="Achieved Bandwidth", unit="GB/s")
    m.value.number_value = 350.0
    m.validation_criteria.name = "efficiency"
    m.validation_criteria.result_status = pb.RESULT_STATUS_SUCCESS

    with (
        mock.patch("chs.html_generator.aggregate_boh_summary") as mock_agg,
        mock.patch(
            "chs.html_generator.extract_validation_data", return_value=[]
        ),
        mock.patch.object(
            generator, "_extract_detailed_reports", return_value=[]
        ),
    ):
        mock_agg.return_value = mock.Mock(
            run_id="r1",
            test_summaries=[],
            errors_and_warnings=[],
        )
        with mock.patch(
            "chs.tests.tpu_collectives.renderer.TpuCollectivesRenderer"
            "._filter_validation_results",
            side_effect=lambda results, nodes: results,
        ) as mock_filter:
            report_data = generator.generate_report_data(boh)

    assert len(report_data["node_insights"]) == 4
    mock_filter.assert_called_once()


def test_html_report_generator_multinode_prefix_boundary():
    """Failing metric on node-10 must not mark node-1 as DEGRADED."""
    generator = HtmlReportGenerator()
    boh = pb.BillOfHealth()
    boh.report_id = "test-prefix"

    res = boh.test_results.add(
        test_name="dcgm",
        test_level="node",
        result_status=pb.RESULT_STATUS_SUCCESS,
    )
    res.target_resources.add(node_name="node-1")
    res.target_resources.add(node_name="node-10")

    m_ok = res.metrics.add(key="Perf")
    m_ok.value.number_value = 100.0
    m_ok.metadata.fields["context"].string_value = "host:node-1"
    m_ok.validation_criteria.name = "perf"
    m_ok.validation_criteria.result_status = pb.RESULT_STATUS_SUCCESS

    m_fail = res.metrics.add(key="Perf")
    m_fail.value.number_value = 10.0
    m_fail.metadata.fields["context"].string_value = "host:node-10"
    m_fail.validation_criteria.name = "perf"
    m_fail.validation_criteria.result_status = pb.RESULT_STATUS_FAILED

    report_data = generator.generate_report_data(boh)
    insights = {n["node_name"]: n for n in report_data["node_insights"]}
    assert insights["node-1"]["status"] == "HEALTHY"
    assert insights["node-10"]["status"] == "DEGRADED"


def test_html_report_generator_handles_unrecognized_test_name():
    """Unrecognized test names (resolve_test_name -> None) do not crash."""
    generator = HtmlReportGenerator()
    boh = pb.BillOfHealth()
    boh.report_id = "test-unknown-plugin"

    res = boh.test_results.add(
        test_name="custom-unregistered-test",
        test_level="node",
        result_status=pb.RESULT_STATUS_SUCCESS,
    )
    res.target_resources.add(node_name="node-1")
    m = res.metrics.add(key="Throughput")
    m.value.number_value = 42.0
    m.validation_criteria.name = "throughput"
    m.validation_criteria.result_status = pb.RESULT_STATUS_SUCCESS

    with (
        mock.patch("chs.html_generator.aggregate_boh_summary") as mock_agg,
        mock.patch(
            "chs.html_generator.extract_validation_data", return_value=[]
        ),
        mock.patch.object(
            generator, "_extract_detailed_reports", return_value=[]
        ),
    ):
        mock_agg.return_value = mock.Mock(
            run_id="r1",
            test_summaries=[],
            errors_and_warnings=[],
        )
        report_data = generator.generate_report_data(boh)
    insights = {n["node_name"]: n for n in report_data["node_insights"]}
    assert insights["node-1"]["status"] == "HEALTHY"
    assert (
        insights["node-1"]["tests"][0]["display_name"]
        == "custom-unregistered-test (node)"
    )


def _timed_out_nccl_target(test_name: str, machine_type: str) -> str:
    boh = pb.BillOfHealth(report_id=f"test-{test_name}-{machine_type}")
    boh.system_info.metadata.fields["machine_type"].string_value = machine_type
    res = boh.test_results.add(
        test_name=test_name,
        test_level="all-reduce",
        result_status=pb.RESULT_STATUS_TIMEOUT,
    )
    res.target_resources.add(node_name="node-0")
    res.target_resources.add(node_name="node-1")
    summary = aggregate_boh_summary(boh)
    assert summary is not None
    return summary.test_summaries[0].target_criteria


def test_fallback_target_skips_rules_the_report_cannot_disambiguate():
    """Rules gated on run conditions absent from BoH (domain_type) skip."""
    # a4-highgpu-8g has an unconditional all-reduce rule: 642 * 0.95 -> 610.
    assert (
        _timed_out_nccl_target("nccl-pairwise", "a4-highgpu-8g")
        == "Target: Peak Detected Bandwidth >= 610 GB/s"
    )
    # a4x-highgpu-4g rules are domain_type-gated (and nccl-cluster only
    # validates intradomain, so interdomain runs have no threshold).
    assert _timed_out_nccl_target("nccl-pairwise", "a4x-highgpu-4g") == ""
    assert _timed_out_nccl_target("nccl-cluster", "a4x-highgpu-4g") == ""


def test_html_summary_reads_slice_topology_from_boh_system_info():
    """Slice topology populated onto boh.system_info drives target fallback."""
    boh = pb.BillOfHealth(report_id="test-sysinfo-topo")
    boh.system_info.metadata.fields[
        "machine_type"
    ].string_value = "tpu7x-standard-4t"
    res = boh.test_results.add(
        test_name="tpu-collectives",
        test_level="psum",
        result_status=pb.RESULT_STATUS_TIMEOUT,
    )
    res.target_resources.add(node_name="node-0")
    builder.populate_system_entities(
        boh, {"node-0": ["c0"]}, [{"topology": "4x4x4", "nodes": ["node-0"]}]
    )

    summary = aggregate_boh_summary(boh)
    assert summary is not None
    assert (
        summary.test_summaries[0].target_criteria
        == "Target: Peak Bandwidth (median) >= 352 GB/s"
    )


def test_mixed_topologies_resolve_per_test_group_in_html():
    """Sub-slice 2x2x1 in a run does not leak onto a 4x4x4 collectives run."""
    boh = pb.BillOfHealth(report_id="test-mixed-topo-html")
    boh.system_info.metadata.fields[
        "machine_type"
    ].string_value = "tpu7x-standard-4t"
    psum = boh.test_results.add(
        test_name="tpu-collectives",
        test_level="psum",
        result_status=pb.RESULT_STATUS_FAILED,
    )
    psum.target_resources.add(node_name="cube-0")
    psum.target_resources.add(node_name="cube-1")
    info = boh.test_results.add(
        test_name="tpu-info",
        test_level="local",
        result_status=pb.RESULT_STATUS_SUCCESS,
    )
    info.target_resources.add(node_name="sub-0")
    builder.populate_system_entities(
        boh,
        {n: [f"{n}-c0"] for n in ("cube-0", "cube-1", "sub-0")},
        [{"topology": "4x4x4", "nodes": ["cube-0", "cube-1"]}],
        {"sub-0": {"tpu_topology": "2x2x1"}},
    )

    summary = aggregate_boh_summary(boh)
    assert summary is not None
    by_name = {
        s.display_name: s.target_criteria for s in summary.test_summaries
    }
    assert (
        by_name["tpu-collectives (psum)"]
        == "Target: Peak Bandwidth (median) >= 352 GB/s"
    )


def test_metric_filter_single_vs_multi_rule_branch():
    """elements/context keep one metric rule and skip ambiguous multi-rules."""
    from chs.core.report_data import configured_thresholds

    # tpu-hbm has a single elements-gated rule for ct6e-standard-4t.
    assert configured_thresholds(
        "tpu-hbm", "check", machine_type="ct6e-standard-4t"
    ) == ["Bandwidth (median) >= 1045 GB/s"]
    # dcgm has multiple distinct context-gated rules; only Result stays.
    assert configured_thresholds(
        "dcgm", "level-2", machine_type="a3-megagpu-8g"
    ) == ["Result == Pass"]


def test_no_fallback_target_when_generator_evaluated_no_thresholds():
    """UNKNOWN results (no matching rule) must not borrow fallback targets."""
    boh = pb.BillOfHealth(report_id="test-no-thresh")
    boh.system_info.metadata.fields[
        "machine_type"
    ].string_value = "ct6e-standard-4t"
    res = boh.test_results.add(
        test_name="tpu-hbm",
        test_level="check",
        result_status=pb.RESULT_STATUS_UNKNOWN,
    )
    res.target_resources.add(node_name="node-0")
    m = res.metrics.add(key="Bandwidth (median)", unit="GB/s")
    m.value.number_value = 1200.0
    m.metadata.fields["context"].string_value = "node-0,1048576,bfloat16"

    report_data = HtmlReportGenerator().generate_report_data(boh)
    card = report_data["test_suite_breakdown"][0]
    assert "No thresholds set" in card["ratio_str"]
    assert card["target_criteria"] == ""

    # A host that hit DEADLINE was never evaluated, so the fallback applies.
    other = boh.test_results.add(
        test_name="tpu-hbm",
        test_level="check",
        result_status=pb.RESULT_STATUS_DEADLINE,
    )
    other.target_resources.add(node_name="node-1")
    summary = aggregate_boh_summary(boh)
    assert summary is not None
    assert (
        summary.test_summaries[0].target_criteria
        == "Target: Bandwidth (median) >= 1045 GB/s"
    )
