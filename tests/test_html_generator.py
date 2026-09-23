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

from chs.html_generator import HtmlReportGenerator
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
