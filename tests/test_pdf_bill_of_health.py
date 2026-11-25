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

"""Unit and scenario tests for the PDF Bill of Health."""

import json
import math
import pathlib
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from chs.html_generator import HtmlReportGenerator
from chs.pdf_generator import (
    PdfReportGenerator,
    builder,
    labels,
    layout,
    model,
    pdfwriter,
    report,
)
from chs.pdf_generator.hardware_ids import (
    BLOCK_SCOPE,
    NO_IDS,
    SUBBLOCK_SCOPE,
    AcceleratorIds,
)
from chs.pdf_generator.pdfwriter import A3_LANDSCAPE
from chs.protos import bill_of_health_pb2 as pb
from chs.ramble_manager import RambleManager
from chs.schedulers.kubernetes import KubernetesScheduler


def _add_result(
    boh: pb.BillOfHealth,
    test_name: str,
    nodes: list[str],
    *,
    test_level: str = "",
    status: int = pb.RESULT_STATUS_SUCCESS,
) -> pb.TestResult:
    res = boh.test_results.add(
        test_name=test_name,
        test_level=test_level,
        result_status=status,
    )
    for node in nodes:
        res.target_resources.add(node_name=node)
    return res


def _add_gte_metric(
    res: pb.TestResult,
    key: str,
    value: float,
    threshold: float,
    *,
    unit: str = "GB/s",
    criteria_name: str = "bandwidth",
) -> None:
    metric = res.metrics.add(key=key, unit=unit)
    metric.value.number_value = value
    vc = metric.validation_criteria
    vc.name = criteria_name
    vc.comparator = pb.COMPARATOR_GTE
    vc.expected_value.number_value = threshold
    vc.result_status = (
        pb.RESULT_STATUS_SUCCESS
        if math.isfinite(value) and value >= threshold
        else pb.RESULT_STATUS_FAILED
    )


def _new_boh(
    report_id: str = "run-123", machine_type: str = ""
) -> pb.BillOfHealth:
    boh = pb.BillOfHealth(report_id=report_id, chs_version="0.4.0")
    if machine_type:
        boh.system_info.metadata.fields[
            "machine_type"
        ].string_value = machine_type
    return boh


def _render_and_validate(
    tmp_path,
    boh: pb.BillOfHealth,
    hardware_ids: dict[str, list[str]] | None = None,
    tpu_slices: list[dict] | None = None,
    host_metadata: dict[str, dict[str, str]] | None = None,
    name: str = "bill_of_health.pdf",
) -> tuple[bytes, str]:
    out = tmp_path / name
    PdfReportGenerator(hardware_ids, tpu_slices, host_metadata).save_pdf(
        boh, str(out)
    )
    pdf_bytes = out.read_bytes()
    assert pdf_bytes.startswith(b"%PDF-1.4")
    assert b"%%EOF" in pdf_bytes[-32:]
    if shutil.which("gs"):
        gs_res = subprocess.run(
            [
                "gs",
                "-dNOPAUSE",
                "-dBATCH",
                "-sDEVICE=nullpage",
                str(out),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert gs_res.returncode == 0, gs_res.stderr
    text = ""
    if shutil.which("pdftotext"):
        txt_res = subprocess.run(
            ["pdftotext", str(out), "-"],
            capture_output=True,
            text=True,
            check=True,
        )
        text = txt_res.stdout
    return pdf_bytes, text


class TestPdfBillOfHealth(unittest.TestCase):
    def setUp(self):
        self.tmp_path = pathlib.Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp_path, ignore_errors=True)

    def test_hardware_ids_deterministic_and_scope_separated(self):
        ids_a = AcceleratorIds({"tpu-1": ["c1", "c0"], "tpu-2": ["c2"]})
        ids_b = AcceleratorIds({"tpu-1": ["c0", "c1"], "tpu-2": ["c2"]})

        assert ids_a.host_id("tpu-1") == ids_b.host_id("tpu-1")
        assert len(ids_a.host_id("tpu-1")) == 32
        assert ids_a.host_id("tpu-missing") == NO_IDS
        assert ids_a.host_id("tpu-missing") == "Unavailable"
        # Group digests differ from 1-host HOST_SCOPE and from one another.
        sub_id = ids_a.group_id(SUBBLOCK_SCOPE, ["tpu-1"])
        blk_id = ids_a.group_id(BLOCK_SCOPE, ["tpu-1"])
        assert len({ids_a.host_id("tpu-1"), sub_id, blk_id}) == 3
        assert (
            ids_a.group_id(SUBBLOCK_SCOPE, ["tpu-1", "tpu-missing"])
            == "Unavailable"
        )

    def test_tpu_subblock_block_slice_and_single_host_grouping(self):
        subblock_nodes = [f"sub-n{i:02d}" for i in range(16)]
        small_nodes = [f"small-n{i:02d}" for i in range(4)]
        single_node = "single-n00"
        all_nodes = subblock_nodes + small_nodes + [single_node]

        boh = _new_boh("tpu-mixed", "tpu7x-standard-4t")
        for n in all_nodes:
            res = _add_result(boh, "tpu-hbm", [n], test_level="check")
            _add_gte_metric(res, "Bandwidth (median)", 3300.0, 3200.0)
        col_sub = _add_result(
            boh, "tpu-collectives", subblock_nodes, test_level="psum"
        )
        _add_gte_metric(col_sub, "Peak Bandwidth (median)", 395.0, 370.0)
        col_small = _add_result(
            boh, "tpu-collectives", small_nodes, test_level="psum"
        )
        _add_gte_metric(col_small, "Peak Bandwidth (median)", 155.0, 140.0)

        hw_ids = {n: [f"{n}-c0", f"{n}-c1"] for n in all_nodes}
        tpu_slices = [
            {"topology": "4x4x4", "nodes": subblock_nodes},
            {"topology": "2x2x4", "nodes": small_nodes},
        ]

        rep = builder.build_report(boh, hw_ids, tpu_slices)
        assert [s.title for s in rep.sections] == [
            "CHS Subblock",
            "CHS Slice (2x2x4)",
            "Hosts",
        ]
        assert rep.sections[0].group_id_label == "CHS Subblock ID"
        assert rep.sections[1].group_id_label is None
        assert rep.sections[1].group_id is None
        assert rep.sections[2].group_id_label is None

        colors = report._test_colors(rep)
        sub_strip = report._section(rep.sections[0], colors)[1]
        assert isinstance(sub_strip, report._Strip)
        assert sub_strip.pairs[:3] == [
            ("Topology", report.Cell("4x4x4", "bold")),
            ("CHS Subblock ID", report.Cell(rep.sections[0].group_id, "bold")),
            ("Hosts", report.Cell("16", "bold")),
        ]
        slice_strip = report._section(rep.sections[1], colors)[1]
        assert isinstance(slice_strip, report._Strip)
        assert all(not k.endswith(" ID") for k, _ in slice_strip.pairs)
        hosts_strip = report._section(rep.sections[2], colors)[1]
        assert isinstance(hosts_strip, report._Strip)
        assert all(
            not k.endswith(" ID") and k != "Status"
            for k, _ in hosts_strip.pairs
        )

        _, text = _render_and_validate(self.tmp_path, boh, hw_ids, tpu_slices)
        if text:
            assert "CHS Subblock" in text
            assert "CHS Slice (2x2x4)" in text
            assert "Slice ID" not in text
            assert "Evaluated metrics per host and multi-host" in text
            assert "host pair" not in text.lower()
            assert "Issues found" not in text

        # On v6e (ct6e-standard-4t), 16x16 is Block and other shapes are Slice.
        v6e_nodes = [f"v6e-b{i:02d}" for i in range(4)]
        v6e_other = [f"v6e-s{i:02d}" for i in range(4)]
        v6e_boh = _new_boh("v6e-block", "ct6e-standard-4t")
        for n in v6e_nodes + v6e_other:
            _add_gte_metric(
                _add_result(v6e_boh, "tpu-gemm", [n], test_level="check"),
                "Throughput",
                900.0,
                850.0,
                unit="TFLOP/s/device",
            )
        v6e_slices = [
            {"topology": "16x16", "nodes": v6e_nodes},
            {"topology": "8x8", "nodes": v6e_other},
        ]
        v6e_rep = builder.build_report(
            v6e_boh, {n: [f"{n}-c0"] for n in v6e_nodes + v6e_other}, v6e_slices
        )
        assert [s.title for s in v6e_rep.sections] == [
            "CHS Block",
            "CHS Slice (8x8)",
        ]
        assert v6e_rep.sections[0].group_id_label == "CHS Block ID"
        assert v6e_rep.sections[1].group_id_label is None
        assert v6e_rep.sections[1].group_id is None

    def test_kubernetes_scheduler_hardware_inventory(self):
        nodes_raw = (
            [
                {
                    "metadata": {
                        "name": f"cube-{i}",
                        "labels": {
                            "cloud.google.com/gke-nodepool": "pool-cube",
                            "cloud.google.com/gke-tpu-accelerator": "tpu7x",
                            "cloud.google.com/gke-tpu-topology": "4x4x4",
                        },
                        "annotations": {
                            "node.gke.io/tpu-chip-ids": f"c{i}a, c{i}b"
                        },
                    }
                }
                for i in range(16)
            ]
            + [
                {
                    "metadata": {
                        "name": f"small-{i}",
                        "labels": {
                            "cloud.google.com/gke-nodepool": "pool-small",
                            "cloud.google.com/gke-tpu-accelerator": "tpu7x",
                            "cloud.google.com/gke-tpu-topology": "2x2x4",
                        },
                        "annotations": {"node.gke.io/tpu-chip-ids": f"s{i}"},
                    }
                }
                for i in range(4)
            ]
            + [
                {
                    "metadata": {
                        "name": "dyn-0",
                        "labels": {
                            "cloud.google.com/gke-nodepool": "pool-dyn",
                            "cloud.google.com/gke-tpu-accelerator": "tpu7x",
                            "cloud.google.com/gke-tpu-topology": "2x2x1",
                        },
                        "annotations": {"node.gke.io/tpu-chip-ids": "d0"},
                    }
                }
            ]
        )
        sched = KubernetesScheduler()
        sched._subslice_nodes = {"dyn-0"}
        with (
            mock.patch.object(
                sched, "_get_node_objects", return_value=nodes_raw
            ),
            mock.patch.object(
                sched, "_fetch_raw_node_resources", return_value=nodes_raw
            ),
        ):
            ids = sched.get_accelerator_ids(
                [n["metadata"]["name"] for n in nodes_raw]
            )
            assert ids["cube-0"] == ["c0a", "c0b"]
            assert ids["dyn-0"] == ["d0"]

            slices = sched.get_tpu_slices_tested(
                [n["metadata"]["name"] for n in nodes_raw]
            )
            assert [s["topology"] for s in slices] == ["4x4x4", "2x2x4"]
            assert len(slices[0]["nodes"]) == 16

            meta = sched.get_host_metadata(
                [n["metadata"]["name"] for n in nodes_raw]
            )
            assert meta["cube-0"]["nodepool"] == "pool-cube"
            assert meta["small-0"]["nodepool"] == "pool-small"
            assert meta["dyn-0"]["nodepool"] == "pool-dyn"

            # Partial 4x4x4 targeting drops the pool from tpu_slices.json.
            partial = sched.get_tpu_slices_tested(
                [f"cube-{i}" for i in range(15)]
                + [f"small-{i}" for i in range(4)]
            )
            assert [s["topology"] for s in partial] == ["2x2x4"]

        # Incomplete 4x4x4 pool missing TPU resources on 1 node is partial.
        incomplete_pool = [
            {
                **nodes_raw[i],
                "status": {"allocatable": {"google.com/tpu": "4"}},
            }
            for i in range(15)
        ]
        sched_incomplete = KubernetesScheduler()
        with mock.patch.object(
            sched_incomplete,
            "_fetch_raw_node_resources",
            return_value=incomplete_pool,
        ):
            assert (
                sched_incomplete.get_tpu_slices_tested(
                    [f"cube-{i}" for i in range(15)]
                )
                == []
            )

        # Explicit --nodes runs test host-by-host and never report slices,
        # even when listing every node of a static pool.
        sched_nodes = KubernetesScheduler(
            nodes=",".join(f"cube-{i}" for i in range(16))
        )
        with mock.patch.object(
            sched_nodes, "_fetch_raw_node_resources", return_value=nodes_raw
        ):
            assert (
                sched_nodes.get_tpu_slices_tested(
                    [f"cube-{i}" for i in range(16)]
                )
                == []
            )

    def test_failure_modes_match_html_and_preserve_all_timeout_chart(self):
        s1_nodes = ["s1-n0", "s1-n1"]
        s2_nodes = ["s2-n0", "s2-n1"]
        boh = _new_boh("fail-run", "tpu7x-standard-4t")

        # Slice 1: s1-n0 passes, s1-n1 fails; s1-n0 has error + warning.
        r0 = _add_result(boh, "tpu-gemm", ["s1-n0"], test_level="check")
        _add_gte_metric(r0, "Throughput", 1040.0, 1000.0, unit="TFLOP/s/device")
        r0.metrics.add(
            key="TPU Hardware / Driver / Network Error Detected"
        ).value.string_value = "ICI link failure detected"
        r0.metrics.add(
            key="Runtime Warning Detected"
        ).value.string_value = "High temp 84C"

        r1 = _add_result(boh, "tpu-gemm", ["s1-n1"], test_level="check")
        _add_gte_metric(r1, "Throughput", 820.0, 1000.0, unit="TFLOP/s/device")

        # Slice 2: s2-n0 times out, s2-n1 cancelled; s2-n2 targeted but not run.
        _add_result(
            boh,
            "tpu-gemm",
            ["s2-n0"],
            test_level="check",
            status=pb.RESULT_STATUS_TIMEOUT,
        )
        _add_result(
            boh,
            "tpu-gemm",
            ["s2-n1"],
            test_level="check",
            status=pb.RESULT_STATUS_CANCELLED,
        )

        tpu_slices = [
            {"topology": "2x2x2", "nodes": s1_nodes},
            {"topology": "2x2x2", "nodes": [*s2_nodes, "s2-n2"]},
        ]
        hw_ids = {n: [f"{n}-0"] for n in [*s1_nodes, *s2_nodes, "s2-n2"]}

        rep = builder.build_report(boh, hw_ids, tpu_slices)
        html_data = HtmlReportGenerator().generate_report_data(boh)
        html_by_node = {n["node_name"]: n for n in html_data["node_insights"]}

        s1, s2 = rep.sections
        s1_rows = {r.name: r for r in s1.hosts}
        s2_rows = {r.name: r for r in s2.hosts}

        # HTML and PDF agree on every node that produced results.
        assert html_by_node["s1-n0"]["status"] == "HEALTHY"
        assert s1_rows["s1-n0"].status == "HEALTHY"
        assert html_by_node["s1-n1"]["status"] == "DEGRADED"
        assert s1_rows["s1-n1"].status == "DEGRADED"
        assert html_by_node["s2-n0"]["status"] == "INCOMPLETE"
        assert s2_rows["s2-n0"].status == "INCOMPLETE"
        assert s2_rows["s2-n2"].status == "INCOMPLETE"

        # Error/warning logged markers appear in s1-n0's tpu-gemm cell.
        s1_n0_texts = [p.text for p in s1_rows["s1-n0"].cells[0]]
        assert "Error logged" in s1_n0_texts
        assert "Warning logged" in s1_n0_texts

        # All-timeout Slice 2 still retains a chart with TIMEOUT/CANCELLED bars.
        assert len(s2.charts) == 1
        assert {p.text for p in s2.charts[0].points} == {
            "TIMEOUT",
            "CANCELLED",
            "INCOMPLETE",
        }

        _, text = _render_and_validate(self.tmp_path, boh, hw_ids, tpu_slices)
        if text:
            assert "Error logged" in text
            assert "Warning logged" in text
            assert "TIMEOUT" in text
            assert "INCOMPLETE" in text
            assert "Not Run" not in text

    def test_tpu_nan_handling_and_empty_boh_and_100_host_pagination(self):
        with self.assertRaisesRegex(ValueError, "report_id"):
            builder.build_report(pb.BillOfHealth(), {}, [])

        empty_boh = _new_boh("empty-run", "tpu7x-standard-4t")
        _render_and_validate(self.tmp_path, empty_boh, name="empty.pdf")

        nan_boh = _new_boh("tpu-nan", "tpu7x-standard-4t")
        _add_gte_metric(
            _add_result(nan_boh, "tpu-gemm", ["n0"], test_level="check"),
            "Throughput",
            math.nan,
            1000.0,
            unit="TFLOP/s/device",
        )
        nan_rep = builder.build_report(nan_boh, {"n0": ["c0"]}, [])
        assert nan_rep.sections[0].hosts[0].status == "DEGRADED"

        big_boh = _new_boh("scale-100", "tpu7x-standard-4t")
        for i in range(100):
            node = f"host-{i:03d}"
            res = _add_result(big_boh, "tpu-gemm", [node], test_level="check")
            _add_gte_metric(
                res,
                "Throughput",
                1040.0 if i != 99 else 820.0,
                1000.0,
                unit="TFLOP/s/device",
            )
            res.metrics.add(
                key="TPU Hardware / Driver / Network Error Detected"
            ).value.string_value = f"ICI link failure on {node}"
        _, text = _render_and_validate(
            self.tmp_path, big_boh, name="scale100.pdf"
        )
        if text:
            # Overview uses host noun and omits Potential errors and warnings.
            per_host_idx = text.index("Per-host results")
            events_idx = text.index("Potential errors and warnings")
            assert (
                text.index("Health summary by host") < per_host_idx < events_idx
            )
            assert "TOTAL HOSTS" in text
            assert "POTENTIAL ERRORS & WARNINGS\n0 detected" not in text
            assert "section" not in text.lower()

    def test_keep_with_next_does_not_orphan_heading_before_tall_table(self):
        tall_table = layout.Table(
            [layout.Column("Col", 1)],
            [["Line\n" * 6]] * 2,
        )
        heading = layout.Paragraph("Section Title", keep_with_next=True)
        # Spacer leaves ~68pt: fits heading + 60pt, not heading+header+row 1.
        spacer = layout.Paragraph("X\n" * 58)
        pdf_bytes = layout.build(
            [spacer, heading, tall_table],
            lambda _p, _n, _t: None,
            "orphan-check",
        )
        assert pdf_bytes.startswith(b"%PDF-1.4")
        # Heading and table must both move onto page 2 (/Count 2).
        assert b"/Count 2" in pdf_bytes

    def test_bugbash_multi_slice_mix_timeout_deadline_and_all_fail(self):
        cube_nodes = [f"c-{i:02d}" for i in range(16)]
        s1_nodes = [f"s1-{i}" for i in range(4)]
        s2_nodes = [f"s2-{i}" for i in range(2)]
        all_nodes = cube_nodes + s1_nodes + s2_nodes

        boh = _new_boh("bb-multi-slice", "tpu7x-standard-4t")
        for n in cube_nodes:
            _add_gte_metric(
                _add_result(boh, "tpu-gemm", [n], test_level="check"),
                "Throughput",
                1040.0,
                1000.0,
                unit="TFLOP/s/device",
            )
        # Slice 1 (2x2x4): mix of TIMEOUT, DEADLINE, and healthy (Bug Bash #6).
        _add_result(
            boh,
            "tpu-gemm",
            ["s1-0"],
            test_level="check",
            status=pb.RESULT_STATUS_TIMEOUT,
        )
        _add_result(
            boh,
            "tpu-gemm",
            ["s1-1"],
            test_level="check",
            status=pb.RESULT_STATUS_DEADLINE,
        )
        for n in ("s1-2", "s1-3"):
            _add_gte_metric(
                _add_result(boh, "tpu-gemm", [n], test_level="check"),
                "Throughput",
                1030.0,
                1000.0,
                unit="TFLOP/s/device",
            )
        # Slice 2 (2x2x2): all hosts miss threshold (Bug Bash #8).
        for n in s2_nodes:
            _add_gte_metric(
                _add_result(boh, "tpu-gemm", [n], test_level="check"),
                "Throughput",
                780.0,
                1000.0,
                unit="TFLOP/s/device",
            )

        tpu_slices = [
            {"topology": "4x4x4", "nodes": cube_nodes},
            {"topology": "2x2x4", "nodes": s1_nodes},
            {"topology": "2x2x2", "nodes": s2_nodes},
        ]
        hw_ids = {n: [f"{n}-c0"] for n in all_nodes}

        rep = builder.build_report(boh, hw_ids, tpu_slices)
        assert [s.title for s in rep.sections] == [
            "CHS Subblock",
            "CHS Slice 1 of 2 (2x2x4)",
            "CHS Slice 2 of 2 (2x2x2)",
        ]
        by_title = {s.title: s for s in rep.sections}
        s1_sec = by_title["CHS Slice 1 of 2 (2x2x4)"]
        s1_rows = {r.name: r for r in s1_sec.hosts}
        assert s1_rows["s1-0"].cells[0][0].text == "TIMEOUT"
        assert s1_rows["s1-1"].cells[0][0].text == "DEADLINE"
        s2_sec = by_title["CHS Slice 2 of 2 (2x2x2)"]
        s2_rows = {r.name: r for r in s2_sec.hosts}
        assert all(r.status == "DEGRADED" for r in s2_rows.values())

        _, text = _render_and_validate(self.tmp_path, boh, hw_ids, tpu_slices)
        if text:
            assert "CHS Slice 1 of 2 (2x2x4)" in text
            assert "CHS Slice 2 of 2 (2x2x2)" in text
            assert "TIMEOUT" in text
            assert "DEADLINE" in text

    def test_bugbash_tiering_potentially_degraded_vs_cleared(self):
        nodes = ["t-0", "t-1"]
        tpu_slices = [{"topology": "2x2x2", "nodes": nodes}]
        hw_ids = {n: [f"{n}-0"] for n in nodes}

        # Tier-1 (tpu-collectives) fails without Tier-2 -> POTENTIALLY DEGRADED.
        boh_unverified = _new_boh("tier-unverified", "tpu7x-standard-4t")
        col = _add_result(
            boh_unverified, "tpu-collectives", nodes, test_level="psum"
        )
        _add_gte_metric(col, "Peak Bandwidth (median)", 150.0, 200.0)

        html_unv = HtmlReportGenerator().generate_report_data(boh_unverified)
        assert all(
            n["status"] == "POTENTIALLY DEGRADED"
            for n in html_unv["node_insights"]
        )
        rep_unv = builder.build_report(boh_unverified, hw_ids, tpu_slices)
        assert all(
            r.status == "POTENTIALLY DEGRADED"
            for r in rep_unv.sections[0].hosts
        )
        _render_and_validate(
            self.tmp_path,
            boh_unverified,
            hw_ids,
            tpu_slices,
            name="tier_unv.pdf",
        )

        # Tier-2 (tpu-network-stress) passes on all nodes -> clears Tier-1!
        boh_cleared = _new_boh("tier-cleared", "tpu7x-standard-4t")
        col2 = _add_result(
            boh_cleared, "tpu-collectives", nodes, test_level="psum"
        )
        _add_gte_metric(col2, "Peak Bandwidth (median)", 150.0, 200.0)
        for n in nodes:
            t2 = _add_result(
                boh_cleared, "tpu-network-stress", [n], test_level="stress"
            )
            _add_gte_metric(t2, "Bandwidth", 175.0, 160.0)

        html_clr = HtmlReportGenerator().generate_report_data(boh_cleared)
        assert all(n["status"] == "HEALTHY" for n in html_clr["node_insights"])
        rep_clr = builder.build_report(boh_cleared, hw_ids, tpu_slices)
        assert all(r.status == "HEALTHY" for r in rep_clr.sections[0].hosts)
        _render_and_validate(
            self.tmp_path, boh_cleared, hw_ids, tpu_slices, name="tier_clr.pdf"
        )

    def test_bugbash_missing_metrics_all_and_partial(self):
        boh = _new_boh("bb-missing", "tpu7x-standard-4t")
        # Slice 1: SUCCESS status but 0 metrics -> MISSING_METRICS on all nodes.
        _add_result(boh, "tpu-gemm", ["m-0"], status=pb.RESULT_STATUS_SUCCESS)
        _add_result(boh, "tpu-gemm", ["m-1"], status=pb.RESULT_STATUS_SUCCESS)
        # Slice 2: partial missing metrics (p-0 empty, p-1 has metric).
        _add_result(boh, "tpu-gemm", ["p-0"], status=pb.RESULT_STATUS_SUCCESS)
        _add_gte_metric(
            _add_result(boh, "tpu-gemm", ["p-1"]), "Throughput", 1040.0, 1000.0
        )

        tpu_slices = [
            {"topology": "2x2x2", "nodes": ["m-0", "m-1"]},
            {"topology": "2x2x2", "nodes": ["p-0", "p-1"]},
        ]
        hw_ids = {n: [f"{n}-0"] for n in ("m-0", "m-1", "p-0", "p-1")}

        html_d = HtmlReportGenerator().generate_report_data(boh)
        assert html_d["summary"]["missing_metrics_count"] == 3
        rep = builder.build_report(boh, hw_ids, tpu_slices)
        s1_rows = {r.name: r for r in rep.sections[0].hosts}
        s2_rows = {r.name: r for r in rep.sections[1].hosts}
        assert s1_rows["m-0"].cells[0][0].text == "MISSING METRICS"
        assert s2_rows["p-0"].status == "INCOMPLETE"
        assert s2_rows["p-1"].status == "HEALTHY"
        _render_and_validate(self.tmp_path, boh, hw_ids, tpu_slices)

    def test_bugbash_skip_validation_and_jax_xla_errors(self):
        boh = _new_boh("bb-jax-err", "tpu7x-standard-4t")
        boh.metadata.fields["skip_validation"].bool_value = True

        r0 = _add_result(boh, "tpu-gemm", ["j-0"], test_level="check")
        r0.metrics.add(key="Throughput").value.number_value = 1040.0
        r0.metrics.add(
            key="TPU/XLA Compilation Error Detected"
        ).value.string_value = "XlaRuntimeError: RESOURCE_EXHAUSTED"
        r0.metrics.add(
            key="Runtime Warning Detected"
        ).value.string_value = "Slow HBM init"

        r1 = _add_result(
            boh,
            "tpu-gemm",
            ["j-1"],
            test_level="check",
            status=pb.RESULT_STATUS_FAILED,
        )
        r1.metrics.add(
            key="TPU/XLA Compilation Error Detected"
        ).value.string_value = "XlaRuntimeError: HBM OOM"

        rep = builder.build_report(boh, {"j-0": ["c0"], "j-1": ["c1"]}, [])
        assert "Threshold validation" not in dict(rep.run_info)
        assert any(
            e.category == "TPU/XLA Compilation" for e in rep.sections[0].events
        )
        hosts = {r.name: r for r in rep.sections[0].hosts}
        assert hosts["j-0"].status == "HEALTHY"
        assert [p.text for p in hosts["j-0"].cells[0]] == [
            "HEALTHY",
            "Error logged",
            "Warning logged",
        ]
        assert hosts["j-1"].status == "INCOMPLETE"
        assert [p.text for p in hosts["j-1"].cells[0]] == [
            "FAILED TO EXECUTE",
            "Error logged",
        ]

        _, text = _render_and_validate(
            self.tmp_path, boh, {"j-0": ["c0"], "j-1": ["c1"]}, []
        )
        if text:
            assert "THRESHOLD VALIDATION" not in text
            assert "TPU/XLA Compilation" in text
            assert "Error logged" in text
            assert "Warning logged" in text
            assert text.index("Per-host results") < text.index(
                "Potential errors and warnings"
            )

    def test_ramble_manager_pdf_save_and_upload_are_non_blocking(self):
        ramble_dir = self.tmp_path / "share" / "ramble"
        ramble_dir.mkdir(parents=True)
        (ramble_dir / "setup-env.sh").write_text("#!/bin/sh\n")

        ws_dir = self.tmp_path / "ws"
        ws_dir.mkdir()
        stale_pdf = ws_dir / "bill_of_health.pdf"
        stale_pdf.write_bytes(b"stale")
        (ws_dir / "bill_of_health.json").write_text("{}")
        (ws_dir / "bill_of_health.html").write_text("<html/>")

        with (
            mock.patch(
                "sysconfig.get_path", lambda *_, **__: str(self.tmp_path)
            ),
            mock.patch("chs.ramble_manager.ChsAppData", mock.MagicMock),
        ):
            manager = RambleManager(
                base_dir=str(self.tmp_path), workspace_name="ws"
            )

            # A failing PDF render removes stale PDFs, warns, and keeps going.
            with (
                mock.patch(
                    "chs.pdf_generator.PdfReportGenerator.from_workspace",
                    side_effect=RuntimeError("render boom"),
                ),
                self.assertLogs(level="WARNING") as save_logs,
            ):
                manager._save_pdf_report(pb.BillOfHealth(), ws_dir)
            assert not stale_pdf.exists()
            assert any(
                "Failed to generate the PDF Bill of Health" in msg
                for msg in save_logs.output
            )

            # When no PDF exists, _upload_boh succeeds without uploading PDF.
            provider = mock.Mock()
            json_dest, html_dest, ok = manager._upload_boh(provider)
            assert ok is True
            assert json_dest == "bills_of_health/ws.json"
            assert html_dest == "bills_of_health/ws.html"
            assert manager.boh_pdf_uploaded is False

            # Failing PDF upload warns, sets boh_pdf_uploaded=False, keeps ok.
            stale_pdf.write_bytes(b"%PDF-1.4")

            def _upload(src, dst):
                if dst.endswith(".pdf"):
                    raise RuntimeError("gcs boom")

            provider.upload_file.side_effect = _upload
            with self.assertLogs(level="WARNING") as upload_logs:
                _, _, ok = manager._upload_boh(provider)
            assert ok is True
            assert manager.boh_pdf_uploaded is False
            assert any(
                "Failed to archive the PDF Bill of Health" in msg
                for msg in upload_logs.output
            )

        # Core upload failure skips PDF upload even when the PDF file exists.
        provider2 = mock.Mock()
        provider2.upload_file.side_effect = RuntimeError("json boom")
        _, _, ok2 = manager._upload_boh(provider2)
        assert ok2 is False
        assert manager.boh_pdf_uploaded is False
        assert provider2.upload_file.call_count == 2

    def test_tpu_multi_subblock_and_scoped_status(self):
        c1 = [f"c1-{i:02d}" for i in range(16)]
        c2 = [f"c2-{i:02d}" for i in range(16)]
        boh = _new_boh("tpu-2subblocks", "tpu7x-standard-4t")
        for n in c1 + c2:
            _add_gte_metric(
                _add_result(boh, "tpu-gemm", [n], test_level="check"),
                "Throughput",
                1040.0,
                1000.0,
                unit="TFLOP/s/device",
            )
        _add_gte_metric(
            _add_result(boh, "tpu-collectives", c1, test_level="psum"),
            "Peak Bandwidth (median)",
            290.0,
            370.0,
        )
        _add_gte_metric(
            _add_result(boh, "tpu-collectives", c2, test_level="psum"),
            "Peak Bandwidth (median)",
            395.0,
            370.0,
        )
        tpu_slices = [
            {"topology": "4x4x4", "nodes": c1},
            {"topology": "4x4x4", "nodes": c2},
        ]
        hw_ids = {n: [f"{n}-0"] for n in c1 + c2}
        host_meta = {
            **{n: {"nodepool": "sub-pool-1"} for n in c1},
            **{n: {"nodepool": "sub-pool-2"} for n in c2},
        }
        rep = builder.build_report(boh, hw_ids, tpu_slices, host_meta)
        assert [s.title for s in rep.sections] == [
            "CHS Subblock 1 of 2",
            "CHS Subblock 2 of 2",
        ]
        # Subblock 1 fails (POTENTIALLY DEGRADED); subblock 2 is healthy.
        assert all(
            r.status == "POTENTIALLY DEGRADED" for r in rep.sections[0].hosts
        )
        assert all(r.status == "HEALTHY" for r in rep.sections[1].hosts)
        run_info = dict(rep.run_info)
        assert "Nodepool" not in run_info and "Nodepools" not in run_info
        assert rep.sections[0].pool_summary == "sub-pool-1"
        assert rep.sections[1].pool_summary == "sub-pool-2"

        # --nodes runs land on Hosts without a slice; no header strip Nodepool.
        nodes_rep = builder.build_report(boh, hw_ids, [], host_meta)
        assert [s.title for s in nodes_rep.sections] == ["Hosts"]
        assert nodes_rep.sections[0].pool_summary == ""
        nodes_strip = report._section(
            nodes_rep.sections[0], report._test_colors(nodes_rep)
        )[1]
        assert all(k != "Nodepool" for k, _ in nodes_strip.pairs)
        overview_flowables = report._overview(rep, report._test_colors(rep))
        health_strip = overview_flowables[3]
        assert [(k, cell.text) for k, cell in health_strip.pairs] == [
            ("Total subblocks", "2"),
            ("Healthy subblocks", "1"),
            ("Degraded subblocks", "0"),
            ("Potentially Degraded", "1"),
            ("Incomplete", "0"),
        ]
        _, text = _render_and_validate(
            self.tmp_path,
            boh,
            hw_ids,
            tpu_slices,
            host_meta,
            name="tpu_2subblocks.pdf",
        )
        if text:
            assert "CHS Subblock 1 of 2" in text
            assert "CHS Subblock 2 of 2" in text
            assert "Cross-Group" not in text
            assert "Issues found" not in text
            assert "Potential errors & warnings" not in text
            assert "Potential Errors & Warnings" not in text

    def test_title_uses_machine_type_only_and_shortens_long_node_names(self):
        nodes = [
            "gke-cluster-v7x-tpu7x-pool1-a1b2c3d4-zcdj",
            "gke-cluster-v7x-tpu7x-pool2-e5f6g7h8-9m2k",
            "gke-cluster-v7x-tpu7x-pool1-a1b2c3d4-pm3h",
        ]
        boh = _new_boh("tpu-title", "tpu7x-standard-4t")
        for n in nodes:
            info = _add_result(boh, "tpu-info", [n])
            info.metrics.add(
                key="Accelerator type"
            ).value.string_value = "TPU v7x Secret Test String"
            jax = info.metrics.add(key="JAX Devices")
            jax.value.number_value = 8
            vc = jax.validation_criteria
            vc.name = "JAX Devices"
            vc.comparator = pb.COMPARATOR_EQ
            vc.expected_value.number_value = 8
            vc.result_status = pb.RESULT_STATUS_SUCCESS
            gemm = _add_result(boh, "tpu-gemm", [n], test_level="check")
            _add_gte_metric(
                gemm, "Throughput", 1040.0, 1000.0, unit="TFLOP/s/device"
            )
        hw_ids = {n: [f"{n}-c{i}" for i in range(4)] for n in nodes}
        meta = {n: {"nodepool": "pool-a"} for n in nodes}

        rep = builder.build_report(boh, hw_ids, [], meta)
        assert rep.title == "tpu7x-standard-4t Bill of Health"
        assert "Secret Test String" not in rep.title
        # tpu-info JAX Devices renders bare "8" without status suffix.
        assert rep.sections[0].hosts[0].cells[0][0].text == "8"
        short_labels = [r.label for r in rep.sections[0].hosts]
        assert len(set(short_labels)) == 3
        assert all(len(lbl) <= 13 and "..." in lbl for lbl in short_labels)
        assert all(
            lbl != "H1" and not lbl.startswith("H") for lbl in short_labels
        )

        _, text = _render_and_validate(
            self.tmp_path, boh, hw_ids, [], meta, name="long_names.pdf"
        )
        if text:
            assert "tpu7x-standard-4t Bill of Health" in text
            for r in rep.sections[0].hosts:
                assert r.hardware_id in text

    def test_shorten_nodes_compacts_or_falls_back_on_collision(self):
        assert labels.shorten_nodes(["node-01", "node-02"]) == {
            "node-01": "node-01",
            "node-02": "node-02",
        }
        colliding = [
            "gke-cluster-v7x-tpu7x-pool1-a1b2c3d4-zcdj",
            "gke-cluster-v7x-tpu7x-pool2-a1b2c3d4-zcdj",
        ]
        assert labels.shorten_nodes(colliding) == {n: n for n in colliding}

    def test_ramble_manager_gates_pdf_to_tpu_only(self):
        fake_app = mock.MagicMock()
        ramble_dir = self.tmp_path / "share" / "ramble"
        ramble_dir.mkdir(parents=True)
        (ramble_dir / "setup-env.sh").write_text("#!/bin/sh\n")

        with (
            mock.patch(
                "sysconfig.get_path", lambda *_, **__: str(self.tmp_path)
            ),
            mock.patch("chs.ramble_manager.ChsAppData", lambda: fake_app),
        ):
            for name, mtype, expect_pdf in (
                ("tpu-ws", "tpu7x-standard-4t", True),
                ("gpu-ws", "a3-megagpu-8g", False),
            ):
                ws_dir = self.tmp_path / name
                ws_dir.mkdir()
                res_json = ws_dir / "results.latest.json"
                res_json.write_text(
                    json.dumps({"workspace_name": name, "experiments": []})
                )
                fake_app.get_run.return_value = {
                    "machine_type": mtype,
                }
                mgr = RambleManager(
                    base_dir=str(self.tmp_path), workspace_name=name
                )
                mgr.generate_boh(str(res_json))
                assert (ws_dir / "bill_of_health.json").exists()
                assert (ws_dir / "bill_of_health.html").exists()
                assert (ws_dir / "bill_of_health.pdf").exists() is expect_pdf

    def test_scale_100_subblocks_overview_exceptions_and_section_order(self):
        num_subblocks = 100
        boh = _new_boh("scale-100-subblocks", "tpu7x-standard-4t")
        slices = []
        hw_ids = {}
        meta = {}

        for c in range(num_subblocks):
            pool = f"pool-{c:03d}"
            c_nodes = [
                f"gke-hyper-v7x-tpu7x-{pool}-a1b2-{h:02d}" for h in range(16)
            ]
            slices.append({"topology": "4x4x4", "nodes": c_nodes})
            for n in c_nodes:
                hw_ids[n] = [f"{n}-chip-{i}" for i in range(4)]
                meta[n] = {"nodepool": pool}
                if c == 92 and n.endswith("-00"):
                    res = _add_result(
                        boh,
                        "tpu-gemm",
                        [n],
                        test_level="check",
                        status=pb.RESULT_STATUS_TIMEOUT,
                    )
                else:
                    res = _add_result(boh, "tpu-gemm", [n], test_level="check")
                    bad = c in (41, 77) and n.endswith("-00")
                    val = 840.0 if bad else 1040.0
                    _add_gte_metric(
                        res, "Throughput", val, 1000.0, unit="TFLOP/s/device"
                    )
                if c < 25:
                    res.metrics.add(
                        key="TPU Hardware / Driver / Network Error Detected"
                    ).value.string_value = f"ICI link failure in rack-{c:03d}"
            col = _add_result(
                boh, "tpu-collectives", c_nodes, test_level="psum"
            )
            _add_gte_metric(
                col,
                "Peak Bandwidth (median)",
                290.0 if c == 77 else 395.0,
                370.0,
                unit="GB/s",
            )

        rep = builder.build_report(boh, hw_ids, slices, meta)
        assert len(rep.sections) == 100
        assert [s.title for s in rep.sections[:3]] == [
            "CHS Subblock 1 of 100",
            "CHS Subblock 2 of 100",
            "CHS Subblock 3 of 100",
        ]

        _, text = _render_and_validate(
            self.tmp_path,
            boh,
            hw_ids,
            slices,
            meta,
            name="scale_100_subblocks.pdf",
        )
        if text:
            assert "Health summary by subblock" in text
            assert "TOTAL SUBBLOCKS" in text
            assert "HEALTHY SUBBLOCKS" in text
            assert "(1552 hosts)" not in text
            overview_end = text.index("CHS Subblock 1 of 100 — Results")
            assert "Potential errors and warnings" not in text[:overview_end]
            assert "CHS Subblock 78 of 100" not in text[:overview_end]
            assert "section" not in text.lower()
            assert (
                text.index("CHS Subblock 1 of 100 — Results")
                < text.index("CHS Subblock 42 of 100 — Results")
                < text.index("CHS Subblock 78 of 100 — Results")
            )
            assert "Issues found" not in text

    def test_scale_100_hosts_chunked_charts_and_events_after_host_table(self):
        nodes = [f"gke-hyperscale-v7x-nodeset-aaaa-{i:04d}" for i in range(100)]
        boh = _new_boh("scale-100-hosts", "tpu7x-standard-4t")
        hw_ids = {n: [f"{n}-c0", f"{n}-c1"] for n in nodes}
        meta = {n: {"nodepool": f"pool-{i // 25}"} for i, n in enumerate(nodes)}
        for i, n in enumerate(nodes):
            res = _add_result(boh, "tpu-gemm", [n], test_level="check")
            _add_gte_metric(
                res,
                "Throughput",
                820.0 if i < 20 else 1045.0,
                1000.0,
                unit="TFLOP/s/device",
            )
            res.metrics.add(
                key="TPU Hardware / Driver / Network Error Detected"
            ).value.string_value = "Hardware error: HBM ECC check"

        rep = builder.build_report(boh, hw_ids, [], meta)
        sec = rep.sections[0]
        assert sec.title == "Hosts"
        # 100 bars split cleanly into <=32-bar chunks (4 charts: 32, 32, 32, 4).
        assert len(sec.charts) == 4
        assert [len(c.points) for c in sec.charts] == [32, 32, 32, 4]
        assert "+98 more" in sec.events[0].hosts

        _, text = _render_and_validate(
            self.tmp_path, boh, hw_ids, [], meta, name="scale_100_hosts.pdf"
        )
        if text:
            assert "(1–32 of 100)" in text
            assert "(97–100 of 100)" in text
            assert text.index("Per-host results") < text.index(
                "Potential errors and warnings"
            )
            assert "Issues found" not in text

    def test_html_parity_columns_numbers_and_clean_cells(self):
        nodes = ["n-0", "n-1"]
        boh = _new_boh("parity-check", "tpu7x-standard-4t")
        for i, n in enumerate(nodes):
            info = _add_result(boh, "tpu-info", [n])
            jax = info.metrics.add(key="JAX Devices")
            jax.value.number_value = 8
            vc = jax.validation_criteria
            vc.name = "JAX Devices"
            vc.comparator = pb.COMPARATOR_EQ
            vc.expected_value.number_value = 8
            vc.result_status = pb.RESULT_STATUS_SUCCESS

            rhd = _add_result(boh, "tpu-host-device", [n], test_level="check")
            _add_gte_metric(
                rhd,
                "H2D Bandwidth",
                225.5 if i == 0 else 229.0,
                214.0,
                unit="GB/s",
                criteria_name="h2d_bandwidth",
            )
            _add_gte_metric(
                rhd,
                "D2H Bandwidth",
                222.45 if i == 0 else 226.0,
                211.0,
                unit="GB/s",
                criteria_name="d2h_bandwidth",
            )
        col = _add_result(boh, "tpu-collectives", nodes, test_level="psum")
        _add_gte_metric(
            col, "Peak Bandwidth (median)", 398.5, 370.0, unit="GB/s"
        )
        tpu_slices = [{"topology": "2x2x2", "nodes": nodes}]
        hw_ids = {n: [f"{n}-c0"] for n in nodes}

        rep = builder.build_report(boh, hw_ids, tpu_slices)
        sec = rep.sections[0]
        assert sec.columns == [
            "tpu-info (JAX Devices)",
            "tpu-host-device (check) · H2D Bandwidth (GB/s)",
            "tpu-host-device (check) · D2H Bandwidth (GB/s)",
            "tpu-collectives (psum) (GB/s)",
        ]
        n0 = next(r for r in sec.hosts if r.name == "n-0")
        assert n0.status == "HEALTHY"
        assert [c[0].text for c in n0.cells] == [
            "8",
            "225.50",
            "222.45",
            "398.50",
        ]

        _, text = _render_and_validate(self.tmp_path, boh, hw_ids, tpu_slices)
        if text:
            assert "8 (Healthy)" not in text
            assert "8 (HEALTHY)" not in text
            assert "(all)" not in text
            assert "225.50" in text
            assert "398.50" in text

    def test_id_and_status_columns_never_wrap_on_wide_tables(self):
        nodes = [
            f"gke-prod-v7x-chs-tpu7x-sub01-8c9d0e1f-{h:02d}ab" for h in range(4)
        ]
        boh = _new_boh("wide-nowrap", "tpu7x-standard-4t")
        for i, n in enumerate(nodes):
            for t_idx in range(12):
                res = _add_result(
                    boh, "tpu-gemm", [n], test_level=f"lvl-{t_idx:02d}"
                )
                res.target_resources[
                    0
                ].instance_id = f"59173845317882587{i:02d}"
                _add_gte_metric(
                    res, "Throughput", 1040.0, 1000.0, unit="TFLOP/s/device"
                )
        # Also make host 0 POTENTIALLY DEGRADED (widest status string).
        col = _add_result(boh, "tpu-collectives", nodes, test_level="psum")
        for i, tr in enumerate(col.target_resources):
            tr.instance_id = f"59173845317882587{i:02d}"
        _add_gte_metric(
            col, "Peak Bandwidth (median)", 290.0, 370.0, unit="GB/s"
        )

        hw_ids = {n: [f"{n}-c{i}" for i in range(4)] for n in nodes}
        meta = {n: {"nodepool": "tpu7x-sub-pool-01"} for n in nodes}
        slices = [{"topology": "2x2x4", "nodes": nodes}]
        rep = builder.build_report(boh, hw_ids, slices, meta)
        tbl = report._host_table(rep.sections[0])
        content_width = A3_LANDSCAPE[0] - 2 * layout.MARGIN
        widths = layout._column_widths(tbl.columns, content_width)
        idx_by_title = {c.title: idx for idx, c in enumerate(tbl.columns)}

        for col_name in ("GCE Instance ID", "CHS Host ID", "Status"):
            col_idx = idx_by_title[col_name]
            for _, slots, _ in tbl._layout(content_width):
                for cell in slots[col_idx]:
                    _, lines = tbl._lines(cell, widths[col_idx])
                    assert len(lines) == 1, f"{col_name} wrapped: {lines}"

        _render_and_validate(self.tmp_path, boh, hw_ids, slices, meta)

    def test_ramble_manager_inventory_gated_to_tpu(self):
        fake_app = mock.MagicMock()
        fake_app.get_images_dir.return_value = None
        ramble_dir = self.tmp_path / "share" / "ramble"
        ramble_dir.mkdir(parents=True)
        (ramble_dir / "setup-env.sh").write_text("#!/bin/sh\n")

        cases = (
            ("tpu-run", "tpu7x-standard-4t", 1),
            ("gpu-run", "a3-megagpu-8g", 0),
        )
        with (
            mock.patch(
                "sysconfig.get_path", lambda *_, **__: str(self.tmp_path)
            ),
            mock.patch("chs.ramble_manager.ChsAppData", lambda: fake_app),
        ):
            for name, mtype, expected_calls in cases:
                fake_app.get_run.return_value = {
                    "machine_type": mtype,
                }
                sched = mock.MagicMock()
                sched.name = "slurm"
                mgr = RambleManager(
                    base_dir=str(self.tmp_path), workspace_name=name
                )
                mgr.configure_workspace(sched, ["tpu-info"], machine_type=mtype)
                calls = sched.write_hardware_inventory.call_count
                assert calls == expected_calls

    def test_write_hardware_inventory_never_raises(self):
        sched = KubernetesScheduler()
        with mock.patch.object(
            sched,
            "get_nodes_under_test",
            side_effect=RuntimeError("lookup boom"),
        ):
            sched.write_hardware_inventory(str(self.tmp_path))
        assert list(self.tmp_path.iterdir()) == []

        with (
            mock.patch.object(
                sched, "get_accelerator_ids", return_value={"n-0": ["c0"]}
            ),
            mock.patch.object(
                sched, "get_host_metadata", side_effect=RuntimeError("boom")
            ),
            mock.patch.object(sched, "get_tpu_slices_tested", return_value=[]),
        ):
            sched.write_hardware_inventory(str(self.tmp_path), ["n-0"])
        assert sorted(p.name for p in self.tmp_path.iterdir()) == [
            "hardware_ids.json"
        ]

    def test_bar_chart_draws_values_for_wide_numbers_on_dense_subblock(self):
        nodes = [f"n{i:02d}" for i in range(16)]
        four_digit = [
            model.Point(n, float(6440 + i), str(6440 + i), False)
            for i, n in enumerate(nodes)
        ]
        chart_w = (A3_LANDSCAPE[0] - 2 * layout.MARGIN - 24) / 4
        page_4d = pdfwriter.Page(*A3_LANDSCAPE)
        layout.BarChart(
            "tpu-hbm-stress (stress)", "GB/s", 6430.0, four_digit
        ).draw(page_4d, layout.MARGIN, layout.TOP, chart_w)
        stream_4d = page_4d.content().decode("latin-1")
        for i in range(16):
            assert f"({6440 + i}) Tj" in stream_4d
        assert "(>=6430) Tj" in stream_4d

        five_digit = [
            model.Point(n, float(12340 + i), str(12340 + i), False)
            for i, n in enumerate(nodes)
        ]
        page_5d = pdfwriter.Page(*A3_LANDSCAPE)
        layout.BarChart("dense-5d", "GB/s", 12000.0, five_digit).draw(
            page_5d, layout.MARGIN, layout.TOP, chart_w
        )
        stream_5d = page_5d.content().decode("latin-1")
        for i in range(16):
            assert f"({12340 + i}) Tj" in stream_5d

    def test_ramble_manager_populates_boh_json_entities(self):
        fake_app = mock.MagicMock()
        ramble_dir = self.tmp_path / "share" / "ramble"
        ramble_dir.mkdir(parents=True)
        (ramble_dir / "setup-env.sh").write_text("#!/bin/sh\n")

        ws_dir = self.tmp_path / "tpu-json-ids"
        ws_dir.mkdir()
        res_json = ws_dir / "results.latest.json"
        res_json.write_text(
            json.dumps({"workspace_name": "tpu-json-ids", "experiments": []})
        )
        sub_nodes = [f"sub-{i:02d}" for i in range(4)]
        slice_nodes = [f"slc-{i:02d}" for i in range(2)]
        all_nodes = sub_nodes + slice_nodes
        hw_ids = {n: [f"{n}-c0", f"{n}-c1"] for n in all_nodes}
        tpu_slices = [
            {"topology": "4x4x4", "nodes": sub_nodes},
            {"topology": "2x2x4", "nodes": slice_nodes},
        ]
        host_meta = {
            **{
                n: {"nodepool": "pool-sub", "instance_id": f"id-{n}"}
                for n in sub_nodes
            },
            **{
                n: {"nodepool": "pool-slc", "instance_id": f"id-{n}"}
                for n in slice_nodes
            },
        }
        (ws_dir / "hardware_ids.json").write_text(json.dumps(hw_ids))
        (ws_dir / "tpu_slices.json").write_text(json.dumps(tpu_slices))
        (ws_dir / "host_metadata.json").write_text(json.dumps(host_meta))

        fake_app.get_run.return_value = {
            "machine_type": "tpu7x-standard-4t",
        }
        with (
            mock.patch(
                "sysconfig.get_path", lambda *_, **__: str(self.tmp_path)
            ),
            mock.patch("chs.ramble_manager.ChsAppData", lambda: fake_app),
        ):
            mgr = RambleManager(
                base_dir=str(self.tmp_path), workspace_name="tpu-json-ids"
            )
            mgr.generate_boh(str(res_json))

        boh_json = json.loads((ws_dir / "bill_of_health.json").read_text())
        entities = boh_json["system_info"]["entities"]
        ids = AcceleratorIds(hw_ids)
        assert entities[0]["type"] == "CHS Subblock"
        assert entities[0]["id"] == ids.group_id(SUBBLOCK_SCOPE, sub_nodes)
        assert entities[0]["metadata"]["topology"] == "4x4x4"
        assert entities[0]["metadata"]["nodepool"] == "pool-sub"
        assert len(entities[0]["sub_entities"]) == 4
        for sub_host in entities[0]["sub_entities"]:
            name = sub_host["metadata"]["node_name"]
            assert sub_host["type"] == "CHS Host"
            assert sub_host["id"] == ids.host_id(name)
            assert sub_host["metadata"]["instance_id"] == f"id-{name}"

        # Non-subblock/non-block slice hosts get CHS Host IDs with no Slice ID.
        ungrouped = entities[1:]
        assert [e["metadata"]["node_name"] for e in ungrouped] == sorted(
            slice_nodes, key=lambda n: ids.host_id(n)
        )
        for host in ungrouped:
            name = host["metadata"]["node_name"]
            assert host["type"] == "CHS Host"
            assert host["id"] == ids.host_id(name)
        assert all("Slice" not in e["type"] for e in entities)

    def test_network_stress_scope_is_slice_wide_despite_per_node_results(self):
        nodes = ["n-0", "n-1", "n-2", "n-3"]
        boh = _new_boh("net-stress-scope", "tpu7x-standard-4t")
        for n in nodes:
            _add_gte_metric(
                _add_result(boh, "tpu-hbm-stress", [n], test_level="stress"),
                "Bandwidth",
                6500.0,
                6430.0,
            )
            _add_gte_metric(
                _add_result(
                    boh, "tpu-network-stress", [n], test_level="stress"
                ),
                "Bandwidth",
                175.0,
                160.0,
            )
        _add_gte_metric(
            _add_result(boh, "tpu-collectives", nodes, test_level="psum"),
            "Peak Bandwidth (median)",
            390.0,
            370.0,
        )

        rep = builder.build_report(
            boh,
            {n: [f"{n}-c0"] for n in nodes},
            [{"topology": "4x4x4", "nodes": nodes}],
        )
        scopes = {t.name: t.scope for t in rep.tests}
        assert scopes == {
            "tpu-hbm-stress (stress)": "Single-Host",
            "tpu-network-stress (stress)": "Slice-Wide",
            "tpu-collectives (psum)": "Slice-Wide",
        }

    def test_overview_shows_configured_threshold_when_all_hosts_fail(self):
        nodes = ["n-0", "n-1", "n-2", "n-3"]
        boh = _new_boh("podkill-thresholds", "tpu7x-standard-4t")
        for n in nodes:
            _add_result(
                boh,
                "tpu-compute-stress",
                [n],
                test_level="stress",
                status=pb.RESULT_STATUS_FAILED,
            )
        _add_result(
            boh,
            "tpu-collectives",
            nodes,
            test_level="psum",
            status=pb.RESULT_STATUS_TIMEOUT,
        )

        rep = builder.build_report(
            boh,
            {n: [f"{n}-c0"] for n in nodes},
            [{"topology": "4x4x4", "nodes": nodes}],
        )
        by_name = {t.name: t.thresholds for t in rep.tests}
        assert by_name["tpu-compute-stress (stress)"] == ["Performance >= 998"]
        assert by_name["tpu-collectives (psum)"] == [
            "Peak Bandwidth (median) >= 352"
        ]

        boh.metadata.fields["skip_validation"].bool_value = True
        skipped_rep = builder.build_report(
            boh,
            {n: [f"{n}-c0"] for n in nodes},
            [{"topology": "4x4x4", "nodes": nodes}],
        )
        assert all(t.thresholds == [] for t in skipped_rep.tests)

    def test_hosts_page_omits_status_and_partial_degraded_group_is_potential(
        self,
    ):
        nodes_partial = [f"p-{i:02d}" for i in range(4)]
        nodes_full = [f"f-{i:02d}" for i in range(4)]
        boh = _new_boh("rollup-rules", "tpu7x-standard-4t")
        for i, n in enumerate(nodes_partial):
            _add_gte_metric(
                _add_result(boh, "tpu-gemm", [n], test_level="check"),
                "Throughput",
                800.0 if i == 0 else 1040.0,
                1000.0,
                unit="TFLOP/s/device",
            )
        for n in nodes_full:
            _add_gte_metric(
                _add_result(boh, "tpu-gemm", [n], test_level="check"),
                "Throughput",
                800.0,
                1000.0,
                unit="TFLOP/s/device",
            )

        hw_ids = {n: [f"{n}-c0"] for n in nodes_partial + nodes_full}
        slices = [
            {"topology": "4x4x4", "nodes": nodes_partial},
            {"topology": "4x4x4", "nodes": nodes_full},
        ]
        rep = builder.build_report(boh, hw_ids, slices)
        colors = report._test_colors(rep)

        partial_strip = dict(report._section(rep.sections[0], colors)[1].pairs)
        full_strip = dict(report._section(rep.sections[1], colors)[1].pairs)
        assert partial_strip["Status"].text == "POTENTIALLY DEGRADED"
        assert full_strip["Status"].text == "DEGRADED"

        overview_strip = dict(report._overview(rep, colors)[3].pairs)
        assert overview_strip["Degraded subblocks"].text == "1"
        assert overview_strip["Potentially Degraded"].text == "1"

        # --nodes / Hosts-only run omits page Status in its header strip.
        hosts_rep = builder.build_report(boh, hw_ids, [])
        assert [s.title for s in hosts_rep.sections] == ["Hosts"]
        hosts_strip = report._section(hosts_rep.sections[0], colors)[1]
        assert all(k != "Status" for k, _ in hosts_strip.pairs)

    def test_column_widths_never_collapse_and_scheduler_uses_system_type(self):
        # When min_width sums approach or exceed total_width, all widths > 0.
        cols = [
            layout.Column("A", 1.8, min_width=180),
            layout.Column("B", 1.8, min_width=180),
            layout.Column("C", 1.0, min_width=0),
            layout.Column("D", 1.0, min_width=0),
        ]
        for total in (400.0, 360.0, 200.0):
            widths = layout._column_widths(cols, total)
            assert len(widths) == 4
            assert all(w > 0 for w in widths)
            assert math.isclose(sum(widths), total, rel_tol=1e-6)

        # boh.system_info.type is primary source for scheduler detection.
        for sys_type, expected_sched in (
            ("SLURM_CLUSTER", "Slurm"),
            ("KUBERNETES_CLUSTER", "GKE"),
            ("STANDALONE_CLUSTER", "Standalone"),
        ):
            boh = _new_boh("sched-detect", "tpu7x-standard-4t")
            boh.system_info.type = sys_type
            _add_gte_metric(
                _add_result(boh, "tpu-hbm", ["n0"], test_level="check"),
                "Bandwidth (median)",
                3300.0,
                3200.0,
            )
            rep = builder.build_report(boh, {"n0": ["c0"]}, [])
            assert dict(rep.run_info)["Scheduler"] == expected_sched

    def test_overview_strip_shows_software_versions_when_reported(self):
        boh = _new_boh("versions", "tpu7x-standard-4t")
        boh.system_info.type = "KUBERNETES_CLUSTER"
        _add_gte_metric(
            _add_result(boh, "tpu-hbm", ["n0"], test_level="check"),
            "Bandwidth (median)",
            3300.0,
            3200.0,
        )
        labels = {"GKE version", "JAX version", "libtpu version"}

        rep = builder.build_report(boh, {"n0": ["c0"]}, [])
        assert not labels & set(dict(rep.run_info))

        fields = boh.system_info.metadata.fields
        fields["gke_version"].string_value = "1.36.4-gke.2377000"
        fields["jax_version"].string_value = "0.12.0"
        fields["libtpu_version"].string_value = "0.0.50"
        rep = builder.build_report(boh, {"n0": ["c0"]}, [])
        assert rep.run_info[-4:] == [
            ("Scheduler", "GKE"),
            ("GKE version", "1.36.4-gke.2377000"),
            ("JAX version", "0.12.0"),
            ("libtpu version", "0.0.50"),
        ]

    def test_review_edge_cases_columns_pulse_nan_strip_charts_and_overview(
        self,
    ):
        # 1. Mixed 4x4x4 + 2x2x4 tpu-collectives shares 1 deduped column.
        cube = [f"c-{i:02d}" for i in range(16)]
        small = [f"s-{i}" for i in range(4)]
        loose = ["d-0", "d-1"]
        boh = _new_boh("edge-cases", "tpu7x-standard-4t")
        _add_gte_metric(
            _add_result(boh, "tpu-collectives", cube, test_level="psum"),
            "Peak Bandwidth (median)",
            390.0,
            352.0,
        )
        _add_gte_metric(
            _add_result(boh, "tpu-collectives", small, test_level="psum"),
            "Peak Bandwidth (median)",
            150.0,
            133.0,
        )

        # 2. tpu-gemm-pulse: no stray HEALTHY, show failed bool, no nan.
        for n, ck_ok in (("d-0", True), ("d-1", False)):
            r = _add_result(boh, "tpu-gemm-pulse", [n], test_level="check")
            _add_gte_metric(
                r, "Throughput", 1010.0, 954.0, unit="TFLOP/s/device"
            )
            for k, val in (("Valid Checksum", True), ("Checksum Match", ck_ok)):
                m = r.metrics.add(key=k)
                m.value.bool_value = val
                vc = m.validation_criteria
                vc.name = k
                vc.comparator = pb.COMPARATOR_EQ
                vc.expected_value.bool_value = True
                vc.result_status = (
                    pb.RESULT_STATUS_SUCCESS if val else pb.RESULT_STATUS_FAILED
                )
        _add_gte_metric(
            _add_result(boh, "tpu-gemm", ["d-0"], test_level="check"),
            "Throughput",
            math.nan,
            950.0,
            unit="TFLOP/s/device",
        )

        hw = {n: [f"{n}-0"] for n in cube + small + loose}
        meta = {n: {"nodepool": "x" * 51} for n in cube}
        rep = builder.build_report(
            boh,
            hw,
            [
                {"topology": "4x4x4", "nodes": cube},
                {"topology": "2x2x4", "nodes": small},
            ],
            host_metadata=meta,
        )
        assert rep.sections[0].columns == ["tpu-collectives (psum) (GB/s)"]
        assert rep.sections[1].columns == ["tpu-collectives (psum) (GB/s)"]

        loose_hosts = {h.name: h for h in rep.sections[2].hosts}
        assert [p.text for p in loose_hosts["d-0"].cells[0]] == ["1010"]
        assert [p.text for p in loose_hosts["d-1"].cells[0]] == [
            "1010",
            "Checksum Match: False",
        ]
        assert [p.text for p in loose_hosts["d-0"].cells[1]] == ["DEGRADED"]

        # 3. Mixed groups + Hosts separates group vs ungrouped host summary.
        colors = report._test_colors(rep)
        ov_blocks = report._overview(rep, colors)
        headings = [
            f.text for f in ov_blocks if isinstance(f, layout.Paragraph)
        ]
        assert "Health summary by group" in headings
        assert "Health summary by ungrouped host" in headings
        group_strip = dict(ov_blocks[3].pairs)
        assert group_strip["Total groups"].text == "2"

        # 4. Long _Strip wraps cleanly across rows instead of splitting values.
        sec_strip = report._section(rep.sections[0], colors)[1]
        rows = sec_strip._rows(1119.0)
        assert len(rows) >= 2
        by_lbl = {lbl: lines for row in rows for lbl, _, lines, _ in row}
        assert by_lbl["CHS Subblock ID"] == [rep.sections[0].group_id]
        assert by_lbl["Status"] == ["HEALTHY"]

        # 5. BarChart clips angled ticks in max_w and wraps multi-status notes.
        long_names = [
            f"gke-hanu-tpu7x-cluster-np-tpu7x-static-c24a{i:02d}-ct9b"
            for i in range(16)
        ]
        tick = layout._fit_tick(long_names[0], 38.0)
        assert layout._TICK.width(tick) <= 38.0
        notes_chart = layout.BarChart(
            "empty",
            "GB/s",
            950.0,
            [
                model.Point("n0", None, "TIMEOUT", True),
                model.Point("n1", None, "DEADLINE", True),
                model.Point("n2", None, "FAILED TO EXECUTE", True),
                model.Point("n3", None, "CANCELLED", True),
            ],
        )
        pdf_bytes = layout.build(
            [notes_chart], lambda _p, _n, _t: None, "chart-wrap"
        )
        assert pdf_bytes.startswith(b"%PDF-1.4")
