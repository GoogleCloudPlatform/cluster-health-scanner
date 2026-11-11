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

"""Tests that error-detector findings reach the reports.

The gemm, stream and tpu-collectives validation filters keep only their
headline metric. Errors must still be reported, and a gemm crash must stay
FAILED instead of turning into UNKNOWN.
"""

import unittest

from chs.core.summary_aggregator import aggregate_boh_summary
from chs.html_generator import HtmlReportGenerator
from chs.protos import bill_of_health_pb2 as pb
from chs.tests.gemm import GemmPlugin

NODE = "node-1"
MET = pb.RESULT_STATUS_SUCCESS


def _result(test_name, status, nodes=(NODE,), test_level="check"):
    res = pb.TestResult(
        test_name=test_name, test_level=test_level, result_status=status
    )
    for n in nodes:
        res.target_resources.add(node_name=n)
    return res


def _metric(res, key, number=None, context="", vc_status=None, text=None):
    m = res.metrics.add(key=key)
    if number is not None:
        m.value.number_value = number
    if text is not None:
        m.value.string_value = text
    m.metadata.fields["context"].string_value = context
    if vc_status is not None:
        m.validation_criteria.name = "performance"
        m.validation_criteria.comparator = pb.Comparator.COMPARATOR_GTE
        m.validation_criteria.expected_value.number_value = 1.0
        m.validation_criteria.result_status = vc_status
    return m


def _boh(*results):
    boh = pb.BillOfHealth(report_id="b581267742")
    boh.test_results.extend(results)
    return boh


def _events(boh):
    summary = aggregate_boh_summary(boh)
    return sorted((e.node, e.category) for e in summary.errors_and_warnings)


def _stream_with_xid():
    """Bug repro 2: passing bandwidth plus an Xid 31 ECC error."""
    res = _result("stream", pb.RESULT_STATUS_SUCCESS)
    _metric(res, "Achieved Bandwidth", 2900.0, "Triad GPU0", MET)
    _metric(res, "Achieved Bandwidth", 2800.0, "Triad GPU1", MET)
    _metric(res, "Hardware Error Detected", text="Xid 31")
    return res


def _gemm_oom(node="node-2", status=pb.RESULT_STATUS_FAILED):
    """Bug repro 1: gemm OOM crash with no performance numbers."""
    res = _result("gemm", status, nodes=(node,))
    _metric(res, "Host OOM Error Detected", text="out of memory")
    return res


def _tpu_collectives_with_errors():
    """Bug repro 3: contextless and non-first-node error metrics."""
    res = _result(
        "tpu-collectives",
        pb.RESULT_STATUS_SUCCESS,
        nodes=("node-0", "node-1"),
        test_level="all_gather",
    )
    _metric(res, "Peak Bandwidth (median)", 160.0, "node-0,peak,1", MET)
    _metric(res, "Peak Bandwidth (median)", 150.0, "node-1,peak,1", MET)
    _metric(res, "TPU OOM Error Detected", text="a")
    _metric(res, "Profiler Error Detected", context="node-1", text="b")
    return res


class ErrorEventsTest(unittest.TestCase):
    def test_stream_error_is_reported_and_test_still_passes(self):
        summary = aggregate_boh_summary(_boh(_stream_with_xid()))
        self.assertEqual(
            [(e.node, e.category) for e in summary.errors_and_warnings],
            [(NODE, "Hardware")],
        )
        self.assertEqual(summary.test_summaries[0].met_threshold, 1)

    def test_gemm_crash_error_is_reported(self):
        self.assertEqual(_events(_boh(_gemm_oom())), [("node-2", "Host OOM")])

    def test_tpu_collectives_errors_reported_for_every_node(self):
        self.assertEqual(
            _events(_boh(_tpu_collectives_with_errors())),
            [
                ("node-0", "Profiler"),
                ("node-0", "TPU OOM"),
                ("node-1", "Profiler"),
                ("node-1", "TPU OOM"),
            ],
        )

    def test_unfiltered_plugin_errors_reported_once(self):
        res = _result("nccl-cluster", pb.RESULT_STATUS_SUCCESS)
        _metric(res, "Network Error Detected", text="ibv timeout")
        self.assertEqual(_events(_boh(res)), [(NODE, "Network")])

    def test_html_report_shows_events_and_crash_status(self):
        data = HtmlReportGenerator().generate_report_data(
            _boh(_stream_with_xid(), _gemm_oom())
        )
        self.assertEqual(data["summary"]["potential_errors_count"], 2)
        nodes = {n["node_name"]: n for n in data["node_insights"]}
        self.assertTrue(nodes[NODE]["has_potential_errors"])
        self.assertEqual(nodes[NODE]["status"], "HEALTHY")
        self.assertEqual(
            nodes[NODE]["tests"][0]["status_str"], "✓ MET THRESHOLD"
        )
        self.assertTrue(nodes["node-2"]["has_potential_errors"])
        self.assertEqual(
            nodes["node-2"]["tests"][0]["status_str"], "✕ FAILED TO EXECUTE"
        )


class GemmCrashStatusTest(unittest.TestCase):
    def test_non_success_status_kept_without_median(self):
        for status in (
            pb.RESULT_STATUS_FAILED,
            pb.RESULT_STATUS_TIMEOUT,
            pb.RESULT_STATUS_CANCELLED,
            pb.RESULT_STATUS_DEADLINE,
        ):
            with self.subTest(status=pb.ResultStatus.Name(status)):
                out = GemmPlugin()._filter_validation_results(
                    [_gemm_oom(NODE, status)], [NODE]
                )
                self.assertEqual(out[0].result_status, status)

    def test_success_without_median_still_unknown(self):
        res = _result("gemm", pb.RESULT_STATUS_SUCCESS)
        _metric(res, "Value", 500.0, context="GPU0")
        out = GemmPlugin()._filter_validation_results([res], [NODE])
        self.assertEqual(out[0].result_status, pb.RESULT_STATUS_UNKNOWN)

    def test_median_still_decides_status(self):
        res = _result("gemm", pb.RESULT_STATUS_FAILED)
        _metric(res, "Value", 510.0, context="MEDIAN", vc_status=MET)
        out = GemmPlugin()._filter_validation_results([res], [NODE])
        self.assertEqual(out[0].result_status, pb.RESULT_STATUS_SUCCESS)


if __name__ == "__main__":
    unittest.main()
