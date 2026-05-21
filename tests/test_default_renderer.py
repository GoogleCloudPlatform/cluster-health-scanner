"""Unit tests for the DefaultRenderer class."""

import unittest
from chs.core.default_plugin.renderer import DefaultRenderer
from chs.protos import bill_of_health_pb2 as pb
from rich.console import Console


class TestDefaultRenderer(unittest.TestCase):
    def test_render_detail_without_thresholds(self):
        renderer = DefaultRenderer()

        res = pb.TestResult()
        res.test_name = "test-default"
        res.test_level = "L1"

        m1 = res.metrics.add()
        m1.key = "metric_without_threshold_1"
        m1.value.number_value = 120.5
        m1.unit = "GB/s"

        m2 = res.metrics.add()
        m2.key = "metric_without_threshold_2"
        m2.value.string_value = "ok"

        console = Console(width=200)
        with console.capture() as capture:
            detail = renderer.render_detail([res])
            console.print(detail)

        output = capture.get()

        # Both metrics should be displayed since there is no validation
        self.assertIn("metric_without_threshold_1", output)
        self.assertIn("120.5", output)
        self.assertIn("metric_without_threshold_2", output)
        self.assertIn("ok", output)

        # Threshold and Status columns should not be printed
        self.assertNotIn("Threshold", output)
        self.assertNotIn("Status", output)

    def test_render_detail_with_thresholds(self):
        renderer = DefaultRenderer()

        res = pb.TestResult()
        res.test_name = "test-default"
        res.test_level = "L1"

        m1 = res.metrics.add()
        m1.key = "metric_with_threshold"
        m1.value.number_value = 85.0
        m1.unit = "GB/s"
        m1.validation_criteria.name = "test_threshold"
        m1.validation_criteria.expected_value.number_value = 80.0
        m1.validation_criteria.comparator = pb.Comparator.COMPARATOR_GTE
        m1.validation_criteria.result_status = pb.RESULT_STATUS_SUCCESS

        m2 = res.metrics.add()
        m2.key = "metric_without_threshold"
        m2.value.number_value = 99.9
        m2.unit = "GB/s"

        console = Console(width=200)
        with console.capture() as capture:
            detail = renderer.render_detail([res])
            console.print(detail)

        output = capture.get()

        # Metric with threshold should be displayed
        self.assertIn("metric_with_threshold", output)
        self.assertIn("85", output)
        self.assertIn(">= 80", output)
        self.assertIn("PASS", output)

        # Metric without threshold should be filtered out
        self.assertNotIn("metric_without_threshold", output)
        self.assertNotIn("99.9", output)

        # Threshold and Status columns should be printed
        self.assertIn("Threshold", output)
        self.assertIn("Status", output)


if __name__ == "__main__":
    unittest.main()
