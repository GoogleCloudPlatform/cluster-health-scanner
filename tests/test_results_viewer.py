"""Unit tests for the ResultsViewer class."""

import json
import logging
import os
import tempfile
import unittest
from unittest import mock

from chs.results_viewer import ResultsViewer
from rich.console import Console


class TestResultsViewer(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.valid_json_path = os.path.join(self.test_dir.name, "boh.json")
        self.invalid_json_path = os.path.join(
            self.test_dir.name, "invalid.json"
        )

        # Valid BOH data
        self.valid_data = {
            "report_id": "test-id",
            "chs_version": "1.0.0",
            "test_results": [
                {
                    "name": "test1",
                    "result_status": "RESULT_STATUS_SUCCESS",
                    "target_resource_ids": ["node1"],
                    "metrics": [{"key": "m1", "value": 1.0, "unit": "GB/s"}],
                },
                {
                    "name": "test2",
                    "result_status": "RESULT_STATUS_FAILED",
                    "target_resource_ids": ["node2"],
                    "metrics": [{"key": "m2", "value": "some_error"}],
                },
            ],
        }

        with open(self.valid_json_path, "w") as f:
            json.dump(self.valid_data, f)

        with open(self.invalid_json_path, "w") as f:
            f.write("{invalid_json")

        logging.disable(logging.CRITICAL)

    def tearDown(self):
        self.test_dir.cleanup()
        logging.disable(logging.NOTSET)

    def test_init_valid_json(self):
        """Test initialization with valid JSON."""
        viewer = ResultsViewer(self.valid_json_path)
        self.assertEqual(viewer.data, self.valid_data)

    def test_init_file_not_found(self):
        """Test initialization with missing file."""
        with mock.patch.object(Console, "print") as mock_print:
            viewer = ResultsViewer("non_existent.json")
            self.assertEqual(viewer.data, {})
            mock_print.assert_called()
            self.assertIn(
                "Error loading Bill of Health", mock_print.call_args[0][0]
            )

    def test_init_invalid_json(self):
        """Test initialization with invalid JSON."""
        with mock.patch.object(Console, "print") as mock_print:
            viewer = ResultsViewer(self.invalid_json_path)
            self.assertEqual(viewer.data, {})
            mock_print.assert_called()

    def test_get_status_style(self):
        """Test status style mapping."""
        viewer = ResultsViewer(self.valid_json_path)

        style, icon = viewer._get_status_style("SUCCESS")
        self.assertEqual(style, "bold green")

        style, icon = viewer._get_status_style("FAILED")
        self.assertEqual(style, "bold red")

        style, icon = viewer._get_status_style("UNKNOWN")
        self.assertEqual(style, "bold cyan")

    @mock.patch.object(Console, "print")
    def test_show_summary_no_data(self, mock_print):
        """Test show_summary with no data."""
        viewer = ResultsViewer("non_existent.json")
        viewer.show_summary()
        # Should safely return without crashing, error printed in init
        pass

    @mock.patch.object(Console, "print")
    def test_show_summary_valid(self, mock_print):
        """Test show_summary with valid data."""
        viewer = ResultsViewer(self.valid_json_path)
        viewer.show_summary()
        self.assertTrue(mock_print.called)
        # Check if Table or specific strings are printed
        # (hard to check exact output with rich)

    @mock.patch.object(Console, "print")
    def test_show_details_all(self, mock_print):
        """Test show_details showing all tests."""
        viewer = ResultsViewer(self.valid_json_path)
        viewer.show_details()
        self.assertTrue(mock_print.called)
        # Verify print call count roughly matches number of tests
        # (each test prints a panel)
        # + calls for "No matching results" if empty
        # In this case we have 2 tests, so at least 2 calls.
        self.assertGreaterEqual(mock_print.call_count, 2)

    @mock.patch.object(Console, "print")
    def test_show_details_filtered_by_test(self, mock_print):
        """Test show_details filtered by test name."""
        viewer = ResultsViewer(self.valid_json_path)
        viewer.show_details(tests=["test1"])
        self.assertTrue(mock_print.called)
        # Should only print test1
        self.assertEqual(mock_print.call_count, 1)  # One panel for test1

    @mock.patch.object(Console, "print")
    def test_show_details_filtered_by_resource(self, mock_print):
        """Test show_details filtered by resource."""
        viewer = ResultsViewer(self.valid_json_path)
        viewer.show_details(resources=["node2"])
        self.assertTrue(mock_print.called)
        # Should only print test2 (which targets node2)
        self.assertEqual(mock_print.call_count, 1)

    @mock.patch.object(Console, "print")
    def test_show_details_no_match(self, mock_print):
        """Test show_details with no matching tests."""
        viewer = ResultsViewer(self.valid_json_path)
        viewer.show_details(tests=["non_existent"])
        mock_print.assert_called_with(
            mock.ANY
        )  # Should print "No matching results found"
        args, _ = mock_print.call_args
        self.assertIn("No matching results found", str(args[0].renderable))

    @mock.patch.object(Console, "print")
    def test_show_details_filtered_by_resource_substring(self, mock_print):
        """Test show_details filtered by resource substring."""
        viewer = ResultsViewer(self.valid_json_path)
        viewer.show_details(resources=["ode2"])  # Should match "node2"
        self.assertTrue(mock_print.called)
        # Should only print test2 (which targets node2)
        self.assertEqual(mock_print.call_count, 1)

    @mock.patch.object(Console, "print")
    def test_show_summary_list_with_description(self, mock_print):
        """Test show_summary_list includes test description."""
        # Create a viewer with a mock console
        viewer = ResultsViewer(self.valid_json_path)
        viewer.console = mock.Mock(spec=Console)

        # Mock data with a known test name
        test_results = [
            {
                "name": "cluster_health_diagnostics-nccl-all",
                "result_status": "RESULT_STATUS_SUCCESS",
                "target_resource_ids": ["node1"],
            }
        ]

        viewer._show_summary_list(test_results)

        # Verify print was called
        self.assertTrue(viewer.console.print.called)

        # Check if the description is in the rendered content
        # The second call to print should be the panel
        args, _ = viewer.console.print.call_args_list[0]
        panel = args[0]
        self.assertIn(
            "Validates cluster-wide GPU communication", panel.renderable.plain
        )

    @mock.patch.object(Console, "print")
    def test_get_test_description(self, mock_print):
        """Test _get_test_description returns correct descriptions."""
        viewer = ResultsViewer(self.valid_json_path)

        desc = viewer._get_test_description("nccl-all-test")
        self.assertIn("Validates cluster-wide", desc)

        desc = viewer._get_test_description("nccl-pairwise-test")
        self.assertIn("Measures point-to-point", desc)

        desc = viewer._get_test_description("dcgm-test")
        self.assertIn("Performs deep hardware diagnostics", desc)

        desc = viewer._get_test_description("unknown-test")
        self.assertEqual(desc, "")


if __name__ == "__main__":
    unittest.main()
