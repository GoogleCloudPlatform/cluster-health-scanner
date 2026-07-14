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

"""Unit tests for the ResultsViewer class."""

import json
import logging
import os
import tempfile
import unittest
from unittest import mock

from chs.results_viewer import ResultsViewer
from chs import exceptions as ex
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
                    "test_name": "test1",
                    "result_status": "RESULT_STATUS_SUCCESS",
                    "target_resources": [{"node_name": "node1"}],
                    "metrics": [{"key": "m1", "value": 1.0, "unit": "GB/s"}],
                },
                {
                    "test_name": "test2",
                    "result_status": "RESULT_STATUS_FAILED",
                    "target_resources": [{"node_name": "node2"}],
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
        self.assertEqual(viewer.boh.report_id, "test-id")
        self.assertEqual(len(viewer.boh.test_results), 2)

    def test_init_file_not_found(self):
        """Test initialization with missing file."""
        with self.assertRaises(ex.BillOfHealthNotFoundException):
            ResultsViewer("non_existent.json")

    def test_init_invalid_json(self):
        """Test initialization with invalid JSON."""
        with self.assertRaises(ex.BillOfHealthLoadException):
            ResultsViewer(self.invalid_json_path)

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
        self.assertEqual(
            mock_print.call_count, 2
        )  # One panel for test1 and one for separator newline

    @mock.patch.object(Console, "print")
    @mock.patch(
        "chs.utils.tests.get_plugin_to_test_map",
        return_value={"tpu-collectives": "collectives"},
    )
    def test_show_details_filtered_by_test_alias(self, mock_map, mock_print):
        """Test show_details filtered by suite test name alias."""
        data = {
            "report_id": "test-id",
            "chs_version": "1.0.0",
            "test_results": [
                {
                    "test_name": "tpu-collectives",
                    "result_status": "RESULT_STATUS_SUCCESS",
                    "target_resources": [{"node_name": "node1"}],
                    "metrics": [],
                }
            ],
        }
        json_path = os.path.join(self.test_dir.name, "boh_alias.json")
        with open(json_path, "w") as f:
            json.dump(data, f)

        viewer = ResultsViewer(json_path)
        results = viewer._filter_results(tests=["collectives"])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].test_name, "tpu-collectives")

    @mock.patch.object(Console, "print")
    def test_show_details_filtered_by_resource(self, mock_print):
        """Test show_details filtered by resource."""
        viewer = ResultsViewer(self.valid_json_path)
        viewer.show_details(resources=["node2"])
        self.assertTrue(mock_print.called)
        # Should print panels for test2 and newlines
        self.assertGreater(mock_print.call_count, 0)

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
        self.assertGreater(mock_print.call_count, 0)

    @mock.patch.object(Console, "print")
    @mock.patch("chs.results_viewer.PluginManager.get_plugin")
    def test_show_details_time_exceeded(self, mock_get_plugin, mock_print):
        """Test that TIMEOUT and DEADLINE statuses are printed and filtered."""
        data = {
            "report_id": "test-id",
            "chs_version": "1.0.0",
            "test_results": [
                {
                    "test_name": "test_timeout",
                    "test_level": "level-1",
                    "result_status": "RESULT_STATUS_TIMEOUT",
                    "target_resources": [{"node_name": "node1"}],
                },
                {
                    "test_name": "test_deadline",
                    "test_level": "level-2",
                    "result_status": "RESULT_STATUS_DEADLINE",
                    "target_resources": [{"node_name": "node2"}],
                },
                {
                    "test_name": "test_success",
                    "test_level": "level-3",
                    "result_status": "RESULT_STATUS_SUCCESS",
                    "target_resources": [{"node_name": "node3"}],
                },
            ],
        }
        time_exceeded_json_path = os.path.join(
            self.test_dir.name, "time_exceeded.json"
        )
        with open(time_exceeded_json_path, "w") as f:
            json.dump(data, f)

        viewer = ResultsViewer(time_exceeded_json_path)

        mock_plugin = mock.MagicMock()
        mock_get_plugin.return_value = mock_plugin
        mock_plugin.render_detail.return_value = None

        viewer.show_details()

        timeout_msg_called = False
        deadline_msg_called = False
        for call in mock_print.call_args_list:
            args = call[0]
            if len(args) > 0 and isinstance(args[0], str):
                if "TIMEOUT" in args[0] and "test_timeout" in args[0]:
                    timeout_msg_called = True
                if "DEADLINE" in args[0] and "test_deadline" in args[0]:
                    deadline_msg_called = True

        self.assertTrue(timeout_msg_called, "TIMEOUT message should be printed")
        self.assertTrue(
            deadline_msg_called, "DEADLINE message should be printed"
        )

        self.assertTrue(mock_plugin.render_detail.called)

        render_args = mock_plugin.render_detail.call_args_list
        valid_passed = False
        for call in render_args:
            results_passed = call[0][0]
            for res in results_passed:
                self.assertEqual(res.test_name, "test_success")
                valid_passed = True
        self.assertTrue(
            valid_passed, "render_detail should be called with valid results"
        )


if __name__ == "__main__":
    unittest.main()
