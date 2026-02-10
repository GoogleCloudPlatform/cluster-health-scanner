"""Unit tests for the Bill of Health Generator."""

import datetime
import json
import os
import tempfile
import unittest
from unittest import mock

from chs.boh_generator import BOHGenerator
from chs.protos import bill_of_health_pb2 as pb
from google.protobuf.timestamp_pb2 import Timestamp


class TestBOHGenerator(unittest.TestCase):

  def setUp(self):
    self.maxDiff = None
    self.test_dir = tempfile.TemporaryDirectory()
    self.test_json_path = os.path.join(self.test_dir.name, "results.json")

  def tearDown(self):
    self.test_dir.cleanup()

  def _write_json(self, data):
    with open(self.test_json_path, "w") as f:
      json.dump(data, f)

  def test_init(self):
    """Test initialization."""
    generator = BOHGenerator(self.test_json_path)
    self.assertEqual(generator.file_path, self.test_json_path)
    self.assertIsInstance(generator.report, pb.BillOfHealth)

  def test_get_timestamp(self):
    """Test timestamp generation."""
    generator = BOHGenerator(self.test_json_path)
    ts = generator._get_timestamp()
    self.assertIsInstance(ts, Timestamp)
    now = datetime.datetime.now(datetime.timezone.utc)
    proto_time = ts.ToDatetime().replace(tzinfo=datetime.timezone.utc)
    self.assertLess(abs((now - proto_time).total_seconds()), 60)

  def test_parse_foms_various_inputs(self):
    """Test parsing of various FOM inputs."""
    generator = BOHGenerator(self.test_json_path)

    foms = [
        {"name": "is_true", "value": "true"},
        {"name": "is_false", "value": "false"},
        {"name": "is_TRUE", "value": "TRUE"},
        {"name": "is_success", "value": "Success"},
        {"name": "is_fail", "value": "Failed"},
        {"name": "is_failure", "value": "Failure"},
        {"name": "is_number", "value": "123.45"},
        {"name": "is_string", "value": "some_string"},
        {"name": "with_unit", "value": "100", "units": "GB/s"},
    ]

    metrics = generator._parse_foms(foms)

    self.assertEqual(len(metrics), 9)

    self.assertTrue(metrics[0].value.bool_value)
    self.assertFalse(metrics[1].value.bool_value)
    self.assertTrue(metrics[2].value.bool_value)
    self.assertEqual(metrics[3].value.string_value, "Success")
    self.assertEqual(metrics[4].value.string_value, "Failed")
    self.assertEqual(metrics[5].value.string_value, "Failure")
    self.assertEqual(metrics[6].value.number_value, 123.45)
    self.assertEqual(metrics[7].value.string_value, "some_string")
    self.assertEqual(metrics[8].value.number_value, 100.0)
    self.assertEqual(metrics[8].unit, "GB/s")

  def test_generate_basic(self):
    """Test basic report generation with minimal data."""
    data = {"workspace_hash": "test_hash", "experiments": []}
    self._write_json(data)

    generator = BOHGenerator(self.test_json_path)
    report = generator.generate()

    self.assertEqual(report.report_id, "test_hash")
    self.assertEqual(report.chs_version, "1.0.0")
    self.assertEqual(report.system_info.type, "SLURM_CLUSTER")
    self.assertEqual(len(report.test_results), 0)

  def test_generate_statuses(self):
    """Test mapping of all supported experiment statuses."""
    experiments = [
        ("SUCCESS", pb.RESULT_STATUS_SUCCESS),
        ("PASS", pb.RESULT_STATUS_UNKNOWN),
        ("PASSED", pb.RESULT_STATUS_UNKNOWN),
        ("FAILED", pb.RESULT_STATUS_FAILED),
        ("FAIL", pb.RESULT_STATUS_UNKNOWN),
        ("FAILURE", pb.RESULT_STATUS_UNKNOWN),
        ("ERROR", pb.RESULT_STATUS_UNKNOWN),
        ("ERR", pb.RESULT_STATUS_UNKNOWN),
        ("SKIPPED", pb.RESULT_STATUS_UNKNOWN),
        ("SKIP", pb.RESULT_STATUS_UNKNOWN),
        ("CANCELLED", pb.RESULT_STATUS_CANCELLED),
        ("TIMEOUT", pb.RESULT_STATUS_TIMEOUT),
        ("TIMEDOUT", pb.RESULT_STATUS_UNKNOWN),
        ("UNKNOWN", pb.RESULT_STATUS_UNKNOWN),
    ]

    data = {
        "experiments": [
            {"name": f"exp_{status}", "EXPERIMENT_STATUS": status}
            for status, _ in experiments
        ]
    }
    self._write_json(data)

    generator = BOHGenerator(self.test_json_path)
    report = generator.generate()

    self.assertEqual(len(report.test_results), len(experiments))
    for i, (_, expected_status) in enumerate(experiments):
      self.assertEqual(
          report.test_results[i].result_status,
          expected_status,
          f"Failed for status {experiments[i][0]}",
      )

  def test_generate_contexts(self):
    """Test handling of default vs explicit contexts."""

    data = {
        "experiments": [
            {
                "name": "default_only",
                "CONTEXTS": [
                    {"name": "null", "foms": [{"name": "m1", "value": "1"}]}
                ],
            },
            {
                "name": "default_and_last_same",
                "CONTEXTS": [
                    {"name": "null", "foms": [{"name": "m1", "value": "1"}]}
                ],
            },
            {
                "name": "default_and_last_diff",
                "CONTEXTS": [
                    {"name": "null", "foms": [{"name": "m1", "value": "1"}]},
                    {"name": "other", "foms": [{"name": "m2", "value": "2"}]},
                ],
            },
        ]
    }
    self._write_json(data)

    generator = BOHGenerator(self.test_json_path)
    report = generator.generate()

    self.assertEqual(len(report.test_results[0].metrics), 1)

    self.assertEqual(len(report.test_results[1].metrics), 1)

    self.assertEqual(len(report.test_results[2].metrics), 2)
    self.assertEqual(report.test_results[2].metrics[0].key, "m1")
    self.assertEqual(report.test_results[2].metrics[1].key, "m2")

  def test_generate_metadata(self):
    """Test extraction of tags and log files."""
    data = {
        "experiments": [{
            "name": "exp1",
            "TAGS": ["tag1", "tag2"],
            "RAMBLE_VARIABLES": {"log_file": "/path/to/log"},
        }]
    }
    self._write_json(data)

    generator = BOHGenerator(self.test_json_path)
    report = generator.generate()

    metadata = report.test_results[0].metadata

    tags = metadata.fields["tags"].list_value.values
    self.assertEqual(len(tags), 2)
    self.assertEqual(tags[0].string_value, "tag1")
    self.assertEqual(tags[1].string_value, "tag2")

    self.assertEqual(metadata.fields["log_file"].string_value, "/path/to/log")

  def test_generate_targets(self):
    """Test extraction of target resource IDs (nodes)."""
    data = {
        "experiments": [
            {"name": "single_node", "RAMBLE_VARIABLES": {"node": "node1"}},
            {
                "name": "multi_node",
                "RAMBLE_RAW_VARIABLES": {"nodes": ["node1", "node2"]},
            },
        ]
    }
    self._write_json(data)

    generator = BOHGenerator(self.test_json_path)
    report = generator.generate()

    self.assertEqual(
        list(report.test_results[0].target_resource_ids), ["node1"]
    )
    self.assertEqual(
        list(report.test_results[1].target_resource_ids), ["node1", "node2"]
    )

  @mock.patch("builtins.print")
  def test_generate_file_error(self, mock_print):
    """Test error handling for missing input file."""
    generator = BOHGenerator("non_existent_file.json")
    report = generator.generate()
    self.assertIsInstance(report, pb.BillOfHealth)
    self.assertEqual(len(report.test_results), 0)
    mock_print.assert_called_once()

  def test_ignore_cluster_health_diagnostics(self):
    """Test that cluster_health_diagnostics experiment is ignored."""
    data = {
        "experiments": [
            {
                "experiment_name": "cluster_health_diagnostics",
                "EXPERIMENT_STATUS": "SUCCESS",
            },
            {"experiment_name": "valid_exp", "EXPERIMENT_STATUS": "SUCCESS"},
        ]
    }
    self._write_json(data)

    generator = BOHGenerator(self.test_json_path)
    report = generator.generate()

    self.assertEqual(len(report.test_results), 1)
    self.assertEqual(report.test_results[0].name, "valid_exp")


if __name__ == "__main__":
  unittest.main()
