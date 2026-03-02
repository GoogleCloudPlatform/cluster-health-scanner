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

    def test_generate_cluster_type(self):
        """Test generating report with different cluster types."""
        data = {"workspace_hash": "test_hash", "experiments": []}
        self._write_json(data)

        # Test default (slurm)
        generator = BOHGenerator(self.test_json_path)
        report = generator.generate()
        self.assertEqual(report.system_info.type, "SLURM_CLUSTER")

        # Test GKE
        generator = BOHGenerator(self.test_json_path, cluster_type="gke")
        report = generator.generate()
        self.assertEqual(report.system_info.type, "GKE_CLUSTER")

        # Test custom
        generator = BOHGenerator(self.test_json_path, cluster_type="custom")
        report = generator.generate()
        self.assertEqual(report.system_info.type, "CUSTOM_CLUSTER")

    def test_generate_statuses(self):
        """Test mapping of all supported experiment statuses."""
        experiments = [
            (
                {"criteria1": "PASSED", "criteria2": "PASSED"},
                pb.RESULT_STATUS_SUCCESS,
            ),
            (
                {"criteria1": "FAILED", "criteria2": "PASSED"},
                pb.RESULT_STATUS_FAILED,
            ),
            (
                {"criteria1": "TIMEOUT", "criteria2": "PASSED"},
                pb.RESULT_STATUS_TIMEOUT,
            ),
            ({"criteria1": "CANCELLED"}, pb.RESULT_STATUS_CANCELLED),
            (
                {"criteria1": "UNKNOWN", "criteria2": "PASSED"},
                pb.RESULT_STATUS_UNKNOWN,
            ),
            ({}, pb.RESULT_STATUS_UNKNOWN),  # Empty criteria
        ]

        data = {
            "experiments": [
                {"name": f"exp_{i}", "SUCCESS_CRITERIA": criteria}
                for i, (criteria, _) in enumerate(experiments)
            ]
        }
        # Add an experiment with no SUCCESS_CRITERIA key
        data["experiments"].append({"name": "exp_missing"})
        expected_statuses = [status_code for _, status_code in experiments] + [
            pb.RESULT_STATUS_UNKNOWN
        ]

        self._write_json(data)

        generator = BOHGenerator(self.test_json_path)
        report = generator.generate()

        self.assertEqual(len(report.test_results), len(expected_statuses))
        for i, expected_status in enumerate(expected_statuses):
            self.assertEqual(
                report.test_results[i].result_status,
                expected_status,
                f"Failed for experiment {data['experiments'][i]['name']}",
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
                        {
                            "name": "null",
                            "foms": [{"name": "m1", "value": "1"}],
                        },
                        {
                            "name": "other",
                            "foms": [{"name": "m2", "value": "2"}],
                        },
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
            "experiments": [
                {
                    "name": "exp1",
                    "TAGS": ["tag1", "tag2"],
                    "RAMBLE_VARIABLES": {"log_file": "/path/to/log"},
                }
            ]
        }
        self._write_json(data)

        generator = BOHGenerator(self.test_json_path)
        report = generator.generate()

        metadata = report.test_results[0].metadata

        tags = metadata.fields["tags"].list_value.values
        self.assertEqual(len(tags), 2)
        self.assertEqual(tags[0].string_value, "tag1")
        self.assertEqual(tags[1].string_value, "tag2")

        self.assertEqual(
            metadata.fields["log_file"].string_value, "/path/to/log"
        )

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
                    "SUCCESS_CRITERIA": {"criteria1": "PASSED"},
                },
                {
                    "experiment_name": "valid_exp",
                    "SUCCESS_CRITERIA": {"criteria1": "PASSED"},
                },
            ]
        }
        self._write_json(data)

        generator = BOHGenerator(self.test_json_path)
        report = generator.generate()

        self.assertEqual(len(report.test_results), 1)
        self.assertEqual(report.test_results[0].test_name, "valid_exp")

    def test_extract_test_info(self):
        """Test extraction of test name and level."""
        generator = BOHGenerator(self.test_json_path)

        # Test NCCL
        name, level = generator._extract_test_info(
            {
                "experiment_name": "nccl-cluster",
                "application_name": "nccl-tests",
                "workload_name": "all_gather_perf",
            }
        )
        self.assertEqual(name, "nccl-cluster")
        self.assertEqual(level, "all_gather_perf")

        # Test DCGM
        name, level = generator._extract_test_info(
            {
                "experiment_name": "nvidia-dcgm-level-3",
                "application_name": "nvidia-dcgm",
            }
        )
        self.assertEqual(name, "dcgm")
        self.assertEqual(level, "level-3")

        # Test DCGM Fallback
        name, level = generator._extract_test_info(
            {
                "experiment_name": "nvidia-dcgm",
                "application_name": "nvidia-dcgm",
                "workload_name": "diag",
            }
        )
        self.assertEqual(name, "dcgm")
        self.assertEqual(level, "diag")

        # Test Generic
        name, level = generator._extract_test_info(
            {
                "experiment_name": "generic_test",
                "workload_name": "workload",
            }
        )
        self.assertEqual(name, "generic_test")
        self.assertEqual(level, "workload")


if __name__ == "__main__":
    unittest.main()
