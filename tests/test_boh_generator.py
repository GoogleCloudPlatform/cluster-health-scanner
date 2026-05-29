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

"""Unit tests for the Bill of Health Generator."""

import datetime
import json
import logging
import os
import tempfile
import unittest
from unittest import mock

import chs.boh_generator
from chs.boh_generator import BOHGenerator
from chs.protos import bill_of_health_pb2 as pb
from google.protobuf.timestamp_pb2 import Timestamp


class TestBOHGenerator(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)
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
        data = {"workspace_name": "test_hash", "experiments": []}
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

        # Test Kubernetes
        generator = BOHGenerator(self.test_json_path, cluster_type="kubernetes")
        report = generator.generate()
        self.assertEqual(report.system_info.type, "KUBERNETES_CLUSTER")

        # Test custom
        generator = BOHGenerator(self.test_json_path, cluster_type="custom")
        report = generator.generate()
        self.assertEqual(report.system_info.type, "CUSTOM_CLUSTER")

    def test_generate_with_thresholds_min(self):
        """Test threshold evaluation with 'min' type."""
        thresholds_data = {
            "default": {
                "my-test": {"type": "min", "thresholds": {"bandwidth": 80}}
            }
        }
        thresh_file = os.path.join(self.test_dir.name, "thresh.yaml")
        with open(thresh_file, "w") as f:
            import yaml

            yaml.dump(thresholds_data, f)

        # Test case 1: Passing metric
        data_pass = {
            "experiments": [
                {
                    "experiment_name": "my-test",
                    "RAMBLE_VARIABLES": {"machine_type": "a3-ultragpu-8g"},
                    "CONTEXTS": [
                        {
                            "name": "null",
                            "foms": [{"name": "bandwidth", "value": 85}],
                        }
                    ],
                }
            ]
        }
        self._write_json(data_pass)
        generator = BOHGenerator(
            self.test_json_path, thresholds_path=thresh_file
        )
        report = generator.generate()
        self.assertEqual(
            report.test_results[0].result_status, pb.RESULT_STATUS_SUCCESS
        )
        self.assertEqual(
            report.test_results[0].metrics[0].validation_criteria.result_status,
            pb.RESULT_STATUS_SUCCESS,
        )

        # Test case 2: Failing metric
        data_fail = {
            "experiments": [
                {
                    "experiment_name": "my-test",
                    "RAMBLE_VARIABLES": {"machine_type": "a3-ultragpu-8g"},
                    "CONTEXTS": [
                        {
                            "name": "null",
                            "foms": [{"name": "bandwidth", "value": 75}],
                        }
                    ],
                }
            ]
        }
        self._write_json(data_fail)
        generator = BOHGenerator(
            self.test_json_path, thresholds_path=thresh_file
        )
        report = generator.generate()
        self.assertEqual(
            report.test_results[0].result_status, pb.RESULT_STATUS_FAILED
        )
        self.assertEqual(
            report.test_results[0].metrics[0].validation_criteria.result_status,
            pb.RESULT_STATUS_FAILED,
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

    def test_generate_file_error(self):
        """Test error handling for missing input file."""
        with mock.patch.object(chs.boh_generator, "logger") as mock_logger:
            generator = BOHGenerator("non_existent_file.json")
            report = generator.generate()
            self.assertIsInstance(report, pb.BillOfHealth)
            self.assertEqual(len(report.test_results), 0)
            mock_logger.error.assert_called_once()

    def test_ignore_chs(self):
        """Test that chs experiment is ignored."""
        data = {
            "experiments": [
                {
                    "experiment_name": "chs",
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
        """Test extraction of test name and level with PluginManager mocks."""
        with mock.patch(
            "chs.boh_generator.chs.core.plugin_manager.PluginManager"
        ) as mock_pm_cls:
            mock_pm = mock_pm_cls.return_value
            mock_plugin = mock.MagicMock()
            mock_pm.get_plugin.return_value = mock_plugin

            mock_pm.resolve_test_name.side_effect = lambda n: n

            generator = BOHGenerator(self.test_json_path)

            def mock_extract(exp):
                if exp.get("application_name") == "nvidia-dcgm":
                    parts = exp.get("experiment_name", "").split("-")
                    if len(parts) > 3 and parts[2] == "level":
                        return "dcgm", f"level-{parts[3]}"
                    return "dcgm", exp.get("workload_name", "")
                if exp.get("application_name") == "nccl-tests":
                    return exp.get("experiment_name", ""), exp.get(
                        "workload_name", ""
                    )
                return exp.get("experiment_name", ""), exp.get(
                    "workload_name", ""
                )

            mock_plugin.extract_test_info.side_effect = mock_extract

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

    def test_extract_test_info_plugin_name(self):
        """Test that _extract_test_info uses class attribute test_name."""
        generator = BOHGenerator(self.test_json_path)

        exp = {
            "experiment_name": "chs-gkeramblett-standard-hvku-stream-gke-"
            "gkeramblett-a3-ultragpu-8g-a3-ult-bf64941f-b8xs",
            "application_name": "nvidia-stream",
            "workload_name": "standard",
        }

        name, level, desc = generator._extract_test_info(exp)
        self.assertEqual(name, "stream")
        self.assertEqual(level, "standard")

    def test_generate_with_thresholds_scalar(self):
        """Test threshold evaluation with 'scalar' type."""
        thresholds_data = {
            "default": {
                "my-test": {"type": "scalar", "thresholds": {"gpus": 8}}
            }
        }
        thresh_file = os.path.join(self.test_dir.name, "thresh.yaml")
        with open(thresh_file, "w") as f:
            import yaml

            yaml.dump(thresholds_data, f)

        # Pass case
        data_pass = {
            "experiments": [
                {
                    "experiment_name": "my-test",
                    "CONTEXTS": [
                        {"name": "null", "foms": [{"name": "gpus", "value": 8}]}
                    ],
                }
            ]
        }
        self._write_json(data_pass)
        generator = BOHGenerator(
            self.test_json_path, thresholds_path=thresh_file
        )
        report = generator.generate()
        self.assertEqual(
            report.test_results[0].result_status, pb.RESULT_STATUS_SUCCESS
        )

        # Fail case
        data_fail = {
            "experiments": [
                {
                    "experiment_name": "my-test",
                    "CONTEXTS": [
                        {"name": "null", "foms": [{"name": "gpus", "value": 7}]}
                    ],
                }
            ]
        }
        self._write_json(data_fail)
        generator = BOHGenerator(
            self.test_json_path, thresholds_path=thresh_file
        )
        report = generator.generate()
        self.assertEqual(
            report.test_results[0].result_status, pb.RESULT_STATUS_FAILED
        )

    def test_generate_with_thresholds_relative(self):
        """Test threshold evaluation with 'relative' type."""
        thresholds_data = {
            "default": {
                "stream": {
                    "type": "relative",
                    "thresholds": {"efficiency": 0.8},
                }
            }
        }
        thresh_file = os.path.join(self.test_dir.name, "thresh.yaml")
        with open(thresh_file, "w") as f:
            import yaml

            yaml.dump(thresholds_data, f)

        # Pass case: Rate=85, PEAK=100 -> 0.85 >= 0.8
        data_pass = {
            "experiments": [
                {
                    "experiment_name": "stream",
                    "CONTEXTS": [
                        {
                            "name": "null",
                            "foms": [{"name": "PEAK Bandwidth", "value": 100}],
                        },
                        {
                            "name": "Device 0",
                            "foms": [{"name": "Rate", "value": 85}],
                        },
                    ],
                }
            ]
        }
        self._write_json(data_pass)
        generator = BOHGenerator(
            self.test_json_path, thresholds_path=thresh_file
        )
        report = generator.generate()
        self.assertEqual(
            report.test_results[0].result_status, pb.RESULT_STATUS_SUCCESS
        )

        # Fail case: Rate=75, PEAK=100 -> 0.75 < 0.8
        data_fail = {
            "experiments": [
                {
                    "experiment_name": "stream",
                    "CONTEXTS": [
                        {
                            "name": "null",
                            "foms": [{"name": "PEAK Bandwidth", "value": 100}],
                        },
                        {
                            "name": "Device 0",
                            "foms": [{"name": "Rate", "value": 75}],
                        },
                    ],
                }
            ]
        }
        self._write_json(data_fail)
        generator = BOHGenerator(
            self.test_json_path, thresholds_path=thresh_file
        )
        report = generator.generate()
        self.assertEqual(
            report.test_results[0].result_status, pb.RESULT_STATUS_FAILED
        )

    def test_generate_fallback_string_match(self):
        """Test fallback string match is disabled when no thresholds defined."""
        data_pass = {
            "experiments": [
                {
                    "experiment_name": "unknown-test",
                    "CONTEXTS": [
                        {
                            "name": "null",
                            "foms": [
                                {
                                    "name": "log",
                                    "value": "Gpu homework PASSED successfully",
                                }
                            ],
                        }
                    ],
                }
            ]
        }
        self._write_json(data_pass)
        generator = BOHGenerator(self.test_json_path)
        report = generator.generate()
        self.assertEqual(
            report.test_results[0].result_status, pb.RESULT_STATUS_UNKNOWN
        )

        data_fail = {
            "experiments": [
                {
                    "experiment_name": "unknown-test",
                    "CONTEXTS": [
                        {
                            "name": "null",
                            "foms": [
                                {"name": "log", "value": "Gpu homework FAILED"}
                            ],
                        }
                    ],
                }
            ]
        }
        self._write_json(data_fail)
        generator = BOHGenerator(self.test_json_path)
        report = generator.generate()
        self.assertEqual(
            report.test_results[0].result_status, pb.RESULT_STATUS_UNKNOWN
        )

    def test_generate_metadata_relative_log(self):
        """Test extraction of tags and log files with relative path."""
        log_path = os.path.join(self.test_dir.name, "experiments/log1.out")
        data = {
            "experiments": [
                {
                    "name": "exp1",
                    "TAGS": ["tag1", "tag2"],
                    "RAMBLE_VARIABLES": {"log_file": log_path},
                }
            ]
        }
        self._write_json(data)

        generator = BOHGenerator(
            self.test_json_path, base_log_dir=self.test_dir.name
        )
        report = generator.generate()

        metadata = report.test_results[0].metadata
        self.assertEqual(
            metadata.fields["log_file"].string_value, "experiments/log1.out"
        )

    def test_generate_metadata_gcs_log(self):
        """Test extraction of tags and log files with GCS path."""
        log_path = os.path.join(self.test_dir.name, "experiments/log1.out")
        data = {
            "workspace_name": "my_workspace",
            "experiments": [
                {
                    "name": "exp1",
                    "TAGS": ["tag1", "tag2"],
                    "RAMBLE_VARIABLES": {"log_file": log_path},
                }
            ],
        }
        self._write_json(data)

        generator = BOHGenerator(
            self.test_json_path,
            base_log_dir=self.test_dir.name,
            gcs_base_path="gs://my_bucket/workspaces/my_workspace",
        )
        report = generator.generate()

        metadata = report.test_results[0].metadata
        self.assertEqual(
            metadata.fields["log_file"].string_value, "experiments/log1.out"
        )
        self.assertEqual(
            metadata.fields["gcs_log_file"].string_value,
            "gs://my_bucket/workspaces/my_workspace/experiments/log1.out",
        )

    def test_generate_relative_zero_peak(self):
        """Test relative threshold evaluation when peak value is zero."""
        thresholds_data = {
            "default": {
                "stream": {
                    "type": "relative",
                    "thresholds": {"efficiency": 0.8},
                }
            }
        }
        thresh_file = os.path.join(self.test_dir.name, "thresh_zero.yaml")
        with open(thresh_file, "w") as f:
            import yaml

            yaml.dump(thresholds_data, f)

        data = {
            "experiments": [
                {
                    "experiment_name": "stream",
                    "CONTEXTS": [
                        {
                            "name": "null",
                            "foms": [{"name": "PEAK Bandwidth", "value": 0}],
                        },
                        {
                            "name": "Device 0",
                            "foms": [{"name": "Rate", "value": 50}],
                        },
                    ],
                }
            ]
        }
        self._write_json(data)
        generator = BOHGenerator(
            self.test_json_path, thresholds_path=thresh_file
        )
        report = generator.generate()
        self.assertEqual(
            report.test_results[0].result_status, pb.RESULT_STATUS_FAILED
        )
        self.assertEqual(
            report.test_results[0].metrics[1].validation_criteria.result_status,
            pb.RESULT_STATUS_FAILED,
        )

    def test_generate_missing_fom_ignored(self):
        """Test that missing FOM for a defined threshold is ignored safely."""
        thresholds_data = {
            "default": {
                "my-test": {
                    "type": "min",
                    "thresholds": {"bandwidth": 80, "non_existent_fom": 50},
                }
            }
        }
        thresh_file = os.path.join(self.test_dir.name, "thresh_missing.yaml")
        with open(thresh_file, "w") as f:
            import yaml

            yaml.dump(thresholds_data, f)

        data = {
            "experiments": [
                {
                    "experiment_name": "my-test",
                    "CONTEXTS": [
                        {
                            "name": "null",
                            "foms": [{"name": "bandwidth", "value": 85}],
                        }
                    ],
                }
            ]
        }
        self._write_json(data)
        generator = BOHGenerator(
            self.test_json_path, thresholds_path=thresh_file
        )
        report = generator.generate()
        self.assertEqual(
            report.test_results[0].result_status, pb.RESULT_STATUS_SUCCESS
        )
        self.assertEqual(
            report.test_results[0].metrics[0].validation_criteria.result_status,
            pb.RESULT_STATUS_SUCCESS,
        )

    def test_generate_machine_override_type(self):
        """Test that machine-specific config can override the default
        comparison type.
        """
        thresholds_data = {
            "default": {
                "my-test": {"type": "min", "thresholds": {"metric1": 80}}
            },
            "a3-ultragpu-8g": {
                "my-test": {"type": "scalar", "thresholds": {"metric1": 80}}
            },
        }
        thresh_file = os.path.join(self.test_dir.name, "thresh_override.yaml")
        with open(thresh_file, "w") as f:
            import yaml

            yaml.dump(thresholds_data, f)

        data = {
            "experiments": [
                {
                    "experiment_name": "my-test",
                    "RAMBLE_VARIABLES": {"machine_type": "a3-ultragpu-8g"},
                    "CONTEXTS": [
                        {
                            "name": "null",
                            "foms": [{"name": "metric1", "value": 85}],
                        }
                    ],
                }
            ]
        }
        self._write_json(data)
        generator = BOHGenerator(
            self.test_json_path, thresholds_path=thresh_file
        )
        report = generator.generate()
        self.assertEqual(
            report.test_results[0].result_status, pb.RESULT_STATUS_FAILED
        )

    def test_generate_skip_validation(self):
        """Test that setting skip_validation=True skips evaluation."""
        thresholds_data = {
            "default": {
                "my-test": {"type": "min", "thresholds": {"bandwidth": 80}}
            }
        }
        thresh_file = os.path.join(self.test_dir.name, "thresh.yaml")
        with open(thresh_file, "w") as f:
            import yaml

            yaml.dump(thresholds_data, f)

        data = {
            "experiments": [
                {
                    "experiment_name": "my-test",
                    "CONTEXTS": [
                        {
                            "name": "null",
                            "foms": [{"name": "bandwidth", "value": 75}],
                        }
                    ],
                }
            ]
        }
        self._write_json(data)

        generator = BOHGenerator(
            self.test_json_path,
            thresholds_path=thresh_file,
            skip_validation=True,
        )
        report = generator.generate()

        self.assertEqual(
            report.test_results[0].result_status, pb.RESULT_STATUS_UNKNOWN
        )
        self.assertFalse(
            report.test_results[0].metrics[0].HasField("validation_criteria")
        )

    @mock.patch("chs.boh_generator.ChsAppData")
    def test_generate_system_info(self, mock_app_data_cls):
        """Test that system info is populated from app_data."""
        mock_app_data = mock_app_data_cls.return_value
        mock_app_data.get_run.return_value = {
            "command_statement": (
                "chs run --scheduler slurm --machine-type a3-high "
                "--dry-run --tests t1,t2"
            ),
        }

        data = {"workspace_name": "test_workspace", "experiments": []}
        self._write_json(data)

        generator = BOHGenerator(self.test_json_path)
        report = generator.generate()

        self.assertEqual(report.report_id, "test_workspace")
        self.assertEqual(
            report.system_info.metadata.fields[
                "command_statement"
            ].string_value,
            (
                "chs run --scheduler slurm --machine-type a3-high "
                "--dry-run --tests t1,t2"
            ),
        )
        # Verify that other fields are NOT there
        self.assertNotIn("scheduler", report.system_info.metadata.fields)

    def test_load_thresholds_invalid_yaml(self):
        """Test that invalid YAML logs warning and falls back."""
        thresh_file = os.path.join(self.test_dir.name, "bad_yaml.yaml")
        with open(thresh_file, "w") as f:
            f.write("invalid: yaml: [")

        with self.assertRaises(ValueError):
            BOHGenerator(self.test_json_path, thresholds_path=thresh_file)


if __name__ == "__main__":
    unittest.main()
