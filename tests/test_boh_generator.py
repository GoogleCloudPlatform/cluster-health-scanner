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
from chs.core.default_plugin import DefaultPlugin
from chs.core.plugin_base import (
    ColumnDef,
    PluginDetailReport,
    TableDataModel,
)
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

    def test_plugin_base_data_contracts(self):
        """Test default plugin base data contracts."""
        col = ColumnDef("key1", "Label 1", align="center")
        self.assertEqual(col.key, "key1")
        self.assertEqual(col.align, "center")

        table = TableDataModel(
            title="Test", headers=[col], rows=[{"key1": "val"}]
        )
        self.assertEqual(table.title, "Test")

        report = PluginDetailReport(
            test_name="t1", test_level="l1", tables=[table]
        )
        self.assertEqual(report.test_name, "t1")

        plugin = DefaultPlugin(test_name="t1")
        self.assertEqual(
            plugin.parse_context_metadata("foo"), {"context": "foo"}
        )
        self.assertEqual(len(plugin.get_column_schema()), 3)
        self.assertIsNone(plugin.get_detail_data([]))

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

        tr1 = report.test_results[0].target_resources[0]
        self.assertEqual(tr1.node_name, "node1")
        self.assertEqual(tr1.instance_id, "")

        trs2 = report.test_results[1].target_resources
        self.assertEqual(trs2[0].node_name, "node1")
        self.assertEqual(trs2[0].instance_id, "")
        self.assertEqual(trs2[1].node_name, "node2")
        self.assertEqual(trs2[1].instance_id, "")

    def test_split_test_result_per_node_preserves_null_context(self):
        """Test per-node splitting retains metrics with 'null' context."""
        test_result = pb.TestResult(test_name="tpu-info")
        test_result.target_resources.add(node_name="node1")
        test_result.target_resources.add(node_name="node2")

        m_null = pb.Metric(key="TPU Hardware / Driver / Network Error Detected")
        m_null.metadata.fields["context"].string_value = "null"
        test_result.metrics.append(m_null)

        m_node1 = pb.Metric(key="Node1 Metric")
        m_node1.metadata.fields["context"].string_value = "node1"
        test_result.metrics.append(m_node1)

        generator = BOHGenerator(self.test_json_path)
        split = generator._split_test_result_per_node(test_result)

        self.assertEqual(len(split), 2)
        node1_keys = [m.key for m in split[0].metrics]
        node2_keys = [m.key for m in split[1].metrics]

        self.assertIn(
            "TPU Hardware / Driver / Network Error Detected", node1_keys
        )
        self.assertIn(
            "TPU Hardware / Driver / Network Error Detected", node2_keys
        )
        self.assertIn("Node1 Metric", node1_keys)
        self.assertNotIn("Node1 Metric", node2_keys)

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

    def test_generate_missing_fom_fails(self):
        """Test that missing FOM for a defined threshold causes failure."""
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
            report.test_results[0].result_status, pb.RESULT_STATUS_FAILED
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

    @mock.patch("chs.boh_generator.ChsAppData")
    def test_generate_time_exceeded(self, mock_app_data_cls):
        """Test handling of TIMEOUT and DEADLINE job statuses."""
        mock_app_data = mock_app_data_cls.return_value
        mock_app_data.get_run.return_value = {
            "job_entries": {
                "job1": {
                    "experiment_dir": "/path/to/exp_timeout",
                    "status": "TIMEOUT",
                },
                "job2": {
                    "experiment_dir": "/path/to/exp_deadline",
                    "status": "DEADLINE",
                },
            }
        }

        data = {
            "workspace_name": "test_workspace",
            "experiments": [
                {
                    "experiment_name": "exp_timeout",
                    "CONTEXTS": [
                        {
                            "name": "null",
                            "foms": [{"name": "metric", "value": 1}],
                        }
                    ],
                },
                {
                    "experiment_name": "exp_deadline",
                    "CONTEXTS": [
                        {
                            "name": "null",
                            "foms": [{"name": "metric", "value": 1}],
                        }
                    ],
                },
            ],
        }
        self._write_json(data)
        generator = BOHGenerator(self.test_json_path)
        report = generator.generate()

        self.assertEqual(len(report.test_results), 2)

        self.assertEqual(report.test_results[0].test_name, "exp_timeout")
        self.assertEqual(
            report.test_results[0].result_status, pb.RESULT_STATUS_TIMEOUT
        )

        self.assertEqual(report.test_results[1].test_name, "exp_deadline")
        self.assertEqual(
            report.test_results[1].result_status, pb.RESULT_STATUS_DEADLINE
        )

    def test_parse_tier_results_from_boh_with_prefixed_test_names(self):
        """Test parse_tier_results_from_boh correctly matches test names."""
        boh = pb.BillOfHealth()
        t1 = boh.test_results.add()
        t1.test_name = "tpu-collectives"
        t1.result_status = pb.RESULT_STATUS_SUCCESS

        t2 = boh.test_results.add()
        t2.test_name = "tpu-hbm"
        t2.result_status = pb.RESULT_STATUS_SUCCESS

        t3 = boh.test_results.add()
        t3.test_name = "tpu-gemm"
        t3.result_status = pb.RESULT_STATUS_FAILED

        expected = ["tpu-collectives", "tpu-hbm", "tpu-gemm", "tpu-info"]
        tier_results, hard_failure = (
            chs.boh_generator.parse_tier_results_from_boh(boh, expected)
        )

        from chs.tier_manager import TestStatus

        self.assertEqual(tier_results["tpu-collectives"], TestStatus.HEALTHY)
        self.assertEqual(tier_results["tpu-hbm"], TestStatus.HEALTHY)
        self.assertEqual(tier_results["tpu-gemm"], TestStatus.DEGRADED)
        self.assertEqual(tier_results["tpu-info"], TestStatus.ERROR)
        self.assertFalse(hard_failure)

    def test_generate_tpu_collectives_peak_bandwidth(self):
        """Test peak bandwidth threshold evaluation for tpu-collectives."""
        thresholds_data = {
            "default": {
                "tpu-collectives": {
                    "type": "min",
                    "thresholds": {
                        "peak_bandwidth": [
                            {
                                "value": 155.0,
                                "conditions": {
                                    "tpu_topology": "2x2x1",
                                    "workload_name": "all_gather",
                                },
                            }
                        ]
                    },
                }
            }
        }
        thresh_file = os.path.join(self.test_dir.name, "thresh.yaml")
        with open(thresh_file, "w") as f:
            import yaml

            yaml.dump(thresholds_data, f)

        data_pass = {
            "experiments": [
                {
                    "experiment_name": "tpu-collectives",
                    "workload_name": "all_gather",
                    "RAMBLE_VARIABLES": {
                        "machine_type": "tpu7x-standard-4t",
                        "tpu_topology": "2x2x1",
                    },
                    "CONTEXTS": [
                        {
                            "name": "node-1,1024,float32",
                            "foms": [
                                {
                                    "name": "Bandwidth (median)",
                                    "value": 100.0,
                                    "units": "GB/s",
                                }
                            ],
                        },
                        {
                            "name": "node-1,16384,float32",
                            "foms": [
                                {
                                    "name": "Bandwidth (median)",
                                    "value": 160.0,
                                    "units": "GB/s",
                                }
                            ],
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

        self.assertEqual(len(report.test_results), 1)
        res = report.test_results[0]
        self.assertEqual(res.result_status, pb.RESULT_STATUS_SUCCESS)

        self.assertEqual(len(res.metrics), 3)

        peak_metric = next(
            (m for m in res.metrics if m.key == "Peak Bandwidth (median)"), None
        )
        self.assertIsNotNone(peak_metric)
        self.assertEqual(peak_metric.value.number_value, 160.0)
        self.assertEqual(
            peak_metric.metadata.fields["context"].string_value,
            "node-1,peak,16384,float32",
        )
        self.assertEqual(
            peak_metric.validation_criteria.result_status,
            pb.RESULT_STATUS_SUCCESS,
        )

        # Test case 2: Failing peak (150.0 < 155.0)
        data_fail = {
            "experiments": [
                {
                    "experiment_name": "tpu-collectives",
                    "workload_name": "all_gather",
                    "RAMBLE_VARIABLES": {
                        "machine_type": "tpu7x-standard-4t",
                        "tpu_topology": "2x2x1",
                    },
                    "CONTEXTS": [
                        {
                            "name": "node-1,1024,float32",
                            "foms": [
                                {
                                    "name": "Bandwidth (median)",
                                    "value": 100.0,
                                    "units": "GB/s",
                                }
                            ],
                        },
                        {
                            "name": "node-1,16384,float32",
                            "foms": [
                                {
                                    "name": "Bandwidth (median)",
                                    "value": 140.0,
                                    "units": "GB/s",
                                }
                            ],
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

        self.assertEqual(len(report.test_results), 1)
        res = report.test_results[0]
        self.assertEqual(res.result_status, pb.RESULT_STATUS_FAILED)
        peak_metric = next(
            (m for m in res.metrics if m.key == "Peak Bandwidth (median)"), None
        )
        self.assertIsNotNone(peak_metric)
        self.assertEqual(peak_metric.value.number_value, 140.0)
        self.assertEqual(
            peak_metric.validation_criteria.result_status,
            pb.RESULT_STATUS_FAILED,
        )

    def test_split_test_result_per_node_prefix_isolation(self):
        """Test per-node splitting isolates node-1 from node-10 metrics."""
        test_result = pb.TestResult(test_name="tpu-info")
        test_result.target_resources.add(node_name="node-1")
        test_result.target_resources.add(node_name="node-10")
        test_result.target_resources.add(node_name="node-100")

        # Global metric (null context) - should be present in all
        m_global = pb.Metric(key="Global Metric")
        m_global.metadata.fields["context"].string_value = "null"
        test_result.metrics.append(m_global)

        # Metric for node-10 via context
        m_node10_ctx = pb.Metric(key="Node 10 Ctx Metric")
        m_node10_ctx.metadata.fields["context"].string_value = "Node node-10"
        test_result.metrics.append(m_node10_ctx)

        # Metric for my-node-1 (hyphenated prefix - should NOT match node-1)
        m_mynode1_ctx = pb.Metric(key="My Node 1 Ctx Metric")
        m_mynode1_ctx.metadata.fields["context"].string_value = "Node my-node-1"
        test_result.metrics.append(m_mynode1_ctx)

        # Metric for node-1-pool (hyphenated suffix - should NOT match node-1)
        m_node1pool_ctx = pb.Metric(key="Node 1 Pool Ctx Metric")
        m_node1pool_ctx.metadata.fields[
            "context"
        ].string_value = "Node node-1-pool"
        test_result.metrics.append(m_node1pool_ctx)

        # Metric for node-100 via explicit node metadata
        m_node100_meta = pb.Metric(key="Node 100 Meta Metric")
        m_node100_meta.metadata.fields["node"].string_value = "node-100"
        m_node100_meta.metadata.fields["context"].string_value = "Node node-100"
        test_result.metrics.append(m_node100_meta)

        # Metric for node-10 via explicit node metadata with null context
        m_node10_meta_null_ctx = pb.Metric(key="Node 10 Meta Null Ctx Metric")
        m_node10_meta_null_ctx.metadata.fields["node"].string_value = "node-10"
        m_node10_meta_null_ctx.metadata.fields["context"].string_value = "null"
        test_result.metrics.append(m_node10_meta_null_ctx)

        # Metric for node-1
        m_node1_ctx = pb.Metric(key="Node 1 Ctx Metric")
        m_node1_ctx.metadata.fields[
            "context"
        ].string_value = "node-1,1024,float32"
        test_result.metrics.append(m_node1_ctx)

        generator = BOHGenerator(self.test_json_path)
        split = generator._split_test_result_per_node(test_result)

        self.assertEqual(len(split), 3)

        # node-1 checks
        node1_keys = [m.key for m in split[0].metrics]
        self.assertIn("Global Metric", node1_keys)
        self.assertIn("Node 1 Ctx Metric", node1_keys)
        self.assertNotIn("Node 10 Ctx Metric", node1_keys)
        self.assertNotIn("My Node 1 Ctx Metric", node1_keys)
        self.assertNotIn("Node 1 Pool Ctx Metric", node1_keys)
        self.assertNotIn("Node 100 Meta Metric", node1_keys)
        self.assertNotIn("Node 10 Meta Null Ctx Metric", node1_keys)

        # node-10 checks
        node10_keys = [m.key for m in split[1].metrics]
        self.assertIn("Global Metric", node10_keys)
        self.assertIn("Node 10 Ctx Metric", node10_keys)
        self.assertIn("Node 10 Meta Null Ctx Metric", node10_keys)
        self.assertNotIn("Node 1 Ctx Metric", node10_keys)
        self.assertNotIn("Node 100 Meta Metric", node10_keys)

        # node-100 checks
        node100_keys = [m.key for m in split[2].metrics]
        self.assertIn("Global Metric", node100_keys)
        self.assertIn("Node 100 Meta Metric", node100_keys)
        self.assertNotIn("Node 1 Ctx Metric", node100_keys)
        self.assertNotIn("Node 10 Ctx Metric", node100_keys)
        self.assertNotIn("Node 10 Meta Null Ctx Metric", node100_keys)

    def test_generate_tpu_collectives_unconfigured_topology_unknown(self):
        """Test unconfigured topology results in UNKNOWN status."""
        thresholds_data = {
            "default": {
                "tpu-collectives": {
                    "type": "min",
                    "thresholds": {
                        "peak_bandwidth": [
                            {
                                "value": 155.0,
                                "conditions": {
                                    "tpu_topology": "2x2x1",
                                    "workload_name": "all_gather",
                                },
                            }
                        ]
                    },
                }
            }
        }
        thresh_file = os.path.join(
            self.test_dir.name, "thresh_unconfigured.yaml"
        )
        with open(thresh_file, "w") as f:
            import yaml

            yaml.dump(thresholds_data, f)

        data = {
            "experiments": [
                {
                    "experiment_name": "tpu-collectives",
                    "workload_name": "all_gather",
                    "RAMBLE_VARIABLES": {
                        "machine_type": "tpu7x-standard-4t",
                        "tpu_topology": "8x8x8",
                    },
                    "CONTEXTS": [
                        {
                            "name": "node-1,1024,float32",
                            "foms": [
                                {
                                    "name": "Bandwidth (median)",
                                    "value": 100.0,
                                    "units": "GB/s",
                                }
                            ],
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

        self.assertEqual(len(report.test_results), 1)
        res = report.test_results[0]
        self.assertEqual(res.result_status, pb.RESULT_STATUS_UNKNOWN)

        from chs.core.summary_aggregator import aggregate_boh_summary

        summary_data = aggregate_boh_summary(report, self.test_json_path)
        self.assertIsNotNone(summary_data)
        self.assertEqual(len(summary_data.test_summaries), 1)
        ts = summary_data.test_summaries[0]
        self.assertTrue(ts.group_has_no_thresholds)
        self.assertEqual(ts.met_threshold, 0)
        self.assertEqual(ts.failed_to_execute, 0)

    def test_generate_tpu_collectives_removed_workload_threshold_unknown(self):
        """Test removed workload thresholds result in UNKNOWN status."""
        thresholds_data = {
            "default": {
                "tpu-collectives": {
                    "type": "min",
                    "thresholds": {
                        "peak_bandwidth": [
                            {
                                "value": 155.0,
                                "conditions": {
                                    "tpu_topology": "2x2x1",
                                    "workload_name": "all_gather",
                                },
                            }
                        ],
                        "some_other_metric_for_all_gather": [
                            {
                                "value": 100.0,
                                "conditions": {
                                    "tpu_topology": "2x2x1",
                                    "workload_name": "all_gather",
                                },
                            }
                        ],
                    },
                }
            }
        }
        thresh_file = os.path.join(
            self.test_dir.name, "thresh_removed_workload.yaml"
        )
        with open(thresh_file, "w") as f:
            import yaml

            yaml.dump(thresholds_data, f)

        data = {
            "experiments": [
                {
                    "experiment_name": "tpu-collectives",
                    "workload_name": "psum",
                    "RAMBLE_VARIABLES": {
                        "machine_type": "tpu7x-standard-4t",
                        "tpu_topology": "2x2x1",
                    },
                    "CONTEXTS": [
                        {
                            "name": "node-1,1024,float32",
                            "foms": [
                                {
                                    "name": "Bandwidth (median)",
                                    "value": 100.0,
                                    "units": "GB/s",
                                }
                            ],
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

        self.assertEqual(len(report.test_results), 1)
        res = report.test_results[0]
        self.assertEqual(res.result_status, pb.RESULT_STATUS_UNKNOWN)

        from chs.core.summary_aggregator import aggregate_boh_summary

        summary_data = aggregate_boh_summary(report, self.test_json_path)
        self.assertIsNotNone(summary_data)
        self.assertEqual(len(summary_data.test_summaries), 1)
        ts = summary_data.test_summaries[0]
        self.assertTrue(ts.group_has_no_thresholds)
        self.assertEqual(ts.missing_metrics, 0)
        self.assertEqual(ts.met_threshold, 0)
        self.assertEqual(ts.failed_to_execute, 0)

    def test_tpu_test_names_prefix(self):
        """Ensure TPU-related tests in thresholds.yaml have 'tpu-' prefix."""
        generator = BOHGenerator(self.test_json_path)
        thresholds = generator._load_thresholds()
        for machine_type, tests in thresholds.items():
            if machine_type == "default":
                continue
            for test_name in tests:
                if "tpu" in test_name.lower():
                    msg = (
                        f"TPU test '{test_name}' under '{machine_type}' "
                        "must start with 'tpu-' prefix."
                    )
                    self.assertTrue(test_name.startswith("tpu-"), msg)


if __name__ == "__main__":
    unittest.main()
