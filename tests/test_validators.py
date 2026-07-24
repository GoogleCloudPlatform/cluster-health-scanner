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

"""Unit tests for the Cluster Health Scanner (CHS) validators package."""

import os
import unittest
import click
from unittest import mock

from chs import validators


class TestValidators(unittest.TestCase):
    """Tests the validators package."""

    def setUp(self):
        # Mock _load_suites and _load_tests to return test data
        self.mock_suites_patcher = mock.patch("chs.utils.suites._load_suites")
        self.mock_tests_patcher = mock.patch("chs.utils.tests._load_tests")

        self.mock_suites = self.mock_suites_patcher.start()
        self.mock_tests = self.mock_tests_patcher.start()

        self.mock_suites.return_value = {
            "suites": [
                {
                    "name": "lite",
                    "tests": [{"name": "nvidia-smi"}],
                }
            ]
        }

        self.mock_tests.return_value = {
            "tests": [
                {
                    "name": "nvidia-smi",
                    "platforms": [
                        {"scheduler": "slurm", "machine_type": "a3-megagpu-8g"},
                        {
                            "scheduler": "slurm",
                            "machine_type": "a3-ultragpu-8g",
                        },
                        {
                            "scheduler": "kubernetes",
                            "machine_type": "a3-megagpu-8g",
                        },
                        {
                            "scheduler": "kubernetes",
                            "machine_type": "a3-ultragpu-8g",
                        },
                    ],
                }
            ]
        }

    def tearDown(self):
        self.mock_suites_patcher.stop()
        self.mock_tests_patcher.stop()

    def test_machine_type_validator_valid(self):
        v = validators.MachineTypeValidator()
        errors = v.validate({"machine_type": "a3-megagpu-8g"})
        self.assertEqual(errors, [])

    def test_machine_type_validator_invalid(self):
        v = validators.MachineTypeValidator()
        errors = v.validate({"machine_type": "unknown"})
        self.assertTrue(any("is not supported" in e for e in errors))

    def test_scheduler_validator_valid(self):
        v = validators.SchedulerValidator()
        errors = v.validate({"scheduler": "slurm"})
        self.assertEqual(errors, [])

    def test_scheduler_validator_invalid(self):
        v = validators.SchedulerValidator()
        errors = v.validate({"scheduler": "unknown"})
        self.assertTrue(any("is not supported" in e for e in errors))

    def test_suite_validator_valid(self):
        v = validators.SuiteValidator()
        errors = v.validate({"suite": "lite"})
        self.assertEqual(errors, [])

    def test_suite_validator_invalid(self):
        v = validators.SuiteValidator()
        errors = v.validate({"suite": "unknown"})
        self.assertTrue(any("is not supported" in e for e in errors))

    def test_project_config_validator_valid(self):
        v = validators.ProjectConfigValidator()
        errors = v.validate(
            {"scheduler": "slurm", "machine_type": "a3-megagpu-8g"}
        )
        self.assertEqual(errors, [])

    def test_project_config_validator_invalid(self):
        v = validators.ProjectConfigValidator()
        errors = v.validate({"scheduler": "unknown", "machine_type": "unknown"})
        self.assertEqual(len(errors), 2)

    def test_validate_non_empty_string_valid(self):
        """Test validate_non_empty_string with valid input."""
        result = validators.validate_non_empty_string(None, None, "  value  ")
        self.assertEqual(result, "value")

    def test_validate_non_empty_string_invalid(self):
        """Test validate_non_empty_string with invalid input."""
        ctx = mock.Mock(spec=click.Context)
        ctx.command = mock.Mock()
        param = mock.Mock(spec=click.Parameter)

        with self.assertRaisesRegex(
            click.BadParameter, "cannot be an empty string"
        ):
            validators.validate_non_empty_string(ctx, param, "")

        with self.assertRaisesRegex(
            click.BadParameter, "cannot be an empty string"
        ):
            validators.validate_non_empty_string(ctx, param, " ")

    def test_validate_non_empty_string_none(self):
        """Test validate_non_empty_string with None input."""
        result = validators.validate_non_empty_string(None, None, None)
        self.assertIsNone(result)

    def test_tests_validator_valid(self):
        v = validators.TestsValidator()
        errors = v.validate({"tests": ["nvidia-smi"]})
        self.assertEqual(errors, [])

    def test_tests_validator_invalid_test(self):
        v = validators.TestsValidator()
        errors = v.validate({"tests": ["unknown"]})
        self.assertTrue(any("are not supported" in e for e in errors))

    def test_tests_validator_unsupported_scheduler(self):
        v = validators.TestsValidator()
        errors = v.validate({"scheduler": "unknown", "tests": ["unknown"]})
        self.assertEqual(errors, [])

    def test_tests_validator_unsupported_machine_type(self):
        v = validators.TestsValidator()
        errors = v.validate({"machine_type": "unknown", "tests": ["unknown"]})
        self.assertEqual(errors, [])

    def test_project_config_validator_invalid_scheduler_and_machine_type(self):
        v = validators.ProjectConfigValidator()
        errors = v.validate(
            {
                "scheduler": "unknown_scheduler",
                "machine_type": "unknown_mt",
            }
        )
        self.assertEqual(len(errors), 2)
        self.assertTrue(any("Scheduler" in e for e in errors))
        self.assertTrue(any("Machine type" in e for e in errors))

    def test_tests_validator_skips_when_scheduler_invalid(self):
        v = validators.TestsValidator()
        errors = v.validate(
            {
                "scheduler": "unknown_scheduler",
                "tests": ["unknown_test"],
            }
        )
        self.assertEqual(errors, [])

    def test_tests_validator_skips_when_machine_type_invalid(self):
        v = validators.TestsValidator()
        errors = v.validate(
            {
                "machine_type": "unknown_mt",
                "tests": ["unknown_test"],
            }
        )
        self.assertEqual(errors, [])

    def test_project_config_validator_invalid_test(self):
        v = validators.ProjectConfigValidator()
        errors = v.validate(
            {
                "tests": ["unknown_test"],
                "scheduler": "kubernetes",
                "machine_type": "a3-megagpu-8g",
            }
        )
        self.assertEqual(len(errors), 1)
        self.assertTrue(any("Tests" in e for e in errors))

    def test_cli_args_validator_conflict_suite_and_tests(self):
        v = validators.CliArgsValidator()
        errors = v.validate({"suite": "lite", "tests": "my-test"})
        self.assertEqual(len(errors), 1)
        self.assertTrue(any("Cannot specify both" in e for e in errors))

    def test_cli_args_validator_conflict_targeting(self):
        v = validators.CliArgsValidator()
        errors = v.validate(
            {"nodes": "node1", "nodelists": "list1", "suite": "lite"}
        )
        self.assertEqual(len(errors), 1)
        self.assertTrue(
            any(
                "Cannot specify more than one targeting option" in e
                for e in errors
            )
        )

    def test_cli_args_validator_invalid(self):
        v = validators.CliArgsValidator()
        errors = v.validate({"suite": "unknown_suite"})
        self.assertEqual(len(errors), 1)
        self.assertTrue(any("Suite" in e for e in errors))

    def test_cli_args_validator_valid(self):
        v = validators.CliArgsValidator()
        errors = v.validate({"suite": "lite"})
        self.assertEqual(len(errors), 0)

    def test_cli_args_validator_invalid_thresholds(self):
        v = validators.CliArgsValidator()
        errors = v.validate(
            {"suite": "lite", "thresholds_path": "/nonexistent/thresholds.yaml"}
        )
        self.assertEqual(len(errors), 1)
        self.assertTrue(any("Thresholds file not found" in e for e in errors))

    def test_thresholds_validator_valid(self):
        import tempfile
        import yaml

        v = validators.ThresholdsValidator()
        valid_data = {
            "default": {
                "my-test": {
                    "type": "min",
                    "thresholds": {"bandwidth": 80},
                }
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            yaml.dump(valid_data, f)
            f_path = f.name

        try:
            errors = v.validate({"thresholds_path": f_path})
            self.assertEqual(errors, [])
        finally:
            os.remove(f_path)

    def test_thresholds_validator_invalid_yaml(self):
        import tempfile

        v = validators.ThresholdsValidator()
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("invalid: yaml: [")
            f_path = f.name

        try:
            errors = v.validate({"thresholds_path": f_path})
            self.assertTrue(any("Failed to parse YAML" in e for e in errors))
        finally:
            os.remove(f_path)

    def test_thresholds_validator_invalid_schema(self):
        import tempfile
        import yaml

        v = validators.ThresholdsValidator()
        invalid_data = {
            "default": {
                "my-test": {
                    "type": "invalid_type",
                    "thresholds": {"bandwidth": 80},
                }
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            yaml.dump(invalid_data, f)
            f_path = f.name

        try:
            errors = v.validate({"thresholds_path": f_path})
            self.assertTrue(
                any("Invalid type 'invalid_type'" in e for e in errors)
            )
        finally:
            os.remove(f_path)

    def test_thresholds_validator_file_not_found(self):
        v = validators.ThresholdsValidator()
        errors = v.validate({"thresholds_path": "/nonexistent/file.yaml"})
        self.assertTrue(any("Thresholds file not found" in e for e in errors))

    def test_thresholds_validator_no_path(self):
        v = validators.ThresholdsValidator()
        errors = v.validate({})
        self.assertEqual(errors, [])

    def test_validate_run_id_valid(self):
        self.assertEqual(
            validators.validate_run_id(None, None, "my-run-id"), "my-run-id"
        )
        self.assertEqual(
            validators.validate_run_id(None, None, "run123"), "run123"
        )
        self.assertEqual(validators.validate_run_id(None, None, "a"), "a")
        self.assertEqual(
            validators.validate_run_id(None, None, "a-b-c"), "a-b-c"
        )
        self.assertEqual(validators.validate_run_id(None, None, None), None)

    def test_validate_run_id_invalid(self):
        import click

        invalid_cases = [
            "",  # Empty
            "   ",  # Whitespace
            "Uppercase-Name",  # Uppercase
            "name_with_underscore",  # Underscore
            "name_with_special!",  # Special character
            "-start-with-dash",  # Starts with dash
            "end-with-dash-",  # Ends with dash
            "a" * 36,  # Too long (36 chars)
        ]
        for case in invalid_cases:
            with self.subTest(case):
                with self.assertRaises(click.BadParameter):
                    validators.validate_run_id(None, None, case)

    def test_homogeneity_validator_valid(self):
        v = validators.HomogeneityValidator()
        errors = v.validate(
            {
                "detected_machine_types": ["a3-megagpu-8g"],
                "scheduler": "kubernetes",
            }
        )
        self.assertEqual(errors, [])

    def test_homogeneity_validator_invalid_kubernetes(self):
        v = validators.HomogeneityValidator()
        errors = v.validate(
            {
                "detected_machine_types": ["a3-megagpu-8g", "a3-ultragpu-8g"],
                "scheduler": "kubernetes",
            }
        )
        self.assertEqual(len(errors), 1)
        self.assertIn("Heterogeneous cluster detected", errors[0])
        self.assertIn("--nodepools", errors[0])

    def test_homogeneity_validator_invalid_slurm(self):
        v = validators.HomogeneityValidator()
        errors = v.validate(
            {
                "detected_machine_types": ["a3-megagpu-8g", "a3-ultragpu-8g"],
                "scheduler": "slurm",
            }
        )
        self.assertEqual(len(errors), 1)
        self.assertIn("Heterogeneous cluster detected", errors[0])
        self.assertIn("--partitions", errors[0])

    def test_tpu_topology_validator_valid(self):
        v = validators.topology.TpuTopologyValidator()
        errors = v.validate(
            {
                "machine_type": "tpu7x-standard-4t",
                "scheduler": "kubernetes",
                "scheduler_obj": mock.Mock(
                    nodes_arg=None, nodelists_arg=None, nodepools_arg="pool1"
                ),
            }
        )
        self.assertEqual(errors, [])

    def test_tpu_topology_validator_invalid_kubernetes(self):
        v = validators.topology.TpuTopologyValidator()
        errors = v.validate(
            {
                "machine_type": "tpu7x-standard-4t",
                "scheduler": "kubernetes",
                "scheduler_obj": mock.Mock(
                    nodes_arg="node1", nodelists_arg=None, nodepools_arg="pool1"
                ),
            }
        )
        self.assertEqual(len(errors), 1)
        self.assertIn(
            "TPU tests need entire slice, please use --nodepools.", errors[0]
        )

    def test_tpu_topology_validator_invalid_slurm(self):
        v = validators.topology.TpuTopologyValidator()
        errors = v.validate(
            {
                "machine_type": "ct5l-hightpu-4t",
                "scheduler": "slurm",
                "scheduler_obj": mock.Mock(
                    nodes_arg=None,
                    nodelists_arg="node-[1-2]",
                    partitions_arg="part1",
                ),
            }
        )
        self.assertEqual(len(errors), 1)
        self.assertIn(
            "TPU tests need entire slice, please use --partitions.", errors[0]
        )

    def test_tpu_topology_validator_missing_nodepool(self):
        v = validators.topology.TpuTopologyValidator()
        errors = v.validate(
            {
                "machine_type": "tpu7x-standard-4t",
                "scheduler": "kubernetes",
                "scheduler_obj": mock.Mock(
                    nodes_arg=None, nodelists_arg=None, nodepools_arg=None
                ),
            }
        )
        self.assertEqual(len(errors), 1)
        self.assertIn(
            "TPU tests need entire slice, please use --nodepools to target "
            "specific nodepools.",
            errors[0],
        )

    def test_tpu_topology_validator_missing_partitions(self):
        v = validators.topology.TpuTopologyValidator()
        errors = v.validate(
            {
                "machine_type": "tpu7x-standard-4t",
                "scheduler": "slurm",
                "scheduler_obj": mock.Mock(
                    nodes_arg=None, nodelists_arg=None, partitions_arg=None
                ),
            }
        )
        self.assertEqual(len(errors), 1)
        self.assertIn(
            "TPU tests need entire slice, please use --partitions to target "
            "specific partitions.",
            errors[0],
        )


if __name__ == "__main__":
    unittest.main()
