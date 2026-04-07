"""Unit tests for the Cluster Health Scanner (CHS) validators package."""

import unittest
import click
from unittest import mock

from chs import validators


class TestValidators(unittest.TestCase):
    """Tests the validators package."""

    def test_machine_type_validator_valid(self):
        v = validators.MachineTypeValidator()
        errors = v.validate({"machine_type": "a3-mega"})
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
        errors = v.validate({"scheduler": "slurm", "machine_type": "a3-mega"})
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


if __name__ == "__main__":
    unittest.main()
