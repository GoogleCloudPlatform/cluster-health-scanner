"""Unit tests for the Cluster Health Scanner (CHS) validators module."""

import unittest
from unittest import mock
from chs import validators
import click


class TestValidators(unittest.TestCase):
    """Tests the validation logic in validators.py."""

    def test_machine_type_validator_valid(self):
        """Test MachineTypeValidator with valid input."""
        validator = validators.MachineTypeValidator()
        errors = validator.validate({"machine_type": "a3-ultra"})
        self.assertEqual(errors, [])

    def test_machine_type_validator_invalid(self):
        """Test MachineTypeValidator with invalid input."""
        validator = validators.MachineTypeValidator()
        errors = validator.validate({"machine_type": "unsupported"})
        self.assertEqual(len(errors), 1)
        self.assertIn("not supported", errors[0])

    def test_machine_type_validator_missing(self):
        """Test MachineTypeValidator with missing input."""
        validator = validators.MachineTypeValidator()
        errors = validator.validate({})
        self.assertEqual(len(errors), 1)
        self.assertIn(
            "'machine_type' is missing in the configuration.", errors[0]
        )

    def test_scheduler_validator_valid(self):
        """Test SchedulerValidator with valid input."""
        validator = validators.SchedulerValidator()
        errors = validator.validate({"scheduler": "slurm"})
        self.assertEqual(errors, [])

    def test_scheduler_validator_invalid(self):
        """Test SchedulerValidator with invalid input."""
        validator = validators.SchedulerValidator()
        errors = validator.validate({"scheduler": "unsupported"})
        self.assertEqual(len(errors), 1)
        self.assertIn("not supported", errors[0])

    def test_scheduler_validator_missing(self):
        """Test SchedulerValidator with missing input."""
        validator = validators.SchedulerValidator()
        errors = validator.validate({})
        self.assertEqual(len(errors), 1)
        self.assertIn("'scheduler' is missing in the configuration.", errors[0])

    def test_tests_validator_valid(self):
        """Test TestsValidator with valid input."""
        validator = validators.TestsValidator()
        errors = validator.validate({"tests": ["nccl-cluster"]})
        self.assertEqual(errors, [])

    def test_tests_validator_invalid(self):
        """Test TestsValidator with invalid input."""
        validator = validators.TestsValidator()
        errors = validator.validate({"tests": ["invalid-test"]})
        self.assertEqual(len(errors), 1)
        self.assertIn("not supported", errors[0])

    def test_project_config_validator(self):
        """Test ProjectConfigValidator aggregates errors."""
        mock_v1 = mock.Mock()
        mock_v1.validate.return_value = ["error1"]
        mock_v2 = mock.Mock()
        mock_v2.validate.return_value = ["error2"]
        validator = validators.ProjectConfigValidator([mock_v1, mock_v2])
        errors = validator.validate({})
        self.assertEqual(errors, ["error1", "error2"])

    def test_validate_non_empty_string_valid(self):
        """Test validate_non_empty_string with valid input."""
        result = validators.validate_non_empty_string(None, None, "  value  ")
        self.assertEqual(result, "value")

    def test_validate_non_empty_string_none(self):
        """Test validate_non_empty_string with None input."""
        result = validators.validate_non_empty_string(None, None, None)
        self.assertIsNone(result)

    def test_validate_non_empty_string_empty(self):
        """Test validate_non_empty_string with empty/whitespace input."""
        with self.assertRaises(click.BadParameter):
            validators.validate_non_empty_string(None, None, "")
        with self.assertRaises(click.BadParameter):
            validators.validate_non_empty_string(None, None, "   ")

    def test_suite_validator_valid(self):
        """Test SuiteValidator with valid input."""
        validator = validators.SuiteValidator()
        errors = validator.validate({"suite": "standard"})
        self.assertEqual(errors, [])

    def test_suite_validator_invalid(self):
        """Test SuiteValidator with invalid input."""
        validator = validators.SuiteValidator()
        errors = validator.validate({"suite": "invalid-suite"})
        self.assertEqual(len(errors), 1)
        self.assertIn("not supported", errors[0])

    def test_suite_validator_missing(self):
        """Test SuiteValidator with missing input."""
        validator = validators.SuiteValidator()
        errors = validator.validate({})
        self.assertEqual(len(errors), 1)
        self.assertIn("'suite' is missing in the configuration.", errors[0])


if __name__ == "__main__":
    unittest.main()
