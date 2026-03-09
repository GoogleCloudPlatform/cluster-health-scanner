"""Validator for tests."""

from typing import Any
from chs import resources
from chs.validators.base import Validator


class TestsValidator(Validator):
    """Validates that the provided tests are supported.
    Checks against the list of supported tests defined in
    `chs.resources.SUPPORTED_TESTS`.
    """

    def validate(self, config: dict[str, Any]) -> list[str]:
        """Validates the tests in the configuration.
        Args:
            config: Configuration dictionary containing 'tests' and optionally
              'machine_type'.
        Returns:
            List of error messages if validation fails, empty list otherwise.
        """
        tests = config.get("tests")
        machine_type = config.get("machine_type")
        if not tests:
            return []
        supported_tests = resources.SUPPORTED_TESTS
        if machine_type:
            supported_tests = resources.TESTS_FOR_MACHINE_TYPE.get(
                machine_type, []
            )
        invalid_tests = [t for t in tests if t not in supported_tests]
        if invalid_tests:
            msg = f"Tests {invalid_tests} are not supported"
            if machine_type:
                msg += f" for machine type '{machine_type}'"
            msg += f". Supported tests: {', '.join(supported_tests)}"
            return [msg]
        return []
