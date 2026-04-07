"""Validator for tests."""

from typing import Any
from chs import resources
from chs.validators.base import Validator
from chs.validators.machine_type import MachineTypeValidator
from chs.validators.scheduler import SchedulerValidator


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
        scheduler = config.get("scheduler")

        if (
            (scheduler and SchedulerValidator().validate(config))
            or (machine_type and MachineTypeValidator().validate(config))
            or (not tests)
        ):
            return []
        supported_tests = resources.SUPPORTED_TESTS
        if scheduler and machine_type:
            tests_map = resources.MAP_TESTS_BY_SCHEDULER_AND_MACHINE_TYPE.get(
                scheduler
            )
            if tests_map:
                supported_tests = tests_map.get(machine_type, [])
        invalid_tests = [t for t in tests if t not in supported_tests]
        if invalid_tests:
            msg = f"Tests {invalid_tests} are not supported"
            if scheduler and machine_type:
                msg += f" for scheduler '{scheduler}' "
                msg += f"and machine type '{machine_type}'"
            msg += f". Supported tests: {', '.join(supported_tests)}"
            return [msg]
        return []
