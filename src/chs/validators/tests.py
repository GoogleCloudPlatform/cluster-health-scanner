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

"""Validator for tests."""

from typing import Any
from chs.utils import tests as utils_tests
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
        supported_tests = utils_tests.get_all_supported_tests()
        if scheduler and machine_type:
            tests_map = utils_tests.get_tests_map().get(scheduler)
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
