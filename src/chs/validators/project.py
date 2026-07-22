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

"""Composite validator for project configuration."""

from typing import Any

from chs.validators.base import Validator
from chs.validators.machine_type import MachineTypeValidator
from chs.validators.scheduler import SchedulerValidator
from chs.validators.tests import TestsValidator


class ProjectConfigValidator(Validator):
    """Composite validator to run multiple validators on the configuration.

    Aggregates errors from all provided validators.
    """

    def __init__(self):
        """Initialize with a list of validators."""

        validators = [
            SchedulerValidator(),
            MachineTypeValidator(),
            TestsValidator(),
        ]
        self.validators = validators

    def validate(self, config: dict[str, Any]) -> list[str]:
        """Validates the configuration using all registered validators.

        Args:
            config: Configuration dictionary.

        Returns:
            Aggregated list of error messages from all validators.
        """
        errors = []
        for validator in self.validators:
            errors.extend(validator.validate(config))
        return errors
