"""Composite validator for project configuration."""

from typing import Any

from chs.validators.base import Validator
from chs.validators.machine_type import MachineTypeValidator
from chs.validators.scheduler import SchedulerValidator
from chs.validators.suite import SuiteValidator
from chs.validators.tests import TestsValidator
from chs.validators.thresholds import ThresholdsValidator


class ProjectConfigValidator(Validator):
    """Composite validator to run multiple validators on the configuration.

    Aggregates errors from all provided validators.
    """

    def __init__(self):
        """Initialize with a list of validators."""

        validators = [
            SchedulerValidator(),
            MachineTypeValidator(),
            SuiteValidator(),
            TestsValidator(),
            ThresholdsValidator(),
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
