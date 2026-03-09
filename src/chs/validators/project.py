"""Composite validator for project configuration."""

from typing import Any

from chs.validators.base import Validator


class ProjectConfigValidator(Validator):
    """Composite validator to run multiple validators on the configuration.

    Aggregates errors from all provided validators.
    """

    def __init__(self, validators: list[Validator]):
        """Initialize with a list of validators.

        Args:
            validators: List of Validator instances to run.
        """
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
