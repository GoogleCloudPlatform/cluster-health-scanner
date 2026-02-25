"""Validator for schedulers."""

from typing import Any
from chs import resources
from chs.validators.base import Validator


class SchedulerValidator(Validator):
    """Validates that the provided scheduler is supported.

    Checks against the list of supported schedulers defined in
    `chs.resources.SUPPORTED_SCHEDULERS`.
    """

    def validate(self, config: dict[str, Any]) -> list[str]:
        """Validates the scheduler in the configuration.

        Args:
            config: Configuration dictionary containing 'scheduler'.

        Returns:
            List of error messages if validation fails, empty list otherwise.
        """
        scheduler = config.get("scheduler")
        if scheduler is None:
            return ["'scheduler' is missing in the configuration."]
        if scheduler not in resources.SUPPORTED_SCHEDULERS:
            return [
                f"Scheduler '{scheduler}' is not supported. "
                "Supported schedulers: "
                f"{', '.join(resources.SUPPORTED_SCHEDULERS)}"
            ]
        return []
