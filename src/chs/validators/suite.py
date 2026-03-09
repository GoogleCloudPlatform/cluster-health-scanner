"""Validator for suites."""

from typing import Any

from chs import resources
from chs.validators.base import Validator


class SuiteValidator(Validator):
    """Validates that the provided suite is supported.

    Checks against the list of supported suites defined in
    `chs.resources.SUPPORTED_SUITES`.
    """

    def validate(self, config: dict[str, Any]) -> list[str]:
        """Validates the suite in the configuration.

        Args:
            config: Configuration dictionary containing 'suite'.

        Returns:
            List of error messages if validation fails, empty list otherwise.
        """
        suite = config.get("suite")

        if suite is None:
            return ["'suite' is missing in the configuration."]
        if suite in resources.SUPPORTED_SUITES:
            return []
        return [
            f"Suite '{suite}' is not supported. "
            "Supported suites: "
            f"{', '.join(list(resources.SUPPORTED_SUITES.keys()))}"
        ]
