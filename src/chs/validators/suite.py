"""Validator for suites."""

from typing import Any

from chs.utils import suites
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

        supported_suites = suites.get_all_suites()
        if suite is None or suite in supported_suites:
            return []
        return [
            f"Suite '{suite}' is not supported. "
            "Supported suites: "
            f"{', '.join(supported_suites)}"
        ]
