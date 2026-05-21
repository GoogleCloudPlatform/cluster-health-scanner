"""Validator for machine types."""

from typing import Any

from chs.utils import tests as utils_tests
from chs.validators.base import Validator


class MachineTypeValidator(Validator):
    """Validates that the provided machine type is supported.

    Checks against the list of supported machine types defined in
    `chs.resources.SUPPORTED_MACHINE_TYPES`.
    """

    def validate(self, config: dict[str, Any]) -> list[str]:
        """Validates the machine type in the configuration.

        Args:
            config: Configuration dictionary containing 'machine_type'.

        Returns:
            List of error messages if validation fails, empty list otherwise.
        """
        machine_type = config.get("machine_type")

        if machine_type is None:
            return ["'machine_type' is missing in the configuration."]

        supported_machine_types = utils_tests.get_all_supported_machine_types()
        if machine_type not in supported_machine_types:
            return [
                f"Machine type '{machine_type}' is not supported. "
                "Supported types: "
                f"{', '.join(supported_machine_types)}"
            ]
        return []
