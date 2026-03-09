"""Base validator protocol for CHS."""

from typing import Any, Protocol


class Validator(Protocol):
    """Abstract base class for validators.

    All specific validators should implement this protocol to ensure
    consistency in validation logic.
    """

    def validate(self, config: dict[str, Any]) -> list[str]:
        """Validates the configuration and returns a list of error messages.

        Args:
            config: A dictionary containing the configuration to validate.

        Returns:
            A list of error strings. If the list is empty, validation passed.
        """
        ...
