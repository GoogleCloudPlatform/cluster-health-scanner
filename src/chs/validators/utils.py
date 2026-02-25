"""Validation utilities for CHS."""

import click


def validate_non_empty_string(
    ctx: click.Context, param: click.Parameter, value: str | None
) -> str | None:
    """Validates that a provided string value is not empty.

    Args:
        ctx: Click context.
        param: Click parameter.
        value: The value to validate.

    Returns:
        The stripped string value, or None if input is None.

    Raises:
        click.BadParameter: If the value is empty or whitespace only.
    """
    if value is None:
        return None
    stripped_value = value.strip()
    if not stripped_value:
        raise click.BadParameter(
            "cannot be an empty string.", ctx=ctx, param=param
        )
    return stripped_value
