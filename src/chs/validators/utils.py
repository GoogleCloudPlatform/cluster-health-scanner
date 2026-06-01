"""Validation utilities for CHS."""

import click
import re


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


def validate_run_id(
    ctx: click.Context, param: click.Parameter, value: str | None
) -> str | None:
    """Validates that the user-provided run ID is DNS-1123 compliant.

    Must be 20 characters or less, contain only lowercase alphanumeric
    characters or '-', and must start and end with an alphanumeric character.
    """
    if value is None:
        return None
    val = validate_non_empty_string(ctx, param, value)
    if val is None:
        return None

    errors = []
    if len(val) > 20:
        errors.append("Run ID must be 20 characters or less.")

    dns_pattern = r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$"
    if not re.match(dns_pattern, val):
        errors.append(
            "Run ID must consist of lowercase alphanumeric characters or '-' "
            "and must start and end with an alphanumeric character (DNS-1123 "
            "compliant)."
        )

    if errors:
        raise click.BadParameter("\n".join(errors), ctx=ctx, param=param)

    return val
