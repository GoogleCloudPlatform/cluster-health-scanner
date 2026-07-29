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

"""Validation utilities for CHS."""

import re

import click


def validate_non_empty_string(
    ctx: click.Context,
    param: click.Parameter,
    value: tuple[str, ...] | str | None,
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

    if isinstance(value, tuple):
        if not value:
            return None
        cleaned = [v.strip() for v in value if v and v.strip()]
        value = ",".join(cleaned)

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
