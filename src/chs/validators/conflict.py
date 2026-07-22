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

"""Validator for conflicting CLI arguments."""

from typing import Any

from chs.validators.base import Validator


class ConflictValidator(Validator):
    """Validates that mutually exclusive CLI arguments are not used together."""

    def validate(self, config: dict[str, Any]) -> list[str]:
        """Validates conflicting configuration options.

        Args:
            config: Configuration dictionary containing tests, suite, nodes,
                nodelists, nodepools, and partitions.

        Returns:
            List of error messages if validation fails, otherwise an empty list.
        """
        errors = []

        tests = config.get("tests")
        suite = config.get("suite")

        if tests and suite:
            errors.append(
                "Cannot specify both --tests/-t and --suite/-s at "
                "the same time."
            )

        targeting_flags = sum(
            bool(config.get(arg))
            for arg in ["nodes", "nodelists", "nodepools", "partitions"]
        )
        if targeting_flags > 1:
            errors.append(
                "Cannot specify more than one targeting option (--nodes, "
                "--nodelists, --nodepools, --partitions)."
            )

        return errors
