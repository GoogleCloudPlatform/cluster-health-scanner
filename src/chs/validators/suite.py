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
