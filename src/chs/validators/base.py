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
