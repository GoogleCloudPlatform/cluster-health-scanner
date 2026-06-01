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

import logging
from typing import Any

from chs.utils.config import load_yaml_config

from chs.exceptions import SuiteNotFoundException

logger = logging.getLogger(__name__)


def load_suite(suite_name: str) -> dict[str, Any]:
    """Loads the full suite configuration from suites.yaml.

    Args:
        suite_name: The name of the suite to load.

    Returns:
        A dictionary containing the suite configuration.

    Raises:
        SuiteNotFoundException: If the suite is not found in suites.yaml.
        FileNotFoundError: If suites.yaml is not found.
    """
    suites_data = _load_suites()
    suites = suites_data.get("suites", [])

    for suite in suites:
        if suite.get("name") == suite_name:
            return suite

    available_suites = [s.get("name") for s in suites]
    raise SuiteNotFoundException(
        f"Suite '{suite_name}' not found in suites.yaml. "
        f"Available suites: {', '.join(available_suites)}"
    )


def _load_suites() -> dict[str, Any]:
    """Loads the full suites configuration from suites.yaml."""
    return load_yaml_config("suites.yaml")


def get_all_suites() -> list[str]:
    """Derives list of all suite names from suites.yaml."""
    data = _load_suites()
    return [suite.get("name") for suite in data.get("suites", [])]
