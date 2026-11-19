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

from chs import resources
from chs.exceptions import SuiteNotFoundException
from chs.utils.config import load_yaml_config

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


def get_all_suites_details() -> list[dict[str, Any]]:
    """Derives list of all suite details from suites.yaml."""
    data = _load_suites()
    return data.get("suites", [])


def is_suite_tiered(suite_cfg: dict) -> bool:
    """Checks if a suite defines any tests with dependencies."""
    return any("trigger_condition" in t for t in suite_cfg.get("tests", []))


def get_suite_deadline(
    suite_cfg: dict[str, Any],
    machine_type: str = "",
    default: int = resources.DEFAULT_DEADLINE,
) -> int:
    """Resolves a suite's deadline (seconds) for the target machine type."""
    raw_deadline = suite_cfg.get("deadline", default)
    if isinstance(raw_deadline, dict):
        cfg_default = raw_deadline.get("default", default)
        if resources.is_tpu(machine_type):
            raw_deadline = raw_deadline.get("tpu", cfg_default)
        elif resources.is_gpu(machine_type):
            raw_deadline = raw_deadline.get("gpu", cfg_default)
        else:
            raw_deadline = cfg_default
    try:
        return int(raw_deadline)
    except (ValueError, TypeError):
        return default
