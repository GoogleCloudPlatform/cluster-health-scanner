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

import os
from typing import Any

import yaml

from chs.testkit import config as chs_config
from chs.testkit import loader


def load_test_thresholds(test_name: str) -> dict[str, Any]:
    path = os.path.join(chs_config.TESTS_ROOT, test_name, "thresholds.yaml")
    if not os.path.exists(path):
        return {}
    with open(path, "r") as handle:
        return yaml.safe_load(handle) or {}


def build_thresholds() -> dict[str, Any]:
    """Groups every test's thresholds by machine."""
    thresholds_by_machine_type: dict[str, Any] = {}
    for test_name, _ in loader.enabled_tests():
        test_thresholds = load_test_thresholds(test_name)
        if not test_thresholds:
            continue
        threshold_type = test_thresholds.pop("type")
        for machine_type, metrics in test_thresholds.items():
            thresholds_by_machine_type.setdefault(machine_type, {})[
                test_name
            ] = {"type": threshold_type, "thresholds": metrics}
    return thresholds_by_machine_type
