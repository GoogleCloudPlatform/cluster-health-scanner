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

import enum
import chs.exceptions as ex


class TestStatus(enum.Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    ERROR = "ERROR"


class SubmissionStatus(enum.Enum):
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"


class TierManager:
    """Manages the execution graph for tiered diagnostic suites."""

    def __init__(self, tests_config: list[dict]):
        self._tests_config = tests_config
        self._cumulative_results = {}
        self._tiers = self._build_tiers_from_dag(tests_config)
        self._current_tier_index = 0

    def _build_tiers_from_dag(self, tests_config: list[dict]) -> list[dict]:
        configs = {t.get("name"): t for t in tests_config if t.get("name")}
        in_degrees = {test_name: 0 for test_name in configs}
        adj = {test_name: [] for test_name in configs}

        for test_name, config in configs.items():
            trigger = config.get("trigger_condition", {})
            for status, deps in trigger.items():
                for dep_name in deps:
                    if dep_name in configs:
                        adj[dep_name].append(test_name)
                        in_degrees[test_name] += 1

        tiers = []
        current_level = [
            test_name for test_name, deg in in_degrees.items() if deg == 0
        ]

        while current_level:
            tier_dict = {
                "tests": [configs[test_name] for test_name in current_level]
            }
            tiers.append(tier_dict)
            next_level = []
            for test_name in current_level:
                for neighbor in adj[test_name]:
                    in_degrees[neighbor] -= 1
                    if in_degrees[neighbor] == 0:
                        next_level.append(neighbor)
            next_level.sort()
            current_level = next_level

        if sum(in_degrees.values()) > 0:
            raise ex.ChsException(
                "Cycle detected in internal test dependencies configuration"
            )

        return tiers

    def fast_forward_to_tier(self, completed_tier: int) -> None:
        """Safely fast-forwards internal pointer past the completed tier."""
        if completed_tier < 1 or completed_tier > len(self._tiers) + 1:
            raise ex.ChsAnalysisException(
                f"Invalid completed tier: {completed_tier}"
            )
        self._current_tier_index = completed_tier

    def _parse_status(self, status_str: str) -> TestStatus:
        """Parses a string into a TestStatus.

        Raises InvalidTestConfigException if the status string is invalid.
        """
        try:
            return TestStatus(status_str.upper())
        except ValueError as e:
            raise ex.InvalidTestConfigException(
                f"Invalid test status: {status_str}"
            ) from e

    def _is_condition_met(self, test_config: dict) -> bool:
        """Checks if the trigger conditions for a test are met.

        The current implementation acts as a pure OR gate. A test will trigger
        if *any* of the conditions specified in the trigger_condition are met.
        For example, if trigger_condition is:
        {"DEGRADED": ["tpu-gemm"], "ERROR": ["tpu-hbm"]},
        the test will run if gemm is DEGRADED OR hbm is ERROR.
        This design currently suits a "fail-fast" use case. To implement AND
        conditions (e.g., run only if both A and B fail), this logic would
        need to be revisited.
        """
        trigger = test_config.get("trigger_condition")
        if not trigger:
            return True

        for status_str, deps in trigger.items():
            if isinstance(deps, str):
                deps = [deps]
            required_status = self._parse_status(status_str)
            for dep_name in deps:
                actual_status = self._cumulative_results.get(
                    dep_name, TestStatus.DEGRADED
                )
                if actual_status == required_status:
                    return True

        return False

    def get_next_tier_tests(
        self, recent_results: dict[str, TestStatus]
    ) -> list[str] | None:
        """Evaluates results and returns tests for next tier,
        or None if done.
        """
        self._cumulative_results.update(recent_results)

        if self._current_tier_index >= len(self._tiers):
            return None

        current_tier = self._tiers[self._current_tier_index]
        self._current_tier_index += 1

        tests_to_run = [
            test_config.get("name")
            for test_config in current_tier.get("tests", [])
            if self._is_condition_met(test_config) and test_config.get("name")
        ]
        return tests_to_run

    def get_all_tests(self) -> list[str]:
        """Returns all tests config names"""
        return [t.get("name") for t in self._tests_config if t.get("name")]
