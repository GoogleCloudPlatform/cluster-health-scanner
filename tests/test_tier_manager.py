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

"""Tests for TierManager."""

import unittest

import chs.exceptions as ex
from chs.tier_manager import (
    TierManager,
    TestStatus as _TestStatus,
)


class TestTierManagerDAGBuilding(unittest.TestCase):
    """Tests for TierManager._build_tiers_from_dag."""

    def test_build_tiers_empty(self):
        """Empty test configs should produce empty tiers."""
        tm = TierManager([])
        self.assertEqual(tm._tiers, [])

    def test_build_tiers_independent(self):
        """Independent tests should all be placed in the first tier (Tier 0)."""
        configs = [
            {"name": "test1"},
            {"name": "test2"},
            {"name": "test3"},
        ]
        tm = TierManager(configs)
        self.assertEqual(len(tm._tiers), 1)
        # Check tests in Tier 0
        tier_0_names = [t["name"] for t in tm._tiers[0]["tests"]]
        self.assertCountEqual(tier_0_names, ["test1", "test2", "test3"])

    def test_build_tiers_simple_dependency(self):
        """A depends on B -> B is in Tier 0, A is in Tier 1."""
        configs = [
            {
                "name": "test_dependent",
                "trigger_condition": {"HEALTHY": ["test_base"]},
            },
            {"name": "test_base"},
        ]
        tm = TierManager(configs)
        self.assertEqual(len(tm._tiers), 2)
        self.assertEqual(
            [t["name"] for t in tm._tiers[0]["tests"]], ["test_base"]
        )
        self.assertEqual(
            [t["name"] for t in tm._tiers[1]["tests"]], ["test_dependent"]
        )

    def test_build_tiers_converging_dependency(self):
        """C depends on A and B -> A, B in Tier 0; C in Tier 1."""
        configs = [
            {
                "name": "test_c",
                "trigger_condition": {"HEALTHY": ["test_a", "test_b"]},
            },
            {"name": "test_a"},
            {"name": "test_b"},
        ]
        tm = TierManager(configs)
        self.assertEqual(len(tm._tiers), 2)
        self.assertCountEqual(
            [t["name"] for t in tm._tiers[0]["tests"]], ["test_a", "test_b"]
        )
        self.assertEqual([t["name"] for t in tm._tiers[1]["tests"]], ["test_c"])

    def test_build_tiers_diverging_dependency(self):
        """B depends on A; C depends on A -> A in Tier 0; B, C in Tier 1."""
        configs = [
            {"name": "test_b", "trigger_condition": {"HEALTHY": ["test_a"]}},
            {"name": "test_c", "trigger_condition": {"DEGRADED": ["test_a"]}},
            {"name": "test_a"},
        ]
        tm = TierManager(configs)
        self.assertEqual(len(tm._tiers), 2)
        self.assertEqual([t["name"] for t in tm._tiers[0]["tests"]], ["test_a"])
        self.assertCountEqual(
            [t["name"] for t in tm._tiers[1]["tests"]], ["test_b", "test_c"]
        )

    def test_build_tiers_multi_level_dependency(self):
        """C -> B; B -> A ==> A in Tier 0, B in Tier 1, C in Tier 2."""
        configs = [
            {"name": "test_c", "trigger_condition": {"HEALTHY": ["test_b"]}},
            {"name": "test_b", "trigger_condition": {"HEALTHY": ["test_a"]}},
            {"name": "test_a"},
        ]
        tm = TierManager(configs)
        self.assertEqual(len(tm._tiers), 3)
        self.assertEqual([t["name"] for t in tm._tiers[0]["tests"]], ["test_a"])
        self.assertEqual([t["name"] for t in tm._tiers[1]["tests"]], ["test_b"])
        self.assertEqual([t["name"] for t in tm._tiers[2]["tests"]], ["test_c"])

    def test_build_tiers_disjoint_graphs(self):
        """Disjoint dependency graphs should be correctly aligned depth."""
        configs = [
            {
                "name": "test_a_dep",
                "trigger_condition": {"HEALTHY": ["test_a_base"]},
            },
            {
                "name": "test_b_dep",
                "trigger_condition": {"HEALTHY": ["test_b_base"]},
            },
            {"name": "test_a_base"},
            {"name": "test_b_base"},
        ]
        tm = TierManager(configs)
        self.assertEqual(len(tm._tiers), 2)
        self.assertCountEqual(
            [t["name"] for t in tm._tiers[0]["tests"]],
            ["test_a_base", "test_b_base"],
        )
        self.assertCountEqual(
            [t["name"] for t in tm._tiers[1]["tests"]],
            ["test_a_dep", "test_b_dep"],
        )

    def test_build_tiers_cycle_detected(self):
        """A cycle in dependencies (A -> B -> A) should raise ChsException."""
        configs = [
            {"name": "test_a", "trigger_condition": {"HEALTHY": ["test_b"]}},
            {"name": "test_b", "trigger_condition": {"HEALTHY": ["test_a"]}},
        ]
        with self.assertRaisesRegex(
            ex.ChsException, "Cycle detected in internal test dependencies"
        ):
            TierManager(configs)

    def test_build_tiers_missing_dependency(self):
        """Test depends on a missing test name, the dependency is ignored."""
        configs = [
            {
                "name": "test_dependent",
                "trigger_condition": {"HEALTHY": ["missing_test"]},
            },
        ]
        tm = TierManager(configs)
        # test_dependent should be in Tier 0 since missing_test was skipped
        self.assertEqual(len(tm._tiers), 1)
        self.assertEqual(
            [t["name"] for t in tm._tiers[0]["tests"]], ["test_dependent"]
        )


class TestTierManagerExecution(unittest.TestCase):
    """Tests for TierManager.get_next_tier_tests."""

    def test_get_next_tier_tests_empty_tiers(self):
        """If no tests configured, get_next_tier_tests returns None."""
        tm = TierManager([])
        self.assertIsNone(tm.get_next_tier_tests({}))

    def test_get_next_tier_tests_independent_tests(self):
        """Independent tests run first, subsequent calls return None."""
        configs = [
            {"name": "test1"},
            {"name": "test2"},
        ]
        tm = TierManager(configs)

        # First call should return both tests
        tests = tm.get_next_tier_tests({})
        self.assertCountEqual(tests, ["test1", "test2"])

        # Second call should return None as we are past the only tier
        self.assertIsNone(tm.get_next_tier_tests({}))

    def test_get_next_tier_tests_simple_dependency_satisfied(self):
        """If dependency is satisfied, the dependent test runs next."""
        configs = [
            {
                "name": "test_dependent",
                "trigger_condition": {"HEALTHY": ["test_base"]},
            },
            {"name": "test_base"},
        ]
        tm = TierManager(configs)

        # Tier 0 runs first
        self.assertEqual(tm.get_next_tier_tests({}), ["test_base"])

        # Tier 1: since test_base is HEALTHY, test_dependent triggers.
        self.assertEqual(
            tm.get_next_tier_tests({"test_base": _TestStatus.HEALTHY}),
            ["test_dependent"],
        )

        # Tier 2 is past the end
        self.assertIsNone(tm.get_next_tier_tests({}))

    def test_get_next_tier_tests_simple_dependency_not_satisfied(self):
        """If dependency is not met, the dependent test does not run."""
        configs = [
            {
                "name": "test_dependent",
                "trigger_condition": {"HEALTHY": ["test_base"]},
            },
            {"name": "test_base"},
        ]
        tm = TierManager(configs)

        # Tier 0 runs first
        self.assertEqual(tm.get_next_tier_tests({}), ["test_base"])

        # Tier 1: test_base is ERROR (not HEALTHY), dependent is skipped.
        self.assertEqual(
            tm.get_next_tier_tests({"test_base": _TestStatus.ERROR}), []
        )

    def test_get_next_tier_tests_or_logic_satisfied(self):
        """Verification of the OR gate logic for trigger conditions."""
        configs = [
            {
                "name": "test_dependent",
                "trigger_condition": {
                    "HEALTHY": ["test_a"],
                    "ERROR": ["test_b"],
                },
            },
            {"name": "test_a"},
            {"name": "test_b"},
        ]
        tm = TierManager(configs)

        # Tier 0 runs test_a and test_b
        self.assertCountEqual(tm.get_next_tier_tests({}), ["test_a", "test_b"])

        # Case 1: test_b is ERROR, matching trigger -> triggers.
        self.assertEqual(
            tm.get_next_tier_tests(
                {"test_a": _TestStatus.DEGRADED, "test_b": _TestStatus.ERROR}
            ),
            ["test_dependent"],
        )

    def test_get_next_tier_tests_or_logic_not_satisfied(self):
        """OR gate logic when none of the conditions are met."""
        configs = [
            {
                "name": "test_dependent",
                "trigger_condition": {
                    "HEALTHY": ["test_a"],
                    "ERROR": ["test_b"],
                },
            },
            {"name": "test_a"},
            {"name": "test_b"},
        ]
        tm = TierManager(configs)

        # Tier 0 runs test_a and test_b
        self.assertCountEqual(
            tm.get_next_tier_tests({}),
            ["test_a", "test_b"],
        )

        # Case 2: neither conditions are met -> skips.
        self.assertEqual(
            tm.get_next_tier_tests(
                {"test_a": _TestStatus.DEGRADED, "test_b": _TestStatus.DEGRADED}
            ),
            [],
        )

    def test_get_next_tier_tests_missing_result_defaults_to_degraded(self):
        """Missing dependency results default to DEGRADED."""
        configs = [
            {
                "name": "test_dependent",
                "trigger_condition": {"DEGRADED": ["test_base"]},
            },
            {"name": "test_base"},
        ]
        tm = TierManager(configs)

        # Tier 0 runs test_base
        self.assertEqual(tm.get_next_tier_tests({}), ["test_base"])

        # Tier 1: no results passed, test_base defaults to DEGRADED.
        self.assertEqual(
            tm.get_next_tier_tests({}),
            ["test_dependent"],
        )

    def test_get_next_tier_tests_invalid_status_raises_error(self):
        """Invalid status in trigger condition raises Exception."""
        configs = [
            {
                "name": "test_dependent",
                "trigger_condition": {"INVALID_STATUS_STR": ["test_base"]},
            },
            {"name": "test_base"},
        ]
        tm = TierManager(configs)

        # Tier 0 runs test_base
        self.assertEqual(tm.get_next_tier_tests({}), ["test_base"])

        # Tier 1 evaluates. INVALID_STATUS_STR raises Exception.
        with self.assertRaises(ex.InvalidTestConfigException):
            tm.get_next_tier_tests({"test_base": _TestStatus.DEGRADED})


if __name__ == "__main__":
    unittest.main()
