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

"""Unit tests for chs.testkit.spec."""

import sys
import unittest
from unittest import mock

from chs.testkit import (
    STRIPPED_WORKSPACE_KEYS,
    ExperimentSpec,
    build_test_request,
)


def machine_configs(**app_config):
    return {
        "machine-x": {
            "variables": {"machine_wide": "yes"},
            "demo": app_config,
        }
    }


class SuiteEntry:
    """Stands in for the suite entry the engine parses out of the workspace."""

    def __init__(self, name, workloads=None, suite_variables=None):
        self.name = name
        self.workloads = workloads or []
        self.variables = suite_variables or {}


class BuildTestRequestTest(unittest.TestCase):
    """Assembling one test's request from the workspace and the suite."""

    def base_vars(self, **overrides):
        values = {
            "scheduler": "kubernetes",
            "machine_type": "machine-x",
            "nodes": ["node-1", "node-2"],
            "n_nodes": 1,
            "n_ranks": 1,
            "processes_per_node": 1,
            "jobset_template": "wrapper",
            "mpi_command": "mpirun",
            "container_image": "wrapper-image",
            "srun_args": "--exclusive",
        }
        values.update(overrides)
        return values

    def test_workspace_defaults_do_not_reach_the_test(self):
        request = build_test_request(
            self.base_vars(), SuiteEntry("demo"), "chs-run"
        )

        for key in STRIPPED_WORKSPACE_KEYS:
            self.assertNotIn(key, request.variables)
        self.assertEqual("machine-x", request.machine_type)
        self.assertEqual(["node-1", "node-2"], request.nodes)

    def test_the_caller_keeps_its_own_dictionary(self):
        base_dict = self.base_vars()
        build_test_request(base_dict, SuiteEntry("demo"), "chs-run")

        self.assertEqual(1, base_dict["n_nodes"])
        self.assertNotIn("chs_test_name", base_dict)

    def test_the_test_name_is_recorded(self):
        request = build_test_request(
            self.base_vars(), SuiteEntry("demo"), "chs-run"
        )

        self.assertEqual("demo", request.variables["chs_test_name"])
        self.assertEqual("chs-run", request.base_prefix)

    def test_suite_variables_win_over_workspace_variables(self):
        request = build_test_request(
            self.base_vars(scheduler="kubernetes"),
            SuiteEntry("demo", suite_variables={"scheduler": "slurm"}),
            "chs-run",
        )

        self.assertEqual("slurm", request.scheduler)

    def test_a_single_workload_is_not_split_into_letters(self):
        request = build_test_request(
            self.base_vars(), SuiteEntry("demo", workloads="all-reduce"), "p"
        )

        self.assertEqual(("all-reduce",), request.workloads)

    def test_no_workloads_leaves_the_choice_to_the_plugin(self):
        request = build_test_request(
            self.base_vars(), SuiteEntry("demo"), "chs-run"
        )

        self.assertEqual((), request.workloads)

    def test_no_configuration_is_carried_unless_one_is_given(self):
        request = build_test_request(
            self.base_vars(), SuiteEntry("demo"), "chs-run"
        )

        self.assertIsNone(request.configs)

    def test_a_supplied_configuration_is_carried_through(self):
        configs = machine_configs(variables={"threads": 8})
        request = build_test_request(
            self.base_vars(), SuiteEntry("demo"), "chs-run", configs=configs
        )

        self.assertIs(configs, request.configs)

    def test_the_workspace_directory_is_carried_through(self):
        request = build_test_request(
            self.base_vars(),
            SuiteEntry("demo"),
            "chs-run",
            workspace_root="/tmp/ws",
        )

        self.assertEqual("/tmp/ws", request.workspace_root)


class EffectiveTimeoutTest(unittest.TestCase):
    """Reconciling the suite's deadline with a test's own timeout."""

    def effective(self, deadline, timeout):
        entry = SuiteEntry("demo")
        if timeout is not None:
            entry.variables = {"timeout": timeout}
        request = build_test_request(
            {}, entry, "chs-run", suite_deadline=deadline
        )
        return request.variables.get("timeout")

    def test_the_shorter_of_the_two_wins(self):
        self.assertEqual(600, self.effective(1800, 600))
        self.assertEqual(600, self.effective(600, 1800))

    def test_either_one_alone_is_used_as_is(self):
        self.assertEqual(1800, self.effective(1800, None))
        self.assertEqual(600, self.effective(None, 600))

    def test_neither_one_leaves_the_timeout_unset(self):
        self.assertIsNone(self.effective(None, None))

    def test_numeric_strings_are_accepted(self):
        self.assertEqual(600, self.effective("1800", "600"))

    def test_an_unreadable_deadline_falls_back_to_the_test(self):
        self.assertEqual(600, self.effective("soon", 600))

    def test_an_unreadable_timeout_falls_back_to_the_deadline(self):
        self.assertEqual(1800, self.effective(1800, "soon"))

    def test_two_unreadable_values_leave_the_timeout_alone(self):
        self.assertEqual("soon", self.effective("later", "soon"))

    def test_the_workspace_deadline_is_used_when_none_is_given(self):
        request = build_test_request(
            {"deadline": 1800}, SuiteEntry("demo"), "chs-run"
        )

        self.assertEqual(1800, request.variables["timeout"])

    def test_the_test_still_wins_when_it_asks_for_less(self):
        request = build_test_request(
            {"deadline": 1800},
            SuiteEntry("demo", suite_variables={"timeout": 600}),
            "chs-run",
        )

        self.assertEqual(600, request.variables["timeout"])

    def test_an_explicit_deadline_beats_the_workspace_one(self):
        request = build_test_request(
            {"deadline": 1800},
            SuiteEntry("demo"),
            "chs-run",
            suite_deadline=600,
        )

        self.assertEqual(600, request.variables["timeout"])

    def test_the_deadline_is_written_into_the_variables(self):
        request = build_test_request(
            {"deadline": 1800},
            SuiteEntry("demo"),
            "chs-run",
            suite_deadline=600,
        )

        self.assertEqual(600, request.variables["deadline"])

    def test_a_test_level_deadline_is_not_overwritten(self):
        request = build_test_request(
            {"deadline": 1800},
            SuiteEntry("demo", suite_variables={"deadline": 300}),
            "chs-run",
            suite_deadline=600,
        )

        self.assertEqual(300, request.variables["deadline"])

    def test_an_unreadable_deadline_is_left_as_it_was(self):
        request = build_test_request(
            {"deadline": "later"}, SuiteEntry("demo"), "chs-run"
        )

        self.assertEqual("later", request.variables["deadline"])


class ExperimentSpecTest(unittest.TestCase):
    """The single point where a built experiment reaches Ramble."""

    def test_a_spec_is_handed_to_ramble_by_name_and_context(self):
        created = mock.Mock()
        ramble_context = mock.Mock(create_context_from_dict=created)

        with mock.patch.dict(
            sys.modules,
            {"ramble": mock.Mock(), "ramble.context": ramble_context},
        ):
            spec = ExperimentSpec(
                application="demo",
                workload="check",
                name="chs-run-demo-check",
                context={"variables": {"a": 1}},
            )
            result = spec.to_ramble_context()

        created.assert_called_once_with(
            "chs-run-demo-check", {"variables": {"a": 1}}
        )
        self.assertIs(created.return_value, result)


if __name__ == "__main__":
    unittest.main()
