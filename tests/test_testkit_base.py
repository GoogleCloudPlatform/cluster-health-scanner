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

"""Unit tests for chs.testkit.base."""

import unittest
from unittest import mock

from chs.testkit import (
    ClusterTestPlugin,
    NodeTestPlugin,
    SliceTestPlugin,
    TestRequest,
)
from chs.testkit import base


def machine_configs(**app_config):
    return {
        "machine-x": {
            "variables": {"machine_wide": "yes"},
            "demo": app_config,
        }
    }


def variables(**overrides):
    values = {
        "scheduler": "kubernetes",
        "machine_type": "machine-x",
        "nodes": ["node-1", "node-2"],
        "timeout": 1200,
        "deadline": 3600,
    }
    values.update(overrides)
    return values


class DemoNodePlugin(NodeTestPlugin):
    name = "demo"
    application = "demo"
    default_workloads = ("check", "verify")


class PluginContractTest(unittest.TestCase):
    def request(self, **overrides):
        return TestRequest(
            variables=variables(),
            configs=machine_configs(variables={}),
            base_prefix="chs-run",
            **overrides,
        )

    def test_a_plugin_must_say_what_it_is(self):
        for attributes, message in (
            ({"application": "a", "default_workloads": ("w",)}, "test name"),
            ({"name": "a", "default_workloads": ("w",)}, "Ramble application"),
            ({"name": "a", "application": "a"}, "default workload"),
        ):
            with self.subTest(attributes=attributes):
                plugin_class = type("P", (NodeTestPlugin,), attributes)
                with self.assertRaisesRegex(ValueError, message):
                    plugin_class()

    def test_the_configuration_key_falls_back_to_the_application(self):
        self.assertEqual(DemoNodePlugin().config_name, "demo")

        renamed = type("P", (DemoNodePlugin,), {"config_key": "demo-variant"})()
        self.assertEqual(renamed.config_name, "demo-variant")

    def test_the_suite_choice_of_workloads_wins(self):
        plugin = DemoNodePlugin()

        self.assertEqual(
            plugin.resolve_workloads(self.request()), ("check", "verify")
        )
        self.assertEqual(
            plugin.resolve_workloads(self.request(workloads=("only",))),
            ("only",),
        )

    def test_a_single_workload_given_as_text_is_not_split_up(self):
        plugin = DemoNodePlugin()

        self.assertEqual(
            plugin.resolve_workloads(self.request(workloads="solo")), ("solo",)
        )

    def test_experiment_names_follow_the_declared_shape(self):
        plugin = DemoNodePlugin()
        self.assertEqual(
            plugin.experiment_name("chs-run", "check"), "chs-run-demo-check"
        )

        quiet = type(
            "P", (DemoNodePlugin,), {"workload_in_experiment_name": False}
        )()
        self.assertEqual(
            quiet.experiment_name("chs-run", "check"), "chs-run-demo"
        )

        suffixed = type(
            "P",
            (DemoNodePlugin,),
            {
                "workload_in_experiment_name": False,
                "experiment_suffix": "-l{lvl}",
            },
        )()
        self.assertEqual(
            suffixed.experiment_name("chs-run", "check"), "chs-run-demo-l{lvl}"
        )

    def test_a_test_below_its_node_floor_is_skipped(self):
        plugin = type("P", (DemoNodePlugin,), {"minimum_nodes": 5})()

        with mock.patch.object(base, "logger") as logger:
            self.assertEqual(plugin.build(self.request()), [])

        logger.warning.assert_called_once()
        self.assertIn("minimum of %d nodes", logger.warning.call_args[0][0])
        self.assertEqual(logger.warning.call_args[0][2], 5)

    def test_a_test_that_has_the_nodes_it_asked_for_runs(self):
        plugin = type("P", (DemoNodePlugin,), {"minimum_nodes": 2})()

        self.assertTrue(plugin.build(self.request()))

    def test_a_plugin_is_identifiable_in_a_log_line(self):
        self.assertEqual("<DemoNodePlugin demo>", repr(DemoNodePlugin()))

    def test_the_job_name_is_shared_by_every_experiment_of_a_workload(self):
        plugin = DemoNodePlugin()

        specs = plugin.build(self.request())

        self.assertEqual(
            [spec.name for spec in specs],
            ["chs-run-demo-check-{node}", "chs-run-demo-verify-{node}"],
        )
        self.assertEqual(
            specs[0].context["variables"]["job_name"], "chs-run-demo-check"
        )

    def test_a_plugin_does_not_modify_the_request(self):
        plugin = DemoNodePlugin()
        request = self.request()

        plugin.build(request)

        self.assertNotIn("job_name", request.variables)
        self.assertEqual(request.configs, machine_configs(variables={}))

    def test_class_level_extras_are_not_shared_between_experiments(self):
        plugin = type(
            "P",
            (SliceTestPlugin,),
            {
                "name": "demo",
                "application": "demo",
                "default_workloads": ("check",),
                "extra_context": {"modifiers": []},
            },
        )()

        specs = plugin.build(
            TestRequest(
                variables=variables(
                    tpu_slices={"a": ["node-1"], "b": ["node-2"]}
                ),
                configs=machine_configs(modifiers=[{"name": "detector"}]),
                base_prefix="chs-run",
            )
        )

        self.assertEqual(len(specs), 2)
        for spec in specs:
            self.assertEqual(spec.context["modifiers"], [{"name": "detector"}])
        self.assertEqual(plugin.extra_context, {"modifiers": []})

    def test_a_test_that_cannot_name_workloads_keeps_its_own(self):
        """Without the workload in the name, two workloads would collide."""
        plugin = type(
            "P", (DemoNodePlugin,), {"workload_in_experiment_name": False}
        )()

        self.assertFalse(plugin.accepts_workload_override)
        self.assertEqual(
            plugin.resolve_workloads(self.request(workloads=("only",))),
            ("check", "verify"),
        )

    def test_the_test_name_reaches_the_context(self):
        specs = DemoNodePlugin().build(self.request())

        for spec in specs:
            self.assertEqual(spec.context["variables"]["chs_test_name"], "demo")

    def test_shared_work_is_prepared_once_per_run_not_per_workload(self):
        calls = []

        class Plugin(ClusterTestPlugin):
            name = "demo"
            application = "demo"
            default_workloads = ("check", "verify")

            def grouping(self, request):
                calls.append(request)
                return base.NodeGrouping(groups=[request.nodes])

        specs = Plugin().build(self.request())

        self.assertEqual(len(specs), 2)
        self.assertEqual(len(calls), 1)

    def test_a_cluster_test_measures_the_whole_allocation_by_default(self):
        """Every shipped cluster test overrides this with something finer."""
        plugin = type(
            "P",
            (ClusterTestPlugin,),
            {
                "name": "demo",
                "application": "demo",
                "default_workloads": ("check",),
            },
        )()

        specs = plugin.build(self.request())

        self.assertEqual(1, len(specs))
        self.assertEqual(
            ["node-1", "node-2"], specs[0].context["variables"]["nodes"]
        )

    def test_extra_context_may_not_replace_the_merged_variables(self):
        plugin = type(
            "P", (DemoNodePlugin,), {"extra_context": {"variables": {"a": 1}}}
        )()

        with self.assertRaisesRegex(ValueError, "must not contain 'variables'"):
            plugin.build(self.request())


if __name__ == "__main__":
    unittest.main()
