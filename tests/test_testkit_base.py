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
    MultiNodePlugin,
    SingleNodePlugin,
    SlicePlugin,
    TestRequest,
)


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


class DemoNodePlugin(SingleNodePlugin):
    name = "demo"
    application = "demo"
    default_workloads = ("check", "verify")


class DemoClusterPlugin(MultiNodePlugin):
    name = "demo"
    application = "demo"
    default_workloads = ("check", "verify")


class PluginContractTest(unittest.TestCase):
    def request(self, **overrides):
        defaults = {
            "variables": variables(),
            "configs": machine_configs(variables={}),
            "base_prefix": "chs-run",
        }
        defaults.update(overrides)
        return TestRequest(**defaults)

    def test_a_plugin_must_say_what_it_is(self):
        for attributes, message in (
            ({"application": "a", "default_workloads": ("w",)}, "test name"),
            ({"name": "a", "default_workloads": ("w",)}, "Ramble application"),
            (
                {"name": "a", "application": "a"},
                "default workload",
            ),
        ):
            with self.subTest(attributes=attributes):
                plugin_class = type("P", (SingleNodePlugin,), attributes)
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
        self.assertEqual(
            plugin.resolve_workloads(self.request(workloads="only")),
            ("only",),
        )

    def test_experiment_names_carry_only_the_pieces_configured(self):
        self.assertEqual(
            DemoNodePlugin().experiment_name("chs-1", "check"),
            "chs-1-demo-check",
        )

        fixed = type(
            "P",
            (DemoNodePlugin,),
            {
                "experiment_label": "custom",
                "experiment_suffix": "-tail",
                "workload_in_experiment_name": False,
            },
        )()
        self.assertEqual(
            fixed.experiment_name("chs-1", "check"), "chs-1-custom-tail"
        )

    def test_a_plugin_displays_its_type_and_name(self):
        self.assertEqual(repr(DemoNodePlugin()), "<DemoNodePlugin demo>")

    def test_produces_one_spec_per_workload_with_the_job_name_fixed(self):
        specs = DemoNodePlugin().build(self.request())

        self.assertEqual(
            [s.name for s in specs],
            ["chs-run-demo-check-{node}", "chs-run-demo-verify-{node}"],
        )
        self.assertEqual([s.workload for s in specs], ["check", "verify"])
        self.assertEqual(
            [s.context["variables"]["job_name"] for s in specs],
            ["chs-run-demo-check", "chs-run-demo-verify"],
        )

    def test_skips_with_a_warning_when_fewer_nodes_are_present(self):
        plugin = type("P", (DemoClusterPlugin,), {"min_nodes_to_run": 5})()

        with mock.patch("chs.testkit.base.logger") as mock_logger:
            specs = plugin.build(self.request())

        self.assertEqual(specs, [])
        mock_logger.warning.assert_called_once()

    def test_runs_when_the_minimum_node_count_is_met(self):
        plugin = type("P", (DemoClusterPlugin,), {"min_nodes_to_run": 2})()

        self.assertEqual(len(plugin.build(self.request())), 2)

    def test_hooks_see_each_workload_separately(self):
        class HookedPlugin(DemoNodePlugin):
            def prepare_variables(self, variables, workload, request):
                variables["chosen"] = workload

            def finalize_context(self, context, workload, request):
                context["variables"]["finalized"] = (
                    f"{context['variables']['chosen']}!"
                )

        specs = HookedPlugin().build(self.request())

        self.assertEqual(
            [s.context["variables"]["finalized"] for s in specs],
            ["check!", "verify!"],
        )

    def test_building_leaves_the_request_untouched(self):
        request = self.request()
        plugin = DemoNodePlugin()

        plugin.build(request)

        self.assertNotIn("job_name", request.variables)
        self.assertEqual(request.configs, machine_configs(variables={}))

    def test_class_level_extras_are_not_shared_between_experiments(self):
        plugin = type(
            "P",
            (SlicePlugin,),
            {
                "name": "demo",
                "application": "demo",
                "default_workloads": ("check",),
                "extra_context": {"modifiers": []},
            },
        )()

        specs = plugin.build(
            self.request(
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

        self.assertEqual(
            plugin.resolve_workloads(self.request(workloads=("only",))),
            ("check", "verify"),
        )

    def test_the_test_name_reaches_the_context(self):
        specs = DemoNodePlugin().build(self.request())

        for spec in specs:
            self.assertEqual(spec.context["variables"]["chs_test_name"], "demo")

    def test_a_cluster_test_measures_the_whole_allocation_by_default(self):
        """Every shipped cluster test overrides this with something finer."""
        plugin = type(
            "P",
            (MultiNodePlugin,),
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

        with self.assertRaisesRegex(ValueError, "extra_context"):
            plugin.build(self.request())


if __name__ == "__main__":
    unittest.main()
