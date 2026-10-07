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

"""Unit tests for chs.testkit.context."""

import os
import shutil
import tempfile
import textwrap
import unittest
from unittest import mock

from chs.testkit import context as ctx


def machine_configs(**app_config):
    """Builds the smallest configuration a context build will accept."""
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


class NodesStringTest(unittest.TestCase):
    def test_slurm_uses_a_comma_separated_list(self):
        self.assertEqual(
            ctx.nodes_str("slurm", ["node-1", "node-2"]), "node-1,node-2"
        )

    def test_kubernetes_uses_an_indented_yaml_sequence(self):
        self.assertEqual(
            ctx.nodes_str("kubernetes", ["node-1", "node-2"]),
            "            - node-1\n            - node-2",
        )

    def test_no_nodes_yields_an_empty_string(self):
        self.assertEqual(ctx.nodes_str("slurm", []), "")
        self.assertEqual(ctx.nodes_str("kubernetes", None), "")

    def test_unknown_scheduler_yields_an_empty_string(self):
        self.assertEqual(ctx.nodes_str("pbs", ["node-1"]), "")


class KubernetesAffinityTest(unittest.TestCase):
    def test_pins_to_nodes_and_keeps_pods_apart(self):
        values = {}
        ctx.apply_kubernetes_affinity(values, "            - node-1")

        self.assertIn("kubernetes.io/hostname", values["affinity_yaml"])
        self.assertIn("podAntiAffinity", values["affinity_yaml"])

    def test_without_nodes_affinity_is_empty(self):
        values = {}
        ctx.apply_kubernetes_affinity(values, "")

        self.assertEqual(values["affinity_yaml"], "")


class SbatchHeaderTest(unittest.TestCase):
    def test_long_and_short_options_are_formatted_differently(self):
        self.assertEqual(
            ctx.sbatch_headers({"-w": "node-1", "--time": "20"}),
            "#SBATCH -w node-1\n#SBATCH --time=20",
        )

    def test_headers_with_no_value_are_dropped(self):
        self.assertEqual(ctx.sbatch_headers({"--time": None}), "")

    def test_time_is_whole_minutes_and_deadline_covers_the_wait(self):
        headers = ctx.common_sbatch_headers(
            {"timeout": 1200, "deadline": 3600}, "node-1"
        )

        self.assertEqual(headers["--time"], "20")
        self.assertEqual(headers["--deadline"], "now+4800seconds")

    def test_time_rounds_up_to_the_next_minute(self):
        for timeout, minutes in ((150, "3"), (30, "1"), (0, "1")):
            with self.subTest(timeout=timeout):
                headers = ctx.common_sbatch_headers(
                    {"timeout": timeout, "deadline": 3600}, "node-1"
                )
                self.assertEqual(headers["--time"], minutes)

    def test_an_unbounded_job_is_refused(self):
        for values in (
            {"deadline": 3600},
            {"timeout": 1200},
            {"timeout": "soon", "deadline": 3600},
            {},
        ):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    ctx.common_sbatch_headers(values, "node-1")


class MergeMachineConfigTest(unittest.TestCase):
    def test_unknown_machine_type_is_refused(self):
        with self.assertRaisesRegex(ValueError, "Machine type 'other'"):
            ctx.merge_machine_config(
                {"variables": {}}, machine_configs(), "other", "demo"
            )

    def test_test_missing_from_a_machine_is_refused(self):
        with self.assertRaisesRegex(ValueError, "Test 'absent' not found"):
            ctx.merge_machine_config(
                {"variables": {}}, machine_configs(), "machine-x", "absent"
            )

    def test_a_context_cannot_be_built_without_a_configuration(self):
        with self.assertRaisesRegex(ValueError, "Machine configuration"):
            ctx.build_base_context(
                {"machine_type": "machine-x"}, None, "demo", "node-1"
            )

    def test_caller_values_win_over_configuration(self):
        context = {"variables": {"shared": "from-caller"}}
        ctx.merge_machine_config(
            context,
            machine_configs(variables={"shared": "from-config"}),
            "machine-x",
            "demo",
        )

        self.assertEqual(context["variables"]["shared"], "from-caller")

    def test_an_empty_caller_value_is_filled_in(self):
        context = {"variables": {"shared": ""}}
        ctx.merge_machine_config(
            context,
            machine_configs(variables={"shared": "from-config"}),
            "machine-x",
            "demo",
        )

        self.assertEqual(context["variables"]["shared"], "from-config")

    def test_a_modifier_named_twice_is_applied_once(self):
        context = {"variables": {}, "modifiers": [{"name": "detector"}]}
        ctx.merge_machine_config(
            context,
            machine_configs(
                modifiers=[{"name": "detector"}, {"name": "persistence"}]
            ),
            "machine-x",
            "demo",
        )

        self.assertEqual(
            context["modifiers"],
            [{"name": "detector"}, {"name": "persistence"}],
        )

    def test_the_domain_section_overrides_the_application_defaults(self):
        context = {"variables": {}}
        ctx.merge_machine_config(
            context,
            machine_configs(
                variables={"expected": 100},
                intradomain={"variables": {"expected": 400}},
            ),
            "machine-x",
            "demo",
            domain_type="intradomain",
        )

        self.assertEqual(context["variables"]["expected"], 400)

    def test_a_configuration_that_is_not_a_mapping_is_refused(self):
        with self.assertRaisesRegex(ValueError, "to be dictionaries"):
            ctx.merge_machine_config({"variables": {}}, [], "machine-x", "demo")

    def test_a_context_that_is_not_a_mapping_is_refused(self):
        with self.assertRaisesRegex(ValueError, "to be dictionaries"):
            ctx.merge_machine_config([], machine_configs(), "machine-x", "demo")


class MergeSectionTest(unittest.TestCase):
    """The merge every configuration layer goes through."""

    def test_a_section_the_context_lacks_is_created(self):
        context = {}

        ctx._merge_section({"env_vars": {"set": {}}}, "env_vars", context, "e")

        self.assertEqual({"set": {}}, context["e"])

    def test_a_list_section_the_context_lacks_is_created(self):
        context = {}

        ctx._merge_section(
            {"modifiers": [{"name": "m"}]}, "modifiers", context, "modifiers"
        )

        self.assertEqual([{"name": "m"}], context["modifiers"])

    def test_a_section_the_source_lacks_is_left_alone(self):
        context = {"env_vars": {"set": {"KEEP": "1"}}}

        ctx._merge_section({}, "env_vars", context, "env_vars")

        self.assertEqual({"set": {"KEEP": "1"}}, context["env_vars"])

    def test_a_scalar_section_is_refused(self):
        """Only mappings and lists have a defined way to merge."""
        with self.assertRaisesRegex(ValueError, "dictionary or list"):
            ctx._merge_section({"count": 3}, "count", {}, "count")

    def test_a_section_that_changes_shape_is_refused(self):
        with self.assertRaisesRegex(ValueError, "dictionaries or lists"):
            ctx._merge_section(
                {"s": [1]}, "s", {"s": {"already": "a mapping"}}, "s"
            )

    def test_a_source_that_is_not_a_mapping_is_refused(self):
        with self.assertRaisesRegex(ValueError, "to be dictionaries"):
            ctx._merge_section([], "s", {}, "s")

    def test_a_destination_that_is_not_a_mapping_is_refused(self):
        with self.assertRaisesRegex(ValueError, "to be dictionaries"):
            ctx._merge_section({}, "s", [], "s")

    def test_override_replaces_a_value_the_context_already_has(self):
        context = {"variables": {"expected": 100}}

        ctx._merge_section(
            {"variables": {"expected": 400}},
            "variables",
            context,
            "variables",
            override=True,
        )

        self.assertEqual(400, context["variables"]["expected"])

    def test_a_caller_that_asked_for_false_or_zero_keeps_it(self):
        for asked_for in (False, 0, 0.0):
            with self.subTest(asked_for=asked_for):
                context = {"variables": {"chosen": asked_for}}

                ctx._merge_section(
                    {"variables": {"chosen": "from-config"}},
                    "variables",
                    context,
                    "variables",
                )

                self.assertIs(asked_for, context["variables"]["chosen"])

    def test_a_value_the_caller_left_empty_comes_from_the_config(self):
        for left_empty in (None, "", [], {}, ()):
            with self.subTest(left_empty=left_empty):
                context = {"variables": {"chosen": left_empty}}

                ctx._merge_section(
                    {"variables": {"chosen": "from-config"}},
                    "variables",
                    context,
                    "variables",
                )

                self.assertEqual("from-config", context["variables"]["chosen"])

    def test_a_value_the_caller_never_mentioned_comes_from_the_config(self):
        context = {"variables": {}}

        ctx._merge_section(
            {"variables": {"chosen": "from-config"}},
            "variables",
            context,
            "variables",
        )

        self.assertEqual("from-config", context["variables"]["chosen"])


class BaseContextTest(unittest.TestCase):
    """What the shared builders put in a context beyond the merge."""

    def test_the_domain_is_recorded_so_results_can_be_judged_by_it(self):
        context = ctx.build_base_context(
            variables(),
            machine_configs(),
            "demo",
            "node-1",
            domain_type="intradomain",
        )

        self.assertEqual("intradomain", context["variables"]["domain_type"])

    def test_no_domain_leaves_no_marker(self):
        context = ctx.build_base_context(
            variables(), machine_configs(), "demo", "node-1"
        )

        self.assertNotIn("domain_type", context["variables"])

    def test_a_test_may_add_its_own_sbatch_headers(self):
        context = ctx.build_base_context(
            variables(scheduler="slurm"),
            machine_configs(),
            "demo",
            "node-1",
            extra_sbatch_headers={"--gpu-bind": "single:1"},
        )

        headers = context["variables"]["extra_sbatch_headers"]
        self.assertIn("#SBATCH --gpu-bind=single:1", headers)
        self.assertIn("#SBATCH --time", headers)

    def test_an_unrecognised_scheduler_gets_no_placement(self):
        """Neither Slurm headers nor Kubernetes affinity would mean anything."""
        context = ctx.build_base_context(
            variables(scheduler="pbs"), machine_configs(), "demo", "node-1"
        )

        self.assertNotIn("extra_sbatch_headers", context["variables"])
        self.assertNotIn("affinity_yaml", context["variables"])

    def test_a_node_test_inlines_its_benchmark_input(self):
        with mock.patch.object(
            ctx, "load_benchmark_config", return_value="cfg"
        ):
            context = ctx.build_node_context(
                variables(),
                machine_configs(),
                "demo",
                benchmark_config="input.yaml",
            )

        self.assertEqual("cfg", context["variables"]["benchmarks_yaml_escaped"])

    def test_a_cluster_test_inlines_its_benchmark_input(self):
        with mock.patch.object(
            ctx, "load_benchmark_config", return_value="cfg"
        ):
            context = ctx.build_cluster_context(
                variables(),
                machine_configs(),
                "demo",
                benchmark_config="input.yaml",
            )

        self.assertEqual("cfg", context["variables"]["benchmarks_yaml_escaped"])

    def test_the_caller_variables_are_not_modified(self):
        """Both builders copy, so one experiment cannot affect the next."""
        original = variables()

        ctx.build_node_context(original, machine_configs(), "demo")
        ctx.build_cluster_context(original, machine_configs(), "demo")

        self.assertEqual(variables(), original)


class SliceContextTest(unittest.TestCase):
    def build(self, values, name="exp"):
        return ctx.build_slice_contexts(
            values, name, machine_configs(variables={}), "demo"
        )

    def test_an_ungrouped_allocation_is_treated_as_one_slice(self):
        contexts = self.build(variables())

        self.assertEqual(len(contexts), 1)
        self.assertEqual(contexts[0][0], "exp")
        self.assertEqual(
            contexts[0][1]["variables"]["nodes"], ["node-1", "node-2"]
        )

    def test_one_slice_keeps_the_plain_experiment_name(self):
        contexts = self.build(variables(tpu_slices={"only": ["node-1"]}))

        self.assertEqual(contexts[0][0], "exp")

    def test_several_slices_are_named_apart_and_sized_separately(self):
        contexts = self.build(
            variables(
                tpu_slices={"Slice_A": ["node-1"], "slice-b": ["n2", "n3"]}
            )
        )

        self.assertEqual(
            [name for name, _ in contexts], ["exp-slice-a", "exp-slice-b"]
        )
        self.assertEqual(contexts[0][1]["variables"]["n_nodes"], 1)
        self.assertEqual(contexts[1][1]["variables"]["n_nodes"], 2)

    def test_each_slice_takes_its_own_topology(self):
        contexts = self.build(
            variables(
                tpu_slices={"a": ["node-1"], "b": ["node-2"]},
                tpu_slice_topologies={"a": "2x2x1", "b": "2x2x2"},
            )
        )

        self.assertEqual(contexts[0][1]["variables"]["tpu_topology"], "2x2x1")
        self.assertEqual(contexts[1][1]["variables"]["tpu_topology"], "2x2x2")

    def test_a_dynamic_slice_asks_for_exclusive_hardware(self):
        contexts = self.build(
            variables(
                tpu_slices={"a": ["node-1"], "b": ["node-2"]},
                tpu_slice_topologies={"b": "2x2x2"},
                tpu_dynamic_pools=["b"],
            )
        )

        static, dynamic = (c[1]["variables"] for c in contexts)
        self.assertNotIn("jobset_annotations_yaml", static)
        self.assertIn("exclusive-topology", dynamic["jobset_annotations_yaml"])
        self.assertIn("2x2x2", dynamic["pod_annotations_yaml"])

    def test_dynamic_annotations_are_added_to_existing_ones(self):
        contexts = self.build(
            variables(
                tpu_slices={"b": ["node-2"]},
                tpu_dynamic_pools=["b"],
                jobset_annotations_yaml="keep: me",
            )
        )

        annotations = contexts[0][1]["variables"]["jobset_annotations_yaml"]
        self.assertTrue(annotations.startswith("keep: me\\n"))
        self.assertIn("exclusive-topology", annotations)

    def test_slurm_collapses_to_a_single_node_matrix(self):
        contexts = self.build(variables(scheduler="slurm"))

        self.assertEqual(len(contexts), 1)
        self.assertEqual(contexts[0][0], "exp-{node}")
        self.assertEqual(
            contexts[0][1]["variables"]["node"], ["node-1", "node-2"]
        )


class BenchmarkConfigTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="chs-benchmarks-")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def write(self, name, content):
        with open(os.path.join(self.dir, name), "w") as f:
            f.write(content)

    def test_the_licence_header_is_dropped(self):
        self.write(
            "input.yaml",
            textwrap.dedent(
                """\
                # Copyright
                # more header

                real: content
                """
            ),
        )

        self.assertEqual(
            ctx.load_benchmark_config("input.yaml", self.dir), "real: content"
        )

    def test_content_is_escaped_for_a_single_quoted_shell_string(self):
        self.write("input.yaml", "path: C:\\tmp\nname: it's\n")
        values = {}

        ctx.apply_benchmark_config(values, "input.yaml", self.dir)

        self.assertEqual(
            values["benchmarks_yaml_escaped"],
            "path: C:\\\\tmp\\nname: it'\\''s",
        )

    def test_an_explicit_choice_beats_the_default(self):
        self.write("chosen.yaml", "chosen: yes")
        self.write("fallback.yaml", "fallback: yes")
        values = {"benchmarks_yaml_file": "chosen.yaml"}

        ctx.apply_benchmark_config(values, "fallback.yaml", self.dir)

        self.assertEqual(values["benchmarks_yaml_escaped"], "chosen: yes")

    def test_a_missing_file_is_not_silently_ignored(self):
        with self.assertRaises(OSError):
            ctx.load_benchmark_config("absent.yaml", self.dir)

    def test_no_file_at_all_yields_no_content(self):
        self.assertEqual(ctx.load_benchmark_config(None, self.dir), "")

    def test_a_file_that_is_all_header_yields_no_content(self):
        self.write("input.yaml", "# Copyright\n\n# more header\n")

        self.assertEqual(ctx.load_benchmark_config("input.yaml", self.dir), "")


if __name__ == "__main__":
    unittest.main()
