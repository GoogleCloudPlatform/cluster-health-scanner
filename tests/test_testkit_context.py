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

import yaml

from chs.testkit import config as chs_config
from chs.testkit import context as ctx


def variables(**overrides):
    values = {
        "chs_test_name": "demo",
        "scheduler": "kubernetes",
        "machine_type": "machine-x",
        "nodes": ["node-1", "node-2"],
        "timeout": 1200,
        "deadline": 3600,
    }
    values.update(overrides)
    return values


class StubbedConfigTestCase(unittest.TestCase):
    """Serves a fixed resolved config instead of reading test files."""

    def setUp(self):
        patcher = mock.patch.object(
            ctx.chs_config,
            "resolve_test",
            return_value={"variables": {"machine_wide": "yes"}},
        )
        self.resolve_test = patcher.start()
        self.addCleanup(patcher.stop)


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
        affinity = yaml.safe_load(values["affinity_yaml"])
        self.assertEqual(
            affinity["podAntiAffinity"],
            {
                "requiredDuringSchedulingIgnoredDuringExecution": [
                    {
                        "labelSelector": {
                            "matchExpressions": [
                                {
                                    "key": "chs.gke.io/worker",
                                    "operator": "Exists",
                                }
                            ]
                        },
                        "topologyKey": "kubernetes.io/hostname",
                    }
                ]
            },
        )

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


class ResolvedConfigTest(unittest.TestCase):
    def test_caller_values_win_over_configuration(self):
        context = {"variables": {"shared": "from-caller"}}
        ctx.merge_resolved_config(
            context, {"variables": {"shared": "from-config"}}
        )

        self.assertEqual(context["variables"]["shared"], "from-caller")

    def test_an_empty_caller_value_is_filled_in(self):
        context = {"variables": {"shared": ""}}
        ctx.merge_resolved_config(
            context, {"variables": {"shared": "from-config"}}
        )

        self.assertEqual(context["variables"]["shared"], "from-config")

    def test_a_modifier_named_twice_is_applied_once(self):
        context = {"variables": {}, "modifiers": [{"name": "detector"}]}
        ctx.merge_resolved_config(
            context,
            {
                "variables": {},
                "modifiers": [{"name": "detector"}, {"name": "persistence"}],
            },
        )

        self.assertEqual(
            context["modifiers"],
            [{"name": "detector"}, {"name": "persistence"}],
        )


class DomainPrecedenceTest(unittest.TestCase):
    """Where a run variable and a domain block disagree, the run wins."""

    @staticmethod
    def platform():
        return {"schedulers": ["kubernetes"], "variables": {}}

    @staticmethod
    def config(**intradomain_vars):
        return {
            "platforms": [{"machine-x": ["kubernetes"]}],
            "machines": {
                "machine-x": {
                    "kubernetes": {},
                    "intradomain": {"variables": dict(intradomain_vars)},
                }
            },
        }

    def resolved(self, **intradomain_vars):
        return chs_config.resolve(
            self.platform(),
            self.config(**intradomain_vars),
            "machine-x",
            "kubernetes",
            domain_type="intradomain",
        )

    def test_a_run_variable_beats_a_domain_variable(self):
        context = {"variables": {"expected": "from-run"}}
        ctx.merge_resolved_config(context, self.resolved(expected=400))

        self.assertEqual(context["variables"]["expected"], "from-run")

    def test_a_domain_variable_still_fills_a_gap(self):
        context = {"variables": {}}
        ctx.merge_resolved_config(context, self.resolved(expected=400))

        self.assertEqual(context["variables"]["expected"], 400)

    def test_a_domain_variable_leaves_unrelated_run_variables_alone(self):
        context = {"variables": {"other": "untouched"}}
        ctx.merge_resolved_config(context, self.resolved(expected=400))

        self.assertEqual(
            context["variables"], {"other": "untouched", "expected": 400}
        )


class BuildBaseContextTest(unittest.TestCase):
    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        platforms_root = os.path.join(temp_dir.name, "platforms")
        tests_root = os.path.join(temp_dir.name, "tests")
        os.makedirs(platforms_root)
        os.makedirs(os.path.join(tests_root, "demo"))
        with open(os.path.join(platforms_root, "machine-x.yaml"), "w") as f:
            f.write("schedulers: [kubernetes]\nvariables:\n  gpus: 8\n")
        with open(os.path.join(tests_root, "demo", "config.yaml"), "w") as f:
            f.write(
                "platforms: [machine-x]\n"
                "defaults:\n  variables:\n    image: demo-image\n"
            )
        for name, root in (
            ("PLATFORMS_ROOT", platforms_root),
            ("TESTS_ROOT", tests_root),
        ):
            patcher = mock.patch.object(chs_config, name, root)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_the_test_config_files_are_resolved(self):
        context = ctx.build_base_context(
            {
                "chs_test_name": "demo",
                "machine_type": "machine-x",
                "scheduler": "kubernetes",
                "image": "from-run",
            },
            "node-1",
        )

        self.assertEqual(context["variables"]["gpus"], 8)
        self.assertEqual(context["variables"]["image"], "from-run")


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


class BaseContextTest(StubbedConfigTestCase):
    """What the shared builders put in a context beyond the merge."""

    def test_the_domain_is_recorded_so_results_can_be_judged_by_it(self):
        context = ctx.build_base_context(
            variables(),
            "node-1",
            domain_type="intradomain",
        )

        self.assertEqual("intradomain", context["variables"]["domain_type"])

    def test_no_domain_leaves_no_marker(self):
        context = ctx.build_base_context(variables(), "node-1")

        self.assertNotIn("domain_type", context["variables"])

    def test_a_test_may_add_its_own_sbatch_headers(self):
        context = ctx.build_base_context(
            variables(scheduler="slurm"),
            "node-1",
            extra_sbatch_headers={"--gpu-bind": "single:1"},
        )

        headers = context["variables"]["extra_sbatch_headers"]
        self.assertIn("#SBATCH --gpu-bind=single:1", headers)
        self.assertIn("#SBATCH --time", headers)

    def test_an_unrecognised_scheduler_gets_no_placement(self):
        """Neither Slurm headers nor Kubernetes affinity would mean anything."""
        context = ctx.build_base_context(variables(scheduler="pbs"), "node-1")

        self.assertNotIn("extra_sbatch_headers", context["variables"])
        self.assertNotIn("affinity_yaml", context["variables"])

    def test_a_node_test_inlines_its_benchmark_input(self):
        with mock.patch.object(
            ctx, "load_benchmark_config", return_value="cfg"
        ):
            context = ctx.build_node_context(
                variables(),
                benchmark_config="input.yaml",
                benchmarks_dir="/tmp/benchmarks",
            )

        self.assertEqual("cfg", context["variables"]["benchmarks_yaml_escaped"])

    def test_a_cluster_test_inlines_its_benchmark_input(self):
        with mock.patch.object(
            ctx, "load_benchmark_config", return_value="cfg"
        ):
            context = ctx.build_cluster_context(
                variables(),
                benchmark_config="input.yaml",
                benchmarks_dir="/tmp/benchmarks",
            )

        self.assertEqual("cfg", context["variables"]["benchmarks_yaml_escaped"])

    def test_the_caller_variables_are_not_modified(self):
        """Both builders copy, so one experiment cannot affect the next."""
        original = variables()

        ctx.build_node_context(original)
        ctx.build_cluster_context(original)

        self.assertEqual(variables(), original)


class SliceContextTest(StubbedConfigTestCase):
    def build(self, values, name="exp"):
        return ctx.build_slice_contexts(values, name)

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

    def test_clears_node_selector_without_slice_topology(self):
        self.resolve_test.return_value = {
            "variables": {
                "node_selector_yaml": (
                    'cloud.google.com/gke-tpu-topology: "{tpu_topology}"'
                )
            }
        }
        contexts = self.build(
            variables(
                tpu_slices={"a": ["node-1"], "b": ["node-2"]},
                tpu_slice_topologies={"a": "2x2x1"},
            )
        )

        self.assertEqual(
            contexts[0][1]["variables"]["node_selector_yaml"],
            'cloud.google.com/gke-tpu-topology: "{tpu_topology}"',
        )
        self.assertEqual(contexts[1][1]["variables"]["node_selector_yaml"], "")

    def test_a_dynamic_slice_asks_for_exclusive_hardware(self):
        contexts = self.build(
            variables(
                tpu_slices={"a": ["node-1"], "b": ["node-2"]},
                tpu_slice_topologies={"b": "2x2x2"},
                tpu_dynamic_pools=["b"],
            )
        )

        static, dynamic = (c[1]["variables"] for c in contexts)
        self.assertNotIn("jobset_annotations_yaml", dynamic)
        self.assertNotIn("podAffinity", static["affinity_yaml"])
        self.assertIn("2x2x2", dynamic["pod_annotations_yaml"])
        for ctx_vars in (static, dynamic):
            self.assertNotIn(ctx.EXCLUSIVE_SLICE_KEY, ctx_vars)

        affinity = yaml.safe_load(
            dynamic["affinity_yaml"].replace("{job_name}", "my-job")
        )
        slice_key = "cloud.google.com/gke-tpu-slice"
        self.assertEqual(
            affinity["podAffinity"],
            {
                "requiredDuringSchedulingIgnoredDuringExecution": [
                    {
                        "labelSelector": {
                            "matchExpressions": [
                                {
                                    "key": "batch.kubernetes.io/job-name",
                                    "operator": "In",
                                    "values": ["my-job"],
                                }
                            ]
                        },
                        "topologyKey": slice_key,
                    }
                ]
            },
        )
        anti_terms = affinity["podAntiAffinity"][
            "requiredDuringSchedulingIgnoredDuringExecution"
        ]
        self.assertEqual(
            [t["topologyKey"] for t in anti_terms],
            ["kubernetes.io/hostname", slice_key],
        )
        self.assertEqual(
            anti_terms[1]["labelSelector"]["matchExpressions"],
            [
                {"key": "chs.gke.io/worker", "operator": "Exists"},
                {
                    "key": "batch.kubernetes.io/job-name",
                    "operator": "NotIn",
                    "values": ["my-job"],
                },
            ],
        )
        self.assertIn(
            "node-2",
            affinity["nodeAffinity"][
                "requiredDuringSchedulingIgnoredDuringExecution"
            ]["nodeSelectorTerms"][0]["matchExpressions"][0]["values"],
        )

    def test_dynamic_slice_keeps_existing_job_annotations(self):
        contexts = self.build(
            variables(
                tpu_slices={"b": ["node-2"]},
                tpu_dynamic_pools=["b"],
                jobset_annotations_yaml="keep: me",
            )
        )

        self.assertEqual(
            contexts[0][1]["variables"]["jobset_annotations_yaml"], "keep: me"
        )

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
