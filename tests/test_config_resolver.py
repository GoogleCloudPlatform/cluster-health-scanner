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

"""Unit tests for chs.testkit.config, the layered configuration resolver."""

import unittest

from chs.testkit import config as chs_config
from tests.testkit_fixtures import FakeTestsTestCase


def platform(
    schedulers=("kubernetes",), variables=None, kubernetes=None, slurm=None
):
    """Builds the contents of a platform file."""
    built = {"schedulers": list(schedulers), "variables": dict(variables or {})}
    if kubernetes is not None:
        built["kubernetes"] = kubernetes
    if slurm is not None:
        built["slurm"] = slurm
    return built


def config(platforms=("machine-x",), defaults=None, machines=None):
    """Builds the contents of a test's config.yaml."""
    built = {"platforms": list(platforms)}
    if defaults is not None:
        built["defaults"] = defaults
    if machines is not None:
        built["machines"] = machines
    return built


def resolve(test_config, machine_platform=None, domain_type=None):
    """Resolves ``test_config`` for machine-x on kubernetes."""
    if machine_platform is None:
        machine_platform = platform()
    return chs_config.resolve(
        machine_platform, test_config, "machine-x", "kubernetes", domain_type
    )


class DeepMergeTest(unittest.TestCase):
    def test_nested_dicts_merge_key_by_key(self):
        merged = chs_config.deep_merge(
            {"variables": {"keep": 1, "change": "old"}},
            {"variables": {"change": "new", "add": 3}},
        )

        self.assertEqual(
            merged, {"variables": {"keep": 1, "change": "new", "add": 3}}
        )

    def test_lists_and_scalars_replace_the_old_value(self):
        merged = chs_config.deep_merge(
            {"modifiers": [{"name": "a"}], "size": {"nested": 1}},
            {"modifiers": [{"name": "b"}], "size": 2},
        )

        self.assertEqual(merged, {"modifiers": [{"name": "b"}], "size": 2})

    def test_merged_values_are_copies(self):
        layer = {"variables": {"nested": {"deep": 1}}, "list": [{"x": 1}]}

        merged = chs_config.deep_merge({}, layer)
        merged["variables"]["nested"]["deep"] = 2
        merged["list"][0]["x"] = 2

        self.assertEqual(
            layer, {"variables": {"nested": {"deep": 1}}, "list": [{"x": 1}]}
        )


class LoadFilesTest(FakeTestsTestCase):
    def test_load_platform_parses_the_file(self):
        self.write_platform("machine-x", "variables:\n  gpus_per_node: 8\n")

        self.assertEqual(
            chs_config.load_platform("machine-x"),
            {"variables": {"gpus_per_node": 8}},
        )

    def test_load_test_config_parses_the_file(self):
        self.write_test("demo", "platforms: [machine-x]\n")

        self.assertEqual(
            chs_config.load_test_config("demo"), {"platforms": ["machine-x"]}
        )


class SupportedPlatformsTest(FakeTestsTestCase):
    def setUp(self):
        super().setUp()
        self.write_platform("machine-x", "schedulers: [kubernetes, slurm]\n")
        self.write_platform("machine-y", "schedulers: [kubernetes]\n")

    def test_a_machine_name_takes_every_scheduler_in_its_platform_file(self):
        self.assertEqual(
            chs_config.supported_platforms(config(["machine-x", "machine-y"])),
            {
                ("kubernetes", "machine-x"),
                ("slurm", "machine-x"),
                ("kubernetes", "machine-y"),
            },
        )

    def test_a_machine_with_schedulers_runs_only_on_those(self):
        self.assertEqual(
            chs_config.supported_platforms(config([{"machine-x": ["slurm"]}])),
            {("slurm", "machine-x")},
        )


class ResolveTest(unittest.TestCase):
    def test_empty_sections_are_dropped_but_variables_is_kept(self):
        self.assertEqual(
            resolve(config(defaults={"env_vars": {}, "modifiers": []})),
            {"variables": {}},
        )

    def test_test_config_overrides_platform_variables(self):
        resolved = resolve(
            config(defaults={"variables": {"size": "test"}}),
            platform(
                variables={"image": "base", "gpus": 8, "size": "platform"},
                kubernetes={"variables": {"image": "k8s"}},
            ),
        )

        self.assertEqual(
            resolved["variables"], {"image": "k8s", "gpus": 8, "size": "test"}
        )

    def test_the_narrowest_test_config_layer_wins(self):
        resolved = resolve(
            config(
                defaults={
                    "variables": {"size": "defaults", "keep": 1},
                    "kubernetes": {"variables": {"size": "defaults-k8s"}},
                },
                machines={
                    "machine-x": {
                        "variables": {"size": "machine"},
                        "kubernetes": {"variables": {"size": "machine-k8s"}},
                    }
                },
            )
        )

        self.assertEqual(
            resolved["variables"], {"size": "machine-k8s", "keep": 1}
        )

    def test_blocks_for_other_schedulers_are_ignored(self):
        resolved = resolve(
            config(
                defaults={
                    "variables": {"size": "defaults"},
                    "slurm": {"variables": {"size": "defaults-slurm"}},
                }
            ),
            platform(
                schedulers=["kubernetes", "slurm"],
                slurm={"variables": {"image": "slurm"}},
            ),
        )

        self.assertEqual(resolved["variables"], {"size": "defaults"})

    def test_success_criteria_lists_are_replaced_by_narrower_layers(self):
        resolved = resolve(
            config(
                defaults={"success_criteria": [{"name": "a"}, {"name": "b"}]},
                machines={"machine-x": {"success_criteria": [{"name": "c"}]}},
            )
        )

        self.assertEqual(resolved["success_criteria"], [{"name": "c"}])

    def test_env_vars_are_merged_key_by_key(self):
        resolved = resolve(
            config(
                defaults={"env_vars": {"set": {"A": "1", "B": "1"}}},
                machines={"machine-x": {"env_vars": {"set": {"B": "2"}}}},
            )
        )

        self.assertEqual(resolved["env_vars"], {"set": {"A": "1", "B": "2"}})

    def test_the_result_shares_no_structure_with_the_inputs(self):
        machine_platform = platform(variables={"nested": {"deep": 1}})
        test_config = config(defaults={"variables": {"other": {"deep": 1}}})

        resolved = resolve(test_config, machine_platform)
        resolved["variables"]["nested"]["deep"] = 2
        resolved["variables"]["other"]["deep"] = 2

        self.assertEqual(machine_platform["variables"]["nested"]["deep"], 1)
        self.assertEqual(
            test_config["defaults"]["variables"]["other"]["deep"], 1
        )


class ResolveModifiersTest(unittest.TestCase):
    def test_modifiers_add_up_across_layers(self):
        resolved = resolve(
            config(
                defaults={
                    "modifiers": [{"name": "common"}],
                    "kubernetes": {"modifiers": [{"name": "k8s"}]},
                    "slurm": {"modifiers": [{"name": "slurm"}]},
                },
                machines={"machine-x": {"modifiers": [{"name": "machine"}]}},
            )
        )

        self.assertEqual(
            [m["name"] for m in resolved["modifiers"]],
            ["common", "k8s", "machine"],
        )

    def test_a_repeated_name_replaces_the_earlier_modifier_in_place(self):
        resolved = resolve(
            config(
                defaults={"modifiers": [{"name": "a"}, {"name": "b"}]},
                machines={
                    "machine-x": {"modifiers": [{"name": "a", "mode": "fast"}]}
                },
            )
        )

        self.assertEqual(
            resolved["modifiers"],
            [{"name": "a", "mode": "fast"}, {"name": "b"}],
        )


class ResolveDomainTest(unittest.TestCase):
    CONFIG = config(
        defaults={
            "variables": {"keep": 1, "size": "base"},
            "intradomain": {"variables": {"size": "intra"}},
            "interdomain": {"variables": {"size": "inter"}},
        }
    )

    def test_the_requested_domain_overrides_its_variables(self):
        self.assertEqual(
            resolve(self.CONFIG, domain_type="intradomain"),
            {"variables": {"keep": 1, "size": "intra"}},
        )

    def test_without_a_domain_the_domain_sections_are_ignored(self):
        self.assertEqual(
            resolve(self.CONFIG), {"variables": {"keep": 1, "size": "base"}}
        )

    def test_the_narrowest_domain_block_wins(self):
        resolved = resolve(
            config(
                defaults={"intradomain": {"variables": {"size": "broad"}}},
                machines={
                    "machine-x": {
                        "intradomain": {"variables": {"size": "narrow"}}
                    }
                },
            ),
            domain_type="intradomain",
        )

        self.assertEqual(resolved["variables"], {"size": "narrow"})


class ResolveTestFromFilesTest(FakeTestsTestCase):
    def setUp(self):
        super().setUp()
        self.write_platform(
            "machine-x",
            "schedulers: [kubernetes]\nvariables:\n  gpus_per_node: 8\n",
        )
        self.write_test(
            "demo",
            "platforms: [machine-x]\n"
            "defaults:\n"
            "  variables:\n    size: 64M\n"
            "  intradomain:\n    variables:\n      size: 8M\n",
        )

    def test_resolves_the_platform_and_config_files(self):
        self.assertEqual(
            chs_config.resolve_test("demo", "machine-x", "kubernetes"),
            {"variables": {"gpus_per_node": 8, "size": "64M"}},
        )

    def test_passes_the_domain_type_through(self):
        resolved = chs_config.resolve_test(
            "demo", "machine-x", "kubernetes", domain_type="intradomain"
        )

        self.assertEqual(resolved["variables"]["size"], "8M")


if __name__ == "__main__":
    unittest.main()
