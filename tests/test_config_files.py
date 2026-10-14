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

"""Consistency checks on the shipped platform, config and threshold files."""

import pathlib
import unittest

from chs.testkit import build_catalog, build_thresholds
from chs.testkit import config as chs_config
from chs.testkit import loader
from chs.testkit import thresholds as chs_thresholds

PLATFORMS_DIR = pathlib.Path(chs_config.PLATFORMS_ROOT)

SCHEDULER_ONLY_MODIFIERS = {
    "gpu-persistence-mode": "kubernetes",
    "pyxis-enroot": "slurm",
}

SCHEDULERS = set(chs_config.SCHEDULERS)
DOMAIN_TYPES = set(chs_config.DOMAIN_TYPES)
SECTIONS = {"variables", "env_vars", "modifiers", "success_criteria"}
TOP_LEVEL_KEYS = {"enabled", "platforms", "defaults", "machines"}
THRESHOLD_TYPES = {"min", "relative", "scalar"}


def _supported_combinations():
    """Returns (test, machine_type, scheduler) for each supported setup."""
    return [
        (test_name, machine, scheduler)
        for test_name in loader.available_tests()
        for scheduler, machine in sorted(
            chs_config.supported_platforms(
                chs_config.load_test_config(test_name)
            )
        )
    ]


class PlatformFilesTest(unittest.TestCase):
    def test_every_platform_file_is_used_by_a_test(self):
        declared = {machine for _, machine, _ in _supported_combinations()}
        on_disk = {path.stem for path in PLATFORMS_DIR.glob("*.yaml")}

        self.assertEqual(on_disk, declared)


class TestConfigFilesTest(unittest.TestCase):
    def test_machines_blocks_only_name_declared_machines(self):
        for test_name in loader.available_tests():
            config = chs_config.load_test_config(test_name)
            declared = {
                machine for _, machine in chs_config.supported_platforms(config)
            }
            with self.subTest(test=test_name):
                self.assertLessEqual(set(config.get("machines", {})), declared)

    def test_scheduler_restrictions_leave_out_a_supported_scheduler(self):
        """A restriction listing every scheduler should be a bare name."""
        for test_name in loader.available_tests():
            config = chs_config.load_test_config(test_name)
            for entry in config["platforms"]:
                if isinstance(entry, str):
                    continue
                ((machine, schedulers),) = entry.items()
                platform = chs_config.load_platform(machine)
                with self.subTest(test=test_name, machine=machine):
                    self.assertLess(
                        set(schedulers), set(platform["schedulers"])
                    )

    def test_scheduler_overrides_differ_from_the_defaults(self):
        for test_name in loader.available_tests():
            defaults = chs_config.load_test_config(test_name).get(
                "defaults", {}
            )
            for scheduler in chs_config.SCHEDULERS:
                for section, values in defaults.get(scheduler, {}).items():
                    if not isinstance(values, dict):
                        continue
                    for key, value in values.items():
                        with self.subTest(
                            test=test_name, scheduler=scheduler, key=key
                        ):
                            self.assertNotEqual(
                                defaults.get(section, {}).get(key, object()),
                                value,
                            )


class ResolveShippedFilesTest(unittest.TestCase):
    def test_every_supported_combination_resolves(self):
        combinations = _supported_combinations()
        self.assertTrue(combinations)
        for test_name, machine, scheduler in combinations:
            for domain_type in (None, *chs_config.DOMAIN_TYPES):
                with self.subTest(
                    test=test_name,
                    machine=machine,
                    scheduler=scheduler,
                    domain=domain_type,
                ):
                    chs_config.resolve_test(
                        test_name, machine, scheduler, domain_type
                    )

    def test_scheduler_only_modifiers_stay_on_their_scheduler(self):
        for test_name, machine, scheduler in _supported_combinations():
            resolved = chs_config.resolve_test(test_name, machine, scheduler)
            names = {m["name"] for m in resolved.get("modifiers", [])}
            with self.subTest(
                test=test_name, machine=machine, scheduler=scheduler
            ):
                for name, only_on in SCHEDULER_ONLY_MODIFIERS.items():
                    if scheduler != only_on:
                        self.assertNotIn(name, names)


class KnownKeysTest(unittest.TestCase):
    """Catches typos such as 'varaibles:' that resolve would ignore."""

    def test_test_configs_use_known_keys(self):
        for test_name in loader.available_tests():
            config = chs_config.load_test_config(test_name)
            with self.subTest(test=test_name):
                self.assertLessEqual(set(config), TOP_LEVEL_KEYS)
                for block in (
                    config.get("defaults", {}),
                    *config.get("machines", {}).values(),
                ):
                    self.assertLessEqual(
                        set(block), SECTIONS | SCHEDULERS | DOMAIN_TYPES
                    )
                    scopes = [block]
                    for scheduler in SCHEDULERS & set(block):
                        self.assertLessEqual(
                            set(block[scheduler]), SECTIONS | DOMAIN_TYPES
                        )
                        scopes.append(block[scheduler])
                    for scope in scopes:
                        for domain_type in DOMAIN_TYPES & set(scope):
                            self.assertLessEqual(
                                set(scope[domain_type]), {"variables"}
                            )

    def test_platform_files_use_known_keys(self):
        for path in PLATFORMS_DIR.glob("*.yaml"):
            platform = chs_config.load_platform(path.stem)
            with self.subTest(machine_type=path.stem):
                self.assertLessEqual(
                    set(platform), {"schedulers", "variables"} | SCHEDULERS
                )
                self.assertLessEqual(set(platform["schedulers"]), SCHEDULERS)
                for scheduler in SCHEDULERS & set(platform):
                    self.assertLessEqual(
                        set(platform[scheduler]), {"variables"}
                    )

    def test_threshold_files_use_known_types_and_machine_types(self):
        for test_name in loader.available_tests():
            thresholds = chs_thresholds.load_test_thresholds(test_name)
            if not thresholds:
                continue
            machine_types = {
                machine_type
                for _, machine_type in chs_config.supported_platforms(
                    chs_config.load_test_config(test_name)
                )
            }
            with self.subTest(test=test_name):
                self.assertIn(thresholds["type"], THRESHOLD_TYPES)
                self.assertLessEqual(
                    set(thresholds) - {"type"}, machine_types | {"default"}
                )


class CatalogTest(unittest.TestCase):
    def test_the_catalog_lists_exactly_the_enabled_tests(self):
        enabled = {test_name for test_name, _ in loader.enabled_tests()}

        self.assertEqual(
            enabled, {entry["name"] for entry in build_catalog()["tests"]}
        )


class ThresholdsTest(unittest.TestCase):
    def test_every_enabled_threshold_file_reaches_the_assembled_map(self):
        expected = {
            test_name
            for test_name, _ in loader.enabled_tests()
            if chs_thresholds.load_test_thresholds(test_name)
        }

        published = {
            test_name
            for tests in build_thresholds().values()
            for test_name in tests
        }

        self.assertEqual(expected, published)


if __name__ == "__main__":
    unittest.main()
