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

import dataclasses
import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
)
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "../src/chs/repos/applications/chs"
        )
    )
)

sys.modules["ramble"] = MagicMock()
sys.modules["ramble.context"] = MagicMock()
sys.modules["ramble.repository"] = MagicMock()
appkit_mock = MagicMock()
appkit_mock.ExecutableApplication = object
appkit_mock.maintainers = MagicMock()
appkit_mock.variable = MagicMock()
appkit_mock.executable = MagicMock()
appkit_mock.workload = MagicMock()
appkit_mock.tags = MagicMock()
sys.modules["ramble.appkit"] = appkit_mock
sys.modules["spack"] = MagicMock()
sys.modules["spack.util.spack_yaml"] = MagicMock()

from chs.repos.applications.chs.test_helper import TestHelper  # noqa: E402
from chs.repos.applications.chs.application import Chs, is_tpu  # noqa: E402


@dataclasses.dataclass
class _DeepMergeTestCase:
    name: str
    dest: dict
    src: dict
    expected_dest: dict = None
    expected_exception: type = None


@dataclasses.dataclass
class _MergeConfigsTestCase:
    name: str
    context_dict: dict
    configs: dict
    machine_type: str
    ramble_app_name: str
    expected_context: dict = None
    expected_exception: type = None


class TestApplicationHelper(unittest.TestCase):
    def test_is_tpu(self):
        self.assertTrue(is_tpu("tpu7x"))
        self.assertTrue(is_tpu("tpuv6e"))
        self.assertFalse(is_tpu("a3-ultra"))
        self.assertFalse(is_tpu("a3-high"))
        self.assertFalse(is_tpu(""))
        self.assertFalse(is_tpu(None))

    def test_deep_merge_cases(self):
        # For _deep_merge(dest, src):
        #    - dest: Base configuration (common.yaml).
        #    - src: Overriding configuration (gke.yaml or slurm.yaml).
        chs_app = Chs()

        circular_src = {}
        circular_src["self"] = circular_src

        test_cases = [
            _DeepMergeTestCase(
                name="precedence_src_overwrites_dest_scalar",
                dest={"var": "common_val"},
                src={"var": "slurm_val"},
                expected_dest={"var": "slurm_val"},
            ),
            _DeepMergeTestCase(
                name="precedence_src_merges_dest_dict",
                dest={"nested": {"var1": "common_val"}},
                src={"nested": {"var2": "slurm_val"}},
                expected_dest={
                    "nested": {"var1": "common_val", "var2": "slurm_val"}
                },
            ),
            _DeepMergeTestCase(
                name="precedence_src_overwrites_dest_dict_with_scalar",
                dest={"key": {"nested": "val"}},
                src={"key": "scalar_val"},
                expected_dest={"key": "scalar_val"},
            ),
            _DeepMergeTestCase(
                name="precedence_src_overwrites_dest_scalar_with_dict",
                dest={"key": "scalar_val"},
                src={"key": {"nested": "val"}},
                expected_dest={"key": {"nested": "val"}},
            ),
            _DeepMergeTestCase(
                name="new_keys_added",
                dest={"var1": "val1"},
                src={"var2": "val2"},
                expected_dest={"var1": "val1", "var2": "val2"},
            ),
            _DeepMergeTestCase(
                name="lists_overwritten_not_extended",
                dest={"list_key": [1, 2]},
                src={"list_key": [3, 4]},
                expected_dest={"list_key": [3, 4]},
            ),
            _DeepMergeTestCase(
                name="none_values",
                dest={"key": "val"},
                src={"key": None},
                expected_dest={"key": None},
            ),
            _DeepMergeTestCase(
                name="empty_src_and_dest",
                dest={},
                src={},
                expected_dest={},
            ),
            _DeepMergeTestCase(
                name="circular_reference",
                dest={},
                src=circular_src,
                expected_dest={"self": circular_src},
            ),
            _DeepMergeTestCase(
                name="crash_when_dest_is_list_and_src_is_dict",
                dest=[],
                src={"key": "val"},
                expected_exception=ValueError,
            ),
            _DeepMergeTestCase(
                name="crash_when_src_is_list",
                dest={},
                src=["a", "b"],
                expected_exception=ValueError,
            ),
        ]

        for case in test_cases:
            with self.subTest(case.name):
                if case.expected_exception:
                    with self.assertRaises(case.expected_exception):
                        chs_app._deep_merge(case.dest, case.src)
                else:
                    result = chs_app._deep_merge(case.dest, case.src)
                    self.assertEqual(result, case.expected_dest)

    def test_merge_configs_cases(self):
        # For TestHelper._merge_configs(context_dict, configs, ...):
        #    - context_dict (dest): Suite variables (higher precedence).
        #    - configs (src): Default configurations from deep_merge of
        #     common.yaml with gke.yaml or slurm.yaml (lower precedence).
        test_cases = [
            _MergeConfigsTestCase(
                name="precedence",
                context_dict={
                    "variables": {"diag_level": [3], "other_var": "suite_val"}
                },
                configs={
                    "machine-type": {
                        "app-name": {
                            "variables": {
                                "diag_level": [1, 2],
                                "default_var": "default_val",
                            }
                        }
                    }
                },
                machine_type="machine-type",
                ramble_app_name="app-name",
                expected_context={
                    "variables": {
                        "diag_level": [3],
                        "other_var": "suite_val",
                        "default_var": "default_val",
                    }
                },
            ),
            _MergeConfigsTestCase(
                name="modifiers",
                context_dict={"modifiers": [{"name": "suite-mod"}]},
                configs={
                    "machine-type": {
                        "app-name": {"modifiers": [{"name": "default-mod"}]}
                    }
                },
                machine_type="machine-type",
                ramble_app_name="app-name",
                expected_context={
                    "modifiers": [
                        {"name": "suite-mod"},
                        {"name": "default-mod"},
                    ]
                },
            ),
            _MergeConfigsTestCase(
                name="precedence_suite_wins_variables",
                context_dict={"variables": {"var": "suite_val"}},
                configs={
                    "machine": {
                        "variables": {"var": "machine_val"},
                        "app": {"variables": {"var": "app_val"}},
                    }
                },
                machine_type="machine",
                ramble_app_name="app",
                expected_context={"variables": {"var": "suite_val"}},
            ),
            _MergeConfigsTestCase(
                name="add_missing_variables",
                context_dict={"variables": {}},
                configs={
                    "machine": {
                        "variables": {"var1": "val1"},
                        "app": {"variables": {"var2": "val2"}},
                    }
                },
                machine_type="machine",
                ramble_app_name="app",
                expected_context={
                    "variables": {"var1": "val1", "var2": "val2"}
                },
            ),
            _MergeConfigsTestCase(
                name="type_mismatch_in_configs",
                context_dict={"variables": {"var": "val"}},
                configs={
                    "machine": {"variables": ["not", "a", "dict"], "app": {}}
                },
                machine_type="machine",
                ramble_app_name="app",
                expected_exception=ValueError,
            ),
            _MergeConfigsTestCase(
                name="type_mismatch_in_context",
                context_dict={"variables": ["not", "a", "dict"]},
                configs={"machine": {"variables": {"var": "val"}, "app": {}}},
                machine_type="machine",
                ramble_app_name="app",
                expected_exception=ValueError,
            ),
            _MergeConfigsTestCase(
                name="missing_machine_type",
                context_dict={"variables": {"var": "val"}},
                configs={},
                machine_type="machine",
                ramble_app_name="app",
                expected_exception=ValueError,
            ),
            _MergeConfigsTestCase(
                name="missing_app",
                context_dict={"variables": {"var": "val"}},
                configs={"machine": {"variables": {"var2": "val2"}}},
                machine_type="machine",
                ramble_app_name="app",
                expected_exception=ValueError,
            ),
            _MergeConfigsTestCase(
                name="empty_context_populated_by_configs",
                context_dict={},
                configs={
                    "machine": {
                        "variables": {"var": "val"},
                        "app": {"modifiers": ["mod"]},
                    }
                },
                machine_type="machine",
                ramble_app_name="app",
                expected_context={
                    "variables": {"var": "val"},
                    "modifiers": ["mod"],
                },
            ),
            _MergeConfigsTestCase(
                name="falsy_value_in_suite_overridden",
                context_dict={"variables": {"var": ""}},
                configs={
                    "machine": {"variables": {"var": "default"}, "app": {}}
                },
                machine_type="machine",
                ramble_app_name="app",
                expected_context={"variables": {"var": "default"}},
            ),
            _MergeConfigsTestCase(
                name="machine_level_wins_over_app_level_in_configs",
                context_dict={"variables": {}},
                configs={
                    "machine": {
                        "variables": {"var": "machine_val"},
                        "app": {"variables": {"var": "app_val"}},
                    }
                },
                machine_type="machine",
                ramble_app_name="app",
                expected_context={"variables": {"var": "machine_val"}},
            ),
            _MergeConfigsTestCase(
                name="duplicate_modifiers_deduplicated",
                context_dict={"modifiers": ["mod1"]},
                configs={"machine": {"app": {"modifiers": ["mod1"]}}},
                machine_type="machine",
                ramble_app_name="app",
                expected_context={"modifiers": ["mod1"]},
            ),
            _MergeConfigsTestCase(
                name="empty_list_in_suite_extended_by_configs",
                context_dict={"modifiers": []},
                configs={"machine": {"app": {"modifiers": ["mod"]}}},
                machine_type="machine",
                ramble_app_name="app",
                expected_context={"modifiers": ["mod"]},
            ),
            _MergeConfigsTestCase(
                name="empty_dict_in_suite_overridden_by_configs",
                context_dict={"variables": {"my_dict": {}}},
                configs={
                    "machine": {
                        "app": {"variables": {"my_dict": {"key": "val"}}}
                    }
                },
                machine_type="machine",
                ramble_app_name="app",
                expected_context={"variables": {"my_dict": {"key": "val"}}},
            ),
            _MergeConfigsTestCase(
                name="invalid_mt_config_type_raises_error",
                context_dict={},
                configs={"machine": "not_a_dict"},
                machine_type="machine",
                ramble_app_name="app",
                expected_exception=ValueError,
            ),
            _MergeConfigsTestCase(
                name="invalid_configs_type_raises_error",
                context_dict={},
                configs="not_a_dict",
                machine_type="machine",
                ramble_app_name="app",
                expected_exception=ValueError,
            ),
            _MergeConfigsTestCase(
                name="invalid_context_dict_type_raises_error",
                context_dict="not_a_dict",
                configs={},
                machine_type="machine",
                ramble_app_name="app",
                expected_exception=ValueError,
            ),
            _MergeConfigsTestCase(
                name="ignores_unsupported_keys_in_configs",
                context_dict={},
                configs={"machine": {"app": {"unsupported_key": {"val": 1}}}},
                machine_type="machine",
                ramble_app_name="app",
                expected_context={},
            ),
        ]

        for case in test_cases:
            with self.subTest(case.name):
                if case.expected_exception:
                    with self.assertRaises(case.expected_exception):
                        TestHelper._merge_configs(
                            case.context_dict,
                            case.configs,
                            case.machine_type,
                            case.ramble_app_name,
                        )
                else:
                    TestHelper._merge_configs(
                        case.context_dict,
                        case.configs,
                        case.machine_type,
                        case.ramble_app_name,
                    )
                    self.assertEqual(case.context_dict, case.expected_context)


if __name__ == "__main__":
    unittest.main()
