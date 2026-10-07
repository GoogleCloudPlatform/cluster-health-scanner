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

import logging
import unittest
from unittest import mock
import os
import tempfile
import yaml
from chs.utils.config import load_yaml_config
from chs.utils.tests import (
    _load_tests,
    get_all_supported_tests,
    get_all_supported_machine_types,
    get_all_supported_schedulers,
    get_tests_map,
    TestsMap as _TestsMap,
    get_tests_by_machine_type,
    get_tests_by_machine_type_and_scheduler,
)


class TestTests(unittest.TestCase):
    def setUp(self):
        load_yaml_config.cache_clear()
        logging.disable(logging.CRITICAL)
        self.test_dir = tempfile.TemporaryDirectory()
        self.tests_path = os.path.join(self.test_dir.name, "tests.yaml")

    def tearDown(self):
        load_yaml_config.cache_clear()
        self.test_dir.cleanup()

    def test__load_tests_success(self):
        """Test loading tests.yaml."""
        data = {
            "tests": [
                {
                    "name": "t1",
                    "platforms": [{"scheduler": "s1", "machine_type": "m1"}],
                }
            ]
        }
        with open(self.tests_path, "w") as f:
            yaml.dump(data, f)

        with mock.patch.dict(
            os.environ, {"CHS_CONFIGS_DIR": self.test_dir.name}
        ):
            tests_data = _load_tests()
        self.assertEqual(tests_data, data)

    def test__load_tests_missing_file(self):
        """Test load_tests with missing file."""
        with mock.patch.dict(
            os.environ, {"CHS_CONFIGS_DIR": self.test_dir.name}
        ):
            with self.assertRaisesRegex(
                FileNotFoundError, "tests.yaml not found"
            ):
                _load_tests()

    def test_get_all_supported_tests(self):
        """Test deriving all supported tests."""
        data = {
            "tests": [
                {"name": "t1"},
                {"name": "t2"},
                {"name": "t3"},
            ]
        }
        with open(self.tests_path, "w") as f:
            yaml.dump(data, f)

        with mock.patch.dict(
            os.environ, {"CHS_CONFIGS_DIR": self.test_dir.name}
        ):
            tests = get_all_supported_tests()
        self.assertEqual(tests, ["t1", "t2", "t3"])

    def test_get_all_supported_machine_types(self):
        """Test deriving all supported machine types."""
        data = {
            "tests": [
                {
                    "name": "t1",
                    "platforms": [
                        {"scheduler": "s1", "machine_type": "m1"},
                        {"scheduler": "s2", "machine_type": "m2"},
                    ],
                },
                {
                    "name": "t2",
                    "platforms": [
                        {"scheduler": "s1", "machine_type": "m2"},
                        {"scheduler": "s2", "machine_type": "m3"},
                    ],
                },
            ]
        }
        with open(self.tests_path, "w") as f:
            yaml.dump(data, f)

        with mock.patch.dict(
            os.environ, {"CHS_CONFIGS_DIR": self.test_dir.name}
        ):
            mts = get_all_supported_machine_types()
        self.assertEqual(mts, ["m1", "m2", "m3"])

    def test_get_all_supported_schedulers(self):
        """Test deriving all supported schedulers."""
        data = {
            "tests": [
                {
                    "name": "t1",
                    "platforms": [
                        {"scheduler": "s1", "machine_type": "m1"},
                        {"scheduler": "s2", "machine_type": "m2"},
                    ],
                },
                {
                    "name": "t2",
                    "platforms": [
                        {"scheduler": "s2", "machine_type": "m2"},
                        {"scheduler": "s3", "machine_type": "m3"},
                    ],
                },
            ]
        }
        with open(self.tests_path, "w") as f:
            yaml.dump(data, f)

        with mock.patch.dict(
            os.environ, {"CHS_CONFIGS_DIR": self.test_dir.name}
        ):
            schedulers = get_all_supported_schedulers()
        self.assertEqual(schedulers, ["s1", "s2", "s3"])

    def test_get_tests_map(self):
        """Test deriving tests map."""
        data = {
            "tests": [
                {
                    "name": "t1",
                    "platforms": [
                        {"scheduler": "s1", "machine_type": "m1"},
                        {"scheduler": "s2", "machine_type": "m1"},
                        {"scheduler": "s1", "machine_type": "m2"},
                        {"scheduler": "s2", "machine_type": "m2"},
                    ],
                },
                {
                    "name": "t2",
                    "platforms": [
                        {"scheduler": "s1", "machine_type": "m2"},
                    ],
                },
            ]
        }
        with open(self.tests_path, "w") as f:
            yaml.dump(data, f)

        with mock.patch.dict(
            os.environ, {"CHS_CONFIGS_DIR": self.test_dir.name}
        ):
            tests_map = get_tests_map()
        expected = {
            "s1": {"m1": ["t1"], "m2": ["t1", "t2"]},
            "s2": {"m1": ["t1"], "m2": ["t1"]},
        }
        self.assertEqual(tests_map, _TestsMap(data=expected))
        self.assertEqual(tests_map.get("s1"), expected["s1"])
        self.assertEqual(tests_map.get("s2"), expected["s2"])
        self.assertIsNone(tests_map.get("unknown_scheduler"))

    def test_load_tests_malformed_yaml(self):
        """Test malformed yaml."""
        with open(self.tests_path, "w") as f:
            f.write("tests: [ unclosed_list")

        with mock.patch.dict(
            os.environ, {"CHS_CONFIGS_DIR": self.test_dir.name}
        ):
            with self.assertRaises(Exception):  # Yaml parser generic exception
                _load_tests()

    def test_get_tests_by_machine_type(self):
        """Test deriving tests by machine type."""
        data = {
            "tests": [
                {
                    "name": "t1",
                    "platforms": [
                        {"scheduler": "s1", "machine_type": "m1"},
                        {"scheduler": "s2", "machine_type": "m2"},
                    ],
                },
                {
                    "name": "t2",
                    "platforms": [
                        {"scheduler": "s1", "machine_type": "m2"},
                        {"scheduler": "s2", "machine_type": "m3"},
                    ],
                },
            ]
        }
        with open(self.tests_path, "w") as f:
            yaml.dump(data, f)

        with mock.patch.dict(
            os.environ, {"CHS_CONFIGS_DIR": self.test_dir.name}
        ):
            tests_m1 = get_tests_by_machine_type("m1")
            tests_m2 = get_tests_by_machine_type("m2")
            tests_m3 = get_tests_by_machine_type("m3")
            tests_m4 = get_tests_by_machine_type("m4")

        self.assertEqual(tests_m1, ["t1"])
        self.assertEqual(tests_m2, ["t1", "t2"])
        self.assertEqual(tests_m3, ["t2"])
        self.assertEqual(tests_m4, [])

    def test_a4x_max_slurm_supported_tests(self):
        """Test that a4x-maxgpu-4g-metal supports expected tests on slurm
        in default config.
        """
        tests = get_tests_by_machine_type_and_scheduler(
            "a4x-maxgpu-4g-metal", "slurm"
        )
        self.assertIn("nvidia-smi", tests)
        self.assertIn("dcgm", tests)
        self.assertIn("gemm", tests)
        self.assertIn("stream", tests)


if __name__ == "__main__":
    unittest.main()
