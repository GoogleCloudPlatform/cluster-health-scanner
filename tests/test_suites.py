import logging
import unittest
from unittest import mock
import os
import tempfile
import yaml
from chs.utils.suites import load_suite
from chs.exceptions import SuiteNotFoundException
from chs.utils.config import load_yaml_config


class TestSuites(unittest.TestCase):
    def setUp(self):
        load_yaml_config.cache_clear()
        logging.disable(logging.CRITICAL)
        self.test_dir = tempfile.TemporaryDirectory()
        self.suites_path = os.path.join(self.test_dir.name, "suites.yaml")
        self.tests_path = os.path.join(self.test_dir.name, "tests.yaml")

    def tearDown(self):
        load_yaml_config.cache_clear()
        self.test_dir.cleanup()

    def test_load_suite_success(self):
        """Test loading a valid suite."""
        data = {
            "suites": [
                {"name": "test_suite", "tests": ["t1", "t2"], "timeout": 100}
            ]
        }
        with open(self.suites_path, "w") as f:
            yaml.dump(data, f)

        with mock.patch.dict(
            os.environ, {"CHS_CONFIGS_DIR": self.test_dir.name}
        ):
            suite = load_suite("test_suite")
        self.assertEqual(
            suite, {"name": "test_suite", "tests": ["t1", "t2"], "timeout": 100}
        )

    def test_load_suite_missing_file(self):
        """Test missing suites.yaml file."""
        with mock.patch.dict(
            os.environ, {"CHS_CONFIGS_DIR": self.test_dir.name}
        ):
            with self.assertRaisesRegex(
                FileNotFoundError, "suites.yaml not found"
            ):
                load_suite("test_suite")

    def test_load_suite_missing_suite(self):
        """Test suite not found in suites.yaml."""
        data = {"suites": [{"name": "other_suite", "tests": ["t1"]}]}
        with open(self.suites_path, "w") as f:
            yaml.dump(data, f)

        with mock.patch.dict(
            os.environ, {"CHS_CONFIGS_DIR": self.test_dir.name}
        ):
            with self.assertRaisesRegex(
                SuiteNotFoundException, "Suite 'test_suite' not found"
            ):
                load_suite("test_suite")

    def test_load_suite_empty_suites(self):
        """Test suites.yaml with no suites."""
        data = {"suites": []}
        with open(self.suites_path, "w") as f:
            yaml.dump(data, f)

        with mock.patch.dict(
            os.environ, {"CHS_CONFIGS_DIR": self.test_dir.name}
        ):
            with self.assertRaisesRegex(
                SuiteNotFoundException, "Suite 'test_suite' not found"
            ):
                load_suite("test_suite")

    def test__load_suites_success(self):
        """Test loading all suites."""
        from chs.utils.suites import _load_suites

        data = {
            "suites": [
                {
                    "name": "test_suite",
                    "tests": [{"name": "t1"}],
                }
            ]
        }
        with open(self.suites_path, "w") as f:
            yaml.dump(data, f)

        with mock.patch.dict(
            os.environ, {"CHS_CONFIGS_DIR": self.test_dir.name}
        ):
            suites_data = _load_suites()
        self.assertEqual(suites_data, data)

    def test_get_all_suites(self):
        """Test deriving all suite names."""
        from chs.utils.suites import get_all_suites

        data = {"suites": [{"name": "suite1"}, {"name": "suite2"}]}
        with open(self.suites_path, "w") as f:
            yaml.dump(data, f)

        with mock.patch.dict(
            os.environ, {"CHS_CONFIGS_DIR": self.test_dir.name}
        ):
            suite_names = get_all_suites()
        self.assertEqual(suite_names, ["suite1", "suite2"])

    def test_load_suite_malformed_yaml(self):
        """Test malformed yaml."""
        with open(self.suites_path, "w") as f:
            f.write("suites: [ unclosed_list")

        with mock.patch.dict(
            os.environ, {"CHS_CONFIGS_DIR": self.test_dir.name}
        ):
            with self.assertRaises(Exception):  # Yaml parser generic exception
                load_suite("test_suite")


if __name__ == "__main__":
    unittest.main()
