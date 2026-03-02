import unittest
import os
import tempfile
import yaml
from chs.utils.suites import load_suite_tests
from chs.exceptions import SuiteNotFoundException


class TestSuites(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.suites_path = os.path.join(self.test_dir.name, "suites.yaml")

    def tearDown(self):
        self.test_dir.cleanup()

    def test_load_suite_tests_success(self):
        """Test loading a valid suite."""
        data = {"suites": [{"name": "test_suite", "tests": ["t1", "t2"]}]}
        with open(self.suites_path, "w") as f:
            yaml.dump(data, f)

        tests = load_suite_tests("test_suite", self.test_dir.name)
        self.assertEqual(tests, ["t1", "t2"])

    def test_load_suite_tests_missing_file(self):
        """Test missing suites.yaml file."""
        with self.assertRaisesRegex(FileNotFoundError, "suites.yaml not found"):
            load_suite_tests("test_suite", self.test_dir.name)

    def test_load_suite_tests_missing_suite(self):
        """Test suite not found in suites.yaml."""
        data = {"suites": [{"name": "other_suite", "tests": ["t1"]}]}
        with open(self.suites_path, "w") as f:
            yaml.dump(data, f)

        with self.assertRaisesRegex(
            SuiteNotFoundException, "Suite 'test_suite' not found"
        ):
            load_suite_tests("test_suite", self.test_dir.name)

    def test_load_suite_tests_empty_suites(self):
        """Test suites.yaml with no suites."""
        data = {"suites": []}
        with open(self.suites_path, "w") as f:
            yaml.dump(data, f)

        with self.assertRaisesRegex(
            SuiteNotFoundException, "Suite 'test_suite' not found"
        ):
            load_suite_tests("test_suite", self.test_dir.name)

    def test_load_suite_tests_malformed_yaml(self):
        """Test malformed yaml."""
        with open(self.suites_path, "w") as f:
            f.write("suites: [ unclosed_list")

        with self.assertRaises(Exception):  # Yaml parser generic exception
            load_suite_tests("test_suite", self.test_dir.name)
