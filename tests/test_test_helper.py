"""Unit tests for TestHelper."""

import unittest
from chs.repos.applications.chs.test_helper import TestHelper


class TestTestHelper(unittest.TestCase):
    """Tests for TestHelper."""

    def test_get_sbatch_headers(self):
        headers_dict = {
            "-w": "node1,node2",
            "--time": "10",
            "--deadline": "2026-04-15T12:00:00",
            "--gpu-bind": "single:1",
            "--empty": None,
        }
        headers = TestHelper._get_sbatch_headers(headers_dict)
        expected = (
            "#SBATCH -w node1,node2\n"
            "#SBATCH --time=10\n"
            "#SBATCH --deadline=2026-04-15T12:00:00\n"
            "#SBATCH --gpu-bind=single:1"
        )
        self.assertEqual(headers, expected)

    def test_build_common_sbatch_headers_valid(self):
        ctx_vars = {"timeout": 1200, "deadline": 3600}
        headers_dict = TestHelper._build_common_sbatch_headers(
            ctx_vars, "node1"
        )
        self.assertEqual(headers_dict["-w"], "node1")
        self.assertEqual(headers_dict["--time"], "20")
        self.assertIn("--deadline", headers_dict)

    def test_merge_configs_variables(self):
        context_dict = {"variables": {"a": 1}}
        configs = {"machine": {"variables": {"b": 2}}}
        TestHelper._merge_configs(context_dict, configs, "machine", "app")
        self.assertEqual(context_dict["variables"], {"a": 1, "b": 2})


if __name__ == "__main__":
    unittest.main()
