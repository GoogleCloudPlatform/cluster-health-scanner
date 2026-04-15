"""Unit tests for TestHelper."""

import sys
import unittest
from unittest import mock

# Mock ramble module dependencies
ramble_mock = mock.Mock()
ramble_mock.context = mock.Mock()
sys.modules["ramble"] = ramble_mock
sys.modules["ramble.context"] = ramble_mock.context

from chs.repos.applications.chs.test_helper import TestHelper  # noqa: E402


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
        configs = {"machine": {"variables": {"b": 2}, "app": {}}}
        TestHelper._merge_configs(context_dict, configs, "machine", "app")
        self.assertEqual(context_dict["variables"], {"a": 1, "b": 2})

    def test_generate_topological_pairs(self):
        topology = {
            "block1": {
                "subblock1": ["node1", "node2", "node7"],
                "subblock2": ["node3", "node4"],
            },
            "block2": {
                "subblock3": ["node5", "node6"],
            },
        }
        expected_pairs = [
            ["node1", "node2"],
            ["node2", "node7"],
            ["node3", "node4"],
            ["node1", "node3"],
            ["node5", "node6"],
            ["node1", "node5"],
        ]
        pairs = TestHelper.generate_topological_pairs(topology)
        self.assertEqual(pairs, expected_pairs)

    def test_generate_topological_pairs_flat_topology(self):
        topology = {
            "rack1": {"default-subblock": ["node1", "node2", "node3", "node4"]},
        }
        expected_pairs = [
            ["node1", "node2"],
            ["node3", "node4"],
        ]
        pairs = TestHelper.generate_topological_pairs(topology)
        self.assertEqual(pairs, expected_pairs)

    def test_generate_topological_pairs_too_few_nodes(self):
        topology = {
            "rack1": {"default-subblock": ["node1"]},
        }
        pairs = TestHelper.generate_topological_pairs(topology)
        self.assertEqual(pairs, [])

    def test_nccl_test_fallback_disjoint(self):
        variables = {
            "nodes": ["node1", "node2", "node3", "node4"],
            "timeout": 600,
            "deadline": 2400,
        }
        with mock.patch(
            "chs.repos.applications.chs.test_helper.create_context_from_dict"
        ) as mock_create:
            mock_create.side_effect = lambda name, ctx: mock.Mock(
                context_name=name
            )
            configs = {
                "a3-ultra": {
                    "nccl-tests": {
                        "variables": {},
                        "env_vars": {},
                        "success_criteria": [],
                        "modifiers": [],
                    }
                }
            }
            results = TestHelper.nccl_test(
                variables, "test-exp", configs, "nccl-pairwise", topology=None
            )
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0].context_name, "test-exp-node1-node2")
            self.assertEqual(results[1].context_name, "test-exp-node3-node4")


if __name__ == "__main__":
    unittest.main()
