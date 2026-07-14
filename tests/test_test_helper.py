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
                "a3-ultragpu-8g": {
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

    def test_prepare_base_context(self):
        ctx_vars = {
            "scheduler": "slurm",
            "machine_type": "a3-ultragpu-8g",
            "timeout": 600,
            "deadline": 3600,
            "nodes": ["node1"],
        }
        configs = {
            "a3-ultragpu-8g": {
                "my-app": {
                    "variables": {"default_var": "val"},
                }
            }
        }
        context_dict = TestHelper._prepare_base_context(
            ctx_vars,
            configs,
            ramble_app_name="my-app",
            nodes_str="node1",
            extra_context={"custom_key": "custom_val"},
            extra_sbatch_headers={"--gpu-bind": "single:1"},
        )
        self.assertEqual(context_dict["custom_key"], "custom_val")
        self.assertEqual(context_dict["variables"]["default_var"], "val")
        self.assertIn(
            "#SBATCH --gpu-bind=single:1",
            context_dict["variables"]["extra_sbatch_headers"],
        )

    def test_prepare_cluster_context(self):
        variables = {
            "scheduler": "kubernetes",
            "machine_type": "tpu7x",
        }
        configs = {
            "tpu7x": {
                "my-app": {
                    "variables": {},
                }
            }
        }
        context_dict = TestHelper._prepare_cluster_context(
            variables,
            configs,
            ramble_app_name="my-app",
            nodes_list=["node1", "node2"],
        )
        self.assertEqual(context_dict["variables"]["n_nodes"], 2)
        self.assertEqual(context_dict["variables"]["nodes"], ["node1", "node2"])
        self.assertIn(
            "device.pool == 'node1'",
            context_dict["variables"]["cel_filter_expression"],
        )

    def test_create_node_test_context(self):
        variables = {
            "scheduler": "slurm",
            "machine_type": "a3-ultragpu-8g",
            "timeout": 600,
            "deadline": 3600,
            "nodes": ["node1"],
        }
        configs = {
            "a3-ultragpu-8g": {
                "my-app": {
                    "variables": {},
                }
            }
        }
        with mock.patch(
            "chs.repos.applications.chs.test_helper.create_context_from_dict"
        ) as mock_create:
            mock_create.return_value = "mock_context"
            result = TestHelper._create_node_test_context(
                variables, "test-node", configs, ramble_app_name="my-app"
            )
            self.assertEqual(result, "mock_context")
            mock_create.assert_called_once()
            call_args = mock_create.call_args[0]
            self.assertEqual(call_args[0], "test-node-{node}")

    def test_tpu_collectives_single_node_test(self):
        variables = {
            "scheduler": "kubernetes",
            "machine_type": "tpuv6e",
            "tpu_topology": "4x4",
            "cores_per_chip": 1,
            "nodes": ["node1"],
        }
        configs = {
            "tpuv6e": {
                "tpu_topology": "4x4",
                "tpu-collectives-single-node": {
                    "variables": {
                        "driver_escape_overrides": "export TPU_TOPOLOGY=2x2x1"
                    },
                },
            }
        }

        mock_yaml_content = """
        benchmarks:
          all_gather:
            mesh_shape: "__MESH__"
            ici_size: __ICI__
        """

        with (
            mock.patch(
                "chs.repos.applications.chs.test_helper.create_context_from_dict"
            ) as mock_create,
            mock.patch(
                "chs.repos.applications.chs.test_helper.TestHelper._load_benchmark_config",
                return_value=mock_yaml_content,
            ),
        ):
            mock_create.side_effect = lambda name, ctx: ctx

            contexts = TestHelper.tpu_collectives_single_node_test(
                variables, "test-collectives", configs
            )
            self.assertEqual(len(contexts), 1)
            ctx = contexts[0]

            escaped_config = ctx["variables"]["benchmarks_yaml_escaped"]
            unescaped = escaped_config.replace("\\n", "\n").replace(
                "\\\\", "\\"
            )

            self.assertIn('mesh_shape: "2x2x2"', unescaped)
            self.assertIn("ici_size: 4", unescaped)


if __name__ == "__main__":
    unittest.main()
