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

import copy
import sys
import unittest
from unittest import mock

# Mock ramble module dependencies
ramble_mock = mock.Mock()
ramble_mock.context = mock.Mock()
ramble_mock.wmkit = mock.Mock()


class _MockWorkflowManagerBase:
    def __init__(self, *args, **kwargs):
        pass


ramble_mock.wmkit.WorkflowManagerBase = _MockWorkflowManagerBase
ramble_mock.wmkit.workflow_manager_variable = mock.Mock()
ramble_mock.wmkit.maintainers = mock.Mock()
ramble_mock.wmkit.tags = mock.Mock()
ramble_mock.wmkit.variable = mock.Mock()
ramble_mock.wmkit.register_template = mock.Mock()
ramble_mock.wmkit.__all__ = [
    "WorkflowManagerBase",
    "workflow_manager_variable",
    "maintainers",
    "tags",
    "variable",
    "register_template",
]
ramble_mock.experiment_result = mock.Mock()
sys.modules["ramble"] = ramble_mock
sys.modules["ramble.context"] = ramble_mock.context
sys.modules["ramble.wmkit"] = ramble_mock.wmkit
sys.modules["ramble.experiment_result"] = ramble_mock.experiment_result

from chs.repos.applications.chs.test_helper import TestHelper as _TestHelper  # noqa: E402
from chs.repos.workflow_managers.kubernetes_jobset.workflow_manager import (  # noqa: E402
    KubernetesJobset,
)


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
        headers = _TestHelper._get_sbatch_headers(headers_dict)
        expected = (
            "#SBATCH -w node1,node2\n"
            "#SBATCH --time=10\n"
            "#SBATCH --deadline=2026-04-15T12:00:00\n"
            "#SBATCH --gpu-bind=single:1"
        )
        self.assertEqual(headers, expected)

    def test_build_common_sbatch_headers_valid(self):
        ctx_vars = {"timeout": 1200, "deadline": 3600}
        headers_dict = _TestHelper._build_common_sbatch_headers(
            ctx_vars, "node1"
        )
        self.assertEqual(headers_dict["-w"], "node1")
        self.assertEqual(headers_dict["--time"], "20")
        self.assertEqual(headers_dict["--deadline"], "now+4800seconds")

        # Verify ceiling rounding for non-multiples of 60 seconds
        ctx_vars_non_multiple = {"timeout": 150, "deadline": 3600}
        headers_dict_nm = _TestHelper._build_common_sbatch_headers(
            ctx_vars_non_multiple, "node1"
        )
        self.assertEqual(headers_dict_nm["--time"], "3")

        ctx_vars_small = {"timeout": 30, "deadline": 3600}
        headers_dict_small = _TestHelper._build_common_sbatch_headers(
            ctx_vars_small, "node1"
        )
        self.assertEqual(headers_dict_small["--time"], "1")

    def test_build_common_sbatch_headers_missing_deadline(self):
        ctx_vars = {"timeout": 1200}
        with self.assertRaisesRegex(
            ValueError, "Invalid deadline value 'None'"
        ):
            _TestHelper._build_common_sbatch_headers(ctx_vars, "node1")

    def test_build_common_sbatch_headers_missing_timeout(self):
        ctx_vars = {"deadline": 3600}
        with self.assertRaisesRegex(ValueError, "Invalid timeout value 'None'"):
            _TestHelper._build_common_sbatch_headers(ctx_vars, "node1")

    def test_build_common_sbatch_headers_missing_both(self):
        ctx_vars = {}
        with self.assertRaisesRegex(ValueError, "Invalid timeout value 'None'"):
            _TestHelper._build_common_sbatch_headers(ctx_vars, "node1")

    def test_build_common_sbatch_headers_invalid_deadline(self):
        ctx_vars = {"timeout": 1200, "deadline": "invalid"}
        with self.assertRaisesRegex(
            ValueError, "Invalid deadline value 'invalid'"
        ):
            _TestHelper._build_common_sbatch_headers(ctx_vars, "node1")

    def test_build_common_sbatch_headers_invalid_timeout(self):
        ctx_vars = {"timeout": "invalid", "deadline": 3600}
        with self.assertRaisesRegex(
            ValueError, "Invalid timeout value 'invalid'"
        ):
            _TestHelper._build_common_sbatch_headers(ctx_vars, "node1")

    def test_merge_configs_variables(self):
        context_dict = {"variables": {"a": 1}}
        configs = {"machine": {"variables": {"b": 2}, "app": {}}}
        _TestHelper._merge_configs(context_dict, configs, "machine", "app")
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
        pairs = _TestHelper.generate_topological_pairs(topology)
        self.assertEqual(pairs, expected_pairs)

    def test_generate_topological_pairs_flat_topology(self):
        topology = {
            "rack1": {"default-subblock": ["node1", "node2", "node3", "node4"]},
        }
        expected_pairs = [
            ["node1", "node2"],
            ["node3", "node4"],
        ]
        pairs = _TestHelper.generate_topological_pairs(topology)
        self.assertEqual(pairs, expected_pairs)

    def test_generate_topological_pairs_too_few_nodes(self):
        topology = {
            "rack1": {"default-subblock": ["node1"]},
        }
        pairs = _TestHelper.generate_topological_pairs(topology)
        self.assertEqual(pairs, [])

    def test_nccl_test_domain_type_variable(self):
        topology = {
            "block1": {
                "subblock1": ["node1", "node2"],
                "subblock2": ["node3", "node4"],
            }
        }
        variables = {
            "machine_type": "a4x-highgpu-4g",
            "nodes": ["node1", "node2", "node3", "node4"],
            "timeout": 600,
            "deadline": 2400,
        }
        configs = {
            "a4x-highgpu-4g": {
                "nccl-tests": {
                    "variables": {},
                }
            }
        }
        contexts_created = []

        def mock_create(name, ctx):
            contexts_created.append((name, ctx))
            return mock.Mock(context_name=name)

        with mock.patch(
            "chs.repos.applications.chs.test_helper.create_context_from_dict",
            side_effect=mock_create,
        ):
            _TestHelper.nccl_test(
                variables, "exp", configs, "nccl-pairwise", topology=topology
            )

        domain_types = {
            name: ctx.get("variables", {}).get("domain_type")
            for name, ctx in contexts_created
        }
        self.assertEqual(domain_types["exp-node1-node2"], "intradomain")
        self.assertEqual(domain_types["exp-node3-node4"], "intradomain")
        self.assertEqual(domain_types["exp-node1-node3"], "interdomain")

        # Fallback to interdomain when topology is None on Kubernetes
        contexts_none = []
        with mock.patch(
            "chs.repos.applications.chs.test_helper.create_context_from_dict",
            side_effect=lambda name, ctx: (
                contexts_none.append((name, ctx)),
                mock.Mock(context_name=name),
            )[1],
        ):
            _TestHelper.nccl_test(
                variables, "exp", configs, "nccl-pairwise", topology=None
            )
        for _, ctx in contexts_none:
            self.assertEqual(
                ctx.get("variables", {}).get("domain_type"), "interdomain"
            )

        # Fallback to intradomain when topology is None and
        # allow_interdomain_pairs is False
        contexts_none = []
        variables_no_interdomain = copy.deepcopy(variables)
        variables_no_interdomain["allow_interdomain_pairs"] = False
        with mock.patch(
            "chs.repos.applications.chs.test_helper.create_context_from_dict",
            side_effect=lambda name, ctx: (
                contexts_none.append((name, ctx)),
                mock.Mock(context_name=name),
            )[1],
        ):
            _TestHelper.nccl_test(
                variables_no_interdomain,
                "exp",
                configs,
                "nccl-pairwise",
                topology=None,
            )
        for _, ctx in contexts_none:
            self.assertEqual(
                ctx.get("variables", {}).get("domain_type"), "intradomain"
            )

        # Fallback to interdomain when node is unmapped
        partial_loc = {"node1": {"block": "b1", "subblock": "sb1"}}
        contexts_unmapped = []
        with mock.patch(
            "chs.repos.applications.chs.test_helper.create_context_from_dict",
            side_effect=lambda name, ctx: (
                contexts_unmapped.append((name, ctx)),
                mock.Mock(context_name=name),
            )[1],
        ):
            _TestHelper._nccl_context(
                variables,
                "exp",
                configs,
                ["node1", "node3"],
                test_type="nccl-pairwise",
                node_to_loc=partial_loc,
            )
        self.assertEqual(
            contexts_unmapped[-1][1]["variables"]["domain_type"], "interdomain"
        )

    def test_generate_domain_groups(self):
        topology = {
            "block1": {
                "subblock1": ["node2", "node1", "node3"],
                "subblock2": ["node4", "node5", "node6"],
            },
            "block2": {
                "subblock3": ["node7", "node8", "node9"],
            },
        }
        expected = [
            ("subblock1", ["node1", "node2", "node3"]),
            ("subblock2", ["node4", "node5", "node6"]),
            ("subblock3", ["node7", "node8", "node9"]),
        ]
        self.assertEqual(_TestHelper.generate_domain_groups(topology), expected)

    def test_generate_domain_groups_filters_smaller_domains(self):
        topology = {
            "block1": {
                "subblock1": ["node1", "node2", "node3"],
                "subblock2": ["node4", "node5"],
            },
        }
        # By default (min_nodes_per_domain=3), subblock2 with 2 nodes is
        # filtered out and logs a warning.
        with mock.patch("chs.utils.topology.logger.warning") as mock_warn:
            self.assertEqual(
                _TestHelper.generate_domain_groups(topology),
                [
                    ("subblock1", ["node1", "node2", "node3"]),
                ],
            )
            mock_warn.assert_called_once()
            self.assertIn("subblock2", mock_warn.call_args[0])
        # When min_nodes_per_domain=2, smaller domains are retained.
        self.assertEqual(
            _TestHelper.generate_domain_groups(
                topology, min_nodes_per_domain=2
            ),
            [
                ("subblock1", ["node1", "node2", "node3"]),
                ("subblock2", ["node4", "node5"]),
            ],
        )

    def test_generate_domain_groups_filters_untargeted_nodes(self):
        topology = {
            "block1": {
                "subblock1": ["node1", "node2", "node3", "node4"],
                "subblock2": ["node5", "node6", "node7", "node8"],
            },
        }
        target_nodes = ["node1", "node2", "node3", "node5", "node6", "node7"]
        groups = _TestHelper.generate_domain_groups(
            topology,
            target_nodes=target_nodes,
        )
        self.assertEqual(
            groups,
            [
                ("subblock1", ["node1", "node2", "node3"]),
                ("subblock2", ["node5", "node6", "node7"]),
            ],
        )

    def test_generate_domain_groups_disambiguates_repeated_subblocks(self):
        topology = {
            "block1": {"default-subblock": ["node1", "node2", "node3"]},
            "block2": {"default-subblock": ["node4", "node5", "node6"]},
        }
        self.assertEqual(
            _TestHelper.generate_domain_groups(topology),
            [
                ("block1-default-subblock", ["node1", "node2", "node3"]),
                ("block2-default-subblock", ["node4", "node5", "node6"]),
            ],
        )

    def test_generate_domain_groups_empty_topology(self):
        self.assertEqual(_TestHelper.generate_domain_groups(None), [])
        self.assertEqual(_TestHelper.generate_domain_groups({}), [])

    def test_generate_domain_groups_empty_target_list_targets_nothing(self):
        topology = {
            "block1": {
                "subblock1": ["node1", "node2", "node3"],
                "subblock2": ["node4", "node5", "node6"],
            },
        }
        self.assertEqual(
            _TestHelper.generate_domain_groups(topology, target_nodes=[]), []
        )

    def test_generate_domain_groups_ignores_unknown_target_nodes(self):
        topology = {"block1": {"subblock1": ["node1", "node2", "node3"]}}
        groups = _TestHelper.generate_domain_groups(
            topology,
            target_nodes=["node1", "node2", "node3", "ghost1", "ghost2"],
        )
        self.assertEqual(groups, [("subblock1", ["node1", "node2", "node3"])])

    def test_generate_domain_groups_skips_empty_subblock(self):
        topology = {
            "block1": {
                "subblock1": [],
                "subblock2": ["node1", "node2", "node3"],
            },
        }
        self.assertEqual(
            _TestHelper.generate_domain_groups(topology),
            [("subblock2", ["node1", "node2", "node3"])],
        )

    def test_generate_domain_groups_filters_domains_below_minimum(self):
        topology = {
            "block1": {
                "subblock1": ["node1"],
                "subblock2": ["node2", "node3"],
            },
            "block2": {"subblock3": ["node4"]},
        }
        # All domains have <= 2 nodes, so default min_nodes_per_domain=3
        # yields none.
        self.assertEqual(
            _TestHelper.generate_domain_groups(topology),
            [],
        )
        # When min_nodes_per_domain=1, all nonempty domains are included.
        self.assertEqual(
            _TestHelper.generate_domain_groups(
                topology, min_nodes_per_domain=1
            ),
            [
                ("subblock1", ["node1"]),
                ("subblock2", ["node2", "node3"]),
                ("subblock3", ["node4"]),
            ],
        )

    def _nccl_configs(self):
        return {
            "a4x-highgpu-4g": {
                "nccl-tests": {
                    "variables": {},
                    "env_vars": {},
                    "success_criteria": [],
                    "modifiers": [],
                    "intradomain": {
                        "variables": {
                            "custom_resources_yaml": "kind: ComputeDomain",
                        },
                    },
                }
            }
        }

    def test_nccl_test_domain_type_variable_slurm(self):
        topology = {
            "block1": {
                "subblock1": ["node1", "node2"],
                "subblock2": ["node3", "node4"],
            }
        }
        variables = {
            "machine_type": "a4x-maxgpu-4g-metal",
            "scheduler": "slurm",
            "nodes": ["node1", "node2", "node3", "node4"],
            "timeout": 600,
            "deadline": 2400,
        }
        configs = {
            "a4x-maxgpu-4g-metal": {
                "nccl-tests": {
                    "variables": {},
                }
            }
        }
        contexts_created = []

        def mock_create(name, ctx):
            contexts_created.append((name, ctx))
            return mock.Mock(context_name=name)

        with mock.patch(
            "chs.repos.applications.chs.test_helper.create_context_from_dict",
            side_effect=mock_create,
        ):
            _TestHelper.nccl_test(
                variables, "exp", configs, "nccl-pairwise", topology=topology
            )

        pair_names = [name for name, _ in contexts_created]
        self.assertEqual(len(pair_names), 3)
        self.assertIn("exp-node1-node2", pair_names)
        self.assertIn("exp-node3-node4", pair_names)
        self.assertIn("exp-node1-node3", pair_names)
        domain_types = {
            name: ctx.get("variables", {}).get("domain_type")
            for name, ctx in contexts_created
        }
        self.assertEqual(domain_types["exp-node1-node2"], "intradomain")
        self.assertEqual(domain_types["exp-node3-node4"], "intradomain")
        self.assertEqual(domain_types["exp-node1-node3"], "interdomain")

    def test_nccl_test_allow_interdomain_pairs_false(self):
        topology = {
            "block1": {
                "subblock1": ["node1", "node2"],
                "subblock2": ["node3", "node4"],
            }
        }
        variables = {
            "machine_type": "a4x-highgpu-4g",
            "nodes": ["node1", "node2", "node3", "node4"],
            "timeout": 600,
            "deadline": 2400,
            "allow_interdomain_pairs": False,
        }
        configs = {
            "a4x-highgpu-4g": {
                "nccl-tests": {
                    "variables": {},
                }
            }
        }
        contexts_created = []

        def mock_create(name, ctx):
            contexts_created.append((name, ctx))
            return mock.Mock(context_name=name)

        with mock.patch(
            "chs.repos.applications.chs.test_helper.create_context_from_dict",
            side_effect=mock_create,
        ):
            _TestHelper.nccl_test(
                variables, "exp", configs, "nccl-pairwise", topology=topology
            )

        pair_names = [name for name, _ in contexts_created]
        self.assertEqual(len(pair_names), 2)
        self.assertIn("exp-node1-node2", pair_names)
        self.assertIn("exp-node3-node4", pair_names)
        for _, ctx in contexts_created:
            self.assertEqual(
                ctx.get("variables", {}).get("domain_type"), "intradomain"
            )

    def test_nccl_test_block_topology_no_pairs_warning(self):
        # 4 subblocks with 1 node each: cannot form any intra-subblock pairs
        topology = {
            f"block{i}": {f"subblock{i}": [f"node{i}"]} for i in range(4)
        }
        variables = {
            "machine_type": "a4x-maxgpu-4g-metal",
            "scheduler": "slurm",
            "allow_interdomain_pairs": False,
            "nodes": [f"node{i}" for i in range(4)],
            "timeout": 600,
            "deadline": 2400,
        }
        configs = {
            "a4x-maxgpu-4g-metal": {
                "nccl-tests": {
                    "variables": {},
                }
            }
        }
        with mock.patch.object(_TestHelper, "_nccl_context") as mock_ctx:
            with mock.patch(
                "chs.repos.applications.chs.test_helper.logger.warning"
            ) as mock_warn:
                res = _TestHelper.nccl_test(
                    variables,
                    "exp",
                    configs,
                    "nccl-pairwise",
                    topology=topology,
                )
            self.assertEqual(res, [])
            mock_ctx.assert_not_called()
            mock_warn.assert_called_once()
            self.assertIn(
                "No node pairs could be formed",
                mock_warn.call_args[0][0],
            )

    def test_nccl_test_single_node_slurm(self):
        variables = {
            "machine_type": "a4x-maxgpu-4g-metal",
            "scheduler": "slurm",
            "nodes": ["node1", "node2"],
            "timeout": 600,
            "deadline": 2400,
        }
        configs = {
            "a4x-maxgpu-4g-metal": {
                "nccl-tests": {
                    "variables": {
                        "container_name": "nccl-gib-a4x-max",
                    },
                }
            }
        }
        contexts_created = []

        def mock_create(name, ctx):
            contexts_created.append((name, ctx))
            return mock.Mock(context_name=name)

        with mock.patch(
            "chs.repos.applications.chs.test_helper.create_context_from_dict",
            side_effect=mock_create,
        ):
            _TestHelper.nccl_test(variables, "exp", configs, "nccl-single-node")

        pair_names = [name for name, _ in contexts_created]
        self.assertEqual(pair_names, ["exp-node1", "exp-node2"])
        for _, ctx in contexts_created:
            self.assertEqual(ctx["env_vars"]["set"]["NCCL_NET"], "Socket")

    def test_nccl_test_domain(self):
        variables = {
            "nodes": [
                "node1",
                "node2",
                "node3",
                "node4",
                "node5",
                "node6",
            ],
            "machine_type": "a4x-highgpu-4g",
            "timeout": 600,
            "deadline": 2400,
        }
        topology = {
            "block1": {
                "subblock1": ["node1", "node2", "node3"],
                "subblock2": ["node4", "node5", "node6"],
            },
        }
        with mock.patch(
            "chs.repos.applications.chs.test_helper.create_context_from_dict"
        ) as mock_create:
            mock_create.side_effect = lambda name, ctx: (name, ctx)
            results = _TestHelper.nccl_test(
                variables,
                "test-exp",
                self._nccl_configs(),
                "nccl-domain",
                topology=topology,
            )

        self.assertEqual(
            [name for name, _ in results],
            ["test-exp-subblock1", "test-exp-subblock2"],
        )
        self.assertEqual(
            [ctx["variables"]["nodes"] for _, ctx in results],
            [["node1", "node2", "node3"], ["node4", "node5", "node6"]],
        )
        self.assertEqual(
            [ctx["variables"]["n_nodes"] for _, ctx in results], [3, 3]
        )
        self.assertEqual(
            [ctx["variables"]["domain_type"] for _, ctx in results],
            ["intradomain", "intradomain"],
        )
        self.assertEqual(
            [ctx["variables"]["custom_resources_yaml"] for _, ctx in results],
            ["kind: ComputeDomain", "kind: ComputeDomain"],
        )

    def test_nccl_test_domain_with_varied_domain_sizes(self):
        variables = {
            "nodes": ["node1", "node2", "node3", "node4"],
            "machine_type": "a4x-highgpu-4g",
            "timeout": 600,
            "deadline": 2400,
        }
        topology = {
            "block1": {
                "subblock1": ["node1"],
                "subblock2": ["node2", "node3", "node4"],
            },
        }
        with mock.patch(
            "chs.repos.applications.chs.test_helper.create_context_from_dict"
        ) as mock_create:
            mock_create.side_effect = lambda name, ctx: (name, ctx)
            results = _TestHelper.nccl_test(
                variables,
                "test-exp",
                self._nccl_configs(),
                "nccl-domain",
                topology=topology,
            )

        # subblock1 has only 1 node, so it is filtered out (requires >= 3
        # nodes per domain)
        self.assertEqual(
            [name for name, _ in results],
            ["test-exp-subblock2"],
        )
        self.assertEqual(
            [ctx["variables"]["nodes"] for _, ctx in results],
            [["node2", "node3", "node4"]],
        )

    def test_nccl_test_domain_skips_fewer_than_3_nodes(self):
        variables = {
            "nodes": ["node1", "node2"],
            "machine_type": "a4x-highgpu-4g",
            "timeout": 600,
            "deadline": 2400,
        }
        topology = {"block1": {"subblock1": ["node1", "node2"]}}
        results = _TestHelper.nccl_test(
            variables,
            "test-exp",
            self._nccl_configs(),
            "nccl-domain",
            topology=topology,
        )
        self.assertEqual(results, [])

    def test_nccl_test_domain_without_topology(self):
        variables = {
            "nodes": ["node1", "node2", "node3"],
            "machine_type": "a4x-highgpu-4g",
            "timeout": 600,
            "deadline": 2400,
        }
        results = _TestHelper.nccl_test(
            variables,
            "test-exp",
            self._nccl_configs(),
            "nccl-domain",
            topology=None,
        )
        self.assertEqual(results, [])

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
            results = _TestHelper.nccl_test(
                variables, "test-exp", configs, "nccl-pairwise", topology=None
            )
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0].context_name, "test-exp-node1-node2")
            self.assertEqual(results[1].context_name, "test-exp-node3-node4")

    def test_nccl_cluster_skipped_when_few_nodes(self):
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
        for node_list in [[], ["node1"], ["node1", "node2"]]:
            variables = {
                "nodes": node_list,
                "timeout": 600,
                "deadline": 2400,
            }
            results = _TestHelper.nccl_test(
                variables, "test-exp", configs, "nccl-cluster"
            )
            self.assertEqual(results, [])

    def test_nccl_cluster_runs_when_three_or_more_nodes(self):
        variables = {
            "nodes": ["node1", "node2", "node3"],
            "timeout": 600,
            "deadline": 2400,
        }
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
        with mock.patch(
            "chs.repos.applications.chs.test_helper.create_context_from_dict"
        ) as mock_create:
            mock_create.side_effect = lambda name, ctx: mock.Mock(
                context_name=name
            )
            results = _TestHelper.nccl_test(
                variables, "test-exp", configs, "nccl-cluster"
            )
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].context_name, "test-exp")

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
        context_dict = _TestHelper._prepare_base_context(
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
        context_dict = _TestHelper._prepare_cluster_context(
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
            result = _TestHelper._create_node_test_context(
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

            contexts = _TestHelper.tpu_collectives_single_node_test(
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

    def test_nccl_pairwise_intradomain_and_interdomain_configs(self):
        topology = {
            "block1": {
                "subblock1": ["node1", "node2"],
                "subblock2": ["node3", "node4"],
            }
        }
        variables = {
            "machine_type": "a4x-highgpu-4g",
            "nodes": ["node1", "node2", "node3", "node4"],
            "timeout": 600,
            "deadline": 2400,
        }
        configs = {
            "a4x-highgpu-4g": {
                "variables": {
                    "pod_annotations_yaml": (
                        "networking.gke.io/default-interface: eth0\n"
                    ),
                    "container_resources_yaml": (
                        "limits:\n  nvidia.com/gpu: 4\n"
                    ),
                },
                "nccl-tests": {
                    "variables": {},
                    "intradomain": {
                        "variables": {
                            "custom_resources_yaml": (
                                "apiVersion: resource.nvidia.com/v1beta1\n"
                                "kind: ComputeDomain\n"
                            ),
                            "pod_resource_claims_yaml": (
                                "- name: compute-domain-channel\n"
                            ),
                            "extra_pod_annotations_yaml": (
                                "compute-domain.nvidia.com/domain-name: cd\n"
                            ),
                            "extra_container_resources_yaml": (
                                "claims:\n- name: compute-domain-channel\n"
                            ),
                        }
                    },
                    "interdomain": {"variables": {}},
                },
            }
        }
        contexts_created = {}

        def mock_create(name, ctx):
            contexts_created[name] = ctx
            return mock.Mock(context_name=name)

        with mock.patch(
            "chs.repos.applications.chs.test_helper.create_context_from_dict",
            side_effect=mock_create,
        ):
            _TestHelper.nccl_test(
                variables, "exp", configs, "nccl-pairwise", topology=topology
            )

        # Pair 1: node1-node2 (intradomain) -> merges intradomain config
        ctx_intra = contexts_created["exp-node1-node2"]["variables"]
        self.assertEqual(ctx_intra["domain_type"], "intradomain")
        self.assertIn("ComputeDomain", ctx_intra["custom_resources_yaml"])
        self.assertIn(
            "compute-domain-channel", ctx_intra["pod_resource_claims_yaml"]
        )
        self.assertIn(
            "compute-domain.nvidia.com", ctx_intra["extra_pod_annotations_yaml"]
        )
        self.assertIn(
            "networking.gke.io/default-interface: eth0",
            ctx_intra["pod_annotations_yaml"],
        )
        self.assertIn(
            "claims:\n- name: compute-domain-channel",
            ctx_intra["extra_container_resources_yaml"],
        )
        self.assertIn(
            "limits:\n  nvidia.com/gpu: 4",
            ctx_intra["container_resources_yaml"],
        )

        # Pair 3: node1-node3 (interdomain) -> base config without ComputeDomain
        ctx_inter = contexts_created["exp-node1-node3"]["variables"]
        self.assertEqual(ctx_inter["domain_type"], "interdomain")
        self.assertNotIn("custom_resources_yaml", ctx_inter)
        self.assertNotIn("pod_resource_claims_yaml", ctx_inter)
        self.assertNotIn("extra_pod_annotations_yaml", ctx_inter)
        self.assertNotIn("extra_container_resources_yaml", ctx_inter)
        self.assertIn(
            "networking.gke.io/default-interface: eth0",
            ctx_inter["pod_annotations_yaml"],
        )
        self.assertIn(
            "limits:\n  nvidia.com/gpu: 4",
            ctx_inter["container_resources_yaml"],
        )

    def test_nccl_pairwise_slurm_domain_variables(self):
        topology = {
            "block1": {
                "subblock1": ["node1", "node2"],
                "subblock2": ["node3", "node4"],
            }
        }
        variables = {
            "machine_type": "a4x-maxgpu-4g-metal",
            "scheduler": "slurm",
            "allow_interdomain_pairs": False,
            "nodes": ["node1", "node2", "node3", "node4"],
            "timeout": 600,
            "deadline": 2400,
        }
        configs = {
            "a4x-maxgpu-4g-metal": {
                "nccl-tests": {
                    "env_vars": {
                        "set": {
                            "NCCL_NET": "gIB",
                            "NCCL_DEBUG": "INFO",
                            "NCCL_MNNVL_ENABLE": "{mnnvl_enable}",
                            "NCCL_NVLS_ENABLE": "{nvls_enable}",
                        }
                    },
                    "variables": {
                        "container_name": "nccl-gib-a4x-max",
                        "mnnvl_enable": 2,
                        "nvls_enable": 2,
                    },
                    "intradomain": {
                        "variables": {"mnnvl_enable": 1, "nvls_enable": 2}
                    },
                    "interdomain": {
                        "variables": {"mnnvl_enable": 0, "nvls_enable": 0}
                    },
                },
            }
        }
        contexts_created = {}

        def mock_create(name, ctx):
            contexts_created[name] = ctx
            return mock.Mock(context_name=name)

        with mock.patch(
            "chs.repos.applications.chs.test_helper.create_context_from_dict",
            side_effect=mock_create,
        ):
            _TestHelper.nccl_test(
                variables, "exp", configs, "nccl-pairwise", topology=topology
            )

        intra = contexts_created["exp-node1-node2"]["variables"]

        self.assertEqual(intra["mnnvl_enable"], 1)
        self.assertEqual(intra["nvls_enable"], 2)

        for name in ("exp-node1-node2", "exp-node3-node4"):
            env = contexts_created[name]["env_vars"]["set"]
            self.assertEqual(env["NCCL_NET"], "gIB")
            self.assertEqual(env["NCCL_DEBUG"], "INFO")
            self.assertEqual(env["NCCL_MNNVL_ENABLE"], "{mnnvl_enable}")
            self.assertEqual(env["NCCL_NVLS_ENABLE"], "{nvls_enable}")

        # Verify interdomain config merges when domain_type is interdomain
        from chs.utils.topology import build_node_to_location

        node_to_loc = build_node_to_location(topology)
        with mock.patch(
            "chs.repos.applications.chs.test_helper.create_context_from_dict",
            side_effect=mock_create,
        ):
            _TestHelper._nccl_context(
                variables,
                "exp",
                configs,
                ["node1", "node3"],
                test_type="nccl-pairwise",
                node_to_loc=node_to_loc,
            )
        inter = contexts_created["exp-node1-node3"]["variables"]
        self.assertEqual(inter["mnnvl_enable"], 0)
        self.assertEqual(inter["nvls_enable"], 0)

    def test_kubernetes_jobset_extra_annotations_and_resources(self):
        wm = KubernetesJobset(mock.Mock())
        wm.app_inst = mock.Mock()

        var_map = {
            "job_name": "test-job",
            "command": "echo test",
            "experiment_run_dir": "/tmp/exp",
            "app_workdir": "",
            "container_image": "image:latest",
            "jobset_labels_yaml": "",
            "jobset_annotations_yaml": "",
            "pod_labels_yaml": "",
            "pod_annotations_yaml": "networking.gke.io/default-interface: eth0",
            "extra_pod_annotations_yaml": (
                "compute-domain.nvidia.com/domain-name: cd"
            ),
            "node_selector_yaml": "",
            "affinity_yaml": "",
            "tolerations_yaml": "",
            "container_resources_yaml": (
                "limits:\n  nvidia.com/gpu: 4\nrequests:\n  nvidia.com/gpu: 4"
            ),
            "extra_container_resources_yaml": (
                "claims:\n- name: compute-domain-channel"
            ),
            "container_security_context_yaml": "",
            "extra_volumes_yaml": "",
            "extra_volume_mounts_yaml": "",
            "init_containers_yaml": "",
            "custom_resources_yaml": "",
            "pod_resource_claims_yaml": "",
            "suspend_jobset": "false",
        }
        wm.app_inst.expander.expand_var_name.side_effect = (
            lambda k: var_map.get(k, "")
        )
        res = wm._jobset_vars()

        # Check annotations merged
        self.assertIn(
            "networking.gke.io/default-interface: eth0",
            res["pod_annotations_section"],
        )
        self.assertIn(
            "compute-domain.nvidia.com/domain-name: cd",
            res["pod_annotations_section"],
        )

        # Check container resources merged
        self.assertIn(
            "limits:\n                    nvidia.com/gpu: 4",
            res["container_resources_section"],
        )
        self.assertIn(
            "claims:\n                  - name: compute-domain-channel",
            res["container_resources_section"],
        )

        # When extra fields are empty
        var_map["extra_pod_annotations_yaml"] = ""
        var_map["extra_container_resources_yaml"] = ""
        res_empty = wm._jobset_vars()
        self.assertIn(
            "networking.gke.io/default-interface: eth0",
            res_empty["pod_annotations_section"],
        )
        self.assertNotIn(
            "compute-domain.nvidia.com", res_empty["pod_annotations_section"]
        )
        self.assertNotIn("claims:", res_empty["container_resources_section"])

    def test_nccl_test_minimum_nodes(self):
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
        with mock.patch(
            "chs.repos.applications.chs.test_helper.create_context_from_dict"
        ) as mock_create:
            mock_create.side_effect = lambda name, ctx: mock.Mock(
                context_name=name
            )

            # 2 nodes should succeed for nccl-pairwise
            vars_2_nodes = {
                "scheduler": "kubernetes",
                "machine_type": "a3-ultragpu-8g",
                "nodes": ["node1", "node2"],
            }
            res_pairwise_2 = _TestHelper.nccl_test(
                vars_2_nodes, "exp", configs, "nccl-pairwise"
            )
            self.assertEqual(len(res_pairwise_2), 1)

            # 1 node should be skipped for nccl-pairwise
            vars_1_node = {
                "scheduler": "kubernetes",
                "machine_type": "a3-ultragpu-8g",
                "nodes": ["node1"],
            }
            res_pairwise_1 = _TestHelper.nccl_test(
                vars_1_node, "exp", configs, "nccl-pairwise"
            )
            self.assertEqual(res_pairwise_1, [])

            # 0 nodes should be skipped for nccl-pairwise
            vars_0_nodes = {
                "scheduler": "kubernetes",
                "machine_type": "a3-ultragpu-8g",
                "nodes": [],
            }
            res_pairwise_0 = _TestHelper.nccl_test(
                vars_0_nodes, "exp", configs, "nccl-pairwise"
            )
            self.assertEqual(res_pairwise_0, [])

            # 3 nodes should succeed for nccl-cluster
            vars_3_nodes = {
                "scheduler": "kubernetes",
                "machine_type": "a3-ultragpu-8g",
                "nodes": ["node1", "node2", "node3"],
            }
            res_cluster_3 = _TestHelper.nccl_test(
                vars_3_nodes, "exp", configs, "nccl-cluster"
            )
            self.assertEqual(len(res_cluster_3), 1)

            # 2 nodes should be skipped for nccl-cluster
            res_cluster_2 = _TestHelper.nccl_test(
                vars_2_nodes, "exp", configs, "nccl-cluster"
            )
            self.assertEqual(res_cluster_2, [])


if __name__ == "__main__":
    unittest.main()
