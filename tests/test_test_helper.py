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

        # Fallback to interdomain when topology is None
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


if __name__ == "__main__":
    unittest.main()
