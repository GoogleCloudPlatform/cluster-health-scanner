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

import copy
import datetime
import logging
import os
import textwrap
from chs.utils import sanitize_dns_1123
from ramble.context import Context, create_context_from_dict

logger = logging.getLogger(__name__)


class TestHelper:
    @staticmethod
    def _yaml_block(value, indent_spaces):
        prefix = " " * indent_spaces
        return textwrap.indent(value.strip(), prefix)

    @staticmethod
    def _k8s_settings(variables, nodes_str=None, nodes_list=None):
        affinity_yaml = ""
        node_affinity_only_yaml = ""
        if nodes_str:
            node_affinity_only_yaml = f"""nodeAffinity:
  requiredDuringSchedulingIgnoredDuringExecution:
    nodeSelectorTerms:
    - matchExpressions:
      - key: kubernetes.io/hostname
        operator: In
        values:
{nodes_str}"""

            affinity_yaml = (
                node_affinity_only_yaml
                + '''
podAntiAffinity:
  requiredDuringSchedulingIgnoredDuringExecution:
  - labelSelector:
      matchExpressions:
      - key: jobset.sigs.k8s.io/jobset-name
        operator: Exists
    topologyKey: "kubernetes.io/hostname"'''
            )

        variables["affinity_yaml"] = affinity_yaml
        variables["node_affinity_only_yaml"] = node_affinity_only_yaml
        lines = node_affinity_only_yaml.split("\n")
        variables["cd_affinity_yaml"] = (
            lines[0] + "\n" + textwrap.indent("\n".join(lines[1:]), "        ")
        )

        if nodes_list:
            expr = " || ".join(
                [f"device.pool == '{name}'" for name in nodes_list]
            )
            variables["cel_filter_expression"] = expr if expr else "true"
        else:
            variables["cel_filter_expression"] = "device.pool == '{node}'"

    @staticmethod
    def _get_nodes_str(scheduler, nodes):
        if not nodes:
            return ""
        if scheduler == "slurm":
            return ",".join(nodes)
        elif scheduler == "kubernetes":
            return "\n".join(
                [TestHelper._yaml_block(f"- {n}", 12) for n in nodes]
            )
        return ""

    @staticmethod
    def _get_sbatch_headers(headers_dict):
        headers = []
        for k, v in headers_dict.items():
            if v is not None:
                if k.startswith("--"):
                    headers.append(f"#SBATCH {k}={v}")
                else:
                    headers.append(f"#SBATCH {k} {v}")
        return "\n".join(headers)

    @staticmethod
    def _build_common_sbatch_headers(ctx_vars, nodes_str):
        headers_dict = {"-w": nodes_str}
        timeout = ctx_vars.get("timeout")

        headers_dict["--time"] = str(int(timeout) // 60)

        deadline_dt = datetime.datetime.now() + datetime.timedelta(
            seconds=int(ctx_vars.get("deadline"))
        )
        headers_dict["--deadline"] = deadline_dt.strftime("%Y-%m-%dT%H:%M:%S")

        return headers_dict

    @staticmethod
    def _merge_configs(context_dict, configs, machine_type, ramble_app_name):
        if not isinstance(configs, dict) or not isinstance(context_dict, dict):
            raise ValueError(
                f"Expected both configs and context_dict to be dictionaries, "
                f"got configs={type(configs).__name__} and context_dict={type(context_dict).__name__}"
            )
        if machine_type not in configs:
            raise ValueError(
                f"Machine type '{machine_type}' not found in configs"
            )

        def _merge(src_dict, src_key, dst_dict, dst_key):
            if not isinstance(src_dict, dict) or not isinstance(dst_dict, dict):
                raise ValueError(
                    f"Expected both source and destination to be dictionaries, "
                    f"got src={type(src_dict).__name__} and dst={type(dst_dict).__name__}"
                )
            if src_key not in src_dict:
                return
            if dst_key not in dst_dict:
                if isinstance(src_dict[src_key], dict):
                    dst_dict[dst_key] = {}
                elif isinstance(src_dict[src_key], list):
                    dst_dict[dst_key] = []
                else:
                    raise ValueError(
                        f"Expected src_dict[{src_key}] to be a dictionary or list, "
                        f"got {type(src_dict[src_key]).__name__}"
                    )
            if isinstance(src_dict[src_key], dict) and isinstance(
                dst_dict[dst_key], dict
            ):
                for k, v in src_dict[src_key].items():
                    if k not in dst_dict[dst_key] or not dst_dict[dst_key][k]:
                        dst_dict[dst_key][k] = v
            elif isinstance(src_dict[src_key], list) and isinstance(
                dst_dict[dst_key], list
            ):
                for item in src_dict[src_key]:
                    if item not in dst_dict[dst_key]:
                        dst_dict[dst_key].append(item)
            else:
                raise ValueError(
                    f"Expected src_dict[{src_key}] and dst_dict[{dst_key}] to be "
                    f"dictionaries or lists, got src={type(src_dict[src_key]).__name__} "
                    f"and dst={type(dst_dict[dst_key]).__name__}"
                )

        mt_config = configs[machine_type]
        _merge(mt_config, "variables", context_dict, "variables")

        if ramble_app_name not in mt_config:
            raise ValueError(
                f"Test '{ramble_app_name}' not found in configs for machine type '{machine_type}'"
            )

        app_cfg = mt_config[ramble_app_name]
        _merge(app_cfg, "env_vars", context_dict, "env_vars")
        _merge(app_cfg, "success_criteria", context_dict, "success")
        _merge(app_cfg, "modifiers", context_dict, "modifiers")
        _merge(app_cfg, "variables", context_dict, "variables")

    @staticmethod
    def _prepare_base_context(
        ctx_vars: dict,
        configs: dict,
        ramble_app_name: str,
        nodes_str: str,
        extra_context: dict = None,
        extra_sbatch_headers: dict = None,
    ) -> dict:
        scheduler = ctx_vars.get("scheduler", "kubernetes")
        machine_type = ctx_vars.get("machine_type", "a3-ultragpu-8g")

        context_dict = {
            "variables": ctx_vars,
        }
        if extra_context:
            context_dict.update(extra_context)

        TestHelper._merge_configs(
            context_dict, configs, machine_type, ramble_app_name
        )

        if scheduler == "slurm":
            headers_dict = TestHelper._build_common_sbatch_headers(
                ctx_vars, nodes_str
            )
            if extra_sbatch_headers:
                headers_dict.update(extra_sbatch_headers)
            ctx_vars["extra_sbatch_headers"] = TestHelper._get_sbatch_headers(
                headers_dict
            )
        elif scheduler == "kubernetes":
            nodes_list = ctx_vars.get("nodes")
            nodes_list = nodes_list if isinstance(nodes_list, list) else None
            TestHelper._k8s_settings(ctx_vars, nodes_str, nodes_list=nodes_list)

        return context_dict

    @staticmethod
    def _prepare_cluster_context(
        variables: dict,
        configs: dict,
        ramble_app_name: str,
        nodes_list: list[str] = None,
        extra_context: dict = None,
        default_config_file: str = None,
    ) -> dict:
        ctx_vars = copy.deepcopy(variables)
        scheduler = ctx_vars.get("scheduler", "kubernetes")

        if nodes_list is not None:
            ctx_vars["nodes"] = nodes_list
        else:
            nodes_list = ctx_vars.get("nodes", [])

        ctx_vars["n_nodes"] = len(nodes_list)
        sched_nodes_str = TestHelper._get_nodes_str(scheduler, nodes_list)

        context_dict = TestHelper._prepare_base_context(
            ctx_vars,
            configs,
            ramble_app_name,
            nodes_str=sched_nodes_str,
            extra_context=extra_context,
        )

        if default_config_file:
            TestHelper._load_and_escape_benchmark_config(
                context_dict["variables"], default_config_file
            )

        return context_dict

    @staticmethod
    def _prepare_node_context(
        variables: dict,
        configs: dict,
        ramble_app_name: str,
        extra_context: dict = None,
        extra_sbatch_headers: dict = None,
    ) -> dict:
        ctx_vars = copy.deepcopy(variables)
        scheduler = ctx_vars.get("scheduler", "kubernetes")

        ctx_vars["node"] = ctx_vars.get("nodes")

        nodes_str = (
            "{node}"
            if scheduler == "slurm"
            else TestHelper._yaml_block(" - {node}", 12)
        )

        return TestHelper._prepare_base_context(
            ctx_vars,
            configs,
            ramble_app_name,
            nodes_str,
            extra_context=extra_context,
            extra_sbatch_headers=extra_sbatch_headers,
        )

    @staticmethod
    def _prepare_slice_context_dicts(
        variables: dict,
        experiment_name: str,
        configs: dict,
        ramble_app_name: str,
        extra_context: dict | None = None,
        extra_sbatch_headers: dict | None = None,
        default_config_file: str | None = None,
    ) -> list[tuple[str, dict]]:
        scheduler = variables.get("scheduler", "kubernetes")
        tpu_slices = variables.get("tpu_slices", {})

        if scheduler == "kubernetes":
            if not tpu_slices:
                tpu_slices = {"default-slice": variables.get("nodes", [])}

            contexts = []
            for slice_id, slice_nodes in tpu_slices.items():
                slice_vars = copy.deepcopy(variables)
                slice_vars["nodes"] = slice_nodes

                if len(tpu_slices) == 1:
                    slice_exp_name = experiment_name
                else:
                    clean_slice_id = sanitize_dns_1123(str(slice_id))
                    slice_exp_name = f"{experiment_name}-{clean_slice_id}"

                context_dict = TestHelper._prepare_cluster_context(
                    slice_vars,
                    configs,
                    ramble_app_name,
                    extra_context=extra_context,
                    default_config_file=default_config_file,
                )
                contexts.append((slice_exp_name, context_dict))
            return contexts
        elif scheduler == "slurm":
            context_dict = TestHelper._prepare_node_context(
                variables,
                configs,
                ramble_app_name,
                extra_context=extra_context,
                extra_sbatch_headers=extra_sbatch_headers,
            )
            if default_config_file:
                TestHelper._load_and_escape_benchmark_config(
                    context_dict["variables"], default_config_file
                )
            exp_name = f"{experiment_name}-{{node}}"
            return [(exp_name, context_dict)]

    @staticmethod
    def _create_node_test_context(
        variables: dict,
        experiment_name: str,
        configs: dict,
        ramble_app_name: str,
        extra_context: dict = None,
        extra_sbatch_headers: dict = None,
        default_config_file: str = None,
    ) -> Context:
        context_dict = TestHelper._prepare_node_context(
            variables,
            configs,
            ramble_app_name,
            extra_context=extra_context,
            extra_sbatch_headers=extra_sbatch_headers,
        )

        if default_config_file:
            TestHelper._load_and_escape_benchmark_config(
                context_dict["variables"], default_config_file
            )

        exp_name = f"{experiment_name}-{{node}}"
        return create_context_from_dict(exp_name, context_dict)

    @staticmethod
    def _nccl_context(
        variables,
        experiment_name,
        configs,
        nodes_list,
        test_type="nccl-cluster",
    ):
        context_dict = TestHelper._prepare_cluster_context(
            variables,
            configs,
            ramble_app_name="nccl-tests",
            nodes_list=nodes_list,
        )

        if test_type == "nccl-pairwise":
            experiment_name = (
                f"{experiment_name}-{nodes_list[0]}-{nodes_list[1]}"
            )
        elif test_type == "nccl-single-node":
            experiment_name = f"{experiment_name}-{nodes_list[0]}"
            context_dict.setdefault("env_vars", {}).setdefault("set", {})[
                "NCCL_NET"
            ] = "Socket"

        return create_context_from_dict(experiment_name, context_dict)

    @staticmethod
    def generate_topological_pairs(
        topology: dict[str, dict[str, list[str]]],
    ) -> list[list[str]]:
        """Generates pairs ordered by topology hierarchy.

        Note: This method generates disjoint pairs (e.g., [n1, n2], [n3, n4]) rather
        than a fully connected ring or all-to-all coverage, as we moved away from
        n-ring pairs.

        topology: dict mapping block names to dicts of subblocks to lists of nodes.
        """
        all_pairs = []
        inter_block_nodes = []

        # 1. Intra-subblock pairs
        for block in sorted(topology.keys()):
            subblocks = topology[block]
            block_nodes = []
            for subblock in sorted(subblocks.keys()):
                nodes = sorted(subblocks[subblock])
                for i in range(0, len(nodes) - 1, 2):
                    all_pairs.append([nodes[i], nodes[i + 1]])

                if len(nodes) % 2 != 0 and len(nodes) > 1:
                    all_pairs.append([nodes[-2], nodes[-1]])

                if nodes:
                    block_nodes.append(nodes[0])

            # 2. Inter-subblock pairs (within the same block)
            for i in range(0, len(block_nodes) - 1, 2):
                all_pairs.append([block_nodes[i], block_nodes[i + 1]])

            if len(block_nodes) % 2 != 0 and len(block_nodes) > 1:
                all_pairs.append([block_nodes[-2], block_nodes[-1]])

            if block_nodes:
                inter_block_nodes.append(block_nodes[0])

        # 3. Inter-block pairs
        for i in range(0, len(inter_block_nodes) - 1, 2):
            all_pairs.append([inter_block_nodes[i], inter_block_nodes[i + 1]])

        if len(inter_block_nodes) % 2 != 0 and len(inter_block_nodes) > 1:
            all_pairs.append([inter_block_nodes[-2], inter_block_nodes[-1]])

        return all_pairs

    @staticmethod
    def nccl_test(
        variables, experiment_name, configs, test_type, topology=None
    ):
        all_nodes = variables.get("nodes", [])

        if test_type == "nccl-pairwise":
            if not all_nodes or len(all_nodes) < 3:
                logger.warning(
                    f"nccl-pairwise test requires at least 3 nodes, but only {len(all_nodes)} were provided. Skipping test generation."
                )
                return []

            if topology:
                node_groups = TestHelper.generate_topological_pairs(topology)
            else:
                # Fallback: create a dummy topology with all nodes in a single block
                dummy_topology = {"default": {"default-subblock": all_nodes}}
                node_groups = TestHelper.generate_topological_pairs(
                    dummy_topology
                )
        elif test_type == "nccl-single-node":
            node_groups = [[n] for n in all_nodes]
        else:
            node_groups = [all_nodes]

        return [
            TestHelper._nccl_context(
                variables,
                experiment_name,
                copy.deepcopy(configs),
                nodes,
                test_type,
            )
            for nodes in node_groups
        ]

    @staticmethod
    def nvidia_smi_test(variables, experiment_name, configs):
        return TestHelper._create_node_test_context(
            variables,
            experiment_name,
            configs,
            ramble_app_name="nvidia-smi",
        )

    @staticmethod
    def dcgm_test(variables, experiment_name, configs):
        return TestHelper._create_node_test_context(
            variables,
            experiment_name,
            configs,
            ramble_app_name="nvidia-dcgm",
            extra_context={
                "matrix": ["node", "diag_level"],
                "modifiers": [],
            },
        )

    @staticmethod
    def stream_test(variables, experiment_name, configs):
        return TestHelper._create_node_test_context(
            variables,
            experiment_name,
            configs,
            ramble_app_name="nvidia-stream",
            extra_context={"modifiers": []},
            extra_sbatch_headers={"--gpu-bind": "single:1"},
        )

    @staticmethod
    def gemm_test(variables, experiment_name, configs):
        return TestHelper._create_node_test_context(
            variables,
            experiment_name,
            configs,
            ramble_app_name="nvidia-gemm",
            extra_context={"modifiers": []},
        )

    @staticmethod
    def tpu_info_test(variables, experiment_name, configs):
        slice_contexts = TestHelper._prepare_slice_context_dicts(
            variables,
            experiment_name,
            configs,
            ramble_app_name="tpu-info",
            extra_context={"modifiers": []},
        )
        return [
            create_context_from_dict(name, dct) for name, dct in slice_contexts
        ]

    @staticmethod
    def tpu_network_diagnostics_test(variables, experiment_name, configs):
        slice_contexts = TestHelper._prepare_slice_context_dicts(
            variables,
            experiment_name,
            configs,
            ramble_app_name="tpu-network-diagnostics",
            extra_context={"modifiers": []},
        )
        return [
            create_context_from_dict(name, dct) for name, dct in slice_contexts
        ]

    @staticmethod
    def _load_benchmark_config(filename):
        if not filename:
            return ""
        chs_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(chs_dir, "configs", "benchmarks", filename)
        try:
            with open(path, "r") as f:
                content = f.read()
            # Strip leading copyright comments / empty lines
            lines = content.splitlines()
            start_idx = 0
            while start_idx < len(lines):
                line = lines[start_idx].strip()
                if not line or line.startswith("#"):
                    start_idx += 1
                else:
                    break
            return "\n".join(lines[start_idx:])
        except Exception as e:
            logger.error(f"Failed to read benchmark config {path}: {e}")
            raise e

    @staticmethod
    def _load_and_escape_benchmark_config(ctx_vars, default_filename):
        benchmarks_yaml_file = ctx_vars.get("benchmarks_yaml_file")
        if not benchmarks_yaml_file:
            benchmarks_yaml_file = default_filename
            ctx_vars["benchmarks_yaml_file"] = benchmarks_yaml_file

        content = TestHelper._load_benchmark_config(benchmarks_yaml_file)
        escaped = (
            content.replace("\\", "\\\\")
            .replace("'", "'\\''")
            .replace("\n", "\\n")
        )
        ctx_vars["benchmarks_yaml_escaped"] = escaped

    @staticmethod
    def tpu_gemm_test(
        variables: dict, experiment_name: str, configs: dict
    ) -> list[Context]:
        slice_contexts = TestHelper._prepare_slice_context_dicts(
            variables,
            experiment_name,
            configs,
            ramble_app_name="tpu-gemm",
            extra_context={"modifiers": []},
            default_config_file="gemm_16384_bf16.yaml",
        )
        return [
            create_context_from_dict(name, dct) for name, dct in slice_contexts
        ]

    @staticmethod
    def tpu_hbm_test(
        variables: dict, experiment_name: str, configs: dict
    ) -> list[Context]:
        slice_contexts = TestHelper._prepare_slice_context_dicts(
            variables,
            experiment_name,
            configs,
            ramble_app_name="tpu-hbm",
            extra_context={"modifiers": []},
            default_config_file="hbm_copy_bf16.yaml",
        )
        return [
            create_context_from_dict(name, dct) for name, dct in slice_contexts
        ]

    @staticmethod
    def tpu_host_device_test(
        variables: dict, experiment_name: str, configs: dict
    ) -> list[Context]:
        slice_contexts = TestHelper._prepare_slice_context_dicts(
            variables,
            experiment_name,
            configs,
            ramble_app_name="tpu-host-device",
            extra_context={"modifiers": []},
            default_config_file="host_device.yaml",
        )
        return [
            create_context_from_dict(name, dct) for name, dct in slice_contexts
        ]

    @staticmethod
    def _inject_tpu_collectives_params(ctx_vars: dict) -> None:
        """Applies mesh shape and topology replacements to the escaped config."""
        topology = ctx_vars.get("execution_topology") or ctx_vars.get(
            "tpu_topology"
        )
        if not topology:
            raise ValueError(
                "Neither execution_topology nor tpu_topology found in variables"
            )
        if "cores_per_chip" not in ctx_vars:
            raise ValueError("cores_per_chip not found in variables")
        cores_per_chip = int(ctx_vars["cores_per_chip"])

        dims = [int(d) for d in topology.split("x") if d.isdigit()]
        if not dims:
            raise ValueError(f"Invalid topology detected {topology}")

        ici = cores_per_chip
        for d in dims:
            ici *= d

        scaled_dims = [dims[0] * cores_per_chip] + dims[1:]
        mesh_shape = "x".join(str(d) for d in scaled_dims)

        escaped = ctx_vars.get("benchmarks_yaml_escaped", "")
        escaped = (
            escaped.replace("__MESH__", mesh_shape)
            .replace("__SHARDING__", mesh_shape)
            .replace("__ICI__", str(ici))
            .replace("__START__", str(max(8, ici)))
        )
        ctx_vars["benchmarks_yaml_escaped"] = escaped

    @staticmethod
    def tpu_collectives_test(
        variables: dict,
        experiment_name: str,
        configs: dict,
    ) -> list[Context]:
        slice_contexts = TestHelper._prepare_slice_context_dicts(
            variables,
            experiment_name,
            configs,
            ramble_app_name="tpu-collectives",
            extra_context={"modifiers": []},
            default_config_file="collectives_all_gather.yaml",
        )
        contexts = []
        for name, dct in slice_contexts:
            TestHelper._inject_tpu_collectives_params(dct["variables"])
            contexts.append(create_context_from_dict(name, dct))
        return contexts

    @staticmethod
    def tpu_collectives_single_node_test(
        variables: dict,
        experiment_name: str,
        configs: dict,
    ) -> list[Context]:
        ctx_vars = copy.deepcopy(variables)
        ctx_vars["execution_topology"] = "2x2x1"
        slice_contexts = TestHelper._prepare_slice_context_dicts(
            ctx_vars,
            experiment_name,
            configs,
            ramble_app_name="tpu-collectives-single-node",
            extra_context={"modifiers": []},
            default_config_file="collectives_all_gather.yaml",
        )
        contexts = []
        for name, dct in slice_contexts:
            TestHelper._inject_tpu_collectives_params(dct["variables"])
            contexts.append(create_context_from_dict(name, dct))
        return contexts
