import copy
import datetime
import logging
import textwrap
from ramble.context import create_context_from_dict

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
    def _get_nccl_nodes_str(scheduler, nodes):
        if not nodes:
            return ""
        if scheduler == "slurm":
            return ",".join(nodes)
        elif scheduler == "gke":
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
    def _nccl_context(
        variables,
        experiment_name,
        configs,
        nodes_list,
        test_type="nccl-cluster",
    ):
        ctx_vars = copy.deepcopy(variables)
        machine_type = ctx_vars.get("machine_type", "a3-ultra")
        scheduler = ctx_vars.get("scheduler", "slurm")
        ctx_vars["nodes"] = nodes_list
        ctx_vars["n_nodes"] = len(nodes_list)
        sched_nodes_str = TestHelper._get_nccl_nodes_str(scheduler, nodes_list)

        context_dict = {
            "variables": ctx_vars,
        }
        TestHelper._merge_configs(
            context_dict, configs, machine_type, "nccl-tests"
        )

        if scheduler == "slurm":
            headers_dict = TestHelper._build_common_sbatch_headers(
                ctx_vars, sched_nodes_str
            )
            ctx_vars["extra_sbatch_headers"] = TestHelper._get_sbatch_headers(
                headers_dict
            )

        elif scheduler == "gke":
            TestHelper._k8s_settings(
                ctx_vars, sched_nodes_str, nodes_list=nodes_list
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
        ctx_vars = copy.deepcopy(variables)
        scheduler = ctx_vars.get("scheduler", "slurm")
        machine_type = ctx_vars.get("machine_type", "a3-ultra")

        ctx_vars["node"] = ctx_vars.get("nodes")

        experiment_template_name = f"{experiment_name}-{{node}}"

        context_dict = {
            "variables": ctx_vars,
        }
        TestHelper._merge_configs(
            context_dict, configs, machine_type, "nvidia-smi"
        )

        if scheduler == "slurm":
            headers_dict = TestHelper._build_common_sbatch_headers(
                ctx_vars, "{node}"
            )
            ctx_vars["extra_sbatch_headers"] = TestHelper._get_sbatch_headers(
                headers_dict
            )
        elif scheduler == "gke":
            TestHelper._k8s_settings(
                ctx_vars, TestHelper._yaml_block(" - {node}", 12)
            )
        return create_context_from_dict(experiment_template_name, context_dict)

    @staticmethod
    def dcgm_test(variables, experiment_name, configs):
        ctx_vars = copy.deepcopy(variables)
        scheduler = ctx_vars.get("scheduler", "slurm")
        machine_type = ctx_vars.get("machine_type", "a3-ultra")

        ctx_vars["node"] = ctx_vars.get("nodes")

        modifiers = []
        experiment_template_name = f"{experiment_name}-{{node}}"

        context_dict = {
            "variables": ctx_vars,
            "matrix": ["node", "diag_level"],
            "modifiers": modifiers,
        }
        TestHelper._merge_configs(
            context_dict, configs, machine_type, "nvidia-dcgm"
        )

        if scheduler == "slurm":
            headers_dict = TestHelper._build_common_sbatch_headers(
                ctx_vars, "{node}"
            )
            ctx_vars["extra_sbatch_headers"] = TestHelper._get_sbatch_headers(
                headers_dict
            )
        elif scheduler == "gke":
            TestHelper._k8s_settings(
                ctx_vars, TestHelper._yaml_block(" - {node}", 12)
            )
        return create_context_from_dict(experiment_template_name, context_dict)

    @staticmethod
    def stream_test(variables, experiment_name, configs):
        ctx_vars = copy.deepcopy(variables)
        scheduler = ctx_vars.get("scheduler", "slurm")
        machine_type = ctx_vars.get("machine_type", "a3-ultra")

        ctx_vars["node"] = ctx_vars.get("nodes")

        modifiers = []
        experiment_template_name = f"{experiment_name}-{{node}}"

        context_dict = {
            "variables": ctx_vars,
            "modifiers": modifiers,
        }
        TestHelper._merge_configs(
            context_dict, configs, machine_type, "nvidia-stream"
        )

        if scheduler == "slurm":
            headers_dict = TestHelper._build_common_sbatch_headers(
                ctx_vars, "{node}"
            )
            headers_dict["--gpu-bind"] = "single:1"
            ctx_vars["extra_sbatch_headers"] = TestHelper._get_sbatch_headers(
                headers_dict
            )
        elif scheduler == "gke":
            TestHelper._k8s_settings(
                ctx_vars, TestHelper._yaml_block(" - {node}", 12)
            )
        return create_context_from_dict(experiment_template_name, context_dict)

    @staticmethod
    def gemm_test(variables, experiment_name, configs):
        ctx_vars = copy.deepcopy(variables)
        scheduler = ctx_vars.get("scheduler", "slurm")
        machine_type = ctx_vars.get("machine_type", "a3-ultra")

        ctx_vars["node"] = ctx_vars.get("nodes")

        modifiers = []
        experiment_template_name = f"{experiment_name}-{{node}}"

        context_dict = {
            "variables": ctx_vars,
            "modifiers": modifiers,
        }
        TestHelper._merge_configs(
            context_dict, configs, machine_type, "nvidia-gemm"
        )

        if scheduler == "slurm":
            headers_dict = TestHelper._build_common_sbatch_headers(
                ctx_vars, "{node}"
            )
            ctx_vars["extra_sbatch_headers"] = TestHelper._get_sbatch_headers(
                headers_dict
            )
        elif scheduler == "gke":
            TestHelper._k8s_settings(
                ctx_vars, TestHelper._yaml_block(" - {node}", 12)
            )
        return create_context_from_dict(experiment_template_name, context_dict)

    @staticmethod
    def tpu_info_test(variables, experiment_name, configs):
        ctx_vars = copy.deepcopy(variables)
        scheduler = ctx_vars.get("scheduler", "gke")
        machine_type = ctx_vars.get("machine_type", "tpu7x")

        ctx_vars["node"] = ctx_vars.get("nodes")
        ctx_vars["n_nodes"] = len(ctx_vars["nodes"])
        modifiers = []
        experiment_template_name = f"{experiment_name}-{{node}}"

        context_dict = {
            "variables": ctx_vars,
            "modifiers": modifiers,
        }
        TestHelper._merge_configs(
            context_dict, configs, machine_type, "tpu-info"
        )

        if scheduler == "gke":
            TestHelper._k8s_settings(
                ctx_vars, TestHelper._yaml_block(" - {node}", 12)
            )
        return create_context_from_dict(experiment_template_name, context_dict)
