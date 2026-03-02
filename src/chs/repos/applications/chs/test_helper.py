import copy
import textwrap
from ramble.context import create_context_from_dict


class TestHelper:
    @staticmethod
    def _yaml_block(value, indent_spaces):
        prefix = " " * indent_spaces
        return textwrap.indent(value.strip(), prefix)

    @staticmethod
    def _k8s_gpu_settings(variables, nodes_str=None):
        affinity_yaml = ""
        if nodes_str:
            affinity_yaml = f'''nodeAffinity:
  requiredDuringSchedulingIgnoredDuringExecution:
    nodeSelectorTerms:
    - matchExpressions:
      - key: kubernetes.io/hostname
        operator: In
        values:
{nodes_str}
podAntiAffinity:
  requiredDuringSchedulingIgnoredDuringExecution:
  - labelSelector:
      matchExpressions:
      - key: jobset.sigs.k8s.io/jobset-name
        operator: Exists
    topologyKey: "kubernetes.io/hostname"'''

        variables["affinity_yaml"] = affinity_yaml

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
    def _merge_configs(context_dict, configs, machine_type, ramble_app_name):
        if machine_type not in configs:
            return

        def _merge(src_dict, src_key, dst_dict, dst_key):
            if src_key not in src_dict:
                return
            if dst_key not in dst_dict:
                if isinstance(src_dict[src_key], dict):
                    dst_dict[dst_key] = {}
                else:
                    dst_dict[dst_key] = []

            # If both are lists, append src_list to dst_list (or vice versa? Tests usually specific)
            # Actually, common configs are defaults. Test vars override.
            # So if dst has variable, keep it. If not, take src.
            # But lists like 'modifiers' we likely want to extend.

            if isinstance(src_dict[src_key], dict) and isinstance(
                dst_dict[dst_key], dict
            ):
                # Update destination with source values (source takes precedence)
                for k, v in src_dict[src_key].items():
                    if k not in dst_dict[dst_key]:
                        dst_dict[dst_key][k] = v
            elif isinstance(src_dict[src_key], list) and isinstance(
                dst_dict[dst_key], list
            ):
                # For lists (modifiers), we extend?
                # Usually we want common modifiers + test modifiers.
                # So append src to dst? No, usually prepend defaults?
                # Actually, duplicate modifiers might differ.
                # Let's assume extending is safe for modifiers.
                dst_dict[dst_key].extend(src_dict[src_key])
            else:
                # Type mismatch or scalar.
                # If dst has it, keep it. If not, take src.
                # But we handled 'not in dst' above.
                pass

        mt_config = configs[machine_type]
        _merge(mt_config, "variables", context_dict, "variables")

        if ramble_app_name in mt_config:
            app_cfg = mt_config[ramble_app_name]
            _merge(app_cfg, "env_vars", context_dict, "env_vars")
            _merge(app_cfg, "success_criteria", context_dict, "success")
            _merge(app_cfg, "modifiers", context_dict, "modifiers")
            _merge(app_cfg, "variables", context_dict, "variables")

    @staticmethod
    def _nccl_context(
        variables, experiment_name, configs, nodes_list, is_pairwise=False
    ):
        ctx_vars = copy.deepcopy(variables)
        machine_type = ctx_vars.get("machine_type", "a3-ultra")
        scheduler = ctx_vars.get("scheduler", "slurm")
        ctx_vars["nodes"] = nodes_list
        ctx_vars["n_nodes"] = len(nodes_list)
        sched_nodes_str = TestHelper._get_nccl_nodes_str(scheduler, nodes_list)
        if scheduler == "slurm":
            ctx_vars["extra_sbatch_headers"] = f"#SBATCH -w {sched_nodes_str}"

        elif scheduler == "gke":
            TestHelper._k8s_gpu_settings(ctx_vars, sched_nodes_str)

        if is_pairwise:
            experiment_name = (
                f"{experiment_name}-{nodes_list[0]}-{nodes_list[1]}"
            )

        context_dict = {
            "variables": ctx_vars,
        }

        TestHelper._merge_configs(
            context_dict, configs, machine_type, "nccl-tests"
        )
        return create_context_from_dict(experiment_name, context_dict)

    @staticmethod
    def nccl_test(variables, experiment_name, configs, test_type):
        all_nodes = variables.get("nodes", [])
        if test_type == "nccl-pairwise":
            if not all_nodes or len(all_nodes) < 3:
                return []

            node_groups = []
            i = 0
            while i < len(all_nodes) - 1:
                node_groups.append([all_nodes[i], all_nodes[i + 1]])
                i += 2

            if i == len(all_nodes) - 1:
                node_groups.append([all_nodes[-2], all_nodes[-1]])

            is_pairwise = True
        else:
            node_groups = [all_nodes]
            is_pairwise = False

        return [
            TestHelper._nccl_context(
                variables, experiment_name, configs, nodes, is_pairwise
            )
            for nodes in node_groups
        ]

    @staticmethod
    def dcgm_test(variables, experiment_name, configs):
        ctx_vars = copy.deepcopy(variables)
        scheduler = ctx_vars.get("scheduler", "slurm")
        machine_type = ctx_vars.get("machine_type", "a3-ultra")

        ctx_vars["node"] = ctx_vars.get("nodes")

        modifiers = []
        if scheduler == "slurm":
            ctx_vars.update(
                {
                    "process_per_node": "1",
                    "n_nodes": 1,
                    "n_ranks": "{n_nodes}*{process_per_node}",
                }
            )
            ctx_vars["extra_sbatch_headers"] = "#SBATCH -w {node}"
        elif scheduler == "gke":
            TestHelper._k8s_gpu_settings(
                ctx_vars, TestHelper._yaml_block(" - {node}", 12)
            )

        experiment_template_name = (
            f"{experiment_name}-dcgm-level-{{diag_level}}-{{node}}"
        )

        context_dict = {
            "variables": ctx_vars,
            "matrix": ["node", "diag_level"],
            "modifiers": modifiers,
        }
        TestHelper._merge_configs(
            context_dict, configs, machine_type, "nvidia-dcgm"
        )
        return create_context_from_dict(experiment_template_name, context_dict)
