import copy
from itertools import combinations
import textwrap
from ramble.context import create_context_from_dict
from ramble.wmkit import *


class TestHelper:

  @staticmethod
  def _yaml_block(value, indent_spaces):
    prefix = " " * indent_spaces
    return textwrap.indent(value.strip(), prefix)

  @staticmethod
  def gpu_settings(variables, nodes_str=None):
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
  def dcgm_test(variables, experiment_name):
    variables["node"] = variables.get("nodes")
    variables["diag_level"] = [1, 2]

    TestHelper.gpu_settings(variables, TestHelper._yaml_block(" - {node}", 12))

    experiment_template_name = experiment_name + "-dcgm-r{diag_level}-{node}"
    exp_context = create_context_from_dict(
        experiment_template_name,
        {
            "variables": variables,
            "matrix": ["node", "diag_level"],
            "modifiers": [{"name": "gpu-persistence-mode"}],
            "variants": {"workflow_manager": "gke-suite-configs"},
        },
    )
    return exp_context

  @staticmethod
  def nccl_test(variables, experiment_name, workload="all-gather"):
    variables["n_nodes"] = len(variables.get("nodes", []))

    nodes_list = variables.get("nodes", [])
    node_str = ""
    if nodes_list:
      nodes_str = "\n".join(
          [TestHelper._yaml_block(f" - {n}", 12) for n in nodes_list]
      )

    TestHelper.gpu_settings(variables, nodes_str)

    machine_type = variables.get("machine_type", "a3-ultra")

    modifiers = [{"name": "gpu-persistence-mode"}]
    if machine_type == "a3-ultra":
      modifiers.append({"name": "nccl-gib"})

    experiment_template_name = experiment_name + f"-nccl-{workload}"
    exp_context = create_context_from_dict(
        experiment_template_name,
        {
            "variables": variables,
            "modifiers": modifiers,
            "variants": {"workflow_manager": "gke-suite-configs"},
        },
    )
    return exp_context

  @staticmethod
  def nccl_pairwise(variables, experiment_name, workload="all-reduce"):
    nodes = variables.get("nodes")
    if not nodes or len(nodes) < 2:
      return []

    machine_type = variables.get("machine_type", "a3-ultra")

    modifiers = [{"name": "gpu-persistence-mode"}]
    if machine_type == "a3-ultra":
      modifiers.append({"name": "nccl-gib"})

    contexts = []
    for i, (n1, n2) in enumerate(combinations(nodes, 2)):
      pair_vars = copy.deepcopy(variables)
      pair_vars["nodes"] = [n1, n2]
      pair_vars["n_nodes"] = 2

      nodes_str = (
          f"{TestHelper._yaml_block(f'- {n1}', 12)}\n{TestHelper._yaml_block(f'- {n2}', 12)}"
      )
      TestHelper.gpu_settings(pair_vars, nodes_str)

      pair_template_name = (
          f"{experiment_name}-nccl-pairwise-{workload}-{n1}-{n2}"
      )
      ctx = create_context_from_dict(
          pair_template_name,
          {
              "variables": pair_vars,
              "modifiers": modifiers,
              "variants": {"workflow_manager": "gke-suite-configs"},
          },
      )
      contexts.append(ctx)

    return contexts
