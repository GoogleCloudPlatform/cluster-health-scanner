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
import logging
import math
import os
import textwrap

from chs.testkit import config as chs_config
from chs.utils.cluster import sanitize_dns_1123

logger = logging.getLogger(__name__)

_NODE_LIST_INDENT = 12


def yaml_block(value: str, indent_spaces: int) -> str:
    return textwrap.indent(value.strip(), " " * indent_spaces)


def nodes_str(scheduler: str, nodes: list[str] | None) -> str:
    if not nodes:
        return ""
    if scheduler == "slurm":
        return ",".join(nodes)
    if scheduler == "kubernetes":
        return "\n".join(yaml_block(f"- {n}", _NODE_LIST_INDENT) for n in nodes)
    return ""


_WORKER_LABEL = "chs.gke.io/worker"
_JOB_NAME_LABEL = "batch.kubernetes.io/job-name"
_TPU_SLICE_TOPOLOGY_KEY = "cloud.google.com/gke-tpu-slice"
EXCLUSIVE_SLICE_KEY = "_chs_exclusive_tpu_slice"

_ONE_POD_PER_NODE_TERM = f'''- labelSelector:
    matchExpressions:
    - key: {_WORKER_LABEL}
      operator: Exists
  topologyKey: "kubernetes.io/hostname"'''

_SAME_SLICE_POD_AFFINITY = f"""podAffinity:
  requiredDuringSchedulingIgnoredDuringExecution:
  - labelSelector:
      matchExpressions:
      - key: {_JOB_NAME_LABEL}
        operator: In
        values:
        - {{job_name}}
    topologyKey: {_TPU_SLICE_TOPOLOGY_KEY}"""

_EXCLUSIVE_SLICE_TERM = f"""- labelSelector:
    matchExpressions:
    - key: {_WORKER_LABEL}
      operator: Exists
    - key: {_JOB_NAME_LABEL}
      operator: NotIn
      values:
      - {{job_name}}
  topologyKey: {_TPU_SLICE_TOPOLOGY_KEY}"""


def node_affinity_yaml(nodes_block: str) -> str:
    return f"""nodeAffinity:
  requiredDuringSchedulingIgnoredDuringExecution:
    nodeSelectorTerms:
    - matchExpressions:
      - key: kubernetes.io/hostname
        operator: In
        values:
{nodes_block}"""


def kubernetes_affinity_yaml(
    nodes_block: str | None, exclusive_slice: bool = False
) -> str:
    """Returns the pod affinity YAML of a CHS test.

    Pods are pinned to nodes_block and at most one CHS pod runs per node.
    With exclusive_slice, all pods of the Job share one TPU slice that no
    other CHS Job uses.
    """
    sections = []
    anti_affinity_terms = []
    if nodes_block:
        sections.append(node_affinity_yaml(nodes_block))
        anti_affinity_terms.append(_ONE_POD_PER_NODE_TERM)
    if exclusive_slice:
        sections.append(_SAME_SLICE_POD_AFFINITY)
        anti_affinity_terms.append(_EXCLUSIVE_SLICE_TERM)
    if anti_affinity_terms:
        terms = "\n".join(textwrap.indent(t, "  ") for t in anti_affinity_terms)
        sections.append(
            "podAntiAffinity:\n"
            "  requiredDuringSchedulingIgnoredDuringExecution:\n"
            f"{terms}"
        )
    return "\n".join(sections)


def mark_exclusive_tpu_slice(variables: dict) -> None:
    """Makes the affinity of the test request an exclusive TPU slice."""
    variables[EXCLUSIVE_SLICE_KEY] = True


def pop_exclusive_tpu_slice(variables: dict) -> bool:
    return bool(variables.pop(EXCLUSIVE_SLICE_KEY, False))


def apply_kubernetes_affinity(
    variables: dict,
    nodes_block: str | None,
) -> None:
    variables["affinity_yaml"] = kubernetes_affinity_yaml(
        nodes_block, pop_exclusive_tpu_slice(variables)
    )


def sbatch_headers(headers: dict) -> str:
    lines = []
    for key, value in headers.items():
        if value is None:
            continue
        if key.startswith("--"):
            lines.append(f"#SBATCH {key}={value}")
        else:
            lines.append(f"#SBATCH {key} {value}")
    return "\n".join(lines)


def _int_var(variables: dict, name: str) -> int:
    value = variables.get(name)
    try:
        return int(value)
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Invalid {name} value '{value}': must be an integer"
        ) from e


def common_sbatch_headers(variables: dict, nodes_block: str) -> dict:
    timeout_seconds = _int_var(variables, "timeout")
    deadline_seconds = _int_var(variables, "deadline")
    return {
        "-w": nodes_block,
        # Slurm takes whole minutes; round up so short timeouts are not 0.
        "--time": str(max(1, math.ceil(timeout_seconds / 60))),
        "--deadline": f"now+{deadline_seconds + timeout_seconds}seconds",
    }


def merge_resolved_config(context_dict: dict, resolved: dict) -> None:
    _merge_section(resolved, "env_vars", context_dict, "env_vars")
    # Ramble's create_context_from_dict expects the key "success_criteria".
    _merge_section(
        resolved, "success_criteria", context_dict, "success_criteria"
    )
    _merge_section(resolved, "modifiers", context_dict, "modifiers")
    _merge_section(resolved, "variables", context_dict, "variables")


_UNSET = object()


def _is_empty(value) -> bool:
    """True when a value carries no choice: unset, null, or empty."""
    return value is _UNSET or value is None or value in ("", [], {}, ())


def _merge_section(
    src: dict,
    src_key: str,
    dst: dict,
    dst_key: str,
) -> None:
    if not isinstance(src, dict) or not isinstance(dst, dict):
        raise ValueError(
            f"Expected both source and destination to be dictionaries, "
            f"got src={type(src).__name__} and dst={type(dst).__name__}"
        )
    if src_key not in src:
        return

    source_value = src[src_key]
    if dst_key not in dst:
        if isinstance(source_value, dict):
            dst[dst_key] = {}
        elif isinstance(source_value, list):
            dst[dst_key] = []
        else:
            raise ValueError(
                f"Expected src_dict[{src_key}] to be a dictionary or list, "
                f"got {type(source_value).__name__}"
            )

    target_value = dst[dst_key]
    if isinstance(source_value, dict) and isinstance(target_value, dict):
        for key, value in source_value.items():
            if _is_empty(target_value.get(key, _UNSET)):
                target_value[key] = value
    elif isinstance(source_value, list) and isinstance(target_value, list):
        for item in source_value:
            if item not in target_value:
                target_value.append(item)
    else:
        raise ValueError(
            f"Expected src_dict[{src_key}] and dst_dict[{dst_key}] to be "
            f"dictionaries or lists, got src={type(source_value).__name__} "
            f"and dst={type(target_value).__name__}"
        )


def build_base_context(
    variables: dict,
    nodes_block: str,
    extra_context: dict | None = None,
    extra_sbatch_headers: dict | None = None,
    domain_type: str | None = None,
) -> dict:
    scheduler = variables.get("scheduler", "kubernetes")
    machine_type = variables.get("machine_type", "a3-ultragpu-8g")

    if domain_type:
        variables["domain_type"] = domain_type

    context_dict = {"variables": variables}
    if extra_context:
        if "variables" in extra_context:
            raise ValueError(
                "Cannot set 'variables' in extra_context; configure variables "
                "via config.yaml or prepare_variables()."
            )
        context_dict.update(copy.deepcopy(extra_context))

    merge_resolved_config(
        context_dict,
        chs_config.resolve_test(
            variables["chs_test_name"],
            machine_type,
            scheduler,
            domain_type,
        ),
    )

    if scheduler == "slurm":
        headers = common_sbatch_headers(variables, nodes_block)
        if extra_sbatch_headers:
            headers.update(extra_sbatch_headers)
        variables["extra_sbatch_headers"] = sbatch_headers(headers)
    elif scheduler == "kubernetes":
        apply_kubernetes_affinity(variables, nodes_block)

    return context_dict


def build_node_context(
    variables: dict,
    extra_context: dict | None = None,
    extra_sbatch_headers: dict | None = None,
    benchmark_config: str | None = None,
    benchmarks_dir: str | None = None,
) -> dict:
    ctx_vars = copy.deepcopy(variables)
    scheduler = ctx_vars.get("scheduler", "kubernetes")

    ctx_vars["node"] = ctx_vars.get("nodes")
    ctx_vars["n_nodes"] = 1

    nodes_block = (
        "{node}"
        if scheduler == "slurm"
        else yaml_block(" - {node}", _NODE_LIST_INDENT)
    )

    context_dict = build_base_context(
        ctx_vars,
        nodes_block,
        extra_context=extra_context,
        extra_sbatch_headers=extra_sbatch_headers,
    )

    if benchmark_config:
        apply_benchmark_config(
            context_dict["variables"], benchmark_config, benchmarks_dir
        )

    return context_dict


def build_cluster_context(
    variables: dict,
    nodes_list: list[str] | None = None,
    extra_context: dict | None = None,
    extra_sbatch_headers: dict | None = None,
    benchmark_config: str | None = None,
    benchmarks_dir: str | None = None,
    domain_type: str | None = None,
) -> dict:
    ctx_vars = copy.deepcopy(variables)
    scheduler = ctx_vars.get("scheduler", "kubernetes")

    if nodes_list is not None:
        ctx_vars["nodes"] = nodes_list
    else:
        nodes_list = ctx_vars.get("nodes", [])

    ctx_vars["n_nodes"] = len(nodes_list)

    context_dict = build_base_context(
        ctx_vars,
        nodes_str(scheduler, nodes_list),
        extra_context=extra_context,
        extra_sbatch_headers=extra_sbatch_headers,
        domain_type=domain_type,
    )

    if benchmark_config:
        apply_benchmark_config(
            context_dict["variables"], benchmark_config, benchmarks_dir
        )

    return context_dict


def build_slice_contexts(
    variables: dict,
    experiment_name: str,
    extra_context: dict | None = None,
    extra_sbatch_headers: dict | None = None,
    benchmark_config: str | None = None,
    benchmarks_dir: str | None = None,
) -> list[tuple[str, dict]]:
    scheduler = variables.get("scheduler", "kubernetes")

    if scheduler == "slurm":
        context_dict = build_node_context(
            variables,
            extra_context=extra_context,
            extra_sbatch_headers=extra_sbatch_headers,
            benchmark_config=benchmark_config,
            benchmarks_dir=benchmarks_dir,
        )
        return [(f"{experiment_name}-{{node}}", context_dict)]

    tpu_slices = variables.get("tpu_slices", {})
    if not tpu_slices:
        tpu_slices = {"default-slice": variables.get("nodes", [])}

    slice_topologies = variables.get("tpu_slice_topologies", {})
    dynamic_pools = variables.get("tpu_dynamic_pools", [])

    contexts = []
    for slice_id, slice_nodes in tpu_slices.items():
        slice_vars = copy.deepcopy(variables)
        slice_vars["nodes"] = slice_nodes

        slice_topology = slice_topologies.get(slice_id)
        if slice_topology:
            slice_vars["tpu_topology"] = slice_topology

        if slice_id in dynamic_pools:
            apply_dynamic_slice_annotations(slice_vars)

        if len(tpu_slices) == 1:
            slice_experiment_name = experiment_name
        else:
            slice_experiment_name = (
                f"{experiment_name}-{sanitize_dns_1123(str(slice_id))}"
            )

        context_dict = build_cluster_context(
            slice_vars,
            extra_context=extra_context,
            benchmark_config=benchmark_config,
            benchmarks_dir=benchmarks_dir,
        )
        if not slice_topology:
            context_dict["variables"]["node_selector_yaml"] = ""
        contexts.append((slice_experiment_name, context_dict))

    return contexts


def apply_dynamic_slice_annotations(slice_vars: dict) -> None:
    """Requests an exclusive, correctly shaped dynamic TPU slice."""
    mark_exclusive_tpu_slice(slice_vars)

    topology = slice_vars.get("tpu_topology")
    shape = f'cloud.google.com/gke-tpu-slice-topology: "{topology}"'
    existing = slice_vars.get("pod_annotations_yaml", "").strip()
    slice_vars["pod_annotations_yaml"] = (
        f"{existing}\\n{shape}" if existing else shape
    )


def load_benchmark_config(filename: str | None, benchmarks_dir: str) -> str:
    if not filename:
        return ""
    path = os.path.join(benchmarks_dir, filename)
    try:
        with open(path, "r") as f:
            content = f.read()
    except OSError as e:
        logger.error(
            "Failed to read the benchmark input '%s' at %s: %s. Check that "
            "the file exists and is readable.",
            filename,
            path,
            e,
        )
        raise

    lines = content.splitlines()
    start = 0
    while start < len(lines):
        stripped = lines[start].strip()
        if not stripped or stripped.startswith("#"):
            start += 1
        else:
            break
    return "\n".join(lines[start:])


def apply_benchmark_config(
    variables: dict,
    default_filename: str | None,
    benchmarks_dir: str,
) -> None:
    filename = variables.get("benchmarks_yaml_file")
    if not filename:
        filename = default_filename
        variables["benchmarks_yaml_file"] = filename

    content = load_benchmark_config(filename, benchmarks_dir)
    variables["benchmarks_yaml_escaped"] = (
        content.replace("\\", "\\\\").replace("'", "'\\''").replace("\n", "\\n")
    )
