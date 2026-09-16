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


"""Kind Integration Cluster Provisioner for Cluster Health Scanner (CHS)."""

from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from typing import Any
import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


MANIFESTS_DIR = Path(__file__).parent / "manifests"
CONFIGS_DIR = Path(__file__).parent / "configs"

DEFAULT_JOBSET_VERSION = "v0.6.0"
DEFAULT_KUEUE_VERSION = "v0.14.5"
KUBE_RBAC_PROXY_IMAGE = "quay.io/brancz/kube-rbac-proxy:v0.13.1"
DEFAULT_CLUSTER_NAME = "chs-tpu-v6e-cluster"
DEFAULT_ZONE = "us-east5-c"
DEFAULT_PROJECT = "chs-test-project"


def get_kind_bin() -> str:
    """Resolves and validates location of the Kind CLI binary."""
    kind_bin = shutil.which("kind") or str(
        Path.home() / ".local" / "bin" / "kind"
    )
    if not Path(kind_bin).exists():
        raise RuntimeError(
            f"Kind binary not found at '{kind_bin}'. Is Kind installed?\n"
            "Please refer to https://kind.sigs.k8s.io/docs/user/quick-start/ "
            "for installation."
        )
    return kind_bin


def resolve_config_path(input_path: str) -> Path:
    """Resolves config path from path, CONFIGS_DIR, or adding .yaml."""
    p = Path(input_path)
    if p.exists() and p.is_file():
        return p

    candidate = CONFIGS_DIR / input_path
    if candidate.exists() and candidate.is_file():
        return candidate

    if not input_path.endswith(".yaml"):
        candidate_yaml = CONFIGS_DIR / f"{input_path}.yaml"
        if candidate_yaml.exists() and candidate_yaml.is_file():
            return candidate_yaml


def resolve_node_scenario(
    scenarios: list[dict[str, Any]],
    nodepool_name: str,
    node_index: int,
    test_name: str | None = None,
) -> dict[str, Any] | None:
    """Resolves test scenario matching a node with pool-wide fallback.

    Args:
        scenarios: List of test_scenario dicts from blueprint config.
        nodepool_name: Name of target nodepool.
        node_index: 0-based index of worker node within the nodepool.
        test_name: Optional test name filter (e.g. 'tpu-hbm').

    Returns:
        Matching scenario dict or None if no matching scenario exists.
    """
    pool_wide_fallback = None

    for s in scenarios:
        if not isinstance(s, dict):
            continue
        if s.get("nodepool") != nodepool_name:
            continue
        if test_name and s.get("test") != test_name:
            continue

        s_index = s.get("node_index")

        # Tier 1: Exact node_index match (e.g. node_index: 0)
        if s_index == node_index:
            return s

        # Tier 2: Pool-wide fallback (node_index omitted, None, or "all")
        if s_index is None or s_index == "all":
            pool_wide_fallback = s

    return pool_wide_fallback


# ==============================================================================
# Helper Functions
# ==============================================================================


def run(
    cmd: list[str] | str,
    input_str: str | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Executes a subprocess command with output logging and error handling.

    Args:
        cmd: Subprocess command to execute (list or shell string).
        input_str: Optional stdin content to pass to command.
        check: Whether to raise CalledProcessError on non-zero exit code.

    Returns:
        CompletedProcess instance containing stdout and stderr.

    Raises:
        subprocess.CalledProcessError: If check=True and process returns
            non-zero exit.
    """
    cmd_str = shlex.join(cmd) if isinstance(cmd, list) else cmd
    logger.info(f"[EXEC] {cmd_str}")

    res = subprocess.run(
        cmd,
        input=input_str,
        text=True,
        capture_output=True,
        check=False,
        shell=isinstance(cmd, str),
    )

    if check and res.returncode != 0:
        logger.error(
            f"[ERROR] Command failed: {cmd_str}\n"
            f"STDOUT:\n{res.stdout}\n"
            f"STDERR:\n{res.stderr}"
        )
        raise subprocess.CalledProcessError(
            res.returncode, cmd, output=res.stdout, stderr=res.stderr
        )

    return res


def sanitize_k8s_name(name: str) -> str:
    """Sanitizes an input string into a valid RFC 1123 Kubernetes name."""
    sanitized = re.sub(r"[^a-z0-9.-]", "-", name.lower()).strip("-.")
    return (sanitized or "default-name")[:63]


def _is_disabled(config: dict[str, Any]) -> bool:
    """Checks whether a dependency component is marked as disabled."""
    if not isinstance(config, dict):
        return False
    state = str(config.get("state", "")).upper()
    return (
        state in ("DISABLED", "MISSING", "MISSING_JOBSET", "MISSING_KUEUE")
        or config.get("enabled") is False
    )


def load_manifest(filename: str, **kwargs: Any) -> str:
    """Loads a YAML manifest template and substitutes placeholder variables."""
    manifest_path = MANIFESTS_DIR / filename
    if not manifest_path.exists() and Path(filename).exists():
        manifest_path = Path(filename)

    content = manifest_path.read_text()
    for key, value in kwargs.items():
        content = content.replace(f"{{{key}}}", str(value))
    return content


def create_kind_cluster(
    cluster_name: str = DEFAULT_CLUSTER_NAME,
    worker_count: int = 2,
    project: str = DEFAULT_PROJECT,
    zone: str = DEFAULT_ZONE,
) -> None:
    """Ensures specified local Kind cluster exists with worker nodes."""
    kind_name = sanitize_k8s_name(cluster_name or "chs-tpu-v6e-cluster")
    res = run([get_kind_bin(), "get", "clusters"], check=False)
    existing_clusters = res.stdout.splitlines() if res.returncode == 0 else []

    if kind_name in existing_clusters:
        run(
            [get_kind_bin(), "export", "kubeconfig", "--name", kind_name],
            check=False,
        )
        run(
            ["kubectl", "config", "use-context", f"kind-{kind_name}"],
            check=False,
        )
        health_check = run(["kubectl", "cluster-info"], check=False)
        existing_workers = (
            len(get_worker_nodes()) if health_check.returncode == 0 else 0
        )

        if health_check.returncode == 0 and existing_workers >= worker_count:
            logger.info(
                f"[INFO] Reusing healthy Kind cluster '{kind_name}' with"
                f" {existing_workers} worker node(s)."
            )
            return

        logger.info(
            f"[INFO] Existing cluster '{kind_name}' is unresponsive or has"
            f" insufficient workers ({existing_workers} < {worker_count})."
            " Recreating cluster..."
        )
        delete_kind_cluster(cluster_name)

    logger.info(
        f"[INFO] Creating Kind cluster '{kind_name}' with"
        f" {worker_count} worker nodes..."
    )
    config_content = (
        "kind: Cluster\n"
        "apiVersion: kind.x-k8s.io/v1alpha4\n"
        "networking:\n"
        "  apiServerPort: 26443\n"
        "nodes:\n"
        "- role: control-plane\n"
        "  kubeadmConfigPatches:\n"
        "  - |\n"
        "    kind: InitConfiguration\n"
        "    nodeRegistration:\n"
        "      kubeletExtraArgs:\n"
        f'        provider-id: "gce://{project}/{zone}/{kind_name}-control-plane"\n'
    )
    for i in range(max(1, worker_count)):
        node_name = f"node-{i + 1}"
        config_content += (
            "- role: worker\n"
            "  kubeadmConfigPatches:\n"
            "  - |\n"
            "    kind: JoinConfiguration\n"
            "    nodeRegistration:\n"
            f"      name: {node_name}\n"
            "      kubeletExtraArgs:\n"
            f'        provider-id: "gce://{project}/{zone}/{node_name}"\n'
        )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as tmp:
        tmp.write(config_content)
        tmp_path = tmp.name

    try:
        run(
            [
                get_kind_bin(),
                "create",
                "cluster",
                "--name",
                kind_name,
                "--config",
                tmp_path,
            ],
            check=True,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    run(
        [get_kind_bin(), "export", "kubeconfig", "--name", kind_name],
        check=True,
    )
    run(["kubectl", "config", "use-context", f"kind-{kind_name}"], check=True)


def delete_kind_cluster(cluster_name: str = "chs-tpu-v6e-cluster") -> None:
    """Tears down and deletes the specified local Kind cluster."""
    kind_name = sanitize_k8s_name(cluster_name or "chs-tpu-v6e-cluster")
    logger.info(f"[INFO] Deleting Kind cluster '{kind_name}'...")
    run([get_kind_bin(), "delete", "cluster", "--name", kind_name], check=False)


def apply_crd_stubs(dependencies: dict[str, Any] | None = None) -> None:
    """Reconciles custom resource definition (CRD) stubs for integration."""
    logger.info("[INFO] Reconciling CRD stubs...")
    deps = (
        dependencies
        if isinstance(dependencies, dict)
        else {"gke_multinetwork": {}}
    )

    builtin_crds = {
        "gke_multinetwork": (
            "networks.networking.gke.io",
            "crd_gke_multinetwork.yaml",
        ),
    }

    all_keys = set(builtin_crds.keys()) | set(deps.keys())
    for key in all_keys:
        cfg = deps.get(key, {})
        if not isinstance(cfg, dict):
            cfg = {}

        if key in builtin_crds:
            crd_name = builtin_crds[key][0]
            manifest_content = load_manifest(builtin_crds[key][1])
        else:
            crd_name = cfg.get("crd_name") or cfg.get("crd")
            manifest_file = cfg.get("manifest") or cfg.get("crd_manifest")
            if isinstance(manifest_file, str) and manifest_file.endswith(
                ".yaml"
            ):
                manifest_content = load_manifest(manifest_file)
            else:
                manifest_content = (
                    manifest_file if isinstance(manifest_file, str) else None
                )

        if _is_disabled(cfg) and crd_name:
            run(
                ["kubectl", "delete", "crd", crd_name, "--ignore-not-found"],
                check=False,
            )
        elif manifest_content:
            run(["kubectl", "apply", "-f", "-"], input_str=manifest_content)


OFFICIAL_MANIFEST_URLS = {
    "jobset": (
        "https://github.com/kubernetes-sigs/jobset/releases/download/"
        "{version}/manifests.yaml"
    ),
    "kueue": (
        "https://github.com/kubernetes-sigs/kueue/releases/download/"
        "{version}/manifests.yaml"
    ),
}

OFFICIAL_IMAGES = {
    "jobset": "registry.k8s.io/jobset/jobset:{version}",
    "kueue": "registry.k8s.io/kueue/kueue:{version}",
}


def _is_deployment_ready(dep_name: str) -> bool:
    ns = f"{dep_name}-system"
    dep_obj = f"deployment/{dep_name}-controller-manager"
    res = run(
        [
            "kubectl",
            "get",
            dep_obj,
            "-n",
            ns,
            "-o",
            "jsonpath={.status.readyReplicas}",
        ],
        check=False,
    )
    if (
        res.returncode == 0
        and res.stdout.strip().isdigit()
        and int(res.stdout.strip()) > 0
    ):
        return True
    return False


def _load_image(img: str, kind_name: str) -> None:
    run(
        [get_kind_bin(), "load", "docker-image", img, "--name", kind_name],
        check=False,
    )


def _crd_exists(crd_fragment: str) -> bool:
    """Checks whether any CRD matching crd_fragment exists on the cluster."""
    res = run(["kubectl", "get", "crd", "-o", "name"], check=False)
    if res.returncode == 0:
        return any(
            crd_fragment.lower() in line.lower()
            for line in res.stdout.splitlines()
        )
    return False


def _uninstall_preflight_deployment(dep_name: str) -> None:
    """Uninstalls a preflight controller deployment, CRDs, and webhooks."""
    logger.info(f"[INFO] Removing preflight deployment '{dep_name}'...")
    run(
        [
            "kubectl",
            "delete",
            "mutatingwebhookconfiguration",
            f"{dep_name}-mutating-webhook-configuration",
            "--ignore-not-found",
        ],
        check=False,
    )
    run(
        [
            "kubectl",
            "delete",
            "validatingwebhookconfiguration",
            f"{dep_name}-validating-webhook-configuration",
            "--ignore-not-found",
        ],
        check=False,
    )
    run(
        [
            "kubectl",
            "delete",
            "deployment",
            f"{dep_name}-controller-manager",
            "-n",
            f"{dep_name}-system",
            "--ignore-not-found",
        ],
        check=False,
    )
    if dep_name == "jobset":
        run(
            [
                "kubectl",
                "delete",
                "crd",
                "jobsets.jobset.x-k8s.io",
                "--ignore-not-found",
            ],
            check=False,
        )
    elif dep_name == "kueue":
        run(
            [
                "kubectl",
                "delete",
                "crd",
                "-l",
                "app.kubernetes.io/name=kueue",
                "--ignore-not-found",
            ],
            check=False,
        )
    run(
        [
            "kubectl",
            "delete",
            "namespace",
            f"{dep_name}-system",
            "--ignore-not-found",
            "--timeout=30s",
        ],
        check=False,
    )


def apply_preflight_deployments(
    dependencies: dict[str, Any] | None = None,
    cluster_name: str = "chs-tpu-v6e-cluster",
) -> None:
    """Provisions preflight controller deployments (JobSet & Kueue)."""
    logger.info("[INFO] Reconciling preflight deployments...")
    default_deps = {
        "jobset": {"version": DEFAULT_JOBSET_VERSION},
        "kueue": {"version": DEFAULT_KUEUE_VERSION},
    }
    deps = dependencies if isinstance(dependencies, dict) else default_deps

    kind_name = sanitize_k8s_name(cluster_name)
    images_to_load: set[str] = set()
    deployments_to_apply: list[tuple[str, str, str]] = []

    for dep_name, info in deps.items():
        if not isinstance(info, dict):
            continue

        if _is_disabled(info):
            if dep_name in OFFICIAL_MANIFEST_URLS:
                if _is_deployment_ready(dep_name) or _crd_exists(dep_name):
                    _uninstall_preflight_deployment(dep_name)
            continue

        if dep_name in OFFICIAL_MANIFEST_URLS:
            if _is_deployment_ready(dep_name):
                logger.info(
                    f"[INFO] Preflight deployment '{dep_name}' is already"
                    " running and ready."
                )
                continue

            version = info.get(
                "version",
                DEFAULT_JOBSET_VERSION
                if dep_name == "jobset"
                else DEFAULT_KUEUE_VERSION,
            )
            manifest_url = OFFICIAL_MANIFEST_URLS[dep_name].format(
                version=version
            )
            img = OFFICIAL_IMAGES[dep_name].format(version=version)

            images_to_load.add(img)
            images_to_load.add(KUBE_RBAC_PROXY_IMAGE)
            deployments_to_apply.append((dep_name, version, manifest_url))

    if not deployments_to_apply and not images_to_load:
        return

    if images_to_load:
        logger.info(
            f"[INFO] Side-loading {len(images_to_load)} docker image(s) in"
            " parallel..."
        )
        with ThreadPoolExecutor(max_workers=len(images_to_load)) as executor:
            list(
                executor.map(
                    lambda img: _load_image(img, kind_name), images_to_load
                )
            )

    def _deploy_and_wait(dep_info: tuple[str, str, str]) -> None:
        dep_name, version, manifest_url = dep_info
        logger.info(f"[INFO] Deploying {dep_name} ({version})...")
        run(
            [
                "kubectl",
                "apply",
                "--server-side",
                "--force-conflicts",
                "-f",
                manifest_url,
            ],
            check=True,
        )
        ns = f"{dep_name}-system"
        dep_obj = f"deployment/{dep_name}-controller-manager"
        run(
            [
                "kubectl",
                "set",
                "image",
                dep_obj,
                f"kube-rbac-proxy={KUBE_RBAC_PROXY_IMAGE}",
                "-n",
                ns,
            ],
            check=False,
        )
        run(
            [
                "kubectl",
                "rollout",
                "status",
                dep_obj,
                "-n",
                ns,
                "--timeout=120s",
            ],
            check=True,
        )

    if deployments_to_apply:
        for dep_info in deployments_to_apply:
            _deploy_and_wait(dep_info)


def get_worker_nodes() -> list[str]:
    """Retrieves list of Kind worker nodes."""
    res = run(
        [
            "kubectl",
            "get",
            "nodes",
            "-l",
            "!node-role.kubernetes.io/control-plane",
            "-o",
            "jsonpath={.items[*].metadata.name}",
        ],
        check=False,
    )
    if res.returncode == 0 and res.stdout.strip():
        return sorted(res.stdout.strip().split())
    return []


def wait_for_worker_nodes(
    expected_workers: int = 1, timeout: int = 60
) -> list[str]:
    """Polls K8s API until expected worker nodes register."""
    logger.info(
        f"[INFO] Waiting up to {timeout}s for {expected_workers} worker node(s)"
        " to register..."
    )
    start_time = time.time()
    while time.time() - start_time < timeout:
        worker_nodes = get_worker_nodes()
        if len(worker_nodes) >= expected_workers:
            logger.info(
                f"[INFO] Found {len(worker_nodes)} worker node(s):"
                f" {', '.join(worker_nodes)}"
            )
            return worker_nodes
        time.sleep(2)

    worker_nodes = get_worker_nodes()
    if not worker_nodes:
        logger.warning(
            f"[WARNING] Timed out waiting for worker nodes after {timeout}s."
        )
    return worker_nodes


def cleanup_nodes() -> None:
    """Resets cordoning, deletes busy pods, and strips non-system labels."""
    logger.info("[INFO] Cleaning up worker node states and integration pods...")
    worker_nodes = get_worker_nodes()
    if not worker_nodes:
        return

    system_prefixes = (
        "kubernetes.io/",
        "node-role.kubernetes.io/",
        "kind.x-k8s.io/",
    )

    # Batch uncordon all nodes in a single command
    run(["kubectl", "uncordon"] + worker_nodes, check=False)

    # Fetch all node metadata in a single JSON request
    res = run(["kubectl", "get", "nodes", "-o", "json"], check=False)
    if res.returncode == 0 and res.stdout:
        try:
            nodes_data = json.loads(res.stdout)
            items = nodes_data.get("items", [])
            for item in items:
                node_name = item.get("metadata", {}).get("name")
                if node_name not in worker_nodes:
                    continue
                labels = item.get("metadata", {}).get("labels", {})
                to_strip = [
                    f"{k}-"
                    for k in labels
                    if not any(k.startswith(p) for p in system_prefixes)
                ]
                if to_strip:
                    run(
                        ["kubectl", "label", "node", node_name] + to_strip,
                        check=False,
                    )
        except Exception as e:
            logger.warning(
                f"[WARNING] Failed to parse node metadata during cleanup: {e}"
            )
    elif res.returncode != 0:
        logger.warning(
            "[WARNING] Failed to fetch nodes for label cleanup:"
            f" {res.stderr.strip()}"
        )

    run(
        [
            "kubectl",
            "delete",
            "pods",
            "-A",
            "-l",
            "chs-integration-stub=busy-pod",
            "--ignore-not-found",
        ],
        check=False,
    )


def _build_node_labels(
    pool_config: dict[str, Any], default_zone: str, cluster_name: str
) -> dict[str, str]:
    """Constructs GKE hardware and topology labels for worker nodes."""
    pool_name = pool_config.get("name", "pool-01")
    accelerator = pool_config.get("accelerator", "")
    topology = pool_config.get("topology", "")
    zone = pool_config.get("zone", default_zone)
    topology_block = pool_config.get("gce_topology_block", "default-block")
    topology_mode = pool_config.get(
        "gke_accelerator_topology_mode"
    ) or pool_config.get("accelerator_topology_mode")

    labels = {
        "cloud.google.com/gke-nodepool": pool_name,
        "chs-integration-nodepool": pool_name,
        "container.googleapis.com/cluster_name": cluster_name,
        "kubernetes.io/os": "linux",
        "kubernetes.io/arch": "amd64",
        "topology.gke.io/zone": zone,
    }

    if accelerator:
        accel_lower = accelerator.lower()
        gpu_ids = ["nvidia", "gpu", "h100", "a100", "l4", "v100"]
        if "tpu" in accel_lower:
            key = "cloud.google.com/gke-tpu-accelerator"
        elif any(g in accel_lower for g in gpu_ids):
            key = "cloud.google.com/gke-gpu"
        else:
            key = "cloud.google.com/gke-accelerator"
        labels[key] = accelerator

    if topology:
        labels["cloud.google.com/gke-tpu-topology"] = topology

    if topology_block:
        labels["cloud.google.com/gce-topology-block"] = topology_block

    partition_topology = pool_config.get("partition_topology")
    if topology_mode or partition_topology:
        if topology_mode:
            labels["cloud.google.com/gke-accelerator-topology-mode"] = (
                topology_mode
            )
        eff_topo = partition_topology or topology or "4x4x4"
        labels[f"cloud.google.com/gke-tpu-partition-{eff_topo}-state"] = (
            pool_config.get("partition_state", "HEALTHY")
        )
        labels[f"cloud.google.com/gke-tpu-partition-{eff_topo}-id"] = (
            pool_config.get("partition_id", f"{pool_name}-partition")
        )

    machine_type = pool_config.get("machine_type") or pool_config.get(
        "instance_type"
    )
    if machine_type:
        labels["node.kubernetes.io/instance-type"] = machine_type
        labels["beta.kubernetes.io/instance-type"] = machine_type

    custom_labels = pool_config.get("labels")
    if isinstance(custom_labels, dict):
        labels.update(custom_labels)

    return labels


def configure_nodes(
    nodepools: list[dict[str, Any]] | None = None,
    default_zone: str = "us-east5-c",
    cluster_name: str = "chs-tpu-v6e-cluster",
    expected_workers: int = 1,
    worker_nodes: list[str] | None = None,
) -> None:
    """Configures Kind worker nodes with GKE labels and cordoning."""
    cleanup_nodes()

    if not nodepools:
        logger.info(
            "[INFO] No nodepools defined in config. Skipping node setup."
        )
        return

    if worker_nodes is None:
        worker_nodes = wait_for_worker_nodes(expected_workers=expected_workers)
    if len(worker_nodes) < expected_workers:
        raise ValueError(
            f"Kind cluster has only {len(worker_nodes)} worker node(s), but"
            f" blueprint requested {expected_workers} worker node(s) across all"
            " nodepools."
        )

    logger.info(f"[INFO] Configuring {len(worker_nodes)} Kind worker nodes...")

    worker_idx = 0
    nodes_to_cordon = []

    for pool in nodepools:
        if not isinstance(pool, dict):
            continue

        health_config = (
            pool.get("health") if isinstance(pool.get("health"), dict) else {}
        )
        count = max(1, pool.get("count", 1))

        ready_cnt = max(0, health_config.get("ready_count", count))
        unsched_cnt = max(0, health_config.get("unschedulable_count", 0))
        busy_cnt = max(0, health_config.get("busy_count", 0))

        labels = _build_node_labels(pool, default_zone, cluster_name)
        labels_args = [f"{k}={v}" for k, v in labels.items()]

        pool_nodes = []
        for i in range(count):
            node_name = worker_nodes[worker_idx]
            worker_idx += 1
            pool_nodes.append(node_name)

            is_unsched = (i >= ready_cnt) or (i >= (count - unsched_cnt))
            if is_unsched:
                nodes_to_cordon.append(node_name)

            is_busy = i < busy_cnt
            if is_busy:
                pod_name = f"busy-pod-{node_name}"
                namespace = (
                    pool.get("busy_pod_namespace")
                    or health_config.get("busy_pod_namespace")
                    or "default"
                )
                if namespace != "default":
                    run(
                        ["kubectl", "create", "namespace", namespace],
                        check=False,
                    )

                logger.info(
                    f"[INFO] Launching busy pod '{pod_name}' on"
                    f" '{node_name}'..."
                )
                overrides = json.dumps({"spec": {"nodeName": node_name}})
                run(
                    [
                        "kubectl",
                        "run",
                        pod_name,
                        "--image=alpine",
                        "--restart=Never",
                        "-n",
                        namespace,
                        "-l",
                        "chs-integration-stub=busy-pod",
                        f"--overrides={overrides}",
                        "--",
                        "sleep",
                        "3600",
                    ],
                    check=False,
                )

        if pool_nodes and labels_args:
            run(
                ["kubectl", "label", "node"]
                + pool_nodes
                + labels_args
                + ["--overwrite"],
                check=True,
            )

    for node_name in nodes_to_cordon:
        logger.info(f"[INFO] Cordoning node '{node_name}'...")
        run(["kubectl", "cordon", node_name], check=True)


def main() -> None:
    """Main orchestrator for Kind cluster turnup and teardown."""
    parser = argparse.ArgumentParser(
        description=(
            "Kind Integration Cluster Manager for Cluster Health Scanner (CHS)"
        )
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="tpu/v6e/standard_2x4",
        help=(
            "Path or shorthand identifier for YAML cluster blueprint"
            " (e.g. tpu/v6e/standard_2x4)."
        ),
    )
    parser.add_argument(
        "--delete",
        "--down",
        action="store_true",
        help="Tear down and delete the specified Kind cluster.",
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List available cluster blueprint configs.",
    )
    args = parser.parse_args()

    if args.list_configs:
        logger.info("Available Cluster Blueprint Configurations:")
        for yaml_path in sorted(CONFIGS_DIR.glob("**/*.yaml")):
            rel_path = yaml_path.relative_to(CONFIGS_DIR)
            logger.info(f"  - {rel_path}")
        return

    config_path = resolve_config_path(args.config)
    logger.info(f"[INFO] Reading config from: {config_path}")

    config: dict[str, Any] = {}
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text()) or {}
    else:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    cluster_name = config.get("cluster_name", "chs-tpu-v6e-cluster")

    if args.delete:
        delete_kind_cluster(cluster_name)
        logger.info(
            f"[SUCCESS] Kind Cluster '{cluster_name}' has been deleted."
        )
        return

    default_zone = config.get("default_zone", "us-east5-c")
    project = config.get("project", "chs-test-project")
    dependencies = config.get("dependencies")
    nodepools = config.get("nodepools", [])

    total_workers = (
        sum(
            pool.get("count", 1) for pool in nodepools if isinstance(pool, dict)
        )
        if nodepools
        else 2
    )

    create_kind_cluster(
        cluster_name,
        worker_count=max(1, total_workers),
        project=project,
        zone=default_zone,
    )
    worker_nodes = wait_for_worker_nodes(expected_workers=max(1, total_workers))
    apply_crd_stubs(dependencies)
    apply_preflight_deployments(dependencies, cluster_name=cluster_name)
    configure_nodes(
        nodepools,
        default_zone=default_zone,
        cluster_name=cluster_name,
        expected_workers=max(1, total_workers),
        worker_nodes=worker_nodes,
    )

    logger.info(
        "\n=======================================================\n"
        f" SUCCESS: Kind Cluster '{cluster_name}' is Ready!\n"
        "=======================================================\n"
    )
    run(
        [
            "kubectl",
            "get",
            "nodes",
            "-L",
            "chs-integration-nodepool,"
            "cloud.google.com/gke-nodepool,"
            "cloud.google.com/gke-tpu-accelerator,"
            "cloud.google.com/gke-tpu-topology",
        ],
        check=False,
    )


if __name__ == "__main__":
    main()
