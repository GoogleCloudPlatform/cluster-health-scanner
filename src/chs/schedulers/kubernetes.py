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

"""Kubernetes scheduler implementation for Cluster Health Scanner."""

import importlib.resources as pkg_resources
import json
import time
import logging
import os
import re
import shutil
import tempfile
import threading
import yaml
import random


import subprocess
from typing import Any
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass, field

from chs.schedulers.base import BaseScheduler
from chs.utils.kueue import KueueManager
from chs.exceptions import (
    ChsSchedulerException,
    ChsValidationError,
    KueueException,
    NoActiveJobsException,
    format_subprocess_error,
)
from chs.utils.lifecycle import JobStatus
from chs.preflights.kubernetes_preflight import KubernetesPreflightChecker
from chs import resources
from chs.constants import (
    CHS_NAMESPACE,
    DELETE_WAIT_TIMEOUT_SEC,
    KUBECTL_TIMEOUT_SEC,
    NAMESPACE_DELETE_TIMEOUT_SEC,
    SUPPORTED_DYNAMIC_SLICE_TOPOLOGY,
)

_BUSY_NODES_BATCH_SIZE = 32
_BUSY_NODES_MAX_RETRIES = 3
_MAX_NODES_UNDER_TEST = 500

_LOG_FETCH_MAX_CONCURRENCY = 32
_LOG_TEMP_PREFIX = ".chs-log-"
_LOG_FETCH_EXECUTOR: ThreadPoolExecutor | None = None
_LOG_FETCH_EXECUTOR_LOCK = threading.Lock()


def _get_log_fetch_executor() -> ThreadPoolExecutor:
    """Returns the shared executor for pod log fetches."""
    global _LOG_FETCH_EXECUTOR
    with _LOG_FETCH_EXECUTOR_LOCK:
        if _LOG_FETCH_EXECUTOR is None:
            _LOG_FETCH_EXECUTOR = ThreadPoolExecutor(
                max_workers=_LOG_FETCH_MAX_CONCURRENCY,
                thread_name_prefix="chs-log-fetch",
            )
        return _LOG_FETCH_EXECUTOR


@dataclass
class TpuSliceData:
    pools: dict[str, list[str]] = field(
        default_factory=lambda: defaultdict(list)
    )
    node_to_pool: dict[str, str] = field(default_factory=dict)
    pool_to_topology: dict[str, str] = field(default_factory=dict)
    dynamic_slice_partitions: dict[str, str] = field(default_factory=dict)
    pool_to_accelerator: dict[str, str] = field(default_factory=dict)
    all_tpu_nodes: set[str] = field(default_factory=set)


def boolean_representer(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:bool", str(data))


yaml.Dumper.add_representer(bool, boolean_representer)

logger = logging.getLogger(__name__)


class KubernetesScheduler(BaseScheduler):
    """Kubernetes scheduler implementation."""

    def __init__(
        self,
        nodes: str | None = None,
        nodelists: str | None = None,
        partitions: str | None = None,
        nodepools: str | None = None,
    ):
        super().__init__()

        if nodelists or partitions:
            raise ChsValidationError(
                "The --nodelists and --partitions flags are Slurm-specific and "
                "cannot be used with the Kubernetes scheduler. Please use "
                "--nodes or --nodepools instead."
            )

        self.nodes_arg = nodes
        self.nodepools_arg = nodepools
        self._resolved_nodes: list[str] | None = None
        self._resolved_raw_nodes: list[str] | None = None
        self._resolved_node_resources: list[dict] | None = None
        self._cluster_name: str | None = None
        self._tpu_slice_data: TpuSliceData | None = None
        self.workspace_config_dir: str | None = None
        self.kueue_supported: bool = False
        self._jobset_labels_yaml: str = ""
        self._suspend_jobset: str = "false"

    def _resolve_project_id(self, node: str) -> str:
        """Resolves GCP project ID from Kubernetes node providerID."""
        try:
            cached_node = next(
                (
                    n
                    for n in (self._resolved_node_resources or [])
                    if n.get("metadata", {}).get("name") == node
                ),
                {},
            )
            provider_id = cached_node.get("spec", {}).get("providerID", "")
            if not provider_id.strip():
                provider_id = self._run_kubectl(
                    ["get", "node", node, "-o", "jsonpath={.spec.providerID}"]
                )
            provider_id = provider_id.strip()
            # providerID format: gce://project-id/zone/instance-name
            match = re.match(r"^gce://([^/]+)/([^/]+)/([^/]+)$", provider_id)
            if not match:
                raise ValueError(f"Unexpected providerID format: {provider_id}")
            return match.group(1)
        except Exception as e:
            raise ChsSchedulerException(
                f"Failed to resolve project ID from node '{node}': {e}"
            ) from e

    def detect_nodes_hardware(self, nodes: list[str]) -> list[str]:
        """Detects machine types and maps instance IDs for the given nodes."""
        if not nodes:
            return []

        try:
            node_objects = self._get_node_objects(nodes)
        except (
            ChsSchedulerException,
            subprocess.SubprocessError,
            OSError,
        ) as e:
            logger.debug(
                "Could not read node objects to detect hardware; querying "
                "the GCE API instead. Error: %s",
                format_subprocess_error(e),
            )
            return super().detect_nodes_hardware(nodes)

        objects_by_name = {
            obj.get("metadata", {}).get("name"): obj for obj in node_objects
        }
        label_types = {}
        gce_nodes = []
        for node in nodes:
            metadata = objects_by_name.get(node, {}).get("metadata", {})
            labels = metadata.get("labels", {})
            annotations = metadata.get("annotations", {})
            machine_type = labels.get(
                "node.kubernetes.io/instance-type"
            ) or labels.get("beta.kubernetes.io/instance-type")
            instance_id = annotations.get(
                "container.googleapis.com/instance_id"
            )
            if machine_type:
                label_types[node] = resources.normalize_machine_type(
                    machine_type
                )
            if machine_type and instance_id:
                self.node_to_instance_id[node] = str(instance_id)
            else:
                gce_nodes.append(node)

        unique_types = set(label_types.values())
        if gce_nodes:
            unique_types.update(
                t
                for t in super().detect_nodes_hardware(gce_nodes)
                if t != "unknown"
            )
        for node in gce_nodes:
            if node in self.node_to_instance_id:
                continue
            if node in label_types:
                logger.info(
                    "Resolved machine type '%s' for node '%s' from "
                    "Kubernetes labels.",
                    label_types[node],
                    node,
                )
                self.node_to_instance_id[node] = node
            else:
                unique_types.add("unknown")

        return sorted(unique_types)

    def run_preflight_checks(
        self,
        tests: list[dict[str, Any]] | None = None,
        workspace_name: str | None = None,
        run_id: str | None = None,
    ) -> list[dict[str, Any]] | None:
        """Checks for JobSet and Kueue dependencies on Kubernetes."""
        preflight_checker = KubernetesPreflightChecker(self)
        filtered_tests, kueue_supported = preflight_checker.check(tests)
        self.kueue_supported = kueue_supported
        return filtered_tests

    def setup_scheduler_resources(
        self,
        workspace_config_dir: str,
        tests: list[dict[str, Any]] | None = None,
        machine_type: str = "",
        workspace_name: str | None = None,
    ) -> list[dict[str, Any]] | None:
        """Sets up scheduler-specific resources like Kueue configurations."""
        self.workspace_config_dir = workspace_config_dir
        os.makedirs(workspace_config_dir, exist_ok=True)

        if self.kueue_supported:
            nodes = self.get_nodes_under_test() or []
            dynamic_slicing = bool(
                self._tpu_slice_data
                and any(
                    self._tpu_slice_data.node_to_pool.get(n)
                    in self._tpu_slice_data.dynamic_slice_partitions
                    for n in nodes
                )
            )
            kueue_mgr = KueueManager(self)
            self._jobset_labels_yaml = kueue_mgr.generate_configs(
                workspace_config_dir,
                machine_type,
                nodes,
                dynamic_slicing=dynamic_slicing,
            )
            self._suspend_jobset = "true"

        return tests

    @property
    def name(self) -> str:
        return "kubernetes"

    def get_cluster_name(self) -> str | None:
        """Retrieves cluster name from cached node labels or kubectl context."""
        if self._cluster_name is None:
            self._cluster_name = self._lookup_cluster_name()
        return self._cluster_name

    def _lookup_cluster_name(self) -> str | None:
        """Looks up the cluster name from cached node labels or kubeconfig."""
        # 1. Try cached node labels
        try:
            for node in self._resolved_node_resources or []:
                labels = node.get("metadata", {}).get("labels", {})
                gke_cluster = labels.get(
                    "container.googleapis.com/cluster_name"
                )
                if gke_cluster:
                    return gke_cluster
        except Exception:
            pass

        # 2. Try current-context (local execution)
        try:
            val = self._run_kubectl(["config", "current-context"], timeout=5)
            if val:
                if val.startswith("gke_"):
                    parts = val.split("_")
                    if len(parts) >= 4:
                        return parts[-1]
                return val
        except Exception:
            pass
        return None

    def _ensure_namespace_exists(self) -> None:
        """Ensures the chs namespace exists before deploying resources."""

        try:
            self._run_kubectl(["get", "namespace", CHS_NAMESPACE])
        except subprocess.CalledProcessError:
            self._run_kubectl(["create", "namespace", CHS_NAMESPACE])

    def _delete_namespace(self) -> bool:
        """Deletes the chs namespace during cleanup."""
        try:
            self._run_kubectl(
                ["delete", "namespace", CHS_NAMESPACE, "--ignore-not-found"],
                timeout=NAMESPACE_DELETE_TIMEOUT_SEC,
            )
        except Exception as e:
            logger.warning(
                "Failed to delete Kubernetes namespace. This may "
                "leave orphaned resources. Error: %s",
                format_subprocess_error(e),
            )
            return False
        return True

    def _run_kubectl(
        self,
        args: list[str],
        timeout: int | None = KUBECTL_TIMEOUT_SEC,
        namespaced: bool = False,
    ) -> str:
        """Runs a kubectl command and returns filtered output."""

        cmd = ["kubectl"]
        if namespaced:
            cmd.extend(["-n", CHS_NAMESPACE])
        cmd.extend(args)

        result = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            check=True,
            timeout=timeout,
        )
        return result.stdout.strip()

    def _delete_job_secrets(self) -> None:
        """Deletes SSH secrets associated with CHS jobs statelessly."""
        try:
            output = self._run_kubectl(
                ["get", "secrets", "-o", "jsonpath={.items[*].metadata.name}"],
                namespaced=True,
            )
            secret_names = output.strip().split()
            secrets_to_delete = [
                s
                for s in secret_names
                if s.startswith("chs-") and s.endswith("-ssh")
            ]

            if secrets_to_delete:
                self._run_kubectl(
                    ["delete", "secrets"]
                    + secrets_to_delete
                    + [
                        "--ignore-not-found",
                        f"--timeout={DELETE_WAIT_TIMEOUT_SEC}s",
                    ],
                    namespaced=True,
                )
                logger.debug(
                    "Successfully deleted secrets: %s",
                    ", ".join(secrets_to_delete),
                )
            else:
                logger.debug("No CHS SSH secrets found to delete.")
        except Exception as e:
            logger.warning(
                "Failed to delete Kubernetes secrets. Error: %s",
                format_subprocess_error(e),
            )

    def _get_node_objects(self, node_names: list[str]) -> list[dict]:
        """Returns the node objects of node_names, in node_names order."""
        if not node_names:
            return []
        index = {
            n.get("metadata", {}).get("name"): n
            for n in self._fetch_raw_node_resources()
            if n.get("metadata", {}).get("name")
        }
        return [index[n] for n in node_names if n in index]

    def _group_nodes_by_slice(
        self, node_names: list[str]
    ) -> dict[str, list[str]]:
        """Groups TPU nodes into slices using placement policy or node pool."""
        if not node_names:
            return {}

        slices = defaultdict(list)
        try:
            for item in self._get_node_objects(node_names):
                node_name = item.get("metadata", {}).get("name")
                if not node_name:
                    continue
                labels = item.get("metadata", {}).get("labels", {})

                slice_id = labels.get("cloud.google.com/gke-nodepool")
                if not slice_id:
                    slice_id = "default-slice"

                slices[slice_id].append(node_name)
        except (ChsSchedulerException, subprocess.SubprocessError) as e:
            logger.warning(
                "Failed to execute 'kubectl' for nodes %s. Error: %s",
                node_names,
                format_subprocess_error(e),
            )
            slices["failed-nodes"].extend(node_names)

        return dict(slices)

    def _discover_networks(self) -> tuple[list[str], list[str]]:
        """Discovers all network names using kubectl and groups them by type.

        # TODO: This is GKE-specific (uses networks.networking.gke.io CRD).

        Returns:
            A tuple of (l3_networks, device_networks).
            L3 networks have 'default' first.
        """
        args = ["get", "networks.networking.gke.io", "-o", "json"]
        try:
            output = self._run_kubectl(args)
            if not output:
                return [], []
            data = json.loads(output)
            items = data.get("items", [])
            if not isinstance(items, list):
                logger.debug(
                    "Unexpected network discovery output format: "
                    "'items' is not a list"
                )
                return [], []

            networks = defaultdict(list)

            for item in items:
                metadata = item.get("metadata", {})
                spec = item.get("spec", {})
                name = metadata.get("name")
                net_type = spec.get("type")

                if name and net_type:
                    networks[net_type].append(name)

            l3_networks = networks["L3"]
            device_networks = networks["Device"]

            if "default" not in l3_networks:
                logger.warning(
                    "The default Layer 3 network configuration "
                    "was not found. Discovered "
                    "networks: %s. Ensure network "
                    "configuration is correct.",
                    ", ".join(l3_networks) or "none",
                )
                return [], []
            l3_networks.sort(key=lambda n: (n != "default", n))
            device_networks.sort()

            logger.debug(
                "Discovered L3 networks: %s, Device networks: %s",
                l3_networks,
                device_networks,
            )
            return l3_networks, device_networks

        except json.JSONDecodeError as e:
            logger.warning(
                "Failed to parse the network discovery JSON "
                "output from the cluster. Error: %s",
                e,
            )
            return [], []
        except Exception as e:
            logger.warning(
                "Encountered an unexpected error during Kubernetes "
                "network topology discovery. Error: %s",
                e,
            )
            return [], []

    def _is_node_busy(self, node_name: str) -> bool:
        """Helper method to determine if a specific node is busy."""
        args = [
            "get",
            "pods",
            "-A",
            "--field-selector",
            f"spec.nodeName={node_name},status.phase!=Succeeded,status.phase!=Failed",
            "-o",
            "custom-columns=NODE:.spec.nodeName,PHASE:.status.phase,"
            "CREQ:.spec.containers[*].resources.requests,CLIM:.spec.containers[*].resources.limits,"
            "IREQ:.spec.initContainers[*].resources.requests,ILIM:.spec.initContainers[*].resources.limits",
        ]

        for attempt in range(_BUSY_NODES_MAX_RETRIES):
            try:
                pod_output = self._run_kubectl(args)
                break
            except (
                subprocess.CalledProcessError,
                subprocess.TimeoutExpired,
            ) as e:
                if attempt == _BUSY_NODES_MAX_RETRIES - 1:
                    raise ChsSchedulerException(
                        "Failed to communicate with API server for "
                        f"node {node_name} "
                        f"after {_BUSY_NODES_MAX_RETRIES} attempts. Error: {e}"
                    ) from e

                # Exponential backoff with jitter to prevent thundering herd
                backoff = (2**attempt) + random.uniform(0, 1)
                time.sleep(backoff)

        for pod_output_row in pod_output.splitlines()[1:]:
            pod_columns = pod_output_row.split()
            if not pod_columns or len(pod_columns) < 2:
                continue
            resources_str = " ".join(pod_columns[2:]).lower()
            if "gpu" in resources_str or "tpu" in resources_str:
                return True
        return False

    def _get_busy_nodes(self, target_nodes: list[str]) -> set[str]:
        """Returns a set of nodes running pods that request accelerators."""
        if not target_nodes:
            raise ValueError(
                "target_nodes cannot be empty when checking for busy nodes."
            )

        if len(target_nodes) > _MAX_NODES_UNDER_TEST:
            raise ChsSchedulerException(
                f"Targeting {len(target_nodes)} nodes exceeds the maximum "
                f"allowed batch size of {_MAX_NODES_UNDER_TEST}. "
                f"Please narrow the test scope to less than "
                f"{_MAX_NODES_UNDER_TEST} nodes at a time."
            )

        busy_nodes = set()
        with ThreadPoolExecutor(max_workers=_BUSY_NODES_BATCH_SIZE) as executor:
            node_status_futures = {
                executor.submit(
                    self._is_node_busy,
                    target_node,
                ): target_node
                for target_node in target_nodes
            }

            try:
                for node_status_future in as_completed(node_status_futures):
                    target_node = node_status_futures[node_status_future]
                    if node_status_future.result():
                        busy_nodes.add(target_node)
            except Exception as e:
                logger.error(
                    "Failed to communicate with the Kubernetes API server "
                    "while determining node availability. Please verify "
                    "your cluster connection and credentials. "
                    "Details: %s",
                    e,
                )

                for future in node_status_futures:
                    future.cancel()

                executor.shutdown(wait=False, cancel_futures=True)
                raise

        return busy_nodes

    def _fetch_raw_node_resources(self) -> list[dict]:
        """Fetches raw node resource dictionaries from kubectl."""
        if self._resolved_node_resources is not None:
            return self._resolved_node_resources

        try:
            args = ["get", "nodes"]
            if self.nodepools_arg:
                pools = [
                    p.strip()
                    for p in self.nodepools_arg.split(",")
                    if p.strip()
                ]
                if pools:
                    pools_str = ",".join(pools)
                    args.extend(
                        [
                            "-l",
                            f"cloud.google.com/gke-nodepool in ({pools_str})",
                        ]
                    )
            else:
                args.extend(["-l", "cloud.google.com/gke-nodepool!=system"])
            args.extend(["-o", "json"])

            kubectl_nodes_output = self._run_kubectl(args)
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            raise ChsSchedulerException(
                f"Failed to get nodes to determine availability: {e}"
            ) from e

        try:
            nodes_manifest = json.loads(kubectl_nodes_output)
            if not isinstance(nodes_manifest, dict):
                node_resources: list[Any] = []
            elif nodes_manifest.get("kind") == "List":
                node_resources = nodes_manifest.get("items") or []
            else:
                node_resources = [nodes_manifest]

            resolved_resources = [
                n for n in node_resources if isinstance(n, dict)
            ]

            if self.nodepools_arg:
                requested_pools = {
                    p.strip()
                    for p in self.nodepools_arg.split(",")
                    if p.strip()
                }
                found_pools = set()
                for n in resolved_resources:
                    labels = (n.get("metadata") or {}).get("labels") or {}
                    pool = labels.get("cloud.google.com/gke-nodepool")
                    if pool:
                        found_pools.add(pool)
                missing_pools = requested_pools - found_pools
                if missing_pools:
                    missing_str = ", ".join(sorted(missing_pools))
                    raise ChsValidationError(
                        "No nodes were found for the following requested "
                        f"nodepool(s): {missing_str}"
                    )

            self._resolved_node_resources = resolved_resources
            return self._resolved_node_resources
        except json.JSONDecodeError as e:
            raise ChsSchedulerException(
                f"Failed to parse node JSON during availability check: {e}"
            ) from e

    def _classify_unhealthy_nodes(
        self,
        node_resources: list[dict],
    ) -> set[str]:
        """Filters node resources for NotReady or unschedulable nodes."""
        unhealthy_node_names = set()
        for node_resource in node_resources:
            node_name = (node_resource.get("metadata") or {}).get("name")
            if not node_name:
                continue

            if (node_resource.get("spec") or {}).get("unschedulable", False):
                unhealthy_node_names.add(node_name)
                continue

            conditions = (node_resource.get("status") or {}).get(
                "conditions"
            ) or []
            is_ready = any(
                isinstance(c, dict)
                and c.get("type") == "Ready"
                and c.get("status") == "True"
                for c in conditions
            )
            if not is_ready:
                unhealthy_node_names.add(node_name)

        return unhealthy_node_names

    def _get_not_ready_nodes(self) -> set[str]:
        """Returns nodes that are NotReady, Unknown, or Cordoned."""
        node_resources = self._fetch_raw_node_resources()
        return self._classify_unhealthy_nodes(node_resources)

    def _purge_unhealthy_pool_data(
        self,
        unhealthy_nodepool: str,
        tpu_nodepools: dict,
        node_to_nodepool: dict,
        pool_to_topology: dict,
        dynamic_slice_partitions: dict,
        pool_to_accelerator: dict,
    ) -> None:
        """Purges unhealthy pool data from dictionaries."""
        for node in tpu_nodepools.pop(unhealthy_nodepool, []):
            node_to_nodepool.pop(node, None)
        pool_to_topology.pop(unhealthy_nodepool, None)
        dynamic_slice_partitions.pop(unhealthy_nodepool, None)
        pool_to_accelerator.pop(unhealthy_nodepool, None)

    def _get_tpu_slices(self) -> TpuSliceData:
        """Returns TPU pools mapping and node-to-pool mapping."""
        if self._tpu_slice_data:
            return self._tpu_slice_data

        try:
            nodes = self._fetch_raw_node_resources()
            node_data = {"items": nodes}
        except Exception as e:
            raise ChsSchedulerException(
                f"Failed to get node labels for TPU topology: {e}"
            ) from e

        all_tpu_nodes = set()
        tpu_nodepools: dict[str, list[str]] = defaultdict(list)
        node_to_nodepool: dict[str, str] = {}
        pool_to_topology: dict[str, str] = {}
        dynamic_slice_partitions: dict[str, str] = {}
        pool_to_accelerator: dict[str, str] = {}

        unhealthy_pools = set()

        for node in node_data.get("items", []):
            node_name = node.get("metadata", {}).get("name")
            if not node_name:
                continue

            labels = node.get("metadata", {}).get("labels", {})
            tpu_type = labels.get("cloud.google.com/gke-tpu-accelerator")
            if not tpu_type:
                continue

            all_tpu_nodes.add(node_name)

            nodepool = labels.get("cloud.google.com/gke-nodepool")
            if not nodepool:
                continue

            accelerator_mode = labels.get(
                "cloud.google.com/gke-accelerator-topology-mode"
            )
            topology = labels.get("cloud.google.com/gke-tpu-topology")

            if accelerator_mode == "PROVISION_ONLY":
                state_label = (
                    "cloud.google.com/gke-tpu-partition-"
                    f"{SUPPORTED_DYNAMIC_SLICE_TOPOLOGY}-state"
                )
                matched_state = labels.get(state_label)

                if matched_state == "HEALTHY":
                    if nodepool not in unhealthy_pools:
                        tpu_nodepools[nodepool].append(node_name)
                        node_to_nodepool[node_name] = nodepool
                        pool_to_topology[nodepool] = (
                            SUPPORTED_DYNAMIC_SLICE_TOPOLOGY
                        )
                        id_label = (
                            "cloud.google.com/gke-tpu-partition-"
                            f"{SUPPORTED_DYNAMIC_SLICE_TOPOLOGY}-id"
                        )
                        partition_id = labels.get(id_label)
                        if partition_id:
                            dynamic_slice_partitions[nodepool] = partition_id
                        if tpu_type:
                            pool_to_accelerator[nodepool] = tpu_type
                else:
                    if nodepool not in unhealthy_pools:
                        if matched_state is None:
                            logger.warning(
                                "Skipping nodepool '%s' because the %s "
                                "topology is unsupported.",
                                nodepool,
                                SUPPORTED_DYNAMIC_SLICE_TOPOLOGY,
                            )
                        else:
                            logger.warning(
                                "Skipping nodepool '%s' because the %s "
                                "topology is currently unhealthy.",
                                nodepool,
                                SUPPORTED_DYNAMIC_SLICE_TOPOLOGY,
                            )
                    unhealthy_pools.add(nodepool)
                    self._purge_unhealthy_pool_data(
                        nodepool,
                        tpu_nodepools,
                        node_to_nodepool,
                        pool_to_topology,
                        dynamic_slice_partitions,
                        pool_to_accelerator,
                    )
            else:
                if not topology:
                    unhealthy_pools.add(nodepool)
                    self._purge_unhealthy_pool_data(
                        nodepool,
                        tpu_nodepools,
                        node_to_nodepool,
                        pool_to_topology,
                        dynamic_slice_partitions,
                        pool_to_accelerator,
                    )
                else:
                    if nodepool not in unhealthy_pools:
                        tpu_nodepools[nodepool].append(node_name)
                        node_to_nodepool[node_name] = nodepool
                        if topology and nodepool not in pool_to_topology:
                            pool_to_topology[nodepool] = topology

        if unhealthy_pools:
            logger.warning(
                "Excluded the following unhealthy dynamic TPU pools from "
                "diagnostics: %s. Please check the TPU node health.",
                ", ".join(unhealthy_pools),
            )

        self._tpu_slice_data = TpuSliceData(
            pools=tpu_nodepools,
            node_to_pool=node_to_nodepool,
            pool_to_topology=pool_to_topology,
            dynamic_slice_partitions=dynamic_slice_partitions,
            pool_to_accelerator=pool_to_accelerator,
            all_tpu_nodes=all_tpu_nodes,
        )
        return self._tpu_slice_data

    def resolve_raw_nodes(self) -> list[str]:
        """Resolves targeted nodes based on CLI arguments.

        Does not drop busy nodes.
        """
        if self._resolved_raw_nodes is not None:
            return self._resolved_raw_nodes

        raw_nodes = []
        if self.nodes_arg:
            raw_nodes = [
                n.strip() for n in self.nodes_arg.split(",") if n.strip()
            ]
            try:
                self._run_kubectl(["get", "node"] + raw_nodes)
            except subprocess.SubprocessError as e:
                err_msg = format_subprocess_error(e)
                raise ChsSchedulerException(
                    f"One or more targeted nodes not found in the cluster. "
                    f"Please verify the nodes exist.\nDetails: {err_msg}"
                ) from e
        else:
            try:
                node_resources = self._fetch_raw_node_resources()
                raw_nodes = [
                    res.get("metadata", {}).get("name")
                    for res in node_resources
                    if (res.get("metadata") or {}).get("name")
                ]
                logger.debug("Resolved raw nodes: %s", raw_nodes)
            except (
                subprocess.SubprocessError,
                FileNotFoundError,
                ChsSchedulerException,
            ) as e:
                logger.warning(
                    "Failed to discover cluster nodes via the "
                    "Kubernetes API. Ensure "
                    "'kubectl' is configured correctly. Error: %s",
                    format_subprocess_error(e),
                )

        self._resolved_raw_nodes = raw_nodes
        return raw_nodes

    def get_nodes_under_test(self) -> list[str]:
        """Returns the final subset of idle, correctly formed resources."""
        if self._resolved_nodes is not None:
            return self._resolved_nodes

        raw_nodes = self.resolve_raw_nodes()
        if not raw_nodes:
            self._resolved_nodes = []
            return []

        topology = self._get_tpu_slices()
        busy_nodes = self._get_busy_nodes(target_nodes=raw_nodes)
        not_ready_nodes = self._get_not_ready_nodes()
        excluded_nodes = busy_nodes.union(not_ready_nodes)

        gpu_nodes = set()
        requested_tpu_slices = defaultdict(set)

        for node in raw_nodes:
            pool = topology.node_to_pool.get(node)
            if pool:
                requested_tpu_slices[pool].add(node)
            elif node in topology.all_tpu_nodes:
                # TPU node and exists without its nodepool/slice mapping
                logger.warning(
                    "Excluded TPU node '%s' from testing because "
                    "its parent slice is unhealthy.",
                    node,
                )
                continue
            else:
                gpu_nodes.add(node)

        valid_nodes = []
        if gpu_nodes:
            excluded_gpus = gpu_nodes.intersection(excluded_nodes)
            if excluded_gpus:
                logger.info(
                    "Identified %d active GPU node(s) with "
                    "existing workloads. These nodes "
                    "will be skipped to avoid disrupting running tasks.",
                    len(excluded_gpus),
                )

            valid_nodes.extend(
                [
                    node
                    for node in raw_nodes
                    if node in gpu_nodes and node not in excluded_nodes
                ]
            )

        for pool_name, requested_slice_nodes in requested_tpu_slices.items():
            expected_slice_nodes = set(topology.pools[pool_name])

            excluded_in_slice = list(
                expected_slice_nodes.intersection(excluded_nodes)
            )

            if excluded_in_slice:
                if len(excluded_in_slice) > 3:
                    formatted_nodes = (
                        f"{', '.join(excluded_in_slice[:3])} and "
                        f"{len(excluded_in_slice) - 3} others"
                    )
                else:
                    formatted_nodes = ", ".join(excluded_in_slice)

                logger.warning(
                    "Skipping TPU slice '%s': TPU tests require the entire "
                    "slice to be available and idle, but the following node(s) "
                    "are currently busy or unavailable: %s",
                    pool_name,
                    formatted_nodes,
                )
            else:
                valid_nodes.extend(expected_slice_nodes)

        self._resolved_nodes = valid_nodes
        return self._resolved_nodes

    def get_topology(self) -> dict[str, dict[str, list[str]]]:
        """Discovers network topology using kubectl to get node labels.

        # TODO: This uses GCE topology labels (gce-topology-block).

        Returns a dictionary mapping block names to dicts of subblocks to lists
        of nodes.
        """
        nodes = self.get_nodes_under_test()
        if not nodes:
            return {}

        try:
            items = self._get_node_objects(nodes)

            topology = defaultdict(lambda: defaultdict(list))
            for item in items:
                metadata = item.get("metadata", {})
                name = metadata.get("name")
                labels = metadata.get("labels", {})

                block = labels.get(
                    "cloud.google.com/gce-topology-block", "default-block"
                )
                subblock = labels.get(
                    "cloud.google.com/gce-topology-subblock", "default-subblock"
                )

                if name:
                    topology[block][subblock].append(name)

            # Convert defaultdict to regular dict
            return {b: dict(s) for b, s in topology.items()}

        except (
            ChsSchedulerException,
            subprocess.SubprocessError,
            FileNotFoundError,
        ) as e:
            logger.warning(
                "Failed to retrieve node topology information from "
                "the Kubernetes cluster. Error: %s",
                format_subprocess_error(e),
            )
            return {}

    def write_ramble_config(
        self,
        workspace_config_dir: str,
        tests: list[dict[str, Any]] | None,
        machine_type: str,
        deadline: int | None = None,
        run_id: str | None = None,
        images_dir: str | None = None,
    ) -> None:
        self.workspace_config_dir = workspace_config_dir
        nodes = self.get_nodes_under_test()
        logger.debug("Configuring workspace with kubernetes_template.yaml")
        template_path = pkg_resources.files("chs.schedulers").joinpath(
            "kubernetes_template.yaml"
        )
        with pkg_resources.as_file(template_path) as p:
            with open(p, "r") as f:
                config = yaml.safe_load(f)

        if resources.is_tpu(machine_type):
            l3_networks = ["default"]
            device_networks = []
        else:
            l3_networks, device_networks = self._discover_networks()
            if not l3_networks or not device_networks:
                logger.warning(
                    "Network discovery did not find any Kubernetes network "
                    "resources. Falling back to standard default network "
                    "interface."
                )
                l3_networks = ["default"]
                device_networks = []
        all_networks = l3_networks + device_networks
        all_interfaces = [
            {"interfaceName": f"eth{i}", "network": net_name}
            for i, net_name in enumerate(all_networks)
        ]

        default_interface_name = "eth0"

        variables = config["ramble"]["variables"]
        variables["nodes"] = nodes or []
        variables["machine_type"] = machine_type
        variables["tests"] = tests or []
        variables["default_interface_name"] = default_interface_name
        variables["all_interfaces_json"] = json.dumps(all_interfaces)
        variables["suspend_jobset"] = self.kueue_supported
        variables["chs_run_id"] = run_id
        if deadline is not None:
            variables["deadline"] = deadline

        variables["kubernetes_namespace"] = CHS_NAMESPACE

        if resources.is_tpu(machine_type) and nodes:
            tpu_slices = self._group_nodes_by_slice(nodes)
            variables["tpu_slices"] = tpu_slices

            slice_data = self._get_tpu_slices()

            slice_topologies = {
                pool: slice_data.pool_to_topology[pool]
                for pool in tpu_slices
                if pool in slice_data.pool_to_topology
            }
            variables["tpu_slice_topologies"] = slice_topologies

            dynamic_pools = [
                pool
                for pool in tpu_slices
                if pool in slice_data.dynamic_slice_partitions
            ]
            variables["tpu_dynamic_pools"] = dynamic_pools

        if self._jobset_labels_yaml:
            variables["jobset_labels_yaml"] = self._jobset_labels_yaml
        if self._suspend_jobset == "true":
            variables["suspend_jobset"] = "true"

        logger.debug("Configuration: %r", config)

        self._write_topology_json(workspace_config_dir)

        with open(os.path.join(workspace_config_dir, "ramble.yaml"), "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    def has_no_failed_jobs(self) -> bool:
        """Checks if there are any failed CHS jobs.

        Returns:
            True if all are healthy (active/completed).
        """
        all_healthy = False
        try:
            args = ["get", "jobset", "-o", "json", "--ignore-not-found"]
            output = self._run_kubectl(args, namespaced=True)

            # If stdout is empty, no jobs were found (all completed and deleted)
            if not output:
                raise NoActiveJobsException("No jobs found on the cluster.")

            data = json.loads(output)

            # If a single item is returned, it's not wrapped in a List
            items = (
                data.get("items", [data])
                if data.get("kind") == "List"
                else [data]
            )

            for job_data in items:
                conditions = job_data.get("status", {}).get("conditions", [])
                failed = False
                for cond in conditions:
                    if (
                        cond.get("type") == "Failed"
                        and cond.get("status") == "True"
                    ):
                        failed = True

                if failed:
                    logger.warning(
                        "The diagnostic JobSet '%s' has entered a failed "
                        "state. Initiating resource cleanup.",
                        job_data.get("metadata", {}).get("name", "Unknown"),
                    )
                    return False
                else:
                    all_healthy = True
                    logger.debug(
                        "JobSet %s is healthy.",
                        job_data.get("metadata", {}).get("name", "Unknown"),
                    )

        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            raise ChsSchedulerException(
                "Failed to query the status of diagnostic JobSets via the "
                f"Kubernetes API: {format_subprocess_error(e)}"
            ) from e
        except (json.JSONDecodeError, AttributeError, TypeError) as e:
            logger.warning(
                "Failed to parse the status payload returned by "
                "the JobSet controller. Error: %s",
                e,
            )

        return all_healthy

    def _check_and_update_deadlines(
        self, failed_jobset_ids: list[str], statuses: dict[str, str]
    ) -> None:
        """Batch checks jobs for deadline exceeded and updates statuses."""
        try:
            label_selector = (
                "jobset.sigs.k8s.io/jobset-name in "
                f"({','.join(failed_jobset_ids)})"
            )
            args = [
                "get",
                "jobs",
                "-l",
                label_selector,
                "-o",
                "json",
            ]
            jobs_output = self._run_kubectl(args, namespaced=True)
            if not jobs_output:
                return

            jobs_data = json.loads(jobs_output).get("items", [])
            deadlined_jobsets = set()

            for job in jobs_data:
                jobset_name = (
                    job.get("metadata", {})
                    .get("labels", {})
                    .get("jobset.sigs.k8s.io/jobset-name")
                )
                if not jobset_name or jobset_name in deadlined_jobsets:
                    continue

                for condition in job.get("status", {}).get("conditions", []):
                    if (
                        condition.get("type") == "Failed"
                        and condition.get("status") == "True"
                        and condition.get("reason") == "DeadlineExceeded"
                    ):
                        deadlined_jobsets.add(jobset_name)
                        statuses[jobset_name] = JobStatus.TIMEOUT.value
                        break
        except (
            subprocess.SubprocessError,
            json.JSONDecodeError,
            AttributeError,
            TypeError,
        ) as e:
            logger.debug("Failed to batch check deadlines: %s", e)

    def _evaluate_jobset_status(self, jobset_data: dict) -> str:
        """Evaluates JobSet conditions and fallbacks to active count."""
        status = jobset_data.get("status", {})

        conditions = status.get("conditions", [])
        for condition in conditions:
            condition_type = condition.get("type", "")
            is_true = condition.get("status") == "True"

            if condition_type == "Completed" and is_true:
                return "Completed"
            elif condition_type == "Failed" and is_true:
                return "Failed"

        replicated_jobs = status.get("replicatedJobsStatus", [])
        is_pending = all(
            job.get("active", 0) == 0 and job.get("failed", 0) == 0
            for job in replicated_jobs
        )
        return "PENDING" if is_pending else "RUNNING"

    def get_job_statuses(self, job_ids: list[str]) -> dict[str, str]:
        """Gets the status of the given jobs using kubectl."""
        if not job_ids:
            return {}

        statuses = {j: JobStatus.UNKNOWN.value for j in job_ids}

        try:
            args = (
                ["get", "jobset"]
                + (job_ids if len(job_ids) == 1 else [])
                + ["-o", "json", "--ignore-not-found"]
            )
            output = self._run_kubectl(args, namespaced=True)
            if not output:
                return statuses
            data = json.loads(output)
            items = (
                data.get("items", []) if data.get("kind") == "List" else [data]
            )

            failed_jobset_ids = []
            for item in items:
                job_id = item.get("metadata", {}).get("name")
                if job_id in statuses:
                    raw_status = self._evaluate_jobset_status(item)
                    if raw_status == "Failed":
                        failed_jobset_ids.append(job_id)
                    statuses[job_id] = JobStatus.from_string(raw_status)

            if failed_jobset_ids:
                self._check_and_update_deadlines(failed_jobset_ids, statuses)
        except (
            subprocess.SubprocessError,
            json.JSONDecodeError,
            AttributeError,
            TypeError,
        ) as e:
            logger.warning(
                "Failed to retrieve the current execution statuses "
                "for the scheduled JobSets. Error: %s",
                e,
            )

        return statuses

    @staticmethod
    def _is_tpu_job(job_id: str) -> bool:
        """Checks if the job is a TPU benchmark based on naming convention."""
        return "tpu-" in job_id

    def _get_pod_names(
        self, job_id: str, head_pod_only: bool = False
    ) -> list[str]:
        """Queries Kubernetes for pod names."""
        if head_pod_only:
            args = [
                "get",
                "pods",
                "-l",
                f"jobset.sigs.k8s.io/jobset-name={job_id},batch.kubernetes.io/job-completion-index=0",
                "-o",
                "jsonpath={.items[0].metadata.name}",
            ]
        else:
            args = [
                "get",
                "pods",
                "-l",
                f"jobset.sigs.k8s.io/jobset-name={job_id}",
                "-o",
                "jsonpath={.items[*].metadata.name}",
            ]
        try:
            output = self._run_kubectl(args, namespaced=True)
            return [p for p in output.split() if p]
        except subprocess.SubprocessError as e:
            logger.debug("Failed to query pods for JobSet %s: %s", job_id, e)
            return []

    def _fetch_pod_log(self, pod_name: str, output_path: str) -> None:
        kubectl_cmd = [
            "kubectl",
            "logs",
            pod_name,
            "-n",
            CHS_NAMESPACE,
        ]
        with open(output_path, "wb") as output_file:
            subprocess.run(
                kubectl_cmd,
                stdout=output_file,
                check=True,
                timeout=KUBECTL_TIMEOUT_SEC,
            )

    def fetch_job_log(self, job_id: str, log_path: str) -> None:
        """Fetches the log for a completed Kubernetes job."""
        logger.debug("Fetching logs for JobSet %s to %s", job_id, log_path)
        temp_paths: list[str] = []
        try:
            is_tpu = self._is_tpu_job(job_id)
            pod_names = self._get_pod_names(job_id, head_pod_only=not is_tpu)
            if not pod_names:
                logger.debug(
                    "No pods found to fetch logs for JobSet %s.", job_id
                )
                return
            logger.debug(
                "Fetching logs from %d pods: %s", len(pod_names), pod_names
            )
            if len(pod_names) == 1:
                self._fetch_pod_log(pod_names[0], log_path)
                logger.debug("Logs saved to %s", log_path)
                return
            log_dir = os.path.dirname(os.path.abspath(log_path))
            for _ in pod_names:
                fd, temp_path = tempfile.mkstemp(
                    prefix=_LOG_TEMP_PREFIX, dir=log_dir
                )
                temp_paths.append(temp_path)
                os.close(fd)
            executor = _get_log_fetch_executor()
            futures = [
                executor.submit(self._fetch_pod_log, pod_name, temp_path)
                for pod_name, temp_path in zip(pod_names, temp_paths)
            ]
            try:
                wait(futures)
            except BaseException:
                for future in futures:
                    future.cancel()
                wait(futures)
                raise
            with open(log_path, "wb") as log_file:
                for temp_path in temp_paths:
                    with open(temp_path, "rb") as temp_file:
                        shutil.copyfileobj(temp_file, log_file)
            logger.debug("Logs saved to %s", log_path)
            for future in futures:
                future.result()
        except subprocess.CalledProcessError as e:
            if e.returncode < 0 or e.returncode == 130:
                logger.debug(
                    "Log fetching interrupted by user for JobSet %s.", job_id
                )
            else:
                logger.error(
                    "Failed to retrieve execution logs for JobSet '%s'. "
                    "Result parsing may be incomplete. Error: %s",
                    job_id,
                    format_subprocess_error(e),
                )
        except subprocess.TimeoutExpired as e:
            logger.error(
                "Timed out retrieving execution logs for JobSet '%s'. "
                "Result parsing may be incomplete. Error: %s",
                job_id,
                e,
            )
        except OSError as e:
            logger.error(
                "Failed to write execution logs to the "
                "local file '%s'. Error: %s",
                log_path,
                e,
            )
        finally:
            for temp_path in temp_paths:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def extract_job_id(self, experiment_dir: str) -> str:
        """Extracts the job ID (JobSet name) from the experiment directory."""
        jobset_file = os.path.join(experiment_dir, "jobset.yaml")
        try:
            with open(jobset_file, "r") as f:
                docs = list(yaml.safe_load_all(f))
                for doc in docs:
                    if doc and doc.get("kind") == "JobSet":
                        name = doc.get("metadata", {}).get("name")
                        if name:
                            return name
            raise ChsSchedulerException(
                f"JobSet kind not found in {jobset_file}"
            )
        except (OSError, yaml.YAMLError) as e:
            raise ChsSchedulerException(
                f"Failed to parse GKE jobset.yaml at {jobset_file}: {e}"
            ) from e

    def _cleanup_dynamic_slices(self) -> None:
        """Deletes dynamic TPU slices created by CHS."""
        try:
            crd_check = self._run_kubectl(
                [
                    "get",
                    "crd",
                    "slices.accelerator.gke.io",
                    "--ignore-not-found",
                ]
            )
            if not crd_check.strip():
                return

            self._run_kubectl(
                [
                    "delete",
                    "slices.accelerator.gke.io",
                    "-A",
                    "-l",
                    "created-by=chs",
                    "--ignore-not-found",
                ],
                timeout=300,
            )
        except Exception as e:
            logger.warning(
                "Failed to delete dynamic TPU slices after execution. "
                "Manual cleanup may be required. Error: %s",
                format_subprocess_error(e),
            )

    def _wait_for_slices_active(
        self, expected_nodepools: list[str], timeout_sec: int = 300
    ) -> tuple[list[str], list[str]]:
        """Waits for dynamically provisioned slices to reach terminal states."""
        start_time = time.time()
        slice_terminal_statuses = {}
        failed_reasons = {
            "FAILED",
            "SliceCreationFailed",
            "ACTIVE_DEGRADED",
            "INCOMPLETE",
        }

        while time.time() - start_time < timeout_sec:
            try:
                output = self._run_kubectl(
                    [
                        "get",
                        "slice",
                        "-l",
                        "created-by=chs",
                        "-o",
                        "json",
                        "--ignore-not-found",
                    ]
                )
                if not output:
                    time.sleep(10)
                    continue
                slices_json = json.loads(output)
                items = slices_json.get("items", [])
                if not items:
                    time.sleep(10)
                    continue

                for slice_item in items:
                    nodepool = (
                        slice_item.get("metadata", {})
                        .get("labels", {})
                        .get("chs-nodepool")
                    )
                    if not nodepool:
                        continue

                    conditions = (
                        slice_item.get("status", {}).get("conditions", [])
                        if slice_item.get("status")
                        else []
                    )
                    ready_cond = next(
                        (c for c in conditions if c.get("type") == "Ready"),
                        {},
                    )
                    reason = ready_cond.get("reason")

                    if reason == "ACTIVE" or reason in failed_reasons:
                        slice_terminal_statuses[nodepool] = reason

                if len(slice_terminal_statuses) >= len(expected_nodepools):
                    break

            except (
                subprocess.SubprocessError,
                json.JSONDecodeError,
            ) as e:
                logger.debug("Failed to check status for dynamic slices: %s", e)
                pass
            time.sleep(10)

        success_nodepools = [
            pool
            for pool, status in slice_terminal_statuses.items()
            if status == "ACTIVE"
        ]
        failed_nodepools = [
            pool for pool in expected_nodepools if pool not in success_nodepools
        ]
        return success_nodepools, failed_nodepools

    def _get_already_fulfilled_partitions(self) -> set[str]:
        """Returns partition IDs of all currently existing slices to prevent
        conflicts.
        """
        fulfilled = set()
        try:
            output = self._run_kubectl(
                ["get", "slice", "-A", "-o", "json", "--ignore-not-found"]
            )
            if output:
                slices = json.loads(output).get("items", [])
                for s in slices:
                    if s.get("metadata", {}).get("deletionTimestamp"):
                        continue
                    partition_ids = s.get("spec", {}).get("partitionIds", [])
                    for pid in partition_ids:
                        fulfilled.add(pid)
        except subprocess.SubprocessError as e:
            logger.debug("Could not retrieve existing slices: %s", e)
        return fulfilled

    def _provision_dynamic_slices(
        self, target_pools: dict[str, list[str]]
    ) -> tuple[list[str], list[str]]:
        """Provisions and waits for dynamic TPU slices."""
        if not (
            self._tpu_slice_data
            and self._tpu_slice_data.dynamic_slice_partitions
            and self.workspace_config_dir
        ):
            return [], []

        fulfilled_partitions = self._get_already_fulfilled_partitions()
        pools_to_provision = []
        already_fulfilled_pools = []

        slice_yaml_path = os.path.join(
            self.workspace_config_dir, "chs-slices.yaml"
        )
        all_slices_yaml = ""
        template_path = pkg_resources.files("chs.schedulers").joinpath(
            "slice.yaml.tpl"
        )
        try:
            with pkg_resources.as_file(template_path) as p:
                with open(p, "r") as f:
                    template_str = f.read()
        except OSError as e:
            raise ChsSchedulerException(f"Failed to read slice template: {e}")

        for (
            pool_name,
            partition_id,
        ) in self._tpu_slice_data.dynamic_slice_partitions.items():
            if pool_name not in target_pools:
                continue

            if partition_id in fulfilled_partitions:
                already_fulfilled_pools.append(pool_name)
                continue

            pools_to_provision.append(pool_name)
            tpu_type = self._tpu_slice_data.pool_to_accelerator.get(
                pool_name, ""
            )
            shape = self._tpu_slice_data.pool_to_topology.get(pool_name, "")

            slice_name = f"chs-slice-{pool_name}"[:63].rstrip("-.")
            rendered = template_str.format(
                slice_name=slice_name,
                pool_name=pool_name,
                tpu_type=tpu_type,
                shape=shape,
                partition_id=partition_id,
            )
            if all_slices_yaml:
                all_slices_yaml += "---\n"
            all_slices_yaml += rendered

        if all_slices_yaml:
            with open(slice_yaml_path, "w") as f:
                f.write(all_slices_yaml)

            try:
                self._run_kubectl(["apply", "-f", slice_yaml_path])
            except subprocess.SubprocessError as e:
                raise ChsSchedulerException(
                    f"Failed to apply dynamic slice configuration: {e}"
                ) from e

            success, failed = self._wait_for_slices_active(
                pools_to_provision, 300
            )
            return success + already_fulfilled_pools, failed

        return already_fulfilled_pools, []

    def apply_scheduler_resources(self) -> None:
        """Applies generated scheduler resources to the cluster."""
        target_pools = {}
        nodes = self.get_nodes_under_test()
        if nodes:
            target_pools = self._group_nodes_by_slice(nodes)

        if self._tpu_slice_data and any(
            pool in self._tpu_slice_data.dynamic_slice_partitions
            for pool in target_pools
        ):
            self._cleanup_dynamic_slices()

        _, failed_nodepools = self._provision_dynamic_slices(target_pools)

        if failed_nodepools and self._tpu_slice_data:
            nodes_to_remove = set()
            for nodepool in failed_nodepools:
                nodes_to_remove.update(target_pools.get(nodepool, []))
                self._purge_unhealthy_pool_data(
                    nodepool,
                    self._tpu_slice_data.pools,
                    self._tpu_slice_data.node_to_pool,
                    self._tpu_slice_data.pool_to_topology,
                    self._tpu_slice_data.dynamic_slice_partitions,
                    self._tpu_slice_data.pool_to_accelerator,
                )

            if self._resolved_nodes is not None:
                self._resolved_nodes = [
                    n for n in self._resolved_nodes if n not in nodes_to_remove
                ]

                if not self._resolved_nodes:
                    raise ChsSchedulerException(
                        "All targeted dynamic slices failed to provision: "
                        f"{', '.join(failed_nodepools)}."
                    )

            logger.warning(
                "Dynamic slice provisioning failed for nodepools: %s. "
                "Tests will skip these and run on the remaining active "
                "slices.",
                ", ".join(failed_nodepools),
            )

        self._ensure_namespace_exists()
        if self.kueue_supported:
            kueue_mgr = KueueManager(self)
            kueue_mgr.apply_configs(self.workspace_config_dir)

    def cleanup_scheduler_resources(self) -> None:
        """Cleans up scheduler-specific resources like Kueue
        configurations statelessly."""

        self._cleanup_dynamic_slices()

        kueue_mgr = KueueManager(self)
        try:
            kueue_mgr.cleanup()
        except KueueException as e:
            logger.warning(
                "Failed to clean up Kueue workload resources. Manual "
                "cleanup may be required. Error: %s",
                e,
            )

        if not self._delete_namespace():
            self._delete_job_secrets()

    def cancel_jobs(self, job_ids: list[str]) -> None:
        """Cancels the given JobSets using kubectl delete."""
        if not job_ids:
            logger.warning("No job IDs to cancel.")
            return

        try:
            args = (
                ["delete", "jobset"]
                + job_ids
                + [
                    "--ignore-not-found",
                    f"--timeout={DELETE_WAIT_TIMEOUT_SEC}s",
                ]
            )
            self._run_kubectl(args, namespaced=True)
            logger.debug("Successfully deleted JobSets: %s", ", ".join(job_ids))
        except FileNotFoundError:
            logger.warning(
                "The 'kubectl' command-line tool was not found in your "
                "PATH. Cannot natively cancel active jobs."
            )
        except subprocess.SubprocessError as e:
            logger.error(
                "Encountered an error while attempting to "
                "delete active JobSets. Error: %s",
                format_subprocess_error(e),
            )

        self._delete_job_secrets()

    def get_active_job_names(self) -> list[str]:
        """Returns a list of active job names (JobSets) on the cluster."""
        try:
            output = self._run_kubectl(
                ["get", "jobset", "-o", "name"], namespaced=True
            )
            if not output:
                return []

            job_names = []
            for line in output.splitlines():
                if line:
                    job_names.append(line.split("/")[-1])
            return job_names
        except FileNotFoundError:
            logger.warning(
                "The 'kubectl' command-line tool was not found in your "
                "PATH. Cannot fetch the status of active jobs."
            )
            return []
        except subprocess.TimeoutExpired as e:
            raise ChsSchedulerException(
                "Timed out listing active JobSets; cannot verify that no "
                f"other CHS run is in progress: {e}"
            ) from e
        except subprocess.CalledProcessError as e:
            logger.warning(
                "Failed to retrieve the list of active JobSets "
                "from the cluster. Error: %s",
                format_subprocess_error(e),
            )
            return []

    def cancel_jobs_by_prefix(self, prefix: str) -> None:
        """Cancels all JobSets matching the specified prefix natively."""
        logger.debug("Attempting native cancel for prefix: %s", prefix)
        # Match the sanitized prefix as a substring in the JobSet name.
        sanitized_match = prefix.replace("_", "-").replace(".", "-").lower()

        try:
            # List all JobSets
            output = self._run_kubectl(
                ["get", "jobset", "-o", "name"], namespaced=True
            )
            if not output:
                return

            job_ids = []
            for line in output.splitlines():
                if sanitized_match in line:
                    job_name = line.split("/")[-1]
                    job_ids.append(job_name)

            if job_ids:
                logger.debug(
                    "Found %d JobSets to cancel matching %s.",
                    len(job_ids),
                    prefix,
                )
                self.cancel_jobs(job_ids)
            else:
                logger.debug("No matching JobSets found for %s.", prefix)
        except FileNotFoundError:
            logger.warning(
                "The 'kubectl' command-line tool was not found in your PATH. "
                "Cannot perform a native job cancellation."
            )
        except Exception as e:
            logger.warning(
                "Failed to execute native job cancellation "
                "via Kubernetes. Error: %s",
                e,
            )
