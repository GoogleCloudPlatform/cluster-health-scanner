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

import datetime
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


from typing import Any
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass, field

from chs.schedulers.base import BaseScheduler
from chs.schedulers.gang_admission import (
    pinned_nodes,
    plan_admissions,
    pod_nodes,
)
from chs.utils import k8s_client
from chs.exceptions import (
    ChsSchedulerException,
    ChsValidationError,
)
from chs.utils.lifecycle import JobStatus
from chs import resources
from chs.constants import (
    CHS_NAMESPACE,
    DELETE_WAIT_TIMEOUT_SEC,
    NAMESPACE_DELETE_TIMEOUT_SEC,
    SUPPORTED_DYNAMIC_SLICE_TOPOLOGY,
)

_BUSY_NODES_BATCH_SIZE = 32
_BUSY_NODES_MAX_RETRIES = 3
_MAX_NODES_UNDER_TEST = 500

GANG_SCHEDULING_TIMEOUT_SECONDS = 300

_JOB_NAME_LABEL = "batch.kubernetes.io/job-name"
_JOB_COMPLETION_INDEX_LABEL = "batch.kubernetes.io/job-completion-index"
_SLICE_GROUP = "accelerator.gke.io"
_SLICE_VERSION = "v1beta1"

WORKER_SERVICE_NAME = "chs"
WORKER_LABEL = "chs.gke.io/worker"

_API_MAX_CONCURRENCY = 32
_APPLY_ATTEMPTS = 5
_CUSTOM_RESOURCE_PLURALS = {
    "ComputeDomain": "computedomains",
}

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


def _read_manifests(experiment_dir: str) -> list[dict]:
    manifest_path = os.path.join(experiment_dir, "jobset.yaml")
    try:
        with open(manifest_path, "r") as f:
            objects = [obj for obj in yaml.safe_load_all(f) if obj]
    except (OSError, yaml.YAMLError) as e:
        raise ChsSchedulerException(
            f"Failed to read {manifest_path}: {e}"
        ) from e
    for obj in objects:
        kind = obj.get("kind")
        if kind not in ("Secret", "Job", *_CUSTOM_RESOURCE_PLURALS):
            raise ChsSchedulerException(
                f"Unsupported kind '{kind}' in {manifest_path}"
            )
    return objects


def _find_job(objects: list[dict], experiment_dir: str) -> dict:
    for obj in objects:
        if obj["kind"] == "Job" and obj.get("metadata", {}).get("name"):
            return obj
    raise ChsSchedulerException(
        f"Job kind not found in {experiment_dir}/jobset.yaml"
    )


def _apply_object(obj: dict) -> None:
    if obj["kind"] == "Secret":
        k8s_client.apply_secret(body=obj)
    elif obj["kind"] == "Job":
        k8s_client.apply_job(body=obj)
    else:
        k8s_client.apply_custom_object(
            plural=_CUSTOM_RESOURCE_PLURALS[obj["kind"]], body=obj
        )


def _is_retryable(error: k8s_client.KubeError) -> bool:
    return error.status is None or error.status == 429 or error.status >= 500


def _apply_objects(objects: list[dict]) -> None:
    for obj in objects:
        for attempt in range(1, _APPLY_ATTEMPTS + 1):
            try:
                _apply_object(obj)
                break
            except k8s_client.KubeError as e:
                if attempt == _APPLY_ATTEMPTS or not _is_retryable(e):
                    raise
                delay = 2 ** (attempt - 1)
                logger.debug(
                    "Retrying %s '%s' in %d seconds. Error: %s",
                    obj["kind"],
                    obj["metadata"]["name"],
                    delay,
                    e,
                )
                time.sleep(delay)


def _job_condition(job: dict, condition_type: str) -> dict | None:
    """Returns the Job's condition of the given type if it is True."""
    for condition in job.get("status", {}).get("conditions") or []:
        if (
            condition.get("type") == condition_type
            and condition.get("status") == "True"
        ):
            return condition
    return None


def _has_accelerator(resources_map: dict | None, accelerator_type: str) -> bool:
    """Returns True if resources has non-zero quantity for accelerator_type."""
    return any(
        accelerator_type in key.lower() and quantity not in (0, "0", None, "")
        for key, quantity in (resources_map or {}).items()
    )


def _is_accelerator_node(node: dict) -> bool:
    """Returns True if the node has allocatable GPU or TPU resources."""
    allocatable = (node.get("status") or {}).get("allocatable")
    return _has_accelerator(allocatable, "gpu") or _has_accelerator(
        allocatable, "tpu"
    )


def _requests_accelerator(pod: dict) -> bool:
    """Returns True if a container of the pod asks for GPUs or TPUs."""
    spec = pod.get("spec") or {}
    for container in (spec.get("containers") or []) + (
        spec.get("initContainers") or []
    ):
        container_resources = container.get("resources") or {}
        for section in ("requests", "limits"):
            res = container_resources.get(section)
            if _has_accelerator(res, "gpu") or _has_accelerator(res, "tpu"):
                return True
    return False


def _is_chs_pod(pod: dict) -> bool:
    """Returns True for pods of CHS's own Jobs."""
    metadata = pod.get("metadata") or {}
    labels = metadata.get("labels") or {}
    return (
        metadata.get("namespace") == CHS_NAMESPACE
        and labels.get(WORKER_LABEL) == "true"
    )


def _worker_service(namespace: str) -> dict:
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": WORKER_SERVICE_NAME, "namespace": namespace},
        "spec": {
            "clusterIP": "None",
            "publishNotReadyAddresses": True,
            "selector": {WORKER_LABEL: "true"},
        },
    }


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

    submits_directly = True

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

    def _resolve_project_id(self, node: str) -> str:
        """Resolves GCP project ID from Kubernetes node providerID or ADC."""
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
                node_obj = k8s_client.get_node(name=node)
                provider_id = node_obj.get("spec", {}).get("providerID", "")
            provider_id = provider_id.strip()
            # providerID format: gce://project-id/zone/instance-name
            match = re.match(r"^gce://([^/]+)/([^/]+)/([^/]+)$", provider_id)
            if not match:
                raise ValueError(f"Unexpected providerID format: {provider_id}")
            return match.group(1)
        except Exception as e:
            try:
                return super()._resolve_project_id()
            except Exception as fallback_err:
                raise ChsSchedulerException(
                    f"Failed to resolve project ID from node '{node}': {e}. "
                    f"{fallback_err}"
                ) from e

    def detect_nodes_hardware(self, nodes: list[str]) -> list[str]:
        """Detects machine types and maps instance IDs for the given nodes."""
        if not nodes:
            return []

        try:
            node_objects = self._get_node_objects(nodes)
        except (
            ChsSchedulerException,
            k8s_client.KubeError,
            OSError,
        ) as e:
            logger.debug(
                "Could not read node objects to detect hardware; querying "
                "the GCE API instead. Error: %s",
                e,
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

    def setup_scheduler_resources(
        self,
        workspace_config_dir: str,
        tests: list[dict[str, Any]] | None = None,
        machine_type: str = "",
        workspace_name: str | None = None,
    ) -> list[dict[str, Any]] | None:
        """Sets up scheduler-specific resources."""
        self.workspace_config_dir = workspace_config_dir
        os.makedirs(workspace_config_dir, exist_ok=True)
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

        try:
            context = (k8s_client.current_context() or "").strip()
            if context:
                if context.startswith("gke_"):
                    parts = context.split("_")
                    if len(parts) >= 4:
                        return parts[-1]
                return context
        except Exception:
            pass
        return None

    def _ensure_namespace_exists(self) -> None:
        """Ensures the chs namespace and worker Service exist."""
        k8s_client.create_namespace_if_not_exists(name=CHS_NAMESPACE)
        k8s_client.apply_service(body=_worker_service(CHS_NAMESPACE))

    def _delete_namespace(self) -> bool:
        """Deletes the chs namespace during cleanup."""
        try:
            k8s_client.delete_namespace(
                name=CHS_NAMESPACE,
                wait_timeout_seconds=NAMESPACE_DELETE_TIMEOUT_SEC,
            )
        except Exception as e:
            logger.warning(
                "Failed to delete Kubernetes namespace. This may "
                "leave orphaned resources. Error: %s",
                e,
            )
            return False
        return True

    def _delete_job_secrets(self, job_ids: list[str] | None = None) -> None:
        """Deletes the SSH secrets of the given jobs, or of all CHS jobs."""
        try:
            secrets = k8s_client.list_secrets(namespace=CHS_NAMESPACE)
            secret_names = [
                secret.get("metadata", {}).get("name", "") for secret in secrets
            ]
            if job_ids is None:
                secrets_to_delete = [
                    s
                    for s in secret_names
                    if s.startswith("chs-") and s.endswith("-ssh")
                ]
            else:
                job_secrets = {f"{job_id}-ssh" for job_id in job_ids}
                secrets_to_delete = [
                    s for s in secret_names if s in job_secrets
                ]

            if secrets_to_delete:
                for name in secrets_to_delete:
                    k8s_client.delete_secret(name=name, namespace=CHS_NAMESPACE)
                logger.debug(
                    "Successfully deleted secrets: %s",
                    ", ".join(secrets_to_delete),
                )
            else:
                logger.debug("No CHS SSH secrets found to delete.")
        except Exception as e:
            logger.warning(
                "Failed to delete Kubernetes secrets. Error: %s",
                e,
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
        except (ChsSchedulerException, k8s_client.KubeError) as e:
            logger.warning(
                "Failed to fetch the Kubernetes node objects for nodes %s. "
                "Error: %s",
                node_names,
                e,
            )
            slices["failed-nodes"].extend(node_names)

        return dict(slices)

    def _discover_networks(self) -> tuple[list[str], list[str]]:
        """Discovers all network names and groups them by type."""
        try:
            gke_networks = k8s_client.list_custom_objects(
                group="networking.gke.io", version="v1", plural="networks"
            )

            networks = defaultdict(list)

            for gke_network in gke_networks:
                metadata = gke_network.get("metadata", {})
                spec = gke_network.get("spec", {})
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

        except Exception as e:
            logger.warning(
                "Encountered an unexpected error during Kubernetes "
                "network topology discovery. Error: %s",
                e,
            )
            return [], []

    def _is_node_busy(self, node_name: str) -> bool:
        """Helper method to determine if a specific node is busy."""
        pods = self._list_pods_on_node(node_name)
        return any(_requests_accelerator(pod) for pod in pods)

    @staticmethod
    def _list_pods_on_node(node_name: str) -> list[dict]:
        """Lists the unfinished pods on a node, retrying failed calls."""
        field_selector = (
            f"spec.nodeName={node_name},"
            "status.phase!=Succeeded,status.phase!=Failed"
        )

        for attempt in range(_BUSY_NODES_MAX_RETRIES):
            try:
                pods = k8s_client.list_pods(field_selector=field_selector)
                break
            except k8s_client.KubeError as e:
                if attempt == _BUSY_NODES_MAX_RETRIES - 1:
                    raise ChsSchedulerException(
                        "Failed to communicate with API server for "
                        f"node {node_name} "
                        f"after {_BUSY_NODES_MAX_RETRIES} attempts. Error: {e}"
                    ) from e

                # Exponential backoff with jitter to prevent thundering herd
                backoff = (2**attempt) + random.uniform(0, 1)
                time.sleep(backoff)

        return pods

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

    def _find_taken_nodes(self, nodes: set[str]) -> dict[str, str]:
        """Maps each node that another workload now uses to one of its pods."""
        if not nodes:
            return {}
        taken = {}
        with ThreadPoolExecutor(
            max_workers=min(len(nodes), _BUSY_NODES_BATCH_SIZE)
        ) as executor:
            futures = {
                executor.submit(self._other_workload_on_node, node): node
                for node in nodes
            }
            for future in as_completed(futures):
                node = futures[future]
                try:
                    pod = future.result()
                except ChsSchedulerException as e:
                    logger.debug(
                        "Failed to check node '%s' for other workloads, so "
                        "it is treated as free. Error: %s",
                        node,
                        e,
                    )
                    continue
                if pod:
                    taken[node] = pod
        return taken

    def _other_workload_on_node(self, node_name: str) -> str | None:
        """Returns a pod outside CHS that uses accelerators on the node."""
        for pod in self._list_pods_on_node(node_name):
            if _requests_accelerator(pod) and not _is_chs_pod(pod):
                metadata = pod.get("metadata") or {}
                return f"{metadata.get('namespace')}/{metadata.get('name')}"
        return None

    def _fetch_raw_node_resources(self) -> list[dict]:
        """Fetches raw node resource dictionaries from the Kubernetes API."""
        if self._resolved_node_resources is not None:
            return self._resolved_node_resources

        label_selector = None
        if self.nodepools_arg:
            pools = [
                p.strip() for p in self.nodepools_arg.split(",") if p.strip()
            ]
            if pools:
                pools_str = ",".join(pools)
                label_selector = (
                    f"cloud.google.com/gke-nodepool in ({pools_str})"
                )
        else:
            label_selector = "cloud.google.com/gke-nodepool!=system"

        try:
            node_resources = k8s_client.list_nodes(
                label_selector=label_selector
            )
        except k8s_client.KubeError as e:
            raise ChsSchedulerException(
                f"Failed to get nodes to determine availability: {e}"
            ) from e

        resolved_resources = [
            node
            for node in node_resources
            if isinstance(node, dict) and _is_accelerator_node(node)
        ]

        if self.nodepools_arg:
            requested_pools = {
                p.strip() for p in self.nodepools_arg.split(",") if p.strip()
            }
            found_pools = set()
            for node in resolved_resources:
                labels = (node.get("metadata") or {}).get("labels") or {}
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

        for node in nodes:
            node_name = node.get("metadata", {}).get("name")
            if not node_name:
                continue

            labels = node.get("metadata", {}).get("labels", {})
            tpu_type = labels.get("cloud.google.com/gke-tpu-accelerator")
            allocatable = (node.get("status") or {}).get("allocatable")
            if not tpu_type and not _has_accelerator(allocatable, "tpu"):
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
        """Resolves targeted nodes without dropping busy nodes."""
        if self._resolved_raw_nodes is not None:
            return self._resolved_raw_nodes

        raw_nodes = []
        if self.nodes_arg:
            raw_nodes = [
                n.strip() for n in self.nodes_arg.split(",") if n.strip()
            ]
            try:
                cluster_nodes = {
                    node.get("metadata", {}).get("name")
                    for node in k8s_client.list_nodes()
                }
            except k8s_client.KubeError as e:
                raise ChsSchedulerException(
                    f"Failed to verify the targeted nodes: {e}"
                ) from e
            missing = [node for node in raw_nodes if node not in cluster_nodes]
            if missing:
                raise ChsSchedulerException(
                    "One or more targeted nodes not found in the cluster. "
                    f"Please verify the nodes exist: {', '.join(missing)}"
                )
        else:
            try:
                node_resources = self._fetch_raw_node_resources()
                raw_nodes = [
                    node.get("metadata", {}).get("name")
                    for node in node_resources
                    if node.get("metadata", {}).get("name")
                ]
                logger.debug("Resolved raw nodes: %s", raw_nodes)
            except (k8s_client.KubeError, ChsSchedulerException) as e:
                logger.warning(
                    "Failed to discover cluster nodes via the "
                    "Kubernetes API. Ensure your kubeconfig (or in-cluster "
                    "configuration) is set up correctly. Error: %s",
                    e,
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
        """Discovers network topology from the labels of the node objects."""
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
                block = labels.get("cloud.google.com/gce-topology-block")
                subblock = labels.get("cloud.google.com/gce-topology-subblock")
                if not (block and subblock):
                    instance_info = self._get_instances_info(nodes).get(
                        name, {}
                    )
                    block = block or instance_info.get("block")
                    subblock = subblock or instance_info.get("subblock")

                if name:
                    topology[block or "default-block"][
                        subblock or "default-subblock"
                    ].append(name)

            return {b: dict(s) for b, s in topology.items()}

        except (
            ChsSchedulerException,
            k8s_client.KubeError,
        ) as e:
            logger.warning(
                "Failed to retrieve node topology information from "
                "the Kubernetes cluster. Error: %s",
                e,
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

        logger.debug("Configuration: %r", config)

        self._write_topology_json(workspace_config_dir)

        with open(os.path.join(workspace_config_dir, "ramble.yaml"), "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    @staticmethod
    def _evaluate_job_status(job: dict) -> str:
        """Returns the JobStatus value of a batch/v1 Job."""
        if _job_condition(job, "Complete"):
            return JobStatus.COMPLETED.value
        failed = _job_condition(job, "Failed")
        if failed:
            if failed.get("reason") == "DeadlineExceeded":
                return JobStatus.TIMEOUT.value
            return JobStatus.UNKNOWN.value
        status = job.get("status", {})
        if status.get("active", 0) == 0:
            return JobStatus.PENDING.value
        return JobStatus.RUNNING.value

    def get_job_statuses(self, job_ids: list[str]) -> dict[str, str]:
        """Gets the status of the given jobs from the Kubernetes API."""
        if not job_ids:
            return {}
        try:
            jobs = self._list_jobs()
        except k8s_client.KubeError as e:
            logger.warning(
                "Failed to retrieve the current execution statuses "
                "for the scheduled Jobs. Error: %s",
                e,
            )
            return {j: JobStatus.UNKNOWN.value for j in job_ids}
        return self._statuses_from_jobs(jobs, job_ids)

    def poll_jobs(self, job_ids: list[str]) -> dict[str, str]:
        """Starts waiting Jobs with free nodes and returns job statuses."""
        if not job_ids:
            return {}
        try:
            jobs = self._list_jobs()
        except k8s_client.KubeError as e:
            raise ChsSchedulerException(
                f"Failed to list the CHS Jobs: {e}"
            ) from e
        statuses = self._statuses_from_jobs(jobs, job_ids)
        queue = list(job_ids)
        checked = set()
        taken = {}
        skipped = []
        while True:
            to_start = plan_admissions(jobs, queue)
            nodes_by_job = pinned_nodes(jobs, to_start)
            unchecked = set().union(*nodes_by_job.values()) - checked
            taken.update(self._find_taken_nodes(unchecked))
            checked |= unchecked
            blocked = [
                name
                for name in to_start
                if not nodes_by_job[name].isdisjoint(taken)
            ]
            if not blocked:
                break
            for name in blocked:
                busy = sorted(nodes_by_job[name].intersection(taken))
                logger.warning(
                    "Job '%s' was not started because node %s is now used by "
                    "another workload (pod %s).",
                    name,
                    ", ".join(busy),
                    ", ".join(taken[node] for node in busy),
                )
                statuses[name] = JobStatus.DEADLINE.value
            skipped += blocked
            # Skipped Jobs hold no nodes, so the Jobs behind them may start.
            queue = [name for name in queue if name not in blocked]
        self._start_jobs(to_start)
        if skipped:
            self.cancel_jobs(skipped)
        return statuses

    @staticmethod
    def _list_jobs() -> list[dict]:
        return k8s_client.list_jobs(namespace=CHS_NAMESPACE)

    def _statuses_from_jobs(
        self, jobs: list[dict], job_ids: list[str]
    ) -> dict[str, str]:
        statuses = {j: JobStatus.UNKNOWN.value for j in job_ids}
        try:
            for job in jobs:
                job_id = job.get("metadata", {}).get("name")
                if job_id in statuses:
                    statuses[job_id] = self._evaluate_job_status(job)
        except (AttributeError, TypeError) as e:
            logger.warning(
                "Failed to retrieve the current execution statuses "
                "for the scheduled Jobs. Error: %s",
                e,
            )
        return statuses

    def _start_jobs(self, names: list[str]) -> None:
        if not names:
            return
        executor = ThreadPoolExecutor(
            max_workers=min(len(names), _API_MAX_CONCURRENCY)
        )
        try:
            list(executor.map(self._start_job, names))
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    @staticmethod
    def _start_job(name: str) -> None:
        try:
            k8s_client.patch_job(
                name=name,
                namespace=CHS_NAMESPACE,
                body={"spec": {"suspend": False}},
            )
            logger.debug("Started Job '%s'.", name)
        except k8s_client.KubeError as e:
            logger.warning("Failed to start Job '%s'. Error: %s", name, e)

    def get_unschedulable_jobs(self, job_ids: list[str]) -> list[str]:
        """Returns Jobs that could not schedule all of their pods.

        A Job is returned once a pod has waited for
        GANG_SCHEDULING_TIMEOUT_SECONDS, or as soon as a pod is waiting and
        another workload uses one of the Job's nodes.
        """
        if not job_ids:
            return []
        job_id_set = set(job_ids)
        try:
            pods = k8s_client.list_pods(
                namespace=CHS_NAMESPACE,
                field_selector="status.phase=Pending",
            )
        except k8s_client.KubeError as e:
            logger.warning(
                "Failed to list pending pods to check for unschedulable "
                "Jobs. Error: %s",
                e,
            )
            return []

        now = datetime.datetime.now(datetime.timezone.utc)
        unschedulable = set()
        waiting = defaultdict(set)
        for pod in pods:
            metadata = pod["metadata"]
            job_name = metadata.get("labels", {}).get(_JOB_NAME_LABEL)
            if job_name not in job_id_set:
                continue
            if not any(
                c.get("type") == "PodScheduled"
                and c.get("status") == "False"
                and c.get("reason") == "Unschedulable"
                for c in pod.get("status", {}).get("conditions", [])
            ):
                continue
            created = datetime.datetime.fromisoformat(
                metadata["creationTimestamp"].replace("Z", "+00:00")
            )
            age_seconds = (now - created).total_seconds()
            if age_seconds >= GANG_SCHEDULING_TIMEOUT_SECONDS:
                unschedulable.add(job_name)
            else:
                waiting[job_name] |= pod_nodes(pod)

        for job_name in unschedulable:
            waiting.pop(job_name, None)
        candidates = self._nodes_without_job_pods(waiting)
        taken = self._find_taken_nodes(set().union(*candidates.values()))
        for job_name, nodes in sorted(candidates.items()):
            busy = sorted(nodes.intersection(taken))
            if busy:
                logger.warning(
                    "Job '%s' cannot schedule all of its pods because node %s "
                    "is now used by another workload (pod %s).",
                    job_name,
                    ", ".join(busy),
                    ", ".join(taken[node] for node in busy),
                )
                unschedulable.add(job_name)
        return sorted(unschedulable)

    @staticmethod
    def _nodes_without_job_pods(
        waiting: dict[str, set[str]],
    ) -> dict[str, set[str]]:
        """Returns each Job's nodes that none of its pods is on yet."""
        if not waiting:
            return {}
        try:
            pods = k8s_client.list_pods(
                namespace=CHS_NAMESPACE,
                label_selector=(
                    f"{_JOB_NAME_LABEL} in ({','.join(sorted(waiting))})"
                ),
                field_selector="status.phase!=Succeeded,status.phase!=Failed",
            )
        except k8s_client.KubeError as e:
            logger.debug(
                "Failed to list the pods of waiting Jobs to check their "
                "nodes. Error: %s",
                e,
            )
            return {}
        used = defaultdict(set)
        for pod in pods:
            job_name = pod["metadata"].get("labels", {}).get(_JOB_NAME_LABEL)
            used[job_name].add(pod.get("spec", {}).get("nodeName"))
        return {job: nodes - used[job] for job, nodes in waiting.items()}

    @staticmethod
    def _is_tpu_job(job_id: str) -> bool:
        """Checks if the job is a TPU benchmark based on naming convention."""
        return "tpu-" in job_id

    def _get_pods(self, job_id: str, head_pod_only: bool = False) -> list[dict]:
        """Queries Kubernetes for the pods of a Job."""
        label_selector = f"{_JOB_NAME_LABEL}={job_id}"
        if head_pod_only:
            label_selector += f",{_JOB_COMPLETION_INDEX_LABEL}=0"
        try:
            pods = k8s_client.list_pods(
                namespace=CHS_NAMESPACE, label_selector=label_selector
            )
        except k8s_client.KubeError as e:
            logger.debug("Failed to query pods for Job %s: %s", job_id, e)
            return []
        return pods[:1] if head_pod_only else pods

    def _fetch_pod_log(self, pod: dict, output_path: str) -> None:
        """Streams the log of one pod into output_path."""
        with open(output_path, "wb") as output_file:
            k8s_client.stream_pod_log(pod=pod, out=output_file)

    def fetch_job_log(self, job_id: str, log_path: str) -> None:
        """Fetches the log for a completed Kubernetes job."""
        logger.debug("Fetching logs for Job %s to %s", job_id, log_path)
        temp_paths: list[str] = []
        try:
            is_tpu = self._is_tpu_job(job_id)
            pods = self._get_pods(job_id, head_pod_only=not is_tpu)
            if not pods:
                logger.debug("No pods found to fetch logs for Job %s.", job_id)
                return
            logger.debug(
                "Fetching logs from %d pods: %s",
                len(pods),
                [pod["metadata"]["name"] for pod in pods],
            )
            if len(pods) == 1:
                self._fetch_pod_log(pods[0], log_path)
                logger.debug("Logs saved to %s", log_path)
                return
            log_dir = os.path.dirname(os.path.abspath(log_path))
            for _ in pods:
                fd, temp_path = tempfile.mkstemp(
                    prefix=_LOG_TEMP_PREFIX, dir=log_dir
                )
                temp_paths.append(temp_path)
                os.close(fd)
            executor = _get_log_fetch_executor()
            futures = [
                executor.submit(self._fetch_pod_log, pod, temp_path)
                for pod, temp_path in zip(pods, temp_paths)
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
        except k8s_client.KubeError as e:
            logger.error(
                "Failed to retrieve execution logs for Job '%s'. "
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
        """Extracts the job ID (Job name) from the experiment directory."""
        objects = _read_manifests(experiment_dir)
        return _find_job(objects, experiment_dir)["metadata"]["name"]

    def submit_experiments(self, experiment_dirs: list[str]) -> dict[str, str]:
        """Creates the Kubernetes objects of each experiment."""
        if not experiment_dirs:
            return {}
        objects_by_dir = {d: _read_manifests(d) for d in experiment_dirs}
        job_by_dir = {
            d: _find_job(objects, d) for d, objects in objects_by_dir.items()
        }
        jobs = list(job_by_dir.values())
        for job in jobs:
            job["spec"]["suspend"] = True
        try:
            existing_jobs = self._list_jobs()
        except k8s_client.KubeError as e:
            raise ChsSchedulerException(
                f"Failed to list the CHS Jobs: {e}"
            ) from e
        queue = [job["metadata"]["name"] for job in jobs]
        start_now = set(plan_admissions(existing_jobs + jobs, queue))
        for job in jobs:
            if job["metadata"]["name"] in start_now:
                job["spec"]["suspend"] = False
        logger.debug(
            "Creating %d experiments, %d of them start immediately.",
            len(experiment_dirs),
            len(start_now),
        )
        submission_order = sorted(
            experiment_dirs, key=lambda d: job_by_dir[d]["spec"]["suspend"]
        )
        self._create_objects(submission_order, objects_by_dir)
        return {job_by_dir[d]["metadata"]["name"]: d for d in experiment_dirs}

    @staticmethod
    def _create_objects(
        experiment_dirs: list[str], objects_by_dir: dict[str, list[dict]]
    ) -> None:
        executor = ThreadPoolExecutor(max_workers=_API_MAX_CONCURRENCY)
        try:
            futures = {
                executor.submit(_apply_objects, objects_by_dir[d]): d
                for d in experiment_dirs
            }
            for future in as_completed(futures):
                try:
                    future.result()
                except k8s_client.KubeError as e:
                    experiment = os.path.basename(futures[future])
                    raise ChsSchedulerException(
                        f"Failed to submit experiment '{experiment}': {e}"
                    ) from e
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    def _cleanup_dynamic_slices(self) -> None:
        """Deletes dynamic TPU slices created by CHS."""
        try:
            slices = k8s_client.list_custom_objects(
                group=_SLICE_GROUP,
                version=_SLICE_VERSION,
                plural="slices",
                label_selector="created-by=chs",
            )
            k8s_client.delete_custom_objects(
                group=_SLICE_GROUP,
                version=_SLICE_VERSION,
                plural="slices",
                object_names=[
                    tpu_slice["metadata"]["name"] for tpu_slice in slices
                ],
                wait_timeout_seconds=300,
            )
        except k8s_client.KubeError as e:
            if e.status == 404:
                return
            logger.warning(
                "Failed to delete dynamic TPU slices after execution. "
                "Manual cleanup may be required. Error: %s",
                e,
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
                slices = k8s_client.list_custom_objects(
                    group=_SLICE_GROUP,
                    version=_SLICE_VERSION,
                    plural="slices",
                    label_selector="created-by=chs",
                )
                if not slices:
                    time.sleep(10)
                    continue

                for tpu_slice in slices:
                    nodepool = (
                        tpu_slice.get("metadata", {})
                        .get("labels", {})
                        .get("chs-nodepool")
                    )
                    if not nodepool:
                        continue

                    conditions = (
                        tpu_slice.get("status", {}).get("conditions", [])
                        if tpu_slice.get("status")
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

            except k8s_client.KubeError as e:
                logger.debug("Failed to check status for dynamic slices: %s", e)
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
        """Returns partition IDs of all existing slices."""
        fulfilled = set()
        try:
            slices = k8s_client.list_custom_objects(
                group=_SLICE_GROUP, version=_SLICE_VERSION, plural="slices"
            )
            for tpu_slice in slices:
                if tpu_slice.get("metadata", {}).get("deletionTimestamp"):
                    continue
                for pid in tpu_slice.get("spec", {}).get("partitionIds", []):
                    fulfilled.add(pid)
        except k8s_client.KubeError as e:
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
                for slice_manifest in yaml.safe_load_all(all_slices_yaml):
                    if slice_manifest:
                        k8s_client.apply_custom_object(
                            plural="slices", body=slice_manifest
                        )
            except k8s_client.KubeError as e:
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

    def cleanup_scheduler_resources(self) -> None:
        """Cleans up scheduler-specific resources statelessly."""
        self._cleanup_dynamic_slices()

        if not self._delete_namespace():
            self._delete_job_secrets()

    def cancel_jobs(self, job_ids: list[str]) -> None:
        """Cancels the given Jobs by deleting them via the Kubernetes API."""
        if not job_ids:
            logger.warning("No job IDs to cancel.")
            return

        try:
            k8s_client.delete_jobs(
                object_names=job_ids,
                namespace=CHS_NAMESPACE,
                wait_timeout_seconds=DELETE_WAIT_TIMEOUT_SEC,
            )
            logger.debug("Successfully deleted Jobs: %s", ", ".join(job_ids))
        except k8s_client.KubeError as e:
            logger.error(
                "Encountered an error while attempting to "
                "delete active Jobs. Error: %s",
                e,
            )

        self._delete_job_secrets(job_ids)

    def get_active_job_names(self) -> list[str]:
        """Returns the names of active CHS Jobs on the cluster."""
        try:
            jobs = self._list_jobs()
            names = [
                job.get("metadata", {}).get("name")
                for job in jobs
                if not _job_condition(job, "Complete")
                and not _job_condition(job, "Failed")
            ]
            return [name for name in names if name]
        except k8s_client.KubeError as e:
            if e.status is None:
                raise ChsSchedulerException(
                    "Failed to list active Jobs; cannot verify that no "
                    f"other CHS run is in progress: {e}"
                ) from e
            logger.warning(
                "Failed to retrieve the list of active Jobs "
                "from the cluster. Error: %s",
                e,
            )
            return []

    def cancel_jobs_by_prefix(self, prefix: str) -> None:
        """Cancels all Jobs matching the specified prefix natively."""
        logger.debug("Attempting native cancel for prefix: %s", prefix)
        sanitized_match = prefix.replace("_", "-").replace(".", "-").lower()

        try:
            jobs = self._list_jobs()
            if not jobs:
                return

            job_names = [job.get("metadata", {}).get("name") for job in jobs]
            job_ids = [
                name for name in job_names if name and sanitized_match in name
            ]

            if job_ids:
                logger.debug(
                    "Found %d Jobs to cancel matching %s.",
                    len(job_ids),
                    prefix,
                )
                self.cancel_jobs(job_ids)
            else:
                logger.debug("No matching Jobs found for %s.", prefix)
        except k8s_client.KubeError as e:
            logger.warning(
                "Failed to execute native job cancellation "
                "via Kubernetes. Error: %s",
                e,
            )
