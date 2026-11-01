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
import hashlib
import importlib.resources as pkg_resources
import json
import math
import time
import logging
import os
import re
import shutil
import tempfile
import threading
import yaml
import random


from typing import Any, Callable
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass, field

from chs.schedulers.base import (
    CHS_JOB_PREFIX,
    BaseScheduler,
    TpuSliceSummary,
)
from chs.schedulers.gang_admission import (
    pinned_nodes,
    plan_admissions,
    pod_nodes,
)
from chs.utils import k8s_client
from chs.utils.formatting import format_names
from chs.exceptions import (
    ChsCancellationException,
    ChsSchedulerException,
    ChsValidationError,
)
from chs.utils.lifecycle import JobStatus
from chs import resources
from chs.constants import (
    CHS_NAMESPACE,
    COMPONENT_LABEL,
    DELETE_WAIT_TIMEOUT_SEC,
    DYNAMIC_SINGLE_HOST_TOPOLOGY,
    INSTANCE_LABEL,
    MANAGED_BY_CHS,
    MANAGED_BY_LABEL,
    NAMESPACE_DELETE_TIMEOUT_SEC,
    SUPPORTED_DYNAMIC_SLICE_TOPOLOGY,
    WORKER_COMPONENT,
)

_BUSY_NODES_BATCH_SIZE = 32
_BUSY_NODES_MAX_RETRIES = 3
_MAX_NODES_UNDER_TEST = 500

GANG_SCHEDULING_TIMEOUT_SECONDS = 300

_JOB_NAME_LABEL = "batch.kubernetes.io/job-name"
_JOB_COMPLETION_INDEX_LABEL = "batch.kubernetes.io/job-completion-index"
_SLICE_GROUP = "accelerator.gke.io"
_SLICE_VERSION = "v1beta1"
_MAX_SLICE_NAME_LENGTH = 49
_FAILED_SLICE_STATES = frozenset({"FAILED", "SliceCreationFailed"})
_UNUSABLE_SLICE_STATES = _FAILED_SLICE_STATES | {
    "ACTIVE_DEGRADED",
    "INCOMPLETE",
}
_CHS_SELECTOR = f"{MANAGED_BY_LABEL}={MANAGED_BY_CHS}"
_TOPOLOGY_MODE_LABEL = "cloud.google.com/gke-accelerator-topology-mode"
_TPU_ACCELERATOR_LABEL = "cloud.google.com/gke-tpu-accelerator"
_TPU_TOPOLOGY_LABEL = "cloud.google.com/gke-tpu-topology"
_PARTITION_LABEL_PREFIX = "cloud.google.com/gke-tpu-partition-"
_TPU7X_PARTITION_TOPOLOGIES = ("2x2x1", "2x2x2", "2x2x4", "2x4x4", "4x4x4")
_PARTITION_ID_LABELS = tuple(
    f"{_PARTITION_LABEL_PREFIX}{topology}-id"
    for topology in _TPU7X_PARTITION_TOPOLOGIES
)
_POOL_PARTITION_ID_LABEL = (
    f"{_PARTITION_LABEL_PREFIX}{SUPPORTED_DYNAMIC_SLICE_TOPOLOGY}-id"
)
_POOL_PARTITION_STATE_LABEL = (
    f"{_PARTITION_LABEL_PREFIX}{SUPPORTED_DYNAMIC_SLICE_TOPOLOGY}-state"
)
_SINGLE_HOST_PARTITION_ID_LABEL = (
    f"{_PARTITION_LABEL_PREFIX}{DYNAMIC_SINGLE_HOST_TOPOLOGY}-id"
)
_SINGLE_HOST_PARTITION_STATE_LABEL = (
    f"{_PARTITION_LABEL_PREFIX}{DYNAMIC_SINGLE_HOST_TOPOLOGY}-state"
)
_MIN_SUBSLICE_GKE_VERSION = "1.36.0-gke.3712000"
_GKE_VERSION_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)-gke\.(\d+)")
_COMPUTE_DOMAIN_GROUP = "resource.nvidia.com"
_COMPUTE_DOMAIN_VERSION = "v1beta1"

_GCE_PROVIDER_ID_RE = re.compile(r"^gce://([^/]+)/([^/]+)/([^/]+)$")

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
    manifest_path = os.path.join(experiment_dir, "job.yaml")
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
        f"Job kind not found in {experiment_dir}/job.yaml"
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


def _with_retries(call: Callable[[], None], description: str) -> None:
    """Runs the Kubernetes API call, retrying throttling and server errors."""
    for attempt in range(1, _APPLY_ATTEMPTS + 1):
        try:
            call()
            return
        except k8s_client.KubeError as e:
            if attempt == _APPLY_ATTEMPTS or not _is_retryable(e):
                raise
            delay = 2 ** (attempt - 1)
            logger.debug(
                "Retrying %s in %d seconds. Error: %s", description, delay, e
            )
            time.sleep(delay)


def _apply_objects(objects: list[dict]) -> None:
    for obj in objects:
        _with_retries(
            lambda: _apply_object(obj),
            f"{obj['kind']} '{obj['metadata']['name']}'",
        )


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


def _is_single_host_slice(
    topology: str | None, allocatable: dict | None
) -> bool:
    try:
        chips = math.prod(int(dim) for dim in (topology or "").split("x"))
        return chips <= int((allocatable or {}).get("google.com/tpu", 0))
    except Exception:
        return False


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
        and labels.get(MANAGED_BY_LABEL) == MANAGED_BY_CHS
        and labels.get(COMPONENT_LABEL) == WORKER_COMPONENT
    )


def _run_labels(run_id: str) -> dict[str, str]:
    """Returns the labels of every object CHS creates for the run."""
    return {MANAGED_BY_LABEL: MANAGED_BY_CHS, INSTANCE_LABEL: run_id}


def _run_selector(run_id: str) -> str:
    """Returns the label selector of every object CHS creates for the run."""
    return ",".join(f"{k}={v}" for k, v in _run_labels(run_id).items())


def _object_names(objects: list[dict]) -> list[str]:
    """Returns the names of the Kubernetes objects that have one."""
    names = (obj.get("metadata", {}).get("name") for obj in objects)
    return [name for name in names if name]


def _slice_name(slice_key: str) -> str:
    """Returns a unique Slice name within the GKE Slice name length limit.

    Long keys keep their tail, which tells nodes of one pool apart.
    """
    prefix = "chs-slice-"
    name = f"{prefix}{slice_key}"
    if len(name) <= _MAX_SLICE_NAME_LENGTH:
        return name
    digest = hashlib.sha256(slice_key.encode()).hexdigest()[:8]
    room = _MAX_SLICE_NAME_LENGTH - len(prefix) - len(digest) - 1
    return f"{prefix}{slice_key[-room:].strip('-.')}-{digest}"


def _slice_state(tpu_slice: dict) -> str | None:
    """Returns the lifecycle state GKE reports in the Slice's Ready reason."""
    conditions = (tpu_slice.get("status") or {}).get("conditions") or []
    ready = next((c for c in conditions if c.get("type") == "Ready"), {})
    return ready.get("reason")


def _holds_partitions(tpu_slice: dict) -> bool:
    """Returns True unless the Slice failed to form and left its hardware."""
    return _slice_state(tpu_slice) not in _FAILED_SLICE_STATES


def _partition_ids(node: dict) -> set[str]:
    """Returns the IDs of every TPU partition, of any shape, the node is in."""
    labels = node.get("metadata", {}).get("labels") or {}
    return {labels[label] for label in _PARTITION_ID_LABELS if label in labels}


def _gke_version(version: str) -> tuple[int, ...]:
    """Returns the comparable parts of a GKE version, or () if unparsable."""
    match = _GKE_VERSION_RE.match(version)
    return tuple(int(part) for part in match.groups()) if match else ()


def _record_subslice_node(
    data: "TpuSliceData", node_name: str, labels: dict[str, str]
) -> None:
    """Records a dynamic node that --nodes tests on its own sub-slice.

    Only the node's own 2x2x1 partition health matters, so the healthy
    hosts of a degraded 4x4x4 cube can still be tested.
    """
    data.provision_only_nodes.add(node_name)
    partition_id = labels.get(_SINGLE_HOST_PARTITION_ID_LABEL)
    if (
        partition_id
        and labels.get(_SINGLE_HOST_PARTITION_STATE_LABEL) == "HEALTHY"
    ):
        data.single_host_slices[node_name] = SliceSpec(
            accelerator=labels.get(_TPU_ACCELERATOR_LABEL, ""),
            topology=DYNAMIC_SINGLE_HOST_TOPOLOGY,
            partition_id=partition_id,
        )


def _record_pool_node(
    data: "TpuSliceData",
    unhealthy_pools: set[str],
    node_name: str,
    labels: dict[str, str],
) -> None:
    """Records a node that is tested as part of its whole node pool."""
    nodepool = labels.get("cloud.google.com/gke-nodepool")
    if not nodepool:
        return
    tpu_type = labels.get(_TPU_ACCELERATOR_LABEL)
    topology = labels.get(_TPU_TOPOLOGY_LABEL)

    if labels.get(_TOPOLOGY_MODE_LABEL) == "PROVISION_ONLY":
        matched_state = labels.get(_POOL_PARTITION_STATE_LABEL)

        if matched_state == "HEALTHY":
            if nodepool not in unhealthy_pools:
                data.pools[nodepool].append(node_name)
                data.node_to_pool[node_name] = nodepool
                data.pool_to_topology[nodepool] = (
                    SUPPORTED_DYNAMIC_SLICE_TOPOLOGY
                )
                partition_id = labels.get(_POOL_PARTITION_ID_LABEL)
                if partition_id:
                    data.dynamic_slice_partitions[nodepool] = partition_id
                if tpu_type:
                    data.pool_to_accelerator[nodepool] = tpu_type
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
            data.purge_pool(nodepool)
    elif not topology:
        unhealthy_pools.add(nodepool)
        data.purge_pool(nodepool)
    elif nodepool not in unhealthy_pools:
        data.pools[nodepool].append(node_name)
        data.node_to_pool[node_name] = nodepool
        if nodepool not in data.pool_to_topology:
            data.pool_to_topology[nodepool] = topology


def _read_slice_template() -> str:
    template_path = pkg_resources.files("chs.schedulers").joinpath(
        "slice.yaml.tpl"
    )
    try:
        with pkg_resources.as_file(template_path) as p:
            with open(p, "r") as f:
                return f.read()
    except OSError as e:
        raise ChsSchedulerException(f"Failed to read slice template: {e}")


def worker_service_name(run_id: str) -> str:
    """Returns the name of the headless Service of the run's worker pods."""
    return f"{CHS_JOB_PREFIX}{run_id}"


def _worker_service(run_id: str) -> dict:
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": worker_service_name(run_id),
            "namespace": CHS_NAMESPACE,
            "labels": _run_labels(run_id),
        },
        "spec": {
            "clusterIP": "None",
            "publishNotReadyAddresses": True,
            "selector": {
                **_run_labels(run_id),
                COMPONENT_LABEL: WORKER_COMPONENT,
            },
        },
    }


@dataclass(frozen=True)
class SliceSpec:
    """The Slice CR that makes the nodes of a slice key schedulable."""

    accelerator: str
    topology: str
    partition_id: str


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
    provision_only_nodes: set[str] = field(default_factory=set)
    single_host_slices: dict[str, SliceSpec] = field(default_factory=dict)
    single_host_pools: set[str] = field(default_factory=set)

    def purge_pool(self, pool: str) -> None:
        """Forgets the pool and its nodes so that no test targets them."""
        for node in self.pools.pop(pool, []):
            self.node_to_pool.pop(node, None)
        self.pool_to_topology.pop(pool, None)
        self.dynamic_slice_partitions.pop(pool, None)
        self.pool_to_accelerator.pop(pool, None)
        self.single_host_pools.discard(pool)


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
        self._subslice_nodes: set[str] = set()
        self._run_id: str | None = None
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
            match = _GCE_PROVIDER_ID_RE.match(provider_id)
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
            node_object = objects_by_name.get(node, {})
            provider_id = node_object.get("spec", {}).get("providerID", "")
            gce_instance = _GCE_PROVIDER_ID_RE.match(provider_id.strip())
            if gce_instance:
                self.project_id = gce_instance.group(1)
                self.node_to_zone[node] = gce_instance.group(2)
            metadata = node_object.get("metadata", {})
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
        """Fails fast if --nodes targets dynamic nodes that cannot sub-slice.

        Raises:
          ChsValidationError: If a targeted node of a provision-only TPU node
            pool runs a GKE version older than dynamic sub-slicing.
        """
        if not self.nodes_arg:
            return tests
        minimum = _gke_version(_MIN_SUBSLICE_GKE_VERSION)
        for node in self._get_node_objects(self.resolve_raw_nodes()):
            labels = node["metadata"].get("labels") or {}
            node_info = (node.get("status") or {}).get("nodeInfo") or {}
            version = node_info.get("kubeletVersion", "")
            if (
                labels.get(_TOPOLOGY_MODE_LABEL) == "PROVISION_ONLY"
                and _gke_version(version) < minimum
            ):
                raise ChsValidationError(
                    f"Node '{node['metadata']['name']}' of a provision-only "
                    f"TPU node pool runs GKE '{version}', but --nodes needs "
                    f"GKE {_MIN_SUBSLICE_GKE_VERSION} or later to "
                    "create single-host slices. Upgrade its node pool, or "
                    "test the whole pool with --nodepools."
                )
        return tests

    def setup_scheduler_resources(
        self,
        workspace_config_dir: str,
        tests: list[dict[str, Any]] | None = None,
        machine_type: str = "",
        workspace_name: str | None = None,
        run_id: str | None = None,
    ) -> list[dict[str, Any]] | None:
        """Sets up scheduler-specific resources."""
        self._run_id = run_id
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
        """Ensures the chs namespace and the run's worker Service exist."""
        k8s_client.create_namespace_if_not_exists(name=CHS_NAMESPACE)
        k8s_client.apply_service(body=_worker_service(self._run_id))

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

    def _delete_job_dependents(self, job_ids: list[str]) -> None:
        """Deletes the SSH Secrets and ComputeDomains of the given Jobs."""
        for job_id in job_ids:
            self._delete_secret(f"{job_id}-ssh")
        self._delete_compute_domains_by_name(
            [f"{job_id}-cd" for job_id in job_ids]
        )

    def _delete_secret(self, name: str) -> bool:
        """Deletes the named Secret and returns whether that succeeded."""
        try:
            k8s_client.delete_secret(name=name, namespace=CHS_NAMESPACE)
        except k8s_client.KubeError as e:
            logger.warning(
                "Failed to delete Kubernetes secret '%s'. Error: %s", name, e
            )
            return False
        return True

    def _delete_secrets(self, label_selector: str) -> bool:
        """Deletes the Secrets that match label_selector.

        Returns:
          Whether every matching Secret was deleted.
        """
        try:
            secrets = k8s_client.list_secrets(
                namespace=CHS_NAMESPACE, label_selector=label_selector
            )
        except k8s_client.KubeError as e:
            logger.warning(
                "Failed to list Kubernetes secrets. Error: %s",
                e,
            )
            return False
        results = [self._delete_secret(name) for name in _object_names(secrets)]
        return all(results)

    def _delete_compute_domains_by_name(self, names: list[str]) -> bool:
        """Deletes the named ComputeDomains; returns whether it succeeded."""
        try:
            k8s_client.delete_custom_objects(
                group=_COMPUTE_DOMAIN_GROUP,
                version=_COMPUTE_DOMAIN_VERSION,
                plural=_CUSTOM_RESOURCE_PLURALS["ComputeDomain"],
                namespace=CHS_NAMESPACE,
                object_names=names,
                wait_timeout_seconds=DELETE_WAIT_TIMEOUT_SEC,
            )
        except k8s_client.KubeError as e:
            logger.warning(
                "Failed to delete Kubernetes ComputeDomains. Error: %s",
                e,
            )
            return False
        return True

    def _delete_compute_domains(self, label_selector: str) -> bool:
        """Deletes the ComputeDomains that match label_selector.

        Returns:
          Whether every matching ComputeDomain was deleted.
        """
        try:
            domains = k8s_client.list_custom_objects(
                group=_COMPUTE_DOMAIN_GROUP,
                version=_COMPUTE_DOMAIN_VERSION,
                plural=_CUSTOM_RESOURCE_PLURALS["ComputeDomain"],
                namespace=CHS_NAMESPACE,
                label_selector=label_selector,
            )
        except k8s_client.KubeError as e:
            if e.status == 404:
                return True
            logger.warning(
                "Failed to list Kubernetes ComputeDomains. Error: %s",
                e,
            )
            return False
        return self._delete_compute_domains_by_name(_object_names(domains))

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
        """Groups TPU nodes by their node pool."""
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

        data = TpuSliceData()
        unhealthy_pools = set()

        for node in nodes:
            metadata = node.get("metadata", {})
            node_name = metadata.get("name")
            labels = metadata.get("labels", {})
            allocatable = (node.get("status") or {}).get("allocatable")
            is_tpu = labels.get(_TPU_ACCELERATOR_LABEL) or _has_accelerator(
                allocatable, "tpu"
            )
            if not node_name or not is_tpu:
                continue

            data.all_tpu_nodes.add(node_name)
            is_provision_only = (
                labels.get(_TOPOLOGY_MODE_LABEL) == "PROVISION_ONLY"
            )
            if self.nodes_arg and is_provision_only:
                _record_subslice_node(data, node_name, labels)
            else:
                _record_pool_node(data, unhealthy_pools, node_name, labels)
                pool = data.node_to_pool.get(node_name)
                if (
                    pool
                    and not is_provision_only
                    and _is_single_host_slice(
                        labels.get(_TPU_TOPOLOGY_LABEL), allocatable
                    )
                ):
                    data.single_host_pools.add(pool)

        if unhealthy_pools:
            logger.warning(
                "Excluded the following unhealthy dynamic TPU pools from "
                "diagnostics: %s. Please check the TPU node health.",
                ", ".join(unhealthy_pools),
            )

        self._tpu_slice_data = data
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

        busy_nodes = self._get_busy_nodes(target_nodes=raw_nodes)
        excluded_nodes = busy_nodes | self._get_not_ready_nodes()
        for run_id, run_nodes in sorted(
            self.find_runs_targeting_nodes(raw_nodes).items()
        ):
            excluded_nodes.update(run_nodes)
            logger.warning(
                "Skipping %d node(s) targeted by CHS run '%s'.",
                len(run_nodes),
                run_id,
            )

        topology = self._get_tpu_slices()
        dynamic_pools = set(topology.dynamic_slice_partitions)
        requested_pools: list[str] = []
        subslice_candidates: list[str] = []
        self.partially_busy_slices = {}

        for node in raw_nodes:
            pool = topology.node_to_pool.get(node)
            if self.nodes_arg and node in topology.provision_only_nodes:
                subslice_candidates.append(node)
            elif pool:
                if pool not in requested_pools:
                    requested_pools.append(pool)
            elif node in topology.all_tpu_nodes:
                excluded_nodes.add(node)
                logger.warning(
                    "Excluded TPU node '%s' from testing because "
                    "its parent slice is unhealthy.",
                    node,
                )

        if subslice_candidates:
            excluded_nodes |= self._nodes_unfit_for_subslice(
                [n for n in subslice_candidates if n not in excluded_nodes]
            )
            self._subslice_nodes = {
                n for n in subslice_candidates if n not in excluded_nodes
            }

        for pool in requested_pools:
            busy = sorted(set(topology.pools[pool]) & excluded_nodes)
            if busy:
                self.partially_busy_slices[pool] = busy

        if self.require_whole_slices:
            for pool, busy in self.partially_busy_slices.items():
                logger.warning(
                    "Cannot run multi-host test(s) on slice '%s': the "
                    "following node(s) are busy or unavailable: %s",
                    pool,
                    format_names(busy),
                )
            valid_nodes = [
                n
                for n in raw_nodes
                if n not in excluded_nodes
                and topology.node_to_pool.get(n)
                not in self.partially_busy_slices
            ]
        else:
            if busy_nodes:
                logger.info(
                    "Identified %d active node(s) with existing workloads. "
                    "These nodes will be skipped to avoid disrupting running "
                    "tasks.",
                    len(busy_nodes),
                )
            busy_dynamic_pools = [
                p for p in self.partially_busy_slices if p in dynamic_pools
            ]
            for pool in busy_dynamic_pools:
                busy = self.partially_busy_slices.pop(pool)
                logger.warning(
                    "Cannot run tests on dynamic TPU pool '%s': dynamic "
                    "slicing requires the entire pool to be available to "
                    "provision the Slice CR, but the following node(s) "
                    "are busy or unavailable: %s",
                    pool,
                    format_names(busy),
                )
                excluded_nodes.update(topology.pools[pool])
            valid_nodes = [n for n in raw_nodes if n not in excluded_nodes]

        self._resolved_nodes = valid_nodes
        return self._resolved_nodes

    def _nodes_unfit_for_subslice(self, nodes: list[str]) -> set[str]:
        """Returns the dynamic nodes that cannot get their own sub-slice.

        A node is unfit if its 2x2x1 partition is not HEALTHY, or if it is
        already in a Slice, which GKE marks with the TPU topology label.
        """
        single_host_slices = self._get_tpu_slices().single_host_slices
        unhealthy = [n for n in nodes if n not in single_host_slices]
        if unhealthy:
            logger.warning(
                "Excluded node(s) %s from testing because their %s TPU "
                "partition is not HEALTHY or not labelled on the node.",
                format_names(unhealthy),
                DYNAMIC_SINGLE_HOST_TOPOLOGY,
            )
        in_slice = [
            node["metadata"]["name"]
            for node in self._get_node_objects(nodes)
            if node["metadata"]["name"] in single_host_slices
            and _TPU_TOPOLOGY_LABEL in (node["metadata"].get("labels") or {})
        ]
        if in_slice:
            logger.warning(
                "Excluded node(s) %s from testing because they are already "
                "part of a TPU slice.",
                format_names(in_slice),
            )
        return set(unhealthy) | set(in_slice)

    def get_tpu_slice_summary(self, nodes: list[str]) -> TpuSliceSummary | None:
        slices = self._tpu_slice_data
        if not slices:
            return None
        tested = set(nodes)
        subslice_nodes = tested & self._subslice_nodes
        pools = {
            slices.node_to_pool[n] for n in tested if n in slices.node_to_pool
        }
        if not pools and not subslice_nodes:
            return None
        topologies = Counter()
        if subslice_nodes:
            topologies[DYNAMIC_SINGLE_HOST_TOPOLOGY] += len(subslice_nodes)
        for pool in pools:
            topology = slices.pool_to_topology.get(pool, "unknown")
            if pool in slices.single_host_pools:
                topologies[topology] += len(tested & set(slices.pools[pool]))
            else:
                topologies[topology] += 1
        is_dynamic = bool(subslice_nodes) or not pools.isdisjoint(
            slices.dynamic_slice_partitions
        )
        return TpuSliceSummary(
            slice_type="dynamic" if is_dynamic else "static",
            topologies=dict(sorted(topologies.items())),
            nodepool_count=None if self.nodes_arg else len(pools),
        )

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
        variables["worker_service_name"] = worker_service_name(run_id)
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
            single_host_pools = [
                pool
                for pool, pool_nodes in tpu_slices.items()
                if self._subslice_nodes.issuperset(pool_nodes)
            ]
            for pool in single_host_pools:
                slice_topologies[pool] = DYNAMIC_SINGLE_HOST_TOPOLOGY
            variables["tpu_slice_topologies"] = slice_topologies

            dynamic_pools = [
                pool
                for pool in tpu_slices
                if pool in slice_data.dynamic_slice_partitions
            ]
            variables["tpu_dynamic_pools"] = dynamic_pools
            variables["tpu_single_host_pools"] = single_host_pools

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

    def _list_slices(self, label_selector: str | None = None) -> list[dict]:
        """Returns the Slices that match label_selector.

        A cluster without the Slice CRD has no Slices.

        Raises:
          ChsSchedulerException: If the Slices cannot be listed.
        """
        try:
            return k8s_client.list_custom_objects(
                group=_SLICE_GROUP,
                version=_SLICE_VERSION,
                plural="slices",
                label_selector=label_selector,
            )
        except k8s_client.KubeError as e:
            if e.status == 404:
                logger.debug("The Slice CRD is not installed: %s", e)
                return []
            raise ChsSchedulerException(
                f"Failed to list the dynamic TPU slices: {e}"
            ) from e

    def _list_slice_names(self, label_selector: str) -> list[str] | None:
        """Returns the names of the dynamic TPU slices matching the selector.

        Returns:
          The slice names, or None if the Slices cannot be listed.
        """
        try:
            slices = self._list_slices(label_selector)
        except ChsSchedulerException as e:
            logger.warning("%s", e)
            return None
        return _object_names(slices)

    def _delete_slices(self, names: list[str]) -> bool:
        """Deletes the named dynamic TPU slices.

        Returns:
          Whether the deletion succeeded.
        """
        if not names:
            return True
        try:
            k8s_client.delete_custom_objects(
                group=_SLICE_GROUP,
                version=_SLICE_VERSION,
                plural="slices",
                object_names=names,
                wait_timeout_seconds=300,
            )
        except k8s_client.KubeError as e:
            logger.warning(
                "Failed to delete dynamic TPU slices. "
                "Manual cleanup may be required. Error: %s",
                e,
            )
            return False
        return True

    def _wait_for_slices_active(
        self, slice_keys: list[str], timeout_sec: int = 300
    ) -> tuple[list[str], list[str]]:
        """Waits for dynamically provisioned slices to reach terminal states."""
        if not slice_keys:
            return [], []
        start_time = time.time()
        key_of_slice = {_slice_name(key): key for key in slice_keys}
        slice_terminal_statuses = {}

        while time.time() - start_time < timeout_sec:
            try:
                slices = self._list_slices(_run_selector(self._run_id))
                if not slices:
                    time.sleep(10)
                    continue

                for tpu_slice in slices:
                    slice_key = key_of_slice.get(
                        tpu_slice.get("metadata", {}).get("name")
                    )
                    if not slice_key:
                        continue

                    state = _slice_state(tpu_slice)
                    if state == "ACTIVE" or state in _UNUSABLE_SLICE_STATES:
                        slice_terminal_statuses[slice_key] = state

                if len(slice_terminal_statuses) >= len(slice_keys):
                    break

            except ChsSchedulerException as e:
                logger.debug("Failed to check status for dynamic slices: %s", e)
            time.sleep(10)

        active = [
            key
            for key, status in slice_terminal_statuses.items()
            if status == "ACTIVE"
        ]
        failed = [key for key in slice_keys if key not in active]
        return active, failed

    def _existing_slice_states(self) -> dict[str, str | None]:
        """Maps partitions held by Slices this run may reuse to their state.

        Slices being deleted and Slices of other CHS runs are left out.
        """
        states = {}
        try:
            slices = self._list_slices()
        except ChsSchedulerException as e:
            logger.debug("Could not retrieve existing slices: %s", e)
            return states
        for tpu_slice in filter(_holds_partitions, slices):
            metadata = tpu_slice.get("metadata", {})
            labels = metadata.get("labels") or {}
            owned_by_other_run = (
                labels.get(MANAGED_BY_LABEL) == MANAGED_BY_CHS
                and labels.get(INSTANCE_LABEL) != self._run_id
            )
            if metadata.get("deletionTimestamp") or owned_by_other_run:
                continue
            for partition_id in tpu_slice.get("spec", {}).get(
                "partitionIds", []
            ):
                states[partition_id] = _slice_state(tpu_slice)
        return states

    def _create_slice(self, slice_manifest: dict, slice_key: str) -> None:
        """Creates the Slice of the slice key.

        Raises:
          ChsSchedulerException: If the Slice exists or cannot be created.
        """
        name = slice_manifest.get("metadata", {}).get("name")
        try:
            _with_retries(
                lambda: k8s_client.create_custom_object(
                    plural="slices", body=slice_manifest
                ),
                f"Slice '{name}'",
            )
        except k8s_client.KubeError as e:
            if e.status == 409:
                if self._is_own_slice(name):
                    return
                raise ChsSchedulerException(
                    f"Dynamic slice '{name}' already exists; another CHS run "
                    f"may be using {slice_key}. Wait for that run to "
                    "finish or cancel it with `chs cancel <run_id>`."
                ) from e
            raise ChsSchedulerException(
                f"Failed to create dynamic slice '{name}': {e}"
            ) from e

    def _is_own_slice(self, name: str) -> bool:
        """Returns True if this run already created the named Slice.

        Creating a Slice is not idempotent: if a retried create's first
        attempt reached the API server but its response was lost, the next
        attempt gets 409 on the Slice this run just created.
        """
        own_slices = self._list_slice_names(_run_selector(self._run_id))
        return name in (own_slices or [])

    def _dynamic_slice_spec(self, slice_key: str) -> SliceSpec | None:
        """Returns the Slice that the slice key needs, if it is dynamic."""
        data = self._get_tpu_slices()
        if slice_key in self._subslice_nodes:
            return data.single_host_slices[slice_key]
        if slice_key in data.dynamic_slice_partitions:
            return SliceSpec(
                accelerator=data.pool_to_accelerator.get(slice_key, ""),
                topology=data.pool_to_topology.get(slice_key, ""),
                partition_id=data.dynamic_slice_partitions[slice_key],
            )
        return None

    def _slice_manifest(
        self, template: str, slice_key: str, spec: SliceSpec
    ) -> dict:
        manifest = yaml.safe_load(
            template.format(
                slice_name=_slice_name(slice_key),
                tpu_type=spec.accelerator,
                shape=spec.topology,
                partition_id=spec.partition_id,
            )
        )
        manifest["metadata"]["labels"] = _run_labels(self._run_id)
        return manifest

    def _create_slices(self, manifests: dict[str, dict]) -> list[str]:
        """Creates the Slices and returns the sub-slice keys that failed.

        Raises:
          ChsSchedulerException: If the Slice of a node pool cannot be
            created.
        """
        failed_subslices = set()
        executor = ThreadPoolExecutor(
            max_workers=min(len(manifests), _API_MAX_CONCURRENCY)
        )
        try:
            futures = {
                executor.submit(self._create_slice, manifest, slice_key): (
                    slice_key
                )
                for slice_key, manifest in manifests.items()
            }
            for future in as_completed(futures):
                slice_key = futures[future]
                try:
                    future.result()
                except ChsSchedulerException as e:
                    if slice_key not in self._subslice_nodes:
                        raise
                    logger.warning("%s", e)
                    failed_subslices.add(slice_key)
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
        return [key for key in manifests if key in failed_subslices]

    def _provision_dynamic_slices(
        self, target_slices: dict[str, list[str]]
    ) -> tuple[list[str], list[str]]:
        """Provisions dynamic TPU slices; returns the active and failed keys.

        Only ACTIVE existing Slices are reused. Slices this run created that
        do not become ACTIVE are deleted.
        """
        if not (self._tpu_slice_data and self.workspace_config_dir):
            return [], []
        specs = {
            slice_key: spec
            for slice_key in target_slices
            if (spec := self._dynamic_slice_spec(slice_key))
        }
        if not specs:
            return [], []

        existing_states = self._existing_slice_states()
        reused = []
        occupied = []
        for slice_key, spec in specs.items():
            if spec.partition_id not in existing_states:
                continue
            state = existing_states[spec.partition_id]
            if state == "ACTIVE":
                reused.append(slice_key)
            else:
                occupied.append(slice_key)
                logger.warning(
                    "Cannot provision a dynamic slice for '%s': its TPU "
                    "partition is held by an existing slice in state %s.",
                    slice_key,
                    state or "unknown",
                )
        template = _read_slice_template()
        manifests = {
            slice_key: self._slice_manifest(template, slice_key, spec)
            for slice_key, spec in specs.items()
            if spec.partition_id not in existing_states
        }
        if not manifests:
            return reused, occupied

        slice_yaml_path = os.path.join(
            self.workspace_config_dir, "chs-slices.yaml"
        )
        with open(slice_yaml_path, "w") as f:
            yaml.safe_dump_all(manifests.values(), f, sort_keys=False)

        failed_to_create = self._create_slices(manifests)
        created = [key for key in manifests if key not in failed_to_create]
        active, inactive = self._wait_for_slices_active(created, 300)
        self._delete_slices([_slice_name(key) for key in inactive])
        return active + reused, occupied + failed_to_create + inactive

    def apply_scheduler_resources(self) -> None:
        """Applies generated scheduler resources to the cluster."""
        target_slices = self._slices_to_provision(self.get_nodes_under_test())
        _, failed_slices = self._provision_dynamic_slices(target_slices)
        if failed_slices and self._tpu_slice_data:
            self._drop_failed_slices(failed_slices, target_slices)

        self._ensure_namespace_exists()

    def _slices_to_provision(self, nodes: list[str]) -> dict[str, list[str]]:
        """Maps each Slice key to its nodes: a pool, or a sub-sliced node."""
        pool_nodes = [n for n in nodes if n not in self._subslice_nodes]
        slices = self._group_nodes_by_slice(pool_nodes)
        for node in nodes:
            if node in self._subslice_nodes:
                slices[node] = [node]
        return slices

    def _drop_failed_slices(
        self, failed_slices: list[str], target_slices: dict[str, list[str]]
    ) -> None:
        """Stops testing the nodes of the slices that could not be used.

        Raises:
          ChsSchedulerException: If no node is left to test.
        """
        failed_subslices = [
            key for key in failed_slices if key in self._subslice_nodes
        ]
        failed_pools = [
            key for key in failed_slices if key not in self._subslice_nodes
        ]
        for pool in failed_pools:
            self._tpu_slice_data.purge_pool(pool)
        self._subslice_nodes.difference_update(failed_subslices)

        nodes_to_remove = {
            node for key in failed_slices for node in target_slices.get(key, [])
        }
        if self._resolved_nodes is not None:
            self._resolved_nodes = [
                n for n in self._resolved_nodes if n not in nodes_to_remove
            ]
            if not self._resolved_nodes:
                raise ChsSchedulerException(
                    "All targeted dynamic slices failed to provision: "
                    f"{', '.join(failed_slices)}."
                )

        if failed_pools:
            logger.warning(
                "Dynamic slice provisioning failed for nodepools: %s. "
                "Tests will skip these and run on the remaining active "
                "slices.",
                ", ".join(failed_pools),
            )
        if failed_subslices:
            logger.warning(
                "Excluded node(s) %s from testing because their %s "
                "single-host slices could not be provisioned. Tests will "
                "run on the remaining nodes.",
                format_names(failed_subslices),
                DYNAMIC_SINGLE_HOST_TOPOLOGY,
            )

    def find_runs_targeting_nodes(
        self, nodes: list[str]
    ) -> dict[str, list[str]]:
        """Finds the CHS runs whose Jobs or dynamic Slices target the nodes.

        Raises:
          ChsSchedulerException: If the Jobs, Slices or nodes cannot be
            listed.
        """
        if not nodes:
            return {}
        requested = set(nodes)
        try:
            targeted_by_jobs = self._runs_targeting_by_jobs(requested)
            targeted_by_slices = self._runs_targeting_by_slices(requested)
        except (k8s_client.KubeError, ChsSchedulerException) as e:
            raise ChsSchedulerException(
                "Failed to list the CHS Jobs, Slices or nodes; cannot verify "
                f"that the nodes are not targeted by another CHS run: {e}"
            ) from e
        targeted = defaultdict(set)
        for runs in (targeted_by_jobs, targeted_by_slices):
            for run_id, run_nodes in runs.items():
                targeted[run_id] |= run_nodes
        return {
            run_id: sorted(run_nodes)
            for run_id, run_nodes in targeted.items()
            if run_nodes
        }

    def _runs_targeting_by_jobs(
        self, requested: set[str]
    ) -> dict[str, set[str]]:
        """Maps run IDs to the requested nodes their Jobs are pinned to."""
        jobs = k8s_client.list_jobs(
            namespace=CHS_NAMESPACE, label_selector=_CHS_SELECTOR
        )
        run_of_job = {}
        named_jobs = []
        for job in jobs:
            metadata = job.get("metadata", {})
            name = metadata.get("name")
            run_id = (metadata.get("labels") or {}).get(INSTANCE_LABEL)
            if name and run_id:
                run_of_job[name] = run_id
                named_jobs.append(job)
        targeted = defaultdict(set)
        for job_name, job_nodes in pinned_nodes(named_jobs, run_of_job).items():
            targeted[run_of_job[job_name]] |= job_nodes & requested
        return targeted

    def _runs_targeting_by_slices(
        self, requested: set[str]
    ) -> dict[str, set[str]]:
        """Maps run IDs to the requested nodes in their Slices' partitions."""
        run_of_partition = {}
        for tpu_slice in self._list_slices(_CHS_SELECTOR):
            labels = tpu_slice.get("metadata", {}).get("labels") or {}
            run_id = labels.get(INSTANCE_LABEL)
            if not run_id:
                continue
            for partition_id in tpu_slice.get("spec", {}).get(
                "partitionIds", []
            ):
                run_of_partition[partition_id] = run_id

        targeted = defaultdict(set)
        if not run_of_partition:
            return targeted
        node_objects = self._resolved_node_resources
        if node_objects is None:
            node_objects = k8s_client.list_nodes()
        for node in node_objects:
            node_name = node.get("metadata", {}).get("name")
            if node_name not in requested:
                continue
            for partition_id in _partition_ids(node) & run_of_partition.keys():
                targeted[run_of_partition[partition_id]].add(node_name)
        return targeted

    def cancel_run(self, run_id: str) -> bool:
        """Deletes the run's Jobs and every object CHS created for it.

        Objects are selected by the run's labels. Every kind of object is
        attempted even if another kind fails to delete. The chs namespace is
        kept for the other runs.

        Returns:
          Whether the run was active, i.e. had Jobs or Slices on the cluster.

        Raises:
          ChsCancellationException: If the run's Jobs cannot be listed or
            some of its objects could not be listed or deleted.
        """
        selector = _run_selector(run_id)
        job_names = self._list_job_names(selector)
        slice_names = self._list_slice_names(selector)
        deletions = {
            "Jobs": self._delete_jobs(job_names),
            "Secrets": self._delete_secrets(selector),
            "ComputeDomains": self._delete_compute_domains(selector),
            "dynamic slices": (
                slice_names is not None and self._delete_slices(slice_names)
            ),
            "worker Service": self._delete_worker_service(run_id),
        }
        failed = [kind for kind, deleted in deletions.items() if not deleted]
        if failed:
            raise ChsCancellationException(
                f"Failed to delete the {', '.join(failed)} of CHS run "
                f"'{run_id}'. Run `chs cancel {run_id}` again to retry."
            )
        return bool(job_names or slice_names)

    def cancel_all_runs(self) -> None:
        """Deletes all CHS Jobs and Slices and the chs namespace.

        Raises:
          ChsCancellationException: If the CHS Jobs cannot be listed or
            some CHS objects could not be listed or deleted.
        """
        job_names = self._list_job_names(_CHS_SELECTOR)
        slice_names = self._list_slice_names(_CHS_SELECTOR)
        deletions = {
            "Jobs": self._delete_jobs(job_names),
            "dynamic slices": (
                slice_names is not None and self._delete_slices(slice_names)
            ),
            "chs namespace": self._delete_namespace(),
        }
        if not deletions["chs namespace"]:
            self._delete_secrets(_CHS_SELECTOR)
            self._delete_compute_domains(_CHS_SELECTOR)
        failed = [kind for kind, deleted in deletions.items() if not deleted]
        if failed:
            raise ChsCancellationException(
                f"Failed to delete the CHS {', '.join(failed)}. Run "
                "`chs cancel --all` again to retry."
            )

    def cancel_jobs(self, job_ids: list[str]) -> None:
        """Deletes the given Jobs and their Secrets and ComputeDomains."""
        if not job_ids:
            logger.warning("No job IDs to cancel.")
            return
        self._delete_jobs(job_ids)
        self._delete_job_dependents(job_ids)

    def _list_job_names(self, label_selector: str) -> list[str]:
        """Returns the names of the CHS Jobs that match label_selector.

        Raises:
          ChsCancellationException: If the Jobs cannot be listed.
        """
        try:
            jobs = k8s_client.list_jobs(
                namespace=CHS_NAMESPACE, label_selector=label_selector
            )
        except k8s_client.KubeError as e:
            raise ChsCancellationException(
                f"Failed to list the Jobs matching '{label_selector}': {e}"
            ) from e
        return _object_names(jobs)

    def _delete_jobs(self, job_ids: list[str]) -> bool:
        """Deletes the given Jobs and returns whether that succeeded."""
        if not job_ids:
            return True
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
            return False
        return True

    def _delete_worker_service(self, run_id: str) -> bool:
        """Deletes the run's worker Service; returns whether it succeeded."""
        try:
            k8s_client.delete_service(
                name=worker_service_name(run_id), namespace=CHS_NAMESPACE
            )
        except k8s_client.KubeError as e:
            logger.warning(
                "Failed to delete the worker Service of run '%s'. Error: %s",
                run_id,
                e,
            )
            return False
        return True
