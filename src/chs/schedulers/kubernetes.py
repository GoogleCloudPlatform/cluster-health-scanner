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

import urllib.request
import importlib.resources as pkg_resources
import json
import time
import logging
import os
import re
import yaml


import subprocess
from typing import Any
from collections import defaultdict
from dataclasses import dataclass, field

from chs.schedulers.base import BaseScheduler
from chs.utils.kueue import KueueManager
from chs.exceptions import (
    ChsSchedulerException,
    KueueException,
    NoActiveJobsException,
)
from chs.utils.lifecycle import JobStatus
from chs.preflights.kubernetes_preflight import KubernetesPreflightChecker
from chs import resources
from chs.constants import CHS_NAMESPACE


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

_PARTITION_STATE_REGEX = re.compile(
    r"cloud\.google\.com/gke-tpu-partition-(.+)-state"
)


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
        self.nodes_arg = nodes
        self.nodepools_arg = nodepools
        self._resolved_nodes: list[str] | None = None
        self._resolved_raw_nodes: list[str] | None = None
        self._tpu_slice_data: TpuSliceData | None = None
        self.workspace_config_dir: str | None = None
        self.kueue_supported: bool = False
        self._jobset_labels_yaml: str = ""
        self._suspend_jobset: str = "false"
        if nodelists:
            logger.warning(
                "nodelists argument is ignored for Kubernetes scheduler."
                " Did you want to use nodes or nodepools?"
            )
        if partitions:
            logger.warning(
                "partitions argument is ignored for Kubernetes scheduler."
                " Did you want to use nodes or nodepools?"
            )

    def _resolve_project_id(self, node: str) -> str:
        """Resolves GCP project ID from Kubernetes node providerID."""
        try:
            provider_id = self._run_kubectl(
                ["get", "node", node, "-o", "jsonpath={.spec.providerID}"]
            ).strip()
            # providerID format: gce://project-id/zone/instance-name
            match = re.match(r"^gce://([^/]+)/([^/]+)/([^/]+)$", provider_id)
            if not match:
                raise ValueError(f"Unexpected providerID format: {provider_id}")
            return match.group(1)
        except Exception as e:
            raise ChsSchedulerException(
                f"Failed to resolve project ID from node: {e}"
            )

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
            try:
                nodes = self.get_nodes_under_test()
                kueue_mgr = KueueManager(self)
                self._jobset_labels_yaml = kueue_mgr.generate_configs(
                    workspace_config_dir, machine_type, nodes or []
                )
                self._suspend_jobset = "true"
            except KueueException as e:
                logger.warning(
                    "Failed to setup Kueue resources: %s. "
                    "Falling back to standard scheduling without Kueue.",
                    e,
                )
                self.kueue_supported = False
                self._jobset_labels_yaml = ""
                self._suspend_jobset = "false"
                return (
                    KubernetesPreflightChecker(
                        self
                    )._filter_tests_without_kueue(tests)
                    or []
                )

        return tests

    @property
    def name(self) -> str:
        return "kubernetes"

    def get_cluster_name(self) -> str | None:
        """Retrieves cluster name from GKE configmap or kubectl."""
        # 1. Try GCP Metadata Server (in-cluster GKE)
        try:
            req = urllib.request.Request(
                "http://metadata.google.internal/computeMetadata/v1/instance/attributes/cluster-name",
                headers={"Metadata-Flavor": "Google"},
            )
            val = urllib.request.urlopen(req, timeout=2).read().decode("utf-8")
            if val:
                return val
        except Exception:
            pass

        # 2. Try ConfigMap (in-cluster GKE)
        try:
            val = self._run_kubectl(
                [
                    "get",
                    "configmap",
                    "gke-common-webhook",
                    "-n",
                    "kube-system",
                    "-o",
                    "jsonpath={.data.clusterName}",
                ],
                timeout=5,
            )
            if val:
                return val
        except Exception:
            pass

        # 3. Try node labels
        try:
            val = self._run_kubectl(
                [
                    "get",
                    "nodes",
                    "-o",
                    "jsonpath={.items[0].metadata.labels}",
                ],
                timeout=5,
            )
            if val:
                labels = json.loads(val)
                gke_cluster = labels.get(
                    "container.googleapis.com/cluster_name"
                )
                if gke_cluster:
                    return gke_cluster
        except Exception:
            pass

        # 4. Try current-context (local execution)
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

    def _delete_namespace(self) -> None:
        """Deletes the chs namespace during cleanup."""
        try:
            self._run_kubectl(
                ["delete", "namespace", CHS_NAMESPACE, "--ignore-not-found"]
            )
        except Exception as e:
            logger.warning("Failed to delete namespace: %s", e)

    def _run_kubectl(
        self,
        args: list[str],
        timeout: int | None = 120,
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
                    + ["--ignore-not-found"],
                    namespaced=True,
                )
                logger.debug(
                    "Successfully deleted secrets: %s",
                    ", ".join(secrets_to_delete),
                )
            else:
                logger.debug("No CHS SSH secrets found to delete.")
        except Exception as e:
            logger.warning("Failed to delete secrets: %s", e)

    def _group_nodes_by_slice(
        self, node_names: list[str]
    ) -> dict[str, list[str]]:
        """Groups TPU nodes into slices using placement policy or node pool."""
        if not node_names:
            return {}

        slices = defaultdict(list)
        try:
            args = ["get", "nodes"] + node_names + ["-o", "json"]
            output = self._run_kubectl(args)
            if output:
                data = json.loads(output)
                # If only one node was queried, kubectl returns a
                # single Node object instead of a List
                items = data.get("items")
                if items is None:
                    items = [data] if data.get("kind") == "Node" else []

                for item in items:
                    node_name = item.get("metadata", {}).get("name")
                    if not node_name:
                        continue
                    labels = item.get("metadata", {}).get("labels", {})

                    slice_id = labels.get("cloud.google.com/gke-nodepool")
                    if not slice_id:
                        slice_id = "default-slice"

                    slices[slice_id].append(node_name)
        except subprocess.CalledProcessError as e:
            logger.warning(
                "Failed to run kubectl for nodes %s: %s", node_names, e
            )
            slices["failed-nodes"].extend(node_names)
        except json.JSONDecodeError as e:
            logger.warning(
                "Failed to parse json for nodes %s: %s", node_names, e
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
                    "Default L3 network not found. Discovered L3: %s",
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
            logger.warning("Failed to parse network discovery JSON: %s", e)
            return [], []
        except Exception as e:
            logger.warning("Unexpected error during network discovery: %s", e)
            return [], []

    def _get_busy_nodes(self) -> set[str]:
        """Returns a set of nodes running pods that request accelerators."""
        try:
            args = [
                "get",
                "pods",
                "-A",
                "-o",
                "custom-columns=NODE:.spec.nodeName,PHASE:.status.phase,"
                "CREQ:.spec.containers[*].resources.requests,CLIM:.spec.containers[*].resources.limits,"
                "IREQ:.spec.initContainers[*].resources.requests,ILIM:.spec.initContainers[*].resources.limits",
            ]
            pod_output = self._run_kubectl(args)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise ChsSchedulerException(
                f"Failed to get pods to determine busy nodes: {e}"
            ) from e

        busy_nodes = set()
        for line in pod_output.splitlines()[1:]:
            parts = line.split()
            if not parts or len(parts) < 2:
                continue

            node_name = parts[0]
            phase = parts[1]

            if phase in ("Succeeded", "Failed"):
                continue

            resources_str = " ".join(parts[2:]).lower()
            if "gpu" in resources_str or "tpu" in resources_str:
                busy_nodes.add(node_name)

        return busy_nodes

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
            node_data = json.loads(
                self._run_kubectl(
                    [
                        "get",
                        "nodes",
                        "-l",
                        "cloud.google.com/gke-tpu-accelerator",
                        "-o",
                        "json",
                    ]
                )
            )
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            json.JSONDecodeError,
        ) as e:
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
            if tpu_type:
                all_tpu_nodes.add(node_name)

            nodepool = labels.get("cloud.google.com/gke-nodepool")
            if not nodepool:
                continue

            accelerator_mode = labels.get(
                "cloud.google.com/gke-accelerator-topology-mode"
            )
            topology = labels.get("cloud.google.com/gke-tpu-topology")

            if accelerator_mode == "PROVISION_ONLY":
                matched_shape = None
                matched_state = None
                for key, val in labels.items():
                    m = _PARTITION_STATE_REGEX.match(key)
                    if m:
                        matched_shape = m.group(1)
                        matched_state = val
                        break

                if matched_shape and matched_state == "HEALTHY":
                    if nodepool not in unhealthy_pools:
                        tpu_nodepools[nodepool].append(node_name)
                        node_to_nodepool[node_name] = nodepool
                        pool_to_topology[nodepool] = matched_shape
                        partition_id = labels.get(
                            f"cloud.google.com/gke-tpu-partition-{matched_shape}-id"
                        )
                        if partition_id:
                            dynamic_slice_partitions[nodepool] = partition_id
                        if tpu_type:
                            pool_to_accelerator[nodepool] = tpu_type
                else:
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
                "Dropped unhealthy dynamic TPU pools: %s",
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
        else:
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
                                "cloud.google.com/gke-nodepool in "
                                f"({pools_str})",
                            ]
                        )
                else:
                    args.extend(["-l", "cloud.google.com/gke-nodepool!=system"])
                args.extend(["-o", "jsonpath={.items[*].metadata.name}"])
                output = self._run_kubectl(args)
                raw_nodes = [n for n in output.split() if n]
                logger.debug("Resolved raw nodes: %s", raw_nodes)
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                logger.warning(
                    "Failed to automatically discover nodes using kubectl: %s",
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
        busy_nodes = self._get_busy_nodes()

        gpu_nodes = set()
        requested_tpu_slices = defaultdict(set)

        for node in raw_nodes:
            pool = topology.node_to_pool.get(node)
            if pool:
                requested_tpu_slices[pool].add(node)
            elif node in topology.all_tpu_nodes:
                # TPU node and exists without its nodepool/slice mapping
                logger.warning(
                    "Dropping TPU node '%s' because its slice is unhealthy "
                    "or malformed.",
                    node,
                )
                continue
            else:
                gpu_nodes.add(node)

        valid_nodes = []
        if gpu_nodes:
            busy_gpus = gpu_nodes.intersection(busy_nodes)
            if busy_gpus:
                logger.info(
                    "Found %d busy GPU node(s). They will be skipped.",
                    len(busy_gpus),
                )

            valid_nodes.extend(gpu_nodes - busy_nodes)

        for pool_name, requested_slice_nodes in requested_tpu_slices.items():
            expected_slice_nodes = set(topology.pools[pool_name])

            busy_in_slice = list(expected_slice_nodes.intersection(busy_nodes))

            if busy_in_slice:
                logger.warning(
                    "Skipping TPU slice '%s': TPU tests require the entire "
                    "slice to be idle, but the following node(s) are currently "
                    "busy: %s",
                    pool_name,
                    ", ".join(busy_in_slice),
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
            args = ["get", "nodes"] + nodes + ["-o", "json"]
            output = self._run_kubectl(args)
            if not output:
                return {}

            data = json.loads(output)
            items = (
                data.get("items", [data])
                if data.get("kind") == "List"
                else [data]
            )

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

        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning("Failed to get node topology via kubectl: %s", e)
            return {}
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse kubectl output as JSON: %s", e)
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
                        "JobSet %s is failed. Will trigger cleanup.",
                        job_data.get("metadata", {}).get("name", "Unknown"),
                    )
                    return False
                else:
                    all_healthy = True
                    logger.debug(
                        "JobSet %s is healthy.",
                        job_data.get("metadata", {}).get("name", "Unknown"),
                    )

        except subprocess.CalledProcessError as e:
            logger.warning("Failed to run kubectl get jobset: %s", e.stderr)
        except (json.JSONDecodeError, AttributeError, TypeError) as e:
            logger.warning("Failed to parse status for JobSets: %s", e)

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
            subprocess.CalledProcessError,
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
                + job_ids
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
            subprocess.CalledProcessError,
            json.JSONDecodeError,
            AttributeError,
            TypeError,
        ) as e:
            logger.warning("Failed to get statuses for JobSets: %s", e)

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
        except subprocess.CalledProcessError as e:
            logger.debug("Failed to query pods for JobSet %s: %s", job_id, e)
            return []

    def fetch_job_log(self, job_id: str, log_path: str) -> None:
        """Fetches the log for a completed Kubernetes job."""
        logger.debug("Fetching logs for JobSet %s to %s", job_id, log_path)
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

            with open(log_path, "w") as f:
                for pod_name in pod_names:
                    subprocess.run(
                        ["kubectl", "logs", pod_name, "-n", CHS_NAMESPACE],
                        stdout=f,
                        check=True,
                        text=True,
                    )
            logger.debug("Logs saved to %s", log_path)

        except subprocess.CalledProcessError as e:
            if e.returncode < 0 or e.returncode == 130:
                logger.debug(
                    "Log fetching interrupted by user for JobSet %s.", job_id
                )
            else:
                logger.error(
                    "Failed to fetch logs for JobSet %s: %s", job_id, e
                )
        except OSError as e:
            logger.error("Failed to write log file %s: %s", log_path, e)

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
                ]
            )
        except subprocess.CalledProcessError as e:
            logger.warning("Failed to delete dynamic slices: %s", e)

    def _wait_for_slices_active(self, timeout_sec: int = 300) -> None:
        """Waits for dynamically provisioned slices to become ACTIVE."""
        start_time = time.time()
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

                statuses = []
                for slice_item in items:
                    conditions = (
                        slice_item.get("status", {}).get("conditions", [])
                        if slice_item.get("status")
                        else []
                    )
                    ready_cond = next(
                        (c for c in conditions if c.get("type") == "Ready"),
                        {},
                    )
                    statuses.append(ready_cond.get("reason"))

                if statuses and all(s == "ACTIVE" for s in statuses):
                    return

                failed_reasons = {
                    "FAILED",
                    "SliceCreationFailed",
                    "ACTIVE_DEGRADED",
                    "INCOMPLETE",
                }
                found_failure = next(
                    (s for s in statuses if s in failed_reasons), None
                )
                if found_failure:
                    logger.error(
                        "Slice provisioning failed with status: %s",
                        found_failure,
                    )
                    raise ChsSchedulerException(
                        f"Slice provisioning failed, status: {found_failure}"
                    )
            except (
                subprocess.CalledProcessError,
                json.JSONDecodeError,
            ) as e:
                logger.debug("Failed to check status for dynamic slices: %s", e)
                pass
            time.sleep(10)

        raise ChsSchedulerException(
            "Timed out waiting for dynamic slices to become ACTIVE."
        )

    def _provision_dynamic_slices(
        self, target_pools: dict[str, list[str]]
    ) -> None:
        """Provisions and waits for dynamic TPU slices."""
        if not (
            self._tpu_slice_data
            and self._tpu_slice_data.dynamic_slice_partitions
            and self.workspace_config_dir
        ):
            return

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
            tpu_type = self._tpu_slice_data.pool_to_accelerator.get(
                pool_name, ""
            )
            shape = self._tpu_slice_data.pool_to_topology.get(pool_name, "")

            slice_name = f"chs-slice-{pool_name}"[:63].rstrip("-.")
            rendered = template_str.format(
                slice_name=slice_name,
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

            self._run_kubectl(["apply", "-f", slice_yaml_path])

            self._wait_for_slices_active(300)

    def apply_scheduler_resources(self) -> None:
        """Applies generated scheduler resources to the cluster."""
        self._cleanup_dynamic_slices()

        target_pools = {}
        nodes = self.get_nodes_under_test()
        if nodes:
            target_pools = self._group_nodes_by_slice(nodes)

        self._provision_dynamic_slices(target_pools)

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
            logger.warning("Failed to clean up Kueue resources: %s", e)

        self._delete_job_secrets()

        self._delete_namespace()

    def cancel_jobs(self, job_ids: list[str]) -> None:
        """Cancels the given JobSets using kubectl delete."""
        if not job_ids:
            logger.warning("No job IDs to cancel.")
            return

        try:
            args = ["delete", "jobset"] + job_ids + ["--ignore-not-found"]
            self._run_kubectl(args, namespaced=True)
            logger.debug("Successfully deleted JobSets: %s", ", ".join(job_ids))
        except FileNotFoundError:
            logger.warning("kubectl command not found. Cannot cancel jobs.")
        except subprocess.CalledProcessError as e:
            logger.error("Failed to delete JobSets. Error: %s", e.stderr)

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
            logger.warning("kubectl command not found. Cannot get jobs.")
            return []
        except subprocess.CalledProcessError as e:
            logger.warning("Failed to get jobsets: %s", e)
            return []

    def cancel_jobs_by_prefix(self, prefix: str) -> None:
        """Cancels all JobSets matching the specified prefix natively."""
        logger.debug("Attempting native cancel for prefix: %s", prefix)
        self._cleanup_dynamic_slices()
        # Match the sanitized prefix as a substring in the JobSet name.
        sanitized_match = prefix.replace("_", "-").replace(".", "-").lower()

        try:
            # List all JobSets
            output = self._run_kubectl(
                ["get", "jobset", "-o", "name"], namespaced=True
            )
            if not output:
                self._delete_namespace()
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

            self._delete_namespace()
        except FileNotFoundError:
            logger.warning(
                "kubectl command not found. Cannot perform native cancel."
            )
        except Exception as e:
            logger.warning("Native cancel failed: %s", e)
