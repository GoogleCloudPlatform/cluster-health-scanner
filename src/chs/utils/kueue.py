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

"""Kueue resource management module for Kubernetes scheduler."""

import importlib.resources as pkg_resources
import logging
import os
import time
from typing import Any

import yaml

from chs.constants import (
    CHS_NAMESPACE,
    DELETE_WAIT_TIMEOUT_SEC,
    SUPPORTED_DYNAMIC_SLICE_TOPOLOGY,
)
from chs.exceptions import KueueException
from chs.resources import MACHINE_TYPE_RESOURCE_MAP
from chs.utils import k8s_client

logger = logging.getLogger(__name__)

KUEUE_API_GROUP = "kueue.x-k8s.io"
TOPOLOGY_NAMES = [
    "chs-node-topology-dynamic",
    "chs-node-topology-static",
]
FLAVOR_NAMES = [
    "chs-flavor-dynamic",
    "chs-flavor-static",
]
_KIND_TO_PLURAL = {
    "Topology": "topologies",
    "ResourceFlavor": "resourceflavors",
    "ClusterQueue": "clusterqueues",
    "LocalQueue": "localqueues",
}
_CLUSTER_QUEUE_TIMEOUT_SECONDS = 300


class KueueManager:
    """Manages Kueue resources for CHS runs."""

    def __init__(self, scheduler: Any) -> None:
        """Initializes the KueueManager."""
        self.scheduler = scheduler

    def apply_and_verify(self, kueue_config_path: str) -> None:
        """Applies Kueue config and waits for ClusterQueue to become active."""
        logger.debug("Applying Kueue configuration: %s", kueue_config_path)
        try:
            with open(kueue_config_path) as f:
                manifests = [
                    manifest for manifest in yaml.safe_load_all(f) if manifest
                ]
            for manifest in manifests:
                k8s_client.apply_custom_object(
                    plural=_KIND_TO_PLURAL[manifest["kind"]], body=manifest
                )
        except k8s_client.KubeError as e:
            raise KueueException(
                "Failed to create ResourceFlavors, ClusterQueues, and/or "
                "LocalQueues. Details: "
                f"{e}"
            ) from e

        cluster_queue = next(
            (
                manifest
                for manifest in manifests
                if manifest.get("kind") == "ClusterQueue"
            ),
            None,
        )
        if cluster_queue is None:
            raise KueueException(
                f"No ClusterQueue document found in {kueue_config_path}."
            )

        logger.debug(
            "Waiting for Kueue ClusterQueue 'chs-cluster-queue' to become "
            "active."
        )
        version = cluster_queue["apiVersion"].split("/")[1]
        try:
            self._wait_for_cluster_queue_active(version)
            logger.debug("Kueue ClusterQueue is active.")
        except k8s_client.KubeError as e:
            raise KueueException(
                "Timeout or error waiting for Kueue ClusterQueue "
                f"'chs-cluster-queue' to become active. Details: {e}"
            ) from e

    def _wait_for_cluster_queue_active(self, version: str) -> None:
        deadline = time.monotonic() + _CLUSTER_QUEUE_TIMEOUT_SECONDS
        while True:
            queue = k8s_client.get_custom_object(
                group=KUEUE_API_GROUP,
                version=version,
                plural="clusterqueues",
                name="chs-cluster-queue",
            )
            conditions = (queue.get("status") or {}).get("conditions") or []
            if any(
                condition.get("type") == "Active"
                and condition.get("status") == "True"
                for condition in conditions
            ):
                return
            if time.monotonic() > deadline:
                raise k8s_client.KubeError(
                    "timed out waiting for clusterqueues/chs-cluster-queue"
                )
            time.sleep(0.5)

    def generate_configs(
        self,
        workspace_config_dir: str,
        machine_type: str,
        nodes: list[str],
        dynamic_slicing: bool = False,
    ) -> str:
        """Renders Kueue config to disk without applying."""
        if machine_type not in MACHINE_TYPE_RESOURCE_MAP:
            raise KueueException(
                f"Machine type '{machine_type}' is not recognized in "
                "MACHINE_TYPE_RESOURCE_MAP. Cannot reliably configure Kueue "
                "ResourceFlavor properties."
            )

        resource_map = MACHINE_TYPE_RESOURCE_MAP.get(machine_type, {})
        accelerators_per_node = resource_map.get("accelerators", 0)
        nominal_quota = len(nodes) * accelerators_per_node
        resource_name = resource_map.get("resource_name", "nvidia.com/gpu")

        tpl_name = (
            "kueue-dynamic-configs.yaml.tpl"
            if dynamic_slicing
            else "kueue-configs.yaml.tpl"
        )
        kueue_tpl_path = pkg_resources.files("chs.schedulers").joinpath(
            tpl_name
        )
        with pkg_resources.as_file(kueue_tpl_path) as p:
            with open(p, "r") as f:
                kueue_tpl = f.read()

        rendered_kueue = kueue_tpl.format(
            instance_type=machine_type,
            nominal_quota=str(nominal_quota),
            resource_name=resource_name,
            namespace=CHS_NAMESPACE,
            kueue_api_version=self._get_kueue_api_version(),
            dynamic_topology=SUPPORTED_DYNAMIC_SLICE_TOPOLOGY,
        )

        kueue_config_path = os.path.join(
            workspace_config_dir, "kueue-configs.yaml"
        )
        with open(kueue_config_path, "w") as f:
            f.write(rendered_kueue)

        return "kueue.x-k8s.io/queue-name: chs-local-queue"

    def apply_configs(self, workspace_config_dir: str) -> None:
        """Applies rendered Kueue configurations to the cluster."""
        self.cleanup()
        kueue_config_path = os.path.join(
            workspace_config_dir, "kueue-configs.yaml"
        )
        self.apply_and_verify(kueue_config_path)

    def cleanup(self) -> None:
        """Deletes Kueue resources by name."""
        logger.debug("Cleaning up Kueue resources...")
        version = self._get_kueue_api_version()
        deletions = [
            ("localqueues", ["chs-local-queue"], CHS_NAMESPACE),
            ("clusterqueues", ["chs-cluster-queue"], None),
            ("resourceflavors", FLAVOR_NAMES, None),
            ("topologies", TOPOLOGY_NAMES, None),
        ]
        try:
            for plural, object_names, namespace in deletions:
                k8s_client.delete_custom_objects(
                    group=KUEUE_API_GROUP,
                    version=version,
                    plural=plural,
                    object_names=object_names,
                    namespace=namespace,
                    wait_timeout_seconds=DELETE_WAIT_TIMEOUT_SEC,
                )
            logger.debug("Successfully deleted Kueue resources.")
        except k8s_client.KubeError as e:
            raise KueueException(
                f"Failed to delete Kueue resources. Details: {e}"
            ) from e

    def _get_kueue_api_version(self) -> str:
        """Determines whether to use v1beta2 or v1beta1."""
        try:
            crd = k8s_client.get_crd(name="clusterqueues.kueue.x-k8s.io")
        except k8s_client.KubeError as e:
            logger.debug(
                "Could not read the Kueue ClusterQueue CRD; defaulting to "
                "v1beta1. Error: %s",
                e,
            )
            return "v1beta1"
        version_names = [
            version.get("name")
            for version in crd.get("spec", {}).get("versions", [])
        ]
        if "v1beta2" in version_names:
            return "v1beta2"
        return "v1beta1"
