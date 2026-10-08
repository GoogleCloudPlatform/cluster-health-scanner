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
import subprocess
from typing import Any

from chs.constants import (
    CHS_NAMESPACE,
    DELETE_WAIT_TIMEOUT_SEC,
    KUBECTL_TIMEOUT_SEC,
    SUPPORTED_DYNAMIC_SLICE_TOPOLOGY,
)
from chs.exceptions import KueueException, format_subprocess_error
from chs.resources import MACHINE_TYPE_RESOURCE_MAP

logger = logging.getLogger(__name__)

TOPOLOGY_NAMES = [
    "chs-node-topology-dynamic",
    "chs-node-topology-static",
]
FLAVOR_NAMES = [
    "chs-flavor-dynamic",
    "chs-flavor-static",
]


class KueueManager:
    """Manages Kueue resources for CHS runs.

    Resources include ResourceFlavor, ClusterQueue, LocalQueue, and Topology.
    """

    def __init__(self, scheduler: Any) -> None:
        """Initializes the KueueManager.

        Args:
            scheduler: The scheduler instance providing _run_kubectl capability.
        """
        self.scheduler = scheduler

    def _run_kubectl(
        self, args: list[str], timeout: int | None = KUBECTL_TIMEOUT_SEC
    ) -> str:
        """Runs kubectl command via the scheduler instance."""
        if hasattr(self.scheduler, "_run_kubectl"):
            return self.scheduler._run_kubectl(args, timeout=timeout)

        cmd = ["kubectl"] + args
        result = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            check=True,
            timeout=timeout,
        )
        return result.stdout.strip()

    def apply_and_verify(self, kueue_config_path: str) -> None:
        """Applies Kueue config and waits for ClusterQueue to become active.

        Args:
            kueue_config_path: Path to the rendered Kueue YAML config file.

        Raises:
            KueueException: If apply fails (e.g., permission error) or
              verification times out.
        """
        logger.debug("Applying Kueue configuration: %s", kueue_config_path)
        try:
            self._run_kubectl(["apply", "-f", kueue_config_path])
        except subprocess.SubprocessError as e:
            raise KueueException(
                "Failed to create ResourceFlavors, ClusterQueues, and/or "
                "LocalQueues. Details: "
                f"{format_subprocess_error(e)}"
            ) from e

        logger.debug(
            "Waiting for Kueue ClusterQueue 'chs-cluster-queue' to become "
            "active."
        )

        try:
            self._run_kubectl(
                [
                    "wait",
                    "--for=condition=Active",
                    "clusterqueue/chs-cluster-queue",
                    "--timeout=300s",
                ],
                timeout=350,
            )
            logger.debug("Kueue ClusterQueue is active.")

        except subprocess.SubprocessError as e:
            raise KueueException(
                "Timeout or error waiting for Kueue ClusterQueue "
                "'chs-cluster-queue' to become active. Details: "
                f"{format_subprocess_error(e)}"
            ) from e

    def generate_configs(
        self,
        workspace_config_dir: str,
        machine_type: str,
        nodes: list[str],
        dynamic_slicing: bool = False,
    ) -> str:
        """Renders Kueue config to disk without applying.

        Args:
            workspace_config_dir: Directory where Kueue configs will be written.
            machine_type: The machine type string (e.g., 'a3-mega').
            nodes: List of target nodes under test.
            dynamic_slicing: If True, renders configs with dynamic slice
              topology and flavor alongside static ones.

        Returns:
            The Kueue local queue label string to be injected into JobSet
            templates.

        Raises:
            KueueException: If machine_type is unsupported or setup fails.
        """
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
        """Deletes Kueue resources by name.

        Raises:
            KueueException: If deletion fails or times out.
        """
        logger.debug("Cleaning up Kueue resources...")
        delete_targets = [
            ["localqueue", "chs-local-queue", "-n", CHS_NAMESPACE],
            ["clusterqueue", "chs-cluster-queue"],
            ["resourceflavor", *FLAVOR_NAMES],
            ["topology", *TOPOLOGY_NAMES],
        ]
        for target in delete_targets:
            try:
                self._run_kubectl(
                    [
                        "delete",
                        *target,
                        "--ignore-not-found",
                        f"--timeout={DELETE_WAIT_TIMEOUT_SEC}s",
                    ]
                )
            except subprocess.SubprocessError as e:
                raise KueueException(
                    f"Failed to delete Kueue resources ({target[0]}). If the "
                    "object is stuck in deletion, check that "
                    "kueue-controller-manager is healthy and can reconcile (a "
                    "blocked 'kueue.x-k8s.io/resource-in-use' finalizer is "
                    f"the usual cause). Details: {format_subprocess_error(e)}"
                ) from e
        logger.debug("Successfully deleted Kueue resources.")

    def _get_kueue_api_version(self) -> str:
        """Determines whether to use v1beta2 or v1beta1 based on
        cluster support."""
        try:
            output = self._run_kubectl(
                [
                    "get",
                    "crd",
                    "clusterqueues.kueue.x-k8s.io",
                    "-o",
                    "jsonpath={.spec.versions[*].name}",
                ]
            )
            if "v1beta2" in output.split():
                return "v1beta2"
            return "v1beta1"
        except subprocess.SubprocessError:
            # Default to v1beta1 if check fails
            return "v1beta1"
