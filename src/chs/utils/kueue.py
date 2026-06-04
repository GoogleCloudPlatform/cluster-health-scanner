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

"""Kueue resource management module for GKE scheduler."""

import importlib.resources as pkg_resources
import logging
import os
import subprocess
import time
from typing import Any

from chs.exceptions import KueueException
from chs.resources import MACHINE_TYPE_RESOURCE_MAP

logger = logging.getLogger(__name__)


class KueueManager:
    """Manages Kueue resources for CHS runs.

    Resources include ResourceFlavor, ClusterQueue, and LocalQueue.
    """

    def __init__(self, scheduler: Any) -> None:
        """Initializes the KueueManager.

        Args:
            scheduler: The scheduler instance providing _run_kubectl capability.
        """
        self.scheduler = scheduler

    def _run_kubectl(self, args: list[str]) -> str:
        """Runs kubectl command via the scheduler instance."""
        if hasattr(self.scheduler, "_run_kubectl"):
            return self.scheduler._run_kubectl(args)

        cmd = ["kubectl"] + args
        result = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            check=True,
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
        except subprocess.CalledProcessError as e:
            raise KueueException(
                "Failed to apply Kueue configuration. Please ensure you have "
                "cluster-level permissions to create ResourceFlavors and "
                "ClusterQueues, or namespace-level permissions for "
                f"LocalQueues. Details: {e.stderr}"
            ) from e

        logger.info(
            "Waiting for Kueue ClusterQueue 'chs-cluster-queue' to become "
            "active."
        )
        start_time = time.time()
        timeout = 300
        verified = False
        last_log_time = start_time

        while time.time() - start_time < timeout:
            try:
                self._run_kubectl(
                    [
                        "wait",
                        "--for=condition=Active",
                        "clusterqueue/chs-cluster-queue",
                        "--timeout=10s",
                    ]
                )
                verified = True
                logger.debug("Kueue ClusterQueue is active.")
                break
            except subprocess.CalledProcessError:
                if time.time() - last_log_time >= 30:
                    logger.info(
                        "Waiting for Kueue ClusterQueue 'chs-cluster-queue' "
                        "to become active."
                    )
                    last_log_time = time.time()

        if not verified:
            raise KueueException(
                "Timeout waiting for Kueue ClusterQueue 'chs-cluster-queue' "
                "to become active."
            )

    def setup(
        self, workspace_config_dir: str, machine_type: str, nodes: list[str]
    ) -> str:
        """Renders Kueue config, applies it, and verifies readiness.

        Args:
            workspace_config_dir: Directory where Kueue configs will be written.
            machine_type: The machine type string (e.g., 'a3-mega').
            nodes: List of target nodes under test.

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
        instance_type = machine_type
        accelerators_per_node = resource_map.get("accelerators", 0)
        nominal_quota = len(nodes) * accelerators_per_node
        resource_name = resource_map.get("resource_name", "nvidia.com/gpu")

        try:
            # Check if flavor matches the current instance_type
            existing_flavor = self._run_kubectl(
                [
                    "get",
                    "resourceflavor",
                    "chs-flavor",
                    "-o",
                    "jsonpath={.spec.nodeLabels['node.kubernetes.io/instance-type']}",
                ]
            )
            if existing_flavor.strip() != instance_type:
                logger.warning(
                    "Existing Kueue resource flavor has instance type %s, "
                    "but current run requires %s. Recreating Kueue "
                    "resources...",
                    existing_flavor.strip(),
                    instance_type,
                )
                # Delete existing kueue resources to force recreation
                self._run_kubectl(
                    [
                        "delete",
                        "clusterqueue",
                        "chs-cluster-queue",
                        "--ignore-not-found",
                    ]
                )
                self._run_kubectl(
                    [
                        "delete",
                        "resourceflavor",
                        "chs-flavor",
                        "--ignore-not-found",
                    ]
                )
            else:
                result = self._run_kubectl(
                    [
                        "get",
                        "clusterqueue",
                        "chs-cluster-queue",
                        "-o",
                        'jsonpath={.status.conditions[?(@.type=="Active")].status}',
                    ]
                )
                if result.strip() == "True":
                    logger.info(
                        "ClusterQueue chs-cluster-queue is already active "
                        "and matches machine type."
                    )
                    return "kueue.x-k8s.io/queue-name: chs-local-queue"
        except subprocess.CalledProcessError:
            pass

        kueue_tpl_path = pkg_resources.files("chs.schedulers").joinpath(
            "kueue-configs.yaml.tpl"
        )
        with pkg_resources.as_file(kueue_tpl_path) as p:
            with open(p, "r") as f:
                kueue_tpl = f.read()

        rendered_kueue = kueue_tpl.format(
            instance_type=instance_type,
            nominal_quota=str(nominal_quota),
            resource_name=resource_name,
        )

        kueue_config_path = os.path.join(
            workspace_config_dir, "kueue-configs.yaml"
        )
        with open(kueue_config_path, "w") as f:
            f.write(rendered_kueue)

        self.apply_and_verify(kueue_config_path)

        return "kueue.x-k8s.io/queue-name: chs-local-queue"

    def cleanup(self, kueue_config_path: str) -> None:
        """Deletes Kueue resources defined in the configuration file.

        Args:
            kueue_config_path: Path to the Kueue YAML configuration file.

        Raises:
            KueueException: If deletion fails.
        """
        if not os.path.exists(kueue_config_path):
            logger.debug(
                "Kueue configuration file not found at %s; skipping deletion.",
                kueue_config_path,
            )
            return

        logger.info("Cleaning up Kueue resources...")
        try:
            self._run_kubectl(["delete", "-f", kueue_config_path])
            logger.info("Successfully deleted Kueue resources.")
        except subprocess.CalledProcessError as e:
            raise KueueException(
                f"Failed to delete Kueue resources. Details: {e.stderr}"
            ) from e
