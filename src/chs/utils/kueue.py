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

from chs.exceptions import KueueException
from chs.resources import MACHINE_TYPE_RESOURCE_MAP
from chs.constants import CHS_NAMESPACE

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

    def _run_kubectl(self, args: list[str], timeout: int | None = None) -> str:
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
        except subprocess.CalledProcessError as e:
            raise KueueException(
                "Failed to create ResourceFlavors, ClusterQueues, and/or "
                "LocalQueues. Details: "
                f"{e.stderr}"
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

        except subprocess.CalledProcessError as e:
            raise KueueException(
                "Timeout or error waiting for Kueue ClusterQueue "
                f"'chs-cluster-queue' to become active. Details: {e}"
            ) from e

    def generate_configs(
        self, workspace_config_dir: str, machine_type: str, nodes: list[str]
    ) -> str:
        """Renders Kueue config to disk without applying.

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
        accelerators_per_node = resource_map.get("accelerators", 0)
        nominal_quota = len(nodes) * accelerators_per_node
        resource_name = resource_map.get("resource_name", "nvidia.com/gpu")

        kueue_tpl_path = pkg_resources.files("chs.schedulers").joinpath(
            "kueue-configs.yaml.tpl"
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
        )

        kueue_config_path = os.path.join(
            workspace_config_dir, "kueue-configs.yaml"
        )
        with open(kueue_config_path, "w") as f:
            f.write(rendered_kueue)

        return "kueue.x-k8s.io/queue-name: chs-local-queue"

    def apply_configs(self, workspace_config_dir: str) -> None:
        """Applies rendered Kueue configurations to the cluster."""
        try:
            self.cleanup()
        except KueueException:
            pass

        kueue_config_path = os.path.join(
            workspace_config_dir, "kueue-configs.yaml"
        )
        self.apply_and_verify(kueue_config_path)

    def cleanup(self) -> None:
        """Deletes Kueue resources by name.

        Raises:
            KueueException: If deletion fails.
        """
        logger.debug("Cleaning up Kueue resources...")
        try:
            self._run_kubectl(
                [
                    "delete",
                    "localqueue",
                    "chs-local-queue",
                    "-n",
                    CHS_NAMESPACE,
                    "--ignore-not-found",
                ]
            )
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
            logger.debug("Successfully deleted Kueue resources.")
        except subprocess.CalledProcessError as e:
            raise KueueException(
                f"Failed to delete Kueue resources. Details: {e.stderr}"
            ) from e

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
        except subprocess.CalledProcessError:
            # Default to v1beta1 if check fails
            return "v1beta1"
