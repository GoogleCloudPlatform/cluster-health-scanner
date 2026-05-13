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
        instance_type = resource_map.get("instance_type")
        if not instance_type:
            raise KueueException(
                f"Missing 'instance_type' mapping for machine type "
                f"'{machine_type}' in MACHINE_TYPE_RESOURCE_MAP."
            )

        gpus_per_node = resource_map.get("gpus", 0)
        nominal_quota = len(nodes) * gpus_per_node

        try:
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
                logger.info("ClusterQueue chs-cluster-queue is already active")
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
            instance_type=instance_type, nominal_quota=str(nominal_quota)
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
