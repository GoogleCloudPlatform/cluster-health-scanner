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

"""Kubernetes preflight checker and cleaner."""

import logging
import re
import subprocess
from typing import Any

from chs.exceptions import ChsPreflightException

logger = logging.getLogger(__name__)


class KubernetesPreflightChecker:
    """Checks for JobSet and Kueue CRDs on Kubernetes.

    Detects missing CRDs, extracts version info to fail first, warn or
    filter tests based on the missing CRDs.
    """

    def __init__(self, scheduler=None):
        self.scheduler = scheduler

    def check(
        self,
        tests: list[dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]] | None, bool]:
        """Runs preflight checks on Kubernetes.

        Args:
            tests: Optional list of tests to be run and potentially filtered.

        Returns:
            A tuple of (filtered_tests, kueue_supported).
        """
        installed_crds = self._get_installed_crds()
        self._check_jobset(installed_crds)

        kueue_supported = self._check_kueue(installed_crds)

        if not kueue_supported:
            return self._filter_tests_without_kueue(tests), kueue_supported

        return tests, kueue_supported

    def _get_installed_crds(self) -> list[str]:
        """Fetches installed CRDs from the cluster."""
        try:
            output = self._run_kubectl(["get", "crd", "-o", "name"])
            installed_crds = [
                line.strip().lower()
                for line in output.splitlines()
                if line.strip()
            ]
            logger.debug(
                "Found %d CRDs installed on the cluster.", len(installed_crds)
            )
            return installed_crds
        except subprocess.CalledProcessError as e:
            raise ChsPreflightException(
                f"Failed to query cluster CRDs: {e}"
            ) from e

    def _check_jobset(self, installed_crds: list[str]) -> None:
        """Checks if JobSet CRD is installed.

        Raises:
            ChsPreflightException: If JobSet CRD is missing.
        """
        if not any("jobset" in crd for crd in installed_crds):
            raise ChsPreflightException(
                "Missing required cluster dependencies: JobSet. "
                "Please install them before continuing. "
                "Reference: https://github.com/kubernetes-sigs/jobset"
            )

    def _check_kueue(self, installed_crds: list[str]) -> bool:
        """Checks if Kueue CRD is installed and version is compatible.

        Returns:
            True if Kueue is installed and compatible, False otherwise.
        """
        if not self._check_kueue_crd_exists(installed_crds):
            return False

        version = self._get_kueue_version()
        if not version:
            return False

        if not self._is_kueue_version_supported(version):
            return False

        return True

    def _check_kueue_crd_exists(self, installed_crds: list[str]) -> bool:
        """Checks if Kueue CRD exists in the list of installed CRDs."""
        return any("kueue" in crd for crd in installed_crds)

    def _get_kueue_version(self) -> str | None:
        """Retrieves the Kueue version string from the deployment."""
        logger.info("Checking version compatibility")
        try:
            image_output = self._run_kubectl(
                [
                    "get",
                    "deployment",
                    "-n",
                    "kueue-system",
                    "kueue-controller-manager",
                    "-o",
                    "jsonpath={.spec.template.spec.containers[0].image}",
                ]
            )
            match = re.search(
                r"(?:v)?(\d+\.\d+\.\d+(?:-[a-zA-Z0-9.]+)?)", image_output
            )
            if match:
                version = match.group(1)
                logger.info("Found Kueue version: v%s", version)
                return version
            else:
                logger.warning(
                    "Could not parse Kueue version from image: %s. "
                    "Skipping 'nccl-cluster' test.",
                    image_output,
                )
        except (subprocess.CalledProcessError, ValueError):
            logger.warning("Could not parse or retrieve Kueue version.")
        return None

    def _is_kueue_version_supported(self, version: str) -> bool:
        """Validates if the installed Kueue version is supported."""
        v_parts = [int(x) for x in version.split("-")[0].split(".")]
        if v_parts < [0, 14, 5]:
            logger.warning(
                "Kueue version %s not supported. "
                "Minimum supported version is v0.14.5. "
                "Reference for upgrading: https://kueue.sigs.k8s.io/docs/installation/#upgrading",
                version,
            )
            return False
        return True

    def _filter_tests_without_kueue(
        self, tests: list[dict[str, Any]] | None
    ) -> list[dict[str, Any]] | None:
        """Filters out tests that require Kueue."""
        filtered_tests = tests or []
        if any(
            (t.get("name") if isinstance(t, dict) else t) == "nccl-cluster"
            for t in filtered_tests
        ):
            logger.warning(
                "Kueue is uninstalled or wrong version. "
                "Skipping 'nccl-cluster' test."
            )
        return [
            t
            for t in filtered_tests
            if (t.get("name") if isinstance(t, dict) else t) != "nccl-cluster"
        ]

    def _run_kubectl(self, args: list[str]) -> str:
        """Runs a kubectl command and returns filtered output."""
        if self.scheduler and hasattr(self.scheduler, "_run_kubectl"):
            return self.scheduler._run_kubectl(args)

        cmd = ["kubectl"] + args
        result = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            check=True,
        )
        return result.stdout.strip()
