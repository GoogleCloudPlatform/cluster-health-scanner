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

import json
import logging
import re
import subprocess
from typing import Any

from packaging.version import InvalidVersion, Version

from chs.exceptions import ChsPreflightException, format_subprocess_error

logger = logging.getLogger(__name__)


MIN_KUEUE_VERSION = "0.14.5"
# We require at least JobSet v0.6.0 which introduces
# `publishNotReadyAddresses` in spec.network.
MIN_JOBSET_VERSION = "0.6.0"


class KubernetesPreflightChecker:
    """Checks for JobSet and Kueue CRDs on Kubernetes.

    Detects missing CRDs, extracts version info to fail first, warn or
    filter tests based on the missing CRDs.
    """

    def __init__(self, scheduler: Any | None = None) -> None:
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
            raise ChsPreflightException(
                "Kueue is missing or incompatible "
                f"(requires v{MIN_KUEUE_VERSION}+). "
                "Please refer to https://kueue.sigs.k8s.io/docs/installation "
                "for installation instructions."
            )

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
        except FileNotFoundError as e:
            raise ChsPreflightException(
                "Kubernetes CLI tool 'kubectl' was not found in PATH. "
                "Please ensure kubectl is installed and configured "
                "to connect to your cluster."
            ) from e
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            msg = format_subprocess_error(e)
            raise ChsPreflightException(
                f"Failed to query cluster CRDs via kubectl: {msg}"
            ) from e

    def _check_jobset(self, installed_crds: list[str]) -> None:
        """Checks if JobSet CRD is installed, healthy, and compatible."""
        if not any("jobset" in crd for crd in installed_crds):
            raise ChsPreflightException(
                "JobSet is missing or incompatible "
                f"(requires v{MIN_JOBSET_VERSION}+). "
                "Please refer to https://jobset.sigs.k8s.io/docs/installation/ "
                "for installation instructions."
            )

        healthy_dep = self._check_jobset_deployment_health()
        version = self._extract_version_from_deployment(
            healthy_dep, app_name="jobset"
        )
        if not version:
            version = self._get_jobset_version()

        if version and not self._is_jobset_version_supported(version):
            raise ChsPreflightException(
                "JobSet is missing or incompatible "
                f"(requires v{MIN_JOBSET_VERSION}+). "
                "Please refer to https://jobset.sigs.k8s.io/docs/installation/ "
                "for installation instructions."
            )

    def _is_deployment_healthy(self, dep: dict[str, Any]) -> bool:
        """Returns True if deployment has >0 replicas and is Available."""
        status = dep.get("status", {})
        available_replicas = status.get("availableReplicas") or 0
        conditions = status.get("conditions", [])
        is_available = any(
            c.get("type") == "Available" and c.get("status") == "True"
            for c in conditions
        )
        return available_replicas > 0 and is_available

    def _check_jobset_deployment_health(self) -> dict[str, Any]:
        """Verifies that JobSet controller deployment is in Available state."""
        deployments = self._discover_deployments("jobset")
        healthy_dep = next(
            (dep for dep in deployments if self._is_deployment_healthy(dep)),
            None,
        )
        if not healthy_dep:
            raise ChsPreflightException(
                "JobSet CRD is installed, but the JobSet controller "
                "deployment is NOT in an Available state. "
                "Please check controller health by running: "
                '`kubectl get deployment -A -l "app.kubernetes.io/name=jobset"`'
            )
        return healthy_dep

    def _discover_deployments(self, app_name: str) -> list[dict[str, Any]]:
        """Discovers controller deployments cluster-wide using app label."""
        try:
            output = self._run_kubectl(
                [
                    "get",
                    "deployment",
                    "-A",
                    "-l",
                    f"app.kubernetes.io/name={app_name}",
                    "-o",
                    "json",
                ]
            )
            return json.loads(output).get("items", [])
        except (
            subprocess.CalledProcessError,
            json.JSONDecodeError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            return []

    def _check_kueue(self, installed_crds: list[str]) -> bool:
        """Checks if Kueue CRD is installed and version is compatible.

        Returns:
            True if Kueue is installed and compatible, False otherwise.
        """
        if not self._check_kueue_crd_exists(installed_crds):
            return False

        deployments = self._discover_deployments("kueue")
        if deployments:
            healthy_dep = self._get_healthy_kueue_deployment(deployments)
            if not healthy_dep:
                logger.warning(
                    "Kueue controller deployment is NOT in an Available "
                    "state. Please check controller health with "
                    "'kubectl get deployment -A -l "
                    "app.kubernetes.io/name=kueue'."
                )
                return False
            version = self._extract_version_from_deployment(
                healthy_dep, app_name="kueue"
            )
            if version:
                return self._is_kueue_version_supported(version)
            return True

        version = self._get_kueue_version()
        if version:
            return self._is_kueue_version_supported(version)

        return True

    def _check_kueue_crd_exists(self, installed_crds: list[str]) -> bool:
        """Checks if Kueue CRD exists in the list of installed CRDs."""
        return any("kueue" in crd for crd in installed_crds)

    def _get_healthy_kueue_deployment(
        self, deployments: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """Finds the first healthy Available deployment."""
        for dep in deployments:
            if self._is_deployment_healthy(dep):
                return dep
        return None

    def _extract_version_from_deployment(
        self, healthy_dep: dict[str, Any], app_name: str
    ) -> str | None:
        """Extracts version string from container image."""
        containers = (
            healthy_dep.get("spec", {})
            .get("template", {})
            .get("spec", {})
            .get("containers", [])
        )
        target_container = None
        for container in containers:
            image = str(container.get("image") or "").lower()
            name = str(container.get("name") or "").lower()
            if app_name in image or name == "manager":
                target_container = container
                break

        if not target_container and containers:
            target_container = containers[0]

        if target_container:
            image_output = str(target_container.get("image") or "")
            if image_output:
                return self._parse_version_string(image_output)
        return None

    def _get_kueue_version(self) -> str | None:
        """Retrieves Kueue version string from cluster deployment."""
        logger.debug("Checking version compatibility")
        try:
            deployments = self._discover_deployments("kueue")
            if deployments:
                healthy_dep = self._get_healthy_kueue_deployment(deployments)
                if healthy_dep:
                    return self._extract_version_from_deployment(
                        healthy_dep, app_name="kueue"
                    )

            output = self._run_kubectl(
                [
                    "get",
                    "deployment",
                    "-n",
                    "kueue-system",
                    "kueue-controller-manager",
                    "-o",
                    "json",
                ]
            )
            if output:
                try:
                    dep_json = json.loads(output)
                    return self._extract_version_from_deployment(
                        dep_json, app_name="kueue"
                    )
                except json.JSONDecodeError:
                    return self._parse_version_string(output)
            return None
        except (
            subprocess.SubprocessError,
            FileNotFoundError,
            ValueError,
            json.JSONDecodeError,
        ):
            return None

    def _parse_version_string(self, text: str) -> str | None:
        """Helper to parse SemVer from image or version text."""
        match = re.search(
            r":v?(\d+\.\d+\.\d+(?:-[a-zA-Z0-9.]+)?(?:\+[a-zA-Z0-9.]+)?)(?:@sha256:[a-fA-F0-9]{64})?$",
            text,
        )
        if match:
            version = match.group(1)
            logger.debug("Found version: v%s", version)
            return version
        return None

    def _is_jobset_version_supported(self, version_str: str) -> bool:
        """Parses the version string and checks if it meets the minimum req."""
        try:
            current_version = Version(version_str)
            min_version = Version(MIN_JOBSET_VERSION)
            return current_version >= min_version
        except InvalidVersion:
            logger.warning(
                "Could not parse JobSet version '%s'. " "Assuming supported.",
                version_str,
            )
            return True

    def _get_jobset_version(self) -> str | None:
        """Retrieves JobSet version string from cluster deployment."""
        try:
            output = self._run_kubectl(
                [
                    "get",
                    "deployment",
                    "-n",
                    "jobset-system",
                    "jobset-controller-manager",
                    "-o",
                    "json",
                ]
            )
            if output:
                try:
                    dep_json = json.loads(output)
                    return self._extract_version_from_deployment(
                        dep_json, app_name="jobset"
                    )
                except json.JSONDecodeError:
                    return self._parse_version_string(output)
            return None
        except (
            subprocess.SubprocessError,
            FileNotFoundError,
            ValueError,
            json.JSONDecodeError,
        ):
            return None

    def _is_kueue_version_supported(self, version: str) -> bool:
        """Validates if the installed Kueue version is supported."""
        try:
            if Version(version) < Version(MIN_KUEUE_VERSION):
                logger.warning(
                    "Kueue version %s not supported. "
                    "Minimum supported version is v%s. "
                    "Reference for upgrading: "
                    "https://kueue.sigs.k8s.io/docs/installation/#upgrading",
                    version,
                    MIN_KUEUE_VERSION,
                )
                return False
            return True
        except InvalidVersion:
            logger.warning(
                "Could not parse Kueue version '%s'. " "Assuming supported.",
                version,
            )
            return True

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
            timeout=120,
        )
        return result.stdout.strip()
