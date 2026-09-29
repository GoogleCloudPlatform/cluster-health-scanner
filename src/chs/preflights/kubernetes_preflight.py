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
from typing import Any

from packaging.version import InvalidVersion, Version

from chs.exceptions import ChsPreflightException
from chs.utils import k8s_client

logger = logging.getLogger(__name__)


MIN_KUEUE_VERSION = "0.14.5"
# We require at least JobSet v0.6.0 which introduces
# `publishNotReadyAddresses` in spec.network.
MIN_JOBSET_VERSION = "0.6.0"


class KubernetesPreflightChecker:
    """Checks for JobSet and Kueue CRDs on Kubernetes."""

    def __init__(self, scheduler: Any | None = None) -> None:
        self.scheduler = scheduler

    def check(
        self,
        tests: list[dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]] | None, bool]:
        """Runs preflight checks on Kubernetes."""
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
            crds = k8s_client.list_crds()
        except k8s_client.KubeError as e:
            raise ChsPreflightException(
                f"Failed to query cluster CRDs from the Kubernetes API: {e}"
            ) from e
        installed_crds = [crd["metadata"]["name"].lower() for crd in crds]
        logger.debug(
            "Found %d CRDs installed on the cluster.", len(installed_crds)
        )
        return installed_crds

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
        if not deployments:
            raise ChsPreflightException(
                (
                    "JobSet CRD is installed, but no JobSet controller "
                    "deployment could be found using standard "
                    "labels or in the default 'jobset-system' "
                    "namespace. If you are using a custom "
                    "deployment name, please add the required "
                    "label by running: `kubectl label deployment "
                    "<your-deployment-name> -n <your-namespace> "
                    "app.kubernetes.io/name=jobset --overwrite`"
                )
            )

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
        discovered_deployments = []
        try:
            discovered_deployments = k8s_client.list_deployments(
                label_selector=f"app.kubernetes.io/name={app_name}"
            )
        except k8s_client.KubeError as e:
            logger.debug("Label query failed: %s", e)

        if not discovered_deployments:
            logger.debug(
                (
                    "%s deployment not found by label, checking for deployment "  # noqa: E501
                    "'%s-controller-manager' in namespace '%s-system'"
                ),
                app_name.capitalize(),
                app_name,
                app_name,
            )
            try:
                discovered_deployments = [
                    k8s_client.get_deployment(
                        name=f"{app_name}-controller-manager",
                        namespace=f"{app_name}-system",
                    )
                ]
            except k8s_client.KubeError as e:
                logger.debug("Namespace fallback query failed: %s", e)

        return discovered_deployments

    def _check_kueue(self, installed_crds: list[str]) -> bool:
        """Checks if Kueue CRD is installed and version is compatible."""
        if not self._check_kueue_crd_exists(installed_crds):
            return False

        deployments = self._discover_deployments("kueue")
        if not deployments:
            logger.warning(
                "Kueue CRD is installed, but no Kueue controller deployment "
                "could be found. Please check controller installation with "
                "'kubectl get deployment -A -l app.kubernetes.io/name=kueue'."
            )
            return False

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
                "Unable to parse the installed JobSet "
                "version '%s'. Proceeding with "
                "the assumption that it is a supported version (>= %s).",
                version_str,
                MIN_JOBSET_VERSION,
            )
            return True

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
                "Unable to parse the installed Kueue version "
                "'%s'. Proceeding with the "
                "assumption that it is a supported version (>= %s).",
                version,
                MIN_KUEUE_VERSION,
            )
            return True
