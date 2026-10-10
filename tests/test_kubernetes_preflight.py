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

"""Unit tests for KubernetesPreflightChecker."""

import unittest
from unittest import mock
from chs.exceptions import ChsPreflightException
from chs.preflights.kubernetes_preflight import KubernetesPreflightChecker
from chs.utils.k8s_client import KubeError
from tests.k8s_fakes import Call, install_fake_client, patch_k8s_client

_JOBSET_CRD = "jobsets.jobset.sigs.k8s.io"
_CRD_QUERY = Call("list_crds", {})


def _label_query(app):
    """Returns the recorded cluster-wide label query for a controller."""
    return Call(
        "list_deployments", {"label_selector": f"app.kubernetes.io/name={app}"}
    )


def _fallback_query(app):
    """Returns the recorded by-name query for a controller."""
    return Call(
        "get_deployment",
        {
            "name": f"{app}-controller-manager",
            "namespace": f"{app}-system",
        },
    )


def _forbidden(what):
    """Returns the error of a request the RBAC rules deny."""
    return KubeError(
        f"Error from server (Forbidden): {what} is forbidden",
        status=403,
    )


def _deployment(
    app, image_tag="v0.7.2", *, available=True, labeled=True, containers=None
):
    """Returns the controller Deployment of `app` in `<app>-system`."""
    metadata = {
        "name": f"{app}-controller-manager",
        "namespace": f"{app}-system",
    }
    if labeled:
        metadata["labels"] = {"app.kubernetes.io/name": app}
    if containers is None:
        containers = [{"image": f"registry.k8s.io/{app}/{app}:{image_tag}"}]
    return {
        "kind": "Deployment",
        "metadata": metadata,
        "spec": {"template": {"spec": {"containers": containers}}},
        "status": {
            "availableReplicas": 1 if available else 0,
            "conditions": [
                {
                    "type": "Available",
                    "status": "True" if available else "False",
                }
            ],
        },
    }


class TestKubernetesPreflightChecker(unittest.TestCase):
    """Tests for KubernetesPreflightChecker."""

    def setUp(self):
        super().setUp()
        self.kube = install_fake_client(self)
        self.mock_scheduler = mock.Mock()
        self.mock_app_data = mock.Mock()
        self.mock_scheduler._app_data = self.mock_app_data
        self.checker = KubernetesPreflightChecker(self.mock_scheduler)

    def _add_crds(self, *crd_names):
        for crd_name in crd_names:
            self.kube.add_crd(crd_name)

    def _add_deployment(self, deployment):
        self.kube.add_object("apps", "deployments", deployment)

    def _add_healthy_jobset(self):
        self._add_deployment(_deployment("jobset", "v0.7.2"))

    def test_check_crds_installed_version_valid(self):
        """Test that preflight check passes if JobSet is valid."""
        self._add_crds(_JOBSET_CRD)
        self._add_healthy_jobset()
        original_tests = [{"name": "test1"}, {"name": "nccl-cluster"}]
        filtered_tests = self.checker.check(original_tests)
        self.assertEqual(
            self.kube.calls,
            [_CRD_QUERY, _label_query("jobset")],
        )
        self.assertEqual(filtered_tests, original_tests)

    def test_check_crds_missing_raises_exception(self):
        """Test that missing JobSet causes it to fail fast."""
        with self.assertRaisesRegex(
            ChsPreflightException,
            "JobSet is missing or incompatible",
        ):
            self.checker.check()
        self.assertEqual(self.kube.calls, [_CRD_QUERY])

    def test_check_jobset_deployment_unhealthy_raises_exception(self):
        """Test that unhealthy deployment (0 replicas) raises exception."""
        self._add_crds(_JOBSET_CRD)
        self._add_deployment(_deployment("jobset", "v0.7.2", available=False))
        with self.assertRaisesRegex(
            ChsPreflightException,
            "JobSet CRD is installed, but the JobSet controller "
            "deployment is NOT in an Available state",
        ):
            self.checker.check()
        self.assertEqual(self.kube.calls, [_CRD_QUERY, _label_query("jobset")])

    def test_check_jobset_deployment_rbac_restricted_raises_exception(self):
        """Test that RBAC query raises exception when deployment unverified."""
        self._add_crds(_JOBSET_CRD)
        self._add_healthy_jobset()
        self.kube.fail_next("list_deployments", _forbidden("deployments.apps"))
        self.kube.fail_next(
            "get_deployment",
            _forbidden('deployments.apps "jobset-controller-manager"'),
        )
        tests = [{"name": "nvidia-smi"}]
        with self.assertRaisesRegex(
            ChsPreflightException,
            "no JobSet controller deployment could be found",
        ):
            self.checker.check(tests)
        self.assertEqual(
            self.kube.calls,
            [_CRD_QUERY, _label_query("jobset"), _fallback_query("jobset")],
        )

    def test_check_jobset_deployment_null_available_replicas(self):
        """Test handling when status.availableReplicas is null in JSON."""
        self._add_crds(_JOBSET_CRD)
        null_replicas_dep = _deployment("jobset", "v0.7.2")
        null_replicas_dep["status"]["availableReplicas"] = None
        self._add_deployment(null_replicas_dep)
        with self.assertRaisesRegex(
            ChsPreflightException,
            "JobSet CRD is installed, but the JobSet controller "
            "deployment is NOT in an Available state",
        ):
            self.checker.check()
        self.assertEqual(self.kube.calls, [_CRD_QUERY, _label_query("jobset")])

    def test_check_crds_query_failure_raises_exception(self):
        """Test that Error when getting CRDs raises ChsPreflightException."""
        err = _forbidden("customresourcedefinitions.apiextensions.k8s.io")
        self.kube.fail_next("list_crds", err)
        with self.assertRaises(ChsPreflightException) as ctx:
            self.checker.check()
        self.assertEqual(
            str(ctx.exception),
            f"Failed to query cluster CRDs from the Kubernetes API: {err}",
        )
        self.assertIs(ctx.exception.__cause__, err)
        self.assertEqual(self.kube.calls, [_CRD_QUERY])

    def test_check_kube_config_error_raises_exception(self):
        """Test that unusable Kubernetes configuration is reported as such."""
        err = KubeError(
            "Kubernetes configuration could not be loaded: No configuration"
        )
        with patch_k8s_client(error=err):
            with self.assertRaises(ChsPreflightException) as ctx:
                self.checker.check()
        self.assertEqual(
            str(ctx.exception),
            f"Failed to query cluster CRDs from the Kubernetes API: {err}",
        )
        self.assertIs(ctx.exception.__cause__, err)

    def test_get_installed_crds_returns_lowercase_kubectl_names(self):
        """Test CRD names are returned in lowercase."""
        self._add_crds(
            "jobsets.jobset.x-k8s.io", "OtherResources.Example.X-K8s.io"
        )
        self.assertEqual(
            self.checker._get_installed_crds(),
            [
                "jobsets.jobset.x-k8s.io",
                "otherresources.example.x-k8s.io",
            ],
        )
        self.assertEqual(self.kube.calls, [_CRD_QUERY])

    def test_check_version_unparsed_tag_passes(self):
        """Test unparsed tag (:latest) passes."""
        self._add_crds(_JOBSET_CRD)
        self._add_deployment(_deployment("jobset", "latest"))
        tests = [{"name": "nccl-cluster"}, {"name": "nvidia-smi"}]
        result_tests = self.checker.check(tests)
        self.assertEqual(result_tests, tests)
        self.assertEqual(
            self.kube.calls,
            [_CRD_QUERY, _label_query("jobset")],
        )

    def test_check_with_none_tests_returns_none(self):
        """Test that checking without tests returns None."""
        self._add_crds(_JOBSET_CRD)
        self._add_healthy_jobset()
        result = self.checker.check(None)
        self.assertIsNone(result)
        self.assertEqual(
            self.kube.calls,
            [_CRD_QUERY, _label_query("jobset")],
        )

    def test_parse_version_string_ip_registry(self):
        """Test parsing JobSet version from registry URL with IP and port."""
        version_output = "192.168.1.100:5000/jobset/jobset:v0.7.2"
        version = self.checker._parse_version_string(version_output)
        self.assertEqual(version, "0.7.2")

    def test_parse_version_string_without_v(self):
        """Test parsing JobSet version without leading 'v'."""
        version_output = "registry.k8s.io/jobset/jobset:0.7.2"
        version = self.checker._parse_version_string(version_output)
        self.assertEqual(version, "0.7.2")

    def test_parse_version_string_with_sha256(self):
        """Test parsing JobSet version with trailing sha256 digest."""
        version_output = (
            "registry.k8s.io/jobset/jobset:v0.7.2@"
            "sha256:5f32b1a5e12f6b8f36c53d9e2a84c8a2b5b3a4f6d8e0c1b2a3f4e5d6c7b8a9f0"
        )
        version = self.checker._parse_version_string(version_output)
        self.assertEqual(version, "0.7.2")

    def test_extract_version_from_deployment_none_fields(self):
        """Test that None image or name in container dict is handled."""
        dep = {
            "spec": {
                "template": {
                    "spec": {
                        "containers": [
                            {"name": None, "image": None},
                            {
                                "name": "manager",
                                "image": "registry.k8s.io/jobset/jobset:v0.7.2",
                            },
                        ]
                    }
                }
            }
        }
        version = self.checker._extract_version_from_deployment(
            dep, app_name="jobset"
        )
        self.assertEqual(version, "0.7.2")

    def test_is_jobset_version_supported_exact(self):
        """Test exact match of minimum supported JobSet version."""
        self.assertTrue(self.checker._is_jobset_version_supported("0.6.0"))
        self.assertTrue(self.checker._is_jobset_version_supported("v0.6.0"))

    def test_is_jobset_version_supported_higher(self):
        """Test JobSet version higher than minimum."""
        self.assertTrue(self.checker._is_jobset_version_supported("0.7.0"))
        self.assertTrue(self.checker._is_jobset_version_supported("1.0.0"))

    def test_is_jobset_version_supported_lower(self):
        """Test JobSet version lower than minimum."""
        self.assertFalse(self.checker._is_jobset_version_supported("0.5.9"))
        self.assertFalse(self.checker._is_jobset_version_supported("v0.5.0"))

    def test_is_jobset_version_supported_invalid(self):
        """Test invalid JobSet version formats are assumed supported."""
        self.assertTrue(
            self.checker._is_jobset_version_supported("invalid-version")
        )
        self.assertTrue(self.checker._is_jobset_version_supported("latest"))

    def test_check_version_with_multicontainer_sidecar(self):
        """Test container selection skips kube-rbac-proxy at container 0."""
        self._add_crds(_JOBSET_CRD)
        containers = [
            {
                "name": "kube-rbac-proxy",
                "image": "gcr.io/kubebuilder/kube-rbac-proxy:v0.13.1",
            },
            {
                "name": "manager",
                "image": "registry.k8s.io/jobset/jobset:v0.7.2",
            },
        ]
        self._add_deployment(_deployment("jobset", containers=containers))
        tests = [{"name": "nccl-cluster"}, {"name": "nvidia-smi"}]
        filtered = self.checker.check(tests)
        self.assertEqual(filtered, tests)
        self.assertEqual(
            self.kube.calls,
            [_CRD_QUERY, _label_query("jobset")],
        )

    def test_discover_deployments_fallback_success(self):
        """Test that missing label triggers fallback to namespace query."""
        self._add_crds(_JOBSET_CRD)
        self._add_deployment(_deployment("jobset", "v0.7.2", labeled=False))
        tests = [{"name": "nccl-cluster"}]
        filtered = self.checker.check(tests)
        self.assertEqual(filtered, tests)
        self.assertEqual(
            self.kube.calls,
            [
                _CRD_QUERY,
                _label_query("jobset"),
                _fallback_query("jobset"),
            ],
        )

    def test_discover_deployments_fallback_fails(self):
        """Test that if both label and fallback name are different, it fails."""
        self._add_crds(_JOBSET_CRD)
        tests = [{"name": "nccl-cluster"}]
        with self.assertRaisesRegex(
            ChsPreflightException,
            "no JobSet controller deployment could be found",
        ):
            self.checker.check(tests)
        self.assertEqual(
            self.kube.calls,
            [_CRD_QUERY, _label_query("jobset"), _fallback_query("jobset")],
        )

    def test_jobset_version_incompatible_raises_exception(self):
        """Test that incompatible JobSet version raises exception."""
        self._add_crds(_JOBSET_CRD)
        self._add_deployment(_deployment("jobset", "v0.1.0"))
        tests = [{"name": "nccl-cluster"}]
        with self.assertRaisesRegex(
            ChsPreflightException, "JobSet is missing or incompatible"
        ):
            self.checker.check(tests)
        self.assertEqual(self.kube.calls, [_CRD_QUERY, _label_query("jobset")])


if __name__ == "__main__":
    unittest.main()
