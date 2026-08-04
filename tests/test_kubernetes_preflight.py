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

import json
import subprocess
import unittest
from unittest import mock
from chs.exceptions import ChsPreflightException
from chs.preflights.kubernetes_preflight import KubernetesPreflightChecker


class TestKubernetesPreflightChecker(unittest.TestCase):
    """Tests for KubernetesPreflightChecker."""

    def setUp(self):
        super().setUp()
        self.mock_scheduler = mock.Mock()
        self.mock_app_data = mock.Mock()
        self.mock_scheduler._app_data = self.mock_app_data
        self.checker = KubernetesPreflightChecker(self.mock_scheduler)

    def _healthy_jobset_deployment_json(self):
        return json.dumps(
            {
                "items": [
                    {
                        "kind": "Deployment",
                        "metadata": {
                            "name": "jobset-controller-manager",
                            "namespace": "jobset-system",
                            "labels": {
                                "app.kubernetes.io/name": "jobset",
                                "app.kubernetes.io/version": "v0.7.2",
                            },
                        },
                        "spec": {
                            "template": {
                                "spec": {
                                    "containers": [
                                        {
                                            "image": (
                                                "registry.k8s.io/jobset/jobset:"
                                                "v0.7.2"
                                            )
                                        }
                                    ]
                                }
                            }
                        },
                        "status": {
                            "availableReplicas": 1,
                            "conditions": [
                                {"type": "Available", "status": "True"}
                            ],
                        },
                    }
                ]
            }
        )

    def _healthy_kueue_deployment_json(self):
        return json.dumps(
            {
                "items": [
                    {
                        "kind": "Deployment",
                        "metadata": {
                            "name": "kueue-controller-manager",
                            "namespace": "kueue-system",
                            "labels": {"app.kubernetes.io/name": "kueue"},
                        },
                        "spec": {
                            "template": {
                                "spec": {
                                    "containers": [
                                        {
                                            "image": (
                                                "registry.k8s.io/kueue/kueue:"
                                                "v0.14.5"
                                            )
                                        }
                                    ]
                                }
                            }
                        },
                        "status": {
                            "availableReplicas": 1,
                            "conditions": [
                                {"type": "Available", "status": "True"}
                            ],
                        },
                    }
                ]
            }
        )

    def _unhealthy_jobset_deployment_json(self):
        return json.dumps(
            {
                "items": [
                    {
                        "kind": "Deployment",
                        "metadata": {
                            "name": "jobset-controller-manager",
                            "namespace": "jobset-system",
                            "labels": {
                                "app.kubernetes.io/name": "jobset",
                                "app.kubernetes.io/version": "v0.7.2",
                            },
                        },
                        "spec": {
                            "template": {
                                "spec": {
                                    "containers": [
                                        {
                                            "image": (
                                                "registry.k8s.io/jobset/jobset:"
                                                "v0.7.2"
                                            )
                                        }
                                    ]
                                }
                            }
                        },
                        "status": {
                            "availableReplicas": 0,
                            "conditions": [
                                {"type": "Available", "status": "False"}
                            ],
                        },
                    }
                ]
            }
        )

    def _kueue_deployment_json(self, image_tag="v0.14.5", available=True):
        return json.dumps(
            {
                "items": [
                    {
                        "kind": "Deployment",
                        "metadata": {
                            "name": "kueue-controller-manager",
                            "namespace": "kueue-system",
                            "labels": {"app.kubernetes.io/name": "kueue"},
                        },
                        "spec": {
                            "template": {
                                "spec": {
                                    "containers": [
                                        {
                                            "image": (
                                                "registry.k8s.io/kueue/kueue:"
                                                f"{image_tag}"
                                            )
                                        }
                                    ]
                                }
                            }
                        },
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
                ]
            }
        )

    def test_check_crds_installed_version_valid(self):
        """Test that preflight check passes if JobSet/Kueue are valid."""
        crd_output = (
            "customresourcedefinition.apiextensions.k8s.io/jobsets.jobset.sigs.k8s.io\n"
            "customresourcedefinition.apiextensions.k8s.io/kueues.kueue.sigs.k8s.io"
        )
        jobset_dep = self._healthy_jobset_deployment_json()
        kueue_dep = self._kueue_deployment_json("v0.14.5")
        with mock.patch.object(self.checker, "_run_kubectl") as mock_run:
            mock_run.side_effect = [crd_output, jobset_dep, kueue_dep]
            original_tests = [{"name": "test1"}, {"name": "nccl-cluster"}]
            filtered_tests, kueue_supported = self.checker.check(original_tests)
            self.assertEqual(mock_run.call_count, 3)
            self.assertEqual(filtered_tests, original_tests)
            self.assertTrue(kueue_supported)

    def test_check_crds_installed_version_invalid(self):
        """Test that invalid Kueue version raises exception."""
        crd_output = (
            "customresourcedefinition.apiextensions.k8s.io/jobsets.jobset.sigs.k8s.io\n"
            "customresourcedefinition.apiextensions.k8s.io/kueues.kueue.sigs.k8s.io"
        )
        jobset_dep = self._healthy_jobset_deployment_json()
        kueue_dep = self._kueue_deployment_json("v0.13.0")
        with mock.patch.object(self.checker, "_run_kubectl") as mock_run:
            mock_run.side_effect = [crd_output, jobset_dep, kueue_dep]
            tests = [
                {"name": "nccl-cluster"},
                {"name": "tpu-info"},
                {"name": "nvidia-smi"},
            ]
            with self.assertRaisesRegex(
                ChsPreflightException, "Kueue is missing or incompatible"
            ):
                self.checker.check(tests)

    def test_check_kueue_deployment_unhealthy_raises_exception(self):
        """Test that unhealthy Kueue deployment raises exception."""
        crd_output = (
            "customresourcedefinition.apiextensions.k8s.io/jobsets.jobset.sigs.k8s.io\n"
            "customresourcedefinition.apiextensions.k8s.io/kueues.kueue.sigs.k8s.io"
        )
        jobset_dep = self._healthy_jobset_deployment_json()
        kueue_dep = self._kueue_deployment_json("v0.14.5", available=False)
        with mock.patch.object(self.checker, "_run_kubectl") as mock_run:
            mock_run.side_effect = [crd_output, jobset_dep, kueue_dep]
            tests = [
                {"name": "nccl-cluster"},
                {"name": "tpu-info"},
                {"name": "nvidia-smi"},
            ]
            with self.assertRaisesRegex(
                ChsPreflightException, "Kueue is missing or incompatible"
            ):
                self.checker.check(tests)

    def test_check_crds_missing_raises_exception(self):
        """Test that missing JobSet causes it to fail fast."""
        with mock.patch.object(self.checker, "_run_kubectl", return_value=""):
            with self.assertRaisesRegex(
                ChsPreflightException,
                "JobSet is missing or incompatible",
            ):
                self.checker.check()

    def test_check_jobset_deployment_unhealthy_raises_exception(self):
        """Test that unhealthy deployment (0 replicas) raises exception."""
        crd_output = (
            "customresourcedefinition.apiextensions.k8s.io/"
            "jobsets.jobset.sigs.k8s.io"
        )
        dep_output = self._unhealthy_jobset_deployment_json()
        with mock.patch.object(self.checker, "_run_kubectl") as mock_run:
            mock_run.side_effect = [crd_output, dep_output]
            with self.assertRaisesRegex(
                ChsPreflightException,
                "JobSet CRD is installed, but the JobSet controller "
                "deployment is NOT in an Available state",
            ):
                self.checker.check()

    def test_check_jobset_deployment_rbac_restricted_raises_exception(self):
        """Test that RBAC query raises exception when deployment unverified."""
        crd_output = (
            "customresourcedefinition.apiextensions.k8s.io/"
            "jobsets.jobset.sigs.k8s.io"
        )
        with mock.patch.object(self.checker, "_run_kubectl") as mock_run:
            mock_run.side_effect = [
                crd_output,
                subprocess.CalledProcessError(1, "kubectl get deployment -A"),
            ]
            tests = [{"name": "nvidia-smi"}]
            with self.assertRaisesRegex(
                ChsPreflightException,
                "JobSet CRD is installed, but the JobSet controller "
                "deployment is NOT in an Available state",
            ):
                self.checker.check(tests)

    def test_check_jobset_deployment_null_available_replicas(self):
        """Test handling when status.availableReplicas is null in JSON."""
        crd_output = (
            "customresourcedefinition.apiextensions.k8s.io/"
            "jobsets.jobset.sigs.k8s.io"
        )
        null_replicas_dep = json.dumps(
            {
                "items": [
                    {
                        "kind": "Deployment",
                        "status": {
                            "availableReplicas": None,
                            "conditions": [
                                {"type": "Available", "status": "True"}
                            ],
                        },
                    }
                ]
            }
        )
        with mock.patch.object(self.checker, "_run_kubectl") as mock_run:
            mock_run.side_effect = [crd_output, null_replicas_dep]
            with self.assertRaisesRegex(
                ChsPreflightException,
                "JobSet CRD is installed, but the JobSet controller "
                "deployment is NOT in an Available state",
            ):
                self.checker.check()

    def test_check_with_none_tests_and_unsupported_kueue_raises_exception(self):
        """Test checking None tests with unsupported Kueue raises exception."""
        crd_output = (
            "customresourcedefinition.apiextensions.k8s.io/"
            "jobsets.jobset.sigs.k8s.io"
        )
        dep_output = self._healthy_jobset_deployment_json()
        with mock.patch.object(
            self.checker, "_run_kubectl", side_effect=[crd_output, dep_output]
        ):
            with self.assertRaisesRegex(
                ChsPreflightException, "Kueue is missing or incompatible"
            ):
                self.checker.check(None)

    def test_check_kueue_missing_raises_exception(self):
        """Test that missing Kueue raises exception."""
        crd_output = (
            "customresourcedefinition.apiextensions.k8s.io/"
            "jobsets.jobset.sigs.k8s.io"
        )
        dep_output = self._healthy_jobset_deployment_json()
        with mock.patch.object(
            self.checker, "_run_kubectl", side_effect=[crd_output, dep_output]
        ):
            tests = [{"name": "nccl-cluster"}, {"name": "nvidia-smi"}]
            with self.assertRaisesRegex(
                ChsPreflightException, "Kueue is missing or incompatible"
            ):
                self.checker.check(tests)

    def test_check_crds_query_failure_raises_exception(self):
        """Test that Error when getting CRDs raises ChsPreflightException."""
        with mock.patch.object(
            self.checker,
            "_run_kubectl",
            side_effect=subprocess.CalledProcessError(1, "kubectl get crd"),
        ):
            with self.assertRaisesRegex(
                ChsPreflightException,
                "Failed to query cluster CRDs via kubectl:",
            ):
                self.checker.check()

    def test_check_version_unparsed_tag_passes(self):
        """Test unparsed tag (:latest) passes."""
        crd_output = (
            "customresourcedefinition.apiextensions.k8s.io/jobsets.jobset.sigs.k8s.io\n"
            "customresourcedefinition.apiextensions.k8s.io/kueues.kueue.sigs.k8s.io"
        )
        jobset_dep = self._healthy_jobset_deployment_json()
        kueue_dep = self._kueue_deployment_json("latest")
        with mock.patch.object(self.checker, "_run_kubectl") as mock_run:
            mock_run.side_effect = [
                crd_output,
                jobset_dep,
                kueue_dep,
            ]
            tests = [{"name": "nccl-cluster"}, {"name": "nvidia-smi"}]
            result_tests, supported = self.checker.check(tests)
            self.assertEqual(result_tests, tests)
            self.assertTrue(supported)

    def test_check_without_kueue_raises_exception(self):
        """Test that missing Kueue raises exception even for non-Kueue tests."""
        crd_output = (
            "customresourcedefinition.apiextensions.k8s.io/"
            "jobsets.jobset.sigs.k8s.io"
        )
        jobset_dep = self._healthy_jobset_deployment_json()
        with mock.patch.object(
            self.checker, "_run_kubectl", side_effect=[crd_output, jobset_dep]
        ):
            tests = [{"name": "nvidia-smi"}, {"name": "dcgm"}]
            with self.assertRaisesRegex(
                ChsPreflightException, "Kueue is missing or incompatible"
            ):
                self.checker.check(tests)

    def test_check_with_none_tests_returns_none(self):
        """Test that checking without tests returns None."""
        crd_output = (
            "customresourcedefinition.apiextensions.k8s.io/jobsets.jobset.sigs.k8s.io\n"
            "customresourcedefinition.apiextensions.k8s.io/kueues.kueue.sigs.k8s.io"
        )
        jobset_dep = self._healthy_jobset_deployment_json()
        kueue_dep = self._kueue_deployment_json("v0.14.5")
        with mock.patch.object(self.checker, "_run_kubectl") as mock_run:
            mock_run.side_effect = [crd_output, jobset_dep, kueue_dep]
            filtered, kueue_supported = self.checker.check(None)
            self.assertIsNone(filtered)
            self.assertTrue(kueue_supported)

    def test_get_kueue_version_ip_registry(self):
        """Test parsing Kueue version from registry URL with IP and port."""
        version_output = "192.168.1.100:5000/kueue/kueue:v0.14.5"
        with mock.patch.object(
            self.checker, "_run_kubectl", return_value=version_output
        ):
            version = self.checker._get_kueue_version()
            self.assertEqual(version, "0.14.5")

    def test_get_kueue_version_without_v(self):
        """Test parsing Kueue version without leading 'v'."""
        version_output = "registry.k8s.io/kueue/kueue:0.14.5"
        with mock.patch.object(
            self.checker, "_run_kubectl", return_value=version_output
        ):
            version = self.checker._get_kueue_version()
            self.assertEqual(version, "0.14.5")

    def test_get_kueue_version_with_sha256(self):
        """Test parsing Kueue version with trailing sha256 digest."""
        version_output = (
            "registry.k8s.io/kueue/kueue:v0.14.5@"
            "sha256:5f32b1a5e12f6b8f36c53d9e2a84c8a2b5b3a4f6d8e0c1b2a3f4e5d6c7b8a9f0"
        )
        with mock.patch.object(
            self.checker, "_run_kubectl", return_value=version_output
        ):
            version = self.checker._get_kueue_version()
            self.assertEqual(version, "0.14.5")

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
                                "image": "registry.k8s.io/kueue/kueue:v0.14.5",
                            },
                        ]
                    }
                }
            }
        }
        version = self.checker._extract_version_from_deployment(
            dep, app_name="kueue"
        )
        self.assertEqual(version, "0.14.5")

    def test_get_kueue_version_timeout(self):
        """Test that TimeoutExpired in kubectl command returns None cleanly."""
        with mock.patch.object(
            self.checker,
            "_run_kubectl",
            side_effect=subprocess.TimeoutExpired(cmd="kubectl", timeout=120),
        ):
            version = self.checker._get_kueue_version()
            self.assertIsNone(version)

    def test_is_kueue_version_supported_exact(self):
        """Test exact match of minimum supported Kueue version."""
        self.assertTrue(self.checker._is_kueue_version_supported("0.14.5"))
        self.assertTrue(self.checker._is_kueue_version_supported("v0.14.5"))

    def test_is_kueue_version_supported_higher(self):
        """Test Kueue version higher than minimum."""
        self.assertTrue(self.checker._is_kueue_version_supported("0.15.0"))
        self.assertTrue(self.checker._is_kueue_version_supported("1.0.0"))

    def test_is_kueue_version_supported_lower(self):
        """Test Kueue version lower than minimum."""
        self.assertFalse(self.checker._is_kueue_version_supported("0.14.4"))
        self.assertFalse(self.checker._is_kueue_version_supported("v0.13.0"))

    def test_is_kueue_version_supported_prerelease(self):
        """Test that a prerelease of the minimum version is not supported."""
        self.assertFalse(self.checker._is_kueue_version_supported("0.14.5-rc1"))
        self.assertTrue(self.checker._is_kueue_version_supported("0.14.6-rc1"))

    def test_is_kueue_version_supported_invalid(self):
        """Test invalid Kueue version formats are assumed supported."""
        self.assertTrue(
            self.checker._is_kueue_version_supported("invalid-version")
        )
        self.assertTrue(self.checker._is_kueue_version_supported("latest"))

    def test_check_version_with_multicontainer_sidecar(self):
        """Test container selection skips kube-rbac-proxy at container 0."""
        crd_output = (
            "customresourcedefinition.apiextensions.k8s.io/jobsets.jobset.sigs.k8s.io\n"
            "customresourcedefinition.apiextensions.k8s.io/kueues.kueue.sigs.k8s.io"
        )
        jobset_dep = self._healthy_jobset_deployment_json()
        containers = [
            {
                "name": "kube-rbac-proxy",
                "image": "gcr.io/kubebuilder/kube-rbac-proxy:v0.13.1",
            },
            {
                "name": "manager",
                "image": "registry.k8s.io/kueue/kueue:v0.14.5",
            },
        ]
        kueue_dep_sidecar = json.dumps(
            {
                "items": [
                    {
                        "kind": "Deployment",
                        "metadata": {
                            "name": "kueue-controller-manager",
                            "namespace": "kueue-system",
                            "labels": {"app.kubernetes.io/name": "kueue"},
                        },
                        "spec": {
                            "template": {"spec": {"containers": containers}}
                        },
                        "status": {
                            "availableReplicas": 1,
                            "conditions": [
                                {"type": "Available", "status": "True"}
                            ],
                        },
                    }
                ]
            }
        )
        with mock.patch.object(self.checker, "_run_kubectl") as mock_run:
            mock_run.side_effect = [crd_output, jobset_dep, kueue_dep_sidecar]
            tests = [{"name": "nccl-cluster"}, {"name": "nvidia-smi"}]
            filtered, kueue_supported = self.checker.check(tests)
            self.assertEqual(filtered, tests)
            self.assertTrue(kueue_supported)


if __name__ == "__main__":
    unittest.main()
