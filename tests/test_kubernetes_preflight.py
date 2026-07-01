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
import subprocess
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

    def test_check_crds_installed_version_valid(self):
        """Test that preflight check passes if JobSet/Kueue are valid."""
        crd_output = (
            "customresourcedefinition.apiextensions.k8s.io/jobsets.jobset.sigs.k8s.io\n"
            "customresourcedefinition.apiextensions.k8s.io/kueues.kueue.sigs.k8s.io"
        )
        version_output = "registry.k8s.io/kueue/kueue:v0.14.5"
        with mock.patch.object(self.checker, "_run_kubectl") as mock_run:
            mock_run.side_effect = [crd_output, version_output]
            original_tests = [{"name": "test1"}, {"name": "nccl-cluster"}]
            filtered_tests, kueue_supported = self.checker.check(original_tests)
            self.assertEqual(mock_run.call_count, 2)
            self.assertEqual(filtered_tests, original_tests)
            self.assertTrue(kueue_supported)

    def test_check_crds_installed_version_invalid(self):
        """Test that 'nccl-cluster' is removed if Kueue version < 0.14.5."""
        crd_output = (
            "customresourcedefinition.apiextensions.k8s.io/jobsets.jobset.sigs.k8s.io\n"
            "customresourcedefinition.apiextensions.k8s.io/kueues.kueue.sigs.k8s.io"
        )
        version_output = "registry.k8s.io/kueue/kueue:v0.13.0"
        with mock.patch.object(self.checker, "_run_kubectl") as mock_run:
            mock_run.side_effect = [crd_output, version_output]
            tests = [{"name": "nccl-cluster"}, {"name": "nvidia-smi"}]
            filtered, kueue_supported = self.checker.check(tests)
            self.assertEqual(filtered, [{"name": "nvidia-smi"}])
            self.assertFalse(kueue_supported)

    def test_check_crds_missing_raises_exception(self):
        """Test that missing JobSet causes it to fail fast."""
        with mock.patch.object(self.checker, "_run_kubectl", return_value=""):
            with self.assertRaisesRegex(
                ChsPreflightException,
                "Missing required cluster dependencies: JobSet",
            ):
                self.checker.check()

    def test_check_kueue_missing_filters_tests(self):
        """Test that missing Kueue causes nccl-cluster to be filtered out."""
        crd_output = (
            "customresourcedefinition.apiextensions.k8s.io/"
            "jobsets.jobset.sigs.k8s.io"
        )
        with mock.patch.object(
            self.checker, "_run_kubectl", return_value=crd_output
        ):
            tests = [{"name": "nccl-cluster"}, {"name": "nvidia-smi"}]
            filtered, kueue_supported = self.checker.check(tests)
            self.assertEqual(filtered, [{"name": "nvidia-smi"}])
            self.assertFalse(kueue_supported)

    def test_check_crds_query_failure_raises_exception(self):
        """Test that Error when getting CRDs raises ChsPreflightException."""
        with mock.patch.object(
            self.checker,
            "_run_kubectl",
            side_effect=subprocess.CalledProcessError(1, "kubectl get crd"),
        ):
            with self.assertRaisesRegex(
                ChsPreflightException,
                "Failed to query cluster CRDs:",
            ):
                self.checker.check()

    def test_check_version_parsing_failure_logs_warning(self):
        """Test that an unparsed image tag emits a warning and filters tests."""
        crd_output = (
            "customresourcedefinition.apiextensions.k8s.io/jobsets.jobset.sigs.k8s.io\n"
            "customresourcedefinition.apiextensions.k8s.io/kueues.kueue.sigs.k8s.io"
        )
        version_output = "registry.k8s.io/kueue/kueue:latest"
        with mock.patch.object(self.checker, "_run_kubectl") as mock_run:
            mock_run.side_effect = [crd_output, version_output]
            tests = [{"name": "nccl-cluster"}, {"name": "nvidia-smi"}]
            with mock.patch(
                "chs.preflights.kubernetes_preflight.logger.warning"
            ) as mock_warn:
                filtered, kueue_supported = self.checker.check(tests)
                self.assertEqual(filtered, [{"name": "nvidia-smi"}])
                self.assertFalse(kueue_supported)
                mock_warn.assert_called()
                self.assertTrue(
                    any(
                        "Could not parse Kueue version from image" in arg[0]
                        for arg, _ in mock_warn.call_args_list
                    )
                )

    def test_check_without_nccl_cluster_returns_tests_unchanged(self):
        """Test that test filtering returns all tests if no 'nccl-cluster'."""
        crd_output = (
            "customresourcedefinition.apiextensions.k8s.io/"
            "jobsets.jobset.sigs.k8s.io"
        )
        with mock.patch.object(
            self.checker, "_run_kubectl", return_value=crd_output
        ):
            tests = [{"name": "nvidia-smi"}, {"name": "dcgm"}]
            filtered, kueue_supported = self.checker.check(tests)
            self.assertEqual(filtered, tests)
            self.assertFalse(kueue_supported)

    def test_check_with_none_tests_returns_none(self):
        """Test that checking without tests returns None."""
        crd_output = (
            "customresourcedefinition.apiextensions.k8s.io/jobsets.jobset.sigs.k8s.io\n"
            "customresourcedefinition.apiextensions.k8s.io/kueues.kueue.sigs.k8s.io"
        )
        version_output = "registry.k8s.io/kueue/kueue:v0.14.5"
        with mock.patch.object(self.checker, "_run_kubectl") as mock_run:
            mock_run.side_effect = [crd_output, version_output]
            filtered, kueue_supported = self.checker.check(None)
            self.assertIsNone(filtered)
            self.assertTrue(kueue_supported)


if __name__ == "__main__":
    unittest.main()
