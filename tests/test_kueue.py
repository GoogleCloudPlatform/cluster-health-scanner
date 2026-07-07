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

"""Unit tests for KueueManager."""

import os
import subprocess
import tempfile
import unittest
from unittest import mock

from chs.exceptions import KueueException
from chs.utils.kueue import KueueManager


class TestKueueManager(unittest.TestCase):
    """Tests for KueueManager."""

    def setUp(self):
        super().setUp()
        self.mock_scheduler = mock.Mock()
        self.mock_scheduler.kueue_supported = False
        self.mock_scheduler.workspace_config_dir = None
        self.kueue_mgr = KueueManager(self.mock_scheduler)
        self.tmp_dir_obj = tempfile.TemporaryDirectory()
        self.tmp_dir = self.tmp_dir_obj.name
        self.addCleanup(self.tmp_dir_obj.cleanup)

    def test_generate_configs_unsupported_machine_type_raises(self):
        """Test generate_configs with unsupported machine type raises
        KueueException."""
        with self.assertRaisesRegex(
            KueueException, "not recognized in MACHINE_TYPE_RESOURCE_MAP"
        ):
            self.kueue_mgr.generate_configs(
                self.tmp_dir, "unsupported-mt", ["node1"]
            )

    def test_generate_configs_success_renders(self):
        """Test generate_configs renders template and returns correct label."""
        label = self.kueue_mgr.generate_configs(
            self.tmp_dir, "a3-megagpu-8g", ["node1", "node2"]
        )
        self.assertEqual(label, "kueue.x-k8s.io/queue-name: chs-local-queue")

        config_path = os.path.join(self.tmp_dir, "kueue-configs.yaml")
        self.assertTrue(os.path.exists(config_path))
        with open(config_path, "r") as f:
            content = f.read()
            self.assertIn("a3-megagpu-8g", content)
            self.assertIn("nvidia.com/gpu", content)
            self.assertIn('nominalQuota: "16"', content)

    def test_generate_configs_tpu_success_renders(self):
        """Test generate_configs for TPU renders correct flavor and
        resource name."""
        label = self.kueue_mgr.generate_configs(
            self.tmp_dir, "tpu7x-standard-4t", ["node1", "node2"]
        )
        self.assertEqual(label, "kueue.x-k8s.io/queue-name: chs-local-queue")

        config_path = os.path.join(self.tmp_dir, "kueue-configs.yaml")
        self.assertTrue(os.path.exists(config_path))
        with open(config_path, "r") as f:
            content = f.read()
            self.assertIn("tpu7x-standard-4t", content)
            self.assertIn("google.com/tpu", content)
            self.assertIn('nominalQuota: "8"', content)

    def test_apply_configs_success(self):
        """Test apply_configs cleans up and applies configurations."""
        self.mock_scheduler._run_kubectl.side_effect = [
            "",  # cleanup localqueue
            "",  # cleanup clusterqueue
            "",  # cleanup resourceflavor
            "",  # apply
            "",  # wait
        ]
        self.kueue_mgr.apply_configs(self.tmp_dir)
        self.assertEqual(self.mock_scheduler._run_kubectl.call_count, 5)

    def test_apply_and_verify_permission_error_raises(self):
        """Test applying config with kubectl error raises KueueException."""
        self.mock_scheduler._run_kubectl.side_effect = (
            subprocess.CalledProcessError(
                1, "kubectl apply", stderr="permission denied"
            )
        )
        with self.assertRaisesRegex(
            KueueException, "ensure you have cluster-level permissions"
        ):
            self.kueue_mgr.apply_and_verify("/tmp/nonexistent.yaml")

    def test_apply_and_verify_success(self):
        """Test apply_and_verify successfully runs apply and wait commands."""
        self.mock_scheduler._run_kubectl.return_value = ""
        self.kueue_mgr.apply_and_verify("/tmp/kueue-configs.yaml")

        self.mock_scheduler._run_kubectl.assert_any_call(
            ["apply", "-f", "/tmp/kueue-configs.yaml"],
            timeout=None,
        )

        self.mock_scheduler._run_kubectl.assert_any_call(
            [
                "wait",
                "--for=condition=Active",
                "clusterqueue/chs-cluster-queue",
                "--timeout=300s",
            ],
            timeout=350,
        )

    def test_apply_and_verify_timeout_raises(self):
        """Test timeout during verification wait loop raises KueueException."""
        self.mock_scheduler._run_kubectl.side_effect = [
            "",
            subprocess.CalledProcessError(
                1, "kubectl wait", stderr="timed out"
            ),
        ]

        with self.assertRaisesRegex(
            KueueException, "Timeout or error waiting for Kueue ClusterQueue"
        ):
            self.kueue_mgr.apply_and_verify("/tmp/nonexistent.yaml")

        self.mock_scheduler._run_kubectl.assert_any_call(
            [
                "wait",
                "--for=condition=Active",
                "clusterqueue/chs-cluster-queue",
                "--timeout=300s",
            ],
            timeout=350,
        )

    def test_cleanup_success(self):
        """Test cleanup deletes resources via kubectl."""
        self.mock_scheduler._run_kubectl.return_value = ""
        self.kueue_mgr.cleanup()
        self.assertEqual(self.mock_scheduler._run_kubectl.call_count, 3)
        self.mock_scheduler._run_kubectl.assert_any_call(
            [
                "delete",
                "localqueue",
                "chs-local-queue",
                "-n",
                "chs",
                "--ignore-not-found",
            ],
            timeout=None,
        )
        self.mock_scheduler._run_kubectl.assert_any_call(
            [
                "delete",
                "clusterqueue",
                "chs-cluster-queue",
                "--ignore-not-found",
            ],
            timeout=None,
        )
        self.mock_scheduler._run_kubectl.assert_any_call(
            ["delete", "resourceflavor", "chs-flavor", "--ignore-not-found"],
            timeout=None,
        )

    def test_cleanup_failure_raises(self):
        """Test cleanup error raises KueueException."""
        self.mock_scheduler._run_kubectl.side_effect = (
            subprocess.CalledProcessError(
                1, "kubectl delete", stderr="conflict"
            )
        )
        with self.assertRaisesRegex(
            KueueException, "Failed to delete Kueue resources"
        ):
            self.kueue_mgr.cleanup()


if __name__ == "__main__":
    unittest.main()
