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

    def test_setup_unsupported_machine_type_raises(self):
        """Test setup with unsupported machine type raises KueueException."""
        with self.assertRaisesRegex(
            KueueException, "not recognized in MACHINE_TYPE_RESOURCE_MAP"
        ):
            self.kueue_mgr.setup(self.tmp_dir, "unsupported-mt", ["node1"])

    def test_setup_already_active_returns_label(self):
        """Test setup returns queue label if ClusterQueue is already active."""
        self.mock_scheduler._run_kubectl.return_value = "True"
        label = self.kueue_mgr.setup(self.tmp_dir, "a3-mega", ["node1"])
        self.assertEqual(label, "kueue.x-k8s.io/queue-name: chs-local-queue")
        self.mock_scheduler._run_kubectl.assert_called_once()

    def test_setup_success_renders_and_verifies(self):
        """Test setup renders template, applies it, and verifies readiness."""

        self.mock_scheduler._run_kubectl.side_effect = [
            subprocess.CalledProcessError(1, "kubectl"),
            "",
            "",
        ]

        label = self.kueue_mgr.setup(
            self.tmp_dir, "a3-mega", ["node1", "node2"]
        )
        self.assertEqual(label, "kueue.x-k8s.io/queue-name: chs-local-queue")

        config_path = os.path.join(self.tmp_dir, "kueue-configs.yaml")
        self.assertTrue(os.path.exists(config_path))
        with open(config_path, "r") as f:
            content = f.read()
            self.assertIn("a3-megagpu-8g", content)
            self.assertIn("nvidia.com/gpu", content)
            self.assertIn('nominalQuota: "16"', content)

    def test_setup_tpu_success_renders_and_verifies(self):
        """Test setup for TPU renders correct resource flavor and name."""
        self.mock_scheduler._run_kubectl.side_effect = [
            subprocess.CalledProcessError(1, "kubectl"),
            "",
            "",
        ]

        label = self.kueue_mgr.setup(self.tmp_dir, "tpu7x", ["node1", "node2"])
        self.assertEqual(label, "kueue.x-k8s.io/queue-name: chs-local-queue")

        config_path = os.path.join(self.tmp_dir, "kueue-configs.yaml")
        self.assertTrue(os.path.exists(config_path))
        with open(config_path, "r") as f:
            content = f.read()
            self.assertIn("tpu7x-standard-4t", content)
            self.assertIn("google.com/tpu", content)
            self.assertIn('nominalQuota: "8"', content)

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

    @mock.patch("chs.utils.kueue.time.time")
    def test_apply_and_verify_timeout_raises(self, mock_time):
        """Test timeout during verification wait loop raises KueueException."""
        # start_time, loop check 1, log check, loop check 2
        mock_time.side_effect = [100, 101, 102, 405]
        self.mock_scheduler._run_kubectl.side_effect = [
            "",
            subprocess.CalledProcessError(
                1, "kubectl wait", stderr="timed out"
            ),
        ]

        with self.assertRaisesRegex(
            KueueException, "Timeout waiting for Kueue ClusterQueue"
        ):
            self.kueue_mgr.apply_and_verify("/tmp/nonexistent.yaml")

    def test_cleanup_nonexistent_file_skips(self):
        """Test cleanup skips if configuration file doesn't exist."""
        self.kueue_mgr.cleanup("/tmp/nonexistent_file_12345.yaml")
        self.mock_scheduler._run_kubectl.assert_not_called()

    def test_cleanup_success(self):
        """Test cleanup deletes resources via kubectl."""
        config_path = os.path.join(self.tmp_dir, "kueue-configs.yaml")
        with open(config_path, "w") as f:
            f.write("dummy")

        self.mock_scheduler._run_kubectl.return_value = ""
        self.kueue_mgr.cleanup(config_path)
        self.mock_scheduler._run_kubectl.assert_called_once_with(
            ["delete", "-f", config_path]
        )

    def test_cleanup_failure_raises(self):
        """Test cleanup error raises KueueException."""
        config_path = os.path.join(self.tmp_dir, "kueue-configs.yaml")
        with open(config_path, "w") as f:
            f.write("dummy")

        self.mock_scheduler._run_kubectl.side_effect = (
            subprocess.CalledProcessError(
                1, "kubectl delete", stderr="conflict"
            )
        )
        with self.assertRaisesRegex(
            KueueException, "Failed to delete Kueue resources"
        ):
            self.kueue_mgr.cleanup(config_path)


if __name__ == "__main__":
    unittest.main()
