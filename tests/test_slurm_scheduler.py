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

import logging
import unittest
from unittest.mock import patch
import os
import subprocess
from chs import exceptions as ex
from chs.constants import SLURM_CMD_TIMEOUT_SEC
from chs.schedulers.slurm import SlurmScheduler


class TestSlurmScheduler(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)
        self.scheduler = SlurmScheduler()

    @patch("subprocess.run")
    def test_get_job_statuses_success(self, mock_run):
        # Mock sacct output (JobID|State)
        mock_output = "123|RUNNING\n124|COMPLETED\n"
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=mock_output
        )

        statuses = self.scheduler.get_job_statuses(["123", "124", "125"])

        self.assertEqual(statuses["123"], "RUNNING")
        self.assertEqual(statuses["124"], "COMPLETED")
        # 125 not in output -> UNKNOWN
        self.assertEqual(statuses["125"], "UNKNOWN")

    @patch("subprocess.run")
    def test_get_job_statuses_command_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError

        statuses = self.scheduler.get_job_statuses(["123"])

        # Should fallback to UNKNOWN
        self.assertEqual(statuses["123"], "UNKNOWN")

    @patch("os.path.exists")
    def test_fetch_job_log_exists(self, mock_exists):
        mock_exists.return_value = True
        # Should not raise
        self.scheduler.fetch_job_log("123", "/path/to/log")
        mock_exists.assert_called_with("/path/to/log")

    @patch("os.path.exists")
    def test_fetch_job_log_missing(self, mock_exists):
        mock_exists.return_value = False
        # Should log warning but not raise
        self.scheduler.fetch_job_log("123", "/path/to/log")

    def test_get_extra_env(self):
        """Test get_extra_env returns expected Enroot paths."""
        env = self.scheduler.get_extra_env()
        home = os.path.expanduser("~")
        self.assertEqual(
            env["ENROOT_RUNTIME_PATH"], os.path.join(home, ".enroot", "run")
        )
        self.assertEqual(
            env["ENROOT_CACHE_PATH"], os.path.join(home, ".enroot", "cache")
        )

    @patch("os.makedirs")
    @patch("os.path.exists")
    def test_ensure_enroot_directories(self, mock_exists, mock_makedirs):
        mock_exists.return_value = False
        self.scheduler._ensure_enroot_directories()

        # Should not create credentials directory
        credentials_path = self.scheduler._enroot_paths[
            "ENROOT_CONFIG_CREDENTIALS"
        ]
        runtime_path = self.scheduler._enroot_paths["ENROOT_RUNTIME_PATH"]
        cache_path = self.scheduler._enroot_paths["ENROOT_CACHE_PATH"]

        mock_makedirs.assert_any_call(runtime_path, exist_ok=True)
        mock_makedirs.assert_any_call(cache_path, exist_ok=True)

        # Assert that credentials_path was NOT called with os.makedirs
        for call in mock_makedirs.call_args_list:
            self.assertNotEqual(call[0][0], credentials_path)

    @patch("google.auth.default")
    @patch("google.cloud.compute_v1.InstancesClient")
    def test_get_topology(self, mock_instances_client, mock_auth_default):
        """Test get_topology uses compute API."""
        mock_auth_default.return_value = (None, "my-project")

        mock_client_instance = mock_instances_client.return_value

        class MockInstance:
            def __init__(self, name, block, subblock):
                self.name = name

                class ResourceStatus:
                    def __init__(self, block, subblock):
                        class PhysicalHostTopology:
                            def __init__(self, block, subblock):
                                self.block = block
                                self.subblock = subblock

                        self.physical_host_topology = PhysicalHostTopology(
                            block, subblock
                        )

                self.resource_status = ResourceStatus(block, subblock)

        class MockInstancesInZone:
            def __init__(self, instances):
                self.instances = instances

        mock_instances = [
            MockInstance("n1", "s0_0", "s0_0_0"),
            MockInstance("n2", "s0_0", "s0_0_0"),
            MockInstance("n3", "s0_0", "s0_1"),
            MockInstance("n4", "s0_0", "s0_1"),
        ]

        mock_aggregated_list = [
            ("zones/us-central1-a", MockInstancesInZone(mock_instances))
        ]

        mock_client_instance.aggregated_list.return_value = mock_aggregated_list

        with patch.object(
            self.scheduler,
            "get_nodes_under_test",
            return_value=["n1", "n2", "n3", "n4"],
        ):
            topology = self.scheduler.get_topology()

        self.assertEqual(len(topology), 1)
        self.assertIn("s0_0", topology)
        self.assertIn("s0_0_0", topology["s0_0"])
        self.assertIn("s0_1", topology["s0_0"])
        self.assertEqual(topology["s0_0"]["s0_0_0"], ["n1", "n2"])
        self.assertEqual(topology["s0_0"]["s0_1"], ["n3", "n4"])

    def test_allow_interdomain_pairs(self):
        self.assertFalse(
            self.scheduler.allow_interdomain_pairs("a4x-highgpu-4g")
        )
        self.assertFalse(
            self.scheduler.allow_interdomain_pairs("a4x-maxgpu-4g-metal")
        )
        self.assertTrue(self.scheduler.allow_interdomain_pairs("a3-megagpu-8g"))
        self.assertTrue(self.scheduler.allow_interdomain_pairs())

    @patch("subprocess.run")
    def test_run_slurm_cmd_passes_timeout(self, mock_run):
        self.scheduler._run_slurm_cmd(["squeue"], check=True)
        mock_run.assert_called_once_with(
            ["squeue"],
            text=True,
            capture_output=True,
            check=True,
            timeout=SLURM_CMD_TIMEOUT_SEC,
        )

    @patch("subprocess.run")
    def test_run_slurm_cmd_timeout_override(self, mock_run):
        self.scheduler._run_slurm_cmd(["sacct"], timeout=5)
        self.assertEqual(mock_run.call_args.kwargs["timeout"], 5)

    @patch("subprocess.run")
    def test_resolve_raw_nodes_sinfo_timeout_raises(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired("sinfo", 60)
        with self.assertRaises(ex.ChsSchedulerException):
            self.scheduler.resolve_raw_nodes()

    @patch("subprocess.run")
    def test_get_job_statuses_timeout_returns_unknown(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired("sacct", 60)
        statuses = self.scheduler.get_job_statuses(["123", "124"])
        self.assertEqual(statuses, {"123": "UNKNOWN", "124": "UNKNOWN"})

    @patch("subprocess.run")
    def test_get_active_job_names_timeout_raises(self, mock_run):
        """A timeout must not look like "no jobs" to the foreign-run check."""
        mock_run.side_effect = subprocess.TimeoutExpired("squeue", 60)
        with self.assertRaisesRegex(
            ex.ChsSchedulerException, "Timed out listing active jobs"
        ):
            self.scheduler.get_active_job_names()


if __name__ == "__main__":
    unittest.main()
