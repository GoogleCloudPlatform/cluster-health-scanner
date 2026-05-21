import logging
import unittest
from unittest.mock import patch
import os
import subprocess
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

    @patch("subprocess.run")
    def test_check_jobs_running_true(self, mock_run):
        mock_run.return_value.stdout = "123"
        self.assertTrue(self.scheduler.check_jobs_running(["123"]))

    @patch("subprocess.run")
    def test_check_jobs_running_false(self, mock_run):
        mock_run.return_value.stdout = ""
        self.assertFalse(self.scheduler.check_jobs_running(["123"]))

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


if __name__ == "__main__":
    unittest.main()
