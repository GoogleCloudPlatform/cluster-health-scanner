import sys
import unittest
from unittest.mock import MagicMock, patch, mock_open

# Mock problematic imports before importing RambleManager
sys.modules["chs.boh_generator"] = MagicMock()
sys.modules["chs.protos"] = MagicMock()
sys.modules["chs.protos.bill_of_health_pb2"] = MagicMock()

# Re-import to ensure mocks are used if it was already imported?
# unittest loader might have tried.
# Safe strategy: keep these mocks active.

from chs.ramble_manager import RambleManager  # noqa: E402
from chs.schedulers.gke import GKEScheduler  # noqa: E402
from chs.schedulers.slurm import SlurmScheduler  # noqa: E402


class TestJobMonitoring(unittest.TestCase):
    def setUp(self):
        self.mock_run_command = MagicMock()
        with patch("os.path.exists", return_value=True), patch("os.makedirs"):
            self.manager = RambleManager(
                base_dir="/tmp/test_workspaces",
                workspace_name="test_run",
                run_command=self.mock_run_command,
            )
        # Default to GKE for existing tests, can override in specific tests
        self.manager._scheduler = GKEScheduler()

    @patch("chs.ramble_manager.get_scheduler")
    @patch("chs.ramble_manager.RambleManager.get_job_ids")
    @patch("os.walk")
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data=(
            "apiVersion: v1\nkind: Service\nmetadata:\n  name: service-1\n---\n"
            "apiVersion: jobset.x-k8s.io/v1alpha2\n"
            "kind: JobSet\nmetadata:\n  name: job-123"
        ),
    )
    @patch("os.path.exists")
    def test_map_jobs_to_log_paths(
        self,
        mock_exists,
        mock_file,
        mock_walk,
        mock_get_job_ids,
        mock_get_scheduler,
    ):
        mock_get_job_ids.return_value = ["job-123"]
        mock_exists.return_value = True
        mock_walk.return_value = [
            (
                "/tmp/test_workspaces/test_run/experiments/exp1",
                [],
                ["jobset.yaml", "exp1.out"],
            )
        ]

        mapping = self.manager._map_jobs_to_log_paths()

        self.assertEqual(
            mapping,
            {
                "job-123": (
                    "/tmp/test_workspaces/test_run/experiments/exp1/exp1.out"
                )
            },
        )

    @patch("chs.ramble_manager.get_scheduler")
    @patch("chs.ramble_manager.RambleManager.get_job_ids")
    @patch("chs.ramble_manager.RambleManager._map_jobs_to_log_paths")
    @patch("time.sleep")
    def test_monitor_jobs_fetches_logs_gke(
        self, mock_sleep, mock_map, mock_get_job_ids, mock_get_scheduler
    ):
        # Setup
        mock_scheduler = MagicMock()
        mock_get_scheduler.return_value = mock_scheduler

        mock_get_job_ids.return_value = ["job-1", "job-2"]
        mock_map.return_value = {
            "job-1": "/path/to/job-1.log",
            "job-2": "/path/to/job-2.log",
        }

        # Mock job completion (job-1 running -> completed, job-2 completed)
        mock_scheduler.get_job_statuses.side_effect = [
            {"job-1": "RUNNING", "job-2": "COMPLETED"},
            {"job-1": "COMPLETED"},
        ]

        # Execute
        self.manager._monitor_jobs(poll_interval=1)

        # Verify calls
        mock_scheduler.fetch_job_log.assert_any_call(
            "job-2", "/path/to/job-2.log"
        )
        mock_scheduler.fetch_job_log.assert_any_call(
            "job-1", "/path/to/job-1.log"
        )
        self.assertEqual(mock_scheduler.fetch_job_log.call_count, 2)

    @patch("chs.ramble_manager.get_scheduler")
    @patch("chs.ramble_manager.RambleManager.get_job_ids")
    @patch("chs.ramble_manager.RambleManager._map_jobs_to_log_paths")
    def test_monitor_jobs_handles_fetch_error(
        self, mock_map, mock_get_job_ids, mock_get_scheduler
    ):
        # Setup
        mock_scheduler = MagicMock()
        mock_get_scheduler.return_value = mock_scheduler

        mock_get_job_ids.return_value = ["job-1"]
        mock_map.return_value = {"job-1": "/path/log"}

        mock_scheduler.get_job_statuses.return_value = {"job-1": "COMPLETED"}
        mock_scheduler.fetch_job_log.side_effect = Exception("Fetch failed")

        # Execute (should not raise)
        self.manager._monitor_jobs(poll_interval=1)

        # Verify
        mock_scheduler.fetch_job_log.assert_called_once_with(
            "job-1", "/path/log"
        )

    @patch("os.walk")
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="invalid: yaml: content",
    )
    @patch("os.path.exists", return_value=True)
    def test_map_jobs_to_log_paths_handles_malformed_yaml(
        self, mock_exists, mock_file, mock_walk
    ):
        mock_walk.return_value = [("/root", [], ["jobset.yaml"])]

        # Should not crash, just warn and return empty dict
        mapping = self.manager._map_jobs_to_log_paths()

        self.assertEqual(mapping, {})

    @patch("chs.ramble_manager.get_scheduler")
    @patch("chs.ramble_manager.RambleManager.get_job_ids")
    @patch("chs.ramble_manager.RambleManager._map_jobs_to_log_paths")
    @patch("time.sleep")
    def test_monitor_jobs_slurm(
        self, mock_sleep, mock_map, mock_get_job_ids, mock_get_scheduler
    ):
        # Setup for Slurm
        mock_scheduler = MagicMock(spec=SlurmScheduler)
        mock_scheduler.name = "slurm"
        mock_get_scheduler.return_value = mock_scheduler

        # Override scheduler on manager (though get_scheduler mock handles it)
        self.manager._scheduler = mock_scheduler

        mock_get_job_ids.return_value = ["job-1"]
        # Slurm might not map logs via jobset.yaml, but if it did, logic holds.
        # If no map, fetch_job_log isn't called, but we still wait.
        mock_map.return_value = {}

        # Mock completion
        mock_scheduler.get_job_statuses.side_effect = [
            {"job-1": "RUNNING"},
            {"job-1": "COMPLETED"},
        ]

        # Execute
        self.manager._monitor_jobs(poll_interval=1)

        # Verify
        # Should have called get_job_statuses twice
        self.assertEqual(mock_scheduler.get_job_statuses.call_count, 2)
        # Should NOT have called fetch_job_log (no map)
        mock_scheduler.fetch_job_log.assert_not_called()

    def test_gke_scheduler_get_nodes_under_test(self):
        # Quick test for GKE scheduler node resolution
        scheduler = GKEScheduler(nodes="node1,node2")
        self.assertEqual(scheduler.get_nodes_under_test(), ["node1", "node2"])


if __name__ == "__main__":
    unittest.main()
