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

import unittest
from unittest.mock import MagicMock, patch

# Standard mocks for problematic imports
MOCK_MODULES = {
    "chs.boh_generator": MagicMock(),
    "chs.protos": MagicMock(),
    "chs.protos.bill_of_health_pb2": MagicMock(),
}

with patch.dict("sys.modules", MOCK_MODULES):
    from chs.ramble_manager import RambleManager  # noqa: E402
    from chs.schedulers.kubernetes import KubernetesScheduler  # noqa: E402
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
        # Default to Kubernetes for existing tests, can override in
        # specific tests
        self.manager._scheduler = KubernetesScheduler()

    @patch("os.listdir")
    @patch("os.path.exists")
    def test_map_jobs_to_log_paths(
        self,
        mock_exists,
        mock_listdir,
    ):
        self.manager._app_data = MagicMock()
        self.manager._app_data.get_run.return_value = {
            "job_entries": {
                "job-123": {
                    "status": "PENDING",
                    "experiment_dir": "/tmp/workspaces/exp1",
                }
            }
        }
        mock_exists.return_value = True
        mock_listdir.return_value = ["job-123.out", "execute.log"]

        mapping = self.manager._map_jobs_to_log_paths()

        self.assertEqual(
            mapping,
            {"job-123": ("/tmp/workspaces/exp1/job-123.out")},
        )

    @patch("chs.ramble_manager.get_scheduler")
    @patch("chs.ramble_manager.RambleManager.get_job_ids")
    @patch("chs.ramble_manager.RambleManager._map_jobs_to_log_paths")
    @patch("time.sleep")
    def test_monitor_jobs_fetches_logs_kubernetes(
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
        self.manager._monitor_jobs(MagicMock(), poll_interval=1)

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
        self.manager._monitor_jobs(MagicMock(), poll_interval=1)

        # Verify
        mock_scheduler.fetch_job_log.assert_called_once_with(
            "job-1", "/path/log"
        )

    @patch("os.path.exists", return_value=False)
    def test_map_jobs_to_log_paths_handles_missing_dir(self, mock_exists):
        self.manager._app_data = MagicMock()
        self.manager._app_data.get_run.return_value = {
            "job_entries": {
                "job-123": {
                    "status": "PENDING",
                    "experiment_dir": "/tmp/workspaces/exp1",
                }
            }
        }
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
        self.manager._monitor_jobs(MagicMock(), poll_interval=1)

        # Verify
        # Should have called get_job_statuses twice
        self.assertEqual(mock_scheduler.get_job_statuses.call_count, 2)
        # Should NOT have called fetch_job_log (no map)
        mock_scheduler.fetch_job_log.assert_not_called()

    def test_kubernetes_scheduler_get_nodes_under_test(self):
        # Quick test for Kubernetes scheduler node resolution
        scheduler = KubernetesScheduler(nodes="node1,node2")
        self.assertEqual(scheduler.get_nodes_under_test(), ["node1", "node2"])


if __name__ == "__main__":
    unittest.main()
