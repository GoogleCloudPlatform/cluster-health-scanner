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
from unittest.mock import Mock, patch
from chs.ramble_manager import RambleManager
from chs.utils.app_data import ChsAppData


class TestRambleManagerStatus(unittest.TestCase):
    @patch("chs.ramble_manager.get_scheduler")
    def test_monitor_jobs_updates_status(self, mock_get_scheduler):
        mock_scheduler = Mock()
        mock_scheduler.get_job_statuses.return_value = {"job1": "COMPLETED"}

        mock_get_scheduler.return_value = mock_scheduler

        # Patch __init__ to do nothing
        with patch.object(RambleManager, "__init__", return_value=None):
            manager = RambleManager()
            manager._workspace_name = "run1"
            manager._run_id = "run1"

            # Mock other methods used in _monitor_jobs
            manager.get_job_ids = Mock(return_value=["job1"])
            manager._map_jobs_to_log_paths = Mock(return_value={})
            manager.get_scheduler_type = Mock(return_value="slurm")

            mock_app_data = Mock(spec=ChsAppData)
            mock_app_data.get_run.return_value = {
                "job_entries": {"job1": {"status": "RUNNING"}},
                "scheduler": "slurm",
            }

            manager._monitor_jobs(poll_interval=1, app_data=mock_app_data)

            mock_app_data.update_run_status.assert_called_once_with(
                "run1", {"job1": "COMPLETED"}
            )

    @patch("chs.ramble_manager.get_scheduler")
    def test_monitor_jobs_preserves_terminal_status(self, mock_get_scheduler):
        mock_scheduler = Mock()
        mock_scheduler.get_job_statuses.return_value = {"job1": "UNKNOWN"}

        mock_get_scheduler.return_value = mock_scheduler

        with patch.object(RambleManager, "__init__", return_value=None):
            manager = RambleManager()
            manager._workspace_name = "run1"
            manager._run_id = "run1"

            manager.get_job_ids = Mock(return_value=["job1"])
            manager._map_jobs_to_log_paths = Mock(return_value={})
            manager.get_scheduler_type = Mock(return_value="slurm")

            mock_app_data = Mock(spec=ChsAppData)
            mock_app_data.get_run.return_value = {
                "job_entries": {"job1": {"status": "COMPLETED"}},
                "scheduler": "slurm",
            }

            manager._monitor_jobs(poll_interval=1, app_data=mock_app_data)

            mock_app_data.update_run_status.assert_called_once_with(
                "run1", {"job1": "UNKNOWN"}
            )


if __name__ == "__main__":
    unittest.main()
