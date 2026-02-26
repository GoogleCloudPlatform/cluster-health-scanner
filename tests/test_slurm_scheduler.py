import unittest
from unittest.mock import patch
import subprocess
from chs.schedulers.slurm import SlurmScheduler


class TestSlurmScheduler(unittest.TestCase):
    def setUp(self):
        self.scheduler = SlurmScheduler()

    @patch("subprocess.run")
    def test_get_job_statuses_success(self, mock_run):
        # Mock squeue output (JobID,State)
        mock_output = "123,R\n124,CD\n"
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=mock_output
        )

        statuses = self.scheduler.get_job_statuses(["123", "124", "125"])

        self.assertEqual(statuses["123"], "R")
        self.assertEqual(statuses["124"], "CD")
        # 125 not in output -> COMPLETED
        self.assertEqual(statuses["125"], "COMPLETED")

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


if __name__ == "__main__":
    unittest.main()
