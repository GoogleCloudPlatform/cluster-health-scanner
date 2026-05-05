import pathlib
import tempfile
import unittest
from unittest import mock

from chs.utils import app_data
import chs.exceptions as ex


class TestChsAppData(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.app_data = app_data.ChsAppData(data_dir=self.test_dir.name)
        self.workspaces_dir = pathlib.Path(self.test_dir.name) / "workspaces"

    def tearDown(self):
        self.test_dir.cleanup()

    def test_get_latest_run_id_success(self):
        self.workspaces_dir.mkdir(parents=True, exist_ok=True)

        # Different days
        (self.workspaces_dir / "2023-01-01_10.00.00.000000").mkdir()

        # Same day, different hour
        (self.workspaces_dir / "2024-05-12_08.00.00.000000").mkdir()

        # Same day, same hour/minute, different seconds
        (self.workspaces_dir / "2024-05-12_10.00.15.000000").mkdir()
        (self.workspaces_dir / "2024-05-12_10.00.18.000000").mkdir()

        # Same day, same hour/minute/second, different milliseconds
        (self.workspaces_dir / "2024-05-12_10.00.18.100000").mkdir()
        (self.workspaces_dir / "2024-05-12_10.00.18.999999").mkdir()
        (self.workspaces_dir / "2024-05-12_10.00.18.150000").mkdir()

        (self.workspaces_dir / "invalid_folder").mkdir()
        (self.workspaces_dir / "2024-05-12_wrong_format").mkdir()
        (
            self.workspaces_dir
            / "not_a_dir_but_valid_name_2024-05-12_13.00.00.000000"
        ).touch()

        latest = self.app_data.get_latest_run_id()
        self.assertEqual(latest, "2024-05-12_10.00.18.999999")

    def test_get_latest_run_id_no_valid_runs(self):
        self.workspaces_dir.mkdir(parents=True, exist_ok=True)
        (self.workspaces_dir / "invalid_folder").mkdir()
        (self.workspaces_dir / "another_invalid_run").mkdir()

        with self.assertRaisesRegex(
            ex.NoPreviousRunsFoundException, "No previous runs found."
        ):
            self.app_data.get_latest_run_id()

    @mock.patch.object(app_data.ChsAppData, "get_workspaces_dir")
    def test_get_latest_run_id_no_workspace_dir(self, mock_get_workspaces_dir):
        non_existent_dir = pathlib.Path(self.test_dir.name) / "does_not_exist"
        mock_get_workspaces_dir.return_value = str(non_existent_dir)

        with self.assertRaisesRegex(
            ex.WorkspaceNotFoundException, "No workspaces directory found."
        ):
            self.app_data.get_latest_run_id()

    def test_get_images_dir(self):
        images_dir = self.app_data.get_images_dir()
        expected_dir = pathlib.Path(self.test_dir.name) / "images"
        self.assertEqual(images_dir, str(expected_dir))
        self.assertTrue(expected_dir.exists())

    @mock.patch("pathlib.Path.mkdir")
    def test_get_images_dir_failure(self, mock_mkdir):
        mock_mkdir.side_effect = OSError("Permission denied")
        images_dir = self.app_data.get_images_dir()
        self.assertIsNone(images_dir)

    def test_update_run_status_success(self):
        self.app_data.update_run(
            "run1", {"job_entries": {"job1": {"status": "RUNNING"}}}
        )

        self.app_data.update_run_status("run1", {"job1": "COMPLETED"})

        run = self.app_data.get_run("run1")
        self.assertEqual(run["job_entries"], {"job1": {"status": "COMPLETED"}})

    def test_update_run_status_preserves_terminal(self):
        self.app_data.update_run(
            "run1", {"job_entries": {"job1": {"status": "COMPLETED"}}}
        )

        self.app_data.update_run_status("run1", {"job1": "UNKNOWN"})

        run = self.app_data.get_run("run1")
        self.assertEqual(run["job_entries"], {"job1": {"status": "COMPLETED"}})

    def test_get_latest_runs(self):
        self.app_data.update_run(
            "run1", {"submission_time": "2023-01-01T12:00:00Z"}
        )
        self.app_data.update_run(
            "run2", {"submission_time": "2023-01-02T12:00:00Z"}
        )
        self.app_data.update_run(
            "run3", {"submission_time": "2023-01-03T12:00:00Z"}
        )

        latest_runs = self.app_data.get_latest_runs(limit=2)
        self.assertEqual(len(latest_runs), 2)
        keys = list(latest_runs.keys())
        self.assertEqual(keys[0], "run3")
        self.assertEqual(keys[1], "run2")


if __name__ == "__main__":
    unittest.main()
