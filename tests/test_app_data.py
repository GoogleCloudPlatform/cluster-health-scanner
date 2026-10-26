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

import fcntl
import multiprocessing
import os
import pathlib
import tempfile
import unittest
from unittest import mock

from chs.utils import app_data
import chs.exceptions as ex


_UPDATES_PER_WORKER = 50


def _record_runs(data_dir: str, worker: int) -> None:
    store = app_data.ChsAppData(data_dir=data_dir)
    for i in range(_UPDATES_PER_WORKER):
        store.update_run(f"run-{worker}-{i}", {"run_id": f"run-{worker}-{i}"})
        store.update_run_status("shared", {f"job-{worker}-{i}": "PENDING"})


class TestChsAppData(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.app_data = app_data.ChsAppData(data_dir=self.test_dir.name)
        self.workspaces_dir = pathlib.Path(self.test_dir.name) / "workspaces"

    def tearDown(self):
        self.test_dir.cleanup()

    def test_get_latest_run_id_success(self):
        self.app_data.update_run(
            "20230101-100000_run-1",
            {
                "run_id": "run-1",
                "submission_time": "2023-01-01T10:00:00Z",
            },
        )
        self.app_data.update_run(
            "20240512-080000_run-2",
            {
                "run_id": "run-2",
                "submission_time": "2024-05-12T08:00:00Z",
            },
        )
        self.app_data.update_run(
            "20240512-100018_run-4",
            {
                "run_id": "run-4",
                "submission_time": "2024-05-12T10:00:18Z",
            },
        )
        self.app_data.update_run(
            "20240512-100015_run-3",
            {
                "run_id": "run-3",
                "submission_time": "2024-05-12T10:00:15Z",
            },
        )

        latest = self.app_data.get_latest_run_id()
        self.assertEqual(latest, "run-4")

    def test_get_latest_run_id_no_runs(self):
        with self.assertRaisesRegex(
            ex.NoPreviousRunsFoundException,
            "No previous runs found in registry.",
        ):
            self.app_data.get_latest_run_id()
            self.app_data.get_latest_run_id()

    def test_get_images_dir(self):
        images_dir = self.app_data.get_images_dir()
        expected_dir = pathlib.Path(self.test_dir.name) / "images"
        self.assertEqual(images_dir, str(expected_dir))
        self.assertTrue(expected_dir.exists())

    @mock.patch("pathlib.Path.mkdir")
    def test_get_images_dir_failure(self, mock_mkdir):
        mock_mkdir.side_effect = PermissionError("Permission denied")
        with self.assertLogs("chs.utils.app_data", level="WARNING") as cm:
            images_dir = self.app_data.get_images_dir()
            self.assertIsNone(images_dir)
            self.assertIn(
                "Failed to create the required image " "assets directory",
                cm.output[0],
            )

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

    def test_update_run_preserves_terminal_status_on_merge(self):
        self.app_data.update_run(
            "run1", {"job_entries": {"job1": {"status": "DEADLINE"}}}
        )

        self.app_data.update_run(
            "run1", {"job_entries": {"job1": {"status": "PENDING"}}}
        )

        run = self.app_data.get_run("run1")
        self.assertEqual(run["job_entries"], {"job1": {"status": "DEADLINE"}})

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

    def test_resolve_run_id_success(self):
        self.app_data.update_run(
            "20260512-120000_run-1",
            {
                "run_id": "run-1",
                "submission_time": "2026-05-12T12:00:00Z",
            },
        )

        resolved = self.app_data.resolve_run_id("run-1")
        self.assertEqual(resolved, "20260512-120000_run-1")

    def test_resolve_run_id_not_found(self):
        with self.assertRaisesRegex(
            ex.WorkspaceNotFoundException,
            "Run ID 'non-existent' not found in registry.",
        ):
            self.app_data.resolve_run_id("non-existent")

    def test_storage_config(self):
        config = {"type": "path", "path": {"path": "/foo"}}
        self.app_data.set_storage_config(config)
        self.assertEqual(self.app_data.get_storage_config(), config)

    def test_concurrent_processes_never_lose_updates(self):
        """Test updates from concurrent CHS processes all persist."""
        workers = 2
        ctx = multiprocessing.get_context("spawn")
        processes = [
            ctx.Process(target=_record_runs, args=(self.test_dir.name, w))
            for w in range(workers)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=60)
            self.assertEqual(process.exitcode, 0)

        runs = self.app_data.get_all_runs()
        expected_runs = {
            f"run-{w}-{i}"
            for w in range(workers)
            for i in range(_UPDATES_PER_WORKER)
        }
        expected_jobs = {
            f"job-{w}-{i}"
            for w in range(workers)
            for i in range(_UPDATES_PER_WORKER)
        }
        self.assertEqual(set(runs) - {"shared"}, expected_runs)
        self.assertEqual(set(runs["shared"]["job_entries"]), expected_jobs)

    def test_failed_update_releases_the_lock(self):
        """Test a failure mid-update releases the lock for the next writer."""
        with mock.patch.object(
            self.app_data, "_save_json", side_effect=RuntimeError("disk")
        ):
            with self.assertRaisesRegex(RuntimeError, "disk"):
                self.app_data.update_run("run-1", {"run_id": "run-1"})

        fd = os.open(
            pathlib.Path(self.test_dir.name) / "app_data.lock", os.O_RDWR
        )
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)
        self.app_data.update_run("run-2", {"run_id": "run-2"})
        self.assertIn("run-2", self.app_data.get_all_runs())


if __name__ == "__main__":
    unittest.main()
