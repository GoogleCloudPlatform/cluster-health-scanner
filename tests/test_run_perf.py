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

"""Tests for the latency optimizations of `chs run`.

Covers the deferred import of the GCE and Kubernetes client libraries, the
Bill of Health upload that overlaps the workspace archive, the job status
poll interval and the resource cleanup that runs in the background after the
analysis.
"""

import contextlib
import os
import pathlib
import subprocess
import sys
import sysconfig
import tempfile
import threading
import unittest
from unittest import mock

import chs
from chs import cli
from chs import exceptions as ex
from chs.cli.background_cleanup import BackgroundCleanup
from chs.protos import bill_of_health_pb2 as pb
from chs.ramble_manager import RambleManager
from chs.utils import k8s_client
from chs.utils.app_data import ChsAppData

# Imported for its guard: no test here may create a real Kubernetes client.
from tests import k8s_fakes  # noqa: F401

# Upper bound for waiting on an event that the code under test should set.
# It is only reached when the behavior under test is broken.
_EVENT_TIMEOUT_S = 5
_CLEANUP_SPINNER = "Cleaning up CHS resources..."


def _analyze_in_guard(app_data, run_id, scheduler):
    """Runs `_analyze_action` under the teardown guard, as `chs run` does."""
    with cli.cancel_jobs_on_failure(scheduler, run_id):
        return cli._analyze_action(app_data, run_id)


_CLEANUP_THREAD = "chs-cleanup"


def _start_patch(test: unittest.TestCase, patcher):
    """Starts a patcher and stops it when the test ends."""
    mocked = patcher.start()
    test.addCleanup(patcher.stop)
    return mocked


def _cleanup_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == _CLEANUP_THREAD]


def _join_cleanup_threads() -> None:
    for thread in _cleanup_threads():
        thread.join(timeout=_EVENT_TIMEOUT_S)


class LazyClientLibraryImportTest(unittest.TestCase):
    """Tests that importing CHS does not load slow client libraries."""

    def _loaded_by_import(self, package):
        """Returns whether importing CHS loads `package` or a submodule.

        Every Ramble subprocess imports chs.utils.topology and the CLI imports
        chs.cli. This runs in a fresh interpreter, since the test process may
        already have loaded the library.
        """
        src_dir = os.path.dirname(
            os.path.dirname(os.path.abspath(chs.__file__))
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            path for path in (src_dir, env.get("PYTHONPATH")) if path
        )
        code = (
            "import sys\n"
            "import chs.utils.topology\n"
            "import chs.cli\n"
            f"package = {package!r}\n"
            "print(any(name == package or name.startswith(package + '.')\n"
            "          for name in sys.modules))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(result.stdout.split()[-1], ("True", "False"))
        return result.stdout.split()[-1] == "True"

    def test_import_does_not_load_compute_v1(self):
        """Test the ~2.9 s google.cloud.compute_v1 import is deferred."""
        self.assertFalse(self._loaded_by_import("google.cloud.compute_v1"))

    def test_import_does_not_load_the_kubernetes_sdk(self):
        """Test the ~1.3 s kubernetes import waits for the first API call."""
        self.assertFalse(self._loaded_by_import("kubernetes"))


class _RambleManagerTestCase(unittest.TestCase):
    """Provides a RambleManager whose workspace is a temporary directory."""

    def setUp(self):
        super().setUp()
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        # RambleManager requires Ramble's setup script under site-packages.
        setup_dir = pathlib.Path(tmp_dir.name, "share", "ramble")
        setup_dir.mkdir(parents=True)
        (setup_dir / "setup-env.sh").write_text("#!/bin/bash\n")
        _start_patch(
            self,
            mock.patch.object(sysconfig, "get_path", return_value=tmp_dir.name),
        )
        _start_patch(
            self, mock.patch("chs.ramble_manager.ChsAppData", autospec=True)
        )
        self.mock_run = mock.Mock(
            return_value=subprocess.CompletedProcess(
                args=[], returncode=0, stdout="", stderr=""
            )
        )
        self.manager = RambleManager(
            os.path.join(tmp_dir.name, "workspaces"),
            "ws1",
            run_command=self.mock_run,
            run_id="run1",
        )
        os.makedirs(os.path.join(self.manager.path, ".ramble-workspace"))


class ArchiveOverlapTest(_RambleManagerTestCase):
    """Tests that archive() overlaps the BoH upload with the tarball."""

    def setUp(self):
        super().setUp()
        self.provider = mock.Mock()
        self.provider.get_uri.side_effect = lambda path: f"gs://bucket/{path}"
        _start_patch(
            self,
            mock.patch(
                "chs.storage.StorageProviderFactory.get_provider_for_run",
                return_value=self.provider,
            ),
        )
        self.archive_file = pathlib.Path(
            self.manager.path, "archive", "ws1.tar.gz"
        )

    def _patch_step(self, name: str, **kwargs) -> mock.Mock:
        return _start_patch(
            self, mock.patch.object(self.manager, name, **kwargs)
        )

    def test_boh_upload_overlaps_tarball_creation(self):
        """Test the BoH is uploaded while the tarball is being created."""
        boh_upload_started = threading.Event()
        tarball_started = threading.Event()
        overlapped = []

        def upload_boh(provider):
            boh_upload_started.set()
            overlapped.append(tarball_started.wait(_EVENT_TIMEOUT_S))
            return "bills_of_health/ws1.json", "bills_of_health/ws1.html", True

        def create_archive():
            tarball_started.set()
            overlapped.append(boh_upload_started.wait(_EVENT_TIMEOUT_S))
            return self.archive_file

        self._patch_step("_upload_boh", side_effect=upload_boh)
        self._patch_step(
            "_create_workspace_archive", side_effect=create_archive
        )
        upload_archive = self._patch_step(
            "_upload_workspace_archive", return_value="workspaces/ws1.tar.gz"
        )

        result = self.manager.archive()

        self.assertEqual(overlapped, [True, True])
        upload_archive.assert_called_once_with(self.provider, self.archive_file)
        self.assertEqual(
            result,
            (
                "gs://bucket/workspaces/ws1.tar.gz",
                "gs://bucket/bills_of_health/ws1.json",
                "gs://bucket/bills_of_health/ws1.html",
                True,
            ),
        )

    def test_tarball_upload_waits_for_boh_upload(self):
        """Test the two uploads never use the provider at the same time."""
        tarball_created = threading.Event()
        boh_upload_done = threading.Event()
        boh_done_at_tarball_upload = []

        def upload_boh(provider):
            tarball_created.wait(_EVENT_TIMEOUT_S)
            boh_upload_done.set()
            return None, None, True

        def create_archive():
            tarball_created.set()
            return self.archive_file

        def upload_archive(provider, archive_file):
            boh_done_at_tarball_upload.append(boh_upload_done.is_set())
            return "workspaces/ws1.tar.gz"

        self._patch_step("_upload_boh", side_effect=upload_boh)
        self._patch_step(
            "_create_workspace_archive", side_effect=create_archive
        )
        self._patch_step(
            "_upload_workspace_archive", side_effect=upload_archive
        )

        self.manager.archive()

        self.assertEqual(boh_done_at_tarball_upload, [True])

    def test_boh_upload_failure_still_uploads_tarball(self):
        """Test a failed BoH upload is reported and the tarball uploaded."""
        self._patch_step("_upload_boh", return_value=(None, None, False))
        self._patch_step(
            "_create_workspace_archive", return_value=self.archive_file
        )
        upload_archive = self._patch_step(
            "_upload_workspace_archive", return_value="workspaces/ws1.tar.gz"
        )

        result = self.manager.archive()

        upload_archive.assert_called_once_with(self.provider, self.archive_file)
        self.assertEqual(
            result, ("gs://bucket/workspaces/ws1.tar.gz", None, None, False)
        )

    def test_tarball_failure_skips_its_upload(self):
        """Test a failed tarball creation is reported and not uploaded."""
        self._patch_step(
            "_upload_boh",
            return_value=(
                "bills_of_health/ws1.json",
                "bills_of_health/ws1.html",
                True,
            ),
        )
        self._patch_step("_create_workspace_archive", return_value=None)
        upload_archive = self._patch_step("_upload_workspace_archive")

        result = self.manager.archive()

        upload_archive.assert_not_called()
        self.assertEqual(
            result,
            (
                None,
                "gs://bucket/bills_of_health/ws1.json",
                "gs://bucket/bills_of_health/ws1.html",
                False,
            ),
        )

    def test_tarball_upload_failure_falls_back_to_archive_file_path(self):
        """Failed tarball upload returns existing local archive_file path."""
        self.archive_file.parent.mkdir(parents=True, exist_ok=True)
        self.archive_file.write_bytes(b"tarball")
        self._patch_step(
            "_upload_boh",
            return_value=(
                "bills_of_health/ws1.json",
                "bills_of_health/ws1.html",
                True,
            ),
        )
        self._patch_step(
            "_create_workspace_archive", return_value=self.archive_file
        )
        self._patch_step("_upload_workspace_archive", return_value=None)

        result = self.manager.archive()

        self.assertEqual(
            result,
            (
                str(self.archive_file),
                "gs://bucket/bills_of_health/ws1.json",
                "gs://bucket/bills_of_health/ws1.html",
                False,
            ),
        )

    def test_unexpected_boh_upload_error_propagates(self):
        """Test an error escaping the BoH upload propagates from archive()."""
        self._patch_step("_upload_boh", side_effect=RuntimeError("boom"))
        self._patch_step(
            "_create_workspace_archive", return_value=self.archive_file
        )
        upload_archive = self._patch_step("_upload_workspace_archive")

        with self.assertRaisesRegex(RuntimeError, "boom"):
            self.manager.archive()

        upload_archive.assert_not_called()

    def test_selects_newest_regular_tarball_and_removes_symlink(self):
        """Test _create_workspace_archive skips symlinks and older tarballs."""
        archive_dir = pathlib.Path(self.manager.path, "archive")
        archive_dir.mkdir()
        older = archive_dir / "ws1-older.tar.gz"
        newer = archive_dir / "ws1-newer.tar.gz"
        older.write_bytes(b"old")
        newer.write_bytes(b"new")
        os.utime(older, (1000, 1000))
        os.utime(newer, (2000, 2000))
        latest_symlink = archive_dir / "archive.latest.tar.gz"
        latest_symlink.symlink_to(newer.name)

        selected = self.manager._create_workspace_archive()

        self.assertEqual(selected, newer)
        self.assertFalse(selected.is_symlink())

        dest = self.manager._upload_workspace_archive(self.provider, selected)

        self.assertEqual(dest, "workspaces/ws1.tar.gz")
        self.assertFalse(newer.exists())
        self.assertFalse(latest_symlink.exists())
        self.assertFalse(latest_symlink.is_symlink())
        self.assertTrue(older.exists())

    def _stage_ramble_archive(
        self, archive_dir, name="ws1-archive-2026", suffix=".tar.gz"
    ):
        """Lays out what `ramble workspace archive -t` leaves behind."""
        archive_dir.mkdir(exist_ok=True)
        staged_dir = archive_dir / name
        (staged_dir / "experiments").mkdir(parents=True)
        (staged_dir / "experiments" / "out.log").write_text("log")
        tarball = archive_dir / f"{name}{suffix}"
        tarball.write_bytes(b"tar")
        (archive_dir / "archive.latest").symlink_to(staged_dir.name)
        (archive_dir / f"archive.latest{suffix}").symlink_to(tarball.name)
        return staged_dir, tarball

    def test_upload_removes_staged_directory_and_symlinks(self):
        """Test the copied workspace directory goes with its tarball."""
        archive_dir = pathlib.Path(self.manager.path, "archive")
        _, tarball = self._stage_ramble_archive(archive_dir)
        other_dir = archive_dir / "ws1-archive-2025"
        other_dir.mkdir()

        dest = self.manager._upload_workspace_archive(self.provider, tarball)

        self.assertEqual(dest, "workspaces/ws1.tar.gz")
        self.assertEqual(sorted(os.listdir(archive_dir)), [other_dir.name])

    def test_upload_removes_staged_directory_of_tgz_archive(self):
        """Test a .tgz tarball's staged directory is removed too."""
        archive_dir = pathlib.Path(self.manager.path, "archive")
        _, tarball = self._stage_ramble_archive(archive_dir, suffix=".tgz")

        self.manager._upload_workspace_archive(self.provider, tarball)

        self.assertEqual(os.listdir(archive_dir), [])

    def test_upload_keeps_directories_outside_the_archive_dir(self):
        """Test a symlinked staged directory and its target are kept."""
        archive_dir = pathlib.Path(self.manager.path, "archive")
        archive_dir.mkdir()
        outside_dir = pathlib.Path(self.manager.path, "outside")
        outside_dir.mkdir()
        (outside_dir / "keep.txt").write_text("keep")
        (archive_dir / "ws1-archive-2026").symlink_to(outside_dir)
        tarball = archive_dir / "ws1-archive-2026.tar.gz"
        tarball.write_bytes(b"tar")

        self.manager._upload_workspace_archive(self.provider, tarball)

        self.assertFalse(tarball.exists())
        self.assertTrue((outside_dir / "keep.txt").exists())
        self.assertTrue((archive_dir / "ws1-archive-2026").is_symlink())

    def test_failed_upload_keeps_local_archive(self):
        """Test nothing is removed if the upload fails."""
        archive_dir = pathlib.Path(self.manager.path, "archive")
        staged_dir, tarball = self._stage_ramble_archive(archive_dir)
        self.provider.upload_file.side_effect = OSError("denied")

        self.assertIsNone(
            self.manager._upload_workspace_archive(self.provider, tarball)
        )

        self.assertTrue(tarball.exists())
        self.assertTrue((staged_dir / "experiments" / "out.log").exists())


class JobPollIntervalTest(_RambleManagerTestCase):
    """Tests the default interval between job status polls."""

    def test_wait_for_jobs_polls_every_5_seconds_by_default(self):
        """Test wait_for_jobs() polls every 5 s unless told otherwise."""
        with mock.patch.object(self.manager, "_monitor_jobs") as monitor:
            self.manager.wait_for_jobs(app_data=mock.sentinel.app_data)
        monitor.assert_called_once_with(mock.sentinel.app_data, 5, None)


class CancelExperimentsTest(_RambleManagerTestCase):
    """Tests how cancel_experiments() picks the scheduler."""

    def setUp(self):
        super().setUp()
        self.get_scheduler = _start_patch(
            self, mock.patch("chs.ramble_manager.get_scheduler", autospec=True)
        )
        self.app_data = self.manager._app_data

    def test_given_scheduler_name_skips_the_registry(self):
        """Test a given scheduler name is used without reading runs.json."""
        self.manager.cancel_experiments(scheduler_name="kubernetes")

        self.app_data.get_run.assert_not_called()
        self.get_scheduler.assert_called_once_with("kubernetes")

    def test_tears_down_only_its_run(self):
        """Test only the run is cancelled, by its run ID."""
        scheduler = self.get_scheduler.return_value
        manager = mock.Mock()
        for name in ("cancel_run", "cancel_all_runs"):
            manager.attach_mock(getattr(scheduler, name), name)

        self.manager.cancel_experiments(scheduler_name="kubernetes")

        self.assertEqual(
            manager.mock_calls,
            [
                mock.call.cancel_run("run1"),
            ],
        )

    def test_default_reads_the_scheduler_from_the_registry(self):
        """Test the scheduler comes from runs.json if no name is given."""
        self.app_data.get_run.return_value = {"scheduler": "kubernetes"}

        self.manager.cancel_experiments()

        self.app_data.get_run.assert_called_once_with("ws1")
        self.get_scheduler.assert_called_once_with("kubernetes")


class BackgroundCleanupTest(unittest.TestCase):
    """Tests the BackgroundCleanup context manager on its own."""

    def setUp(self):
        super().setUp()
        self.spinner = _start_patch(
            self, mock.patch.object(cli.theme, "get_progress_spinner")
        )
        self.addCleanup(_join_cleanup_threads)

    def test_exit_without_start_does_nothing(self):
        """Test nothing is waited for if no cleanup was started."""
        with BackgroundCleanup("run1"):
            pass
        self.spinner.assert_not_called()
        self.assertEqual(_cleanup_threads(), [])

    def test_start_twice_is_noop(self):
        """Test a second start() is a no-op once cleanup is running."""
        ramble_manager = mock.Mock()
        ramble_manager.get_scheduler_type.return_value = "slurm"

        with BackgroundCleanup("run1") as cleanup:
            cleanup.start(ramble_manager)
            first_thread = cleanup._thread
            cleanup.start(ramble_manager)
            self.assertIs(cleanup._thread, first_thread)

        ramble_manager.cancel_experiments.assert_called_once_with(
            scheduler_name="slurm"
        )
        self.assertEqual(_cleanup_threads(), [])

    def test_cleanup_thread_is_daemon(self):
        """Test the chs-cleanup worker thread is created with daemon=True."""
        ramble_manager = mock.Mock()
        ramble_manager.get_scheduler_type.return_value = "slurm"
        release = threading.Event()
        ramble_manager.cancel_experiments.side_effect = (
            lambda scheduler_name=None: release.wait(_EVENT_TIMEOUT_S)
        )

        with BackgroundCleanup("run1") as cleanup:
            cleanup.start(ramble_manager)
            self.assertIsNotNone(cleanup._thread)
            self.assertTrue(cleanup._thread.daemon)
            release.set()

    def test_start_failure_resets_thread(self):
        """Test _thread is reset to None if thread.start() raises."""
        ramble_manager = mock.Mock()
        ramble_manager.get_scheduler_type.return_value = "slurm"

        with BackgroundCleanup("run1") as cleanup:
            with mock.patch.object(
                threading.Thread,
                "start",
                side_effect=RuntimeError("cannot start thread"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "cannot start thread"
                ):
                    cleanup.start(ramble_manager)
            self.assertIsNone(cleanup._thread)

    def test_keyboard_interrupt_during_wait_logs_hint_and_skips_outcome(self):
        """Test Ctrl+C during _wait() logs a hint and skips _handle_outcome."""
        ramble_manager = mock.Mock()
        ramble_manager.get_scheduler_type.return_value = "slurm"
        started = threading.Event()
        release = threading.Event()

        def cancel_experiments(scheduler_name=None):
            started.set()
            release.wait(_EVENT_TIMEOUT_S)
            raise RuntimeError("cleanup error that should not be handled")

        ramble_manager.cancel_experiments.side_effect = cancel_experiments

        @contextlib.contextmanager
        def interrupting_spinner(description, show_time=True):
            release.set()
            raise KeyboardInterrupt
            yield

        self.spinner.side_effect = interrupting_spinner

        cleanup = BackgroundCleanup("run1")
        with (
            self.assertLogs("chs.cli", level="WARNING") as logs,
            mock.patch.object(
                cleanup, "_handle_outcome", wraps=cleanup._handle_outcome
            ) as handle_outcome,
            self.assertRaises(KeyboardInterrupt),
        ):
            with cleanup:
                cleanup.start(ramble_manager)
                self.assertTrue(started.wait(_EVENT_TIMEOUT_S))

        handle_outcome.assert_not_called()
        self.assertIn(
            "Interrupted while cleaning up CHS resources for run run1; "
            "cleanup may be incomplete. Run 'chs cancel run1' to finish "
            "cleaning up.",
            "\n".join(logs.output),
        )


class AnalyzeBackgroundCleanupTest(unittest.TestCase):
    """Tests that _analyze_action cleans up in the background."""

    def setUp(self):
        super().setUp()
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        ws_dir = os.path.join(tmp_dir.name, "run1")
        os.makedirs(os.path.join(ws_dir, ".ramble-workspace"))

        self.app_data = mock.MagicMock()
        self.app_data.get_workspaces_dir.return_value = tmp_dir.name
        self.app_data.resolve_run_id.return_value = "run1"
        # A non-tiered run whose jobs have not been analyzed yet.
        self.run_entry = {"is_tiered": False, "scheduler": "kubernetes"}
        self.app_data.get_run.return_value = self.run_entry
        self.scheduler = mock.Mock()

        ramble_manager_cls = _start_patch(
            self, mock.patch("chs.cli.RambleManager", autospec=True)
        )
        self.rm = ramble_manager_cls.return_value
        self.rm.path = ws_dir
        self.rm.workspace_name = "run1"
        self.rm.is_workspace_created = True
        self.rm.get_scheduler_type.return_value = "kubernetes"
        self.rm.analyze_experiments.return_value = os.path.join(
            ws_dir, "results.latest.json"
        )
        self.rm.generate_boh.return_value = ws_dir

        # Steps taken on the calling thread, in order.
        self.steps = []
        self.show_results = _start_patch(
            self, mock.patch("chs.cli._show_results", autospec=True)
        )
        self.show_results.side_effect = lambda *args, **kwargs: (
            self.steps.append("results")
        )
        self.archive_and_log = _start_patch(
            self, mock.patch("chs.cli._archive_and_log", autospec=True)
        )
        self.archive_and_log.side_effect = lambda *args, **kwargs: (
            self.steps.append("archive")
        )
        _start_patch(self, mock.patch("chs.cli.Console"))
        self.on_cleanup_spinner = None
        _start_patch(
            self,
            mock.patch.object(
                cli.theme, "get_progress_spinner", new=self._fake_spinner
            ),
        )
        self.addCleanup(_join_cleanup_threads)

    @contextlib.contextmanager
    def _fake_spinner(self, description, show_time=True):
        self.steps.append(f"spinner: {description}")
        if description == _CLEANUP_SPINNER and self.on_cleanup_spinner:
            self.on_cleanup_spinner()
        yield mock.Mock(), "task"

    def _assert_cleanup_overlaps_results_and_archive(self):
        cleanup_started = threading.Event()
        release_cleanup = threading.Event()
        cleanup_done = threading.Event()
        observed = {}

        def cancel_experiments(scheduler_name=None):
            cleanup_started.set()
            observed["released"] = release_cleanup.wait(_EVENT_TIMEOUT_S)
            cleanup_done.set()

        def show_results(*args, **kwargs):
            observed["running_during_results"] = (
                cleanup_started.wait(_EVENT_TIMEOUT_S)
                and not cleanup_done.is_set()
            )

        def archive_and_log(*args, **kwargs):
            observed["running_during_archive"] = not cleanup_done.is_set()
            release_cleanup.set()

        self.rm.cancel_experiments.side_effect = cancel_experiments
        self.show_results.side_effect = show_results
        self.archive_and_log.side_effect = archive_and_log

        self.assertEqual(
            _analyze_in_guard(self.app_data, "run1", self.scheduler), "run1"
        )

        self.assertEqual(
            observed,
            {
                "released": True,
                "running_during_results": True,
                "running_during_archive": True,
            },
        )
        self.assertTrue(cleanup_done.is_set())
        self.assertEqual(_cleanup_threads(), [])
        self.rm.cancel_experiments.assert_called_once_with(
            scheduler_name="kubernetes"
        )

    def test_cleanup_overlaps_results_and_archive(self):
        """Test the cleanup runs while results are shown and archived."""
        self._assert_cleanup_overlaps_results_and_archive()
        self.assertTrue(self.run_entry["is_boh_final"])
        self.app_data.update_run.assert_called_with(
            "run1", {"is_boh_final": True}
        )

    def test_waits_for_running_cleanup_after_archive(self):
        """Test a still running cleanup is awaited, with a spinner, last."""
        release_cleanup = threading.Event()
        self.rm.cancel_experiments.side_effect = (
            lambda scheduler_name=None: release_cleanup.wait(_EVENT_TIMEOUT_S)
        )
        # The cleanup can only finish once its spinner is shown.
        self.on_cleanup_spinner = release_cleanup.set

        self.assertEqual(
            _analyze_in_guard(self.app_data, "run1", self.scheduler), "run1"
        )

        self.assertEqual(
            self.steps,
            [
                "spinner: Analyzing logs...",
                "results",
                "archive",
                f"spinner: {_CLEANUP_SPINNER}",
            ],
        )
        self.assertEqual(_cleanup_threads(), [])

    def test_no_spinner_if_cleanup_already_finished(self):
        """Test no spinner is shown if the cleanup finished meanwhile."""

        def archive_and_log(*args, **kwargs):
            self.steps.append("archive")
            _join_cleanup_threads()

        self.archive_and_log.side_effect = archive_and_log

        _analyze_in_guard(self.app_data, "run1", self.scheduler)

        self.assertEqual(
            self.steps, ["spinner: Analyzing logs...", "results", "archive"]
        )

    def test_expected_cleanup_errors_log_warning(self):
        """Expected cleanup errors log a warning."""
        errors = [
            ex.ChsSchedulerException("scheduler down"),
            k8s_client.KubeError(
                "Error from server (Forbidden): forbidden", status=403
            ),
            subprocess.CalledProcessError(1, ["scancel", "--name=chs-run1"]),
            FileNotFoundError(2, "No such file or directory", "scancel"),
        ]
        for error in errors:
            with self.subTest(error=type(error).__name__):
                self.run_entry.pop("is_boh_final", None)
                self.archive_and_log.reset_mock()
                self.rm.cancel_experiments.side_effect = error

                with self.assertLogs("chs.cli", level="WARNING") as logs:
                    self.assertEqual(
                        _analyze_in_guard(
                            self.app_data, "run1", self.scheduler
                        ),
                        "run1",
                    )

                self.assertIn(
                    "Failed to clean CHS resources. Run 'chs cancel run1' to "
                    "finish cleaning up. Error:",
                    "\n".join(logs.output),
                )
                self.archive_and_log.assert_called_once_with(self.rm)

    def test_unexpected_error_is_raised_after_archive(self):
        """Test an unexpected cleanup error is raised once archived."""
        self.rm.cancel_experiments.side_effect = RuntimeError("kaboom")

        with self.assertRaisesRegex(RuntimeError, "kaboom"):
            _analyze_in_guard(self.app_data, "run1", self.scheduler)

        self.archive_and_log.assert_called_once_with(self.rm)

    def test_cleanup_error_does_not_mask_a_failure(self):
        """Test a failure after the cleanup started is what propagates."""
        self.show_results.side_effect = ValueError("results broke")
        self.rm.cancel_experiments.side_effect = RuntimeError("cleanup broke")

        with self.assertLogs("chs.cli", level="WARNING") as logs:
            with self.assertRaisesRegex(ValueError, "results broke"):
                _analyze_in_guard(self.app_data, "run1", self.scheduler)

        self.assertIn(
            "Failed to clean CHS resources. Run 'chs cancel run1' to finish "
            "cleaning up. Error: cleanup broke",
            "\n".join(logs.output),
        )
        self.archive_and_log.assert_called_once_with(self.rm)
        self.assertEqual(_cleanup_threads(), [])

    def test_scheduler_is_resolved_on_the_calling_thread(self):
        """Test the worker never reads runs.json to find the scheduler."""
        resolved_on = []

        def get_scheduler_type():
            resolved_on.append(threading.current_thread().name)
            return "slurm"

        self.rm.get_scheduler_type.side_effect = get_scheduler_type

        _analyze_in_guard(self.app_data, "run1", self.scheduler)

        self.assertEqual(resolved_on, [threading.current_thread().name])
        self.rm.cancel_experiments.assert_called_once_with(
            scheduler_name="slurm"
        )

    def test_tiered_run_starts_cleanup_only_after_final_tier(self):
        """Test a tiered run cleans up only after tier 2 finishes analysis."""
        self.run_entry.update(
            {
                "is_tiered": True,
                "suite": "standard",
                "active_tier": 1,
                "active_tier_tests": ["test-tier-1"],
            }
        )
        events = []

        def wait_for_jobs(app_data, tier=None):
            events.append(f"wait tier {tier}")

        def analyze_experiments():
            events.append(f"analyze tier {self.run_entry['active_tier']}")
            return os.path.join(self.rm.path, "results.latest.json")

        def execute_experiments(tier=1, active_tests=None):
            events.append(f"submit tier {tier}")
            return [f"job-tier-{tier}"]

        def cancel_experiments(scheduler_name=None):
            events.append("cleanup")

        self.rm.wait_for_jobs.side_effect = wait_for_jobs
        self.rm.analyze_experiments.side_effect = analyze_experiments
        self.rm.execute_experiments.side_effect = execute_experiments
        self.rm.cancel_experiments.side_effect = cancel_experiments

        tier_manager = mock.MagicMock()
        tier_manager.get_next_tier_tests.side_effect = [
            ["test-tier-2"],
            [],
        ]

        with (
            mock.patch(
                "chs.cli.load_suite",
                return_value={
                    "tests": [{"name": "test-tier-1"}, {"name": "test-tier-2"}]
                },
            ),
            mock.patch("chs.cli.TierManager", return_value=tier_manager),
            mock.patch(
                "chs.cli.parse_tier_results_from_boh",
                return_value=({}, False),
            ),
        ):
            self.assertEqual(
                _analyze_in_guard(self.app_data, "run1", self.scheduler),
                "run1",
            )

        self.assertEqual(
            events,
            [
                "wait tier 1",
                "analyze tier 1",
                "submit tier 2",
                "wait tier 2",
                "analyze tier 2",
                "cleanup",
            ],
        )
        self.assertEqual(self.rm.wait_for_jobs.call_count, 2)
        self.rm.cancel_experiments.assert_called_once_with(
            scheduler_name="kubernetes"
        )

    def test_results_json_shortcut_starts_background_cleanup(self):
        """Path 2 (results.latest.json exists) cleans up in the background."""
        results_json = os.path.join(self.rm.path, "results.latest.json")
        pathlib.Path(results_json).write_text("{}")

        self._assert_cleanup_overlaps_results_and_archive()
        self.rm.generate_boh.assert_called_once_with(results_json)
        self.rm.wait_for_jobs.assert_not_called()


class ShowResultsCachedUriTest(unittest.TestCase):
    """Tests that _show_results uses cached URIs from runs.json first."""

    def setUp(self):
        super().setUp()
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.ws_dir = pathlib.Path(tmp_dir.name) / "ws1"
        self.ws_dir.mkdir()
        self.boh_path = self.ws_dir / "bill_of_health.json"
        self.boh_path.write_text("{}")

        self.app_data = mock.Mock()
        self.app_data.get_workspaces_dir.return_value = tmp_dir.name
        self.app_data.resolve_run_id.return_value = "ws1"

        self.provider = mock.Mock()
        _start_patch(
            self,
            mock.patch(
                "chs.cli.StorageProviderFactory.get_provider_for_run",
                return_value=self.provider,
            ),
        )
        self.viewer_cls = _start_patch(
            self, mock.patch("chs.cli.ResultsViewer", autospec=True)
        )
        # autospec omits attributes set in __init__; telemetry reads boh.
        self.viewer_cls.return_value.boh = pb.BillOfHealth()

    def test_cached_boh_html_uri_skips_provider_exists(self):
        """Cached boh_html_uri in runs.json skips storage exists() calls."""
        self.app_data.get_run.return_value = {
            "boh_html_uri": "gs://bucket/bills_of_health/ws1.html",
            "boh_uri": "gs://bucket/bills_of_health/ws1.json",
        }

        cli._show_results(self.app_data, "run1")

        self.provider.exists.assert_not_called()
        self.viewer_cls.assert_called_once_with(
            self.boh_path, boh_uri="gs://bucket/bills_of_health/ws1.html"
        )

    def test_cached_boh_uri_used_when_html_uri_absent(self):
        """Cached boh_uri is used when boh_html_uri is absent."""
        self.app_data.get_run.return_value = {
            "boh_uri": "gs://bucket/bills_of_health/ws1.json",
        }

        cli._show_results(self.app_data, "run1")

        self.provider.exists.assert_not_called()
        self.viewer_cls.assert_called_once_with(
            self.boh_path, boh_uri="gs://bucket/bills_of_health/ws1.json"
        )

    def test_falls_back_to_provider_exists_when_no_cached_uri(self):
        """Without cached URIs in runs.json, storage exists() is queried."""
        self.app_data.get_run.return_value = {}
        self.provider.exists.side_effect = [True]
        self.provider.get_uri.return_value = (
            "gs://bucket/bills_of_health/ws1.html"
        )

        cli._show_results(self.app_data, "run1")

        self.provider.exists.assert_called_once_with("bills_of_health/ws1.html")
        self.viewer_cls.assert_called_once_with(
            self.boh_path, boh_uri="gs://bucket/bills_of_health/ws1.html"
        )


class AppDataDirtyCheckAndTerminalProtectionTest(unittest.TestCase):
    """Tests update_run_status dirty check and update_run terminal guard."""

    def setUp(self):
        super().setUp()
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.app_data = ChsAppData(data_dir=tmp_dir.name)

    def test_update_run_status_skips_save_when_unchanged(self):
        """No disk write occurs when polled statuses match stored statuses."""
        self.app_data.update_run(
            "run1",
            {
                "job_entries": {
                    "job1": {"status": "RUNNING"},
                    "job2": {"status": "COMPLETED"},
                }
            },
        )
        with mock.patch.object(
            self.app_data, "_save_json", wraps=self.app_data._save_json
        ) as mock_save:
            self.app_data.update_run_status(
                "run1", {"job1": "RUNNING", "job2": "RUNNING"}
            )

        mock_save.assert_not_called()
        self.assertEqual(
            self.app_data.get_run("run1")["job_entries"],
            {
                "job1": {"status": "RUNNING"},
                "job2": {"status": "COMPLETED"},
            },
        )

    def test_update_run_status_saves_when_status_changes(self):
        """A status transition writes runs.json once."""
        self.app_data.update_run(
            "run1", {"job_entries": {"job1": {"status": "RUNNING"}}}
        )
        with mock.patch.object(
            self.app_data, "_save_json", wraps=self.app_data._save_json
        ) as mock_save:
            self.app_data.update_run_status("run1", {"job1": "COMPLETED"})

        mock_save.assert_called_once()
        self.assertEqual(
            self.app_data.get_run("run1")["job_entries"]["job1"]["status"],
            "COMPLETED",
        )

    def test_update_run_protects_terminal_status_from_non_terminal(self):
        """Terminal job statuses are never overwritten by non-terminal ones."""
        self.app_data.update_run(
            "run1",
            {
                "job_entries": {
                    "job1": {"status": "COMPLETED"},
                    "job2": {"status": "DEADLINE"},
                }
            },
        )
        for non_terminal in ("PENDING", "RUNNING"):
            with self.subTest(non_terminal=non_terminal):
                self.app_data.update_run(
                    "run1",
                    {
                        "job_entries": {
                            "job1": {"status": non_terminal},
                            "job2": {"status": non_terminal},
                        }
                    },
                )
                entries = self.app_data.get_run("run1")["job_entries"]
                self.assertEqual(entries["job1"]["status"], "COMPLETED")
                self.assertEqual(entries["job2"]["status"], "DEADLINE")


if __name__ == "__main__":
    unittest.main()
