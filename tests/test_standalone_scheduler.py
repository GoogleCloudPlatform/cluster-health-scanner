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

"""Unit tests for the standalone scheduler and its detection."""

import fcntl
import logging
import os
import pathlib
import shutil
import tempfile
import unittest
from unittest import mock

import yaml

from chs import schedulers
from chs.exceptions import (
    ChsSchedulerException,
    ChsValidationError,
    HostCheckException,
    TpuRuntimeException,
)
from chs.schedulers import standalone
from chs.schedulers.standalone import job_spec
from chs.schedulers.standalone.host import GceInstance
from chs.utils.lifecycle import JobStatus

_INSTANCE = GceInstance(
    name="tpu-vm-1",
    instance_id="123",
    machine_type="ct6e-standard-4t",
    zone="us-east5-b",
    project_id="my-project",
)


class StandaloneSchedulerTestCase(unittest.TestCase):
    """Fakes the VM: metadata, lock, host checks, and runtime."""

    def setUp(self):
        logging.disable(logging.NOTSET)
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        os.makedirs(self.tmp + "/run/lock")
        os.makedirs(self.tmp + "/dev/vfio")
        for n in range(4):
            open(f"{self.tmp}/dev/vfio/{n}", "w").close()
        self.host = {}
        for target, name, value in (
            (standalone.host, "gce_instance", _INSTANCE),
            (standalone.host, "check_host_access", None),
            (standalone.host, "check_tpus_idle", None),
            (standalone.host, "check_tpu_access", None),
            (standalone.tpu_runtime, "find_runtime", pathlib.Path("/rt")),
            (standalone.tpu_runtime, "ensure_runtime", pathlib.Path("/rt")),
        ):
            self.host[name] = self._patch(
                target, name, mock.Mock(return_value=value)
            )
        self._patch(
            standalone.host,
            "host_path",
            mock.Mock(side_effect=lambda p: self.tmp + p),
        )
        self.which = self._patch(
            standalone.shutil, "which", mock.Mock(return_value="/usr/bin/x")
        )

    def _patch(self, target, name, value):
        patcher = mock.patch.object(target, name, value)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def _scheduler(self):
        scheduler = standalone.StandaloneScheduler()
        self.addCleanup(self._release, scheduler)
        return scheduler

    @staticmethod
    def _release(scheduler):
        if scheduler._lock_fd is not None:
            os.close(scheduler._lock_fd)
            scheduler._lock_fd = None

    def _nodes(self):
        """Runs the node check of a new scheduler, then releases its lock."""
        scheduler = self._scheduler()
        nodes = scheduler.get_nodes_under_test()
        self._release(scheduler)
        return nodes

    def _skip_reason(self):
        scheduler = self._scheduler()
        with self.assertLogs(standalone.logger, "WARNING") as logs:
            self.assertEqual(scheduler.get_nodes_under_test(), [])
        self.assertIsNone(scheduler._lock_fd)
        return logs.output[0]


class NodesTest(StandaloneSchedulerTestCase):
    def test_local_vm_resolution_and_hardware(self):
        scheduler = self._scheduler()
        self.assertEqual(scheduler.name, "standalone")
        self.assertIsNone(scheduler.nodes_arg)
        self.assertEqual(scheduler.resolve_raw_nodes(), ["tpu-vm-1"])
        self.assertEqual(scheduler.get_cluster_name(), "tpu-vm-1")
        self.assertEqual(
            scheduler.detect_nodes_hardware(["tpu-vm-1"]), ["ct6e-standard-4t"]
        )
        self.assertEqual(scheduler.node_to_instance_id, {"tpu-vm-1": "123"})
        self.assertEqual(scheduler.node_to_zone, {"tpu-vm-1": "us-east5-b"})
        self.assertEqual(scheduler.project_id, "my-project")
        self.assertIs(schedulers.get_scheduler("standalone"), scheduler)

    def test_rejects_remote_and_cluster_targets(self):
        for kwargs in (
            {"nodes": "tpu-vm-1"},
            {"nodelists": "n[1-2]"},
            {"nodepools": "pool"},
            {"partitions": "p"},
        ):
            with self.assertRaises(ChsValidationError):
                schedulers.get_scheduler("standalone", **kwargs)


class NodeCheckTest(StandaloneSchedulerTestCase):
    def test_idle_vm_passes_and_caches_result(self):
        scheduler = self._scheduler()
        self.assertEqual(scheduler.get_nodes_under_test(), ["tpu-vm-1"])
        self.assertEqual(scheduler.get_nodes_under_test(), ["tpu-vm-1"])
        self.assertIsNotNone(scheduler._lock_fd)
        self.host["check_tpus_idle"].assert_called_once_with()
        self.host["check_tpu_access"].assert_called_once_with()

    def test_failed_check_skips_the_node_with_its_reason(self):
        self.host["check_tpus_idle"].side_effect = HostCheckException(
            "TPUs are busy: used by process 1 (python)."
        )
        self.assertIn("TPUs are busy", self._skip_reason())
        self.host["check_tpus_idle"].side_effect = None
        self.host["check_tpu_access"].side_effect = HostCheckException(
            "The memlock limit is 65536 bytes"
        )
        self.assertIn("memlock", self._skip_reason())

    def test_host_access_is_checked_first(self):
        self.host["check_host_access"].side_effect = HostCheckException(
            "no host"
        )
        self.assertIn("no host", self._skip_reason())
        self.host["check_tpus_idle"].assert_not_called()

    def test_vm_without_tpus_is_skipped(self):
        shutil.rmtree(self.tmp + "/dev/vfio")
        self.assertIn("no TPU devices", self._skip_reason())
        self.host["check_tpus_idle"].assert_not_called()

    def test_runtime_missing_and_unbuildable_skips_the_node(self):
        self.host["find_runtime"].return_value = None
        self.assertEqual(self._nodes(), ["tpu-vm-1"])
        self.which.return_value = None
        self.assertIn("can't be built: git not found", self._skip_reason())
        with mock.patch.object(
            standalone.importlib.util, "find_spec", return_value=None
        ):
            self.assertIn(
                "can't be built: uv and git not found", self._skip_reason()
            )
        self.host["find_runtime"].return_value = pathlib.Path("/rt")
        self.assertEqual(self._nodes(), ["tpu-vm-1"])

    def test_uv_is_found_in_the_chs_environment_not_on_path(self):
        self.host["find_runtime"].return_value = None
        self.which.side_effect = lambda tool: None if tool == "uv" else "/x"
        self.assertEqual(self._nodes(), ["tpu-vm-1"])

    def test_new_lock_file_is_writable_by_everyone(self):
        old_umask = os.umask(0o077)
        self.addCleanup(os.umask, old_umask)
        self.assertEqual(self._nodes(), ["tpu-vm-1"])
        mode = os.stat(self.tmp + standalone.LOCK_PATH).st_mode & 0o777
        self.assertEqual(mode, 0o666)

    def test_read_only_lock_file_still_locks(self):
        lock = self.tmp + standalone.LOCK_PATH
        open(lock, "w").close()
        real_open = os.open

        def read_only_open(path, flags, *args):
            if path == lock and flags & os.O_RDWR:
                raise PermissionError(13, "Permission denied", path)
            return real_open(path, flags, *args)

        scheduler = self._scheduler()
        with mock.patch.object(standalone.os, "open", read_only_open):
            self.assertEqual(scheduler.get_nodes_under_test(), ["tpu-vm-1"])
        self.assertFalse(scheduler._lock_writable)

    def test_node_check_clears_the_previous_run_id(self):
        lock = self.tmp + standalone.LOCK_PATH
        with open(lock, "w") as f:
            f.write("old-run\n")
        self.assertEqual(self._nodes(), ["tpu-vm-1"])
        with open(lock) as f:
            self.assertEqual(f.read(), "")

    def test_lock_file_of_another_user_is_opened_without_o_creat(self):
        """fs.protected_regular rejects O_CREAT on another user's file."""
        lock = self.tmp + standalone.LOCK_PATH
        open(lock, "w").close()
        real_open = os.open

        def protected_open(path, flags, *args):
            if path == lock and flags & os.O_CREAT:
                raise PermissionError(13, "Permission denied", path)
            return real_open(path, flags, *args)

        with mock.patch.object(standalone.os, "open", protected_open):
            self.assertEqual(self._nodes(), ["tpu-vm-1"])

    def test_lock_file_is_created_when_missing(self):
        lock = self.tmp + standalone.LOCK_PATH
        self.assertFalse(os.path.exists(lock))
        self.assertEqual(self._nodes(), ["tpu-vm-1"])
        self.assertTrue(os.path.exists(lock))

    def test_held_and_symlinked_locks_skip_the_node(self):
        first = self._scheduler()
        self.assertEqual(first.get_nodes_under_test(), ["tpu-vm-1"])
        self.assertIn(standalone.LOCK_PATH, self._skip_reason())

        self._release(first)
        os.remove(self.tmp + standalone.LOCK_PATH)
        os.symlink(self.tmp + "/elsewhere", self.tmp + standalone.LOCK_PATH)
        self.assertIn("Could not open the host lock", self._skip_reason())

    def test_find_runs_targeting_nodes(self):
        scheduler = self._scheduler()
        self.assertEqual(scheduler.find_runs_targeting_nodes(["tpu-vm-1"]), {})


class RunTest(StandaloneSchedulerTestCase):
    def setUp(self):
        super().setUp()
        self.configs = os.path.join(self.tmp, "configs")
        os.makedirs(self.configs)
        self.backend = mock.Mock()
        self.backend.exit_code.return_value = None
        self.backend_cls = self._patch(
            standalone, "NativeBackend", mock.Mock(return_value=self.backend)
        )

    def _experiment(self, name, order, runtime="tpu"):
        exp_dir = os.path.join(self.tmp, name)
        os.makedirs(exp_dir)
        with open(os.path.join(exp_dir, job_spec.SPEC_FILE), "w") as f:
            yaml.safe_dump(
                {
                    "job_name": name,
                    "node": "tpu-vm-1",
                    "image": "",
                    "script": "/s",
                    "timeout": "60",
                    "order": str(order),
                    "privileged": "false",
                    "run_id": "run-1",
                    "runtime": runtime,
                },
                f,
            )
        return exp_dir

    def _started_scheduler(self):
        scheduler = self._scheduler()
        scheduler.write_ramble_config(
            self.configs,
            tests=[{"name": "tpu-info"}],
            machine_type="ct6e-standard-4t",
            deadline=600,
            run_id="run-1",
        )
        return scheduler

    def _variables(self):
        with open(os.path.join(self.configs, "ramble.yaml")) as f:
            config = yaml.safe_load(f)["ramble"]
        self.assertEqual(config["variants"]["workflow_manager"], "standalone")
        return config["variables"]

    def test_write_ramble_config_records_the_run_id(self):
        self._started_scheduler()
        with open(self.tmp + standalone.LOCK_PATH) as f:
            self.assertEqual(f.read(), "run-1\n")

    def test_write_ramble_config_readies_the_runtime(self):
        os.remove(f"{self.tmp}/dev/vfio/3")
        scheduler = self._started_scheduler()
        variables = self._variables()
        for key in ("mpi_command", "n_ranks", "n_nodes", "processes_per_node"):
            self.assertIn(key, variables)
        self.assertEqual(
            {
                k: variables[k]
                for k in ("nodes", "node", "chs_run_id", "deadline")
            },
            {
                "nodes": ["tpu-vm-1"],
                "node": "tpu-vm-1",
                "chs_run_id": "run-1",
                "deadline": 600,
            },
        )
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "topology.json")))
        self.backend_cls.assert_called_once_with(
            pathlib.Path("/rt"),
            chip_count=4,
        )
        self.assertIsNotNone(scheduler._runner)

    def test_skipped_vm_builds_no_runtime(self):
        self.host["check_tpus_idle"].side_effect = HostCheckException("busy")
        with self.assertLogs(standalone.logger, "WARNING"):
            scheduler = self._started_scheduler()
        self.host["ensure_runtime"].assert_not_called()
        self.assertIsNone(scheduler._runner)
        variables = self._variables()
        self.assertEqual(variables["nodes"], [])
        self.assertNotIn("node", variables)

    def test_runtime_build_failure_fails_the_run(self):
        self.host["ensure_runtime"].side_effect = TpuRuntimeException(
            "no network"
        )
        with self.assertRaisesRegex(TpuRuntimeException, "no network"):
            self._started_scheduler()

    def test_submit_and_poll_run_jobs_in_order(self):
        scheduler = self._started_scheduler()
        dirs = [self._experiment("b", 2), self._experiment("a", 1)]
        self.assertEqual(
            scheduler.submit_experiments(dirs),
            {"b": dirs[0], "a": dirs[1]},
        )
        self.assertEqual(scheduler.extract_job_id(dirs[0]), "b")
        self.assertEqual(
            scheduler.get_job_statuses(["a", "b"]),
            dict.fromkeys(["a", "b"], JobStatus.PENDING.value),
        )
        self.backend.start.assert_not_called()

        self.assertEqual(
            scheduler.poll_jobs(["a", "b"]),
            {"a": JobStatus.RUNNING.value, "b": JobStatus.PENDING.value},
        )
        self.backend.exit_code.side_effect = lambda spec: (
            0 if spec.job_name == "a" else None
        )
        self.assertEqual(
            scheduler.poll_jobs(["a", "b"]),
            {"a": JobStatus.COMPLETED.value, "b": JobStatus.RUNNING.value},
        )
        scheduler.fetch_job_log("a", os.path.join(self.tmp, "a.out"))
        self.backend.fetch_log.assert_called_once()
        self.backend.remove.assert_called_once()

        scheduler.cancel_jobs(["b"])
        self.assertEqual(
            scheduler.get_job_statuses(["b"]), {"b": JobStatus.CANCELLED.value}
        )

    def test_container_spec_is_refused(self):
        scheduler = self._started_scheduler()
        with self.assertRaisesRegex(ChsValidationError, "can't run"):
            scheduler.submit_experiments([self._experiment("a", 1, "")])

    def test_cancel_run_and_all_runs(self):
        scheduler = self._started_scheduler()
        scheduler.submit_experiments([self._experiment("a", 1)])
        scheduler.poll_jobs(["a"])
        self.assertFalse(scheduler.cancel_run("run-other"))
        self.assertTrue(scheduler.cancel_run("run-1"))
        self.backend.stop.assert_called_once()
        self.assertEqual(
            scheduler.get_job_statuses(["a"]), {"a": JobStatus.CANCELLED.value}
        )

        scheduler.submit_experiments([self._experiment("b", 2)])
        scheduler.cancel_all_runs()
        self.assertEqual(
            scheduler.get_job_statuses(["b"]), {"b": JobStatus.CANCELLED.value}
        )

    def test_without_a_passed_node_nothing_runs(self):
        scheduler = self._scheduler()
        with self.assertRaises(ChsSchedulerException):
            scheduler.get_job_statuses(["a"])
        with self.assertRaises(ChsSchedulerException):
            scheduler.submit_experiments([])
        scheduler.cancel_jobs(["a"])
        self.assertFalse(scheduler.cancel_run("run-1"))
        scheduler.cancel_all_runs()


class CancelFromAnotherProcessTest(StandaloneSchedulerTestCase):
    """`chs cancel` in a new process stops the CHS run holding the lock."""

    def setUp(self):
        super().setUp()
        lock = self.tmp + standalone.LOCK_PATH
        self.holder_fd = os.open(lock, os.O_RDONLY | os.O_CREAT, 0o644)
        self.addCleanup(os.close, self.holder_fd)
        fcntl.flock(self.holder_fd, fcntl.LOCK_EX)
        st = os.stat(lock)
        self.lock_id = (
            f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}"
        )
        self.kill = self._patch(standalone.os, "kill", mock.Mock())

    def _record(self, run_id):
        with open(self.tmp + standalone.LOCK_PATH, "w") as f:
            f.write(f"{run_id}\n" if run_id else "")

    def _write_proc_locks(self, *lines):
        os.makedirs(self.tmp + "/proc", exist_ok=True)
        with open(self.tmp + "/proc/locks", "w") as f:
            f.write("".join(line + "\n" for line in lines))

    def test_sends_sigterm_to_the_lock_holder(self):
        self._write_proc_locks(
            "1: POSIX  ADVISORY  WRITE 99 00:01:1 0 EOF",
            f"2: FLOCK  ADVISORY  WRITE 4242 {self.lock_id} 0 EOF",
        )
        self._record("any-run")
        scheduler = self._scheduler()
        self.assertTrue(scheduler.cancel_run("any-run"))
        self.kill.assert_called_once_with(4242, standalone.signal.SIGTERM)
        self.kill.reset_mock()
        scheduler.cancel_all_runs()
        self.kill.assert_called_once_with(4242, standalone.signal.SIGTERM)
        self.assertIsNone(scheduler._lock_fd)

    def test_no_holder_found_or_holder_gone(self):
        self._record("run-1")
        self._write_proc_locks("1: FLOCK  ADVISORY  WRITE 4242 00:01:1 0 EOF")
        self.assertFalse(self._scheduler().cancel_run("run-1"))
        self.kill.assert_not_called()

        self._write_proc_locks(
            f"1: FLOCK  ADVISORY  WRITE 4242 {self.lock_id} 0 EOF"
        )
        self.kill.side_effect = ProcessLookupError
        self.assertFalse(self._scheduler().cancel_run("run-1"))

    def test_cancel_run_stops_only_the_recorded_run(self):
        self._write_proc_locks(
            f"1: FLOCK  ADVISORY  WRITE 4242 {self.lock_id} 0 EOF"
        )
        self._record("run-a")
        with self.assertLogs(standalone.logger, "WARNING") as logs:
            self.assertFalse(self._scheduler().cancel_run("run-b"))
        self.assertIn("the active run is 'run-a'", logs.output[0])
        self.kill.assert_not_called()

        self._record("")
        with self.assertLogs(standalone.logger, "WARNING") as logs:
            self.assertFalse(self._scheduler().cancel_run("run-b"))
        self.assertIn("hasn't recorded its ID yet", logs.output[0])
        self.kill.assert_not_called()

        self._record("run-a")
        self.assertTrue(self._scheduler().cancel_run("run-a"))
        self.kill.assert_called_once_with(4242, standalone.signal.SIGTERM)

    def test_cancel_all_runs_ignores_the_recorded_run(self):
        self._write_proc_locks(
            f"1: FLOCK  ADVISORY  WRITE 4242 {self.lock_id} 0 EOF"
        )
        self._record("")
        self._scheduler().cancel_all_runs()
        self.kill.assert_called_once_with(4242, standalone.signal.SIGTERM)

    def test_holder_of_another_user_fails_clearly(self):
        self._write_proc_locks(
            f"1: FLOCK  ADVISORY  WRITE 4242 {self.lock_id} 0 EOF"
        )
        self.kill.side_effect = PermissionError("not permitted")
        with self.assertRaisesRegex(
            standalone.ChsCancellationException, "process 4242.*sudo"
        ):
            self._scheduler().cancel_all_runs()

    def test_run_holding_the_lock_does_not_signal_itself(self):
        os.close(self.holder_fd)
        self.holder_fd = os.open(os.devnull, os.O_RDONLY)
        scheduler = self._scheduler()
        self.assertEqual(scheduler.get_nodes_under_test(), ["tpu-vm-1"])
        self.assertFalse(scheduler.cancel_run("run-1"))
        self.kill.assert_not_called()


if __name__ == "__main__":
    unittest.main()
