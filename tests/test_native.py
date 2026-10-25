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

"""Unit tests for the native job runner, using real processes."""

import functools
import os
import resource
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest import mock

from chs.exceptions import ChsExecutionException, ChsValidationError
from chs.schedulers.standalone import native

_DEADLINE_SECONDS = 10


def _wait_for(condition, what: str) -> None:
    deadline = time.monotonic() + _DEADLINE_SECONDS
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError(f"Timed out waiting for {what}.")
        time.sleep(0.02)


def _proc_state(pid: int) -> str | None:
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    return text.rpartition(")")[2].split()[0]


def _has_exited(pid: int) -> bool:
    return _proc_state(pid) in (None, "Z")


class BuildCommandTest(unittest.TestCase):
    def test_build_command(self):
        self.assertEqual(
            native.build_command("echo 'a b'", 600, 7),
            [
                "bash",
                "-c",
                "( read -r _ <&7; kill -TERM 0 ) & exec 7<&-; "
                "ulimit -l unlimited || "
                '{ echo "chs: cannot raise RLIMIT_MEMLOCK" >&2; exit 125; }; '
                "exec timeout -k 30 600 bash -c "
                "'echo '\"'\"'a b'\"'\"''",
            ],
        )
        with self.assertRaises(ChsValidationError):
            native.build_command("true", 0, 7)


class BuildEnvTest(unittest.TestCase):
    def setUp(self):
        self.runtime = Path("/rt/tpu-abc")
        self.base = {
            "HOME": "/home/u",
            "PATH": "/opt/venv/bin:/usr/bin:/opt/venv/bin/:/bin",
            "VIRTUAL_ENV": "/opt/venv",
            "PYTHONPATH": "/x",
            "PYTHONHOME": "/y",
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": "--foo",
            "LIBTPU_INIT_ARGS": "--bar",
            "TPU_NAME": "t",
            "TPU_CHIPS_PER_HOST_BOUNDS": "9,9,9",
            "MY_TPU": "kept",
        }

    def _build(self, chip_count=4):
        with (
            mock.patch.object(sys, "prefix", "/opt/venv"),
            mock.patch.object(sys, "base_prefix", "/usr"),
        ):
            return native.build_env(self.base, self.runtime, "vm-1", chip_count)

    def test_scrubs_and_sets_single_host_tpu_env(self):
        self.assertEqual(
            self._build(),
            {
                "HOME": "/home/u",
                "MY_TPU": "kept",
                "PATH": "/rt/tpu-abc/venv/bin:/usr/bin:/bin",
                "CHS_AMB_DIR": "/rt/tpu-abc/code",
                "MY_NODE_NAME": "vm-1",
                "TPU_SKIP_MDS_QUERY": "true",
                "TPU_WORKER_ID": "0",
                "TPU_WORKER_HOSTNAMES": "localhost",
                "ENABLE_TPUNETD_CLIENT": "false",
                "TPU_HOST_BOUNDS": "1,1,1",
                "TPU_CHIPS_PER_HOST_BOUNDS": "2,2,1",
                "LIBTPU_INIT_ARGS": native._LIBTPU_INIT_ARGS,
            },
        )

    def test_chips_per_host_bounds_and_invalid_chip_count(self):
        self.assertEqual(
            self._build(chip_count=4)["TPU_CHIPS_PER_HOST_BOUNDS"],
            "2,2,1",
        )
        for chips in (1, 2, 8):
            with self.assertRaises(ChsValidationError):
                self._build(chip_count=chips)

    def test_keeps_path_when_chs_is_not_in_a_venv(self):
        with (
            mock.patch.object(sys, "prefix", "/usr"),
            mock.patch.object(sys, "base_prefix", "/usr"),
        ):
            env = native.build_env(
                {"PATH": "/usr/bin:/bin"}, self.runtime, "vm-1", 4
            )
        self.assertEqual(env["PATH"], "/rt/tpu-abc/venv/bin:/usr/bin:/bin")


class NativeJobTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.out = self.dir / "exp" / "tpu-info.out"

    def _start(self, script, timeout_seconds=60):
        job = native.NativeJob.start(
            job_name="tpu-info",
            script=script,
            timeout_seconds=timeout_seconds,
            runtime_dir=self.dir / "runtime",
            node="vm-1",
            chip_count=4,
            out_path=self.out,
        )
        self.addCleanup(job.kill)
        return job

    def _wait(self, job):
        _wait_for(lambda: job.poll() is not None, "the job to exit")
        return job.returncode

    def _assert_reaped(self, pid):
        with self.assertRaises(ChildProcessError):
            os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG)

    def test_writes_output_and_reaps_leader(self):
        job = self._start("echo out; echo err >&2; exit 3")
        self.assertEqual(self._wait(job), 3)
        self.assertEqual(self.out.read_text("utf-8"), "out\nerr\n")
        self._assert_reaped(job.pid)

    def test_runs_in_own_session_with_tpu_env(self):
        job = self._start(
            'ps -o sid= -p $$; echo "$CHS_AMB_DIR $TPU_WORKER_ID $PATH"'
        )
        self.assertEqual(self._wait(job), 0)
        sid, env_line = self.out.read_text("utf-8").splitlines()
        self.assertEqual(int(sid), job.pid)
        amb_dir, worker_id, path = env_line.split(" ")
        self.assertEqual(amb_dir, str(self.dir / "runtime/code"))
        self.assertEqual(worker_id, "0")
        self.assertTrue(path.startswith(str(self.dir / "runtime/venv/bin:")))

    def test_memlock_is_unlimited(self):
        if (
            resource.getrlimit(resource.RLIMIT_MEMLOCK)[1]
            != resource.RLIM_INFINITY
        ):
            self.skipTest("memlock hard limit is not unlimited")
        job = self._start("ulimit -l")
        self.assertEqual(self._wait(job), 0)
        self.assertEqual(self.out.read_text("utf-8"), "unlimited\n")

    def test_memlock_failure_exits_125(self):
        if os.geteuid() == 0:
            self.skipTest("root can raise the memlock hard limit")
        hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)[1]
        limit = 64 * 1024
        if hard != resource.RLIM_INFINITY:
            limit = min(limit, hard)
        popen = functools.partial(
            subprocess.Popen,
            preexec_fn=lambda: resource.setrlimit(
                resource.RLIMIT_MEMLOCK, (limit, limit)
            ),
        )
        with mock.patch.object(native.subprocess, "Popen", popen):
            job = self._start("echo ran")
        self.assertEqual(self._wait(job), 125)
        self.assertEqual(
            self.out.read_text("utf-8").splitlines()[-1],
            "chs: cannot raise RLIMIT_MEMLOCK",
        )

    def test_timeout_exit_code(self):
        job = self._start("sleep 30", timeout_seconds=1)
        self.assertEqual(self._wait(job), 124)

    def test_group_is_killed_after_leader_exits_and_before_reaping(self):
        pid_file = self.dir / "child.pid"
        job = self._start(f"sleep 300 & echo $! > {pid_file}; exit 0")
        _wait_for(lambda: _proc_state(job.pid) == "Z", "the leader to exit")
        child = int(pid_file.read_text("utf-8"))
        self.assertFalse(_has_exited(child))
        self.assertEqual(job.poll(), 0)
        _wait_for(lambda: _has_exited(child), "the child to die")
        self._assert_reaped(job.pid)

    def test_finish_stops_waiting_when_group_outlives_deadline(self):
        job = self._start("true")
        _wait_for(lambda: _proc_state(job.pid) == "Z", "the leader to exit")
        with (
            mock.patch.object(native, "_GROUP_EXIT_WAIT_SECONDS", 0.02),
            mock.patch.object(native.os, "killpg", return_value=None),
        ):
            self.assertEqual(job.poll(), 0)

    def test_kill_stops_group_and_reaps(self):
        pid_file = self.dir / "child.pid"
        job = self._start(f"sleep 300 & echo $! > {pid_file}; sleep 300")
        _wait_for(
            lambda: pid_file.exists() and bool(pid_file.read_text("utf-8")),
            "the child pid",
        )
        child = int(pid_file.read_text("utf-8"))
        self.assertIsNone(job.poll())
        self.assertEqual(job.kill(), -signal.SIGKILL)
        self.assertEqual(job.poll(), -signal.SIGKILL)
        self.assertEqual(job.kill(), -signal.SIGKILL)
        self.assertTrue(_has_exited(child))
        self._assert_reaped(job.pid)
        self.assertTrue(self.out.exists())

    def test_job_stops_when_chs_is_killed(self):
        pid_file = self.dir / "child.pid"
        starter = subprocess.Popen(
            [
                sys.executable,
                "-c",
                textwrap.dedent(
                    f"""
                    import pathlib, time
                    from chs.schedulers.standalone import native
                    native.NativeJob.start(
                        job_name="tpu-info",
                        script="sleep 300 & echo $! > {pid_file}; wait",
                        timeout_seconds=600,
                        runtime_dir=pathlib.Path({str(self.dir / "runtime")!r}),
                        node="vm-1",
                        chip_count=4,
                        out_path=pathlib.Path({str(self.out)!r}),
                    )
                    time.sleep(300)
                    """
                ),
            ],
        )
        self.addCleanup(starter.kill)
        _wait_for(
            lambda: pid_file.exists() and bool(pid_file.read_text("utf-8")),
            "the test's child pid",
        )
        child = int(pid_file.read_text("utf-8"))
        self.assertFalse(_has_exited(child))
        starter.kill()
        starter.wait()
        _wait_for(lambda: _has_exited(child), "the test to stop with CHS")

    def test_start_failure_raises_execution_exception(self):
        with mock.patch.object(
            native.subprocess, "Popen", side_effect=FileNotFoundError("bash")
        ):
            with self.assertRaisesRegex(ChsExecutionException, "Cannot start"):
                self._start("true")


if __name__ == "__main__":
    unittest.main()
