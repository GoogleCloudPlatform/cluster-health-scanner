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

"""Runs standalone TPU tests as native processes from a TPU runtime."""

import logging
import os
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path

from chs.exceptions import ChsExecutionException, ChsValidationError

logger = logging.getLogger(__name__)

KILL_AFTER_SECONDS = 30
_GROUP_EXIT_WAIT_SECONDS = 5

_SCRUBBED_ENV_PREFIXES = ("PYTHON", "JAX_", "XLA_", "LIBTPU_", "TPU_")
_SCRUBBED_ENV_VARS = ("VIRTUAL_ENV",)
_CHIPS_PER_HOST_BOUNDS = {4: "2,2,1"}
_LIBTPU_INIT_ARGS = (
    "--deepsea_hal_test_skip_slicebuilder=true"
    " --deepsea_hal_test_allow_multichip_skip_slicebuilder=true"
    " --xla_tpu_use_enhanced_launch_barrier=false"
    " --megascale_error_reporter_abort_on_hang=true"
)


def build_command(
    script: str, timeout_seconds: int, watch_fd: int
) -> list[str]:
    """Returns the argv to run `script` with memlock, timeout and watchdog."""
    if timeout_seconds <= 0:
        raise ChsValidationError(
            f"Native job timeout must be positive, got {timeout_seconds}."
        )
    return [
        "bash",
        "-c",
        f"( read -r _ <&{watch_fd}; kill -TERM 0 ) & exec {watch_fd}<&-; "
        "ulimit -l unlimited || "
        '{ echo "chs: cannot raise RLIMIT_MEMLOCK" >&2; exit 125; }; '
        f"exec timeout -k {KILL_AFTER_SECONDS} "
        f"{timeout_seconds} bash -c {shlex.quote(script)}",
    ]


def build_env(
    base_env: Mapping[str, str],
    runtime_dir: Path,
    node: str,
    chip_count: int,
) -> dict[str, str]:
    """Returns the environment for a native TPU job."""
    if chip_count not in _CHIPS_PER_HOST_BOUNDS:
        raise ChsValidationError(
            f"Unsupported TPU chip count per host: {chip_count}."
        )
    env = {
        key: value
        for key, value in base_env.items()
        if key not in _SCRUBBED_ENV_VARS
        and not key.startswith(_SCRUBBED_ENV_PREFIXES)
    }
    chs_venv_bin = (
        os.path.normpath(os.path.join(sys.prefix, "bin"))
        if sys.prefix != sys.base_prefix
        else None
    )
    path = [
        entry
        for entry in env.get("PATH", "").split(os.pathsep)
        if entry and os.path.normpath(entry) != chs_venv_bin
    ]
    env.update(
        {
            "PATH": os.pathsep.join([str(runtime_dir / "venv" / "bin"), *path]),
            "CHS_AMB_DIR": str(runtime_dir / "code"),
            "MY_NODE_NAME": node,
            "TPU_SKIP_MDS_QUERY": "true",
            "TPU_WORKER_ID": "0",
            "TPU_WORKER_HOSTNAMES": "localhost",
            "ENABLE_TPUNETD_CLIENT": "false",
            "TPU_HOST_BOUNDS": "1,1,1",
            "TPU_CHIPS_PER_HOST_BOUNDS": _CHIPS_PER_HOST_BOUNDS[chip_count],
            "LIBTPU_INIT_ARGS": _LIBTPU_INIT_ARGS,
        }
    )
    return env


def _kill_group(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _wait_for_group_exit(pgid: int) -> None:
    deadline = time.monotonic() + _GROUP_EXIT_WAIT_SECONDS
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.01)


class NativeJob:
    """A running native TPU test."""

    def __init__(
        self,
        process: subprocess.Popen[bytes],
        job_name: str,
    ) -> None:
        self._process = process
        self.job_name = job_name
        self.pid = process.pid
        self.returncode: int | None = None
        self._watch_fd: int | None = None

    @classmethod
    def start(
        cls,
        *,
        job_name: str,
        script: str,
        timeout_seconds: int,
        runtime_dir: Path,
        node: str,
        chip_count: int,
        out_path: Path,
    ) -> "NativeJob":
        """Starts a test in its own session that stops when CHS exits."""
        env = build_env(os.environ, runtime_dir, node, chip_count)
        read_fd, write_fd = os.pipe()
        try:
            command = build_command(script, timeout_seconds, read_fd)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=True,
                    pass_fds=(read_fd,),
                )
        except OSError as e:
            os.close(write_fd)
            raise ChsExecutionException(
                f"Cannot start native job {job_name}: {e}"
            ) from e
        except BaseException:
            os.close(write_fd)
            raise
        finally:
            os.close(read_fd)
        logger.debug("Started native job %s (pid %d).", job_name, process.pid)
        job = cls(process, job_name)
        job._watch_fd = write_fd
        return job

    def poll(self) -> int | None:
        """Returns the exit code (negative for signal), or None if running."""
        if self.returncode is not None:
            return self.returncode
        info = os.waitid(
            os.P_PID,
            self._process.pid,
            os.WEXITED | os.WNOHANG | os.WNOWAIT,
        )
        return None if info is None else self._finish()

    def kill(self) -> int:
        """Kills the job's process group and finishes the job."""
        if self.returncode is not None:
            return self.returncode
        _kill_group(self._process.pid)
        os.waitid(os.P_PID, self._process.pid, os.WEXITED | os.WNOWAIT)
        return self._finish()

    def _finish(self) -> int:
        _kill_group(self._process.pid)
        self.returncode = self._process.wait()
        _wait_for_group_exit(self._process.pid)
        if self._watch_fd is not None:
            os.close(self._watch_fd)
            self._watch_fd = None
        log = logger.debug if self.returncode == 0 else logger.warning
        log("Native job %s exited with %d.", self.job_name, self.returncode)
        return self.returncode
