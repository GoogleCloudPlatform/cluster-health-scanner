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

"""Runs `runtime: tpu` job specs as native processes for the runner."""

import pathlib
import shutil
import tempfile

from chs.exceptions import (
    ChsExecutionException,
    ChsSchedulerException,
    ChsValidationError,
)
from chs.schedulers.standalone import native
from chs.schedulers.standalone.job_spec import JobSpec


class NativeBackend:
    """Runs job specs natively from a TPU runtime (runner.Backend)."""

    def __init__(
        self,
        runtime_dir: pathlib.Path,
        chip_count: int,
    ):
        self._runtime_dir = runtime_dir
        self._chip_count = chip_count
        self._out_dir = pathlib.Path(tempfile.mkdtemp(prefix="chs-native-"))
        self._jobs: dict[str, native.NativeJob] = {}

    def _out_path(self, spec: JobSpec) -> pathlib.Path:
        return self._out_dir / f"{spec.job_name}.out"

    def start(self, spec: JobSpec) -> None:
        """Starts the spec's native job."""
        try:
            self._jobs[spec.job_name] = native.NativeJob.start(
                job_name=spec.job_name,
                script=spec.read_script(),
                timeout_seconds=spec.timeout,
                runtime_dir=self._runtime_dir,
                node=spec.node,
                chip_count=self._chip_count,
                out_path=self._out_path(spec),
            )
        except (ChsExecutionException, ChsValidationError, OSError) as e:
            raise ChsSchedulerException(str(e)) from e

    def exit_code(self, spec: JobSpec) -> int | None:
        """Returns the job's exit code (128 + signal if killed), or None."""
        code = self._jobs[spec.job_name].poll()
        return 128 - code if code is not None and code < 0 else code

    def stop(self, spec: JobSpec) -> None:
        """Kills the job's process group, keeping its output."""
        self._jobs[spec.job_name].kill()

    def fetch_log(self, spec: JobSpec, log_path: str) -> None:
        """Copies the job's output to log_path."""
        shutil.copyfile(self._out_path(spec), log_path)

    def remove(self, spec: JobSpec) -> None:
        """Deletes the job's saved output."""
        self._out_path(spec).unlink(missing_ok=True)
