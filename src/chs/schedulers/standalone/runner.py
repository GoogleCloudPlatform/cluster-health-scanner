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

"""Runs a standalone run's job specs one at a time."""

import logging
import time
from typing import Mapping, Protocol

from chs.exceptions import ChsSchedulerException, ChsValidationError
from chs.schedulers.standalone.job_spec import JobSpec
from chs.utils.lifecycle import JobStatus

logger = logging.getLogger(__name__)

_TIMEOUT_EXIT_CODE = 124
_KILLED_EXIT_CODE = 137


class Backend(Protocol):
    """Runs one job spec at a time for the runner."""

    def start(self, spec: JobSpec) -> None: ...
    def exit_code(self, spec: JobSpec) -> int | None: ...
    def stop(self, spec: JobSpec) -> None: ...
    def fetch_log(self, spec: JobSpec, log_path: str) -> None: ...
    def remove(self, spec: JobSpec) -> None: ...


def status_for_exit(code: int, elapsed: float, timeout: int) -> str:
    """Maps a finished job's exit code to its JobStatus value."""
    if code == _TIMEOUT_EXIT_CODE or (
        code == _KILLED_EXIT_CODE and elapsed >= timeout
    ):
        return JobStatus.TIMEOUT.value
    return JobStatus.COMPLETED.value


class SequentialRunner:
    """Runs job specs one at a time in `order` on their runtime's backend."""

    def __init__(self, backends: Mapping[str, Backend]):
        self._backends = dict(backends)
        self._specs: dict[str, JobSpec] = {}
        self._queue: list[JobSpec] = []
        self._statuses: dict[str, str] = {}
        self._errors: dict[str, str] = {}
        self._current: JobSpec | None = None
        self._started_at = 0.0

    def submit(self, specs: list[JobSpec]) -> list[str]:
        """Queues specs and returns their job IDs (job names)."""
        seen = set(self._specs)
        for spec in specs:
            if spec.runtime not in self._backends:
                raise ChsValidationError(
                    f"Job '{spec.job_name}' needs runtime '{spec.runtime}',"
                    " which this host can't run."
                )
            if spec.job_name in seen:
                raise ChsValidationError(
                    f"Duplicate job name '{spec.job_name}'."
                )
            seen.add(spec.job_name)
        for spec in specs:
            self._specs[spec.job_name] = spec
            self._statuses[spec.job_name] = JobStatus.PENDING.value
        self._queue = sorted(
            self._queue + list(specs), key=lambda spec: spec.order
        )
        return [spec.job_name for spec in specs]

    def poll(self) -> None:
        """Finishes the running job if it exited and starts the next one."""
        if self._current is not None and not self._finish_current():
            return
        while self._queue and self._current is None:
            spec = self._queue.pop(0)
            try:
                self._backends[spec.runtime].start(spec)
            except ChsSchedulerException as e:
                logger.error("Could not start job %s: %s", spec.job_name, e)
                self._errors[spec.job_name] = str(e)
                self._statuses[spec.job_name] = JobStatus.UNKNOWN.value
            else:
                self._current = spec
                self._started_at = time.monotonic()
                self._statuses[spec.job_name] = JobStatus.RUNNING.value

    def _finish_current(self) -> bool:
        spec = self._current
        try:
            code = self._backends[spec.runtime].exit_code(spec)
        except ChsSchedulerException as e:
            logger.error("Lost track of job %s: %s", spec.job_name, e)
            self._errors[spec.job_name] = str(e)
            status = JobStatus.UNKNOWN.value
        else:
            if code is None:
                return False
            elapsed = time.monotonic() - self._started_at
            status = status_for_exit(code, elapsed, spec.timeout)
            logger.debug(
                "Job %s exited with %d after %.0fs: %s.",
                spec.job_name,
                code,
                elapsed,
                status,
            )
        self._statuses[spec.job_name] = status
        self._current = None
        return True

    def statuses(self, job_ids: list[str]) -> dict[str, str]:
        """Returns the JobStatus value of each job."""
        return {
            job_id: self._statuses.get(job_id, JobStatus.UNKNOWN.value)
            for job_id in job_ids
        }

    def fetch_log(self, job_id: str, log_path: str) -> None:
        """Writes a finished job's output to log_path and removes the job."""
        spec = self._specs[job_id]
        try:
            if job_id in self._errors:
                with open(log_path, "w") as log:
                    log.write(self._errors[job_id] + "\n")
            else:
                self._backends[spec.runtime].fetch_log(spec, log_path)
        finally:
            self._backends[spec.runtime].remove(spec)

    def cancel(self, job_ids: list[str]) -> bool:
        """Cancels queued and running jobs; returns whether any was active."""
        cancelled = set(job_ids) & set(self._specs)
        found = False
        for spec in list(self._queue):
            if spec.job_name in cancelled:
                self._queue.remove(spec)
                self._errors[spec.job_name] = "Cancelled before it started."
                self._statuses[spec.job_name] = JobStatus.CANCELLED.value
                found = True
        if self._current is not None and self._current.job_name in cancelled:
            self._backends[self._current.runtime].stop(self._current)
            self._statuses[self._current.job_name] = JobStatus.CANCELLED.value
            self._current = None
            found = True
        return found

    def cancel_all(self) -> bool:
        """Cancels every queued and running job; returns whether any was."""
        return self.cancel(list(self._specs))
