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

"""Unit tests for chs.schedulers.standalone.runner."""

import os
import shutil
import tempfile
import unittest
from unittest import mock

from chs.exceptions import ChsSchedulerException, ChsValidationError
from chs.schedulers.standalone import runner
from chs.schedulers.standalone.job_spec import JobSpec
from chs.utils.lifecycle import JobStatus


def _spec(name: str, order: int, runtime: str = "", timeout: int = 60):
    return JobSpec(
        job_name=name,
        node="vm",
        image="img",
        script="/s",
        timeout=timeout,
        order=order,
        privileged=False,
        run_id="run",
        runtime=runtime,
    )


class FakeBackend:
    """Records calls; jobs run until finish() sets their exit code."""

    def __init__(self):
        self.calls = []
        self.exit_codes = {}
        self.fail_start = set()
        self.lost = set()

    def finish(self, name, code=0):
        self.exit_codes[name] = code

    def start(self, spec):
        self.calls.append(("start", spec.job_name))
        if spec.job_name in self.fail_start:
            raise ChsSchedulerException("image not found")

    def exit_code(self, spec):
        if spec.job_name in self.lost:
            raise ChsSchedulerException("no such container")
        return self.exit_codes.get(spec.job_name)

    def stop(self, spec):
        self.calls.append(("stop", spec.job_name))

    def fetch_log(self, spec, log_path):
        self.calls.append(("fetch_log", spec.job_name))
        with open(log_path, "w") as f:
            f.write(f"output of {spec.job_name}\n")

    def remove(self, spec):
        self.calls.append(("remove", spec.job_name))


class SequentialRunnerTest(unittest.TestCase):
    def setUp(self):
        self.backend = FakeBackend()
        self.runner = runner.SequentialRunner({"": self.backend})
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)

    def _status(self, name):
        return self.runner.statuses([name])[name]

    def test_status_for_exit(self):
        for code, elapsed, expected in (
            (0, 5, JobStatus.COMPLETED),
            (1, 5, JobStatus.COMPLETED),
            (124, 5, JobStatus.TIMEOUT),
            (137, 60, JobStatus.TIMEOUT),
            (137, 5, JobStatus.COMPLETED),
            (125, 5, JobStatus.COMPLETED),
            (126, 5, JobStatus.COMPLETED),
            (127, 5, JobStatus.COMPLETED),
        ):
            self.assertEqual(
                runner.status_for_exit(code, elapsed, 60), expected.value
            )

    def test_runs_one_at_a_time_in_order(self):
        self.assertEqual(self._status("nope"), JobStatus.UNKNOWN.value)
        ids = self.runner.submit([_spec("b", 2), _spec("a", 1), _spec("c", 3)])
        self.assertEqual(ids, ["b", "a", "c"])
        self.assertEqual(
            self.runner.statuses(ids),
            dict.fromkeys(ids, JobStatus.PENDING.value),
        )
        self.assertEqual(self.backend.calls, [])

        self.runner.poll()
        self.assertEqual(self.backend.calls, [("start", "a")])
        self.assertEqual(self._status("a"), JobStatus.RUNNING.value)
        self.runner.poll()
        self.assertEqual(self.backend.calls, [("start", "a")])

        self.backend.finish("a", 1)
        self.runner.poll()
        self.assertEqual(self._status("a"), JobStatus.COMPLETED.value)
        self.assertEqual(self._status("b"), JobStatus.RUNNING.value)

        self.backend.finish("b", 124)
        self.runner.poll()
        self.backend.finish("c", 0)
        self.runner.poll()
        self.assertEqual(
            self.runner.statuses(["a", "b", "c"]),
            {
                "a": JobStatus.COMPLETED.value,
                "b": JobStatus.TIMEOUT.value,
                "c": JobStatus.COMPLETED.value,
            },
        )
        log = os.path.join(self.tmp, "a.out")
        self.runner.fetch_log("a", log)
        self.assertEqual(
            self.backend.calls[-2:], [("fetch_log", "a"), ("remove", "a")]
        )
        with open(log) as f:
            self.assertEqual(f.read(), "output of a\n")

    def test_killed_after_timeout_is_timeout(self):
        self.runner.submit([_spec("a", 1, timeout=10)])
        with mock.patch.object(runner.time, "monotonic", return_value=100):
            self.runner.poll()
        self.backend.finish("a", 137)
        with mock.patch.object(runner.time, "monotonic", return_value=140):
            self.runner.poll()
        self.assertEqual(self._status("a"), JobStatus.TIMEOUT.value)

    def test_start_failure_and_lost_job_are_unknown(self):
        self.backend.fail_start.add("a")
        self.runner.submit([_spec("a", 1), _spec("b", 2), _spec("c", 3)])
        self.runner.poll()
        self.assertEqual(self._status("a"), JobStatus.UNKNOWN.value)
        self.assertEqual(self._status("b"), JobStatus.RUNNING.value)

        log = os.path.join(self.tmp, "a.out")
        self.runner.fetch_log("a", log)
        with open(log) as f:
            self.assertIn("image not found", f.read())
        self.assertNotIn(("fetch_log", "a"), self.backend.calls)
        self.assertEqual(self.backend.calls[-1], ("remove", "a"))

        self.backend.lost.add("b")
        self.runner.poll()
        self.assertEqual(self._status("b"), JobStatus.UNKNOWN.value)
        self.assertEqual(self._status("c"), JobStatus.RUNNING.value)

    def test_cancel_running_and_queued(self):
        self.runner.submit([_spec("a", 1), _spec("b", 2), _spec("c", 3)])
        self.runner.poll()
        self.assertTrue(self.runner.cancel(["a", "b"]))
        self.assertIn(("stop", "a"), self.backend.calls)
        self.assertEqual(self._status("a"), JobStatus.CANCELLED.value)
        self.assertEqual(self._status("b"), JobStatus.CANCELLED.value)

        self.runner.poll()
        self.assertEqual(self._status("c"), JobStatus.RUNNING.value)
        self.assertNotIn(("start", "b"), self.backend.calls)

        log = os.path.join(self.tmp, "b.out")
        self.runner.fetch_log("b", log)
        with open(log) as f:
            self.assertIn("Cancelled before it started", f.read())

        self.backend.finish("c")
        self.runner.poll()
        self.assertFalse(self.runner.cancel_all())
        self.assertEqual(self._status("c"), JobStatus.COMPLETED.value)

    def test_cancel_all(self):
        self.runner.submit([_spec("a", 1), _spec("b", 2)])
        self.runner.poll()
        self.assertTrue(self.runner.cancel_all())
        self.runner.poll()
        self.assertEqual(
            self.runner.statuses(["a", "b"]),
            dict.fromkeys(["a", "b"], JobStatus.CANCELLED.value),
        )
        self.assertEqual(self.backend.calls, [("start", "a"), ("stop", "a")])

    def test_dispatches_by_runtime_and_validates_submit(self):
        other = FakeBackend()
        r = runner.SequentialRunner({"": self.backend, "tpu": other})
        r.submit([_spec("a", 1, runtime="tpu"), _spec("b", 2)])
        r.poll()
        self.assertEqual(other.calls, [("start", "a")])
        other.finish("a")
        r.poll()
        self.assertEqual(self.backend.calls, [("start", "b")])

        with self.assertRaisesRegex(ChsValidationError, "runtime 'tpu'"):
            self.runner.submit([_spec("x", 1), _spec("y", 2, runtime="tpu")])
        self.assertEqual(self._status("x"), JobStatus.UNKNOWN.value)
        with self.assertRaisesRegex(ChsValidationError, "Duplicate"):
            r.submit([_spec("a", 3)])
        with self.assertRaisesRegex(ChsValidationError, "Duplicate"):
            self.runner.submit([_spec("x", 1), _spec("x", 2)])


if __name__ == "__main__":
    unittest.main()
