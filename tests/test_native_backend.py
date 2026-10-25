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

"""Unit tests for chs.schedulers.standalone.native_backend."""

import os
import pathlib
import shutil
import tempfile
import time
import unittest

from chs.exceptions import ChsSchedulerException
from chs.schedulers.standalone.job_spec import JobSpec
from chs.schedulers.standalone.native_backend import NativeBackend


class NativeBackendTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        self.runtime = pathlib.Path(self.tmp, "tpu")
        (self.runtime / "venv" / "bin").mkdir(parents=True)
        self.backend = NativeBackend(
            self.runtime,
            chip_count=4,
        )

    def _spec(self, script, name="job-1"):
        path = os.path.join(self.tmp, f"{name}.sh")
        with open(path, "w") as f:
            f.write(script)
        return JobSpec(name, "vm", "", path, 60, 1, False, "run", "tpu")

    def _wait(self, spec):
        for _ in range(100):
            code = self.backend.exit_code(spec)
            if code is not None:
                return code
            time.sleep(0.05)
        self.fail("job did not exit")

    def test_runs_the_script_natively_with_the_tpu_env(self):
        spec = self._spec(
            'echo "$CHS_AMB_DIR $TPU_CHIPS_PER_HOST_BOUNDS"; exit 3'
        )
        self.backend.start(spec)
        self.assertEqual(self._wait(spec), 3)
        log = os.path.join(self.tmp, "out.log")
        self.backend.fetch_log(spec, log)
        with open(log) as f:
            self.assertEqual(f.read(), f"{self.runtime}/code 2,2,1\n")
        self.backend.remove(spec)
        self.assertFalse(os.path.exists(self.backend._out_path(spec)))

    def test_stop_kills_and_reports_the_signal_like_a_shell(self):
        spec = self._spec("echo started; sleep 60")
        self.backend.start(spec)
        time.sleep(0.2)
        self.backend.stop(spec)
        self.assertEqual(self.backend.exit_code(spec), 137)
        log = os.path.join(self.tmp, "out.log")
        self.backend.fetch_log(spec, log)
        with open(log) as f:
            self.assertEqual(f.read(), "started\n")

    def test_invalid_job_fails_to_start(self):
        spec = self._spec("true")
        backend = NativeBackend(self.runtime, chip_count=3)
        with self.assertRaises(ChsSchedulerException):
            backend.start(spec)


if __name__ == "__main__":
    unittest.main()
