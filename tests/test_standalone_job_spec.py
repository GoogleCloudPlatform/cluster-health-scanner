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

"""Unit tests for chs.schedulers.standalone.job_spec."""

import os
import shutil
import tempfile
import unittest

import yaml

from chs.exceptions import ChsValidationError
from chs.schedulers.standalone import job_spec
from chs.schedulers.standalone.job_spec import JobSpec

_RAW_SPEC = {
    "job_name": "run1-gemm-node1-abc123",
    "node": "vm-1",
    "image": "nvcr.io/nvidia/cuda:12",
    "script": "/ws/exp/job_script",
    "timeout": "600",
    "order": "3",
    "privileged": "false",
    "run_id": "vm-1-lite-1234",
    "runtime": "",
}


class LoadJobSpecTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)

    def _write(self, raw):
        with open(os.path.join(self.dir, job_spec.SPEC_FILE), "w") as f:
            yaml.safe_dump(raw, f)

    def test_loads_and_parses_values(self):
        self._write(_RAW_SPEC)
        self.assertEqual(
            job_spec.load_job_spec(self.dir),
            JobSpec(
                job_name="run1-gemm-node1-abc123",
                node="vm-1",
                image="nvcr.io/nvidia/cuda:12",
                script="/ws/exp/job_script",
                timeout=600,
                order=3,
                privileged=False,
                run_id="vm-1-lite-1234",
                runtime="",
            ),
        )
        self._write({**_RAW_SPEC, "runtime": "tpu", "privileged": "True"})
        spec = job_spec.load_job_spec(self.dir)
        self.assertEqual(spec.runtime, "tpu")
        self.assertTrue(spec.privileged)

    def test_invalid_specs_raise(self):
        with self.assertRaises(ChsValidationError):
            job_spec.load_job_spec(self.dir)
        for raw in (
            {**_RAW_SPEC, "job_name": "../etc"},
            {**_RAW_SPEC, "node": "{node}"},
            {**_RAW_SPEC, "node": ""},
            {**_RAW_SPEC, "timeout": "{timeout}"},
            {**_RAW_SPEC, "timeout": "0"},
            {**_RAW_SPEC, "privileged": "yes please"},
            {k: v for k, v in _RAW_SPEC.items() if k != "runtime"},
        ):
            self._write(raw)
            with self.assertRaises(ChsValidationError):
                job_spec.load_job_spec(self.dir)

    def test_read_script(self):
        script = os.path.join(self.dir, job_spec.SCRIPT_FILE)
        with open(script, "w") as f:
            f.write("echo hi\n")
        spec = JobSpec("j", "n", "i", script, 1, 0, False, "r")
        self.assertEqual(spec.read_script(), "echo hi\n")


if __name__ == "__main__":
    unittest.main()
