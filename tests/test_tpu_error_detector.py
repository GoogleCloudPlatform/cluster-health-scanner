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

import re
import unittest


class TestTpuErrorDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import os

        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        cls.modifier_path = os.path.join(
            project_root,
            "src",
            "chs",
            "repos",
            "modifiers",
            "tpu-error-detector",
            "modifier.py",
        )

        # Read the file content
        with open(cls.modifier_path, "r") as f:
            cls.content = f.read()

        # Extract regexes
        cls.oom_regex = cls.extract_regex("TPU OOM Error Detected")
        cls.host_mem_regex = cls.extract_regex("Host Memory Error Detected")
        cls.hardware_regex = cls.extract_regex(
            "TPU Hardware / Driver / Network Error Detected"
        )
        cls.comp_regex = cls.extract_regex("TPU/XLA Compilation Error Detected")
        cls.profiler_regex = cls.extract_regex("Profiler Error Detected")
        cls.env_config_regex = cls.extract_regex(
            "TPU Environment / Configuration Error Detected"
        )
        cls.bench_fail_regex = cls.extract_regex(
            "TPU Benchmark Failure Detected"
        )

    @classmethod
    def extract_regex(cls, name):
        pattern = (
            rf'figure_of_merit\(\s*"{name}",\s*'
            r'fom_regex=r"((?:\(\?i\))?\(\?P<[^>]+>[^"]+\))"'
        )
        match = re.search(pattern, cls.content, re.DOTALL)
        if match:
            return match.group(1)
        raise ValueError(f"Could not find regex for {name}")

    def test_oom_regex(self):
        regex = self.oom_regex
        self.assertTrue(re.match(regex, "jax.errors.ResourceExhaustedError"))
        self.assertTrue(re.match(regex, "Resource exhausted during allocation"))
        self.assertTrue(
            re.match(regex, "[node-1] Resource exhausted: Out of memory")
        )
        self.assertTrue(re.match(regex, "Out of memory on device"))
        self.assertFalse(re.match(regex, "Normal allocation succeeded"))

    def test_host_mem_regex(self):
        regex = self.host_mem_regex
        self.assertTrue(re.match(regex, "MemoryError"))
        self.assertTrue(re.match(regex, "Killed"))
        self.assertTrue(
            re.match(regex, "[node-2] Host process Killed by OOM-killer")
        )
        self.assertFalse(re.match(regex, "Memory usage is 10%"))

    def test_hardware_regex(self):
        regex = self.hardware_regex
        self.assertTrue(re.match(regex, "TPU device reset"))
        self.assertTrue(re.match(regex, "Hardware error occurred"))
        self.assertTrue(re.match(regex, "Failed to initialize PJRT client"))
        self.assertTrue(re.match(regex, "PjRtClient initialization failed"))
        self.assertTrue(re.match(regex, "Collective operation timed out"))
        self.assertTrue(re.match(regex, "ICI link failure detected"))
        self.assertTrue(re.match(regex, "ERROR: libtpu not found."))
        self.assertTrue(
            re.match(
                regex,
                "PermissionError: [Errno 13] Permission denied: '/dev/vfio/0'",
            )
        )
        self.assertTrue(
            re.match(
                regex,
                "PermissionError: [Errno 13] Permission denied: '/dev/accel0'",
            )
        )
        self.assertFalse(re.match(regex, "TPU initialized successfully"))

    def test_comp_regex(self):
        regex = self.comp_regex
        self.assertTrue(
            re.match(regex, "jax.errors.XlaRuntimeError: compilation failed")
        )
        self.assertTrue(re.match(regex, "XLA compiler failure"))
        self.assertTrue(re.match(regex, "failed to compile graph"))
        self.assertTrue(
            re.match(regex, "Mesh geometry does not match physical devices")
        )
        self.assertTrue(
            re.match(regex, "TypeError: unsupported operand type(s)")
        )
        self.assertFalse(re.match(regex, "Compilation succeeded in 1.2s"))

    def test_profiler_regex(self):
        regex = self.profiler_regex
        self.assertTrue(re.match(regex, "Failed to start profiler"))
        self.assertTrue(
            re.match(regex, "profiler runtime error: already running")
        )
        self.assertTrue(
            re.match(regex, "permission error writing to trace directory")
        )
        self.assertFalse(re.match(regex, "Profiler finished trace"))

    def test_env_config_regex(self):
        regex = self.env_config_regex
        self.assertTrue(
            re.match(regex, "libtpu version: N/A (incompatible environment)")
        )
        self.assertTrue(re.match(regex, "tpu-info: No such file or directory"))
        self.assertFalse(re.match(regex, "libtpu version: 0.0.24"))

    def test_bench_fail_regex(self):
        regex = self.bench_fail_regex
        self.assertTrue(
            re.match(regex, "Benchmark func failed: ResourceExhaustedError")
        )
        self.assertTrue(
            re.match(
                regex,
                "Benchmark func failed: 'Failed to initialize PJRT client'",
            )
        )
        self.assertFalse(re.match(regex, "Benchmark succeeded"))


if __name__ == "__main__":
    unittest.main()
