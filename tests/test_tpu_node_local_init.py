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

import os
import unittest
from chs.testkit import config as chs_config


class TestTpuNodeLocalInitModifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        cls.modifier_path = os.path.join(
            project_root,
            "src",
            "chs",
            "repos",
            "modifiers",
            "tpu-node-local-init",
            "modifier.py",
        )

        with open(cls.modifier_path, "r") as f:
            cls.content = f.read()

    def test_modifier_file_exists(self):
        self.assertTrue(os.path.isfile(self.modifier_path))

    def test_modifier_exports_env_vars(self):
        self.assertIn('name = "tpu-node-local-init"', self.content)
        self.assertIn("TPU_HOST_BOUNDS=1,1,1", self.content)
        self.assertIn("TPU_SKIP_MDS_QUERY=true", self.content)
        self.assertIn("TPU_WORKER_ID=0", self.content)
        self.assertIn("TPU_WORKER_HOSTNAMES=localhost", self.content)
        self.assertIn("--deepsea_hal_test_skip_slicebuilder=true", self.content)
        self.assertIn(
            "--deepsea_hal_test_allow_multichip_skip_slicebuilder=true",
            self.content,
        )
        self.assertIn(
            "--xla_tpu_use_enhanced_launch_barrier=false", self.content
        )
        self.assertIn(
            "--megascale_error_reporter_abort_on_hang=true", self.content
        )

    def test_single_node_tests_include_modifier(self):
        tests_to_check = [
            ("tpu-compute-stress", "tpu7x-standard-4t", "kubernetes"),
            ("tpu-hbm-stress", "tpu7x-standard-4t", "kubernetes"),
            ("tpu-info", "tpu7x-standard-4t", "kubernetes"),
        ]
        for test_name, machine, scheduler in tests_to_check:
            with self.subTest(test=test_name):
                resolved = chs_config.resolve_test(
                    test_name, machine, scheduler
                )
                modifier_names = [
                    m["name"] for m in resolved.get("modifiers", [])
                ]
                self.assertIn("tpu-node-local-init", modifier_names)


if __name__ == "__main__":
    unittest.main()
