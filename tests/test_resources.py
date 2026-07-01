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

import unittest
from chs.resources import normalize_machine_type, is_tpu


class TestResources(unittest.TestCase):
    def test_normalize_machine_type_url(self):
        self.assertEqual(
            normalize_machine_type(
                "zones/us-central1-a/machineTypes/a3-megagpu-8g"
            ),
            "a3-megagpu-8g",
        )

    def test_normalize_machine_type_prefix(self):
        self.assertEqual(
            normalize_machine_type("a3-megagpu-8g-custom"), "a3-megagpu-8g"
        )

    def test_normalize_machine_type_unknown(self):
        self.assertEqual(normalize_machine_type(""), "unknown")
        self.assertEqual(normalize_machine_type(None), "unknown")

    def test_normalize_machine_type_custom(self):
        self.assertEqual(
            normalize_machine_type(
                "zones/us-central1-a/machineTypes/custom-machine-type"
            ),
            "custom-machine-type",
        )

    def test_is_tpu(self):
        self.assertTrue(is_tpu("tpu7x-standard-4t"))
        self.assertTrue(is_tpu("ct6e-standard-4t"))
        self.assertFalse(is_tpu("a3-megagpu-8g"))
        self.assertFalse(is_tpu("a4x-highgpu-4g"))
        self.assertFalse(is_tpu("unknown"))
        self.assertFalse(is_tpu(""))
        self.assertFalse(is_tpu(None))


if __name__ == "__main__":
    unittest.main()
