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

"""Tests for chs.utils.lifecycle."""

import json
import unittest

from chs.utils.lifecycle import JobStatus


class JobStatusTest(unittest.TestCase):
    def test_formats_as_value(self):
        for status in JobStatus:
            with self.subTest(status=status.name):
                self.assertEqual(str(status), status.value)
                self.assertEqual(f"{status}", status.value)
                self.assertEqual("%s" % status, status.value)
                self.assertEqual(json.dumps(status), f'"{status.value}"')

    def test_equals_its_value(self):
        self.assertEqual(JobStatus.RUNNING, "RUNNING")
        self.assertIn(JobStatus.PENDING, ("PENDING", "UNKNOWN"))

    def test_from_string_is_case_insensitive(self):
        self.assertEqual(JobStatus.from_string("completed"), "COMPLETED")
        self.assertEqual(JobStatus.from_string("Running"), "RUNNING")

    def test_from_string_falls_back_to_unknown(self):
        self.assertEqual(JobStatus.from_string("BOOT_FAIL"), "UNKNOWN")

    def test_is_terminal(self):
        terminal = {
            JobStatus.COMPLETED,
            JobStatus.TIMEOUT,
            JobStatus.CANCELLED,
            JobStatus.DEADLINE,
            JobStatus.UNKNOWN,
        }
        for status in JobStatus:
            with self.subTest(status=status.name):
                self.assertEqual(
                    JobStatus.is_terminal(status.value), status in terminal
                )


if __name__ == "__main__":
    unittest.main()
