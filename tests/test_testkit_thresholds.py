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

"""Unit tests for chs.testkit.thresholds."""

import unittest

from chs.testkit import thresholds
from tests.testkit_fixtures import FakeTestsTestCase

CONFIG = "platforms: [machine-x]\n"


class BuildThresholdsTest(FakeTestsTestCase):
    def test_the_type_is_copied_onto_every_machine(self):
        self.write_test(
            "demo",
            CONFIG,
            """
            type: min
            default:
              bandwidth: 50
            machine-x:
              bandwidth: 100
            """,
        )

        self.assertEqual(
            {
                "default": {
                    "demo": {"type": "min", "thresholds": {"bandwidth": 50}}
                },
                "machine-x": {
                    "demo": {"type": "min", "thresholds": {"bandwidth": 100}}
                },
            },
            thresholds.build_thresholds(),
        )

    def test_two_tests_land_under_the_same_machine(self):
        for name in ("demo-a", "demo-b"):
            self.write_test(name, CONFIG, "type: min\nmachine-x: {gbps: 10}\n")

        assembled = thresholds.build_thresholds()

        self.assertEqual({"demo-a", "demo-b"}, set(assembled["machine-x"]))

    def test_tests_without_usable_thresholds_contribute_nothing(self):
        self.write_test("no-file", CONFIG)
        self.write_test(
            "disabled",
            "enabled: false\n" + CONFIG,
            "type: min\ndefault: {gbps: 50}\n",
        )

        self.assertEqual({}, thresholds.build_thresholds())


if __name__ == "__main__":
    unittest.main()
