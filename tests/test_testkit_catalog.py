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

"""Unit tests for chs.testkit.catalog."""

import unittest

from chs.testkit import catalog
from tests.testkit_fixtures import FakeTestsTestCase


class BuildCatalogTest(FakeTestsTestCase):
    def setUp(self):
        super().setUp()
        self.write_platform("machine-x", "schedulers: [kubernetes, slurm]\n")

    def test_an_entry_lists_every_scheduler_the_machine_supports(self):
        self.write_test("demo", "platforms: [machine-x]\n")

        self.assertEqual(
            [
                {
                    "name": "demo",
                    "category": "Compute",
                    "description": "Short description for demo",
                    "long_description": "Long description for demo",
                    "platforms": [
                        {
                            "scheduler": "kubernetes",
                            "machine_type": "machine-x",
                        },
                        {"scheduler": "slurm", "machine_type": "machine-x"},
                    ],
                }
            ],
            catalog.build_catalog()["tests"],
        )

    def test_a_scheduler_restriction_limits_the_listed_platforms(self):
        self.write_test("demo", "platforms:\n  - machine-x: [slurm]\n")

        (entry,) = catalog.build_catalog()["tests"]

        self.assertEqual(
            [{"scheduler": "slurm", "machine_type": "machine-x"}],
            entry["platforms"],
        )

    def test_disabled_tests_are_left_out(self):
        self.write_test("disabled", "enabled: false\nplatforms: [machine-x]\n")

        self.assertEqual([], catalog.build_catalog()["tests"])


if __name__ == "__main__":
    unittest.main()
