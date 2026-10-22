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

"""Tests for the catalog readers in ``chs.utils.tests``."""

import logging
import unittest
from unittest import mock

from chs.utils.tests import (
    _load_tests,
    get_all_supported_machine_types,
    get_all_supported_schedulers,
    get_all_supported_tests,
    get_tests_by_machine_type,
    get_tests_map,
    TestsMap as _TestsMap,
    get_tests_by_machine_type_and_scheduler,
)


def catalog(*tests) -> dict:
    """Builds a catalog of the shape ``build_catalog`` returns."""
    return {"tests": list(tests)}


def entry(name: str, *platforms) -> dict:
    """Builds one catalog entry from ``(scheduler, machine)`` pairs."""
    return {
        "name": name,
        "platforms": [
            {"scheduler": scheduler, "machine_type": machine}
            for scheduler, machine in platforms
        ],
    }


class TestTests(unittest.TestCase):
    """Each reader gets a catalog and is checked on what it makes of it."""

    def setUp(self):
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)
        _load_tests.cache_clear()
        self.addCleanup(_load_tests.cache_clear)

    def use(self, data: dict):
        """Makes ``data`` the catalog for the duration of the test."""
        patcher = mock.patch("chs.utils.tests._load_tests", return_value=data)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_get_all_supported_tests(self):
        """Every test in the catalog is offered, in catalog order."""
        self.use(catalog(entry("t1"), entry("t2"), entry("t3")))
        self.assertEqual(get_all_supported_tests(), ["t1", "t2", "t3"])

    def test_get_all_supported_machine_types(self):
        """A machine named by any test counts, listed once and sorted."""
        self.use(
            catalog(
                entry("t1", ("s1", "m1"), ("s2", "m2")),
                entry("t2", ("s1", "m2"), ("s2", "m3")),
            )
        )
        self.assertEqual(get_all_supported_machine_types(), ["m1", "m2", "m3"])

    def test_get_all_supported_schedulers(self):
        """Likewise for schedulers."""
        self.use(
            catalog(
                entry("t1", ("s1", "m1"), ("s2", "m2")),
                entry("t2", ("s2", "m2"), ("s3", "m3")),
            )
        )
        self.assertEqual(get_all_supported_schedulers(), ["s1", "s2", "s3"])

    def test_get_tests_map(self):
        """The catalog is inverted into scheduler -> machine -> tests."""
        self.use(
            catalog(
                entry(
                    "t1",
                    ("s1", "m1"),
                    ("s2", "m1"),
                    ("s1", "m2"),
                    ("s2", "m2"),
                ),
                entry("t2", ("s1", "m2")),
            )
        )
        expected = {
            "s1": {"m1": ["t1"], "m2": ["t1", "t2"]},
            "s2": {"m1": ["t1"], "m2": ["t1"]},
        }
        tests_map = get_tests_map()
        self.assertEqual(tests_map, _TestsMap(data=expected))
        self.assertEqual(tests_map.get("s1"), expected["s1"])
        self.assertEqual(tests_map.get("s2"), expected["s2"])
        self.assertIsNone(tests_map.get("unknown_scheduler"))

    def test_get_tests_by_machine_type(self):
        """A machine gets the tests that name it, on any scheduler."""
        self.use(
            catalog(
                entry("t1", ("s1", "m1"), ("s2", "m2")),
                entry("t2", ("s1", "m2"), ("s2", "m3")),
            )
        )
        self.assertEqual(get_tests_by_machine_type("m1"), ["t1"])
        self.assertEqual(get_tests_by_machine_type("m2"), ["t1", "t2"])
        self.assertEqual(get_tests_by_machine_type("m3"), ["t2"])
        self.assertEqual(get_tests_by_machine_type("m4"), [])

    def test_readers_empty_catalog(self):
        """No tests on disk is an empty answer, not a crash."""
        self.use(catalog())
        self.assertEqual(get_all_supported_tests(), [])
        self.assertEqual(get_all_supported_machine_types(), [])
        self.assertEqual(get_all_supported_schedulers(), [])
        self.assertEqual(get_tests_map(), _TestsMap(data={}))


class TestTestsOnDisk(unittest.TestCase):
    """Runs the readers against the tests that are actually on disk."""

    def setUp(self):
        _load_tests.cache_clear()
        self.addCleanup(_load_tests.cache_clear)

    def test_get_all_supported_tests_not_empty(self):
        """CHS ships tests, so deriving the catalog must find some."""
        self.assertTrue(get_all_supported_tests())

    def test__load_tests_cached(self):
        """Deriving the catalog walks the whole tests tree, so it is cached."""
        self.assertIs(_load_tests(), _load_tests())

    def test_get_all_supported_machine_types_matches_catalog(self):
        """The flat list and the inverted map agree on the machines."""
        from_map = set(get_all_supported_machine_types())
        from_catalog = {
            platform["machine_type"]
            for test in _load_tests()["tests"]
            for platform in test["platforms"]
        }
        self.assertEqual(from_map, from_catalog)

    def test_get_tests_by_machine_type_covers_all_tests(self):
        """No test is listed yet unreachable, which would make it dead."""
        reachable = set()
        for machine in get_all_supported_machine_types():
            reachable.update(get_tests_by_machine_type(machine))
        self.assertEqual(reachable, set(get_all_supported_tests()))

    def test_a4x_max_slurm_supported_tests(self):
        """Test that a4x-maxgpu-4g-metal supports expected tests on slurm
        in default config.
        """
        tests = get_tests_by_machine_type_and_scheduler(
            "a4x-maxgpu-4g-metal", "slurm"
        )
        self.assertIn("nvidia-smi", tests)
        self.assertIn("dcgm", tests)
        self.assertIn("gemm", tests)
        self.assertIn("stream", tests)


if __name__ == "__main__":
    unittest.main()
