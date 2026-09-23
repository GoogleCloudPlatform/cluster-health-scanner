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

"""Unit tests for chs.testkit.loader.

The loader registers every plugin it imports under the ``chs`` module
namespace. A plugin written to a scratch directory would therefore be
measured as CHS source, so these tests never write one: the shipped plugins
cover the happy path, and the rejection paths are driven by modules built in
memory.
"""

import os
import sys
import types
import unittest
from unittest import mock

from chs.testkit import available_tests, clear_cache, load_plugin
from chs.testkit import loader
from chs.testkit import NodeTestPlugin


class DemoPlugin(NodeTestPlugin):
    """A valid plugin that does not match any shipped test directory."""

    name = "demo"
    application = "demo"
    default_workloads = ("check",)


def plugin_module(**attributes) -> types.ModuleType:
    """Builds a stand-in for an imported plugin.py."""
    module = types.ModuleType("stand_in_plugin")
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


class LoaderTest(unittest.TestCase):
    def setUp(self):
        clear_cache()
        self.addCleanup(clear_cache)

    def load_with(self, module, test_name="dcgm"):
        """Loads a shipped test, but with the module contents we choose."""
        with mock.patch.object(loader, "_import_module", return_value=module):
            return load_plugin(test_name)

    def test_the_shipped_tests_are_all_discoverable(self):
        discovered = available_tests()

        self.assertIn("dcgm", discovered)
        self.assertIn("tpu-collectives", discovered)
        for test_name in discovered:
            self.assertEqual(load_plugin(test_name).name, test_name)

    def test_a_test_without_a_plugin_is_skipped_not_fatal(self):
        with mock.patch.object(loader, "logger") as logger:
            self.assertIsNone(load_plugin("not-a-test"))

        logger.warning.assert_called_once()

    def test_a_missing_directory_lists_nothing(self):
        absent = os.path.join(loader.TESTS_ROOT, "absent")

        with mock.patch.object(loader, "logger") as logger:
            self.assertEqual(available_tests(absent), [])

        logger.warning.assert_called_once()

    def test_a_plugin_is_loaded_once_and_reused(self):
        self.assertIs(load_plugin("dcgm"), load_plugin("dcgm"))

    def test_two_spellings_of_one_directory_load_the_same_plugin(self):
        first = load_plugin("dcgm")
        second = load_plugin("dcgm", os.path.join(loader.TESTS_ROOT, "."))

        self.assertIs(first, second)

    def test_a_file_without_a_plugin_class_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not define a class"):
            self.load_with(plugin_module(VALUE=1))

    def test_something_that_is_not_a_plugin_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must subclass"):
            self.load_with(plugin_module(Plugin=type("Plugin", (), {})))

    def test_a_plugin_in_the_wrong_directory_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must match"):
            self.load_with(plugin_module(Plugin=DemoPlugin))

    def test_a_plugin_python_cannot_load_is_reported(self):
        with mock.patch.object(
            loader.importlib.util, "spec_from_file_location", return_value=None
        ):
            with self.assertRaisesRegex(ValueError, "Could not load plugin"):
                load_plugin("dcgm")

    def test_a_plugin_that_fails_to_import_is_not_left_half_loaded(self):
        spec = mock.Mock()
        spec.loader.exec_module.side_effect = RuntimeError("boom")

        with (
            mock.patch.object(
                loader.importlib.util,
                "spec_from_file_location",
                return_value=spec,
            ),
            mock.patch.object(
                loader.importlib.util,
                "module_from_spec",
                return_value=plugin_module(),
            ),
        ):
            with self.assertRaises(RuntimeError):
                load_plugin("dcgm")

        self.assertNotIn(
            loader._module_name("dcgm", loader.TESTS_ROOT), sys.modules
        )


if __name__ == "__main__":
    unittest.main()
