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

"""Fake test packages shared by the catalog and thresholds tests."""

import os
import tempfile
import textwrap
import unittest
from unittest import mock

from chs.testkit import config as chs_config
from chs.testkit import loader
from chs.testkit import SingleNodePlugin


def build_plugin(name):
    return type(
        "Plugin",
        (SingleNodePlugin,),
        {
            "name": name,
            "application": name,
            "default_workloads": ("check",),
            "category": "Compute",
            "description": f"Short description for {name}",
            "long_description": f"Long description for {name}",
        },
    )()


class FakeTestsTestCase(unittest.TestCase):
    """Serves fake tests to the testkit.

    Plugins are kept in memory and returned by a patched loader; config,
    threshold and platform files are written to temporary directories.
    """

    def setUp(self):
        super().setUp()
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.tests_root = os.path.join(temp_dir.name, "tests")
        self.platforms_root = os.path.join(temp_dir.name, "platforms")
        os.makedirs(self.tests_root)
        os.makedirs(self.platforms_root)

        self.test_names = []
        self.plugins = {}
        for patcher in (
            mock.patch.object(chs_config, "TESTS_ROOT", self.tests_root),
            mock.patch.object(
                chs_config, "PLATFORMS_ROOT", self.platforms_root
            ),
            mock.patch.object(
                loader,
                "available_tests",
                side_effect=lambda: sorted(self.test_names),
            ),
            mock.patch.object(
                loader, "load_plugin", side_effect=self.plugins.get
            ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def write_platform(self, machine, text):
        path = os.path.join(self.platforms_root, f"{machine}.yaml")
        with open(path, "w") as f:
            f.write(textwrap.dedent(text))

    def write_test(self, name, config_text, thresholds_text=None):
        """Adds a test with a plugin, a config.yaml and optional thresholds."""
        directory = os.path.join(self.tests_root, name)
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "config.yaml"), "w") as f:
            f.write(textwrap.dedent(config_text))
        if thresholds_text is not None:
            with open(os.path.join(directory, "thresholds.yaml"), "w") as f:
                f.write(textwrap.dedent(thresholds_text))
        if name not in self.test_names:
            self.test_names.append(name)
        self.plugins[name] = build_plugin(name)
