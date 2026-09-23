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

"""Toolkit for defining CHS tests.

A test is a directory under ``chs/repos/tests`` holding a ``plugin.py`` that
subclasses one of the plugin shapes exported here. Everything a test needs to
say about itself is said in that one file, and the runner discovers it without
being told the test exists.
"""

from chs.testkit.base import (
    BaseTestPlugin,
    ClusterTestPlugin,
    NodeTestPlugin,
    SliceTestPlugin,
)
from chs.testkit.loader import available_tests, clear_cache, load_plugin
from chs.testkit.spec import (
    STRIPPED_WORKSPACE_KEYS,
    ExperimentSpec,
    NodeGrouping,
    TestRequest,
    build_test_request,
)

__all__ = [
    "BaseTestPlugin",
    "ClusterTestPlugin",
    "NodeTestPlugin",
    "SliceTestPlugin",
    "ExperimentSpec",
    "NodeGrouping",
    "STRIPPED_WORKSPACE_KEYS",
    "TestRequest",
    "available_tests",
    "build_test_request",
    "clear_cache",
    "load_plugin",
]
