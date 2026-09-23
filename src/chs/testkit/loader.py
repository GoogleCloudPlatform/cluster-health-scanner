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

import hashlib
import importlib.util
import logging
import os
import sys

from chs.testkit.base import BaseTestPlugin

logger = logging.getLogger(__name__)

TESTS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "repos",
    "tests",
)

PLUGIN_FILENAME = "plugin.py"

# Parent module name for dynamically loaded plugins. Keeping them under one
# namespace stops a plugin from shadowing a real module.
_MODULE_NAMESPACE = "chs._test_plugins"

_cache: dict[tuple[str, str], BaseTestPlugin] = {}


def clear_cache() -> None:
    _cache.clear()


def plugin_path(test_name: str, tests_root: str | None = None) -> str:
    return os.path.join(tests_root or TESTS_ROOT, test_name, PLUGIN_FILENAME)


def available_tests(tests_root: str | None = None) -> list[str]:
    root = tests_root or TESTS_ROOT
    if not os.path.isdir(root):
        logger.warning("Test directory '%s' does not exist.", root)
        return []
    return sorted(
        entry
        for entry in os.listdir(root)
        if os.path.isfile(os.path.join(root, entry, PLUGIN_FILENAME))
    )


def load_plugin(
    test_name: str, tests_root: str | None = None
) -> BaseTestPlugin | None:
    root = os.path.abspath(tests_root or TESTS_ROOT)
    key = (root, test_name)
    if key in _cache:
        return _cache[key]

    path = plugin_path(test_name, root)
    if not os.path.isfile(path):
        logger.warning(
            "No plugin found for test '%s' at '%s'.", test_name, path
        )
        return None

    plugin = _instantiate(test_name, path, root)
    _cache[key] = plugin
    return plugin


def _instantiate(test_name: str, path: str, root: str) -> BaseTestPlugin:
    module = _import_module(test_name, path, root)

    plugin_class = getattr(module, "Plugin", None)
    if plugin_class is None:
        raise ValueError(
            f"Plugin file '{path}' does not define a class named 'Plugin'."
        )
    if not (
        isinstance(plugin_class, type)
        and issubclass(plugin_class, BaseTestPlugin)
    ):
        raise ValueError(
            f"'Plugin' in '{path}' must subclass BaseTestPlugin, "
            f"got {plugin_class!r}."
        )

    plugin = plugin_class()
    if plugin.name != test_name:
        raise ValueError(
            f"Plugin in '{path}' declares test name '{plugin.name}' but lives "
            f"in directory '{test_name}'. These must match, since the "
            f"directory name is how a test is requested."
        )
    return plugin


def _import_module(test_name: str, path: str, root: str):
    module_name = _module_name(test_name, root)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load plugin module from '{path}'.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _module_name(test_name: str, root: str) -> str:
    digest = hashlib.sha1(os.path.abspath(root).encode()).hexdigest()[:8]
    return f"{_MODULE_NAMESPACE}.{digest}.{test_name.replace('-', '_')}"
