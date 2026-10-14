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

import importlib.util
import logging
import os
import sys
from collections.abc import Iterator

from chs.testkit import config as chs_config
from chs.testkit.base import BaseTestPlugin

logger = logging.getLogger(__name__)

PLUGIN_FILENAME = "plugin.py"

# Parent module name for dynamically loaded plugins. Keeping them under one
# namespace stops a plugin from shadowing a real module.
_MODULE_NAMESPACE = "chs._test_plugins"

_cache: dict[str, BaseTestPlugin] = {}


def clear_cache() -> None:
    _cache.clear()


def plugin_path(test_name: str) -> str:
    return os.path.join(chs_config.TESTS_ROOT, test_name, PLUGIN_FILENAME)


def available_tests() -> list[str]:
    root = chs_config.TESTS_ROOT
    if not os.path.isdir(root):
        logger.warning("Test directory '%s' does not exist.", root)
        return []
    return sorted(
        entry
        for entry in os.listdir(root)
        if os.path.isfile(os.path.join(root, entry, PLUGIN_FILENAME))
    )


def enabled_tests() -> Iterator[tuple[str, dict]]:
    """Yields (test_name, config) for every test not set to enabled: false."""
    for test_name in available_tests():
        config = chs_config.load_test_config(test_name)
        if config.get("enabled", True):
            yield test_name, config


def load_plugin(test_name: str) -> BaseTestPlugin | None:
    if test_name in _cache:
        return _cache[test_name]

    path = plugin_path(test_name)
    if not os.path.isfile(path):
        logger.warning(
            "No plugin found for test '%s' at '%s'.", test_name, path
        )
        return None

    plugin = _instantiate(test_name, path)
    _cache[test_name] = plugin
    return plugin


def _instantiate(test_name: str, path: str) -> BaseTestPlugin:
    module = _import_module(test_name, path)

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
            f"Plugin name '{plugin.name}' in '{path}' does not match its "
            f"directory '{test_name}'."
        )
    return plugin


def _import_module(test_name: str, path: str):
    module_name = _module_name(test_name)
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


def _module_name(test_name: str) -> str:
    return f"{_MODULE_NAMESPACE}.{test_name.replace('-', '_')}"
