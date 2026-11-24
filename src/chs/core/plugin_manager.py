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

import logging
from typing import Type

from chs.core.plugin_base import TestPlugin
from chs.core.default_plugin import DefaultPlugin
from chs.tests import TEST_PACKAGES

logger = logging.getLogger(__name__)


class PluginManager:
    """Discovers and manages CHS visualization plugins."""

    def __init__(self):
        self._plugins: dict[str, Type[TestPlugin]] = {}

        self._discover_plugins()

    def _discover_plugins(self) -> None:
        """Iterates through chs.tests to register plugins."""
        for package in TEST_PACKAGES:
            logger.debug("Discovered package plugin: %s", package.__name__)
            self._register_plugin_from_module(package, package.__name__)

    def _register_plugin_from_module(self, module, modname: str) -> None:
        """Helper to register a plugin from a loaded module."""
        if hasattr(module, "__chs_plugin__"):
            plugin_info = getattr(module, "__chs_plugin__")
            test_name = plugin_info.get("test_name")
            plugin_cls = plugin_info.get("plugin") or plugin_info.get(
                "renderer"
            )
            if test_name and plugin_cls:
                self._plugins[test_name] = plugin_cls
                logger.debug(
                    "Registered CHS visualization plugin for %s from %s",
                    test_name,
                    modname,
                )

    def _find_registered_name(self, test_name: str) -> str | None:
        """Finds the longest registered prefix match."""
        if test_name in self._plugins:
            return test_name

        for registered_name in sorted(
            self._plugins.keys(), key=len, reverse=True
        ):
            if registered_name in test_name:
                return registered_name
        return None

    def resolve_test_name(self, test_name: str) -> str | None:
        """Returns the registered plugin name that matches the raw Ramble test name."""  # noqa: E501
        if not isinstance(test_name, str):
            return None
        return self._find_registered_name(test_name)

    def _load_and_instantiate_plugin(
        self, plugin_cls: Type[TestPlugin], test_name: str
    ) -> TestPlugin | None:
        """Helper to safely instantiate and configure a plugin class."""
        try:
            plugin_instance = plugin_cls()
            plugin_instance.test_name = test_name
            return plugin_instance
        except Exception:
            logger.exception(
                "Failed to load plugin for %s. Using default.",
                test_name,
            )
            return None

    def get_plugin(self, name: str) -> TestPlugin:
        """Returns the specific plugin for a test, or the default fallback."""
        # Use existing if already resolved/exact match
        if name in self._plugins:
            plugin_instance = self._load_and_instantiate_plugin(
                self._plugins[name], name
            )
            return (
                plugin_instance
                if plugin_instance
                else DefaultPlugin(test_name=name)
            )

        resolved_name = self.resolve_test_name(name)
        plugin_cls = self._plugins.get(resolved_name) if resolved_name else None

        if plugin_cls:
            plugin_instance = self._load_and_instantiate_plugin(
                plugin_cls, resolved_name
            )
            if plugin_instance:
                return plugin_instance

        return DefaultPlugin(test_name=resolved_name or name)
