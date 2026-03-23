import importlib
import logging
import pkgutil
from typing import Type

from chs.core.plugin_base import TestPlugin
from chs.core.default_plugin import DefaultPlugin

logger = logging.getLogger(__name__)


class PluginManager:
    """Discovers and manages CHS visualization plugins."""

    def __init__(self):
        self._plugins: dict[str, Type[TestPlugin]] = {}
        self._default_plugin = DefaultPlugin()

        self._discover_plugins()

    def _discover_plugins(self) -> None:
        """Iterates through chs.plugins and external dir to register plugins."""
        # 1. Discover internal plugins
        try:
            import chs.plugins

            package = chs.plugins
            if hasattr(package, "__path__"):
                prefix = package.__name__ + "."
                for _, modname, ispkg in pkgutil.iter_modules(
                    package.__path__, prefix
                ):
                    try:
                        module = importlib.import_module(modname)
                        plugin_type = "package" if ispkg else "module"
                        logger.debug(
                            "Discovered %s plugin: %s", plugin_type, modname
                        )
                        self._register_plugin_from_module(module, modname)
                    except Exception as e:
                        logger.warning(
                            "Failed to load CHS visualization plugin %s: %s",
                            modname,
                            e,
                        )
        except ImportError:
            logger.debug("No chs.plugins package found")

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
            if test_name.startswith(registered_name):
                return registered_name
        return None

    def resolve_test_name(self, test_name: str) -> str | None:
        """Returns the registered plugin name that matches the raw Ramble test name."""  # noqa: E501
        if not isinstance(test_name, str):
            return None
        return self._find_registered_name(test_name)

    def get_plugin(self, name: str) -> TestPlugin:
        """Returns the specific plugin for a test, or the default fallback."""
        # Use existing if already resolved/exact match
        if name in self._plugins:
            return self._plugins[name]()

        resolved_name = self.resolve_test_name(name)
        plugin_cls = self._plugins.get(resolved_name) if resolved_name else None

        if plugin_cls:
            try:
                return plugin_cls()
            except Exception:
                logger.exception(
                    "Failed to load plugin for %s. Using default.",
                    name,
                )

        return DefaultPlugin()
