from chs.core.plugin_base import TestPlugin
from .renderer import DefaultRenderer
from .extractor import DefaultExtractor


class DefaultPlugin(DefaultRenderer, DefaultExtractor, TestPlugin):
    """Fallback plugin for tests without a specific handler extension."""

    def __init__(self, test_name: str | None = None):
        self.test_name = test_name
