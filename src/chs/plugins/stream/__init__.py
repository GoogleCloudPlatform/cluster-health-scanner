from chs.core.default_plugin.extractor import DefaultExtractor
from chs.core.plugin_base import TestPlugin
from .renderer import StreamRenderer


class StreamPlugin(StreamRenderer, DefaultExtractor, TestPlugin):
    pass


__chs_plugin__ = {
    "test_name": "stream",
    "plugin": StreamPlugin,
}
