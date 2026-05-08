from chs.core.default_plugin.extractor import DefaultExtractor
from chs.core.plugin_base import TestPlugin
from .renderer import StreamRenderer


class StreamPlugin(StreamRenderer, DefaultExtractor, TestPlugin):
    def get_relative_fom_keys(self) -> list[tuple[str, str]]:
        return [("Rate", "PEAK Bandwidth")]


__chs_plugin__ = {
    "test_name": "stream",
    "plugin": StreamPlugin,
}
