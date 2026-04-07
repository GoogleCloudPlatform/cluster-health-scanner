from chs.core.plugin_base import TestPlugin
from .extractor import GemmExtractor
from .renderer import GemmRenderer


class GemmPlugin(GemmRenderer, GemmExtractor, TestPlugin):
    pass


__chs_plugin__ = {
    "test_name": "gemm",
    "plugin": GemmPlugin,
}
