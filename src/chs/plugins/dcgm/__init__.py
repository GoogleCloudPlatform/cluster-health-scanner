from chs.core.plugin_base import TestPlugin
from .extractor import DcgmExtractor
from .renderer import DcgmRenderer


class DcgmPlugin(DcgmRenderer, DcgmExtractor, TestPlugin):
    pass


__chs_plugin__ = {
    "test_name": "dcgm",
    "plugin": DcgmPlugin,
}
