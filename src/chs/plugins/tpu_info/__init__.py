from chs.core.plugin_base import TestPlugin
from .extractor import TpuInfoExtractor
from .renderer import TpuInfoRenderer


class TpuInfoPlugin(TpuInfoRenderer, TpuInfoExtractor, TestPlugin):
    pass


__chs_plugin__ = {
    "test_name": "tpu-info",
    "plugin": TpuInfoPlugin,
}
