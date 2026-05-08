from chs.core.plugin_base import TestPlugin
from .extractor import TpuHbmExtractor
from .renderer import TpuHbmRenderer


class TpuHbmPlugin(TpuHbmRenderer, TpuHbmExtractor, TestPlugin):
    pass


__chs_plugin__ = {
    "test_name": "tpu-hbm",
    "plugin": TpuHbmPlugin,
}
