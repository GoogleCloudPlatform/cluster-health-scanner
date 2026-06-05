from chs.core.plugin_base import TestPlugin
from .extractor import TpuCollectivesExtractor
from .renderer import TpuCollectivesRenderer


class TpuCollectivesPlugin(
    TpuCollectivesRenderer, TpuCollectivesExtractor, TestPlugin
):
    pass


__chs_plugin__ = {
    "test_name": "tpu-collectives",
    "plugin": TpuCollectivesPlugin,
}
