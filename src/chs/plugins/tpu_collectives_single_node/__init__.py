from chs.core.plugin_base import TestPlugin
from .extractor import TpuCollectivesSingleNodeExtractor
from .renderer import TpuCollectivesSingleNodeRenderer


class TpuCollectivesSingleNodePlugin(
    TpuCollectivesSingleNodeRenderer,
    TpuCollectivesSingleNodeExtractor,
    TestPlugin,
):
    pass


__chs_plugin__ = {
    "test_name": "tpu-collectives-single-node",
    "plugin": TpuCollectivesSingleNodePlugin,
}
