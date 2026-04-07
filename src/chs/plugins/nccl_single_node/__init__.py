from chs.core.plugin_base import TestPlugin
from .extractor import NcclSingleNodeExtractor
from .renderer import NcclSingleNodeRenderer


class NcclSingleNodePlugin(
    NcclSingleNodeRenderer, NcclSingleNodeExtractor, TestPlugin
):
    pass


__chs_plugin__ = {
    "test_name": "nccl-single-node",
    "plugin": NcclSingleNodePlugin,
}
