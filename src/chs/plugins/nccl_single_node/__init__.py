from chs.core.plugin_base import TestPlugin
from .extractor import NcclSingleNodeExtractor
from .renderer import NcclSingleNodeRenderer


class NcclSingleNodePlugin(
    NcclSingleNodeRenderer, NcclSingleNodeExtractor, TestPlugin
):
    def get_fom_aliases(self) -> dict[str, str]:
        return {"bandwidth": "Avg. Bus Bandwidth"}


__chs_plugin__ = {
    "test_name": "nccl-single-node",
    "plugin": NcclSingleNodePlugin,
}
