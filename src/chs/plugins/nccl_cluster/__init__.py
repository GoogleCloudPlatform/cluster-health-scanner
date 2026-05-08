from chs.core.plugin_base import TestPlugin
from .extractor import NcclClusterExtractor
from .renderer import NcclClusterRenderer


class NcclClusterPlugin(NcclClusterRenderer, NcclClusterExtractor, TestPlugin):
    def get_fom_aliases(self) -> dict[str, str]:
        return {"bandwidth": "Avg. Bus Bandwidth"}


__chs_plugin__ = {
    "test_name": "nccl-cluster",
    "plugin": NcclClusterPlugin,
}
