from chs.core.plugin_base import TestPlugin
from .extractor import NcclPairwiseExtractor
from .renderer import NcclPairwiseRenderer


class NcclPairwisePlugin(
    NcclPairwiseRenderer, NcclPairwiseExtractor, TestPlugin
):
    pass


__chs_plugin__ = {
    "test_name": "nccl-pairwise",
    "plugin": NcclPairwisePlugin,
}
