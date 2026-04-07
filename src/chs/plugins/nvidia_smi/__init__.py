from chs.core.plugin_base import TestPlugin
from .extractor import NvidiaSmiExtractor
from .renderer import NvidiaSmiRenderer


class NvidiaSmiPlugin(NvidiaSmiRenderer, NvidiaSmiExtractor, TestPlugin):
    pass


__chs_plugin__ = {
    "test_name": "nvidia-smi",
    "plugin": NvidiaSmiPlugin,
}
