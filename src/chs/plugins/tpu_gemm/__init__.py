from chs.core.plugin_base import TestPlugin
from .extractor import TpuGemmExtractor
from .renderer import TpuGemmRenderer


class TpuGemmPlugin(TpuGemmRenderer, TpuGemmExtractor, TestPlugin):
    pass


__chs_plugin__ = {
    "test_name": "tpu-gemm",
    "plugin": TpuGemmPlugin,
}
