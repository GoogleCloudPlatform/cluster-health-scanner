from chs.core.plugin_base import TestPlugin
from .extractor import GemmExtractor
from .renderer import GemmRenderer


class GemmPlugin(GemmRenderer, GemmExtractor, TestPlugin):
    def get_fom_aliases(self) -> dict[str, str]:
        return {"performance": "Value"}


__chs_plugin__ = {
    "test_name": "gemm",
    "plugin": GemmPlugin,
}
