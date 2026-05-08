from chs.core.plugin_base import TestPlugin
from .extractor import NvidiaSmiExtractor
from .renderer import NvidiaSmiRenderer


class NvidiaSmiPlugin(NvidiaSmiRenderer, NvidiaSmiExtractor, TestPlugin):
    def get_fom_aliases(self) -> dict[str, str]:
        return {"gpus": "GPU Count"}


__chs_plugin__ = {
    "test_name": "nvidia-smi",
    "plugin": NvidiaSmiPlugin,
}
