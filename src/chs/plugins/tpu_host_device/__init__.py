from chs.core.plugin_base import TestPlugin
from .extractor import TpuHostDeviceExtractor
from .renderer import TpuHostDeviceRenderer


class TpuHostDevicePlugin(
    TpuHostDeviceRenderer, TpuHostDeviceExtractor, TestPlugin
):
    pass


__chs_plugin__ = {
    "test_name": "tpu-host-device",
    "plugin": TpuHostDevicePlugin,
}
