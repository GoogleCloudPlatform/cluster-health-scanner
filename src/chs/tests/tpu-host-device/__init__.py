# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from chs.core.plugin_base import TestPlugin
from .extractor import TpuHostDeviceExtractor
from .renderer import TpuHostDeviceRenderer


class TpuHostDevicePlugin(
    TpuHostDeviceRenderer, TpuHostDeviceExtractor, TestPlugin
):
    def get_fom_aliases(self) -> dict[str, str]:
        return {
            "h2d_bandwidth": "H2D Bandwidth",
            "d2h_bandwidth": "D2H Bandwidth",
        }

    def creates_per_node_results(self) -> bool:
        return True


__chs_plugin__ = {
    "test_name": "tpu-host-device",
    "plugin": TpuHostDevicePlugin,
}
