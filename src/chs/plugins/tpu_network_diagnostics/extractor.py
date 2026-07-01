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

from typing import Any
from chs.core.default_plugin.extractor import DefaultExtractor


class TpuNetworkDiagnosticsExtractor(DefaultExtractor):
    """Overrides DefaultExtractor to preserve test naming and contexts."""

    def extract_test_info(self, exp: dict[str, Any]) -> tuple[str, str, str]:
        return (
            "tpu-network-diagnostics",
            exp.get("workload_name", ""),
            self.get_description(),
        )

    def get_required_contexts(
        self, contexts: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return contexts

    def get_fom_aliases(self) -> dict[str, str]:
        return {
            "bandwidth": "Bandwidth",
        }
