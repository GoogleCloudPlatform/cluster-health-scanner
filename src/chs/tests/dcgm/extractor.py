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


class DcgmExtractor(DefaultExtractor):
    def parse_context_metadata(self, ctx_name: str) -> dict[str, Any]:
        """Parses DCGM test group or sub-test context into structured fields."""
        if ctx_name.startswith("GPU"):
            return {"gpu_id": ctx_name, "context": ctx_name}
        return {"test_group": ctx_name, "context": ctx_name}

    def extract_test_info(self, exp: dict[str, Any]) -> tuple[str, str, str]:
        name = "dcgm"
        level_val = exp.get("RAMBLE_VARIABLES", {}).get("diag_level")
        level = str(level_val) if level_val is not None else ""
        return name, level, self.get_description(name)

    def get_description(self, test_name: str | None = None) -> str:
        return "NVIDIA Data Center GPU Manager Diagnostics"
