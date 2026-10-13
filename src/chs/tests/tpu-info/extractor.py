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

import re
from typing import Any
from chs.core.default_plugin.extractor import DefaultExtractor


class TpuInfoExtractor(DefaultExtractor):
    """Overrides DefaultExtractor to preserve test naming and contexts."""

    def parse_context_metadata(self, ctx_name: str) -> dict[str, Any]:
        """Parses TPU Info context string into structured fields."""
        node_match = re.search(r"Node ([a-zA-Z0-9-]+)", ctx_name)
        if node_match:
            return {"node": node_match.group(1), "context": ctx_name}
        return {"context": ctx_name}

    def extract_test_info(self, exp: dict[str, Any]) -> tuple[str, str, str]:
        return "tpu-info", exp.get("workload_name", ""), self.get_description()

    def get_description(self, test_name: str | None = None) -> str:
        return "Cloud TPU Information and Diagnostics"

    def get_required_contexts(
        self, contexts: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return contexts
