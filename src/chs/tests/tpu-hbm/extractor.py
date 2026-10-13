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


class TpuHbmExtractor(DefaultExtractor):
    """Overrides DefaultExtractor to preserve test naming and contexts."""

    def parse_context_metadata(self, ctx_name: str) -> dict[str, Any]:
        """Parses TPU HBM context string into structured fields."""
        parts = ctx_name.split(",")
        if len(parts) == 3:
            try:
                return {
                    "node": parts[0],
                    "num_elements": int(parts[1]),
                    "dtype": parts[2],
                    "context": ctx_name,
                }
            except ValueError:
                pass
        return {"context": ctx_name}

    def extract_test_info(self, exp: dict[str, Any]) -> tuple[str, str, str]:
        return "tpu-hbm", exp.get("workload_name", ""), self.get_description()

    def get_required_contexts(
        self, contexts: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return contexts
