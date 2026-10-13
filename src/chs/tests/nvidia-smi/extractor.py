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


class NvidiaSmiExtractor(DefaultExtractor):
    def parse_context_metadata(self, ctx_name: str) -> dict[str, Any]:
        """Parses NVIDIA SMI context (e.g. GPU 0) into metadata."""
        if ctx_name.startswith("GPU "):
            gpu_idx = ctx_name.replace("GPU ", "")
            return {
                "gpu_id": ctx_name,
                "gpu_index": gpu_idx,
                "context": ctx_name,
            }
        return {"context": ctx_name}

    def extract_test_info(self, exp: dict[str, Any]) -> tuple[str, str, str]:
        """Prevent DefaultExtractor from stripping name to 'nvidia'."""
        workload = exp.get("workload_name", "")
        return "nvidia-smi", workload, self.get_description("nvidia-smi")

    def get_required_contexts(
        self, contexts: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Preserve all contexts for NVIDIA SMI results parsing."""
        return contexts
