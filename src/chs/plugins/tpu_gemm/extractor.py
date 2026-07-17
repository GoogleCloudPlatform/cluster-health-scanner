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


class TpuGemmExtractor(DefaultExtractor):
    @classmethod
    def parse_context_metadata(cls, ctx_name: str) -> dict[str, Any]:
        """Parses TPU GEMM context string into structured fields."""
        parts = ctx_name.split(",")
        if len(parts) == 3:
            return {
                "node": parts[0],
                "matrix_shape": parts[1],
                "dtype": parts[2],
                "sweep_param": f"{parts[1]},{parts[2]}",
                "context": ctx_name,
            }
        return {"context": ctx_name}

    def get_required_contexts(
        self, contexts: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Preserve all contexts for TPU GEMM results parsing."""
        return contexts

    def extract_test_info(self, exp: dict[str, Any]) -> tuple[str, str, str]:
        """Extracts test name and level for TPU GEMM."""
        name = "tpu-gemm"
        level = exp.get("workload_name", "check")
        return name, level, self.get_description(name)
