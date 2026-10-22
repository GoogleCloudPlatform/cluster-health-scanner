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


class GemmExtractor(DefaultExtractor):
    def extract_test_info(self, exp: dict[str, Any]) -> tuple[str, str, str]:
        """Prevent DefaultExtractor from stripping name to 'check'."""
        workload = exp.get("workload_name", "")
        return "gemm", workload, self.get_description("gemm")

    def get_required_contexts(
        self, contexts: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Preserve all contexts for GEMM results parsing."""
        return contexts
