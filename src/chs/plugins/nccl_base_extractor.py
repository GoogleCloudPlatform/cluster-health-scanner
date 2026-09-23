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
from chs.plugins.utils.nccl_utils import parse_nccl_context_metadata


class NcclBaseExtractor(DefaultExtractor):
    def get_fom_aliases(self) -> dict[str, str]:
        return {"bandwidth": "Peak Detected Bandwidth"}

    def parse_context_metadata(self, ctx_name: str) -> dict[str, Any]:
        return parse_nccl_context_metadata(ctx_name)

    def get_required_contexts(
        self, contexts: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Filter contexts to 'null' and the last recorded context.

        This maintains backward compatibility with legacy Ramble
        result structures for NCCL tests.
        """
        if not contexts:
            return []

        extracted = []
        default_ctx = next(
            (c for c in contexts if c.get("name") == "null"), None
        )
        if default_ctx is not None:
            extracted.append(default_ctx)
        last_ctx = contexts[-1]
        if last_ctx != default_ctx:
            extracted.append(last_ctx)

        # We only want to inject Peak Detected Bandwidth into the last context
        # (which corresponds to the maximum message size) to avoid false
        # negatives from validating the null context against peak thresholds.

        for ctx in extracted:
            # We skip injecting into 'null' context unless it's the only one.
            if len(extracted) > 1 and ctx.get("name") == "null":
                continue

            foms = ctx.get("foms", [])
            max_bw = 0.0
            for fom in foms:
                name = fom.get("name", "")
                val = fom.get("value")
                if "Bus Bandwidth" in name:
                    try:
                        max_bw = max(max_bw, float(val))
                    except (ValueError, TypeError):
                        pass

            if max_bw > 0:
                # Add synthetic metric so thresholds can validate against it
                foms.append(
                    {
                        "name": "Peak Detected Bandwidth",
                        "value": max_bw,
                        "units": "GB/s",
                    }
                )

        return extracted
