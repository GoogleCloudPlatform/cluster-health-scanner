from typing import Any
from chs.core.default_plugin.extractor import DefaultExtractor


class NcclBaseExtractor(DefaultExtractor):
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

        return extracted
