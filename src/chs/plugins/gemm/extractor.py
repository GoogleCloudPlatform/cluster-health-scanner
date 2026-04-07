from typing import Any
from chs.core.default_plugin.extractor import DefaultExtractor


class GemmExtractor(DefaultExtractor):
    def get_required_contexts(
        self, contexts: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Preserve all contexts for GEMM results parsing."""
        return contexts
