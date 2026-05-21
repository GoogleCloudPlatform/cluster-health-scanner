from typing import Any
from chs.core.default_plugin.extractor import DefaultExtractor


class TpuInfoExtractor(DefaultExtractor):
    """Overrides DefaultExtractor to preserve test naming and contexts."""

    def extract_test_info(self, exp: dict[str, Any]) -> tuple[str, str, str]:
        return "tpu-info", exp.get("workload_name", ""), self.get_description()

    def get_description(self, test_name: str | None = None) -> str:
        return "Cloud TPU Information and Diagnostics"

    def get_required_contexts(
        self, contexts: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return contexts
