from typing import Any
from chs.core.default_plugin.extractor import DefaultExtractor


class NvidiaSmiExtractor(DefaultExtractor):
    def extract_test_info(self, exp: dict[str, Any]) -> tuple[str, str, str]:
        """Prevent DefaultExtractor from stripping name to 'nvidia'."""
        workload = exp.get("workload_name", "")
        return "nvidia-smi", workload, self.get_description("nvidia-smi")

    def get_required_contexts(
        self, contexts: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Preserve all contexts for NVIDIA SMI results parsing."""
        return contexts
