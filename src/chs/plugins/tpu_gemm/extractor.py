from typing import Any
from chs.core.default_plugin.extractor import DefaultExtractor


class TpuGemmExtractor(DefaultExtractor):
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
