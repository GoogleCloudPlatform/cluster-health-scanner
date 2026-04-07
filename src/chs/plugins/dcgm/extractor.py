from typing import Any
from chs.core.default_plugin.extractor import DefaultExtractor


class DcgmExtractor(DefaultExtractor):
    def extract_test_info(self, exp: dict[str, Any]) -> tuple[str, str, str]:
        name = "dcgm"
        level = exp.get("RAMBLE_VARIABLES", {}).get("diag_level")
        return name, level, self.get_description(name)

    def get_description(self, test_name: str | None = None) -> str:
        return "NVIDIA Data Center GPU Manager Diagnostics"
