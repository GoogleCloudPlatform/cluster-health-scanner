from typing import Any
from chs.core.plugin_base import TestPlugin


class DefaultExtractor:
    """Default logic for extracting test metadata/metrics."""

    def extract_test_info(
        self: TestPlugin, exp: dict[str, Any]
    ) -> tuple[str, str, str]:
        level = exp.get("workload_name", "")
        name = exp.get("RAMBLE_VARIABLES", {}).get("chs_test_name")
        if not name and hasattr(self, "test_name") and self.test_name:
            name = self.test_name
        return name, level, self.get_description(name)
