from typing import Any
from chs.core.plugin_base import TestPlugin


class DefaultExtractor:
    """Default logic for extracting test metadata/metrics."""

    def extract_test_info(
        self: TestPlugin, exp: dict[str, Any]
    ) -> tuple[str, str, str]:
        """Extracts test metadata with basic canonicalization."""
        name = exp.get("experiment_name", "")
        level = exp.get("workload_name", "")

        # Strip common CHS prefix if no specific plugin handled it
        if name.startswith("chs-"):
            parts = name.split("-")
            if len(parts) >= 2:
                name = parts[1]

        return name, level, self.get_description(name)
