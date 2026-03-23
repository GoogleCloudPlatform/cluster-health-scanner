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
        if name.startswith("cluster_health_diagnostics-"):
            parts = name.split("-")
            if len(parts) >= 2:
                name = parts[1]

        return name, level, self.get_description(name)

    def get_required_contexts(
        self, contexts: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Filter contexts to 'null' and the last recorded context.

        This maintains backward compatibility with legacy Ramble
        result structures.
        """
        if not contexts:
            return []

        extracted = []
        default_ctx = next(
            (c for c in contexts if c.get("name") == "null"), None
        )
        if default_ctx:
            extracted.append(default_ctx)

        last_ctx = contexts[-1]
        if last_ctx and last_ctx != default_ctx:
            extracted.append(last_ctx)

        return extracted
