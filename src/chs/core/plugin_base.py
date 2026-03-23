from abc import ABC, abstractmethod
from typing import Any
from chs.protos import bill_of_health_pb2 as pb
from rich.console import RenderableType


class TestPlugin(ABC):
    """Plugin for visualization and data extraction lifecycle."""

    @abstractmethod
    def render_summary(
        self, test_results: list[pb.TestResult]
    ) -> RenderableType:
        """Returns a Renderable for summary views aggregating across nodes."""
        pass

    @abstractmethod
    def render_detail(
        self, test_results: list[pb.TestResult]
    ) -> RenderableType:
        """Returns a Rich renderable for detailed views across nodes."""
        pass

    def get_description(self, test_name: str | None = None) -> str:
        """Returns a human-readable description of the test."""
        return ""

    def extract_test_info(self, exp: dict[str, Any]) -> tuple[str, str, str]:
        """Extracts test metadata from raw Ramble experiment data.

        Returns:
            tuple: (test_name, test_level, description)
        """
        name = exp.get("experiment_name", "")
        level = exp.get("workload_name", "")
        return name, level, self.get_description(name)

    def get_required_contexts(
        self, contexts: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Filter which Ramble contexts to extract metrics from.

        Defaults to returning all contexts for maximum data extraction.
        """
        return contexts
