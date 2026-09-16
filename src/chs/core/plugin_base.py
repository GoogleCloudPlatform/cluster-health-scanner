# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from chs.protos import bill_of_health_pb2 as pb
from rich.console import RenderableType


@dataclass
class ColumnDef:
    """Defines a table column's key, display label, and alignment."""

    key: str
    label: str
    align: str = "left"


@dataclass
class TableDataModel:
    """Structured table representation decoupled from UI rendering."""

    title: str
    headers: list[ColumnDef]
    rows: list[dict[str, Any]]


@dataclass
class PluginDetailReport:
    """Structured detail report containing extracted tables and metrics."""

    test_name: str
    test_level: str
    tables: list[TableDataModel] = field(default_factory=list)


class TestPlugin(ABC):
    """Plugin for visualization and data extraction lifecycle."""

    @abstractmethod
    def render_detail(
        self, test_results: list[pb.TestResult]
    ) -> RenderableType:
        """Returns a Rich renderable for detailed views across nodes."""
        pass

    def parse_context_metadata(self, ctx_name: str) -> dict[str, Any]:
        """Parses a raw context string into structured key-value pairs.

        Default implementation returns the context string under 'context'.
        Plugins override this to break context into explicit metadata fields
        (e.g., node, matrix_shape, dtype, gpu_id, msg_size_bytes).
        """
        return {"context": ctx_name}

    def get_column_schema(self) -> list[ColumnDef]:
        """Returns the list of column definitions for raw metric tables."""
        return [
            ColumnDef("key", "Key"),
            ColumnDef("value", "Value"),
            ColumnDef("unit", "Unit"),
        ]

    def get_detail_data(
        self, test_results: list[pb.TestResult]
    ) -> PluginDetailReport | None:
        """Extracts structured data model for detail views (HTML/JSON/CLI)."""
        return None

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

    def get_fom_aliases(self) -> dict[str, str]:
        """Maps clean keys to raw FOM names."""
        return {}

    def get_relative_fom_keys(self) -> list[tuple[str, str]]:
        """For relative tests, returns list of (achieved_fom_key,
        peak_fom_key)."""
        return []

    def creates_per_node_results(self) -> bool:
        """Returns True if the test generates one TestResult per node."""
        return False

    def show_context_in_validation(self) -> bool:
        """Returns True if context should be shown in validation tables."""
        return False

    def post_process_test_result(self, test_result: pb.TestResult) -> None:
        """Hook to modify the TestResult before threshold evaluation."""
        pass
