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

from dataclasses import dataclass
from rich.console import Group
from rich.text import Text
from chs.utils import create_styled_table
from chs.protos import bill_of_health_pb2 as pb
from chs.plugins.utils.rendering import format_nodes, format_val
from chs.core.default_plugin.renderer import DefaultRenderer
from chs.core.plugin_base import ColumnDef, TableDataModel, PluginDetailReport


@dataclass
class HbmResult:
    """Holds HBM benchmark results for a specific node, size, and dtype."""

    node: str
    num_elements: int
    dtype: str
    bandwidth: float | None = None
    time_ms: float | None = None
    tensor_size: float | None = None


class TpuHbmRenderer(DefaultRenderer):
    """Renders details for tpu-hbm tests."""

    def get_column_schema(self) -> list[ColumnDef]:
        return [
            ColumnDef("num_elements", "Elements", align="right"),
            ColumnDef("tensor_size", "Tensor Size (MB)", align="right"),
            ColumnDef("dtype", "Dtype", align="left"),
            ColumnDef("bandwidth", "Bandwidth (median) (GB/s)", align="right"),
            ColumnDef("time_ms", "Time Taken (median) (ms)", align="right"),
        ]

    def get_detail_data(
        self, test_results: list[pb.TestResult]
    ) -> PluginDetailReport | None:
        """Extracts structured TPU HBM data model."""
        if not test_results:
            return None

        node_data = self._extract_node_data(test_results)
        if not node_data:
            return None

        tables = []
        for node in sorted(node_data.keys()):
            node_results = node_data[node]
            rows = []
            for r in node_results:
                rows.append(
                    {
                        "num_elements": f"{r.num_elements:,}",
                        "tensor_size": format_val(r.tensor_size, precision=1),
                        "dtype": r.dtype,
                        "bandwidth": format_val(r.bandwidth, precision=2),
                        "time_ms": format_val(r.time_ms, precision=3),
                    }
                )
            tables.append(
                TableDataModel(
                    title=f"Node: {node}",
                    headers=self.get_column_schema(),
                    rows=rows,
                )
            )

        test_name = test_results[0].test_name
        test_level = test_results[0].test_level

        return PluginDetailReport(
            test_name=test_name,
            test_level=test_level,
            tables=tables,
        )

    def render_detail(self, test_results: list[pb.TestResult]) -> Group | Text:
        """Renders detail tables for hbm tests per node."""
        report = self.get_detail_data(test_results)
        if not report or not report.tables:
            return Text("No detailed metrics available.", style="yellow")

        elements = [Text("\nTPU HBM Bandwidth Detail", style="bold cyan")]

        for table_data in report.tables:
            table = create_styled_table(expand=False)
            table.add_column("Elements", justify="right")
            table.add_column("Tensor Size (MB)", justify="right")
            table.add_column("Dtype", justify="left")
            table.add_column("Bandwidth (median) (GB/s)", justify="right")
            table.add_column("Time Taken (median) (ms)", justify="right")

            for r in table_data.rows:
                table.add_row(
                    r["num_elements"],
                    r["tensor_size"],
                    r["dtype"],
                    r["bandwidth"],
                    r["time_ms"],
                )

            elements.append(Text(f"\n{table_data.title}", style="bold yellow"))
            elements.append(table)

        val_table = self.render_validation_results(test_results)
        if val_table:
            elements.append(val_table)

        return Group(*elements)

    def _parse_context(
        self, ctx_name: str
    ) -> tuple[str | None, int, str] | None:
        """Parses node, num_elements, and dtype from context name."""
        parts = ctx_name.split(",")
        if len(parts) == 3:
            try:
                return parts[0], int(parts[1]), parts[2]
            except ValueError:
                pass
        return None

    def _extract_node_data(
        self, test_results: list[pb.TestResult]
    ) -> dict[str, list[HbmResult]]:
        """Extracts metrics from test results and groups them by node."""
        aggregated: dict[tuple[str, int, str], HbmResult] = {}

        for res in test_results:
            default_node = (
                format_nodes(res.target_resources)
                if res.target_resources
                else "unknown"
            )
            for m in res.metrics:
                if not m.value.HasField("number_value"):
                    continue

                ctx_field = m.metadata.fields.get("context")
                ctx_name = ctx_field.string_value if ctx_field else ""

                parsed_ctx = self._parse_context(ctx_name)
                if not parsed_ctx:
                    continue

                node, num_elements, dtype = parsed_ctx
                if not node:
                    node = default_node

                key = (node, num_elements, dtype)

                if key not in aggregated:
                    aggregated[key] = HbmResult(
                        node=node, num_elements=num_elements, dtype=dtype
                    )

                val = m.value.number_value
                if m.key == "Bandwidth (median)":
                    aggregated[key].bandwidth = val
                elif m.key == "Time Taken (median)":
                    aggregated[key].time_ms = val
                elif m.key == "Tensor Size":
                    aggregated[key].tensor_size = val

        node_data: dict[str, list[HbmResult]] = {}
        for r in sorted(
            aggregated.values(), key=lambda x: (x.node, x.num_elements)
        ):
            node_data.setdefault(r.node, []).append(r)

        return node_data
