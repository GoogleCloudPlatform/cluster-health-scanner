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
from chs.core.default_plugin.renderer import DefaultRenderer
from chs.core.plugin_base import ColumnDef, PluginDetailReport, TableDataModel
from chs.core.utils.rendering import format_nodes, format_val
from chs.protos import bill_of_health_pb2 as pb


@dataclass
class HostDeviceResult:
    """Holds bandwidth results for a specific node and transfer size."""

    node: str
    size_mib: int
    h2d: float | None = None
    d2h: float | None = None


class TpuHostDeviceRenderer(DefaultRenderer):
    """Renders details for tpu-host-device tests."""

    def get_column_schema(self) -> list[ColumnDef]:
        return [
            ColumnDef("node", "Node", align="left"),
            ColumnDef("size_mib", "Size (MiB)", align="right"),
            ColumnDef("h2d", "H2D Bandwidth (GB/s)", align="right"),
            ColumnDef("d2h", "D2H Bandwidth (GB/s)", align="right"),
        ]

    def get_detail_data(
        self, test_results: list[pb.TestResult]
    ) -> PluginDetailReport | None:
        """Extracts structured host-device detail table data model."""
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
                h2d_str = format_val(r.h2d)
                d2h_str = format_val(r.d2h)
                rows.append(
                    {
                        "node": node,
                        "size_mib": str(r.size_mib),
                        "h2d": h2d_str,
                        "d2h": d2h_str,
                    }
                )
            tables.append(
                TableDataModel(
                    title=f"Host-Device I/O Bandwidth Detail ({node})",
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
        """Renders detail tables for host-device tests per node and size."""
        report = self.get_detail_data(test_results)
        if not report or not report.tables:
            return Text("No detailed metrics available.", style="yellow")

        elements = [
            Text("\nHost-Device I/O Bandwidth Detail", style="bold cyan")
        ]
        for table_model in report.tables:
            elements.append(Text(f"\n{table_model.title}", style="bold yellow"))
            table = create_styled_table(expand=False)
            table.add_column("Size (MiB)", justify="right")
            table.add_column("H2D Bandwidth (GB/s)", justify="right")
            table.add_column("D2H Bandwidth (GB/s)", justify="right")

            for r in table_model.rows:
                table.add_row(r["size_mib"], r["h2d"], r["d2h"])
            elements.append(table)

        val_table = self.render_validation_results(test_results)
        if val_table:
            elements.append(val_table)

        return Group(*elements)

    def _parse_context(self, ctx_name: str) -> tuple[str | None, int] | None:
        """Parses node and size from context name."""
        parts = ctx_name.split(",")
        if len(parts) == 2:
            node = parts[0]
            size_part = parts[1]
        else:
            return None

        if size_part.endswith("MiB"):
            try:
                return node, int(size_part[:-3])
            except ValueError:
                pass
        return None

    def _extract_node_data(
        self, test_results: list[pb.TestResult]
    ) -> dict[str, list[HostDeviceResult]]:
        """Extracts metrics from test results and groups them by node."""
        aggregated: dict[tuple[str, int], HostDeviceResult] = {}

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
                if parsed_ctx is None:
                    continue

                node, size = parsed_ctx
                if not node:
                    node = default_node

                key = (node, size)

                if key not in aggregated:
                    aggregated[key] = HostDeviceResult(node=node, size_mib=size)

                val = m.value.number_value
                if m.key == "H2D Bandwidth":
                    aggregated[key].h2d = val
                elif m.key == "D2H Bandwidth":
                    aggregated[key].d2h = val

        node_data: dict[str, list[HostDeviceResult]] = {}
        for r in sorted(
            aggregated.values(), key=lambda x: (x.node, x.size_mib)
        ):
            node_data.setdefault(r.node, []).append(r)

        return node_data
