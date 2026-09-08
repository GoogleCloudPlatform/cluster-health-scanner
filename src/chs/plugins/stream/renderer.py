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

import re
from dataclasses import dataclass

from chs.core.default_plugin.renderer import DefaultRenderer
from chs.core.plugin_base import ColumnDef, PluginDetailReport, TableDataModel
from chs.plugins.utils.rendering import format_nodes, format_val
from chs.protos import bill_of_health_pb2 as pb
from chs.utils import create_styled_table
from rich.console import Group
from rich.text import Text


@dataclass
class StreamParsedResult:
    node: str
    device_id: str
    copy_rate: float | None = None
    scale_rate: float | None = None
    add_rate: float | None = None
    triad_rate: float | None = None


class StreamRenderer(DefaultRenderer):
    """Renders details for stream tests."""

    validation_render_mode = "all"

    def get_column_schema(self) -> list[ColumnDef]:
        return [
            ColumnDef("node", "Node"),
            ColumnDef("device_id", "Device"),
            ColumnDef("copy_rate", "Copy (GB/s)", align="right"),
            ColumnDef("scale_rate", "Scale (GB/s)", align="right"),
            ColumnDef("add_rate", "Add (GB/s)", align="right"),
            ColumnDef("triad_rate", "Triad (GB/s)", align="right"),
        ]

    def get_detail_data(
        self, test_results: list[pb.TestResult]
    ) -> PluginDetailReport | None:
        """Extracts structured STREAM detail data model."""
        if not test_results:
            return None

        results_map, node_peak_bw = self._parse_metrics(test_results)
        if not results_map:
            return None

        node_groups: dict[str, list[StreamParsedResult]] = {}
        for key in sorted(results_map.keys()):
            item = results_map[key]
            if item.node not in node_groups:
                node_groups[item.node] = []
            node_groups[item.node].append(item)

        tables = []
        for node, items in node_groups.items():
            peak = node_peak_bw.get(node)
            peak_str = (
                f" (Peak BW: {peak:.2f} GB/s)" if peak is not None else ""
            )
            rows = []
            for item in items:
                rows.append(
                    {
                        "node": node,
                        "device_id": str(item.device_id),
                        "copy_rate": format_val(item.copy_rate),
                        "scale_rate": format_val(item.scale_rate),
                        "add_rate": format_val(item.add_rate),
                        "triad_rate": format_val(item.triad_rate),
                    }
                )

            tables.append(
                TableDataModel(
                    title=f"STREAM Performance Detail ({node}){peak_str}",
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

    def _parse_metrics(
        self, test_results: list[pb.TestResult]
    ) -> tuple[dict[tuple[str, int], StreamParsedResult], dict[str, float]]:
        """Parses raw test results and extracts peak BW."""

        results_map: dict[tuple[str, int], StreamParsedResult] = {}

        node_peak_bw: dict[str, float] = {}

        # Regex to parse context like "Device 0 Copy" or "Device 0"
        context_regex = re.compile(
            r"Device\s+(?P<device_id>[0-9]+)(?:\s+(?P<function>\w+))?"
        )

        func_attr_map = {
            "Copy": "copy_rate",
            "Scale": "scale_rate",
            "Add": "add_rate",
            "Triad": "triad_rate",
        }

        for res in test_results:
            node = (
                format_nodes(res.target_resources)
                if res.target_resources
                else "Unknown"
            )

            for m in res.metrics:
                ctx_str = (
                    m.metadata.fields["context"].string_value
                    if "context" in m.metadata.fields
                    else ""
                )

                val = (
                    m.value.number_value
                    if m.value.HasField("number_value")
                    else None
                )

                if val is None:
                    continue

                if (
                    not ctx_str or ctx_str == "null"
                ) and m.key == "Peak Bandwidth":
                    node_peak_bw[node] = val
                    continue

                match = context_regex.search(ctx_str)
                if not match:
                    continue

                device_id = match.group("device_id")
                function = match.group("function")

                key = (node, int(device_id))

                if key not in results_map:
                    results_map[key] = StreamParsedResult(
                        node=node, device_id=device_id
                    )

                # Convert rates from MB/s to GB/s if not already converted
                if m.key == "Achieved Bandwidth":
                    attr_name = func_attr_map.get(function)
                    if attr_name:
                        rate_val = val if m.unit == "GB/s" else val / 1000.0
                        setattr(results_map[key], attr_name, rate_val)

        return results_map, node_peak_bw

    def render_detail(self, test_results: list[pb.TestResult]) -> Group | Text:
        """Renders detail tables for stream tests per node and device."""
        report = self.get_detail_data(test_results)
        if not report or not report.tables:
            return Text("No detailed metrics available.", style="yellow")

        group_elements = [Text("\nSTREAM Detail", style="bold")]

        for table_model in report.tables:
            group_elements.append(Text(f"\n{table_model.title}", style="cyan"))

            table = create_styled_table(expand=False)
            table.add_column("Device")
            table.add_column("Copy (GB/s)", justify="right")
            table.add_column("Scale (GB/s)", justify="right")
            table.add_column("Add (GB/s)", justify="right")
            table.add_column("Triad (GB/s)", justify="right")

            for r in table_model.rows:
                table.add_row(
                    r["device_id"],
                    r["copy_rate"],
                    r["scale_rate"],
                    r["add_rate"],
                    r["triad_rate"],
                )

            group_elements.append(table)

        val_table = self.render_validation_results(test_results)
        if val_table:
            group_elements.append(val_table)

        return Group(*group_elements)
