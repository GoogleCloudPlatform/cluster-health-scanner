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

from rich.console import Group
from rich.text import Text
from chs.utils import create_styled_table
from chs.core.default_plugin.renderer import DefaultRenderer
from chs.core.plugin_base import ColumnDef, PluginDetailReport, TableDataModel
from chs.core.utils.rendering import format_nodes, format_val
from chs.protos import bill_of_health_pb2 as pb


class TpuNetworkStressRenderer(DefaultRenderer):
    """Renders details for tpu-network-stress tests."""

    def get_column_schema(self) -> list[ColumnDef]:
        return [
            ColumnDef("node", "Node Name"),
            ColumnDef("bandwidth", "Bandwidth (GB/s)", align="right"),
        ]

    def get_detail_data(
        self, test_results: list[pb.TestResult]
    ) -> PluginDetailReport | None:
        """Extracts structured network stress detail table data model."""
        if not test_results:
            return None

        node_bandwidths = self._extract_node_data(test_results)
        if not node_bandwidths:
            return None

        rows = []
        for node_name in sorted(node_bandwidths.keys()):
            bandwidth_val = node_bandwidths[node_name]
            formatted_bandwidth = format_val(bandwidth_val)
            rows.append(
                {
                    "node": node_name,
                    "bandwidth": formatted_bandwidth,
                }
            )

        table_model = TableDataModel(
            title="TPU Network Stress Detail",
            headers=self.get_column_schema(),
            rows=rows,
        )

        test_name = test_results[0].test_name
        test_level = test_results[0].test_level

        return PluginDetailReport(
            test_name=test_name,
            test_level=test_level,
            tables=[table_model],
        )

    def render_detail(self, test_results: list[pb.TestResult]) -> Group | Text:
        """Renders detail tables for network stress tests per node."""
        report = self.get_detail_data(test_results)
        if not report or not report.tables:
            return Text("No detailed metrics available.", style="yellow")

        table_model = report.tables[0]
        elements = [Text(f"\n{table_model.title}", style="bold cyan")]

        table = create_styled_table(expand=False)
        table.add_column("Node Name", justify="left")
        table.add_column("Bandwidth (GB/s)", justify="right")

        for r in table_model.rows:
            table.add_row(r["node"], r["bandwidth"])

        elements.append(table)

        val_table = self.render_validation_results(test_results)
        if val_table:
            elements.append(val_table)

        return Group(*elements)

    def _extract_node_data(
        self, test_results: list[pb.TestResult]
    ) -> dict[str, float]:
        """Extracts node name to minimum bandwidth mapping from test results."""
        node_bandwidths: dict[str, float] = {}

        for test_result in test_results:
            default_node = (
                format_nodes(test_result.target_resources)
                if test_result.target_resources
                else "unknown"
            )
            for metric in test_result.metrics:
                if not metric.value.HasField("number_value"):
                    continue

                if metric.key.lower() not in ("bandwidth", "axis bandwidth"):
                    continue

                context_field = metric.metadata.fields.get("context")
                context_name = (
                    context_field.string_value
                    if context_field
                    else default_node
                )
                node_name = context_name.split(" Axis-")[0].strip()
                bandwidth_val = metric.value.number_value

                if (
                    node_name not in node_bandwidths
                    or bandwidth_val < node_bandwidths[node_name]
                ):
                    node_bandwidths[node_name] = bandwidth_val

        return node_bandwidths
