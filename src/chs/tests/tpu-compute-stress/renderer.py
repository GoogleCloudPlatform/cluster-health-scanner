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


class TpuComputeStressRenderer(DefaultRenderer):
    def get_column_schema(self) -> list[ColumnDef]:
        return [
            ColumnDef("node", "Node Name"),
            ColumnDef("performance", "Performance (TFLOPS)", align="right"),
        ]

    def get_detail_data(
        self, test_results: list[pb.TestResult]
    ) -> PluginDetailReport | None:
        if not test_results:
            return None

        node_metrics = self._extract_node_data(test_results)
        if not node_metrics:
            return None

        rows = []
        for node_name in sorted(node_metrics.keys()):
            perf_val = node_metrics[node_name]
            formatted_perf = format_val(perf_val)
            rows.append(
                {
                    "node": node_name,
                    "performance": formatted_perf,
                }
            )

        table_model = TableDataModel(
            title="TPU Compute Stress Detail",
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
        report = self.get_detail_data(test_results)
        if not report or not report.tables:
            return Text("No detailed metrics available.", style="yellow")

        table_model = report.tables[0]
        elements = [Text(f"\n{table_model.title}", style="bold cyan")]

        table = create_styled_table(expand=False)
        for col in table_model.headers:
            table.add_column(col.label, justify=col.align)

        for r in table_model.rows:
            table.add_row(r["node"], r["performance"])

        elements.append(table)
        val_table = self.render_validation_results(test_results)
        if val_table:
            elements.append(val_table)

        return Group(*elements)

    def _extract_node_data(
        self, test_results: list[pb.TestResult]
    ) -> dict[str, float]:
        node_metrics: dict[str, float] = {}

        for test_result in test_results:
            default_node = (
                format_nodes(test_result.target_resources)
                if test_result.target_resources
                else "unknown"
            )
            for metric in test_result.metrics:
                if not metric.value.HasField("number_value"):
                    continue

                if metric.key != "Performance":
                    continue

                context_field = metric.metadata.fields.get("context")
                node_name = (
                    context_field.string_value
                    if context_field
                    else default_node
                )

                perf_val = metric.value.number_value
                node_metrics[node_name] = perf_val

        return node_metrics
