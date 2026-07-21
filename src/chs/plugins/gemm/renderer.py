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

from chs.core.default_plugin.renderer import DefaultRenderer
from chs.core.plugin_base import ColumnDef, PluginDetailReport, TableDataModel
from chs.plugins.utils.rendering import format_nodes, format_val
from chs.protos import bill_of_health_pb2 as pb
from chs.utils import create_styled_table


def _to_tflops(val: float | None) -> float | None:
    if val is None:
        return None
    return val / 1000.0 if val >= 0 else val


class GemmRenderer(DefaultRenderer):
    """Renders details for gemm tests."""

    def get_column_schema(self) -> list[ColumnDef]:
        return [
            ColumnDef("workload", "Workload"),
            ColumnDef("node", "Node"),
            ColumnDef("min_tflops", "Min (TFLOPS)", align="right"),
            ColumnDef("median_tflops", "Median (TFLOPS)", align="right"),
            ColumnDef("max_tflops", "Max (TFLOPS)", align="right"),
        ]

    def get_detail_data(
        self, test_results: list[pb.TestResult]
    ) -> PluginDetailReport | None:
        """Extracts structured GEMM detail data model."""
        if not test_results:
            return None

        summary_rows = []
        for res in test_results:
            min_val = None
            median_val = None
            max_val = None
            for m in res.metrics:
                if m.value.HasField("number_value") and m.key == "Value":
                    ctx = (
                        m.metadata.fields["context"].string_value
                        if "context" in m.metadata.fields
                        else ""
                    )
                    if ctx == "MAX":
                        max_val = m.value.number_value
                    elif ctx == "MIN":
                        min_val = m.value.number_value
                    elif ctx == "MEDIAN":
                        median_val = m.value.number_value

            nodes = (
                format_nodes(res.target_resources)
                if res.target_resources
                else "Unknown Node"
            )
            test_level = res.test_level if res.test_level else "Unknown"

            min_tflops, median_tflops, max_tflops = map(
                _to_tflops, [min_val, median_val, max_val]
            )
            min_str, median_str, max_str = map(
                format_val, [min_tflops, median_tflops, max_tflops]
            )

            summary_rows.append(
                {
                    "workload": test_level,
                    "node": nodes,
                    "min_tflops": min_str,
                    "median_tflops": median_str,
                    "max_tflops": max_str,
                }
            )

        table_model = TableDataModel(
            title="GEMM Detail",
            headers=self.get_column_schema(),
            rows=summary_rows,
        )

        test_name = test_results[0].test_name
        test_level = test_results[0].test_level

        return PluginDetailReport(
            test_name=test_name,
            test_level=test_level,
            tables=[table_model],
        )

    def render_detail(self, test_results: list[pb.TestResult]) -> Group | Text:
        """Renders detail tables for gemm tests per node."""
        report = self.get_detail_data(test_results)
        if not report or not report.tables:
            return Text("No detailed metrics available.", style="yellow")

        table_model = report.tables[0]
        table = create_styled_table(expand=False)
        table.add_column("Workload")
        table.add_column("Node", overflow="fold")
        table.add_column("Min (TFLOPS)", justify="right")
        table.add_column("Median (TFLOPS)", justify="right")
        table.add_column("Max (TFLOPS)", justify="right")

        for r in table_model.rows:
            table.add_row(
                r["workload"],
                r["node"],
                r["min_tflops"],
                r["median_tflops"],
                r["max_tflops"],
            )

        elements = [
            Text(f"\n{table_model.title}", style="bold"),
            table,
        ]
        val_table = self.render_validation_results(test_results)
        if val_table:
            elements.append(val_table)

        return Group(*elements)
