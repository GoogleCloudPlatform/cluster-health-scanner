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

from dataclasses import dataclass
from chs.protos import bill_of_health_pb2 as pb
from chs.core.default_plugin.renderer import DefaultRenderer
from chs.core.plugin_base import ColumnDef, TableDataModel, PluginDetailReport
from chs.plugins.tpu_gemm.extractor import TpuGemmExtractor


@dataclass
class TpuGemmParsedResult:
    test_level: str
    nodes: str
    sweep_param: str
    throughput: float | None = None
    total_throughput: float | None = None
    mfu: float | None = None
    step_time: float | None = None
    total_flops: float | None = None


class TpuGemmRenderer(DefaultRenderer):
    """Renders details for tpu-gemm tests."""

    def get_column_schema(self) -> list[ColumnDef]:
        return [
            ColumnDef("sweep_param", "Parameters (MxNxK,Dtype)"),
            ColumnDef("nodes", "Nodes"),
            ColumnDef("throughput", "Throughput\n(TFLOP/s/dev)", align="right"),
            ColumnDef("total_throughput", "Total\n(TFLOPs/s)", align="right"),
            ColumnDef("mfu", "MFU (%)", align="right"),
            ColumnDef("step_time", "Step Time (s)", align="right"),
            ColumnDef("total_flops", "Total FLOPs", align="right"),
        ]

    def get_detail_data(
        self, test_results: list[pb.TestResult]
    ) -> PluginDetailReport | None:
        """Extracts structured TPU GEMM data model."""
        if not test_results:
            return None

        parsed_map: dict[str, list[TpuGemmParsedResult]] = {}

        for res in test_results:
            test_level = res.test_level if res.test_level else "check"

            for m in res.metrics:
                node = (
                    m.metadata.fields["node"].string_value
                    if "node" in m.metadata.fields
                    else None
                )
                sweep_param = (
                    m.metadata.fields["sweep_param"].string_value
                    if "sweep_param" in m.metadata.fields
                    else None
                )

                if not sweep_param or not node:
                    ctx = (
                        m.metadata.fields["context"].string_value
                        if "context" in m.metadata.fields
                        else "unknown"
                    )
                    if ctx != "unknown" and ctx != "null":
                        parsed_meta = TpuGemmExtractor.parse_context_metadata(
                            ctx
                        )
                        if not sweep_param:
                            sweep_param = parsed_meta.get("sweep_param", ctx)
                        if not node:
                            node = parsed_meta.get("node", "N/A")

                resolved_node = node if node else "N/A"
                if not sweep_param:
                    sweep_param = "unknown"

                if sweep_param not in parsed_map:
                    parsed_map[sweep_param] = []

                parsed = None
                for p in parsed_map[sweep_param]:
                    if p.nodes == resolved_node:
                        parsed = p
                        break

                if parsed is None:
                    parsed = TpuGemmParsedResult(
                        test_level=test_level,
                        nodes=resolved_node,
                        sweep_param=sweep_param,
                    )
                    parsed_map[sweep_param].append(parsed)

                val = (
                    m.value.number_value
                    if m.value.HasField("number_value")
                    else None
                )

                if m.key == "Throughput":
                    parsed.throughput = val
                elif m.key == "Total Throughput":
                    parsed.total_throughput = val
                elif m.key == "MFU":
                    parsed.mfu = val
                elif m.key == "Step Time":
                    parsed.step_time = val
                elif m.key == "Total FLOPs":
                    parsed.total_flops = val

        rows = []
        for context in sorted(parsed_map.keys()):
            items = parsed_map[context]
            for i, item in enumerate(items):
                rows.append(
                    {
                        "sweep_param": context if i == 0 else "",
                        "nodes": item.nodes,
                        "throughput": (
                            f"{item.throughput:.2f}"
                            if item.throughput is not None
                            else "N/A"
                        ),
                        "total_throughput": (
                            f"{item.total_throughput:.2f}"
                            if item.total_throughput is not None
                            else "N/A"
                        ),
                        "mfu": (
                            f"{item.mfu:.1f}%"
                            if item.mfu is not None
                            else "N/A"
                        ),
                        "step_time": (
                            f"{item.step_time:.2f}"
                            if item.step_time is not None
                            else "N/A"
                        ),
                        "total_flops": (
                            f"{item.total_flops:.2e}"
                            if item.total_flops is not None
                            else "N/A"
                        ),
                    }
                )

        table_model = TableDataModel(
            title="TPU GEMM Performance Summary",
            headers=self.get_column_schema(),
            rows=rows,
        )

        test_name = test_results[0].test_name if test_results else "tpu-gemm"
        test_level = test_results[0].test_level if test_results else ""

        return PluginDetailReport(
            test_name=test_name,
            test_level=test_level,
            tables=[table_model],
        )

    def render_detail(self, test_results: list[pb.TestResult]) -> Group | Text:
        """Renders detail tables for tpu-gemm tests per sweep parameter."""
        report = self.get_detail_data(test_results)
        if not report or not report.tables:
            return Text("No detailed metrics available.", style="yellow")

        table_data = report.tables[0]
        table = create_styled_table(expand=False)
        table.add_column("Parameters (MxNxK,Dtype)", style="cyan")
        table.add_column("Nodes", overflow="fold", style="bold white")
        table.add_column(
            "Throughput\n(TFLOP/s/dev)", justify="right", style="green"
        )
        table.add_column("Total\n(TFLOPs/s)", justify="right", style="green")
        table.add_column("MFU (%)", justify="right", style="magenta")
        table.add_column("Step Time (s)", justify="right", style="blue")
        table.add_column("Total FLOPs", justify="right", style="dim")

        for row in table_data.rows:
            table.add_row(
                row["sweep_param"],
                row["nodes"],
                row["throughput"],
                row["total_throughput"],
                row["mfu"],
                row["step_time"],
                row["total_flops"],
            )

        elements = [
            Text(f"\n{table_data.title}", style="bold underline cyan"),
            table,
        ]
        val_table = self.render_validation_results(test_results)
        if val_table:
            elements.append(val_table)
        return Group(*elements)
