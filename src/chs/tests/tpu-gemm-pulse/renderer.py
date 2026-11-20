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

from chs.core.default_plugin.renderer import DefaultRenderer
from chs.core.plugin_base import ColumnDef, PluginDetailReport, TableDataModel
from chs.core.utils.rendering import format_val, parse_value
from chs.protos import bill_of_health_pb2 as pb
from chs.tests.tpu_gemm_pulse.extractor import TpuGemmPulseExtractor
from chs.utils import create_styled_table


@dataclass
class TpuGemmPulseParsedResult:
    nodes: str
    sweep_param: str
    throughput: float | None = None
    chip_throughput: float | None = None
    slowdown_ratio: float | None = None
    mfu: float | None = None
    step_time: float | None = None
    valid_checksum: str = "N/A"
    checksum_match: str = "N/A"


class TpuGemmPulseRenderer(DefaultRenderer):
    """Renders details for tpu-gemm-pulse tests."""

    def get_column_schema(self) -> list[ColumnDef]:
        return [
            ColumnDef("sweep_param", "Parameters (MxNxK,Dtype,Chain,Cycles)"),
            ColumnDef("nodes", "Nodes"),
            ColumnDef("throughput", "Throughput\n(TFLOP/s/dev)", align="right"),
            ColumnDef(
                "chip_throughput",
                "Throughput per chip\n(TFLOP/s/chip)",
                align="right",
            ),
            ColumnDef("slowdown_ratio", "Slowdown Ratio", align="right"),
            ColumnDef("mfu", "MFU (%)", align="right"),
            ColumnDef("step_time", "Step Time (ms)", align="right"),
            ColumnDef("valid_checksum", "Valid Checksum"),
            ColumnDef("checksum_match", "Checksum Match"),
        ]

    def get_detail_data(
        self, test_results: list[pb.TestResult]
    ) -> PluginDetailReport | None:
        """Extracts structured TPU GEMM pulse data model."""
        if not test_results:
            return None

        parsed_map: dict[str, list[TpuGemmPulseParsedResult]] = {}
        parsed_by_key: dict[tuple[str, str], TpuGemmPulseParsedResult] = {}

        for res in test_results:
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
                        else ""
                    )
                    parsed_meta = TpuGemmPulseExtractor.parse_context_metadata(
                        ctx
                    )
                    sweep_param = sweep_param or parsed_meta.get("sweep_param")
                    node = node or parsed_meta.get("node")

                if not sweep_param:
                    continue

                resolved_node = node or "N/A"
                resolved_param = sweep_param

                entries = parsed_map.setdefault(resolved_param, [])
                parsed = parsed_by_key.get((resolved_param, resolved_node))
                if parsed is None:
                    parsed = TpuGemmPulseParsedResult(
                        nodes=resolved_node,
                        sweep_param=resolved_param,
                    )
                    entries.append(parsed)
                    parsed_by_key[(resolved_param, resolved_node)] = parsed

                num_val = (
                    m.value.number_value
                    if m.value.HasField("number_value")
                    else None
                )

                if m.key == "Throughput":
                    parsed.throughput = num_val
                elif m.key == "Throughput per chip":
                    parsed.chip_throughput = num_val
                elif m.key == "Slowdown Ratio":
                    parsed.slowdown_ratio = num_val
                elif m.key == "MFU":
                    parsed.mfu = num_val
                elif m.key == "Step Time":
                    parsed.step_time = num_val
                elif m.key == "Valid Checksum":
                    parsed.valid_checksum = str(parse_value(m.value))
                elif m.key == "Checksum Match":
                    parsed.checksum_match = str(parse_value(m.value))

        rows = []
        for context in sorted(parsed_map.keys()):
            for i, item in enumerate(parsed_map[context]):
                mfu_str = (
                    f"{format_val(item.mfu, precision=1)}%"
                    if item.mfu is not None
                    else "N/A"
                )
                rows.append(
                    {
                        "sweep_param": context if i == 0 else "",
                        "nodes": item.nodes,
                        "throughput": format_val(item.throughput),
                        "chip_throughput": format_val(item.chip_throughput),
                        "slowdown_ratio": format_val(item.slowdown_ratio),
                        "mfu": mfu_str,
                        "step_time": format_val(item.step_time),
                        "valid_checksum": item.valid_checksum,
                        "checksum_match": item.checksum_match,
                    }
                )

        table_model = TableDataModel(
            title="TPU GEMM Pulse Summary",
            headers=self.get_column_schema(),
            rows=rows,
        )

        return PluginDetailReport(
            test_name=test_results[0].test_name,
            test_level=test_results[0].test_level,
            tables=[table_model],
        )

    def render_detail(self, test_results: list[pb.TestResult]) -> Group | Text:
        """Renders detail tables for tpu-gemm-pulse tests."""
        report = self.get_detail_data(test_results)
        if not report or not report.tables:
            return Text("No detailed metrics available.", style="yellow")

        table_data = report.tables[0]
        table = create_styled_table(expand=False)
        table.add_column("Parameters (MxNxK,Dtype,Chain,Cycles)", style="cyan")
        table.add_column("Nodes", overflow="fold", style="bold white")
        table.add_column(
            "Throughput\n(TFLOP/s/dev)", justify="right", style="green"
        )
        table.add_column(
            "Throughput per chip\n(TFLOP/s/chip)",
            justify="right",
            style="green",
        )
        table.add_column("Slowdown Ratio", justify="right", style="yellow")
        table.add_column("MFU (%)", justify="right", style="magenta")
        table.add_column("Step Time (ms)", justify="right", style="blue")
        table.add_column("Valid Checksum", style="bold white")
        table.add_column("Checksum Match", style="bold white")

        for row in table_data.rows:
            table.add_row(
                row["sweep_param"],
                row["nodes"],
                row["throughput"],
                row["chip_throughput"],
                row["slowdown_ratio"],
                row["mfu"],
                row["step_time"],
                row["valid_checksum"],
                row["checksum_match"],
            )

        elements = [
            Text(f"\n{table_data.title}", style="bold underline cyan"),
            table,
        ]
        val_table = self.render_validation_results(test_results)
        if val_table:
            elements.append(val_table)
        return Group(*elements)
