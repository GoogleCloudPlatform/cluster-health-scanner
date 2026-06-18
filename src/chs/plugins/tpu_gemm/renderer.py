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

    def render_detail(self, test_results: list[pb.TestResult]) -> Group | Text:
        """Renders detail tables for tpu-gemm tests per sweep parameter."""
        if not test_results:
            return Text("No detailed metrics available.", style="yellow")

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

        parsed_map: dict[str, list[TpuGemmParsedResult]] = {}

        for res in test_results:
            test_level = res.test_level if res.test_level else "check"

            for m in res.metrics:
                ctx = (
                    m.metadata.fields["context"].string_value
                    if "context" in m.metadata.fields
                    else "unknown"
                )
                if ctx == "unknown" or ctx == "null":
                    continue

                parts = ctx.split(",")
                resolved_node = "N/A"
                sweep_param = ctx
                if len(parts) == 3:
                    resolved_node = parts[0]
                    sweep_param = f"{parts[1]},{parts[2]}"

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

        for context in sorted(parsed_map.keys()):
            items = parsed_map[context]

            for i, item in enumerate(items):
                tflops_str = (
                    f"{item.throughput:.2f}"
                    if item.throughput is not None
                    else "N/A"
                )
                total_tflops_str = (
                    f"{item.total_throughput:.2f}"
                    if item.total_throughput is not None
                    else "N/A"
                )
                mfu_str = f"{item.mfu:.1f}%" if item.mfu is not None else "N/A"
                step_str = (
                    f"{item.step_time:.2f}"
                    if item.step_time is not None
                    else "N/A"
                )
                flops_str = (
                    f"{item.total_flops:.2e}"
                    if item.total_flops is not None
                    else "N/A"
                )

                param_column = context if i == 0 else ""

                table.add_row(
                    param_column,
                    item.nodes,
                    tflops_str,
                    total_tflops_str,
                    mfu_str,
                    step_str,
                    flops_str,
                )

        elements = [
            Text("\nTPU GEMM Performance Summary", style="bold underline cyan"),
            table,
        ]
        val_table = self.render_validation_results(test_results)
        if val_table:
            elements.append(val_table)
        return Group(*elements)
