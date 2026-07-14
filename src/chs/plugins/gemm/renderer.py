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
from dataclasses import dataclass

from chs.utils import create_styled_table
from chs.protos import bill_of_health_pb2 as pb
from chs.plugins.utils.rendering import format_nodes
from chs.core.default_plugin.renderer import DefaultRenderer


@dataclass
class GemmParsedResult:
    test_level: str
    nodes: str
    min_val: float | None
    median_val: float | None
    max_val: float | None
    validation_status: str


class GemmRenderer(DefaultRenderer):
    """Renders details for gemm tests."""

    def render_detail(self, test_results: list[pb.TestResult]) -> Group | Text:
        """Renders detail tables for gemm tests per node."""
        if not test_results:
            return Text("No detailed metrics available.", style="yellow")

        has_thresholds = not all(
            r.result_status == pb.RESULT_STATUS_UNKNOWN for r in test_results
        )

        table = create_styled_table(expand=False)
        table.add_column("Workload")
        table.add_column("Node", overflow="fold")
        table.add_column("Min (TFLOPS)", justify="right")
        table.add_column("Median (TFLOPS)", justify="right")
        table.add_column("Max (TFLOPS)", justify="right")
        if has_thresholds:
            table.add_column("Validation")

        parsed_results: list[GemmParsedResult] = []
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

            validation_status = "N/A"
            if res.result_status == pb.RESULT_STATUS_SUCCESS:
                validation_status = "[green]MET[/]"
            elif res.result_status == pb.RESULT_STATUS_FAILED:
                failed_threshold = any(
                    m.HasField("validation_criteria")
                    and m.validation_criteria.result_status
                    == pb.RESULT_STATUS_FAILED
                    for m in res.metrics
                )
                if failed_threshold:
                    validation_status = "[red]NOT MET[/]"
                else:
                    validation_status = "[yellow]ERROR[/]"

            parsed_results.append(
                GemmParsedResult(
                    test_level=test_level,
                    nodes=nodes,
                    min_val=min_val,
                    median_val=median_val,
                    max_val=max_val,
                    validation_status=validation_status,
                )
            )

        def to_tflops(val):
            if val is None:
                return None
            return val / 1000.0 if val >= 0 else val

        for item in parsed_results:
            min_tflops, median_tflops, max_tflops = map(
                to_tflops, [item.min_val, item.median_val, item.max_val]
            )

            def format_val(v):
                return f"{v:.2f}" if v is not None else "N/A"

            min_str, median_str, max_str = map(
                format_val, [min_tflops, median_tflops, max_tflops]
            )
            row = [
                item.test_level,
                item.nodes,
                min_str,
                median_str,
                max_str,
            ]
            if has_thresholds:
                row.append(item.validation_status)
            table.add_row(*row)

        elements = [
            Text("\nGEMM Detail", style="bold"),
            table,
        ]
        val_table = self.render_validation_results(test_results)
        if val_table:
            elements.append(val_table)

        return Group(*elements)
