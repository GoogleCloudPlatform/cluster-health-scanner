from rich.console import Group
from rich.text import Text

from chs.utils import create_styled_table

from dataclasses import dataclass
from chs.protos import bill_of_health_pb2 as pb
from chs.core.default_plugin.renderer import DefaultRenderer


@dataclass
class GemmParsedResult:
    test_level: str
    nodes: str
    min_val: float | None
    median_val: float | None
    max_val: float | None


class GemmRenderer(DefaultRenderer):
    """Renders details for gemm tests."""

    def render_detail(self, test_results: list[pb.TestResult]) -> Group | Text:
        """Renders detail tables for gemm tests per node."""
        if not test_results:
            return Text("No detailed metrics available.", style="yellow")

        table = create_styled_table(expand=False)
        table.add_column("Workload")
        table.add_column("Node")
        table.add_column("Min (TFLOPS)", justify="right")
        table.add_column("Median (TFLOPS)", justify="right")
        table.add_column("Max (TFLOPS)", justify="right")

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
                ", ".join(res.target_resource_ids)
                if res.target_resource_ids
                else "Unknown Node"
            )
            test_level = res.test_level if res.test_level else "Unknown"
            parsed_results.append(
                GemmParsedResult(
                    test_level, nodes, min_val, median_val, max_val
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
            table.add_row(
                item.test_level,
                item.nodes,
                min_str,
                median_str,
                max_str,
            )

        return Group(
            Text("\nGEMM Detail", style="bold"),
            table,
        )
