from chs.protos import bill_of_health_pb2 as pb
from rich.console import RenderableType, Group
from rich.panel import Panel
from rich.text import Text
from chs.utils import create_styled_table


class DefaultRenderer:
    """Default visualization logic using Rich."""

    def render_detail(
        self, test_results: list[pb.TestResult]
    ) -> RenderableType:
        if not test_results:
            return Text("No detailed metrics available.", style="yellow")

        panels = []
        for test_result in test_results:
            is_success = test_result.result_status == pb.RESULT_STATUS_SUCCESS
            border_color = "green" if is_success else "red"

            table = create_styled_table(
                header_style=f"bold {border_color}",
                expand=True,
                show_lines=True,
            )
            table.add_column("Key", style="cyan")
            table.add_column("Value", style="bold white")
            table.add_column("Unit", style="dim")

            for metric in test_result.metrics:
                if metric.value.HasField("number_value"):
                    val = metric.value.number_value
                elif metric.value.HasField("bool_value"):
                    val = metric.value.bool_value
                else:
                    val = metric.value.string_value

                ctx_name = (
                    metric.metadata.fields["context"].string_value
                    if "context" in metric.metadata.fields
                    else ""
                )
                prefix = (
                    f"[{ctx_name}] "
                    if ctx_name and ctx_name != "unknown"
                    else ""
                )
                table.add_row(f"{prefix}{metric.key}", str(val), metric.unit)

            nodes = (
                ", ".join(test_result.target_resource_ids)
                if test_result.target_resource_ids
                else "Cluster"
            )
            panels.append(
                Panel(
                    table
                    if test_result.metrics
                    else Text("No metrics generated."),
                    title=(
                        f"Raw Metrics: {test_result.test_name} "
                        f"({test_result.test_level}) - \\[{nodes}\\]"
                    ),
                    expand=True,
                    border_style=border_color,
                )
            )

        return Group(*panels)
