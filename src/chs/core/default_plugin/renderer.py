from chs.protos import bill_of_health_pb2 as pb
from rich.console import RenderableType, Group
from rich.panel import Panel
from rich.text import Text
from chs.utils import create_styled_table


class DefaultRenderer:
    """Default visualization logic using Rich."""

    @staticmethod
    def _has_validation_criteria(test_results: list[pb.TestResult]) -> bool:
        """Checks if any metric in the test results has validation criteria."""
        return any(
            m.HasField("validation_criteria") and m.validation_criteria.name
            for res in test_results
            for m in res.metrics
        )

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
            table.add_column("Threshold", style="yellow")
            table.add_column("Status", style="bold")

            for metric in test_result.metrics:
                if not metric.HasField("validation_criteria"):
                    continue
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

                th_str = "N/A"
                status_str = "N/A"
                status_style = "white"

                if metric.HasField("validation_criteria"):
                    vc = metric.validation_criteria
                    if vc.name:
                        if vc.HasField("expected_value"):
                            if vc.expected_value.HasField("number_value"):
                                th_val = vc.expected_value.number_value
                                if th_val.is_integer():
                                    th_val = int(th_val)
                            elif vc.expected_value.HasField("bool_value"):
                                th_val = vc.expected_value.bool_value
                            else:
                                th_val = vc.expected_value.string_value
                        else:
                            th_val = "complex"

                        comp_str = ""
                        if vc.comparator == pb.Comparator.COMPARATOR_GTE:
                            comp_str = ">="
                        elif vc.comparator == pb.Comparator.COMPARATOR_EQ:
                            comp_str = "=="
                        elif vc.comparator == pb.Comparator.COMPARATOR_CONTAINS:
                            comp_str = "contains"

                        th_str = f"{comp_str} {th_val}".strip()

                        if vc.result_status == pb.RESULT_STATUS_SUCCESS:
                            status_str = "PASS"
                            status_style = "green"
                        elif vc.result_status == pb.RESULT_STATUS_FAILED:
                            status_str = "FAIL"
                            status_style = "red"
                        else:
                            status_str = "UNKNOWN"
                            status_style = "yellow"

                table.add_row(
                    f"{prefix}{metric.key}",
                    str(val),
                    metric.unit,
                    th_str,
                    Text(status_str, style=status_style),
                )

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
