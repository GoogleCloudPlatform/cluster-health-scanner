from dataclasses import dataclass
from chs.protos import bill_of_health_pb2 as pb
from chs.utils import theme

from rich.table import Table
from rich.text import Text
from rich.console import Group, RenderableType

BYTES_PER_GB = 1024**3


@dataclass
class PeakMetrics:
    max_bw: float
    max_size: str


def get_peak_metrics(metrics: list[pb.Metric]) -> PeakMetrics:
    max_bw = 0.0
    max_size = "Unknown"
    for m in metrics:
        if m.value.HasField("number_value") and "Bandwidth" in m.key:
            if m.value.number_value > max_bw:
                max_bw = m.value.number_value
                if "context" in m.metadata.fields:
                    try:
                        size_bytes = float(
                            m.metadata.fields["context"].string_value
                        )
                        max_size = f"{size_bytes / BYTES_PER_GB:.1f}"
                    except ValueError:
                        max_size = "Unknown"
    return PeakMetrics(max_bw, max_size)


def create_nccl_table(resource_column_name: str) -> Table:
    """Creates a standardized Table for NCCL metrics."""
    table = theme.create_styled_table(
        header_style=theme.CHS_HEADER_STYLE,
        expand=True,
    )
    table.add_column("Workload")
    table.add_column(resource_column_name)
    table.add_column("Peak Bandwidth (GB/s)", justify="right")
    table.add_column("Msg Size (GB)", justify="right")
    return table


def render_nccl_detail(
    title: str,
    resource_column_name: str,
    test_results: list[pb.TestResult],
    get_resource_id_func,
) -> RenderableType:
    if not test_results:
        return Text(
            "No detailed metrics available.", style=theme.CHS_WARNING_STYLE
        )

    table = create_nccl_table(resource_column_name)

    for res in test_results:
        peak = get_peak_metrics(res.metrics)
        max_bw = peak.max_bw
        max_size = peak.max_size

        resource_id = get_resource_id_func(res)
        test_level = res.test_level if res.test_level else "Unknown"
        table.add_row(test_level, resource_id, f"{max_bw:.2f}", max_size)

    renderables = [Text(f"\n{title}", style=theme.CHS_TITLE_STYLE), table]

    return Group(*renderables)
