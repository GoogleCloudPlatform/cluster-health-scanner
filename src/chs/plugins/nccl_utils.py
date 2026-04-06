from dataclasses import dataclass
from chs.protos import bill_of_health_pb2 as pb
from chs.utils import create_styled_table, CHS_HEADER_STYLE
from rich.table import Table

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
    table = create_styled_table(
        header_style=CHS_HEADER_STYLE,
        expand=True,
    )
    table.add_column("Workload")
    table.add_column(resource_column_name)
    table.add_column("Peak Bandwidth (GB/s)", justify="right")
    table.add_column("Msg Size (GB)", justify="right")
    return table
