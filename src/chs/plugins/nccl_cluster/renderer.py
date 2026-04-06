from chs.protos import bill_of_health_pb2 as pb
from chs.core.default_plugin.renderer import DefaultRenderer
from rich.text import Text
from rich.console import Group, RenderableType
from chs.plugins.nccl_utils import (
    get_peak_metrics,
    create_nccl_table,
)
from chs.utils import CHS_WARNING_STYLE, CHS_TITLE_STYLE


class NcclClusterRenderer(DefaultRenderer):
    def render_detail(
        self, test_results: list[pb.TestResult]
    ) -> RenderableType:
        if not test_results:
            return Text(
                "No detailed metrics available.", style=CHS_WARNING_STYLE
            )

        table = create_nccl_table("Cluster Metric")

        for res in test_results:
            peak = get_peak_metrics(res.metrics)
            max_bw = peak.max_bw
            max_size = peak.max_size

            test_level = res.test_level if res.test_level else "Unknown"
            table.add_row(
                test_level,
                "Peak Detected Cluster Bandwidth",
                f"{max_bw:.2f}",
                max_size,
            )

        return Group(Text("\nCluster Detail", style=CHS_TITLE_STYLE), table)
