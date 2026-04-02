from chs.protos import bill_of_health_pb2 as pb
from rich.text import Text
from rich.console import Group, RenderableType
from chs.core.default_plugin.renderer import DefaultRenderer
from chs.plugins.nccl_utils import (
    get_peak_metrics,
    create_nccl_table,
)
from chs.utils.theme import CHS_WARNING_STYLE, CHS_TITLE_STYLE


class NcclPairwiseRenderer(DefaultRenderer):
    def render_detail(
        self, test_results: list[pb.TestResult]
    ) -> RenderableType:
        if not test_results:
            return Text(
                "No detailed metrics available.", style=CHS_WARNING_STYLE
            )

        table = create_nccl_table("Pair (Node 1 <-> Node 2)")

        for res in test_results:
            peak = get_peak_metrics(res.metrics)
            max_bw = peak.max_bw
            max_size = peak.max_size

            nodes = (
                " <-> ".join(res.target_resource_ids)
                if res.target_resource_ids
                else "Unknown Pair"
            )
            test_level = res.test_level if res.test_level else "Unknown"
            table.add_row(test_level, nodes, f"{max_bw:.2f}", max_size)

        return Group(
            Text("\nPairwise Detail (Aggregated)", style=CHS_TITLE_STYLE),
            table,
        )
