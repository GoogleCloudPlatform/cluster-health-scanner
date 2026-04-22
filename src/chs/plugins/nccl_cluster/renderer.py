from chs.protos import bill_of_health_pb2 as pb
from chs.core.default_plugin.renderer import DefaultRenderer
from rich.console import RenderableType
from chs.plugins.nccl_utils import render_nccl_detail


class NcclClusterRenderer(DefaultRenderer):
    def render_detail(
        self, test_results: list[pb.TestResult]
    ) -> RenderableType:
        return render_nccl_detail(
            "Cluster Detail",
            "Cluster Metric",
            test_results,
            lambda res: "Peak Detected Cluster Bandwidth",
        )
