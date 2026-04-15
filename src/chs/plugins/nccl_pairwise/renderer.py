from chs.protos import bill_of_health_pb2 as pb
from rich.console import RenderableType
from chs.core.default_plugin.renderer import DefaultRenderer
from chs.plugins.nccl_utils import render_nccl_detail


class NcclPairwiseRenderer(DefaultRenderer):
    def __init__(self):
        super().__init__()
        self.topology = None

    def set_topology(self, topology):
        self.topology = topology

    def render_detail(
        self, test_results: list[pb.TestResult]
    ) -> RenderableType:
        return render_nccl_detail(
            "Pairwise Detail (Aggregated)",
            "Pair (Node 1 <-> Node 2)",
            test_results,
            lambda res: " <-> ".join(res.target_resource_ids)
            if res.target_resource_ids
            else "Unknown Pair",
            self.topology,
        )
