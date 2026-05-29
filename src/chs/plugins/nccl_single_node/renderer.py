from chs.protos import bill_of_health_pb2 as pb
from rich.console import RenderableType
from chs.core.default_plugin.renderer import DefaultRenderer
from chs.plugins.utils.nccl_utils import render_nccl_detail


class NcclSingleNodeRenderer(DefaultRenderer):
    def render_detail(
        self, test_results: list[pb.TestResult]
    ) -> RenderableType:
        return render_nccl_detail(
            "Single Node Detail",
            "Node",
            test_results,
            lambda res: ", ".join(res.target_resource_ids)
            if res.target_resource_ids
            else "Unknown Node",
        )
