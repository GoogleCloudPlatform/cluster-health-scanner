from chs.plugins.nccl_base_extractor import NcclBaseExtractor
from chs.core.plugin_base import TestPlugin


class NcclClusterExtractor(NcclBaseExtractor):
    def extract_test_info(self: TestPlugin, exp) -> tuple[str, str, str]:
        level = exp.get("workload_name", "")
        return "nccl-cluster", level, self.get_description("nccl-cluster")
