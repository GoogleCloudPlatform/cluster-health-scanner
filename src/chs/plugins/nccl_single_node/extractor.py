from chs.plugins.nccl_base_extractor import NcclBaseExtractor


class NcclSingleNodeExtractor(NcclBaseExtractor):
    def extract_test_info(self, exp) -> tuple[str, str, str]:
        level = exp.get("workload_name", "")
        return (
            "nccl-single-node",
            level,
            self.get_description("nccl-single-node"),
        )
