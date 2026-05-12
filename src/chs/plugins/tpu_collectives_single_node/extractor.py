from typing import Any
from chs.plugins.tpu_collectives.extractor import TpuCollectivesExtractor


class TpuCollectivesSingleNodeExtractor(TpuCollectivesExtractor):
    def extract_test_info(self, exp: dict[str, Any]) -> tuple[str, str, str]:
        """Extracts test name and level for TPU Collectives Single Node."""
        name = "tpu-collectives-single-node"
        level = exp.get("workload_name", "check")
        return name, level, self.get_description(name)
