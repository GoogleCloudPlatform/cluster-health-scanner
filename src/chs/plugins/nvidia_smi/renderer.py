from rich.console import Group
from rich.text import Text

from chs.utils import create_styled_table

from dataclasses import dataclass
from chs.protos import bill_of_health_pb2 as pb
from chs.core.default_plugin.renderer import DefaultRenderer


@dataclass
class NvidiaSmiParsedResult:
    test_level: str
    nodes: str
    driver_version: str | None
    cuda_version: str | None
    gpu_count: int | None


class NvidiaSmiRenderer(DefaultRenderer):
    """Renders details for nvidia-smi tests."""

    def render_detail(self, test_results: list[pb.TestResult]) -> Group | Text:
        """Renders detail tables for nvidia-smi tests per node."""
        if not test_results:
            return Text("No detailed metrics available.", style="yellow")

        table = create_styled_table(expand=False)
        table.add_column("Workload")
        table.add_column("Node")
        table.add_column("Driver Version")
        table.add_column("CUDA Version")
        table.add_column("GPU Count", justify="right")

        parsed_results: list[NvidiaSmiParsedResult] = []
        for res in test_results:
            driver_version = None
            cuda_version = None
            gpu_count = None

            for m in res.metrics:
                key = m.key.lower()
                val = (
                    m.value.string_value
                    if m.value.HasField("string_value")
                    else ""
                )
                if not val and m.value.HasField("number_value"):
                    val = str(m.value.number_value)

                if not val:
                    continue

                if "driver" in key:
                    driver_version = val
                elif "cuda" in key:
                    cuda_version = val
                elif "gpu count" in key:
                    try:
                        gpu_count = int(float(val))
                    except ValueError:
                        pass

            nodes = (
                ", ".join(res.target_resource_ids)
                if res.target_resource_ids
                else "Unknown Node"
            )
            test_level = res.test_level if res.test_level else "Unknown"
            parsed_results.append(
                NvidiaSmiParsedResult(
                    test_level, nodes, driver_version, cuda_version, gpu_count
                )
            )

        for item in parsed_results:
            table.add_row(
                item.test_level,
                item.nodes,
                item.driver_version or "N/A",
                item.cuda_version or "N/A",
                str(item.gpu_count) if item.gpu_count is not None else "N/A",
            )

        return Group(
            Text("\nNVIDIA SMI Detail", style="bold"),
            table,
        )
