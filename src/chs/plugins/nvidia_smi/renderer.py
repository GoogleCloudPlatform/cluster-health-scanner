from rich.console import Group
from rich.text import Text

from chs.utils import create_styled_table

from chs.protos import bill_of_health_pb2 as pb
from chs.core.default_plugin.renderer import DefaultRenderer


class NvidiaSmiRenderer(DefaultRenderer):
    """Renders details for nvidia-smi tests."""

    def render_detail(self, test_results: list[pb.TestResult]) -> Group | Text:
        """Renders detail tables for nvidia-smi tests per node."""
        if not test_results:
            return Text("No detailed metrics available.", style="yellow")

        node_data = self._extract_node_data(test_results)

        render_elements = [Text("\nNVIDIA SMI Summary", style="bold")]
        render_elements.append(self._build_summary_table(node_data))
        render_elements.extend(self._build_detailed_tables(node_data))

        return Group(*render_elements)

    def _extract_node_data(self, test_results: list[pb.TestResult]) -> dict:
        """Extracts metrics from test results into a structured dict."""
        node_data = {}
        for res in test_results:
            nodes = (
                ", ".join(res.target_resource_ids)
                if res.target_resource_ids
                else "Unknown Node"
            )
            if nodes not in node_data:
                node_data[nodes] = {
                    "summary": {
                        "Driver Version": "N/A",
                        "CUDA Version": "N/A",
                        "GPU Count": "N/A",
                    },
                    "gpus": {},
                }

            for m in res.metrics:
                key = m.key
                val = (
                    m.value.string_value
                    if m.value.HasField("string_value")
                    else ""
                )
                if not val and m.value.HasField("number_value"):
                    num_val = m.value.number_value
                    if key == "GPU Count" and num_val.is_integer():
                        val = str(int(num_val))
                    else:
                        val = str(num_val)

                if not val:
                    continue

                ctx = (
                    m.metadata.fields["context"].string_value
                    if "context" in m.metadata.fields
                    else ""
                )

                if not ctx or ctx == "null":
                    # Global metrics
                    if key in ["Driver Version", "CUDA Version", "GPU Count"]:
                        node_data[nodes]["summary"][key] = val
                else:
                    # Per-GPU metrics
                    # ctx is "GPU 0", "GPU 1", etc.
                    gpu_idx = ctx.replace("GPU ", "")
                    if gpu_idx not in node_data[nodes]["gpus"]:
                        node_data[nodes]["gpus"][gpu_idx] = {}

                    node_data[nodes]["gpus"][gpu_idx][key] = val

                    # Fallback: if global driver is N/A, take it from GPU 0
                    if (
                        key == "Driver Version"
                        and node_data[nodes]["summary"]["Driver Version"]
                        == "N/A"
                    ):
                        node_data[nodes]["summary"]["Driver Version"] = val
        return node_data

    def _build_summary_table(self, node_data: dict):
        """Builds the summary table for all nodes."""
        summary_table = create_styled_table(expand=False)
        summary_table.add_column("Node")
        summary_table.add_column("Driver Version")
        summary_table.add_column("CUDA Version")
        summary_table.add_column("GPU Count", justify="right")

        for node, data in node_data.items():
            summary_table.add_row(
                node,
                data["summary"]["Driver Version"],
                data["summary"]["CUDA Version"],
                data["summary"]["GPU Count"],
            )
        return summary_table

    def _build_detailed_tables(self, node_data: dict) -> list:
        """Builds detailed GPU tables for each node."""
        render_elements = []
        for node, data in node_data.items():
            if not data["gpus"]:
                continue

            render_elements.append(
                Text(f"\nGPU Details for {node}", style="bold")
            )

            det_table = create_styled_table(expand=False)
            det_table.add_column("GPU")
            det_table.add_column("Name")
            det_table.add_column("PState")
            det_table.add_column("Power Draw")
            det_table.add_column("Power Limit")
            det_table.add_column("Graphics Clock")
            det_table.add_column("Memory Clock")

            # Sort by GPU index safely
            def safe_gpu_idx(x):
                try:
                    return int(x)
                except ValueError:
                    return 999

            sorted_gpus = sorted(data["gpus"].keys(), key=safe_gpu_idx)

            for gpu_idx in sorted_gpus:
                gpu_info = data["gpus"][gpu_idx]
                det_table.add_row(
                    f"GPU {gpu_idx}",
                    gpu_info.get("GPU Name", "N/A"),
                    gpu_info.get("Performance State", "N/A"),
                    gpu_info.get("Power Draw", "N/A"),
                    gpu_info.get("Power Limit", "N/A"),
                    gpu_info.get("Graphics Clock", "N/A"),
                    gpu_info.get("Memory Clock", "N/A"),
                )
            render_elements.append(det_table)
        return render_elements
