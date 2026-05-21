from collections import defaultdict
from rich.console import Group
from rich.text import Text

from chs.utils import create_styled_table
from chs.protos import bill_of_health_pb2 as pb
from chs.core.default_plugin.renderer import DefaultRenderer


class TpuInfoRenderer(DefaultRenderer):
    """Renders details for tpu-info tests."""

    def render_detail(self, test_results: list[pb.TestResult]) -> Group | Text:
        if not test_results:
            return Text("No detailed metrics available.", style="yellow")

        node_data = self._extract_node_data(test_results)
        elements = [
            Text("\nTPU Info Summary", style="bold"),
            self._build_summary_table(node_data),
        ]
        elements.extend(self._build_detailed_tables(node_data))
        return Group(*elements)

    def _extract_node_data(self, test_results: list[pb.TestResult]) -> dict:
        """Extracts metrics from test results into structured dicts by node."""
        node_data = {}
        for res in test_results:
            nodes = (
                ", ".join(res.target_resource_ids)
                if res.target_resource_ids
                else "Unknown Node"
            )

            if nodes not in node_data:
                node_data[nodes] = {
                    "summary": {},
                    "chips": defaultdict(dict),
                }

            for m in res.metrics:
                if m.value.HasField("string_value"):
                    val = m.value.string_value
                elif m.value.HasField("number_value"):
                    val = m.value.number_value
                    val = str(int(val)) if val.is_integer() else str(val)
                else:
                    continue

                if not val:
                    continue

                context = m.metadata.fields["context"].string_value or ""
                key = m.key

                if not context or context == "null":
                    node_data[nodes]["summary"][key] = val
                elif context.startswith("Chip"):
                    node_data[nodes]["chips"][context.removeprefix("Chip ")][
                        key
                    ] = val

        return node_data

    def _build_summary_table(self, node_data: dict):
        table = create_styled_table(expand=False)
        table.add_column("Node")
        table.add_column("Libtpu Version")
        table.add_column("Accelerator Type")
        table.add_column("TPU Chip Count", justify="right")

        for node, data in node_data.items():
            summary = data["summary"]
            table.add_row(
                node,
                summary.get("Libtpu version", "N/A"),
                summary.get("Accelerator type", "N/A"),
                summary.get("TPU Chip Count", "N/A"),
            )
        return table

    def _build_detailed_tables(self, node_data: dict) -> list:
        elements = []
        for node, data in node_data.items():
            if chips := data["chips"]:
                elements.append(
                    Text(f"\nTPU Chip Details for {node}", style="bold")
                )
                table = create_styled_table(expand=False)
                table.add_column("Chip")
                table.add_column("Type")
                table.add_column("Devices", justify="right")

                for chip_path, info in sorted(chips.items()):
                    table.add_row(
                        chip_path,
                        info.get("Chip Type", "N/A"),
                        info.get("Chip Devices", "N/A"),
                    )
                elements.append(table)

        return elements
