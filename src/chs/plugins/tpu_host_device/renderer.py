from dataclasses import dataclass
from rich.console import Group
from rich.text import Text
from chs.utils import create_styled_table
from chs.protos import bill_of_health_pb2 as pb
from chs.core.default_plugin.renderer import DefaultRenderer


@dataclass
class HostDeviceResult:
    """Holds bandwidth results for a specific node and transfer size."""

    node: str
    size_mib: int
    h2d: float | None = None
    d2h: float | None = None


class TpuHostDeviceRenderer(DefaultRenderer):
    """Renders details for tpu-host-device tests."""

    def render_detail(self, test_results: list[pb.TestResult]) -> Group | Text:
        """Renders detail tables for host-device tests per node and size."""
        if not test_results:
            return Text("No detailed metrics available.", style="yellow")

        node_data = self._extract_node_data(test_results)
        if not node_data:
            return Text("No parsed Host-Device metrics found.", style="yellow")

        elements = [
            Text("\nHost-Device I/O Bandwidth Detail", style="bold cyan")
        ]
        elements.extend(self._build_bandwidth_tables(node_data))

        val_table = self.render_validation_results(test_results)
        if val_table:
            elements.append(val_table)

        return Group(*elements)

    def _parse_context(self, ctx_name: str) -> int | None:
        """Parses size from context name (e.g. '1024MiB')."""
        if ctx_name.endswith("MiB"):
            try:
                return int(ctx_name[:-3])
            except ValueError:
                pass
        return None

    def _extract_node_data(
        self, test_results: list[pb.TestResult]
    ) -> dict[str, list[HostDeviceResult]]:
        """Extracts metrics from test results and groups them by node."""
        aggregated: dict[tuple[str, int], HostDeviceResult] = {}

        for res in test_results:
            node = (
                res.target_resource_ids[0]
                if res.target_resource_ids
                else "unknown"
            )
            for m in res.metrics:
                if not m.value.HasField("number_value"):
                    continue

                ctx_field = m.metadata.fields.get("context")
                ctx_name = ctx_field.string_value if ctx_field else ""

                parsed_ctx = self._parse_context(ctx_name)
                if parsed_ctx is None:
                    continue

                size = parsed_ctx
                key = (node, size)

                if key not in aggregated:
                    aggregated[key] = HostDeviceResult(node=node, size_mib=size)

                val = m.value.number_value
                if m.key == "H2D Bandwidth":
                    aggregated[key].h2d = val
                elif m.key == "D2H Bandwidth":
                    aggregated[key].d2h = val

        node_data: dict[str, list[HostDeviceResult]] = {}
        for r in sorted(
            aggregated.values(), key=lambda x: (x.node, x.size_mib)
        ):
            node_data.setdefault(r.node, []).append(r)

        return node_data

    def _build_bandwidth_tables(
        self, node_data: dict[str, list[HostDeviceResult]]
    ) -> list:
        """Builds bandwidth tables for each node."""
        render_elements = []

        for node in sorted(node_data.keys()):
            node_results = node_data[node]

            table = create_styled_table(expand=False)
            table.add_column("Size (MiB)", justify="right")
            table.add_column("H2D Bandwidth (GB/s)", justify="right")
            table.add_column("D2H Bandwidth (GB/s)", justify="right")

            for r in node_results:
                h2d_str = f"{r.h2d:.2f}" if r.h2d is not None else "N/A"
                d2h_str = f"{r.d2h:.2f}" if r.d2h is not None else "N/A"
                table.add_row(str(r.size_mib), h2d_str, d2h_str)

            render_elements.append(Text(f"\nNode: {node}", style="bold yellow"))
            render_elements.append(table)

        return render_elements
