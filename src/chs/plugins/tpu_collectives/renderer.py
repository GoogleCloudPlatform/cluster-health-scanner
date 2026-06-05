from collections import defaultdict
from dataclasses import dataclass

from rich.console import Group
from rich.text import Text

from chs.core.default_plugin.renderer import DefaultRenderer
from chs.protos import bill_of_health_pb2 as pb
from chs.utils import create_styled_table


@dataclass
class TpuCollectivesParsedResult:
    test_level: str
    nodes: str
    op_type: str
    matrix_dim: str
    dtype: str
    bandwidth: float | None = None
    time_taken: float | None = None


class TpuCollectivesRenderer(DefaultRenderer):
    """Renders details for tpu-collectives tests."""

    def _get_summary_title(self) -> Text:
        return Text(
            "\nTPU Collectives Performance Summary", style="bold underline cyan"
        )

    def render_detail(self, test_results: list[pb.TestResult]) -> Group | Text:
        """Renders detail tables for tpu-collectives tests."""
        if not test_results:
            return Text("No detailed metrics available.", style="yellow")

        parsed_map: dict[tuple[str, str, str], TpuCollectivesParsedResult] = {}

        for res in test_results:
            test_level = res.test_level if res.test_level else "check"
            node = (
                ", ".join(res.target_resource_ids)
                if res.target_resource_ids
                else "Unknown Node"
            )

            for m in res.metrics:
                ctx = (
                    m.metadata.fields["context"].string_value
                    if "context" in m.metadata.fields
                    else "unknown"
                )
                if ctx == "unknown" or ctx == "null":
                    continue

                parts = ctx.split(",")
                resolved_node = node
                matrix_dim = ctx
                dtype = "N/A"
                if len(parts) == 2:
                    matrix_dim = parts[0]
                    dtype = parts[1]
                key = (resolved_node, test_level, ctx)
                if key not in parsed_map:
                    parsed_map[key] = TpuCollectivesParsedResult(
                        test_level=test_level,
                        nodes=resolved_node,
                        op_type=test_level,
                        matrix_dim=matrix_dim,
                        dtype=dtype,
                    )

                parsed = parsed_map[key]
                val = (
                    m.value.number_value
                    if m.value.HasField("number_value")
                    else None
                )

                if m.key == "Bandwidth (median)":
                    parsed.bandwidth = val
                elif m.key == "Time Taken (median)":
                    parsed.time_taken = val

        nodes_map = defaultdict(list)
        for parsed in parsed_map.values():
            nodes_map[parsed.nodes].append(parsed)

        def get_matrix_dim_key(res: TpuCollectivesParsedResult) -> int:
            try:
                return int(res.matrix_dim)
            except ValueError:
                return 999999

        render_elements = [self._get_summary_title()]

        for node in sorted(nodes_map.keys()):
            render_elements.append(
                Text(f"\nTarget Resource(s): {node}", style="bold magenta")
            )

            node_ops_map = defaultdict(list)
            for item in nodes_map[node]:
                node_ops_map[item.op_type].append(item)

            node_table = create_styled_table(expand=False)
            node_table.add_column("Operation", style="cyan")
            node_table.add_column("Matrix Dim", style="bold white")
            node_table.add_column("Dtype", style="dim")
            node_table.add_column(
                "Bandwidth\n(GB/s)", justify="right", style="green"
            )

            for op_type in sorted(node_ops_map.keys()):
                sorted_items = sorted(
                    node_ops_map[op_type], key=get_matrix_dim_key
                )

                for i, item in enumerate(sorted_items):
                    bw_str = (
                        f"{item.bandwidth:.2f}"
                        if item.bandwidth is not None
                        else "N/A"
                    )

                    op_column = item.op_type if i == 0 else ""

                    node_table.add_row(
                        op_column,
                        item.matrix_dim,
                        item.dtype,
                        bw_str,
                    )

            render_elements.append(node_table)

        return Group(*render_elements)
