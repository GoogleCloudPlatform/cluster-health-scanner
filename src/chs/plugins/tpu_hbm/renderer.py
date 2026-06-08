# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from rich.console import Group
from rich.text import Text
from chs.utils import create_styled_table
from chs.protos import bill_of_health_pb2 as pb
from chs.core.default_plugin.renderer import DefaultRenderer


@dataclass
class HbmResult:
    """Holds HBM benchmark results for a specific node, size, and dtype."""

    node: str
    num_elements: int
    dtype: str
    bandwidth: float | None = None
    time_ms: float | None = None
    tensor_size: float | None = None


class TpuHbmRenderer(DefaultRenderer):
    """Renders details for tpu-hbm tests."""

    def render_detail(self, test_results: list[pb.TestResult]) -> Group | Text:
        """Renders detail tables for hbm tests per node."""
        if not test_results:
            return Text("No detailed metrics available.", style="yellow")

        node_data = self._extract_node_data(test_results)
        if not node_data:
            return Text("No parsed HBM metrics found.", style="yellow")

        elements = [Text("\nTPU HBM Bandwidth Detail", style="bold cyan")]
        elements.extend(self._build_bandwidth_tables(node_data))

        val_table = self.render_validation_results(test_results)
        if val_table:
            elements.append(val_table)

        return Group(*elements)

    def _parse_context(
        self, ctx_name: str
    ) -> tuple[str | None, int, str] | None:
        """Parses node, num_elements, and dtype from context name."""
        parts = ctx_name.split(",")
        if len(parts) == 3:
            try:
                return parts[0], int(parts[1]), parts[2]
            except ValueError:
                pass
        return None

    def _extract_node_data(
        self, test_results: list[pb.TestResult]
    ) -> dict[str, list[HbmResult]]:
        """Extracts metrics from test results and groups them by node."""
        aggregated: dict[tuple[str, int, str], HbmResult] = {}

        for res in test_results:
            default_node = (
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
                if not parsed_ctx:
                    continue

                node, num_elements, dtype = parsed_ctx
                if not node:
                    node = default_node

                key = (node, num_elements, dtype)

                if key not in aggregated:
                    aggregated[key] = HbmResult(
                        node=node, num_elements=num_elements, dtype=dtype
                    )

                val = m.value.number_value
                if m.key == "Bandwidth (median)":
                    aggregated[key].bandwidth = val
                elif m.key == "Time Taken (median)":
                    aggregated[key].time_ms = val
                elif m.key == "Tensor Size":
                    aggregated[key].tensor_size = val

        node_data: dict[str, list[HbmResult]] = {}
        for r in sorted(
            aggregated.values(), key=lambda x: (x.node, x.num_elements)
        ):
            node_data.setdefault(r.node, []).append(r)

        return node_data

    def _build_bandwidth_tables(
        self, node_data: dict[str, list[HbmResult]]
    ) -> list:
        """Builds bandwidth tables for each node."""
        render_elements = []

        for node in sorted(node_data.keys()):
            node_results = node_data[node]

            table = create_styled_table(expand=False)
            table.add_column("Elements", justify="right")
            table.add_column("Tensor Size (MB)", justify="right")
            table.add_column("Dtype", justify="left")
            table.add_column("Bandwidth (median) (GB/s)", justify="right")
            table.add_column("Time Taken (median) (ms)", justify="right")

            for r in node_results:
                bw_str = (
                    f"{r.bandwidth:.2f}" if r.bandwidth is not None else "N/A"
                )
                time_str = (
                    f"{r.time_ms:.3f}" if r.time_ms is not None else "N/A"
                )
                ts_str = (
                    f"{r.tensor_size:.1f}"
                    if r.tensor_size is not None
                    else "N/A"
                )
                elements_str = f"{r.num_elements:,}"
                table.add_row(elements_str, ts_str, r.dtype, bw_str, time_str)

            render_elements.append(Text(f"\nNode: {node}", style="bold yellow"))
            render_elements.append(table)

        return render_elements
