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
from chs.plugins.utils.rendering import format_nodes
from chs.core.default_plugin.renderer import DefaultRenderer


@dataclass
class NetworkDiagnosticResult:
    """Holds Network Diagnostic benchmark results for a node and axis."""

    node: str
    axis: str
    bandwidth: float | None = None


class TpuNetworkDiagnosticsRenderer(DefaultRenderer):
    """Renders details for tpu-network-diagnostics tests."""

    def render_detail(self, test_results: list[pb.TestResult]) -> Group | Text:
        """Renders detail tables for network diagnostic tests per node."""
        if not test_results:
            return Text("No detailed metrics available.", style="yellow")

        node_data = self._extract_node_data(test_results)
        if not node_data:
            return Text(
                "No parsed Network Diagnostics metrics found.", style="yellow"
            )

        elements = [Text("\nTPU Network Diagnostics Detail", style="bold cyan")]
        elements.extend(self._build_bandwidth_tables(node_data))

        val_table = self.render_validation_results(test_results)
        if val_table:
            elements.append(val_table)

        return Group(*elements)

    def _parse_context(
        self, ctx_name: str
    ) -> tuple[str | None, str | None] | None:
        """Parses node and axis from context name."""
        # Context is formatted as "{host} Axis-{axis}"
        parts = ctx_name.split(" Axis-")
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()
        return None

    def _extract_node_data(
        self, test_results: list[pb.TestResult]
    ) -> dict[str, list[NetworkDiagnosticResult]]:
        """Extracts metrics from test results and groups them by node."""
        aggregated: dict[tuple[str, str], NetworkDiagnosticResult] = {}

        for res in test_results:
            default_node = (
                format_nodes(res.target_resources)
                if res.target_resources
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

                node, axis = parsed_ctx
                if not node:
                    node = default_node

                key = (node, axis)

                if key not in aggregated:
                    aggregated[key] = NetworkDiagnosticResult(
                        node=node, axis=axis
                    )

                if m.key == "Bandwidth":
                    aggregated[key].bandwidth = m.value.number_value

        node_data: dict[str, list[NetworkDiagnosticResult]] = {}
        for r in sorted(aggregated.values(), key=lambda x: (x.node, x.axis)):
            node_data.setdefault(r.node, []).append(r)

        return node_data

    def _build_bandwidth_tables(
        self, node_data: dict[str, list[NetworkDiagnosticResult]]
    ) -> list:
        """Builds bandwidth tables for each node."""
        render_elements = []

        for node in sorted(node_data.keys()):
            node_results = node_data[node]

            table = create_styled_table(expand=False)
            table.add_column("Axis", justify="left")
            table.add_column("Bandwidth (GB/s)", justify="right")

            for r in node_results:
                bw_str = (
                    f"{r.bandwidth:.2f}" if r.bandwidth is not None else "N/A"
                )
                table.add_row(r.axis, bw_str)

            render_elements.append(Text(f"\nNode: {node}", style="bold yellow"))
            render_elements.append(table)

        return render_elements
