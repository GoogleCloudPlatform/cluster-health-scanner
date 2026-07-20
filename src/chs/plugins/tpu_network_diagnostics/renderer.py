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

from rich.console import Group
from rich.table import Table
from rich.text import Text
from chs.utils import create_styled_table
from chs.protos import bill_of_health_pb2 as pb
from chs.plugins.utils.rendering import format_nodes
from chs.core.default_plugin.renderer import DefaultRenderer


class TpuNetworkDiagnosticsRenderer(DefaultRenderer):
    """Renders details for tpu-network-diagnostics tests."""

    def render_detail(self, test_results: list[pb.TestResult]) -> Group | Text:
        """Renders detail tables for network diagnostic tests per node."""
        if not test_results:
            return Text("No detailed metrics available.", style="yellow")

        node_bandwidths = self._extract_node_data(test_results)
        if not node_bandwidths:
            return Text(
                "No parsed Network Diagnostics metrics found.", style="yellow"
            )

        elements = [Text("\nTPU Network Diagnostics Detail", style="bold cyan")]
        elements.extend(self._build_bandwidth_table(node_bandwidths))

        val_table = self.render_validation_results(test_results)
        if val_table:
            elements.append(val_table)

        return Group(*elements)

    def _extract_node_data(
        self, test_results: list[pb.TestResult]
    ) -> dict[str, float]:
        """Extracts node name to minimum bandwidth mapping from test results."""
        node_bandwidths: dict[str, float] = {}

        for test_result in test_results:
            default_node = (
                format_nodes(test_result.target_resources)
                if test_result.target_resources
                else "unknown"
            )
            for metric in test_result.metrics:
                if not metric.value.HasField("number_value"):
                    continue

                if metric.key not in ("Bandwidth", "Axis Bandwidth"):
                    continue

                context_field = metric.metadata.fields.get("context")
                context_name = (
                    context_field.string_value
                    if context_field
                    else default_node
                )
                node_name = context_name.split(" Axis-")[0].strip()
                bandwidth_val = metric.value.number_value

                if (
                    node_name not in node_bandwidths
                    or bandwidth_val < node_bandwidths[node_name]
                ):
                    node_bandwidths[node_name] = bandwidth_val

        return node_bandwidths

    def _build_bandwidth_table(
        self, node_bandwidths: dict[str, float]
    ) -> list[Table]:
        """Builds a summary table showing node name vs min bandwidth."""
        table = create_styled_table(expand=False)
        table.add_column("Node Name", justify="left")
        table.add_column("Bandwidth (GB/s)", justify="right")

        for node_name in sorted(node_bandwidths.keys()):
            bandwidth_val = node_bandwidths[node_name]
            formatted_bandwidth = f"{bandwidth_val:.2f}"
            table.add_row(node_name, formatted_bandwidth)

        return [table]
