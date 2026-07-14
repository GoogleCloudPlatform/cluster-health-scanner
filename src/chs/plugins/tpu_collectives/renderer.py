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

from collections import defaultdict
from dataclasses import dataclass

from rich.console import Group
from rich.text import Text

from chs.core.default_plugin.renderer import DefaultRenderer
from chs.plugins.utils.rendering import extract_node_names
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

    def _should_only_show_first_host(self) -> bool:
        return True

    def _filter_validation_results(
        self, test_results: list[pb.TestResult], sorted_nodes: list[str]
    ) -> list[pb.TestResult]:
        """Filters the validation results to only include metrics
        for the first node."""
        if not test_results or not sorted_nodes:
            return test_results

        filtered_result = pb.TestResult()
        filtered_result.CopyFrom(test_results[0])

        first_node = sorted_nodes[0]

        filtered_metrics = [
            metric
            for metric in filtered_result.metrics
            if (context := metric.metadata.fields.get("context")) is None
            or first_node in context.string_value
        ]

        del filtered_result.metrics[:]
        filtered_result.metrics.extend(filtered_metrics)

        return [filtered_result]

    def render_detail(self, test_results: list[pb.TestResult]) -> Group | Text:
        """Renders detail tables for tpu-collectives tests."""
        if not test_results:
            return Text("No detailed metrics available.", style="yellow")

        parsed_map: dict[tuple[str, str, str], TpuCollectivesParsedResult] = {}

        for res in test_results:
            test_level = res.test_level if res.test_level else "check"

            for m in res.metrics:
                if m.key == "Peak Bandwidth (median)":
                    continue

                ctx = (
                    m.metadata.fields["context"].string_value
                    if "context" in m.metadata.fields
                    else "unknown"
                )
                if ctx == "unknown" or ctx == "null":
                    continue

                parts = ctx.split(",")
                resolved_node = "N/A"
                matrix_dim = ctx
                dtype = "N/A"
                if len(parts) == 3:
                    resolved_node = parts[0]
                    matrix_dim = parts[1]
                    dtype = parts[2]
                remaining_ctx = f"{matrix_dim},{dtype}"
                key = (resolved_node, test_level, remaining_ctx)
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

        all_target_resources = set()
        for res in test_results:
            if res.target_resources:
                all_target_resources.update(
                    extract_node_names(res.target_resources)
                )
        all_resources_str = (
            ", ".join(sorted(all_target_resources))
            if all_target_resources
            else "Unknown Node"
        )

        render_elements = [self._get_summary_title()]

        sorted_nodes = sorted(nodes_map.keys())
        if self._should_only_show_first_host() and sorted_nodes:
            nodes_to_render = [sorted_nodes[0]]
            header_node = all_resources_str
        else:
            nodes_to_render = sorted_nodes
            header_node = None

        for node in nodes_to_render:
            display_node = header_node if header_node else node
            render_elements.append(
                Text(
                    f"\nTarget Resource(s): {display_node}",
                    style="bold magenta",
                )
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

        validation_test_results = self._filter_validation_results(
            test_results, sorted_nodes
        )

        val_table = self.render_validation_results(validation_test_results)
        if val_table:
            render_elements.append(val_table)
        return Group(*render_elements)
