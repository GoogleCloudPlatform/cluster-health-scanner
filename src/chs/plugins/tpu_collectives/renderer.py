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
from chs.core.plugin_base import ColumnDef, PluginDetailReport, TableDataModel
from chs.plugins.utils.rendering import extract_node_names, format_val
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

    def get_column_schema(self) -> list[ColumnDef]:
        return [
            ColumnDef("node", "Node", align="left"),
            ColumnDef("operation", "Operation"),
            ColumnDef("matrix_dim", "Matrix Dim"),
            ColumnDef("dtype", "Dtype"),
            ColumnDef("bandwidth", "Bandwidth (GB/s)", align="right"),
        ]

    def get_detail_data(
        self, test_results: list[pb.TestResult]
    ) -> PluginDetailReport | None:
        """Extracts structured TPU Collectives detail data model."""
        if not test_results:
            return None

        tables = []
        for res in test_results:
            test_level = res.test_level if res.test_level else "check"
            parsed_map: dict[
                tuple[str, str, str], TpuCollectivesParsedResult
            ] = {}

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

            def get_matrix_dim_key(p: TpuCollectivesParsedResult) -> int:
                try:
                    return int(p.matrix_dim)
                except ValueError:
                    # Fallback to a large number so unknown/empty matrix_dim
                    # items are pushed to the end of the sorted list.
                    return 999999

            nodepool_nodes = extract_node_names(res.target_resources)
            all_resources_str = (
                ", ".join(sorted(nodepool_nodes))
                if nodepool_nodes
                else "Unknown Node"
            )

            sorted_nodes = sorted(nodes_map.keys())
            if self._should_only_show_first_host() and sorted_nodes:
                nodes_to_render = [sorted_nodes[0]]
                header_node = all_resources_str
            else:
                nodes_to_render = sorted_nodes
                header_node = None

            for node in nodes_to_render:
                display_node = header_node if header_node else node
                node_ops_map = defaultdict(list)
                for item in nodes_map[node]:
                    node_ops_map[item.op_type].append(item)

                rows = []
                for op_type in sorted(node_ops_map.keys()):
                    sorted_items = sorted(
                        node_ops_map[op_type], key=get_matrix_dim_key
                    )
                    for i, item in enumerate(sorted_items):
                        bw_str = format_val(item.bandwidth)
                        op_column = item.op_type if i == 0 else ""
                        rows.append(
                            {
                                "node": display_node,
                                "operation": op_column,
                                "matrix_dim": item.matrix_dim,
                                "dtype": item.dtype,
                                "bandwidth": bw_str,
                            }
                        )

                title_level = f" ({test_level})" if test_level else ""
                table_title = (
                    f"TPU Collectives{title_level} "
                    f"Bandwidth Detail ({display_node})"
                )
                tables.append(
                    TableDataModel(
                        title=table_title,
                        headers=self.get_column_schema(),
                        rows=rows,
                    )
                )

        test_name = test_results[0].test_name
        test_level = test_results[0].test_level

        return PluginDetailReport(
            test_name=test_name,
            test_level=test_level,
            tables=tables,
        )

    def _get_summary_title(self) -> Text:
        return Text(
            "\nTPU Collectives Performance Summary", style="bold underline cyan"
        )

    def _should_only_show_first_host(self) -> bool:
        return True

    def _filter_validation_results(
        self, test_results: list[pb.TestResult], sorted_nodes: list[str]
    ) -> list[pb.TestResult]:
        """Filters validation results to keep metrics for the first node of each
        nodepool (each TestResult)."""
        if not test_results or not sorted_nodes:
            return test_results

        filtered_results = []
        for res in test_results:
            filtered_result = pb.TestResult()
            filtered_result.CopyFrom(res)

            res_nodes = sorted(extract_node_names(res.target_resources))
            first_node = res_nodes[0] if res_nodes else sorted_nodes[0]

            filtered_metrics = [
                metric
                for metric in filtered_result.metrics
                if (context := metric.metadata.fields.get("context")) is None
                or first_node in context.string_value
            ]

            del filtered_result.metrics[:]
            filtered_result.metrics.extend(filtered_metrics)

            has_failed_metric = any(
                m.HasField("validation_criteria")
                and m.validation_criteria.result_status
                == pb.RESULT_STATUS_FAILED
                for m in filtered_metrics
            )
            if has_failed_metric:
                filtered_result.result_status = pb.RESULT_STATUS_FAILED
            elif any(
                m.HasField("validation_criteria") for m in filtered_metrics
            ):
                filtered_result.result_status = pb.RESULT_STATUS_SUCCESS

            filtered_results.append(filtered_result)

        return filtered_results

    def render_detail(self, test_results: list[pb.TestResult]) -> Group | Text:
        """Renders detail tables for tpu-collectives tests."""
        report = self.get_detail_data(test_results)
        if not report or not report.tables:
            return Text("No detailed metrics available.", style="yellow")

        render_elements = [self._get_summary_title()]

        for table_model in report.tables:
            render_elements.append(
                Text(f"\n{table_model.title}", style="bold magenta")
            )
            node_table = create_styled_table(expand=False)
            node_table.add_column("Operation", style="cyan")
            node_table.add_column("Matrix Dim", style="bold white")
            node_table.add_column("Dtype", style="dim")
            node_table.add_column(
                "Bandwidth (GB/s)", justify="right", style="green"
            )

            for r in table_model.rows:
                node_table.add_row(
                    r["operation"],
                    r["matrix_dim"],
                    r["dtype"],
                    r["bandwidth"],
                )
            render_elements.append(node_table)

        nodes_map_keys = sorted(
            list(
                {
                    res.target_resources[0].node_name
                    for res in test_results
                    if res.target_resources
                }
            )
        )
        validation_test_results = self._filter_validation_results(
            test_results, nodes_map_keys
        )

        val_table = self.render_validation_results(validation_test_results)
        if val_table:
            render_elements.append(val_table)
        return Group(*render_elements)
