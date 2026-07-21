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
from chs.core.plugin_base import ColumnDef, PluginDetailReport, TableDataModel
from chs.protos import bill_of_health_pb2 as pb
from chs.plugins.utils.rendering import extract_node_names, format_val
from chs.utils import theme
from . import validation_renderer

from rich.table import Table
from rich.text import Text
from rich.console import Group, RenderableType

BYTES_PER_GB = 1024**3


@dataclass
class PeakMetrics:
    max_bw: float
    max_size: str


def get_peak_metrics(metrics: list[pb.Metric]) -> PeakMetrics:
    max_bw = 0.0
    max_size = "Unknown"
    for m in metrics:
        if m.value.HasField("number_value") and "Bandwidth" in m.key:
            if m.value.number_value > max_bw:
                max_bw = m.value.number_value
                if "context" in m.metadata.fields:
                    try:
                        size_bytes = float(
                            m.metadata.fields["context"].string_value
                        )
                        max_size = f"{size_bytes / BYTES_PER_GB:.1f}"
                    except ValueError:
                        max_size = "Unknown"
    return PeakMetrics(max_bw, max_size)


def get_topological_context(
    node1: str, node2: str, node_to_location: dict[str, dict[str, str]]
) -> str:
    if not node_to_location:
        return "Unknown"

    loc1 = node_to_location.get(node1)
    loc2 = node_to_location.get(node2)

    if loc1 and loc2:
        if loc1["block"] != loc2["block"]:
            return "Inter-Block"
        elif loc1["subblock"] != loc2["subblock"]:
            return "Inter-Subblock"
        else:
            return "Intra-Subblock"

    return "Unknown"


def create_nccl_table(
    resource_column_name: str,
    include_topology: bool = False,
    has_thresholds: bool = False,
) -> Table:
    """Creates a standardized Table for NCCL metrics."""
    table = theme.create_styled_table(
        header_style=theme.CHS_HEADER_STYLE,
        expand=True,
    )
    table.add_column("Workload")
    table.add_column(resource_column_name, overflow="fold")
    if include_topology:
        table.add_column("Topological Context")
    table.add_column("Peak Bandwidth (GB/s)", justify="right")
    table.add_column("Msg Size (GB)", justify="right")
    if has_thresholds:
        table.add_column("Validation")
    return table


def parse_nccl_context_metadata(ctx_name: str) -> dict[str, float | str]:
    """Parses NCCL context (message size in bytes) into structured metadata."""
    try:
        size_bytes = float(ctx_name)
        return {
            "msg_size_bytes": size_bytes,
            "msg_size_gb": size_bytes / BYTES_PER_GB,
            "context": ctx_name,
        }
    except ValueError:
        return {"context": ctx_name}


def extract_nccl_detail_data(
    title: str,
    resource_column_name: str,
    test_results: list[pb.TestResult],
    get_resource_id_func,
    topology: dict[str, dict[str, list[str]]] | None = None,
) -> PluginDetailReport | None:
    """Extracts UI-decoupled NCCL detail table data model."""
    if not test_results:
        return None

    has_thresholds = not all(
        r.result_status == pb.RESULT_STATUS_UNKNOWN for r in test_results
    )

    node_to_location = {}
    if topology:
        for block, subblocks in topology.items():
            for subblock, nodes in subblocks.items():
                for node in nodes:
                    node_to_location[node] = {
                        "block": block,
                        "subblock": subblock,
                    }

    include_topology = False
    if node_to_location:
        for res in test_results:
            if res.target_resources and len(res.target_resources) >= 2:
                include_topology = True
                break

    headers = [
        ColumnDef("workload", "Workload"),
        ColumnDef("resource", resource_column_name),
    ]
    if include_topology:
        headers.append(ColumnDef("topology", "Topological Context"))
    headers.extend(
        [
            ColumnDef("peak_bw", "Peak Bandwidth (GB/s)", align="right"),
            ColumnDef("msg_size", "Msg Size (GB)", align="right"),
        ]
    )
    if has_thresholds:
        headers.append(ColumnDef("validation", "Validation"))

    rows = []
    for res in test_results:
        peak = get_peak_metrics(res.metrics)
        max_bw = peak.max_bw
        max_size = peak.max_size

        resource_id = get_resource_id_func(res)

        topo_context = "Unknown"
        if (
            node_to_location
            and res.target_resources
            and len(res.target_resources) >= 2
        ):
            names = extract_node_names(res.target_resources)
            topo_context = get_topological_context(
                names[0],
                names[1],
                node_to_location,
            )

        test_level = res.test_level if res.test_level else "Unknown"

        row_dict = {
            "workload": test_level,
            "resource": resource_id,
            "peak_bw": format_val(max_bw),
            "msg_size": max_size,
        }
        if include_topology:
            row_dict["topology"] = topo_context

        if has_thresholds:
            validation_status = "N/A"
            bw_metric = next(
                (
                    m
                    for m in res.metrics
                    if m.HasField("validation_criteria")
                    and "bandwidth" in m.validation_criteria.name.lower()
                ),
                None,
            )
            if bw_metric:
                if (
                    bw_metric.validation_criteria.result_status
                    == pb.RESULT_STATUS_SUCCESS
                ):
                    validation_status = "PASS"
                elif (
                    bw_metric.validation_criteria.result_status
                    == pb.RESULT_STATUS_FAILED
                ):
                    validation_status = "FAIL"
            else:
                if res.result_status == pb.RESULT_STATUS_FAILED:
                    validation_status = "ERROR"
            row_dict["validation"] = validation_status

        rows.append(row_dict)

    table_model = TableDataModel(
        title=title,
        headers=headers,
        rows=rows,
    )

    test_name = test_results[0].test_name
    test_level = test_results[0].test_level

    return PluginDetailReport(
        test_name=test_name,
        test_level=test_level,
        tables=[table_model],
    )


def render_nccl_detail(
    title: str,
    resource_column_name: str,
    test_results: list[pb.TestResult],
    get_resource_id_func,
    topology: dict[str, dict[str, list[str]]] | None = None,
) -> RenderableType:
    report = extract_nccl_detail_data(
        title,
        resource_column_name,
        test_results,
        get_resource_id_func,
        topology,
    )
    if not report or not report.tables:
        return Text(
            "No detailed metrics available.", style=theme.CHS_WARNING_STYLE
        )

    table_data = report.tables[0]
    include_topology = any(h.key == "topology" for h in table_data.headers)
    has_thresholds = any(h.key == "validation" for h in table_data.headers)

    table = create_nccl_table(
        resource_column_name,
        include_topology=include_topology,
        has_thresholds=has_thresholds,
    )

    for r in table_data.rows:
        row = [r["workload"], r["resource"]]
        if include_topology:
            row.append(r.get("topology", "Unknown"))
        row.extend([r["peak_bw"], r["msg_size"]])

        if has_thresholds:
            val_status = r.get("validation", "N/A")
            if val_status == "PASS":
                val_status = "[green]PASS[/]"
            elif val_status == "FAIL":
                val_status = "[red]FAIL[/]"
            elif val_status == "ERROR":
                val_status = "[yellow]ERROR[/]"
            row.append(val_status)

        table.add_row(*row)

    renderables = [Text(f"\n{title}", style=theme.CHS_TITLE_STYLE), table]

    val_table = validation_renderer.render_validation_results(test_results)
    if val_table:
        renderables.append(val_table)

    return Group(*renderables)
