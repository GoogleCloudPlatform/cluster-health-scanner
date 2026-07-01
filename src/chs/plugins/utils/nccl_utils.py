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
from chs.protos import bill_of_health_pb2 as pb
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
    table.add_column(resource_column_name)
    if include_topology:
        table.add_column("Topological Context")
    table.add_column("Peak Bandwidth (GB/s)", justify="right")
    table.add_column("Msg Size (GB)", justify="right")
    if has_thresholds:
        table.add_column("Validation")
    return table


def render_nccl_detail(
    title: str,
    resource_column_name: str,
    test_results: list[pb.TestResult],
    get_resource_id_func,
    topology: dict[str, dict[str, list[str]]] | None = None,
) -> RenderableType:
    if not test_results:
        return Text(
            "No detailed metrics available.", style=theme.CHS_WARNING_STYLE
        )

    has_thresholds = validation_renderer.has_validation_criteria(test_results)

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
            if res.target_resource_ids and len(res.target_resource_ids) >= 2:
                include_topology = True
                break

    table = create_nccl_table(
        resource_column_name,
        include_topology=include_topology,
        has_thresholds=has_thresholds,
    )

    for res in test_results:
        peak = get_peak_metrics(res.metrics)
        max_bw = peak.max_bw
        max_size = peak.max_size

        resource_id = get_resource_id_func(res)

        topo_context = "Unknown"
        if (
            node_to_location
            and res.target_resource_ids
            and len(res.target_resource_ids) >= 2
        ):
            topo_context = get_topological_context(
                res.target_resource_ids[0],
                res.target_resource_ids[1],
                node_to_location,
            )

        test_level = res.test_level if res.test_level else "Unknown"

        row = [test_level, resource_id]
        if include_topology:
            row.append(topo_context)
        row.extend([f"{max_bw:.2f}", max_size])

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
                    validation_status = "[green]PASS[/]"
                elif (
                    bw_metric.validation_criteria.result_status
                    == pb.RESULT_STATUS_FAILED
                ):
                    validation_status = "[red]FAIL[/]"
            row.append(validation_status)

        table.add_row(*row)

    renderables = [Text(f"\n{title}", style=theme.CHS_TITLE_STYLE), table]

    val_table = validation_renderer.render_validation_results(test_results)
    if val_table:
        renderables.append(val_table)

    return Group(*renderables)
