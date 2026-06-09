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

import re
from dataclasses import dataclass

from chs.core.default_plugin.renderer import DefaultRenderer
from chs.protos import bill_of_health_pb2 as pb
from chs.utils import create_styled_table
from rich.console import Group
from rich.text import Text


@dataclass
class StreamParsedResult:
    node: str
    device_id: str
    copy_rate: float | None = None
    scale_rate: float | None = None
    add_rate: float | None = None
    triad_rate: float | None = None


class StreamRenderer(DefaultRenderer):
    """Renders details for stream tests."""

    validation_render_mode = "all"

    def _parse_metrics(
        self, test_results: list[pb.TestResult]
    ) -> tuple[dict[tuple[str, int], StreamParsedResult], dict[str, float]]:
        """Parses raw test results and extracts peak BW."""

        results_map: dict[tuple[str, int], StreamParsedResult] = {}

        node_peak_bw: dict[str, float] = {}

        # Regex to parse context like "Device 0 Copy" or "Device 0"
        context_regex = re.compile(
            r"Device\s+(?P<device_id>[0-9]+)(?:\s+(?P<function>\w+))?"
        )

        func_attr_map = {
            "Copy": "copy_rate",
            "Scale": "scale_rate",
            "Add": "add_rate",
            "Triad": "triad_rate",
        }

        for res in test_results:
            node = (
                ", ".join(res.target_resource_ids)
                if res.target_resource_ids
                else "Unknown Node"
            )

            for m in res.metrics:
                ctx_str = (
                    m.metadata.fields["context"].string_value
                    if "context" in m.metadata.fields
                    else ""
                )

                val = (
                    m.value.number_value
                    if m.value.HasField("number_value")
                    else None
                )

                if val is None:
                    continue

                if (
                    not ctx_str or ctx_str == "null"
                ) and m.key == "PEAK Bandwidth":
                    node_peak_bw[node] = val
                    continue

                match = context_regex.search(ctx_str)
                if not match:
                    continue

                device_id = match.group("device_id")
                function = match.group("function")

                key = (node, int(device_id))

                if key not in results_map:
                    results_map[key] = StreamParsedResult(
                        node=node, device_id=device_id
                    )

                # Convert rates from MB/s to GB/s
                if m.key == "Rate":
                    attr_name = func_attr_map.get(function)
                    if attr_name:
                        setattr(results_map[key], attr_name, val / 1000.0)

        return results_map, node_peak_bw

    def render_detail(self, test_results: list[pb.TestResult]) -> Group | Text:
        """Renders detail tables for stream tests per node and device."""
        if not test_results:
            return Text("No detailed metrics available.", style="yellow")

        results_map, node_peak_bw = self._parse_metrics(test_results)

        # Group results by node
        node_groups: dict[str, list[StreamParsedResult]] = {}
        for key in sorted(results_map.keys()):
            item = results_map[key]
            if item.node not in node_groups:
                node_groups[item.node] = []
            node_groups[item.node].append(item)

        group_elements = [Text("\nSTREAM Detail", style="bold")]

        for node, items in node_groups.items():
            peak = node_peak_bw.get(node)
            peak_str = (
                f" (Peak BW: {peak:.2f} GB/s)" if peak is not None else ""
            )

            group_elements.append(
                Text(f"\nNode: {node}{peak_str}", style="cyan")
            )

            table = create_styled_table(expand=False)
            table.add_column("Device")
            table.add_column("Copy (GB/s)", justify="right")
            table.add_column("Scale (GB/s)", justify="right")
            table.add_column("Add (GB/s)", justify="right")
            table.add_column("Triad (GB/s)", justify="right")

            def format_val(v):
                return f"{v:.2f}" if v is not None else "N/A"

            for item in items:
                table.add_row(
                    item.device_id,
                    format_val(item.copy_rate),
                    format_val(item.scale_rate),
                    format_val(item.add_rate),
                    format_val(item.triad_rate),
                )

            group_elements.append(table)

        val_table = self.render_validation_results(test_results)
        if val_table:
            group_elements.append(val_table)

        return Group(*group_elements)
