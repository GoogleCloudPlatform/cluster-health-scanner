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
from rich.console import Group
from rich.text import Text

from chs.utils import create_styled_table
from chs.protos import bill_of_health_pb2 as pb
from chs.plugins.utils.rendering import format_nodes
from chs.core.default_plugin.renderer import DefaultRenderer
from chs.core.plugin_base import ColumnDef, TableDataModel, PluginDetailReport


class TpuInfoRenderer(DefaultRenderer):
    """Renders details for tpu-info tests."""

    def get_column_schema(self) -> list[ColumnDef]:
        return [
            ColumnDef("node", "Node"),
            ColumnDef("libtpu_version", "Libtpu Version"),
            ColumnDef("accelerator_type", "Accelerator Type"),
            ColumnDef("jax_devices", "JAX Devices", align="right"),
        ]

    def get_detail_data(
        self, test_results: list[pb.TestResult]
    ) -> PluginDetailReport | None:
        """Extracts structured TPU Info data model."""
        if not test_results:
            return None

        node_data = self._extract_node_data(test_results)
        rows = []
        for node, data in node_data.items():
            summary = data["summary"]
            if not any(
                k in summary
                for k in ["Libtpu version", "Accelerator type", "JAX Devices"]
            ):
                continue
            rows.append(
                {
                    "node": node,
                    "libtpu_version": summary.get("Libtpu version", "N/A"),
                    "accelerator_type": summary.get("Accelerator type", "N/A"),
                    "jax_devices": summary.get("JAX Devices", "N/A"),
                }
            )

        table_model = TableDataModel(
            title="TPU Info Summary",
            headers=self.get_column_schema(),
            rows=rows,
        )

        test_name = test_results[0].test_name if test_results else "tpu-info"
        test_level = test_results[0].test_level if test_results else ""

        return PluginDetailReport(
            test_name=test_name,
            test_level=test_level,
            tables=[table_model],
        )

    def render_detail(self, test_results: list[pb.TestResult]) -> Group | Text:
        report = self.get_detail_data(test_results)
        if not report or not report.tables or not report.tables[0].rows:
            return Text("No detailed metrics available.", style="yellow")

        table_data = report.tables[0]
        table = create_styled_table(expand=False)
        table.add_column("Node", overflow="fold")
        table.add_column("Libtpu Version")
        table.add_column("Accelerator Type")
        table.add_column("JAX Devices", justify="right")

        for r in table_data.rows:
            table.add_row(
                r["node"],
                r["libtpu_version"],
                r["accelerator_type"],
                str(r["jax_devices"]),
            )

        elements = [
            Text(f"\n{table_data.title}", style="bold"),
            table,
        ]
        val_table = self.render_validation_results(test_results)
        if val_table:
            elements.append(val_table)
        return Group(*elements)

    def _extract_node_data(self, test_results: list[pb.TestResult]) -> dict:
        """Extracts metrics from test results into structured dicts by node."""
        node_data = {}
        for res in test_results:
            default_nodes = (
                format_nodes(res.target_resources)
                if res.target_resources
                else "Unknown Node"
            )

            for m in res.metrics:
                if m.value.HasField("string_value"):
                    val = m.value.string_value
                elif m.value.HasField("number_value"):
                    val = m.value.number_value
                    val = str(int(val)) if val.is_integer() else str(val)
                else:
                    continue

                if not val:
                    continue

                context = m.metadata.fields["context"].string_value or ""
                key = m.key

                node_match = re.search(r"Node ([a-zA-Z0-9-]+)", context)
                nodes = node_match.group(1) if node_match else default_nodes

                if nodes not in node_data:
                    node_data[nodes] = {
                        "summary": {},
                    }

                node_data[nodes]["summary"][key] = val

        return node_data
