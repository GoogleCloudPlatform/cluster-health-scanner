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
from rich.text import Text

from chs.utils import create_styled_table
from chs.protos import bill_of_health_pb2 as pb
from chs.core.utils.rendering import format_nodes
from chs.core.default_plugin.renderer import DefaultRenderer
from chs.core.plugin_base import ColumnDef, TableDataModel, PluginDetailReport


class NvidiaSmiRenderer(DefaultRenderer):
    """Renders details for nvidia-smi tests."""

    def get_column_schema(self) -> list[ColumnDef]:
        return [
            ColumnDef("node", "Node"),
            ColumnDef("driver_version", "Driver Version"),
            ColumnDef("cuda_version", "CUDA Version"),
            ColumnDef("gpu_count", "GPU Count", align="right"),
        ]

    def get_detail_data(
        self, test_results: list[pb.TestResult]
    ) -> PluginDetailReport | None:
        """Extracts structured NVIDIA SMI data model."""
        if not test_results:
            return None

        node_data = self._extract_node_data(test_results)

        summary_rows = [
            {
                "node": node,
                "driver_version": data["summary"]["Driver Version"],
                "cuda_version": data["summary"]["CUDA Version"],
                "gpu_count": str(data["summary"]["GPU Count"]),
            }
            for node, data in node_data.items()
        ]

        tables = [
            TableDataModel(
                title="NVIDIA SMI Summary",
                headers=self.get_column_schema(),
                rows=summary_rows,
            )
        ]

        # Per-GPU Detailed Tables
        gpu_headers = [
            ColumnDef("gpu", "GPU"),
            ColumnDef("name", "Name"),
            ColumnDef("pstate", "PState"),
            ColumnDef("power_draw", "Power Draw"),
            ColumnDef("power_limit", "Power Limit"),
            ColumnDef("graphics_clock", "Graphics Clock"),
            ColumnDef("memory_clock", "Memory Clock"),
        ]

        def safe_gpu_idx(x):
            try:
                return int(x)
            except ValueError:
                return 999

        for node, data in node_data.items():
            if not data["gpus"]:
                continue

            gpu_rows = []
            sorted_gpus = sorted(data["gpus"].keys(), key=safe_gpu_idx)
            for gpu_idx in sorted_gpus:
                gpu_info = data["gpus"][gpu_idx]
                gpu_rows.append(
                    {
                        "gpu": f"GPU {gpu_idx}",
                        "name": gpu_info.get("GPU Name", "N/A"),
                        "pstate": gpu_info.get("Performance State", "N/A"),
                        "power_draw": gpu_info.get("Power Draw", "N/A"),
                        "power_limit": gpu_info.get("Power Limit", "N/A"),
                        "graphics_clock": gpu_info.get("Graphics Clock", "N/A"),
                        "memory_clock": gpu_info.get("Memory Clock", "N/A"),
                    }
                )

            tables.append(
                TableDataModel(
                    title=f"GPU Details for {node}",
                    headers=gpu_headers,
                    rows=gpu_rows,
                )
            )

        test_name = test_results[0].test_name if test_results else "nvidia-smi"
        test_level = test_results[0].test_level if test_results else ""

        return PluginDetailReport(
            test_name=test_name,
            test_level=test_level,
            tables=tables,
        )

    def render_detail(self, test_results: list[pb.TestResult]) -> Group | Text:
        """Renders detail tables for nvidia-smi tests per node."""
        report = self.get_detail_data(test_results)
        if not report or not report.tables:
            return Text("No detailed metrics available.", style="yellow")

        render_elements = [Text(f"\n{report.tables[0].title}", style="bold")]
        summary_table = create_styled_table(expand=False)
        summary_table.add_column("Node", overflow="fold")
        summary_table.add_column("Driver Version")
        summary_table.add_column("CUDA Version")
        summary_table.add_column("GPU Count", justify="right")

        for r in report.tables[0].rows:
            summary_table.add_row(
                r["node"],
                r["driver_version"],
                r["cuda_version"],
                str(r["gpu_count"]),
            )

        render_elements.append(summary_table)

        for det_table_data in report.tables[1:]:
            render_elements.append(
                Text(f"\n{det_table_data.title}", style="bold")
            )
            det_table = create_styled_table(expand=False)
            det_table.add_column("GPU")
            det_table.add_column("Name")
            det_table.add_column("PState")
            det_table.add_column("Power Draw")
            det_table.add_column("Power Limit")
            det_table.add_column("Graphics Clock")
            det_table.add_column("Memory Clock")

            for r in det_table_data.rows:
                det_table.add_row(
                    r["gpu"],
                    r["name"],
                    r["pstate"],
                    r["power_draw"],
                    r["power_limit"],
                    r["graphics_clock"],
                    r["memory_clock"],
                )
            render_elements.append(det_table)

        val_table = self.render_validation_results(test_results)
        if val_table:
            render_elements.append(val_table)

        return Group(*render_elements)

    def _extract_node_data(self, test_results: list[pb.TestResult]) -> dict:
        """Extracts metrics from test results into a structured dict."""
        node_data = {}
        for res in test_results:
            nodes = (
                format_nodes(res.target_resources)
                if res.target_resources
                else "Unknown Node"
            )
            if nodes not in node_data:
                node_data[nodes] = {
                    "summary": {
                        "Driver Version": "N/A",
                        "CUDA Version": "N/A",
                        "GPU Count": "N/A",
                    },
                    "gpus": {},
                }

            for m in res.metrics:
                key = m.key
                val = (
                    m.value.string_value
                    if m.value.HasField("string_value")
                    else ""
                )
                if not val and m.value.HasField("number_value"):
                    num_val = m.value.number_value
                    if key == "GPU Count" and num_val.is_integer():
                        val = str(int(num_val))
                    else:
                        val = str(num_val)

                if not val:
                    continue

                ctx = (
                    m.metadata.fields["context"].string_value
                    if "context" in m.metadata.fields
                    else ""
                )

                if not ctx or ctx == "null":
                    # Global metrics
                    if key in ["Driver Version", "CUDA Version", "GPU Count"]:
                        node_data[nodes]["summary"][key] = val
                else:
                    # Per-GPU metrics
                    # ctx is "GPU 0", "GPU 1", etc.
                    gpu_idx = ctx.replace("GPU ", "")
                    if gpu_idx not in node_data[nodes]["gpus"]:
                        node_data[nodes]["gpus"][gpu_idx] = {}

                    node_data[nodes]["gpus"][gpu_idx][key] = val

                    # Fallback: if global driver is N/A, take it from GPU 0
                    if (
                        key == "Driver Version"
                        and node_data[nodes]["summary"]["Driver Version"]
                        == "N/A"
                    ):
                        node_data[nodes]["summary"]["Driver Version"] = val
        return node_data
