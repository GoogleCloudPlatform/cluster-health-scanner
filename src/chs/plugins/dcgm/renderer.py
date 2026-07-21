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
from chs.protos import bill_of_health_pb2 as pb
from chs.plugins.utils.rendering import extract_node_names
from rich.text import Text
from rich.console import Group, RenderableType
from chs.core.default_plugin.renderer import DefaultRenderer
from chs.core.plugin_base import ColumnDef, TableDataModel, PluginDetailReport
from chs.utils import theme


class DcgmRenderer(DefaultRenderer):
    def get_column_schema(self) -> list[ColumnDef]:
        return [
            ColumnDef("node", "Node"),
            ColumnDef("level", "Level"),
            ColumnDef("test_group", "Test Group"),
            ColumnDef("status", "Status"),
        ]

    def get_detail_data(
        self, test_results: list[pb.TestResult]
    ) -> PluginDetailReport | None:
        """Extracts structured DCGM detail data model."""
        if not test_results:
            return None

        node_results = defaultdict(list)
        for result in test_results:
            for rid in extract_node_names(result.target_resources):
                node_results[rid].append(result)

        rows = []
        for node_name, results in sorted(node_results.items()):
            level_to_metrics = self._get_level_metrics(results)
            if not level_to_metrics:
                rows.append(
                    {
                        "node": node_name,
                        "level": "N/A",
                        "test_group": "N/A",
                        "status": "No diagnostics found",
                        "passed_gpus": 0,
                        "total_gpus": 0,
                    }
                )
            else:
                first_node_row = True
                for level, tests in sorted(level_to_metrics.items()):
                    level_str = level.replace("-", " ")
                    first_level_row = True
                    for test_name, metrics in sorted(tests.items()):
                        passed_gpus = {s for s in metrics["passed"]}
                        failed_gpus = {s for s in metrics["failed"]}
                        total = len(passed_gpus) + len(failed_gpus)
                        if failed_gpus:
                            failed_str = ", ".join(sorted(failed_gpus))
                            status_plain = (
                                f"{len(passed_gpus)}/{total} GPUs passed "
                                f"({failed_str} failed)"
                            )
                        else:
                            status_plain = f"{total}/{total} GPUs passed"

                        disp_node = node_name if first_node_row else ""
                        disp_level = level_str if first_level_row else ""
                        rows.append(
                            {
                                "node": disp_node,
                                "level": disp_level,
                                "test_group": test_name,
                                "status": status_plain,
                                "passed_gpus": len(passed_gpus),
                                "total_gpus": total,
                                "failed_gpus": sorted(list(failed_gpus)),
                            }
                        )
                        first_node_row = False
                        first_level_row = False

        table_model = TableDataModel(
            title="DCGM",
            headers=self.get_column_schema(),
            rows=rows,
        )

        test_name = test_results[0].test_name
        test_level = test_results[0].test_level

        return PluginDetailReport(
            test_name=test_name,
            test_level=test_level,
            tables=[table_model],
        )

    def _get_level_metrics(self, results: list[pb.TestResult]) -> dict:
        """Groups sub-tests by level and provides a structured dictionary."""
        level_to_metrics = defaultdict(
            lambda: defaultdict(lambda: {"passed": set(), "failed": set()})
        )
        for result in results:
            level = result.test_level or "unknown"
            for m in result.metrics:
                status = (
                    "passed"
                    if "PASS" in m.value.string_value.upper()
                    else "failed"
                )
                sub_test = m.key[6:] if m.key.startswith("diag: ") else m.key

                ctx_field = m.metadata.fields.get("context")
                test_name = (
                    ctx_field.string_value
                    if ctx_field and ctx_field.string_value
                    else "unknown"
                )

                # Extracting GPU specific metrics GPU0, GPU1, etc.
                if sub_test.startswith("GPU"):
                    level_to_metrics[level][test_name][status].add(sub_test)
        return level_to_metrics

    def render_detail(
        self, test_results: list[pb.TestResult]
    ) -> RenderableType:
        if not test_results:
            return Text(
                "No detailed metrics available.", style=theme.CHS_WARNING_STYLE
            )

        report = self.get_detail_data(test_results)

        all_dcgm_versions = {
            m.value.string_value
            for r in test_results
            for m in r.metrics
            if m.key == "DCGM Version" and m.value.string_value
        }
        all_driver_versions = {
            m.value.string_value
            for r in test_results
            for m in r.metrics
            if m.key == "Driver Version Detected" and m.value.string_value
        }

        header_elements = [Text("\nDCGM", style=theme.CHS_TITLE_STYLE)]
        if all_dcgm_versions or all_driver_versions:
            v_parts = []
            if all_dcgm_versions:
                v_parts.append(
                    f"DCGM Version: {', '.join(sorted(all_dcgm_versions))}"
                )
            if all_driver_versions:
                v_parts.append(
                    f"Driver Version: {', '.join(sorted(all_driver_versions))}"
                )
            header_elements.append(
                Text.from_markup(
                    f"[{theme.CHS_NEUTRAL_STYLE}]Versions: "
                    f"{', '.join(v_parts)}[/]\n"
                )
            )

        table = theme.create_styled_table(
            "Node", "Level", "Test Group", "Status", expand=False
        )

        if report and report.tables:
            table_model = report.tables[0]
            current_node = None
            for row in table_model.rows:
                if (
                    current_node is not None
                    and row["node"]
                    and row["node"] != current_node
                ):
                    table.add_section()
                if row["node"]:
                    current_node = row["node"]

                table.add_row(
                    row["node"],
                    row["level"],
                    row["test_group"],
                    row["status"],
                )
            table.add_section()

        val_table = self.render_validation_results(test_results)
        if val_table:
            return Group(*header_elements, table, val_table)
        return Group(*header_elements, table)
