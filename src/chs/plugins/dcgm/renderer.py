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

from chs.protos import bill_of_health_pb2 as pb
from rich.text import Text
from rich.console import Group, RenderableType
from collections import defaultdict
from chs.core.default_plugin.renderer import DefaultRenderer
from chs.utils import theme


class DcgmRenderer(DefaultRenderer):
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

        node_results = defaultdict(list)
        for result in test_results:
            for rid in result.target_resource_ids:
                node_results[rid].append(result)

        table = theme.create_styled_table(
            "Node", "Level", "Test Group", "Status", expand=False
        )

        for node_name, results in sorted(node_results.items()):
            level_to_metrics = self._get_level_metrics(results)

            first_node_row = True
            if not level_to_metrics:
                table.add_row(
                    node_name,
                    "N/A",
                    "N/A",
                    f"[{theme.CHS_NEUTRAL_STYLE}]No diagnostics found[/]",
                )
            else:
                for level, tests in sorted(level_to_metrics.items()):
                    level_str = level.replace("-", " ")
                    first_level_row = True
                    for test_name, metrics in sorted(tests.items()):
                        passed_gpus = {s for s in metrics["passed"]}
                        failed_gpus = {s for s in metrics["failed"]}
                        total = len(passed_gpus) + len(failed_gpus)
                        if failed_gpus:
                            failed_str = ", ".join(sorted(failed_gpus))
                            status_str = (
                                f"[{theme.CHS_FAILURE_STYLE}]"
                                f"{len(passed_gpus)}/{total} "
                                f"GPUs passed ({failed_str} failed)[/]"
                            )
                        else:
                            status_str = (
                                f"[{theme.CHS_SUCCESS_STYLE}]{total}/{total} "
                                "GPUs passed[/]"
                            )

                        table.add_row(
                            node_name if first_node_row else "",
                            level_str if first_level_row else "",
                            test_name,
                            status_str,
                        )
                        first_node_row = False
                        first_level_row = False
            table.add_section()

        val_table = self.render_validation_results(test_results)
        if val_table:
            return Group(*header_elements, table, val_table)
        return Group(*header_elements, table)
