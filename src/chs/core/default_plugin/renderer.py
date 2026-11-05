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

from chs.core.utils import rendering, validation_renderer
from chs.protos import bill_of_health_pb2 as pb
from rich.console import RenderableType, Group
from rich.panel import Panel
from rich.text import Text
from chs.utils import create_styled_table
from rich.markup import escape


class DefaultRenderer:
    """Default visualization logic using Rich."""

    def render_raw_metrics(
        self, test_results: list[pb.TestResult]
    ) -> RenderableType:
        """Renders raw metrics in a table."""
        if not test_results:
            return Text("No detailed metrics available.", style="yellow")

        panels = []
        for test_result in test_results:
            is_success = test_result.result_status == pb.RESULT_STATUS_SUCCESS
            border_color = "green" if is_success else "red"

            table = create_styled_table(
                header_style=f"bold {border_color}",
                expand=True,
                show_lines=True,
            )
            table.add_column("Key", style="cyan")
            table.add_column("Value", style="bold white")
            table.add_column("Unit", style="dim")

            for metric in test_result.metrics:
                val = rendering.parse_value(metric.value)

                prefix = rendering.format_context_prefix(metric)

                table.add_row(
                    escape(f"{prefix}{metric.key}"), str(val), metric.unit
                )

            nodes_str = rendering.format_nodes(test_result.target_resources)
            panels.append(
                Panel(
                    table
                    if test_result.metrics
                    else Text("No metrics generated."),
                    title=Text(
                        f"Raw Metrics: {test_result.test_name} "
                        f"({test_result.test_level}) - [{nodes_str}]"
                    ),
                    expand=True,
                    border_style=border_color,
                )
            )

        return Group(*panels)

    def render_validation_results(
        self, test_results: list[pb.TestResult]
    ) -> RenderableType | None:
        """Renders validation results using the shared utility."""
        return validation_renderer.render_validation_results(test_results)

    def render_detail(
        self, test_results: list[pb.TestResult]
    ) -> RenderableType:
        raw_metrics = self.render_raw_metrics(test_results)
        validation = self.render_validation_results(test_results)
        if validation:
            return Group(raw_metrics, validation)
        return raw_metrics
