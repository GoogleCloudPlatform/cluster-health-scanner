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

from typing import Any
from rich.console import RenderableType, Group
from rich.text import Text
from chs.protos import bill_of_health_pb2 as pb
from chs.utils import create_styled_table
from . import rendering
from . import validation_analyzer


def has_validation_criteria(test_results: list[pb.TestResult]) -> bool:
    """Checks if any metric in the test results has validation criteria."""
    return validation_analyzer.has_validation_criteria(test_results)


def _parse_threshold(validation_criteria: pb.ValidationCriteria) -> str:
    """Parses expected value and comparator into a string."""
    return validation_analyzer.parse_threshold(validation_criteria)


def _parse_status(
    validation_criteria: pb.ValidationCriteria,
) -> tuple[str, str]:
    """Returns (status_str, status_style) for validation criteria."""
    return validation_analyzer.parse_status(validation_criteria)


def render_validation_results(
    test_results: list[pb.TestResult],
    plugin_manager: Any = None,
) -> RenderableType | None:
    """Renders tables showing only the metrics that have validation criteria.

    Args:
        test_results: List of TestResult protos.

    Returns:
        A Rich Renderable (Group of Tables) if validation criteria are present,
        otherwise None.
    """
    groups = validation_analyzer.extract_validation_data(
        test_results, max_display_nodes=5, plugin_manager=plugin_manager
    )
    if not groups:
        return None

    renderables = []

    for group in groups:
        header = (
            f"\nValidation Results: {group.test_name} "
            f"({group.test_level}) - [{group.nodes_str}]"
        )
        title_text = Text(
            header,
            style=f"bold {group.status_summary}",
        )
        renderables.append(title_text)

        table = create_styled_table(
            header_style=f"bold {group.status_summary}",
            expand=False,
            show_lines=False,
        )
        table.add_column("Target", style="magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold white")
        table.add_column("Unit", style="dim")
        table.add_column("Threshold", style="yellow")
        table.add_column("Status", style="bold")

        for row in group.rows:
            table.add_row(
                row.target,
                rendering.escape(row.metric),
                row.formatted_value,
                row.unit,
                row.threshold,
                Text(row.status, style=row.status_style),
                style=row.status_style,
            )

        renderables.append(table)
        renderables.append(Text())

    if (
        renderables
        and isinstance(renderables[-1], Text)
        and not renderables[-1].plain
    ):
        renderables.pop()

    return Group(*renderables)
