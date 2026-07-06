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

import collections

from rich.console import RenderableType, Group
from rich.text import Text
from chs.protos import bill_of_health_pb2 as pb
from chs.utils import create_styled_table
from . import rendering

MAX_DISPLAY_NODES = 5


def has_validation_criteria(test_results: list[pb.TestResult]) -> bool:
    """Checks if any metric in the test results has validation criteria."""
    return any(
        m.HasField("validation_criteria") and m.validation_criteria.name
        for res in test_results
        for m in res.metrics
    )


def _parse_threshold(validation_criteria: pb.ValidationCriteria) -> str:
    """Parses expected value and comparator into a string."""
    threshold_value = "N/A"
    if validation_criteria.HasField("expected_value"):
        threshold_value = rendering.parse_value(
            validation_criteria.expected_value
        )
    else:
        threshold_value = "complex"

    comparator_str = ""
    if validation_criteria.comparator == pb.Comparator.COMPARATOR_GTE:
        comparator_str = ">="
    elif validation_criteria.comparator == pb.Comparator.COMPARATOR_EQ:
        comparator_str = "=="
    elif validation_criteria.comparator == pb.Comparator.COMPARATOR_CONTAINS:
        comparator_str = "contains"

    return f"{comparator_str} {threshold_value}".strip()


def _parse_status(
    validation_criteria: pb.ValidationCriteria,
) -> tuple[str, str]:
    """Returns (status_str, status_style) for validation criteria."""
    if validation_criteria.result_status == pb.RESULT_STATUS_SUCCESS:
        return "MET", "green"
    elif validation_criteria.result_status == pb.RESULT_STATUS_FAILED:
        return "NOT MET", "red"
    return "UNKNOWN", "yellow"


def render_validation_results(
    test_results: list[pb.TestResult],
) -> RenderableType | None:
    """Renders tables showing only the metrics that have validation criteria.

    Args:
        test_results: List of TestResult protos.

    Returns:
        A Rich Renderable (Group of Tables) if validation criteria are present,
        otherwise None.
    """
    if not test_results or not has_validation_criteria(test_results):
        return None

    grouped_results = collections.defaultdict(list)
    for res in test_results:
        grouped_results[(res.test_name, res.test_level)].append(res)

    renderables = []

    for (test_name, test_level), group in grouped_results.items():
        if not has_validation_criteria(group):
            continue

        all_success = all(
            res.result_status == pb.RESULT_STATUS_SUCCESS for res in group
        )
        all_failed = all(
            res.result_status == pb.RESULT_STATUS_FAILED for res in group
        )

        if all_success:
            border_color = "green"
        elif all_failed:
            border_color = "red"
        else:
            border_color = "yellow"

        unique_targets = {}
        for res in group:
            for target in res.target_resources:
                unique_targets[target.SerializeToString()] = target
        all_targets = list(unique_targets.values())

        nodes_str = rendering.format_nodes(all_targets[:MAX_DISPLAY_NODES])
        if len(all_targets) > MAX_DISPLAY_NODES:
            nodes_str += (
                f", ... and {len(all_targets) - MAX_DISPLAY_NODES} more"
            )

        title_text = Text(
            f"\nValidation Results: {test_name}"
            f" ({test_level}) - [{nodes_str}]",
            style=f"bold {border_color}",
        )
        renderables.append(title_text)

        table = create_styled_table(
            header_style=f"bold {border_color}",
            expand=False,
            show_lines=True,
        )
        table.add_column("Target", style="magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold white")
        table.add_column("Unit", style="dim")
        table.add_column("Threshold", style="yellow")
        table.add_column("Status", style="bold")

        for res in group:
            target_str = rendering.format_nodes(res.target_resources)
            for metric in res.metrics:
                if not metric.HasField("validation_criteria"):
                    continue

                validation_criteria = metric.validation_criteria
                if not validation_criteria.name:
                    continue

                metric_value = rendering.parse_value(metric.value)
                threshold_str = _parse_threshold(validation_criteria)
                status_str, status_style = _parse_status(validation_criteria)

                prefix = rendering.format_context_prefix(metric)
                row_style = status_style

                metric_value_2f = (
                    f"{metric_value:.2f}"
                    if isinstance(metric_value, float)
                    else str(metric_value)
                )

                table.add_row(
                    target_str,
                    f"{prefix}{metric.key}",
                    metric_value_2f,
                    metric.unit,
                    threshold_str,
                    Text(status_str, style=status_style),
                    style=row_style,
                )

        renderables.append(table)
        renderables.append(Text())

    if not renderables:
        return None

    if (
        renderables
        and isinstance(renderables[-1], Text)
        and not renderables[-1].plain
    ):
        renderables.pop()

    return Group(*renderables)
