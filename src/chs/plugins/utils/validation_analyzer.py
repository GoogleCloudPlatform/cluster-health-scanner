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

"""Validation analysis and data extraction decoupled from UI rendering."""

import collections
from dataclasses import dataclass
from typing import Any
from chs.protos import bill_of_health_pb2 as pb
from . import rendering


@dataclass
class ValidationRowData:
    target: str
    metric: str
    value: Any
    formatted_value: str
    unit: str
    threshold: str
    status: str
    status_style: str


@dataclass
class ValidationGroupData:
    test_name: str
    test_level: str
    status_summary: str  # "green", "red", "yellow"
    targets: list[pb.TargetResource]
    nodes_str: str
    rows: list[ValidationRowData]


def has_validation_criteria(test_results: list[pb.TestResult]) -> bool:
    """Checks if any metric in the test results has validation criteria."""
    return any(
        m.HasField("validation_criteria") and m.validation_criteria.name
        for res in test_results
        for m in res.metrics
    )


def parse_threshold(validation_criteria: pb.ValidationCriteria) -> str:
    """Parses expected value and comparator into threshold string."""
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


def parse_status(
    validation_criteria: pb.ValidationCriteria,
) -> tuple[str, str]:
    """Returns (status_str, status_style) for validation criteria."""
    if validation_criteria.result_status == pb.RESULT_STATUS_SUCCESS:
        return "MET", "green"
    elif validation_criteria.result_status == pb.RESULT_STATUS_FAILED:
        return "NOT MET", "red"
    return "UNKNOWN", "yellow"


def extract_validation_data(
    test_results: list[pb.TestResult], max_display_nodes: int = 5
) -> list[ValidationGroupData]:
    """Extracts structured validation results independently of UI formatting."""
    if not test_results or not has_validation_criteria(test_results):
        return []

    grouped_results = collections.defaultdict(list)
    for res in test_results:
        grouped_results[(res.test_name, res.test_level)].append(res)

    validation_groups = []

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

        nodes_str = rendering.format_nodes(all_targets[:max_display_nodes])
        if len(all_targets) > max_display_nodes:
            more_cnt = len(all_targets) - max_display_nodes
            nodes_str += f", ... and {more_cnt} more"

        rows = []
        for res in group:
            target_str = rendering.format_nodes(res.target_resources)
            for metric in res.metrics:
                if not metric.HasField("validation_criteria"):
                    continue

                validation_criteria = metric.validation_criteria
                if not validation_criteria.name:
                    continue

                metric_value = rendering.parse_value(metric.value)
                threshold_str = parse_threshold(validation_criteria)
                status_str, status_style = parse_status(validation_criteria)

                metric_value_2f = (
                    f"{metric_value:.2f}"
                    if isinstance(metric_value, float)
                    else str(metric_value)
                )

                rows.append(
                    ValidationRowData(
                        target=target_str,
                        metric=metric.key,
                        value=metric_value,
                        formatted_value=metric_value_2f,
                        unit=metric.unit,
                        threshold=threshold_str,
                        status=status_str,
                        status_style=status_style,
                    )
                )

        validation_groups.append(
            ValidationGroupData(
                test_name=test_name,
                test_level=test_level,
                status_summary=border_color,
                targets=all_targets,
                nodes_str=nodes_str,
                rows=rows,
            )
        )

    return validation_groups
