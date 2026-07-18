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

"""Aggregates Bill of Health summary metrics independently of rendering."""

import os
import re
import datetime
from dataclasses import dataclass, field
from chs.protos import bill_of_health_pb2 as pb
from chs.core.plugin_manager import PluginManager
from chs.utils import lifecycle
from chs.utils.app_data import ChsAppData
from chs.utils.tests import get_plugin_to_test_map


@dataclass
class HardErrorEvent:
    type: str
    category: str
    message: str
    node: str
    test: str


@dataclass
class TestGroupSummary:
    canonical_name: str
    test_level: str
    met_threshold: int
    below_threshold: int
    failed_to_execute: int
    denominator: int
    entity_noun: str
    group_has_no_thresholds: bool
    explicit_skip_validation: bool
    below_threshold_resources: list[str]
    missing_metrics: int
    missing_metrics_resources: list[str]
    display_name: str
    failed_to_execute_resources: list[str] = field(default_factory=list)


@dataclass
class RunSummaryData:
    run_id: str
    submission_time: str
    total_nodes: int
    test_summaries: list[TestGroupSummary]
    errors_and_warnings: list[HardErrorEvent]
    has_tests_or_resources_filter: bool


def filter_test_results(
    boh: pb.BillOfHealth,
    tests: list[str] | None = None,
    resources: list[str] | None = None,
) -> list[pb.TestResult]:
    import sys

    rv = sys.modules.get("chs.results_viewer")
    if rv and hasattr(rv, "get_plugin_to_test_map"):
        plugin_map = rv.get_plugin_to_test_map()
    else:
        plugin_map = get_plugin_to_test_map()
    filtered_results = []
    for result in boh.test_results:
        match_tests = True
        match_resources = True

        if tests:
            base_name = plugin_map.get(result.test_name, result.test_name)
            match_tests = any(t in (result.test_name, base_name) for t in tests)
        if resources:
            match_resources = False
            for target in result.target_resources:
                if target.node_name in resources or any(
                    res in target.node_name for res in resources
                ):
                    match_resources = True
                    break

        if match_tests and match_resources:
            filtered_results.append(result)
    return filtered_results


def aggregate_boh_summary(
    boh: pb.BillOfHealth,
    boh_path: str | os.PathLike[str] | None = None,
    tests: list[str] | None = None,
    resources: list[str] | None = None,
    plugin_manager: PluginManager | None = None,
) -> RunSummaryData | None:
    """Extracts and aggregates Bill of Health data into a structured model."""
    if not boh.report_id:
        return None

    pm = plugin_manager or PluginManager()

    filtered_results = filter_test_results(boh, tests, resources)

    if not filtered_results:
        return RunSummaryData(
            run_id="",
            submission_time="",
            total_nodes=0,
            test_summaries=[],
            errors_and_warnings=[],
            has_tests_or_resources_filter=bool(tests or resources),
        )

    explicit_skip_validation = False
    if "skip_validation" in boh.metadata.fields:
        explicit_skip_validation = boh.metadata.fields[
            "skip_validation"
        ].bool_value

    report_id = boh.report_id
    run_id = (
        os.path.basename(os.path.dirname(os.path.abspath(boh_path)))
        if boh_path
        else report_id
    )

    submission_time = None
    app_data = ChsAppData()
    run_entry = app_data.get_run(report_id)
    if run_entry:
        run_id = run_entry.get("run_id", run_id)
        submission_time = run_entry.get("submission_time", submission_time)

    time_str = ""
    if submission_time:
        try:
            dt = datetime.datetime.fromisoformat(submission_time)
            time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            time_str = str(submission_time)
    elif boh.created_at:
        time_str = boh.created_at.ToDatetime().strftime("%Y-%m-%d %H:%M:%S")

    all_nodes = set()
    for r in boh.test_results:
        for t in r.target_resources:
            all_nodes.add(t.node_name)
    total_nodes = len(all_nodes)

    grouped_results = {}
    for result in filtered_results:
        canonical_name = pm.resolve_test_name(result.test_name)
        key = (canonical_name, result.test_level)
        if key not in grouped_results:
            grouped_results[key] = []
        grouped_results[key].append(result)

    errors_and_warnings = []
    test_summaries = []

    for (canonical_name, test_level), results in grouped_results.items():
        met_threshold = 0
        below_threshold = 0

        group_all_unknown = all(
            r.result_status == pb.RESULT_STATUS_UNKNOWN for r in results
        )
        group_has_no_thresholds = group_all_unknown and len(results) > 0

        display_name = (
            f"{canonical_name} ({test_level})"
            if test_level
            else str(canonical_name)
        )

        test_nodes = set()
        for res in results:
            for t in res.target_resources:
                test_nodes.add(t.node_name)

            for m in res.metrics:
                if (
                    m.key.endswith("Error Detected")
                    or m.key.endswith("Failure Detected")
                    or "Warning" in m.key
                ):
                    val = ""
                    if m.value.HasField("string_value"):
                        val = m.value.string_value
                    elif m.value.HasField("bool_value"):
                        val = str(m.value.bool_value)

                    is_warn = "Warning" in m.key
                    err_type = "WARNING" if is_warn else "FAIL"

                    category = re.sub(
                        r"(?i)\s*(error|failure|warning)\s*detected",
                        "",
                        m.key,
                    ).strip()

                    for t in res.target_resources:
                        errors_and_warnings.append(
                            HardErrorEvent(
                                type=err_type,
                                category=category,
                                message=val,
                                node=t.node_name,
                                test=display_name,
                            )
                        )

        test_nodes_count = len(test_nodes)
        is_pairwise = "pairwise" in canonical_name.lower()
        below_threshold_resources = []
        missing_metrics_resources = []
        failed_to_execute_resources = []
        missing_metrics = 0
        failed_threshold = False
        failed_to_execute = 0
        for res in results:
            has_real_metrics = any(
                not (
                    m.key.endswith("Error Detected")
                    or m.key.endswith("Failure Detected")
                    or "Warning" in m.key
                )
                for m in res.metrics
            )

            failed_threshold = False
            for m in res.metrics:
                if (
                    m.HasField("validation_criteria")
                    and m.validation_criteria.result_status
                    == pb.RESULT_STATUS_FAILED
                ):
                    failed_threshold = True

            if res.result_status == pb.RESULT_STATUS_SUCCESS:
                met_threshold += 1 if is_pairwise else len(res.target_resources)
            elif res.result_status in (
                pb.RESULT_STATUS_TIMEOUT,
                pb.RESULT_STATUS_DEADLINE,
                pb.RESULT_STATUS_CANCELLED,
            ):
                if is_pairwise and len(res.target_resources) == 2:
                    failed_to_execute_resources.append(
                        f"{res.target_resources[0].node_name} <-> "
                        f"{res.target_resources[1].node_name}"
                    )
                else:
                    for t in res.target_resources:
                        failed_to_execute_resources.append(t.node_name)
            else:
                count = 1 if is_pairwise else len(res.target_resources)
                if failed_threshold:
                    below_threshold += count
                    if is_pairwise and len(res.target_resources) == 2:
                        below_threshold_resources.append(
                            f"{res.target_resources[0].node_name} <-> "
                            f"{res.target_resources[1].node_name}"
                        )
                    else:
                        for t in res.target_resources:
                            below_threshold_resources.append(t.node_name)
                else:
                    missing_metrics += count
                    if is_pairwise and len(res.target_resources) == 2:
                        missing_metrics_resources.append(
                            f"{res.target_resources[0].node_name} <-> "
                            f"{res.target_resources[1].node_name}"
                        )
                    else:
                        for t in res.target_resources:
                            missing_metrics_resources.append(t.node_name)

            if (
                not has_real_metrics
                and res.result_status
                in lifecycle.TIME_EXCEEDED_PB_STATUSES.values()
            ):
                status_name = pb.ResultStatus.Name(res.result_status)
                status_clean = status_name.replace("RESULT_STATUS_", "")
                for t in res.target_resources:
                    errors_and_warnings.append(
                        HardErrorEvent(
                            type="FAIL",
                            category=f"Status: {status_clean}",
                            message=(
                                "Test exceeded time limits or was " "cancelled."
                            ),
                            node=t.node_name,
                            test=display_name,
                        )
                    )

        entity_noun = "pairs" if is_pairwise else "nodes"
        denominator = len(results) if is_pairwise else test_nodes_count
        failed_to_execute = (
            denominator - met_threshold - below_threshold - missing_metrics
        )

        test_summaries.append(
            TestGroupSummary(
                canonical_name=canonical_name,
                test_level=test_level,
                met_threshold=met_threshold,
                below_threshold=below_threshold,
                failed_to_execute=failed_to_execute,
                denominator=denominator,
                entity_noun=entity_noun,
                group_has_no_thresholds=group_has_no_thresholds,
                explicit_skip_validation=explicit_skip_validation,
                below_threshold_resources=below_threshold_resources,
                missing_metrics=missing_metrics,
                missing_metrics_resources=missing_metrics_resources,
                display_name=display_name,
                failed_to_execute_resources=failed_to_execute_resources,
            )
        )

    return RunSummaryData(
        run_id=run_id,
        submission_time=time_str,
        total_nodes=total_nodes,
        test_summaries=test_summaries,
        errors_and_warnings=errors_and_warnings,
        has_tests_or_resources_filter=bool(tests or resources),
    )
