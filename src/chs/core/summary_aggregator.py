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
from chs.utils.app_data import ChsAppData
from chs.plugins.utils.validation_analyzer import parse_threshold


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
    unmet_threshold: int
    failed_to_execute: int
    denominator: int
    entity_noun: str
    group_has_no_thresholds: bool
    explicit_skip_validation: bool
    unmet_threshold_resources: list[str]
    missing_metrics: int
    missing_metrics_resources: list[str]
    display_name: str
    failed_to_execute_resources: list[str] = field(default_factory=list)
    target_criteria: str = ""
    timeout_count: int = 0
    deadline_count: int = 0
    cancelled_count: int = 0


@dataclass
class RunSummaryData:
    run_id: str
    submission_time: str
    total_nodes: int
    test_summaries: list[TestGroupSummary]
    errors_and_warnings: list[HardErrorEvent] = field(default_factory=list)
    has_tests_or_resources_filter: bool = False


def filter_test_results(
    boh: pb.BillOfHealth,
    tests: list[str] | None = None,
    resources: list[str] | None = None,
) -> list[pb.TestResult]:
    filtered_results = []
    for result in boh.test_results:
        match_tests = True
        match_resources = True

        if tests:
            match_tests = result.test_name in tests
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
        plugin = pm.get_plugin(canonical_name)
        if plugin and hasattr(plugin, "_filter_validation_results"):
            sorted_nodes = sorted(
                list(
                    {
                        target.node_name
                        for res in results
                        for target in res.target_resources
                    }
                )
            )
            results = plugin._filter_validation_results(results, sorted_nodes)

        met_threshold = 0
        unmet_threshold = 0

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
        is_domain = "domain" in canonical_name.lower()
        is_grouped_entity = is_pairwise or is_domain
        unmet_threshold_resources = []
        missing_metrics_resources = []
        failed_to_execute_resources = []
        missing_metrics = 0
        failed_threshold = False
        failed_to_execute = 0
        timeout_count = 0
        deadline_count = 0
        cancelled_count = 0
        target_criteria = ""

        for res in results:
            for m in res.metrics:
                if (
                    m.HasField("validation_criteria")
                    and m.validation_criteria.name
                ):
                    vc = m.validation_criteria
                    t_str = parse_threshold(vc)
                    unit_str = f" {m.unit}" if m.unit else ""
                    target_criteria = f"Target: {vc.name} {t_str}{unit_str}"
                    break
            if target_criteria:
                break
        for res in results:
            failed_threshold = False
            for m in res.metrics:
                if (
                    m.HasField("validation_criteria")
                    and m.validation_criteria.result_status
                    == pb.RESULT_STATUS_FAILED
                ):
                    failed_threshold = True

            if res.result_status == pb.RESULT_STATUS_SUCCESS:
                met_threshold += (
                    1 if is_grouped_entity else len(res.target_resources)
                )
            elif res.result_status in (
                pb.RESULT_STATUS_TIMEOUT,
                pb.RESULT_STATUS_DEADLINE,
                pb.RESULT_STATUS_CANCELLED,
            ):
                cnt = (
                    1
                    if is_grouped_entity
                    else max(1, len(res.target_resources))
                )
                if res.result_status == pb.RESULT_STATUS_TIMEOUT:
                    timeout_count += cnt
                elif res.result_status == pb.RESULT_STATUS_DEADLINE:
                    deadline_count += cnt
                elif res.result_status == pb.RESULT_STATUS_CANCELLED:
                    cancelled_count += cnt

                if is_pairwise and len(res.target_resources) == 2:
                    failed_to_execute_resources.append(
                        f"{res.target_resources[0].node_name} <-> "
                        f"{res.target_resources[1].node_name}"
                    )
                else:
                    for t in res.target_resources:
                        failed_to_execute_resources.append(t.node_name)
            elif res.result_status == pb.RESULT_STATUS_UNKNOWN:
                pass
            else:
                count = 1 if is_grouped_entity else len(res.target_resources)
                if failed_threshold:
                    unmet_threshold += count
                    if is_pairwise and len(res.target_resources) == 2:
                        unmet_threshold_resources.append(
                            f"{res.target_resources[0].node_name} <-> "
                            f"{res.target_resources[1].node_name}"
                        )
                    else:
                        for t in res.target_resources:
                            unmet_threshold_resources.append(t.node_name)
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

        if is_pairwise:
            entity_noun = "pairs"
        elif is_domain:
            entity_noun = "domains"
        else:
            entity_noun = "nodes"

        denominator = len(results) if is_grouped_entity else test_nodes_count
        if group_has_no_thresholds:
            failed_to_execute = 0
            for r in results:
                if len(r.metrics) == 0:
                    cnt = (
                        1
                        if is_grouped_entity
                        else max(1, len(r.target_resources))
                    )
                    failed_to_execute += cnt
                    if is_pairwise and len(r.target_resources) == 2:
                        failed_to_execute_resources.append(
                            f"{r.target_resources[0].node_name} <-> "
                            f"{r.target_resources[1].node_name}"
                        )
                    else:
                        for t in r.target_resources:
                            failed_to_execute_resources.append(t.node_name)
        else:
            failed_to_execute = (
                denominator - met_threshold - unmet_threshold - missing_metrics
            )

        test_summaries.append(
            TestGroupSummary(
                canonical_name=canonical_name,
                test_level=test_level,
                met_threshold=met_threshold,
                unmet_threshold=unmet_threshold,
                failed_to_execute=failed_to_execute,
                denominator=denominator,
                entity_noun=entity_noun,
                group_has_no_thresholds=group_has_no_thresholds,
                explicit_skip_validation=explicit_skip_validation,
                unmet_threshold_resources=unmet_threshold_resources,
                missing_metrics=missing_metrics,
                missing_metrics_resources=missing_metrics_resources,
                display_name=display_name,
                failed_to_execute_resources=failed_to_execute_resources,
                target_criteria=target_criteria,
                timeout_count=timeout_count,
                deadline_count=deadline_count,
                cancelled_count=cancelled_count,
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
