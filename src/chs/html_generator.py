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

"""HTML Bill of Health Report Generator for Cluster Health Scanner (CHS)."""

import json
import logging
import os
from typing import Any


from chs.core.plugin_manager import PluginManager
from chs.core.summary_aggregator import aggregate_boh_summary
from chs.core.utils.validation_analyzer import extract_validation_data
from chs.protos import bill_of_health_pb2 as pb
from chs.utils.config import load_yaml_config
from chs.utils.tests import get_all_supported_tests_details
from chs.version import get_version

DEFAULT_TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__), "templates", "boh_template.html"
)

logger = logging.getLogger(__name__)


class HtmlReportGenerator:
    """Generates standalone, interactive HTML Bill of Health reports."""

    def __init__(
        self,
        template_path: str | None = None,
        topology: dict[str, Any] | None = None,
    ):
        self.template_path = template_path or DEFAULT_TEMPLATE_PATH
        self.topology = topology or {}

    def _extract_tier2_mappings(self) -> dict[str, list[str]]:
        """Extracts tier1-to-tier2 test mappings."""
        try:
            tiering_config = load_yaml_config("tiering.yaml")
            raw_tiering = tiering_config.get("tiering", {})
            return {
                k.lower(): [v.lower() for v in vals]
                for k, vals in raw_tiering.items()
            }
        except Exception as e:
            logger.debug(
                "Failed to safely extract tier2_mappings: %s", e, exc_info=True
            )
            return {}

    def generate_report_data(
        self,
        boh: pb.BillOfHealth,
        run_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Converts BillOfHealth protobuf into pure-Python JSON-serializable
        dictionary."""
        run_info = run_info or {}
        plugin_manager = PluginManager()
        summary = aggregate_boh_summary(boh)

        all_nodes = set()
        for tr in boh.test_results:
            for t in tr.target_resources:
                all_nodes.add(t.node_name)

        def _extract_nodes(resources: list[str]) -> set[str]:
            res = set()
            for r in resources:
                for n in r.split("<->"):
                    n_clean = n.strip()
                    if n_clean:
                        res.add(n_clean)
            return res

        degraded_nodes = set()
        missing_metrics_nodes = set()
        for ts in summary.test_summaries:
            if (
                not ts.group_has_no_thresholds
                and not ts.explicit_skip_validation
            ):
                degraded_nodes.update(
                    _extract_nodes(ts.unmet_threshold_resources)
                )
                missing_metrics_nodes.update(
                    _extract_nodes(ts.missing_metrics_resources)
                )
                missing_metrics_nodes.update(
                    _extract_nodes(ts.failed_to_execute_resources)
                )

        total_nodes = len(all_nodes)
        failed_count = sum(
            1 for e in summary.errors_and_warnings if e.type == "ERROR"
        )

        report_id = (
            summary.run_id
            or (
                boh.metadata.fields["run_id"].string_value
                if "run_id" in boh.metadata.fields
                else ""
            )
            or boh.report_id
            or run_info.get("report_id", "N/A")
        )

        created_at_str = run_info.get("created_at")
        if not created_at_str and boh.HasField("created_at"):
            created_at_str = boh.created_at.ToDatetime().strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            )
        created_at_str = created_at_str or "N/A"

        system_type = (
            boh.metadata.fields["system_type"].string_value
            if "system_type" in boh.metadata.fields
            else run_info.get("system_type", "KUBERNETES_CLUSTER")
        )

        skip_validation = (
            boh.metadata.fields["skip_validation"].bool_value
            if "skip_validation" in boh.metadata.fields
            else False
        )
        chs_version = (
            boh.chs_version or run_info.get("chs_version") or get_version()
        )
        tier1_to_tier2_tests = self._extract_tier2_mappings()

        executed_tests = {tr.test_name.lower() for tr in boh.test_results}

        test_long_descriptions = {}
        for t in get_all_supported_tests_details():
            long_desc = t.get("long_description", "") or t.get(
                "description", ""
            )
            if t.get("name"):
                test_long_descriptions[t["name"]] = long_desc
            if t.get("plugin_name"):
                test_long_descriptions[t["plugin_name"]] = long_desc

        test_suite_breakdown = []
        for ts in summary.test_summaries:
            long_description = test_long_descriptions.get(ts.canonical_name, "")
            evaluated_nodes = ts.met_threshold + ts.unmet_threshold
            badge_tooltip = None
            if ts.explicit_skip_validation:
                ratio_str = "Validation skipped"
                status_badge = "SKIPPED"
                badge_style = "badge-slate"
            elif ts.group_has_no_thresholds:
                executed_count = ts.denominator - ts.failed_to_execute
                ratio_str = (
                    f"{executed_count} / {ts.denominator} {ts.entity_noun}"
                    " executed (No thresholds set)"
                )
                if ts.failed_to_execute > 0:
                    status_badge = f"{ts.failed_to_execute} FAILED TO EXECUTE"
                    badge_style = "badge-failed"
                else:
                    status_badge = "EXECUTED"
                    badge_style = "badge-info"
            else:
                if evaluated_nodes == ts.denominator:
                    ratio_str = (
                        f"{ts.met_threshold} / {ts.denominator}"
                        f" {ts.entity_noun} met threshold"
                    )
                elif evaluated_nodes > 0:
                    ratio_str = (
                        f"{ts.met_threshold} / {evaluated_nodes} evaluated"
                        f" {ts.entity_noun} met threshold ({ts.denominator}"
                        " total)"
                    )
                else:
                    ratio_str = (
                        f"0 / {ts.denominator} {ts.entity_noun} evaluated"
                        f" against thresholds ({ts.denominator} total)"
                    )

                badge_tooltip = None

                if ts.unmet_threshold > 0:
                    canonical_lower = ts.canonical_name.lower()
                    if canonical_lower in tier1_to_tier2_tests:
                        tier2_list = tier1_to_tier2_tests[canonical_lower]
                        tier2_names = " or ".join(tier2_list)
                        status_badge = (
                            f"{ts.unmet_threshold} POTENTIALLY DEGRADED"
                        )

                        tier2_executed = any(
                            d in executed_tests for d in tier2_list
                        )
                        if tier2_executed:
                            badge_tooltip = (
                                f"Consider checking the {tier2_names} "
                                "test results for more details"
                            )
                        else:
                            badge_tooltip = (
                                f"Consider running the {tier2_names} test "
                                "for more details"
                            )
                    else:
                        status_badge = f"{ts.unmet_threshold} DEGRADED"
                    badge_style = "badge-degraded"
                elif ts.timeout_count > 0:
                    status_badge = f"{ts.timeout_count} TIMEOUT"
                    badge_style = "badge-failed"
                elif ts.deadline_count > 0:
                    status_badge = f"{ts.deadline_count} DEADLINE"
                    badge_style = "badge-failed"
                elif ts.failed_to_execute > 0:
                    status_badge = f"{ts.failed_to_execute} FAILED TO EXECUTE"
                    badge_style = "badge-failed"
                elif ts.missing_metrics > 0:
                    status_badge = f"{ts.missing_metrics} MISSING METRICS"
                    badge_style = "badge-slate"
                else:
                    status_badge = "PASSED"
                    badge_style = "badge-success"

            test_suite_breakdown.append(
                {
                    "display_name": ts.display_name,
                    "long_description": long_description,
                    "ratio_str": ratio_str,
                    "target_criteria": ts.target_criteria,
                    "status_badge": status_badge,
                    "badge_style": badge_style,
                    "badge_tooltip": badge_tooltip,
                    "unmet_threshold_resources": ts.unmet_threshold_resources,
                    "missing_metrics_resources": ts.missing_metrics_resources,
                    "failed_to_execute_resources": (
                        ts.failed_to_execute_resources
                    ),
                }
            )

        nodes_with_errors = {
            e.node for e in summary.errors_and_warnings if e.node
        }

        node_insights = []
        for node in sorted(all_nodes):
            node_tests = []
            node_has_unmet_threshold = False
            node_has_missing_metrics = False
            node_has_exec_error = False
            node_failed_tests = set()
            node_passed_tests = set()

            for tr in boh.test_results:
                target_nodes = {t.node_name for t in tr.target_resources}
                if node in target_nodes:
                    canonical_name = plugin_manager.resolve_test_name(
                        tr.test_name
                    )
                    plugin = plugin_manager.get_plugin(canonical_name)
                    has_custom_filter = bool(
                        plugin and hasattr(plugin, "_filter_validation_results")
                    )

                    eval_tr = tr
                    if has_custom_filter:
                        sorted_nodes = sorted(list(target_nodes))
                        filtered_list = plugin._filter_validation_results(
                            [tr], sorted_nodes
                        )
                        if filtered_list:
                            eval_tr = filtered_list[0]

                    display_name = (
                        f"{canonical_name} ({tr.test_level})"
                        if tr.test_level
                        else canonical_name
                    )

                    if has_custom_filter:
                        node_metrics = list(eval_tr.metrics)
                    else:
                        node_metrics = []
                        for m in eval_tr.metrics:
                            ctx = m.metadata.fields.get("context")
                            if ctx and ctx.string_value:
                                ctx_str = ctx.string_value
                                if any(
                                    t_node in ctx_str for t_node in target_nodes
                                ):
                                    if node not in ctx_str:
                                        continue
                            node_metrics.append(m)

                    has_unmet_metric = any(
                        m.HasField("validation_criteria")
                        and m.validation_criteria.result_status
                        == pb.RESULT_STATUS_FAILED
                        for m in node_metrics
                    )
                    if has_unmet_metric:
                        node_has_unmet_threshold = True
                        node_failed_tests.add(canonical_name.lower())

                    res_status = eval_tr.result_status

                    if res_status == pb.RESULT_STATUS_TIMEOUT:
                        status_str = "✕ TIMEOUT"
                        is_met = False
                        node_has_exec_error = True
                    elif res_status == pb.RESULT_STATUS_DEADLINE:
                        status_str = "✕ DEADLINE"
                        is_met = False
                        node_has_exec_error = True
                    elif res_status == pb.RESULT_STATUS_CANCELLED:
                        status_str = "✕ CANCELLED"
                        is_met = False
                        node_has_exec_error = True
                    elif has_unmet_metric:
                        status_str = "✕ DID NOT MEET THRESHOLD"
                        is_met = False
                    elif res_status == pb.RESULT_STATUS_FAILED:
                        status_str = "✕ FAILED TO EXECUTE"
                        is_met = False
                        node_has_exec_error = True
                    else:
                        has_val_criteria = any(
                            m.HasField("validation_criteria")
                            and m.validation_criteria.name
                            for m in node_metrics
                        )
                        has_metrics = any(
                            not (
                                m.key.endswith("Error Detected")
                                or m.key.endswith("Failure Detected")
                                or "Warning" in m.key
                            )
                            for m in node_metrics
                        )
                        if not has_val_criteria and has_metrics:
                            status_str = "• EXECUTED (No thresholds set)"
                            is_met = False
                        elif has_metrics:
                            status_str = "✓ MET THRESHOLD"
                            is_met = True
                            node_passed_tests.add(canonical_name.lower())
                        else:
                            status_str = "✕ MISSING METRICS"
                            is_met = False
                            node_has_missing_metrics = True

                    node_tests.append(
                        {
                            "display_name": display_name,
                            "status_str": status_str,
                            "is_met": is_met,
                        }
                    )

            validated_tier1_tests = set()
            for (
                tier1_test,
                tier2_tests,
            ) in tier1_to_tier2_tests.items():
                if any(
                    tier2_test in node_passed_tests
                    for tier2_test in tier2_tests
                ):
                    validated_tier1_tests.add(tier1_test)

            if node_has_unmet_threshold and node_failed_tests:
                if node_failed_tests.issubset(validated_tier1_tests):
                    node_has_unmet_threshold = False

            if node_has_unmet_threshold:
                tier1_tests = set(tier1_to_tier2_tests.keys())
                if node_failed_tests and node_failed_tests.issubset(
                    tier1_tests
                ):
                    node_status = "POTENTIALLY DEGRADED"
                else:
                    node_status = "DEGRADED"
                node_badge_style = "badge-degraded"
            elif node_has_exec_error or node_has_missing_metrics:
                node_status = "INCOMPLETE"
                node_badge_style = "badge-slate"
            else:
                node_status = "HEALTHY"
                node_badge_style = "badge-success"

            node_insights.append(
                {
                    "node_name": node,
                    "status": node_status,
                    "badge_style": node_badge_style,
                    "has_potential_errors": node in nodes_with_errors,
                    "has_threshold_breach": node_has_unmet_threshold,
                    "is_healthy": (node_status == "HEALTHY"),
                    "tests": node_tests,
                }
            )

        healthy_count = sum(
            1 for n in node_insights if n["status"] == "HEALTHY"
        )
        degraded_count = sum(
            1 for n in node_insights if n["status"] == "DEGRADED"
        )
        potentially_degraded_count = sum(
            1 for n in node_insights if n["status"] == "POTENTIALLY DEGRADED"
        )
        missing_metrics_count = sum(
            1 for n in node_insights if n["status"] == "INCOMPLETE"
        )

        detailed_reports = self._extract_detailed_reports(boh, plugin_manager)

        validation_groups = extract_validation_data(
            boh.test_results, plugin_manager=plugin_manager
        )
        validation_tables = [
            {
                "test_name": vg.test_name,
                "test_level": vg.test_level,
                "status_summary": vg.status_summary,
                "target_nodes": vg.target_nodes,
                "nodes_str": vg.nodes_str,
                "formula": vg.formula,
                "validation_name": vg.validation_name,
                "rows": [
                    {
                        "target": r.target,
                        "target_nodes": r.target_nodes,
                        "metric": r.metric,
                        "value": r.value,
                        "formatted_value": r.formatted_value,
                        "unit": r.unit,
                        "threshold": r.threshold,
                        "status": r.status,
                        "status_style": r.status_style,
                    }
                    for r in vg.rows
                ],
            }
            for vg in validation_groups
        ]

        return {
            "metadata": {
                "report_id": report_id,
                "chs_version": chs_version,
                "created_at": created_at_str,
                "system_type": system_type,
                "command_statement": run_info.get("command_statement", ""),
                "skip_validation": skip_validation,
            },
            "summary": {
                "total_nodes": total_nodes,
                "healthy_count": healthy_count,
                "degraded_count": degraded_count,
                "potentially_degraded_count": potentially_degraded_count,
                "missing_metrics_count": missing_metrics_count,
                "failed_count": failed_count,
                "total_test_suites": len(summary.test_summaries),
                "potential_errors_count": len(summary.errors_and_warnings),
            },
            "test_suite_breakdown": test_suite_breakdown,
            "potential_errors_list": [
                {
                    "type": e.type,
                    "category": e.category,
                    "message": e.message,
                    "node": e.node,
                    "test": e.test,
                }
                for e in summary.errors_and_warnings
            ],
            "node_insights": node_insights,
            "detailed_reports": detailed_reports,
            "validation_tables": validation_tables,
        }

    def _extract_detailed_reports(
        self, boh: pb.BillOfHealth, plugin_manager: PluginManager
    ) -> list[dict[str, Any]]:
        """Helper to extract structured detailed reports from plugins."""
        detailed_reports = []
        grouped_results: dict[tuple[str, str], list[pb.TestResult]] = {}
        for result in boh.test_results:
            canonical_name = plugin_manager.resolve_test_name(result.test_name)
            key = (canonical_name or result.test_name, result.test_level)
            grouped_results.setdefault(key, []).append(result)

        for (canonical_name, test_level), results in grouped_results.items():
            plugin = plugin_manager.get_plugin(canonical_name)
            if hasattr(plugin, "set_topology") and self.topology:
                plugin.set_topology(self.topology)
            detail_report = plugin.get_detail_data(results)
            if detail_report and detail_report.tables:
                tables_data = [
                    {
                        "title": table.title,
                        "headers": [
                            {"key": c.key, "label": c.label, "align": c.align}
                            for c in table.headers
                        ],
                        "rows": table.rows,
                    }
                    for table in detail_report.tables
                ]
                raw_name = results[0].test_name
                display_name = (
                    f"{raw_name} ({test_level})" if test_level else raw_name
                )
                target_nodes = sorted(
                    {
                        t.node_name
                        for res in results
                        for t in res.target_resources
                    }
                )

                detailed_reports.append(
                    {
                        "test_name": raw_name,
                        "test_level": test_level,
                        "display_name": display_name,
                        "canonical_name": canonical_name,
                        "target_nodes": target_nodes,
                        "tables": tables_data,
                    }
                )
        return detailed_reports

    def render_html(
        self,
        boh: pb.BillOfHealth,
        run_info: dict[str, Any] | None = None,
    ) -> str:
        """Renders HTML string by populating the template with serialized JSON
        data."""
        data_dict = self.generate_report_data(boh, run_info)
        raw_json = json.dumps(data_dict, indent=2)

        with open(self.template_path, "r", encoding="utf-8") as f:
            template_content = f.read()

        return template_content.replace(
            "__BOH_OVERVIEW_RAW_DATA_JSON__", raw_json
        ).replace("__BOH_RAW_DATA_JSON__", raw_json)

    def save_html(
        self,
        boh: pb.BillOfHealth,
        output_path: str,
        run_info: dict[str, Any] | None = None,
    ) -> str:
        """Generates and saves the HTML report to specified output path."""
        html_str = self.render_html(boh, run_info)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_str)
        return output_path
