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

"""Module for displaying Bill of Health results in the terminal."""

import logging
import os
import datetime
import re
import json
from collections import defaultdict
from dataclasses import dataclass
from google.protobuf import text_format, json_format
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from chs.utils.theme import create_error_table as create_theme_error_table

from chs.protos import bill_of_health_pb2 as pb
from chs.core.plugin_manager import PluginManager
from chs import exceptions as ex
from chs.plugins.utils import rendering
from chs.utils import lifecycle
from chs.utils.app_data import ChsAppData

logger = logging.getLogger(__name__)


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
    display_name: str


@dataclass
class RunSummaryData:
    run_id: str
    submission_time: str
    total_nodes: int
    test_summaries: list[TestGroupSummary]
    errors_and_warnings: list[HardErrorEvent]
    has_tests_or_resources_filter: bool


class ResultsViewer:
    """Renders Bill of Health results using Rich formatting via Plugins."""

    def __init__(self, boh_path: str):
        """Initializes the viewer with the path to the Bill of Health JSON."""
        self.boh_path = boh_path
        self.console = Console()
        self.boh = pb.BillOfHealth()
        self.plugin_manager = PluginManager()
        self.topology = {}

        workspace_dir = os.path.dirname(self.boh_path)
        topo_path = os.path.join(workspace_dir, "topology.json")
        if os.path.exists(topo_path):
            try:
                with open(topo_path, "r") as f:
                    self.topology = json.load(f)
            except json.JSONDecodeError as e:
                logger.warning(
                    "Failed to parse topology JSON from %s: %s",
                    topo_path,
                    e,
                )
            except IOError as e:
                logger.warning(
                    "Failed to read topology file %s: %s",
                    topo_path,
                    e,
                )

        try:
            with open(self.boh_path, "r") as f:
                content = f.read()
            try:
                json_format.Parse(content, self.boh, ignore_unknown_fields=True)
            except json_format.ParseError:
                # Fallback to TextProto (in case it was written as such)
                text_format.Parse(content, self.boh)

        except FileNotFoundError as e:
            logger.error(
                "Bill of Health file not found at %s: %s", self.boh_path, e
            )
            raise ex.BillOfHealthNotFoundException(
                f"Bill of Health file not found at {self.boh_path}: {e}"
            ) from e
        except Exception as e:
            logger.error(
                "Error parsing Bill of Health from %s: %s", self.boh_path, e
            )
            raise ex.BillOfHealthLoadException(
                f"Error parsing Bill of Health from {self.boh_path}: {e}"
            ) from e

    def _filter_results(
        self, tests: list[str] | None = None, resources: list[str] | None = None
    ) -> list[pb.TestResult]:
        filtered_results = []
        for result in self.boh.test_results:
            match_tests = True
            match_resources = True

            if tests:
                match_tests = any(t == result.test_name for t in tests)
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

    def _append_errors(self, results: list[pb.TestResult]) -> None:
        """Checks for errors (keys ending in 'Detected') and prints them."""
        all_errors = []
        for res in results:
            for m in res.metrics:
                if m.key.endswith("Error Detected") or m.key.endswith(
                    "Failure Detected"
                ):
                    all_errors.append(m)

        if all_errors:
            self.console.print(
                Text("\nDetected Errors/Warnings", style="bold red")
            )
            error_data = []
            for m in all_errors:
                if m.value.HasField("string_value"):
                    val = m.value.string_value
                elif m.value.HasField("bool_value"):
                    val = str(m.value.bool_value)
                else:
                    val = "<unknown value type>"
                error_data.append((m.key, val))

            table = create_theme_error_table(error_data)
            self.console.print(table)

    def _aggregate_summary_data(
        self,
        tests: list[str] | None = None,
        resources: list[str] | None = None,
    ) -> RunSummaryData | None:
        """Extracts and normalizes data into a structured format."""
        if not self.boh.report_id:
            return None

        filtered_results = self._filter_results(tests, resources)

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
        if "skip_validation" in self.boh.metadata.fields:
            explicit_skip_validation = self.boh.metadata.fields[
                "skip_validation"
            ].bool_value

        report_id = self.boh.report_id
        run_id = os.path.basename(
            os.path.dirname(os.path.abspath(self.boh_path))
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
        elif self.boh.created_at:
            time_str = self.boh.created_at.ToDatetime().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

        all_nodes = set()
        for r in self.boh.test_results:
            for t in r.target_resources:
                all_nodes.add(t.node_name)
        total_nodes = len(all_nodes)

        grouped_results = {}
        for result in filtered_results:
            canonical_name = self.plugin_manager.resolve_test_name(
                result.test_name
            )
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

                if res.result_status == pb.RESULT_STATUS_SUCCESS:
                    met_threshold += (
                        1 if is_pairwise else len(res.target_resources)
                    )
                elif res.result_status in (
                    pb.RESULT_STATUS_TIMEOUT,
                    pb.RESULT_STATUS_CANCELLED,
                ):
                    pass
                else:
                    below_threshold += (
                        1 if is_pairwise else len(res.target_resources)
                    )
                    if is_pairwise and len(res.target_resources) == 2:
                        below_threshold_resources.append(
                            f"{res.target_resources[0].node_name} <-> "
                            f"{res.target_resources[1].node_name}"
                        )
                    else:
                        for t in res.target_resources:
                            below_threshold_resources.append(t.node_name)

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
                                    "Test exceeded time limits or was "
                                    "cancelled."
                                ),
                                node=t.node_name,
                                test=display_name,
                            )
                        )

            entity_noun = "pairs" if is_pairwise else "nodes"
            denominator = len(results) if is_pairwise else test_nodes_count
            failed_to_execute = denominator - met_threshold - below_threshold

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
                    display_name=display_name,
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

    def show_summary_and_details(
        self,
        tests: list[str] | None = None,
        resources: list[str] | None = None,
        show_details: bool = False,
    ) -> None:
        summary_data = self._aggregate_summary_data(tests, resources)
        if not summary_data:
            return

        if summary_data.run_id == "" and not summary_data.test_summaries:
            self.console.print(
                Panel(
                    "[yellow]No matching results found.[/yellow]",
                    title="Info",
                    border_style="yellow",
                )
            )
            return

        self.console.print(f"Run ID: {summary_data.run_id}")
        self.console.print(f"Submission Time: {summary_data.submission_time}")
        self.console.print(f"Scope: {summary_data.total_nodes} nodes")
        self.console.print("\nTest Suite Breakdown:")

        for summary in summary_data.test_summaries:
            if summary.explicit_skip_validation:
                self.console.print(
                    f"  • {summary.display_name:<30} : Executed on "
                    f"{summary.denominator} {summary.entity_noun} "
                    "(Validation skipped)"
                )
            elif summary.group_has_no_thresholds:
                self.console.print(
                    f"  • {summary.display_name:<30} : Executed on "
                    f"{summary.denominator} {summary.entity_noun} "
                    "(No thresholds set)"
                )
            else:
                if summary.failed_to_execute > 0:
                    self.console.print(
                        f"  • {summary.display_name:<30} : "
                        f"{summary.met_threshold} / "
                        f"{summary.denominator} {summary.entity_noun} "
                        "met the threshold "
                        f"({summary.failed_to_execute} failed to complete)"
                    )
                else:
                    self.console.print(
                        f"  • {summary.display_name:<30} : "
                        f"{summary.met_threshold} / "
                        f"{summary.denominator} {summary.entity_noun} "
                        "met the threshold"
                    )

                if (
                    summary.below_threshold > 0
                    and summary.below_threshold_resources
                ):
                    below_threshold_unique = sorted(
                        list(set(summary.below_threshold_resources))
                    )
                    if len(below_threshold_unique) > 5:
                        self.console.print(
                            f"    - {summary.entity_noun.title()} That "
                            "Didn't Meet Threshold (First 5):"
                        )
                        for node in below_threshold_unique[:5]:
                            self.console.print(f"      * {node}")
                        self.console.print(
                            f"      ... and "
                            f"{len(below_threshold_unique) - 5} more."
                        )
                    else:
                        self.console.print(
                            f"    - {summary.entity_noun.title()} That "
                            "Didn't Meet Threshold:"
                        )
                        for node in below_threshold_unique:
                            self.console.print(f"      * {node}")

        if summary_data.errors_and_warnings:
            self.console.print("\nPOTENTIAL ERRORS / WARNINGS:")
            self.console.print(
                "───────────────────────────────────────────────────────"
            )

            grouped_errors = defaultdict(list)
            for err in summary_data.errors_and_warnings:
                grouped_errors[(err.category, err.test)].append(err)

            sorted_errors = sorted(
                grouped_errors.items(), key=lambda x: len(x[1]), reverse=True
            )

            total_categories = len(sorted_errors)
            shown_categories = sorted_errors[:5]

            for (category, test_name), errs in shown_categories:
                err_type = errs[0].type
                msg = errs[0].message
                nodes = sorted(list(set(e.node for e in errs)))
                node_count = len(nodes)

                self.console.print(
                    f"{err_type} Category: {category} - Test: {test_name} "
                    f"({node_count} nodes)"
                )
                if msg:
                    self.console.print(f"  • Message: {msg}")

                if node_count > 5:
                    self.console.print("  • Affected Nodes (First 5 shown):")
                    self.console.print(f"    - {', '.join(nodes[:5])}")
                    self.console.print(
                        f"    ... and {node_count - 5} other nodes."
                    )
                else:
                    self.console.print("  • Affected Nodes:")
                    for n in nodes:
                        self.console.print(f"    - {n}")
                self.console.print()

            if total_categories > 5:
                hidden = total_categories - 5
                hidden_nodes = sum(
                    len(set(e.node for e in errs))
                    for _, errs in sorted_errors[5:]
                )
                self.console.print(
                    f"[!] ... and {hidden} other failure/warning categories "
                    f"affecting {hidden_nodes} nodes."
                )

            self.console.print(
                "───────────────────────────────────────────────────────"
            )
        else:
            if summary_data.has_tests_or_resources_filter:
                self.console.print(
                    "\nNo warnings or errors for the selected resources."
                )

        abs_path = os.path.abspath(self.boh_path)
        self.console.print(
            "For full raw metrics and detailed results for all nodes, "
            "please use [bold cyan]chs results --details[/bold cyan] "
            "or check the Bill of Health in:\n"
            f"{abs_path}\n"
        )

        if show_details:
            self.show_details(tests, resources)

    def show_details(
        self, tests: list[str] | None = None, resources: list[str] | None = None
    ) -> None:
        """Displays detailed views for specific tests or resources.

        Uses dedicated plugins for rendering details.
        """
        if not self.boh.report_id:
            return

        filtered_results = self._filter_results(tests, resources)

        if not filtered_results:
            self.console.print(
                Panel(
                    "[yellow]No matching results found.[/yellow]",
                    title="Info",
                    border_style="yellow",
                )
            )
            return

        grouped_results = {}
        for result in filtered_results:
            canonical_name = self.plugin_manager.resolve_test_name(
                result.test_name
            )
            key = (canonical_name, result.test_level)
            if key not in grouped_results:
                grouped_results[key] = []
            grouped_results[key].append(result)

        for (canonical_name, test_level), results in grouped_results.items():
            plugin = self.plugin_manager.get_plugin(canonical_name)
            if hasattr(plugin, "set_topology"):
                plugin.set_topology(self.topology)
            valid_results = []
            for res in results:
                if (
                    res.result_status
                    in lifecycle.TIME_EXCEEDED_PB_STATUSES.values()
                ):
                    nodes_str = rendering.format_nodes(res.target_resources)
                    status_name = pb.ResultStatus.Name(res.result_status)
                    status_clean = status_name.replace("RESULT_STATUS_", "")
                    self.console.print(
                        f"[bold red]{status_clean}[/bold red] • "
                        f"{res.test_name} ({res.test_level}) on {nodes_str}"
                    )
                else:
                    valid_results.append(res)

            if not valid_results:
                self.console.print()
                continue

            detail = plugin.render_detail(valid_results)

            has_errors = any(
                (
                    m.key.endswith("Error Detected")
                    or m.key.endswith("Failure Detected")
                )
                for res in valid_results
                for m in res.metrics
            )

            if detail is not None or has_errors:
                if detail is not None:
                    self.console.print(detail)

                self._append_errors(valid_results)
                self.console.print()
