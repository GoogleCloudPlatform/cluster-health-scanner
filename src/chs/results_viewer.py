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
import json
from collections import defaultdict
from google.protobuf import text_format, json_format
from rich.console import Console
from rich.panel import Panel

from chs.protos import bill_of_health_pb2 as pb
from chs.core.plugin_manager import PluginManager
from chs import exceptions as ex
from chs.plugins.utils import rendering
from chs.utils import lifecycle
from chs.utils.tests import get_plugin_to_test_map  # noqa: F401
from chs.core.summary_aggregator import (
    RunSummaryData,
    aggregate_boh_summary,
    filter_test_results,
)

logger = logging.getLogger(__name__)


class ResultsViewer:
    """Renders Bill of Health results using Rich formatting via Plugins."""

    def __init__(
        self,
        boh_path: str | os.PathLike[str],
        boh_uri: str | None = None,
    ):
        """Initializes the viewer with the path to the Bill of Health JSON."""
        self.boh_path = boh_path
        self.boh_uri = boh_uri
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
        return filter_test_results(self.boh, tests, resources)

    def _aggregate_summary_data(
        self,
        tests: list[str] | None = None,
        resources: list[str] | None = None,
    ) -> RunSummaryData | None:
        """Extracts and normalizes data into a structured format."""
        return aggregate_boh_summary(
            boh=self.boh,
            boh_path=self.boh_path,
            tests=tests,
            resources=resources,
            plugin_manager=self.plugin_manager,
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
        self.console.print(
            f"Nodes under test: {summary_data.total_nodes} nodes"
        )
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
                total_nodes = summary.denominator
                evaluated_nodes = (
                    summary.met_threshold + summary.unmet_threshold
                )

                if evaluated_nodes > 0:
                    if evaluated_nodes == total_nodes:
                        self.console.print(
                            f"  • {summary.display_name:<30} : "
                            f"{summary.met_threshold} / "
                            f"{total_nodes} {summary.entity_noun} "
                            "met the threshold"
                        )
                    else:
                        self.console.print(
                            f"  • {summary.display_name:<30} : "
                            f"{summary.met_threshold} / "
                            f"{evaluated_nodes} evaluated "
                            f"{summary.entity_noun} "
                            f"met the threshold ({total_nodes} total)"
                        )
                else:
                    self.console.print(
                        f"  • {summary.display_name:<30} : "
                        f"0 {summary.entity_noun} evaluated against "
                        f"thresholds ({total_nodes} total)"
                    )

                if (
                    summary.unmet_threshold > 0
                    and summary.unmet_threshold_resources
                ):
                    unmet_threshold_unique = sorted(
                        list(set(summary.unmet_threshold_resources))
                    )
                    if len(unmet_threshold_unique) > 5:
                        self.console.print(
                            f"    - {summary.entity_noun.title()} That "
                            "Didn't Meet Threshold (First 5):"
                        )
                        for node in unmet_threshold_unique[:5]:
                            self.console.print(f"      * {node}")
                        self.console.print(
                            f"      ... and "
                            f"{len(unmet_threshold_unique) - 5} more."
                        )
                    else:
                        self.console.print(
                            f"    - {summary.entity_noun.title()} That "
                            "Didn't Meet Threshold:"
                        )
                        for node in unmet_threshold_unique:
                            self.console.print(f"      * {node}")

                if (
                    summary.missing_metrics > 0
                    and summary.missing_metrics_resources
                ):
                    missing_unique = sorted(
                        list(set(summary.missing_metrics_resources))
                    )
                    if len(missing_unique) > 5:
                        self.console.print(
                            f"    - {summary.entity_noun.title()} Missing "
                            "Required Metrics (First 5):"
                        )
                        for node in missing_unique[:5]:
                            self.console.print(f"      * {node}")
                        self.console.print(
                            f"      ... and " f"{len(missing_unique) - 5} more."
                        )
                    else:
                        self.console.print(
                            f"    - {summary.entity_noun.title()} Missing "
                            "Required Metrics:"
                        )
                        for node in missing_unique:
                            self.console.print(f"      * {node}")

                if summary.failed_to_execute > 0:
                    self.console.print(
                        f"    - {summary.entity_noun.title()} Failed "
                        f"To Execute: {summary.failed_to_execute}"
                    )

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

        base_msg = (
            "For full raw metrics and detailed results for all nodes, "
            "please use [bold cyan]chs results --details[/bold cyan]"
        )
        if self.boh_uri:
            self.console.print(f"{base_msg} or check the Bill of Health in:")
            href = (
                self.boh_uri.replace(
                    "gs://", "https://storage.cloud.google.com/", 1
                )
                if self.boh_uri.startswith("gs://")
                else self.boh_uri
            )
            self.console.print(
                f"[link={href}][cyan]{self.boh_uri}[/cyan][/link]\n"
            )
        else:
            self.console.print(base_msg)

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

            if detail is not None:
                self.console.print(detail)
                self.console.print()
