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
from google.protobuf import text_format, json_format
from rich.console import Console
from rich.panel import Panel

from chs.protos import bill_of_health_pb2 as pb
from chs.core.plugin_manager import PluginManager
from chs import exceptions as ex
from chs.plugins.utils import rendering
from chs.utils import lifecycle
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
    ) -> RunSummaryData | None:
        """Extracts and normalizes data into a structured format."""
        return aggregate_boh_summary(
            boh=self.boh,
            boh_path=self.boh_path,
            plugin_manager=self.plugin_manager,
        )

    def show_summary_and_details(
        self,
    ) -> None:
        summary_data = self._aggregate_summary_data()
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

        if self.boh_uri:
            href = (
                self.boh_uri.replace(
                    "gs://", "https://storage.cloud.google.com/", 1
                )
                if self.boh_uri.startswith("gs://")
                else self.boh_uri
            )
            self.console.print(
                "Check the interactive health report in\n"
                f"[link={href}][cyan]{self.boh_uri}[/cyan][/link]"
            )

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
