"""Module for displaying Bill of Health results in the terminal."""

import logging
from chs.protos import bill_of_health_pb2 as pb
from google.protobuf import text_format, json_format
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from chs.core.plugin_manager import PluginManager
from chs import exceptions as ex

logger = logging.getLogger(__name__)


class ResultsViewer:
    """Renders Bill of Health results using Rich formatting via Plugins."""

    def __init__(self, boh_path: str):
        """Initializes the viewer with the path to the Bill of Health JSON."""
        self.boh_path = boh_path
        self.console = Console()
        self.boh = pb.BillOfHealth()
        self.plugin_manager = PluginManager()

        try:
            with open(self.boh_path, "r") as f:
                content = f.read()
                try:
                    json_format.Parse(
                        content, self.boh, ignore_unknown_fields=True
                    )
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
                match_resources = any(
                    r in result.target_resource_ids for r in resources
                )

            if match_tests and match_resources:
                filtered_results.append(result)
        return filtered_results

    def show_summary(
        self,
        tests: list[str] | None = None,
        resources: list[str] | None = None,
    ) -> None:
        """Displays the summary view of the Bill of Health."""
        if not self.boh.report_id:
            return

        # Header
        grid = Table.grid(expand=True)
        grid.add_column(justify="center", ratio=1)
        grid.add_column(justify="right")
        grid.add_row(
            f"[b]ID:[/b] {self.boh.report_id}",
            f"[b]v{self.boh.chs_version}[/b]",
        )

        self.console.print(
            Panel(
                grid,
                title="[bold white on blue]Cluster Bill of Health Report[/]",
                border_style="blue",
                padding=(1, 2),
            )
        )

        filtered_results = self._filter_results(tests, resources)
        if not filtered_results:
            self.console.print(
                Panel(
                    "[yellow]No test results found.[/yellow]",
                    title="Warning",
                    border_style="yellow",
                )
            )
            return

        self.console.print("\n[bold]Test Summaries[/bold]")
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
            summary = plugin.render_summary(results)
            if summary is not None:
                self.console.print(summary)

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
            detail = plugin.render_detail(results)
            if detail is not None:
                self.console.print(detail)
                self.console.print()
