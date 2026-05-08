"""Module for displaying Bill of Health results in the terminal."""

import logging
import os
import json
from google.protobuf import text_format, json_format
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from chs.utils.theme import create_error_table as create_theme_error_table

from chs.protos import bill_of_health_pb2 as pb
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
                match_resources = any(
                    r in result.target_resource_ids for r in resources
                )

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

        from chs.core.default_plugin import DefaultPlugin
        from chs.core.default_plugin.renderer import DefaultRenderer

        for (canonical_name, test_level), results in grouped_results.items():
            plugin = self.plugin_manager.get_plugin(canonical_name)
            if hasattr(plugin, "set_topology"):
                plugin.set_topology(self.topology)
            detail = plugin.render_detail(results)

            has_errors = any(
                (
                    m.key.endswith("Error Detected")
                    or m.key.endswith("Failure Detected")
                )
                for res in results
                for m in res.metrics
            )

            has_thresholds = any(
                m.HasField("validation_criteria") and m.validation_criteria.name
                for res in results
                for m in res.metrics
            )

            if detail is not None or has_errors:
                if detail is not None:
                    self.console.print(detail)

                if (
                    detail is not None
                    and not isinstance(plugin, DefaultPlugin)
                    and has_thresholds
                ):
                    raw_detail = DefaultRenderer().render_detail(results)
                    self.console.print(raw_detail)

                self._append_errors(results)
                self.console.print()
