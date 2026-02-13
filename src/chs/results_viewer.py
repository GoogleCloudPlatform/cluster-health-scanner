"""Module for displaying Bill of Health results in the terminal."""

import json
import logging
from typing import Any
from chs.protos import bill_of_health_pb2 as pb
from google.protobuf import text_format
from google.protobuf.json_format import MessageToDict
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

logger = logging.getLogger(__name__)


class ResultsViewer:
  """Renders Bill of Health results using Rich."""

  def __init__(self, boh_path: str):
    """Initializes the viewer with the path to the Bill of Health JSON."""
    self.boh_path = boh_path
    self.console = Console()
    try:
      with open(self.boh_path, "r") as f:
        self.data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
      # Fallback: Try parsing as TextProto (in case it was written as such)
      try:
        with open(self.boh_path, "r") as f:
          content = f.read()
          boh = pb.BillOfHealth()
          text_format.Parse(content, boh)
          self.data = MessageToDict(boh, preserving_proto_field_name=True)
      except (text_format.ParseError, UnicodeDecodeError) as e:
        self.console.print(
            f"[bold red]Error loading Bill of Health:[/bold red] {e}"
        )
        self.data = {}
    except FileNotFoundError as e:
      self.console.print(
          f"[bold red]Error loading Bill of Health:[/bold red] {e}"
      )
      self.data = {}

  def _get_status_style(self, status: str) -> tuple[str, str]:
    """Returns a rich style string and icon for a given status."""
    status = status.upper()
    if status == "SUCCESS":
      return "bold green", ""
    elif status == "FAILED":
      return "bold red", ""
    elif status == "TIMEOUT":
      return "bold magenta", ""
    elif status == "CANCELLED":
      return "bold white", ""
    else:
      return "bold cyan", ""

  def show_summary(self) -> None:
    """Displays the summary view of the Bill of Health."""
    if not self.data:
      return

    report_id = self.data.get("report_id", "Unknown")
    chs_version = self.data.get("chs_version", "Unknown")

    # Header
    grid = Table.grid(expand=True)
    grid.add_column(justify="center", ratio=1)
    grid.add_column(justify="right")
    grid.add_row(
        f"[b]ID:[/b] {report_id}",
        f"[b]v{chs_version}[/b]",
    )

    self.console.print(
        Panel(
            grid,
            title=(
                "[bold white on blue] Cluster Bill of Health Report [/bold"
                " white on blue]"
            ),
            border_style="blue",
            padding=(1, 2),
        )
    )

    test_results = self.data.get("test_results", [])
    if not test_results:
      self.console.print(
          Panel(
              "[yellow]No test results found.[/yellow]",
              title="Warning",
              border_style="yellow",
          )
      )
      return

    if self.console.width < 150:
      self._show_summary_list(test_results)
    else:
      self._show_summary_table(test_results)

  def _show_summary_table(self, test_results: list[dict[str, Any]]) -> None:
    """Displays test results as a table."""
    table = Table(
        title="🔬 Test Results",
        show_lines=True,
        box=box.ROUNDED,
        header_style="bold white on navy_blue",
    )
    table.add_column("Status", justify="center")
    table.add_column("Test Name", style="cyan")
    table.add_column("Target Resources", style="green")

    for result in test_results:
      status_enum = result.get("result_status", "RESULT_STATUS_UNKNOWN")
      status = status_enum.replace("RESULT_STATUS_", "")
      name = result.get("name", "Unknown")

      targets = result.get("target_resource_ids", [])
      targets_str = "\n".join(targets) if targets else "Cluster"

      style, icon = self._get_status_style(status)
      table.add_row(
          f"[{style}]{icon} {status}[/{style}]",
          name,
          targets_str,
      )

    self.console.print(table)

  def _get_test_description(self, name: str) -> str:
    """Returns a description for the given test name."""
    if "nccl-cluster" in name or "nccl-all" in name:
      return (
          "Validates cluster-wide GPU communication bandwidth/latency using"
          " NCCL AllReduce and AllGather."
      )
    elif "nccl-pairwise" in name:
      return (
          "Measures point-to-point interconnect bandwidth between node pairs to"
          " isolate network bottlenecks."
      )
    elif "dcgm" in name:
      return (
          "Performs deep hardware diagnostics (memory, PCIe, replay limits) to"
          " detect GPU faults."
      )
    return ""

  def _show_summary_list(self, test_results: list[dict[str, Any]]) -> None:
    """Displays test results as a list of panels for smaller screens."""
    self.console.print(
        "[bold italic]Switch to wider terminal (>150 cols) for table view[/bold"
        " italic]",
        style="dim",
    )
    for result in test_results:
      status_enum = result.get("result_status", "RESULT_STATUS_UNKNOWN")
      status = status_enum.replace("RESULT_STATUS_", "")
      name = result.get("name", "Unknown")
      targets = result.get("target_resource_ids", [])
      targets_str = ", ".join(targets) if targets else "Cluster"

      style, icon = self._get_status_style(status)
      description = self._get_test_description(name)

      content = Text()
      content.append(f"{name}\n", style="cyan bold")
      if description:
        content.append(f"{description}\n", style="italic dim")
      content.append("Targets: ", style="bold")
      content.append(f"{targets_str}\n", style="green")

      panel = Panel(
          content,
          title=f"[{style}]{icon} {status}[/{style}]",
          expand=True,
          border_style=style.split()[-1],  # extract color from style string
      )
      self.console.print(panel)

  def show_details(
      self, tests: list[str] | None = None, resources: list[str] | None = None
  ) -> None:
    """Displays detailed views for specific tests or resources."""
    if not self.data:
      return

    test_results = self.data.get("test_results", [])

    # Filter results
    filtered_results = []
    for result in test_results:
      name = result.get("name", "")
      targets = result.get("target_resource_ids", [])

      include = False
      if tests:
        if name in tests:
          include = True

      if resources:
        if any(r in targets for r in resources):
          include = True

      if tests or resources:
        if include:
          filtered_results.append(result)
      else:
        filtered_results.append(result)

    if not filtered_results:
      self.console.print(
          Panel(
              "[yellow]No matching results found.[/yellow]",
              title="Info",
              border_style="yellow",
          )
      )
      return

    for result in filtered_results:
      self._print_test_detail(result)

  def _print_test_detail(self, result: dict[str, Any]) -> None:
    name = result.get("name", "Unknown")
    status_enum = result.get("result_status", "RESULT_STATUS_UNKNOWN")
    status = status_enum.replace("RESULT_STATUS_", "")
    style, icon = self._get_status_style(status)
    border_color = style.split()[-1]

    # Metrics Table
    metrics = result.get("metrics", [])
    m_table = Table(
        title=f" Metrics for {name}",
        show_lines=True,
        box=box.SIMPLE,
        header_style=f"bold {border_color}",
        expand=True,
    )
    m_table.add_column("Key", style="cyan")
    m_table.add_column("Value", style="bold white")
    m_table.add_column("Unit", style="dim")

    for m in metrics:
      key = m.get("key", "")
      val = m.get("value")

      str_val = ""
      if isinstance(val, dict):
        # Handle TextProto-like structure where value is a dict wrapper
        if "number_value" in val:
          str_val = str(val["number_value"])
        elif "string_value" in val:
          str_val = val["string_value"]
        elif "bool_value" in val:
          str_val = str(val["bool_value"])
      elif val is not None:
        # Handle direct JSON value
        str_val = str(val)

      unit = m.get("unit", "")

      m_table.add_row(key, str_val, unit)

    self.console.print(
        Panel(
            m_table if metrics else "[italic]No metrics available[/italic]",
            title=f"[{style}]{icon} {status}[/{style}] - [bold]{name}[/bold]",
            expand=True,
            border_style=border_color,
        )
    )
