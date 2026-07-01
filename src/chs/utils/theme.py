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

from rich import box
from rich.table import Table

CHS_HEADER_STYLE = "bold cyan"
CHS_WARNING_STYLE = "yellow"
CHS_TITLE_STYLE = "bold"
CHS_SUCCESS_STYLE = "green"
CHS_FAILURE_STYLE = "red"
CHS_NEUTRAL_STYLE = "dim"


def create_styled_table(
    *columns: str,
    header_style: str = CHS_HEADER_STYLE,
    expand: bool = True,
    show_lines: bool = False,
) -> Table:
    """Creates a Table with CHS standard styling.

    If columns are provided, they are added to the table.
    If no columns are provided, an empty styled table is returned,
    allowing caller to add columns with custom styles.
    """
    table = Table(
        box=box.SIMPLE,
        show_lines=show_lines,
        header_style=header_style,
        expand=expand,
    )
    for col in columns:
        table.add_column(col)
    return table


def create_error_table(errors: list[tuple[str, str]]) -> Table:
    """Creates a table for displaying errors and warnings."""
    table = create_styled_table(
        "Type",
        "Detail",
        header_style="bold red",
        expand=True,
    )
    for error_type, detail in errors:
        table.add_row(error_type, detail)
    return table
