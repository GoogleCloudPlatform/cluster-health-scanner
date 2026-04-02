from rich import box
from rich.table import Table

CHS_HEADER_STYLE = "bold cyan"
CHS_WARNING_STYLE = "yellow"
CHS_TITLE_STYLE = "bold"


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
