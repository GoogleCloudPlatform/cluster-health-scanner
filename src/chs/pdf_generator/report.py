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

"""Draws a model.Report as the PDF Bill of Health.

Page 1 is the run overview; each results page (CHS Subblock, CHS Block,
CHS Slice or Hosts) then starts on a new page with its charts, per-host
results and potential errors and warnings.
"""

import dataclasses
import functools

from chs.pdf_generator import layout
from chs.pdf_generator import model
from chs.pdf_generator.layout import Cell, Column, Paragraph, Table
from chs.pdf_generator.pdfwriter import Page, Style, wrap_text
from chs.utils import file_utils

STATUSES = ("HEALTHY", "DEGRADED", "POTENTIALLY DEGRADED", "INCOMPLETE")
_STATUS_STYLE = {  # status -> (text colour, background)
    "HEALTHY": ("#137333", "#e6f4ea"),
    "DEGRADED": ("#c5221f", "#fce8e6"),
    "POTENTIALLY DEGRADED": ("#b06000", "#fef7e0"),
    "INCOMPLETE": ("#5f6368", "#f1f3f4"),
}
# Per-test colours, assigned by test order; degraded bars are always red.
_TEST_COLORS = (
    "#1E40AF #16A34A #0F766E #6D28D9 #D97706 #0369A1 #4D7C0F #9F1239 #B45309"
    " #475569"
).split()
_EVENT_COLOR = {"Error": "#c5221f", "Warning": "#b06000"}
_CHART_COLUMNS = ((4, 16), (2, 40))  # (charts per row, most bars per chart)
_STRIP_LABEL = Style("bold", 7, layout.MUTED)
_FOOTER = Style("regular", 7, layout.MUTED)
_MAX_POINTS = 12  # Lines per host-table cell.


def render(report: model.Report, path: str) -> None:
    """Writes the report to path atomically (temp file + rename)."""
    pdf = layout.build(
        _flowables(report), functools.partial(_decorate, report), report.title
    )
    with file_utils.atomic_open(path, "wb") as f:
        f.write(pdf)


@dataclasses.dataclass
class _Strip(layout.Flowable):
    """A shaded band of labelled values wrapping onto rows when needed."""

    pairs: list[tuple[str, Cell]]

    def _rows(self, width):
        styles = [Style(v.face, 10, v.color) for _, v in self.pairs]
        cap = max(width / 3, 1.0)  # Longer values wrap rather than crowd.
        items = []
        for (label, value), style in zip(self.pairs, styles):
            inner = max(
                _STRIP_LABEL.width(label.upper()),
                min(style.width(value.text), cap),
            )
            items.append((label, style, value.text, min(width, 25 + inner)))
        rows, row, row_w = [], [], 0.0
        for item in items:
            if row and row_w + item[3] > width:
                rows.append(self._fit_row(row, width))
                row, row_w = [], 0.0
            row.append(item)
            row_w += item[3]
        if row:
            rows.append(self._fit_row(row, width))
        return rows

    @staticmethod
    def _fit_row(row, width):
        total = sum(w for _, _, _, w in row)
        extra = (total - width) / len(row) if total > width else 0.0
        fitted = []
        for label, style, text, w in row:
            col_w = w - extra
            lines = wrap_text(text, style, max(col_w - 24, 1.0))
            fitted.append((label, style, lines, col_w))
        return fitted

    def _columns(self, width):
        """Yields (label, value style, value lines, column width)."""
        for row in self._rows(width):
            yield from row

    def height(self, width):
        rows = self._rows(width)
        return 8 + sum(18 + 13 * max(len(c[2]) for c in row) for row in rows)

    def draw(self, page, x, y, width):
        rows = self._rows(width)
        page.rect(x, y, width, self.height(width), layout.PANEL, layout.RULE)
        ry = y
        for row in rows:
            cx = x
            for label, style, lines, w in row:
                page.text(cx + 12, ry + 14, label.upper(), _STRIP_LABEL)
                page.text_lines(cx + 12, ry + 18, lines, style, 13)
                cx += w
            ry += 18 + 13 * max(len(c[2]) for c in row)


def _status(status: str, face: str = "bold") -> Cell:
    color, fill = _STATUS_STYLE.get(status, _STATUS_STYLE["INCOMPLETE"])
    return Cell(status, face, color, fill)


def _counts(rows: list[model.Row]) -> dict[str, int]:
    return {s: sum(row.status == s for row in rows) for s in STATUSES}


def _rollup(rows: list[model.Row]) -> str:
    counts = _counts(rows)
    if counts["DEGRADED"]:
        return (
            "DEGRADED"
            if counts["DEGRADED"] == len(rows)
            else "POTENTIALLY DEGRADED"
        )
    if counts["POTENTIALLY DEGRADED"]:
        return "POTENTIALLY DEGRADED"
    if counts["INCOMPLETE"]:
        return "INCOMPLETE"
    return "HEALTHY" if counts["HEALTHY"] else "INCOMPLETE"


def _heading(text: str) -> Paragraph:
    return Paragraph(text, Style("bold", 11, layout.INK), keep_with_next=True)


def _title(text: str) -> Paragraph:
    return Paragraph(text, Style("bold", 17, layout.INK))


def _point_cell(point: model.Point) -> Cell:
    if point.degraded:
        color, fill = _STATUS_STYLE["DEGRADED"]
        return Cell(point.text, "regular", color, fill, max_lines=2)
    if point.text == "Warning logged":
        return Cell(point.text, "bold", _EVENT_COLOR["Warning"], max_lines=2)
    return Cell(point.text, "regular", layout.INK, max_lines=2)


def _points(points: list[model.Point]) -> list[Cell]:
    """One cell line per point, capped so a row always fits on a page."""
    ordered = (
        sorted(
            points,
            key=lambda p: (not p.degraded, p.text != "Warning logged"),
        )
        if len(points) > _MAX_POINTS
        else points
    )
    cells = [_point_cell(p) for p in ordered[:_MAX_POINTS]]
    if len(ordered) > _MAX_POINTS:
        hidden = ordered[_MAX_POINTS - 1 :]
        hidden_bad = sum(p.degraded for p in hidden)
        suffix = f" ({hidden_bad} degraded)" if hidden_bad else ""
        more = f"+{len(hidden)} more{suffix} in the test log"
        cells[-1] = Cell(
            more,
            "regular",
            _STATUS_STYLE["DEGRADED"][0] if hidden_bad else layout.MUTED,
        )
    return cells


def _events_strip(events: list[model.Event]) -> _Strip:
    """The "Potential errors and warnings" summary strip."""
    label = "Potential errors and warnings"
    if not events:
        return _Strip([(label, Cell("0 detected"))])
    by_type = {
        kind: sum(e.count for e in events if e.type == kind)
        for kind in _EVENT_COLOR
    }
    detected = Cell(
        f"{sum(e.count for e in events)} detected",
        "bold",
        _EVENT_COLOR["Error" if by_type["Error"] else "Warning"],
    )
    return _Strip(
        [(label, detected)]
        + [(f"{kind}s", Cell(str(n))) for kind, n in by_type.items()]
        + [("Distinct messages", Cell(str(len(events))))]
    )


def _events_table(
    events: list[model.Event], group_label: str = "Group"
) -> Table:
    columns = [
        Column("Host", 2.2),
        Column(group_label, 1),
        Column("Test", 1.5, "bold"),
        Column("Type", 0.6),
        Column("Category", 1.2),
        Column("Diagnostic message", 5),
    ]
    rows = [
        [
            e.hosts,
            e.section,
            e.test,
            Cell(e.type, "bold", _EVENT_COLOR.get(e.type, layout.INK)),
            e.category,
            Cell(e.message, "mono", max_lines=3, overflow="… (see test log)"),
        ]
        for e in events
    ]
    return Table(columns, rows)


def _events(
    events: list[model.Event], group_label: str = "Group"
) -> list[layout.Flowable]:
    """The "Potential errors and warnings" band and, if any, its table."""
    strip = _events_strip(events)
    return [strip, _events_table(events, group_label)] if events else [strip]


def _section_kind(title: str) -> str | None:
    if title.startswith("CHS Subblock"):
        return "Subblock"
    if title.startswith("CHS Block"):
        return "Block"
    if title.startswith("CHS Slice"):
        return "Slice"
    if title == "Hosts":
        return "Hosts"
    return None


def _overview_noun(sections: list[model.Section]) -> tuple[str, str]:
    kinds = {k for s in sections if (k := _section_kind(s.title)) is not None}
    if kinds == {"Subblock"}:
        return "subblock", "subblocks"
    if kinds == {"Block"}:
        return "block", "blocks"
    if kinds == {"Slice"}:
        return "slice", "slices"
    if kinds == {"Hosts"}:
        return "host", "hosts"
    return "group", "groups"


def _health_strip(
    heading: str, plural: str, statuses: list[str]
) -> list[layout.Flowable]:
    healthy_count = sum(1 for st in statuses if st == "HEALTHY")
    by_state = {
        st: sum(1 for s_st in statuses if s_st == st)
        for st in ("DEGRADED", "POTENTIALLY DEGRADED", "INCOMPLETE")
    }
    strip_items = [
        (f"Total {plural}", Cell(str(len(statuses)), "bold")),
        (
            f"Healthy {plural}",
            Cell(
                str(healthy_count),
                "bold",
                _STATUS_STYLE["HEALTHY"][0] if healthy_count else layout.INK,
            ),
        ),
        (
            f"Degraded {plural}",
            Cell(
                str(by_state["DEGRADED"]),
                "bold",
                _STATUS_STYLE["DEGRADED"][0]
                if by_state["DEGRADED"]
                else layout.INK,
            ),
        ),
        (
            "Potentially Degraded",
            Cell(
                str(by_state["POTENTIALLY DEGRADED"]),
                "bold",
                _STATUS_STYLE["POTENTIALLY DEGRADED"][0]
                if by_state["POTENTIALLY DEGRADED"]
                else layout.INK,
            ),
        ),
        ("Incomplete", Cell(str(by_state["INCOMPLETE"]), "bold")),
    ]
    return [_heading(heading), _Strip(strip_items)]


def _overview(
    report: model.Report, colors: dict[str, str]
) -> list[layout.Flowable]:
    flowables = [
        _title(f"{report.title} — Overview"),
        _Strip([(k, Cell(v, "bold")) for k, v in report.run_info]),
    ]
    if report.sections:
        grouped = [s for s in report.sections if s.title != "Hosts"]
        loose = [s for s in report.sections if s.title == "Hosts"]
        if grouped:
            singular, plural = _overview_noun(grouped)
            flowables += _health_strip(
                f"Health summary by {singular}",
                plural,
                [_rollup(s.hosts) for s in grouped],
            )
        if loose:
            heading = (
                "Health summary by ungrouped host"
                if grouped
                else "Health summary by host"
            )
            flowables += _health_strip(
                heading,
                "hosts",
                [r.status for s in loose for r in s.hosts],
            )
    if report.tests:
        columns = [
            Column("Test", 1.5, "bold"),
            Column("Scope", 0.8),
            Column("Workload & validation methodology", 5),
            Column("Healthy threshold", 2.5),
        ]
        rows = [
            [
                Cell(t.name, "bold", marker=colors[t.name]),
                t.scope,
                t.description,
                "\n".join(t.thresholds) or "—",
            ]
            for t in report.tests
        ]
        flowables += [
            _heading("Workloads and healthy thresholds"),
            Table(columns, rows),
        ]
    return flowables


def _test_colors(report: model.Report) -> dict[str, str]:
    """Test name (incl. level) -> its colour, stable across the report."""
    return {
        test.name: _TEST_COLORS[i % len(_TEST_COLORS)]
        for i, test in enumerate(report.tests)
    }


def _charts(
    charts: list[model.Chart], colors: dict[str, str]
) -> list[layout.Flowable]:
    if not charts:
        return []
    most_bars = max(len(chart.points) for chart in charts)
    per_row = next((n for n, bars in _CHART_COLUMNS if most_bars <= bars), 1)
    return [
        _heading("Evaluated metrics per host and multi-host"),
        layout.Grid(
            [
                layout.BarChart(
                    c.title,
                    c.unit,
                    c.threshold,
                    c.points,
                    colors.get(c.test, layout.BAR),
                )
                for c in charts
            ],
            per_row,
        ),
    ]


def _single_line_width(
    header: str, values, face: str, floor: float = 0
) -> float:
    header_w = Style("bold", 7.5).width(header)
    val_style = Style(face, 7.5)
    val_w = max((val_style.width(v) for v in values if v), default=0.0)
    return max(floor, max(header_w, val_w) + 12)


def _host_table(section: model.Section) -> Table:
    show_pool = bool(section.pool_label) and any(r.pool for r in section.hosts)
    show_instance = any(r.instance_id for r in section.hosts)
    columns = [
        Column(section.host_label, 1.8, "bold"),
    ]
    if show_pool:
        columns.append(Column(section.pool_label, 1.1))
    if show_instance:
        inst_min = _single_line_width(
            "GCE Instance ID",
            (r.instance_id or "—" for r in section.hosts),
            "mono",
        )
        columns.append(
            Column("GCE Instance ID", 1.2, "mono", min_width=inst_min)
        )
    host_id_min = _single_line_width(
        "CHS Host ID", (r.hardware_id for r in section.hosts), "mono", floor=158
    )
    status_min = _single_line_width(
        "Status", (r.status for r in section.hosts), "bold"
    )
    columns += [
        Column("CHS Host ID", 1.8, "mono", min_width=host_id_min),
        Column("Status", 0.95, min_width=status_min),
        *[Column(test, 1) for test in section.columns],
    ]
    rows = []
    for r in section.hosts:
        row = [r.name]
        if show_pool:
            row.append(r.pool or "—")
        if show_instance:
            row.append(r.instance_id or "—")
        row += [r.hardware_id, _status(r.status), *map(_points, r.cells)]
        rows.append(row)
    return Table(columns, rows)


def _section(
    section: model.Section, colors: dict[str, str]
) -> list[layout.Flowable]:
    pairs: list[tuple[str, Cell]] = []
    if section.topology:
        pairs.append(("Topology", Cell(section.topology, "bold")))
    if section.group_id is not None and section.group_id_label:
        pairs.append((section.group_id_label, Cell(section.group_id, "bold")))
    pairs.append(("Hosts", Cell(str(len(section.hosts)), "bold")))
    pairs.extend(
        (k, Cell(str(n), "bold")) for k, n in _counts(section.hosts).items()
    )
    if section.pool_label and section.pool_summary:
        pairs.append((section.pool_label, Cell(section.pool_summary, "bold")))
    events = sum(e.count for e in section.events)
    pairs.append(("Potential errors and warnings", Cell(str(events), "bold")))
    if section.title != "Hosts":
        pairs.append(("Status", _status(_rollup(section.hosts))))
    flowables = [
        _title(f"{section.title} — Results"),
        _Strip(pairs),
    ]
    flowables += _charts(section.charts, colors)
    event_group_col = _section_kind(section.title) or "Group"
    if event_group_col == "Hosts":
        event_group_col = "Group"
    flowables += [
        _heading("Per-host results"),
        _host_table(section),
        _heading("Potential errors and warnings"),
        *_events(section.events, event_group_col),
    ]
    return flowables


def _flowables(report: model.Report) -> list:
    colors = _test_colors(report)
    flowables = _overview(report, colors)
    for section in report.sections:
        flowables += [layout.PAGE_BREAK, *_section(section, colors)]
    return flowables


def _decorate(report: model.Report, page: Page, number: int, total: int):
    left, right = layout.MARGIN, page.width - layout.MARGIN
    footer_y = page.height - 22
    page.text(left, 26, report.title, Style("bold", 8, layout.MUTED))
    page.line(left, 33, right, 33, layout.RULE)
    page.line(left, footer_y - 10, right, footer_y - 10, layout.RULE)
    page.text(right, footer_y, f"Page {number} of {total}", _FOOTER, "right")
    footer = wrap_text(report.id_footer, _FOOTER, right - left - 80)
    for i, line in enumerate(footer[:2]):
        page.text(left, footer_y + 9 * i, line, _FOOTER)
