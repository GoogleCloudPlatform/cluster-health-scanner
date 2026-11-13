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

"""Top-to-bottom flow layout on top of pdfwriter.

A document is a list of flowables. A flowable reports its height for a
width, draws itself at a top-left corner and may split itself when it does
not fit in the space left on a page: tables split between rows and repeat
their header, grids split between rows of cells.
"""

import dataclasses
import math
from typing import Callable

from chs.pdf_generator.pdfwriter import Document, Page, Style, wrap_text

INK, MUTED, RULE, PANEL = "#202124", "#5f6368", "#dadce0", "#f8f9fa"
BAR, BAR_DEGRADED = "#1a73e8", "#c5221f"
PAGE_BREAK = object()
MARGIN, TOP, BOTTOM, GAP = 36, 48, 40, 10  # Page geometry in points.
MARKER = 6  # Side of a test's colour square.

_TICK = Style("regular", 5.5, MUTED)
_BAR_VALUE = Style("bold", 5.5, INK)
_BAR_NOTE = Style("bold", 5, BAR_DEGRADED)
_CHART_TITLE = Style("bold", 8.5, INK)
_CHART_CAPTION = Style("regular", 7, MUTED)
_NO_VALUES = Style("bold", 8, BAR_DEGRADED)


class Flowable:
    keep_with_next = False

    def height(self, width: float) -> float:
        raise NotImplementedError

    def min_height(self, width: float) -> float:
        """Smallest non-splittable leading slice (used by keep_with_next)."""
        return self.height(width)

    def draw(self, page: Page, x: float, y: float, width: float) -> None:
        raise NotImplementedError

    def split(self, width: float, available: float):
        """Returns (part that fits in available, remainder) or None."""
        return None


@dataclasses.dataclass
class Paragraph(Flowable):
    text: str
    style: Style = Style("regular", 9, INK)
    keep_with_next: bool = False

    def height(self, width):
        lines = wrap_text(self.text, self.style, width)
        return len(lines) * self.style.size * 1.3

    def draw(self, page, x, y, width):
        lines = wrap_text(self.text, self.style, width)
        page.text_lines(x, y, lines, self.style, self.style.size * 1.3)


@dataclasses.dataclass
class Cell:
    """A styled table cell; a list of cells in one slot stacks vertically."""

    text: str
    face: str = "regular"
    color: str = INK
    fill: str | None = None
    max_lines: int | None = None
    overflow: str = "…"
    marker: str | None = None  # Colour of a square drawn before the text.

    def lines(self, style: Style, width: float) -> list[str]:
        lines = wrap_text(self.text, style, width)
        if self.max_lines is None or len(lines) <= self.max_lines:
            return lines
        last = lines[self.max_lines - 1]
        while last and style.width(last + self.overflow) > width:
            last = last[:-1]
        return lines[: self.max_lines - 1] + [last.rstrip() + self.overflow]


def _cells(slot, face: str) -> list[Cell]:
    if isinstance(slot, str):
        return [Cell(slot, face)]
    return [slot] if isinstance(slot, Cell) else slot


@dataclasses.dataclass
class Column:
    title: str
    weight: float = 1
    face: str = "regular"
    min_width: float = 0


@dataclasses.dataclass
class Table(Flowable):
    """Weighted columns, wrapped cells, row split with header repeat."""

    columns: list[Column]
    rows: list[list]  # Each slot: str, Cell or list[Cell].
    size: float = 7.5
    pad: float = 4
    _heights: dict = dataclasses.field(  # width -> row heights, header first
        default_factory=dict, init=False, repr=False, compare=False
    )

    def _layout(self, width):
        """Yields (row index, slots, column widths) with the header first."""
        widths = _column_widths(self.columns, width)
        header = [Cell(c.title, "bold", MUTED) for c in self.columns]
        for index, row in enumerate([header, *self.rows]):
            cells = [_cells(slot, c.face) for slot, c in zip(row, self.columns)]
            yield index, cells, widths

    def _lines(self, cell, width):
        style = Style(cell.face, self.size, cell.color)
        indent = MARKER + 4 if cell.marker else 0
        return style, cell.lines(style, width - 2 * self.pad - indent)

    def _row_height(self, slots, widths):
        line_counts = (
            sum(len(self._lines(cell, w)[1]) for cell in slot)
            for slot, w in zip(slots, widths)
        )
        return 2 * self.pad + self.size * 1.25 * max(line_counts)

    def _row_heights(self, width):
        if width not in self._heights:
            self._heights[width] = [
                self._row_height(s, w) for _, s, w in self._layout(width)
            ]
        return self._heights[width]

    def height(self, width):
        return sum(self._row_heights(width))

    def min_height(self, width):
        return sum(self._row_heights(width)[:2])

    def split(self, width, available):
        header, *heights = self._row_heights(width)
        fits = _fit_count(heights, available - header)
        if not 1 <= fits < len(self.rows):
            return None
        head = dataclasses.replace(self, rows=self.rows[:fits])
        tail = dataclasses.replace(self, rows=self.rows[fits:])
        head._heights[width] = [header, *heights[:fits]]
        tail._heights[width] = [header, *heights[fits:]]
        return head, tail

    def draw(self, page, x, y, width):
        leading = self.size * 1.25
        heights = self._row_heights(width)
        for index, slots, widths in self._layout(width):
            height = heights[index]
            if index % 2 == 0:
                page.rect(x, y, width, height, PANEL if index else "#f1f3f4")
            cell_x = x
            for slot, w in zip(slots, widths):
                line_y = y + self.pad
                for cell in slot:
                    style, lines = self._lines(cell, w)
                    box_x, box_y = cell_x + self.pad / 2, line_y - 1
                    box = (w - self.pad, len(lines) * leading + 1)
                    page.rect(box_x, box_y, *box, cell.fill)  # No-op if None.
                    text_x = cell_x + self.pad
                    square = (MARKER, MARKER, cell.marker)
                    page.rect(text_x, line_y + 1.5, *square)  # Ditto.
                    text_x += (MARKER + 4) * bool(cell.marker)
                    line_y = page.text_lines(
                        text_x, line_y, lines, style, leading
                    )
                cell_x += w
            page.line(x, y + height, x + width, y + height, RULE, 0.4)
            y += height


def _column_widths(columns: list[Column], total_width: float) -> list[float]:
    if not columns or total_width <= 0:
        return [0.0] * len(columns)
    n = len(columns)
    min_floor = min(1.0, total_width / n)
    floors = [max(min_floor, c.min_width) for c in columns]
    floor_sum = sum(floors)
    if floor_sum > total_width:
        scale = total_width / floor_sum
        floors = [f * scale for f in floors]
    widths = [0.0] * n
    locked = [False] * n
    remaining = total_width
    for _ in range(n):
        free_weight = sum(
            c.weight for i, c in enumerate(columns) if not locked[i]
        )
        if free_weight <= 0:
            break
        newly_locked = False
        for i, col in enumerate(columns):
            if locked[i]:
                continue
            share = remaining * (col.weight / free_weight)
            if share < floors[i]:
                widths[i] = floors[i]
                locked[i] = True
                remaining -= floors[i]
                newly_locked = True
            else:
                widths[i] = share
        if not newly_locked:
            break
    return widths


def _fit_count(heights: list[float], available: float, gap: float = 0) -> int:
    """How many leading items, gap apart, fit in available."""
    used = -gap
    for count, height in enumerate(heights):
        used += height + gap
        if used > available:
            return count
    return len(heights)


def _short(value: float) -> str:
    return f"{value:.0f}" if abs(value) >= 100 else f"{value:.3g}"


def _value_label_style(bars: list, slot: float) -> tuple[Style, bool]:
    """Picks a horizontal font size that fits in slot, or vertical text."""
    labels = [_short(b.value) for b in bars if b.value is not None]
    for size in (5.5, 5.0, 4.5):
        style = Style("bold", size, INK)
        if max((style.width(s) for s in labels), default=0) <= slot * 0.85:
            return style, False
    return Style("bold", 5.0, INK), True


def _fit_tick(label: str, max_width: float) -> str:
    """Shortens an oversized chart tick label to fit within max_width."""
    if _TICK.width(label) <= max_width:
        return label
    ellipsis = "..."
    if _TICK.width(ellipsis) >= max_width:
        return ellipsis
    if "-" in label:
        tail = label.rsplit("-", 1)[-1]
        if (
            len(tail) <= 6
            and _TICK.width(f"{label[:4]}{ellipsis}{tail}") <= max_width
        ):
            prefix = 4
            while (
                _TICK.width(f"{label[: prefix + 1]}{ellipsis}{tail}")
                <= max_width
            ):
                prefix += 1
            return f"{label[:prefix]}{ellipsis}{tail}"
    lo, hi = 0, len(label)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _TICK.width(label[:mid] + ellipsis) <= max_width:
            lo = mid
        else:
            hi = mid - 1
    return label[:lo] + ellipsis


@dataclasses.dataclass
class BarChart(Flowable):
    """One bar per point, a dashed threshold line, degraded bars in red.

    bars: objects with label, value (float | None), text and degraded, such
    as model.Point; text replaces the bar when there is no value.
    """

    title: str
    unit: str
    threshold: float | None
    bars: list
    color: str = BAR
    chart_height: float = 170

    def height(self, width):
        return self.chart_height

    def draw(self, page, x, y, width):
        page.rect(x, y, width, self.chart_height, stroke=RULE)
        page.rect(x + 8, y + 7, MARKER, MARKER, self.color)
        indent = MARKER + 4
        title = wrap_text(self.title, _CHART_TITLE, width - 16 - indent)[:2]
        top = page.text_lines(x + 8 + indent, y + 3, title, _CHART_TITLE, 11)
        caption = self.unit
        if self.threshold is not None:
            caption = f"≥ {self.threshold:g} {self.unit}".strip()
        page.text(x + 8, top + 11, caption, _CHART_CAPTION)
        values = [bar.value for bar in self.bars if bar.value is not None]
        chart_left, chart_right = x + 40, x + width - 16
        slot = (chart_right - chart_left) / max(1, len(self.bars))
        widest = max((_TICK.width(bar.label) for bar in self.bars), default=0)
        angled = widest + 2 > slot
        value_style, value_angled = _value_label_style(self.bars, slot)
        extra_top = (
            max(0.0, max(value_style.width(_short(v)) for v in values) - 12)
            if value_angled and values
            else 0.0
        )
        top = top + 27 + extra_top
        bottom = y + self.chart_height - (42 if angled else 18)
        if values:
            self._draw_bars(
                page,
                chart_left,
                chart_right,
                top,
                bottom,
                values,
                angled,
                value_style,
                value_angled,
            )
        else:
            notes = ", ".join(sorted({bar.text for bar in self.bars}))
            lines = wrap_text(notes, _NO_VALUES, max(width - 16, 1.0))
            step = 11
            start_y = (top + bottom - (len(lines) - 1) * step) / 2
            for i, line in enumerate(lines):
                page.text(
                    x + width / 2,
                    start_y + i * step,
                    line,
                    _NO_VALUES,
                    "center",
                )

    def _draw_bars(
        self,
        page,
        left,
        right,
        top,
        bottom,
        values,
        angled,
        value_style,
        value_angled,
    ):
        bounds = values + [self.threshold] * (self.threshold is not None)
        low, high = min(bounds), max(bounds)
        margin = (high - low) * 0.1 or abs(high) * 0.1 or 1
        low, high = low - margin, high + margin

        def to_y(value):
            return bottom - (value - low) / (high - low) * (bottom - top)

        level = to_y(self.threshold) if self.threshold is not None else None
        for tick in range(4):
            value = low + (high - low) * tick / 3
            ty = to_y(value)
            page.line(left, ty, right, ty, "#eeeeee", 0.4)
            if level is None or abs(ty - level) >= 9.0:
                page.text(left - 3, ty + 2, _short(value), _TICK, "right")
        slot = (right - left) / len(self.bars)
        stride = (
            max(1, math.ceil(8.0 / slot))
            if angled
            else max(
                1,
                math.ceil(
                    (max(_TICK.width(b.label) for b in self.bars) + 4) / slot
                ),
            )
        )
        # Angled ticks start 7pt below the axis with 42pt to the box bottom
        # and 16pt + half a slot to its right edge.
        angle = math.radians(48)
        max_tick_w = min(
            35 / math.sin(angle), (18 + slot / 2) / math.cos(angle)
        )
        for i, bar in enumerate(self.bars):
            center = left + (i + 0.5) * slot
            if i % stride == 0:
                if angled:
                    page.text(
                        center - 2,
                        bottom + 7,
                        _fit_tick(bar.label, max_tick_w),
                        _TICK,
                        angle=-48,
                    )
                else:
                    page.text(center, bottom + 9, bar.label, _TICK, "center")
            if bar.value is None:
                if _BAR_NOTE.width(bar.text) <= slot:
                    page.text(center, bottom - 3, bar.text, _BAR_NOTE, "center")
                continue
            half = min(slot * 0.35, 14)
            color = BAR_DEGRADED if bar.degraded else self.color
            top_y = to_y(bar.value)
            page.rect(center - half, top_y, 2 * half, bottom - top_y, color)
            label = _short(bar.value)
            if value_angled:
                page.text(
                    center + value_style.size * 0.35,
                    top_y - 3,
                    label,
                    value_style,
                    angle=90,
                )
            else:
                page.text(center, top_y - 2, label, value_style, "center")
        if level is not None:
            page.line(left, level, right, level, INK, 0.8, dash=(3, 2))
            page.text(
                left - 3,
                level + 2,
                f">={_short(self.threshold)}",
                Style("bold", 5.5, INK),
                "right",
            )


@dataclasses.dataclass
class Grid(Flowable):
    """Equal-width cells laid out row by row; splits between rows."""

    cells: list[Flowable]
    columns: int = 4
    gap: float = 8

    def _rows(self, width):
        """Yields (first cell index, cells, cell width, row height)."""
        cell_width = (width - self.gap * (self.columns - 1)) / self.columns
        for start in range(0, len(self.cells), self.columns):
            row = self.cells[start : start + self.columns]
            yield start, row, cell_width, max(c.height(cell_width) for c in row)

    def height(self, width):
        heights = [height for *_, height in self._rows(width)]
        return sum(heights) + self.gap * (len(heights) - 1)

    def min_height(self, width):
        rows = list(self._rows(width))
        return rows[0][-1] if rows else 0

    def split(self, width, available):
        rows = list(self._rows(width))
        fits = _fit_count([row[-1] for row in rows], available, self.gap)
        if not 1 <= fits < len(rows):
            return None
        cut = rows[fits][0]
        return (
            dataclasses.replace(self, cells=self.cells[:cut]),
            dataclasses.replace(self, cells=self.cells[cut:]),
        )

    def draw(self, page, x, y, width):
        for _, row, cell_width, height in self._rows(width):
            for i, cell in enumerate(row):
                cell.draw(page, x + i * (cell_width + self.gap), y, cell_width)
            y += height + self.gap


def build(
    flowables: list, decorate: Callable[[Page, int, int], None], title: str
) -> bytes:
    """Lays flowables out page by page and returns the PDF bytes.

    PAGE_BREAK starts a new page. decorate(page, number, total) draws the
    per-page header and footer once the page count is known.
    """
    doc = Document(title=title)
    width = doc.page_size[0] - 2 * MARGIN
    limit = doc.page_size[1] - BOTTOM
    queue, page, y = list(flowables), None, TOP
    while queue:
        item = queue.pop(0)
        if item is PAGE_BREAK:
            page = None
            continue
        if page is None:
            page, y = doc.add_page(), TOP
        needed = item.height(width)
        if item.keep_with_next and queue and queue[0] is not PAGE_BREAK:
            needed += GAP + min(queue[0].min_height(width), limit - TOP)
        parts = None
        if y + needed > limit and not item.keep_with_next:
            parts = item.split(width, limit - y)
        if parts:
            parts[0].draw(page, MARGIN, y, width)
            queue.insert(0, parts[1])
            page = None
        elif y + needed > limit and y > TOP:
            queue.insert(0, item)
            page = None
        else:  # Fits, or is taller than a whole page and drawn anyway.
            item.draw(page, MARGIN, y, width)
            y += item.height(width) + GAP
    for number, page in enumerate(doc.pages, start=1):
        decorate(page, number, len(doc.pages))
    return doc.to_bytes()
