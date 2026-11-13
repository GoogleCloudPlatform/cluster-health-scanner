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

"""Minimal dependency-free PDF 1.4 writer.

Text uses three of the PDF base-14 fonts, which every conforming viewer
ships, so nothing is embedded. Strings are WinAnsi (cp1252) encoded; any
other character is transliterated or replaced, so untrusted test output can
never break the file. Text is measured with the fonts' AFM widths, which
also drives word wrapping. Coordinates are points with the origin at the
TOP-left of the page and y growing downwards, which suits top-to-bottom
layout.
"""

import dataclasses
import functools
import math
import unicodedata
import zlib

A3_LANDSCAPE = (1191.0, 842.0)
FONTS = {"regular": "Helvetica", "bold": "Helvetica-Bold", "mono": "Courier"}

# Advance widths (1/1000 em) of ASCII 32..126 from the Adobe core-14 AFMs.
# fmt: off
_ASCII_WIDTHS = {
    "regular": [
        278, 278, 355, 556, 556, 889, 667, 191, 333, 333, 389, 584, 278, 333,
        278, 278, 556, 556, 556, 556, 556, 556, 556, 556, 556, 556, 278, 278,
        584, 584, 584, 556, 1015, 667, 667, 722, 722, 667, 611, 778, 722, 278,
        500, 667, 556, 833, 722, 778, 667, 778, 722, 667, 611, 722, 667, 944,
        667, 667, 611, 278, 278, 278, 469, 556, 333, 556, 556, 500, 556, 556,
        278, 556, 556, 222, 222, 500, 222, 833, 556, 556, 556, 556, 333, 500,
        278, 556, 500, 722, 500, 500, 500, 334, 260, 334, 584,
    ],
    "bold": [
        278, 333, 474, 556, 556, 889, 722, 238, 333, 333, 389, 584, 278, 333,
        278, 278, 556, 556, 556, 556, 556, 556, 556, 556, 556, 556, 333, 333,
        584, 584, 584, 611, 975, 722, 722, 722, 722, 667, 611, 778, 722, 278,
        556, 722, 611, 833, 722, 778, 667, 778, 722, 667, 611, 722, 667, 944,
        667, 667, 611, 333, 278, 333, 584, 556, 333, 556, 611, 556, 611, 556,
        333, 611, 611, 278, 278, 556, 278, 889, 611, 611, 611, 611, 389, 556,
        333, 611, 556, 778, 556, 556, 500, 389, 280, 389, 584,
    ],
}
# fmt: on
_SYMBOL_WIDTHS = {"–": 556, "—": 1000, "·": 278, "×": 584, "•": 350, "…": 1000}
_COURIER_WIDTH = 600

# fmt: off
_TRANSLITERATIONS = str.maketrans({
    "≥": ">=", "≤": "<=", "≠": "!=", "\u2212": "-", "↔": "<->", "→": "->",
    "←": "<-", "Δ": "delta ", "✓": "OK", "✗": "X", "\u00a0": " ",
    "\t": "    ",
})
# fmt: on


# ASCII fast path of to_winansi: drop control characters, expand tabs.
_ASCII_TRANS = {c: None for c in (*range(32), 127)} | {ord("\t"): "    "}
# Longer strings (e.g. raw diagnostic messages) are measured and wrapped
# uncached, so the caches below only ever hold short, repeated strings.
_CACHE_MAX_CHARS = 512


def _byte_widths(font: str) -> tuple[int, ...]:
    widths = [556] * 256
    for byte in range(256):
        try:
            char = bytes([byte]).decode("cp1252")
        except UnicodeDecodeError:
            continue
        if 32 <= ord(char) <= 126:
            widths[byte] = _ASCII_WIDTHS[font][ord(char) - 32]
        elif char in _SYMBOL_WIDTHS:
            widths[byte] = _SYMBOL_WIDTHS[char]
    return tuple(widths)


_BYTE_WIDTHS = {font: _byte_widths(font) for font in ("regular", "bold")}


def to_winansi(text: str) -> bytes:
    """Encodes text as WinAnsi; never raises.

    Known symbols are transliterated, letters outside cp1252 lose their
    accent, control characters and stray combining marks are dropped and
    anything else becomes '?'.
    """
    raw = str(text)
    if raw.isascii():
        return raw.translate(_ASCII_TRANS).encode("ascii")
    if len(raw) > _CACHE_MAX_CHARS:
        return _to_winansi_slow.__wrapped__(raw)
    return _to_winansi_slow(raw)


@functools.lru_cache(maxsize=4096)
def _to_winansi_slow(raw: str) -> bytes:
    out = bytearray()
    text = unicodedata.normalize("NFC", raw)
    for char in text.translate(_TRANSLITERATIONS):
        if unicodedata.category(char) in ("Cc", "Cf", "Cs", "Mn"):
            continue
        try:
            out += char.encode("cp1252")
        except UnicodeEncodeError:
            base = unicodedata.normalize("NFKD", char)
            out += base.encode("ascii", "ignore") or b"?"
    return bytes(out)


def _measure(text: str, font: str) -> int:
    """Width of text in 1/1000 em, from the WinAnsi bytes drawn."""
    raw = to_winansi(text)
    if font == "mono":
        return len(raw) * _COURIER_WIDTH
    return sum(map(_BYTE_WIDTHS[font].__getitem__, raw))


_measure_cached = functools.lru_cache(maxsize=16384)(_measure)


def text_width(text: str, font: str, size: float) -> float:
    """Width in points of text as it will be drawn."""
    text = str(text)
    measure = _measure if len(text) > _CACHE_MAX_CHARS else _measure_cached
    return measure(text, font) * size / 1000


@dataclasses.dataclass(frozen=True)
class Style:
    """How a run of text is drawn; face is a FONTS key."""

    face: str = "regular"
    size: float = 9
    color: str = "#000000"

    def width(self, text: str) -> float:
        return text_width(text, self.face, self.size)


def _wrap_lines(
    raw: str, face: str, size: float, width: float
) -> tuple[str, ...]:
    style = Style(face, size)
    lines = []
    for paragraph in raw.split("\n"):
        line = ""
        for word in paragraph.split(" "):
            candidate = f"{line} {word}" if line else word
            if style.width(candidate) <= width:
                line = candidate
                continue
            if line:
                lines.append(line)
            while len(word) > 1 and style.width(word) > width:
                cut = _cut(word, style, width)
                lines.append(word[:cut])
                word = word[cut:]
            line = word
        lines.append(line)
    return tuple(lines)


_wrap_lines_cached = functools.lru_cache(maxsize=8192)(_wrap_lines)


def wrap_text(text: str, style: Style, width: float) -> list[str]:
    """Greedy word wrap; over-long words are broken by _cut."""
    width = max(width, 1.0)  # At worst one character per line, never a hang.
    text = str(text)
    wrap = _wrap_lines if len(text) > _CACHE_MAX_CHARS else _wrap_lines_cached
    return list(wrap(text, style.face, style.size, width))


def _cut(word: str, style: Style, width: float) -> int:
    """Where to break a word wider than width.

    After the last hyphen that fits (node and test names), else into
    near-equal pieces, so a 64-hex digest becomes two halves rather than a
    line and a stub.
    """
    hyphens = [
        i + 1
        for i, char in enumerate(word)
        if char == "-" and style.width(word[: i + 1]) <= width
    ]
    if hyphens:
        return hyphens[-1]
    cut = math.ceil(len(word) / math.ceil(style.width(word) / width))
    while cut > 1 and style.width(word[:cut]) > width:
        cut -= 1
    return max(1, cut)


def _escape(raw: bytes) -> bytes:
    return (
        raw.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")
    )


def _pdf_text(text: str) -> str:
    """text as an escaped WinAnsi PDF string literal body."""
    return _escape(to_winansi(text)).decode("latin-1")


_pdf_text_cached = functools.lru_cache(maxsize=8192)(_pdf_text)


@functools.lru_cache(maxsize=128)
def _rgb(color: str) -> str:
    """'#rrggbb' -> PDF colour operands."""
    return " ".join(f"{int(color[i : i + 2], 16) / 255:.3f}" for i in (1, 3, 5))


class Page:
    """One page's content stream, built from drawing calls."""

    def __init__(self, width: float, height: float):
        self.width, self.height = width, height
        self._ops: list[str] = []

    def rect(self, x, y, w, h, fill=None, stroke=None, line_width=0.6):
        """Draws a rectangle whose top-left corner is (x, y)."""
        if not (fill or stroke):
            return
        paint = "B" if fill and stroke else "f" if fill else "S"
        ops = ["q"]
        if fill:
            ops.append(f"{_rgb(fill)} rg")
        if stroke:
            ops.append(f"{_rgb(stroke)} RG {line_width:.2f} w")
        ops.append(f"{x:.2f} {self.height - y - h:.2f} {w:.2f} {h:.2f} re")
        self._ops.append(" ".join(ops) + f" {paint} Q")

    def line(self, x1, y1, x2, y2, color="#000000", width=0.6, dash=None):
        dash_op = f"[{' '.join(map(str, dash))}] 0 d " if dash else ""
        self._ops.append(
            f"q {_rgb(color)} RG {width:.2f} w {dash_op}"
            f"{x1:.2f} {self.height - y1:.2f} m "
            f"{x2:.2f} {self.height - y2:.2f} l S Q"
        )

    def text(self, x, y, text, style=Style(), align="left", angle: float = 0):
        """Draws one line of text whose baseline is at y."""
        if align != "left":
            x -= style.width(text) / (2 if align == "center" else 1)
        font_ref = list(FONTS).index(style.face) + 1
        string = str(text)
        string = (
            _pdf_text(string)
            if len(string) > _CACHE_MAX_CHARS
            else _pdf_text_cached(string)
        )
        if angle:
            rad = math.radians(angle)
            cos, sin = math.cos(rad), math.sin(rad)
            self._ops.append(
                f"BT /F{font_ref} {style.size:.2f} Tf {_rgb(style.color)} rg "
                f"{cos:.4f} {sin:.4f} {-sin:.4f} {cos:.4f} "
                f"{x:.2f} {self.height - y:.2f} Tm ({string}) Tj ET"
            )
            return
        self._ops.append(
            f"BT /F{font_ref} {style.size:.2f} Tf {_rgb(style.color)} rg "
            f"{x:.2f} {self.height - y:.2f} Td ({string}) Tj ET"
        )

    def text_lines(self, x, top, lines, style, leading):
        """Draws lines downwards from top; returns the y below the last."""
        for line in lines:
            top += leading
            self.text(x, top - style.size * 0.3, line, style)
        return top

    def content(self) -> bytes:
        return "\n".join(self._ops).encode("latin-1")


class Document:
    """An ordered list of pages, serialised by to_bytes()."""

    def __init__(self, page_size=A3_LANDSCAPE, title=""):
        self.page_size = page_size
        self.title = title
        self.pages: list[Page] = []

    def add_page(self) -> Page:
        self.pages.append(Page(*self.page_size))
        return self.pages[-1]

    def to_bytes(self) -> bytes:
        objects: list[bytes] = []  # objects[i] is object number i + 1.

        def add(body: bytes) -> int:
            objects.append(body)
            return len(objects)

        add(b"<< /Type /Catalog /Pages 2 0 R >>")
        add(b"")  # The page tree, written once the page numbers are known.
        fonts = " ".join(
            f"/F{index} %d 0 R"
            % add(
                f"<< /Type /Font /Subtype /Type1 /BaseFont /{name} "
                "/Encoding /WinAnsiEncoding >>".encode()
            )
            for index, name in enumerate(FONTS.values(), start=1)
        )
        width, height = self.page_size
        kids = []
        for page in self.pages:
            data = zlib.compress(page.content())
            stream = add(
                b"<< /Length %d /Filter /FlateDecode >>\nstream\n" % len(data)
                + data
                + b"\nendstream"
            )
            kids.append(
                "%d 0 R"
                % add(
                    f"<< /Type /Page /Parent 2 0 R "
                    f"/MediaBox [0 0 {width} {height}] "
                    f"/Resources << /Font << {fonts} >> >> "
                    f"/Contents {stream} 0 R >>".encode()
                )
            )
        objects[1] = (
            f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
        ).encode()
        title = ("\ufeff" + self.title).encode("utf-16-be").hex()
        info = add(
            f"<< /Title <{title}> "
            "/Producer (Cluster Health Scanner) >>".encode()
        )

        out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = []
        for number, body in enumerate(objects, start=1):
            offsets.append(len(out))
            out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
        xref = len(out)
        out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
        out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
        out += b"trailer\n<< /Size %d /Root 1 0 R /Info %d 0 R >>\n" % (
            len(objects) + 1,
            info,
        )
        out += b"startxref\n%d\n%%%%EOF\n" % xref
        return bytes(out)
