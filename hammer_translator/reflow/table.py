"""Structured table planning and drawing for reflowed PDFs."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any

import fitz


_CELL_PADDING_PT = 1.5
_MIN_TABLE_FONT_PT = 4.0
_DEFAULT_TABLE_FONT_PT = 10.0
_MIN_OVERLAP_RATIO = 0.05
_CROSS_PAGE_X_TOLERANCE_PT = 3.0
_CROSS_PAGE_MAX_Y_FRACTION = 0.55
_DEFAULT_TABLE_FONT_ID = "zh_body_regular"


@dataclass(frozen=True)
class HtmlCell:
    text: str
    colspan: int
    rowspan: int


@dataclass(frozen=True)
class TableCellPlacement:
    page_idx: int
    bbox: tuple[float, float, float, float]
    text: str
    lines: tuple[str, ...]
    font_size: float


@dataclass(frozen=True)
class TableRenderJob:
    item_index: int
    block_id: str
    unit_id: str
    regions: tuple[tuple[int, tuple[float, float, float, float]], ...]
    cells: tuple[TableCellPlacement, ...]
    font_id: str
    color: tuple[float, float, float]


@dataclass(frozen=True)
class TablePlan:
    jobs: tuple[TableRenderJob, ...]
    covered_block_ids: frozenset[str]
    errors: dict[str, str]


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[HtmlCell]] = []
        self._row: list[HtmlCell] | None = None
        self._cell: dict[str, Any] | None = None

    @staticmethod
    def _span(attrs: list[tuple[str, str | None]], name: str) -> int:
        value = dict(attrs).get(name)
        try:
            return max(1, int(str(value)))
        except (TypeError, ValueError):
            return 1

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"}:
            if self._row is None:
                self._row = []
            self._cell = {
                "text": "",
                "colspan": self._span(attrs, "colspan"),
                "rowspan": self._span(attrs, "rowspan"),
            }
        elif tag == "br" and self._cell is not None:
            self._cell["text"] += " "

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell["text"] += data

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self._cell is not None:
            if self._row is None:
                self._row = []
            self._row.append(
                HtmlCell(
                    text=str(self._cell["text"]),
                    colspan=int(self._cell["colspan"]),
                    rowspan=int(self._cell["rowspan"]),
                )
            )
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None

    def close(self) -> None:
        super().close()
        if self._row:
            self.rows.append(self._row)
            self._row = None


_LATEX_SYMBOLS = {
    "ldots": "…",
    "dots": "…",
    "cdots": "…",
    "in": "∈",
    "notin": "∉",
    "times": "×",
    "le": "≤",
    "leq": "≤",
    "ge": "≥",
    "geq": "≥",
    "neq": "≠",
    "ne": "≠",
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ε",
    "theta": "θ",
    "lambda": "λ",
    "mu": "μ",
    "sigma": "σ",
    "tau": "τ",
    "phi": "φ",
    "psi": "ψ",
    "omega": "ω",
    "sum": "∑",
    "prod": "∏",
    "infty": "∞",
    "rightarrow": "→",
    "leftarrow": "←",
    "to": "→",
    "cdot": "·",
    "pm": "±",
    "approx": "≈",
    "equiv": "≡",
    "subset": "⊂",
    "subseteq": "⊆",
    "cup": "∪",
    "cap": "∩",
    "prime": "′",
}

_LATEX_FORMATTING = {
    "left",
    "right",
    "boldsymbol",
    "mathbf",
    "mathrm",
    "mathit",
    "text",
    "displaystyle",
    "textstyle",
    "quad",
    "qquad",
    "begin",
    "end",
    "matrix",
    "aligned",
    "overline",
    "bar",
    "breve",
    "check",
    "ddot",
    "dot",
    "grave",
    "hat",
    "tilde",
    "vec",
}


_SUPERSCRIPT_TRANSLATION = str.maketrans(
    {
        "0": "⁰",
        "1": "¹",
        "2": "²",
        "3": "³",
        "4": "⁴",
        "5": "⁵",
        "6": "⁶",
        "7": "⁷",
        "8": "⁸",
        "9": "⁹",
        "+": "⁺",
        "-": "⁻",
        "(": "⁽",
        ")": "⁾",
        "n": "ⁿ",
        "i": "ⁱ",
        "L": "ᴸ",
        "R": "ᴿ",
        "E": "ᴱ",
        "k": "ᵏ",
        "p": "ᵖ",
    }
)

_SUBSCRIPT_TRANSLATION = str.maketrans(
    {
        "0": "₀",
        "1": "₁",
        "2": "₂",
        "3": "₃",
        "4": "₄",
        "5": "₅",
        "6": "₆",
        "7": "₇",
        "8": "₈",
        "9": "₉",
        "+": "₊",
        "-": "₋",
        "(": "₍",
        ")": "₎",
        "a": "ₐ",
        "e": "ₑ",
        "i": "ᵢ",
        "k": "ₖ",
        "m": "ₘ",
        "n": "ₙ",
        "o": "ₒ",
        "p": "ₚ",
        "r": "ᵣ",
        "s": "ₛ",
        "t": "ₜ",
        "u": "ᵤ",
        "v": "ᵥ",
        "x": "ₓ",
    }
)

_SUBSCRIPT_CHARACTERS = frozenset("₀₁₂₃₄₅₆₇₈₉₊₋₍₎ₐₑᵢₖₘₙₒₚᵣₛₜᵤᵥₓ")
_SCRIPT_PATTERN = re.compile(r"([_^])\{([^{}]*)\}")


def _convert_latex_scripts(text: str) -> str:
    """Render simple LaTeX sub/superscripts with Unicode where possible.

    Table cells are drawn as plain text, so leaving ``^``/``_`` markers makes
    formulas such as ``10^{-3}`` much harder to read than the source PDF.
    The bundled CJK font covers the Unicode super/subscript ranges used here.
    """

    text = re.sub(r"\s*([_^])\s*", r"\1", text)
    previous = None
    while previous != text:
        previous = text

        def replace(match: re.Match[str]) -> str:
            marker, content = match.group(1), match.group(2)
            content = re.sub(r"\s+", "", _convert_latex_scripts(content))
            if marker == "^":
                return "".join(
                    character
                    if character in _SUBSCRIPT_CHARACTERS
                    else character.translate(_SUPERSCRIPT_TRANSLATION)
                    for character in content
                )
            return content.translate(_SUBSCRIPT_TRANSLATION)

        text = _SCRIPT_PATTERN.sub(replace, text)
    text = re.sub(
        r"\^([A-Za-z0-9])",
        lambda match: match.group(1).translate(_SUPERSCRIPT_TRANSLATION),
        text,
    )
    text = re.sub(
        r"_([A-Za-z0-9])",
        lambda match: match.group(1).translate(_SUBSCRIPT_TRANSLATION),
        text,
    )
    return text


def _plain_table_text(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", value)
    text = text.replace("\\\\", " ")
    text = re.sub(
        r"\\([A-Za-z]+)",
        lambda match: (
            ""
            if match.group(1) in _LATEX_FORMATTING
            else _LATEX_SYMBOLS.get(match.group(1), match.group(1))
        ),
        text,
    )
    text = text.replace("\\{", "⟦").replace("\\}", "⟧").replace("\\_", "_")
    text = text.replace("\\%", "%").replace("\\&", "&")
    text = text.replace("⟦", "{").replace("⟧", "}")
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s*([_^])\s*", r"\1", text)
    text = _convert_latex_scripts(text)
    text = text.replace("{", "").replace("}", "").replace("$", "")
    text = re.sub(r"\s*_\s*", "_", text)
    text = re.sub(r"\s*\^\s*", "^", text)
    text = re.sub(r"(?<=\d)\s+(?=\d)", "", text)
    text = re.sub(r"(?<=\d)\s*\.\s*(?=\d)", ".", text)
    text = re.sub(r"(?<=\d)\s*\.\s*(?=,)", ".", text)
    text = re.sub(r"\.\s+(?=\d)", ".", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(
        r"(?<![₀₁₂₃₄₅₆₇₈₉])\s*,\s*(?![₀₁₂₃₄₅₆₇₈₉])",
        ", ",
        text,
    )
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _parse_html_table(markup: str) -> list[list[HtmlCell]]:
    parser = _TableParser()
    parser.feed(markup)
    parser.close()
    return parser.rows


def _split_merged_row_labels(rows: list[list[HtmlCell]]) -> list[list[HtmlCell]]:
    """Split labels such as ``456`` that MinerU merged across three rows.

    The source PDF keeps one label per row, but when a table has no internal
    horizontal rules MinerU can concatenate the first-column labels.  Only an
    exactly consecutive run is split, so ordinary multi-digit labels such as
    ``10`` are left untouched.
    """

    result = [list(row) for row in rows]
    for index, row in enumerate(result):
        if not row:
            continue
        label = row[0].text.strip()
        if len(label) < 2 or not label.isdigit():
            continue
        span = 0
        while (
            index + span + 1 < len(result)
            and result[index + span + 1]
            and not result[index + span + 1][0].text.strip()
        ):
            span += 1
        if span == 0 or len(label) != span + 1:
            continue
        if any(int(label[offset]) != int(label[0]) + offset for offset in range(len(label))):
            continue
        for offset, digit in enumerate(label):
            target = result[index + offset]
            first = target[0]
            result[index + offset] = [
                HtmlCell(text=digit, colspan=first.colspan, rowspan=first.rowspan),
                *target[1:],
            ]
    return result


def _rect_tuple(rect: Any) -> tuple[float, float, float, float]:
    value = fitz.Rect(rect)
    return (float(value.x0), float(value.y0), float(value.x1), float(value.y1))


def _overlap_ratio(first: Any, second: Any) -> float:
    left = fitz.Rect(first)
    right = fitz.Rect(second)
    intersection = left & right
    if intersection.is_empty:
        return 0.0
    denominator = min(left.get_area(), right.get_area())
    return intersection.get_area() / denominator if denominator else 0.0


def _cluster_axis(values: list[float], *, tolerance: float = 1.0) -> list[float]:
    ordered = sorted(values)
    groups: list[list[float]] = []
    for value in ordered:
        if groups and abs(value - groups[-1][-1]) <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [sum(group) / len(group) for group in groups]


def _axis_with_endpoints(
    values: list[float],
    start: float,
    end: float,
    *,
    tolerance: float = 1.5,
) -> list[float]:
    bounds = [value for value in values if start - tolerance <= value <= end + tolerance]
    if not bounds or abs(bounds[0] - start) > tolerance:
        bounds.insert(0, start)
    else:
        bounds[0] = start
    if abs(bounds[-1] - end) > tolerance:
        bounds.append(end)
    else:
        bounds[-1] = end
    return bounds


def _grid_rows_from_drawings(
    page: fitz.Page,
    table_bbox: fitz.Rect,
    target_rows: int,
) -> list[list[tuple[float, float, float, float]]] | None:
    """Reconstruct under-detected table rows from the drawn rule segments.

    PyMuPDF's ``find_tables`` merges data rows when a table has no internal
    horizontal rules.  The PDF still contains one vertical rule segment per
    row, so their endpoints provide the true row grid.
    """

    x_values: list[float] = []
    y_values: list[float] = []
    tolerance = 1.5
    for drawing in page.get_drawings():
        rect = fitz.Rect(drawing.get("rect") or (0, 0, 0, 0))
        if rect.width == 0 and rect.height == 0:
            continue
        if (
            rect.x1 < table_bbox.x0 - tolerance
            or rect.x0 > table_bbox.x1 + tolerance
            or rect.y1 < table_bbox.y0 - tolerance
            or rect.y0 > table_bbox.y1 + tolerance
        ):
            continue
        if rect.width <= tolerance and rect.height > tolerance:
            x_values.append(rect.x0)
            y_values.extend((rect.y0, rect.y1))
        elif rect.height <= tolerance and rect.width > tolerance:
            y_values.append(rect.y0)
            x_values.extend((rect.x0, rect.x1))
    x_bounds = _axis_with_endpoints(
        _cluster_axis(x_values), table_bbox.x0, table_bbox.x1
    )
    y_bounds = _axis_with_endpoints(
        _cluster_axis(y_values), table_bbox.y0, table_bbox.y1
    )
    if len(x_bounds) < 2 or len(y_bounds) - 1 != target_rows:
        return None
    return [
        [
            (x_bounds[column], y_bounds[row], x_bounds[column + 1], y_bounds[row + 1])
            for column in range(len(x_bounds) - 1)
        ]
        for row in range(len(y_bounds) - 1)
    ]


def _find_table_regions(document: fitz.Document, item: dict[str, Any], required_rows: int):
    page_idx = int(item.get("page_idx") or 0)
    if page_idx >= document.page_count:
        return []
    page = document[page_idx]
    candidates = list(page.find_tables().tables)
    if not candidates:
        return []
    primary = max(candidates, key=lambda table: _overlap_ratio(table.bbox, item.get("bbox")))
    if _overlap_ratio(primary.bbox, item.get("bbox")) <= _MIN_OVERLAP_RATIO:
        return []
    regions: list[tuple[int, Any]] = [(page_idx, primary)]
    total_rows = len(primary.rows)
    next_page = page_idx + 1
    x0, _, x1, _ = primary.bbox
    while total_rows < required_rows and next_page < document.page_count:
        next_candidates = list(document[next_page].find_tables().tables)
        candidates = [
            table
            for table in next_candidates
            if abs(float(table.bbox[0]) - float(x0)) <= _CROSS_PAGE_X_TOLERANCE_PT
            and abs(float(table.bbox[2]) - float(x1)) <= _CROSS_PAGE_X_TOLERANCE_PT
            and float(table.bbox[1]) <= document[next_page].rect.height * _CROSS_PAGE_MAX_Y_FRACTION
        ]
        if not candidates:
            break
        continuation = min(candidates, key=lambda table: float(table.bbox[1]))
        if not continuation.rows:
            break
        regions.append((next_page, continuation))
        total_rows += len(continuation.rows)
        next_page += 1
    return regions


def _row_cells(row: Any) -> list[tuple[float, float, float, float] | None]:
    cells: list[tuple[float, float, float, float] | None] = []
    for cell in row.cells:
        cells.append(None if cell is None else _rect_tuple(cell))
    return cells


class TableStructureError(ValueError):
    """The translated HTML cannot be mapped onto the source table grid."""


def _html_row_width(html_row: list[HtmlCell]) -> int:
    return sum(max(1, html_cell.colspan) for html_cell in html_row)


def _union_cells(
    cells: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    return (
        min(cell[0] for cell in cells),
        min(cell[1] for cell in cells),
        max(cell[2] for cell in cells),
        max(cell[3] for cell in cells),
    )


def _pair_row_cells(
    html_row: list[HtmlCell],
    destination_cells: list[tuple[float, float, float, float] | None],
) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Pair translated HTML cells with source cells without shifting columns.

    Exact grid matches keep empty cells positional and merge the destination
    boxes of ``colspan`` cells.  A fallback is allowed only for unspanned rows
    where the number of non-empty translated cells equals the number of
    concrete source cells; anything else is ambiguous and must be reviewed.
    """

    html_width = _html_row_width(html_row)
    if html_width == len(destination_cells):
        pairs: list[tuple[str, tuple[float, float, float, float]]] = []
        column = 0
        for html_cell in html_row:
            span = max(1, html_cell.colspan)
            text = _plain_table_text(html_cell.text)
            segment = destination_cells[column : column + span]
            columns = [cell for cell in segment if cell is not None]
            column += span
            if not columns:
                if text:
                    raise TableStructureError(
                        f"table_nonempty_cell_has_no_source_column:{column - span}"
                    )
                continue
            if span > 1 or len(columns) > 1:
                if len(columns) != span:
                    raise TableStructureError(
                        f"table_colspan_mismatch:{span}:{len(columns)}"
                    )
                destination = _union_cells(columns)
            else:
                destination = columns[0]
            pairs.append((text, destination))
        return pairs

    if any(
        html_cell.colspan != 1 or html_cell.rowspan != 1 for html_cell in html_row
    ):
        raise TableStructureError(
            f"table_span_grid_mismatch:{html_width}:{len(destination_cells)}"
        )
    translated = [
        _plain_table_text(html_cell.text)
        for html_cell in html_row
        if _plain_table_text(html_cell.text)
    ]
    concrete = [cell for cell in destination_cells if cell is not None]
    if len(translated) != len(concrete):
        raise TableStructureError(
            f"table_column_count_mismatch:{len(translated)}:{len(concrete)}"
        )
    return list(zip(translated, concrete))


def _text_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:[._:/+\-][A-Za-z0-9]+)*|\s+|.", text)


def _wrap_text(text: str, width: float, font: fitz.Font, font_size: float) -> list[str]:
    if width <= 0 or not text:
        return []
    lines: list[str] = []
    current = ""
    for token in _text_tokens(text):
        if token.isspace():
            if current and not current.endswith(" "):
                current += " "
            continue
        candidate = current + token
        if current and font.text_length(candidate, fontsize=font_size) > width:
            lines.append(current.rstrip())
            current = token
            continue
        if not current and font.text_length(token, fontsize=font_size) > width:
            broken = ""
            for character in token:
                if broken and font.text_length(broken + character, fontsize=font_size) > width:
                    lines.append(broken)
                    broken = character
                else:
                    broken += character
            current = broken
            continue
        current = candidate
    if current.strip():
        lines.append(current.rstrip())
    return lines


def _cell_fits(
    text: str,
    bbox: tuple[float, float, float, float],
    font: fitz.Font,
    font_size: float,
) -> bool:
    x0, y0, x1, y1 = bbox
    width = max(0.0, x1 - x0 - 2 * _CELL_PADDING_PT)
    height = max(0.0, y1 - y0 - 2 * _CELL_PADDING_PT)
    lines = _wrap_text(text, width, font, font_size)
    if not lines:
        return True
    required_height = len(lines) * font_size * 1.15
    return required_height <= height + 0.01


def _choose_font_size(
    cells: list[tuple[str, tuple[float, float, float, float]]],
    font: fitz.Font,
    base_size: float,
) -> float:
    size = max(_MIN_TABLE_FONT_PT, min(_DEFAULT_TABLE_FONT_PT, base_size))
    while size >= _MIN_TABLE_FONT_PT:
        if all(_cell_fits(text, bbox, font, size) for text, bbox in cells):
            return size
        size -= 0.25
    return _MIN_TABLE_FONT_PT


def plan_table_jobs(
    artifact: dict[str, Any],
    source_pdf: Path,
    profile: dict[str, Any],
) -> TablePlan:
    items = artifact.get("items") or []
    table_items = [
        (index, item)
        for index, item in enumerate(items)
        if str(item.get("type") or "").lower() == "table"
        or str(item.get("detected_type") or "").lower() == "table"
    ]
    if not table_items:
        return TablePlan((), frozenset(), {})

    document = fitz.open(source_pdf)
    jobs: list[TableRenderJob] = []
    errors: dict[str, str] = {}
    table_roles = (profile.get("roles") or {}).get("table") or {}
    font_id = str((table_roles.get("target_style") or {}).get("font_id") or _DEFAULT_TABLE_FONT_ID)
    base_size = float((table_roles.get("target_style") or {}).get("font_size_pt") or _DEFAULT_TABLE_FONT_PT)
    font_record = (profile.get("fonts") or {}).get(font_id) or {}
    font = fitz.Font(fontfile=str(font_record.get("path") or ""))
    color_value = ((table_roles.get("source_statistics") or {}).get("dominant_color") or "#000000")
    color = tuple(int(str(color_value)[index : index + 2], 16) / 255.0 for index in (1, 3, 5))
    try:
        for item_index, item in table_items:
            block_id = str(item.get("block_id") or "")
            html_rows = _split_merged_row_labels(
                _parse_html_table(str(item.get("translated") or ""))
            )
            if not html_rows:
                errors[block_id] = "table_translation_has_no_rows"
                continue
            regions = _find_table_regions(document, item, len(html_rows))
            if not regions:
                errors[block_id] = "table_structure_unavailable"
                continue
            physical_rows: list[tuple[int, list[tuple[float, float, float, float] | None]]] = []
            for page_idx, table in regions:
                for row in table.rows:
                    physical_rows.append((page_idx, _row_cells(row)))
            if len(regions) == 1 and len(physical_rows) < len(html_rows):
                page_idx, table = regions[0]
                grid = _grid_rows_from_drawings(
                    document[page_idx],
                    fitz.Rect(table.bbox),
                    len(html_rows),
                )
                if grid:
                    physical_rows = [(page_idx, row) for row in grid]
            if len(physical_rows) != len(html_rows):
                errors[block_id] = (
                    f"table_row_count_mismatch:{len(physical_rows)}:{len(html_rows)}"
                )
                continue
            specs: list[tuple[int, str, tuple[float, float, float, float]]] = []
            try:
                for row_index, html_row in enumerate(html_rows):
                    page_idx, physical = physical_rows[row_index]
                    specs.extend(
                        (page_idx, text, destination_cell)
                        for text, destination_cell in _pair_row_cells(html_row, physical)
                    )
            except TableStructureError as exc:
                errors[block_id] = str(exc)
                continue
            if not specs:
                errors[block_id] = "table_translation_has_no_text"
                continue
            font_size = _choose_font_size([(text, bbox) for _, text, bbox in specs], font, base_size)
            cells = tuple(
                TableCellPlacement(
                    page_idx=page_idx,
                    bbox=bbox,
                    text=text,
                    lines=tuple(
                        _wrap_text(
                            text,
                            max(0.0, bbox[2] - bbox[0] - 2 * _CELL_PADDING_PT),
                            font,
                            font_size,
                        )
                    ),
                    font_size=font_size,
                )
                for page_idx, text, bbox in specs
            )
            jobs.append(
                TableRenderJob(
                    item_index=item_index,
                    block_id=block_id,
                    unit_id=str(item.get("translation_unit_id") or ""),
                    regions=tuple(
                        (page_idx, _rect_tuple(table.bbox)) for page_idx, table in regions
                    ),
                    cells=cells,
                    font_id=font_id,
                    color=color,
                )
            )
    finally:
        document.close()

    table_block_ids = {job.block_id for job in jobs}
    covered: set[str] = set()
    for item in items:
        block_id = str(item.get("block_id") or "")
        if block_id in table_block_ids:
            continue
        item_bbox = item.get("bbox")
        page_idx = int(item.get("page_idx") or 0)
        if not isinstance(item_bbox, (list, tuple)) or len(item_bbox) != 4:
            continue
        for job in jobs:
            for region_page, region_bbox in job.regions:
                if region_page == page_idx and _overlap_ratio(item_bbox, region_bbox) >= 0.20:
                    covered.add(block_id)
                    break
            if block_id in covered:
                break
    return TablePlan(tuple(jobs), frozenset(covered), errors)


def draw_table_job(document: fitz.Document, job: TableRenderJob, font: fitz.Font) -> None:
    writers: dict[int, fitz.TextWriter] = {}
    for cell in job.cells:
        if not cell.lines:
            continue
        page = document[cell.page_idx]
        writer = writers.get(cell.page_idx)
        if writer is None:
            writer = fitz.TextWriter(page.rect, color=job.color)
            writers[cell.page_idx] = writer
        rect = fitz.Rect(cell.bbox)
        line_height = cell.font_size * 1.15
        y = rect.y0 + _CELL_PADDING_PT
        for line in cell.lines:
            baseline = y + (line_height - cell.font_size) / 2 + font.ascender * cell.font_size
            prefix = ""
            for character in line:
                if not character.isspace():
                    x = rect.x0 + _CELL_PADDING_PT + font.text_length(
                        prefix, fontsize=cell.font_size
                    )
                    writer.append((x, baseline), character, font=font, fontsize=cell.font_size)
                prefix += character
            y += line_height
    for page_idx, writer in writers.items():
        writer.write_text(document[page_idx], color=job.color, overlay=1)
    for page_idx, region in job.regions:
        page = document[page_idx]
        page.draw_rect(fitz.Rect(region), color=job.color, width=0.6, overlay=True)
    for cell in job.cells:
        document[cell.page_idx].draw_rect(
            fitz.Rect(cell.bbox), color=job.color, width=0.5, overlay=True
        )


def verify_table_job(document: fitz.Document, job: TableRenderJob) -> dict[str, Any]:
    expected = 0
    rendered = 0
    for cell in job.cells:
        if not cell.text.strip():
            continue
        expected += 1
        extracted = document[cell.page_idx].get_text("text", clip=fitz.Rect(cell.bbox)).strip()
        if extracted:
            rendered += 1
    region_nonempty = all(
        bool(document[page_idx].get_text("text", clip=fitz.Rect(region)).strip())
        for page_idx, region in job.regions
    )
    return {
        "text_layer_expected": True,
        "text_layer_nonempty": rendered > 0 and region_nonempty,
        "bbox_unchanged": True,
        "page_size_unchanged": True,
        "table_regions": len(job.regions),
        "table_cells_expected": expected,
        "table_cells_nonempty": rendered,
        "table_font_size_pt": job.cells[0].font_size if job.cells else None,
    }
