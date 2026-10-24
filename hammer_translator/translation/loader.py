"""Deterministic input discovery and validation for Translation Stage."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import fitz
from pydantic import ValidationError

from .authors import AFFILIATION_RE
from .schema import INPUT_SCHEMA_VERSION, InputBlock
from .taxonomy import (
    ElementType,
    TaxonomyDecision,
    TranslationPolicy,
    caption_element_type,
    classify_element,
)


class InputDiscoveryError(ValueError):
    """Raised when an input path cannot resolve to exactly one source JSON."""


class InputValidationError(ValueError):
    """Raised when a selected JSON does not satisfy the input contract."""


@dataclass(frozen=True)
class DiscoveredInput:
    source_json: Path
    selection_reason: str


@dataclass(frozen=True)
class LoadedBlock:
    block_id: str
    source_index: int
    data: InputBlock
    taxonomy: TaxonomyDecision
    raw: dict[str, Any]


@dataclass(frozen=True)
class LoadedDocument:
    schema_version: str
    document_id: str
    source_json: Path
    source_pdf: Path
    selection_reason: str
    source_hash: str
    page_count: int
    blocks: tuple[LoadedBlock, ...]

    def public_summary(self) -> dict[str, Any]:
        type_counts = Counter(block.taxonomy.element_type.value for block in self.blocks)
        policy_counts = Counter(block.taxonomy.policy.value for block in self.blocks)
        return {
            "schema_version": self.schema_version,
            "document_id": self.document_id,
            "source_json": str(self.source_json),
            "source_pdf": str(self.source_pdf),
            "selection_reason": self.selection_reason,
            "source_hash": self.source_hash,
            "page_count": self.page_count,
            "block_count": len(self.blocks),
            "type_counts": dict(sorted(type_counts.items())),
            "policy_counts": dict(sorted(policy_counts.items())),
        }


def _is_adapted_content_list(path: Path) -> bool:
    name = path.name
    return (
        name.endswith("_content_list.json")
        and not name.endswith("_mineru_content_list.json")
        and not name.endswith("_content_list_v2.json")
        and not name.endswith("_content_list_translated.json")
    )


def _one_candidate(candidates: list[Path], *, directory: Path, label: str) -> Path | None:
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise InputDiscoveryError(
            f"multiple {label} candidates in {directory}: {names}; pass one JSON file explicitly"
        )
    return candidates[0] if candidates else None


def discover_source(path: str | Path) -> DiscoveredInput:
    """Resolve one file or one parse-result directory using the approved order."""

    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise InputDiscoveryError(f"input not found: {source}")
    if source.is_file():
        if source.suffix.lower() != ".json":
            raise InputDiscoveryError(f"explicit input must be a JSON file: {source}")
        return DiscoveredInput(source_json=source, selection_reason="explicit_json")
    if not source.is_dir():
        raise InputDiscoveryError(f"input is neither a JSON file nor directory: {source}")

    enriched = sorted(source.glob("*_enriched.json"))
    selected = _one_candidate(enriched, directory=source, label="enriched JSON")
    if selected is not None:
        return DiscoveredInput(selected, "directory_preferred_enriched")

    adapted = sorted(path for path in source.glob("*_content_list.json") if _is_adapted_content_list(path))
    selected = _one_candidate(adapted, directory=source, label="adapted content-list JSON")
    if selected is not None:
        return DiscoveredInput(selected, "directory_fallback_adapted_content_list")

    raise InputDiscoveryError(
        f"no supported JSON in {source}; expected one *_enriched.json or adapted "
        "*_content_list.json. Native *_mineru_content_list.json and "
        "*_content_list_v2.json are never selected automatically"
    )


def discover_sources(paths: Sequence[str | Path]) -> list[DiscoveredInput]:
    if not paths:
        raise InputDiscoveryError("dry-run requires at least one input JSON file or directory")
    discovered = [discover_source(path) for path in paths]
    seen: set[Path] = set()
    for item in discovered:
        if item.source_json in seen:
            raise InputDiscoveryError(f"duplicate input resolves to {item.source_json}")
        seen.add(item.source_json)
    return discovered


def document_id_from_json(path: Path) -> str:
    stem = path.stem
    for suffix in (
        "_mineru_content_list",
        "_content_list_v2",
        "_content_list",
        "_enriched",
    ):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def discover_pdf(source_json: Path, explicit_pdf: str | Path | None = None) -> Path:
    if explicit_pdf is not None:
        pdf = Path(explicit_pdf).expanduser().resolve()
        if not pdf.is_file() or pdf.suffix.lower() != ".pdf":
            raise InputDiscoveryError(f"explicit PDF not found or not a PDF file: {pdf}")
        return pdf

    document_id = document_id_from_json(source_json)
    parent = source_json.parent
    for candidate in (parent / f"{document_id}_origin.pdf", parent / f"{document_id}.pdf"):
        if candidate.is_file():
            return candidate.resolve()

    origin_pdfs = sorted(parent.glob("*_origin.pdf"))
    selected = _one_candidate(origin_pdfs, directory=parent, label="origin PDF")
    if selected is not None:
        return selected.resolve()

    pdfs = sorted(path for path in parent.glob("*.pdf") if not path.name.endswith("_layout.pdf"))
    selected = _one_candidate(pdfs, directory=parent, label="PDF")
    if selected is not None:
        return selected.resolve()
    raise InputDiscoveryError(
        f"no source PDF found beside {source_json}; pass --pdf for a single input"
    )


def _load_json_array(path: Path) -> tuple[bytes, list[dict[str, Any]]]:
    raw_bytes = path.read_bytes()
    try:
        value = json.loads(raw_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise InputValidationError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, list):
        raise InputValidationError(f"translation input root must be a JSON array: {path}")
    if not value:
        raise InputValidationError(f"translation input contains no blocks: {path}")
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise InputValidationError(f"block[{index}] must be a JSON object")
    return raw_bytes, value


def _normalized_pdf_text(value: str) -> str:
    without_markup = re.sub(r"</?su[bp]>", "", value or "", flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", without_markup).strip().casefold()


def _caption_bbox_from_pdf(
    item: dict[str, Any],
    document: fitz.Document,
) -> tuple[float, float, float, float] | None:
    """Locate a chart block's real caption text instead of its graphic bbox."""

    text = str(item.get("text") or "")
    if caption_element_type(text) is None:
        return None
    page_idx = item.get("page_idx")
    if (
        not isinstance(page_idx, int)
        or isinstance(page_idx, bool)
        or page_idx < 0
        or page_idx >= document.page_count
    ):
        return None
    needle = _normalized_pdf_text(text)[:64]
    if not needle:
        return None

    blocks = document[page_idx].get_text("blocks")
    for match_contains in (False, True):
        for block in blocks:
            if len(block) < 7 or int(block[6]) != 0:
                continue
            block_text = _normalized_pdf_text(str(block[4]))
            matched = needle in block_text if match_contains else block_text.startswith(needle)
            if not matched:
                continue
            x0, y0, x1, y1 = (float(value) for value in block[:4])
            if x1 > x0 and y1 > y0:
                return (x0, y0, x1, y1)
    return None


MARGIN_BAND_RATIO = 0.08
REPEATED_MARGIN_PAGE_COUNT = 3
IMAGE_OVERLAP_RATIO = 0.45
PDF_BBOX_TOLERANCE = 2.0
MIN_RECOVERED_WORDS = 6
CAPTION_MARKER_RE = re.compile(
    r"\b(?:Fig(?:ure)?\.?\s*\d+|Table\s*\d+)\s*[.:：]",
    re.IGNORECASE,
)


def _bbox_union(
    rects: Sequence[Sequence[float]],
) -> tuple[float, float, float, float] | None:
    values = list(rects)
    if not values:
        return None
    return (
        min(float(rect[0]) for rect in values),
        min(float(rect[1]) for rect in values),
        max(float(rect[2]) for rect in values),
        max(float(rect[3]) for rect in values),
    )


def _horizontal_overlap(first: Sequence[float], second: Sequence[float]) -> float:
    return max(
        0.0,
        min(float(first[2]), float(second[2]))
        - max(float(first[0]), float(second[0])),
    )


def _vertical_gap(first: Sequence[float], second: Sequence[float]) -> float:
    return max(
        0.0,
        float(first[1]) - float(second[3]),
        float(second[1]) - float(first[3]),
    )


def _horizontal_gap(first: Sequence[float], second: Sequence[float]) -> float:
    return max(
        0.0,
        float(first[0]) - float(second[2]),
        float(second[0]) - float(first[2]),
    )


def _line_geometry_bbox(
    item: dict[str, Any],
) -> tuple[float, float, float, float] | None:
    rects: list[Sequence[float]] = []
    for line in item.get("lines") or []:
        if not isinstance(line, dict):
            continue
        line_bbox = line.get("bbox")
        if isinstance(line_bbox, (list, tuple)) and len(line_bbox) == 4:
            rects.append(line_bbox)
        for span in line.get("spans") or []:
            if not isinstance(span, dict):
                continue
            span_bbox = span.get("bbox")
            if isinstance(span_bbox, (list, tuple)) and len(span_bbox) == 4:
                rects.append(span_bbox)
    return _bbox_union(rects)


def _has_formula_span(item: dict[str, Any]) -> bool:
    for line in item.get("lines") or []:
        if not isinstance(line, dict):
            continue
        for span in line.get("spans") or []:
            if isinstance(span, dict) and str(span.get("type") or "") in {
                "inline_equation",
                "interline_equation",
                "equation",
            }:
                return True
    return False


def _caption_blocks_for_chart(
    item: dict[str, Any],
    document: fitz.Document,
    *,
    chart_text: str,
) -> tuple[list[Sequence[Any]], str] | None:
    """Return the PDF caption lines that belong to a mixed chart block."""

    page_idx = item.get("page_idx")
    bbox = item.get("bbox")
    if (
        not isinstance(page_idx, int)
        or isinstance(page_idx, bool)
        or not 0 <= page_idx < document.page_count
        or not isinstance(bbox, (list, tuple))
        or len(bbox) != 4
    ):
        return None
    marker = CAPTION_MARKER_RE.search(chart_text)
    if marker is None:
        return None
    page = document[page_idx]
    chart_bbox = tuple(float(value) for value in bbox)
    image_rects = _image_rects(page)
    parser_caption = _normalized_pdf_text(chart_text[marker.start() :])
    needle = parser_caption[:64]
    text_blocks = [
        block
        for block in page.get_text("blocks")
        if len(block) >= 7 and int(block[6]) == 0
    ]

    def block_text(block: Sequence[Any]) -> str:
        return re.sub(r"\s+", " ", str(block[4])).strip()

    def block_bbox(block: Sequence[Any]) -> tuple[float, float, float, float]:
        return tuple(float(value) for value in block[:4])

    candidates: list[
        tuple[float, int, Sequence[Any], tuple[float, float, float, float]]
    ] = []
    for index, block in enumerate(text_blocks):
        text = block_text(block)
        if caption_element_type(text) is None:
            continue
        candidate_bbox = block_bbox(block)
        if _in_page_margin(candidate_bbox, float(page.rect.height)):
            continue
        if _max_image_overlap(candidate_bbox, image_rects) >= IMAGE_OVERLAP_RATIO:
            continue
        normalized = _normalized_pdf_text(text)
        if needle and not (
            needle in normalized
            or normalized.startswith(needle[:40])
            or needle.startswith(normalized[:40])
        ):
            continue
        vertical_gap = _vertical_gap(candidate_bbox, chart_bbox)
        horizontal_gap = _horizontal_gap(candidate_bbox, chart_bbox)
        candidates.append(
            (
                vertical_gap * vertical_gap + horizontal_gap * horizontal_gap,
                index,
                block,
                candidate_bbox,
            )
        )

    if not candidates:
        # The parser text and the PDF text can diverge around inline formulas.
        # Fall back to the nearest real caption outside the chart image.
        for index, block in enumerate(text_blocks):
            text = block_text(block)
            if not text or caption_element_type(text) is None:
                continue
            candidate_bbox = block_bbox(block)
            if _max_image_overlap(candidate_bbox, image_rects) >= IMAGE_OVERLAP_RATIO:
                continue
            vertical_gap = _vertical_gap(candidate_bbox, chart_bbox)
            horizontal_gap = _horizontal_gap(candidate_bbox, chart_bbox)
            candidates.append(
                (
                    vertical_gap * vertical_gap + horizontal_gap * horizontal_gap,
                    index,
                    block,
                    candidate_bbox,
                )
            )
    if not candidates:
        return None

    _, start_index, _, start_bbox = min(
        candidates,
        key=lambda value: (value[0], value[1]),
    )
    selected: list[Sequence[Any]] = [text_blocks[start_index]]
    selected_bboxes: list[tuple[float, float, float, float]] = [start_bbox]
    last_bbox = start_bbox
    last_height = max(1.0, last_bbox[3] - last_bbox[1])
    for block in text_blocks[start_index + 1 :]:
        text = block_text(block)
        if not text:
            continue
        current_text = " ".join(block_text(item) for item in selected)
        if current_text.rstrip().endswith((".", "!", "?")):
            break
        next_bbox = block_bbox(block)
        if caption_element_type(text) is not None:
            break
        if _max_image_overlap(next_bbox, image_rects) >= IMAGE_OVERLAP_RATIO:
            break
        vertical_gap = next_bbox[1] - last_bbox[3]
        if vertical_gap > max(2.5, last_height * 1.5):
            break
        if _horizontal_overlap(next_bbox, start_bbox) <= 0:
            break
        selected.append(block)
        selected_bboxes.append(next_bbox)
        last_bbox = next_bbox
        last_height = max(1.0, last_bbox[3] - last_bbox[1])
        if len(selected) >= 4:
            break

    caption_text = " ".join(block_text(block) for block in selected)
    union = _bbox_union(selected_bboxes)
    if not caption_text or union is None:
        return None
    return selected, caption_text


def _split_mixed_chart_captions(
    items: list[dict[str, Any]],
    document: fitz.Document,
) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for item in items:
        raw_type = str(item.get("type") or "").strip().lower()
        text = str(item.get("text") or "")
        marker = CAPTION_MARKER_RE.search(text)
        if raw_type != "chart" or marker is None:
            expanded.append(item)
            continue
        geometry = _caption_blocks_for_chart(item, document, chart_text=text)
        if geometry is None:
            expanded.append(item)
            continue
        blocks, caption_text = geometry
        caption_bbox = _bbox_union(
            tuple(float(value) for value in block[:4]) for block in blocks
        )
        if caption_bbox is None:
            expanded.append(item)
            continue
        chart = deepcopy(item)
        chart["type"] = "image"
        chart["text"] = ""
        chart.pop("lines", None)
        chart.pop("rich_spans", None)
        caption_type = caption_element_type(caption_text)
        caption = {
            "type": caption_type.value if caption_type is not None else "figure_caption",
            "text": caption_text,
            "context": caption_text,
            "page_idx": item.get("page_idx"),
            "bbox": list(caption_bbox),
            "logical_para_id": item.get("logical_para_id"),
            "recovered_from_pdf": True,
        }
        expanded.extend((chart, caption))
    return expanded


def _split_textless_tables(
    items: list[dict[str, Any]],
    document: fitz.Document,
    page_data: Callable[[int], PageData],
) -> list[dict[str, Any]]:
    """Recover a parser-missed table as independently translatable cells."""

    expanded: list[dict[str, Any]] = []
    next_logical_id = max(
        (
            int(item["logical_para_id"])
            for item in items
            if isinstance(item.get("logical_para_id"), int)
        ),
        default=0,
    ) + 1
    for item in items:
        raw_type = str(item.get("type") or "").strip().lower()
        table_body = str(item.get("table_body") or "").strip()
        page_idx = item.get("page_idx")
        bbox = item.get("bbox")
        if (
            raw_type != "table"
            or table_body
            or not isinstance(page_idx, int)
            or isinstance(page_idx, bool)
            or not 0 <= page_idx < document.page_count
            or not isinstance(bbox, (list, tuple))
            or len(bbox) != 4
        ):
            expanded.append(item)
            continue
        table_bbox = tuple(float(value) for value in bbox)
        blocks, _ = page_data(page_idx)
        cells: list[tuple[Sequence[Any], str]] = []
        for block in blocks:
            if len(block) < 7 or int(block[6]) != 0:
                continue
            text = re.sub(r"\s+", " ", str(block[4])).strip()
            if not text:
                continue
            cell_bbox = tuple(float(value) for value in block[:4])
            area = _bbox_area(cell_bbox)
            if area <= 0:
                continue
            if _intersection_area(cell_bbox, table_bbox) / area < 0.7:
                continue
            cells.append((block, text))
        if len(cells) < 2:
            expanded.append(item)
            continue
        for block, text in cells:
            cell_bbox = tuple(float(value) for value in block[:4])
            expanded.append(
                {
                    "type": "text",
                    "text": text,
                    "context": text,
                    "page_idx": page_idx,
                    "bbox": list(cell_bbox),
                    "logical_para_id": next_logical_id,
                    "style_role": "table",
                    "recovered_from_pdf": True,
                }
            )
            next_logical_id += 1
    return expanded


def _bbox_area(bbox: Sequence[float]) -> float:
    x0, y0, x1, y1 = (float(value) for value in bbox)
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _intersection_area(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    x0 = max(float(first[0]), float(second[0]))
    y0 = max(float(first[1]), float(second[1]))
    x1 = min(float(first[2]), float(second[2]))
    y1 = min(float(first[3]), float(second[3]))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def _contains_bbox(
    outer: Sequence[float],
    inner: Sequence[float],
    *,
    tolerance: float = PDF_BBOX_TOLERANCE,
) -> bool:
    return (
        float(inner[0]) >= float(outer[0]) - tolerance
        and float(inner[1]) >= float(outer[1]) - tolerance
        and float(inner[2]) <= float(outer[2]) + tolerance
        and float(inner[3]) <= float(outer[3]) + tolerance
    )


def _image_rects(page: fitz.Page) -> list[tuple[float, float, float, float]]:
    rects: list[tuple[float, float, float, float]] = []
    for info in page.get_image_info():
        bbox = info.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        rect = tuple(float(value) for value in bbox)
        if _bbox_area(rect) > 0:
            rects.append(rect)
    return rects


def _max_image_overlap(
    bbox: Sequence[float],
    image_rects: Sequence[Sequence[float]],
) -> float:
    area = _bbox_area(bbox)
    if area <= 0:
        return 0.0
    return max(
        (_intersection_area(bbox, rect) / area for rect in image_rects),
        default=0.0,
    )


def _in_page_margin(bbox: Sequence[float], page_height: float) -> bool:
    band = page_height * MARGIN_BAND_RATIO
    return float(bbox[1]) <= band or float(bbox[3]) >= page_height - band


def _line_content(line: dict[str, Any]) -> str:
    parts: list[str] = []
    for span in line.get("spans") or []:
        if not isinstance(span, dict):
            continue
        content = str(span.get("content") or "")
        if not content:
            continue
        if str(span.get("type") or "").endswith("equation"):
            if not content.startswith("$"):
                content = f"${content}$"
        parts.append(content)
    return "".join(parts).strip()


def _join_line_text(lines: Sequence[str]) -> str:
    joined = ""
    for raw_line in lines:
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        if not joined:
            joined = line
        elif joined.endswith("-") and line[:1].islower():
            joined = joined[:-1] + line
        else:
            joined = f"{joined} {line}"
    return joined


def _split_multicolumn_text_item(item: dict[str, Any], document: fitz.Document) -> list[dict[str, Any]]:
    """Split a parser block whose lines belong to two disjoint columns."""

    if str(item.get("type") or "").strip().lower() != "text":
        return [item]
    page_idx = item.get("page_idx")
    bbox = item.get("bbox")
    lines = item.get("lines")
    if (
        not isinstance(page_idx, int)
        or isinstance(page_idx, bool)
        or not 0 <= page_idx < document.page_count
        or not isinstance(bbox, (list, tuple))
        or len(bbox) != 4
        or not isinstance(lines, list)
        or len(lines) < 2
    ):
        return [item]
    page_width = float(document[page_idx].rect.width)
    if float(bbox[2]) - float(bbox[0]) < page_width * 0.55:
        return [item]
    page_mid = page_width / 2
    groups: dict[str, list[dict[str, Any]]] = {"left": [], "right": []}
    for line in lines:
        if not isinstance(line, dict):
            return [item]
        line_bbox = line.get("bbox")
        if not isinstance(line_bbox, (list, tuple)) or len(line_bbox) != 4:
            return [item]
        width = float(line_bbox[2]) - float(line_bbox[0])
        if width <= 0 or width >= page_width * 0.55:
            return [item]
        center = (float(line_bbox[0]) + float(line_bbox[2])) / 2
        groups["left" if center < page_mid else "right"].append(line)
    if not groups["left"] or not groups["right"]:
        return [item]
    left_right = max(float(line["bbox"][2]) for line in groups["left"])
    right_left = min(float(line["bbox"][0]) for line in groups["right"])
    if right_left - left_right < 4.0:
        return [item]

    split: list[dict[str, Any]] = []
    for group in sorted(
        groups.values(), key=lambda values: min(float(line["bbox"][1]) for line in values)
    ):
        part = deepcopy(item)
        part["lines"] = deepcopy(group)
        part["bbox"] = list(
            _bbox_union(tuple(float(value) for value in line["bbox"]) for line in group)
            or bbox
        )
        text = _join_line_text([_line_content(line) for line in group])
        if not text:
            return [item]
        part["text"] = text
        part["context"] = text
        part.pop("block_id", None)
        rich_spans = item.get("rich_spans")
        if isinstance(rich_spans, list):
            part_is_right = float(part["bbox"][0]) >= page_mid - 4.0
            part["rich_spans"] = [
                span
                for span in rich_spans
                if isinstance(span, dict)
                and isinstance(span.get("bbox"), (list, tuple))
                and len(span["bbox"]) == 4
                and (
                    (float(span["bbox"][0]) + float(span["bbox"][2])) / 2 >= page_mid
                    if part_is_right
                    else (float(span["bbox"][0]) + float(span["bbox"][2])) / 2 < page_mid
                )
            ]
        split.append(part)
    return split


def _split_multicolumn_text_items(
    items: list[dict[str, Any]],
    document: fitz.Document,
) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    next_logical_id = max(
        (
            int(item["logical_para_id"])
            for item in items
            if isinstance(item.get("logical_para_id"), int)
        ),
        default=0,
    ) + 1
    for item in items:
        parts = _split_multicolumn_text_item(item, document)
        if len(parts) == 1:
            expanded.append(item)
            continue
        for index, part in enumerate(parts):
            if index:
                part["logical_para_id"] = next_logical_id
                next_logical_id += 1
            expanded.append(part)
    return expanded


def _compact_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _looks_like_math_fragment(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    if re.search(r"\(\s*[A-Z]?\d+[a-z]?\s*\)\s*$", text) and not re.search(
        r"[A-Za-z]{3,}", text
    ):
        return True
    words = re.findall(r"[A-Za-z]{2,}", text)
    return bool(re.search(r"[<>=≤≥^_{}\\]", text)) and len(words) < 2


def _compact_prefix_end(text: str, compact_length: int) -> int | None:
    seen = 0
    for index, character in enumerate(text):
        if re.fullmatch(r"[a-z0-9]", character.casefold()):
            seen += 1
            if seen == compact_length:
                return index + 1
    return None


def _text_match_score(source: str, target: str) -> float:
    if not source or not target:
        return 0.0
    source_compact = _compact_text(source)
    target_compact = _compact_text(target)
    if not source_compact or not target_compact:
        return 0.0
    if source == target:
        return 1.0
    if source_compact == target_compact:
        return 1.0
    if len(source_compact) >= len(target_compact) + 40 and source_compact.startswith(target_compact):
        return 0.95
    if len(source) >= 30 and source in target:
        coverage = len(source) / max(len(target), 1)
        if coverage >= 0.70:
            return 0.96
    if len(target) >= 30 and target in source:
        coverage = len(target) / max(len(source), 1)
        if coverage >= 0.70:
            return 0.90
    source_tokens = set(source.split())
    target_tokens = set(target.split())
    if len(source_tokens) >= MIN_RECOVERED_WORDS:
        coverage = len(source_tokens & target_tokens) / len(source_tokens)
        if coverage >= 0.85:
            return 0.70 + 0.15 * coverage
    return 0.0


def _match_pdf_text_block(
    text: str,
    blocks: Sequence[Sequence[Any]],
    *,
    source_bbox: Sequence[float] | None = None,
) -> tuple[tuple[float, float, float, float], str] | None:
    needle = _normalized_pdf_text(text)
    if len(needle) < 8:
        return None
    best_score = 0.0
    best_bbox: tuple[float, float, float, float] | None = None
    best_text = ""
    for block in blocks:
        if len(block) < 7 or int(block[6]) != 0:
            continue
        candidate = _normalized_pdf_text(str(block[4]))
        score = _text_match_score(needle, candidate)
        if score <= best_score:
            continue
        bbox = tuple(float(value) for value in block[:4])
        if _bbox_area(bbox) <= 0:
            continue
        best_score = score
        best_bbox = bbox
        best_text = str(block[4])
    if best_bbox is not None and best_score >= 0.85:
        return best_bbox, best_text
    if source_bbox is None:
        return None
    probe = _compact_text(needle)[:48]
    if len(probe) < 20:
        return None
    source_area = _bbox_area(source_bbox)
    if source_area <= 0:
        return None
    best_overlap = 0.0
    best_bbox = None
    best_text = ""
    for block in blocks:
        if len(block) < 7 or int(block[6]) != 0:
            continue
        bbox = tuple(float(value) for value in block[:4])
        overlap = _intersection_area(source_bbox, bbox) / source_area
        if overlap < 0.6:
            continue
        candidate_compact = _compact_text(_normalized_pdf_text(str(block[4])))
        if probe not in candidate_compact:
            continue
        if overlap > best_overlap:
            best_overlap = overlap
            best_bbox = bbox
            best_text = str(block[4])
    return (best_bbox, best_text) if best_bbox is not None else None


def _mark_repeated_margin_items(
    items: list[dict[str, Any]],
    document: fitz.Document,
) -> None:
    pages_by_text: defaultdict[str, set[int]] = defaultdict(set)
    for item in items:
        text = _normalized_pdf_text(str(item.get("text") or ""))
        page_idx = item.get("page_idx")
        bbox = item.get("bbox")
        if (
            len(text) < 6
            or not isinstance(page_idx, int)
            or isinstance(page_idx, bool)
            or not 0 <= page_idx < document.page_count
            or not isinstance(bbox, (list, tuple))
            or len(bbox) != 4
        ):
            continue
        if _in_page_margin(bbox, float(document[page_idx].rect.height)):
            pages_by_text[text].add(page_idx)

    repeated = {
        text
        for text, pages in pages_by_text.items()
        if len(pages) >= REPEATED_MARGIN_PAGE_COUNT
    }
    if not repeated:
        return

    for item in items:
        text = _normalized_pdf_text(str(item.get("text") or ""))
        page_idx = item.get("page_idx")
        bbox = item.get("bbox")
        if (
            text not in repeated
            or not isinstance(page_idx, int)
            or isinstance(page_idx, bool)
            or not 0 <= page_idx < document.page_count
            or not isinstance(bbox, (list, tuple))
            or len(bbox) != 4
        ):
            continue
        page_height = float(document[page_idx].rect.height)
        if not _in_page_margin(bbox, page_height):
            continue
        item["type"] = "header" if (float(bbox[1]) + float(bbox[3])) / 2 < page_height / 2 else "footer"


PageData = tuple[list[Sequence[Any]], list[tuple[float, float, float, float]]]
_AFFILIATION_LINE_RE = re.compile(
    rf"(?:{AFFILIATION_RE.pattern})|observator(?:y|ies)",
    re.IGNORECASE,
)
_AUTHOR_NAME_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'’.-]*")
_TEXT_BBOX_SKIP_TYPES = {
    "image",
    "table",
    "chart",
    "header",
    "footer",
    "page_number",
    "reference",
    "equation",
}


def _correct_text_bbox(
    item: dict[str, Any],
    document: fitz.Document,
    page_data: Callable[[int], PageData],
    *,
    allow_match: bool = True,
) -> None:
    raw_type = str(item.get("type") or "").strip().lower()
    if raw_type in _TEXT_BBOX_SKIP_TYPES:
        return
    text = str(item.get("text") or "")
    if not text.strip():
        return
    page_idx = item.get("page_idx")
    bbox = item.get("bbox")
    if (
        not isinstance(page_idx, int)
        or isinstance(page_idx, bool)
        or not 0 <= page_idx < document.page_count
        or not isinstance(bbox, (list, tuple))
        or len(bbox) != 4
    ):
        return
    if not allow_match:
        return
    blocks, image_rects = page_data(page_idx)
    matched = _match_pdf_text_block(text, blocks, source_bbox=bbox)
    if matched is None:
        return
    matched_bbox, matched_text = matched
    line_bbox = _line_geometry_bbox(item)
    if line_bbox is not None:
        line_area = _bbox_area(line_bbox)
        geometry_overlap = _intersection_area(matched_bbox, line_bbox) / max(line_area, 1.0)
        if geometry_overlap < 0.75:
            return
    source_normalized = _normalized_pdf_text(text)
    matched_normalized = _normalized_pdf_text(matched_text)
    source_compact = _compact_text(source_normalized)
    matched_compact = _compact_text(matched_normalized)
    if matched_compact.startswith(source_compact) and len(matched_compact) > len(source_compact):
        missing = len(matched_compact) - len(source_compact)
        tail_start = _compact_prefix_end(matched_text, len(source_compact))
        if tail_start is not None and missing <= max(160, len(source_compact) // 4):
            tail_raw = matched_text[tail_start:]
            tail = re.sub(r"\s+", " ", tail_raw).strip()
            if tail.startswith("-") and tail[1:2].islower():
                tail = tail[1:]
                separator = ""
            elif tail_raw[:1].isalnum() and not text[-1:].isspace():
                separator = ""
            else:
                separator = " "
            if tail and "$" not in tail and "\\" not in tail:
                extended = f"{text.rstrip()}{separator}{tail}"
                extended = re.sub(r"\s+", " ", extended).strip()
                item["text"] = extended
                item["context"] = extended
    matched_prefix_rejected = False
    if len(source_compact) >= 20 and len(matched_compact) > len(source_compact):
        probe_length = min(len(source_compact), 64)
        source_position = matched_compact.find(source_compact[:probe_length])
        if (
            0 < source_position
            and source_position + len(source_compact) >= len(matched_compact) * 0.85
        ):
            prefix_end = None
            leading_words = re.findall(r"[A-Za-z]{2,}", str(item.get("text") or ""))[:4]
            if leading_words:
                leading_pattern = r"\s+".join(re.escape(word) for word in leading_words)
                leading_match = re.search(leading_pattern, matched_text)
                if leading_match is not None and leading_match.start() > 0:
                    prefix_end = leading_match.start()
            if prefix_end is None:
                prefix_end = _compact_prefix_end(matched_text, source_position)
            if prefix_end is not None:
                prefix = re.sub(r"\s+", " ", matched_text[:prefix_end]).strip()
                if prefix and "$" not in prefix and "\\" not in prefix:
                    if _looks_like_math_fragment(prefix):
                        matched_prefix_rejected = True
                    else:
                        extended = f"{prefix} {str(item.get('text') or '').lstrip()}"
                        extended = re.sub(r"\s+", " ", extended).strip()
                        item["text"] = extended
                        item["context"] = extended
    if matched_prefix_rejected:
        return
    current = tuple(float(value) for value in bbox)
    overlap = _max_image_overlap(current, image_rects)
    if overlap >= IMAGE_OVERLAP_RATIO or not _contains_bbox(current, matched_bbox):
        corrected = matched_bbox
        if line_bbox is not None:
            union = _bbox_union((matched_bbox, line_bbox))
            corrected = union or matched_bbox
        item["bbox"] = list(corrected)


def _ensure_bbox_contains_line_geometry(item: dict[str, Any]) -> None:
    if _has_formula_span(item):
        # Formula placement has its own contract that validates the original
        # block/line/span geometry.  Expanding the block first would silently
        # turn an out-of-bounds formula into a seemingly valid one.
        return
    bbox = item.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return
    line_bbox = _line_geometry_bbox(item)
    if line_bbox is None or _contains_bbox(bbox, line_bbox, tolerance=0.0):
        return
    line_area = _bbox_area(line_bbox)
    overlap = _intersection_area(bbox, line_bbox) / max(line_area, 1.0)
    if overlap >= 0.5:
        union = _bbox_union((bbox, line_bbox))
        item["bbox"] = list(union or line_bbox)
    else:
        item["bbox"] = list(line_bbox)


def _recover_missing_text_items(
    items: list[dict[str, Any]],
    document: fitz.Document,
    page_data: Callable[[int], PageData],
) -> list[dict[str, Any]]:
    by_page: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        page_idx = item.get("page_idx")
        if isinstance(page_idx, int) and not isinstance(page_idx, bool):
            by_page[page_idx].append(item)

    next_logical_id = max(
        (
            int(item["logical_para_id"])
            for item in items
            if isinstance(item.get("logical_para_id"), int)
        ),
        default=0,
    ) + 1

    for page_idx in sorted(by_page):
        if not 0 <= page_idx < document.page_count:
            continue
        page = document[page_idx]
        blocks, image_rects = page_data(page_idx)
        page_height = float(page.rect.height)
        page_items = by_page[page_idx]
        existing_texts = [
            _normalized_pdf_text(str(item.get("text") or ""))
            for item in page_items
            if len(_normalized_pdf_text(str(item.get("text") or ""))) >= 30
        ]
        occupied = [
            tuple(float(value) for value in item["bbox"])
            for item in page_items
            if isinstance(item.get("bbox"), (list, tuple)) and len(item["bbox"]) == 4
        ]
        recovered: list[dict[str, Any]] = []
        for block in blocks:
            if len(block) < 7 or int(block[6]) != 0:
                continue
            text = re.sub(r"\s+", " ", str(block[4])).strip()
            normalized = _normalized_pdf_text(text)
            if len(re.findall(r"[A-Za-z]{3,}", text)) < MIN_RECOVERED_WORDS:
                continue
            caption_type = caption_element_type(text)
            bbox = tuple(float(value) for value in block[:4])
            bbox_height = bbox[3] - bbox[1]
            if _in_page_margin(bbox, page_height) and bbox_height < max(30.0, page_height * 0.05):
                continue
            if _max_image_overlap(bbox, image_rects) >= IMAGE_OVERLAP_RATIO:
                continue
            if any(_intersection_area(bbox, rect) / max(_bbox_area(bbox), 1.0) >= 0.35 for rect in occupied):
                continue
            covered = sum(_intersection_area(bbox, rect) for rect in occupied) / max(_bbox_area(bbox), 1.0)
            if covered >= 0.5:
                continue
            if any(
                normalized in existing
                or (len(existing) >= 30 and existing in normalized)
                for existing in existing_texts
            ):
                continue
            recovered.append(
                {
                    "type": caption_type.value if caption_type is not None else "text",
                    "text": text,
                    "page_idx": page_idx,
                    "bbox": list(bbox),
                    "logical_para_id": next_logical_id,
                    "context": text,
                    "recovered_from_pdf": True,
                }
            )
            next_logical_id += 1

        for new_item in sorted(recovered, key=lambda value: float(value["bbox"][1])):
            insert_at = len(page_items)
            for index, existing in enumerate(page_items):
                existing_bbox = existing.get("bbox")
                if (
                    isinstance(existing_bbox, (list, tuple))
                    and len(existing_bbox) == 4
                    and float(existing_bbox[1]) > float(new_item["bbox"][1])
                ):
                    insert_at = index
                    break
            page_items.insert(insert_at, new_item)

    return [item for page_idx in sorted(by_page) for item in by_page[page_idx]]


def _shared_pdf_text_matches(
    items: Sequence[dict[str, Any]],
    document: fitz.Document,
    page_data: Callable[[int], PageData],
) -> set[int]:
    """Return items whose PDF text block also contains another parser item.

    Expanding either duplicate to the shared block would make both items render
    the same text in the same rectangle, so their authoritative parser geometry
    must be preserved instead.
    """

    page_texts: defaultdict[int, dict[int, str]] = defaultdict(dict)
    page_boxes: defaultdict[int, list[tuple[int, tuple[float, float, float, float]]]] = defaultdict(list)
    for index, item in enumerate(items):
        page_idx = item.get("page_idx")
        raw_type = str(item.get("type") or "").strip().lower()
        if (
            not isinstance(page_idx, int)
            or isinstance(page_idx, bool)
            or raw_type in _TEXT_BBOX_SKIP_TYPES
        ):
            continue
        compact = _compact_text(_normalized_pdf_text(str(item.get("text") or "")))
        if len(compact) >= 30:
            page_texts[page_idx][index] = compact
        if (
            isinstance(item.get("bbox"), (list, tuple))
            and len(item["bbox"]) == 4
            and _bbox_area(item["bbox"]) > 0
        ):
            page_boxes[page_idx].append(
                (index, tuple(float(value) for value in item["bbox"]))
            )

    counts: Counter[tuple[float, float, float, float]] = Counter()
    keys: dict[int, tuple[float, float, float, float]] = {}
    ambiguous: set[int] = set()
    for index, item in enumerate(items):
        raw_type = str(item.get("type") or "").strip().lower()
        if raw_type in _TEXT_BBOX_SKIP_TYPES:
            continue
        text = str(item.get("text") or "")
        if not text.strip():
            continue
        page_idx = item.get("page_idx")
        bbox = item.get("bbox")
        if (
            not isinstance(page_idx, int)
            or isinstance(page_idx, bool)
            or not 0 <= page_idx < document.page_count
            or not isinstance(bbox, (list, tuple))
            or len(bbox) != 4
        ):
            continue
        blocks, _ = page_data(page_idx)
        matched = _match_pdf_text_block(text, blocks, source_bbox=bbox)
        if matched is None:
            continue
        source_bbox = tuple(float(value) for value in bbox)
        matched_bbox = tuple(float(value) for value in matched[0])
        key = tuple(round(float(value), 1) for value in matched[0])
        keys[index] = key
        counts[key] += 1
        matched_compact = _compact_text(_normalized_pdf_text(matched[1]))
        if any(
            other_index != index and compact in matched_compact
            for other_index, compact in page_texts[page_idx].items()
        ):
            ambiguous.add(index)
        if any(
            other_index != index
            and _bbox_area(other_bbox) > 0
            and not _contains_bbox(source_bbox, other_bbox, tolerance=2.0)
            and _intersection_area(matched_bbox, other_bbox) / _bbox_area(other_bbox) >= 0.25
            for other_index, other_bbox in page_boxes[page_idx]
        ):
            ambiguous.add(index)
    ambiguous.update(index for index, key in keys.items() if counts[key] > 1)
    return ambiguous


def _deduplicate_prepared_items(items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    deduplicated: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for item in items:
        text = _normalized_pdf_text(str(item.get("text") or ""))
        bbox = item.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            key = (
                item.get("page_idx"),
                text,
                tuple(round(float(value), 1) for value in bbox),
            )
        else:
            key = (item.get("page_idx"), text, None)
        if text and key in seen:
            continue
        seen.add(key)
        deduplicated.append(item)
    return deduplicated


def _is_text_paragraph(item: dict[str, Any]) -> bool:
    if str(item.get("type") or "").strip().lower() != "text":
        return False
    if str(item.get("sub_type") or "").strip().lower() in {
        "title",
        "doc_title",
        "paragraph_title",
        "abstract",
    }:
        return False
    return "text_level" not in item


def _looks_like_continuation(previous_text: str, current_text: str) -> bool:
    previous = previous_text.rstrip()
    if len(previous) < 40:
        return False
    if re.search(r"[.!?。！？:：;；]\s*$", previous):
        return False
    current = re.sub(r"</?su[bp]>", " ", current_text or "", flags=re.IGNORECASE).lstrip()
    current = re.sub(r"^[\s$\\{}^_]+", "", current)
    if not current:
        return False
    return current[0].islower()


def _merge_continuation_logical_ids(
    items: list[dict[str, Any]],
    document: fitz.Document,
) -> None:
    """Join a paragraph that continues at the top of the next column or block."""

    by_page: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        page_idx = item.get("page_idx")
        bbox = item.get("bbox")
        if (
            isinstance(page_idx, int)
            and not isinstance(page_idx, bool)
            and 0 <= page_idx < document.page_count
            and isinstance(bbox, (list, tuple))
            and len(bbox) == 4
        ):
            by_page[page_idx].append(item)
    for page_idx, page_items in by_page.items():
        page_width = float(document[page_idx].rect.width)
        page_mid = page_width / 2
        left: list[dict[str, Any]] = []
        right: list[dict[str, Any]] = []
        for item in sorted(
            page_items,
            key=lambda value: (float(value["bbox"][1]), float(value["bbox"][0])),
        ):
            bbox = item["bbox"]
            center = (float(bbox[0]) + float(bbox[2])) / 2
            (left if center < page_mid else right).append(item)
        left_paragraphs = [item for item in left if _is_text_paragraph(item)]
        right_paragraphs = [item for item in right if _is_text_paragraph(item)]
        for paragraphs in (left_paragraphs, right_paragraphs):
            for previous, current in zip(paragraphs, paragraphs[1:]):
                if not _looks_like_continuation(
                    str(previous.get("text") or ""), str(current.get("text") or "")
                ):
                    continue
                gap = float(current["bbox"][1]) - float(previous["bbox"][3])
                if gap > 24.0:
                    continue
                previous_id = previous.get("logical_para_id")
                if isinstance(previous_id, int):
                    current["logical_para_id"] = previous_id
        if left_paragraphs and right_paragraphs:
            previous = left_paragraphs[-1]
            current = right_paragraphs[0]
            if _looks_like_continuation(
                str(previous.get("text") or ""), str(current.get("text") or "")
            ):
                previous_id = previous.get("logical_para_id")
                if isinstance(previous_id, int):
                    current["logical_para_id"] = previous_id


def _strip_line_suffix(item: dict[str, Any], suffix: str) -> None:
    for line in reversed(item.get("lines") or []):
        if not isinstance(line, dict):
            continue
        spans = line.get("spans")
        if not isinstance(spans, list):
            continue
        for span in reversed(spans):
            if not isinstance(span, dict):
                continue
            content = span.get("content")
            if isinstance(content, str) and content.endswith(suffix):
                span["content"] = content[: -len(suffix)]
                return


def _prepend_line_prefix(item: dict[str, Any], prefix: str) -> None:
    for line in item.get("lines") or []:
        if not isinstance(line, dict):
            continue
        spans = line.get("spans")
        if not isinstance(spans, list):
            continue
        for span in spans:
            if not isinstance(span, dict):
                continue
            content = span.get("content")
            if isinstance(content, str) and content.strip():
                span["content"] = prefix + content.lstrip()
                return


def _repair_cross_page_hyphenation(
    items: list[dict[str, Any]],
    document: fitz.Document,
) -> None:
    """Join a word that MinerU split at a page boundary with a trailing hyphen."""

    pages: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        page_idx = item.get("page_idx")
        if (
            isinstance(page_idx, int)
            and not isinstance(page_idx, bool)
            and 0 <= page_idx < document.page_count
            and _is_text_paragraph(item)
        ):
            pages[page_idx].append(item)

    for page_idx in sorted(pages):
        current_items = sorted(
            pages[page_idx],
            key=lambda value: (float(value["bbox"][1]), float(value["bbox"][0])),
        )
        current = current_items[-1]
        current_text = str(current.get("text") or "").rstrip()
        match = re.search(r"([A-Za-z]{3,})-$", current_text)
        if match is None:
            continue
        next_item: dict[str, Any] | None = None
        for offset in range(1, 4):
            next_page = page_idx + offset
            if next_page >= document.page_count or not pages.get(next_page):
                continue
            next_item = sorted(
                pages[next_page],
                key=lambda value: (float(value["bbox"][1]), float(value["bbox"][0])),
            )[0]
            break
        if next_item is None:
            continue
        next_text = str(next_item.get("text") or "").lstrip()
        if not next_text or not next_text[0].islower():
            continue

        prefix = match.group(1)
        current["text"] = current_text[: match.start()].rstrip()
        _strip_line_suffix(current, f"{prefix}-")
        next_item["text"] = prefix + next_text
        _prepend_line_prefix(next_item, prefix)
        if isinstance(current.get("context"), str) and current["context"].rstrip().endswith(
            f"{prefix}-"
        ):
            current["context"] = current["context"].rstrip()[: -len(prefix) - 1].rstrip()
        if isinstance(next_item.get("context"), str):
            next_item["context"] = prefix + next_item["context"].lstrip()


def _looks_like_author_name_line(text: str) -> bool:
    plain = re.sub(r"</?su[bp]>", " ", text or "", flags=re.IGNORECASE)
    if not plain.strip() or _AFFILIATION_LINE_RE.search(plain):
        return False
    if any(marker in plain for marker in (";", ":", "?", "!", "@", "$")):
        return False
    if "http" in plain.casefold():
        return False
    words = [
        word
        for word in _AUTHOR_NAME_TOKEN_RE.findall(plain)
        if word.casefold() not in {"and"}
    ]
    if len(words) < 3 or any(word[0].islower() for word in words):
        return False
    return True


def _front_matter_author_indices(
    items: Sequence[dict[str, Any]],
    document: fitz.Document,
) -> set[int]:
    if document.page_count < 1:
        return set()
    page_items = [
        (index, item)
        for index, item in enumerate(items)
        if item.get("page_idx") == 0
    ]
    title_bottoms: list[float] = []
    boundary: float | None = None
    for _, item in page_items:
        bbox = item.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        element_type = classify_element(item).element_type
        if element_type is ElementType.TITLE:
            title_bottoms.append(float(bbox[3]))
        elif element_type is ElementType.ABSTRACT:
            top = float(bbox[1])
            boundary = top if boundary is None else min(boundary, top)
    if not title_bottoms:
        return set()
    title_bottom = max(title_bottoms)
    if boundary is None:
        header_tops = [
            float(item["bbox"][1])
            for _, item in page_items
            if isinstance(item.get("bbox"), (list, tuple))
            and len(item["bbox"]) == 4
            and classify_element(item).element_type is ElementType.SECTION_HEADER
        ]
        if header_tops:
            boundary = min(header_tops)
    if boundary is None or boundary <= title_bottom:
        above_boundary = [value for value in title_bottoms if value < boundary] if boundary else []
        if not above_boundary:
            return set()
        title_bottom = max(above_boundary)
    if boundary is None or boundary <= title_bottom:
        return set()
    authors: set[int] = set()
    for index, item in page_items:
        if str(item.get("type") or "").strip().lower() != "text":
            continue
        bbox = item.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        if float(bbox[1]) < title_bottom - 2.0 or float(bbox[3]) > boundary + 2.0:
            continue
        if _looks_like_author_name_line(str(item.get("text") or "")):
            authors.add(index)
    return authors


def _prepare_items(
    raw_items: Iterable[dict[str, Any]],
    document: fitz.Document,
) -> list[dict[str, Any]]:
    prepared = [deepcopy(item) for item in raw_items]
    _mark_repeated_margin_items(prepared, document)
    prepared = _split_multicolumn_text_items(prepared, document)
    cache: dict[int, PageData] = {}

    def page_data(page_idx: int) -> PageData:
        if page_idx not in cache:
            page = document[page_idx]
            cache[page_idx] = (page.get_text("blocks"), _image_rects(page))
        return cache[page_idx]

    shared_matches = _shared_pdf_text_matches(prepared, document, page_data)
    for item in prepared:
        _repair_replacement_glyphs(item, document)
    for index, item in enumerate(prepared):
        _correct_text_bbox(
            item,
            document,
            page_data,
            allow_match=index not in shared_matches,
        )
    for item in prepared:
        _ensure_bbox_contains_line_geometry(item)
    prepared = _split_mixed_chart_captions(prepared, document)
    prepared = _split_textless_tables(prepared, document, page_data)
    prepared = _recover_missing_text_items(prepared, document, page_data)
    prepared = _deduplicate_prepared_items(prepared)
    _repair_cross_page_hyphenation(prepared, document)
    _merge_continuation_logical_ids(prepared, document)
    return prepared


def _clean_pdf_context(value: str) -> str:
    value = re.sub(r"\\[A-Za-z]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\s+([,.;:!?])", r"\1", value)
    return value.strip()


def _replacement_from_pdf(source_text: str, start: int, end: int, pdf_text: str) -> str | None:
    prefix = source_text[max(0, start - 120) : start].rsplit("\ufffd", 1)[-1]
    prefix = prefix.rsplit("$", 1)[-1]
    suffix_segments = [
        segment
        for segment in re.split("\ufffd+", source_text[end : end + 160])
        if segment.strip()
    ]
    prefix = unicodedata.normalize("NFKC", _clean_pdf_context(prefix))
    if not prefix or not suffix_segments:
        return None
    for prefix_size in (60, 40, 24, 12):
        probe = prefix[-prefix_size:]
        if not probe:
            continue
        position = pdf_text.rfind(probe)
        if position < 0:
            continue
        content_start = position + len(probe)
        for segment in suffix_segments:
            segment = segment.split("$", 1)[0]
            suffix = unicodedata.normalize("NFKC", _clean_pdf_context(segment))
            for suffix_size in (60, 40, 24, 12):
                tail = suffix[:suffix_size]
                if not tail:
                    continue
                content_end = pdf_text.find(tail, content_start)
                if content_end < 0 or content_end == content_start:
                    continue
                candidate = pdf_text[content_start:content_end].strip()
                if 0 < len(candidate) <= 80:
                    return unicodedata.normalize("NFKC", candidate)
    return None


def _repair_replacement_glyphs(item: dict[str, Any], document: fitz.Document) -> None:
    text = item.get("text")
    if not isinstance(text, str) or "\ufffd" not in text:
        return
    page_idx = int(item.get("page_idx") or 0)
    bbox = item.get("bbox")
    if page_idx >= document.page_count or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return
    pdf_text = unicodedata.normalize(
        "NFKC",
        re.sub(r"\s+", " ", document[page_idx].get_text("text", clip=fitz.Rect(bbox))),
    )
    if not pdf_text:
        return
    repaired = text
    cursor = 0
    while True:
        match = re.search("\ufffd+", repaired[cursor:])
        if match is None:
            break
        start = cursor + match.start()
        end = cursor + match.end()
        replacement = _replacement_from_pdf(repaired, start, end, pdf_text)
        if replacement:
            repaired = repaired[:start] + replacement + repaired[end:]
            cursor = start + len(replacement)
        else:
            cursor = end
    item["text"] = unicodedata.normalize("NFKC", repaired)


def _validate_pdf_bounds(block: InputBlock, *, index: int, document: fitz.Document) -> None:
    if block.page_idx >= document.page_count:
        raise InputValidationError(
            f"block[{index}].page_idx={block.page_idx} is outside PDF page range "
            f"0..{document.page_count - 1}"
        )
    page = document[block.page_idx]
    width, height = float(page.rect.width), float(page.rect.height)
    x0, y0, x1, y1 = block.bbox
    tolerance = 1.0
    if x0 < -tolerance or y0 < -tolerance or x1 > width + tolerance or y1 > height + tolerance:
        raise InputValidationError(
            f"block[{index}].bbox={list(block.bbox)} is outside page {block.page_idx} "
            f"bounds [0, 0, {width:.2f}, {height:.2f}]"
        )


def _validate_blocks(raw_items: Iterable[dict[str, Any]], document: fitz.Document) -> tuple[LoadedBlock, ...]:
    loaded: list[LoadedBlock] = []
    page_ordinals: defaultdict[int, int] = defaultdict(int)
    used_ids: set[str] = set()
    prepared_items = _prepare_items(raw_items, document)
    author_indices = _front_matter_author_indices(prepared_items, document)

    for index, prepared in enumerate(prepared_items):
        if str(prepared.get("type") or "").strip().lower() == "chart":
            corrected_bbox = _caption_bbox_from_pdf(prepared, document)
            if corrected_bbox is not None:
                prepared["bbox"] = list(corrected_bbox)
        try:
            block = InputBlock.model_validate(prepared)
        except ValidationError as exc:
            details = "; ".join(
                f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                for error in exc.errors(include_url=False)
            )
            raise InputValidationError(f"block[{index}] failed schema validation: {details}") from exc

        _validate_pdf_bounds(block, index=index, document=document)
        page_ordinals[block.page_idx] += 1
        generated_id = f"p{block.page_idx + 1:04d}-b{page_ordinals[block.page_idx]:04d}"
        block_id = block.block_id or generated_id
        if block_id in used_ids:
            raise InputValidationError(f"duplicate block_id {block_id!r} at block[{index}]")
        used_ids.add(block_id)
        taxonomy = classify_element(prepared)
        if index in author_indices:
            taxonomy = TaxonomyDecision(
                ElementType.AUTHOR,
                TranslationPolicy.PRESERVE,
                "front-matter author names preserve the original identity layout",
            )
        loaded.append(
            LoadedBlock(
                block_id=block_id,
                source_index=index,
                data=block,
                taxonomy=taxonomy,
                raw=deepcopy(prepared),
            )
        )
    return tuple(loaded)


def load_document(
    discovered: DiscoveredInput,
    *,
    explicit_pdf: str | Path | None = None,
) -> LoadedDocument:
    source_json = discovered.source_json
    source_pdf = discover_pdf(source_json, explicit_pdf)
    raw_bytes, raw_items = _load_json_array(source_json)
    try:
        document = fitz.open(source_pdf)
    except Exception as exc:
        raise InputValidationError(f"cannot open source PDF {source_pdf}: {exc}") from exc
    try:
        if document.page_count < 1:
            raise InputValidationError(f"source PDF has no pages: {source_pdf}")
        blocks = _validate_blocks(raw_items, document)
        page_count = document.page_count
    finally:
        document.close()

    return LoadedDocument(
        schema_version=INPUT_SCHEMA_VERSION,
        document_id=document_id_from_json(source_json),
        source_json=source_json,
        source_pdf=source_pdf,
        selection_reason=discovered.selection_reason,
        source_hash=hashlib.sha256(raw_bytes).hexdigest(),
        page_count=page_count,
        blocks=blocks,
    )


def load_documents(
    paths: Sequence[str | Path],
    *,
    explicit_pdf: str | Path | None = None,
) -> list[LoadedDocument]:
    discovered = discover_sources(paths)
    if explicit_pdf is not None and len(discovered) != 1:
        raise InputDiscoveryError("--pdf can only be used when exactly one input is selected")
    documents = [
        load_document(item, explicit_pdf=explicit_pdf if len(discovered) == 1 else None)
        for item in discovered
    ]
    document_ids = Counter(document.document_id for document in documents)
    duplicates = sorted(document_id for document_id, count in document_ids.items() if count > 1)
    if duplicates:
        raise InputDiscoveryError(
            "duplicate document_id values in one run: "
            f"{', '.join(duplicates)}; select only one parse result per document"
        )
    return documents
