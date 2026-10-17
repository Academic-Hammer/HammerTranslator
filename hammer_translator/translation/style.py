"""Document-level golden style statistics derived from the original PDF layout."""

from __future__ import annotations

import hashlib
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import fitz

from .loader import LoadedBlock
from .taxonomy import ElementType
from .units import TranslationPlan, TranslationUnit


STYLE_SCHEMA_VERSION = "layout-style-v1"
MEASUREMENT_ENGINE_VERSION = "layout-metrics-v2"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FONTS_DIR = PROJECT_ROOT / "external" / "Colayout-pure" / "fonts"


@dataclass(frozen=True)
class WeightedValue:
    value: float
    weight: float


@dataclass(frozen=True)
class SpanSample:
    text: str
    size: float
    font: str
    flags: int
    color: int
    weight: float
    source: str


ROLE_DEFAULTS: dict[str, dict[str, Any]] = {
    "document_title": {"scale": 1.70, "line": 1.20, "font_category": "title", "shrink": 0.08},
    "section_header_l1": {"scale": 1.25, "line": 1.20, "font_category": "heading", "shrink": 0.08},
    "section_header_l2": {"scale": 1.15, "line": 1.20, "font_category": "heading", "shrink": 0.08},
    "section_header_l3": {"scale": 1.08, "line": 1.20, "font_category": "heading", "shrink": 0.08},
    "body": {"scale": 1.00, "line": 1.25, "font_category": "body", "shrink": 0.10},
    "abstract": {"scale": 1.00, "line": 1.25, "font_category": "body", "shrink": 0.10},
    "caption": {"scale": 0.90, "line": 1.20, "font_category": "caption", "shrink": 0.12},
    "header": {"scale": 0.85, "line": 1.15, "font_category": "body", "shrink": 0.12},
    "footer": {"scale": 0.85, "line": 1.15, "font_category": "caption", "shrink": 0.12},
    "footnote": {"scale": 0.85, "line": 1.15, "font_category": "caption", "shrink": 0.12},
    "author": {"scale": 0.95, "line": 1.15, "font_category": "body", "shrink": 0.10},
    "affiliation": {"scale": 0.90, "line": 1.15, "font_category": "body", "shrink": 0.10},
    "reference": {"scale": 0.90, "line": 1.20, "font_category": "body", "shrink": 0.10},
    "table": {"scale": 0.90, "line": 1.15, "font_category": "body", "shrink": 0.12},
    "equation": {"scale": 1.00, "line": 1.20, "font_category": "body", "shrink": 0.10},
    "other": {"scale": 1.00, "line": 1.20, "font_category": "body", "shrink": 0.10},
}


FONT_FILES = {
    "title": "SimHei.ttf",
    "heading": "SimHei.ttf",
    "body": "SimSun.ttf",
    "caption": "SimKai.ttf",
}


def style_role_for(block: LoadedBlock, element_type: ElementType) -> str:
    raw_level = block.raw.get("text_level")
    try:
        text_level = int(raw_level) if raw_level is not None else None
    except (TypeError, ValueError):
        text_level = None

    if element_type is ElementType.TITLE:
        if text_level is None or text_level <= 1:
            return "document_title"
        return f"section_header_l{min(3, max(1, text_level - 1))}"
    if element_type is ElementType.SECTION_HEADER:
        heading_level = 1 if text_level is None else min(3, max(1, text_level - 1))
        return f"section_header_l{heading_level}"
    if element_type is ElementType.ABSTRACT:
        return "abstract"
    if element_type is ElementType.TEXT:
        return "body"
    if element_type in {
        ElementType.CAPTION,
        ElementType.IMAGE_CAPTION,
        ElementType.FIGURE_CAPTION,
        ElementType.TABLE_CAPTION,
    }:
        return "caption"
    if element_type is ElementType.HEADER:
        return "header"
    if element_type is ElementType.FOOTER:
        return "footer"
    if element_type in {ElementType.FOOTNOTE, ElementType.PAGE_FOOTNOTE}:
        return "footnote"
    if element_type is ElementType.AUTHOR:
        return "author"
    if element_type is ElementType.AFFILIATION:
        return "affiliation"
    if element_type is ElementType.REFERENCE:
        return "reference"
    if element_type is ElementType.TABLE:
        return "table"
    if element_type is ElementType.EQUATION:
        return "equation"
    return "other"


def style_role_for_unit(plan: TranslationPlan, unit: TranslationUnit) -> str:
    blocks = {block.block_id: block for block in plan.document.blocks}
    return style_role_for(blocks[unit.block_ids[0]], unit.element_type)


def _weighted_median(values: Iterable[WeightedValue]) -> float | None:
    ordered = sorted((item for item in values if item.weight > 0), key=lambda item: item.value)
    if not ordered:
        return None
    threshold = sum(item.weight for item in ordered) / 2
    cumulative = 0.0
    for item in ordered:
        cumulative += item.weight
        if cumulative >= threshold:
            return item.value
    return ordered[-1].value


def _robust_weighted_median(values: list[WeightedValue]) -> tuple[float | None, int]:
    center = _weighted_median(values)
    if center is None:
        return None, 0
    deviations = [WeightedValue(abs(item.value - center), item.weight) for item in values]
    mad = _weighted_median(deviations) or 0.0
    tolerance = max(0.75, mad * 3.0)
    filtered = [item for item in values if abs(item.value - center) <= tolerance]
    return _weighted_median(filtered), len(filtered)


def _safe_span(raw: dict[str, Any], source: str) -> SpanSample | None:
    text = str(raw.get("text") or raw.get("content") or "")
    try:
        size = float(raw.get("size") or 0)
        flags = int(raw.get("flags") or 0)
        color = int(raw.get("color") or 0)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(size) or not 3.0 <= size <= 72.0 or not text.strip():
        return None
    weight = max(1, len("".join(text.split())))
    return SpanSample(
        text=text,
        size=size,
        font=str(raw.get("font") or "unknown"),
        flags=flags,
        color=color,
        weight=float(weight),
        source=source,
    )


def _pdf_spans(block: LoadedBlock, document: fitz.Document) -> list[SpanSample]:
    page = document[block.data.page_idx]
    result: list[SpanSample] = []
    for pdf_block in page.get_text("dict", clip=fitz.Rect(block.data.bbox)).get("blocks", []):
        for line in pdf_block.get("lines", []):
            for span in line.get("spans", []):
                sample = _safe_span(span, "pdf_clip")
                if sample is not None:
                    result.append(sample)
    return result


def _block_spans(block: LoadedBlock, document: fitz.Document) -> list[SpanSample]:
    rich_spans = block.raw.get("rich_spans")
    if isinstance(rich_spans, list):
        result = [
            sample
            for span in rich_spans
            if isinstance(span, dict)
            for sample in [_safe_span(span, "rich_spans")]
            if sample is not None
        ]
        if result:
            return result
    return _pdf_spans(block, document)


def _line_spacing_samples(block: LoadedBlock, representative_size: float) -> list[WeightedValue]:
    raw_lines = block.raw.get("lines")
    if not isinstance(raw_lines, list) or representative_size <= 0:
        return []
    centers: list[float] = []
    for line in raw_lines:
        bbox = line.get("bbox") if isinstance(line, dict) else None
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            y0, y1 = float(bbox[1]), float(bbox[3])
        except (TypeError, ValueError):
            continue
        if y1 > y0:
            centers.append((y0 + y1) / 2)
    centers = sorted(set(centers))
    samples = []
    for previous, current in zip(centers, centers[1:]):
        ratio = (current - previous) / representative_size
        if 0.85 <= ratio <= 2.50:
            samples.append(WeightedValue(ratio, 1.0))
    return samples


def _alignment_vote(block: LoadedBlock, role: str, page_width: float) -> str:
    explicit = str(block.raw.get("alignment") or block.raw.get("align") or "").lower()
    if explicit in {"left", "center", "right", "justify"}:
        return explicit
    if role == "document_title":
        x0, _, x1, _ = block.data.bbox
        if abs(((x0 + x1) / 2) - page_width / 2) / max(page_width, 1.0) <= 0.08:
            return "center"
        return "left"
    if role != "author":
        return "left"
    raw_lines = block.raw.get("lines")
    if not isinstance(raw_lines, list) or not raw_lines:
        return "left"
    x0, _, x1, _ = block.data.bbox
    width = x1 - x0
    offsets: list[float] = []
    for line in raw_lines:
        bbox = line.get("bbox") if isinstance(line, dict) else None
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            line_center = (float(bbox[0]) + float(bbox[2])) / 2
        except (TypeError, ValueError):
            continue
        offsets.append(abs(line_center - (x0 + x1) / 2) / max(width, 1.0))
    if offsets and statistics.median(offsets) <= 0.04:
        return "center"
    return "left"


def _rgb_hex(color: int) -> str:
    return f"#{color & 0xFFFFFF:06x}"


def _font_record(category: str) -> dict[str, Any]:
    filename = FONT_FILES[category]
    path = (FONTS_DIR / filename).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"required bundled Chinese font is missing: {path}")
    data = path.read_bytes()
    font = fitz.Font(fontfile=str(path))
    if font.text_length("中文", fontsize=10) <= 0:
        raise ValueError(f"font cannot measure Chinese glyphs: {path}")
    return {
        "font_id": f"zh_{category}_regular",
        "family": font.name,
        "path": str(path),
        "project_relative_path": str(path.relative_to(PROJECT_ROOT)),
        "sha256": hashlib.sha256(data).hexdigest(),
        "ascender": round(float(font.ascender), 6),
        "descender": round(float(font.descender), 6),
    }


def _dominant(counter: Counter, fallback: Any) -> Any:
    return counter.most_common(1)[0][0] if counter else fallback


def build_golden_style_profile(plan: TranslationPlan) -> dict[str, Any]:
    """Build robust per-role styles from original blocks without consulting translations."""

    role_spans: defaultdict[str, list[SpanSample]] = defaultdict(list)
    role_lines: defaultdict[str, list[WeightedValue]] = defaultdict(list)
    role_alignments: defaultdict[str, Counter] = defaultdict(Counter)
    unit_roles: dict[str, str] = {}
    block_to_unit = {
        block_id: unit
        for unit in plan.units
        for block_id in unit.block_ids
    }

    document = fitz.open(plan.document.source_pdf)
    try:
        for block in plan.document.blocks:
            unit = block_to_unit[block.block_id]
            role = style_role_for(block, unit.element_type)
            unit_roles[unit.translation_unit_id] = role
            spans = _block_spans(block, document)
            role_spans[role].extend(spans)
            representative = _weighted_median(
                [WeightedValue(span.size, span.weight) for span in spans]
            )
            if representative:
                role_lines[role].extend(_line_spacing_samples(block, representative))
            page_width = float(document[block.data.page_idx].rect.width)
            role_alignments[role][_alignment_vote(block, role, page_width)] += max(
                1, len(block.data.text.strip())
            )
    finally:
        document.close()

    all_spans = [span for spans in role_spans.values() for span in spans]
    body_candidates = role_spans.get("body") or role_spans.get("abstract") or all_spans
    body_size, _ = _robust_weighted_median(
        [WeightedValue(span.size, span.weight) for span in body_candidates]
    )
    body_size = float(body_size or 10.0)

    requested_roles = list(dict.fromkeys(unit_roles.values()))
    fonts: dict[str, dict[str, Any]] = {}
    roles: dict[str, dict[str, Any]] = {}
    for role in requested_roles:
        defaults = ROLE_DEFAULTS.get(role, ROLE_DEFAULTS["other"])
        spans = role_spans.get(role, [])
        measured_size, retained_count = _robust_weighted_median(
            [WeightedValue(span.size, span.weight) for span in spans]
        )
        provenance = "measured" if measured_size is not None else "derived_from_body"
        if measured_size is None:
            measured_size = body_size * float(defaults["scale"])

        measured_line = _weighted_median(role_lines.get(role, []))
        line_provenance = "measured" if measured_line is not None else "role_default"
        if measured_line is None:
            measured_line = float(defaults["line"])
        measured_line = min(2.0, max(1.0, float(measured_line)))

        font_counter: Counter[str] = Counter()
        color_counter: Counter[int] = Counter()
        total_weight = sum(span.weight for span in spans)
        bold_weight = 0.0
        italic_weight = 0.0
        sources: Counter[str] = Counter()
        for span in spans:
            font_counter[span.font] += span.weight
            color_counter[span.color] += span.weight
            sources[span.source] += 1
            if span.flags & 16:
                bold_weight += span.weight
            if span.flags & 2:
                italic_weight += span.weight

        category = str(defaults["font_category"])
        font_record = fonts.setdefault(f"zh_{category}_regular", _font_record(category))
        golden_size = round(float(measured_size), 3)
        golden_line = round(measured_line, 4)
        char_spacing = 0.0
        shrink = float(defaults["shrink"])
        roles[role] = {
            "role": role,
            "source_statistics": {
                "font_size_pt": golden_size,
                "line_height_multiplier": golden_line,
                "dominant_font": _dominant(font_counter, "unknown"),
                "dominant_color": _rgb_hex(int(_dominant(color_counter, 0))),
                "bold_ratio": round(bold_weight / total_weight, 4) if total_weight else 0.0,
                "italic_ratio": round(italic_weight / total_weight, 4) if total_weight else 0.0,
                "alignment": _dominant(role_alignments.get(role, Counter()), "left"),
                "span_sample_count": len(spans),
                "retained_size_sample_count": retained_count,
                "line_gap_sample_count": len(role_lines.get(role, [])),
                "weighted_character_count": int(total_weight),
                "sample_sources": dict(sorted(sources.items())),
            },
            "target_style": {
                "font_id": font_record["font_id"],
                "font_size_pt": golden_size,
                "line_height_multiplier": golden_line,
                "char_spacing_em": char_spacing,
                "alignment": _dominant(role_alignments.get(role, Counter()), "left"),
            },
            "guardrails": {
                "bbox_locked": True,
                "font_size_pt": {
                    "min": round(golden_size * (1.0 - shrink), 3),
                    "max": golden_size,
                    "max_shrink_ratio": shrink,
                },
                "line_height_multiplier": {
                    "min": round(golden_line * 0.92, 4),
                    "max": round(golden_line * 1.05, 4),
                },
                "char_spacing_em": {"min": -0.04, "max": 0.02},
            },
            "provenance": {
                "font_size": provenance,
                "line_height": line_provenance,
                "source": "original_pdf_and_parsed_rich_spans",
            },
        }

    return {
        "schema_version": STYLE_SCHEMA_VERSION,
        "measurement_engine": MEASUREMENT_ENGINE_VERSION,
        "document_id": plan.document.document_id,
        "source_hash": plan.document.source_hash,
        "statistics_basis": "original_pdf",
        "translated_text_used_for_statistics": False,
        "body_reference_font_size_pt": round(body_size, 3),
        "fonts": fonts,
        "roles": roles,
        "unit_roles": unit_roles,
    }
