"""Formula provenance and protected-token bindings for Translation/Reflow."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Literal

from .loader import LoadedBlock
from .protection import ProtectedToken


FORMULA_CONTRACT_VERSION = "formula-contract-v1"
# MinerU emits integer bboxes for blocks, lines, and spans independently.  A
# formula with an accent or superscript can therefore extend two points above
# the rounded block while remaining exactly inside its source line.  Keep the
# tolerance deliberately small: it absorbs rounding/font extents without
# accepting geometry from another line or block.
MINERU_NESTED_BBOX_TOLERANCE = 3.0
FORMULA_TOKEN_KINDS = {
    "latex_inline",
    "latex_parenthesis",
    "latex_display",
    "latex_bracket",
    "latex_environment",
}

FormulaKind = Literal["inline", "display"]
FormulaBindingStatus = Literal[
    "bound_exact",
    "bound_normalized",
    "latex_mismatch",
    "protected_token_only",
    "source_span_only",
    "preserved_display",
]
FormulaRenderPolicy = Literal["move_with_text", "preserve_in_place", "needs_review"]
FormulaAssetPreference = Literal["source_pdf_then_mathjax", "mathjax_svg", "source_pdf"]


class FormulaContractError(ValueError):
    """Raised when nested MinerU formula geometry is malformed or unsafe."""


@dataclass(frozen=True)
class FormulaSpan:
    block_id: str
    page_idx: int
    latex: str
    bbox: tuple[float, float, float, float]
    line_bbox: tuple[float, float, float, float]
    block_bbox: tuple[float, float, float, float]
    line_index: int
    span_index: int
    span_path: str
    confidence: float | None
    kind: FormulaKind
    geometry_flags: tuple[str, ...]


@dataclass(frozen=True)
class FormulaBinding:
    formula_id: str
    kind: FormulaKind
    translation_unit_id: str
    block_id: str | None
    page_idx: int | None
    latex: str
    latex_hash: str
    source_bbox: tuple[float, float, float, float] | None
    source_line_bbox: tuple[float, float, float, float] | None
    source_block_bbox: tuple[float, float, float, float] | None
    source_line_index: int | None
    source_span_index: int | None
    source_span_path: str | None
    recognition_confidence: float | None
    geometry_flags: tuple[str, ...]
    protected_placeholder: str | None
    source_text_start: int | None
    source_text_end: int | None
    binding_status: FormulaBindingStatus
    render_policy: FormulaRenderPolicy
    asset_preference: FormulaAssetPreference
    asset_status: Literal["uninspected"] = "uninspected"

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("source_bbox", "source_line_bbox", "source_block_bbox"):
            if value[key] is not None:
                value[key] = list(value[key])
        return value


def _bbox(value: object, *, location: str) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise FormulaContractError(f"{location} must contain four coordinates")
    try:
        result = tuple(float(coordinate) for coordinate in value)
    except (TypeError, ValueError) as exc:
        raise FormulaContractError(f"{location} coordinates must be numeric") from exc
    if not all(math.isfinite(coordinate) for coordinate in result):
        raise FormulaContractError(f"{location} coordinates must be finite")
    if result[2] <= result[0] or result[3] <= result[1]:
        raise FormulaContractError(f"{location} must satisfy x0 < x1 and y0 < y1")
    return result  # type: ignore[return-value]


def _contains(
    outer: tuple[float, float, float, float],
    inner: tuple[float, float, float, float],
    *,
    tolerance: float = MINERU_NESTED_BBOX_TOLERANCE,
) -> bool:
    return (
        inner[0] >= outer[0] - tolerance
        and inner[1] >= outer[1] - tolerance
        and inner[2] <= outer[2] + tolerance
        and inner[3] <= outer[3] + tolerance
    )


def _confidence(value: object, *, location: str) -> float | None:
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise FormulaContractError(f"{location} must be numeric") from exc
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise FormulaContractError(f"{location} must be within 0..1")
    return score


def extract_formula_spans(block: LoadedBlock) -> tuple[FormulaSpan, ...]:
    """Extract formula-local bboxes while retaining block and line provenance."""

    raw_lines = block.raw.get("lines")
    if not isinstance(raw_lines, list):
        return ()
    block_bbox = _bbox(block.raw.get("bbox"), location=f"{block.block_id}.bbox")
    records: list[FormulaSpan] = []
    for line_index, line in enumerate(raw_lines):
        if not isinstance(line, dict):
            continue
        spans = line.get("spans")
        if not isinstance(spans, list):
            continue
        formula_spans = [
            (span_index, span)
            for span_index, span in enumerate(spans)
            if isinstance(span, dict)
            and str(span.get("type") or "") in {"inline_equation", "interline_equation", "equation"}
        ]
        if not formula_spans:
            continue
        line_bbox = _bbox(
            line.get("bbox"), location=f"{block.block_id}.lines[{line_index}].bbox"
        )
        line_geometry_flags: list[str] = []
        # MinerU's independently rounded block and line boxes can disagree by a
        # few points.  The formula-local box remains authoritative; retain the
        # discrepancy as provenance instead of rejecting an otherwise usable
        # formula span.
        if not _contains(block_bbox, line_bbox, tolerance=0.0):
            line_geometry_flags.append("line_bbox_outside_block")
        for span_index, span in formula_spans:
            span_type = str(span.get("type") or "")
            kind: FormulaKind = "inline" if span_type == "inline_equation" else "display"
            location = f"{block.block_id}.lines[{line_index}].spans[{span_index}]"
            latex = str(span.get("content") or "").strip()
            if not latex:
                raise FormulaContractError(f"{location}.content cannot be empty")
            formula_bbox = _bbox(span.get("bbox"), location=f"{location}.bbox")
            if not _contains(line_bbox, formula_bbox):
                raise FormulaContractError(f"{location}.bbox is outside its line bbox")
            if not _contains(block_bbox, formula_bbox):
                raise FormulaContractError(f"{location}.bbox is outside its block bbox")
            geometry_flags = list(line_geometry_flags)
            if not _contains(line_bbox, formula_bbox, tolerance=0.0):
                geometry_flags.append("formula_bbox_outside_line_tolerance_used")
            if not _contains(block_bbox, formula_bbox, tolerance=0.0):
                geometry_flags.append("formula_bbox_outside_block_tolerance_used")
            records.append(
                FormulaSpan(
                    block_id=block.block_id,
                    page_idx=block.data.page_idx,
                    latex=latex,
                    bbox=formula_bbox,
                    line_bbox=line_bbox,
                    block_bbox=block_bbox,
                    line_index=line_index,
                    span_index=span_index,
                    span_path=f"blocks/{block.block_id}/lines/{line_index}/spans/{span_index}",
                    confidence=_confidence(span.get("score"), location=f"{location}.score"),
                    kind=kind,
                    geometry_flags=tuple(geometry_flags),
                )
            )
    return tuple(records)


def _token_latex(token: ProtectedToken) -> str:
    value = token.original.strip()
    if token.kind == "latex_inline" and value.startswith("$") and value.endswith("$"):
        return value[1:-1].strip()
    if token.kind == "latex_display" and value.startswith("$$") and value.endswith("$$"):
        return value[2:-2].strip()
    if token.kind == "latex_parenthesis" and value.startswith(r"\(") and value.endswith(r"\)"):
        return value[2:-2].strip()
    if token.kind == "latex_bracket" and value.startswith(r"\[") and value.endswith(r"\]"):
        return value[2:-2].strip()
    return value


def _normalized_latex(value: str) -> str:
    return re.sub(r"\s+", "", value)


def _formula_id(
    document_id: str,
    translation_unit_id: str,
    *,
    span: FormulaSpan | None,
    token: ProtectedToken | None,
    kind: FormulaKind,
) -> str:
    identity = span.span_path if span is not None else token.placeholder if token is not None else ""
    latex = span.latex if span is not None else _token_latex(token) if token is not None else ""
    payload = "\0".join((document_id, translation_unit_id, kind, identity, latex))
    return f"fm-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def _bind(
    document_id: str,
    translation_unit_id: str,
    span: FormulaSpan | None,
    token: ProtectedToken | None,
    *,
    status: FormulaBindingStatus,
    kind: FormulaKind,
) -> FormulaBinding:
    latex = _token_latex(token) if token is not None else span.latex if span is not None else ""
    if kind == "display" and status == "preserved_display":
        policy: FormulaRenderPolicy = "preserve_in_place"
        preference: FormulaAssetPreference = "source_pdf"
    elif status == "source_span_only":
        policy = "needs_review"
        preference = "source_pdf_then_mathjax"
    elif status == "latex_mismatch":
        policy = "move_with_text"
        preference = "mathjax_svg"
    else:
        policy = "move_with_text"
        preference = "source_pdf_then_mathjax" if span is not None else "mathjax_svg"
    return FormulaBinding(
        formula_id=_formula_id(
            document_id,
            translation_unit_id,
            span=span,
            token=token,
            kind=kind,
        ),
        kind=kind,
        translation_unit_id=translation_unit_id,
        block_id=span.block_id if span is not None else None,
        page_idx=span.page_idx if span is not None else None,
        latex=latex,
        latex_hash=hashlib.sha256(latex.encode("utf-8")).hexdigest(),
        source_bbox=span.bbox if span is not None else None,
        source_line_bbox=span.line_bbox if span is not None else None,
        source_block_bbox=span.block_bbox if span is not None else None,
        source_line_index=span.line_index if span is not None else None,
        source_span_index=span.span_index if span is not None else None,
        source_span_path=span.span_path if span is not None else None,
        recognition_confidence=span.confidence if span is not None else None,
        geometry_flags=span.geometry_flags if span is not None else (),
        protected_placeholder=token.placeholder if token is not None else None,
        source_text_start=token.start if token is not None else None,
        source_text_end=token.end if token is not None else None,
        binding_status=status,
        render_policy=policy,
        asset_preference=preference,
    )


def _lcs_pairs(tokens: list[ProtectedToken], spans: list[FormulaSpan]) -> list[tuple[int, int]]:
    token_keys = [_normalized_latex(_token_latex(token)) for token in tokens]
    span_keys = [_normalized_latex(span.latex) for span in spans]
    rows, columns = len(tokens), len(spans)
    table = [[0] * (columns + 1) for _ in range(rows + 1)]
    for row in range(rows - 1, -1, -1):
        for column in range(columns - 1, -1, -1):
            if token_keys[row] == span_keys[column]:
                table[row][column] = table[row + 1][column + 1] + 1
            else:
                table[row][column] = max(table[row + 1][column], table[row][column + 1])
    pairs: list[tuple[int, int]] = []
    row = column = 0
    while row < rows and column < columns:
        if token_keys[row] == span_keys[column]:
            pairs.append((row, column))
            row += 1
            column += 1
        elif table[row + 1][column] >= table[row][column + 1]:
            row += 1
        else:
            column += 1
    return pairs


def build_formula_bindings(
    *,
    document_id: str,
    translation_unit_id: str,
    blocks: Iterable[LoadedBlock],
    protected_tokens: Iterable[ProtectedToken],
) -> tuple[FormulaBinding, ...]:
    """Bind immutable formula tokens to MinerU formula spans without guessing."""

    spans = [span for block in blocks for span in extract_formula_spans(block)]
    display_spans = [span for span in spans if span.kind == "display"]
    inline_spans = [span for span in spans if span.kind == "inline"]
    tokens = [token for token in protected_tokens if token.kind in FORMULA_TOKEN_KINDS]
    inline_tokens = [
        token
        for token in tokens
        if token.kind in {"latex_inline", "latex_parenthesis"}
    ]
    display_tokens = [
        token
        for token in tokens
        if token.kind not in {"latex_inline", "latex_parenthesis"}
    ]

    bindings: list[FormulaBinding] = []
    paired_display_tokens: set[int] = set()
    paired_display_spans: set[int] = set()
    for token_index, span_index in _lcs_pairs(display_tokens, display_spans):
        bindings.append(
            _bind(
                document_id,
                translation_unit_id,
                display_spans[span_index],
                display_tokens[token_index],
                status="preserved_display",
                kind="display",
            )
        )
        paired_display_tokens.add(token_index)
        paired_display_spans.add(span_index)
    for index, span in enumerate(display_spans):
        if index not in paired_display_spans:
            bindings.append(
                _bind(
                    document_id,
                    translation_unit_id,
                    span,
                    None,
                    status="preserved_display",
                    kind="display",
                )
            )

    paired_tokens: set[int] = set()
    paired_spans: set[int] = set()
    for token_index, span_index in _lcs_pairs(inline_tokens, inline_spans):
        token = inline_tokens[token_index]
        span = inline_spans[span_index]
        exact = _token_latex(token) == span.latex
        bindings.append(
            _bind(
                document_id,
                translation_unit_id,
                span,
                token,
                status="bound_exact" if exact else "bound_normalized",
                kind="inline",
            )
        )
        paired_tokens.add(token_index)
        paired_spans.add(span_index)

    remaining_tokens = [
        (index, token) for index, token in enumerate(inline_tokens) if index not in paired_tokens
    ]
    remaining_spans = [
        (index, span) for index, span in enumerate(inline_spans) if index not in paired_spans
    ]
    mismatch_count = min(len(remaining_tokens), len(remaining_spans))
    for position in range(mismatch_count):
        token_index, token = remaining_tokens[position]
        span_index, span = remaining_spans[position]
        bindings.append(
            _bind(
                document_id,
                translation_unit_id,
                span,
                token,
                status="latex_mismatch",
                kind="inline",
            )
        )
        paired_tokens.add(token_index)
        paired_spans.add(span_index)

    for index, token in enumerate(inline_tokens):
        if index not in paired_tokens:
            bindings.append(
                _bind(
                    document_id,
                    translation_unit_id,
                    None,
                    token,
                    status="protected_token_only",
                    kind="inline",
                )
            )
    for index, span in enumerate(inline_spans):
        if index not in paired_spans:
            bindings.append(
                _bind(
                    document_id,
                    translation_unit_id,
                    span,
                    None,
                    status="source_span_only",
                    kind="inline",
                )
            )
    for index, token in enumerate(display_tokens):
        if index not in paired_display_tokens:
            bindings.append(
                _bind(
                    document_id,
                    translation_unit_id,
                    None,
                    token,
                    status="protected_token_only",
                    kind="display",
                )
            )

    def order_key(binding: FormulaBinding) -> tuple[int, int, int, int, str]:
        if binding.kind == "inline" and binding.source_text_start is not None:
            return (0, binding.source_text_start, 0, 0, binding.formula_id)
        kind_order = 1 if binding.kind == "inline" else 2
        return (
            kind_order,
            binding.page_idx if binding.page_idx is not None else 10**9,
            binding.source_line_index if binding.source_line_index is not None else 10**9,
            binding.source_span_index if binding.source_span_index is not None else 10**9,
            binding.formula_id,
        )

    return tuple(sorted(bindings, key=order_key))
