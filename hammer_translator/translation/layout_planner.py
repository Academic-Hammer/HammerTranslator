"""Translation-owned text allocation and bounded layout optimization.

This module deliberately stops before PDF drawing.  It assigns one complete
unit translation to its original blocks, searches typography only inside the
golden-style guardrails, and emits a deterministic contract for Reflow.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

from .cache import TranslationCache
from .layout import (
    FormulaRunSpec,
    LayoutMetricError,
    LayoutMetrics,
    LayoutResult,
    bind_formula_run_specs,
)
from .protection import ProtectionError, STANDARD_PATTERNS
from .provider import ProviderError, ProviderResponse, TranslationProvider, TranslationRequest
from .quality import QualityReport, evaluate_translation
from .runner import RunnerConfig, RunnerResult
from .taxonomy import TranslationPolicy
from .units import TranslationPlan, TranslationUnit


LAYOUT_OPTIMIZER_VERSION = "layout-optimizer-v2"
LAYOUT_CONTRACT_VERSION = "translation-reflow-v2"

STRONG_BREAKS = set("。！？!?\n")
MEDIUM_BREAKS = set("；：;:")
WEAK_BREAKS = set("，,.")
OPENING_PUNCTUATION = set("（([{【《“‘")
CLOSING_PUNCTUATION = set("）)]}】》”’，。！？；：,.!?;:")
VISIBLE_RE = re.compile(r"\s+")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rounded_range(start: float, stop: float, step: float) -> tuple[float, ...]:
    """Return a stable descending range that includes both endpoints."""

    start = float(start)
    stop = float(stop)
    if start < stop:
        start, stop = stop, start
    values = [start]
    value = start
    while value - step > stop + 1e-7:
        value -= step
        values.append(round(value, 6))
    if not math.isclose(values[-1], stop, abs_tol=1e-7):
        values.append(stop)
    return tuple(dict.fromkeys(round(item, 6) for item in values))


@dataclass(frozen=True)
class LayoutSearchConfig:
    font_size_step_pt: float = 0.1
    line_height_step: float = 0.01
    char_spacing_step_em: float = 0.005
    boundary_candidates_per_cut: int = 9
    max_allocation_candidates: int = 192
    capacity_target_tolerance_ratio: float = 0.04
    capacity_target_tolerance_min_chars: int = 2
    max_retranslation_attempts: int = 2
    capacity_safety_ratio: float = 0.94
    exact_pruning: bool = True
    exact_allocation_pruning: bool = True

    def __post_init__(self) -> None:
        positive = (
            self.font_size_step_pt,
            self.line_height_step,
            self.char_spacing_step_em,
            self.boundary_candidates_per_cut,
            self.max_allocation_candidates,
            self.capacity_target_tolerance_ratio,
            self.capacity_target_tolerance_min_chars,
            self.capacity_safety_ratio,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("layout search settings must be positive")
        if self.max_retranslation_attempts < 0:
            raise ValueError("max_retranslation_attempts cannot be negative")
        if self.capacity_safety_ratio > 1:
            raise ValueError("capacity_safety_ratio cannot exceed 1")
        if self.capacity_target_tolerance_ratio > 1:
            raise ValueError("capacity_target_tolerance_ratio cannot exceed 1")


@dataclass(frozen=True)
class StyleOptimization:
    status: str
    style: dict[str, Any]
    measurement: LayoutResult
    score: float | None
    evaluated_candidates: int
    feasible_candidates: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "style": dict(self.style),
            "measurement": self.measurement.as_dict(),
            "score": self.score,
            "evaluated_candidates": self.evaluated_candidates,
            "feasible_candidates": self.feasible_candidates,
        }


@dataclass(frozen=True)
class BlockAllocation:
    block_id: str
    text: str
    optimization: StyleOptimization

    def as_dict(self) -> dict[str, Any]:
        value = self.optimization.as_dict()
        return {
            "block_id": self.block_id,
            "translated": self.text,
            "fit_status": value["status"],
            "recommended_style": value["style"],
            "measurement": value["measurement"],
            "optimization_score": value["score"],
            "evaluated_style_candidates": value["evaluated_candidates"],
            "feasible_style_candidates": value["feasible_candidates"],
        }


@dataclass(frozen=True)
class UnitAllocation:
    status: str
    allocation_id: str | None
    score: float | None
    evaluated_allocations: int
    feasible_allocations: int
    segments: tuple[BlockAllocation, ...]
    retranslation_constraints: dict[str, Any] | None = None

    @property
    def all_blocks_fit(self) -> bool:
        return bool(self.segments) and all(
            segment.optimization.measurement.fits for segment in self.segments
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "optimizer_version": LAYOUT_OPTIMIZER_VERSION,
            "all_blocks_fit": self.all_blocks_fit,
            "selected_allocation_id": self.allocation_id,
            "score": self.score,
            "evaluated_allocations": self.evaluated_allocations,
            "feasible_allocations": self.feasible_allocations,
            "retranslation_constraints": self.retranslation_constraints,
            "segments": [segment.as_dict() for segment in self.segments],
        }


@dataclass(frozen=True)
class AcceptedCandidate:
    candidate_id: str
    text: str
    reason: str
    constraints: dict[str, Any]
    quality: QualityReport
    created_at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "mode": "constrained",
            "text": self.text,
            "reason": self.reason,
            "constraints": self.constraints,
            "quality": self.quality.as_dict(),
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class UnitLayoutOutcome:
    unit_id: str
    selected_text: str | None
    selected_version: str | None
    candidates: tuple[AcceptedCandidate, ...]
    allocation: UnitAllocation
    retranslation_attempts: int = 0
    retranslation_errors: tuple[dict[str, Any], ...] = ()

    def layout_dict(self) -> dict[str, Any]:
        value = self.allocation.as_dict()
        value["retranslation_attempts"] = self.retranslation_attempts
        value["retranslation_errors"] = list(self.retranslation_errors)
        return value


@dataclass
class LayoutPlanningStats:
    units_total: int = 0
    fits_golden: int = 0
    fits_with_adjustment: int = 0
    not_applicable: int = 0
    needs_review: int = 0
    requires_retranslation: int = 0
    constrained_requests: int = 0
    constrained_cache_hits: int = 0
    constrained_provider_calls: int = 0
    constrained_successes: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class LayoutPlanningResult:
    outcomes: dict[str, UnitLayoutOutcome]
    stats: LayoutPlanningStats


def _normalized_distance(value: float, golden: float, lower: float) -> float:
    span = max(golden - lower, 1e-9)
    return min(1.0, abs(golden - value) / span)


def _style_base_score(
    style: dict[str, Any],
    golden: dict[str, Any],
    guardrails: dict[str, Any],
) -> float:
    """Return the result-independent lower bound of the style score."""

    font_delta = _normalized_distance(
        float(style["font_size_pt"]),
        float(golden["font_size_pt"]),
        float(guardrails["font_size_pt"]["min"]),
    )
    line_delta = _normalized_distance(
        float(style["line_height_multiplier"]),
        float(golden["line_height_multiplier"]),
        float(guardrails["line_height_multiplier"]["min"]),
    )
    char_delta = _normalized_distance(
        float(style["char_spacing_em"]),
        float(golden["char_spacing_em"]),
        float(guardrails["char_spacing_em"]["min"]),
    )
    deviation = 4.0 * font_delta**2 + 2.0 * line_delta**2 + 1.5 * char_delta**2
    edge = 0.2 * (font_delta**6 + line_delta**6 + char_delta**6)
    return deviation + edge


def _style_score(
    result: LayoutResult,
    style: dict[str, Any],
    golden: dict[str, Any],
    guardrails: dict[str, Any],
) -> float:
    """Penalize visual deviation non-linearly and add line-quality costs."""

    widow = 0.0
    if result.line_count > 1 and result.line_widths_pt:
        last_fill = result.line_widths_pt[-1] / max(result.available_width_pt, 1e-9)
        if last_fill < 0.12:
            widow = 0.8
        elif last_fill < 0.2:
            widow = 0.25
    return round(_style_base_score(style, golden, guardrails) + widow, 8)


def optimize_style(
    metrics: LayoutMetrics,
    text: str,
    bbox: Iterable[float],
    *,
    role: str,
    config: LayoutSearchConfig | None = None,
    formula_runs: tuple[FormulaRunSpec, ...] = (),
) -> StyleOptimization:
    """Return the globally best feasible style using exact branch-and-bound."""

    settings = config or LayoutSearchConfig()
    golden, guardrails = metrics.style_for_role(role)
    font_values = _rounded_range(
        golden["font_size_pt"], guardrails["font_size_pt"]["min"], settings.font_size_step_pt
    )
    line_values = _rounded_range(
        golden["line_height_multiplier"],
        guardrails["line_height_multiplier"]["min"],
        settings.line_height_step,
    )
    char_values = _rounded_range(
        golden["char_spacing_em"],
        guardrails["char_spacing_em"]["min"],
        settings.char_spacing_step_em,
    )
    bbox_values = list(bbox)
    best: tuple[tuple[Any, ...], dict[str, Any], LayoutResult, float] | None = None
    most_compact: tuple[dict[str, Any], LayoutResult] | None = None
    evaluated = 0
    feasible = 0
    golden_result: LayoutResult | None = None
    if settings.exact_pruning:
        golden_result = metrics.measure(
            text,
            bbox_values,
            role=role,
            style=golden,
            formula_runs=formula_runs,
        )
        evaluated = 1
        if golden_result.fits:
            feasible = 1
            golden_score = _style_score(golden_result, golden, golden, guardrails)
            golden_key = (
                golden_score,
                -float(golden["font_size_pt"]),
                0.0,
                0.0,
            )
            best = (golden_key, dict(golden), golden_result, golden_score)
            # Every score component is non-negative, so zero is the proven
            # global lower bound rather than a heuristic first-fit decision.
            if golden_score == 0.0:
                return StyleOptimization(
                    "fits_golden",
                    dict(golden),
                    golden_result,
                    golden_score,
                    evaluated,
                    feasible,
                )

    wrapping_groups = []
    for font_size, char_spacing in itertools.product(font_values, char_values):
        lower_bound_style = dict(golden)
        lower_bound_style.update(
            font_size_pt=font_size,
            line_height_multiplier=line_values[0],
            char_spacing_em=char_spacing,
        )
        wrapping_groups.append(
            (
                _style_base_score(lower_bound_style, golden, guardrails),
                -float(font_size),
                abs(float(char_spacing) - float(golden["char_spacing_em"])),
                font_size,
                char_spacing,
            )
        )
    if settings.exact_pruning:
        wrapping_groups.sort()

    for group_lower_bound, _, _, font_size, char_spacing in wrapping_groups:
        if (
            settings.exact_pruning
            and best is not None
            and round(group_lower_bound, 8) > best[3]
        ):
            break
        wrapping_style = dict(golden)
        wrapping_style.update(
            font_size_pt=font_size,
            line_height_multiplier=line_values[0],
            char_spacing_em=char_spacing,
        )
        is_golden_wrapping = math.isclose(
            font_size, float(golden["font_size_pt"]), abs_tol=1e-7
        ) and math.isclose(
            char_spacing, float(golden["char_spacing_em"]), abs_tol=1e-7
        )
        wrapping_result = (
            golden_result
            if is_golden_wrapping and golden_result is not None
            else metrics.measure(
                text,
                bbox_values,
                role=role,
                style=wrapping_style,
                formula_runs=formula_runs,
            )
        )
        for line_height in line_values:
            style = dict(wrapping_style)
            style["line_height_multiplier"] = line_height
            is_precomputed_golden = (
                settings.exact_pruning
                and golden_result is not None
                and is_golden_wrapping
                and math.isclose(
                    line_height,
                    float(golden["line_height_multiplier"]),
                    abs_tol=1e-7,
                )
            )
            if is_precomputed_golden:
                continue
            base_score = _style_base_score(style, golden, guardrails)
            if (
                settings.exact_pruning
                and best is not None
                and round(base_score, 8) > best[3]
            ):
                break
            result = (
                wrapping_result
                if math.isclose(line_height, line_values[0], abs_tol=1e-7)
                else metrics.with_line_height(
                    wrapping_result,
                    line_height_multiplier=line_height,
                )
            )
            evaluated += 1
            if (
                math.isclose(font_size, font_values[-1], abs_tol=1e-7)
                and math.isclose(line_height, line_values[-1], abs_tol=1e-7)
                and math.isclose(char_spacing, char_values[-1], abs_tol=1e-7)
            ):
                most_compact = (style, result)
            if not result.fits:
                continue
            feasible += 1
            score = _style_score(result, style, golden, guardrails)
            tie_break = (
                score,
                -float(style["font_size_pt"]),
                abs(
                    float(style["line_height_multiplier"])
                    - float(golden["line_height_multiplier"])
                ),
                abs(float(style["char_spacing_em"]) - float(golden["char_spacing_em"])),
            )
            if best is None or tie_break < best[0]:
                best = (tie_break, style, result, score)

    if best is not None:
        _, style, result, score = best
        status = "fits_golden" if style == golden else "fits_with_adjustment"
        return StyleOptimization(status, style, result, score, evaluated, feasible)
    if most_compact is None:  # pragma: no cover - ranges always contain their endpoints
        raise RuntimeError("layout search failed to evaluate the compact style")
    style, result = most_compact
    return StyleOptimization(
        "requires_retranslation", style, result, None, evaluated, feasible
    )


def _protected_intervals(text: str, protected_values: Iterable[str]) -> tuple[tuple[int, int], ...]:
    intervals: list[tuple[int, int]] = []
    for value in protected_values:
        if not value:
            continue
        start = 0
        while True:
            index = text.find(value, start)
            if index < 0:
                break
            intervals.append((index, index + len(value)))
            start = index + len(value)
    return tuple(sorted(intervals))


def _boundary_penalty(text: str, position: int) -> float:
    before = text[position - 1] if position > 0 else ""
    after = text[position] if position < len(text) else ""
    if before in STRONG_BREAKS:
        penalty = 0.0
    elif before in MEDIUM_BREAKS:
        penalty = 0.08
    elif before in WEAK_BREAKS:
        penalty = 0.16
    elif before.isspace():
        penalty = 0.12
    else:
        penalty = 0.45
    if before in OPENING_PUNCTUATION or after in CLOSING_PUNCTUATION:
        penalty += 0.8
    return penalty


def _valid_boundaries(text: str, protected_values: Iterable[str]) -> tuple[int, ...]:
    intervals = list(_protected_intervals(text, protected_values))
    formula_kinds = {
        "latex_environment",
        "latex_display",
        "latex_bracket",
        "latex_parenthesis",
        "latex_inline",
    }
    for kind, pattern in STANDARD_PATTERNS:
        if kind not in formula_kinds:
            continue
        intervals.extend(match.span() for match in pattern.finditer(text))
    intervals.sort()
    values = []
    for position in range(1, len(text)):
        if any(start < position < end for start, end in intervals):
            continue
        values.append(position)
    return tuple(values)


def _candidate_cut_sets(
    text: str,
    capacities: list[int],
    protected_values: Iterable[str],
    config: LayoutSearchConfig,
) -> tuple[tuple[int, ...], ...]:
    if len(capacities) <= 1:
        return ((),)
    boundaries = _valid_boundaries(text, protected_values)
    if not boundaries:
        return ()
    targets = _capacity_targets(len(text), capacities)
    tolerance = _capacity_target_tolerance(len(text), config)
    choices: list[tuple[int, ...]] = []
    components: list[dict[int, tuple[float, float, float]]] = []
    for target in targets:
        ranked = sorted(
            boundaries,
            key=lambda position: (
                *_cut_priority_component(text, position, target, tolerance),
                position,
            ),
        )
        selected = tuple(ranked[: config.boundary_candidates_per_cut])
        choices.append(selected)
        components.append(
            {
                position: _cut_priority_component(text, position, target, tolerance)
                for position in selected
            }
        )

    # Keep only the top K prefixes for each final cut.  Future validity depends
    # only on that final cut, so a worse prefix ending at the same position can
    # never enter the global top K.  This avoids materializing the full Cartesian
    # product while preserving the exact ordering defined by _cut_priority_key.
    states: dict[int, list[tuple[float, float, float, tuple[int, ...]]]] = {}
    for index, selected in enumerate(choices):
        next_states: dict[int, list[tuple[float, float, float, tuple[int, ...]]]] = {}
        for position in selected:
            excess, boundary, distance = components[index][position]
            if index == 0:
                records = [(excess, boundary, distance, (position,))]
            else:
                records = [
                    (
                        prefix_excess + excess,
                        prefix_boundary + boundary,
                        prefix_distance + distance,
                        (*prefix, position),
                    )
                    for previous, prefixes in states.items()
                    if previous < position
                    for prefix_excess, prefix_boundary, prefix_distance, prefix in prefixes
                ]
            if not records:
                continue
            records.sort(key=lambda record: (*record[:3], record[3]))
            next_states[position] = records[: config.max_allocation_candidates]
        states = next_states
        if not states:
            return ()

    combinations = [record for records in states.values() for record in records]
    combinations.sort(key=lambda record: (*record[:3], record[3]))
    return tuple(
        record[3] for record in combinations[: config.max_allocation_candidates]
    )


def _capacity_targets(text_length: int, capacities: list[int]) -> tuple[float, ...]:
    """Return cumulative cut targets derived from effective block capacity."""

    total_capacity = max(sum(max(0, value) for value in capacities), 1)
    targets: list[float] = []
    cumulative = 0
    for capacity in capacities[:-1]:
        cumulative += max(0, capacity)
        targets.append(text_length * cumulative / total_capacity)
    return tuple(targets)


def _capacity_target_tolerance(text_length: int, config: LayoutSearchConfig) -> float:
    """Treat nearby cut points as equivalent before applying punctuation quality."""

    return max(
        float(config.capacity_target_tolerance_min_chars),
        float(text_length) * config.capacity_target_tolerance_ratio,
    )


def _cut_priority_component(
    text: str,
    position: int,
    target: float,
    tolerance: float,
) -> tuple[float, float, float]:
    """Rank capacity first, then natural boundaries inside its tolerance band."""

    length = max(len(text), 1)
    distance = abs(position - target)
    excess = max(0.0, distance - tolerance)
    return (
        round(excess / length, 8),
        _boundary_penalty(text, position),
        round(distance / length, 8),
    )


def _cut_priority_key(
    text: str,
    cuts: tuple[int, ...],
    capacities: list[int],
    config: LayoutSearchConfig,
) -> tuple[float, float, float, tuple[int, ...]]:
    """Return the capacity-first deterministic priority of one complete split."""

    targets = _capacity_targets(len(text), capacities)
    tolerance = _capacity_target_tolerance(len(text), config)
    components = [
        _cut_priority_component(text, cut, target, tolerance)
        for cut, target in zip(cuts, targets, strict=True)
    ]
    return (
        round(sum(value[0] for value in components), 8),
        round(sum(value[1] for value in components), 8),
        round(sum(value[2] for value in components), 8),
        cuts,
    )


def _split_at(text: str, cuts: tuple[int, ...]) -> tuple[str, ...]:
    points = (0, *cuts, len(text))
    return tuple(text[points[index] : points[index + 1]] for index in range(len(points) - 1))


def _allocation_id(unit_id: str, cuts: tuple[int, ...], text: str) -> str:
    material = f"{unit_id}\0{','.join(map(str, cuts))}\0{text}"
    return f"alloc-{hashlib.sha256(material.encode('utf-8')).hexdigest()[:16]}"


def _allocation_score(
    text: str,
    cuts: tuple[int, ...],
    segments: tuple[BlockAllocation, ...],
) -> float:
    style_cost = sum(segment.optimization.score or 0.0 for segment in segments)
    boundary_cost = sum(_boundary_penalty(text, cut) for cut in cuts)
    fill_ratios = [segment.optimization.measurement.height_fill_ratio for segment in segments]
    balance_cost = (max(fill_ratios) - min(fill_ratios)) ** 2 if len(fill_ratios) > 1 else 0.0
    font_sizes = [float(segment.optimization.style["font_size_pt"]) for segment in segments]
    consistency_cost = (max(font_sizes) - min(font_sizes)) ** 2 if len(font_sizes) > 1 else 0.0
    return round(style_cost + boundary_cost + 0.8 * balance_cost + 0.5 * consistency_cost, 8)


def _overflow_rank(segments: tuple[BlockAllocation, ...]) -> tuple[float, float, float]:
    height = sum(
        segment.optimization.measurement.overflow_height_pt
        / max(segment.optimization.measurement.available_height_pt, 1e-9)
        for segment in segments
    )
    width = sum(
        segment.optimization.measurement.overflow_width_pt
        / max(segment.optimization.measurement.available_width_pt, 1e-9)
        for segment in segments
    )
    imbalance = 0.0
    fills = [segment.optimization.measurement.height_fill_ratio for segment in segments]
    if len(fills) > 1:
        imbalance = max(fills) - min(fills)
    return round(height, 8), round(width, 8), round(imbalance, 8)


def _retranslation_constraints(
    text: str,
    role: str,
    segments: tuple[BlockAllocation, ...],
    protected_values: Iterable[str],
    config: LayoutSearchConfig,
) -> dict[str, Any]:
    protected_values = tuple(protected_values)
    capacity = sum(segment.optimization.measurement.estimated_cjk_capacity for segment in segments)
    raw_target = max(0, math.floor(capacity * config.capacity_safety_ratio))
    target = max(1, raw_target)
    visible = len(VISIBLE_RE.sub("", text))
    protected_visible = sum(len(VISIBLE_RE.sub("", value)) for value in protected_values)
    ratio = max(0.0, 1.0 - target / max(visible, 1))
    overflow_height = sum(
        segment.optimization.measurement.overflow_height_pt for segment in segments
    )
    overflow_width = sum(
        segment.optimization.measurement.overflow_width_pt for segment in segments
    )
    protected_only_overflow = any(
        not segment.optimization.measurement.fits
        and any(segment.text.strip() == value.strip() for value in protected_values)
        for segment in segments
    )
    zero_line_capacity = any(
        not segment.optimization.measurement.fits
        and segment.optimization.measurement.estimated_cjk_capacity == 0
        for segment in segments
    )
    impossibility_reason = None
    if protected_only_overflow:
        impossibility_reason = "protected_content_does_not_fit_assigned_bbox"
    elif zero_line_capacity:
        impossibility_reason = "minimum_allowed_line_height_exceeds_bbox"
    retranslation_possible = bool(
        raw_target > 0
        and target < visible
        and target >= protected_visible
        and not protected_only_overflow
        and not zero_line_capacity
    )
    return {
        "reason": "fixed_bbox_overflow",
        "style_role": role,
        "current_visible_characters": visible,
        "target_max_characters": target,
        "protected_visible_characters": protected_visible,
        "required_reduction_ratio": round(ratio, 4),
        "overflow_height_pt": round(overflow_height, 4),
        "overflow_width_pt": round(overflow_width, 4),
        "preserve_meaning": True,
        "summary_not_allowed": True,
        "preserve_terms_numbers_citations_formulas": True,
        "protected_values": list(protected_values),
        "bbox_locked": True,
        "retranslation_possible": retranslation_possible,
        "retranslation_impossibility_reason": impossibility_reason,
    }


def _formula_specs_for_unit(
    unit: TranslationUnit,
    text: str,
    formula_asset_contract: dict[str, Any] | None,
) -> tuple[FormulaRunSpec, ...]:
    if formula_asset_contract is None or not unit.formula_bindings:
        return ()
    return bind_formula_run_specs(
        text,
        [binding.as_dict() for binding in unit.formula_bindings],
        [
            {
                "placeholder": token.placeholder,
                "original": token.original,
                "kind": token.kind,
            }
            for token in unit.protected_tokens
        ],
        list(formula_asset_contract.get("assets") or []),
    )


def _formula_specs_for_segment(
    formula_runs: tuple[FormulaRunSpec, ...],
    start: int,
    end: int,
) -> tuple[FormulaRunSpec, ...]:
    selected: list[FormulaRunSpec] = []
    for run in formula_runs:
        if run.end <= start or run.start >= end:
            continue
        if run.start < start or run.end > end:
            raise LayoutMetricError(f"formula run {run.formula_id} crosses a block allocation cut")
        selected.append(run.shifted(start))
    return tuple(selected)


def allocate_unit_text(
    unit: TranslationUnit,
    text: str,
    *,
    blocks: list[Any],
    role: str,
    metrics: LayoutMetrics,
    config: LayoutSearchConfig | None = None,
    formula_asset_contract: dict[str, Any] | None = None,
) -> UnitAllocation:
    """Allocate a complete translation contiguously across its source blocks."""

    settings = config or LayoutSearchConfig()
    formula_runs = _formula_specs_for_unit(unit, text, formula_asset_contract)
    if not blocks:
        raise ValueError(f"translation unit {unit.translation_unit_id} has no blocks")
    if len(blocks) == 1:
        optimization = optimize_style(
            metrics,
            text,
            blocks[0].data.bbox,
            role=role,
            config=settings,
            formula_runs=formula_runs,
        )
        segment = BlockAllocation(blocks[0].block_id, text, optimization)
        constraints = None
        if not optimization.measurement.fits:
            constraints = _retranslation_constraints(
                text, role, (segment,), (token.original for token in unit.protected_tokens), settings
            )
        return UnitAllocation(
            status=optimization.status,
            allocation_id=_allocation_id(unit.translation_unit_id, (), text),
            score=optimization.score,
            evaluated_allocations=1,
            feasible_allocations=1 if optimization.measurement.fits else 0,
            segments=(segment,),
            retranslation_constraints=constraints,
        )

    effective_capacities: list[int] = []
    capacity_style, _ = metrics.style_for_role(role)
    for block in blocks:
        effective_capacities.append(
            metrics.measure(text[:1] or "汉", block.data.bbox, role=role, style=capacity_style)
            .estimated_cjk_capacity
        )
    cuts_to_try = _candidate_cut_sets(
        text,
        effective_capacities,
        (token.original for token in unit.protected_tokens),
        settings,
    )
    if not cuts_to_try:
        cuts_to_try = ((),)

    best: tuple[tuple[Any, ...], tuple[int, ...], tuple[BlockAllocation, ...], float] | None = None
    best_overflow: tuple[
        tuple[Any, ...], tuple[int, ...], tuple[BlockAllocation, ...]
    ] | None = None
    optimization_cache: dict[
        tuple[str, tuple[float, ...], tuple[FormulaRunSpec, ...]], StyleOptimization
    ] = {}

    def optimize_segment(
        block: Any,
        part: str,
        start: int,
        end: int,
    ) -> StyleOptimization:
        segment_formulas = _formula_specs_for_segment(formula_runs, start, end)
        cache_key = (
            part,
            tuple(float(value) for value in block.data.bbox),
            segment_formulas,
        )
        if settings.exact_allocation_pruning and cache_key in optimization_cache:
            return optimization_cache[cache_key]
        optimization = optimize_style(
            metrics,
            part,
            block.data.bbox,
            role=role,
            config=settings,
            formula_runs=segment_formulas,
        )
        if settings.exact_allocation_pruning:
            optimization_cache[cache_key] = optimization
        return optimization

    evaluated = 0
    feasible = 0
    for cuts in cuts_to_try:
        parts = _split_at(text, cuts)
        if len(parts) != len(blocks) or any(not part.strip() for part in parts):
            continue
        cut_priority = _cut_priority_key(text, cuts, effective_capacities, settings)
        if best is not None and settings.exact_allocation_pruning:
            if cut_priority[0] > best[0][0]:
                break
            if cut_priority[0] == best[0][0] and cut_priority[1] > best[3]:
                evaluated += 1
                continue
        points = (0, *cuts, len(text))
        evaluated += 1
        allocation_values: list[BlockAllocation] = []
        style_lower_bound = cut_priority[1]
        pruned = False
        for index, (block, part) in enumerate(zip(blocks, parts, strict=True)):
            optimization = optimize_segment(
                block,
                part,
                points[index],
                points[index + 1],
            )
            allocation_values.append(BlockAllocation(block.block_id, part, optimization))
            if not optimization.measurement.fits:
                if best is not None and settings.exact_allocation_pruning:
                    pruned = True
                    break
                continue
            style_lower_bound += float(optimization.score or 0.0)
            if (
                best is not None
                and settings.exact_allocation_pruning
                and cut_priority[0] == best[0][0]
                and round(style_lower_bound, 8) > best[3]
            ):
                pruned = True
                break
        if pruned:
            continue
        allocations = tuple(allocation_values)
        if all(segment.optimization.measurement.fits for segment in allocations):
            feasible += 1
            score = _allocation_score(text, cuts, allocations)
            key = (cut_priority[0], score, cut_priority[2], cuts)
            if best is None or key < best[0]:
                best = (key, cuts, allocations, score)
        else:
            key = (*_overflow_rank(allocations), sum(_boundary_penalty(text, cut) for cut in cuts), cuts)
            if best_overflow is None or key < best_overflow[0]:
                best_overflow = (key, cuts, allocations)

    if best is not None:
        _, cuts, allocations, score = best
        status = (
            "fits_golden"
            if all(segment.optimization.status == "fits_golden" for segment in allocations)
            else "fits_with_adjustment"
        )
        return UnitAllocation(
            status=status,
            allocation_id=_allocation_id(unit.translation_unit_id, cuts, text),
            score=score,
            evaluated_allocations=evaluated,
            feasible_allocations=feasible,
            segments=allocations,
        )

    if best_overflow is None:
        # Text shorter than the number of blocks cannot produce non-empty segments.
        parts = tuple(text if index == 0 else "" for index in range(len(blocks)))
        allocations = tuple(
            BlockAllocation(
                block.block_id,
                part,
                optimize_style(
                    metrics,
                    part,
                    block.data.bbox,
                    role=role,
                    config=settings,
                    formula_runs=_formula_specs_for_segment(
                        formula_runs,
                        0 if index == 0 else len(text),
                        len(text),
                    ),
                ),
            )
            for index, (block, part) in enumerate(zip(blocks, parts, strict=True))
        )
        cuts = ()
    else:
        _, cuts, allocations = best_overflow
    constraints = _retranslation_constraints(
        text, role, allocations, (token.original for token in unit.protected_tokens), settings
    )
    return UnitAllocation(
        status="requires_retranslation",
        allocation_id=_allocation_id(unit.translation_unit_id, cuts, text),
        score=None,
        evaluated_allocations=evaluated,
        feasible_allocations=0,
        segments=allocations,
        retranslation_constraints=constraints,
    )


def _non_layout_outcome(unit: TranslationUnit, status: str) -> UnitLayoutOutcome:
    allocation = UnitAllocation(status, None, None, 0, 0, ())
    selected = unit.source_text if unit.policy is TranslationPolicy.PRESERVE else None
    version = "source" if selected is not None else None
    return UnitLayoutOutcome(unit.translation_unit_id, selected, version, (), allocation)


def _review_if_retranslation_cannot_help(allocation: UnitAllocation) -> UnitAllocation:
    if allocation.status != "requires_retranslation" or bool(
        (allocation.retranslation_constraints or {}).get("retranslation_possible", True)
    ):
        return allocation
    constraints = allocation.retranslation_constraints or {}
    return UnitAllocation(
        status="needs_review",
        allocation_id=allocation.allocation_id,
        score=allocation.score,
        evaluated_allocations=allocation.evaluated_allocations,
        feasible_allocations=allocation.feasible_allocations,
        segments=allocation.segments,
        retranslation_constraints={
            **constraints,
            "reason": constraints.get("retranslation_impossibility_reason")
            or "overflow_cannot_be_reduced_by_retranslation",
        },
    )


def _constrained_request(
    unit: TranslationUnit,
    provider: TranslationProvider,
    config: RunnerConfig,
    constraints: dict[str, Any],
    current_translation: str,
) -> TranslationRequest:
    values = dict(constraints)
    values["current_translation"] = current_translation
    return TranslationRequest(
        translation_unit_id=unit.translation_unit_id,
        protected_text=unit.protected_text,
        source_language=config.source_language,
        target_language=config.target_language,
        element_type=unit.element_type.value,
        provider=provider.name,
        model=provider.model,
        prompt_version=config.prompt_version,
        glossary_version=config.glossary_version,
        mode="constrained",
        constraints=values,
    )


def _candidate_from_response(
    unit: TranslationUnit,
    request: TranslationRequest,
    response: ProviderResponse,
    *,
    simulation: bool,
    ordinal: int,
) -> AcceptedCandidate:
    restored = unit.restore_protected_content(response.text)
    quality = evaluate_translation(
        unit.protected_text,
        response.text,
        target_language=request.target_language,
        simulation=simulation,
    )
    if quality.should_retry:
        raise ProviderError(
            "; ".join(flag.message for flag in quality.flags),
            code="constrained_translation_quality_rejected",
            retryable=True,
        )
    candidate_id = f"candidate-{ordinal:02d}-{hashlib.sha256(restored.encode('utf-8')).hexdigest()[:10]}"
    return AcceptedCandidate(
        candidate_id=candidate_id,
        text=restored,
        reason=str(request.constraints.get("reason") or "fixed_bbox_overflow"),
        constraints={key: value for key, value in request.constraints.items() if key != "current_translation"},
        quality=quality,
        created_at=_utc_now(),
    )


def plan_translation_layout(
    plan: TranslationPlan,
    result: RunnerResult,
    style_profile: dict[str, Any],
    *,
    provider: TranslationProvider | None = None,
    cache: TranslationCache | None = None,
    runner_config: RunnerConfig | None = None,
    search_config: LayoutSearchConfig | None = None,
    formula_asset_contract: dict[str, Any] | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> LayoutPlanningResult:
    """Plan item text and optionally execute bounded constrained retranslations."""

    settings = search_config or LayoutSearchConfig()
    run_settings = runner_config or RunnerConfig()
    metrics = LayoutMetrics(style_profile)
    block_by_id = {block.block_id: block for block in plan.document.blocks}
    stats = LayoutPlanningStats(units_total=len(plan.units))
    outcomes: dict[str, UnitLayoutOutcome] = {}

    for unit_index, unit in enumerate(plan.units, start=1):
        unit_id = unit.translation_unit_id
        if progress is not None:
            progress(
                {
                    "event": "layout_unit_started",
                    "unit_index": unit_index,
                    "unit_total": len(plan.units),
                    "translation_unit_id": unit_id,
                    "element_type": unit.element_type.value,
                }
            )
        if unit.policy is TranslationPolicy.SKIP:
            outcomes[unit_id] = _non_layout_outcome(unit, "not_applicable")
            stats.not_applicable += 1
            continue
        if unit.policy is TranslationPolicy.NEEDS_REVIEW or unit_id not in result.outputs:
            outcomes[unit_id] = _non_layout_outcome(unit, "needs_review")
            stats.needs_review += 1
            continue

        text = result.outputs[unit_id]
        selected_version = "source" if unit.policy is TranslationPolicy.PRESERVE else "initial"
        role = style_profile["unit_roles"][unit_id]
        blocks = [block_by_id[block_id] for block_id in unit.block_ids]
        try:
            allocation = allocate_unit_text(
                unit,
                text,
                blocks=blocks,
                role=role,
                metrics=metrics,
                config=settings,
                formula_asset_contract=formula_asset_contract,
            )
        except LayoutMetricError as exc:
            outcomes[unit_id] = UnitLayoutOutcome(
                unit_id=unit_id,
                selected_text=text,
                selected_version=selected_version,
                candidates=(),
                allocation=UnitAllocation(
                    status="needs_review",
                    allocation_id=None,
                    score=None,
                    evaluated_allocations=0,
                    feasible_allocations=0,
                    segments=(),
                    retranslation_constraints={
                        "reason": "formula_layout_contract_invalid",
                        "detail": str(exc),
                        "retranslation_possible": False,
                    },
                ),
            )
            stats.needs_review += 1
            continue
        allocation = _review_if_retranslation_cannot_help(allocation)
        if (
            unit.policy is TranslationPolicy.PRESERVE
            and not allocation.all_blocks_fit
        ):
            allocation = UnitAllocation(
                status="needs_review",
                allocation_id=allocation.allocation_id,
                score=allocation.score,
                evaluated_allocations=allocation.evaluated_allocations,
                feasible_allocations=allocation.feasible_allocations,
                segments=allocation.segments,
                retranslation_constraints={
                    **(allocation.retranslation_constraints or {}),
                    "reason": "immutable_content_overflow",
                    "retranslation_allowed": False,
                },
            )
        candidates: list[AcceptedCandidate] = []
        errors: list[dict[str, Any]] = []
        attempts = 0

        can_retranslate = (
            allocation.status == "requires_retranslation"
            and unit.policy not in {TranslationPolicy.PRESERVE, TranslationPolicy.SKIP}
            and provider is not None
            and cache is not None
            and bool(
                (allocation.retranslation_constraints or {}).get(
                    "retranslation_possible", True
                )
            )
        )
        while can_retranslate and attempts < settings.max_retranslation_attempts:
            attempts += 1
            stats.constrained_requests += 1
            if progress is not None:
                progress(
                    {
                        "event": "constrained_request",
                        "unit_index": unit_index,
                        "unit_total": len(plan.units),
                        "translation_unit_id": unit_id,
                        "element_type": unit.element_type.value,
                        "attempt": attempts,
                        "max_attempts": settings.max_retranslation_attempts,
                    }
                )
            constraints = dict(allocation.retranslation_constraints or {})
            request = _constrained_request(unit, provider, run_settings, constraints, text)
            response: ProviderResponse | None = None
            abort_retranslation = False
            cached = cache.get(request.cache_key())
            if cached is not None:
                stats.constrained_cache_hits += 1
                response = ProviderResponse(
                    text=cached.response_text,
                    provider=cached.provider,
                    model=cached.model,
                    request_id=cached.metadata.get("request_id"),
                    metadata=cached.metadata,
                )
            else:
                try:
                    stats.constrained_provider_calls += 1
                    response = provider.translate(request)
                    cache.put(request, response)
                except ProviderError as exc:
                    errors.append(exc.safe_details())
                    abort_retranslation = not exc.retryable
                except Exception as exc:  # provider boundary: never leak raw details
                    errors.append(
                        ProviderError(
                            f"unexpected constrained provider error: {type(exc).__name__}",
                            code="unexpected_constrained_provider_error",
                            retryable=False,
                        ).safe_details()
                    )
                    abort_retranslation = True
            if response is None:
                if abort_retranslation:
                    break
                continue
            try:
                candidate = _candidate_from_response(
                    unit,
                    request,
                    response,
                    simulation=provider.name == "fake",
                    ordinal=len(candidates) + 1,
                )
            except ProtectionError as exc:
                cache.delete(request.cache_key())
                errors.append(
                    ProviderError(
                        str(exc), code="protected_content_invalid", retryable=True
                    ).safe_details()
                )
                continue
            except ProviderError as exc:
                cache.delete(request.cache_key())
                errors.append(exc.safe_details())
                continue
            candidates.append(candidate)
            candidate_allocation = allocate_unit_text(
                unit,
                candidate.text,
                blocks=blocks,
                role=role,
                metrics=metrics,
                config=settings,
                formula_asset_contract=formula_asset_contract,
            )
            text = candidate.text
            allocation = _review_if_retranslation_cannot_help(candidate_allocation)
            selected_version = candidate.candidate_id
            if allocation.all_blocks_fit:
                stats.constrained_successes += 1
                break
            can_retranslate = (
                allocation.status == "requires_retranslation"
                and bool(
                    (allocation.retranslation_constraints or {}).get(
                        "retranslation_possible", True
                    )
                )
            )

        outcome = UnitLayoutOutcome(
            unit_id=unit_id,
            selected_text=text,
            selected_version=selected_version,
            candidates=tuple(candidates),
            allocation=allocation,
            retranslation_attempts=attempts,
            retranslation_errors=tuple(errors),
        )
        outcomes[unit_id] = outcome
        if allocation.status == "fits_golden":
            stats.fits_golden += 1
        elif allocation.status == "fits_with_adjustment":
            stats.fits_with_adjustment += 1
        elif allocation.status == "requires_retranslation":
            stats.requires_retranslation += 1
        else:
            stats.needs_review += 1

    return LayoutPlanningResult(outcomes=outcomes, stats=stats)
