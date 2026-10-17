"""Build and atomically write the readable Translation Stage artifact."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any

from .formula import FORMULA_CONTRACT_VERSION, FormulaBinding
from .formula_assets import build_formula_asset_contract
from .jsonio import write_json_atomic
from .layout_planner import (
    LAYOUT_CONTRACT_VERSION,
    LAYOUT_OPTIMIZER_VERSION,
    LayoutPlanningResult,
    UnitLayoutOutcome,
    plan_translation_layout,
)
from .output_schema import OUTPUT_SCHEMA_VERSION, TranslationArtifactOutput
from .provider import TranslationProvider
from .quality import QualityReport
from .runner import RunnerConfig, RunnerResult
from .style import build_golden_style_profile, style_role_for_unit
from .taxonomy import TranslationPolicy
from .units import TranslationPlan, TranslationUnit


def _source_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _status_for(
    unit: TranslationUnit,
    *,
    provider: TranslationProvider,
    output: str | None,
    quality: QualityReport,
) -> str:
    if unit.policy is TranslationPolicy.PRESERVE:
        return "preserved"
    if unit.policy is TranslationPolicy.SKIP:
        return "skipped"
    if unit.policy is TranslationPolicy.NEEDS_REVIEW:
        return "needs_review"
    if output is None:
        return "failed"
    if provider.name == "fake":
        return "simulated"
    if quality.needs_review:
        return "needs_review"
    return "translated"


def _translation_payload(
    unit: TranslationUnit,
    *,
    provider: TranslationProvider,
    result: RunnerResult,
    layout_outcome: UnitLayoutOutcome,
) -> tuple[str, dict[str, Any]]:
    unit_id = unit.translation_unit_id
    output = result.outputs.get(unit_id)
    quality = result.quality_reports.get(unit_id, QualityReport.not_applicable())
    state = result.unit_states[unit_id]
    status = _status_for(unit, provider=provider, output=output, quality=quality)

    if status == "preserved":
        initial_text = None
        selected_text = unit.source_text
        selected_version = "source"
    elif status in {"translated", "simulated"} or (
        status == "needs_review" and output is not None and unit.policy is not TranslationPolicy.NEEDS_REVIEW
    ):
        initial_text = output
        selected_text = output
        selected_version = "initial"
    else:
        initial_text = None
        selected_text = None
        selected_version = None

    quality_flags = list(dict.fromkeys((*unit.quality_flags, *(flag.code for flag in quality.flags))))
    selected_quality = quality
    if layout_outcome.candidates:
        selected_quality = layout_outcome.candidates[-1].quality
    # A QA warning changes publication status to needs_review, but it does not
    # discard an otherwise usable provider output.  Layout may select a bounded
    # retranslation of that output, so the canonical text must follow the
    # layout outcome for every unit that actually has selected text.
    if layout_outcome.selected_text is not None:
        selected_text = layout_outcome.selected_text
        selected_version = layout_outcome.selected_version

    formula_by_placeholder = {
        binding.protected_placeholder: binding.formula_id
        for binding in unit.formula_bindings
        if binding.protected_placeholder is not None
    }
    payload = {
        "status": status,
        "initial_text": initial_text,
        "candidates": [candidate.as_dict() for candidate in layout_outcome.candidates],
        "selected_text": selected_text,
        "selected_version": selected_version,
        "source_hash": _source_hash(unit.source_text),
        "quality_flags": quality_flags,
        "quality": selected_quality.as_dict(),
        "protected_tokens": [
            {
                "placeholder": token.placeholder,
                "original": token.original,
                "kind": token.kind,
                "start": token.start,
                "end": token.end,
                "formula_id": formula_by_placeholder.get(token.placeholder),
            }
            for token in unit.protected_tokens
        ],
        "execution": {
            "attempts": int(state.get("attempts") or 0),
            "cache_hit": bool(state.get("cache_hit")),
            "cache_key": state.get("cache_key"),
            "response_hash": state.get("response_hash"),
            "request_id": state.get("request_id"),
            "last_error": state.get("last_error"),
        },
    }
    return status, payload


def build_translation_artifact(
    plan: TranslationPlan,
    *,
    provider: TranslationProvider,
    requested_provider: str,
    result: RunnerResult,
    config: RunnerConfig,
    style_profile: dict[str, Any] | None = None,
    layout_planning: LayoutPlanningResult | None = None,
    formula_asset_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    golden_profile = style_profile or build_golden_style_profile(plan)
    planning = layout_planning or plan_translation_layout(
        plan,
        result,
        golden_profile,
        formula_asset_contract=formula_asset_contract,
    )
    unit_records: list[dict[str, Any]] = []
    unit_statuses: dict[str, str] = {}
    block_to_unit: dict[str, TranslationUnit] = {}
    formula_bindings: list[FormulaBinding] = [
        binding for unit in plan.units for binding in unit.formula_bindings
    ]
    assets_by_formula = {
        str(asset["formula_id"]): asset
        for asset in (formula_asset_contract or {}).get("assets", [])
    }

    for unit in plan.units:
        style_role = style_role_for_unit(plan, unit)
        golden_style = golden_profile["roles"][style_role]
        layout_outcome = planning.outcomes[unit.translation_unit_id]
        status, translation = _translation_payload(
            unit,
            provider=provider,
            result=result,
            layout_outcome=layout_outcome,
        )
        unit_statuses[unit.translation_unit_id] = status
        for block_id in unit.block_ids:
            block_to_unit[block_id] = unit
        unit_records.append(
            {
                "translation_unit_id": unit.translation_unit_id,
                "logical_para_id": unit.logical_para_id,
                "element_type": unit.element_type.value,
                "policy": unit.policy.value,
                "style_role": style_role,
                "golden_style": golden_style,
                "block_ids": list(unit.block_ids),
                "formula_ids": [binding.formula_id for binding in unit.formula_bindings],
                "source_text": unit.source_text,
                "translation": translation,
                "layout": layout_outcome.layout_dict(),
            }
        )

    segment_by_block = {
        segment.block_id: (outcome, segment)
        for outcome in planning.outcomes.values()
        for segment in outcome.allocation.segments
    }
    items: list[dict[str, Any]] = []
    for block in plan.document.blocks:
        unit = block_to_unit[block.block_id]
        item = deepcopy(block.raw)
        item["block_id"] = block.block_id
        item["translation_unit_id"] = unit.translation_unit_id
        item["detected_type"] = unit.element_type.value
        item["translation_status"] = unit_statuses[unit.translation_unit_id]
        source_formula_ids = [
            binding.formula_id
            for binding in unit.formula_bindings
            if binding.block_id == block.block_id
        ]
        fixed_formula_ids = [
            binding.formula_id
            for binding in unit.formula_bindings
            if binding.block_id == block.block_id
            and not (
                binding.kind == "inline"
                and binding.render_policy == "move_with_text"
            )
        ]
        item["formula_ids"] = source_formula_ids
        role = golden_profile["unit_roles"][unit.translation_unit_id]
        segment_record = segment_by_block.get(block.block_id)
        display_status = "not_applicable"
        if segment_record is not None:
            outcome, segment = segment_record
            item["translated"] = segment.text
            item["golden_style"] = deepcopy(golden_profile["roles"][role])
            item["recommended_style"] = dict(segment.optimization.style)
            item["layout_preflight"] = {
                "status": segment.optimization.status,
                "optimizer_version": LAYOUT_OPTIMIZER_VERSION,
                "measurement": segment.optimization.measurement.as_dict(),
                "optimization_score": segment.optimization.score,
            }
            mixed_formula_ids = [
                run.formula_id
                for line in segment.optimization.measurement.mixed_lines
                for run in line.runs
                if run.kind == "formula" and run.formula_id is not None
            ]
            if formula_asset_contract is not None:
                # Only movable inline formulas follow the allocated translation.
                # Display formulas and review-only source formulas retain their
                # original block link even though they are absent from mixed text
                # measurement runs.
                item["formula_ids"] = list(
                    dict.fromkeys((*fixed_formula_ids, *mixed_formula_ids))
                )
            display_status = (
                "ready_for_reflow"
                if segment.optimization.measurement.fits
                else outcome.allocation.status
            )
        elif layout_outcome := planning.outcomes.get(unit.translation_unit_id):
            display_status = layout_outcome.allocation.status
        item["translation"] = {
            "translation_unit_id": unit.translation_unit_id,
            "canonical_text_field": "translation_units[].translation.selected_text",
            "display_text_status": display_status,
            "style_role": role,
        }
        items.append(item)

    allocation_statuses = {
        outcome.allocation.status for outcome in planning.outcomes.values()
    }
    if "requires_retranslation" in allocation_statuses:
        layout_status = "needs_retranslation"
    elif "needs_review" in allocation_statuses:
        layout_status = "needs_review"
    else:
        layout_status = "ready_for_reflow"

    formula_inventory = []
    for binding in formula_bindings:
        record = binding.as_dict()
        asset = assets_by_formula.get(binding.formula_id)
        if asset is not None:
            record["asset_id"] = asset["asset_id"]
            record["asset_status"] = asset["status"]
        else:
            record["asset_id"] = None
        formula_inventory.append(record)

    artifact = {
        "schema_version": OUTPUT_SCHEMA_VERSION if formula_asset_contract is not None else "translation-v2",
        "document": {
            "document_id": plan.document.document_id,
            "source_pdf": str(plan.document.source_pdf),
            "source_json": str(plan.document.source_json),
            "source_hash": plan.document.source_hash,
            "source_language": config.source_language,
            "target_language": config.target_language,
            "page_count": plan.document.page_count,
            "block_count": len(plan.document.blocks),
            "translation_unit_count": len(plan.units),
            "formula_count": len(formula_bindings),
        },
        "translation_run": {
            "mode": "simulation" if provider.name == "fake" else "provider",
            "requested_provider": requested_provider,
            "provider": provider.name,
            "model": provider.model,
            "prompt_version": config.prompt_version,
            "glossary_version": config.glossary_version,
            "started_at": result.started_at,
            "completed_at": result.completed_at,
            "network_requests": 0 if provider.name == "fake" else result.stats.provider_calls,
            "stats": result.stats.as_dict(),
        },
        "layout_status": layout_status,
        "layout_run": {
            "contract_version": LAYOUT_CONTRACT_VERSION,
            "optimizer_version": LAYOUT_OPTIMIZER_VERSION,
            "measurement_engine": golden_profile["measurement_engine"],
            "stats": planning.stats.as_dict(),
        },
        "golden_style_profile": golden_profile,
        "formula_contract": {
            "schema_version": FORMULA_CONTRACT_VERSION,
            "inventory": formula_inventory,
            "stats": {
                "total": len(formula_bindings),
                "inline": sum(binding.kind == "inline" for binding in formula_bindings),
                "display": sum(binding.kind == "display" for binding in formula_bindings),
                "protected": sum(
                    binding.protected_placeholder is not None
                    for binding in formula_bindings
                ),
                "source_geometry": sum(
                    binding.source_bbox is not None for binding in formula_bindings
                ),
                "needs_review": sum(
                    binding.render_policy == "needs_review"
                    for binding in formula_bindings
                ),
            },
        },
        "formula_assets": formula_asset_contract,
        "translation_units": unit_records,
        "items": items,
    }
    return TranslationArtifactOutput.model_validate(artifact).model_dump(mode="json")


def write_translation_artifact(
    path: str | Path,
    plan: TranslationPlan,
    *,
    provider: TranslationProvider,
    requested_provider: str,
    result: RunnerResult,
    config: RunnerConfig,
    style_profile: dict[str, Any] | None = None,
    layout_planning: LayoutPlanningResult | None = None,
    formula_asset_contract: dict[str, Any] | None = None,
) -> Path:
    target = Path(path).expanduser().resolve()
    resolved_formula_assets = formula_asset_contract or build_formula_asset_contract(plan, target)
    artifact = build_translation_artifact(
        plan,
        provider=provider,
        requested_provider=requested_provider,
        result=result,
        config=config,
        style_profile=style_profile,
        layout_planning=layout_planning,
        formula_asset_contract=resolved_formula_assets,
    )
    return write_json_atomic(target, artifact)
