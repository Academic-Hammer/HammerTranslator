"""Render measured text and vector formulas without changing translation wording."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz

from hammer_translator._hashing import sha256_file
from ..font_assets import FontAssetError, resolve_registered_font
from pydantic import ValidationError

from hammer_translator.translation.layout import (
    LayoutMetricError,
    LayoutMetrics,
    bind_formula_run_specs,
)
from hammer_translator.translation.output_schema import TranslationArtifactOutput

from .schema import REFLOW_FEEDBACK_VERSION, ReflowReportOutput
from .table import TablePlan, TableRenderJob, draw_table_job, plan_table_jobs, verify_table_job

FORMULA_BBOX_TOLERANCE_PT = 0.75
TEXT_BBOX_OVERLAP_RATIO = 0.5


class ReflowContractError(ValueError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_color(value: Any) -> tuple[float, float, float]:
    if isinstance(value, str) and re.fullmatch(r"#[0-9A-Fa-f]{6}", value):
        return tuple(int(value[index : index + 2], 16) / 255.0 for index in (1, 3, 5))
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        channels = [float(channel) for channel in value[:3]]
        if any(channel > 1 for channel in channels):
            channels = [channel / 255.0 for channel in channels]
        return tuple(max(0.0, min(1.0, channel)) for channel in channels)
    return 0.0, 0.0, 0.0


@dataclass(frozen=True)
class RenderJob:
    item_index: int
    block_id: str
    unit_id: str
    page_idx: int
    bbox: tuple[float, float, float, float]
    text: str
    role: str
    style: dict[str, Any]
    measurement: dict[str, Any]
    color: tuple[float, float, float]


@dataclass(frozen=True)
class FormulaPlacement:
    block_id: str
    page_idx: int
    formula_id: str
    asset_id: str
    asset_kind: str
    line_index: int
    run_index: int
    bbox: tuple[float, float, float, float]
    baseline_y: float
    baseline_error_pt: float
    draw_method: str


class ThinReflowRenderer:
    """Contract-first renderer: measure, draw, report; never rewrite."""

    def __init__(self, *, allow_partial: bool = False):
        self.allow_partial = allow_partial

    @staticmethod
    def _load_artifact(path: str | Path) -> tuple[Path, dict[str, Any]]:
        artifact_path = Path(path).expanduser().resolve()
        if not artifact_path.is_file():
            raise ReflowContractError(f"translation artifact not found: {artifact_path}")
        try:
            raw = json.loads(artifact_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ReflowContractError(f"invalid translation artifact JSON: {exc}") from exc
        try:
            TranslationArtifactOutput.model_validate(raw)
        except ValidationError as exc:
            first = exc.errors(include_url=False)[0]
            location = ".".join(str(part) for part in first["loc"])
            raise ReflowContractError(
                f"translation artifact contract failed at {location}: {first['msg']}"
            ) from exc
        return artifact_path, raw

    @staticmethod
    def _verify_fonts(profile: dict[str, Any]) -> None:
        for font_id, record in profile.get("fonts", {}).items():
            try:
                path = resolve_registered_font(record)
            except FontAssetError as exc:
                raise ReflowContractError(f"cannot resolve Reflow font {font_id}: {exc}") from exc
            record["path"] = str(path)

    @staticmethod
    def _verify_formula_assets(
        artifact_path: Path,
        artifact: dict[str, Any],
        source_pdf: Path,
    ) -> dict[str, dict[str, Any]]:
        contract = artifact.get("formula_assets")
        if not isinstance(contract, dict):
            return {}
        source_hash = sha256_file(source_pdf)
        if source_hash != contract.get("source_pdf_sha256"):
            raise ReflowContractError("source PDF hash no longer matches formula assets")
        records: dict[str, dict[str, Any]] = {}
        for asset in contract.get("assets") or []:
            asset_id = str(asset.get("asset_id") or "")
            formula_id = str(asset.get("formula_id") or "")
            if not asset_id or not formula_id or asset.get("status") != "ready":
                continue
            if asset.get("kind") == "source_pdf_clip":
                source = asset.get("source_pdf_clip") or {}
                if source.get("source_pdf_sha256") != source_hash:
                    raise ReflowContractError(
                        f"source formula asset hash changed: {formula_id}"
                    )
                locator = json.dumps(
                    {
                        "source_pdf_sha256": source_hash,
                        "page_idx": source.get("page_idx"),
                        "bbox": source.get("clip_bbox"),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if hashlib.sha256(locator.encode("utf-8")).hexdigest() != asset.get("sha256"):
                    raise ReflowContractError(
                        f"source formula locator changed: {formula_id}"
                    )
            elif asset.get("kind") == "mathjax_svg":
                relative = str(asset.get("relative_path") or "")
                path = (artifact_path.parent / relative).resolve()
                try:
                    path.relative_to(artifact_path.parent)
                except ValueError as exc:
                    raise ReflowContractError(
                        f"formula SVG escapes artifact directory: {formula_id}"
                    ) from exc
                if not path.is_file() or sha256_file(path) != asset.get("sha256"):
                    raise ReflowContractError(
                        f"formula SVG is missing or changed: {formula_id}"
                    )
                asset = {**asset, "resolved_path": str(path)}
            else:
                raise ReflowContractError(f"unsupported formula asset kind: {formula_id}")
            records[formula_id] = asset
        return records

    @staticmethod
    def _feedback(
        item: dict[str, Any],
        *,
        status: str,
        reason: str,
        action: str,
        role: str | None = None,
        style: dict[str, Any] | None = None,
        measurement: dict[str, Any] | None = None,
        translation_request: dict[str, Any] | None = None,
        verification: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "block_id": str(item.get("block_id") or ""),
            "translation_unit_id": str(item.get("translation_unit_id") or ""),
            "page_idx": int(item.get("page_idx") or 0),
            "bbox": [float(value) for value in (item.get("bbox") or [])],
            "status": status,
            "reason": reason,
            "action": action,
            "style_role": role,
            "recommended_style": dict(style) if style else None,
            "measurement": measurement,
            "translation_request": translation_request,
            "verification": verification,
        }

    @staticmethod
    def _has_blocking_feedback(feedback: list[dict[str, Any]]) -> bool:
        """Items preserved in place are reviewable but do not block output."""

        return any(
            item["status"] in {"needs_review", "failed"}
            and item["action"] != "preserve_original"
            for item in feedback
        )

    @staticmethod
    def _has_review_feedback(feedback: list[dict[str, Any]]) -> bool:
        return any(item["status"] in {"needs_review", "failed"} for item in feedback)

    @staticmethod
    def _measurement_matches(stored: dict[str, Any], current: dict[str, Any]) -> bool:
        exact_keys = ("fits", "is_overflow", "line_count", "font_id")
        if any(stored.get(key) != current.get(key) for key in exact_keys):
            return False
        numeric_keys = (
            "available_width_pt",
            "available_height_pt",
            "used_width_pt",
            "used_height_pt",
            "font_size_pt",
            "line_height_multiplier",
            "char_spacing_em",
        )
        if not all(
            math.isclose(float(stored.get(key, 0)), float(current.get(key, 0)), abs_tol=0.02)
            for key in numeric_keys
        ):
            return False
        stored_mixed = stored.get("mixed_lines") or []
        current_mixed = current.get("mixed_lines") or []
        if len(stored_mixed) != len(current_mixed):
            return False
        for stored_line, current_line in zip(stored_mixed, current_mixed, strict=True):
            if len(stored_line.get("runs") or []) != len(current_line.get("runs") or []):
                return False
            for stored_run, current_run in zip(
                stored_line.get("runs") or [], current_line.get("runs") or [], strict=True
            ):
                if any(
                    stored_run.get(key) != current_run.get(key)
                    for key in ("kind", "text", "formula_id", "asset_id", "asset_kind")
                ):
                    return False
                if any(
                    not math.isclose(
                        float(stored_run.get(key) or 0),
                        float(current_run.get(key) or 0),
                        abs_tol=0.02,
                    )
                    for key in (
                        "x_offset_pt",
                        "width_pt",
                        "height_pt",
                        "baseline_from_top_pt",
                        "asset_scale_to_pt",
                    )
                ):
                    return False
        return True

    @staticmethod
    def _formula_runs(
        artifact: dict[str, Any],
        unit: dict[str, Any],
        item: dict[str, Any],
        text: str,
    ):
        formula_ids = set(item.get("formula_ids") or [])
        if not formula_ids:
            return ()
        contract = artifact.get("formula_contract") or {}
        assets = artifact.get("formula_assets") or {}
        records = [
            record
            for record in contract.get("inventory") or []
            if record.get("formula_id") in formula_ids
        ]
        return bind_formula_run_specs(
            text,
            records,
            list((unit.get("translation") or {}).get("protected_tokens") or []),
            list(assets.get("assets") or []),
        )

    @staticmethod
    def _unavailable_movable_formula_ids(
        artifact: dict[str, Any],
        item: dict[str, Any],
    ) -> tuple[str, ...]:
        """Return inline formulas that cannot be drawn as part of the text.

        Display formulas and review-only source formulas stay in place, so a
        failed asset for those does not prevent the surrounding text from
        being reflowed.
        """

        formula_ids = set(item.get("formula_ids") or [])
        if not formula_ids:
            return ()
        records = {
            str(record.get("formula_id") or ""): record
            for record in (artifact.get("formula_contract") or {}).get("inventory") or []
        }
        assets = {
            str(record.get("formula_id") or ""): record
            for record in (artifact.get("formula_assets") or {}).get("assets") or []
        }
        unavailable: list[str] = []
        for formula_id in sorted(formula_ids):
            record = records.get(formula_id)
            if not isinstance(record, dict):
                continue
            if record.get("kind") != "inline" or record.get("render_policy") != "move_with_text":
                continue
            asset = assets.get(formula_id)
            if not isinstance(asset, dict) or asset.get("status") != "ready":
                unavailable.append(formula_id)
        return tuple(unavailable)

    @staticmethod
    def _translation_request(unit: dict[str, Any], measurement: dict[str, Any]) -> dict[str, Any]:
        layout = unit.get("layout") or {}
        constraints = dict(layout.get("retranslation_constraints") or {})
        constraints.update(
            reason="reflow_measurement_overflow",
            overflow_width_pt=measurement.get("overflow_width_pt", 0.0),
            overflow_height_pt=measurement.get("overflow_height_pt", 0.0),
            bbox_locked=True,
        )
        return {
            "translation_unit_id": unit.get("translation_unit_id"),
            "selected_version": (unit.get("translation") or {}).get("selected_version"),
            "constraints": constraints,
        }

    _TEXT_CONTAINER_TYPES = frozenset(
        {
            "text",
            "abstract",
            "figure_caption",
            "table_caption",
            "reference",
            "page_footnote",
            "title",
            "author",
        }
    )
    _NON_TEXT_BLOCK_TYPES = frozenset(
        {"image", "table", "figure", "equation", "page_number", "header"}
    )

    @staticmethod
    def _source_tokens(text: Any) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", str(text or "").lower()))

    @staticmethod
    def _containment_ratio(inner: Any, outer: Any) -> float:
        inner_rect = fitz.Rect(inner)
        area = inner_rect.get_area()
        if area <= 0:
            return 0.0
        overlap = inner_rect & fitz.Rect(outer)
        return max(0.0, overlap.get_area()) / area

    @classmethod
    def _merged_container_candidate(cls, item: dict[str, Any]) -> bool:
        """Return whether an item can represent several merged duplicate blocks."""

        if str(item.get("detected_type") or "").lower() not in cls._TEXT_CONTAINER_TYPES:
            return False
        if str(item.get("type") or "").lower() in cls._NON_TEXT_BLOCK_TYPES:
            return False
        if item.get("translation_status") in {"preserved", "skipped"}:
            return False
        if not isinstance(item.get("translated"), str) or not item["translated"].strip():
            return False
        if not isinstance(item.get("recommended_style"), dict):
            return False
        if not (item.get("translation") or {}).get("style_role"):
            return False
        if len(cls._source_tokens(item.get("text"))) < 40:
            return False
        measurement = (item.get("layout_preflight") or {}).get("measurement")
        return isinstance(measurement, dict) and measurement.get("fits") is True

    @classmethod
    def _dedup_nested_candidate(cls, item: dict[str, Any]) -> bool:
        if str(item.get("detected_type") or "").lower() in cls._NON_TEXT_BLOCK_TYPES:
            return False
        if str(item.get("type") or "").lower() in cls._NON_TEXT_BLOCK_TYPES:
            return False
        if item.get("translation_status") in {"preserved", "skipped"}:
            return False
        if not isinstance(item.get("translated"), str) or not item["translated"].strip():
            return False
        return len(cls._source_tokens(item.get("text"))) >= 20

    @classmethod
    def _overlapping_container_groups(
        cls, artifact: dict[str, Any]
    ) -> dict[str, frozenset[str]]:
        """Find merged blocks that repeat text already covered by a larger block.

        MinerU occasionally emits one oversized block (often a figure caption
        merged with the surrounding columns) while also emitting the same
        paragraphs as separate blocks.  Reflowing both copies paints the same
        region twice.  Keep the larger block, whose translation contains the
        complete text, and skip the covered duplicates.
        """

        pages: dict[int, list[dict[str, Any]]] = {}
        for item in artifact.get("items") or []:
            if isinstance(item, dict):
                pages.setdefault(int(item.get("page_idx") or 0), []).append(item)

        groups: dict[str, set[str]] = {}
        for page_items in pages.values():
            containers = [
                item
                for item in page_items
                if cls._merged_container_candidate(item)
                and not cls._unavailable_movable_formula_ids(artifact, item)
            ]
            if not containers:
                continue
            parent_of: dict[str, str] = {}
            for nested in page_items:
                if not cls._dedup_nested_candidate(nested):
                    continue
                nested_id = str(nested.get("block_id") or "")
                nested_tokens = cls._source_tokens(nested.get("text"))
                if not nested_id or len(nested_tokens) < 20:
                    continue
                matches: list[dict[str, Any]] = []
                for container in containers:
                    container_id = str(container.get("block_id") or "")
                    if not container_id or container_id == nested_id:
                        continue
                    container_rect = fitz.Rect(container.get("bbox") or ())
                    nested_rect = fitz.Rect(nested.get("bbox") or ())
                    if container_rect.get_area() < nested_rect.get_area() * 1.5:
                        continue
                    if cls._containment_ratio(nested_rect, container_rect) < 0.9:
                        continue
                    container_tokens = cls._source_tokens(container.get("text"))
                    coverage = len(nested_tokens & container_tokens) / len(nested_tokens)
                    if coverage >= 0.9:
                        matches.append(container)
                if matches:
                    tightest = min(
                        matches,
                        key=lambda candidate: fitz.Rect(candidate.get("bbox") or ()).get_area(),
                    )
                    parent_of[nested_id] = str(tightest.get("block_id") or "")

            for nested_id in parent_of:
                current = nested_id
                visited: set[str] = set()
                while current in parent_of and current not in visited:
                    visited.add(current)
                    current = parent_of[current]
                if current != nested_id:
                    groups.setdefault(current, set()).add(nested_id)
        return {container_id: frozenset(nested) for container_id, nested in groups.items()}

    @staticmethod
    def _overlapping_job_block_ids(jobs: list[RenderJob]) -> dict[str, set[str]]:
        """Return text jobs that would paint over most of another text job."""

        by_page: dict[int, list[RenderJob]] = {}
        for job in jobs:
            by_page.setdefault(job.page_idx, []).append(job)
        conflicts: dict[str, set[str]] = {}
        for page_jobs in by_page.values():
            for index, first in enumerate(page_jobs):
                first_rect = fitz.Rect(first.bbox)
                first_area = first_rect.get_area()
                for second in page_jobs[index + 1 :]:
                    second_rect = fitz.Rect(second.bbox)
                    second_area = second_rect.get_area()
                    smaller = min(first_area, second_area)
                    if smaller <= 0:
                        continue
                    overlap = (first_rect & second_rect).get_area()
                    if overlap / smaller < TEXT_BBOX_OVERLAP_RATIO:
                        continue
                    conflicts.setdefault(first.block_id, set()).add(second.block_id)
                    conflicts.setdefault(second.block_id, set()).add(first.block_id)
        return conflicts

    def _preflight(
        self,
        artifact: dict[str, Any],
        metrics: LayoutMetrics,
        table_plan: TablePlan | None = None,
    ) -> tuple[list[RenderJob], list[dict[str, Any]]]:
        table_plan = table_plan or TablePlan((), frozenset(), {})
        table_jobs = {job.block_id: job for job in table_plan.jobs}
        merged_groups = self._overlapping_container_groups(artifact)
        units = {unit["translation_unit_id"]: unit for unit in artifact["translation_units"]}
        jobs: list[RenderJob] = []
        feedback: list[dict[str, Any]] = []
        for item_index, item in enumerate(artifact["items"]):
            unit = units[item["translation_unit_id"]]
            translation_status = item.get("translation_status")
            block_id = str(item.get("block_id") or "")
            if block_id in table_plan.covered_block_ids:
                feedback.append(
                    self._feedback(
                        item,
                        status="rendered",
                        reason="covered_by_structured_table_reflow",
                        action="none",
                        verification={
                            "text_layer_nonempty": True,
                            "bbox_unchanged": True,
                            "page_size_unchanged": True,
                            "covered_by_table_reflow": True,
                        },
                    )
                )
                continue
            if block_id in table_jobs:
                table_job = table_jobs[block_id]
                feedback.append(
                    self._feedback(
                        item,
                        status="rendered",
                        reason="structured_table_reflow_confirmed",
                        action="none",
                        role="table",
                        style=item.get("recommended_style"),
                        measurement={
                            "table_regions": len(table_job.regions),
                            "table_cells": len(table_job.cells),
                            "font_size_pt": table_job.cells[0].font_size if table_job.cells else None,
                        },
                    )
                )
                continue
            if block_id in table_plan.errors:
                table_error = table_plan.errors[block_id]
                feedback.append(
                    self._feedback(
                        item,
                        status="needs_review",
                        reason=f"table_reflow_unavailable:{table_error}",
                        action="preserve_original",
                        measurement={"table_error": table_error},
                    )
                )
                continue
            if translation_status == "preserved":
                feedback.append(
                    self._feedback(
                        item,
                        status="preserved",
                        reason="translation_policy_preserves_original_content",
                        action="preserve_original",
                    )
                )
                continue
            if translation_status == "skipped" or item.get("type", "").lower() in {
                "image",
                "table",
                "figure",
            }:
                feedback.append(
                    self._feedback(
                        item,
                        status="skipped",
                        reason="element_is_outside_text_reflow",
                        action="preserve_original",
                    )
                )
                continue

            text = item.get("translated")
            role = (item.get("translation") or {}).get("style_role")
            style = item.get("recommended_style")
            stored_preflight = item.get("layout_preflight") or {}
            if not isinstance(text, str) or not text or not isinstance(style, dict) or not role:
                feedback.append(
                    self._feedback(
                        item,
                        status="needs_review",
                        reason="approved_element_translation_or_style_is_missing",
                        action="manual_review",
                        role=role,
                        style=style if isinstance(style, dict) else None,
                    )
                )
                continue
            unavailable_formulas = self._unavailable_movable_formula_ids(artifact, item)
            if unavailable_formulas:
                feedback.append(
                    self._feedback(
                        item,
                        status="needs_review",
                        reason=(
                            "formula_asset_unavailable:"
                            + ",".join(unavailable_formulas)
                        ),
                        action="preserve_original",
                        role=role,
                        style=style,
                        measurement={"formula_ids": list(unavailable_formulas)},
                    )
                )
                continue
            try:
                formula_runs = self._formula_runs(artifact, unit, item, text)
                measured = metrics.measure(
                    text,
                    item["bbox"],
                    role=role,
                    style=style,
                    enforce_guardrails=False,
                    formula_runs=formula_runs,
                ).as_dict()
            except (LayoutMetricError, KeyError, TypeError, ValueError) as exc:
                feedback.append(
                    self._feedback(
                        item,
                        status="failed",
                        reason=f"reflow_measurement_failed:{type(exc).__name__}",
                        action="manual_review",
                        role=role,
                        style=style,
                    )
                )
                continue

            stored_measurement = stored_preflight.get("measurement")
            if not isinstance(stored_measurement, dict) or not self._measurement_matches(
                stored_measurement, measured
            ):
                feedback.append(
                    self._feedback(
                        item,
                        status="failed",
                        reason="translation_and_reflow_measurements_do_not_match",
                        action="manual_review",
                        role=role,
                        style=style,
                        measurement=measured,
                    )
                )
                continue
            if not measured["fits"]:
                constraints = (unit.get("layout") or {}).get("retranslation_constraints") or {}
                action = (
                    "request_constrained_retranslation"
                    if constraints.get("retranslation_possible", True)
                    else "manual_review"
                )
                feedback.append(
                    self._feedback(
                        item,
                        status="needs_review",
                        reason="fixed_bbox_overflow_confirmed_by_reflow",
                        action=action,
                        role=role,
                        style=style,
                        measurement=measured,
                        translation_request=(
                            self._translation_request(unit, measured)
                            if action == "request_constrained_retranslation"
                            else None
                        ),
                    )
                )
                continue

            color = _parse_color(
                artifact["golden_style_profile"]["roles"][role]["source_statistics"].get(
                    "dominant_color"
                )
            )
            jobs.append(
                RenderJob(
                    item_index=item_index,
                    block_id=item["block_id"],
                    unit_id=item["translation_unit_id"],
                    page_idx=int(item["page_idx"]),
                    bbox=tuple(float(value) for value in item["bbox"]),
                    text=text,
                    role=role,
                    style=dict(style),
                    measurement=measured,
                    color=color,
                )
            )
            feedback.append(
                self._feedback(
                    item,
                    status="rendered",
                    reason=(
                        "shared_mixed_measurement_confirmed"
                        if formula_runs
                        else "shared_measurement_confirmed"
                    ),
                    action="none",
                    role=role,
                    style=style,
                    measurement=measured,
                )
            )
        rendered_block_ids = {job.block_id for job in jobs}
        items_by_block = {
            str(item.get("block_id") or ""): item for item in artifact["items"]
        }
        feedback_by_block = {entry["block_id"]: entry for entry in feedback}
        removed_block_ids: set[str] = set()
        for container_id, nested_ids in merged_groups.items():
            if container_id not in rendered_block_ids:
                continue
            container_removed: set[str] = set()
            for nested_id in nested_ids:
                nested_item = items_by_block.get(nested_id)
                if nested_item is None or nested_id in table_plan.covered_block_ids:
                    continue
                replacement = self._feedback(
                    nested_item,
                    status="skipped",
                    reason="merged_into_overlapping_container_block",
                    action="none",
                    role=(nested_item.get("translation") or {}).get("style_role"),
                    style=nested_item.get("recommended_style"),
                    verification={
                        "covered_by_container_block": True,
                        "container_block_id": container_id,
                    },
                )
                existing = feedback_by_block.get(nested_id)
                if existing is None:
                    feedback.append(replacement)
                    feedback_by_block[nested_id] = replacement
                else:
                    existing.clear()
                    existing.update(replacement)
                removed_block_ids.add(nested_id)
                container_removed.add(nested_id)
            container_feedback = feedback_by_block.get(container_id)
            if container_removed and container_feedback is not None:
                container_feedback["verification"] = {
                    **(container_feedback.get("verification") or {}),
                    "merged_duplicate_block_ids": sorted(container_removed),
                }
        if removed_block_ids:
            jobs = [job for job in jobs if job.block_id not in removed_block_ids]
        conflicts = self._overlapping_job_block_ids(jobs)
        if conflicts:
            for entry in feedback:
                peers = conflicts.get(entry["block_id"])
                if not peers or entry["status"] != "rendered":
                    continue
                entry["status"] = "needs_review"
                entry["reason"] = "overlapping_text_bboxes"
                entry["action"] = "manual_review"
                entry["measurement"] = {
                    **(entry.get("measurement") or {}),
                    "overlapping_block_ids": sorted(peers),
                }
                entry["translation_request"] = None
            jobs = [job for job in jobs if job.block_id not in conflicts]
        return jobs, feedback

    @staticmethod
    def _line_start_x(rect: fitz.Rect, width: float, alignment: str) -> float:
        if alignment == "center":
            return rect.x0 + max(0.0, (rect.width - width) / 2)
        if alignment == "right":
            return rect.x1 - width
        return rect.x0

    @classmethod
    def _formula_placements(cls, job: RenderJob) -> list[FormulaPlacement]:
        lines = job.measurement.get("mixed_lines") or []
        if not lines:
            return []
        rect = fitz.Rect(job.bbox)
        alignment = str(job.style.get("alignment") or "left")
        y = rect.y0
        placements: list[FormulaPlacement] = []
        for line_index, line in enumerate(lines):
            line_width = float(line["width_pt"])
            line_baseline = y + float(line["baseline_from_top_pt"])
            start_x = cls._line_start_x(rect, line_width, alignment)
            for run_index, run in enumerate(line.get("runs") or []):
                if run.get("kind") != "formula":
                    continue
                x0 = start_x + float(run["x_offset_pt"])
                y0 = line_baseline - float(run["baseline_from_top_pt"])
                x1 = x0 + float(run["width_pt"])
                y1 = y0 + float(run["height_pt"])
                expected_baseline = y0 + float(run["baseline_from_top_pt"])
                asset_kind = str(run["asset_kind"])
                placements.append(
                    FormulaPlacement(
                        block_id=job.block_id,
                        page_idx=job.page_idx,
                        formula_id=str(run["formula_id"]),
                        asset_id=str(run["asset_id"]),
                        asset_kind=asset_kind,
                        line_index=line_index,
                        run_index=run_index,
                        bbox=(x0, y0, x1, y1),
                        baseline_y=line_baseline,
                        baseline_error_pt=abs(expected_baseline - line_baseline),
                        draw_method=(
                            "show_pdf_page_source_clip"
                            if asset_kind == "source_pdf_clip"
                            else "show_pdf_page_svg_vector"
                        ),
                    )
                )
            y += float(line["height_pt"])
        return placements

    @classmethod
    def _draw_text_job(
        cls,
        writer: fitz.TextWriter,
        job: RenderJob,
        profile: dict[str, Any],
        font_cache: dict[str, fitz.Font],
    ) -> None:
        font_record = profile["fonts"][job.style["font_id"]]
        # _verify_fonts resolves and verifies each record once before rendering.
        font_path = str(Path(font_record["path"]).resolve())
        font = font_cache.get(job.style["font_id"])
        if font is None:
            font = fitz.Font(fontfile=font_path)
            font_cache[job.style["font_id"]] = font
        font_size = float(job.style["font_size_pt"])
        line_height = font_size * float(job.style["line_height_multiplier"])
        char_spacing = font_size * float(job.style["char_spacing_em"])
        rect = fitz.Rect(job.bbox)
        alignment = str(job.style.get("alignment") or "left")

        mixed_lines = job.measurement.get("mixed_lines") or []
        if mixed_lines:
            y = rect.y0
            for line in mixed_lines:
                start_x = cls._line_start_x(rect, float(line["width_pt"]), alignment)
                baseline = y + float(line["baseline_from_top_pt"])
                for run in line.get("runs") or []:
                    if run.get("kind") != "text":
                        continue
                    run_x = start_x + float(run["x_offset_pt"])
                    prefix = ""
                    for char_index, character in enumerate(str(run.get("text") or "")):
                        x = run_x + float(font.text_length(prefix, fontsize=font_size))
                        x += char_index * char_spacing
                        if not character.isspace():
                            writer.append(
                                (x, baseline),
                                character,
                                font=font,
                                fontsize=font_size,
                            )
                        prefix += character
                y += float(line["height_pt"])
            return

        for line_index, (line, measured_width) in enumerate(
            zip(job.measurement["lines"], job.measurement["line_widths_pt"], strict=True)
        ):
            start_x = cls._line_start_x(rect, measured_width, alignment)
            baseline = (
                rect.y0
                + line_index * line_height
                + max(0.0, (line_height - font_size) / 2)
                + float(font.ascender) * font_size
            )
            prefix = ""
            for char_index, character in enumerate(line):
                x = start_x + float(font.text_length(prefix, fontsize=font_size))
                x += char_index * char_spacing
                if not character.isspace():
                    writer.append(
                        (x, baseline),
                        character,
                        font=font,
                        fontsize=font_size,
                    )
                prefix += character

    @staticmethod
    def _draw_formula(
        page: fitz.Page,
        placement: FormulaPlacement,
        asset: dict[str, Any],
        source_formula_document: fitz.Document,
        svg_documents: dict[str, fitz.Document],
    ) -> int:
        target = fitz.Rect(placement.bbox)
        if placement.asset_kind == "source_pdf_clip":
            source = asset.get("source_pdf_clip") or {}
            clip = fitz.Rect(source.get("clip_bbox") or [])
            page_idx = int(source.get("page_idx"))
            return int(
                page.show_pdf_page(
                    target,
                    source_formula_document,
                    page_idx,
                    clip=clip,
                    keep_proportion=False,
                    overlay=True,
                )
            )
        if placement.asset_kind != "mathjax_svg":
            raise ReflowContractError(
                f"unsupported formula draw asset: {placement.formula_id}"
            )
        vector_document = svg_documents.get(placement.asset_id)
        if vector_document is None:
            svg_bytes = Path(str(asset["resolved_path"])).read_bytes()
            svg_document = fitz.open(stream=svg_bytes, filetype="svg")
            try:
                pdf_bytes = svg_document.convert_to_pdf()
            finally:
                svg_document.close()
            vector_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            svg_documents[placement.asset_id] = vector_document
        return int(
            page.show_pdf_page(
                target,
                vector_document,
                0,
                keep_proportion=False,
                overlay=True,
            )
        )

    @staticmethod
    def _atomic_save(document: fitz.Document, output: Path) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp.pdf",
            delete=False,
        )
        temp_path = Path(handle.name)
        handle.close()
        try:
            # TextWriter embeds complete font programs. Subset them only after all
            # text has been written so the PDF keeps just the glyphs it uses.
            document.subset_fonts(verbose=False, fallback=False)
            document.save(temp_path, garbage=4, deflate=True)
            os.replace(temp_path, output)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    @staticmethod
    def _verify_output(
        output: Path,
        source_pdf: Path,
        jobs: list[RenderJob],
        placements: list[FormulaPlacement],
        drawn_formula_ids: list[str],
        table_jobs: list[TableRenderJob] | None = None,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        source = fitz.open(source_pdf)
        rendered = fitz.open(output)
        try:
            if rendered.page_count != source.page_count:
                raise ReflowContractError("rendered PDF page count changed")
            for index in range(source.page_count):
                if rendered[index].rect != source[index].rect:
                    raise ReflowContractError(f"rendered PDF page {index} size changed")
            results: dict[str, dict[str, Any]] = {}
            placements_by_block: dict[str, list[FormulaPlacement]] = {}
            for placement in placements:
                placements_by_block.setdefault(placement.block_id, []).append(placement)
            for job in jobs:
                page = rendered[job.page_idx]
                extracted = page.get_text("text", clip=fitz.Rect(job.bbox)).strip()
                mixed_lines = job.measurement.get("mixed_lines") or []
                text_expected = (
                    any(
                        run.get("kind") == "text" and str(run.get("text") or "").strip()
                        for line in mixed_lines
                        for run in line.get("runs") or []
                    )
                    if mixed_lines
                    else bool(job.text.strip())
                )
                job_placements = placements_by_block.get(job.block_id, [])
                formula_boxes = [fitz.Rect(placement.bbox) for placement in job_placements]
                overlap_count = sum(
                    1
                    for index, first in enumerate(formula_boxes)
                    for second in formula_boxes[index + 1 :]
                    if (first & second).get_area() > 0.01
                )
                locked = fitz.Rect(job.bbox)
                inside = all(
                    box.x0 >= locked.x0 - FORMULA_BBOX_TOLERANCE_PT
                    and box.y0 >= locked.y0 - FORMULA_BBOX_TOLERANCE_PT
                    and box.x1 <= locked.x1 + FORMULA_BBOX_TOLERANCE_PT
                    and box.y1 <= locked.y1 + FORMULA_BBOX_TOLERANCE_PT
                    for box in formula_boxes
                )
                ink_checks: list[bool] = []
                for box in formula_boxes:
                    pixmap = page.get_pixmap(
                        matrix=fitz.Matrix(2, 2),
                        clip=box,
                        colorspace=fitz.csGRAY,
                        alpha=False,
                    )
                    ink_checks.append(any(sample < 245 for sample in pixmap.samples))
                results[job.block_id] = {
                    "text_layer_expected": text_expected,
                    "text_layer_nonempty": bool(extracted) if text_expected else True,
                    "extracted_text": extracted[:500],
                    "bbox_unchanged": True,
                    "page_size_unchanged": True,
                    "formula_expected_count": len(job_placements),
                    "formula_drawn_count": sum(
                        placement.formula_id in drawn_formula_ids
                        for placement in job_placements
                    ),
                    "formula_ids": [placement.formula_id for placement in job_placements],
                    "formula_order_verified": [
                        placement.formula_id for placement in job_placements
                    ]
                    == [
                        formula_id
                        for formula_id in drawn_formula_ids
                        if formula_id
                        in {placement.formula_id for placement in job_placements}
                    ],
                    "formula_bbox_inside_locked_bbox": inside,
                    "formula_baseline_max_error_pt": round(
                        max(
                            (placement.baseline_error_pt for placement in job_placements),
                            default=0.0,
                        ),
                        6,
                    ),
                    "formula_overlap_count": overlap_count,
                    "formula_target_nonempty": all(ink_checks),
                    "formula_vector_only": all(
                        placement.draw_method.startswith("show_pdf_page_")
                        for placement in job_placements
                    ),
                    "formula_draw_methods": [
                        placement.draw_method for placement in job_placements
                    ],
                }
            for table_job in table_jobs or []:
                results[table_job.block_id] = verify_table_job(rendered, table_job)
            expected_ids = [placement.formula_id for placement in placements]
            formula_summary = {
                "expected": len(expected_ids),
                "drawn": len(drawn_formula_ids),
                "source_pdf_clip": sum(
                    placement.asset_kind == "source_pdf_clip" for placement in placements
                ),
                "mathjax_svg": sum(
                    placement.asset_kind == "mathjax_svg" for placement in placements
                ),
                "order_verified": expected_ids == drawn_formula_ids,
                "bbox_verified": all(
                    result.get("formula_bbox_inside_locked_bbox", True)
                    for result in results.values()
                ),
                "baseline_verified": all(
                    result.get("formula_baseline_max_error_pt", 0.0) <= 0.02
                    for result in results.values()
                ),
                "overlap_count": sum(
                    result.get("formula_overlap_count", 0) for result in results.values()
                ),
                "nonempty_verified": all(
                    result.get("formula_target_nonempty", True) for result in results.values()
                ),
                "vector_only": all(
                    result.get("formula_vector_only", True) for result in results.values()
                ),
            }
            return results, formula_summary
        finally:
            rendered.close()
            source.close()

    def render(
        self,
        artifact_path: str | Path,
        output_pdf: str | Path,
    ) -> dict[str, Any]:
        source_artifact, artifact = self._load_artifact(artifact_path)
        if artifact["layout_run"]["contract_version"] not in {
            "translation-reflow-v1",
            "translation-reflow-v2",
        }:
            raise ReflowContractError("unsupported Translation/Reflow contract")
        profile = artifact["golden_style_profile"]
        self._verify_fonts(profile)
        metrics = LayoutMetrics(profile)
        source_pdf = Path(artifact["document"]["source_pdf"]).expanduser().resolve()
        if not source_pdf.is_file():
            raise ReflowContractError(f"source PDF not found: {source_pdf}")
        output = Path(output_pdf).expanduser().resolve()
        if output == source_pdf:
            raise ReflowContractError("Reflow output must not overwrite the source PDF")
        formula_assets = self._verify_formula_assets(source_artifact, artifact, source_pdf)
        table_plan = plan_table_jobs(artifact, source_pdf, profile)

        jobs, feedback = self._preflight(artifact, metrics, table_plan=table_plan)
        placements = [placement for job in jobs for placement in self._formula_placements(job)]
        missing_assets = sorted(
            placement.formula_id
            for placement in placements
            if placement.formula_id not in formula_assets
        )
        if missing_assets:
            missing_set = set(missing_assets)
            affected_blocks = {
                placement.block_id
                for placement in placements
                if placement.formula_id in missing_set
            }
            for item in feedback:
                if item["block_id"] in affected_blocks:
                    item["status"] = "needs_review"
                    item["reason"] = (
                        "formula_asset_missing_at_draw:"
                        + ",".join(sorted(missing_set))
                    )
                    item["action"] = "preserve_original"
                    item["measurement"] = {"formula_ids": sorted(missing_set)}
                    item["translation_request"] = None
            jobs = [job for job in jobs if job.block_id not in affected_blocks]
            placements = [
                placement
                for placement in placements
                if placement.formula_id not in missing_set
            ]
        blocking = self._has_blocking_feedback(feedback)
        has_degraded_preserve = any(
            item["status"] == "needs_review"
            and item["action"] == "preserve_original"
            for item in feedback
        )
        output_value: str | None = None
        status = "blocked"
        formula_verification = {
            "expected": 0,
            "drawn": 0,
            "source_pdf_clip": 0,
            "mathjax_svg": 0,
            "order_verified": True,
            "bbox_verified": True,
            "baseline_verified": True,
            "overlap_count": 0,
            "nonempty_verified": True,
            "vector_only": True,
        }
        if (jobs or table_plan.jobs or has_degraded_preserve) and (
            self.allow_partial or not blocking
        ):
            document = fitz.open(source_pdf)
            source_formula_document = fitz.open(source_pdf)
            svg_documents: dict[str, fitz.Document] = {}
            drawn_formula_ids: list[str] = []
            try:
                table_pages: set[int] = set()
                for table_job in table_plan.jobs:
                    for page_idx, region in table_job.regions:
                        document[page_idx].add_redact_annot(
                            fitz.Rect(region), fill=(1, 1, 1), cross_out=False
                        )
                        table_pages.add(page_idx)
                for page_index in sorted(table_pages):
                    document[page_index].apply_redactions(
                        images=fitz.PDF_REDACT_IMAGE_NONE,
                        graphics=fitz.PDF_REDACT_LINE_ART_REMOVE_IF_COVERED,
                        text=fitz.PDF_REDACT_TEXT_REMOVE,
                    )
                pages_with_redactions: set[int] = set()
                for job in jobs:
                    document[job.page_idx].add_redact_annot(
                        fitz.Rect(job.bbox), fill=None, cross_out=False
                    )
                    pages_with_redactions.add(job.page_idx)
                for page_index in sorted(pages_with_redactions):
                    document[page_index].apply_redactions(
                        images=fitz.PDF_REDACT_IMAGE_NONE,
                        graphics=fitz.PDF_REDACT_LINE_ART_NONE,
                        text=fitz.PDF_REDACT_TEXT_REMOVE,
                    )
                formulas = {
                    record["formula_id"]: record
                    for record in (artifact.get("formula_contract") or {}).get("inventory", [])
                }
                formula_redaction_pages: set[int] = set()
                for placement in placements:
                    formula = formulas[placement.formula_id]
                    source_bbox = formula.get("source_bbox")
                    source_page = formula.get("page_idx")
                    if source_bbox is None or source_page is None:
                        continue
                    document[int(source_page)].add_redact_annot(
                        fitz.Rect(source_bbox), fill=None, cross_out=False
                    )
                    formula_redaction_pages.add(int(source_page))
                for page_index in sorted(formula_redaction_pages):
                    document[page_index].apply_redactions(
                        images=fitz.PDF_REDACT_IMAGE_NONE,
                        graphics=fitz.PDF_REDACT_LINE_ART_REMOVE_IF_COVERED,
                        text=fitz.PDF_REDACT_TEXT_REMOVE,
                    )
                writers: dict[tuple[int, tuple[float, float, float]], fitz.TextWriter] = {}
                font_cache: dict[str, fitz.Font] = {}
                for job in jobs:
                    key = (job.page_idx, job.color)
                    writer = writers.get(key)
                    if writer is None:
                        writer = fitz.TextWriter(document[job.page_idx].rect, color=job.color)
                        writers[key] = writer
                    self._draw_text_job(writer, job, profile, font_cache)
                for placement in placements:
                    xref = self._draw_formula(
                        document[placement.page_idx],
                        placement,
                        formula_assets[placement.formula_id],
                        source_formula_document,
                        svg_documents,
                    )
                    if xref <= 0:
                        raise ReflowContractError(
                            f"formula vector insertion failed: {placement.formula_id}"
                        )
                    drawn_formula_ids.append(placement.formula_id)
                for table_job in table_plan.jobs:
                    table_font = font_cache.get(table_job.font_id)
                    if table_font is None:
                        table_font = fitz.Font(
                            fontfile=str(Path(profile["fonts"][table_job.font_id]["path"]).resolve())
                        )
                        font_cache[table_job.font_id] = table_font
                    draw_table_job(document, table_job, table_font)
                for (page_index, color), writer in writers.items():
                    writer.write_text(document[page_index], color=color, overlay=1)
                self._atomic_save(document, output)
            finally:
                for vector_document in svg_documents.values():
                    vector_document.close()
                source_formula_document.close()
                document.close()
            try:
                verification, formula_verification = self._verify_output(
                    output,
                    source_pdf,
                    jobs,
                    placements,
                    drawn_formula_ids,
                    table_plan.jobs,
                )
                formula_checks = (
                    formula_verification["expected"] == formula_verification["drawn"],
                    formula_verification["order_verified"],
                    formula_verification["bbox_verified"],
                    formula_verification["baseline_verified"],
                    formula_verification["overlap_count"] == 0,
                    formula_verification["nonempty_verified"],
                    formula_verification["vector_only"],
                )
                if not all(formula_checks):
                    raise ReflowContractError("formula output verification failed")
            except Exception:
                if output.exists():
                    output.unlink()
                raise
            for item in feedback:
                if item["block_id"] in verification:
                    item["verification"] = {
                        **(item.get("verification") or {}),
                        **verification[item["block_id"]],
                    }
                    if not verification[item["block_id"]]["text_layer_nonempty"]:
                        item["status"] = "failed"
                        item["reason"] = "rendered_text_layer_is_empty"
                        item["action"] = "manual_review"
            status = (
                "partial_needs_review"
                if self._has_review_feedback(feedback)
                else "rendered"
            )
            output_value = str(output)

        counts = Counter(item["status"] for item in feedback)
        report = {
            "schema_version": REFLOW_FEEDBACK_VERSION,
            "contract_version": artifact["layout_run"]["contract_version"],
            "optimizer_version": artifact["layout_run"]["optimizer_version"],
            "measurement_engine": artifact["layout_run"]["measurement_engine"],
            "created_at": _utc_now(),
            "source_translation_artifact": str(source_artifact),
            "source_pdf": str(source_pdf),
            "output_pdf": output_value,
            "status": status,
            "allow_partial": self.allow_partial,
            "summary": {
                "items_total": len(feedback),
                "rendered": counts["rendered"],
                "preserved": counts["preserved"],
                "skipped": counts["skipped"],
                "needs_review": counts["needs_review"],
                "failed": counts["failed"],
            },
            "formula_verification": formula_verification,
            "items": feedback,
        }
        return ReflowReportOutput.model_validate(report).model_dump(mode="json")
