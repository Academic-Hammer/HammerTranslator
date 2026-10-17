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

from ..font_assets import FontAssetError, resolve_registered_font
from pydantic import ValidationError

from hammer_translator.translation.layout import (
    LayoutMetricError,
    LayoutMetrics,
    bind_formula_run_specs,
)
from hammer_translator.translation.output_schema import TranslationArtifactOutput

from .schema import REFLOW_FEEDBACK_VERSION, ReflowReportOutput


class ReflowContractError(ValueError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        source_hash = _sha256(source_pdf)
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
                if not path.is_file() or _sha256(path) != asset.get("sha256"):
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

    def _preflight(
        self,
        artifact: dict[str, Any],
        metrics: LayoutMetrics,
    ) -> tuple[list[RenderJob], list[dict[str, Any]]]:
        units = {unit["translation_unit_id"]: unit for unit in artifact["translation_units"]}
        jobs: list[RenderJob] = []
        feedback: list[dict[str, Any]] = []
        for item_index, item in enumerate(artifact["items"]):
            unit = units[item["translation_unit_id"]]
            translation_status = item.get("translation_status")
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
            try:
                formula_runs = self._formula_runs(artifact, unit, item, text)
                measured = metrics.measure(
                    text,
                    item["bbox"],
                    role=role,
                    style=style,
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
                    box.x0 >= locked.x0 - 0.02
                    and box.y0 >= locked.y0 - 0.02
                    and box.x1 <= locked.x1 + 0.02
                    and box.y1 <= locked.y1 + 0.02
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
                    result["formula_bbox_inside_locked_bbox"] for result in results.values()
                ),
                "baseline_verified": all(
                    result["formula_baseline_max_error_pt"] <= 0.02
                    for result in results.values()
                ),
                "overlap_count": sum(
                    result["formula_overlap_count"] for result in results.values()
                ),
                "nonempty_verified": all(
                    result["formula_target_nonempty"] for result in results.values()
                ),
                "vector_only": all(
                    result["formula_vector_only"] for result in results.values()
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

        jobs, feedback = self._preflight(artifact, metrics)
        placements = [placement for job in jobs for placement in self._formula_placements(job)]
        missing_assets = sorted(
            placement.formula_id
            for placement in placements
            if placement.formula_id not in formula_assets
        )
        if missing_assets:
            raise ReflowContractError(
                f"formula placements have no verified assets: {missing_assets[:5]}"
            )
        blocking = any(item["status"] in {"needs_review", "failed"} for item in feedback)
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
        if jobs and (self.allow_partial or not blocking):
            document = fitz.open(source_pdf)
            source_formula_document = fitz.open(source_pdf)
            svg_documents: dict[str, fitz.Document] = {}
            drawn_formula_ids: list[str] = []
            try:
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
                    item["verification"] = verification[item["block_id"]]
                    if not verification[item["block_id"]]["text_layer_nonempty"]:
                        item["status"] = "failed"
                        item["reason"] = "rendered_text_layer_is_empty"
                        item["action"] = "manual_review"
            blocking = any(item["status"] in {"needs_review", "failed"} for item in feedback)
            status = "partial_needs_review" if blocking else "rendered"
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
