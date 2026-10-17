"""Versioned schema for readable, unit-level translation artifacts."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Literal
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, model_validator


OUTPUT_SCHEMA_VERSION = "translation-v3"
TranslationStatus = Literal[
    "translated",
    "simulated",
    "preserved",
    "skipped",
    "failed",
    "needs_review",
]


class StrictOutputModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QualityFlagOutput(StrictOutputModel):
    code: StrictStr
    severity: Literal["info", "warning", "error"]
    message: StrictStr


class QualityOutput(StrictOutputModel):
    disposition: Literal["pass", "review", "reject", "simulation", "not_applicable"]
    flags: list[QualityFlagOutput]
    metrics: dict[str, Any]


class ProtectedTokenOutput(StrictOutputModel):
    placeholder: StrictStr
    original: StrictStr
    kind: StrictStr
    start: StrictInt
    end: StrictInt
    formula_id: StrictStr | None = None


class FormulaRecordOutput(StrictOutputModel):
    formula_id: StrictStr
    kind: Literal["inline", "display"]
    translation_unit_id: StrictStr
    block_id: StrictStr | None
    page_idx: StrictInt | None = Field(default=None, ge=0)
    latex: StrictStr
    latex_hash: StrictStr
    source_bbox: list[float] | None
    source_line_bbox: list[float] | None
    source_block_bbox: list[float] | None
    source_line_index: StrictInt | None = Field(default=None, ge=0)
    source_span_index: StrictInt | None = Field(default=None, ge=0)
    source_span_path: StrictStr | None
    recognition_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    geometry_flags: list[
        Literal[
            "line_bbox_outside_block",
            "formula_bbox_outside_line_tolerance_used",
            "formula_bbox_outside_block_tolerance_used",
        ]
    ]
    protected_placeholder: StrictStr | None
    source_text_start: StrictInt | None = Field(default=None, ge=0)
    source_text_end: StrictInt | None = Field(default=None, ge=0)
    binding_status: Literal[
        "bound_exact",
        "bound_normalized",
        "latex_mismatch",
        "protected_token_only",
        "source_span_only",
        "preserved_display",
    ]
    render_policy: Literal["move_with_text", "preserve_in_place", "needs_review"]
    asset_preference: Literal["source_pdf_then_mathjax", "mathjax_svg", "source_pdf"]
    asset_id: StrictStr | None = None
    asset_status: Literal["uninspected", "ready", "failed"]

    @model_validator(mode="after")
    def validate_formula_record(self) -> Self:
        expected_hash = hashlib.sha256(self.latex.encode("utf-8")).hexdigest()
        if self.latex_hash != expected_hash:
            raise ValueError("formula latex_hash does not match latex")
        geometry = (
            self.source_bbox,
            self.source_line_bbox,
            self.source_block_bbox,
            self.page_idx,
            self.block_id,
            self.source_line_index,
            self.source_span_index,
            self.source_span_path,
        )
        if any(value is not None for value in geometry) and any(
            value is None for value in geometry
        ):
            raise ValueError("formula source geometry must be either complete or absent")
        if self.source_text_start is not None or self.source_text_end is not None:
            if self.source_text_start is None or self.source_text_end is None:
                raise ValueError("formula source text range must be complete")
            if self.source_text_end <= self.source_text_start:
                raise ValueError("formula source text range must satisfy start < end")
        if self.kind == "display" and self.binding_status == "preserved_display":
            if self.render_policy != "preserve_in_place":
                raise ValueError("preserved display formulas must remain in place")
        if self.binding_status == "source_span_only" and self.render_policy != "needs_review":
            raise ValueError("unprotected source formulas must require review")
        if len(self.geometry_flags) != len(set(self.geometry_flags)):
            raise ValueError("formula geometry_flags must be unique")
        if self.source_bbox is None and self.geometry_flags:
            raise ValueError("formula geometry_flags require source geometry")
        if self.asset_status == "uninspected" and self.asset_id is not None:
            raise ValueError("uninspected formula cannot reference an asset")
        if self.asset_status != "uninspected" and self.asset_id is None:
            raise ValueError("resolved formula must reference an asset")
        return self


class FormulaStatsOutput(StrictOutputModel):
    total: StrictInt = Field(ge=0)
    inline: StrictInt = Field(ge=0)
    display: StrictInt = Field(ge=0)
    protected: StrictInt = Field(ge=0)
    source_geometry: StrictInt = Field(ge=0)
    needs_review: StrictInt = Field(ge=0)


class FormulaContractOutput(StrictOutputModel):
    schema_version: Literal["formula-contract-v1"]
    inventory: list[FormulaRecordOutput]
    stats: FormulaStatsOutput

    @model_validator(mode="after")
    def validate_inventory(self) -> Self:
        ids = [record.formula_id for record in self.inventory]
        if len(ids) != len(set(ids)):
            raise ValueError("formula_id values must be unique")
        expected = {
            "total": len(self.inventory),
            "inline": sum(record.kind == "inline" for record in self.inventory),
            "display": sum(record.kind == "display" for record in self.inventory),
            "protected": sum(
                record.protected_placeholder is not None for record in self.inventory
            ),
            "source_geometry": sum(
                record.source_bbox is not None for record in self.inventory
            ),
            "needs_review": sum(
                record.render_policy == "needs_review" for record in self.inventory
            ),
        }
        if self.stats.model_dump() != expected:
            raise ValueError("formula stats do not match formula inventory")
        return self


class FormulaAssetMetricsOutput(StrictOutputModel):
    coordinate_system: Literal["pdf_points", "mathjax_ex"]
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    baseline_from_top: float | None = Field(default=None, ge=0)
    depth: float | None = Field(default=None, ge=0)
    aspect_ratio: float = Field(gt=0)
    view_box: list[float] | None

    @model_validator(mode="after")
    def validate_metrics(self) -> Self:
        if self.baseline_from_top is not None and self.baseline_from_top > self.height + 1e-6:
            raise ValueError("formula baseline must be within asset height")
        if self.depth is not None and self.depth > self.height + 1e-6:
            raise ValueError("formula depth cannot exceed asset height")
        if not abs(self.aspect_ratio - self.width / self.height) <= 1e-6:
            raise ValueError("formula aspect_ratio does not match width and height")
        if self.coordinate_system == "mathjax_ex":
            if self.view_box is None or len(self.view_box) != 4:
                raise ValueError("MathJax metrics require a four-coordinate view_box")
            if self.view_box[2] <= 0 or self.view_box[3] <= 0:
                raise ValueError("MathJax view_box dimensions must be positive")
        elif self.view_box is not None:
            raise ValueError("source PDF clip metrics cannot contain an SVG view_box")
        return self


class SourcePdfClipOutput(StrictOutputModel):
    source_pdf_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    page_idx: StrictInt = Field(ge=0)
    requested_bbox: list[float]
    clip_bbox: list[float]
    text_match: Literal["exact", "refined", "not_checked"]
    text_span_count: StrictInt = Field(ge=0)
    drawing_count: StrictInt = Field(ge=0)
    raster_image_count: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def validate_bbox(self) -> Self:
        for name, bbox in (
            ("requested_bbox", self.requested_bbox),
            ("clip_bbox", self.clip_bbox),
        ):
            if len(bbox) != 4 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                raise ValueError(f"source PDF clip {name} is invalid")
        if (
            self.clip_bbox[0] < self.requested_bbox[0] - 1e-6
            or self.clip_bbox[1] < self.requested_bbox[1] - 1e-6
            or self.clip_bbox[2] > self.requested_bbox[2] + 1e-6
            or self.clip_bbox[3] > self.requested_bbox[3] + 1e-6
        ):
            raise ValueError("source PDF clip bbox must stay inside requested bbox")
        if self.text_match == "refined" and self.clip_bbox == self.requested_bbox:
            raise ValueError("refined source PDF clip must change its bbox")
        if self.raster_image_count:
            raise ValueError("source PDF vector clip cannot include raster images")
        if not self.text_span_count and not self.drawing_count:
            raise ValueError("source PDF vector clip must contain vector content")
        return self


class FormulaAssetEngineOutput(StrictOutputModel):
    name: Literal["pymupdf-source-clip", "mathjax"]
    version: StrictStr


class FormulaAssetRecordOutput(StrictOutputModel):
    asset_id: StrictStr
    formula_id: StrictStr
    status: Literal["ready", "failed"]
    kind: Literal["source_pdf_clip", "mathjax_svg"] | None
    storage: Literal["source_pdf_reference", "file", "none"]
    relative_path: StrictStr | None
    media_type: Literal["application/pdf", "image/svg+xml"] | None
    sha256: StrictStr | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    metrics: FormulaAssetMetricsOutput | None
    source_pdf_clip: SourcePdfClipOutput | None
    engine: FormulaAssetEngineOutput | None
    fallback_reason: StrictStr | None
    failure_code: StrictStr | None

    @model_validator(mode="after")
    def validate_asset(self) -> Self:
        if self.asset_id != f"fa-{self.formula_id}":
            raise ValueError("formula asset_id must be derived from formula_id")
        if self.status == "failed":
            if any(
                value is not None
                for value in (
                    self.kind,
                    self.relative_path,
                    self.media_type,
                    self.sha256,
                    self.metrics,
                    self.source_pdf_clip,
                    self.engine,
                )
            ) or self.storage != "none" or not self.failure_code:
                raise ValueError("failed formula asset must contain only failure metadata")
            return self
        if any(
            value is None for value in (self.kind, self.media_type, self.sha256, self.metrics, self.engine)
        ):
            raise ValueError("ready formula asset metadata is incomplete")
        if self.failure_code is not None:
            raise ValueError("ready formula asset cannot have failure_code")
        if self.kind == "source_pdf_clip":
            if (
                self.storage != "source_pdf_reference"
                or self.relative_path is not None
                or self.media_type != "application/pdf"
                or self.source_pdf_clip is None
                or self.metrics.coordinate_system != "pdf_points"
            ):
                raise ValueError("source PDF formula asset is inconsistent")
        elif (
            self.storage != "file"
            or not self.relative_path
            or self.media_type != "image/svg+xml"
            or self.source_pdf_clip is not None
            or self.metrics.coordinate_system != "mathjax_ex"
        ):
            raise ValueError("MathJax formula asset is inconsistent")
        if self.relative_path:
            path = self.relative_path.replace("\\", "/")
            if path.startswith("/") or ".." in path.split("/"):
                raise ValueError("formula asset relative_path must stay within artifact directory")
        return self


class FormulaAssetStatsOutput(StrictOutputModel):
    total: StrictInt = Field(ge=0)
    ready: StrictInt = Field(ge=0)
    failed: StrictInt = Field(ge=0)
    source_pdf_clip: StrictInt = Field(ge=0)
    mathjax_svg: StrictInt = Field(ge=0)
    mathjax_fallback: StrictInt = Field(ge=0)


class MathJaxRuntimeOutput(StrictOutputModel):
    available: StrictBool
    version: StrictStr | None
    script: StrictStr


class FormulaAssetContractOutput(StrictOutputModel):
    schema_version: Literal["formula-assets-v1"]
    asset_root: StrictStr
    source_pdf_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    mathjax: MathJaxRuntimeOutput
    assets: list[FormulaAssetRecordOutput]
    stats: FormulaAssetStatsOutput

    @model_validator(mode="after")
    def validate_asset_contract(self) -> Self:
        root = self.asset_root.replace("\\", "/")
        if root.startswith("/") or ".." in root.split("/"):
            raise ValueError("formula asset_root must stay within artifact directory")
        ids = [asset.asset_id for asset in self.assets]
        formula_ids = [asset.formula_id for asset in self.assets]
        if len(ids) != len(set(ids)) or len(formula_ids) != len(set(formula_ids)):
            raise ValueError("formula asset and formula IDs must be unique")
        expected = {
            "total": len(self.assets),
            "ready": sum(asset.status == "ready" for asset in self.assets),
            "failed": sum(asset.status == "failed" for asset in self.assets),
            "source_pdf_clip": sum(asset.kind == "source_pdf_clip" for asset in self.assets),
            "mathjax_svg": sum(asset.kind == "mathjax_svg" for asset in self.assets),
            "mathjax_fallback": sum(
                asset.kind == "mathjax_svg"
                and asset.fallback_reason != "source_geometry_unavailable"
                for asset in self.assets
            ),
        }
        if self.stats.model_dump() != expected:
            raise ValueError("formula asset stats do not match assets")
        return self


class TranslationCandidateOutput(StrictOutputModel):
    candidate_id: StrictStr
    mode: Literal["constrained"]
    text: StrictStr
    reason: StrictStr
    constraints: dict[str, Any]
    quality: QualityOutput
    created_at: StrictStr


class UnitExecutionOutput(StrictOutputModel):
    attempts: StrictInt = Field(ge=0)
    cache_hit: StrictBool
    cache_key: StrictStr | None
    response_hash: StrictStr | None
    request_id: StrictStr | None
    last_error: dict[str, Any] | None


class UnitTranslationOutput(StrictOutputModel):
    status: TranslationStatus
    initial_text: StrictStr | None
    candidates: list[TranslationCandidateOutput]
    selected_text: StrictStr | None
    selected_version: Literal["source", "initial"] | StrictStr | None
    source_hash: StrictStr
    quality_flags: list[StrictStr]
    quality: QualityOutput
    protected_tokens: list[ProtectedTokenOutput]
    execution: UnitExecutionOutput

    @model_validator(mode="after")
    def validate_selected_version(self) -> Self:
        candidate_by_id = {candidate.candidate_id: candidate for candidate in self.candidates}
        if len(candidate_by_id) != len(self.candidates):
            raise ValueError("candidate_id values must be unique within one translation unit")
        if self.selected_version == "initial":
            if self.initial_text is None or self.selected_text != self.initial_text:
                raise ValueError("selected_version=initial must select initial_text")
        elif self.selected_version not in {None, "source"}:
            candidate = candidate_by_id.get(self.selected_version)
            if candidate is None or self.selected_text != candidate.text:
                raise ValueError("selected_version must reference the selected candidate text")
        if self.status in {"translated", "simulated"} and self.selected_version is None:
            raise ValueError(f"{self.status} output must have a selected version")
        if self.status in {"failed", "skipped"} and any(
            value is not None for value in (self.initial_text, self.selected_text, self.selected_version)
        ):
            raise ValueError(f"{self.status} output cannot publish translated text")
        return self


class TranslationUnitOutput(StrictOutputModel):
    translation_unit_id: StrictStr
    logical_para_id: StrictInt = Field(ge=0)
    element_type: StrictStr
    policy: StrictStr
    style_role: StrictStr
    golden_style: dict[str, Any]
    block_ids: list[StrictStr]
    formula_ids: list[StrictStr] = Field(default_factory=list)
    source_text: StrictStr
    translation: UnitTranslationOutput
    layout: "UnitLayoutOutput"

    @model_validator(mode="after")
    def validate_unit_invariants(self) -> Self:
        if not self.block_ids or len(set(self.block_ids)) != len(self.block_ids):
            raise ValueError("block_ids must be non-empty and unique")
        expected_hash = hashlib.sha256(self.source_text.encode("utf-8")).hexdigest()
        if self.translation.source_hash != expected_hash:
            raise ValueError("translation source_hash does not match source_text")
        if self.translation.status == "preserved":
            if self.translation.selected_version != "source":
                raise ValueError("preserved output must select the source version")
            if self.translation.selected_text != self.source_text:
                raise ValueError("preserved output must keep source_text exactly")
        return self


class DocumentOutput(StrictOutputModel):
    document_id: StrictStr
    source_pdf: StrictStr
    source_json: StrictStr
    source_hash: StrictStr
    source_language: StrictStr
    target_language: StrictStr
    page_count: StrictInt = Field(ge=1)
    block_count: StrictInt = Field(ge=1)
    translation_unit_count: StrictInt = Field(ge=1)
    formula_count: StrictInt = Field(default=0, ge=0)


class TranslationRunOutput(StrictOutputModel):
    mode: Literal["simulation", "provider"]
    requested_provider: StrictStr
    provider: StrictStr
    model: StrictStr
    prompt_version: StrictStr
    glossary_version: StrictStr
    started_at: StrictStr
    completed_at: StrictStr
    network_requests: StrictInt = Field(ge=0)
    stats: dict[str, StrictInt]


class GoldenFontOutput(StrictOutputModel):
    font_id: StrictStr
    family: StrictStr
    path: StrictStr
    project_relative_path: StrictStr
    sha256: StrictStr
    ascender: float
    descender: float


class GoldenSourceStatisticsOutput(StrictOutputModel):
    font_size_pt: float
    line_height_multiplier: float
    dominant_font: StrictStr
    dominant_color: StrictStr
    bold_ratio: float
    italic_ratio: float
    alignment: Literal["left", "center", "right", "justify"]
    span_sample_count: StrictInt = Field(ge=0)
    retained_size_sample_count: StrictInt = Field(ge=0)
    line_gap_sample_count: StrictInt = Field(ge=0)
    weighted_character_count: StrictInt = Field(ge=0)
    sample_sources: dict[str, StrictInt]


class GoldenTargetStyleOutput(StrictOutputModel):
    font_id: StrictStr
    font_size_pt: float
    line_height_multiplier: float
    char_spacing_em: float
    alignment: Literal["left", "center", "right", "justify"]


class NumericRangeOutput(StrictOutputModel):
    min: float
    max: float


class FontSizeRangeOutput(NumericRangeOutput):
    max_shrink_ratio: float


class GoldenGuardrailsOutput(StrictOutputModel):
    bbox_locked: Literal[True]
    font_size_pt: FontSizeRangeOutput
    line_height_multiplier: NumericRangeOutput
    char_spacing_em: NumericRangeOutput


class GoldenProvenanceOutput(StrictOutputModel):
    font_size: Literal["measured", "derived_from_body"]
    line_height: Literal["measured", "role_default"]
    source: Literal["original_pdf_and_parsed_rich_spans"]


class GoldenRoleOutput(StrictOutputModel):
    role: StrictStr
    source_statistics: GoldenSourceStatisticsOutput
    target_style: GoldenTargetStyleOutput
    guardrails: GoldenGuardrailsOutput
    provenance: GoldenProvenanceOutput


class GoldenStyleProfileOutput(StrictOutputModel):
    schema_version: Literal["layout-style-v1"]
    measurement_engine: Literal["layout-metrics-v1", "layout-metrics-v2"]
    document_id: StrictStr
    source_hash: StrictStr
    statistics_basis: Literal["original_pdf"]
    translated_text_used_for_statistics: Literal[False]
    body_reference_font_size_pt: float
    fonts: dict[str, GoldenFontOutput]
    roles: dict[str, GoldenRoleOutput]
    unit_roles: dict[str, StrictStr]

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        for key, role in self.roles.items():
            if role.role != key:
                raise ValueError("golden style role key does not match its role field")
            if role.target_style.font_id not in self.fonts:
                raise ValueError(f"golden style role {key} references an unknown font")
        unknown_roles = sorted(set(self.unit_roles.values()) - set(self.roles))
        if unknown_roles:
            raise ValueError(f"unit_roles references unknown roles: {unknown_roles}")
        return self


class RecommendedStyleOutput(StrictOutputModel):
    font_id: StrictStr
    font_size_pt: float
    line_height_multiplier: float
    char_spacing_em: float
    alignment: Literal["left", "center", "right", "justify"]


class MixedRunOutput(StrictOutputModel):
    kind: Literal["text", "formula"]
    text: StrictStr
    x_offset_pt: float = Field(ge=0)
    width_pt: float = Field(ge=0)
    height_pt: float = Field(ge=0)
    baseline_from_top_pt: float = Field(ge=0)
    formula_id: StrictStr | None = None
    asset_id: StrictStr | None = None
    asset_kind: Literal["source_pdf_clip", "mathjax_svg"] | None = None
    asset_scale_to_pt: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_run(self) -> Self:
        formula_fields = (self.formula_id, self.asset_id, self.asset_kind, self.asset_scale_to_pt)
        if self.kind == "formula":
            if any(value is None for value in formula_fields):
                raise ValueError("formula run must reference one ready formula asset")
        elif any(value is not None for value in formula_fields):
            raise ValueError("text run cannot reference a formula asset")
        if self.baseline_from_top_pt > self.height_pt + 0.02:
            raise ValueError("run baseline cannot fall below its height")
        return self


class MixedLineOutput(StrictOutputModel):
    text: StrictStr
    width_pt: float = Field(ge=0)
    height_pt: float = Field(gt=0)
    baseline_from_top_pt: float = Field(ge=0)
    runs: list[MixedRunOutput]

    @model_validator(mode="after")
    def validate_line(self) -> Self:
        if self.baseline_from_top_pt > self.height_pt + 0.02:
            raise ValueError("line baseline cannot fall below its height")
        if "".join(run.text for run in self.runs) != self.text:
            raise ValueError("mixed runs must reconstruct their line text")
        expected_x = 0.0
        for run in self.runs:
            if not math.isclose(run.x_offset_pt, expected_x, abs_tol=0.03):
                raise ValueError("mixed run x offsets must be contiguous")
            expected_x += run.width_pt
        if not math.isclose(expected_x, self.width_pt, abs_tol=0.03):
            raise ValueError("mixed run widths must reconstruct line width")
        return self


class LayoutMeasurementOutput(StrictOutputModel):
    fits: StrictBool
    is_overflow: StrictBool
    available_width_pt: float
    available_height_pt: float
    used_width_pt: float
    used_height_pt: float
    overflow_width_pt: float
    overflow_height_pt: float
    line_count: StrictInt = Field(ge=0)
    line_height_pt: float
    width_fill_ratio: float
    height_fill_ratio: float
    area_fill_ratio: float
    estimated_cjk_capacity: StrictInt = Field(ge=0)
    lines: list[StrictStr]
    line_widths_pt: list[float]
    font_id: StrictStr
    font_size_pt: float
    line_height_multiplier: float
    char_spacing_em: float
    mixed_lines: list[MixedLineOutput] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_mixed_lines(self) -> Self:
        if self.mixed_lines:
            if len(self.mixed_lines) != self.line_count:
                raise ValueError("mixed line count must match line_count")
            if [line.text for line in self.mixed_lines] != self.lines:
                raise ValueError("mixed lines must match legacy line text")
            for line, width in zip(self.mixed_lines, self.line_widths_pt, strict=True):
                if not math.isclose(line.width_pt, width, abs_tol=0.03):
                    raise ValueError("mixed line width must match legacy line width")
        return self


class LayoutSegmentOutput(StrictOutputModel):
    block_id: StrictStr
    translated: StrictStr
    fit_status: Literal["fits_golden", "fits_with_adjustment", "requires_retranslation"]
    recommended_style: RecommendedStyleOutput
    measurement: LayoutMeasurementOutput
    optimization_score: float | None
    evaluated_style_candidates: StrictInt = Field(ge=1)
    feasible_style_candidates: StrictInt = Field(ge=0)


class UnitLayoutOutput(StrictOutputModel):
    status: Literal[
        "fits_golden",
        "fits_with_adjustment",
        "requires_retranslation",
        "needs_review",
        "not_applicable",
    ]
    optimizer_version: Literal["layout-optimizer-v1", "layout-optimizer-v2"]
    all_blocks_fit: StrictBool
    selected_allocation_id: StrictStr | None
    score: float | None
    evaluated_allocations: StrictInt = Field(ge=0)
    feasible_allocations: StrictInt = Field(ge=0)
    retranslation_constraints: dict[str, Any] | None
    segments: list[LayoutSegmentOutput]
    retranslation_attempts: StrictInt = Field(ge=0)
    retranslation_errors: list[dict[str, Any]]


class LayoutRunOutput(StrictOutputModel):
    contract_version: Literal["translation-reflow-v1", "translation-reflow-v2"]
    optimizer_version: Literal["layout-optimizer-v1", "layout-optimizer-v2"]
    measurement_engine: Literal["layout-metrics-v1", "layout-metrics-v2"]
    stats: dict[str, StrictInt]

    @model_validator(mode="after")
    def validate_versions(self) -> Self:
        expected = {
            "translation-reflow-v1": "layout-metrics-v1",
            "translation-reflow-v2": "layout-metrics-v2",
        }
        if self.measurement_engine != expected[self.contract_version]:
            raise ValueError("Translation/Reflow contract and measurement engine do not match")
        return self


class TranslationArtifactOutput(StrictOutputModel):
    schema_version: Literal["translation-v1", "translation-v2", "translation-v3"]
    document: DocumentOutput
    translation_run: TranslationRunOutput
    layout_status: Literal["ready_for_reflow", "needs_retranslation", "needs_review"]
    layout_run: LayoutRunOutput
    golden_style_profile: GoldenStyleProfileOutput
    formula_contract: FormulaContractOutput | None = None
    formula_assets: FormulaAssetContractOutput | None = None
    translation_units: list[TranslationUnitOutput]
    items: list[dict[str, Any]]

    @model_validator(mode="after")
    def validate_layout_contract(self) -> Self:
        if self.layout_run.measurement_engine != self.golden_style_profile.measurement_engine:
            raise ValueError("layout run and golden style measurement engines do not match")
        if self.schema_version in {"translation-v2", "translation-v3"} and self.formula_contract is None:
            raise ValueError(f"{self.schema_version} requires formula_contract")
        if self.schema_version == "translation-v3" and self.formula_assets is None:
            raise ValueError("translation-v3 requires formula_assets")
        formulas = {
            record.formula_id: record
            for record in (self.formula_contract.inventory if self.formula_contract else [])
        }
        if self.document.formula_count != len(formulas):
            raise ValueError("document formula_count does not match formula inventory")
        if self.formula_assets is not None:
            assets = {asset.formula_id: asset for asset in self.formula_assets.assets}
            if set(assets) != set(formulas):
                raise ValueError("formula asset contract must cover every formula exactly once")
            for formula_id, formula in formulas.items():
                asset = assets[formula_id]
                if formula.asset_id != asset.asset_id or formula.asset_status != asset.status:
                    raise ValueError("formula inventory asset link does not match asset contract")
        elif any(formula.asset_status != "uninspected" for formula in formulas.values()):
            raise ValueError("formula without asset contract must remain uninspected")
        units = {unit.translation_unit_id: unit for unit in self.translation_units}
        if len(units) != len(self.translation_units):
            raise ValueError("translation_unit_id values must be unique")
        item_by_block: dict[str, dict[str, Any]] = {}
        for item in self.items:
            block_id = item.get("block_id")
            if not isinstance(block_id, str) or not block_id:
                raise ValueError("every item must have a non-empty block_id")
            if block_id in item_by_block:
                raise ValueError("item block_id values must be unique")
            item_by_block[block_id] = item

        for unit in self.translation_units:
            if len(unit.formula_ids) != len(set(unit.formula_ids)):
                raise ValueError("unit formula_ids must be unique")
            for formula_id in unit.formula_ids:
                formula = formulas.get(formula_id)
                if formula is None:
                    raise ValueError("unit references an unknown formula_id")
                if formula.translation_unit_id != unit.translation_unit_id:
                    raise ValueError("formula translation_unit_id does not match unit")
            token_formula_ids = {
                token.formula_id
                for token in unit.translation.protected_tokens
                if token.formula_id is not None
            }
            if not token_formula_ids.issubset(set(unit.formula_ids)):
                raise ValueError("protected token references formula outside its unit")
            segment_by_block = {segment.block_id: segment for segment in unit.layout.segments}
            if len(segment_by_block) != len(unit.layout.segments):
                raise ValueError("layout segment block_id values must be unique within a unit")
            if unit.layout.segments:
                if set(segment_by_block) != set(unit.block_ids):
                    raise ValueError("layout segments must cover exactly the unit block_ids")
                combined = "".join(segment.translated for segment in unit.layout.segments)
                if combined != unit.translation.selected_text:
                    raise ValueError("layout segment text must exactly reconstruct selected_text")
            mixed_formula_ids: list[str] = []
            for segment in unit.layout.segments:
                for line in segment.measurement.mixed_lines:
                    for run in line.runs:
                        if run.kind != "formula":
                            continue
                        formula = formulas.get(str(run.formula_id))
                        if formula is None or formula.translation_unit_id != unit.translation_unit_id:
                            raise ValueError("mixed formula run references a formula outside its unit")
                        if run.asset_id != formula.asset_id:
                            raise ValueError("mixed formula run asset does not match formula inventory")
                        mixed_formula_ids.append(formula.formula_id)
            if (
                self.layout_run.contract_version == "translation-reflow-v2"
                and self.formula_assets is not None
                and unit.layout.segments
            ):
                expected_mixed = [
                    formula_id
                    for formula_id in unit.formula_ids
                    if formulas[formula_id].kind == "inline"
                    and formulas[formula_id].render_policy == "move_with_text"
                ]
                if mixed_formula_ids != expected_mixed:
                    raise ValueError(
                        "v2 mixed layout must contain every movable inline formula exactly once"
                    )
            for block_id, segment in segment_by_block.items():
                item = item_by_block.get(block_id)
                if item is None:
                    raise ValueError(f"layout segment references unknown block {block_id}")
                if item.get("translation_unit_id") != unit.translation_unit_id:
                    raise ValueError("item translation_unit_id does not match layout segment")
                if item.get("translated") != segment.translated:
                    raise ValueError("item translated text does not match layout segment")
        linked_formula_ids = {
            formula_id for unit in self.translation_units for formula_id in unit.formula_ids
        }
        if linked_formula_ids != set(formulas):
            raise ValueError("formula inventory must be linked from exactly one unit")
        formula_item_links: dict[str, list[dict[str, Any]]] = {formula_id: [] for formula_id in formulas}
        for item in self.items:
            for formula_id in item.get("formula_ids", []):
                formula = formulas.get(str(formula_id))
                if formula is None:
                    raise ValueError("item references an unknown formula_id")
                if item.get("translation_unit_id") != formula.translation_unit_id:
                    raise ValueError("item formula belongs to another translation unit")
                formula_item_links[formula.formula_id].append(item)
        for formula in formulas.values():
            links = formula_item_links[formula.formula_id]
            if formula.kind == "inline" and formula.render_policy == "move_with_text":
                unit = units[formula.translation_unit_id]
                if unit.layout.segments and len(links) != 1:
                    raise ValueError("movable inline formula must be linked from its allocated item")
            elif formula.block_id is not None:
                source_item = item_by_block.get(formula.block_id)
                if source_item is None or source_item not in links:
                    raise ValueError("preserved formula must remain linked from its source item")
        return self
