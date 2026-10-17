"""Versioned report emitted by the thin Reflow renderer."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, model_validator


REFLOW_FEEDBACK_VERSION = "reflow-feedback-v2"


class StrictReflowModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReflowItemOutput(StrictReflowModel):
    block_id: StrictStr
    translation_unit_id: StrictStr
    page_idx: StrictInt = Field(ge=0)
    bbox: list[float]
    status: Literal["rendered", "preserved", "skipped", "needs_review", "failed"]
    reason: StrictStr
    action: Literal[
        "none",
        "preserve_original",
        "request_constrained_retranslation",
        "manual_review",
    ]
    style_role: StrictStr | None
    recommended_style: dict[str, Any] | None
    measurement: dict[str, Any] | None
    translation_request: dict[str, Any] | None
    verification: dict[str, Any] | None


class ReflowSummaryOutput(StrictReflowModel):
    items_total: StrictInt = Field(ge=0)
    rendered: StrictInt = Field(ge=0)
    preserved: StrictInt = Field(ge=0)
    skipped: StrictInt = Field(ge=0)
    needs_review: StrictInt = Field(ge=0)
    failed: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def validate_total(self) -> Self:
        subtotal = self.rendered + self.preserved + self.skipped + self.needs_review + self.failed
        if subtotal != self.items_total:
            raise ValueError("reflow summary counts must add up to items_total")
        return self


class FormulaVerificationSummaryOutput(StrictReflowModel):
    expected: StrictInt = Field(ge=0)
    drawn: StrictInt = Field(ge=0)
    source_pdf_clip: StrictInt = Field(ge=0)
    mathjax_svg: StrictInt = Field(ge=0)
    order_verified: StrictBool
    bbox_verified: StrictBool
    baseline_verified: StrictBool
    overlap_count: StrictInt = Field(ge=0)
    nonempty_verified: StrictBool
    vector_only: StrictBool

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.source_pdf_clip + self.mathjax_svg != self.expected:
            raise ValueError("formula asset counts must add up to expected")
        return self


class ReflowReportOutput(StrictReflowModel):
    schema_version: Literal["reflow-feedback-v1", "reflow-feedback-v2"]
    contract_version: Literal["translation-reflow-v1", "translation-reflow-v2"]
    optimizer_version: Literal["layout-optimizer-v1", "layout-optimizer-v2"]
    measurement_engine: Literal["layout-metrics-v1", "layout-metrics-v2"]
    created_at: StrictStr
    source_translation_artifact: StrictStr
    source_pdf: StrictStr
    output_pdf: StrictStr | None
    status: Literal["rendered", "partial_needs_review", "blocked", "failed"]
    allow_partial: StrictBool
    summary: ReflowSummaryOutput
    formula_verification: FormulaVerificationSummaryOutput | None = None
    items: list[ReflowItemOutput]

    @model_validator(mode="after")
    def validate_items(self) -> Self:
        if len(self.items) != self.summary.items_total:
            raise ValueError("reflow item count does not match summary")
        if self.status in {"rendered", "partial_needs_review"} and self.output_pdf is None:
            raise ValueError("rendered reports must reference output_pdf")
        if self.status == "blocked" and self.output_pdf is not None:
            raise ValueError("blocked reports cannot publish output_pdf")
        if self.schema_version == "reflow-feedback-v2" and self.formula_verification is None:
            raise ValueError("reflow-feedback-v2 requires formula verification")
        return self
