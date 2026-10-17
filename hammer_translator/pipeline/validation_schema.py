"""Versioned output contract for the final PDF validation gate."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, model_validator


VALIDATION_REPORT_VERSION = "pipeline-validation-v1"


class StrictValidationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ValidationCheckOutput(StrictValidationModel):
    name: StrictStr
    status: Literal["passed", "warning", "failed"]
    message: StrictStr
    detail: dict[str, Any] = Field(default_factory=dict)


class ValidationSummaryOutput(StrictValidationModel):
    checks_total: StrictInt = Field(ge=0)
    passed: StrictInt = Field(ge=0)
    warnings: StrictInt = Field(ge=0)
    failed: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def validate_total(self) -> Self:
        if self.passed + self.warnings + self.failed != self.checks_total:
            raise ValueError("validation check counts must add up to checks_total")
        return self


class PDFValidationReportOutput(StrictValidationModel):
    schema_version: Literal["pipeline-validation-v1"] = VALIDATION_REPORT_VERSION
    status: Literal["passed", "passed_with_warnings", "failed"]
    created_at: StrictStr
    source_pdf: StrictStr
    translated_pdf: StrictStr
    reflow_feedback: StrictStr
    source_pdf_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    translated_pdf_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    page_count: StrictInt = Field(ge=1)
    reflow_status: Literal["rendered", "partial_needs_review"]
    review_item_count: StrictInt = Field(ge=0)
    failed_item_count: StrictInt = Field(ge=0)
    publishable: StrictBool
    summary: ValidationSummaryOutput
    checks: list[ValidationCheckOutput]

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if len(self.checks) != self.summary.checks_total:
            raise ValueError("validation checks do not match summary")
        expected_status = (
            "failed"
            if self.summary.failed
            else "passed_with_warnings"
            if self.summary.warnings
            else "passed"
        )
        if self.status != expected_status:
            raise ValueError("validation status does not match check results")
        if self.publishable and self.status == "failed":
            raise ValueError("failed validation cannot be publishable")
        return self
