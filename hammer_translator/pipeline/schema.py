"""Versioned contracts for the local V1 pipeline.

The web API, local runner, and future remote workers must communicate through
these models instead of sharing implementation-specific paths or objects.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


JOB_SCHEMA_VERSION = "pipeline-job-v1"
EVENT_SCHEMA_VERSION = "pipeline-event-v1"
STAGE_RESULT_SCHEMA_VERSION = "pipeline-stage-result-v1"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PipelineStage(StrEnum):
    PARSING = "parsing"
    TRANSLATING = "translating"
    REFLOWING = "reflowing"
    VALIDATING = "validating"


PIPELINE_STAGES: tuple[PipelineStage, ...] = (
    PipelineStage.PARSING,
    PipelineStage.TRANSLATING,
    PipelineStage.REFLOWING,
    PipelineStage.VALIDATING,
)


class JobStatus(StrEnum):
    QUEUED = "queued"
    PARSING = "parsing"
    TRANSLATING = "translating"
    REFLOWING = "reflowing"
    VALIDATING = "validating"
    COMPLETED = "completed"
    COMPLETED_WITH_WARNINGS = "completed_with_warnings"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_JOB_STATUSES = {
    JobStatus.COMPLETED,
    JobStatus.COMPLETED_WITH_WARNINGS,
    JobStatus.FAILED,
    JobStatus.CANCELLED,
}


STAGE_TO_JOB_STATUS = {
    PipelineStage.PARSING: JobStatus.PARSING,
    PipelineStage.TRANSLATING: JobStatus.TRANSLATING,
    PipelineStage.REFLOWING: JobStatus.REFLOWING,
    PipelineStage.VALIDATING: JobStatus.VALIDATING,
}


class StageStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    COMPLETED_WITH_WARNINGS = "completed_with_warnings"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EventType(StrEnum):
    JOB_CREATED = "job_created"
    STAGE_STARTED = "stage_started"
    STAGE_PROGRESS = "stage_progress"
    ARTIFACT_CREATED = "artifact_created"
    WARNING = "warning"
    STAGE_COMPLETED = "stage_completed"
    JOB_QUEUED = "job_queued"
    JOB_COMPLETED = "job_completed"
    JOB_FAILED = "job_failed"
    CANCEL_REQUESTED = "cancel_requested"
    JOB_CANCELLED = "job_cancelled"
    JOB_RESUMED = "job_resumed"


class ProgressRecord(StrictModel):
    current: int = Field(default=0, ge=0)
    total: int | None = Field(default=None, ge=0)
    percent: float | None = Field(default=None, ge=0.0, le=100.0)
    message: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_bounds(self) -> "ProgressRecord":
        if self.total is not None and self.current > self.total:
            raise ValueError("progress current cannot exceed total")
        return self


class WarningRecord(StrictModel):
    code: str
    message: str
    stage: PipelineStage | None = None
    block_id: str | None = None
    page_index: int | None = Field(default=None, ge=0)
    detail: dict[str, Any] = Field(default_factory=dict)


class ErrorRecord(StrictModel):
    code: str
    message: str
    stage: PipelineStage | None = None
    retryable: bool = False
    detail: dict[str, Any] = Field(default_factory=dict)
    occurred_at: datetime = Field(default_factory=utc_now)


class ArtifactRecord(StrictModel):
    name: str
    kind: str
    relative_path: str
    node_type: Literal["file", "directory"] = "file"
    media_type: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    stage: PipelineStage | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        normalized = value.replace("\\", "/")
        path = PurePosixPath(normalized)
        if path.is_absolute() or normalized in {"", "."} or ".." in path.parts:
            raise ValueError("artifact path must be a safe job-relative path")
        if normalized.startswith("./") or "//" in normalized:
            raise ValueError("artifact path must be normalized")
        return str(path)


class StageRecord(StrictModel):
    stage: PipelineStage
    status: StageStatus = StageStatus.PENDING
    attempt: int = Field(default=0, ge=0)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    progress: ProgressRecord = Field(default_factory=ProgressRecord)
    artifacts: list[ArtifactRecord] = Field(default_factory=list)
    warnings: list[WarningRecord] = Field(default_factory=list)
    error: ErrorRecord | None = None


class PipelineConfig(StrictModel):
    execution_mode: Literal["real", "simulation"] = "real"
    environment_profile: Literal["local_mac", "linux_gpu"] = "local_mac"
    source_language: str = "en"
    target_language: str = "zh-CN"
    parser_backend: Literal["pipeline", "vlm-engine", "hybrid-engine"] = "pipeline"
    parser_method: Literal["auto", "txt", "ocr"] = "auto"
    parser_device: Literal["auto", "mps", "cpu"] = "auto"
    parser_language: str = "ch"
    parser_formula: bool = True
    parser_table: bool = True
    parser_formula_ch: bool = True
    parser_image_analysis: bool = True
    parser_effort: Literal["medium", "high"] = "medium"
    parser_start_page: int = Field(default=0, ge=0)
    parser_end_page: int | None = Field(default=None, ge=0)
    parser_render_threads: int = Field(default=4, ge=1)
    parser_window_size: int = Field(default=8, ge=1)
    translation_provider: str = "deepseek"
    translation_model: str | None = None
    allow_partial_pdf: bool = True
    max_translation_attempts: int = Field(default=3, ge=1)
    max_retranslation_attempts: int = Field(default=2, ge=0)
    translation_request_timeout_seconds: float = Field(default=90.0, gt=0)

    @model_validator(mode="after")
    def validate_v1_languages(self) -> "PipelineConfig":
        if (self.source_language, self.target_language) != ("en", "zh-CN"):
            raise ValueError("V1 supports only English to Simplified Chinese")
        if (
            self.parser_end_page is not None
            and self.parser_end_page < self.parser_start_page
        ):
            raise ValueError("parser_end_page cannot be smaller than parser_start_page")
        return self


def initial_stage_records() -> list[StageRecord]:
    return [StageRecord(stage=stage) for stage in PIPELINE_STAGES]


class JobRecord(StrictModel):
    schema_version: Literal["pipeline-job-v1"] = JOB_SCHEMA_VERSION
    job_id: str
    display_name: str
    source_pdf: str = "input/source.pdf"
    status: JobStatus = JobStatus.QUEUED
    current_stage: PipelineStage | None = None
    config: PipelineConfig = Field(default_factory=PipelineConfig)
    cancel_requested: bool = False
    revision: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    stages: list[StageRecord] = Field(default_factory=initial_stage_records)
    artifacts: list[ArtifactRecord] = Field(default_factory=list)
    warnings: list[WarningRecord] = Field(default_factory=list)
    error: ErrorRecord | None = None

    @field_validator("job_id")
    @classmethod
    def validate_job_id(cls, value: str) -> str:
        try:
            parsed = UUID(value)
        except ValueError as exc:
            raise ValueError("job_id must be a UUID") from exc
        if str(parsed) != value:
            raise ValueError("job_id must use canonical UUID form")
        return value

    @field_validator("source_pdf")
    @classmethod
    def validate_source_pdf(cls, value: str) -> str:
        return ArtifactRecord(
            name="source", kind="source_pdf", relative_path=value
        ).relative_path

    @model_validator(mode="after")
    def validate_stage_set(self) -> "JobRecord":
        stages = [record.stage for record in self.stages]
        if stages != list(PIPELINE_STAGES):
            raise ValueError("job must contain every pipeline stage exactly once in order")
        if self.status in TERMINAL_JOB_STATUSES and self.current_stage is not None:
            raise ValueError("terminal jobs cannot have a current stage")
        return self

    def stage_record(self, stage: PipelineStage) -> StageRecord:
        return next(record for record in self.stages if record.stage is stage)


class PipelineEvent(StrictModel):
    schema_version: Literal["pipeline-event-v1"] = EVENT_SCHEMA_VERSION
    sequence: int = Field(ge=1)
    job_id: str
    event_type: EventType
    timestamp: datetime = Field(default_factory=utc_now)
    stage: PipelineStage | None = None
    status: JobStatus | None = None
    progress: ProgressRecord | None = None
    message: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)

    _validate_job_id = field_validator("job_id")(JobRecord.validate_job_id.__func__)


class ProducedArtifact(StrictModel):
    name: str
    kind: str
    relative_path: str
    node_type: Literal["file", "directory"] = "file"
    media_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    _validate_relative_path = field_validator("relative_path")(
        ArtifactRecord.validate_relative_path.__func__
    )


class StageExecutionResult(StrictModel):
    schema_version: Literal["pipeline-stage-result-v1"] = STAGE_RESULT_SCHEMA_VERSION
    artifacts: list[ProducedArtifact] = Field(default_factory=list)
    warnings: list[WarningRecord] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
