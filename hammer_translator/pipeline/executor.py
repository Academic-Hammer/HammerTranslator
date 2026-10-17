"""Execution boundary between pipeline orchestration and machine-specific work."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .schema import (
    ArtifactRecord,
    PipelineStage,
    ProducedArtifact,
    StageExecutionResult,
    WarningRecord,
)


class StageExecutionError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "stage_execution_failed",
        retryable: bool = False,
        detail: dict | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.detail = detail or {}


class StageCancelledError(StageExecutionError):
    def __init__(self, message: str = "pipeline stage was cancelled"):
        super().__init__(message, code="cancelled", retryable=True)


ProgressCallback = Callable[[int, int | None, str | None, dict | None], None]
CancellationCheck = Callable[[], bool]


@dataclass(frozen=True)
class StageContext:
    job_id: str
    job_dir: Path
    stage: PipelineStage
    attempt: int
    source_pdf: Path
    document_name: str
    config: dict
    artifacts: tuple[ArtifactRecord, ...] = ()


class PipelineExecutor(Protocol):
    name: str

    def execute(
        self,
        context: StageContext,
        *,
        progress: ProgressCallback,
        is_cancel_requested: CancellationCheck,
    ) -> StageExecutionResult: ...


@dataclass
class FakePipelineExecutor:
    """Deterministic Step 1 executor used to test orchestration without real work."""

    name: str = "fake-pipeline-executor"
    steps: dict[PipelineStage, int] = field(
        default_factory=lambda: {
            PipelineStage.PARSING: 3,
            PipelineStage.TRANSLATING: 4,
            PipelineStage.REFLOWING: 2,
            PipelineStage.VALIDATING: 2,
        }
    )
    warnings: dict[PipelineStage, list[WarningRecord]] = field(default_factory=dict)
    fail_stages: set[PipelineStage] = field(default_factory=set)
    fail_once_stages: set[PipelineStage] = field(default_factory=set)
    _attempts: dict[PipelineStage, int] = field(default_factory=dict, init=False)

    def execute(
        self,
        context: StageContext,
        *,
        progress: ProgressCallback,
        is_cancel_requested: CancellationCheck,
    ) -> StageExecutionResult:
        attempt = self._attempts.get(context.stage, 0) + 1
        self._attempts[context.stage] = attempt
        if context.stage in self.fail_stages or (
            context.stage in self.fail_once_stages and attempt == 1
        ):
            raise StageExecutionError(
                f"simulated failure in {context.stage.value}",
                code="simulated_stage_failure",
                retryable=True,
            )

        total = self.steps.get(context.stage, 1)
        for current in range(1, total + 1):
            if is_cancel_requested():
                raise StageCancelledError()
            progress(
                current,
                total,
                f"simulated {context.stage.value} {current}/{total}",
                {"executor": self.name},
            )

        relative = {
            PipelineStage.PARSING: "parse/stage_result.json",
            PipelineStage.TRANSLATING: "translation/stage_result.json",
            PipelineStage.REFLOWING: "output/reflow_stage_result.json",
            PipelineStage.VALIDATING: "output/validation_stage_result.json",
        }[context.stage]
        output = context.job_dir / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "simulated": True,
                    "job_id": context.job_id,
                    "stage": context.stage.value,
                    "attempt": context.attempt,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return StageExecutionResult(
            artifacts=[
                ProducedArtifact(
                    name=f"{context.stage.value}_result",
                    kind="stage_manifest",
                    relative_path=relative,
                    media_type="application/json",
                    metadata={"simulated": True},
                )
            ],
            warnings=self.warnings.get(context.stage, []),
            metadata={"executor": self.name},
        )
