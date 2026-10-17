"""Stage router for the complete single-machine V1 pipeline."""

from __future__ import annotations

from collections.abc import Callable

from .executor import (
    CancellationCheck,
    PipelineExecutor,
    ProgressCallback,
    StageContext,
    StageExecutionError,
)
from .mac_parser import MacMinerUParsingExecutor
from .reflow_executor import ReflowStageExecutor
from .schema import PipelineStage, StageExecutionResult
from .translation_executor import TranslationStageExecutor
from .validation_executor import PDFValidationStageExecutor


class LocalPipelineExecutor:
    """Route each pipeline stage to its real local implementation."""

    name = "local-v1-pipeline"

    def __init__(
        self,
        *,
        parsing_executor: PipelineExecutor | None = None,
        translation_executor: PipelineExecutor | None = None,
        reflow_executor: PipelineExecutor | None = None,
        validation_executor: PipelineExecutor | None = None,
        progress_sink: Callable[[str], None] | None = None,
    ):
        self.executors: dict[PipelineStage, PipelineExecutor] = {
            PipelineStage.PARSING: parsing_executor
            or MacMinerUParsingExecutor(progress_sink=progress_sink),
            PipelineStage.TRANSLATING: translation_executor
            or TranslationStageExecutor(
                simulate=False,
                progress_sink=progress_sink,
            ),
            PipelineStage.REFLOWING: reflow_executor
            or ReflowStageExecutor(progress_sink=progress_sink),
            PipelineStage.VALIDATING: validation_executor
            or PDFValidationStageExecutor(),
        }

    def execute(
        self,
        context: StageContext,
        *,
        progress: ProgressCallback,
        is_cancel_requested: CancellationCheck,
    ) -> StageExecutionResult:
        executor = self.executors.get(context.stage)
        if executor is None:
            raise StageExecutionError(
                f"no local executor is registered for {context.stage.value}",
                code="stage_not_supported",
            )
        return executor.execute(
            context,
            progress=progress,
            is_cancel_requested=is_cancel_requested,
        )
