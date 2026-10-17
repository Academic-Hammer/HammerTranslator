"""Real Reflow Stage adapter for the unified pipeline."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from hammer_translator.reflow.schema import ReflowReportOutput
from hammer_translator.translation.jsonio import write_json_atomic
from hammer_translator.translation.output_schema import TranslationArtifactOutput

from .executor import (
    CancellationCheck,
    ProgressCallback,
    StageContext,
    StageExecutionError,
)
from .process import CancellableProcessRunner
from .schema import PipelineStage, ProducedArtifact, StageExecutionResult, WarningRecord


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFLOW_SCRIPT = PROJECT_ROOT / "render_translated_pdf.py"
REFLOW_STAGE_MANIFEST_VERSION = "pipeline-reflow-v1"


class ReflowStageExecutor:
    """Render one registered Translation artifact into a job-local PDF."""

    name = "reflow-stage"

    def __init__(
        self,
        *,
        python_executable: str | Path | None = None,
        reflow_script: str | Path = REFLOW_SCRIPT,
        poll_interval: float = 0.25,
        heartbeat_interval: float = 5.0,
        progress_sink: Callable[[str], None] | None = None,
    ):
        self.python_executable = str(python_executable or sys.executable)
        self.reflow_script = Path(reflow_script).expanduser().resolve()
        self.process_runner = CancellableProcessRunner(
            poll_interval=poll_interval,
            heartbeat_interval=heartbeat_interval,
            progress_sink=progress_sink,
        )

    @staticmethod
    def _job_path(context: StageContext, relative_path: str) -> Path:
        path = (context.job_dir / relative_path).resolve()
        if context.job_dir.resolve() not in path.parents:
            raise StageExecutionError(
                "registered Reflow input escaped the job directory",
                code="reflow_input_unsafe",
            )
        return path

    def _translation_input(self, context: StageContext) -> Path:
        matches = [
            artifact
            for artifact in context.artifacts
            if artifact.kind == "translation_artifact"
            and artifact.stage is PipelineStage.TRANSLATING
        ]
        if len(matches) != 1:
            raise StageExecutionError(
                "Reflow requires exactly one registered translation_artifact",
                code="reflow_input_contract_invalid",
                detail={"candidate_count": len(matches)},
            )
        path = self._job_path(context, matches[0].relative_path)
        if not path.is_file():
            raise StageExecutionError(
                "registered Translation artifact is missing",
                code="reflow_input_missing",
                retryable=True,
            )
        return path

    @staticmethod
    def _load_translation(path: Path) -> TranslationArtifactOutput:
        try:
            return TranslationArtifactOutput.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError, ValueError) as exc:
            raise StageExecutionError(
                "Reflow input failed Translation artifact validation",
                code="reflow_input_contract_invalid",
            ) from exc

    @staticmethod
    def _load_feedback(path: Path) -> ReflowReportOutput:
        try:
            return ReflowReportOutput.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError, ValueError) as exc:
            raise StageExecutionError(
                "Reflow feedback failed schema validation",
                code="reflow_contract_invalid",
            ) from exc

    @staticmethod
    def _relative(context: StageContext, path: Path) -> str:
        try:
            return path.resolve().relative_to(context.job_dir.resolve()).as_posix()
        except ValueError as exc:
            raise StageExecutionError(
                "Reflow artifact escaped the pipeline job directory",
                code="reflow_artifact_unsafe",
            ) from exc

    def build_command(
        self,
        context: StageContext,
        translation_path: Path,
        output_pdf: Path,
        feedback_path: Path,
    ) -> list[str]:
        command = [
            self.python_executable,
            "-u",
            str(self.reflow_script),
            str(translation_path),
            "--output",
            str(output_pdf),
            "--feedback",
            str(feedback_path),
        ]
        if context.config["allow_partial_pdf"]:
            command.append("--allow-partial")
        return command

    def _run_process(
        self,
        context: StageContext,
        translation_path: Path,
        output_pdf: Path,
        feedback_path: Path,
        *,
        progress: ProgressCallback,
        is_cancel_requested: CancellationCheck,
    ) -> None:
        if not self.reflow_script.is_file():
            raise StageExecutionError(
                f"Reflow dependency was not found: {self.reflow_script}",
                code="reflow_entrypoint_missing",
            )
        result = self.process_runner.run(
            self.build_command(
                context,
                translation_path,
                output_pdf,
                feedback_path,
            ),
            cwd=PROJECT_ROOT,
            log_path=context.job_dir / "logs" / "reflowing.log",
            executor_name=self.name,
            heartbeat_message="Reflow is still running",
            start_error_message="could not start the Reflow Stage",
            start_error_code="reflow_process_start_failed",
            progress=progress,
            is_cancel_requested=is_cancel_requested,
        )
        if result.return_code == 3:
            raise StageExecutionError(
                "Reflow was blocked by unresolved layout items",
                code="reflow_blocked",
                detail={
                    "allow_partial_pdf": context.config["allow_partial_pdf"],
                    "log_tail": list(result.log_tail),
                },
            )
        if result.return_code != 0:
            raise StageExecutionError(
                f"Reflow Stage exited with code {result.return_code}",
                code="reflow_process_failed",
                retryable=True,
                detail={
                    "return_code": result.return_code,
                    "log_tail": list(result.log_tail),
                },
            )

    def _validate_report_links(
        self,
        context: StageContext,
        report: ReflowReportOutput,
        translation_path: Path,
        output_pdf: Path,
    ) -> None:
        if Path(report.source_translation_artifact).resolve() != translation_path.resolve():
            raise StageExecutionError(
                "Reflow report references a different Translation artifact",
                code="reflow_contract_invalid",
            )
        if Path(report.source_pdf).resolve() != context.source_pdf.resolve():
            raise StageExecutionError(
                "Reflow report references a different source PDF",
                code="reflow_contract_invalid",
            )
        if report.status not in {"rendered", "partial_needs_review"}:
            raise StageExecutionError(
                f"Reflow did not produce a usable PDF: {report.status}",
                code="reflow_contract_invalid",
            )
        if report.output_pdf is None or Path(report.output_pdf).resolve() != output_pdf.resolve():
            raise StageExecutionError(
                "Reflow report does not reference the expected output PDF",
                code="reflow_contract_invalid",
            )
        try:
            with output_pdf.open("rb") as stream:
                pdf_signature = stream.read(5)
        except OSError:
            pdf_signature = b""
        if pdf_signature != b"%PDF-":
            raise StageExecutionError(
                "Reflow output is missing or is not a PDF",
                code="reflow_contract_missing",
                retryable=True,
            )

    def _write_stage_manifest(
        self,
        context: StageContext,
        translation_path: Path,
        output_pdf: Path,
        feedback_path: Path,
        report: ReflowReportOutput,
        *,
        publishable: bool,
    ) -> Path:
        path = context.job_dir / "output" / "pipeline_reflow_manifest.json"
        payload: dict[str, Any] = {
            "schema_version": REFLOW_STAGE_MANIFEST_VERSION,
            "source_translation_artifact": self._relative(context, translation_path),
            "output_pdf": self._relative(context, output_pdf),
            "feedback": self._relative(context, feedback_path),
            "status": report.status,
            "allow_partial": report.allow_partial,
            "publishable": publishable,
            "summary": report.summary.model_dump(mode="json"),
            "formula_verification": (
                report.formula_verification.model_dump(mode="json")
                if report.formula_verification is not None
                else None
            ),
        }
        write_json_atomic(path, payload)
        return path

    def _warnings(
        self,
        translation: TranslationArtifactOutput,
        report: ReflowReportOutput,
    ) -> list[WarningRecord]:
        warnings: list[WarningRecord] = []
        if report.status == "partial_needs_review":
            review_blocks = [
                item.block_id
                for item in report.items
                if item.status in {"needs_review", "failed"}
            ]
            warnings.append(
                WarningRecord(
                    code="reflow_partial_needs_review",
                    message=(
                        "Reflow produced a partial PDF and preserved unresolved "
                        "source regions"
                    ),
                    stage=PipelineStage.REFLOWING,
                    detail={
                        "summary": report.summary.model_dump(mode="json"),
                        "review_block_ids": review_blocks[:50],
                    },
                )
            )
        if translation.translation_run.mode == "simulation":
            warnings.append(
                WarningRecord(
                    code="reflow_simulated_translation",
                    message="Reflow output uses simulated translations and is not publishable",
                    stage=PipelineStage.REFLOWING,
                )
            )
        return warnings

    def execute(
        self,
        context: StageContext,
        *,
        progress: ProgressCallback,
        is_cancel_requested: CancellationCheck,
    ) -> StageExecutionResult:
        if context.stage is not PipelineStage.REFLOWING:
            raise StageExecutionError(
                f"{self.name} cannot execute {context.stage.value}",
                code="stage_not_supported",
            )
        translation_path = self._translation_input(context)
        translation = self._load_translation(translation_path)
        output_dir = context.job_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_pdf = output_dir / "translated.pdf"
        feedback_path = output_dir / "reflow_feedback.json"
        progress(
            0,
            1,
            "starting Reflow Stage",
            {
                "executor": self.name,
                "allow_partial": context.config["allow_partial_pdf"],
            },
        )
        self._run_process(
            context,
            translation_path,
            output_pdf,
            feedback_path,
            progress=progress,
            is_cancel_requested=is_cancel_requested,
        )
        report = self._load_feedback(feedback_path)
        self._validate_report_links(
            context,
            report,
            translation_path,
            output_pdf,
        )
        publishable = translation.translation_run.mode == "provider"
        manifest_path = self._write_stage_manifest(
            context,
            translation_path,
            output_pdf,
            feedback_path,
            report,
            publishable=publishable,
        )
        warnings = self._warnings(translation, report)
        artifacts = [
            ProducedArtifact(
                name="translated_pdf",
                kind="translated_pdf",
                relative_path=self._relative(context, output_pdf),
                media_type="application/pdf",
                metadata={
                    "document_id": translation.document.document_id,
                    "reflow_status": report.status,
                    "publishable": publishable,
                },
            ),
            ProducedArtifact(
                name="reflow_feedback",
                kind="reflow_feedback",
                relative_path=self._relative(context, feedback_path),
                media_type="application/json",
            ),
            ProducedArtifact(
                name="reflow_stage_manifest",
                kind="reflow_stage_manifest",
                relative_path=self._relative(context, manifest_path),
                media_type="application/json",
            ),
        ]
        log_path = context.job_dir / "logs" / "reflowing.log"
        if log_path.is_file():
            artifacts.append(
                ProducedArtifact(
                    name="reflow_log",
                    kind="stage_log",
                    relative_path=self._relative(context, log_path),
                    media_type="text/plain",
                )
            )
        progress(
            1,
            1,
            "Reflow Stage completed",
            {
                "executor": self.name,
                "status": report.status,
                "publishable": publishable,
                "warning_count": len(warnings),
            },
        )
        return StageExecutionResult(
            artifacts=artifacts,
            warnings=warnings,
            metadata={
                "executor": self.name,
                "status": report.status,
                "publishable": publishable,
                "summary": report.summary.model_dump(mode="json"),
                "network_requests": 0,
            },
        )
