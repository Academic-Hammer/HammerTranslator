"""Real Translation Stage adapter for the unified pipeline."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from hammer_translator.translation.jsonio import write_json_atomic
from hammer_translator.translation.output_schema import TranslationArtifactOutput
from hammer_translator.runtime_paths import child_process_cwd, runtime_paths

from .executor import (
    CancellationCheck,
    ProgressCallback,
    StageContext,
    StageExecutionError,
)
from .process import CancellableProcessRunner
from .schema import PipelineStage, ProducedArtifact, StageExecutionResult, WarningRecord


TRANSLATION_MODULE = "hammer_translator.translation"
TRANSLATION_CONFIG = runtime_paths().config_path
TRANSLATION_ENV_FILE = runtime_paths().env_file
TRANSLATION_STAGE_MANIFEST_VERSION = "pipeline-translation-v1"
UNIT_PROGRESS_RE = re.compile(r"\]\s+(?:layout\s+)?(\d+)/(\d+)\s+")


class TranslationStageExecutor:
    """Execute the existing translation implementation against one registered input."""

    name = "translation-stage"

    def __init__(
        self,
        *,
        simulate: bool = False,
        python_executable: str | Path | None = None,
        translation_module: str = TRANSLATION_MODULE,
        translation_script: str | Path | None = None,
        config_path: str | Path = TRANSLATION_CONFIG,
        env_file: str | Path = TRANSLATION_ENV_FILE,
        poll_interval: float = 0.25,
        heartbeat_interval: float = 5.0,
        progress_sink: Callable[[str], None] | None = None,
    ):
        self.simulate = simulate
        self.python_executable = str(python_executable or sys.executable)
        self.translation_module = translation_module
        self.translation_script = (
            Path(translation_script).expanduser().resolve()
            if translation_script is not None
            else None
        )
        self.config_path = Path(config_path).expanduser().resolve()
        self.env_file = Path(env_file).expanduser().resolve()
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
                "registered translation input escaped the job directory",
                code="translation_input_unsafe",
            )
        return path

    def _translation_input(self, context: StageContext) -> Path:
        matches = [
            artifact
            for artifact in context.artifacts
            if artifact.kind == "enriched_content_list"
            and artifact.stage is PipelineStage.PARSING
        ]
        if len(matches) != 1:
            raise StageExecutionError(
                "translation requires exactly one registered enriched_content_list",
                code="translation_input_contract_invalid",
                detail={"candidate_count": len(matches)},
            )
        path = self._job_path(context, matches[0].relative_path)
        if not path.is_file():
            raise StageExecutionError(
                "registered enriched_content_list is missing",
                code="translation_input_missing",
                retryable=True,
            )
        return path

    def build_command(self, context: StageContext, source_json: Path) -> list[str]:
        config = context.config
        entrypoint = (
            [str(self.translation_script)]
            if self.translation_script is not None
            else ["-m", self.translation_module]
        )
        command = [
            self.python_executable,
            "-u",
            *entrypoint,
            str(source_json),
            "--config",
            str(self.config_path),
            "--env-file",
            str(self.env_file),
            "--provider",
            str(config["translation_provider"]),
            "--output",
            str(context.job_dir / "translation"),
            "--pdf",
            str(context.source_pdf),
            "--source-language",
            str(config["source_language"]),
            "--target-language",
            str(config["target_language"]),
            "--max-attempts",
            str(config["max_translation_attempts"]),
            "--max-retranslation-attempts",
            str(config["max_retranslation_attempts"]),
            "--request-timeout",
            str(config["translation_request_timeout_seconds"]),
            "--simulate" if self.simulate else "--run",
        ]
        if config.get("translation_model"):
            command.extend(["--model", str(config["translation_model"])])
        if context.attempt > 1:
            command.append("--retry-failed")
        return command

    def _run_process(
        self,
        context: StageContext,
        source_json: Path,
        *,
        progress: ProgressCallback,
        is_cancel_requested: CancellationCheck,
    ) -> None:
        dependencies = [(self.config_path, "translation_config_missing")]
        if self.translation_script is not None:
            dependencies.insert(
                0, (self.translation_script, "translation_entrypoint_missing")
            )
        for path, code in dependencies:
            if not path.is_file():
                raise StageExecutionError(
                    f"translation dependency was not found: {path}",
                    code=code,
                )

        def translation_progress(
            current: int,
            total: int | None,
            message: str | None,
            detail: dict | None,
        ) -> None:
            match = UNIT_PROGRESS_RE.search(message or "")
            if match:
                current = int(match.group(1))
                total = int(match.group(2))
            progress(current, total, message, detail)

        result = self.process_runner.run(
            self.build_command(context, source_json),
            cwd=child_process_cwd(context.job_dir),
            log_path=context.job_dir / "logs" / "translating.log",
            executor_name=self.name,
            heartbeat_message="Translation is still running",
            start_error_message="could not start the Translation Stage",
            start_error_code="translation_process_start_failed",
            progress=translation_progress,
            is_cancel_requested=is_cancel_requested,
        )
        if result.return_code != 0:
            raise StageExecutionError(
                f"Translation Stage exited with code {result.return_code}",
                code="translation_process_failed",
                retryable=True,
                detail={
                    "return_code": result.return_code,
                    "log_tail": list(result.log_tail),
                },
            )

    @staticmethod
    def _load_artifact(path: Path) -> TranslationArtifactOutput:
        try:
            return TranslationArtifactOutput.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError, ValueError) as exc:
            raise StageExecutionError(
                "translation output failed schema validation",
                code="translation_contract_invalid",
            ) from exc

    @staticmethod
    def _one_output(output_dir: Path) -> Path:
        candidates = sorted(output_dir.glob("*_translated.json"))
        if len(candidates) != 1:
            raise StageExecutionError(
                "translation must publish exactly one *_translated.json artifact",
                code="translation_contract_missing",
                retryable=True,
                detail={"candidate_count": len(candidates)},
            )
        return candidates[0]

    def _reusable_partial_output(
        self,
        context: StageContext,
        source_json: Path,
    ) -> tuple[Path, TranslationArtifactOutput] | None:
        """Reuse a validated prior artifact when only allowed partial units failed."""

        if context.attempt <= 1 or not context.config["allow_partial_pdf"]:
            return None
        candidates = sorted((context.job_dir / "translation").glob("*_translated.json"))
        if len(candidates) != 1:
            return None
        try:
            artifact = self._load_artifact(candidates[0])
        except StageExecutionError:
            return None
        expected_mode = "simulation" if self.simulate else "provider"
        source_hash = hashlib.sha256(source_json.read_bytes()).hexdigest()
        failed_units = [
            unit
            for unit in artifact.translation_units
            if unit.translation.status == "failed"
        ]
        if not failed_units:
            return None
        if (
            artifact.translation_run.mode != expected_mode
            or artifact.translation_run.requested_provider
            != context.config["translation_provider"]
            or artifact.document.source_hash != source_hash
            or Path(artifact.document.source_pdf).resolve()
            != context.source_pdf.resolve()
        ):
            return None
        return candidates[0], artifact

    @staticmethod
    def _relative(context: StageContext, path: Path) -> str:
        try:
            return path.resolve().relative_to(context.job_dir.resolve()).as_posix()
        except ValueError as exc:
            raise StageExecutionError(
                "translation artifact escaped the pipeline job directory",
                code="translation_artifact_unsafe",
            ) from exc

    def _write_stage_manifest(
        self,
        context: StageContext,
        source_json: Path,
        translation_path: Path,
        artifact: TranslationArtifactOutput,
    ) -> Path:
        path = context.job_dir / "translation" / "pipeline_translation_manifest.json"
        status_counts: dict[str, int] = {}
        for unit in artifact.translation_units:
            status = unit.translation.status
            status_counts[status] = status_counts.get(status, 0) + 1
        payload: dict[str, Any] = {
            "schema_version": TRANSLATION_STAGE_MANIFEST_VERSION,
            "mode": "simulation" if self.simulate else "provider",
            "source_artifact": self._relative(context, source_json),
            "translation_artifact": self._relative(context, translation_path),
            "document_id": artifact.document.document_id,
            "requested_provider": artifact.translation_run.requested_provider,
            "provider": artifact.translation_run.provider,
            "model": artifact.translation_run.model,
            "layout_status": artifact.layout_status,
            "network_requests": artifact.translation_run.network_requests,
            "translation_status_counts": dict(sorted(status_counts.items())),
        }
        write_json_atomic(path, payload)
        return path

    def _collect_artifacts(
        self,
        context: StageContext,
        translation_path: Path,
        artifact: TranslationArtifactOutput,
        stage_manifest: Path,
    ) -> list[ProducedArtifact]:
        output_dir = context.job_dir / "translation"
        document_id = artifact.document.document_id
        required: Sequence[tuple[str, str, Path, str]] = (
            (
                "translation_artifact",
                "translation_artifact",
                translation_path,
                "application/json",
            ),
            (
                "translation_checkpoint",
                "translation_checkpoint",
                output_dir / f"{document_id}_translation_manifest.json",
                "application/json",
            ),
            (
                "translation_stage_manifest",
                "translation_stage_manifest",
                stage_manifest,
                "application/json",
            ),
        )
        artifacts: list[ProducedArtifact] = []
        for name, kind, path, media_type in required:
            if not path.is_file():
                raise StageExecutionError(
                    f"required translation artifact is missing: {path.name}",
                    code="translation_contract_missing",
                    retryable=True,
                )
            artifacts.append(
                ProducedArtifact(
                    name=name,
                    kind=kind,
                    relative_path=self._relative(context, path),
                    media_type=media_type,
                )
            )
        for name, kind, path, media_type in (
            (
                "translation_cache",
                "translation_cache",
                output_dir / "translation_cache.sqlite3",
                "application/vnd.sqlite3",
            ),
            (
                "translation_log",
                "stage_log",
                context.job_dir / "logs" / "translating.log",
                "text/plain",
            ),
        ):
            if path.is_file():
                artifacts.append(
                    ProducedArtifact(
                        name=name,
                        kind=kind,
                        relative_path=self._relative(context, path),
                        media_type=media_type,
                    )
                )
        formula_assets = output_dir / f"{document_id}_formula_assets"
        if formula_assets.is_dir():
            artifacts.append(
                ProducedArtifact(
                    name="formula_assets",
                    kind="formula_assets",
                    relative_path=self._relative(context, formula_assets),
                    node_type="directory",
                )
            )
        return artifacts

    def _warnings(
        self, artifact: TranslationArtifactOutput
    ) -> list[WarningRecord]:
        warnings: list[WarningRecord] = []
        if self.simulate:
            warnings.append(
                WarningRecord(
                    code="translation_simulated",
                    message="translation used FakeProvider and is not publishable",
                    stage=PipelineStage.TRANSLATING,
                )
            )
        review_count = sum(
            unit.translation.status == "needs_review"
            for unit in artifact.translation_units
        )
        failed_count = sum(
            unit.translation.status == "failed" for unit in artifact.translation_units
        )
        if artifact.layout_status != "ready_for_reflow" or review_count or failed_count:
            warnings.append(
                WarningRecord(
                    code="translation_needs_review",
                    message=(
                        "translation contains unresolved review or layout items"
                    ),
                    stage=PipelineStage.TRANSLATING,
                    detail={
                        "layout_status": artifact.layout_status,
                        "needs_review_units": review_count,
                        "failed_units": failed_count,
                    },
                )
            )
        return warnings

    @staticmethod
    def _enforce_failed_unit_policy(
        context: StageContext, failed_units: list[str]
    ) -> None:
        if failed_units and not context.config["allow_partial_pdf"]:
            raise StageExecutionError(
                "Translation Stage contains failed translation units",
                code="translation_units_failed",
                retryable=True,
                detail={
                    "failed_unit_count": len(failed_units),
                    "failed_unit_ids": failed_units[:50],
                },
            )

    def execute(
        self,
        context: StageContext,
        *,
        progress: ProgressCallback,
        is_cancel_requested: CancellationCheck,
    ) -> StageExecutionResult:
        if context.stage is not PipelineStage.TRANSLATING:
            raise StageExecutionError(
                f"{self.name} cannot execute {context.stage.value}",
                code="stage_not_supported",
            )
        source_json = self._translation_input(context)
        progress(
            0,
            None,
            "starting Translation Stage",
            {
                "executor": self.name,
                "provider": context.config["translation_provider"],
                "simulation": self.simulate,
            },
        )
        reusable = self._reusable_partial_output(context, source_json)
        if reusable is None:
            self._run_process(
                context,
                source_json,
                progress=progress,
                is_cancel_requested=is_cancel_requested,
            )
            translation_path = self._one_output(context.job_dir / "translation")
            artifact = self._load_artifact(translation_path)
        else:
            translation_path, artifact = reusable
            progress(
                1,
                1,
                "reusing validated partial Translation artifact",
                {"executor": self.name, "reused_partial_artifact": True},
            )
        expected_mode = "simulation" if self.simulate else "provider"
        if artifact.translation_run.mode != expected_mode:
            raise StageExecutionError(
                "translation output mode does not match the requested execution mode",
                code="translation_contract_invalid",
            )
        failed_units = [
            unit.translation_unit_id
            for unit in artifact.translation_units
            if unit.translation.status == "failed"
        ]
        self._enforce_failed_unit_policy(context, failed_units)
        stage_manifest = self._write_stage_manifest(
            context, source_json, translation_path, artifact
        )
        artifacts = self._collect_artifacts(
            context, translation_path, artifact, stage_manifest
        )
        warnings = self._warnings(artifact)
        progress(
            1,
            1,
            "Translation Stage completed",
            {
                "executor": self.name,
                "artifact_count": len(artifacts),
                "warning_count": len(warnings),
                "layout_status": artifact.layout_status,
            },
        )
        return StageExecutionResult(
            artifacts=artifacts,
            warnings=warnings,
            metadata={
                "executor": self.name,
                "mode": expected_mode,
                "document_id": artifact.document.document_id,
                "layout_status": artifact.layout_status,
                "network_requests": artifact.translation_run.network_requests,
            },
        )
