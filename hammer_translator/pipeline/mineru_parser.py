"""Real MinerU parsing stage for local macOS and Linux GPU runtimes."""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .executor import (
    CancellationCheck,
    ProgressCallback,
    StageContext,
    StageExecutionError,
)
from .process import CancellableProcessRunner
from .schema import PipelineStage, ProducedArtifact, StageExecutionResult, WarningRecord


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARSER_MODULE = "hammer_translator.parsing"
PARSE_MANIFEST_SCHEMA_VERSION = "mineru-parse-v1"


def safe_document_stem(display_name: str) -> str:
    name = Path(display_name).name
    stem = Path(name).stem if Path(name).suffix.lower() == ".pdf" else name
    normalized = re.sub(r"[^\w.-]+", "_", stem, flags=re.UNICODE).strip("._")
    if not normalized:
        return "document"
    return normalized


def parse_mode(backend: str, method: str) -> str:
    if backend == "pipeline":
        return method
    if backend == "vlm-engine":
        return "vlm"
    if backend == "hybrid-engine":
        return f"hybrid_{method}"
    raise StageExecutionError(
        f"unsupported parser backend: {backend}",
        code="invalid_parser_config",
    )


class MinerUParsingExecutor:
    """Run the packaged MinerU entry point in a cancellable child process."""

    name = "mineru-parser"

    def __init__(
        self,
        *,
        python_executable: str | Path | None = None,
        parser_module: str = PARSER_MODULE,
        poll_interval: float = 0.25,
        heartbeat_interval: float = 5.0,
        progress_sink: Callable[[str], None] | None = None,
    ):
        self.python_executable = str(python_executable or sys.executable)
        self.parser_module = parser_module
        self.process_runner = CancellableProcessRunner(
            poll_interval=poll_interval,
            heartbeat_interval=heartbeat_interval,
            progress_sink=progress_sink,
        )

    @staticmethod
    def _bool_flag(name: str, enabled: bool) -> str:
        return f"--{name}" if enabled else f"--no-{name}"

    def build_command(self, context: StageContext, document_name: str) -> list[str]:
        config = context.config
        device = str(config["parser_device"])
        if device == "auto" and config.get("environment_profile") == "linux_gpu":
            device = "cuda"
        command = [
            self.python_executable,
            "-u",
            "-m",
            self.parser_module,
            str(context.source_pdf),
            "--output",
            str(context.job_dir / "parse"),
            "--document-name",
            document_name,
            "--backend",
            str(config["parser_backend"]),
            "--device",
            device,
            "--method",
            str(config["parser_method"]),
            "--lang",
            str(config["parser_language"]),
            "--effort",
            str(config["parser_effort"]),
            "--start-page",
            str(config["parser_start_page"]),
            "--render-threads",
            str(config["parser_render_threads"]),
            "--window-size",
            str(config["parser_window_size"]),
            self._bool_flag("formula", bool(config["parser_formula"])),
            self._bool_flag("table", bool(config["parser_table"])),
            self._bool_flag("formula-ch", bool(config["parser_formula_ch"])),
            self._bool_flag(
                "image-analysis", bool(config["parser_image_analysis"])
            ),
            "--adapt",
            "--no-skip-existing",
            "--no-continue-on-error",
        ]
        if config.get("parser_end_page") is not None:
            command.extend(["--end-page", str(config["parser_end_page"])])
        return command

    def _run_process(
        self,
        context: StageContext,
        document_name: str,
        *,
        progress: ProgressCallback,
        is_cancel_requested: CancellationCheck,
    ) -> None:
        result = self.process_runner.run(
            self.build_command(context, document_name),
            cwd=PROJECT_ROOT,
            log_path=context.job_dir / "logs" / "parsing.log",
            executor_name=self.name,
            heartbeat_message="MinerU parsing is still running",
            start_error_message="could not start the MinerU parser",
            start_error_code="parser_process_start_failed",
            progress=progress,
            is_cancel_requested=is_cancel_requested,
        )
        if result.return_code != 0:
            raise StageExecutionError(
                f"MinerU parser exited with code {result.return_code}",
                code="mineru_parse_failed",
                retryable=True,
                detail={
                    "return_code": result.return_code,
                    "log_tail": list(result.log_tail),
                },
            )

    @staticmethod
    def _read_manifest(path: Path, expected: dict[str, str]) -> dict[str, Any]:
        if not path.is_file():
            raise StageExecutionError(
                f"MinerU parse manifest is missing: {path.name}",
                code="parse_contract_missing",
                retryable=True,
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StageExecutionError(
                "MinerU parse manifest is unreadable",
                code="parse_contract_invalid",
            ) from exc
        if not isinstance(payload, dict):
            raise StageExecutionError(
                "MinerU parse manifest must be a JSON object",
                code="parse_contract_invalid",
            )
        for key, value in expected.items():
            if payload.get(key) != value:
                raise StageExecutionError(
                    f"MinerU parse manifest has an unexpected {key}",
                    code="parse_contract_invalid",
                    detail={"field": key},
                )
        warnings = payload.get("adapter_warnings", [])
        if not isinstance(warnings, list) or not all(
            isinstance(item, str) for item in warnings
        ):
            raise StageExecutionError(
                "MinerU parse manifest has invalid adapter_warnings",
                code="parse_contract_invalid",
            )
        return payload

    @staticmethod
    def _relative(context: StageContext, path: Path) -> str:
        try:
            return path.resolve().relative_to(context.job_dir.resolve()).as_posix()
        except ValueError as exc:
            raise StageExecutionError(
                "MinerU artifact escaped the pipeline job directory",
                code="parse_artifact_unsafe",
            ) from exc

    def _collect_artifacts(
        self,
        context: StageContext,
        document_name: str,
        result_dir: Path,
        manifest_path: Path,
    ) -> list[ProducedArtifact]:
        files: Sequence[tuple[str, str, Path, str]] = (
            ("parse_manifest", "parse_manifest", manifest_path, "application/json"),
            (
                "middle_json",
                "mineru_middle_json",
                result_dir / f"{document_name}_middle.json",
                "application/json",
            ),
            (
                "content_list",
                "adapted_content_list",
                result_dir / f"{document_name}_content_list.json",
                "application/json",
            ),
            (
                "enriched_json",
                "enriched_content_list",
                result_dir / f"{document_name}_enriched.json",
                "application/json",
            ),
            (
                "origin_pdf",
                "parsed_origin_pdf",
                result_dir / f"{document_name}_origin.pdf",
                "application/pdf",
            ),
        )
        artifacts: list[ProducedArtifact] = []
        for name, kind, path, media_type in files:
            if not path.is_file():
                raise StageExecutionError(
                    f"required MinerU artifact is missing: {path.name}",
                    code="parse_contract_missing",
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

        optional_files: Sequence[tuple[str, str, Path, str]] = (
            (
                "mineru_content_list",
                "mineru_native_content_list",
                result_dir / f"{document_name}_mineru_content_list.json",
                "application/json",
            ),
            (
                "model_output",
                "mineru_model_output",
                result_dir / f"{document_name}_model.json",
                "application/json",
            ),
            (
                "markdown",
                "mineru_markdown",
                result_dir / f"{document_name}.md",
                "text/markdown",
            ),
        )
        for name, kind, path, media_type in optional_files:
            if path.is_file():
                artifacts.append(
                    ProducedArtifact(
                        name=name,
                        kind=kind,
                        relative_path=self._relative(context, path),
                        media_type=media_type,
                    )
                )
        images = result_dir / "images"
        if images.is_dir():
            artifacts.append(
                ProducedArtifact(
                    name="images",
                    kind="mineru_images",
                    relative_path=self._relative(context, images),
                    node_type="directory",
                )
            )
        log_path = context.job_dir / "logs" / "parsing.log"
        if log_path.is_file():
            artifacts.append(
                ProducedArtifact(
                    name="parsing_log",
                    kind="stage_log",
                    relative_path=self._relative(context, log_path),
                    media_type="text/plain",
                )
            )
        artifacts.append(
            ProducedArtifact(
                name="parse_result_directory",
                kind="parse_result_directory",
                relative_path=self._relative(context, result_dir),
                node_type="directory",
            )
        )
        return artifacts

    def execute(
        self,
        context: StageContext,
        *,
        progress: ProgressCallback,
        is_cancel_requested: CancellationCheck,
    ) -> StageExecutionResult:
        if context.stage is not PipelineStage.PARSING:
            raise StageExecutionError(
                f"{self.name} cannot execute {context.stage.value}",
                code="stage_not_supported",
            )
        document_name = safe_document_stem(context.document_name)
        backend = str(context.config["parser_backend"])
        method = str(context.config["parser_method"])
        result_dir = (
            context.job_dir / "parse" / document_name / parse_mode(backend, method)
        )
        progress(
            0,
            None,
            "starting MinerU parser process",
            {"executor": self.name, "backend": backend, "method": method},
        )
        self._run_process(
            context,
            document_name,
            progress=progress,
            is_cancel_requested=is_cancel_requested,
        )
        manifest_path = result_dir / f"{document_name}_parse_manifest.json"
        manifest = self._read_manifest(
            manifest_path,
            {
                "schema_version": PARSE_MANIFEST_SCHEMA_VERSION,
                "document_name": document_name,
                "backend": backend,
                "method": method,
            },
        )
        artifacts = self._collect_artifacts(
            context,
            document_name,
            result_dir,
            manifest_path,
        )
        warnings = [
            WarningRecord(
                code="mineru_adapter_warning",
                message=message[:2000],
                stage=PipelineStage.PARSING,
            )
            for message in manifest["adapter_warnings"]
        ]
        progress(
            1,
            1,
            "MinerU parsing completed",
            {
                "executor": self.name,
                "artifact_count": len(artifacts),
                "warning_count": len(warnings),
            },
        )
        return StageExecutionResult(
            artifacts=artifacts,
            warnings=warnings,
            metadata={
                "executor": self.name,
                "backend": backend,
                "method": method,
                "document_name": document_name,
                "result_directory": self._relative(context, result_dir),
            },
        )
