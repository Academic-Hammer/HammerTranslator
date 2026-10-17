"""Independent final validation gate for pipeline-produced PDFs."""

from __future__ import annotations

import hashlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import fitz
from pydantic import ValidationError

from hammer_translator.reflow.schema import ReflowReportOutput
from hammer_translator.translation.jsonio import write_json_atomic

from .executor import (
    CancellationCheck,
    ProgressCallback,
    StageCancelledError,
    StageContext,
    StageExecutionError,
)
from .schema import ArtifactRecord, PipelineStage, ProducedArtifact, StageExecutionResult, WarningRecord
from .validation_schema import (
    PDFValidationReportOutput,
    VALIDATION_REPORT_VERSION,
    ValidationCheckOutput,
    ValidationSummaryOutput,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PDFValidationStageExecutor:
    """Validate registered Reflow artifacts without modifying the PDF."""

    name = "pdf-validation-stage"

    @staticmethod
    def _job_path(context: StageContext, artifact: ArtifactRecord) -> Path:
        path = (context.job_dir / artifact.relative_path).resolve()
        if context.job_dir.resolve() not in path.parents:
            raise StageExecutionError(
                "registered validation input escaped the job directory",
                code="validation_input_unsafe",
            )
        return path

    @staticmethod
    def _one_artifact(context: StageContext, kind: str, stage: PipelineStage) -> ArtifactRecord:
        matches = [
            artifact
            for artifact in context.artifacts
            if artifact.kind == kind and artifact.stage is stage
        ]
        if len(matches) != 1:
            raise StageExecutionError(
                f"validation requires exactly one registered {kind}",
                code="validation_input_contract_invalid",
                detail={"kind": kind, "candidate_count": len(matches)},
            )
        return matches[0]

    @staticmethod
    def _load_feedback(path: Path) -> ReflowReportOutput:
        try:
            return ReflowReportOutput.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError, ValueError) as exc:
            raise StageExecutionError(
                "registered Reflow feedback failed schema validation",
                code="validation_input_contract_invalid",
            ) from exc

    @staticmethod
    def _relative(context: StageContext, path: Path) -> str:
        try:
            return path.resolve().relative_to(context.job_dir.resolve()).as_posix()
        except ValueError as exc:
            raise StageExecutionError(
                "validation artifact escaped the pipeline job directory",
                code="validation_artifact_unsafe",
            ) from exc

    @staticmethod
    def _check(
        name: str,
        passed: bool,
        passed_message: str,
        failed_message: str,
        *,
        warning: bool = False,
        detail: dict | None = None,
    ) -> ValidationCheckOutput:
        if passed:
            status = "warning" if warning else "passed"
            message = failed_message if warning else passed_message
        else:
            status = "failed"
            message = failed_message
        return ValidationCheckOutput(
            name=name,
            status=status,
            message=message,
            detail=detail or {},
        )

    def _validate_hash(
        self,
        artifact: ArtifactRecord,
        path: Path,
    ) -> tuple[bool, str | None]:
        if not path.is_file():
            return False, None
        actual = _sha256(path)
        return artifact.sha256 == actual, actual

    def _inspect_pdf(
        self,
        source_pdf: Path,
        translated_pdf: Path,
        *,
        progress: ProgressCallback,
        is_cancel_requested: CancellationCheck,
    ) -> tuple[int, list[ValidationCheckOutput]]:
        checks: list[ValidationCheckOutput] = []
        try:
            source = fitz.open(source_pdf)
            translated = fitz.open(translated_pdf)
        except Exception as exc:
            raise StageExecutionError(
                f"could not open PDF for validation: {type(exc).__name__}",
                code="validation_pdf_unreadable",
            ) from exc
        try:
            readable = not source.needs_pass and not translated.needs_pass
            checks.append(
                self._check(
                    "pdf_readable",
                    readable,
                    "source and translated PDFs opened without encryption",
                    "source or translated PDF requires a password",
                )
            )
            page_count_matches = source.page_count == translated.page_count
            checks.append(
                self._check(
                    "page_count_unchanged",
                    page_count_matches and source.page_count > 0,
                    "translated PDF page count matches the source",
                    "translated PDF page count differs from the source",
                    detail={
                        "source": source.page_count,
                        "translated": translated.page_count,
                    },
                )
            )
            comparable_pages = min(source.page_count, translated.page_count)
            size_mismatches: list[int] = []
            rotation_mismatches: list[int] = []
            render_failures: list[dict[str, str | int]] = []
            for index in range(comparable_pages):
                if is_cancel_requested():
                    raise StageCancelledError()
                source_page = source[index]
                translated_page = translated[index]
                if source_page.rect != translated_page.rect:
                    size_mismatches.append(index)
                if source_page.rotation != translated_page.rotation:
                    rotation_mismatches.append(index)
                try:
                    pixmap = translated_page.get_pixmap(
                        matrix=fitz.Matrix(0.5, 0.5),
                        colorspace=fitz.csRGB,
                        alpha=False,
                    )
                    if pixmap.width <= 0 or pixmap.height <= 0 or not pixmap.samples:
                        raise ValueError("empty raster output")
                except Exception as exc:
                    render_failures.append(
                        {"page_index": index, "error_type": type(exc).__name__}
                    )
                progress(
                    index + 1,
                    comparable_pages,
                    f"validating PDF page {index + 1}/{comparable_pages}",
                    {"executor": self.name},
                )
            checks.append(
                self._check(
                    "page_geometry_unchanged",
                    not size_mismatches and not rotation_mismatches,
                    "all page sizes and rotations match the source",
                    "one or more page geometries changed",
                    detail={
                        "size_mismatch_pages": size_mismatches,
                        "rotation_mismatch_pages": rotation_mismatches,
                    },
                )
            )
            checks.append(
                self._check(
                    "all_pages_renderable",
                    not render_failures and comparable_pages > 0,
                    "every translated PDF page passed a raster render smoke test",
                    "one or more translated PDF pages could not be rasterized",
                    detail={"failures": render_failures},
                )
            )
            return translated.page_count, checks
        finally:
            translated.close()
            source.close()

    @staticmethod
    def _feedback_checks(report: ReflowReportOutput) -> list[ValidationCheckOutput]:
        rendered_failures: list[str] = []
        failed_blocks: list[str] = []
        review_blocks: list[str] = []
        for item in report.items:
            if item.status == "failed":
                failed_blocks.append(item.block_id)
            elif item.status == "needs_review":
                review_blocks.append(item.block_id)
            if item.status != "rendered":
                continue
            verification = item.verification or {}
            if not (
                verification.get("text_layer_nonempty")
                and verification.get("bbox_unchanged")
                and verification.get("page_size_unchanged")
            ):
                rendered_failures.append(item.block_id)

        checks = [
            PDFValidationStageExecutor._check(
                "rendered_items_verified",
                not rendered_failures and not failed_blocks,
                "all rendered items passed text-layer and geometry verification",
                "one or more rendered items failed verification",
                detail={
                    "rendered_failure_block_ids": rendered_failures[:50],
                    "failed_block_ids": failed_blocks[:50],
                },
            )
        ]
        formula = report.formula_verification
        formula_passed = bool(
            formula is not None
            and formula.expected == formula.drawn
            and formula.order_verified
            and formula.bbox_verified
            and formula.baseline_verified
            and formula.overlap_count == 0
            and formula.nonempty_verified
            and formula.vector_only
        )
        checks.append(
            PDFValidationStageExecutor._check(
                "formula_output_verified",
                formula_passed,
                "all expected formulas passed vector-output verification",
                "formula output verification failed",
                detail=(formula.model_dump(mode="json") if formula is not None else {}),
            )
        )
        checks.append(
            PDFValidationStageExecutor._check(
                "review_items_preserved",
                True,
                "no unresolved review items remain",
                "unresolved items remain preserved from the source PDF",
                warning=bool(review_blocks),
                detail={"review_item_count": len(review_blocks), "block_ids": review_blocks[:50]},
            )
        )
        return checks

    def execute(
        self,
        context: StageContext,
        *,
        progress: ProgressCallback,
        is_cancel_requested: CancellationCheck,
    ) -> StageExecutionResult:
        if context.stage is not PipelineStage.VALIDATING:
            raise StageExecutionError(
                f"{self.name} cannot execute {context.stage.value}",
                code="stage_not_supported",
            )
        pdf_artifact = self._one_artifact(
            context, "translated_pdf", PipelineStage.REFLOWING
        )
        feedback_artifact = self._one_artifact(
            context, "reflow_feedback", PipelineStage.REFLOWING
        )
        source_artifacts = [
            artifact for artifact in context.artifacts if artifact.kind == "source_pdf"
        ]
        if len(source_artifacts) != 1:
            raise StageExecutionError(
                "validation requires exactly one registered source_pdf",
                code="validation_input_contract_invalid",
                detail={"candidate_count": len(source_artifacts)},
            )
        source_artifact = source_artifacts[0]
        translated_pdf = self._job_path(context, pdf_artifact)
        feedback_path = self._job_path(context, feedback_artifact)
        source_pdf = self._job_path(context, source_artifact)
        progress(0, None, "starting final PDF validation", {"executor": self.name})

        source_hash_ok, source_hash = self._validate_hash(source_artifact, source_pdf)
        output_hash_ok, output_hash = self._validate_hash(pdf_artifact, translated_pdf)
        checks = [
            self._check(
                "registered_artifact_hashes",
                source_hash_ok and output_hash_ok,
                "source and translated PDF hashes match their registered artifacts",
                "source or translated PDF changed after artifact registration",
                detail={
                    "source_hash_matches": source_hash_ok,
                    "translated_hash_matches": output_hash_ok,
                },
            )
        ]
        if source_hash is None or output_hash is None:
            raise StageExecutionError(
                "registered PDF input is missing",
                code="validation_input_missing",
                retryable=True,
            )
        feedback = self._load_feedback(feedback_path)
        expected_output = (
            Path(feedback.output_pdf).resolve() if feedback.output_pdf is not None else None
        )
        links_match = (
            expected_output == translated_pdf.resolve()
            and Path(feedback.source_pdf).resolve() == source_pdf.resolve()
        )
        checks.append(
            self._check(
                "reflow_provenance_links",
                links_match,
                "Reflow feedback points to the registered source and output PDFs",
                "Reflow feedback provenance does not match registered artifacts",
            )
        )
        page_count, pdf_checks = self._inspect_pdf(
            source_pdf,
            translated_pdf,
            progress=progress,
            is_cancel_requested=is_cancel_requested,
        )
        checks.extend(pdf_checks)
        checks.extend(self._feedback_checks(feedback))

        publishable_candidate = bool(pdf_artifact.metadata.get("publishable"))
        checks.append(
            self._check(
                "publishable_translation_source",
                True,
                "Reflow output was produced from a real provider translation",
                "Reflow output uses a simulated translation and is not publishable",
                warning=not publishable_candidate,
            )
        )
        counts = Counter(check.status for check in checks)
        summary = ValidationSummaryOutput(
            checks_total=len(checks),
            passed=counts["passed"],
            warnings=counts["warning"],
            failed=counts["failed"],
        )
        status = (
            "failed"
            if summary.failed
            else "passed_with_warnings"
            if summary.warnings
            else "passed"
        )
        publishable = publishable_candidate and not summary.failed
        review_item_count = sum(
            item.status == "needs_review" for item in feedback.items
        )
        failed_item_count = sum(item.status == "failed" for item in feedback.items)
        report = PDFValidationReportOutput(
            schema_version=VALIDATION_REPORT_VERSION,
            status=status,
            created_at=_utc_now(),
            source_pdf=self._relative(context, source_pdf),
            translated_pdf=self._relative(context, translated_pdf),
            reflow_feedback=self._relative(context, feedback_path),
            source_pdf_sha256=source_hash,
            translated_pdf_sha256=output_hash,
            page_count=page_count,
            reflow_status=feedback.status,
            review_item_count=review_item_count,
            failed_item_count=failed_item_count,
            publishable=publishable,
            summary=summary,
            checks=checks,
        )
        report_path = context.job_dir / "output" / "validation_report.json"
        write_json_atomic(report_path, report.model_dump(mode="json"))
        if summary.failed:
            raise StageExecutionError(
                "final PDF validation failed",
                code="pdf_validation_failed",
                detail={
                    "report": self._relative(context, report_path),
                    "failed_checks": [
                        check.name for check in checks if check.status == "failed"
                    ],
                },
            )

        warnings: list[WarningRecord] = []
        if review_item_count:
            warnings.append(
                WarningRecord(
                    code="validation_review_items_preserved",
                    message=(
                        "validated PDF contains unresolved items preserved from "
                        "the source PDF"
                    ),
                    stage=PipelineStage.VALIDATING,
                    detail={"review_item_count": review_item_count},
                )
            )
        if not publishable_candidate:
            warnings.append(
                WarningRecord(
                    code="validation_not_publishable",
                    message="validated PDF uses simulated translations and cannot be published",
                    stage=PipelineStage.VALIDATING,
                )
            )
        progress(
            page_count,
            page_count,
            "final PDF validation completed",
            {
                "executor": self.name,
                "status": status,
                "publishable": publishable,
                "warning_count": len(warnings),
            },
        )
        return StageExecutionResult(
            artifacts=[
                ProducedArtifact(
                    name="validation_report",
                    kind="validation_report",
                    relative_path=self._relative(context, report_path),
                    media_type="application/json",
                    metadata={"status": status, "publishable": publishable},
                )
            ],
            warnings=warnings,
            metadata={
                "executor": self.name,
                "status": status,
                "publishable": publishable,
                "page_count": page_count,
                "network_requests": 0,
            },
        )
