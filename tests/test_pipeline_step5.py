from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import fitz

from hammer_translator.pipeline.cli import main
from hammer_translator.pipeline.executor import StageContext, StageExecutionError
from hammer_translator.pipeline.reflow_executor import ReflowStageExecutor
from hammer_translator.pipeline.runner import PipelineRunner
from hammer_translator.pipeline.schema import (
    EventType,
    JobStatus,
    PipelineConfig,
    PipelineStage,
    ProducedArtifact,
    StageExecutionResult,
    StageStatus,
)
from hammer_translator.pipeline.storage import LocalJobStore
from hammer_translator.pipeline.translation_executor import TranslationStageExecutor
from hammer_translator.pipeline.validation_executor import PDFValidationStageExecutor
from hammer_translator.pipeline.validation_schema import PDFValidationReportOutput


def _pdf(path: Path) -> Path:
    document = fitz.open()
    page = document.new_page(width=300, height=400)
    page.insert_text((30, 50), "Hello world.")
    document.save(path)
    document.close()
    return path


class FixtureParsingExecutor:
    name = "fixture-parser"

    def execute(self, context, *, progress, is_cancel_requested):
        output = context.job_dir / "parse" / "paper" / "auto"
        output.mkdir(parents=True)
        enriched = output / "paper_enriched.json"
        enriched.write_text(
            json.dumps(
                [
                    {
                        "type": "text",
                        "text": "Hello world.",
                        "page_idx": 0,
                        "bbox": [30, 35, 200, 65],
                        "logical_para_id": 0,
                    }
                ]
            ),
            encoding="utf-8",
        )
        progress(1, 1, "fixture parsing complete", None)
        return StageExecutionResult(
            artifacts=[
                ProducedArtifact(
                    name="enriched_json",
                    kind="enriched_content_list",
                    relative_path="parse/paper/auto/paper_enriched.json",
                    media_type="application/json",
                )
            ]
        )


def _reflowed_job(root: Path):
    store = LocalJobStore(root / "jobs")
    job = store.create_job(_pdf(root / "paper.pdf"))
    PipelineRunner(store, FixtureParsingExecutor()).run_stage(
        job.job_id, PipelineStage.PARSING
    )
    PipelineRunner(
        store,
        TranslationStageExecutor(
            simulate=True,
            poll_interval=0.02,
            heartbeat_interval=1.0,
        ),
    ).run_stage(job.job_id, PipelineStage.TRANSLATING)
    reflowed = PipelineRunner(
        store,
        ReflowStageExecutor(
            poll_interval=0.02,
            heartbeat_interval=1.0,
        ),
    ).run_stage(job.job_id, PipelineStage.REFLOWING)
    if reflowed.stage_record(PipelineStage.REFLOWING).status not in {
        StageStatus.COMPLETED,
        StageStatus.COMPLETED_WITH_WARNINGS,
    }:
        raise AssertionError("fixture Reflow stage failed")
    return store, job


class PDFValidationStageExecutorTests(unittest.TestCase):
    def test_missing_registered_reflow_artifacts_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            context = StageContext(
                job_id="00000000-0000-0000-0000-000000000001",
                job_dir=root,
                stage=PipelineStage.VALIDATING,
                attempt=1,
                source_pdf=_pdf(root / "source.pdf"),
                document_name="paper.pdf",
                config=PipelineConfig().model_dump(mode="json"),
            )

            with self.assertRaisesRegex(StageExecutionError, "exactly one registered"):
                PDFValidationStageExecutor().execute(
                    context,
                    progress=lambda *_: None,
                    is_cancel_requested=lambda: False,
                )

    def test_validation_cli_completes_job_and_writes_versioned_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store, job = _reflowed_job(root)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = main(
                    [
                        "--jobs-root",
                        str(root / "jobs"),
                        "validate",
                        job.job_id,
                    ]
                )

            payload = json.loads(stdout.getvalue())
            self.assertEqual(code, 0)
            self.assertEqual(payload["status"], "completed_with_warnings")
            self.assertIn("validation stage: completed_with_warnings", stderr.getvalue())
            completed = store.load_job(job.job_id)
            self.assertEqual(completed.status, JobStatus.COMPLETED_WITH_WARNINGS)
            self.assertEqual(
                completed.stage_record(PipelineStage.VALIDATING).status,
                StageStatus.COMPLETED_WITH_WARNINGS,
            )
            self.assertEqual(
                list(store.iter_events(job.job_id))[-1].event_type,
                EventType.JOB_COMPLETED,
            )
            report_artifact = next(
                artifact
                for artifact in completed.artifacts
                if artifact.kind == "validation_report"
            )
            report = PDFValidationReportOutput.model_validate_json(
                store.resolve_artifact(
                    job.job_id, report_artifact.relative_path
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(report.schema_version, "pipeline-validation-v1")
            self.assertEqual(report.status, "passed_with_warnings")
            self.assertFalse(report.publishable)
            self.assertEqual(report.page_count, 1)
            self.assertEqual(report.summary.failed, 0)
            self.assertIn(
                "validation_not_publishable",
                [warning.code for warning in completed.warnings],
            )

    def test_registered_pdf_tampering_fails_validation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store, job = _reflowed_job(root)
            reflowed = store.load_job(job.job_id)
            pdf_artifact = next(
                artifact for artifact in reflowed.artifacts if artifact.kind == "translated_pdf"
            )
            output_pdf = store.resolve_artifact(job.job_id, pdf_artifact.relative_path)
            with output_pdf.open("ab") as stream:
                stream.write(b"\n% post-registration change\n")

            failed = PipelineRunner(store, PDFValidationStageExecutor()).run_stage(
                job.job_id, PipelineStage.VALIDATING
            )

            self.assertEqual(failed.status, JobStatus.FAILED)
            self.assertEqual(failed.error.code, "pdf_validation_failed")
            self.assertIn(
                "registered_artifact_hashes",
                failed.error.detail["failed_checks"],
            )
            report = PDFValidationReportOutput.model_validate_json(
                (store.job_dir(job.job_id) / "output" / "validation_report.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(report.status, "failed")
            self.assertFalse(report.publishable)


if __name__ == "__main__":
    unittest.main()
