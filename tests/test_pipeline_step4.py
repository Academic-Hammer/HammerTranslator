from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import fitz

from hammer_translator.pipeline.executor import StageContext, StageExecutionError
from hammer_translator.pipeline.reflow_executor import ReflowStageExecutor
from hammer_translator.pipeline.runner import PipelineRunner
from hammer_translator.pipeline.schema import (
    JobStatus,
    PipelineConfig,
    PipelineStage,
    ProducedArtifact,
    StageExecutionResult,
    StageStatus,
)
from hammer_translator.pipeline.storage import LocalJobStore
from hammer_translator.pipeline.translation_executor import TranslationStageExecutor
from hammer_translator.reflow.schema import ReflowReportOutput


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


class ReflowStageExecutorTests(unittest.TestCase):
    def test_command_uses_registered_artifact_and_partial_policy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            translation = root / "translation" / "paper_translated.json"
            translation.parent.mkdir()
            translation.write_text("{}", encoding="utf-8")
            context = StageContext(
                job_id="00000000-0000-0000-0000-000000000001",
                job_dir=root,
                stage=PipelineStage.REFLOWING,
                attempt=1,
                source_pdf=_pdf(root / "source.pdf"),
                document_name="paper.pdf",
                config=PipelineConfig(allow_partial_pdf=True).model_dump(mode="json"),
            )
            command = ReflowStageExecutor().build_command(
                context,
                translation,
                root / "output" / "translated.pdf",
                root / "output" / "reflow_feedback.json",
            )

            self.assertEqual(command[2:4], ["-m", "hammer_translator.reflow"])
            self.assertEqual(command[4], str(translation))
            self.assertIn("--allow-partial", command)

    def test_missing_registered_input_is_rejected_without_directory_guessing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            guessed = root / "translation" / "guessed_translated.json"
            guessed.parent.mkdir()
            guessed.write_text("{}", encoding="utf-8")
            context = StageContext(
                job_id="00000000-0000-0000-0000-000000000001",
                job_dir=root,
                stage=PipelineStage.REFLOWING,
                attempt=1,
                source_pdf=_pdf(root / "source.pdf"),
                document_name="paper.pdf",
                config=PipelineConfig().model_dump(mode="json"),
            )

            with self.assertRaisesRegex(StageExecutionError, "exactly one registered"):
                ReflowStageExecutor().execute(
                    context,
                    progress=lambda *_: None,
                    is_cancel_requested=lambda: False,
                )

    def test_existing_renderer_creates_registered_pdf_and_queues_validation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = LocalJobStore(root / "jobs")
            job = store.create_job(_pdf(root / "paper.pdf"))
            PipelineRunner(store, FixtureParsingExecutor()).run_stage(
                job.job_id, PipelineStage.PARSING
            )
            translated = PipelineRunner(
                store,
                TranslationStageExecutor(
                    simulate=True,
                    poll_interval=0.02,
                    heartbeat_interval=1.0,
                ),
            ).run_stage(job.job_id, PipelineStage.TRANSLATING)
            self.assertEqual(translated.status, JobStatus.QUEUED)

            reflowed = PipelineRunner(
                store,
                ReflowStageExecutor(
                    poll_interval=0.02,
                    heartbeat_interval=1.0,
                ),
            ).run_stage(job.job_id, PipelineStage.REFLOWING)

            self.assertEqual(reflowed.status, JobStatus.QUEUED)
            self.assertEqual(
                reflowed.stage_record(PipelineStage.REFLOWING).status,
                StageStatus.COMPLETED_WITH_WARNINGS,
            )
            self.assertEqual(
                reflowed.stage_record(PipelineStage.VALIDATING).status,
                StageStatus.PENDING,
            )
            by_kind = {artifact.kind: artifact for artifact in reflowed.artifacts}
            self.assertIn("translated_pdf", by_kind)
            self.assertIn("reflow_feedback", by_kind)
            self.assertIn("reflow_stage_manifest", by_kind)
            self.assertFalse(by_kind["translated_pdf"].metadata["publishable"])

            output_pdf = store.resolve_artifact(
                job.job_id, by_kind["translated_pdf"].relative_path
            )
            source = fitz.open(store.job_dir(job.job_id) / "input" / "source.pdf")
            output = fitz.open(output_pdf)
            try:
                self.assertEqual(output.page_count, source.page_count)
                self.assertEqual(output[0].rect, source[0].rect)
            finally:
                output.close()
                source.close()

            feedback_path = store.resolve_artifact(
                job.job_id, by_kind["reflow_feedback"].relative_path
            )
            feedback = ReflowReportOutput.model_validate_json(
                feedback_path.read_text(encoding="utf-8")
            )
            self.assertEqual(feedback.status, "rendered")
            self.assertEqual(feedback.summary.rendered, 1)

            manifest = json.loads(
                store.resolve_artifact(
                    job.job_id,
                    by_kind["reflow_stage_manifest"].relative_path,
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["output_pdf"], "output/translated.pdf")
            self.assertFalse(manifest["publishable"])
            self.assertIn(
                "reflow_simulated_translation",
                [warning.code for warning in reflowed.warnings],
            )


if __name__ == "__main__":
    unittest.main()
