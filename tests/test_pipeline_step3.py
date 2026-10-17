from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import fitz

from hammer_translator.pipeline.executor import (
    StageContext,
    StageExecutionError,
)
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
from hammer_translator.translation.output_schema import TranslationArtifactOutput


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
                        "bbox": [30, 35, 120, 55],
                        "logical_para_id": 0,
                    }
                ]
            ),
            encoding="utf-8",
        )
        origin = output / "paper_origin.pdf"
        origin.write_bytes(context.source_pdf.read_bytes())
        progress(1, 1, "fixture parsing complete", None)
        return StageExecutionResult(
            artifacts=[
                ProducedArtifact(
                    name="enriched_json",
                    kind="enriched_content_list",
                    relative_path="parse/paper/auto/paper_enriched.json",
                    media_type="application/json",
                ),
                ProducedArtifact(
                    name="origin_pdf",
                    kind="parsed_origin_pdf",
                    relative_path="parse/paper/auto/paper_origin.pdf",
                    media_type="application/pdf",
                ),
            ]
        )


class TranslationStageExecutorTests(unittest.TestCase):
    def test_retry_reuses_matching_partial_artifact(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_json = root / "parse" / "paper_enriched.json"
            source_json.parent.mkdir()
            source_json.write_text("[]", encoding="utf-8")
            source_pdf = _pdf(root / "source.pdf")
            translation_path = root / "translation" / "paper_translated.json"
            translation_path.parent.mkdir()
            translation_path.write_text("{}", encoding="utf-8")
            context = StageContext(
                job_id="00000000-0000-0000-0000-000000000001",
                job_dir=root,
                stage=PipelineStage.TRANSLATING,
                attempt=2,
                source_pdf=source_pdf,
                document_name="paper.pdf",
                config=PipelineConfig(
                    allow_partial_pdf=True,
                    translation_provider="deepseek",
                ).model_dump(mode="json"),
            )
            artifact = MagicMock()
            failed = MagicMock()
            failed.translation.status = "failed"
            artifact.translation_units = [failed]
            artifact.translation_run.mode = "provider"
            artifact.translation_run.requested_provider = "deepseek"
            artifact.document.source_hash = hashlib.sha256(
                source_json.read_bytes()
            ).hexdigest()
            artifact.document.source_pdf = str(source_pdf)
            executor = TranslationStageExecutor(simulate=False)

            with patch.object(executor, "_load_artifact", return_value=artifact):
                reusable = executor._reusable_partial_output(context, source_json)

            self.assertEqual(reusable, (translation_path, artifact))

    def test_retry_does_not_reuse_partial_artifact_from_another_input(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_json = root / "parse" / "paper_enriched.json"
            source_json.parent.mkdir()
            source_json.write_text("[]", encoding="utf-8")
            source_pdf = _pdf(root / "source.pdf")
            translation_path = root / "translation" / "paper_translated.json"
            translation_path.parent.mkdir()
            translation_path.write_text("{}", encoding="utf-8")
            context = StageContext(
                job_id="00000000-0000-0000-0000-000000000001",
                job_dir=root,
                stage=PipelineStage.TRANSLATING,
                attempt=2,
                source_pdf=source_pdf,
                document_name="paper.pdf",
                config=PipelineConfig(
                    allow_partial_pdf=True,
                    translation_provider="deepseek",
                ).model_dump(mode="json"),
            )
            artifact = MagicMock()
            failed = MagicMock()
            failed.translation.status = "failed"
            artifact.translation_units = [failed]
            artifact.translation_run.mode = "provider"
            artifact.translation_run.requested_provider = "deepseek"
            artifact.document.source_hash = "different-source"
            artifact.document.source_pdf = str(source_pdf)
            executor = TranslationStageExecutor(simulate=False)

            with patch.object(executor, "_load_artifact", return_value=artifact):
                reusable = executor._reusable_partial_output(context, source_json)

            self.assertIsNone(reusable)

    def test_failed_units_are_warnings_when_partial_pdf_is_allowed(self):
        artifact = MagicMock()
        failed = MagicMock()
        failed.translation.status = "failed"
        review = MagicMock()
        review.translation.status = "needs_review"
        artifact.translation_units = [failed, review]
        artifact.layout_status = "needs_review"

        warnings = TranslationStageExecutor(simulate=False)._warnings(artifact)

        self.assertEqual([warning.code for warning in warnings], ["translation_needs_review"])
        self.assertEqual(warnings[0].detail["failed_units"], 1)

        context = MagicMock()
        context.config = {"allow_partial_pdf": True}
        TranslationStageExecutor._enforce_failed_unit_policy(context, ["failed-unit"])

        context.config = {"allow_partial_pdf": False}
        with self.assertRaises(StageExecutionError) as raised:
            TranslationStageExecutor._enforce_failed_unit_policy(
                context, ["failed-unit"]
            )
        self.assertEqual(raised.exception.code, "translation_units_failed")

    def test_command_uses_registered_json_and_explicit_source_pdf(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "parse" / "paper_enriched.json"
            source.parent.mkdir()
            source.write_text("[]", encoding="utf-8")
            context = StageContext(
                job_id="00000000-0000-0000-0000-000000000001",
                job_dir=root,
                stage=PipelineStage.TRANSLATING,
                attempt=2,
                source_pdf=_pdf(root / "source.pdf"),
                document_name="paper.pdf",
                config=PipelineConfig(
                    translation_provider="deepseek",
                    translation_model="deepseek-chat-test",
                ).model_dump(mode="json"),
            )
            command = TranslationStageExecutor(simulate=False).build_command(
                context, source
            )

            self.assertEqual(command[command.index("--pdf") + 1], str(context.source_pdf))
            self.assertEqual(command[command.index("--provider") + 1], "deepseek")
            self.assertEqual(
                command[command.index("--model") + 1], "deepseek-chat-test"
            )
            self.assertIn("--run", command)
            self.assertIn("--retry-failed", command)

    def test_missing_registered_input_is_rejected_without_directory_guessing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "parse").mkdir()
            (root / "parse" / "guessed_enriched.json").write_text(
                "[]", encoding="utf-8"
            )
            context = StageContext(
                job_id="00000000-0000-0000-0000-000000000001",
                job_dir=root,
                stage=PipelineStage.TRANSLATING,
                attempt=1,
                source_pdf=_pdf(root / "source.pdf"),
                document_name="paper.pdf",
                config=PipelineConfig().model_dump(mode="json"),
            )

            with self.assertRaisesRegex(
                StageExecutionError, "exactly one registered"
            ):
                TranslationStageExecutor(simulate=True).execute(
                    context,
                    progress=lambda *_: None,
                    is_cancel_requested=lambda: False,
                )

    def test_simulated_translation_runs_existing_implementation_and_queues_reflow(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = LocalJobStore(root / "jobs")
            job = store.create_job(_pdf(root / "paper.pdf"))
            parsed = PipelineRunner(store, FixtureParsingExecutor()).run_stage(
                job.job_id, PipelineStage.PARSING
            )
            self.assertEqual(parsed.status, JobStatus.QUEUED)

            translated = PipelineRunner(
                store,
                TranslationStageExecutor(
                    simulate=True,
                    poll_interval=0.02,
                    heartbeat_interval=1.0,
                ),
            ).run_stage(job.job_id, PipelineStage.TRANSLATING)

            self.assertEqual(translated.status, JobStatus.QUEUED)
            self.assertEqual(
                translated.stage_record(PipelineStage.TRANSLATING).status,
                StageStatus.COMPLETED_WITH_WARNINGS,
            )
            self.assertEqual(
                translated.stage_record(PipelineStage.REFLOWING).status,
                StageStatus.PENDING,
            )
            by_kind = {artifact.kind: artifact for artifact in translated.artifacts}
            self.assertIn("translation_artifact", by_kind)
            self.assertIn("translation_stage_manifest", by_kind)
            artifact_path = store.resolve_artifact(
                job.job_id, by_kind["translation_artifact"].relative_path
            )
            artifact = TranslationArtifactOutput.model_validate_json(
                artifact_path.read_text(encoding="utf-8")
            )
            self.assertEqual(artifact.translation_run.mode, "simulation")
            self.assertEqual(artifact.document.document_id, "paper")
            self.assertIn(
                "translation_simulated",
                [warning.code for warning in translated.warnings],
            )
            stage_manifest = json.loads(
                store.resolve_artifact(
                    job.job_id,
                    by_kind["translation_stage_manifest"].relative_path,
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                stage_manifest["source_artifact"],
                "parse/paper/auto/paper_enriched.json",
            )
            self.assertEqual(stage_manifest["network_requests"], 0)


if __name__ == "__main__":
    unittest.main()
