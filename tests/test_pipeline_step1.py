from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from hammer_translator.pipeline.cli import main
from hammer_translator.pipeline.executor import FakePipelineExecutor
from hammer_translator.pipeline.runner import PipelineRunner
from hammer_translator.pipeline.schema import (
    ArtifactRecord,
    EventType,
    JobStatus,
    PipelineStage,
    StageStatus,
    WarningRecord,
)
from hammer_translator.pipeline.storage import LocalJobStore


def _pdf(path: Path) -> Path:
    path.write_bytes(b"%PDF-1.4\n% pipeline contract fixture\n%%EOF\n")
    return path


class PipelineStorageTests(unittest.TestCase):
    def test_create_job_uses_isolated_atomic_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = LocalJobStore(root / "jobs")
            job = store.create_job(_pdf(root / "sample.pdf"))
            job_dir = store.job_dir(job.job_id)

            self.assertEqual(job.status, JobStatus.QUEUED)
            self.assertEqual(job.source_pdf, "input/source.pdf")
            self.assertTrue((job_dir / "input" / "source.pdf").is_file())
            self.assertTrue((job_dir / "job.json").is_file())
            self.assertTrue((job_dir / "events.jsonl").is_file())
            self.assertEqual(
                [record.stage for record in job.stages],
                list(PipelineStage),
            )
            self.assertEqual(list(job_dir.glob(".*.tmp")), [])
            events = list(store.iter_events(job.job_id))
            self.assertEqual([event.event_type for event in events], [EventType.JOB_CREATED])

    def test_artifact_paths_cannot_escape_job_directory(self):
        with self.assertRaisesRegex(ValueError, "safe job-relative"):
            ArtifactRecord(name="bad", kind="bad", relative_path="../secret")
        with self.assertRaisesRegex(ValueError, "safe job-relative"):
            ArtifactRecord(name="bad", kind="bad", relative_path=".")

    def test_symlink_artifact_cannot_escape_job_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = LocalJobStore(root / "jobs")
            job = store.create_job(_pdf(root / "sample.pdf"))
            outside = root / "outside.json"
            outside.write_text("{}", encoding="utf-8")
            link = store.job_dir(job.job_id) / "output" / "escaped.json"
            link.symlink_to(outside)

            with self.assertRaisesRegex(Exception, "escaped the job directory"):
                store.resolve_artifact(job.job_id, "output/escaped.json")

    def test_non_pdf_signature_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fake = root / "not-a-pdf.pdf"
            fake.write_text("not a PDF", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "PDF signature"):
                LocalJobStore(root / "jobs").create_job(fake)


class PipelineRunnerTests(unittest.TestCase):
    def test_fake_pipeline_completes_and_persists_ordered_events(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = LocalJobStore(root / "jobs")
            job = store.create_job(_pdf(root / "sample.pdf"))
            result = PipelineRunner(store, FakePipelineExecutor()).run(job.job_id)

            self.assertEqual(result.status, JobStatus.COMPLETED)
            self.assertIsNone(result.current_stage)
            self.assertEqual(len(result.artifacts), 5)
            self.assertTrue(
                all(record.status is StageStatus.COMPLETED for record in result.stages)
            )
            events = list(store.iter_events(job.job_id))
            self.assertEqual(
                [event.sequence for event in events], list(range(1, len(events) + 1))
            )
            self.assertEqual(events[-1].event_type, EventType.JOB_COMPLETED)

    def test_warning_produces_completed_with_warnings(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = LocalJobStore(root / "jobs")
            job = store.create_job(_pdf(root / "sample.pdf"))
            executor = FakePipelineExecutor(
                warnings={
                    PipelineStage.REFLOWING: [
                        WarningRecord(
                            code="needs_review",
                            message="one block preserved for manual review",
                            stage=PipelineStage.REFLOWING,
                        )
                    ]
                }
            )
            result = PipelineRunner(store, executor).run(job.job_id)

            self.assertEqual(result.status, JobStatus.COMPLETED_WITH_WARNINGS)
            self.assertEqual(len(result.warnings), 1)
            self.assertEqual(
                result.stage_record(PipelineStage.REFLOWING).status,
                StageStatus.COMPLETED_WITH_WARNINGS,
            )

    def test_failed_stage_can_resume_without_repeating_completed_stages(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = LocalJobStore(root / "jobs")
            job = store.create_job(_pdf(root / "sample.pdf"))
            executor = FakePipelineExecutor(fail_once_stages={PipelineStage.TRANSLATING})
            runner = PipelineRunner(store, executor)

            failed = runner.run(job.job_id)
            self.assertEqual(failed.status, JobStatus.FAILED)
            self.assertEqual(
                failed.stage_record(PipelineStage.PARSING).attempt,
                1,
            )
            self.assertEqual(
                failed.stage_record(PipelineStage.TRANSLATING).status,
                StageStatus.FAILED,
            )

            resumed = runner.run(job.job_id, retry_failed=True)
            self.assertEqual(resumed.status, JobStatus.COMPLETED)
            self.assertEqual(resumed.stage_record(PipelineStage.PARSING).attempt, 1)
            self.assertEqual(resumed.stage_record(PipelineStage.TRANSLATING).attempt, 2)
            self.assertIn(
                EventType.JOB_RESUMED,
                [event.event_type for event in store.iter_events(job.job_id)],
            )

    def test_cancel_before_run_preserves_all_stages(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = LocalJobStore(root / "jobs")
            job = store.create_job(_pdf(root / "sample.pdf"))
            runner = PipelineRunner(store, FakePipelineExecutor())
            runner.request_cancel(job.job_id)
            cancelled = runner.run(job.job_id)

            self.assertEqual(cancelled.status, JobStatus.CANCELLED)
            self.assertTrue(
                all(record.status is StageStatus.PENDING for record in cancelled.stages)
            )

    def test_running_stage_honors_cooperative_cancellation(self):
        class CancellingExecutor(FakePipelineExecutor):
            runner: PipelineRunner | None = None

            def execute(self, context, *, progress, is_cancel_requested):
                assert self.runner is not None
                self.runner.request_cancel(context.job_id)
                return super().execute(
                    context,
                    progress=progress,
                    is_cancel_requested=is_cancel_requested,
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = LocalJobStore(root / "jobs")
            job = store.create_job(_pdf(root / "sample.pdf"))
            executor = CancellingExecutor()
            runner = PipelineRunner(store, executor)
            executor.runner = runner

            cancelled = runner.run(job.job_id)

            self.assertEqual(cancelled.status, JobStatus.CANCELLED)
            self.assertEqual(
                cancelled.stage_record(PipelineStage.PARSING).status,
                StageStatus.CANCELLED,
            )
            self.assertEqual(
                [
                    event.event_type
                    for event in store.iter_events(job.job_id)
                    if event.event_type
                    in {EventType.CANCEL_REQUESTED, EventType.JOB_CANCELLED}
                ],
                [EventType.CANCEL_REQUESTED, EventType.JOB_CANCELLED],
            )


class PipelineCliTests(unittest.TestCase):
    def test_simulated_cli_returns_machine_readable_job(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = main(
                    [
                        "--jobs-root",
                        str(root / "jobs"),
                        "run",
                        str(_pdf(root / "sample.pdf")),
                        "--simulate",
                    ]
                )
            payload = json.loads(stdout.getvalue())
            self.assertEqual(code, 0)
            self.assertEqual(stderr.getvalue(), "")
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(payload["schema_version"], "pipeline-job-v1")


if __name__ == "__main__":
    unittest.main()
