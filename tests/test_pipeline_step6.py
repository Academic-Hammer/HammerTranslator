from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hammer_translator.pipeline.cli import main
from hammer_translator.pipeline.executor import FakePipelineExecutor, StageContext
from hammer_translator.pipeline.local_executor import LocalPipelineExecutor
from hammer_translator.pipeline.runner import PipelineRunner
from hammer_translator.pipeline.schema import (
    JobStatus,
    PipelineConfig,
    PipelineStage,
    StageExecutionResult,
)
from hammer_translator.pipeline.storage import LocalJobStore
from hammer_translator.pipeline.translation_executor import UNIT_PROGRESS_RE


def _pdf(path: Path) -> Path:
    path.write_bytes(b"%PDF-1.4\n% step 6 pipeline fixture\n%%EOF\n")
    return path


class RecordingExecutor:
    name = "recording-executor"

    def __init__(self) -> None:
        self.calls: list[PipelineStage] = []

    def execute(self, context, *, progress, is_cancel_requested):
        self.calls.append(context.stage)
        progress(1, 1, context.stage.value, None)
        return StageExecutionResult(metadata={"stage": context.stage.value})


class LocalPipelineExecutorTests(unittest.TestCase):
    def test_routes_each_stage_to_its_registered_real_adapter(self):
        delegates = {stage: RecordingExecutor() for stage in PipelineStage}
        executor = LocalPipelineExecutor(
            parsing_executor=delegates[PipelineStage.PARSING],
            translation_executor=delegates[PipelineStage.TRANSLATING],
            reflow_executor=delegates[PipelineStage.REFLOWING],
            validation_executor=delegates[PipelineStage.VALIDATING],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = _pdf(root / "source.pdf")
            for stage in PipelineStage:
                context = StageContext(
                    job_id="job",
                    job_dir=root,
                    stage=stage,
                    attempt=1,
                    source_pdf=source,
                    document_name="source.pdf",
                    config=PipelineConfig().model_dump(mode="json"),
                )
                executor.execute(
                    context,
                    progress=lambda *_: None,
                    is_cancel_requested=lambda: False,
                )

        for stage, delegate in delegates.items():
            self.assertEqual(delegate.calls, [stage])

    def test_translation_progress_recognizes_layout_retry_lines(self):
        match = UNIT_PROGRESS_RE.search(
            "[paper] layout 63/203 title: constrained retranslation"
        )
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.groups(), ("63", "203"))


class FullPipelineCliTests(unittest.TestCase):
    def test_runtime_profile_rejects_the_other_platform_accelerator(self):
        with self.assertRaisesRegex(ValueError, "CUDA"):
            PipelineConfig(environment_profile="local_mac", parser_device="cuda")
        with self.assertRaisesRegex(ValueError, "MPS"):
            PipelineConfig(environment_profile="linux_gpu", parser_device="mps")

    def test_run_real_mode_uses_local_executor_and_persists_options(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stdout = io.StringIO()
            stderr = io.StringIO()
            fake = FakePipelineExecutor()
            with patch(
                "hammer_translator.pipeline.cli.LocalPipelineExecutor",
                return_value=fake,
            ) as local_executor, contextlib.redirect_stdout(
                stdout
            ), contextlib.redirect_stderr(stderr):
                code = main(
                    [
                        "--jobs-root",
                        str(root / "jobs"),
                        "run",
                        str(_pdf(root / "sample.pdf")),
                        "--run",
                        "--parser-backend",
                        "vlm-engine",
                        "--parser-device",
                        "mps",
                        "--provider",
                        "deepseek",
                        "--model",
                        "deepseek-chat",
                        "--no-allow-partial-pdf",
                    ]
                )

            payload = json.loads(stdout.getvalue())
            self.assertEqual(code, 0)
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(payload["config"]["execution_mode"], "real")
            self.assertEqual(payload["config"]["parser_backend"], "vlm-engine")
            self.assertEqual(payload["config"]["parser_device"], "mps")
            self.assertEqual(payload["config"]["translation_model"], "deepseek-chat")
            self.assertFalse(payload["config"]["allow_partial_pdf"])
            self.assertIn(payload["job_id"], stderr.getvalue())
            local_executor.assert_called_once()

    def test_linux_profile_and_cuda_are_persisted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stdout = io.StringIO()
            fake = FakePipelineExecutor()
            with patch(
                "hammer_translator.pipeline.cli.LocalPipelineExecutor",
                return_value=fake,
            ), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(
                io.StringIO()
            ):
                code = main(
                    [
                        "--jobs-root",
                        str(root / "jobs"),
                        "run",
                        str(_pdf(root / "sample.pdf")),
                        "--run",
                        "--environment-profile",
                        "linux_gpu",
                        "--parser-device",
                        "cuda",
                    ]
                )

            payload = json.loads(stdout.getvalue())
            self.assertEqual(code, 0)
            self.assertEqual(payload["config"]["environment_profile"], "linux_gpu")
            self.assertEqual(payload["config"]["parser_device"], "cuda")

    def test_resume_real_job_skips_completed_stage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            jobs_root = root / "jobs"
            store = LocalJobStore(jobs_root)
            job = store.create_job(
                _pdf(root / "sample.pdf"),
                config=PipelineConfig(execution_mode="real"),
            )
            failed = PipelineRunner(
                store,
                FakePipelineExecutor(
                    fail_once_stages={PipelineStage.TRANSLATING}
                ),
            ).run(job.job_id)
            self.assertEqual(failed.status, JobStatus.FAILED)

            stdout = io.StringIO()
            stderr = io.StringIO()
            with patch(
                "hammer_translator.pipeline.cli.LocalPipelineExecutor",
                return_value=FakePipelineExecutor(),
            ), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = main(
                    [
                        "--jobs-root",
                        str(jobs_root),
                        "resume",
                        job.job_id,
                        "--run",
                    ]
                )

            payload = json.loads(stdout.getvalue())
            self.assertEqual(code, 0)
            self.assertEqual(payload["status"], "completed")
            stages = {item["stage"]: item for item in payload["stages"]}
            self.assertEqual(stages["parsing"]["attempt"], 1)
            self.assertEqual(stages["translating"]["attempt"], 2)

    def test_resume_rejects_executor_mode_change(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            jobs_root = root / "jobs"
            store = LocalJobStore(jobs_root)
            job = store.create_job(
                _pdf(root / "sample.pdf"),
                config=PipelineConfig(execution_mode="real"),
            )
            PipelineRunner(
                store,
                FakePipelineExecutor(fail_stages={PipelineStage.PARSING}),
            ).run(job.job_id)

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = main(
                    [
                        "--jobs-root",
                        str(jobs_root),
                        "resume",
                        job.job_id,
                        "--simulate",
                    ]
                )

            self.assertEqual(code, 2)
            self.assertIn("resume mode does not match", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
