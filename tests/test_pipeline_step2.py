from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from hammer_translator.pipeline.cli import main
from hammer_translator.pipeline.executor import (
    FakePipelineExecutor,
    StageCancelledError,
    StageContext,
    StageExecutionError,
)
from hammer_translator.pipeline.mac_parser import (
    MacMinerUParsingExecutor,
    PARSER_MODULE,
    parse_mode,
    safe_document_stem,
)
from hammer_translator.pipeline.process import CancellableProcessRunner
from hammer_translator.pipeline.runner import PipelineRunner
from hammer_translator.pipeline.schema import (
    EventType,
    JobStatus,
    PipelineConfig,
    PipelineStage,
    StageStatus,
)
from hammer_translator.pipeline.storage import LocalJobStore


def _pdf(path: Path) -> Path:
    path.write_bytes(b"%PDF-1.4\n% pipeline parser fixture\n%%EOF\n")
    return path


class FixtureMacParser(MacMinerUParsingExecutor):
    def _run_process(
        self,
        context,
        document_name,
        *,
        progress,
        is_cancel_requested,
    ):
        self.calls = getattr(self, "calls", 0) + 1
        backend = context.config["parser_backend"]
        method = context.config["parser_method"]
        result_dir = (
            context.job_dir
            / "parse"
            / document_name
            / parse_mode(backend, method)
        )
        result_dir.mkdir(parents=True)
        (result_dir / "images").mkdir()
        (result_dir / "images" / "figure.jpg").write_bytes(b"jpeg")
        (result_dir / f"{document_name}_middle.json").write_text(
            '{"pdf_info": []}\n', encoding="utf-8"
        )
        (result_dir / f"{document_name}_content_list.json").write_text(
            "[]\n", encoding="utf-8"
        )
        (result_dir / f"{document_name}_enriched.json").write_text(
            "[]\n", encoding="utf-8"
        )
        (result_dir / f"{document_name}_origin.pdf").write_bytes(
            context.source_pdf.read_bytes()
        )
        (result_dir / f"{document_name}.md").write_text(
            "# Parsed\n", encoding="utf-8"
        )
        manifest = {
            "schema_version": "mineru-parse-v1",
            "document_name": document_name,
            "source_filename": context.source_pdf.name,
            "backend": backend,
            "method": method,
            "language": context.config["parser_language"],
            "adapter_enabled": True,
            "adapter_warnings": ["fixture adapter warning"],
            "files": [],
        }
        (result_dir / f"{document_name}_parse_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        progress(1, 1, "fixture parsing completed", {"fixture": True})


class MacParserExecutorTests(unittest.TestCase):
    def test_command_uses_job_config_and_safe_document_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            context = StageContext(
                job_id="00000000-0000-0000-0000-000000000001",
                job_dir=root,
                stage=PipelineStage.PARSING,
                attempt=1,
                source_pdf=_pdf(root / "source.pdf"),
                document_name="A paper (final).pdf",
                config=PipelineConfig(
                    parser_backend="vlm-engine",
                    parser_method="ocr",
                    parser_device="auto",
                    parser_formula=False,
                    parser_end_page=4,
                ).model_dump(mode="json"),
            )
            executor = MacMinerUParsingExecutor()
            document_name = safe_document_stem(context.document_name)
            command = executor.build_command(context, document_name)

            self.assertEqual(document_name, "A_paper_final")
            self.assertEqual(command[2:4], ["-m", PARSER_MODULE])
            self.assertIn("vlm-engine", command)
            self.assertIn("--no-formula", command)
            self.assertEqual(command[command.index("--end-page") + 1], "4")
            self.assertEqual(
                command[command.index("--document-name") + 1], document_name
            )

    def test_parser_child_process_can_be_cancelled(self):
        class SleepingParser(MacMinerUParsingExecutor):
            def build_command(self, context, document_name):
                return [
                    sys.executable,
                    "-u",
                    "-c",
                    "import time; print('ready', flush=True); time.sleep(30)",
                ]

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "logs").mkdir()
            context = StageContext(
                job_id="00000000-0000-0000-0000-000000000001",
                job_dir=root,
                stage=PipelineStage.PARSING,
                attempt=1,
                source_pdf=_pdf(root / "source.pdf"),
                document_name="paper.pdf",
                config=PipelineConfig().model_dump(mode="json"),
            )
            executor = SleepingParser(poll_interval=0.02)
            started = time.monotonic()

            with self.assertRaises(StageCancelledError):
                executor._run_process(
                    context,
                    "paper",
                    progress=lambda *_: None,
                    is_cancel_requested=lambda: time.monotonic() - started > 0.1,
                )

            self.assertLess(time.monotonic() - started, 3.0)

    def test_keyboard_interrupt_terminates_child_and_is_retryable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runner = CancellableProcessRunner(poll_interval=0.02)
            started = time.monotonic()

            def interrupt_on_output(*_args):
                raise KeyboardInterrupt

            with self.assertRaises(StageExecutionError) as raised:
                runner.run(
                    [
                        sys.executable,
                        "-u",
                        "-c",
                        "import time; print('ready', flush=True); time.sleep(30)",
                    ],
                    cwd=root,
                    log_path=root / "interrupted.log",
                    executor_name="test-child",
                    heartbeat_message="still running",
                    start_error_message="could not start child",
                    start_error_code="child_start_failed",
                    progress=interrupt_on_output,
                    is_cancel_requested=lambda: False,
                )

            self.assertEqual(raised.exception.code, "stage_interrupted")
            self.assertTrue(raised.exception.retryable)
            self.assertLess(time.monotonic() - started, 3.0)

    def test_real_parse_stage_contract_queues_translation_and_is_not_repeated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = LocalJobStore(root / "jobs")
            job = store.create_job(
                _pdf(root / "Research Paper.pdf"),
                config=PipelineConfig(parser_backend="pipeline", parser_method="auto"),
            )
            executor = FixtureMacParser()
            parsed = PipelineRunner(store, executor).run_stage(
                job.job_id, PipelineStage.PARSING
            )

            self.assertEqual(parsed.status, JobStatus.QUEUED)
            self.assertEqual(
                parsed.stage_record(PipelineStage.PARSING).status,
                StageStatus.COMPLETED_WITH_WARNINGS,
            )
            self.assertEqual(parsed.stage_record(PipelineStage.PARSING).attempt, 1)
            self.assertEqual(executor.calls, 1)
            self.assertIn(
                "parse/Research_Paper/auto/Research_Paper_enriched.json",
                [artifact.relative_path for artifact in parsed.artifacts],
            )
            self.assertEqual(
                list(store.iter_events(job.job_id))[-1].event_type,
                EventType.JOB_QUEUED,
            )

            completed = PipelineRunner(store, FakePipelineExecutor()).run(job.job_id)
            self.assertEqual(completed.status, JobStatus.COMPLETED_WITH_WARNINGS)
            self.assertEqual(completed.stage_record(PipelineStage.PARSING).attempt, 1)
            self.assertEqual(executor.calls, 1)

    def test_manifest_mismatch_fails_stage_without_publishing_artifacts(self):
        class BadManifestParser(FixtureMacParser):
            def _run_process(self, context, document_name, **kwargs):
                super()._run_process(context, document_name, **kwargs)
                result_dir = (
                    context.job_dir
                    / "parse"
                    / document_name
                    / parse_mode(
                        context.config["parser_backend"],
                        context.config["parser_method"],
                    )
                )
                manifest_path = result_dir / f"{document_name}_parse_manifest.json"
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                payload["backend"] = "wrong-backend"
                manifest_path.write_text(json.dumps(payload), encoding="utf-8")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = LocalJobStore(root / "jobs")
            job = store.create_job(_pdf(root / "paper.pdf"))
            result = PipelineRunner(store, BadManifestParser()).run_stage(
                job.job_id, PipelineStage.PARSING
            )

            self.assertEqual(result.status, JobStatus.FAILED)
            self.assertEqual(result.error.code, "parse_contract_invalid")
            self.assertEqual(len(result.artifacts), 1)

    def test_parse_cli_creates_a_job_and_runs_only_parsing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with patch(
                "hammer_translator.pipeline.cli.MacMinerUParsingExecutor",
                return_value=FixtureMacParser(),
            ):
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(
                    stderr
                ):
                    code = main(
                        [
                            "--jobs-root",
                            str(root / "jobs"),
                            "parse",
                            str(_pdf(root / "paper.pdf")),
                            "--backend",
                            "pipeline",
                            "--run",
                        ]
                    )

            payload = json.loads(stdout.getvalue())
            self.assertEqual(code, 0)
            self.assertIn(payload["job_id"], stderr.getvalue())
            self.assertIn("parsing stage: completed_with_warnings", stderr.getvalue())
            self.assertEqual(payload["status"], "queued")
            self.assertEqual(payload["stages"][0]["status"], "completed_with_warnings")
            self.assertEqual(payload["stages"][1]["status"], "pending")


if __name__ == "__main__":
    unittest.main()
