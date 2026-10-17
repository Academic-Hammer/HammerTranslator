"""Unified command line for local pipeline jobs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from hammer_translator.parsing import default_environment_profile

from .executor import FakePipelineExecutor
from .local_executor import LocalPipelineExecutor
from .mineru_parser import MinerUParsingExecutor
from .reflow_executor import ReflowStageExecutor
from .runner import PipelineRunner
from .schema import JobStatus, PipelineConfig, PipelineStage, StageStatus
from .storage import LocalJobStore, PipelineStorageError
from .translation_executor import TranslationStageExecutor
from .validation_executor import PDFValidationStageExecutor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JOBS_ROOT = PROJECT_ROOT / "var" / "jobs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description="Unified local PDF parsing, translation, Reflow, and validation pipeline.",
    )
    parser.add_argument(
        "--jobs-root", type=Path, default=DEFAULT_JOBS_ROOT, help="local job storage root"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="create and execute a local pipeline job")
    run.add_argument("pdf", type=Path)
    run_mode = run.add_mutually_exclusive_group(required=True)
    run_mode.add_argument("--run", action="store_true")
    run_mode.add_argument("--simulate", action="store_true")
    run.add_argument(
        "--environment-profile",
        choices=["local_mac", "linux_gpu"],
        default=default_environment_profile(),
    )
    run.add_argument(
        "--parser-backend",
        choices=["pipeline", "vlm-engine", "hybrid-engine"],
        default="pipeline",
    )
    run.add_argument("--parser-method", choices=["auto", "txt", "ocr"], default="auto")
    run.add_argument(
        "--parser-device", choices=["auto", "mps", "cuda", "cpu"], default="auto"
    )
    run.add_argument("--parser-lang", default="ch")
    run.add_argument(
        "--parser-formula", action=argparse.BooleanOptionalAction, default=True
    )
    run.add_argument(
        "--parser-table", action=argparse.BooleanOptionalAction, default=True
    )
    run.add_argument(
        "--parser-formula-ch", action=argparse.BooleanOptionalAction, default=True
    )
    run.add_argument(
        "--parser-image-analysis",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    run.add_argument("--parser-effort", choices=["medium", "high"], default="medium")
    run.add_argument("--parser-start-page", type=int, default=0)
    run.add_argument("--parser-end-page", type=int)
    run.add_argument("--parser-render-threads", type=int, default=4)
    run.add_argument("--parser-window-size", type=int, default=8)
    run.add_argument("--provider", default="deepseek")
    run.add_argument("--model")
    run.add_argument("--max-translation-attempts", type=int, default=3)
    run.add_argument("--max-retranslation-attempts", type=int, default=2)
    run.add_argument("--request-timeout", type=float, default=90.0)
    run.add_argument(
        "--allow-partial-pdf",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parse = commands.add_parser(
        "parse",
        help="create a job and run only the real MinerU parsing stage",
    )
    parse.add_argument("pdf", type=Path)
    parse.add_argument("--run", action="store_true", required=True)
    parse.add_argument(
        "--environment-profile",
        choices=["local_mac", "linux_gpu"],
        default=default_environment_profile(),
    )
    parse.add_argument(
        "--backend",
        choices=["pipeline", "vlm-engine", "hybrid-engine"],
        default="pipeline",
    )
    parse.add_argument("--method", choices=["auto", "txt", "ocr"], default="auto")
    parse.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parse.add_argument("--lang", default="ch")
    parse.add_argument("--formula", action=argparse.BooleanOptionalAction, default=True)
    parse.add_argument("--table", action=argparse.BooleanOptionalAction, default=True)
    parse.add_argument(
        "--formula-ch", action=argparse.BooleanOptionalAction, default=True
    )
    parse.add_argument(
        "--image-analysis", action=argparse.BooleanOptionalAction, default=True
    )
    parse.add_argument("--effort", choices=["medium", "high"], default="medium")
    parse.add_argument("--start-page", type=int, default=0)
    parse.add_argument("--end-page", type=int)
    parse.add_argument("--render-threads", type=int, default=4)
    parse.add_argument("--window-size", type=int, default=8)
    parse.add_argument("--provider", default="deepseek")
    parse.add_argument("--model")
    parse.add_argument("--max-translation-attempts", type=int, default=3)
    parse.add_argument("--max-retranslation-attempts", type=int, default=2)
    parse.add_argument("--request-timeout", type=float, default=90.0)

    translate = commands.add_parser(
        "translate",
        help="run only Translation for a job whose parsing stage is complete",
    )
    translate.add_argument("job_id")
    translation_mode = translate.add_mutually_exclusive_group(required=True)
    translation_mode.add_argument("--run", action="store_true")
    translation_mode.add_argument("--simulate", action="store_true")
    translate.add_argument("--retry-failed", action="store_true")

    reflow = commands.add_parser(
        "reflow",
        help="run only Reflow for a job whose Translation stage is complete",
    )
    reflow.add_argument("job_id")
    reflow.add_argument("--retry-failed", action="store_true")

    validate = commands.add_parser(
        "validate",
        help="run final PDF validation for a job whose Reflow stage is complete",
    )
    validate.add_argument("job_id")
    validate.add_argument("--retry-failed", action="store_true")

    status = commands.add_parser("status", help="print one job state")
    status.add_argument("job_id")

    events = commands.add_parser("events", help="print persisted job events")
    events.add_argument("job_id")
    events.add_argument("--after", type=int, default=0)

    cancel = commands.add_parser("cancel", help="request cooperative cancellation")
    cancel.add_argument("job_id")

    resume = commands.add_parser("resume", help="resume a failed pipeline job")
    resume.add_argument("job_id")
    resume_mode = resume.add_mutually_exclusive_group(required=True)
    resume_mode.add_argument("--run", action="store_true")
    resume_mode.add_argument("--simulate", action="store_true")
    return parser


def _print(value) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store = LocalJobStore(args.jobs_root)
    runner = PipelineRunner(store, FakePipelineExecutor())
    try:
        if args.command == "run":
            config = PipelineConfig(
                execution_mode="simulation" if args.simulate else "real",
                environment_profile=args.environment_profile,
                parser_backend=args.parser_backend,
                parser_method=args.parser_method,
                parser_device=args.parser_device,
                parser_language=args.parser_lang,
                parser_formula=args.parser_formula,
                parser_table=args.parser_table,
                parser_formula_ch=args.parser_formula_ch,
                parser_image_analysis=args.parser_image_analysis,
                parser_effort=args.parser_effort,
                parser_start_page=args.parser_start_page,
                parser_end_page=args.parser_end_page,
                parser_render_threads=args.parser_render_threads,
                parser_window_size=args.parser_window_size,
                translation_provider=args.provider,
                translation_model=args.model,
                allow_partial_pdf=args.allow_partial_pdf,
                max_translation_attempts=args.max_translation_attempts,
                max_retranslation_attempts=args.max_retranslation_attempts,
                translation_request_timeout_seconds=args.request_timeout,
            )
            job = store.create_job(args.pdf, config=config)
            if not args.simulate:
                print(
                    f"[{job.job_id}] queued real local pipeline",
                    file=sys.stderr,
                    flush=True,
                )
            execution = (
                FakePipelineExecutor()
                if args.simulate
                else LocalPipelineExecutor(
                    progress_sink=lambda message: print(
                        f"[{job.job_id}] {message}",
                        file=sys.stderr,
                        flush=True,
                    )
                )
            )
            result = PipelineRunner(store, execution).run(job.job_id)
            _print(result.model_dump(mode="json"))
            return 0 if result.status in {
                JobStatus.COMPLETED,
                JobStatus.COMPLETED_WITH_WARNINGS,
            } else 1
        if args.command == "parse":
            config = PipelineConfig(
                environment_profile=args.environment_profile,
                parser_backend=args.backend,
                parser_method=args.method,
                parser_device=args.device,
                parser_language=args.lang,
                parser_formula=args.formula,
                parser_table=args.table,
                parser_formula_ch=args.formula_ch,
                parser_image_analysis=args.image_analysis,
                parser_effort=args.effort,
                parser_start_page=args.start_page,
                parser_end_page=args.end_page,
                parser_render_threads=args.render_threads,
                parser_window_size=args.window_size,
                translation_provider=args.provider,
                translation_model=args.model,
                max_translation_attempts=args.max_translation_attempts,
                max_retranslation_attempts=args.max_retranslation_attempts,
                translation_request_timeout_seconds=args.request_timeout,
            )
            job = store.create_job(args.pdf, config=config)
            print(
                f"[{job.job_id}] queued real MinerU parsing ({args.environment_profile})",
                file=sys.stderr,
                flush=True,
            )
            real_runner = PipelineRunner(
                store,
                MinerUParsingExecutor(
                    progress_sink=lambda message: print(
                        f"[{job.job_id}] {message}",
                        file=sys.stderr,
                        flush=True,
                    )
                ),
            )
            result = real_runner.run_stage(job.job_id, PipelineStage.PARSING)
            print(
                f"[{job.job_id}] parsing stage: "
                f"{result.stage_record(PipelineStage.PARSING).status.value}",
                file=sys.stderr,
                flush=True,
            )
            _print(result.model_dump(mode="json"))
            parse_status = result.stage_record(PipelineStage.PARSING).status
            return 0 if parse_status in {
                StageStatus.COMPLETED,
                StageStatus.COMPLETED_WITH_WARNINGS,
            } else 1
        if args.command == "translate":
            print(
                f"[{args.job_id}] queued Translation Stage",
                file=sys.stderr,
                flush=True,
            )
            translation_runner = PipelineRunner(
                store,
                TranslationStageExecutor(
                    simulate=args.simulate,
                    progress_sink=lambda message: print(
                        f"[{args.job_id}] {message}",
                        file=sys.stderr,
                        flush=True,
                    ),
                ),
            )
            result = translation_runner.run_stage(
                args.job_id,
                PipelineStage.TRANSLATING,
                retry_failed=args.retry_failed,
            )
            print(
                f"[{args.job_id}] translation stage: "
                f"{result.stage_record(PipelineStage.TRANSLATING).status.value}",
                file=sys.stderr,
                flush=True,
            )
            _print(result.model_dump(mode="json"))
            translation_status = result.stage_record(PipelineStage.TRANSLATING).status
            return 0 if translation_status in {
                StageStatus.COMPLETED,
                StageStatus.COMPLETED_WITH_WARNINGS,
            } else 1
        if args.command == "reflow":
            print(
                f"[{args.job_id}] queued Reflow Stage",
                file=sys.stderr,
                flush=True,
            )
            reflow_runner = PipelineRunner(
                store,
                ReflowStageExecutor(
                    progress_sink=lambda message: print(
                        f"[{args.job_id}] {message}",
                        file=sys.stderr,
                        flush=True,
                    ),
                ),
            )
            result = reflow_runner.run_stage(
                args.job_id,
                PipelineStage.REFLOWING,
                retry_failed=args.retry_failed,
            )
            print(
                f"[{args.job_id}] reflow stage: "
                f"{result.stage_record(PipelineStage.REFLOWING).status.value}",
                file=sys.stderr,
                flush=True,
            )
            _print(result.model_dump(mode="json"))
            reflow_status = result.stage_record(PipelineStage.REFLOWING).status
            return 0 if reflow_status in {
                StageStatus.COMPLETED,
                StageStatus.COMPLETED_WITH_WARNINGS,
            } else 1
        if args.command == "validate":
            print(
                f"[{args.job_id}] queued final PDF validation",
                file=sys.stderr,
                flush=True,
            )
            validation_runner = PipelineRunner(store, PDFValidationStageExecutor())
            result = validation_runner.run_stage(
                args.job_id,
                PipelineStage.VALIDATING,
                retry_failed=args.retry_failed,
            )
            print(
                f"[{args.job_id}] validation stage: "
                f"{result.stage_record(PipelineStage.VALIDATING).status.value}",
                file=sys.stderr,
                flush=True,
            )
            _print(result.model_dump(mode="json"))
            validation_status = result.stage_record(PipelineStage.VALIDATING).status
            return 0 if validation_status in {
                StageStatus.COMPLETED,
                StageStatus.COMPLETED_WITH_WARNINGS,
            } else 1
        if args.command == "status":
            _print(store.load_job(args.job_id).model_dump(mode="json"))
            return 0
        if args.command == "events":
            _print(
                [
                    event.model_dump(mode="json")
                    for event in store.iter_events(args.job_id, after=args.after)
                ]
            )
            return 0
        if args.command == "cancel":
            _print(runner.request_cancel(args.job_id).model_dump(mode="json"))
            return 0
        if args.command == "resume":
            job = store.load_job(args.job_id)
            requested_mode = "simulation" if args.simulate else "real"
            if job.config.execution_mode != requested_mode:
                raise ValueError(
                    "resume mode does not match the mode recorded when the job was created"
                )
            execution = (
                FakePipelineExecutor()
                if args.simulate
                else LocalPipelineExecutor(
                    progress_sink=lambda message: print(
                        f"[{args.job_id}] {message}",
                        file=sys.stderr,
                        flush=True,
                    )
                )
            )
            result = PipelineRunner(store, execution).run(
                args.job_id, retry_failed=True
            )
            _print(result.model_dump(mode="json"))
            return 0 if result.status in {
                JobStatus.COMPLETED,
                JobStatus.COMPLETED_WITH_WARNINGS,
            } else 1
    except (OSError, ValueError, PipelineStorageError) as exc:
        print(f"pipeline error: {exc}", file=sys.stderr)
        return 2
    raise AssertionError("unreachable pipeline command")
