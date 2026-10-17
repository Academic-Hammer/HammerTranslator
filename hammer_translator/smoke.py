"""Offline installed-package smoke test for Translation and Reflow."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

import fitz

from hammer_translator.reflow.schema import ReflowReportOutput
from hammer_translator.runtime_paths import runtime_paths
from hammer_translator.translation.config import ConfigError, load_config
from hammer_translator.translation.output_schema import TranslationArtifactOutput


class SmokeTestError(RuntimeError):
    """Raised when one of the offline smoke-test stages fails."""

    def __init__(self, stage: str, message: str, log_tail: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.stage = stage
        self.log_tail = tuple(log_tail)


def _run(command: Sequence[str], *, cwd: Path, stage: str, timeout: float) -> None:
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SmokeTestError(stage, f"could not execute {stage}: {type(exc).__name__}") from exc
    if completed.returncode == 0:
        return
    combined = [
        line[-1000:]
        for line in f"{completed.stdout}\n{completed.stderr}".splitlines()
        if line.strip()
    ]
    raise SmokeTestError(
        stage,
        f"{stage} exited with code {completed.returncode}",
        combined[-20:],
    )


def _write_fixture(work_dir: Path) -> tuple[Path, Path]:
    source_pdf = work_dir / "smoke_source.pdf"
    source_json = work_dir / "smoke_enriched.json"
    document = fitz.open()
    page = document.new_page(width=500, height=700)
    page.insert_text((48, 70), "A short local smoke test.", fontsize=12)
    document.save(source_pdf)
    document.close()
    source_json.write_text(
        json.dumps(
            [
                {
                    "type": "text",
                    "text": "A short local smoke test.",
                    "page_idx": 0,
                    "bbox": [48, 48, 420, 108],
                    "logical_para_id": 0,
                }
            ],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return source_pdf, source_json


def run_core_smoke(work_dir: Path, *, timeout: float = 180.0) -> dict[str, Any]:
    """Run a deterministic, no-network Translation-to-Reflow smoke test."""

    started = time.monotonic()
    paths = runtime_paths()
    config = load_config(paths.config_path)
    providers = config.get("providers")
    if not isinstance(providers, dict) or not providers:
        raise SmokeTestError("prepare", "translation config has no providers")
    provider_name = sorted(str(name) for name in providers)[0]
    work_dir = work_dir.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    reserved = (
        work_dir / "smoke_source.pdf",
        work_dir / "smoke_enriched.json",
        work_dir / "translation",
        work_dir / "output",
    )
    occupied = [path.name for path in reserved if path.exists()]
    if occupied:
        raise SmokeTestError(
            "prepare",
            f"work directory contains reserved smoke-test paths: {', '.join(occupied)}",
        )

    source_pdf, source_json = _write_fixture(work_dir)
    translation_dir = work_dir / "translation"
    output_dir = work_dir / "output"
    translated_pdf = output_dir / "translated.pdf"
    feedback_path = output_dir / "reflow_feedback.json"

    _run(
        [
            sys.executable,
            "-u",
            "-m",
            "hammer_translator.translation",
            str(source_json),
            "--config",
            str(paths.config_path),
            "--env-file",
            str(paths.env_file),
            "--provider",
            provider_name,
            "--output",
            str(translation_dir),
            "--pdf",
            str(source_pdf),
            "--simulate",
            "--quiet",
        ],
        cwd=work_dir,
        stage="translation",
        timeout=timeout,
    )
    artifacts = sorted(translation_dir.glob("*_translated.json"))
    if len(artifacts) != 1:
        raise SmokeTestError(
            "translation",
            f"expected one Translation artifact, found {len(artifacts)}",
        )
    translation = TranslationArtifactOutput.model_validate_json(
        artifacts[0].read_text(encoding="utf-8")
    )
    if translation.translation_run.mode != "simulation":
        raise SmokeTestError("translation", "smoke artifact was not marked simulation")

    output_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            "-u",
            "-m",
            "hammer_translator.reflow",
            str(artifacts[0]),
            "--output",
            str(translated_pdf),
            "--feedback",
            str(feedback_path),
            "--allow-partial",
        ],
        cwd=work_dir,
        stage="reflow",
        timeout=timeout,
    )
    feedback = ReflowReportOutput.model_validate_json(
        feedback_path.read_text(encoding="utf-8")
    )
    if feedback.status not in {"rendered", "partial_needs_review"}:
        raise SmokeTestError("reflow", f"unexpected Reflow status: {feedback.status}")

    source = fitz.open(source_pdf)
    rendered = fitz.open(translated_pdf)
    try:
        same_geometry = (
            rendered.page_count == source.page_count == 1
            and rendered[0].rect == source[0].rect
            and rendered[0].rotation == source[0].rotation
        )
    finally:
        rendered.close()
        source.close()
    if not same_geometry:
        raise SmokeTestError("validation", "output PDF page geometry changed")

    return {
        "status": "passed",
        "mode": "offline_simulation",
        "network_requests": 0,
        "provider_config": provider_name,
        "parser_tested": False,
        "translation_schema": translation.schema_version,
        "reflow_schema": feedback.schema_version,
        "reflow_status": feedback.status,
        "page_count": 1,
        "page_geometry_preserved": True,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "artifacts": {
            "translation": str(artifacts[0]),
            "pdf": str(translated_pdf),
            "feedback": str(feedback_path),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hammer-smoke",
        description=(
            "Run an offline Translation-to-Reflow smoke test. This never invokes "
            "MinerU or a remote translation API."
        ),
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="retain smoke-test artifacts in this otherwise unused directory",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.timeout <= 0:
        print(json.dumps({"status": "failed", "error": "timeout must be positive"}))
        return 2

    temporary = None
    if args.work_dir is None:
        temp_root = runtime_paths().temp_dir
        temp_root.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="smoke-", dir=temp_root)
        work_context = temporary
    else:
        work_context = nullcontext(str(args.work_dir))

    try:
        with work_context as selected:
            report = run_core_smoke(Path(selected), timeout=args.timeout)
            if temporary is not None:
                report["artifacts"] = None
                report["artifacts_retained"] = False
            else:
                report["artifacts_retained"] = True
            print(json.dumps(report, ensure_ascii=False, indent=2))
    except (SmokeTestError, ConfigError, OSError, ValueError) as exc:
        failure = {
            "status": "failed",
            "stage": getattr(exc, "stage", "validation"),
            "error": str(exc),
            "log_tail": list(getattr(exc, "log_tail", ())),
        }
        print(json.dumps(failure, ensure_ascii=False, indent=2))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
