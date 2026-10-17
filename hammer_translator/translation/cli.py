"""The single command-line entry for the rewritten Translation Stage."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from .artifact import write_translation_artifact
from .cache import TranslationCache
from .checkpoint import CheckpointError
from .config import (
    ConfigError,
    ProviderConfig,
    load_config,
    load_environment_file,
    resolve_provider,
)
from .formula_assets import build_formula_asset_contract
from .loader import InputDiscoveryError, InputValidationError, LoadedDocument, load_documents
from .layout_planner import LayoutSearchConfig, plan_translation_layout
from .provider import FakeProvider, OpenAICompatibleProvider, TranslationProvider
from .runner import RunnerConfig, execute_translation_plan
from .style import build_golden_style_profile
from .units import TranslationPlan, build_translation_plan


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "translation_config.yaml"
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
SUPPORTED_LANGUAGE_PAIR = ("en", "zh-CN")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="translate_pdfs.py",
        description=(
            "PDF translation-stage entry point. Supports local preflight, plan dry-run, "
            "FakeProvider simulation, and explicit billable provider execution."
        ),
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="One or more parsed JSON files or parse-result directories",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Safe YAML configuration (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help=(
            "Local environment-variable file; existing process variables take "
            f"precedence (default: {DEFAULT_ENV_FILE})"
        ),
    )
    parser.add_argument(
        "--provider",
        required=True,
        help="Provider name from config; must be selected explicitly for every run",
    )
    parser.add_argument("--model", help="Optional model override for this run")
    parser.add_argument("--output", type=Path, help="Translation output directory")
    parser.add_argument(
        "--pdf",
        type=Path,
        help="Explicit source PDF; allowed only when exactly one input is selected",
    )
    parser.add_argument("--source-language", default="en")
    parser.add_argument("--target-language", default="zh-CN")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check",
        action="store_true",
        help="Validate configuration and require the selected provider credential",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate local inputs and print the plan without requiring a credential",
    )
    mode.add_argument(
        "--simulate",
        action="store_true",
        help=(
            "Execute the plan locally with FakeProvider, SQLite cache, and JSON "
            "checkpoints, then write readable *_translated.json artifacts; "
            "requires --output and never calls a remote API"
        ),
    )
    mode.add_argument(
        "--run",
        action="store_true",
        help=(
            "Execute translation with the selected remote provider; requires --output "
            "and the configured credential, and may incur API charges"
        ),
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum provider attempts per unit in --simulate or --run mode (default: 3)",
    )
    parser.add_argument(
        "--max-retranslation-attempts",
        type=int,
        default=2,
        help=(
            "Maximum constrained shortening attempts after fixed-bbox overflow "
            "(default: 2)"
        ),
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=90.0,
        help="Timeout in seconds for each remote API request (default: 90)",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry units marked failed in an existing compatible checkpoint",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress live progress messages; the final JSON report is still printed",
    )
    return parser


def _validate_language_pair(source: str, target: str) -> None:
    if (source, target) != SUPPORTED_LANGUAGE_PAIR:
        raise ConfigError(
            "Step 1 supports only English to Simplified Chinese: "
            "--source-language en --target-language zh-CN"
        )


def _print_report(
    *,
    mode: str,
    provider: ProviderConfig,
    plans: list[TranslationPlan],
    output: Path | None,
    network_requests: int = 0,
    simulation: dict | None = None,
    execution: dict | None = None,
    env_file: Path | None = None,
) -> None:
    summary = provider.public_summary()
    report = {
        "status": "ok",
        "mode": mode,
        "network_requests": network_requests,
        "provider": summary,
        "language_pair": {"source": "en", "target": "zh-CN"},
        "documents": [plan.public_summary() for plan in plans],
        "output": str(output.expanduser().resolve()) if output else None,
        "environment_file": {
            "path": str(env_file.expanduser().resolve()) if env_file else None,
            "exists": bool(env_file and env_file.expanduser().resolve().is_file()),
        },
    }
    if simulation is not None:
        report["simulation"] = simulation
    if execution is not None:
        report["execution"] = execution
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _execute_documents(
    plans: list[TranslationPlan],
    *,
    provider: TranslationProvider,
    requested_provider: str,
    output: Path,
    runner_config: RunnerConfig,
    max_retranslation_attempts: int,
    show_progress: bool = True,
) -> tuple[TranslationCache, list[dict], int]:
    output.mkdir(parents=True, exist_ok=True)
    cache = TranslationCache(output / "translation_cache.sqlite3")
    document_runs = []
    network_requests = 0
    for plan in plans:
        document_id = plan.document.document_id

        def progress(event: dict) -> None:
            if not show_progress:
                return
            name = event.get("event")
            index = event.get("unit_index")
            total = event.get("unit_total")
            element = event.get("element_type")
            prefix = f"[{document_id}] {index}/{total} {element}"
            if name == "unit_started":
                message = f"{prefix}: preparing"
            elif name == "provider_request":
                message = (
                    f"{prefix}: API request "
                    f"{event.get('attempt')}/{event.get('max_attempts')}"
                )
            elif name == "provider_retry":
                message = (
                    f"{prefix}: retry after {event.get('error_code')} "
                    f"in {event.get('delay_seconds')}s"
                )
            elif name == "unit_finished":
                cache_label = " (cache)" if event.get("cache_hit") else ""
                message = f"{prefix}: {event.get('status')}{cache_label}"
            elif name == "layout_unit_started":
                message = f"[{document_id}] layout {index}/{total} {element}"
            elif name == "constrained_request":
                message = (
                    f"[{document_id}] layout {index}/{total} {element}: constrained API "
                    f"request {event.get('attempt')}/{event.get('max_attempts')}"
                )
            else:
                return
            print(message, file=sys.stderr, flush=True)

        if show_progress:
            print(
                f"[{document_id}] starting translation; use conda run "
                "--no-capture-output to see these messages immediately",
                file=sys.stderr,
                flush=True,
            )
        checkpoint_path = output / f"{plan.document.document_id}_translation_manifest.json"
        translation_path = output / f"{plan.document.document_id}_translated.json"
        result = execute_translation_plan(
            plan,
            provider=provider,
            cache=cache,
            checkpoint_path=checkpoint_path,
            config=runner_config,
            progress=progress,
        )
        if show_progress:
            print(f"[{document_id}] building style and formula assets", file=sys.stderr, flush=True)
        style_profile = build_golden_style_profile(plan)
        formula_asset_contract = build_formula_asset_contract(plan, translation_path)
        if show_progress:
            print(f"[{document_id}] planning fixed-bbox layout", file=sys.stderr, flush=True)
        layout_planning = plan_translation_layout(
            plan,
            result,
            style_profile,
            provider=provider,
            cache=cache,
            runner_config=runner_config,
            search_config=LayoutSearchConfig(
                max_retranslation_attempts=max_retranslation_attempts
            ),
            formula_asset_contract=formula_asset_contract,
            progress=progress,
        )
        if show_progress:
            print(f"[{document_id}] writing translation artifact", file=sys.stderr, flush=True)
        translation_output = write_translation_artifact(
            translation_path,
            plan,
            provider=provider,
            requested_provider=requested_provider,
            result=result,
            config=runner_config,
            style_profile=style_profile,
            layout_planning=layout_planning,
            formula_asset_contract=formula_asset_contract,
        )
        network_requests += (
            result.stats.provider_calls + layout_planning.stats.constrained_provider_calls
        )
        document_runs.append(
            {
                "document_id": plan.document.document_id,
                "checkpoint": str(result.checkpoint_path),
                "translation_output": str(translation_output),
                "stats": result.stats.as_dict(),
                "layout_stats": layout_planning.stats.as_dict(),
            }
        )
    return cache, document_runs, network_requests


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        _validate_language_pair(args.source_language, args.target_language)
        load_environment_file(args.env_file)
        config = load_config(args.config)
        provider = resolve_provider(
            config,
            args.provider,
            model_override=args.model,
        )
        if (args.simulate or args.run) and args.output is None:
            raise ConfigError("--simulate and --run require --output")
        if args.max_attempts < 1:
            raise ConfigError("--max-attempts must be at least 1")
        if args.max_retranslation_attempts < 0:
            raise ConfigError("--max-retranslation-attempts cannot be negative")
        if args.request_timeout <= 0:
            raise ConfigError("--request-timeout must be greater than zero")
        documents: list[LoadedDocument] = (
            load_documents(args.inputs, explicit_pdf=args.pdf)
            if args.dry_run or args.simulate or args.run
            else []
        )
        plans = [build_translation_plan(document) for document in documents]
        if (args.check or args.run) and not provider.credential_configured:
            raise ConfigError(
                f"credential is missing: set environment variable {provider.api_key_env}"
            )
        simulation = None
        execution = None
        network_requests = 0
        if args.simulate:
            output = args.output.expanduser().resolve()
            fake_provider = FakeProvider(model=f"fake::{provider.name}::{provider.model}")
            runner_config = RunnerConfig(
                source_language=args.source_language,
                target_language=args.target_language,
                max_attempts=args.max_attempts,
                retry_failed=args.retry_failed,
            )
            cache, document_runs, _ = _execute_documents(
                plans,
                provider=fake_provider,
                requested_provider=provider.name,
                output=output,
                runner_config=runner_config,
                max_retranslation_attempts=args.max_retranslation_attempts,
                show_progress=not args.quiet,
            )
            simulation = {
                "provider": "fake",
                "cache": str(cache.path),
                "cache_entries": cache.count(),
                "fake_provider_calls": fake_provider.total_calls,
                "documents": document_runs,
            }
        elif args.run:
            output = args.output.expanduser().resolve()
            real_provider = OpenAICompatibleProvider(
                name=provider.name,
                base_url=provider.base_url,
                model=provider.model,
                api_key=os.environ[provider.api_key_env],
                timeout_seconds=args.request_timeout,
            )
            runner_config = RunnerConfig(
                source_language=args.source_language,
                target_language=args.target_language,
                max_attempts=args.max_attempts,
                retry_failed=args.retry_failed,
            )
            cache, document_runs, network_requests = _execute_documents(
                plans,
                provider=real_provider,
                requested_provider=provider.name,
                output=output,
                runner_config=runner_config,
                max_retranslation_attempts=args.max_retranslation_attempts,
                show_progress=not args.quiet,
            )
            execution = {
                "provider": provider.name,
                "model": provider.model,
                "usage": real_provider.usage_summary(),
                "cache": str(cache.path),
                "cache_entries": cache.count(),
                "documents": document_runs,
            }
    except (
        ConfigError,
        InputDiscoveryError,
        InputValidationError,
        CheckpointError,
        OSError,
        ValueError,
        RuntimeError,
    ) as exc:
        print(f"preflight failed: {exc}", file=sys.stderr)
        return 2

    _print_report(
        mode=(
            "check"
            if args.check
            else "simulate"
            if args.simulate
            else "run"
            if args.run
            else "dry-run"
        ),
        provider=provider,
        plans=plans,
        output=args.output,
        network_requests=network_requests,
        simulation=simulation,
        execution=execution,
        env_file=args.env_file,
    )
    if args.dry_run and not provider.credential_configured:
        print(
            f"warning: {provider.api_key_env} is not set; this is allowed in dry-run only",
            file=sys.stderr,
        )
    elif args.simulate and not provider.credential_configured:
        print(
            f"warning: {provider.api_key_env} is not set; this is allowed because "
            "FakeProvider makes no remote API requests",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
