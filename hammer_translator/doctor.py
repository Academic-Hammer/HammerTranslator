"""Deployment preflight for local macOS and Linux GPU installations."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

from hammer_translator.font_assets import FontAssetError, required_font_path
from hammer_translator.parsing.cli import RuntimeConfig, preflight
from hammer_translator.runtime_paths import distribution_is_installed, runtime_paths
from hammer_translator.translation.config import (
    ConfigError,
    load_config,
    load_environment_file,
    resolve_provider,
)
from hammer_translator.translation.formula_assets import MathJaxSvgRenderer


def _check(name: str, status: str, detail: str, **metadata: Any) -> dict[str, Any]:
    return {"name": name, "status": status, "detail": detail, **metadata}


def _writable_destination(path: Path) -> bool:
    candidate = path.expanduser().resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate.is_dir() and os.access(candidate, os.W_OK)


def inspect_runtime(
    *,
    profile: str,
    backend: str,
    provider: str,
    require_provider_key: bool = False,
) -> dict[str, Any]:
    paths = runtime_paths()
    checks: list[dict[str, Any]] = []
    supported_python = (3, 11) <= sys.version_info[:2] < (3, 14)
    checks.append(
        _check(
            "python",
            "ok" if supported_python else "error",
            platform.python_version(),
            executable=sys.executable,
        )
    )

    expected_system = "Darwin" if profile == "local_mac" else "Linux"
    actual_system = platform.system()
    checks.append(
        _check(
            "host_profile",
            "ok" if actual_system == expected_system else "error",
            f"profile={profile}, host={actual_system}/{platform.machine()}",
        )
    )
    installed = distribution_is_installed()
    checks.append(
        _check(
            "application_install",
            "ok" if installed else "warning",
            (
                "installed distribution is available to child processes"
                if installed
                else (
                    "running from a source checkout; install the project before "
                    "using arbitrary directories"
                )
            ),
            source_root=str(paths.source_root) if paths.source_root else None,
        )
    )

    try:
        load_environment_file(paths.env_file)
        config = load_config(paths.config_path)
        selected = resolve_provider(config, provider)
    except ConfigError as exc:
        checks.append(_check("translation_config", "error", str(exc)))
    else:
        checks.append(
            _check(
                "translation_config",
                "ok",
                f"provider={selected.name}, model={selected.model}",
                config_path=str(paths.config_path),
                env_file=str(paths.env_file),
            )
        )
        credential_status = (
            "ok"
            if selected.credential_configured
            else ("error" if require_provider_key else "warning")
        )
        checks.append(
            _check(
                "provider_credential",
                credential_status,
                (
                    f"{selected.api_key_env} is configured"
                    if selected.credential_configured
                    else f"{selected.api_key_env} is not configured"
                ),
                variable=selected.api_key_env,
            )
        )

    try:
        font_paths = {
            category: str(required_font_path(category))
            for category in ("title", "heading", "body", "caption")
        }
    except FontAssetError as exc:
        checks.append(_check("fonts", "error", str(exc)))
    else:
        checks.append(
            _check(
                "fonts",
                "ok",
                "required fonts passed SHA-256 verification",
                directory=str(paths.fonts_dir),
                resolved=font_paths,
            )
        )

    mathjax = MathJaxSvgRenderer()
    node = shutil.which(mathjax.node_executable)
    if node and mathjax.available:
        checks.append(
            _check(
                "formula_runtime",
                "ok",
                f"Node.js and MathJax {mathjax.version} are available",
                node=node,
                script=str(mathjax.script_path),
            )
        )
    else:
        missing = []
        if not node:
            missing.append("node")
        if not mathjax.available:
            missing.append("MathJax package")
        checks.append(
            _check(
                "formula_runtime",
                "error",
                f"missing {', '.join(missing)}; restore formula_runtime with npm ci",
                runtime_dir=str(paths.formula_runtime_dir),
            )
        )

    for name, path in (("jobs_root", paths.jobs_root), ("temp_dir", paths.temp_dir)):
        writable = _writable_destination(path)
        checks.append(
            _check(
                name,
                "ok" if writable else "error",
                f"destination {'is' if writable else 'is not'} writable: {path}",
                path=str(path),
            )
        )

    parser_config = RuntimeConfig(
        backend=backend,
        method="auto",
        language="ch",
        device="cuda" if profile == "linux_gpu" else "auto",
        formula=True,
        table=True,
        formula_ch=True,
        image_analysis=True,
        effort="medium",
        start_page=0,
        end_page=None,
        draw_bboxes=False,
        adapt=True,
        render_threads=4,
        window_size=8,
    )
    try:
        parser_ready, parser_messages = preflight(parser_config)
    except Exception as exc:  # diagnostic boundary: report import/runtime failures
        parser_ready = False
        parser_messages = [f"ERROR  parser preflight raised {type(exc).__name__}"]
    checks.append(
        _check(
            "parser_runtime",
            "ok" if parser_ready else "error",
            (
                "parser dependencies are ready"
                if parser_ready
                else "parser dependencies are incomplete"
            ),
            backend=backend,
            messages=parser_messages,
        )
    )

    errors = sum(check["status"] == "error" for check in checks)
    warnings = sum(check["status"] == "warning" for check in checks)
    return {
        "status": "ready" if errors == 0 else "not_ready",
        "profile": profile,
        "backend": backend,
        "summary": {"errors": errors, "warnings": warnings},
        "paths": {
            "app_home": str(paths.app_home),
            "jobs_root": str(paths.jobs_root),
            "config": str(paths.config_path),
            "env_file": str(paths.env_file),
            "fonts": str(paths.fonts_dir),
            "formula_runtime": str(paths.formula_runtime_dir),
            "model_cache": str(paths.model_cache_dir),
            "temp": str(paths.temp_dir),
        },
        "checks": checks,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hammer-doctor",
        description="Check whether a HammerTranslator host is ready to run.",
    )
    parser.add_argument("--profile", choices=["local_mac", "linux_gpu"], required=True)
    parser.add_argument(
        "--backend",
        choices=["pipeline", "vlm-engine", "hybrid-engine"],
        default="pipeline",
    )
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--require-provider-key", action="store_true")
    parser.add_argument(
        "--json", action="store_true", help="print machine-readable JSON"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = inspect_runtime(
        profile=args.profile,
        backend=args.backend,
        provider=args.provider,
        require_provider_key=args.require_provider_key,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(
            f"HammerTranslator {report['status']}: "
            f"{report['summary']['errors']} error(s), "
            f"{report['summary']['warnings']} warning(s)"
        )
        for check in report["checks"]:
            print(f"{check['status'].upper():7} {check['name']}: {check['detail']}")
            for message in check.get("messages", []):
                print(f"        {message}")
    return 0 if report["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
