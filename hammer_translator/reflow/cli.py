"""CLI for the thin Translation-to-Reflow compatibility layer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from hammer_translator.translation.jsonio import write_json_atomic

from .renderer import ReflowContractError, ThinReflowRenderer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="render_translated_pdf.py",
        description=(
            "Render a translation artifact with shared text/formula layout metrics. "
            "This command never rewrites translation text and never calls an LLM."
        ),
    )
    parser.add_argument("translation_json", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="Output PDF path")
    parser.add_argument(
        "--feedback",
        type=Path,
        help="Feedback JSON path (default: <output stem>_feedback.json)",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Render verified elements while preserving unresolved elements in the source PDF",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    feedback_path = (
        args.feedback.expanduser().resolve()
        if args.feedback
        else output.with_name(f"{output.stem}_feedback.json")
    )
    try:
        report = ThinReflowRenderer(allow_partial=args.allow_partial).render(
            args.translation_json, output
        )
        write_json_atomic(feedback_path, report)
    except (ReflowContractError, OSError, ValueError) as exc:
        print(f"reflow failed: {exc}", file=sys.stderr)
        return 2
    printable = {
        "status": report["status"],
        "network_requests": 0,
        "source_translation_artifact": report["source_translation_artifact"],
        "source_pdf": report["source_pdf"],
        "output_pdf": report["output_pdf"],
        "feedback": str(feedback_path),
        "summary": report["summary"],
    }
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    return 0 if report["status"] in {"rendered", "partial_needs_review"} else 3


if __name__ == "__main__":
    raise SystemExit(main())
