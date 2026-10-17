"""Command-line entry point for post-processing one MinerU document."""

from __future__ import annotations

import argparse
from pathlib import Path

from .adapter import AdapterOptions, MinerUAdapter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert MinerU middle.json to downstream content JSON")
    parser.add_argument("middle_json", type=Path, help="MinerU *_middle.json file")
    parser.add_argument("--pdf", type=Path, help="Original PDF; enables rich style extraction")
    parser.add_argument("--output-dir", type=Path, help="Output directory (defaults to the input directory)")
    parser.add_argument("--name", help="Output document name (defaults to the middle JSON stem)")
    parser.add_argument("--image-dir", default="images", help="Image path prefix stored in JSON")
    parser.add_argument("--no-styles", action="store_true", help="Do not extract PDF font spans")
    parser.add_argument("--drop-discarded", action="store_true", help="Drop headers, footers and page numbers")
    parser.add_argument("--no-merge", action="store_true", help="Do not merge continuation blocks")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    adapter = MinerUAdapter(
        AdapterOptions(
            image_dir=args.image_dir,
            preserve_discarded=not args.drop_discarded,
            merge_paragraphs=not args.no_merge,
            enrich_styles=not args.no_styles,
        )
    )
    result = adapter.adapt_file(args.middle_json, args.pdf, args.output_dir, args.name)
    for kind, path in result.output_files.items():
        print(f"{kind}: {path}")
    for warning in result.warnings:
        print(f"warning: {warning}")
    return 0
