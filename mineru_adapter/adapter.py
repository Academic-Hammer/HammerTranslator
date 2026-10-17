"""Public orchestration API for MinerU output adaptation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .content import ConversionOptions, MiddleJsonConverter
from .paragraphs import classify_document, group_logical_paragraphs
from .styles import StyleEnricher


@dataclass(frozen=True)
class AdapterOptions:
    image_dir: str = "images"
    preserve_discarded: bool = True
    split_outside_lines: bool = True
    classify_sections: bool = True
    merge_paragraphs: bool = True
    enrich_styles: bool = True
    clip_margin: float = 1.0


@dataclass
class AdaptationResult:
    content_list: list[dict[str, Any]]
    enriched: list[dict[str, Any]] | None = None
    warnings: list[str] = field(default_factory=list)
    output_files: dict[str, Path] = field(default_factory=dict)


class MinerUAdapter:
    """Turn MinerU ``middle.json`` into the stable downstream JSON contract."""

    def __init__(self, options: AdapterOptions | None = None):
        self.options = options or AdapterOptions()

    def adapt(
        self,
        middle_json: dict[str, Any],
        pdf_path: str | Path | None = None,
    ) -> AdaptationResult:
        converter = MiddleJsonConverter(
            ConversionOptions(
                image_dir=self.options.image_dir,
                preserve_discarded=self.options.preserve_discarded,
                split_outside_lines=self.options.split_outside_lines,
            )
        )
        content_list = converter.convert(middle_json)
        if self.options.classify_sections:
            content_list = classify_document(content_list)
        if self.options.merge_paragraphs:
            content_list = group_logical_paragraphs(content_list)

        warnings = list(converter.warnings)
        enriched = None
        if pdf_path is not None and self.options.enrich_styles:
            enriched, style_warnings = StyleEnricher(self.options.clip_margin).enrich(content_list, pdf_path)
            warnings.extend(style_warnings)
        return AdaptationResult(content_list=content_list, enriched=enriched, warnings=warnings)

    def adapt_file(
        self,
        middle_json_path: str | Path,
        pdf_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        document_name: str | None = None,
    ) -> AdaptationResult:
        middle_path = Path(middle_json_path)
        with middle_path.open("r", encoding="utf-8") as handle:
            middle_json = json.load(handle)
        if not isinstance(middle_json, dict):
            raise ValueError(f"Expected a JSON object in {middle_path}")

        result = self.adapt(middle_json, pdf_path)
        target_dir = Path(output_dir) if output_dir is not None else middle_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        stem = document_name or _document_stem(middle_path)
        content_path = target_dir / f"{stem}_content_list.json"
        _write_json(content_path, result.content_list)
        result.output_files["content_list"] = content_path
        if result.enriched is not None:
            enriched_path = target_dir / f"{stem}_enriched.json"
            _write_json(enriched_path, result.enriched)
            result.output_files["enriched"] = enriched_path
        return result


def _document_stem(path: Path) -> str:
    suffix = "_middle"
    return path.stem[: -len(suffix)] if path.stem.endswith(suffix) else path.stem


def _write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
