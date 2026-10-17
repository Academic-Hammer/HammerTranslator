"""Deterministic input discovery and validation for Translation Stage."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import fitz
from pydantic import ValidationError

from .schema import INPUT_SCHEMA_VERSION, InputBlock
from .taxonomy import TaxonomyDecision, classify_element


class InputDiscoveryError(ValueError):
    """Raised when an input path cannot resolve to exactly one source JSON."""


class InputValidationError(ValueError):
    """Raised when a selected JSON does not satisfy the input contract."""


@dataclass(frozen=True)
class DiscoveredInput:
    source_json: Path
    selection_reason: str


@dataclass(frozen=True)
class LoadedBlock:
    block_id: str
    source_index: int
    data: InputBlock
    taxonomy: TaxonomyDecision
    raw: dict[str, Any]


@dataclass(frozen=True)
class LoadedDocument:
    schema_version: str
    document_id: str
    source_json: Path
    source_pdf: Path
    selection_reason: str
    source_hash: str
    page_count: int
    blocks: tuple[LoadedBlock, ...]

    def public_summary(self) -> dict[str, Any]:
        type_counts = Counter(block.taxonomy.element_type.value for block in self.blocks)
        policy_counts = Counter(block.taxonomy.policy.value for block in self.blocks)
        return {
            "schema_version": self.schema_version,
            "document_id": self.document_id,
            "source_json": str(self.source_json),
            "source_pdf": str(self.source_pdf),
            "selection_reason": self.selection_reason,
            "source_hash": self.source_hash,
            "page_count": self.page_count,
            "block_count": len(self.blocks),
            "type_counts": dict(sorted(type_counts.items())),
            "policy_counts": dict(sorted(policy_counts.items())),
        }


def _is_adapted_content_list(path: Path) -> bool:
    name = path.name
    return (
        name.endswith("_content_list.json")
        and not name.endswith("_mineru_content_list.json")
        and not name.endswith("_content_list_v2.json")
        and not name.endswith("_content_list_translated.json")
    )


def _one_candidate(candidates: list[Path], *, directory: Path, label: str) -> Path | None:
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise InputDiscoveryError(
            f"multiple {label} candidates in {directory}: {names}; pass one JSON file explicitly"
        )
    return candidates[0] if candidates else None


def discover_source(path: str | Path) -> DiscoveredInput:
    """Resolve one file or one parse-result directory using the approved order."""

    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise InputDiscoveryError(f"input not found: {source}")
    if source.is_file():
        if source.suffix.lower() != ".json":
            raise InputDiscoveryError(f"explicit input must be a JSON file: {source}")
        return DiscoveredInput(source_json=source, selection_reason="explicit_json")
    if not source.is_dir():
        raise InputDiscoveryError(f"input is neither a JSON file nor directory: {source}")

    enriched = sorted(source.glob("*_enriched.json"))
    selected = _one_candidate(enriched, directory=source, label="enriched JSON")
    if selected is not None:
        return DiscoveredInput(selected, "directory_preferred_enriched")

    adapted = sorted(path for path in source.glob("*_content_list.json") if _is_adapted_content_list(path))
    selected = _one_candidate(adapted, directory=source, label="adapted content-list JSON")
    if selected is not None:
        return DiscoveredInput(selected, "directory_fallback_adapted_content_list")

    raise InputDiscoveryError(
        f"no supported JSON in {source}; expected one *_enriched.json or adapted "
        "*_content_list.json. Native *_mineru_content_list.json and "
        "*_content_list_v2.json are never selected automatically"
    )


def discover_sources(paths: Sequence[str | Path]) -> list[DiscoveredInput]:
    if not paths:
        raise InputDiscoveryError("dry-run requires at least one input JSON file or directory")
    discovered = [discover_source(path) for path in paths]
    seen: set[Path] = set()
    for item in discovered:
        if item.source_json in seen:
            raise InputDiscoveryError(f"duplicate input resolves to {item.source_json}")
        seen.add(item.source_json)
    return discovered


def document_id_from_json(path: Path) -> str:
    stem = path.stem
    for suffix in (
        "_mineru_content_list",
        "_content_list_v2",
        "_content_list",
        "_enriched",
    ):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def discover_pdf(source_json: Path, explicit_pdf: str | Path | None = None) -> Path:
    if explicit_pdf is not None:
        pdf = Path(explicit_pdf).expanduser().resolve()
        if not pdf.is_file() or pdf.suffix.lower() != ".pdf":
            raise InputDiscoveryError(f"explicit PDF not found or not a PDF file: {pdf}")
        return pdf

    document_id = document_id_from_json(source_json)
    parent = source_json.parent
    for candidate in (parent / f"{document_id}_origin.pdf", parent / f"{document_id}.pdf"):
        if candidate.is_file():
            return candidate.resolve()

    origin_pdfs = sorted(parent.glob("*_origin.pdf"))
    selected = _one_candidate(origin_pdfs, directory=parent, label="origin PDF")
    if selected is not None:
        return selected.resolve()

    pdfs = sorted(path for path in parent.glob("*.pdf") if not path.name.endswith("_layout.pdf"))
    selected = _one_candidate(pdfs, directory=parent, label="PDF")
    if selected is not None:
        return selected.resolve()
    raise InputDiscoveryError(
        f"no source PDF found beside {source_json}; pass --pdf for a single input"
    )


def _load_json_array(path: Path) -> tuple[bytes, list[dict[str, Any]]]:
    raw_bytes = path.read_bytes()
    try:
        value = json.loads(raw_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise InputValidationError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, list):
        raise InputValidationError(f"translation input root must be a JSON array: {path}")
    if not value:
        raise InputValidationError(f"translation input contains no blocks: {path}")
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise InputValidationError(f"block[{index}] must be a JSON object")
    return raw_bytes, value


def _validate_pdf_bounds(block: InputBlock, *, index: int, document: fitz.Document) -> None:
    if block.page_idx >= document.page_count:
        raise InputValidationError(
            f"block[{index}].page_idx={block.page_idx} is outside PDF page range "
            f"0..{document.page_count - 1}"
        )
    page = document[block.page_idx]
    width, height = float(page.rect.width), float(page.rect.height)
    x0, y0, x1, y1 = block.bbox
    tolerance = 1.0
    if x0 < -tolerance or y0 < -tolerance or x1 > width + tolerance or y1 > height + tolerance:
        raise InputValidationError(
            f"block[{index}].bbox={list(block.bbox)} is outside page {block.page_idx} "
            f"bounds [0, 0, {width:.2f}, {height:.2f}]"
        )


def _validate_blocks(raw_items: Iterable[dict[str, Any]], document: fitz.Document) -> tuple[LoadedBlock, ...]:
    loaded: list[LoadedBlock] = []
    page_ordinals: defaultdict[int, int] = defaultdict(int)
    used_ids: set[str] = set()

    for index, raw in enumerate(raw_items):
        try:
            block = InputBlock.model_validate(raw)
        except ValidationError as exc:
            details = "; ".join(
                f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                for error in exc.errors(include_url=False)
            )
            raise InputValidationError(f"block[{index}] failed schema validation: {details}") from exc

        _validate_pdf_bounds(block, index=index, document=document)
        page_ordinals[block.page_idx] += 1
        generated_id = f"p{block.page_idx + 1:04d}-b{page_ordinals[block.page_idx]:04d}"
        block_id = block.block_id or generated_id
        if block_id in used_ids:
            raise InputValidationError(f"duplicate block_id {block_id!r} at block[{index}]")
        used_ids.add(block_id)
        loaded.append(
            LoadedBlock(
                block_id=block_id,
                source_index=index,
                data=block,
                taxonomy=classify_element(raw),
                raw=deepcopy(raw),
            )
        )
    return tuple(loaded)


def load_document(
    discovered: DiscoveredInput,
    *,
    explicit_pdf: str | Path | None = None,
) -> LoadedDocument:
    source_json = discovered.source_json
    source_pdf = discover_pdf(source_json, explicit_pdf)
    raw_bytes, raw_items = _load_json_array(source_json)
    try:
        document = fitz.open(source_pdf)
    except Exception as exc:
        raise InputValidationError(f"cannot open source PDF {source_pdf}: {exc}") from exc
    try:
        if document.page_count < 1:
            raise InputValidationError(f"source PDF has no pages: {source_pdf}")
        blocks = _validate_blocks(raw_items, document)
        page_count = document.page_count
    finally:
        document.close()

    return LoadedDocument(
        schema_version=INPUT_SCHEMA_VERSION,
        document_id=document_id_from_json(source_json),
        source_json=source_json,
        source_pdf=source_pdf,
        selection_reason=discovered.selection_reason,
        source_hash=hashlib.sha256(raw_bytes).hexdigest(),
        page_count=page_count,
        blocks=blocks,
    )


def load_documents(
    paths: Sequence[str | Path],
    *,
    explicit_pdf: str | Path | None = None,
) -> list[LoadedDocument]:
    discovered = discover_sources(paths)
    if explicit_pdf is not None and len(discovered) != 1:
        raise InputDiscoveryError("--pdf can only be used when exactly one input is selected")
    documents = [
        load_document(item, explicit_pdf=explicit_pdf if len(discovered) == 1 else None)
        for item in discovered
    ]
    document_ids = Counter(document.document_id for document in documents)
    duplicates = sorted(document_id for document_id, count in document_ids.items() if count > 1)
    if duplicates:
        raise InputDiscoveryError(
            "duplicate document_id values in one run: "
            f"{', '.join(duplicates)}; select only one parse result per document"
        )
    return documents
