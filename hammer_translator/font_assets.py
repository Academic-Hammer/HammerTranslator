"""Project-owned font discovery and integrity verification."""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FONTS_DIR = PROJECT_ROOT / "assets" / "fonts"
FONT_MANIFEST_PATH = DEFAULT_FONTS_DIR / "manifest.json"
FONT_DIRECTORY_ENV = "HAMMERTRANSLATOR_FONTS_DIR"


class FontAssetError(RuntimeError):
    """Raised when a required font is missing, changed, or misconfigured."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def font_manifest() -> dict[str, Any]:
    try:
        payload = json.loads(FONT_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FontAssetError(f"font manifest is unreadable: {FONT_MANIFEST_PATH}") from exc
    if payload.get("schema_version") != "hammertranslator-font-assets-v1":
        raise FontAssetError("font manifest has an unsupported schema version")
    fonts = payload.get("fonts")
    if not isinstance(fonts, dict) or not fonts:
        raise FontAssetError("font manifest does not define any fonts")
    return payload


def configured_fonts_dir() -> Path:
    configured = os.environ.get(FONT_DIRECTORY_ENV, "").strip()
    return Path(configured).expanduser().resolve() if configured else DEFAULT_FONTS_DIR


def font_filename(category: str) -> str:
    for filename, record in font_manifest()["fonts"].items():
        if category in record.get("categories", []):
            return str(filename)
    raise FontAssetError(f"font category is not registered: {category}")


def expected_font_hash(filename: str) -> str:
    record = font_manifest()["fonts"].get(filename)
    if not isinstance(record, dict) or not record.get("sha256"):
        raise FontAssetError(f"font file is not registered: {filename}")
    return str(record["sha256"])


def _verified(path: Path, expected_hash: str) -> bool:
    return path.is_file() and _sha256(path) == expected_hash


def required_font_path(category: str) -> Path:
    filename = font_filename(category)
    path = configured_fonts_dir() / filename
    if not path.is_file():
        raise FontAssetError(
            f"required font is missing: {path}; provide assets/fonts or set "
            f"{FONT_DIRECTORY_ENV}"
        )
    if not _verified(path, expected_font_hash(filename)):
        raise FontAssetError(f"required font hash does not match manifest: {path}")
    return path.resolve()


def canonical_project_relative_path(category: str) -> str:
    return (Path("assets") / "fonts" / font_filename(category)).as_posix()


def resolve_registered_font(record: Mapping[str, Any]) -> Path:
    """Resolve an artifact font locally and require its recorded hash.

    Translation artifacts retain their original absolute path, but Reflow may
    run after a checkout or job has moved. The configured font directory and
    project-relative path provide deterministic fallbacks.
    """

    expected_hash = str(record.get("sha256") or "")
    if not expected_hash:
        raise FontAssetError("registered font does not contain a SHA-256 hash")

    raw_path = str(record.get("path") or "").strip()
    relative = str(record.get("project_relative_path") or "").strip()
    filename = Path(relative or raw_path).name
    candidates: list[Path] = []
    if raw_path:
        candidates.append(Path(raw_path).expanduser().resolve())
    if filename:
        candidates.append((configured_fonts_dir() / filename).resolve())
    if relative:
        relative_path = Path(relative)
        if not relative_path.is_absolute() and ".." not in relative_path.parts:
            candidates.append((PROJECT_ROOT / relative_path).resolve())

    seen: set[Path] = set()
    changed: list[Path] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if not candidate.is_file():
            continue
        if _verified(candidate, expected_hash):
            return candidate
        changed.append(candidate)
    if changed:
        raise FontAssetError(f"registered font hash changed: {changed[0]}")
    raise FontAssetError(f"registered font is unavailable: {filename or raw_path}")
