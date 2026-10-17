"""SQLite content-hash cache for provider responses."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .provider import ProviderResponse, TranslationRequest, redact_sensitive_text


CACHE_SCHEMA_VERSION = "translation-cache-v1"
SAFE_METADATA_KEYS = {
    "request_id",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "finish_reason",
}


class CacheError(ValueError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_metadata(metadata: Mapping[str, Any], request_id: str | None) -> dict[str, Any]:
    safe = {key: value for key, value in metadata.items() if key in SAFE_METADATA_KEYS}
    if request_id:
        safe["request_id"] = request_id
    return {
        key: redact_sensitive_text(value) if isinstance(value, str) else value
        for key, value in safe.items()
    }


@dataclass(frozen=True)
class CacheEntry:
    cache_key: str
    provider: str
    model: str
    response_text: str
    response_hash: str
    metadata: dict[str, Any]
    created_at: str


class TranslationCache:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    cache_key TEXT PRIMARY KEY,
                    schema_version TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    source_language TEXT NOT NULL,
                    target_language TEXT NOT NULL,
                    element_type TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    glossary_version TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    response_text TEXT NOT NULL,
                    response_hash TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(cache_entries)").fetchall()
            }
            required_columns = {
                "cache_key",
                "schema_version",
                "provider",
                "model",
                "source_language",
                "target_language",
                "element_type",
                "prompt_version",
                "glossary_version",
                "mode",
                "response_text",
                "response_hash",
                "metadata_json",
                "created_at",
            }
            missing = sorted(required_columns - columns)
            if missing:
                raise CacheError(
                    "incompatible translation cache schema; use a new output directory "
                    f"(missing columns: {', '.join(missing)})"
                )

    def get(self, cache_key: str) -> CacheEntry | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM cache_entries WHERE cache_key = ?", (cache_key,)
            ).fetchone()
        if row is None:
            return None
        actual_hash = hashlib.sha256(row["response_text"].encode("utf-8")).hexdigest()
        if actual_hash != row["response_hash"]:
            self.delete(cache_key)
            return None
        return CacheEntry(
            cache_key=row["cache_key"],
            provider=row["provider"],
            model=row["model"],
            response_text=row["response_text"],
            response_hash=row["response_hash"],
            metadata=json.loads(row["metadata_json"]),
            created_at=row["created_at"],
        )

    def put(self, request: TranslationRequest, response: ProviderResponse) -> CacheEntry:
        cache_key = request.cache_key()
        response_hash = hashlib.sha256(response.text.encode("utf-8")).hexdigest()
        metadata = _safe_metadata(response.metadata, response.request_id)
        created_at = _utc_now()
        values = (
            cache_key,
            CACHE_SCHEMA_VERSION,
            request.provider,
            request.model,
            request.source_language,
            request.target_language,
            request.element_type,
            request.prompt_version,
            request.glossary_version,
            request.mode,
            response.text,
            response_hash,
            json.dumps(metadata, ensure_ascii=False, sort_keys=True),
            created_at,
        )
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO cache_entries (
                    cache_key, schema_version, provider, model,
                    source_language, target_language, element_type,
                    prompt_version, glossary_version, mode, response_text,
                    response_hash, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
        entry = self.get(cache_key)
        if entry is None:
            raise RuntimeError(f"failed to read cache entry after insert: {cache_key}")
        return entry

    def delete(self, cache_key: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM cache_entries WHERE cache_key = ?", (cache_key,))

    def count(self) -> int:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM cache_entries").fetchone()
        return int(row["count"])
