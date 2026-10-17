"""Atomic per-document execution checkpoint."""

from __future__ import annotations

import json
import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .jsonio import write_json_atomic


CHECKPOINT_SCHEMA_VERSION = "translation-checkpoint-v1"


class CheckpointError(ValueError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CheckpointStore:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._data: dict[str, Any] | None = None

    def initialize(
        self,
        *,
        document_id: str,
        source_hash: str,
        run_signature: Mapping[str, Any],
        unit_ids: Iterable[str],
    ) -> None:
        with self._lock:
            if self.path.exists():
                data = self._read()
                self._validate_existing(data, document_id, source_hash, run_signature)
            else:
                now = _utc_now()
                data = {
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "document_id": document_id,
                    "source_hash": source_hash,
                    "run_signature": dict(run_signature),
                    "run_status": "in_progress",
                    "created_at": now,
                    "updated_at": now,
                    "units": {},
                }
            units = data.setdefault("units", {})
            for unit_id in unit_ids:
                units.setdefault(
                    unit_id,
                    {
                        "execution_status": "pending",
                        "attempts": 0,
                        "cache_hit": False,
                        "cache_key": None,
                        "response_hash": None,
                        "request_id": None,
                        "last_error": None,
                        "quality": None,
                        "updated_at": _utc_now(),
                    },
                )
            self._data = data
            self._write_atomic()

    def _read(self) -> dict[str, Any]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointError(f"invalid checkpoint {self.path}: {exc}") from exc
        if not isinstance(value, dict):
            raise CheckpointError(f"checkpoint root must be an object: {self.path}")
        return value

    def _validate_existing(
        self,
        data: Mapping[str, Any],
        document_id: str,
        source_hash: str,
        run_signature: Mapping[str, Any],
    ) -> None:
        expected = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "document_id": document_id,
            "source_hash": source_hash,
            "run_signature": dict(run_signature),
        }
        for key, value in expected.items():
            if data.get(key) != value:
                raise CheckpointError(
                    f"checkpoint {key} mismatch; use a new output directory instead of "
                    "silently reusing incompatible progress"
                )

    def unit_state(self, unit_id: str) -> dict[str, Any]:
        with self._lock:
            if self._data is None:
                raise CheckpointError("checkpoint is not initialized")
            return deepcopy(self._data["units"][unit_id])

    def update_unit(self, unit_id: str, **changes: Any) -> None:
        with self._lock:
            if self._data is None:
                raise CheckpointError("checkpoint is not initialized")
            state = self._data["units"][unit_id]
            state.update(changes)
            state["updated_at"] = _utc_now()
            self._data["updated_at"] = state["updated_at"]
            self._write_atomic()

    def finalize(self) -> None:
        with self._lock:
            if self._data is None:
                raise CheckpointError("checkpoint is not initialized")
            statuses = {
                state.get("execution_status") for state in self._data["units"].values()
            }
            if "failed" in statuses:
                run_status = "completed_with_errors"
            elif statuses <= {"completed", "preserved", "skipped", "needs_review"}:
                run_status = "completed"
            else:
                run_status = "in_progress"
            self._data["run_status"] = run_status
            self._data["updated_at"] = _utc_now()
            self._write_atomic()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            if self._data is None:
                raise CheckpointError("checkpoint is not initialized")
            return deepcopy(self._data)

    def _write_atomic(self) -> None:
        if self._data is None:
            raise CheckpointError("checkpoint is not initialized")
        write_json_atomic(self.path, self._data)
