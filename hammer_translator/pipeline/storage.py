"""Local, atomic storage for V1 pipeline jobs."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Protocol
from uuid import UUID, uuid4

from hammer_translator._hashing import sha256_file

from .schema import (
    ArtifactRecord,
    EventType,
    JobRecord,
    PipelineConfig,
    PipelineEvent,
    PipelineStage,
    ProgressRecord,
    utc_now,
)


class PipelineStorageError(RuntimeError):
    pass


class JobNotFoundError(PipelineStorageError):
    pass


class ConcurrentJobUpdateError(PipelineStorageError):
    pass


class JobStore(Protocol):
    root: Path

    def create_job(
        self,
        source_pdf: str | Path,
        *,
        config: PipelineConfig | None = None,
        display_name: str | None = None,
    ) -> JobRecord: ...

    def load_job(self, job_id: str) -> JobRecord: ...

    def update_job(
        self, job_id: str, mutate: Callable[[JobRecord], JobRecord]
    ) -> JobRecord: ...

    def append_event(
        self,
        job_id: str,
        event_type: EventType,
        *,
        stage: PipelineStage | None = None,
        progress: ProgressRecord | None = None,
        message: str | None = None,
        payload: dict | None = None,
    ) -> PipelineEvent: ...

    def iter_events(self, job_id: str, *, after: int = 0) -> Iterator[PipelineEvent]: ...

    def job_dir(self, job_id: str) -> Path: ...

    def resolve_artifact(self, job_id: str, relative_path: str) -> Path: ...


def _safe_filename(name: str) -> str:
    cleaned = Path(name).name.replace("\x00", "").strip()
    return cleaned or "source.pdf"


class LocalJobStore:
    """Filesystem-backed store used by the local Mac V1.

    The per-process locks are intentionally an implementation detail. A future
    multi-user deployment can replace this class with a database-backed store
    without changing the pipeline contracts.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.RLock] = {}
        self._locks_guard = threading.Lock()

    def _lock(self, job_id: str) -> threading.RLock:
        with self._locks_guard:
            return self._locks.setdefault(job_id, threading.RLock())

    @staticmethod
    def _validate_job_id(job_id: str) -> str:
        try:
            parsed = UUID(job_id)
        except ValueError as exc:
            raise JobNotFoundError("invalid job id") from exc
        canonical = str(parsed)
        if canonical != job_id:
            raise JobNotFoundError("invalid job id")
        return canonical

    def job_dir(self, job_id: str) -> Path:
        canonical = self._validate_job_id(job_id)
        path = (self.root / canonical).resolve()
        if path.parent != self.root:
            raise JobNotFoundError("job path escaped the jobs root")
        return path

    def _job_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "job.json"

    def _events_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "events.jsonl"

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _validate_pdf(path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"PDF input does not exist: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError("pipeline input must use the .pdf extension")
        with path.open("rb") as stream:
            if stream.read(5) != b"%PDF-":
                raise ValueError("pipeline input does not have a PDF signature")

    def create_job(
        self,
        source_pdf: str | Path,
        *,
        config: PipelineConfig | None = None,
        display_name: str | None = None,
    ) -> JobRecord:
        source = Path(source_pdf).expanduser().resolve()
        self._validate_pdf(source)
        job_id = str(uuid4())
        directory = self.job_dir(job_id)
        directory.mkdir(parents=False, exist_ok=False)
        for relative in ("input", "parse", "translation", "output", "logs"):
            (directory / relative).mkdir()
        destination = directory / "input" / "source.pdf"
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        try:
            with source.open("rb") as reader, temporary.open("wb") as writer:
                shutil.copyfileobj(reader, writer, length=1024 * 1024)
                writer.flush()
                os.fsync(writer.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()

        source_artifact = ArtifactRecord(
            name="source_pdf",
            kind="source_pdf",
            relative_path="input/source.pdf",
            media_type="application/pdf",
            size_bytes=destination.stat().st_size,
            sha256=sha256_file(destination),
            metadata={"original_filename": _safe_filename(source.name)},
        )
        job = JobRecord(
            job_id=job_id,
            display_name=display_name or _safe_filename(source.name),
            config=config or PipelineConfig(),
            artifacts=[source_artifact],
        )
        self._write_json_atomic(self._job_path(job_id), job.model_dump(mode="json"))
        self.append_event(
            job_id,
            EventType.JOB_CREATED,
            message="pipeline job created",
            payload={"source_artifact": source_artifact.model_dump(mode="json")},
        )
        return self.load_job(job_id)

    def load_job(self, job_id: str) -> JobRecord:
        path = self._job_path(job_id)
        if not path.is_file():
            raise JobNotFoundError(f"pipeline job not found: {job_id}")
        try:
            return JobRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise PipelineStorageError(f"pipeline job state is unreadable: {job_id}") from exc

    def save_job(self, job: JobRecord, *, expected_revision: int) -> JobRecord:
        with self._lock(job.job_id):
            current = self.load_job(job.job_id)
            if current.revision != expected_revision:
                raise ConcurrentJobUpdateError(
                    f"job revision changed: expected {expected_revision}, got {current.revision}"
                )
            saved = job.model_copy(
                update={"revision": current.revision + 1, "updated_at": utc_now()}
            )
            validated = JobRecord.model_validate(saved.model_dump(mode="python"))
            self._write_json_atomic(
                self._job_path(job.job_id), validated.model_dump(mode="json")
            )
            return validated

    def update_job(
        self, job_id: str, mutate: Callable[[JobRecord], JobRecord]
    ) -> JobRecord:
        with self._lock(job_id):
            current = self.load_job(job_id)
            changed = mutate(current.model_copy(deep=True))
            if changed.job_id != current.job_id:
                raise PipelineStorageError("job mutation cannot change job_id")
            return self.save_job(changed, expected_revision=current.revision)

    def _next_event_sequence(self, job_id: str) -> int:
        path = self._events_path(job_id)
        if not path.is_file():
            return 1
        sequence = 0
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    sequence = PipelineEvent.model_validate_json(line).sequence
        return sequence + 1

    def append_event(
        self,
        job_id: str,
        event_type: EventType,
        *,
        stage: PipelineStage | None = None,
        progress: ProgressRecord | None = None,
        message: str | None = None,
        payload: dict | None = None,
    ) -> PipelineEvent:
        with self._lock(job_id):
            job = self.load_job(job_id)
            event = PipelineEvent(
                sequence=self._next_event_sequence(job_id),
                job_id=job_id,
                event_type=event_type,
                stage=stage,
                status=job.status,
                progress=progress,
                message=message,
                payload=payload or {},
            )
            path = self._events_path(job_id)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event.model_dump(mode="json"), ensure_ascii=False))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            return event

    def iter_events(self, job_id: str, *, after: int = 0) -> Iterator[PipelineEvent]:
        self.load_job(job_id)
        path = self._events_path(job_id)
        if not path.is_file():
            return
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                event = PipelineEvent.model_validate_json(line)
                if event.sequence > after:
                    yield event

    def resolve_artifact(self, job_id: str, relative_path: str) -> Path:
        validated = ArtifactRecord(
            name="artifact", kind="unknown", relative_path=relative_path
        ).relative_path
        directory = self.job_dir(job_id)
        path = (directory / validated).resolve()
        if directory not in path.parents:
            raise PipelineStorageError("artifact path escaped the job directory")
        return path
