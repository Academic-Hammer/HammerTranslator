"""Resumable, observable pipeline state machine."""

from __future__ import annotations

from pathlib import Path

from hammer_translator._hashing import sha256_file

from .executor import (
    PipelineExecutor,
    StageCancelledError,
    StageContext,
    StageExecutionError,
)
from .schema import (
    PIPELINE_STAGES,
    STAGE_TO_JOB_STATUS,
    ArtifactRecord,
    ErrorRecord,
    EventType,
    JobRecord,
    JobStatus,
    PipelineStage,
    ProgressRecord,
    StageStatus,
    TERMINAL_JOB_STATUSES,
    utc_now,
)
from .storage import JobStore, PipelineStorageError


class PipelineRunner:
    def __init__(self, store: JobStore, executor: PipelineExecutor):
        self.store = store
        self.executor = executor

    def request_cancel(self, job_id: str) -> JobRecord:
        job = self.store.load_job(job_id)
        if job.status in TERMINAL_JOB_STATUSES:
            return job

        def mutate(current: JobRecord) -> JobRecord:
            current.cancel_requested = True
            return current

        updated = self.store.update_job(job_id, mutate)
        self.store.append_event(
            job_id,
            EventType.CANCEL_REQUESTED,
            stage=updated.current_stage,
            message="pipeline cancellation requested",
        )
        return updated

    def _cancel_job(self, job_id: str, stage: PipelineStage | None) -> JobRecord:
        def mutate(job: JobRecord) -> JobRecord:
            if stage is not None:
                record = job.stage_record(stage)
                if record.status is StageStatus.RUNNING:
                    record.status = StageStatus.CANCELLED
                    record.finished_at = utc_now()
            job.status = JobStatus.CANCELLED
            job.current_stage = None
            job.error = None
            return job

        updated = self.store.update_job(job_id, mutate)
        self.store.append_event(
            job_id,
            EventType.JOB_CANCELLED,
            stage=stage,
            message="pipeline job cancelled",
        )
        return updated

    def _resume_failed_job(self, job_id: str) -> JobRecord:
        def mutate(job: JobRecord) -> JobRecord:
            if job.status is not JobStatus.FAILED:
                return job
            failed = next(
                (record for record in job.stages if record.status is StageStatus.FAILED),
                None,
            )
            if failed is None:
                raise PipelineStorageError("failed job has no failed stage")
            failed.status = StageStatus.PENDING
            failed.finished_at = None
            failed.progress = ProgressRecord()
            failed.error = None
            job.status = JobStatus.QUEUED
            job.current_stage = None
            job.cancel_requested = False
            job.error = None
            return job

        updated = self.store.update_job(job_id, mutate)
        self.store.append_event(
            job_id,
            EventType.JOB_RESUMED,
            message="failed pipeline job queued for resume",
        )
        return updated

    def _start_stage(self, job_id: str, stage: PipelineStage) -> JobRecord:
        def mutate(job: JobRecord) -> JobRecord:
            record = job.stage_record(stage)
            record.status = StageStatus.RUNNING
            record.attempt += 1
            record.started_at = utc_now()
            record.finished_at = None
            record.progress = ProgressRecord()
            record.error = None
            job.status = STAGE_TO_JOB_STATUS[stage]
            job.current_stage = stage
            job.error = None
            return job

        updated = self.store.update_job(job_id, mutate)
        self.store.append_event(
            job_id,
            EventType.STAGE_STARTED,
            stage=stage,
            message=f"{stage.value} stage started",
            payload={"attempt": updated.stage_record(stage).attempt},
        )
        return updated

    def _record_progress(
        self,
        job_id: str,
        stage: PipelineStage,
        current: int,
        total: int | None,
        message: str | None,
        detail: dict | None,
    ) -> None:
        percent = None
        if total:
            percent = round(current * 100.0 / total, 2)
        progress = ProgressRecord(
            current=current,
            total=total,
            percent=percent,
            message=message,
            detail=detail or {},
        )

        def mutate(job: JobRecord) -> JobRecord:
            job.stage_record(stage).progress = progress
            return job

        self.store.update_job(job_id, mutate)
        self.store.append_event(
            job_id,
            EventType.STAGE_PROGRESS,
            stage=stage,
            progress=progress,
            message=message,
        )

    def _artifact_record(
        self, job_id: str, stage: PipelineStage, produced
    ) -> ArtifactRecord:
        path = self.store.resolve_artifact(job_id, produced.relative_path)
        if produced.node_type == "file":
            if not path.is_file():
                raise StageExecutionError(
                    f"stage did not create declared artifact: {produced.relative_path}",
                    code="declared_artifact_missing",
                )
            size = path.stat().st_size
            digest = sha256_file(path)
        else:
            if not path.is_dir():
                raise StageExecutionError(
                    f"stage did not create declared directory: {produced.relative_path}",
                    code="declared_artifact_missing",
                )
            size = None
            digest = None
        return ArtifactRecord(
            name=produced.name,
            kind=produced.kind,
            relative_path=produced.relative_path,
            node_type=produced.node_type,
            media_type=produced.media_type,
            size_bytes=size,
            sha256=digest,
            stage=stage,
            metadata=produced.metadata,
        )

    def _finish_stage(self, job_id: str, stage: PipelineStage, result) -> JobRecord:
        artifacts = [self._artifact_record(job_id, stage, item) for item in result.artifacts]

        def mutate(job: JobRecord) -> JobRecord:
            record = job.stage_record(stage)
            record.status = (
                StageStatus.COMPLETED_WITH_WARNINGS
                if result.warnings
                else StageStatus.COMPLETED
            )
            record.finished_at = utc_now()
            record.artifacts = artifacts
            record.warnings = list(result.warnings)
            record.error = None
            job.artifacts.extend(artifacts)
            job.warnings.extend(result.warnings)
            return job

        updated = self.store.update_job(job_id, mutate)
        for artifact in artifacts:
            self.store.append_event(
                job_id,
                EventType.ARTIFACT_CREATED,
                stage=stage,
                message=f"artifact created: {artifact.name}",
                payload={"artifact": artifact.model_dump(mode="json")},
            )
        for warning in result.warnings:
            self.store.append_event(
                job_id,
                EventType.WARNING,
                stage=stage,
                message=warning.message,
                payload={"warning": warning.model_dump(mode="json")},
            )
        self.store.append_event(
            job_id,
            EventType.STAGE_COMPLETED,
            stage=stage,
            message=f"{stage.value} stage completed",
            payload={
                "status": updated.stage_record(stage).status.value,
                "artifact_count": len(artifacts),
                "warning_count": len(result.warnings),
            },
        )
        return updated

    def _fail_job(
        self,
        job_id: str,
        stage: PipelineStage,
        error: StageExecutionError,
    ) -> JobRecord:
        safe_error = ErrorRecord(
            code=error.code,
            message=str(error),
            stage=stage,
            retryable=error.retryable,
            detail=error.detail,
        )

        def mutate(job: JobRecord) -> JobRecord:
            record = job.stage_record(stage)
            record.status = StageStatus.FAILED
            record.finished_at = utc_now()
            record.error = safe_error
            job.status = JobStatus.FAILED
            job.current_stage = None
            job.error = safe_error
            return job

        updated = self.store.update_job(job_id, mutate)
        self.store.append_event(
            job_id,
            EventType.JOB_FAILED,
            stage=stage,
            message=safe_error.message,
            payload={"error": safe_error.model_dump(mode="json")},
        )
        return updated

    def run(self, job_id: str, *, retry_failed: bool = False) -> JobRecord:
        job = self.store.load_job(job_id)
        if job.status in {JobStatus.COMPLETED, JobStatus.COMPLETED_WITH_WARNINGS}:
            return job
        if job.status is JobStatus.CANCELLED:
            return job
        if job.status is JobStatus.FAILED:
            if not retry_failed:
                return job
            job = self._resume_failed_job(job_id)
        if job.cancel_requested:
            return self._cancel_job(job_id, job.current_stage)

        for stage in PIPELINE_STAGES:
            job = self.store.load_job(job_id)
            stage_record = job.stage_record(stage)
            if stage_record.status in {
                StageStatus.COMPLETED,
                StageStatus.COMPLETED_WITH_WARNINGS,
            }:
                continue
            if job.cancel_requested:
                return self._cancel_job(job_id, job.current_stage)
            job = self._start_stage(job_id, stage)
            context = StageContext(
                job_id=job_id,
                job_dir=self.store.job_dir(job_id),
                stage=stage,
                attempt=job.stage_record(stage).attempt,
                source_pdf=self.store.job_dir(job_id) / job.source_pdf,
                document_name=job.display_name,
                config=job.config.model_dump(mode="json"),
                artifacts=tuple(job.artifacts),
            )
            try:
                result = self.executor.execute(
                    context,
                    progress=lambda current, total, message, detail: self._record_progress(
                        job_id, stage, current, total, message, detail
                    ),
                    is_cancel_requested=lambda: self.store.load_job(
                        job_id
                    ).cancel_requested,
                )
                if self.store.load_job(job_id).cancel_requested:
                    raise StageCancelledError()
                self._finish_stage(job_id, stage, result)
            except StageCancelledError:
                return self._cancel_job(job_id, stage)
            except StageExecutionError as exc:
                return self._fail_job(job_id, stage, exc)
            except Exception as exc:
                safe = StageExecutionError(
                    f"unexpected stage error: {type(exc).__name__}",
                    code="unexpected_stage_error",
                    retryable=False,
                )
                return self._fail_job(job_id, stage, safe)

        def complete(current: JobRecord) -> JobRecord:
            current.status = (
                JobStatus.COMPLETED_WITH_WARNINGS
                if current.warnings
                else JobStatus.COMPLETED
            )
            current.current_stage = None
            current.error = None
            return current

        completed = self.store.update_job(job_id, complete)
        self.store.append_event(
            job_id,
            EventType.JOB_COMPLETED,
            message="pipeline job completed",
            payload={
                "status": completed.status.value,
                "warning_count": len(completed.warnings),
            },
        )
        return completed

    def run_stage(
        self,
        job_id: str,
        stage: PipelineStage,
        *,
        retry_failed: bool = False,
    ) -> JobRecord:
        """Run one ready stage and leave the job queued for its successor.

        This is the worker-facing entry point used while real stages are added
        incrementally. It also lets a future multi-worker deployment schedule
        stages independently without changing the persisted job contract.
        """

        job = self.store.load_job(job_id)
        if job.status in {JobStatus.COMPLETED, JobStatus.COMPLETED_WITH_WARNINGS}:
            return job
        if job.status is JobStatus.CANCELLED:
            return job
        if job.status is JobStatus.FAILED:
            failed = next(
                (record for record in job.stages if record.status is StageStatus.FAILED),
                None,
            )
            if failed is None or failed.stage is not stage or not retry_failed:
                return job
            job = self._resume_failed_job(job_id)
        stage_record = job.stage_record(stage)
        if stage_record.status in {
            StageStatus.COMPLETED,
            StageStatus.COMPLETED_WITH_WARNINGS,
        }:
            return job
        stage_index = PIPELINE_STAGES.index(stage)
        incomplete_predecessors = [
            record.stage.value
            for record in job.stages[:stage_index]
            if record.status
            not in {StageStatus.COMPLETED, StageStatus.COMPLETED_WITH_WARNINGS}
        ]
        if incomplete_predecessors:
            raise PipelineStorageError(
                f"stage {stage.value} is not ready; incomplete predecessors: "
                + ", ".join(incomplete_predecessors)
            )
        if job.cancel_requested:
            return self._cancel_job(job_id, job.current_stage)

        job = self._start_stage(job_id, stage)
        context = StageContext(
            job_id=job_id,
            job_dir=self.store.job_dir(job_id),
            stage=stage,
            attempt=job.stage_record(stage).attempt,
            source_pdf=self.store.job_dir(job_id) / job.source_pdf,
            document_name=job.display_name,
            config=job.config.model_dump(mode="json"),
            artifacts=tuple(job.artifacts),
        )
        try:
            result = self.executor.execute(
                context,
                progress=lambda current, total, message, detail: self._record_progress(
                    job_id, stage, current, total, message, detail
                ),
                is_cancel_requested=lambda: self.store.load_job(job_id).cancel_requested,
            )
            if self.store.load_job(job_id).cancel_requested:
                raise StageCancelledError()
            finished = self._finish_stage(job_id, stage, result)
        except StageCancelledError:
            return self._cancel_job(job_id, stage)
        except StageExecutionError as exc:
            return self._fail_job(job_id, stage, exc)
        except Exception as exc:
            safe = StageExecutionError(
                f"unexpected stage error: {type(exc).__name__}",
                code="unexpected_stage_error",
                retryable=False,
            )
            return self._fail_job(job_id, stage, safe)

        if stage is PIPELINE_STAGES[-1]:
            def complete(current: JobRecord) -> JobRecord:
                current.status = (
                    JobStatus.COMPLETED_WITH_WARNINGS
                    if current.warnings
                    else JobStatus.COMPLETED
                )
                current.current_stage = None
                current.error = None
                return current

            completed = self.store.update_job(job_id, complete)
            self.store.append_event(
                job_id,
                EventType.JOB_COMPLETED,
                message="pipeline job completed",
                payload={
                    "status": completed.status.value,
                    "warning_count": len(completed.warnings),
                },
            )
            return completed

        def queue_next(current: JobRecord) -> JobRecord:
            current.status = JobStatus.QUEUED
            current.current_stage = None
            return current

        queued = self.store.update_job(job_id, queue_next)
        self.store.append_event(
            job_id,
            EventType.JOB_QUEUED,
            message="pipeline job queued for next stage",
            payload={"next_stage": PIPELINE_STAGES[stage_index + 1].value},
        )
        return queued
