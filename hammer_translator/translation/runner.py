"""Cache-aware, resumable execution of a translation plan."""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .cache import TranslationCache
from .checkpoint import CheckpointStore
from .protection import ProtectionError
from .provider import ProviderError, TranslationProvider, TranslationRequest
from .quality import QualityReport, evaluate_translation
from .taxonomy import TranslationPolicy
from .units import TranslationPlan, TranslationUnit


ProgressCallback = Callable[[dict[str, Any]], None]


def _emit(progress: ProgressCallback | None, **event: Any) -> None:
    if progress is not None:
        progress(event)


@dataclass(frozen=True)
class RunnerConfig:
    source_language: str = "en"
    target_language: str = "zh-CN"
    prompt_version: str = "translation-prompt-v1"
    glossary_version: str = "none"
    max_attempts: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    retry_failed: bool = False

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.base_delay_seconds < 0 or self.max_delay_seconds < 0:
            raise ValueError("retry delays cannot be negative")


@dataclass
class RunnerStats:
    units_total: int = 0
    provider_calls: int = 0
    cache_hits: int = 0
    resumed_from_checkpoint: int = 0
    retries: int = 0
    completed: int = 0
    preserved: int = 0
    skipped: int = 0
    needs_review: int = 0
    qa_needs_review: int = 0
    failed: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class RunnerResult:
    outputs: dict[str, str]
    quality_reports: dict[str, QualityReport]
    unit_states: dict[str, dict]
    stats: RunnerStats
    checkpoint_path: Path
    started_at: str
    completed_at: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_signature(provider: TranslationProvider, config: RunnerConfig) -> dict[str, str]:
    return {
        "provider": provider.name,
        "model": provider.model,
        "source_language": config.source_language,
        "target_language": config.target_language,
        "prompt_version": config.prompt_version,
        "glossary_version": config.glossary_version,
    }


def _request(unit: TranslationUnit, provider: TranslationProvider, config: RunnerConfig) -> TranslationRequest:
    return TranslationRequest(
        translation_unit_id=unit.translation_unit_id,
        protected_text=unit.protected_text,
        source_language=config.source_language,
        target_language=config.target_language,
        element_type=unit.element_type.value,
        provider=provider.name,
        model=provider.model,
        prompt_version=config.prompt_version,
        glossary_version=config.glossary_version,
        mode="initial",
    )


def _retry_delay(error: ProviderError, attempt: int, config: RunnerConfig) -> float:
    if error.retry_after is not None:
        return max(0.0, min(error.retry_after, config.max_delay_seconds))
    exponential = min(
        config.base_delay_seconds * (2 ** max(0, attempt - 1)),
        config.max_delay_seconds,
    )
    return min(exponential + random.uniform(0, exponential * 0.1), config.max_delay_seconds)


def execute_translation_plan(
    plan: TranslationPlan,
    *,
    provider: TranslationProvider,
    cache: TranslationCache,
    checkpoint_path: str | Path,
    config: RunnerConfig | None = None,
    sleep: Callable[[float], None] = time.sleep,
    progress: ProgressCallback | None = None,
) -> RunnerResult:
    settings = config or RunnerConfig()
    started_at = _utc_now()
    checkpoint = CheckpointStore(checkpoint_path)
    checkpoint.initialize(
        document_id=plan.document.document_id,
        source_hash=plan.document.source_hash,
        run_signature=_run_signature(provider, settings),
        unit_ids=(unit.translation_unit_id for unit in plan.units),
    )
    stats = RunnerStats(units_total=len(plan.units))
    outputs: dict[str, str] = {}
    quality_reports: dict[str, QualityReport] = {}

    for unit_index, unit in enumerate(plan.units, start=1):
        unit_id = unit.translation_unit_id
        common_event = {
            "unit_index": unit_index,
            "unit_total": len(plan.units),
            "translation_unit_id": unit_id,
            "element_type": unit.element_type.value,
        }
        _emit(progress, event="unit_started", **common_event)
        if unit.policy is TranslationPolicy.PRESERVE:
            outputs[unit_id] = unit.source_text
            quality_reports[unit_id] = QualityReport.not_applicable()
            checkpoint.update_unit(
                unit_id,
                execution_status="preserved",
                last_error=None,
                quality=quality_reports[unit_id].as_dict(),
            )
            stats.preserved += 1
            _emit(progress, event="unit_finished", status="preserved", **common_event)
            continue
        if unit.policy is TranslationPolicy.SKIP:
            outputs[unit_id] = unit.source_text
            quality_reports[unit_id] = QualityReport.not_applicable()
            checkpoint.update_unit(
                unit_id,
                execution_status="skipped",
                last_error=None,
                quality=quality_reports[unit_id].as_dict(),
            )
            stats.skipped += 1
            _emit(progress, event="unit_finished", status="skipped", **common_event)
            continue
        if unit.policy is TranslationPolicy.NEEDS_REVIEW:
            outputs[unit_id] = unit.source_text
            quality_reports[unit_id] = QualityReport.not_applicable()
            checkpoint.update_unit(
                unit_id,
                execution_status="needs_review",
                last_error=None,
                quality=quality_reports[unit_id].as_dict(),
            )
            stats.needs_review += 1
            _emit(progress, event="unit_finished", status="needs_review", **common_event)
            continue

        request = _request(unit, provider, settings)
        cache_key = request.cache_key()
        state = checkpoint.unit_state(unit_id)
        if state.get("execution_status") == "failed" and not settings.retry_failed:
            stats.failed += 1
            quality = state.get("quality")
            if isinstance(quality, dict):
                quality_reports[unit_id] = QualityReport.from_dict(quality)
            _emit(
                progress,
                event="unit_finished",
                status="failed_from_checkpoint",
                **common_event,
            )
            continue

        cached = cache.get(cache_key)
        if cached is not None:
            try:
                outputs[unit_id] = unit.restore_protected_content(cached.response_text)
            except ProtectionError:
                cache.delete(cache_key)
            else:
                quality = evaluate_translation(
                    unit.protected_text,
                    cached.response_text,
                    target_language=settings.target_language,
                    simulation=provider.name == "fake",
                )
                if quality.should_retry:
                    del outputs[unit_id]
                    cache.delete(cache_key)
                else:
                    quality_reports[unit_id] = quality
                    execution_status = "needs_review" if quality.needs_review else "completed"
                    checkpoint.update_unit(
                        unit_id,
                        execution_status=execution_status,
                        cache_key=cache_key,
                        cache_hit=True,
                        response_hash=cached.response_hash,
                        request_id=cached.metadata.get("request_id"),
                        last_error=None,
                        quality=quality.as_dict(),
                    )
                    stats.cache_hits += 1
                    if quality.needs_review:
                        stats.qa_needs_review += 1
                    else:
                        stats.completed += 1
                    if state.get("execution_status") in {"completed", "needs_review"}:
                        stats.resumed_from_checkpoint += 1
                    _emit(
                        progress,
                        event="unit_finished",
                        status=execution_status,
                        cache_hit=True,
                        **common_event,
                    )
                    continue

        previous_attempts = int(state.get("attempts") or 0)
        checkpoint.update_unit(
            unit_id,
            execution_status="in_progress",
            cache_key=cache_key,
            cache_hit=False,
            last_error=None,
        )
        final_error: ProviderError | None = None
        final_quality: QualityReport | None = None
        attempts_made = 0
        for attempt in range(1, settings.max_attempts + 1):
            attempts_made = attempt
            final_quality = None
            checkpoint.update_unit(
                unit_id,
                execution_status="in_progress",
                attempts=previous_attempts + attempt,
            )
            _emit(
                progress,
                event="provider_request",
                attempt=attempt,
                max_attempts=settings.max_attempts,
                **common_event,
            )
            try:
                stats.provider_calls += 1
                response = provider.translate(request)
                restored = unit.restore_protected_content(response.text)
                quality = evaluate_translation(
                    unit.protected_text,
                    response.text,
                    target_language=settings.target_language,
                    simulation=provider.name == "fake",
                )
                final_quality = quality
                if quality.should_retry:
                    raise ProviderError(
                        "; ".join(flag.message for flag in quality.flags),
                        code="translation_quality_rejected",
                        retryable=True,
                    )
                entry = cache.put(request, response)
                outputs[unit_id] = restored
                quality_reports[unit_id] = quality
                execution_status = "needs_review" if quality.needs_review else "completed"
                checkpoint.update_unit(
                    unit_id,
                    execution_status=execution_status,
                    attempts=previous_attempts + attempt,
                    cache_key=cache_key,
                    cache_hit=False,
                    response_hash=entry.response_hash,
                    request_id=response.request_id,
                    last_error=None,
                    quality=quality.as_dict(),
                )
                if quality.needs_review:
                    stats.qa_needs_review += 1
                else:
                    stats.completed += 1
                final_error = None
                break
            except ProtectionError as exc:
                final_quality = QualityReport.protected_content_rejected(str(exc))
                final_error = ProviderError(
                    str(exc),
                    code="protected_content_invalid",
                    retryable=True,
                )
            except ProviderError as exc:
                final_error = exc
            except Exception as exc:
                final_error = ProviderError(
                    f"unexpected provider error: {type(exc).__name__}",
                    code="unexpected_provider_error",
                    retryable=False,
                )

            checkpoint.update_unit(
                unit_id,
                execution_status="in_progress",
                attempts=previous_attempts + attempt,
                last_error=final_error.safe_details(),
                quality=final_quality.as_dict() if final_quality else None,
            )
            if not final_error.retryable or attempt >= settings.max_attempts:
                break
            stats.retries += 1
            delay = _retry_delay(final_error, attempt, settings)
            _emit(
                progress,
                event="provider_retry",
                attempt=attempt,
                error_code=final_error.code,
                delay_seconds=round(delay, 2),
                **common_event,
            )
            sleep(delay)

        if final_error is not None:
            checkpoint.update_unit(
                unit_id,
                execution_status="failed",
                attempts=previous_attempts + attempts_made,
                last_error=final_error.safe_details(),
                quality=final_quality.as_dict() if final_quality else None,
            )
            if final_quality is not None:
                quality_reports[unit_id] = final_quality
            stats.failed += 1
            _emit(
                progress,
                event="unit_finished",
                status="failed",
                error_code=final_error.code,
                **common_event,
            )
        else:
            _emit(
                progress,
                event="unit_finished",
                status=(
                    "needs_review"
                    if quality_reports[unit_id].needs_review
                    else "completed"
                ),
                cache_hit=False,
                **common_event,
            )

    checkpoint.finalize()
    snapshot = checkpoint.snapshot()
    return RunnerResult(
        outputs=outputs,
        quality_reports=quality_reports,
        unit_states=snapshot["units"],
        stats=stats,
        checkpoint_path=checkpoint.path,
        started_at=started_at,
        completed_at=_utc_now(),
    )
