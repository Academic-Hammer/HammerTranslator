"""Cache-aware, resumable execution of a translation plan."""

from __future__ import annotations

import random
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .cache import TranslationCache
from .checkpoint import CheckpointStore
from .protection import (
    ProtectedToken,
    ProtectionError,
    ProtectionResult,
    mask_protected_tokens,
)
from .provider import (
    ProviderError,
    ProviderResponse,
    TranslationProvider,
    TranslationRequest,
)
from .quality import QualityReport, evaluate_translation, strip_prompt_wrapper
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
    prompt_version: str = "translation-prompt-v2"
    glossary_version: str = "none"
    max_attempts: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    retry_failed: bool = False
    segment_fallback: bool = True

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


def _request(
    unit: TranslationUnit,
    provider: TranslationProvider,
    config: RunnerConfig,
    *,
    mode: str = "initial",
    constraints: dict[str, Any] | None = None,
    protected_text: str | None = None,
) -> TranslationRequest:
    return TranslationRequest(
        translation_unit_id=unit.translation_unit_id,
        protected_text=unit.protected_text if protected_text is None else protected_text,
        source_language=config.source_language,
        target_language=config.target_language,
        element_type=unit.element_type.value,
        provider=provider.name,
        model=provider.model,
        prompt_version=(
            f"{config.prompt_version}-table-v2"
            if unit.element_type.value == "table"
            else config.prompt_version
        ),
        glossary_version=config.glossary_version,
        mode=mode,
        constraints=constraints or {},
    )


def _normalized_response(response: ProviderResponse) -> ProviderResponse:
    text = strip_prompt_wrapper(response.text)
    if text == response.text:
        return response
    return ProviderResponse(
        text=text,
        provider=response.provider,
        model=response.model,
        request_id=response.request_id,
        metadata=response.metadata,
    )


def _restore_with_cleanup(
    unit: TranslationUnit,
    response: ProviderResponse,
) -> tuple[str, ProviderResponse]:
    """Restore protected spans, dropping unknown or duplicated placeholders."""

    try:
        return unit.restore_protected_content(response.text, order_policy="typed"), response
    except ProtectionError:
        cleaned = unit.clean_protected_candidate(response.text)
        if cleaned == response.text:
            raise
        restored = unit.restore_protected_content(cleaned, order_policy="typed")
        return restored, ProviderResponse(
            text=cleaned,
            provider=response.provider,
            model=response.model,
            request_id=response.request_id,
            metadata=response.metadata,
        )


def _retry_delay(error: ProviderError, attempt: int, config: RunnerConfig) -> float:
    if error.retry_after is not None:
        return max(0.0, min(error.retry_after, config.max_delay_seconds))
    exponential = min(
        config.base_delay_seconds * (2 ** max(0, attempt - 1)),
        config.max_delay_seconds,
    )
    return min(exponential + random.uniform(0, exponential * 0.1), config.max_delay_seconds)


def _repair_request(
    unit: TranslationUnit,
    provider: TranslationProvider,
    config: RunnerConfig,
    error: ProtectionError,
    previous_candidate: str,
) -> TranslationRequest:
    return _request(
        unit,
        provider,
        config,
        mode="repair",
        constraints={
            "validation_error": mask_protected_tokens("; ".join(error.issues)),
            "previous_candidate": mask_protected_tokens(previous_candidate)[:12000],
        },
    )


def _segment_request(
    unit: TranslationUnit,
    provider: TranslationProvider,
    config: RunnerConfig,
    *,
    text: str,
    index: int,
    count: int,
    context_before: str,
    context_after: str,
    use_context: bool = True,
) -> TranslationRequest:
    return _request(
        unit,
        provider,
        config,
        mode="segment",
        protected_text=text,
        constraints={
            "segment_index": index,
            "segment_count": count,
            "context_before": mask_protected_tokens(context_before)[:2000] if use_context else "",
            "context_after": mask_protected_tokens(context_after)[:2000] if use_context else "",
        },
    )


def _split_protected_segments(
    unit: TranslationUnit,
) -> list[tuple[str, tuple[ProtectedToken, ...]]]:
    """Split a failed unit into sentence/clause fragments for exact fallback."""

    text = unit.protected_text
    tokens = unit.protected_tokens
    if not text.strip() or not tokens:
        return []
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    segments: list[str] = []
    for sentence in sentences:
        token_count = sum(1 for token in tokens if token.placeholder in sentence)
        if token_count > 3:
            clauses = [
                part.strip()
                for part in re.split(r"(?<=[,;:])\s+", sentence)
                if part.strip()
            ]
            segments.extend(clauses or [sentence])
        else:
            segments.append(sentence)
    result = [
        (segment, tuple(token for token in tokens if token.placeholder in segment))
        for segment in segments
    ]
    expanded: list[tuple[str, tuple[ProtectedToken, ...]]] = []
    for segment, segment_tokens in result:
        expanded.extend(_split_segment_by_tokens(segment, segment_tokens))
    result = expanded
    if len(result) < 2:
        return []
    covered = sorted(token.placeholder for _, segment_tokens in result for token in segment_tokens)
    expected = sorted(token.placeholder for token in tokens)
    return result if covered == expected else []


def _split_segment_by_tokens(
    text: str,
    tokens: tuple[ProtectedToken, ...],
) -> list[tuple[str, tuple[ProtectedToken, ...]]]:
    """Split one fragment until it contains at most one protected token.

    With a single token per request, a provider cannot move that token across
    another token. Concatenating the translated fragments in source order then
    preserves the global protected-token order without discarding any formula.
    """

    if len(tokens) <= 1:
        return [(text, tokens)]
    ordered = sorted(tokens, key=lambda token: text.find(token.placeholder))
    boundary = ordered[len(ordered) // 2]
    index = text.find(boundary.placeholder)
    if index <= 0:
        return [(text, tokens)]
    left_text = text[:index].strip()
    right_text = text[index:].strip()
    left_tokens = tuple(token for token in ordered if text.find(token.placeholder) < index)
    right_tokens = tuple(token for token in ordered if text.find(token.placeholder) >= index)
    result: list[tuple[str, tuple[ProtectedToken, ...]]] = []
    if left_text:
        result.extend(_split_segment_by_tokens(left_text, left_tokens))
    if right_text:
        result.extend(_split_segment_by_tokens(right_text, right_tokens))
    return result


def _is_order_error(error: ProtectionError) -> bool:
    return any("order" in issue for issue in error.issues)


def _translate_segmented(
    unit: TranslationUnit,
    *,
    provider: TranslationProvider,
    cache: TranslationCache,
    config: RunnerConfig,
    stats: RunnerStats,
) -> tuple[str, str, QualityReport]:
    """Translate text fragments separately and reinsert protected spans in order."""

    segments = _split_protected_segments(unit)
    if len(segments) < 2:
        raise ProviderError(
            "protected token order changed and the unit cannot be split safely",
            code="protected_content_invalid",
            retryable=False,
        )
    candidate_parts: list[str] = []
    for index, (segment_text, segment_tokens) in enumerate(segments, start=1):
        segment_result = ProtectionResult(
            original_text=segment_text,
            protected_text=segment_text,
            tokens=segment_tokens,
        )
        for segment_attempt in range(2):
            request = _segment_request(
                unit,
                provider,
                config,
                text=segment_text,
                index=index,
                count=len(segments),
                context_before=segments[index - 2][0] if index > 1 else "",
                context_after=segments[index][0] if index < len(segments) else "",
                use_context=segment_attempt == 0,
            )
            cached = cache.get(request.cache_key())
            if cached is not None:
                response = ProviderResponse(
                    text=cached.response_text,
                    provider=cached.provider,
                    model=cached.model,
                    request_id=cached.metadata.get("request_id"),
                    metadata=cached.metadata,
                )
            else:
                stats.provider_calls += 1
                response = provider.translate(request)
            response = _normalized_response(response)
            if segment_text.strip() and not response.text.strip():
                raise ProviderError(
                    "provider returned an empty segment translation",
                    code="empty_segment_translation",
                    retryable=True,
                )
            cleaned = segment_result.clean_candidate(response.text)
            try:
                segment_result.restore(cleaned, order_policy="typed")
            except ProtectionError:
                cache.delete(request.cache_key())
                if segment_attempt == 0:
                    continue
                raise
            if cleaned != response.text:
                response = ProviderResponse(
                    text=cleaned,
                    provider=response.provider,
                    model=response.model,
                    request_id=response.request_id,
                    metadata=response.metadata,
                )
            cache.put(request, response)
            candidate_parts.append(cleaned.strip())
            break
    candidate = " ".join(candidate_parts)
    restored = unit.restore_protected_content(candidate, order_policy="typed")
    quality = evaluate_translation(
        unit.protected_text,
        candidate,
        target_language=config.target_language,
        simulation=provider.name == "fake",
    )
    if quality.should_retry:
        raise ProviderError(
            "; ".join(flag.message for flag in quality.flags),
            code="segment_translation_quality_rejected",
            retryable=True,
        )
    return restored, candidate, quality


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

        initial_request = _request(unit, provider, settings)
        cache_key = initial_request.cache_key()
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
                cached_text = strip_prompt_wrapper(cached.response_text)
                outputs[unit_id] = unit.restore_protected_content(
                    cached_text, order_policy="typed"
                )
            except ProtectionError:
                cache.delete(cache_key)
            else:
                quality = evaluate_translation(
                    unit.protected_text,
                    cached_text,
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
        current_request = initial_request
        last_protection_error: ProtectionError | None = None
        last_candidate = ""
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
            fallback_error = last_protection_error
            last_protection_error = None
            if (
                attempt == settings.max_attempts
                and settings.segment_fallback
                and fallback_error is not None
                and _is_order_error(fallback_error)
            ):
                try:
                    restored, candidate, quality = _translate_segmented(
                        unit,
                        provider=provider,
                        cache=cache,
                        config=settings,
                        stats=stats,
                    )
                except ProtectionError as exc:
                    final_quality = QualityReport.protected_content_rejected(str(exc))
                    final_error = ProviderError(
                        str(exc),
                        code="protected_content_invalid",
                        retryable=False,
                    )
                    break
                except ProviderError as exc:
                    final_error = exc
                    break
                except Exception as exc:
                    final_error = ProviderError(
                        f"unexpected segmented fallback error: {type(exc).__name__}",
                        code="unexpected_segment_fallback_error",
                        retryable=False,
                    )
                    break
                response = ProviderResponse(
                    text=candidate,
                    provider=provider.name,
                    model=provider.model,
                    metadata={"finish_reason": "segmented_fallback"},
                )
                entry = cache.put(initial_request, response)
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
                    request_id=None,
                    last_error=None,
                    quality=quality.as_dict(),
                )
                if quality.needs_review:
                    stats.qa_needs_review += 1
                else:
                    stats.completed += 1
                final_error = None
                break
            try:
                response: ProviderResponse | None = None
                cached_attempt = cache.get(current_request.cache_key())
                if cached_attempt is not None:
                    response = ProviderResponse(
                        text=cached_attempt.response_text,
                        provider=cached_attempt.provider,
                        model=cached_attempt.model,
                        request_id=cached_attempt.metadata.get("request_id"),
                        metadata=cached_attempt.metadata,
                    )
                else:
                    stats.provider_calls += 1
                    response = provider.translate(current_request)
                response = _normalized_response(response)
                restored, response = _restore_with_cleanup(unit, response)
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
                entry = cache.put(initial_request, response)
                if current_request.cache_key() != cache_key:
                    cache.put(current_request, response)
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
                last_protection_error = exc
                last_candidate = response.text if response is not None else ""
                final_quality = QualityReport.protected_content_rejected(str(exc))
                final_error = ProviderError(
                    str(exc),
                    code="protected_content_invalid",
                    retryable=True,
                )
                if attempt < settings.max_attempts:
                    current_request = _repair_request(
                        unit,
                        provider,
                        settings,
                        exc,
                        last_candidate,
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
