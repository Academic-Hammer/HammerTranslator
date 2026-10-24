"""Translation-only quality checks, deliberately independent of layout checks."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from typing import Any, Literal


QualityDisposition = Literal["pass", "review", "reject", "simulation", "not_applicable"]
QualitySeverity = Literal["info", "warning", "error"]


@dataclass(frozen=True)
class QualityFlag:
    code: str
    severity: QualitySeverity
    message: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class QualityReport:
    disposition: QualityDisposition
    flags: tuple[QualityFlag, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def should_retry(self) -> bool:
        return self.disposition == "reject"

    @property
    def needs_review(self) -> bool:
        return self.disposition == "review"

    def as_dict(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition,
            "flags": [flag.as_dict() for flag in self.flags],
            "metrics": dict(self.metrics),
        }

    @classmethod
    def not_applicable(cls) -> "QualityReport":
        return cls(disposition="not_applicable")

    @classmethod
    def protected_content_rejected(cls, message: str) -> "QualityReport":
        return cls(
            disposition="reject",
            flags=(
                QualityFlag(
                    code="protected_content_invalid",
                    severity="error",
                    message=message,
                ),
            ),
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "QualityReport":
        allowed = {"pass", "review", "reject", "simulation", "not_applicable"}
        disposition = value.get("disposition")
        if disposition not in allowed:
            disposition = "reject"
        flags = tuple(
            QualityFlag(
                code=str(flag.get("code") or "invalid_quality_record"),
                severity=(
                    flag.get("severity")
                    if flag.get("severity") in {"info", "warning", "error"}
                    else "error"
                ),
                message=str(flag.get("message") or "invalid persisted quality record"),
            )
            for flag in value.get("flags", [])
            if isinstance(flag, dict)
        )
        metrics = value.get("metrics")
        return cls(
            disposition=disposition,
            flags=flags,
            metrics=dict(metrics) if isinstance(metrics, dict) else {},
        )


META_OUTPUT_PATTERNS = (
    re.compile(r"^\s*(?:here is|here's)\s+(?:the\s+)?translation\b", re.IGNORECASE),
    re.compile(r"^\s*(?:translation|answer|output|result)\s*[:\uff1a]", re.IGNORECASE),
    re.compile(r"^\s*(?:\u8bd1\u6587|\u7ffb\u8bd1\u7ed3\u679c|\u7b54\u6848|\u8f93\u51fa)\s*[:\uff1a]"),
    re.compile(r"\b(?:as an ai|language model|system prompt|user prompt)\b", re.IGNORECASE),
    re.compile(r"(?:\u8bf7\u63d0\u4f9b\u9700\u8981\u7ffb\u8bd1|\u4ec5\u8f93\u51fa\u7ffb\u8bd1\u5185\u5bb9|\u4e0a\u4e00\u6b21\u7684\u5c1d\u8bd5)"),
    re.compile(r"^\s*(?:element type|source language|target language)\s*[:\uff1a]", re.IGNORECASE),
    re.compile(r"^\s*\u5143\u7d20\u7c7b\u578b\s*[:\uff1a]"),
    re.compile(r"^\s*(?:\u6e90\u8bed\u8a00|\u76ee\u6807\u8bed\u8a00)\s*[:\uff1a]"),
    re.compile(r"^\s*(?:\u5728\u5b9a\u754c\u7b26\u4e4b\u95f4|\u7ffb\u8bd1\u5206\u9694\u7b26\u4e4b\u95f4)"),
    re.compile(r"</?\s*(?:SOURCE|\u6e90\u6587\u672c)\s*>", re.IGNORECASE),
)


WRAPPER_TAG_PAIRS = (
    (
        re.compile(r"<\s*SOURCE\s*>", re.IGNORECASE),
        re.compile(r"</\s*SOURCE\s*>", re.IGNORECASE),
    ),
    (re.compile(r"<\s*\u6e90\u6587\u672c\s*>"), re.compile(r"</\s*\u6e90\u6587\u672c\s*>")),
)
WRAPPER_MARKERS = (
    "element type",
    "source language",
    "target language",
    "translate the source",
    "\u5143\u7d20\u7c7b\u578b",
    "\u6e90\u8bed\u8a00",
    "\u76ee\u6807\u8bed\u8a00",
    "\u5b9a\u754c\u7b26\u4e4b\u95f4",
    "\u7ffb\u8bd1\u5206\u9694\u7b26",
)


def strip_prompt_wrapper(text: str) -> str:
    """Remove an echoed prompt envelope while preserving the translated body."""

    value = text or ""
    for opening_pattern, closing_pattern in WRAPPER_TAG_PAIRS:
        opening = opening_pattern.search(value)
        if opening is None:
            continue
        closing = closing_pattern.search(value, opening.end())
        if closing is None:
            continue
        prefix = value[: opening.start()].strip()
        suffix = value[closing.end() :].strip()

        def known_envelope(part: str) -> bool:
            if not part:
                return True
            lowered = part.casefold()
            return any(marker in lowered for marker in WRAPPER_MARKERS) or any(
                pattern.search(part) for pattern in META_OUTPUT_PATTERNS
            )

        if not (known_envelope(prefix) and known_envelope(suffix)):
            continue
        inner = value[opening.end() : closing.start()].strip()
        if inner:
            return inner
    return value


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _latin_word_count(text: str) -> int:
    return len(re.findall(r"\b[A-Za-z]{3,}\b", text or ""))


def _cjk_count(text: str) -> int:
    return len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", text or ""))


def _visible_character_count(text: str) -> int:
    without_placeholders = re.sub(r"\[\[HTP_[A-F0-9]+_\d{4}\]\]", "", text or "")
    return len(re.sub(r"\s+", "", without_placeholders))


EMAIL_OR_URL_RE = re.compile(
    r"^(?:(?:https?://|www\.)\S+|[^\s@]+@[^\s@]+\.[^\s@]+)$",
    re.IGNORECASE,
)
NUMERIC_LITERAL_RE = re.compile(
    r"^[\s\d.,:;+\-−–—/%=<>≤≥±×*()\[\]]+$"
)


def _is_nonlinguistic_literal(text: str) -> bool:
    """Return true for values that must stay byte-for-byte unchanged."""

    value = _normalized(text)
    if not value:
        return False
    if EMAIL_OR_URL_RE.fullmatch(value):
        return True
    return bool(
        NUMERIC_LITERAL_RE.fullmatch(value)
        and re.search(r"\d", value)
    )


def evaluate_translation(
    source_text: str,
    candidate_text: str,
    *,
    target_language: str,
    simulation: bool = False,
) -> QualityReport:
    """Evaluate semantic output only; bbox capacity and typography are out of scope."""

    source = _normalized(source_text)
    candidate = _normalized(candidate_text)
    flags: list[QualityFlag] = []

    if not candidate:
        return QualityReport(
            disposition="reject",
            flags=(
                QualityFlag(
                    code="empty_translation",
                    severity="error",
                    message="provider returned an empty translation",
                ),
            ),
            metrics={"source_latin_words": _latin_word_count(source)},
        )

    if _is_nonlinguistic_literal(source):
        metrics = {
            "source_latin_words": _latin_word_count(source),
            "candidate_latin_words": _latin_word_count(candidate),
            "candidate_cjk_chars": _cjk_count(candidate),
            "source_candidate_similarity": round(
                SequenceMatcher(None, source[:4000], candidate[:4000]).ratio(),
                4,
            ),
            "candidate_visible_chars": _visible_character_count(candidate),
            "nonlinguistic_literal": True,
        }
        if candidate != source:
            return QualityReport(
                disposition="review",
                flags=(
                    QualityFlag(
                        code="literal_value_changed",
                        severity="warning",
                        message="email, URL, or numeric value changed during translation",
                    ),
                ),
                metrics=metrics,
            )
        return QualityReport(disposition="pass", metrics=metrics)

    source_has_meta = any(pattern.search(source) for pattern in META_OUTPUT_PATTERNS)
    if not source_has_meta and any(pattern.search(candidate) for pattern in META_OUTPUT_PATTERNS):
        flags.append(
            QualityFlag(
                code="prompt_or_meta_leakage",
                severity="error",
                message="output contains likely prompt or response-wrapper text",
            )
        )
        return QualityReport(disposition="reject", flags=tuple(flags))

    source_latin = _latin_word_count(source)
    candidate_latin = _latin_word_count(candidate)
    candidate_cjk = _cjk_count(candidate)
    similarity = SequenceMatcher(None, source[:4000], candidate[:4000]).ratio()
    metrics: dict[str, Any] = {
        "source_latin_words": source_latin,
        "candidate_latin_words": candidate_latin,
        "candidate_cjk_chars": candidate_cjk,
        "source_candidate_similarity": round(similarity, 4),
        "candidate_visible_chars": _visible_character_count(candidate),
    }

    if simulation:
        return QualityReport(
            disposition="simulation",
            flags=(
                QualityFlag(
                    code="simulation_output_not_quality_scored",
                    severity="info",
                    message="FakeProvider output is structural test data, not a real translation",
                ),
            ),
            metrics=metrics,
        )

    if source and candidate == source:
        flags.append(
            QualityFlag(
                code="source_text_unchanged",
                severity="warning",
                message="candidate is identical to the source text",
            )
        )

    if target_language == "zh-CN" and source_latin >= 4:
        if candidate_cjk == 0:
            flags.append(
                QualityFlag(
                    code="target_language_missing",
                    severity="warning",
                    message="English source has no Chinese characters in the candidate",
                )
            )
        elif candidate_latin >= max(12, int(source_latin * 0.65)) and candidate_cjk < source_latin:
            flags.append(
                QualityFlag(
                    code="excessive_latin_residue",
                    severity="warning",
                    message="candidate may contain too much untranslated English prose",
                )
            )

    candidate_visible = metrics["candidate_visible_chars"]
    if source_latin >= 12 and candidate_visible < max(6, int(source_latin * 0.22)):
        flags.append(
            QualityFlag(
                code="possibly_truncated",
                severity="warning",
                message="candidate is unusually short relative to the source",
            )
        )

    disposition: QualityDisposition = "review" if flags else "pass"
    return QualityReport(disposition=disposition, flags=tuple(flags), metrics=metrics)
