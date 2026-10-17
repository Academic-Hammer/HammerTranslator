"""Conservative author/affiliation line splitting."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .protection import ProtectedSpan


EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
ORCID_RE = re.compile(r"(?:https?://orcid\.org/)?\d{4}-\d{4}-\d{4}-\d{3}[\dX]", re.IGNORECASE)
AFFILIATION_RE = re.compile(
    r"\b(?:university|institute|institution|laborator(?:y|ies)|\blab\b|department|"
    r"school|college|academy|centre|center|hospital|faculty|research|technology|"
    r"corporation|company|inc\.?|ltd\.?)\b",
    re.IGNORECASE,
)
NAME_RE = re.compile(r"^[A-Za-zÀ-ÖØ-öø-ÿ'’.-]+(?:[ ,]+[A-Za-zÀ-ÖØ-öø-ÿ'’.-]+){0,9}$")


@dataclass(frozen=True)
class AuthorSegment:
    text: str
    start: int
    end: int
    role: str


@dataclass(frozen=True)
class AuthorSplit:
    segments: tuple[AuthorSegment, ...]
    protected_spans: tuple[ProtectedSpan, ...]
    ambiguous: bool
    reason: str
    has_affiliation: bool


def _identity_role(text: str) -> str | None:
    if EMAIL_RE.search(text):
        return "email"
    if ORCID_RE.search(text):
        return "orcid"
    if URL_RE.search(text):
        return "url"
    return None


def split_author_block(text: str) -> AuthorSplit:
    """Split names/identity fields from affiliation lines without guessing."""

    lines = [match for match in re.finditer(r"[^\r\n]+", text) if match.group(0).strip()]
    if not lines:
        return AuthorSplit((), (), True, "empty author block", False)

    affiliation_start: int | None = None
    for index, match in enumerate(lines):
        value = match.group(0).strip()
        if _identity_role(value) is None and AFFILIATION_RE.search(value):
            affiliation_start = index
            break

    segments: list[AuthorSegment] = []
    ambiguous = False
    reason = "author identity and affiliation lines separated"
    for index, match in enumerate(lines):
        raw = match.group(0)
        stripped = raw.strip()
        offset = match.start() + (len(raw) - len(raw.lstrip()))
        end = offset + len(stripped)
        identity_role = _identity_role(stripped)
        if identity_role:
            role = identity_role
        elif affiliation_start is not None and index >= affiliation_start:
            role = "affiliation"
        elif NAME_RE.fullmatch(stripped):
            role = "name"
        else:
            role = "ambiguous"
            ambiguous = True
        segments.append(AuthorSegment(stripped, offset, end, role))

    has_affiliation = any(segment.role == "affiliation" for segment in segments)
    if affiliation_start is None:
        non_identity = [segment for segment in segments if segment.role not in {"email", "url", "orcid"}]
        if non_identity and all(segment.role == "name" for segment in non_identity):
            reason = "no affiliation detected; all author content will be preserved"
        else:
            ambiguous = True
            reason = "affiliation boundary could not be identified reliably"
    elif ambiguous:
        reason = "content before the affiliation boundary is not reliably identifiable as a name"

    if ambiguous:
        protected = (ProtectedSpan(0, len(text), "author_ambiguous"),) if text else ()
    else:
        protected = tuple(
            ProtectedSpan(segment.start, segment.end, f"author_{segment.role}")
            for segment in segments
            if segment.role != "affiliation"
        )
    return AuthorSplit(tuple(segments), protected, ambiguous, reason, has_affiliation)

