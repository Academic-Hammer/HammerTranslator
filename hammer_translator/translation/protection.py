"""Lossless protection of content that translation must not modify."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Literal


OrderPolicy = Literal["strict", "typed"]

# Formulas, code, markup, and table values are positional: moving them can
# change the meaning or corrupt the layout. Citations, URLs, and similar
# anchors may legitimately move when the target-language word order changes.
ORDER_SENSITIVE_KINDS = frozenset(
    {
        "code_fence",
        "latex_environment",
        "latex_display",
        "latex_bracket",
        "latex_parenthesis",
        "inline_code",
        "latex_inline",
        "html_tag",
        "html_entity",
        "stat_expression",
        "number",
        "stat_symbol",
    }
)


@dataclass(frozen=True)
class ProtectedSpan:
    start: int
    end: int
    kind: str


@dataclass(frozen=True)
class ProtectedToken:
    placeholder: str
    original: str
    kind: str
    start: int
    end: int


class ProtectionError(ValueError):
    def __init__(self, issues: Iterable[str]):
        self.issues = tuple(issues)
        super().__init__("; ".join(self.issues))


PROTECTED_TOKEN_RE = re.compile(r"\[\[HTP_[A-F0-9]+_\d{4}\]\]")


def mask_protected_tokens(text: str, *, replacement: str = "…") -> str:
    """Replace protected placeholders so they cannot leak through context."""

    masked = PROTECTED_TOKEN_RE.sub(replacement, text)
    return re.sub(r"[ \t]{2,}", " ", masked).strip()


def clean_protected_candidate(candidate: str, expected: Iterable[str]) -> str:
    """Remove unknown and duplicate placeholders while preserving expected ones."""

    expected_counts = Counter(expected)
    observed_counts: Counter[str] = Counter()

    def replace(match: re.Match[str]) -> str:
        placeholder = match.group(0)
        if placeholder not in expected_counts:
            return " "
        observed_counts[placeholder] += 1
        return placeholder if observed_counts[placeholder] <= expected_counts[placeholder] else " "

    cleaned = PROTECTED_TOKEN_RE.sub(replace, candidate)
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip()


@dataclass(frozen=True)
class ProtectionResult:
    original_text: str
    protected_text: str
    tokens: tuple[ProtectedToken, ...]

    def clean_candidate(self, candidate: str) -> str:
        """Return a candidate with unknown or duplicated placeholders removed."""

        return clean_protected_candidate(
            candidate,
            (token.placeholder for token in self.tokens),
        )

    def restore(self, candidate: str, *, order_policy: OrderPolicy = "strict") -> str:
        """Validate token count/order, then restore exact original strings.

        ``strict`` keeps the historical behavior and requires every protected
        token to remain in source order. ``typed`` enforces order only for
        position-sensitive kinds (formulas, code, markup, and table values),
        allowing citations and URLs to follow natural target-language order.
        """

        if order_policy not in {"strict", "typed"}:
            raise ValueError(f"unsupported order policy: {order_policy}")

        expected = [token.placeholder for token in self.tokens]
        issues: list[str] = []
        for placeholder, count in Counter(expected).items():
            actual = candidate.count(placeholder)
            if actual == 0:
                issues.append(f"missing protected token {placeholder}")
            elif actual > count:
                issues.append(f"duplicated protected token {placeholder}")
        order_expected = (
            expected
            if order_policy == "strict"
            else [
                token.placeholder
                for token in self.tokens
                if token.kind in ORDER_SENSITIVE_KINDS
            ]
        )
        positions = [candidate.find(placeholder) for placeholder in order_expected]
        if all(position >= 0 for position in positions) and positions != sorted(positions):
            issues.append("protected token order changed")

        expected_set = set(expected)
        observed = set(PROTECTED_TOKEN_RE.findall(candidate))
        for placeholder in sorted(observed - expected_set):
            issues.append(f"unexpected protected token {placeholder}")
        if issues:
            raise ProtectionError(issues)

        restored = candidate
        for token in self.tokens:
            restored = restored.replace(token.placeholder, token.original)
        return restored


STANDARD_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("code_fence", re.compile(r"```[\s\S]*?```")),
    ("latex_environment", re.compile(r"\\begin\{([A-Za-z*]+)\}[\s\S]*?\\end\{\1\}")),
    ("latex_display", re.compile(r"\$\$[\s\S]*?\$\$")),
    ("latex_bracket", re.compile(r"\\\[[\s\S]*?\\\]")),
    ("latex_parenthesis", re.compile(r"\\\([\s\S]*?\\\)")),
    ("inline_code", re.compile(r"`[^`\n]+`")),
    ("latex_inline", re.compile(r"(?<!\$)\$(?!\$)(?:\\.|[^$\n])+\$(?!\$)")),
    (
        "latex_reference",
        re.compile(r"\\(?:cite|citep|citet|citealp|ref|eqref|autoref)(?:\[[^\]]*\])?\{[^{}]+\}"),
    ),
    ("url", re.compile(r"(?:https?://|www\.)[^\s<>\[\]\"']+", re.IGNORECASE)),
    ("doi", re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)),
    ("orcid", re.compile(r"\b(?:https?://orcid\.org/)?\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b", re.IGNORECASE)),
    ("numeric_citation", re.compile(r"\[(?:\d+[a-z]?(?:\s*[-–,]\s*\d+[a-z]?)*)(?:\s*,\s*\d+[a-z]?)*\]", re.IGNORECASE)),
    (
        "author_year_citation",
        re.compile(r"\([A-Z][A-Za-z'’.-]+(?:\s+et\s+al\.)?,\s*\d{4}[a-z]?(?:;\s*[^()]+,\s*\d{4}[a-z]?)*\)"),
    ),
)

TABLE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("html_tag", re.compile(r"<[^>]+>")),
    ("html_entity", re.compile(r"&(?:#\d+|#x[0-9A-Fa-f]+|[A-Za-z][A-Za-z0-9]+);")),
    (
        "stat_expression",
        re.compile(r"\b(?:p|r|R|F|t|z|χ2|χ²)\s*(?:=|<|>|≤|≥)\s*[+-]?(?:\d+(?:\.\d+)?|\.\d+)", re.IGNORECASE),
    ),
    ("number", re.compile(r"(?<![\w.])[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?%?(?:\*+)?(?![\w.])")),
    ("stat_symbol", re.compile(r"(?:\*{1,4}|±|≤|≥)")),
)


class ContentProtector:
    """Replace immutable spans with deterministic, auditable placeholders."""

    def protect(
        self,
        text: str,
        *,
        extra_spans: Iterable[ProtectedSpan] = (),
        table_mode: bool = False,
    ) -> ProtectionResult:
        candidates: list[ProtectedSpan] = list(extra_spans)
        patterns = STANDARD_PATTERNS + (TABLE_PATTERNS if table_mode else ())
        for kind, pattern in patterns:
            for match in pattern.finditer(text):
                start, end = match.span()
                if kind == "url":
                    while end > start and text[end - 1] in ".,;:!?":
                        end -= 1
                if end > start:
                    candidates.append(ProtectedSpan(start, end, kind))

        selected = self._select_non_overlapping(candidates, len(text))
        nonce = self._nonce(text)
        tokens: list[ProtectedToken] = []
        chunks: list[str] = []
        cursor = 0
        for index, span in enumerate(selected, start=1):
            placeholder = f"[[HTP_{nonce}_{index:04d}]]"
            chunks.append(text[cursor : span.start])
            chunks.append(placeholder)
            tokens.append(
                ProtectedToken(
                    placeholder=placeholder,
                    original=text[span.start : span.end],
                    kind=span.kind,
                    start=span.start,
                    end=span.end,
                )
            )
            cursor = span.end
        chunks.append(text[cursor:])
        return ProtectionResult(text, "".join(chunks), tuple(tokens))

    @staticmethod
    def _select_non_overlapping(spans: Iterable[ProtectedSpan], text_length: int) -> list[ProtectedSpan]:
        valid = [
            span
            for span in spans
            if 0 <= span.start < span.end <= text_length
        ]
        valid.sort(key=lambda span: (span.start, -(span.end - span.start), span.kind))
        selected: list[ProtectedSpan] = []
        cursor = -1
        for span in valid:
            if span.start < cursor:
                continue
            selected.append(span)
            cursor = span.end
        return selected

    @staticmethod
    def _nonce(text: str) -> str:
        salt = 0
        while True:
            value = hashlib.sha256(f"{salt}\0{text}".encode("utf-8")).hexdigest()[:10].upper()
            if f"[[HTP_{value}_" not in text:
                return value
            salt += 1

