"""Text extraction from MinerU line/span structures."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

TEXT_SPAN_TYPES = {"text"}
INLINE_EQUATION_TYPES = {"inline_equation"}
DISPLAY_EQUATION_TYPES = {"interline_equation", "equation"}


def _half_width_ascii(text: str) -> str:
    result: list[str] = []
    for char in text:
        code = ord(char)
        if 0xFF10 <= code <= 0xFF19 or 0xFF21 <= code <= 0xFF3A or 0xFF41 <= code <= 0xFF5A:
            result.append(chr(code - 0xFEE0))
        else:
            result.append(char)
    return "".join(result)


def _span_text(span: dict[str, Any]) -> str:
    span_type = str(span.get("type", "text"))
    content = _half_width_ascii(str(span.get("content", ""))).strip()
    if not content:
        return ""
    if span_type in INLINE_EQUATION_TYPES:
        return f"${content}$"
    if span_type in DISPLAY_EQUATION_TYPES:
        return f"$$\n{re.sub(r'\s+', ' ', content)}\n$$"
    if span_type in TEXT_SPAN_TYPES:
        return content
    return ""


def _join_fragments(fragments: Iterable[str]) -> str:
    output = ""
    for fragment in fragments:
        fragment = fragment.strip()
        if not fragment:
            continue
        if output.endswith("-") and re.search(r"[A-Za-z]-$", output):
            output = output[:-1] + fragment
        elif not output or output.endswith(("\n", " ")):
            output += fragment
        elif _needs_space(output[-1], fragment[0]):
            output += " " + fragment
        else:
            output += fragment
    return output.strip()


def _needs_space(left: str, right: str) -> bool:
    cjk = lambda char: "\u2e80" <= char <= "\u9fff"
    return not cjk(left) and not cjk(right) and right not in ",.;:!?)]}"


def extract_text(block: dict[str, Any]) -> str:
    """Extract readable text without mutating the MinerU object."""
    lines: list[dict[str, Any]] = []
    if isinstance(block.get("lines"), list):
        lines.extend(line for line in block["lines"] if isinstance(line, dict))
    elif isinstance(block.get("blocks"), list):
        for child in block["blocks"]:
            if isinstance(child, dict) and isinstance(child.get("lines"), list):
                lines.extend(line for line in child["lines"] if isinstance(line, dict))

    line_texts: list[str] = []
    for line in lines:
        spans = line.get("spans", [])
        if not isinstance(spans, list):
            continue
        text = _join_fragments(_span_text(span) for span in spans if isinstance(span, dict))
        if text:
            line_texts.append(text)
    return _join_fragments(line_texts)


def first_alpha(text: str) -> str:
    return next((char for char in text.strip() if char.isalpha()), "")


def sentence_is_complete(text: str) -> bool:
    stripped = text.rstrip()
    return not stripped or stripped[-1] in ".!?。！？:："
