"""Semantic classification and logical-paragraph grouping."""

from __future__ import annotations

import re
from collections import defaultdict
from copy import deepcopy
from typing import Any

from .text import first_alpha, sentence_is_complete

ABSTRACT_RE = re.compile(r"^(abstract|\u6458\u8981)\b", re.IGNORECASE)
REFERENCE_RE = re.compile(r"^(references?|bibliography|\u53c2\u8003\u6587\u732e)\s*:?\s*$", re.IGNORECASE)
URL_RE = re.compile(r"^(?:https?://|www\.)\S+$", re.IGNORECASE)


def classify_document(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = deepcopy(items)
    _classify_references(output)
    _classify_authors(output)
    for item in output:
        if item.get("type") == "text" and URL_RE.fullmatch(str(item.get("text", "")).strip()):
            item["type"] = "url"
    return output


def _classify_references(items: list[dict[str, Any]]) -> None:
    in_references = False
    for item in items:
        text = str(item.get("text", "")).strip()
        is_heading = "text_level" in item
        if item.get("type") == "text" and REFERENCE_RE.match(text):
            in_references = True
            item["type"] = "reference"
            continue
        if in_references and is_heading:
            in_references = False
        elif in_references and item.get("type") == "text":
            item["type"] = "reference"


def _classify_authors(items: list[dict[str, Any]]) -> None:
    """Conservatively label text before an explicit Abstract heading on page 1."""
    abstract_index = next(
        (
            index
            for index, item in enumerate(items)
            if item.get("page_idx") == 0
            and item.get("type") == "text"
            and ABSTRACT_RE.match(str(item.get("text", "")).strip())
            and "text_level" in item
        ),
        None,
    )
    if abstract_index is None:
        return
    title_indices = [
        index
        for index, item in enumerate(items[:abstract_index])
        if item.get("page_idx") == 0 and item.get("type") == "text" and "text_level" in item
    ]
    start = title_indices[0] + 1 if title_indices else 0
    for item in items[start:abstract_index]:
        if item.get("page_idx") == 0 and item.get("type") == "text" and "text_level" not in item:
            item["type"] = "author"


def group_logical_paragraphs(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = deepcopy(items)
    _merge_body_continuations(output)
    _merge_reference_continuations(output)
    _add_context(output)
    return output


def _merge_body_continuations(items: list[dict[str, Any]]) -> None:
    abstract_seen = False
    previous: dict[str, Any] | None = None
    for item in items:
        text = str(item.get("text", "")).strip()
        if item.get("type") == "text" and ABSTRACT_RE.match(text):
            abstract_seen = True
        if not abstract_seen or item.get("type") != "text" or not text:
            continue
        if "text_level" in item:
            previous = None
            continue
        if previous is not None:
            leading = first_alpha(text)
            if not sentence_is_complete(str(previous.get("text", ""))) and leading and not leading.isupper():
                item["logical_para_id"] = previous["logical_para_id"]
        previous = item


def _merge_reference_continuations(items: list[dict[str, Any]]) -> None:
    previous: dict[str, Any] | None = None
    for item in items:
        if item.get("type") != "reference" or not str(item.get("text", "")).strip():
            continue
        text = str(item["text"]).strip()
        leading = first_alpha(text)
        if previous is not None and "text_level" not in item and leading and not leading.isupper():
            item["logical_para_id"] = previous["logical_para_id"]
        previous = item


def _add_context(items: list[dict[str, Any]]) -> None:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        logical_id = item.get("logical_para_id")
        if isinstance(logical_id, int):
            groups[logical_id].append(item)
    for group in groups.values():
        if len(group) < 2:
            continue
        context = " ".join(str(item.get("text", "")).strip() for item in group if item.get("text")).strip()
        for item in group:
            item["context"] = context
