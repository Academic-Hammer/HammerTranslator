"""Canonical element taxonomy and translation policies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class ElementType(str, Enum):
    TEXT = "text"
    ABSTRACT = "abstract"
    TITLE = "title"
    SECTION_HEADER = "section_header"
    CAPTION = "caption"
    IMAGE_CAPTION = "image_caption"
    FIGURE_CAPTION = "figure_caption"
    TABLE_CAPTION = "table_caption"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_FOOTNOTE = "page_footnote"
    FOOTNOTE = "footnote"
    AUTHOR = "author"
    AFFILIATION = "affiliation"
    REFERENCE = "reference"
    URL = "url"
    PAGE_NUMBER = "page_number"
    EQUATION = "equation"
    IMAGE = "image"
    TABLE = "table"
    UNKNOWN = "unknown"


class TranslationPolicy(str, Enum):
    TRANSLATE_TEXT = "translate_text"
    TRANSLATE_AUTHOR_AFFILIATION = "translate_author_affiliation"
    TRANSLATE_TABLE_HTML = "translate_table_html"
    PRESERVE = "preserve"
    SKIP = "skip"
    NEEDS_REVIEW = "needs_review"


@dataclass(frozen=True)
class TaxonomyDecision:
    element_type: ElementType
    policy: TranslationPolicy
    reason: str


TEXT_TYPES = {
    ElementType.TEXT,
    ElementType.ABSTRACT,
    ElementType.TITLE,
    ElementType.SECTION_HEADER,
    ElementType.CAPTION,
    ElementType.IMAGE_CAPTION,
    ElementType.FIGURE_CAPTION,
    ElementType.TABLE_CAPTION,
    ElementType.HEADER,
    ElementType.FOOTER,
    ElementType.PAGE_FOOTNOTE,
    ElementType.FOOTNOTE,
    ElementType.AFFILIATION,
}
PRESERVE_TYPES = {
    ElementType.REFERENCE,
    ElementType.URL,
    ElementType.PAGE_NUMBER,
    ElementType.EQUATION,
}


def classify_element(item: Mapping[str, Any]) -> TaxonomyDecision:
    """Map one source block to exactly one conservative policy.

    MinerUAdapter represents headings as ``type=text`` with ``sub_type`` and
    ``text_level``.  The normalization below exposes those distinctions without
    changing the original source object.
    """

    raw_type = str(item.get("type") or "").strip().lower()
    sub_type = str(item.get("sub_type") or "").strip().lower()

    if raw_type == "text":
        if sub_type == "abstract":
            element_type = ElementType.ABSTRACT
        elif sub_type in {"title", "doc_title"}:
            element_type = ElementType.TITLE
        elif sub_type == "paragraph_title" or "text_level" in item:
            element_type = ElementType.SECTION_HEADER
        elif sub_type == "formula_number":
            element_type = ElementType.EQUATION
        else:
            element_type = ElementType.TEXT
    else:
        try:
            element_type = ElementType(raw_type)
        except ValueError:
            element_type = ElementType.UNKNOWN

    if element_type in TEXT_TYPES:
        policy = TranslationPolicy.TRANSLATE_TEXT
        reason = "textual element approved by the translation policy"
    elif element_type is ElementType.AUTHOR:
        policy = TranslationPolicy.TRANSLATE_AUTHOR_AFFILIATION
        reason = "preserve identity fields and translate affiliation fields in Step 3"
    elif element_type is ElementType.TABLE:
        policy = TranslationPolicy.TRANSLATE_TABLE_HTML
        reason = "translate HTML text nodes while preserving markup and numeric content"
    elif element_type in PRESERVE_TYPES:
        policy = TranslationPolicy.PRESERVE
        reason = "element must remain unchanged"
    elif element_type is ElementType.IMAGE:
        policy = TranslationPolicy.SKIP
        reason = "image body is outside the text translation stage"
    else:
        policy = TranslationPolicy.NEEDS_REVIEW
        reason = f"unknown source type {raw_type!r}; unknown types are never auto-translated"

    return TaxonomyDecision(element_type=element_type, policy=policy, reason=reason)
