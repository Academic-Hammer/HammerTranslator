"""Build deterministic translation units from validated input blocks."""

from __future__ import annotations

import hashlib
import re
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Any

from .authors import AuthorSplit, split_author_block
from .formula import FORMULA_CONTRACT_VERSION, FormulaBinding, build_formula_bindings
from .loader import LoadedBlock, LoadedDocument
from .protection import ContentProtector, OrderPolicy, ProtectedToken, ProtectionResult
from .taxonomy import ElementType, TaxonomyDecision, TranslationPolicy


CAPTION_TYPES = {
    ElementType.CAPTION,
    ElementType.IMAGE_CAPTION,
    ElementType.FIGURE_CAPTION,
    ElementType.TABLE_CAPTION,
}


_SUPERSCRIPT_CHARS = str.maketrans(
    "0123456789+-−=()abcdefghijklmnoprstuvwxyz",
    "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁻⁼⁽⁾ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻ",
)
_SUBSCRIPT_CHARS = str.maketrans(
    "0123456789+-−=()aehijklmnoprstuvx",
    "₀₁₂₃₄₅₆₇₈₉₊₋₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ",
)


def _normalize_inline_markup(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        content = match.group(2)
        table = _SUPERSCRIPT_CHARS if match.group(1).lower() == "sup" else _SUBSCRIPT_CHARS
        return content.translate(table)

    normalized = re.sub(r"<(sup|sub)>(.*?)</\1>", replace, value, flags=re.IGNORECASE | re.DOTALL)
    return re.sub(r"</?su[bp]>", "", normalized, flags=re.IGNORECASE)


@dataclass(frozen=True)
class TranslationUnit:
    translation_unit_id: str
    logical_para_id: int
    element_type: ElementType
    policy: TranslationPolicy
    block_ids: tuple[str, ...]
    source_text: str
    protected_text: str
    protected_tokens: tuple[ProtectedToken, ...]
    formula_bindings: tuple[FormulaBinding, ...] = ()
    quality_flags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def clean_protected_candidate(self, candidate: str) -> str:
        return ProtectionResult(
            original_text=self.source_text,
            protected_text=self.protected_text,
            tokens=self.protected_tokens,
        ).clean_candidate(candidate)

    def restore_protected_content(
        self, candidate: str, *, order_policy: OrderPolicy = "strict"
    ) -> str:
        restored = ProtectionResult(
            original_text=self.source_text,
            protected_text=self.protected_text,
            tokens=self.protected_tokens,
        ).restore(candidate, order_policy=order_policy)
        return _normalize_inline_markup(restored)


@dataclass(frozen=True)
class TranslationPlan:
    document: LoadedDocument
    units: tuple[TranslationUnit, ...]

    def public_summary(self) -> dict[str, Any]:
        type_counts = Counter(unit.element_type.value for unit in self.units)
        policy_counts = Counter(unit.policy.value for unit in self.units)
        return {
            **self.document.public_summary(),
            "translation_unit_count": len(self.units),
            "translation_unit_type_counts": dict(sorted(type_counts.items())),
            "translation_unit_policy_counts": dict(sorted(policy_counts.items())),
            "protected_token_count": sum(len(unit.protected_tokens) for unit in self.units),
            "formula_count": sum(len(unit.formula_bindings) for unit in self.units),
            "needs_review_count": sum(
                unit.policy is TranslationPolicy.NEEDS_REVIEW for unit in self.units
            ),
        }


def _caption_decisions(document: LoadedDocument) -> dict[str, TaxonomyDecision]:
    bodies: defaultdict[int, set[ElementType]] = defaultdict(set)
    for block in document.blocks:
        if block.taxonomy.element_type in {ElementType.TABLE, ElementType.IMAGE}:
            bodies[block.data.logical_para_id].add(block.taxonomy.element_type)

    decisions: dict[str, TaxonomyDecision] = {}
    for block in document.blocks:
        decision = block.taxonomy
        if decision.element_type is not ElementType.CAPTION:
            decisions[block.block_id] = decision
            continue
        related = bodies.get(block.data.logical_para_id, set())
        if related == {ElementType.TABLE}:
            decisions[block.block_id] = TaxonomyDecision(
                ElementType.TABLE_CAPTION,
                TranslationPolicy.TRANSLATE_TEXT,
                "generic caption resolved from its table logical paragraph",
            )
        elif related == {ElementType.IMAGE}:
            decisions[block.block_id] = TaxonomyDecision(
                ElementType.IMAGE_CAPTION,
                TranslationPolicy.TRANSLATE_TEXT,
                "generic caption resolved from its image logical paragraph",
            )
        else:
            decisions[block.block_id] = TaxonomyDecision(
                ElementType.CAPTION,
                TranslationPolicy.NEEDS_REVIEW,
                "generic caption has both or neither table/image body",
            )
    return decisions


def _group_key(block: LoadedBlock, decision: TaxonomyDecision) -> tuple[Any, ...]:
    logical_id = block.data.logical_para_id
    if decision.element_type in {ElementType.TABLE, ElementType.IMAGE}:
        return ("block", block.block_id)
    if decision.element_type in {
        ElementType.HEADER,
        ElementType.FOOTER,
        ElementType.PAGE_NUMBER,
        ElementType.PAGE_FOOTNOTE,
    }:
        # Running headers and footers can reuse a body paragraph's logical id
        # across a page break. They are page furniture and must never be folded
        # into the body text they happen to precede or follow.
        return ("margin", decision.element_type.value, block.block_id)
    if decision.element_type in CAPTION_TYPES:
        # MinerU can assign the same logical paragraph to a whole-figure caption
        # and labels detected inside the figure. A caption therefore never shares
        # a translation unit merely because its logical id matches another block.
        return ("caption", decision.element_type.value, block.block_id)
    if decision.element_type is ElementType.AUTHOR:
        return ("author", logical_id)
    if decision.policy is TranslationPolicy.TRANSLATE_TEXT:
        return ("text", logical_id)
    return (decision.policy.value, decision.element_type.value, logical_id)


def _merge_text(blocks: list[LoadedBlock]) -> str:
    if len(blocks) == 1:
        return blocks[0].data.text.strip()
    contexts = {
        str(block.raw.get("context")).strip()
        for block in blocks
        if isinstance(block.raw.get("context"), str) and str(block.raw.get("context")).strip()
    }
    if len(contexts) == 1:
        context = next(iter(contexts))
        normalized_context = re.sub(r"\s+", " ", context).strip().casefold()
        if all(
            not block.data.text.strip()
            or re.sub(r"\s+", " ", block.data.text).strip().casefold()
            in normalized_context
            for block in blocks
        ):
            return context
    merged = ""
    for block in blocks:
        text = block.data.text.strip()
        if not text:
            continue
        if merged.endswith("-") and text[:1].islower():
            merged += text
        elif merged:
            merged += " " + text
        else:
            merged = text
    return merged.strip()


def _unit_id(document_id: str, element_type: ElementType, logical_id: int, block_ids: tuple[str, ...]) -> str:
    payload = "\0".join((document_id, element_type.value, str(logical_id), *block_ids))
    return f"tu-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def _author_metadata(split: AuthorSplit) -> dict[str, Any]:
    return {
        "author_split": {
            "ambiguous": split.ambiguous,
            "reason": split.reason,
            "has_affiliation": split.has_affiliation,
            "segments": [
                {"text": segment.text, "role": segment.role, "start": segment.start, "end": segment.end}
                for segment in split.segments
            ],
        }
    }


def build_translation_plan(
    document: LoadedDocument,
    *,
    protector: ContentProtector | None = None,
) -> TranslationPlan:
    """Create one semantic unit per logical paragraph or structural body."""

    content_protector = protector or ContentProtector()
    decisions = _caption_decisions(document)
    groups: OrderedDict[tuple[Any, ...], list[LoadedBlock]] = OrderedDict()
    for block in document.blocks:
        decision = decisions[block.block_id]
        groups.setdefault(_group_key(block, decision), []).append(block)

    units: list[TranslationUnit] = []
    for blocks in groups.values():
        first = blocks[0]
        decision = decisions[first.block_id]
        element_type = decision.element_type
        policy = decision.policy
        flags: list[str] = []
        metadata: dict[str, Any] = {"policy_reason": decision.reason}

        if element_type is ElementType.TABLE:
            source_text = str(first.raw.get("table_body") or "")
            if not source_text:
                policy = TranslationPolicy.NEEDS_REVIEW
                flags.append("missing_table_body")
        elif element_type is ElementType.AUTHOR:
            source_text = "\n".join(
                block.data.text.strip() for block in blocks if block.data.text.strip()
            ).strip()
        else:
            source_text = _merge_text(blocks)

        extra_spans = ()
        table_mode = element_type is ElementType.TABLE
        if element_type is ElementType.AUTHOR and policy is not TranslationPolicy.PRESERVE:
            author_split = split_author_block(source_text)
            metadata.update(_author_metadata(author_split))
            extra_spans = author_split.protected_spans
            if author_split.ambiguous:
                policy = TranslationPolicy.NEEDS_REVIEW
                flags.append("ambiguous_author_block")
            elif not author_split.has_affiliation:
                policy = TranslationPolicy.PRESERVE

        should_protect = policy in {
            TranslationPolicy.TRANSLATE_TEXT,
            TranslationPolicy.TRANSLATE_AUTHOR_AFFILIATION,
            TranslationPolicy.TRANSLATE_TABLE_HTML,
        }
        if should_protect:
            protection = content_protector.protect(
                source_text,
                extra_spans=extra_spans,
                table_mode=table_mode,
            )
        else:
            protection = content_protector.protect(source_text)

        block_ids = tuple(block.block_id for block in blocks)
        logical_id = first.data.logical_para_id
        translation_unit_id = _unit_id(
            document.document_id, element_type, logical_id, block_ids
        )
        formula_bindings = build_formula_bindings(
            document_id=document.document_id,
            translation_unit_id=translation_unit_id,
            blocks=blocks,
            protected_tokens=protection.tokens,
        )
        statuses = {binding.binding_status for binding in formula_bindings}
        if "latex_mismatch" in statuses:
            flags.append("formula_latex_mismatch")
        if "protected_token_only" in statuses:
            flags.append("formula_source_span_missing")
        if "source_span_only" in statuses:
            flags.append("formula_source_span_not_protected")
            policy = TranslationPolicy.NEEDS_REVIEW
        if formula_bindings:
            metadata["formula_contract_version"] = FORMULA_CONTRACT_VERSION
        units.append(
            TranslationUnit(
                translation_unit_id=translation_unit_id,
                logical_para_id=logical_id,
                element_type=element_type,
                policy=policy,
                block_ids=block_ids,
                source_text=source_text,
                protected_text=protection.protected_text,
                protected_tokens=protection.tokens,
                formula_bindings=formula_bindings,
                quality_flags=tuple(flags),
                metadata=metadata,
            )
        )
    return TranslationPlan(document=document, units=tuple(units))
