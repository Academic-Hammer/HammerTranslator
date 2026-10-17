from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import fitz

from hammer_translator.translation.loader import DiscoveredInput, load_document
from hammer_translator.translation.protection import ContentProtector, ProtectionError
from hammer_translator.translation.taxonomy import ElementType, TranslationPolicy
from hammer_translator.translation.units import build_translation_plan


def block(index: int, *, block_type: str = "text", logical_id: int | None = None, text: str = "Text", **extra):
    y0 = 10 + index * 20
    value = {
        "type": block_type,
        "text": text,
        "page_idx": 0,
        "bbox": [10, y0, 180, y0 + 15],
        "logical_para_id": index if logical_id is None else logical_id,
    }
    value.update(extra)
    return value


def load_plan(root: Path, blocks: list[dict]):
    source = root / "paper_enriched.json"
    source.write_text(json.dumps(blocks), encoding="utf-8")
    pdf_path = root / "paper_origin.pdf"
    pdf = fitz.open()
    pdf.new_page(width=200, height=500)
    pdf.save(pdf_path)
    pdf.close()
    document = load_document(DiscoveredInput(source.resolve(), "test"))
    return build_translation_plan(document)


class ContentProtectionTests(unittest.TestCase):
    def test_supported_immutable_content_round_trips_exactly(self):
        source = (
            "See $x+y$, $$z=1$$, \\(a+b\\), \\[c=d\\], "
            "\\begin{equation}e=mc^2\\end{equation}, \\citep{smith2024}, "
            "\\ref{sec:intro}, [1–3], (Smith et al., 2024), "
            "https://example.org/a, doi 10.1234/ABC.9 and `code()`."
        )
        result = ContentProtector().protect(source)
        self.assertGreaterEqual(len(result.tokens), 12)
        translated_candidate = "译文 " + result.protected_text
        restored = result.restore(translated_candidate)
        self.assertEqual(restored, "译文 " + source)

    def test_missing_duplicate_and_reordered_tokens_are_rejected(self):
        result = ContentProtector().protect("$a$ and https://example.org")
        first, second = (token.placeholder for token in result.tokens)
        with self.assertRaisesRegex(ProtectionError, "missing"):
            result.restore(result.protected_text.replace(first, ""))
        with self.assertRaisesRegex(ProtectionError, "duplicated"):
            result.restore(result.protected_text + first)
        swapped = result.protected_text.replace(first, "TEMP").replace(second, first).replace("TEMP", second)
        with self.assertRaisesRegex(ProtectionError, "order"):
            result.restore(swapped)

    def test_table_mode_protects_markup_numbers_and_statistics(self):
        source = '<table><tr><td rowspan="2">Accuracy</td><td>91.5***</td></tr></table>'
        result = ContentProtector().protect(source, table_mode=True)
        kinds = {token.kind for token in result.tokens}
        self.assertIn("html_tag", kinds)
        self.assertIn("number", kinds)
        self.assertEqual(result.restore(result.protected_text), source)


class TranslationUnitTests(unittest.TestCase):
    def test_same_logical_paragraph_becomes_one_unit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            blocks = [
                block(0, logical_id=7, text="First part", context="First part second part."),
                block(1, logical_id=7, text="second part.", context="First part second part."),
            ]
            plan = load_plan(Path(temp_dir), blocks)
            self.assertEqual(len(plan.units), 1)
            self.assertEqual(plan.units[0].source_text, "First part second part.")
            self.assertEqual(len(plan.units[0].block_ids), 2)

    def test_table_and_caption_are_separate_units(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            blocks = [
                block(0, block_type="caption", logical_id=3, text="Table 1: Results"),
                block(
                    1,
                    block_type="table",
                    logical_id=3,
                    text="",
                    table_body="<table><tr><td>Score</td><td>2.0</td></tr></table>",
                ),
            ]
            plan = load_plan(Path(temp_dir), blocks)
            self.assertEqual(len(plan.units), 2)
            by_type = {unit.element_type: unit for unit in plan.units}
            self.assertIn(ElementType.TABLE_CAPTION, by_type)
            self.assertIn(ElementType.TABLE, by_type)
            self.assertEqual(by_type[ElementType.TABLE].policy, TranslationPolicy.TRANSLATE_TABLE_HTML)

    def test_image_caption_uses_requested_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            blocks = [
                block(0, block_type="image", logical_id=4, text="", img_path="images/a.png"),
                block(1, block_type="caption", logical_id=4, text="Figure 1: Pipeline"),
            ]
            plan = load_plan(Path(temp_dir), blocks)
            caption = next(unit for unit in plan.units if unit.element_type is ElementType.IMAGE_CAPTION)
            self.assertEqual(caption.policy, TranslationPolicy.TRANSLATE_TEXT)

    def test_whole_image_caption_and_internal_labels_are_never_aggregated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            shared_context = "Fig. 7. Grad-CAM Heatmap Breast Cancer Image"
            blocks = [
                block(
                    0,
                    block_type="image",
                    logical_id=7,
                    text="",
                    img_path="images/a.png",
                ),
                block(
                    1,
                    block_type="caption",
                    logical_id=7,
                    text="Fig. 7. Grad-CAM Heatmap",
                    context=shared_context,
                ),
                block(
                    2,
                    block_type="caption",
                    logical_id=7,
                    text="Breast Cancer Image",
                    context=shared_context,
                ),
            ]
            plan = load_plan(Path(temp_dir), blocks)
            captions = [
                unit
                for unit in plan.units
                if unit.element_type is ElementType.IMAGE_CAPTION
            ]

            self.assertEqual(len(captions), 2)
            self.assertEqual(
                [unit.source_text for unit in captions],
                ["Fig. 7. Grad-CAM Heatmap", "Breast Cancer Image"],
            )
            self.assertTrue(all(len(unit.block_ids) == 1 for unit in captions))

    def test_unresolved_generic_caption_requires_review(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            plan = load_plan(
                Path(temp_dir),
                [block(0, block_type="caption", logical_id=5, text="Unattached caption")],
            )
            self.assertEqual(plan.units[0].element_type, ElementType.CAPTION)
            self.assertEqual(plan.units[0].policy, TranslationPolicy.NEEDS_REVIEW)

    def test_author_names_and_identity_fields_are_protected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            author = (
                "Jia-Ling Shi\nBeijing Institute of\nTechnology\n"
                "Beijing, China\nsjl@bit.edu.cn"
            )
            plan = load_plan(
                Path(temp_dir),
                [block(0, block_type="author", text=author)],
            )
            unit = plan.units[0]
            self.assertEqual(unit.policy, TranslationPolicy.TRANSLATE_AUTHOR_AFFILIATION)
            self.assertNotIn("Jia-Ling Shi", unit.protected_text)
            self.assertNotIn("sjl@bit.edu.cn", unit.protected_text)
            self.assertIn("Beijing Institute of", unit.protected_text)
            self.assertEqual(unit.restore_protected_content(unit.protected_text), author)
            # The unit metadata records the exact split used for custom protection.
            roles = [segment["role"] for segment in unit.metadata["author_split"]["segments"]]
            self.assertEqual(roles, ["name", "affiliation", "affiliation", "affiliation", "email"])

    def test_ambiguous_author_is_preserved_for_review(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            plan = load_plan(
                Path(temp_dir),
                [block(0, block_type="author", text="Alpha & Beta Consortium\nSomewhere")],
            )
            unit = plan.units[0]
            self.assertEqual(unit.policy, TranslationPolicy.NEEDS_REVIEW)
            self.assertIn("ambiguous_author_block", unit.quality_flags)

    def test_reference_is_preserved_and_image_is_skipped(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            plan = load_plan(
                Path(temp_dir),
                [
                    block(0, block_type="reference", text="Smith (2024). Paper."),
                    block(1, block_type="image", text="", img_path="images/a.png"),
                ],
            )
            policies = {unit.element_type: unit.policy for unit in plan.units}
            self.assertEqual(policies[ElementType.REFERENCE], TranslationPolicy.PRESERVE)
            self.assertEqual(policies[ElementType.IMAGE], TranslationPolicy.SKIP)

    def test_plan_is_deterministic_and_does_not_mutate_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            blocks = [block(0, text="Text with $x$."), block(1, text="More text")]
            original = deepcopy(blocks)
            first = load_plan(Path(temp_dir), blocks)
            second = load_plan(Path(temp_dir), blocks)
            self.assertEqual(
                [unit.translation_unit_id for unit in first.units],
                [unit.translation_unit_id for unit in second.units],
            )
            self.assertEqual(blocks, original)


if __name__ == "__main__":
    unittest.main()
