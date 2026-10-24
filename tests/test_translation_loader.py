from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import fitz

from hammer_translator.translation.loader import (
    DiscoveredInput,
    InputDiscoveryError,
    InputValidationError,
    discover_source,
    load_document,
    load_documents,
)
from hammer_translator.translation.taxonomy import ElementType, TranslationPolicy
from hammer_translator.translation.units import build_translation_plan


def write_pdf(path: Path, pages: int = 1, width: float = 200, height: float = 300) -> None:
    document = fitz.open()
    for _ in range(pages):
        document.new_page(width=width, height=height)
    document.save(path)
    document.close()


def write_caption_pdf(path: Path, caption: str) -> None:
    document = fitz.open()
    page = document.new_page(width=600, height=800)
    remaining = page.insert_textbox(
        fitz.Rect(90, 220, 523, 300),
        caption,
        fontsize=9,
        fontname="helv",
        lineheight=1.1,
    )
    if remaining < 0:
        raise AssertionError("test caption did not fit in its text box")
    document.save(path)
    document.close()


def valid_block(**updates):
    block = {
        "type": "text",
        "text": "Source text",
        "page_idx": 0,
        "bbox": [10, 10, 100, 30],
        "logical_para_id": 0,
        "rich_spans": [{"font": "Times", "size": 10}],
    }
    block.update(updates)
    return block


class TranslationDiscoveryTests(unittest.TestCase):
    def test_directory_prefers_single_enriched_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "paper_enriched.json").write_text("[]", encoding="utf-8")
            (root / "paper_content_list.json").write_text("[]", encoding="utf-8")
            selected = discover_source(root)
            self.assertEqual(selected.source_json.name, "paper_enriched.json")
            self.assertEqual(selected.selection_reason, "directory_preferred_enriched")

    def test_directory_fallback_ignores_native_and_v2_backups(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "paper_mineru_content_list.json").write_text("[]", encoding="utf-8")
            (root / "paper_content_list_v2.json").write_text("[]", encoding="utf-8")
            (root / "paper_content_list.json").write_text("[]", encoding="utf-8")
            selected = discover_source(root)
            self.assertEqual(selected.source_json.name, "paper_content_list.json")
            self.assertEqual(
                selected.selection_reason,
                "directory_fallback_adapted_content_list",
            )

    def test_multiple_same_priority_candidates_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "a_enriched.json").write_text("[]", encoding="utf-8")
            (root / "b_enriched.json").write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(InputDiscoveryError, "multiple enriched"):
                discover_source(root)

    def test_explicit_json_is_used_exactly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "custom.json"
            path.write_text("[]", encoding="utf-8")
            selected = discover_source(path)
            self.assertEqual(selected.source_json, path.resolve())
            self.assertEqual(selected.selection_reason, "explicit_json")


class TranslationSchemaTests(unittest.TestCase):
    def load(self, root: Path, blocks: list[dict], pages: int = 1):
        source = root / "paper_enriched.json"
        source.write_text(json.dumps(blocks), encoding="utf-8")
        pdf = root / "paper_origin.pdf"
        write_pdf(pdf, pages=pages)
        return load_document(DiscoveredInput(source.resolve(), "test"))

    def test_valid_blocks_receive_stable_ids_and_keep_extra_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            blocks = [
                valid_block(),
                valid_block(
                    type="table",
                    text="",
                    bbox=[20, 40, 150, 120],
                    logical_para_id=1,
                    table_body="<table><tr><td>A</td></tr></table>",
                ),
            ]
            first = self.load(root, blocks)
            second = self.load(root, blocks)
            self.assertEqual([b.block_id for b in first.blocks], ["p0001-b0001", "p0001-b0002"])
            self.assertEqual(
                [b.block_id for b in first.blocks],
                [b.block_id for b in second.blocks],
            )
            self.assertEqual(first.blocks[1].raw["table_body"], blocks[1]["table_body"])
            self.assertEqual(first.blocks[1].taxonomy.policy, TranslationPolicy.TRANSLATE_TABLE_HTML)

    def test_partial_context_does_not_replace_merged_block_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            first = "First part of the paragraph"
            second = "second part of the paragraph"
            document = self.load(
                Path(temp_dir),
                [
                    valid_block(text=first, bbox=[10, 10, 180, 30], logical_para_id=3),
                    valid_block(
                        text=second,
                        bbox=[10, 35, 180, 55],
                        logical_para_id=3,
                        context=second,
                    ),
                ],
            )
            unit = build_translation_plan(document).units[0]
            self.assertEqual(unit.source_text, f"{first} {second}")

    def test_complete_shared_context_still_reconstructs_a_paragraph(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            first = "First part of the paragraph"
            second = "second part of the paragraph"
            context = f"{first} {second}"
            document = self.load(
                Path(temp_dir),
                [
                    valid_block(
                        text=first,
                        bbox=[10, 10, 180, 30],
                        logical_para_id=4,
                        context=context,
                    ),
                    valid_block(
                        text=second,
                        bbox=[10, 35, 180, 55],
                        logical_para_id=4,
                        context=context,
                    ),
                ],
            )
            unit = build_translation_plan(document).units[0]
            self.assertEqual(unit.source_text, context)

    def test_word_split_across_pages_is_rejoined_before_translation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            document = self.load(
                Path(temp_dir),
                [
                    valid_block(
                        text="This paragraph ends with Conven-",
                        bbox=[10, 10, 180, 30],
                        logical_para_id=5,
                    ),
                    valid_block(
                        text="tional charge trapping.",
                        bbox=[10, 10, 180, 30],
                        logical_para_id=6,
                        page_idx=1,
                    ),
                ],
                pages=2,
            )
            self.assertEqual(document.blocks[0].data.text, "This paragraph ends with")
            self.assertEqual(
                document.blocks[1].data.text, "Conventional charge trapping."
            )

    def test_heading_is_normalized_without_changing_raw_type(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            document = self.load(
                Path(temp_dir),
                [valid_block(sub_type="doc_title", text_level=1)],
            )
            block = document.blocks[0]
            self.assertEqual(block.raw["type"], "text")
            self.assertEqual(block.taxonomy.element_type, ElementType.TITLE)
            self.assertEqual(block.taxonomy.policy, TranslationPolicy.TRANSLATE_TEXT)

    def test_unknown_type_requires_review(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            document = self.load(Path(temp_dir), [valid_block(type="future_widget")])
            self.assertEqual(document.blocks[0].taxonomy.element_type, ElementType.UNKNOWN)
            self.assertEqual(document.blocks[0].taxonomy.policy, TranslationPolicy.NEEDS_REVIEW)

    def test_chart_caption_uses_real_pdf_text_bbox_and_is_translated(self):
        cases = (
            (
                "Figure 5: The DBSCAN filtering workflow. (a) Raw 3D probability volume.",
                ElementType.FIGURE_CAPTION,
            ),
            (
                "Table 1: Number of labeled inlines and crosslines used for training.",
                ElementType.TABLE_CAPTION,
            ),
        )
        for caption, expected_type in cases:
            with self.subTest(expected_type=expected_type), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                source = root / "paper_enriched.json"
                source.write_text(
                    json.dumps(
                        [
                            valid_block(
                                type="chart",
                                text=caption,
                                bbox=[376, 84, 510, 205],
                            )
                        ]
                    ),
                    encoding="utf-8",
                )
                write_caption_pdf(root / "paper_origin.pdf", caption)
                document = load_document(DiscoveredInput(source.resolve(), "test"))
                caption_blocks = [
                    block
                    for block in document.blocks
                    if block.taxonomy.element_type is expected_type
                ]
                self.assertEqual(len(caption_blocks), 1)
                block = caption_blocks[0]
                self.assertEqual(block.taxonomy.element_type, expected_type)
                self.assertEqual(block.taxonomy.policy, TranslationPolicy.TRANSLATE_TEXT)
                self.assertAlmostEqual(block.data.bbox[0], 90, delta=2)
                self.assertGreater(block.data.bbox[2], 300)
                self.assertGreater(block.data.bbox[1], 210)
                self.assertLess(block.data.bbox[3], 310)
                self.assertTrue(
                    any(
                        candidate.taxonomy.element_type is ElementType.IMAGE
                        for candidate in document.blocks
                    )
                )

    def test_mixed_chart_text_is_split_into_graphic_and_translated_caption(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            caption = (
                "Fig 5. The rationale for rejecting Loan Application 2 with a 0.98 "
                "probability, supported by a confidence level within the threshold."
            )
            source = root / "paper_enriched.json"
            source.write_text(
                json.dumps(
                    [
                        valid_block(
                            type="chart",
                            text=(
                                "Loan Application 1: Decision={(Fund,0.02),(Reject,0.98)} "
                                + caption
                            ),
                            bbox=[60, 100, 540, 400],
                        )
                    ]
                ),
                encoding="utf-8",
            )
            pdf = fitz.open()
            page = pdf.new_page(width=600, height=800)
            pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 20, 20), 0)
            pixmap.set_rect(pixmap.irect, (220, 220, 220))
            page.insert_image(fitz.Rect(60, 100, 540, 400), pixmap=pixmap)
            remaining = page.insert_textbox(
                fitz.Rect(70, 420, 540, 470),
                caption,
                fontsize=9,
            )
            self.assertGreaterEqual(remaining, 0)
            pdf.save(root / "paper_origin.pdf")
            pdf.close()

            document = load_document(DiscoveredInput(source.resolve(), "test"))
            by_type = {block.taxonomy.element_type: block for block in document.blocks}
            self.assertEqual(len(document.blocks), 2)
            self.assertIn(ElementType.IMAGE, by_type)
            self.assertIn(ElementType.FIGURE_CAPTION, by_type)
            self.assertEqual(by_type[ElementType.IMAGE].taxonomy.policy, TranslationPolicy.SKIP)
            caption_block = by_type[ElementType.FIGURE_CAPTION]
            self.assertEqual(caption_block.taxonomy.policy, TranslationPolicy.TRANSLATE_TEXT)
            self.assertIn("Fig 5.", caption_block.data.text)
            self.assertGreaterEqual(caption_block.data.bbox[1], 410)
            self.assertLessEqual(by_type[ElementType.IMAGE].data.bbox[3], 400)

    def test_whole_missing_figure_caption_is_recovered_from_pdf(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            caption = (
                "Figure 3. Probability distributions of mass, age, and radius "
                "for KIC 8379927 on the corner plot (Foreman-Mackey 2016)."
            )
            source = root / "paper_enriched.json"
            source.write_text(
                json.dumps(
                    [
                        valid_block(
                            type="image",
                            text="",
                            bbox=[60, 100, 540, 400],
                            logical_para_id=1,
                        )
                    ]
                ),
                encoding="utf-8",
            )
            pdf = fitz.open()
            page = pdf.new_page(width=600, height=800)
            pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 20, 20), 0)
            pixmap.set_rect(pixmap.irect, (220, 220, 220))
            page.insert_image(fitz.Rect(60, 100, 540, 400), pixmap=pixmap)
            remaining = page.insert_textbox(
                fitz.Rect(70, 420, 540, 470),
                caption,
                fontsize=9,
            )
            self.assertGreaterEqual(remaining, 0)
            pdf.save(root / "paper_origin.pdf")
            pdf.close()

            document = load_document(DiscoveredInput(source.resolve(), "test"))
            captions = [
                block
                for block in document.blocks
                if block.taxonomy.element_type is ElementType.FIGURE_CAPTION
            ]
            self.assertEqual(len(captions), 1)
            caption_block = captions[0]
            self.assertEqual(caption_block.taxonomy.policy, TranslationPolicy.TRANSLATE_TEXT)
            self.assertIn("Figure 3.", caption_block.data.text)
            self.assertGreaterEqual(caption_block.data.bbox[1], 410)

    def test_head_truncated_body_is_recovered_from_pdf(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prefix = (
                "timescale for core collapse scales as a steep power of concentration, "
                "where the concentration follows the NFW profile."
            )
            suffix = (
                "We aim to understand how a prompt cusp at the center of a low-mass halo "
                "impacts the time evolution of the halo profile."
            )
            pdf = fitz.open()
            page = pdf.new_page(width=600, height=800)
            remaining = page.insert_textbox(
                fitz.Rect(50, 50, 300, 220),
                f"{prefix} {suffix}",
                fontsize=9,
                fontname="helv",
                lineheight=1.1,
            )
            self.assertGreaterEqual(remaining, 0)
            pdf.save(root / "paper_origin.pdf")
            pdf.close()
            source = root / "paper_enriched.json"
            source.write_text(
                json.dumps([valid_block(text=suffix, bbox=[50, 110, 300, 190])]),
                encoding="utf-8",
            )
            document = load_document(DiscoveredInput(source.resolve(), "test"))
            block = document.blocks[0]
            self.assertTrue(block.data.text.startswith("timescale for core collapse"))
            self.assertLess(block.data.bbox[1], 70)

    def test_formula_tail_is_not_absorbed_into_following_body_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            formula_tail = "h < 1 0, x h > 1 (A6)"
            sentence = "Applying this to Equation A4, we obtain"
            pdf = fitz.open()
            page = pdf.new_page(width=300, height=400)
            remaining = page.insert_textbox(
                fitz.Rect(50, 80, 290, 140),
                f"{formula_tail}\n{sentence}",
                fontsize=9,
                fontname="helv",
            )
            self.assertGreaterEqual(remaining, 0)
            pdf.save(root / "paper_origin.pdf")
            pdf.close()

            with fitz.open(root / "paper_origin.pdf") as source_pdf:
                sentence_rect = source_pdf[0].search_for(sentence)[0]
            source = root / "paper_enriched.json"
            source.write_text(
                json.dumps(
                    [
                        valid_block(
                            text=sentence,
                            bbox=[
                                sentence_rect.x0 - 2,
                                sentence_rect.y0 - 2,
                                sentence_rect.x1 + 2,
                                sentence_rect.y1 + 2,
                            ],
                            lines=[
                                {
                                    "bbox": list(sentence_rect),
                                    "spans": [
                                        {
                                            "type": "text",
                                            "content": sentence,
                                            "bbox": list(sentence_rect),
                                        }
                                    ],
                                }
                            ],
                        )
                    ]
                ),
                encoding="utf-8",
            )
            document = load_document(DiscoveredInput(source.resolve(), "test"))
            block = document.blocks[0]
            self.assertEqual(block.data.text, sentence)
            self.assertGreater(block.data.bbox[1], sentence_rect.y0 - 3)

    def test_multicolumn_text_block_is_split_by_column(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            left = "Left column text describes the halo density profile."
            right = "Right column text describes the velocity dispersion."
            pdf = fitz.open()
            page = pdf.new_page(width=600, height=800)
            page.insert_textbox(
                fitz.Rect(50, 100, 250, 180), left, fontsize=10, fontname="helv"
            )
            page.insert_textbox(
                fitz.Rect(320, 100, 520, 180), right, fontsize=10, fontname="helv"
            )
            pdf.save(root / "paper_origin.pdf")
            pdf.close()
            source = root / "paper_enriched.json"
            source.write_text(
                json.dumps(
                    [
                        valid_block(
                            text=f"{left} {right}",
                            bbox=[50, 100, 520, 180],
                            lines=[
                                {
                                    "bbox": [50, 105, 250, 120],
                                    "spans": [{"type": "text", "content": left}],
                                },
                                {
                                    "bbox": [320, 105, 520, 120],
                                    "spans": [{"type": "text", "content": right}],
                                },
                            ],
                        )
                    ]
                ),
                encoding="utf-8",
            )
            document = load_document(DiscoveredInput(source.resolve(), "test"))
            text_blocks = [
                block
                for block in document.blocks
                if block.taxonomy.element_type is ElementType.TEXT
            ]
            self.assertEqual(len(text_blocks), 2)
            self.assertLess(text_blocks[0].data.bbox[0], 300)
            self.assertGreater(text_blocks[1].data.bbox[0], 300)
            self.assertIn("Left column", text_blocks[0].data.text)
            self.assertIn("Right column", text_blocks[1].data.text)

    def test_cross_column_continuation_ignores_top_margin_page_number(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            left = (
                "This section describes the simulation setup and the initial halo "
                "profiles, including the model for the prompt cusps. Section II B de-"
            )
            right = "tails the numerical setup of the N-body simulations."
            source = root / "paper_enriched.json"
            source.write_text(
                json.dumps(
                    [
                        valid_block(text=left, bbox=[50, 100, 280, 200], logical_para_id=1),
                        valid_block(
                            text=right,
                            bbox=[320, 80, 560, 140],
                            logical_para_id=2,
                        ),
                        valid_block(
                            type="page_number",
                            text="2",
                            bbox=[555, 20, 565, 35],
                            logical_para_id=3,
                        ),
                    ]
                ),
                encoding="utf-8",
            )
            pdf = fitz.open()
            page = pdf.new_page(width=600, height=800)
            page.insert_textbox(fitz.Rect(50, 100, 280, 200), left, fontsize=10)
            page.insert_textbox(fitz.Rect(320, 80, 560, 140), right, fontsize=10)
            page.insert_text((555, 32), "2", fontsize=10)
            pdf.save(root / "paper_origin.pdf")
            pdf.close()

            document = load_document(DiscoveredInput(source.resolve(), "test"))
            left_block = next(block for block in document.blocks if block.data.text == left)
            right_block = next(block for block in document.blocks if block.data.text == right)
            self.assertEqual(
                left_block.data.logical_para_id,
                right_block.data.logical_para_id,
            )
            plan = build_translation_plan(document)
            unit = next(
                unit
                for unit in plan.units
                if left_block.block_id in unit.block_ids
            )
            self.assertIn(right_block.block_id, unit.block_ids)
            self.assertIn("de-tails", unit.source_text)
            self.assertNotIn("de- tails", unit.source_text)

    def test_textless_table_is_recovered_as_translatable_cells(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "paper_enriched.json"
            source.write_text(
                json.dumps(
                    [
                        valid_block(
                            type="table",
                            text="",
                            bbox=[70, 60, 560, 115],
                            logical_para_id=4,
                        )
                    ]
                ),
                encoding="utf-8",
            )
            pdf = fitz.open()
            page = pdf.new_page(width=600, height=400)
            page.draw_rect(fitz.Rect(70, 60, 560, 115), color=(0, 0, 0), width=1)
            page.insert_text((80, 80), "Learning", fontsize=10)
            page.insert_text((80, 96), "rate", fontsize=10)
            page.insert_text((170, 80), "The step size at each iteration", fontsize=10)
            page.insert_text((170, 96), "while moving toward a minimum", fontsize=10)
            page.insert_text((390, 80), "0.0001", fontsize=10)
            pdf.save(root / "paper_origin.pdf")
            pdf.close()

            document = load_document(DiscoveredInput(source.resolve(), "test"))
            self.assertFalse(any(block.data.type == "table" for block in document.blocks))
            self.assertGreaterEqual(len(document.blocks), 2)
            texts = " ".join(block.data.text for block in document.blocks)
            self.assertIn("Learning", texts)
            self.assertIn("step size", texts)
            self.assertIn("0.0001", texts)
            self.assertTrue(
                all(block.raw.get("style_role") == "table" for block in document.blocks)
            )
            plan = build_translation_plan(document)
            self.assertTrue(
                all(unit.policy is TranslationPolicy.TRANSLATE_TEXT for unit in plan.units)
            )

    def test_repeated_running_footer_is_not_merged_with_body_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            footer = "J. Luis Silva, J. Gomes, and A. Cerqueira"
            body = "This body paragraph continues across the page break and must stay separate."
            source = root / "paper_enriched.json"
            source.write_text(
                json.dumps(
                    [
                        valid_block(page_idx=0, text=footer, bbox=[20, 370, 280, 392], logical_para_id=7),
                        valid_block(page_idx=1, text=footer, bbox=[20, 370, 280, 392], logical_para_id=7),
                        valid_block(page_idx=2, text=footer, bbox=[20, 370, 280, 392], logical_para_id=7),
                        valid_block(page_idx=2, text=body, bbox=[20, 80, 280, 140], logical_para_id=7),
                    ]
                ),
                encoding="utf-8",
            )
            pdf = fitz.open()
            for page_index in range(3):
                page = pdf.new_page(width=300, height=400)
                page.insert_text((20, 380), footer, fontsize=8)
                if page_index == 2:
                    page.insert_textbox(fitz.Rect(20, 80, 280, 140), body, fontsize=9)
            pdf.save(root / "paper_origin.pdf")
            pdf.close()

            document = load_document(DiscoveredInput(source.resolve(), "test"))
            footers = [block for block in document.blocks if block.data.text == footer]
            self.assertEqual(len(footers), 3)
            self.assertTrue(
                all(block.taxonomy.element_type is ElementType.FOOTER for block in footers)
            )
            plan = build_translation_plan(document)
            for unit in plan.units:
                texts = [block.data.text for block in document.blocks if block.block_id in unit.block_ids]
                self.assertFalse(footer in texts and body in texts)

    def test_text_bbox_is_corrected_when_mineru_box_covers_a_figure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            text = "The recovered paragraph starts below the figure and remains fully visible."
            source = root / "paper_enriched.json"
            source.write_text(
                json.dumps([valid_block(text=text, bbox=[20, 20, 280, 380])]),
                encoding="utf-8",
            )
            pdf = fitz.open()
            page = pdf.new_page(width=300, height=400)
            pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 20, 20), 0)
            pixmap.set_rect(pixmap.irect, (255, 0, 0))
            page.insert_image(fitz.Rect(20, 20, 280, 250), pixmap=pixmap)
            remaining = page.insert_textbox(
                fitz.Rect(20, 280, 280, 360), text, fontsize=9
            )
            self.assertGreaterEqual(remaining, 0)
            pdf.save(root / "paper_origin.pdf")
            pdf.close()

            document = load_document(DiscoveredInput(source.resolve(), "test"))
            bbox = document.blocks[0].data.bbox
            self.assertGreaterEqual(bbox[1], 270)
            self.assertLessEqual(bbox[3], 370)

    def test_remote_pdf_text_match_does_not_replace_line_geometry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            text = "The recovered paragraph remains at its authoritative line geometry."
            source = root / "paper_enriched.json"
            source.write_text(
                json.dumps(
                    [
                        valid_block(
                            text=text,
                            bbox=[20, 40, 280, 80],
                            lines=[
                                {
                                    "bbox": [20, 40, 280, 58],
                                    "spans": [
                                        {
                                            "type": "text",
                                            "content": text,
                                            "bbox": [20, 40, 280, 58],
                                        }
                                    ],
                                }
                            ],
                        )
                    ]
                ),
                encoding="utf-8",
            )
            pdf = fitz.open()
            page = pdf.new_page(width=300, height=400)
            remaining = page.insert_textbox(
                fitz.Rect(20, 200, 280, 260), text, fontsize=9
            )
            self.assertGreaterEqual(remaining, 0)
            pdf.save(root / "paper_origin.pdf")
            pdf.close()

            document = load_document(DiscoveredInput(source.resolve(), "test"))
            bbox = document.blocks[0].data.bbox
            self.assertGreaterEqual(bbox[1], 40)
            self.assertLessEqual(bbox[3], 80)

    def test_symbol_only_pdf_block_does_not_replace_a_body_bbox(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            text = (
                "The immutability of a blockchain ledger enables trust among multiple "
                "organizations sharing expertise to improve data representation."
            )
            source = root / "paper_enriched.json"
            source.write_text(
                json.dumps([valid_block(text=text, bbox=[20, 40, 280, 100])]),
                encoding="utf-8",
            )
            pdf = fitz.open()
            page = pdf.new_page(width=300, height=400)
            page.insert_text((20, 25), "()", fontsize=9)
            remaining = page.insert_textbox(
                fitz.Rect(20, 40, 280, 100), text, fontsize=9
            )
            self.assertGreaterEqual(remaining, 0)
            pdf.save(root / "paper_origin.pdf")
            pdf.close()

            document = load_document(DiscoveredInput(source.resolve(), "test"))
            self.assertEqual(tuple(document.blocks[0].data.bbox), (20.0, 40.0, 280.0, 100.0))

    def test_pdf_text_block_split_at_inline_formula_does_not_truncate_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_pdf_block = (
                "The design of the proposed Context-Fusion Attention U-Net was validated "
                "through an ablation study with a spatial head (CFAS"
            )
            second_pdf_block = (
                "U-Net) that successfully addressed the low-recall problem and improved "
                "coverage at ten-line spacing."
            )
            full_text = (
                "The design of the proposed Context-Fusion Attention U-Net was validated "
                "through an ablation study with a spatial head $( C F A ^ { S }$ U-Net) "
                "that successfully addressed the low-recall problem and improved coverage "
                "at ten-line spacing."
            )
            block = valid_block(
                text=full_text,
                bbox=[20, 40, 280, 140],
                logical_para_id=1,
                lines=[
                    {
                        "bbox": [20, 40, 280, 58],
                        "spans": [
                            {
                                "type": "text",
                                "content": "The design of the proposed Context-Fusion Attention "
                                "U-Net was validated through an ablation study with a spatial head (",
                                "bbox": [20, 40, 240, 58],
                            },
                            {
                                "type": "inline_equation",
                                "content": "( C F A ^ { S }",
                                "bbox": [240, 40, 270, 58],
                                "score": 0.9,
                            },
                        ],
                    },
                    {
                        "bbox": [20, 60, 280, 78],
                        "spans": [
                            {
                                "type": "text",
                                "content": second_pdf_block,
                                "bbox": [20, 60, 280, 78],
                            }
                        ],
                    },
                ],
            )
            source = root / "paper_enriched.json"
            source.write_text(json.dumps([block]), encoding="utf-8")
            pdf = fitz.open()
            page = pdf.new_page(width=300, height=400)
            page.insert_textbox(fitz.Rect(20, 40, 280, 100), first_pdf_block, fontsize=9)
            page.insert_textbox(fitz.Rect(20, 104, 280, 140), second_pdf_block, fontsize=9)
            pdf.save(root / "paper_origin.pdf")
            pdf.close()

            document = load_document(DiscoveredInput(source.resolve(), "test"))
            self.assertEqual(document.blocks[0].data.text, full_text)
            unit = build_translation_plan(document).units[0]
            self.assertEqual(unit.policy, TranslationPolicy.TRANSLATE_TEXT)
            self.assertNotIn("formula_source_span_not_protected", unit.quality_flags)
            self.assertEqual(len(unit.formula_bindings), 1)

    def test_parser_truncated_paragraph_tail_is_recovered_from_pdf(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            truncated = (
                "This complex structural and stratigraphic setting serves as an excellent "
                "test case for the robustness of the"
            )
            full_text = f"{truncated} proposed segmentation framework."
            source = root / "paper_enriched.json"
            source.write_text(
                json.dumps([valid_block(text=truncated, bbox=[20, 40, 280, 90])]),
                encoding="utf-8",
            )
            pdf = fitz.open()
            page = pdf.new_page(width=300, height=400)
            remaining = page.insert_textbox(
                fitz.Rect(20, 40, 280, 100), full_text, fontsize=9
            )
            self.assertGreaterEqual(remaining, 0)
            pdf.save(root / "paper_origin.pdf")
            pdf.close()

            document = load_document(DiscoveredInput(source.resolve(), "test"))
            self.assertEqual(document.blocks[0].data.text, full_text)
            self.assertEqual(document.blocks[0].raw["context"], full_text)

    def test_parser_items_sharing_one_pdf_block_are_not_duplicated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = (
                "<sup>1</sup>The internal structures of dark matter halos encode "
                "information about the particle nature of dark matter."
            )
            second = (
                "Most analyses examine core collapse in the context of classical "
                "dwarf galaxies and their observed density profiles.<sup>∗</sup>"
            )
            pdf_first = first.replace("<sup>1</sup>", "1")
            pdf_second = second.replace("<sup>∗</sup>", "∗")
            source = root / "paper_enriched.json"
            source.write_text(
                json.dumps(
                    [
                        valid_block(text=first, bbox=[20, 40, 280, 100]),
                        valid_block(text=second, bbox=[20, 100, 280, 180], logical_para_id=1),
                    ]
                ),
                encoding="utf-8",
            )
            pdf = fitz.open()
            page = pdf.new_page(width=300, height=500)
            remaining = page.insert_textbox(
                fitz.Rect(20, 40, 280, 180),
                f"{pdf_first} {pdf_second}",
                fontsize=9,
            )
            self.assertGreaterEqual(remaining, 0)
            pdf.save(root / "paper_origin.pdf")
            pdf.close()

            document = load_document(DiscoveredInput(source.resolve(), "test"))
            texts = [block.data.text for block in document.blocks]
            self.assertEqual(texts.count(first), 1)
            self.assertEqual(texts.count(second), 1)
            first_block = next(block for block in document.blocks if block.data.text == first)
            second_block = next(block for block in document.blocks if block.data.text == second)
            self.assertLess(first_block.data.bbox[3], 110)
            self.assertGreater(second_block.data.bbox[1], 90)

    def test_front_matter_author_names_are_preserved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            title = "The role of prompt cusps in dark matter halos"
            authors = "Alice Smith, Bob Jones, and Carol Lee"
            affiliation = (
                "Department of Physics, Example University, Cambridge, MA 02139, USA"
            )
            abstract = (
                "We study the evolution of dark matter halos with high-resolution "
                "simulations and compare them with analytic models."
            )
            source = root / "paper_enriched.json"
            source.write_text(
                json.dumps(
                    [
                        valid_block(
                            type="text",
                            sub_type="doc_title",
                            text=title,
                            bbox=[100, 40, 500, 60],
                            logical_para_id=0,
                        ),
                        valid_block(
                            text=authors,
                            bbox=[150, 80, 450, 100],
                            logical_para_id=1,
                        ),
                        valid_block(
                            text=affiliation,
                            bbox=[100, 110, 500, 130],
                            logical_para_id=2,
                        ),
                        valid_block(
                            type="text",
                            sub_type="abstract",
                            text=abstract,
                            bbox=[80, 150, 520, 240],
                            logical_para_id=3,
                        ),
                    ]
                ),
                encoding="utf-8",
            )
            pdf = fitz.open()
            page = pdf.new_page(width=600, height=800)
            for rect, text in (
                ((100, 40, 500, 60), title),
                ((150, 80, 450, 100), authors),
                ((100, 110, 500, 130), affiliation),
                ((80, 150, 520, 240), abstract),
            ):
                remaining = page.insert_textbox(fitz.Rect(*rect), text, fontsize=9)
                self.assertGreaterEqual(remaining, 0)
            pdf.save(root / "paper_origin.pdf")
            pdf.close()

            document = load_document(DiscoveredInput(source.resolve(), "test"))
            author_block = next(block for block in document.blocks if block.data.text == authors)
            affiliation_block = next(
                block for block in document.blocks if block.data.text == affiliation
            )
            self.assertEqual(author_block.taxonomy.element_type, ElementType.AUTHOR)
            self.assertEqual(author_block.taxonomy.policy, TranslationPolicy.PRESERVE)
            self.assertEqual(
                affiliation_block.taxonomy.policy, TranslationPolicy.TRANSLATE_TEXT
            )
            plan = build_translation_plan(document)
            unit = next(unit for unit in plan.units if author_block.block_id in unit.block_ids)
            self.assertEqual(unit.policy, TranslationPolicy.PRESERVE)

    def test_body_text_missing_from_enriched_json_is_recovered_from_pdf(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = "The first body paragraph is present in the enriched parse result."
            missing = "A second body paragraph was omitted by parsing and must be recovered."
            source = root / "paper_enriched.json"
            source.write_text(
                json.dumps([valid_block(text=first, bbox=[20, 60, 280, 120])]),
                encoding="utf-8",
            )
            pdf = fitz.open()
            page = pdf.new_page(width=300, height=400)
            page.insert_textbox(fitz.Rect(20, 60, 280, 120), first, fontsize=9)
            page.insert_textbox(fitz.Rect(20, 150, 280, 210), missing, fontsize=9)
            pdf.save(root / "paper_origin.pdf")
            pdf.close()

            document = load_document(DiscoveredInput(source.resolve(), "test"))
            texts = {block.data.text for block in document.blocks}
            self.assertIn(missing, texts)
            recovered = next(block for block in document.blocks if block.data.text == missing)
            self.assertTrue(recovered.raw.get("recovered_from_pdf"))
            self.assertGreaterEqual(recovered.data.bbox[1], 140)

    def test_missing_logical_paragraph_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            block = valid_block()
            del block["logical_para_id"]
            with self.assertRaisesRegex(InputValidationError, "logical_para_id"):
                self.load(Path(temp_dir), [block])

    def test_duplicate_provided_block_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            blocks = [
                valid_block(block_id="same"),
                valid_block(block_id="same", bbox=[10, 40, 100, 60], logical_para_id=1),
            ]
            with self.assertRaisesRegex(InputValidationError, "duplicate block_id"):
                self.load(Path(temp_dir), blocks)

    def test_page_out_of_range_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(InputValidationError, "outside PDF page range"):
                self.load(Path(temp_dir), [valid_block(page_idx=1)], pages=1)

    def test_bbox_out_of_page_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(InputValidationError, "outside page"):
                self.load(Path(temp_dir), [valid_block(bbox=[10, 10, 250, 30])])

    def test_batch_accepts_distinct_documents(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inputs = []
            for name in ("paper_a", "paper_b"):
                directory = root / name
                directory.mkdir()
                (directory / f"{name}_enriched.json").write_text(
                    json.dumps([valid_block()]), encoding="utf-8"
                )
                write_pdf(directory / f"{name}_origin.pdf")
                inputs.append(directory)
            documents = load_documents(inputs)
            self.assertEqual([doc.document_id for doc in documents], ["paper_a", "paper_b"])

    def test_batch_rejects_duplicate_document_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inputs = []
            for folder in ("first", "second"):
                directory = root / folder
                directory.mkdir()
                (directory / "paper_enriched.json").write_text(
                    json.dumps([valid_block()]), encoding="utf-8"
                )
                write_pdf(directory / "paper_origin.pdf")
                inputs.append(directory)
            with self.assertRaisesRegex(InputDiscoveryError, "duplicate document_id"):
                load_documents(inputs)


if __name__ == "__main__":
    unittest.main()
