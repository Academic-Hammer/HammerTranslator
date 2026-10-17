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


def write_pdf(path: Path, pages: int = 1, width: float = 200, height: float = 300) -> None:
    document = fitz.open()
    for _ in range(pages):
        document.new_page(width=width, height=height)
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
