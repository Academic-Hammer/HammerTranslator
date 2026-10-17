from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import fitz

from hammer_translator.translation.artifact import build_translation_artifact
from hammer_translator.translation.cache import TranslationCache
from hammer_translator.translation.formula import FormulaContractError
from hammer_translator.translation.formula_assets import build_formula_asset_contract
from hammer_translator.translation.loader import DiscoveredInput, load_document
from hammer_translator.translation.output_schema import TranslationArtifactOutput
from hammer_translator.translation.provider import FakeProvider
from hammer_translator.translation.runner import RunnerConfig, execute_translation_plan
from hammer_translator.translation.taxonomy import TranslationPolicy
from hammer_translator.translation.units import build_translation_plan


def _load_plan(root: Path, blocks: list[dict]):
    source = root / "paper_enriched.json"
    source.write_text(json.dumps(blocks), encoding="utf-8")
    pdf = fitz.open()
    pdf.new_page(width=300, height=400)
    pdf.save(root / "paper_origin.pdf")
    pdf.close()
    document = load_document(DiscoveredInput(source.resolve(), "test"))
    return build_translation_plan(document)


def _inline_block(*, formula_bbox: list[float] | None = None, text: str | None = None) -> dict:
    formula_bbox = formula_bbox or [100, 12, 135, 28]
    text = text if text is not None else "Accuracy is $x_i^2$."
    return {
        "type": "text",
        "text": text,
        "page_idx": 0,
        "bbox": [10, 10, 280, 50],
        "logical_para_id": 1,
        "lines": [
            {
                "bbox": [10, 10, 280, 30],
                "spans": [
                    {"type": "text", "content": "Accuracy is", "bbox": [10, 10, 98, 30]},
                    {
                        "type": "inline_equation",
                        "content": "x_i^2",
                        "bbox": formula_bbox,
                        "score": 0.93,
                    },
                    {"type": "text", "content": ".", "bbox": [136, 10, 140, 30]},
                ],
            }
        ],
        "rich_spans": [
            {
                "text": "Accuracy is x_i^2.",
                "size": 10,
                "font": "Times-Roman",
                "flags": 4,
                "color": 0,
                "bbox": [10, 10, 140, 30],
            }
        ],
    }


class FormulaInventoryTests(unittest.TestCase):
    def test_inline_formula_uses_its_own_bbox_and_binds_to_protected_token(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = _load_plan(root, [_inline_block()])
            second = _load_plan(root, [_inline_block()])
            binding = first.units[0].formula_bindings[0]
            self.assertEqual(binding.kind, "inline")
            self.assertEqual(binding.binding_status, "bound_exact")
            self.assertEqual(binding.source_bbox, (100.0, 12.0, 135.0, 28.0))
            self.assertEqual(binding.source_line_bbox, (10.0, 10.0, 280.0, 30.0))
            self.assertEqual(binding.source_block_bbox, (10.0, 10.0, 280.0, 50.0))
            self.assertEqual(binding.latex, "x_i^2")
            self.assertEqual(binding.recognition_confidence, 0.93)
            self.assertEqual(
                binding.formula_id,
                second.units[0].formula_bindings[0].formula_id,
            )
            self.assertEqual(
                binding.protected_placeholder,
                first.units[0].protected_tokens[0].placeholder,
            )

    def test_formula_bbox_outside_line_is_rejected_before_translation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(FormulaContractError, "outside its line bbox"):
                _load_plan(Path(temp_dir), [_inline_block(formula_bbox=[100, 12, 135, 45])])

    def test_small_mineru_geometry_disagreement_is_retained_as_provenance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            block = _inline_block(formula_bbox=[100, 12, 135, 28])
            block["bbox"] = [11, 10, 134, 29]
            binding = _load_plan(Path(temp_dir), [block]).units[0].formula_bindings[0]
            self.assertEqual(
                binding.geometry_flags,
                (
                    "line_bbox_outside_block",
                    "formula_bbox_outside_block_tolerance_used",
                ),
            )

    def test_two_point_accent_overflow_is_accepted_as_mineru_rounding(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            block = _inline_block(formula_bbox=[100, 8, 135, 28])
            block["bbox"] = [10, 10, 280, 50]
            block["lines"][0]["bbox"] = [10, 8, 280, 30]
            binding = _load_plan(Path(temp_dir), [block]).units[0].formula_bindings[0]
            self.assertEqual(
                binding.geometry_flags,
                (
                    "line_bbox_outside_block",
                    "formula_bbox_outside_block_tolerance_used",
                ),
            )

    def test_geometry_disagreement_beyond_three_points_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            block = _inline_block(formula_bbox=[100, 6, 135, 28])
            block["lines"][0]["bbox"] = [10, 6, 280, 30]
            with self.assertRaisesRegex(FormulaContractError, "outside its block bbox"):
                _load_plan(Path(temp_dir), [block])

    def test_formula_without_mineru_span_is_reserved_for_mathjax(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            block = _inline_block(text="Accuracy is $x_i^2$.")
            block["lines"][0]["spans"] = [
                {"type": "text", "content": "Accuracy is x_i^2.", "bbox": [10, 10, 140, 30]}
            ]
            plan = _load_plan(Path(temp_dir), [block])
            binding = plan.units[0].formula_bindings[0]
            self.assertEqual(binding.binding_status, "protected_token_only")
            self.assertEqual(binding.asset_preference, "mathjax_svg")
            self.assertIsNone(binding.source_bbox)
            self.assertIn("formula_source_span_missing", plan.units[0].quality_flags)

    def test_unprotected_mineru_formula_forces_manual_review(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            plan = _load_plan(Path(temp_dir), [_inline_block(text="Accuracy is x_i^2.")])
            binding = plan.units[0].formula_bindings[0]
            self.assertEqual(binding.binding_status, "source_span_only")
            self.assertEqual(binding.render_policy, "needs_review")
            self.assertEqual(plan.units[0].policy, TranslationPolicy.NEEDS_REVIEW)

    def test_display_formula_is_inventoried_and_preserved_in_place(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            block = {
                "type": "equation",
                "text": "",
                "page_idx": 0,
                "bbox": [40, 80, 260, 130],
                "logical_para_id": 2,
                "lines": [
                    {
                        "bbox": [40, 80, 260, 130],
                        "spans": [
                            {
                                "type": "interline_equation",
                                "content": r"E=mc^2\tag{1}",
                                "bbox": [40, 80, 260, 130],
                                "score": 0.88,
                            }
                        ],
                    }
                ],
            }
            plan = _load_plan(Path(temp_dir), [block])
            binding = plan.units[0].formula_bindings[0]
            self.assertEqual(binding.kind, "display")
            self.assertEqual(binding.binding_status, "preserved_display")
            self.assertEqual(binding.render_policy, "preserve_in_place")
            self.assertEqual(binding.asset_preference, "source_pdf")
            self.assertEqual(plan.units[0].policy, TranslationPolicy.PRESERVE)

    def test_display_text_token_and_mineru_span_are_one_formula(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            block = {
                "type": "equation",
                "text": "$$E=mc^2$$",
                "page_idx": 0,
                "bbox": [40, 80, 260, 130],
                "logical_para_id": 2,
                "lines": [
                    {
                        "bbox": [40, 80, 260, 130],
                        "spans": [
                            {
                                "type": "interline_equation",
                                "content": "E=mc^2",
                                "bbox": [40, 80, 260, 130],
                            }
                        ],
                    }
                ],
            }
            plan = _load_plan(Path(temp_dir), [block])
            self.assertEqual(len(plan.units[0].formula_bindings), 1)
            binding = plan.units[0].formula_bindings[0]
            self.assertEqual(binding.binding_status, "preserved_display")
            self.assertIsNotNone(binding.protected_placeholder)
            self.assertIsNotNone(binding.source_bbox)


class FormulaArtifactTests(unittest.TestCase):
    def test_translation_v3_keeps_preserved_display_formula_on_source_item(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = _load_plan(
                root,
                [
                    {
                        "type": "equation",
                        "text": "$$E=mc^2$$",
                        "page_idx": 0,
                        "bbox": [40, 80, 260, 130],
                        "logical_para_id": 2,
                        "lines": [
                            {
                                "bbox": [40, 80, 260, 130],
                                "spans": [
                                    {
                                        "type": "interline_equation",
                                        "content": "E=mc^2",
                                        "bbox": [40, 80, 260, 130],
                                        "score": 0.88,
                                    }
                                ],
                            }
                        ],
                    }
                ],
            )
            provider = FakeProvider()
            config = RunnerConfig()
            result = execute_translation_plan(
                plan,
                provider=provider,
                cache=TranslationCache(root / "cache.sqlite3"),
                checkpoint_path=root / "manifest.json",
                config=config,
            )
            output_path = root / "paper_translated.json"
            assets = build_formula_asset_contract(plan, output_path)
            artifact = build_translation_artifact(
                plan,
                provider=provider,
                requested_provider="fake",
                result=result,
                config=config,
                formula_asset_contract=assets,
            )

            formula = artifact["formula_contract"]["inventory"][0]
            self.assertEqual(artifact["schema_version"], "translation-v3")
            self.assertEqual(formula["render_policy"], "preserve_in_place")
            self.assertEqual(artifact["items"][0]["formula_ids"], [formula["formula_id"]])

    def test_translation_v2_links_formula_inventory_unit_token_and_item(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = _load_plan(root, [_inline_block()])
            provider = FakeProvider()
            config = RunnerConfig()
            result = execute_translation_plan(
                plan,
                provider=provider,
                cache=TranslationCache(root / "cache.sqlite3"),
                checkpoint_path=root / "manifest.json",
                config=config,
            )
            artifact = build_translation_artifact(
                plan,
                provider=provider,
                requested_provider="fake",
                result=result,
                config=config,
            )
            self.assertEqual(artifact["schema_version"], "translation-v2")
            self.assertEqual(artifact["layout_run"]["contract_version"], "translation-reflow-v2")
            contract = artifact["formula_contract"]
            self.assertEqual(contract["schema_version"], "formula-contract-v1")
            self.assertEqual(contract["stats"]["total"], 1)
            formula_id = contract["inventory"][0]["formula_id"]
            self.assertEqual(artifact["document"]["formula_count"], 1)
            self.assertEqual(artifact["translation_units"][0]["formula_ids"], [formula_id])
            self.assertEqual(artifact["items"][0]["formula_ids"], [formula_id])
            formula_tokens = [
                token
                for token in artifact["translation_units"][0]["translation"]["protected_tokens"]
                if token["formula_id"] is not None
            ]
            self.assertEqual([token["formula_id"] for token in formula_tokens], [formula_id])

    def test_legacy_translation_v1_artifact_remains_readable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = _load_plan(
                root,
                [
                    {
                        **_inline_block(text="Plain text"),
                        "lines": [
                            {
                                "bbox": [10, 10, 280, 30],
                                "spans": [
                                    {"type": "text", "content": "Plain text", "bbox": [10, 10, 80, 30]}
                                ],
                            }
                        ],
                    }
                ],
            )
            provider = FakeProvider()
            config = RunnerConfig()
            result = execute_translation_plan(
                plan,
                provider=provider,
                cache=TranslationCache(root / "cache.sqlite3"),
                checkpoint_path=root / "manifest.json",
                config=config,
            )
            artifact = build_translation_artifact(
                plan,
                provider=provider,
                requested_provider="fake",
                result=result,
                config=config,
            )
            legacy = deepcopy(artifact)
            legacy["schema_version"] = "translation-v1"
            legacy.pop("formula_contract")
            legacy["document"].pop("formula_count")
            for unit in legacy["translation_units"]:
                unit.pop("formula_ids")
                for token in unit["translation"]["protected_tokens"]:
                    token.pop("formula_id", None)
            for item in legacy["items"]:
                item.pop("formula_ids", None)
            validated = TranslationArtifactOutput.model_validate(legacy)
            self.assertEqual(validated.schema_version, "translation-v1")


if __name__ == "__main__":
    unittest.main()
