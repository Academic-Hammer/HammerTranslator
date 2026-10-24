from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import fitz

from hammer_translator.translation.artifact import build_translation_artifact
from hammer_translator.translation.cache import TranslationCache
from hammer_translator.translation.formula_assets import (
    FormulaAssetError,
    MathJaxSvgRenderer,
    _repair_latex,
    _source_probe,
    build_formula_asset_contract,
)
from hammer_translator.translation.formula import FormulaBinding
from hammer_translator.translation.loader import DiscoveredInput, load_document
from hammer_translator.translation.output_schema import FormulaAssetContractOutput
from hammer_translator.translation.provider import FakeProvider
from hammer_translator.translation.runner import RunnerConfig, execute_translation_plan
from hammer_translator.translation.units import build_translation_plan


def _plan(root: Path, *, with_formula_span: bool, formula_latex: str = "x^2"):
    bbox = [20, 20, 220, 55]
    spans = [{"type": "text", "content": "Value x^2", "bbox": [20, 20, 90, 40]}]
    if with_formula_span:
        spans = [
            {"type": "text", "content": "Value", "bbox": [20, 20, 60, 40]},
            {
                "type": "inline_equation",
                "content": formula_latex,
                "bbox": [64, 20, 90, 40],
                "score": 0.95,
            },
        ]
    source_json = root / "paper_enriched.json"
    source_json.write_text(
        json.dumps(
            [
                {
                    "type": "text",
                    "text": f"Value ${formula_latex}$",
                    "page_idx": 0,
                    "bbox": bbox,
                    "logical_para_id": 1,
                    "lines": [{"bbox": bbox, "spans": spans}],
                    "rich_spans": [
                        {
                            "text": f"Value {formula_latex}",
                            "size": 12,
                            "font": "Helvetica",
                            "flags": 0,
                            "color": 0,
                            "bbox": [20, 20, 90, 40],
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    source_pdf = root / "paper_origin.pdf"
    document = fitz.open()
    page = document.new_page(width=250, height=180)
    page.insert_text((20, 36), "Value", fontsize=12)
    page.insert_text((64, 36), "x^2", fontsize=12)
    document.save(source_pdf)
    document.close()
    loaded = load_document(DiscoveredInput(source_json.resolve(), "test"))
    return build_translation_plan(loaded)


class UnsafeSvgRenderer(MathJaxSvgRenderer):
    @property
    def available(self) -> bool:
        return True

    @property
    def version(self) -> str | None:
        return "test"

    def render_batch(self, expressions):
        return (
            {
                expression.formula_id: (
                    '<svg xmlns="http://www.w3.org/2000/svg" width="1ex" height="1ex" '
                    'viewBox="0 0 1 1"><script>alert(1)</script></svg>'
                )
                for expression in expressions
            },
            {},
        )


VALID_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="2ex" height="1ex" '
    'viewBox="0 0 2 1"><path d="M0 0L1 1"/></svg>'
)


class RecordingSvgRenderer(MathJaxSvgRenderer):
    def __init__(self) -> None:
        super().__init__()
        self.latex: list[str] = []

    @property
    def available(self) -> bool:
        return True

    @property
    def version(self) -> str | None:
        return "test"

    def render_batch(self, expressions):
        self.latex = [expression.latex for expression in expressions]
        return ({expression.formula_id: VALID_SVG for expression in expressions}, {})


class ErrorSvgRenderer(MathJaxSvgRenderer):
    @property
    def available(self) -> bool:
        return True

    @property
    def version(self) -> str | None:
        return "test"

    def render_batch(self, expressions):
        return (
            {
                expression.formula_id: (
                    '<svg xmlns="http://www.w3.org/2000/svg" width="2ex" height="1ex" '
                    'viewBox="0 0 2 1"><g data-mml-node="merror" '
                    'data-mjx-error="Missing close brace"><rect width="2" height="1"/></g></svg>'
                )
                for expression in expressions
            },
            {},
        )


class FormulaAssetTests(unittest.TestCase):
    def test_accent_command_swallowing_a_delimiter_is_repaired(self):
        corrupt = (
            r"\{ \check { \alpha } _ { 0 } ^ { k , p } "
            r"\check { \} ,   \{ \check { \beta } _ { 0 } ^ { k , p } \}"
        )

        repaired = _repair_latex(corrupt)

        self.assertEqual(
            repaired,
            r"\{ \check { \alpha } _ { 0 } ^ { k , p } \},   "
            r"\{ \check { \beta } _ { 0 } ^ { k , p } \}",
        )

    def test_repaired_latex_is_what_reaches_mathjax(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            corrupt = r"\{ \check { \alpha } _ { 0 } ^ { k , p } \check { \} ,   \{ \check { \beta } _ { 0 } ^ { k , p } \}"
            plan = _plan(root, with_formula_span=False, formula_latex=corrupt)
            renderer = RecordingSvgRenderer()

            contract = build_formula_asset_contract(
                plan,
                root / "paper_translated.json",
                renderer=renderer,
            )

            self.assertEqual(len(renderer.latex), 1)
            self.assertNotIn(r"\check { \}", renderer.latex[0])
            asset = FormulaAssetContractOutput.model_validate(contract).assets[0]
            self.assertEqual(asset.status, "ready")

    def test_mathjax_error_svg_is_never_marked_ready(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = _plan(root, with_formula_span=False)

            contract = build_formula_asset_contract(
                plan,
                root / "paper_translated.json",
                renderer=ErrorSvgRenderer(),
            )

            asset = FormulaAssetContractOutput.model_validate(contract).assets[0]
            self.assertEqual(asset.status, "failed")
            self.assertIn("mathjax_svg_rendering_error", str(asset.failure_code))

    def test_refinement_that_cannot_change_bbox_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            document = fitz.open()
            page = document.new_page(width=250, height=180)
            page.insert_text((92, 27), "g", fontsize=10)
            page.insert_text((54, 36), "x2", fontsize=12)
            binding = FormulaBinding(
                formula_id="fm-test",
                kind="inline",
                translation_unit_id="tu-test",
                block_id="p0001-b0001",
                page_idx=0,
                latex="x^2",
                latex_hash="test",
                source_bbox=(52, 26, 112, 38),
                source_line_bbox=(20, 20, 200, 50),
                source_block_bbox=(20, 20, 200, 50),
                source_line_index=0,
                source_span_index=0,
                source_span_path="blocks/p0001-b0001/lines/0/spans/0",
                recognition_confidence=None,
                geometry_flags=(),
                protected_placeholder=None,
                source_text_start=None,
                source_text_end=None,
                binding_status="bound_exact",
                render_policy="move_with_text",
                asset_preference="source_pdf_then_mathjax",
            )
            with self.assertRaisesRegex(
                FormulaAssetError, "source_clip_refinement_unchanged"
            ):
                _source_probe(page, binding)
            document.close()

    def test_source_pdf_formula_becomes_verified_vector_clip_reference(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = _plan(root, with_formula_span=True)
            contract = build_formula_asset_contract(plan, root / "paper_translated.json")
            validated = FormulaAssetContractOutput.model_validate(contract)
            self.assertEqual(validated.stats.source_pdf_clip, 1)
            self.assertEqual(validated.stats.mathjax_svg, 0)
            asset = validated.assets[0]
            self.assertEqual(asset.status, "ready")
            self.assertEqual(asset.kind, "source_pdf_clip")
            self.assertEqual(asset.storage, "source_pdf_reference")
            self.assertIsNone(asset.relative_path)
            self.assertGreater(asset.source_pdf_clip.text_span_count, 0)
            self.assertEqual(asset.source_pdf_clip.raster_image_count, 0)
            self.assertIn(asset.source_pdf_clip.text_match, {"exact", "refined"})
            self.assertEqual(asset.metrics.coordinate_system, "pdf_points")
            self.assertAlmostEqual(asset.metrics.width, 26.0)
            self.assertAlmostEqual(asset.metrics.height, 20.0)

    def test_missing_source_span_generates_self_contained_mathjax_svg(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = _plan(root, with_formula_span=False)
            contract = build_formula_asset_contract(plan, root / "paper_translated.json")
            validated = FormulaAssetContractOutput.model_validate(contract)
            self.assertEqual(validated.stats.mathjax_svg, 1)
            asset = validated.assets[0]
            self.assertEqual(asset.kind, "mathjax_svg")
            svg_path = root / asset.relative_path
            self.assertTrue(svg_path.is_file())
            svg = svg_path.read_text(encoding="utf-8")
            self.assertIn("<svg", svg)
            self.assertIn("<path", svg)
            self.assertNotIn("<script", svg)
            self.assertEqual(
                hashlib.sha256(svg_path.read_bytes()).hexdigest(),
                asset.sha256,
            )
            self.assertEqual(asset.metrics.coordinate_system, "mathjax_ex")
            self.assertGreater(asset.metrics.baseline_from_top, 0)

    def test_unsafe_mathjax_svg_is_rejected_as_failed_asset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = _plan(root, with_formula_span=False)
            contract = build_formula_asset_contract(
                plan,
                root / "paper_translated.json",
                renderer=UnsafeSvgRenderer(),
            )
            asset = FormulaAssetContractOutput.model_validate(contract).assets[0]
            self.assertEqual(asset.status, "failed")
            self.assertEqual(asset.failure_code, "mathjax_svg_contains_script")
            self.assertFalse((root / "paper_formula_assets" / f"{asset.formula_id}.svg").exists())

    def test_translation_v3_links_inventory_to_resolved_asset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = _plan(root, with_formula_span=True)
            provider = FakeProvider()
            config = RunnerConfig()
            result = execute_translation_plan(
                plan,
                provider=provider,
                cache=TranslationCache(root / "cache.sqlite3"),
                checkpoint_path=root / "manifest.json",
                config=config,
            )
            contract = build_formula_asset_contract(plan, root / "paper_translated.json")
            artifact = build_translation_artifact(
                plan,
                provider=provider,
                requested_provider="fake",
                result=result,
                config=config,
                formula_asset_contract=contract,
            )
            self.assertEqual(artifact["schema_version"], "translation-v3")
            formula = artifact["formula_contract"]["inventory"][0]
            asset = artifact["formula_assets"]["assets"][0]
            self.assertEqual(formula["asset_id"], asset["asset_id"])
            self.assertEqual(formula["asset_status"], "ready")


if __name__ == "__main__":
    unittest.main()
