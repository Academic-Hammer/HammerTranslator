from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import fitz

from hammer_translator.reflow.renderer import ReflowContractError, ThinReflowRenderer
from hammer_translator.translation.artifact import build_translation_artifact
from hammer_translator.translation.cache import TranslationCache
from hammer_translator.translation.jsonio import write_json_atomic
from hammer_translator.translation.loader import DiscoveredInput, load_document
from hammer_translator.translation.provider import FakeProvider
from hammer_translator.translation.runner import RunnerConfig, execute_translation_plan
from hammer_translator.translation.units import build_translation_plan


def _artifact(root: Path, *, bbox: list[float] | None = None, source_text: str = "Source text"):
    bbox = bbox or [20, 20, 220, 55]
    source_json = root / "paper_enriched.json"
    source_json.write_text(
        json.dumps(
            [
                {
                    "type": "text",
                    "text": source_text,
                    "page_idx": 0,
                    "bbox": bbox,
                    "logical_para_id": 1,
                    "rich_spans": [
                        {
                            "text": source_text,
                            "size": 10,
                            "font": "Times-Roman",
                            "flags": 4,
                            "color": 0,
                            "bbox": bbox,
                        }
                    ],
                    "lines": [{"bbox": bbox}],
                }
            ]
        ),
        encoding="utf-8",
    )
    source_pdf = root / "paper_origin.pdf"
    document = fitz.open()
    page = document.new_page(width=250, height=180)
    page.insert_text((bbox[0], bbox[1] + 12), source_text, fontsize=10)
    document.save(source_pdf)
    document.close()
    plan = build_translation_plan(load_document(DiscoveredInput(source_json.resolve(), "test")))
    provider = FakeProvider(transform=lambda request: "测试译文")
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
    artifact_path = write_json_atomic(root / "paper_translated.json", artifact)
    return artifact_path, artifact, source_pdf


class ThinReflowTests(unittest.TestCase):
    def test_ready_artifact_renders_without_rewriting_or_page_geometry_change(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact_path, artifact, source_pdf = _artifact(root)
            output = root / "rendered.pdf"
            report = ThinReflowRenderer().render(artifact_path, output)
            self.assertEqual(report["status"], "rendered")
            self.assertEqual(report["summary"]["rendered"], 1)
            self.assertTrue(output.is_file())
            self.assertEqual(artifact["items"][0]["translated"], "测试译文")

            source = fitz.open(source_pdf)
            rendered = fitz.open(output)
            try:
                self.assertEqual(rendered.page_count, source.page_count)
                self.assertEqual(rendered[0].rect, source[0].rect)
                clipped = rendered[0].get_text("text", clip=fitz.Rect([20, 20, 220, 55]))
                self.assertIn("测试译文", clipped.replace("\n", ""))
                self.assertNotIn("Source text", clipped)
            finally:
                rendered.close()
                source.close()

    def test_blocking_item_refuses_output_without_allow_partial(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact_path, artifact, _ = _artifact(root)
            artifact["items"][0].pop("translated")
            artifact["items"][0].pop("recommended_style")
            artifact["items"][0].pop("layout_preflight")
            artifact["translation_units"][0]["layout"]["segments"] = []
            artifact["translation_units"][0]["layout"]["status"] = "needs_review"
            artifact["translation_units"][0]["layout"]["all_blocks_fit"] = False
            artifact["translation_units"][0]["translation"]["status"] = "needs_review"
            artifact["translation_units"][0]["translation"]["initial_text"] = None
            artifact["translation_units"][0]["translation"]["selected_text"] = None
            artifact["translation_units"][0]["translation"]["selected_version"] = None
            artifact["items"][0]["translation_status"] = "needs_review"
            artifact["layout_status"] = "needs_review"
            write_json_atomic(artifact_path, artifact)
            output = root / "blocked.pdf"
            report = ThinReflowRenderer().render(artifact_path, output)
            self.assertEqual(report["status"], "blocked")
            self.assertEqual(report["summary"]["needs_review"], 1)
            self.assertFalse(output.exists())

    def test_tampered_preflight_is_rejected_before_pdf_write(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact_path, artifact, _ = _artifact(root)
            artifact["items"][0]["layout_preflight"]["measurement"]["line_count"] += 1
            write_json_atomic(artifact_path, artifact)
            output = root / "tampered.pdf"
            report = ThinReflowRenderer().render(artifact_path, output)
            self.assertEqual(report["status"], "blocked")
            self.assertEqual(report["summary"]["failed"], 1)
            self.assertFalse(output.exists())

    def test_inline_formula_is_preserved_for_manual_review(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact_path, artifact, _ = _artifact(root)
            unit = artifact["translation_units"][0]
            item = artifact["items"][0]
            unit["translation"]["initial_text"] = "结果 $x+y$"
            unit["translation"]["selected_text"] = "结果 $x+y$"
            item["translated"] = "结果 $x+y$"
            unit["layout"]["segments"][0]["translated"] = "结果 $x+y$"
            write_json_atomic(artifact_path, artifact)
            output = root / "formula.pdf"
            report = ThinReflowRenderer().render(artifact_path, output)
            self.assertEqual(report["status"], "blocked")
            self.assertEqual(
                report["items"][0]["reason"],
                "translation_and_reflow_measurements_do_not_match",
            )

    def test_source_pdf_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact_path, _, source_pdf = _artifact(root)
            with self.assertRaisesRegex(ReflowContractError, "must not overwrite"):
                ThinReflowRenderer().render(artifact_path, source_pdf)


if __name__ == "__main__":
    unittest.main()
