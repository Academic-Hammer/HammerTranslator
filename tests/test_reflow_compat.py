from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import fitz

from hammer_translator.reflow.renderer import (
    ReflowContractError,
    RenderJob,
    ThinReflowRenderer,
)
from hammer_translator.reflow.table import TablePlan
from hammer_translator.translation.artifact import build_translation_artifact
from hammer_translator.translation.cache import TranslationCache
from hammer_translator.translation.jsonio import write_json_atomic
from hammer_translator.translation.layout import LayoutMetrics
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
    def test_overlapping_text_jobs_are_reported_and_not_drawn(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _, artifact, _ = _artifact(root)
            first = artifact["items"][0]
            second = copy.deepcopy(first)
            second["block_id"] = "p0001-b0002"
            artifact["items"] = [first, second]

            renderer = ThinReflowRenderer()
            jobs, feedback = renderer._preflight(
                artifact, LayoutMetrics(artifact["golden_style_profile"])
            )

            self.assertEqual(jobs, [])
            for item in feedback:
                self.assertEqual(item["status"], "needs_review")
                self.assertEqual(item["reason"], "overlapping_text_bboxes")
                self.assertEqual(item["action"], "manual_review")
                self.assertEqual(
                    item["measurement"]["overlapping_block_ids"],
                    ["p0001-b0002"]
                    if item["block_id"] == "p0001-b0001"
                    else ["p0001-b0001"],
                )

    def test_text_overlap_guard_ignores_small_vertical_contact(self):
        def job(block_id: str, bbox: tuple[float, float, float, float]) -> RenderJob:
            return RenderJob(
                item_index=0,
                block_id=block_id,
                unit_id="tu",
                page_idx=0,
                bbox=bbox,
                text="text",
                role="body",
                style={},
                measurement={},
                color=(0.0, 0.0, 0.0),
            )

        renderer = ThinReflowRenderer()
        small = renderer._overlapping_job_block_ids(
            [job("a", (0, 0, 100, 100)), job("b", (0, 60, 100, 160))]
        )
        large = renderer._overlapping_job_block_ids(
            [job("a", (0, 0, 100, 100)), job("b", (0, 40, 100, 140))]
        )

        self.assertEqual(small, {})
        self.assertEqual(large, {"a": {"b"}, "b": {"a"}})

    def test_overlapping_merged_block_suppresses_duplicate_nested_blocks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _, artifact, _ = _artifact(root)
            base = artifact["items"][0]
            words = [
                "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
                "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho",
                "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega", "one", "two",
                "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
                "seventeen", "eighteen", "nineteen", "twenty",
            ]

            container = copy.deepcopy(base)
            container.update(
                block_id="p0001-b0004",
                type="figure_caption",
                detected_type="figure_caption",
                text=" ".join(words),
            )
            duplicate = copy.deepcopy(base)
            duplicate.update(
                block_id="p0001-b0005",
                bbox=[24, 24, 100, 40],
                text=" ".join(words[:24]),
            )
            unrelated = copy.deepcopy(base)
            unrelated.update(
                block_id="p0001-b0006",
                bbox=[110, 24, 200, 40],
                text=(
                    "quasar nebula parsec redshift spectroscopy photometry interferometry "
                    "exoplanet magnetohydrodynamics asteroseismology helioseismology "
                    "chromosphere corona prominence filament reconnection turbulence "
                    "convection granulation facula"
                ),
            )
            artifact["items"] = [container, duplicate, unrelated]

            renderer = ThinReflowRenderer()
            groups = renderer._overlapping_container_groups(artifact)
            self.assertEqual(groups, {"p0001-b0004": frozenset({"p0001-b0005"})})

            jobs, feedback = renderer._preflight(
                artifact, LayoutMetrics(artifact["golden_style_profile"])
            )
            by_block = {item["block_id"]: item for item in feedback}
            self.assertEqual([job.block_id for job in jobs], ["p0001-b0004"])
            self.assertEqual(
                by_block["p0001-b0004"]["verification"]["merged_duplicate_block_ids"],
                ["p0001-b0005"],
            )
            self.assertEqual(by_block["p0001-b0005"]["status"], "skipped")
            self.assertEqual(
                by_block["p0001-b0005"]["reason"],
                "merged_into_overlapping_container_block",
            )
            self.assertEqual(by_block["p0001-b0005"]["action"], "none")

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

    def test_table_structure_error_is_visible_and_preserves_the_original(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _, artifact, _ = _artifact(root)
            renderer = ThinReflowRenderer()
            block_id = artifact["items"][0]["block_id"]
            table_plan = TablePlan(
                (),
                frozenset(),
                {block_id: "table_structure_unavailable"},
            )

            jobs, feedback = renderer._preflight(
                artifact,
                LayoutMetrics(artifact["golden_style_profile"]),
                table_plan=table_plan,
            )

            self.assertEqual(jobs, [])
            self.assertEqual(feedback[0]["status"], "needs_review")
            self.assertEqual(feedback[0]["action"], "preserve_original")
            self.assertEqual(
                feedback[0]["reason"],
                "table_reflow_unavailable:table_structure_unavailable",
            )
            self.assertFalse(renderer._has_blocking_feedback(feedback))

    def test_failed_inline_formula_asset_only_degrades_its_block(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _, artifact, _ = _artifact(root)
            artifact["items"][0]["formula_ids"] = ["fm-inline", "fm-display"]
            artifact["formula_contract"] = {
                "inventory": [
                    {
                        "formula_id": "fm-inline",
                        "kind": "inline",
                        "render_policy": "move_with_text",
                    },
                    {
                        "formula_id": "fm-display",
                        "kind": "display",
                        "render_policy": "preserve_in_place",
                    },
                ]
            }
            artifact["formula_assets"] = {
                "assets": [
                    {"formula_id": "fm-inline", "status": "failed"},
                    {"formula_id": "fm-display", "status": "failed"},
                ]
            }
            renderer = ThinReflowRenderer()

            unavailable = renderer._unavailable_movable_formula_ids(
                artifact, artifact["items"][0]
            )
            _, feedback = renderer._preflight(
                artifact,
                LayoutMetrics(artifact["golden_style_profile"]),
            )

            self.assertEqual(unavailable, ("fm-inline",))
            self.assertEqual(feedback[0]["status"], "needs_review")
            self.assertEqual(feedback[0]["action"], "preserve_original")
            self.assertEqual(
                feedback[0]["reason"], "formula_asset_unavailable:fm-inline"
            )
            self.assertFalse(renderer._has_blocking_feedback(feedback))

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
