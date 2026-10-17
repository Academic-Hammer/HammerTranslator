from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import fitz

from hammer_translator.reflow.renderer import ReflowContractError, ThinReflowRenderer
from hammer_translator.translation.artifact import build_translation_artifact
from hammer_translator.translation.cache import TranslationCache
from hammer_translator.translation.formula_assets import build_formula_asset_contract
from hammer_translator.translation.jsonio import write_json_atomic
from hammer_translator.translation.layout import FormulaRunSpec, LayoutMetrics
from hammer_translator.translation.layout_planner import plan_translation_layout
from hammer_translator.translation.loader import DiscoveredInput, load_document
from hammer_translator.translation.provider import FakeProvider
from hammer_translator.translation.runner import RunnerConfig, execute_translation_plan
from hammer_translator.translation.style import build_golden_style_profile
from hammer_translator.translation.units import build_translation_plan


def _plan(root: Path, *, with_source_span: bool = False):
    text = "Value $x^2$ improves accuracy."
    bbox = [20, 20, 280, 70]
    source = root / "paper_enriched.json"
    lines = [{"bbox": [20, 25, 230, 45]}]
    if with_source_span:
        lines = [
            {
                "bbox": [20, 25, 230, 45],
                "spans": [
                    {"type": "text", "content": "Value", "bbox": [20, 26, 55, 42]},
                    {
                        "type": "inline_equation",
                        "content": "x^2",
                        "bbox": [69, 26, 91, 42],
                        "score": 0.99,
                    },
                    {
                        "type": "text",
                        "content": "improves accuracy.",
                        "bbox": [105, 26, 230, 42],
                    },
                ],
            }
        ]
    source.write_text(
        json.dumps(
            [
                {
                    "type": "text",
                    "text": text,
                    "page_idx": 0,
                    "bbox": bbox,
                    "logical_para_id": 1,
                    "rich_spans": [
                        {
                            "text": text,
                            "size": 10,
                            "font": "Times-Roman",
                            "flags": 4,
                            "color": 0,
                            "bbox": bbox,
                        }
                    ],
                    "lines": lines,
                }
            ]
        ),
        encoding="utf-8",
    )
    pdf = fitz.open()
    page = pdf.new_page(width=300, height=120)
    if with_source_span:
        page.insert_text((20, 38), "Value", fontsize=10)
        page.insert_text((70, 38), "x^2", fontsize=10)
        page.insert_text((105, 38), "improves accuracy.", fontsize=10)
    else:
        page.insert_text((20, 38), text, fontsize=10)
    pdf.save(root / "paper_origin.pdf")
    pdf.close()
    return build_translation_plan(load_document(DiscoveredInput(source.resolve(), "test")))


class MixedFormulaLayoutTests(unittest.TestCase):
    def test_formula_is_one_atomic_run_with_shared_baseline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            plan = _plan(Path(temp_dir))
            profile = build_golden_style_profile(plan)
            metrics = LayoutMetrics(profile)
            text = "结果 $x^2$ 继续"
            start = text.index("$x^2$")
            spec = FormulaRunSpec(
                formula_id="fm-test",
                asset_id="fa-fm-test",
                asset_kind="mathjax_svg",
                text="$x^2$",
                start=start,
                end=start + len("$x^2$"),
                coordinate_system="mathjax_ex",
                natural_width=2.4,
                natural_height=2.2,
                natural_baseline_from_top=1.6,
                natural_depth=0.6,
            )
            result = metrics.measure(
                text,
                [0, 0, 65, 80],
                role="body",
                formula_runs=(spec,),
            )
            formula_runs = [
                run
                for line in result.mixed_lines
                for run in line.runs
                if run.kind == "formula"
            ]
            self.assertEqual(len(formula_runs), 1)
            self.assertEqual(formula_runs[0].text, "$x^2$")
            self.assertEqual(formula_runs[0].formula_id, "fm-test")
            containing_line = next(
                line for line in result.mixed_lines if any(run.kind == "formula" for run in line.runs)
            )
            self.assertGreaterEqual(
                containing_line.baseline_from_top_pt,
                formula_runs[0].baseline_from_top_pt,
            )
            self.assertEqual(
                "".join(line.text for line in result.mixed_lines),
                "".join(result.lines),
            )

    def test_pdf_formula_scales_with_surrounding_text_size(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            plan = _plan(Path(temp_dir))
            profile = build_golden_style_profile(plan)
            metrics = LayoutMetrics(profile)
            text = "$x$"
            spec = FormulaRunSpec(
                formula_id="fm-pdf",
                asset_id="fa-fm-pdf",
                asset_kind="source_pdf_clip",
                text=text,
                start=0,
                end=len(text),
                coordinate_system="pdf_points",
                natural_width=20,
                natural_height=10,
                natural_baseline_from_top=7,
                natural_depth=3,
            )
            golden, guardrails = metrics.style_for_role("body")
            smaller = dict(golden)
            smaller["font_size_pt"] = guardrails["font_size_pt"]["min"]
            normal = metrics.measure(text, [0, 0, 100, 40], role="body", formula_runs=(spec,))
            compact = metrics.measure(
                text,
                [0, 0, 100, 40],
                role="body",
                style=smaller,
                formula_runs=(spec,),
            )
            normal_run = normal.mixed_lines[0].runs[0]
            compact_run = compact.mixed_lines[0].runs[0]
            self.assertEqual(normal_run.asset_kind, "source_pdf_clip")
            self.assertLess(compact_run.width_pt, normal_run.width_pt)
            self.assertAlmostEqual(
                compact_run.width_pt / normal_run.width_pt,
                smaller["font_size_pt"] / golden["font_size_pt"],
                places=3,
            )

    def test_tall_formula_expands_line_height_and_can_trigger_vertical_overflow(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            plan = _plan(Path(temp_dir))
            metrics = LayoutMetrics(build_golden_style_profile(plan))
            text = "$\\frac{a}{b}$"
            spec = FormulaRunSpec(
                formula_id="fm-tall",
                asset_id="fa-fm-tall",
                asset_kind="mathjax_svg",
                text=text,
                start=0,
                end=len(text),
                coordinate_system="mathjax_ex",
                natural_width=3,
                natural_height=8,
                natural_baseline_from_top=6,
                natural_depth=2,
            )
            result = metrics.measure(
                text,
                [0, 0, 100, 20],
                role="body",
                formula_runs=(spec,),
            )
            self.assertFalse(result.fits)
            self.assertGreater(result.used_height_pt, result.line_height_pt)
            self.assertGreater(result.overflow_height_pt, 0)

    def test_translation_and_reflow_share_v2_formula_measurement(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = _plan(root)
            provider = FakeProvider()
            config = RunnerConfig()
            result = execute_translation_plan(
                plan,
                provider=provider,
                cache=TranslationCache(root / "cache.sqlite3"),
                checkpoint_path=root / "manifest.json",
                config=config,
            )
            profile = build_golden_style_profile(plan)
            artifact_path = root / "paper_translated.json"
            assets = build_formula_asset_contract(plan, artifact_path)
            planning = plan_translation_layout(
                plan,
                result,
                profile,
                formula_asset_contract=assets,
            )
            artifact = build_translation_artifact(
                plan,
                provider=provider,
                requested_provider="fake",
                result=result,
                config=config,
                style_profile=profile,
                layout_planning=planning,
                formula_asset_contract=assets,
            )
            write_json_atomic(artifact_path, artifact)

            self.assertEqual(artifact["layout_run"]["contract_version"], "translation-reflow-v2")
            self.assertEqual(artifact["layout_run"]["measurement_engine"], "layout-metrics-v2")
            measurement = artifact["items"][0]["layout_preflight"]["measurement"]
            runs = [run for line in measurement["mixed_lines"] for run in line["runs"]]
            formula_runs = [run for run in runs if run["kind"] == "formula"]
            self.assertEqual(len(formula_runs), 1)
            self.assertEqual(formula_runs[0]["asset_kind"], "mathjax_svg")
            self.assertEqual(artifact["items"][0]["formula_ids"], [formula_runs[0]["formula_id"]])

            output = root / "step8d-vector-formula.pdf"
            report = ThinReflowRenderer().render(artifact_path, output)
            self.assertEqual(report["status"], "rendered")
            self.assertEqual(
                report["items"][0]["reason"],
                "shared_mixed_measurement_confirmed",
            )
            self.assertTrue(output.is_file())
            verification = report["formula_verification"]
            self.assertEqual(verification["expected"], 1)
            self.assertEqual(verification["drawn"], 1)
            self.assertEqual(verification["mathjax_svg"], 1)
            self.assertTrue(verification["order_verified"])
            self.assertTrue(verification["bbox_verified"])
            self.assertTrue(verification["baseline_verified"])
            self.assertTrue(verification["nonempty_verified"])
            self.assertTrue(verification["vector_only"])
            self.assertEqual(verification["overlap_count"], 0)

    def test_changed_mathjax_svg_is_rejected_before_rendering(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = _plan(root)
            provider = FakeProvider()
            config = RunnerConfig()
            result = execute_translation_plan(
                plan,
                provider=provider,
                cache=TranslationCache(root / "cache.sqlite3"),
                checkpoint_path=root / "manifest.json",
                config=config,
            )
            profile = build_golden_style_profile(plan)
            artifact_path = root / "paper_translated.json"
            assets = build_formula_asset_contract(plan, artifact_path)
            planning = plan_translation_layout(
                plan,
                result,
                profile,
                formula_asset_contract=assets,
            )
            artifact = build_translation_artifact(
                plan,
                provider=provider,
                requested_provider="fake",
                result=result,
                config=config,
                style_profile=profile,
                layout_planning=planning,
                formula_asset_contract=assets,
            )
            write_json_atomic(artifact_path, artifact)

            svg_path = artifact_path.parent / assets["assets"][0]["relative_path"]
            svg_path.write_text(
                svg_path.read_text(encoding="utf-8") + "\n<!-- changed -->\n",
                encoding="utf-8",
            )
            output = root / "must-not-exist.pdf"
            with self.assertRaisesRegex(
                ReflowContractError,
                "formula SVG is missing or changed",
            ):
                ThinReflowRenderer().render(artifact_path, output)
            self.assertFalse(output.exists())

    def test_source_pdf_clip_is_replayed_as_vector_formula(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = _plan(root, with_source_span=True)
            provider = FakeProvider()
            config = RunnerConfig()
            result = execute_translation_plan(
                plan,
                provider=provider,
                cache=TranslationCache(root / "cache.sqlite3"),
                checkpoint_path=root / "manifest.json",
                config=config,
            )
            profile = build_golden_style_profile(plan)
            artifact_path = root / "paper_translated.json"
            assets = build_formula_asset_contract(plan, artifact_path)
            self.assertEqual(assets["assets"][0]["kind"], "source_pdf_clip")
            planning = plan_translation_layout(
                plan,
                result,
                profile,
                formula_asset_contract=assets,
            )
            artifact = build_translation_artifact(
                plan,
                provider=provider,
                requested_provider="fake",
                result=result,
                config=config,
                style_profile=profile,
                layout_planning=planning,
                formula_asset_contract=assets,
            )
            write_json_atomic(artifact_path, artifact)
            output = root / "source-vector-formula.pdf"
            report = ThinReflowRenderer().render(artifact_path, output)
            self.assertEqual(report["status"], "rendered")
            verification = report["formula_verification"]
            self.assertEqual(verification["source_pdf_clip"], 1)
            self.assertEqual(verification["drawn"], 1)
            self.assertTrue(verification["vector_only"])
            self.assertTrue(verification["nonempty_verified"])


if __name__ == "__main__":
    unittest.main()
