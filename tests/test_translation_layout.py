from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import fitz

from hammer_translator.translation.layout import LayoutMetricError, LayoutMetrics
from hammer_translator.translation.loader import DiscoveredInput, load_document
from hammer_translator.translation.style import build_golden_style_profile
from hammer_translator.translation.units import build_translation_plan


def _write_pdf(path: Path) -> None:
    document = fitz.open()
    document.new_page(width=200, height=400)
    document.save(path)
    document.close()


def _span(text: str, size: float, *, font: str = "Times-Roman", flags: int = 4):
    return {
        "text": text,
        "size": size,
        "font": font,
        "flags": flags,
        "color": 0,
        "bbox": [10, 10, 100, 20],
    }


def _block(
    index: int,
    *,
    text: str,
    bbox: list[float],
    block_type: str = "text",
    sub_type: str | None = None,
    text_level: int | None = None,
    rich_spans: list[dict] | None = None,
    lines: list[dict] | None = None,
):
    value = {
        "type": block_type,
        "text": text,
        "page_idx": 0,
        "bbox": bbox,
        "logical_para_id": index,
    }
    if sub_type is not None:
        value["sub_type"] = sub_type
    if text_level is not None:
        value["text_level"] = text_level
    if rich_spans is not None:
        value["rich_spans"] = rich_spans
    if lines is not None:
        value["lines"] = lines
    return value


def _make_style_plan(root: Path):
    blocks = [
        _block(
            0,
            text="Document title",
            bbox=[10, 20, 190, 50],
            sub_type="title",
            text_level=1,
            rich_spans=[_span("Document title", 20, flags=20)],
            lines=[{"bbox": [45, 25, 155, 45]}],
        ),
        _block(
            1,
            text="1 Introduction",
            bbox=[10, 60, 190, 80],
            sub_type="title",
            text_level=2,
            rich_spans=[_span("1 Introduction", 14, flags=20)],
            lines=[{"bbox": [10, 62, 90, 76]}],
        ),
        _block(
            2,
            text="A sufficiently long body paragraph for stable measurement.",
            bbox=[10, 100, 190, 126],
            rich_spans=[
                _span("A sufficiently long body", 10),
                _span("paragraph for stable measurement", 10.2),
                _span("x", 30),
            ],
            lines=[
                {"bbox": [10, 100, 175, 110]},
                {"bbox": [10, 112, 180, 122]},
            ],
        ),
        _block(
            3,
            block_type="caption",
            text="Figure 1. Caption",
            bbox=[20, 150, 180, 170],
            rich_spans=[],
            lines=[{"bbox": [20, 151, 110, 165]}],
        ),
    ]
    source = root / "paper_enriched.json"
    source.write_text(json.dumps(blocks), encoding="utf-8")
    _write_pdf(root / "paper_origin.pdf")
    document = load_document(DiscoveredInput(source.resolve(), "test"))
    return build_translation_plan(document)


class GoldenStyleTests(unittest.TestCase):
    def test_profile_uses_original_styles_per_hierarchy_and_filters_outlier(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            plan = _make_style_plan(Path(temp_dir))
            profile = build_golden_style_profile(plan)
            self.assertEqual(profile["schema_version"], "layout-style-v1")
            self.assertEqual(profile["measurement_engine"], "layout-metrics-v2")
            self.assertEqual(profile["statistics_basis"], "original_pdf")
            self.assertFalse(profile["translated_text_used_for_statistics"])
            self.assertIn("document_title", profile["roles"])
            self.assertIn("section_header_l1", profile["roles"])
            self.assertIn("body", profile["roles"])
            self.assertIn("caption", profile["roles"])

            body = profile["roles"]["body"]
            self.assertAlmostEqual(body["source_statistics"]["font_size_pt"], 10.2)
            self.assertAlmostEqual(
                body["source_statistics"]["line_height_multiplier"], 1.1765
            )
            self.assertEqual(body["provenance"]["font_size"], "measured")
            self.assertTrue(body["guardrails"]["bbox_locked"])
            self.assertAlmostEqual(body["guardrails"]["font_size_pt"]["min"], 9.18)
            self.assertAlmostEqual(body["guardrails"]["font_size_pt"]["max"], 10.2)
            self.assertEqual(body["guardrails"]["char_spacing_em"], {"min": -0.04, "max": 0.02})

            title = profile["roles"]["document_title"]
            self.assertEqual(title["source_statistics"]["alignment"], "center")
            self.assertEqual(title["target_style"]["font_id"], "zh_title_regular")
            font_path = Path(profile["fonts"]["zh_title_regular"]["path"])
            self.assertTrue(font_path.is_file())

            caption = profile["roles"]["caption"]
            self.assertEqual(caption["provenance"]["font_size"], "derived_from_body")
            self.assertAlmostEqual(caption["target_style"]["font_size_pt"], 9.18)

    def test_each_unit_has_one_explicit_style_role(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            plan = _make_style_plan(Path(temp_dir))
            profile = build_golden_style_profile(plan)
            self.assertEqual(len(profile["unit_roles"]), len(plan.units))
            self.assertEqual(
                set(profile["unit_roles"].values()),
                {"document_title", "section_header_l1", "body", "caption"},
            )

    def test_body_line_height_guardrail_keeps_a_practical_floor(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "paper_enriched.json"
            source.write_text(
                json.dumps(
                    [
                        _block(
                            0,
                            text="A body paragraph with generous original leading.",
                            bbox=[10, 20, 190, 70],
                            rich_spans=[_span("Body paragraph", 10)],
                            lines=[
                                {"bbox": [10, 20, 180, 30]},
                                {"bbox": [10, 37.5, 180, 47.5]},
                            ],
                        )
                    ]
                ),
                encoding="utf-8",
            )
            _write_pdf(root / "paper_origin.pdf")
            document = load_document(DiscoveredInput(source.resolve(), "test"))
            profile = build_golden_style_profile(build_translation_plan(document))
            body = profile["roles"]["body"]
            self.assertAlmostEqual(
                body["source_statistics"]["line_height_multiplier"], 1.75
            )
            self.assertAlmostEqual(
                body["guardrails"]["line_height_multiplier"]["min"], 1.15
            )


class LayoutMetricsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.plan = _make_style_plan(self.root)
        self.profile = build_golden_style_profile(self.plan)
        self.metrics = LayoutMetrics(self.profile)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_measurement_is_deterministic_and_uses_registered_font(self):
        text = "这是一段用于测量的中文文本"
        first = self.metrics.measure(text, [0, 0, 80, 60], role="body")
        second = self.metrics.measure(text, [0, 0, 80, 60], role="body")
        self.assertEqual(first, second)
        self.assertEqual(first.font_id, "zh_body_regular")
        self.assertGreaterEqual(first.line_count, 2)
        self.assertEqual(len(first.lines), len(first.line_widths_pt))

    def test_measurement_cache_is_exact_and_reuses_widths_and_tokens(self):
        cached = LayoutMetrics(self.profile)
        uncached = LayoutMetrics(self.profile, measurement_cache_max_entries=0)
        normal_style, _ = cached.style_for_role("body")
        tight_style = dict(normal_style)
        tight_style["char_spacing_em"] = -0.04
        cases = (
            ("这是一段重复测量的中文文本", [0, 0, 80, 60], normal_style),
            ("这是一段重复测量的中文文本", [0, 0, 80, 60], tight_style),
            ("Chinese words and 中文 mixed together", [0, 0, 95, 60], normal_style),
        )
        first_pass = []
        for text, bbox, style in cases:
            expected = uncached.measure(text, bbox, role="body", style=style)
            actual = cached.measure(text, bbox, role="body", style=style)
            self.assertEqual(actual, expected)
            first_pass.append(actual)
        for (text, bbox, style), expected in zip(cases, first_pass, strict=True):
            self.assertEqual(
                cached.measure(text, bbox, role="body", style=style),
                expected,
            )
        stats = cached.measurement_cache_stats()
        self.assertGreater(stats["glyph_width_hits"], 0)
        self.assertGreater(stats["glyph_width_misses"], 0)
        self.assertGreater(stats["token_hits"], 0)
        self.assertGreater(stats["token_misses"], 0)

    def test_measurement_cache_is_bounded(self):
        metrics = LayoutMetrics(self.profile, measurement_cache_max_entries=2)
        for text in ("甲乙丙丁", "戊己庚辛", "壬癸子丑"):
            metrics.measure(text, [0, 0, 100, 60], role="body")
        stats = metrics.measurement_cache_stats()
        self.assertLessEqual(stats["glyph_width_entries"], 2)
        self.assertLessEqual(stats["token_entries"], 2)
        self.assertGreater(stats["glyph_width_evictions"], 0)
        self.assertGreater(stats["token_evictions"], 0)

    def test_narrower_box_requires_at_least_as_many_lines(self):
        text = "中文排版需要使用与渲染一致的字体度量"
        wide = self.metrics.measure(text, [0, 0, 140, 100], role="body")
        narrow = self.metrics.measure(text, [0, 0, 55, 100], role="body")
        self.assertGreaterEqual(narrow.line_count, wide.line_count)
        self.assertGreaterEqual(narrow.height_fill_ratio, wide.height_fill_ratio)

    def test_height_overflow_and_fixed_bbox_are_reported(self):
        result = self.metrics.measure(
            "这是一段会换成多行的中文文本",
            [10, 20, 55, 31],
            role="body",
        )
        self.assertFalse(result.fits)
        self.assertTrue(result.is_overflow)
        self.assertEqual(result.available_width_pt, 45.0)
        self.assertEqual(result.available_height_pt, 11.0)
        self.assertGreater(result.overflow_height_pt, 0)

    def test_sub_point_rounding_overflow_is_accepted(self):
        text = "U-Net 编码器路径"
        style, _ = self.metrics.style_for_role("body")
        full = self.metrics.measure(text, [0, 0, 200, 100], role="body", style=style)
        allowed = self.metrics.measure(
            text,
            [0, 0, 200, full.used_height_pt - 0.5],
            role="body",
            style=style,
        )
        rejected = self.metrics.measure(
            text,
            [0, 0, 200, full.used_height_pt - 1.5],
            role="body",
            style=style,
        )
        self.assertTrue(allowed.fits)
        self.assertAlmostEqual(allowed.overflow_height_pt, 0.5, delta=0.1)
        self.assertFalse(rejected.fits)
        self.assertAlmostEqual(rejected.overflow_height_pt, 1.5, delta=0.1)

    def test_guardrails_reject_excessive_font_shrink(self):
        style, _ = self.metrics.style_for_role("body")
        style["font_size_pt"] = 9.0
        with self.assertRaisesRegex(LayoutMetricError, "outside role body guardrails"):
            self.metrics.measure("test", [0, 0, 100, 100], role="body", style=style)

    def test_allowed_negative_char_spacing_reduces_width(self):
        normal_style, _ = self.metrics.style_for_role("body")
        tight_style = dict(normal_style)
        tight_style["char_spacing_em"] = -0.04
        normal = self.metrics.measure("ChineseABC", [0, 0, 200, 100], role="body")
        tight = self.metrics.measure(
            "ChineseABC", [0, 0, 200, 100], role="body", style=tight_style
        )
        self.assertLess(tight.used_width_pt, normal.used_width_pt)

    def test_malformed_latex_residue_is_rendered_as_readable_text(self):
        source = "其中 r _ { \\mathrm { s } } , 与 \\rho _ { \\mathrm { s } } , 是参数。"
        result = self.metrics.measure(source, [0, 0, 200, 100], role="body")
        rendered = "".join(line.text for line in result.mixed_lines)
        self.assertNotIn("\\", rendered)
        self.assertNotIn("$", rendered)
        self.assertIn("r_s", rendered)
        self.assertIn("ρ_s", rendered)

    def test_closing_punctuation_is_kept_off_the_start_of_a_line(self):
        style, _ = self.metrics.style_for_role("body")
        font = self.metrics._font(style["font_id"])
        text = "甲" * 20 + "。乙"
        width = (
            self.metrics._text_width(
                font,
                "甲" * 20 + "。",
                font_size=float(style["font_size_pt"]),
                char_spacing_em=float(style["char_spacing_em"]),
            )
            - 0.1
        )

        result = self.metrics.measure(text, [0, 0, width, 100], role="body", style=style)

        self.assertFalse(any(line.startswith("。") for line in result.lines))
        self.assertTrue(result.lines[1].startswith("甲。"))

    def test_closing_punctuation_is_kept_off_the_start_of_a_mixed_line(self):
        style, _ = self.metrics.style_for_role("body")
        font = self.metrics._font(style["font_id"])
        text = "甲" * 20 + "。乙"
        width = (
            self.metrics._text_width(
                font,
                "甲" * 20 + "。",
                font_size=float(style["font_size_pt"]),
                char_spacing_em=float(style["char_spacing_em"]),
            )
            - 0.1
        )

        lines, _, _ = self.metrics._wrap_mixed(
            text,
            formula_runs=(),
            role="body",
            font=font,
            font_size=float(style["font_size_pt"]),
            line_height_multiplier=float(style["line_height_multiplier"]),
            char_spacing_em=float(style["char_spacing_em"]),
            available_width=width,
        )

        self.assertFalse(any(line.startswith("。") for line in lines))
        self.assertTrue(lines[1].startswith("甲。"))


if __name__ == "__main__":
    unittest.main()
