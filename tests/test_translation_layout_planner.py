from __future__ import annotations

import json
import itertools
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz

from hammer_translator.translation.artifact import build_translation_artifact
from hammer_translator.translation.cache import TranslationCache
from hammer_translator.translation.layout import LayoutMetrics
from hammer_translator.translation.layout_planner import (
    LayoutSearchConfig,
    _block_style_bounds,
    _candidate_from_response,
    _candidate_cut_sets,
    _capacity_target_tolerance,
    _capacity_targets,
    _cut_priority_component,
    _cut_priority_key,
    _valid_boundaries,
    allocate_unit_text,
    optimize_style,
    plan_translation_layout,
)
from hammer_translator.translation.loader import DiscoveredInput, load_document
from hammer_translator.translation.provider import (
    FakeProvider,
    ProviderResponse,
    TranslationRequest,
)
from hammer_translator.translation.runner import RunnerConfig, execute_translation_plan
from hammer_translator.translation.style import build_golden_style_profile
from hammer_translator.translation.units import build_translation_plan


def _make_plan(root: Path, blocks: list[dict]):
    source = root / "paper_enriched.json"
    source.write_text(json.dumps(blocks), encoding="utf-8")
    pdf = fitz.open()
    pdf.new_page(width=300, height=400)
    pdf.save(root / "paper_origin.pdf")
    pdf.close()
    document = load_document(DiscoveredInput(source.resolve(), "test"))
    return build_translation_plan(document)


def _block(
    index: int,
    text: str,
    bbox: list[float],
    *,
    logical_id: int = 7,
    **extra,
) -> dict:
    value = {
        "type": "text",
        "text": text,
        "context": "First source sentence. Second source sentence.",
        "page_idx": 0,
        "bbox": bbox,
        "logical_para_id": logical_id,
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
        "lines": [{"bbox": bbox}],
    }
    value.update(extra)
    return value


class BoundedStyleOptimizerTests(unittest.TestCase):
    def test_small_heading_is_capped_by_its_measured_source_font(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            large_heading = _block(
                0,
                "A long section heading used to raise the role size",
                [10, 10, 250, 40],
                logical_id=7,
                text_level=2,
                sub_type="title",
            )
            large_heading["rich_spans"] = [
                {
                    "text": large_heading["text"],
                    "size": 14.0,
                    "font": "Times-Bold",
                    "flags": 16,
                    "color": 0,
                    "bbox": large_heading["bbox"],
                }
            ]
            small_heading = _block(
                1,
                "Abstract",
                [200, 254, 244, 265],
                logical_id=8,
                text_level=2,
                sub_type="title",
            )
            small_heading["rich_spans"] = [
                {
                    "text": "Abstract",
                    "size": 8.966,
                    "font": "Times-Bold",
                    "flags": 16,
                    "color": 0,
                    "bbox": small_heading["bbox"],
                }
            ]
            plan = _make_plan(root, [large_heading, small_heading])
            profile = build_golden_style_profile(plan)
            metrics = LayoutMetrics(profile)
            small_block = next(
                block for block in plan.document.blocks if block.data.text == "Abstract"
            )
            unit = next(
                unit for unit in plan.units if unit.source_text == "Abstract"
            )
            role = "section_header_l1"
            golden, _ = metrics.style_for_role(role)
            capped, guardrails = _block_style_bounds(metrics, role, small_block)
            self.assertEqual(golden["font_size_pt"], 14.0)
            self.assertEqual(capped["font_size_pt"], 8.966)
            self.assertEqual(guardrails["line_height_multiplier"]["min"], 1.0)

            outcome = allocate_unit_text(
                unit,
                "摘要",
                blocks=[small_block],
                role=role,
                metrics=metrics,
            )
            self.assertTrue(outcome.all_blocks_fit)
            self.assertLessEqual(
                outcome.segments[0].optimization.style["font_size_pt"], 8.966
            )

    def test_zero_score_golden_fit_stops_at_proven_global_optimum(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = _make_plan(root, [_block(0, "Source", [10, 10, 250, 50])])
            profile = build_golden_style_profile(plan)
            metrics = LayoutMetrics(profile)
            result = optimize_style(metrics, "简短译文", [10, 10, 250, 50], role="body")
            golden, _ = metrics.style_for_role("body")
            self.assertEqual(result.status, "fits_golden")
            self.assertEqual(result.style, golden)
            self.assertEqual(result.score, 0.0)
            self.assertEqual(result.evaluated_candidates, 1)
            self.assertEqual(result.feasible_candidates, 1)

    def test_exact_pruning_matches_exhaustive_global_optimum(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = _make_plan(root, [_block(0, "Source", [10, 10, 250, 50])])
            profile = build_golden_style_profile(plan)
            metrics = LayoutMetrics(profile)
            cases = (
                ("简短译文", [10, 10, 250, 50]),
                ("这是一段需要调整字号与行距的较长译文" * 3, [10, 10, 120, 45]),
                ("这是一段无法放入极小固定边界框的译文", [10, 10, 35, 20]),
            )
            reduced = False
            for text, bbox in cases:
                exhaustive = optimize_style(
                    metrics,
                    text,
                    bbox,
                    role="body",
                    config=LayoutSearchConfig(exact_pruning=False),
                )
                pruned = optimize_style(
                    metrics,
                    text,
                    bbox,
                    role="body",
                    config=LayoutSearchConfig(exact_pruning=True),
                )
                self.assertEqual(pruned.status, exhaustive.status)
                self.assertEqual(pruned.style, exhaustive.style)
                self.assertEqual(pruned.score, exhaustive.score)
                self.assertEqual(
                    pruned.measurement.as_dict(), exhaustive.measurement.as_dict()
                )
                self.assertLessEqual(
                    pruned.evaluated_candidates, exhaustive.evaluated_candidates
                )
                reduced = reduced or (
                    pruned.evaluated_candidates < exhaustive.evaluated_candidates
                )
            self.assertTrue(reduced)

    def test_impossible_fit_returns_compact_boundary_and_constraints(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = _make_plan(root, [_block(0, "Source", [10, 10, 35, 20])])
            profile = build_golden_style_profile(plan)
            metrics = LayoutMetrics(profile)
            unit = plan.units[0]
            outcome = allocate_unit_text(
                unit,
                "这是一段明显无法放入超小固定边界框的长译文",
                blocks=list(plan.document.blocks),
                role=profile["unit_roles"][unit.translation_unit_id],
                metrics=metrics,
            )
            self.assertEqual(outcome.status, "requires_retranslation")
            self.assertFalse(outcome.all_blocks_fit)
            self.assertTrue(outcome.retranslation_constraints["bbox_locked"])
            self.assertGreater(
                outcome.retranslation_constraints["required_reduction_ratio"], 0
            )


class TranslationAllocationTests(unittest.TestCase):
    def test_capacity_target_precedes_distant_punctuation(self):
        config = LayoutSearchConfig()
        text = "甲" * 9 + "。" + "乙" * 30
        cuts = _candidate_cut_sets(text, [3, 1], (), config)
        self.assertTrue(cuts)
        self.assertEqual(cuts[0], (30,))

    def test_punctuation_wins_inside_capacity_tolerance(self):
        config = LayoutSearchConfig()
        text = "甲" * 17 + "。" + "乙" * 22
        cuts = _candidate_cut_sets(text, [1, 1], (), config)
        self.assertTrue(cuts)
        self.assertEqual(cuts[0], (18,))

    def test_formula_and_protected_content_are_hard_cut_constraints(self):
        config = LayoutSearchConfig(boundary_candidates_per_cut=20)
        text = "甲甲$y=x+1$乙乙PROTECTED丙丙"
        cuts = _candidate_cut_sets(text, [1, 1], ("PROTECTED",), config)
        formula_start = text.index("$y=x+1$")
        formula_end = formula_start + len("$y=x+1$")
        protected_start = text.index("PROTECTED")
        protected_end = protected_start + len("PROTECTED")
        self.assertTrue(cuts)
        for (cut,) in cuts:
            self.assertFalse(formula_start < cut < formula_end)
            self.assertFalse(protected_start < cut < protected_end)

    def test_bounded_cut_generator_matches_brute_force_top_k(self):
        config = LayoutSearchConfig(
            boundary_candidates_per_cut=6,
            max_allocation_candidates=20,
        )
        text = "甲。乙，丙；丁甲乙丙丁甲乙丙丁甲乙丙丁甲乙丙丁甲乙丙丁甲"
        capacities = [2, 3, 2, 1]
        boundaries = _valid_boundaries(text, ())
        targets = _capacity_targets(len(text), capacities)
        tolerance = _capacity_target_tolerance(len(text), config)
        choices = []
        for target in targets:
            ranked = sorted(
                boundaries,
                key=lambda position: (
                    *_cut_priority_component(text, position, target, tolerance),
                    position,
                ),
            )
            choices.append(ranked[: config.boundary_candidates_per_cut])
        brute_force = [
            cuts
            for cuts in itertools.product(*choices)
            if tuple(sorted(cuts)) == cuts and len(set(cuts)) == len(cuts)
        ]
        brute_force.sort(
            key=lambda cuts: _cut_priority_key(text, cuts, capacities, config)
        )
        expected = tuple(brute_force[: config.max_allocation_candidates])
        self.assertEqual(
            _candidate_cut_sets(text, capacities, (), config),
            expected,
        )

    def test_pruned_allocation_matches_reference_and_reduces_style_calls(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = _make_plan(
                root,
                [
                    _block(0, "First source sentence.", [10, 10, 125, 42]),
                    _block(1, "Second source sentence.", [10, 48, 180, 84]),
                    _block(2, "Third source sentence.", [10, 90, 150, 124]),
                ],
            )
            profile = build_golden_style_profile(plan)
            unit = plan.units[0]
            text = "第一段内容。第二段内容稍长，包含更多细节。第三段内容收尾。"
            common = dict(
                font_size_step_pt=1.0,
                line_height_step=0.1,
                char_spacing_step_em=0.05,
                boundary_candidates_per_cut=7,
                max_allocation_candidates=32,
            )
            with patch(
                "hammer_translator.translation.layout_planner.optimize_style",
                wraps=optimize_style,
            ) as reference_calls:
                reference = allocate_unit_text(
                    unit,
                    text,
                    blocks=list(plan.document.blocks),
                    role=profile["unit_roles"][unit.translation_unit_id],
                    metrics=LayoutMetrics(profile),
                    config=LayoutSearchConfig(
                        exact_pruning=True,
                        exact_allocation_pruning=False,
                        **common,
                    ),
                )
            with patch(
                "hammer_translator.translation.layout_planner.optimize_style",
                wraps=optimize_style,
            ) as pruned_calls:
                pruned = allocate_unit_text(
                    unit,
                    text,
                    blocks=list(plan.document.blocks),
                    role=profile["unit_roles"][unit.translation_unit_id],
                    metrics=LayoutMetrics(profile),
                    config=LayoutSearchConfig(
                        exact_pruning=True,
                        exact_allocation_pruning=True,
                        **common,
                    ),
                )
            self.assertEqual(pruned.status, reference.status)
            self.assertEqual(pruned.score, reference.score)
            self.assertEqual(
                [segment.text for segment in pruned.segments],
                [segment.text for segment in reference.segments],
            )
            self.assertEqual(
                [segment.optimization.style for segment in pruned.segments],
                [segment.optimization.style for segment in reference.segments],
            )
            self.assertLess(pruned_calls.call_count, reference_calls.call_count)

    def test_multi_block_split_is_contiguous_lossless_and_semantic(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = _make_plan(
                root,
                [
                    _block(0, "First source sentence.", [10, 10, 150, 42]),
                    _block(1, "Second source sentence.", [10, 48, 150, 80]),
                ],
            )
            profile = build_golden_style_profile(plan)
            metrics = LayoutMetrics(profile)
            unit = plan.units[0]
            text = "第一个完整句子。第二个完整句子。"
            outcome = allocate_unit_text(
                unit,
                text,
                blocks=list(plan.document.blocks),
                role=profile["unit_roles"][unit.translation_unit_id],
                metrics=metrics,
            )
            self.assertTrue(outcome.all_blocks_fit)
            self.assertEqual("".join(segment.text for segment in outcome.segments), text)
            self.assertTrue(outcome.segments[0].text.endswith("。"))

    def test_protected_formula_is_never_split_between_blocks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = _make_plan(
                root,
                [
                    _block(0, "Value $x+y$ appears.", [10, 10, 120, 42]),
                    _block(1, "Then discussion.", [10, 48, 120, 80]),
                ],
            )
            profile = build_golden_style_profile(plan)
            metrics = LayoutMetrics(profile)
            unit = plan.units[0]
            text = "前文说明较长的内容$x+y$后文继续说明。"
            outcome = allocate_unit_text(
                unit,
                text,
                blocks=list(plan.document.blocks),
                role=profile["unit_roles"][unit.translation_unit_id],
                metrics=metrics,
            )
            containing = [segment for segment in outcome.segments if "$x+y$" in segment.text]
            self.assertEqual(len(containing), 1)
            self.assertEqual("".join(segment.text for segment in outcome.segments), text)

    def test_constrained_candidate_is_selected_only_after_it_fits(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = _make_plan(root, [_block(0, "Long source text", [10, 10, 70, 28])])
            unit = plan.units[0]
            provider = FakeProvider(
                scripted={
                    unit.translation_unit_id: [
                        "这是一段非常非常长的初始译文它无法放入指定的固定框中",
                        "简短译文",
                    ]
                }
            )
            cache = TranslationCache(root / "cache.sqlite3")
            config = RunnerConfig(max_attempts=1)
            result = execute_translation_plan(
                plan,
                provider=provider,
                cache=cache,
                checkpoint_path=root / "manifest.json",
                config=config,
            )
            profile = build_golden_style_profile(plan)
            planning = plan_translation_layout(
                plan,
                result,
                profile,
                provider=provider,
                cache=cache,
                runner_config=config,
                search_config=LayoutSearchConfig(max_retranslation_attempts=2),
            )
            outcome = planning.outcomes[unit.translation_unit_id]
            self.assertTrue(outcome.allocation.all_blocks_fit)
            self.assertEqual(outcome.selected_text, "简短译文")
            self.assertTrue(outcome.selected_version.startswith("candidate-"))
            self.assertEqual(len(outcome.candidates), 1)
            self.assertEqual(planning.stats.constrained_successes, 1)

            artifact = build_translation_artifact(
                plan,
                provider=provider,
                requested_provider="fake",
                result=result,
                config=config,
                style_profile=profile,
                layout_planning=planning,
            )
            self.assertEqual(artifact["layout_status"], "ready_for_reflow")
            self.assertEqual(artifact["items"][0]["translated"], "简短译文")
            self.assertEqual(
                artifact["translation_units"][0]["translation"]["selected_text"],
                "简短译文",
            )

    def test_review_output_keeps_layout_selected_candidate_canonical(self):
        class RealLikeProvider(FakeProvider):
            name = "local-test"

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = "This deliberately long English source remains untranslated for QA."
            plan = _make_plan(root, [_block(0, source, [10, 10, 70, 28])])
            unit = plan.units[0]
            provider = RealLikeProvider(
                scripted={unit.translation_unit_id: [source, "简短译文"]}
            )
            cache = TranslationCache(root / "cache.sqlite3")
            config = RunnerConfig(max_attempts=1)
            result = execute_translation_plan(
                plan,
                provider=provider,
                cache=cache,
                checkpoint_path=root / "manifest.json",
                config=config,
            )
            self.assertEqual(
                result.unit_states[unit.translation_unit_id]["execution_status"],
                "needs_review",
            )
            profile = build_golden_style_profile(plan)
            planning = plan_translation_layout(
                plan,
                result,
                profile,
                provider=provider,
                cache=cache,
                runner_config=config,
                search_config=LayoutSearchConfig(max_retranslation_attempts=1),
            )

            artifact = build_translation_artifact(
                plan,
                provider=provider,
                requested_provider="local-test",
                result=result,
                config=config,
                style_profile=profile,
                layout_planning=planning,
            )
            translation = artifact["translation_units"][0]["translation"]
            self.assertEqual(translation["status"], "needs_review")
            self.assertEqual(translation["selected_text"], "简短译文")
            self.assertEqual(artifact["items"][0]["translated"], "简短译文")


class ConstrainedRetranslationTests(unittest.TestCase):
    def test_candidate_strips_prompt_wrapper_before_acceptance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            plan = _make_plan(
                Path(temp_dir),
                [_block(0, "Translate me", [10, 10, 70, 28])],
            )
            unit = plan.units[0]
            request = TranslationRequest(
                translation_unit_id=unit.translation_unit_id,
                protected_text=unit.protected_text,
                source_language="en",
                target_language="zh-CN",
                element_type=unit.element_type.value,
                provider="fake",
                model="fake-v1",
                prompt_version="translation-prompt-v2",
                glossary_version="none",
                mode="constrained",
                constraints={"reason": "fixed_bbox_overflow"},
            )
            response = ProviderResponse(
                text=(
                    "元素类型：文本。\n"
                    "源语言：英语。\n"
                    "目标语言：简体中文。\n"
                    "在定界符之间翻译源文本。\n"
                    "<源文本>这是一段简短的中文译文。</源文本>"
                ),
                provider="fake",
                model="fake-v1",
            )
            candidate = _candidate_from_response(
                unit,
                request,
                response,
                simulation=False,
                ordinal=1,
            )
            self.assertEqual(candidate.text, "这是一段简短的中文译文。")


if __name__ == "__main__":
    unittest.main()
