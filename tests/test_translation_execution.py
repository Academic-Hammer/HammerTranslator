from __future__ import annotations

import contextlib
import io
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz

from hammer_translator.translation.artifact import build_translation_artifact
from hammer_translator.translation.cache import TranslationCache
from hammer_translator.translation.checkpoint import CheckpointError, CheckpointStore
from hammer_translator.translation.cli import main
from hammer_translator.translation.loader import DiscoveredInput, load_document
from hammer_translator.translation.provider import (
    FakeProvider,
    ProviderError,
    ProviderResponse,
    TranslationRequest,
)
from hammer_translator.translation.runner import RunnerConfig, execute_translation_plan
from hammer_translator.translation.units import build_translation_plan


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "translation_config.yaml"


def _block(
    index: int,
    *,
    block_type: str = "text",
    text: str = "Translate me",
    logical_id: int | None = None,
    **extra,
) -> dict:
    value = {
        "type": block_type,
        "text": text,
        "page_idx": 0,
        "bbox": [10, 10 + index * 30, 190, 30 + index * 30],
        "logical_para_id": index if logical_id is None else logical_id,
    }
    value.update(extra)
    return value


def _make_plan(root: Path, blocks: list[dict]):
    source = root / "paper_enriched.json"
    source.write_text(json.dumps(blocks), encoding="utf-8")
    pdf_path = root / "paper_origin.pdf"
    pdf = fitz.open()
    pdf.new_page(width=200, height=400)
    pdf.save(pdf_path)
    pdf.close()
    document = load_document(DiscoveredInput(source.resolve(), "test"))
    return build_translation_plan(document)


def _request(**overrides) -> TranslationRequest:
    values = {
        "translation_unit_id": "tu-a",
        "protected_text": "Hello",
        "source_language": "en",
        "target_language": "zh-CN",
        "element_type": "text",
        "provider": "fake",
        "model": "fake-v1",
        "prompt_version": "prompt-v1",
        "glossary_version": "none",
    }
    values.update(overrides)
    return TranslationRequest(**values)


class ProviderAndCacheTests(unittest.TestCase):
    def test_cache_key_is_content_based_but_semantically_versioned(self):
        base = _request()
        same_content_other_document = _request(translation_unit_id="tu-other")
        self.assertEqual(base.cache_key(), same_content_other_document.cache_key())
        for change in (
            {"element_type": "title"},
            {"model": "fake-v2"},
            {"prompt_version": "prompt-v2"},
            {"glossary_version": "glossary-v1"},
            {"mode": "constrained"},
            {"constraints": {"max_chars": 20}},
        ):
            self.assertNotEqual(base.cache_key(), _request(**change).cache_key())

    def test_cache_keeps_only_allowlisted_redacted_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = TranslationCache(Path(temp_dir) / "cache.sqlite3")
            request = _request()
            secret = "sk-1234567890abcdef"
            response = ProviderResponse(
                text="你好",
                provider="fake",
                model="fake-v1",
                request_id=f"request Bearer {secret}",
                metadata={"total_tokens": 2, "api_key": secret, "secret": secret},
            )
            entry = cache.put(request, response)
            self.assertEqual(entry.metadata["total_tokens"], 2)
            self.assertNotIn("api_key", entry.metadata)
            self.assertNotIn("secret", json.dumps(entry.metadata))
            self.assertNotIn(secret, json.dumps(entry.metadata))

    def test_corrupted_cached_response_is_discarded(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = TranslationCache(Path(temp_dir) / "cache.sqlite3")
            request = _request()
            cache.put(
                request,
                ProviderResponse(text="你好", provider="fake", model="fake-v1"),
            )
            with sqlite3.connect(cache.path) as connection:
                connection.execute(
                    "UPDATE cache_entries SET response_text = ? WHERE cache_key = ?",
                    ("被篡改", request.cache_key()),
                )
            self.assertIsNone(cache.get(request.cache_key()))
            self.assertEqual(cache.count(), 0)


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_rejects_incompatible_resume_and_leaves_no_temp_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manifest.json"
            store = CheckpointStore(path)
            signature = {"provider": "fake", "model": "v1"}
            store.initialize(
                document_id="doc",
                source_hash="hash-a",
                run_signature=signature,
                unit_ids=["tu-a"],
            )
            store.update_unit("tu-a", execution_status="completed")
            self.assertTrue(path.exists())
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])
            with self.assertRaisesRegex(CheckpointError, "source_hash mismatch"):
                CheckpointStore(path).initialize(
                    document_id="doc",
                    source_hash="hash-b",
                    run_signature=signature,
                    unit_ids=["tu-a"],
                )


class TranslationRunnerTests(unittest.TestCase):
    def test_failed_checkpoint_unit_can_be_explicitly_retried(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = _make_plan(root, [_block(0, text="Hello")])
            unit = plan.units[0]
            checkpoint = root / "manifest.json"
            cache = TranslationCache(root / "cache.sqlite3")
            first = FakeProvider(
                model="same-model",
                scripted={
                    unit.translation_unit_id: [
                        ProviderError(
                            "temporary server failure",
                            code="provider_server_error",
                            retryable=False,
                            status_code=503,
                        )
                    ]
                },
            )
            failed = execute_translation_plan(
                plan,
                provider=first,
                cache=cache,
                checkpoint_path=checkpoint,
            )
            self.assertEqual(failed.stats.failed, 1)

            second = FakeProvider(model="same-model", transform=lambda request: "你好")
            resumed = execute_translation_plan(
                plan,
                provider=second,
                cache=cache,
                checkpoint_path=checkpoint,
                config=RunnerConfig(retry_failed=True),
            )
            self.assertEqual(second.total_calls, 1)
            self.assertEqual(resumed.stats.completed, 1)
            self.assertEqual(resumed.outputs[unit.translation_unit_id], "你好")

    def test_retryable_failure_retries_then_completes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = _make_plan(root, [_block(0, text="Hello")])
            unit = plan.units[0]
            provider = FakeProvider(
                scripted={
                    unit.translation_unit_id: [
                        ProviderError(
                            "rate limited",
                            code="rate_limit",
                            retryable=True,
                            retry_after=0,
                        ),
                        "你好",
                    ]
                }
            )
            delays: list[float] = []
            result = execute_translation_plan(
                plan,
                provider=provider,
                cache=TranslationCache(root / "cache.sqlite3"),
                checkpoint_path=root / "manifest.json",
                sleep=delays.append,
            )
            self.assertEqual(provider.total_calls, 2)
            self.assertEqual(result.stats.retries, 1)
            self.assertEqual(result.stats.completed, 1)
            self.assertEqual(result.outputs[unit.translation_unit_id], "你好")
            self.assertEqual(delays, [0.0])

    def test_non_retryable_failure_stops_immediately_and_redacts_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = _make_plan(root, [_block(0)])
            unit = plan.units[0]
            secret = "sk-1234567890abcdef"
            provider = FakeProvider(
                scripted={
                    unit.translation_unit_id: [
                        ProviderError(
                            f"authorization={secret}",
                            code="bad_request",
                            retryable=False,
                        )
                    ]
                }
            )
            cache = TranslationCache(root / "cache.sqlite3")
            result = execute_translation_plan(
                plan,
                provider=provider,
                cache=cache,
                checkpoint_path=root / "manifest.json",
                sleep=lambda _: None,
            )
            manifest_text = (root / "manifest.json").read_text(encoding="utf-8")
            self.assertEqual(provider.total_calls, 1)
            self.assertEqual(result.stats.failed, 1)
            self.assertEqual(cache.count(), 0)
            self.assertIn('"run_status": "completed_with_errors"', manifest_text)
            self.assertNotIn(secret, manifest_text)

    def test_second_run_uses_cache_without_provider_call(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = _make_plan(root, [_block(0)])
            cache = TranslationCache(root / "cache.sqlite3")
            checkpoint = root / "manifest.json"
            first = FakeProvider(model="same-model")
            execute_translation_plan(
                plan, provider=first, cache=cache, checkpoint_path=checkpoint
            )
            second = FakeProvider(model="same-model")
            result = execute_translation_plan(
                plan, provider=second, cache=cache, checkpoint_path=checkpoint
            )
            self.assertEqual(first.total_calls, 1)
            self.assertEqual(second.total_calls, 0)
            self.assertEqual(result.stats.cache_hits, 1)
            self.assertEqual(result.stats.resumed_from_checkpoint, 1)

    def test_interruption_leaves_resumable_in_progress_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = _make_plan(root, [_block(0)])
            unit = plan.units[0]
            cache = TranslationCache(root / "cache.sqlite3")
            checkpoint = root / "manifest.json"
            interrupted = FakeProvider(
                model="same-model",
                scripted={unit.translation_unit_id: [KeyboardInterrupt()]},
            )
            with self.assertRaises(KeyboardInterrupt):
                execute_translation_plan(
                    plan,
                    provider=interrupted,
                    cache=cache,
                    checkpoint_path=checkpoint,
                )
            state = json.loads(checkpoint.read_text(encoding="utf-8"))
            self.assertEqual(
                state["units"][unit.translation_unit_id]["execution_status"],
                "in_progress",
            )
            self.assertEqual(state["units"][unit.translation_unit_id]["attempts"], 1)
            resumed = FakeProvider(model="same-model")
            result = execute_translation_plan(
                plan,
                provider=resumed,
                cache=cache,
                checkpoint_path=checkpoint,
            )
            self.assertEqual(resumed.total_calls, 1)
            self.assertEqual(result.stats.completed, 1)

    def test_invalid_protected_content_is_never_cached(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = _make_plan(root, [_block(0, text="Value $x$ stays")])
            unit = plan.units[0]
            provider = FakeProvider(
                scripted={unit.translation_unit_id: ["遗失公式", "仍然遗失", "还是遗失"]}
            )
            cache = TranslationCache(root / "cache.sqlite3")
            result = execute_translation_plan(
                plan,
                provider=provider,
                cache=cache,
                checkpoint_path=root / "manifest.json",
                config=RunnerConfig(base_delay_seconds=0, max_delay_seconds=0),
                sleep=lambda _: None,
            )
            self.assertEqual(provider.total_calls, 3)
            self.assertEqual(result.stats.failed, 1)
            self.assertEqual(cache.count(), 0)

    def test_nontranslatable_policies_never_call_provider(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = _make_plan(
                root,
                [
                    _block(0, block_type="reference", text="Smith (2024)"),
                    _block(1, block_type="image", text="", img_path="images/a.png"),
                    _block(2, block_type="mystery", text="Unknown"),
                ],
            )
            provider = FakeProvider()
            result = execute_translation_plan(
                plan,
                provider=provider,
                cache=TranslationCache(root / "cache.sqlite3"),
                checkpoint_path=root / "manifest.json",
            )
            self.assertEqual(provider.total_calls, 0)
            self.assertEqual(result.stats.preserved, 1)
            self.assertEqual(result.stats.skipped, 1)
            self.assertEqual(result.stats.needs_review, 1)

    def test_real_provider_like_unchanged_english_is_visible_as_needs_review(self):
        class LocalProvider(FakeProvider):
            name = "local-test"

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = _make_plan(root, [_block(0, text="This paragraph should be translated fully.")])
            provider = LocalProvider(transform=lambda request: request.protected_text)
            result = execute_translation_plan(
                plan,
                provider=provider,
                cache=TranslationCache(root / "cache.sqlite3"),
                checkpoint_path=root / "manifest.json",
            )
            unit_id = plan.units[0].translation_unit_id
            self.assertEqual(result.stats.qa_needs_review, 1)
            self.assertEqual(result.unit_states[unit_id]["execution_status"], "needs_review")
            codes = {flag.code for flag in result.quality_reports[unit_id].flags}
            self.assertIn("source_text_unchanged", codes)
            self.assertIn("target_language_missing", codes)

    def test_empty_translation_is_retried_then_failed_without_cache(self):
        class LocalProvider(FakeProvider):
            name = "local-test"

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = _make_plan(root, [_block(0)])
            provider = LocalProvider(transform=lambda request: "")
            cache = TranslationCache(root / "cache.sqlite3")
            result = execute_translation_plan(
                plan,
                provider=provider,
                cache=cache,
                checkpoint_path=root / "manifest.json",
                config=RunnerConfig(base_delay_seconds=0, max_delay_seconds=0),
                sleep=lambda _: None,
            )
            unit_id = plan.units[0].translation_unit_id
            self.assertEqual(provider.total_calls, 3)
            self.assertEqual(result.stats.failed, 1)
            self.assertEqual(cache.count(), 0)
            self.assertNotIn(unit_id, result.outputs)
            self.assertEqual(
                result.quality_reports[unit_id].flags[0].code,
                "empty_translation",
            )

    def test_prompt_wrapper_is_rejected_and_never_published(self):
        class LocalProvider(FakeProvider):
            name = "local-test"

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = _make_plan(root, [_block(0)])
            provider = LocalProvider(transform=lambda request: "Here is the translation: 译文")
            cache = TranslationCache(root / "cache.sqlite3")
            result = execute_translation_plan(
                plan,
                provider=provider,
                cache=cache,
                checkpoint_path=root / "manifest.json",
                config=RunnerConfig(max_attempts=1),
            )
            unit_id = plan.units[0].translation_unit_id
            self.assertEqual(result.stats.failed, 1)
            self.assertNotIn(unit_id, result.outputs)
            self.assertEqual(cache.count(), 0)
            self.assertEqual(
                result.quality_reports[unit_id].flags[0].code,
                "prompt_or_meta_leakage",
            )

    def test_translation_artifact_keeps_one_canonical_text_and_block_references(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_text = "First part second part."
            plan = _make_plan(
                root,
                [
                    _block(0, text="First part", logical_id=7, context=source_text),
                    _block(1, text="second part.", logical_id=7, context=source_text),
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
                requested_provider="deepseek",
                result=result,
                config=config,
            )
            self.assertEqual(artifact["schema_version"], "translation-v2")
            self.assertEqual(len(artifact["translation_units"]), 1)
            unit = artifact["translation_units"][0]
            self.assertEqual(unit["source_text"], source_text)
            self.assertEqual(unit["translation"]["status"], "simulated")
            self.assertEqual(unit["translation"]["selected_version"], "initial")
            self.assertEqual(len(artifact["items"]), 2)
            self.assertEqual(
                {item["translation_unit_id"] for item in artifact["items"]},
                {unit["translation_unit_id"]},
            )
            self.assertTrue(all("translated" in item for item in artifact["items"]))
            self.assertEqual(
                "".join(item["translated"] for item in artifact["items"]),
                unit["translation"]["selected_text"],
            )
            self.assertTrue(all("recommended_style" in item for item in artifact["items"]))
            self.assertEqual(
                artifact["layout_status"],
                "ready_for_reflow",
            )
            self.assertEqual(
                artifact["layout_run"]["contract_version"],
                "translation-reflow-v2",
            )
            self.assertEqual(
                artifact["golden_style_profile"]["statistics_basis"],
                "original_pdf",
            )
            self.assertFalse(
                artifact["golden_style_profile"]["translated_text_used_for_statistics"]
            )


class SimulationCliTests(unittest.TestCase):
    def test_simulate_writes_only_local_execution_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = _make_plan(root, [_block(0, text="Hello $x$")])
            source = plan.document.source_json
            output = root / "simulation"
            stdout = io.StringIO()
            stderr = io.StringIO()
            with patch.dict(os.environ, {}, clear=True):
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    code = main(
                        [
                            str(source),
                            "--config",
                            str(CONFIG_PATH),
                            "--env-file",
                            str(root / "missing.env"),
                            "--provider",
                            "deepseek",
                            "--output",
                            str(output),
                            "--simulate",
                        ]
                    )
            report = json.loads(stdout.getvalue())
            self.assertEqual(code, 0)
            self.assertEqual(report["mode"], "simulate")
            self.assertEqual(report["network_requests"], 0)
            self.assertEqual(report["simulation"]["fake_provider_calls"], 1)
            self.assertTrue((output / "translation_cache.sqlite3").exists())
            self.assertEqual(len(list(output.glob("*_translation_manifest.json"))), 1)
            translated_files = list(output.glob("*_translated.json"))
            self.assertEqual(len(translated_files), 1)
            artifact = json.loads(translated_files[0].read_text(encoding="utf-8"))
            self.assertEqual(artifact["schema_version"], "translation-v3")
            self.assertEqual(artifact["formula_assets"]["schema_version"], "formula-assets-v1")
            self.assertEqual(
                artifact["translation_units"][0]["translation"]["status"],
                "simulated",
            )
            self.assertIn("source_text", artifact["translation_units"][0])
            self.assertIn("initial_text", artifact["translation_units"][0]["translation"])
            self.assertEqual(artifact["translation_units"][0]["translation"]["candidates"], [])
            self.assertIn("translation_output", report["simulation"]["documents"][0])
            self.assertIn("FakeProvider", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
