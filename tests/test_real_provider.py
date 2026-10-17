from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import fitz

from hammer_translator.reflow.renderer import ThinReflowRenderer
from hammer_translator.translation.cli import main
from hammer_translator.translation.provider import (
    FakeProvider,
    OpenAICompatibleProvider,
    ProviderError,
    TranslationRequest,
    build_translation_messages,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "translation_config.yaml"


def _request(*, mode: str = "initial", constraints: dict | None = None):
    return TranslationRequest(
        translation_unit_id="tu-real",
        protected_text="Value [[HTP_ABCD1234_0001]] is stable.",
        source_language="en",
        target_language="zh-CN",
        element_type="text",
        provider="deepseek",
        model="deepseek-flash",
        prompt_version="translation-prompt-v1",
        glossary_version="none",
        mode=mode,
        constraints=constraints or {},
    )


class _RecordingCompletions:
    def __init__(self, *, error: BaseException | None = None):
        self.error = error
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            id="request-123",
            model="deepseek-flash",
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content="数值 [[HTP_ABCD1234_0001]] 是稳定的。"
                    ),
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=30,
                completion_tokens=12,
                total_tokens=42,
            ),
        )


class _Client:
    def __init__(self, completions: _RecordingCompletions):
        self.chat = SimpleNamespace(completions=completions)


class RealProviderTests(unittest.TestCase):
    def test_deepseek_request_is_non_thinking_and_returns_safe_usage(self):
        completions = _RecordingCompletions()
        provider = OpenAICompatibleProvider(
            name="deepseek",
            base_url="https://api.deepseek.com",
            model="deepseek-flash",
            api_key="not-a-real-key",
            client=_Client(completions),
        )
        response = provider.translate(_request())
        self.assertEqual(response.text, "数值 [[HTP_ABCD1234_0001]] 是稳定的。")
        self.assertEqual(response.request_id, "request-123")
        self.assertEqual(response.metadata["total_tokens"], 42)
        self.assertEqual(provider.usage_summary()["total_tokens"], 42)
        call = completions.calls[0]
        self.assertEqual(call["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertFalse(call["stream"])
        self.assertNotIn("not-a-real-key", json.dumps(call))

    def test_constrained_prompt_uses_source_and_limit_not_unprotected_draft(self):
        messages = build_translation_messages(
            _request(
                mode="constrained",
                constraints={
                    "target_max_characters": 18,
                    "current_translation": "不应放入提示词的旧译文 $x$",
                },
            )
        )
        prompt = json.dumps(messages, ensure_ascii=False)
        self.assertIn("18 visible characters", prompt)
        self.assertIn("[[HTP_ABCD1234_0001]]", prompt)
        self.assertNotIn("不应放入提示词", prompt)

    def test_rate_limit_is_classified_without_remote_body(self):
        class RateLimitFailure(Exception):
            status_code = 429
            response = SimpleNamespace(headers={"retry-after": "2.5"})

        completions = _RecordingCompletions(
            error=RateLimitFailure("Bearer sk-do-not-log-this-secret")
        )
        provider = OpenAICompatibleProvider(
            name="deepseek",
            base_url="https://api.deepseek.com",
            model="deepseek-flash",
            api_key="not-a-real-key",
            client=_Client(completions),
        )
        with self.assertRaises(ProviderError) as caught:
            provider.translate(_request())
        error = caught.exception
        self.assertEqual(error.code, "rate_limit")
        self.assertTrue(error.retryable)
        self.assertEqual(error.retry_after, 2.5)
        self.assertNotIn("sk-do-not-log", str(error))


class RealCliTests(unittest.TestCase):
    @staticmethod
    def _source(root: Path) -> Path:
        source = root / "paper_enriched.json"
        source.write_text(
            json.dumps(
                [
                    {
                        "type": "text",
                        "text": "Translate this sentence.",
                        "page_idx": 0,
                        "bbox": [10, 10, 190, 50],
                        "logical_para_id": 0,
                    }
                ]
            ),
            encoding="utf-8",
        )
        pdf = fitz.open()
        pdf.new_page(width=200, height=100)
        pdf.save(root / "paper_origin.pdf")
        pdf.close()
        return source

    def test_run_requires_credential_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self._source(root)
            output = root / "real-output"
            stderr = io.StringIO()
            with patch.dict(os.environ, {}, clear=True):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
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
                            "--run",
                        ]
                    )
            self.assertEqual(code, 2)
            self.assertIn("TRANSLATION_API_KEY", stderr.getvalue())
            self.assertFalse(output.exists())

    def test_run_executes_real_provider_path_and_reports_requests(self):
        class StubRemoteProvider(FakeProvider):
            name = "deepseek"

            def __init__(self):
                super().__init__(
                    model="deepseek-flash",
                    transform=lambda request: "这个句子已翻译。",
                )

            def usage_summary(self):
                return {
                    "successful_responses": self.total_calls,
                    "input_tokens": 10,
                    "output_tokens": 8,
                    "total_tokens": 18,
                }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self._source(root)
            output = root / "real-output"
            stdout = io.StringIO()
            stderr = io.StringIO()
            stub = StubRemoteProvider()
            with patch.dict(
                os.environ,
                {"TRANSLATION_API_KEY": "unit-test-only"},
                clear=True,
            ), patch(
                "hammer_translator.translation.cli.OpenAICompatibleProvider",
                return_value=stub,
            ):
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    code = main(
                        [
                            str(source),
                            "--config",
                            str(CONFIG_PATH),
                            "--provider",
                            "deepseek",
                            "--output",
                            str(output),
                            "--run",
                        ]
                    )
            report = json.loads(stdout.getvalue())
            self.assertEqual(code, 0)
            progress = stderr.getvalue()
            self.assertIn("1/1 text: API request 1/3", progress)
            self.assertIn("planning fixed-bbox layout", progress)
            self.assertNotIn("unit-test-only", progress)
            self.assertEqual(report["mode"], "run")
            self.assertEqual(report["network_requests"], 1)
            self.assertEqual(report["execution"]["provider"], "deepseek")
            self.assertEqual(report["execution"]["usage"]["total_tokens"], 18)
            artifact_path = next(output.glob("*_translated.json"))
            artifact_text = artifact_path.read_text(encoding="utf-8")
            artifact = json.loads(artifact_text)
            self.assertEqual(
                artifact["translation_units"][0]["translation"]["status"],
                "translated",
            )
            self.assertNotIn("unit-test-only", artifact_text)
            pdf_output = root / "translated.pdf"
            reflow_report = ThinReflowRenderer().render(artifact_path, pdf_output)
            self.assertEqual(reflow_report["status"], "rendered")
            self.assertEqual(reflow_report["summary"]["rendered"], 1)
            rendered = fitz.open(pdf_output)
            try:
                self.assertEqual(len(rendered), 1)
                self.assertIn("这个句子已翻译", rendered[0].get_text())
            finally:
                rendered.close()


if __name__ == "__main__":
    unittest.main()
