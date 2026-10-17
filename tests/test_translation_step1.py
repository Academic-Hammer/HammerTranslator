from __future__ import annotations

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz

from hammer_translator.translation.cli import main
from hammer_translator.translation.config import (
    ConfigError,
    load_config,
    load_environment_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "translation_config.yaml"


class TranslationStep1Tests(unittest.TestCase):
    def run_cli(self, args: list[str], env: dict[str, str] | None = None):
        stdout = io.StringIO()
        stderr = io.StringIO()
        clean_env = {} if env is None else env
        with patch.dict(os.environ, clean_env, clear=True):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = main(args)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_inline_api_key_is_rejected_without_echoing_secret(self):
        secret = "do-not-print-this-key"
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "unsafe.yaml"
            path.write_text(
                "providers:\n"
                "  unsafe:\n"
                "    base_url: https://example.invalid/v1\n"
                "    model: test\n"
                f"    api_key: {secret}\n",
                encoding="utf-8",
            )
            with self.assertRaises(ConfigError) as caught:
                load_config(path)
        self.assertNotIn(secret, str(caught.exception))

    def test_check_requires_selected_provider_credential(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            code, stdout, stderr = self.run_cli(
                [
                    "--config",
                    str(CONFIG_PATH),
                    "--env-file",
                    str(Path(temp_dir) / "missing.env"),
                    "--provider",
                    "deepseek",
                    "--check",
                ]
            )
        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("TRANSLATION_API_KEY", stderr)

    def test_check_report_redacts_credential(self):
        secret = "unit-test-secret"
        code, stdout, stderr = self.run_cli(
            ["--config", str(CONFIG_PATH), "--provider", "deepseek", "--check"],
            {"TRANSLATION_API_KEY": secret},
        )
        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        self.assertIn('"api_key": "<redacted>"', stdout)
        self.assertNotIn(secret, stdout)

    def test_check_can_load_credential_from_explicit_env_file(self):
        secret = "dotenv-unit-test-secret"
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(f"TRANSLATION_API_KEY={secret}\n", encoding="utf-8")
            code, stdout, stderr = self.run_cli(
                [
                    "--config",
                    str(CONFIG_PATH),
                    "--env-file",
                    str(env_file),
                    "--provider",
                    "deepseek",
                    "--check",
                ]
            )
        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        self.assertIn('"credential_configured": true', stdout)
        self.assertIn('"exists": true', stdout)
        self.assertNotIn(secret, stdout)

    def test_process_environment_takes_precedence_over_env_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text("TRANSLATION_API_KEY=file-value\n", encoding="utf-8")
            environment = {"TRANSLATION_API_KEY": "process-value"}
            loaded = load_environment_file(env_file, environ=environment)
        self.assertEqual(loaded, ())
        self.assertEqual(environment["TRANSLATION_API_KEY"], "process-value")

    def test_dry_run_accepts_file_without_credential_and_makes_no_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "paper_enriched.json"
            source.write_text(
                '[{"type":"text","text":"Hello","page_idx":0,'
                '"bbox":[10,10,100,30],"logical_para_id":0}]',
                encoding="utf-8",
            )
            pdf = fitz.open()
            pdf.new_page(width=200, height=300)
            pdf.save(Path(temp_dir) / "paper_origin.pdf")
            pdf.close()
            output = Path(temp_dir) / "out"
            code, stdout, stderr = self.run_cli(
                [
                    str(source),
                    "--config",
                    str(CONFIG_PATH),
                    "--env-file",
                    str(Path(temp_dir) / "missing.env"),
                    "--provider",
                    "deepseek",
                    "--output",
                    str(output),
                    "--dry-run",
                ]
            )
            self.assertEqual(code, 0)
            self.assertIn('"network_requests": 0', stdout)
            self.assertIn("dry-run only", stderr)
            self.assertFalse(output.exists())

    def test_only_en_to_zh_cn_is_accepted(self):
        code, _, stderr = self.run_cli(
            [
                "--config",
                str(CONFIG_PATH),
                "--provider",
                "deepseek",
                "--source-language",
                "zh",
                "--target-language",
                "en",
                "--check",
            ],
            {"TRANSLATION_API_KEY": "test"},
        )
        self.assertEqual(code, 2)
        self.assertIn("only English to Simplified Chinese", stderr)


if __name__ == "__main__":
    unittest.main()
