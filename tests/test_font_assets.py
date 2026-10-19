from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hammer_translator.font_assets import (
    FONT_DIRECTORY_ENV,
    FontAssetError,
    canonical_project_relative_path,
    font_manifest,
    required_font_path,
    resolve_registered_font,
)


class FontAssetTests(unittest.TestCase):
    def test_default_fonts_are_packaged_and_verified(self) -> None:
        self.assertEqual(
            set(font_manifest()["fonts"]),
            {
                "LXGWWenKai-Medium.ttf",
                "LXGWWenKai-Regular.ttf",
            },
        )
        for category in ("title", "heading", "body", "caption"):
            path = required_font_path(category)
            self.assertTrue(path.is_file())
            self.assertEqual(path.parent.name, "fonts")
            self.assertEqual(canonical_project_relative_path(category), "")

    def test_custom_font_directory_can_supply_a_pinned_legacy_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            font = directory / "Legacy.ttf"
            font.write_bytes(b"legacy font fixture")
            digest = hashlib.sha256(font.read_bytes()).hexdigest()
            (directory / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": "hammertranslator-font-assets-v1",
                        "fonts": {
                            "Legacy.ttf": {
                                "categories": ["body"],
                                "sha256": digest,
                            }
                        },
                    }
                )
            )
            with patch.dict(os.environ, {FONT_DIRECTORY_ENV: str(directory)}):
                self.assertEqual(required_font_path("body"), font.resolve())
                self.assertEqual(font_manifest()["fonts"]["Legacy.ttf"]["sha256"], digest)

    def test_registered_font_uses_configured_directory_after_checkout_moves(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            font = directory / "Fixture.ttf"
            font.write_bytes(b"font fixture")
            digest = hashlib.sha256(font.read_bytes()).hexdigest()
            record = {
                "path": "/missing/old-checkout/Fixture.ttf",
                "project_relative_path": "assets/fonts/Fixture.ttf",
                "sha256": digest,
            }
            with patch.dict(os.environ, {FONT_DIRECTORY_ENV: str(directory)}):
                self.assertEqual(resolve_registered_font(record), font.resolve())

    def test_registered_font_rejects_silent_substitution(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            font = directory / "Fixture.ttf"
            font.write_bytes(b"changed font")
            record = {
                "path": str(font),
                "project_relative_path": "assets/fonts/Fixture.ttf",
                "sha256": "0" * 64,
            }
            with self.assertRaisesRegex(FontAssetError, "hash changed"):
                resolve_registered_font(record)


if __name__ == "__main__":
    unittest.main()
