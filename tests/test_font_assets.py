from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hammer_translator.font_assets import (
    FONT_DIRECTORY_ENV,
    FontAssetError,
    canonical_project_relative_path,
    required_font_path,
    resolve_registered_font,
)


class FontAssetTests(unittest.TestCase):
    def test_required_v1_fonts_are_project_owned_and_verified(self) -> None:
        for category in ("title", "heading", "body", "caption"):
            path = required_font_path(category)
            self.assertTrue(path.is_file())
            self.assertEqual(path.parent.name, "fonts")
            self.assertTrue(canonical_project_relative_path(category).startswith("assets/fonts/"))

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
