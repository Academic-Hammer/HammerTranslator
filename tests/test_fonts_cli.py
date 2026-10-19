from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hammer_translator.fonts_cli import FontInstallError, install_default_fonts


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class FontInstallerTests(unittest.TestCase):
    def test_install_is_verified_idempotent_and_preserves_custom_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            resources = root / "resources"
            resources.mkdir()
            fonts = root / "fonts"
            payload = b"open font fixture"
            manifest = {
                "schema_version": "hammertranslator-font-assets-v1",
                "fonts": {
                    "Fixture.ttf": {
                        "categories": ["body"],
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "size_bytes": len(payload),
                        "license_file": "OFL.txt",
                        "source": "https://raw.githubusercontent.com/example/fonts/commit/Fixture.ttf",
                    }
                },
            }
            (resources / "manifest.json").write_text(json.dumps(manifest))
            (resources / "OFL.txt").write_text("license fixture")
            with patch(
                "hammer_translator.fonts_cli._bundled_manifest",
                return_value=(resources, manifest),
            ), patch(
                "hammer_translator.fonts_cli.runtime_paths",
                return_value=SimpleNamespace(fonts_dir=fonts),
            ), patch(
                "hammer_translator.fonts_cli.urllib.request.urlopen",
                side_effect=lambda *_args, **_kwargs: _Response(payload),
            ) as urlopen:
                first = install_default_fonts()
                self.assertIn("SHA-256 verified", first[-1])
                self.assertEqual((fonts / "Fixture.ttf").read_bytes(), payload)
                self.assertTrue((fonts / "manifest.json").is_file())
                self.assertEqual((fonts / "OFL.txt").read_text(), "license fixture")
                self.assertEqual(len(first), 2)
                second = install_default_fonts()
                self.assertIn("already verified", second[0])
                self.assertEqual(urlopen.call_count, 1)

                (fonts / "Fixture.ttf").write_bytes(b"different")
                with self.assertRaisesRegex(FontInstallError, "does not match"):
                    install_default_fonts()
                install_default_fonts(repair=True)
                self.assertEqual((fonts / "Fixture.ttf").read_bytes(), payload)

    def test_bad_download_does_not_install_a_font(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            resources = root / "resources"
            resources.mkdir()
            fonts = root / "fonts"
            manifest = {
                "schema_version": "hammertranslator-font-assets-v1",
                "fonts": {
                    "Fixture.ttf": {
                        "sha256": hashlib.sha256(b"expected").hexdigest(),
                        "size_bytes": 8,
                        "license_file": "OFL.txt",
                        "source": "https://raw.githubusercontent.com/example/fonts/commit/Fixture.ttf",
                    }
                },
            }
            (resources / "manifest.json").write_text(json.dumps(manifest))
            (resources / "OFL.txt").write_text("license fixture")
            with patch(
                "hammer_translator.fonts_cli._bundled_manifest",
                return_value=(resources, manifest),
            ), patch(
                "hammer_translator.fonts_cli.runtime_paths",
                return_value=SimpleNamespace(fonts_dir=fonts),
            ), patch(
                "hammer_translator.fonts_cli.urllib.request.urlopen",
                side_effect=lambda *_args, **_kwargs: _Response(b"bad data"),
            ):
                with self.assertRaisesRegex(FontInstallError, "verification failed"):
                    install_default_fonts()
            self.assertFalse((fonts / "Fixture.ttf").exists())
            self.assertFalse((fonts / "manifest.json").exists())

    def test_offline_install_verifies_source_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            resources = root / "resources"
            resources.mkdir()
            source_dir = root / "downloaded"
            source_dir.mkdir()
            fonts = root / "fonts"
            payload = b"verified offline font"
            (source_dir / "Fixture.ttf").write_bytes(payload)
            manifest = {
                "schema_version": "hammertranslator-font-assets-v1",
                "fonts": {
                    "Fixture.ttf": {
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "size_bytes": len(payload),
                        "license_file": "OFL.txt",
                        "source": "https://raw.githubusercontent.com/example/fonts/commit/Fixture.ttf",
                    }
                },
            }
            (resources / "manifest.json").write_text(json.dumps(manifest))
            (resources / "OFL.txt").write_text("license fixture")
            with patch(
                "hammer_translator.fonts_cli._bundled_manifest",
                return_value=(resources, manifest),
            ), patch(
                "hammer_translator.fonts_cli.runtime_paths",
                return_value=SimpleNamespace(fonts_dir=fonts),
            ), patch("hammer_translator.fonts_cli.urllib.request.urlopen") as urlopen:
                install_default_fonts(source_dir=source_dir)
                urlopen.assert_not_called()
                self.assertEqual((fonts / "Fixture.ttf").read_bytes(), payload)


if __name__ == "__main__":
    unittest.main()
