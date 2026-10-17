from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hammer_translator.runtime_paths import (
    APP_HOME_ENV,
    CONFIG_PATH_ENV,
    ENV_FILE_ENV,
    FONTS_DIR_ENV,
    FORMULA_RUNTIME_DIR_ENV,
    JOBS_ROOT_ENV,
    MODEL_CACHE_DIR_ENV,
    TEMP_DIR_ENV,
    child_process_cwd,
    portable_resource_label,
    runtime_paths,
)


class RuntimePathsTests(unittest.TestCase):
    def test_operator_overrides_are_resolved_centrally(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            values = {
                JOBS_ROOT_ENV: str(root / "jobs"),
                CONFIG_PATH_ENV: str(root / "config.yaml"),
                ENV_FILE_ENV: str(root / "secrets.env"),
                FONTS_DIR_ENV: str(root / "fonts"),
                FORMULA_RUNTIME_DIR_ENV: str(root / "formula"),
                MODEL_CACHE_DIR_ENV: str(root / "models"),
                TEMP_DIR_ENV: str(root / "temp"),
            }
            paths = runtime_paths(values)

            self.assertEqual(paths.jobs_root, (root / "jobs").resolve())
            self.assertEqual(paths.config_path, (root / "config.yaml").resolve())
            self.assertEqual(paths.env_file, (root / "secrets.env").resolve())
            self.assertEqual(paths.fonts_dir, (root / "fonts").resolve())
            self.assertEqual(paths.formula_runtime_dir, (root / "formula").resolve())
            self.assertEqual(paths.model_cache_dir, (root / "models").resolve())
            self.assertEqual(paths.temp_dir, (root / "temp").resolve())

    def test_explicit_application_home_moves_operator_owned_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = runtime_paths({APP_HOME_ENV: str(root)})

            self.assertEqual(paths.jobs_root, (root / "var" / "jobs").resolve())
            self.assertEqual(paths.env_file, (root / ".env").resolve())
            self.assertEqual(paths.fonts_dir, (root / "fonts").resolve())

    def test_uninstalled_source_process_uses_checkout_as_import_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "hammer_translator.runtime_paths.distribution_is_installed",
                return_value=False,
            ):
                selected = child_process_cwd(Path(temp_dir))
        self.assertEqual(selected, runtime_paths().source_root)

    def test_source_resource_label_preserves_existing_artifact_value(self) -> None:
        source_root = runtime_paths().source_root
        self.assertIsNotNone(source_root)
        assert source_root is not None
        self.assertEqual(
            portable_resource_label(source_root / "formula_runtime" / "bridge.mjs"),
            "formula_runtime/bridge.mjs",
        )


if __name__ == "__main__":
    unittest.main()
