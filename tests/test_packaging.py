from __future__ import annotations

import tomllib
import unittest
from importlib import import_module
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PackagingContractTest(unittest.TestCase):
    def test_application_metadata_and_commands_are_declared(self) -> None:
        with (PROJECT_ROOT / "pyproject.toml").open("rb") as stream:
            data = tomllib.load(stream)
        project = data["project"]
        self.assertEqual(project["version"], "1.0.0")
        self.assertEqual(project["requires-python"], ">=3.11,<3.14")
        self.assertEqual(
            set(project["scripts"]),
            {
                "hammer-translator",
                "hammer-parse",
                "hammer-translate",
                "hammer-reflow",
                "hammer-mineru-adapt",
                "hammer-doctor",
                "hammer-smoke",
            },
        )
        for target in project["scripts"].values():
            module_name, attribute = target.split(":", maxsplit=1)
            self.assertTrue(callable(getattr(import_module(module_name), attribute)))
        self.assertEqual(project["readme"], "README.md")
        self.assertTrue((PROJECT_ROOT / project["readme"]).is_file())

    def test_installed_runtime_resources_are_declared_and_match_source(self) -> None:
        package_resources = PROJECT_ROOT / "hammer_translator" / "resources"
        pairs = (
            (
                PROJECT_ROOT / "translation_config.yaml",
                package_resources / "translation_config.yaml",
            ),
            (
                PROJECT_ROOT / "assets" / "fonts" / "manifest.json",
                package_resources / "fonts" / "manifest.json",
            ),
            (
                PROJECT_ROOT / "formula_runtime" / "mathjax_to_svg.mjs",
                package_resources / "formula_runtime" / "mathjax_to_svg.mjs",
            ),
            (
                PROJECT_ROOT / "formula_runtime" / "package.json",
                package_resources / "formula_runtime" / "package.json",
            ),
            (
                PROJECT_ROOT / "formula_runtime" / "package-lock.json",
                package_resources / "formula_runtime" / "package-lock.json",
            ),
        )
        for source, packaged in pairs:
            self.assertEqual(source.read_bytes(), packaged.read_bytes())

    def test_core_does_not_contain_parser_or_gpu_frameworks(self) -> None:
        pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
        core = (PROJECT_ROOT / "requirements" / "core.txt").read_text(encoding="utf-8").lower()
        combined = pyproject + core
        for package in ("mineru[", "torch==", "mlx==", "vllm==", "nvidia-"):
            self.assertNotIn(package, combined)

    def test_platform_overlays_pin_mineru_344(self) -> None:
        macos = (PROJECT_ROOT / "requirements" / "macos.txt").read_text(encoding="utf-8")
        linux = (PROJECT_ROOT / "requirements" / "linux-gpu.txt").read_text(encoding="utf-8")
        macos_parser = (PROJECT_ROOT / "requirements" / "macos-parser.txt").read_text(
            encoding="utf-8"
        )
        linux_parser = (
            PROJECT_ROOT / "requirements" / "linux-gpu-parser.txt"
        ).read_text(encoding="utf-8")
        self.assertIn("-r macos-parser.txt", macos)
        self.assertIn("-r linux-gpu-parser.txt", linux)
        self.assertIn("mineru[pipeline,mlx]==3.4.4", macos_parser)
        self.assertIn("mineru[pipeline,vllm]==3.4.4", linux_parser)
        self.assertNotIn("vllm", macos_parser.lower())
        self.assertNotIn("mlx", linux_parser.lower())

    def test_old_linux_freeze_is_preserved_but_not_active(self) -> None:
        legacy = PROJECT_ROOT / "legacy" / "requirements" / "linux-cuda-freeze-pre-v1.txt"
        self.assertTrue(legacy.is_file())
        self.assertIn("nvidia-cuda-runtime", legacy.read_text(encoding="utf-8"))
        active_lines = [
            line
            for line in (PROJECT_ROOT / "requirements.txt")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip() and not line.startswith("#")
        ]
        self.assertEqual(active_lines, ["-r requirements/core.txt"])


if __name__ == "__main__":
    unittest.main()
