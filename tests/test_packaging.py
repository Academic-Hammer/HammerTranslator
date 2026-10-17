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
            },
        )
        for target in project["scripts"].values():
            module_name, attribute = target.split(":", maxsplit=1)
            self.assertTrue(callable(getattr(import_module(module_name), attribute)))

    def test_core_does_not_contain_parser_or_gpu_frameworks(self) -> None:
        pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
        core = (PROJECT_ROOT / "requirements" / "core.txt").read_text(encoding="utf-8").lower()
        combined = pyproject + core
        for package in ("mineru[", "torch==", "mlx==", "vllm==", "nvidia-"):
            self.assertNotIn(package, combined)

    def test_platform_overlays_pin_mineru_344(self) -> None:
        macos = (PROJECT_ROOT / "requirements" / "macos.txt").read_text(encoding="utf-8")
        linux = (PROJECT_ROOT / "requirements" / "linux-gpu.txt").read_text(encoding="utf-8")
        self.assertIn("mineru[pipeline,mlx]==3.4.4", macos)
        self.assertIn("mineru[pipeline,vllm]==3.4.4", linux)
        self.assertNotIn("vllm", macos.lower())
        self.assertNotIn("mlx", linux.lower())

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
