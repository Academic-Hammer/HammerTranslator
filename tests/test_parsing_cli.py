from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import batch_parse_pdfs_mac as compatibility_shim
from hammer_translator.parsing import (
    RuntimeConfig,
    build_parser,
    collect_pdfs,
    config_from_args,
    configure_mineru_import,
    default_environment_profile,
    expected_parse_dir,
    normalize_document_name,
    preflight,
    select_pipeline_device,
)


def config(backend: str = "pipeline") -> RuntimeConfig:
    return RuntimeConfig(
        backend=backend,
        method="auto",
        language="ch",
        device="auto",
        formula=True,
        table=True,
        formula_ch=True,
        image_analysis=True,
        effort="medium",
        start_page=0,
        end_page=None,
        draw_bboxes=False,
        adapt=True,
        render_threads=4,
        window_size=8,
    )


class MacBatchParserTest(unittest.TestCase):
    def test_root_script_remains_a_compatible_thin_shim(self) -> None:
        self.assertIs(compatibility_shim.RuntimeConfig, RuntimeConfig)
        self.assertIs(compatibility_shim.collect_pdfs, collect_pdfs)

    def test_collect_pdfs_preserves_relative_parent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nested").mkdir()
            (root / "nested" / "paper.PDF").write_bytes(b"pdf")
            (root / "ignored.txt").write_text("x")
            files = collect_pdfs(root)
            self.assertEqual(files[0][0].name, "paper.PDF")
            self.assertEqual(files[0][1], Path("nested"))

    def test_output_directory_matches_mineru_contract(self) -> None:
        root = Path("out")
        self.assertEqual(expected_parse_dir(root, "paper", config()), root / "paper" / "auto")
        self.assertEqual(
            expected_parse_dir(root, "paper", config("vlm-engine")), root / "paper" / "vlm"
        )
        self.assertEqual(
            expected_parse_dir(root, "paper", config("hybrid-engine")),
            root / "paper" / "hybrid_auto",
        )

    def test_cli_defaults_are_memory_conservative(self) -> None:
        args = build_parser().parse_args([])
        runtime = config_from_args(args)
        self.assertEqual(runtime.backend, "pipeline")
        self.assertEqual(runtime.device, "auto")
        self.assertEqual(runtime.window_size, 8)
        self.assertEqual(runtime.render_threads, 4)
        self.assertTrue(runtime.adapt)

    def test_invalid_page_range_is_rejected(self) -> None:
        args = build_parser().parse_args(["--start-page", "3", "--end-page", "2"])
        with self.assertRaisesRegex(ValueError, "end-page"):
            config_from_args(args)

    def test_cli_accepts_linux_cuda_device(self) -> None:
        args = build_parser().parse_args(["--device", "cuda"])
        self.assertEqual(config_from_args(args).device, "cuda")

    def test_auto_device_is_platform_aware(self) -> None:
        self.assertEqual(
            select_pipeline_device(
                "auto",
                system="Darwin",
                accelerators={"mps_available": True, "cuda_available": False},
            ),
            "mps",
        )
        self.assertEqual(
            select_pipeline_device(
                "auto",
                system="Linux",
                accelerators={"mps_available": False, "cuda_available": True},
            ),
            "cuda",
        )
        self.assertEqual(default_environment_profile("Linux"), "linux_gpu")

    def test_linux_vlm_preflight_requires_vllm(self) -> None:
        accelerators = {
            "mps_available": False,
            "cuda_available": True,
            "cuda_device_count": 1,
            "cuda_devices": ["Test GPU"],
            "mlx_vlm_available": False,
        }
        with patch("hammer_translator.parsing.cli.configure_mineru_import"), patch(
            "hammer_translator.parsing.cli.platform.system", return_value="Linux"
        ), patch(
            "hammer_translator.parsing.cli.inspect_accelerators",
            return_value=accelerators,
        ), patch(
            "hammer_translator.parsing.cli.module_available",
            side_effect=lambda name: name != "vllm",
        ):
            ready, messages = preflight(config("vlm-engine"))

        self.assertFalse(ready)
        self.assertIn("MISSING  vllm", "\n".join(messages))

    def test_explicit_document_name_is_safe_and_stable(self) -> None:
        self.assertEqual(normalize_document_name("A paper (final).pdf"), "A_paper_final")
        with self.assertRaisesRegex(ValueError, "plain filename"):
            normalize_document_name("../paper.pdf")

    def test_installed_mineru_does_not_require_repository_source(self) -> None:
        with patch.dict(os.environ, {}, clear=False), patch(
            "hammer_translator.parsing.cli.module_available", return_value=True
        ):
            os.environ.pop("HAMMERTRANSLATOR_MINERU_SOURCE", None)
            self.assertIsNone(configure_mineru_import())

    def test_explicit_mineru_source_override_is_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            with patch.dict(
                os.environ,
                {"HAMMERTRANSLATOR_MINERU_SOURCE": str(source)},
                clear=False,
            ):
                with self.assertRaisesRegex(FileNotFoundError, "does not point"):
                    configure_mineru_import()


if __name__ == "__main__":
    unittest.main()
