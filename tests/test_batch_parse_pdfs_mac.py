from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from batch_parse_pdfs_mac import (
    RuntimeConfig,
    collect_pdfs,
    config_from_args,
    expected_parse_dir,
    build_parser,
    normalize_document_name,
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

    def test_explicit_document_name_is_safe_and_stable(self) -> None:
        self.assertEqual(normalize_document_name("A paper (final).pdf"), "A_paper_final")
        with self.assertRaisesRegex(ValueError, "plain filename"):
            normalize_document_name("../paper.pdf")


if __name__ == "__main__":
    unittest.main()
