from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import fitz

from hammer_translator.smoke import SmokeTestError, run_core_smoke


class InstalledCoreSmokeTests(unittest.TestCase):
    def test_offline_smoke_runs_translation_and_reflow_modules(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir) / "work"
            report = run_core_smoke(work_dir)

            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["network_requests"], 0)
            self.assertFalse(report["parser_tested"])
            self.assertEqual(report["translation_schema"], "translation-v3")
            self.assertEqual(report["reflow_schema"], "reflow-feedback-v2")
            self.assertTrue(report["page_geometry_preserved"])
            output = fitz.open(report["artifacts"]["pdf"])
            try:
                self.assertEqual(output.page_count, 1)
            finally:
                output.close()

    def test_smoke_refuses_to_overwrite_reserved_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            (work_dir / "smoke_source.pdf").write_bytes(b"existing")

            with self.assertRaisesRegex(SmokeTestError, "reserved"):
                run_core_smoke(work_dir)


if __name__ == "__main__":
    unittest.main()
