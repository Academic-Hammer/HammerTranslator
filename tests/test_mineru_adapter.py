from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from mineru_adapter import AdapterOptions, MinerUAdapter


def line(text: str, bbox: list[int], span_type: str = "text") -> dict:
    return {"bbox": bbox, "spans": [{"type": span_type, "content": text, "bbox": bbox}]}


def cross_page_line(text: str, bbox: list[int]) -> dict:
    return {
        "bbox": bbox,
        "spans": [
            {"type": "text", "content": text, "bbox": bbox, "cross_page": True}
        ],
    }


class MinerUAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.middle = {
            "pdf_info": [
                {
                    "page_idx": 0,
                    "page_size": [600, 800],
                    "para_blocks": [
                        {"type": "title", "level": 1, "index": 1, "bbox": [50, 50, 400, 80], "lines": [line("A Paper", [50, 50, 400, 80])]},
                        {"type": "text", "index": 2, "bbox": [50, 90, 400, 110], "lines": [line("Alice and Bob", [50, 90, 400, 110])]},
                        {"type": "title", "level": 1, "index": 3, "bbox": [50, 120, 180, 140], "lines": [line("Abstract", [50, 120, 180, 140])]},
                        {"type": "text", "index": 4, "bbox": [50, 150, 400, 180], "lines": [line("This continues", [50, 150, 400, 180])]},
                        {"type": "text", "index": 5, "bbox": [50, 185, 400, 210], "lines": [line("on the next block.", [50, 185, 400, 210])]},
                        {"type": "text", "index": 6, "bbox": [50, 220, 300, 240], "lines": [line("https://example.com", [50, 220, 300, 240])]},
                        {"type": "abstract", "index": 7, "bbox": [50, 250, 400, 280], "lines": [line("Abstract body.", [50, 250, 400, 280])]},
                    ],
                    "discarded_blocks": [
                        {"type": "header", "index": 0, "bbox": [50, 10, 300, 30], "lines": [line("Conference", [50, 10, 300, 30])]},
                    ],
                },
                {
                    "page_idx": 1,
                    "page_size": [600, 800],
                    "para_blocks": [
                        {"type": "title", "level": 1, "index": 8, "bbox": [50, 50, 200, 70], "lines": [line("References", [50, 50, 200, 70])]},
                        {"type": "text", "index": 9, "bbox": [50, 80, 400, 100], "lines": [line("Smith. A work", [50, 80, 400, 100])]},
                        {"type": "text", "index": 10, "bbox": [50, 105, 400, 125], "lines": [line("continued here", [50, 105, 400, 125])]},
                    ],
                    "discarded_blocks": [],
                },
            ]
        }

    def test_contract_classification_and_grouping(self) -> None:
        adapter = MinerUAdapter(AdapterOptions(enrich_styles=False))
        result = adapter.adapt(self.middle)
        items = result.content_list
        self.assertEqual(items[0]["type"], "header")
        self.assertEqual(items[1]["text_level"], 1)
        self.assertEqual(items[2]["type"], "author")
        self.assertEqual(items[6]["type"], "url")
        self.assertEqual(items[7]["type"], "text")
        self.assertEqual(items[7]["sub_type"], "abstract")
        self.assertEqual(items[4]["logical_para_id"], items[5]["logical_para_id"])
        self.assertEqual(items[4]["context"], "This continues on the next block.")
        self.assertEqual(items[-3]["type"], "reference")
        self.assertEqual(items[-2]["type"], "reference")
        self.assertEqual(items[-1]["logical_para_id"], items[-2]["logical_para_id"])
        self.assertEqual(items[1]["bbox"], [50, 50, 400, 80])

    def test_writes_stable_filenames(self) -> None:
        adapter = MinerUAdapter(AdapterOptions(enrich_styles=False))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "paper_middle.json"
            source.write_text(json.dumps(self.middle), encoding="utf-8")
            result = adapter.adapt_file(source)
            output = root / "paper_content_list.json"
            self.assertEqual(result.output_files["content_list"], output)
            self.assertTrue(output.exists())
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), result.content_list)

    def test_bad_middle_json_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "pdf_info"):
            MinerUAdapter(AdapterOptions(enrich_styles=False)).adapt({})

    def test_same_column_coordinate_reset_moves_continuation_to_next_page(self) -> None:
        middle = {
            "pdf_info": [
                {
                    "page_idx": 0,
                    "page_size": [600, 800],
                    "para_blocks": [
                        {
                            "type": "text",
                            "index": 1,
                            "bbox": [50, 700, 400, 740],
                            "lines": [
                                line("A paragraph that", [50, 700, 400, 715]),
                                line("starts on page one", [50, 720, 400, 735]),
                                line("and continues on page two.", [50, 100, 400, 115]),
                            ],
                        }
                    ],
                    "discarded_blocks": [],
                },
                {
                    "page_idx": 1,
                    "page_size": [600, 800],
                    "para_blocks": [],
                    "discarded_blocks": [],
                },
            ]
        }

        result = MinerUAdapter(AdapterOptions(enrich_styles=False)).adapt(middle)

        self.assertEqual(len(result.content_list), 2)
        self.assertEqual(result.content_list[0]["page_idx"], 0)
        self.assertEqual(result.content_list[1]["page_idx"], 1)
        self.assertTrue(result.content_list[1]["cross_page"])
        self.assertNotIn("cross_column", result.content_list[1])
        self.assertEqual(result.content_list[1]["bbox"], [50, 100, 400, 115])

    def test_horizontal_jump_remains_a_same_page_column_split(self) -> None:
        middle = {
            "pdf_info": [
                {
                    "page_idx": 0,
                    "page_size": [600, 800],
                    "para_blocks": [
                        {
                            "type": "text",
                            "index": 1,
                            "bbox": [50, 500, 250, 540],
                            "lines": [
                                line("Left column", [50, 500, 250, 515]),
                                line("Right column", [330, 100, 550, 115]),
                            ],
                        }
                    ],
                    "discarded_blocks": [],
                }
            ]
        }

        result = MinerUAdapter(AdapterOptions(enrich_styles=False)).adapt(middle)

        self.assertEqual(result.content_list[1]["page_idx"], 0)
        self.assertTrue(result.content_list[1]["cross_column"])
        self.assertNotIn("cross_page", result.content_list[1])

    def test_cross_page_marked_lines_move_to_next_page(self) -> None:
        middle = {
            "pdf_info": [
                {
                    "page_idx": 0,
                    "page_size": [600, 800],
                    "para_blocks": [
                        {
                            "type": "text",
                            "index": 1,
                            "bbox": [320, 600, 560, 720],
                            "lines": [
                                line("Right column body", [320, 600, 560, 615]),
                                line("and the paragraph ends here", [320, 620, 560, 635]),
                                cross_page_line("continuation at next page top", [50, 260, 300, 275]),
                            ],
                        }
                    ],
                    "discarded_blocks": [],
                },
                {
                    "page_idx": 1,
                    "page_size": [600, 800],
                    "para_blocks": [
                        {
                            "type": "text",
                            "index": 4,
                            "bbox": [50, 260, 300, 275],
                            "lines": [],
                            "lines_deleted": True,
                        }
                    ],
                    "discarded_blocks": [],
                },
            ]
        }

        result = MinerUAdapter(AdapterOptions(enrich_styles=False)).adapt(middle)

        self.assertEqual(len(result.content_list), 2)
        self.assertEqual(result.content_list[0]["page_idx"], 0)
        self.assertNotIn("continuation", result.content_list[0]["text"])
        self.assertEqual(result.content_list[1]["page_idx"], 1)
        self.assertTrue(result.content_list[1]["cross_page"])
        self.assertEqual(result.content_list[1]["bbox"], [50, 260, 300, 275])
        self.assertIn("continuation at next page top", result.content_list[1]["text"])

    def test_overlapping_cross_page_lines_move_to_next_page(self) -> None:
        middle = {
            "pdf_info": [
                {
                    "page_idx": 0,
                    "page_size": [600, 800],
                    "para_blocks": [
                        {
                            "type": "text",
                            "index": 1,
                            "bbox": [50, 100, 400, 140],
                            "lines": [
                                line("page one line one", [50, 100, 400, 115]),
                                line("page one line two", [50, 120, 400, 135]),
                                cross_page_line("page two line one", [50, 100, 400, 115]),
                                cross_page_line("page two line two", [50, 120, 400, 135]),
                            ],
                        }
                    ],
                    "discarded_blocks": [],
                },
                {
                    "page_idx": 1,
                    "page_size": [600, 800],
                    "para_blocks": [],
                    "discarded_blocks": [],
                },
            ]
        }

        result = MinerUAdapter(AdapterOptions(enrich_styles=False)).adapt(middle)

        self.assertEqual(result.content_list[0]["page_idx"], 0)
        self.assertEqual(
            result.content_list[0]["text"], "page one line one page one line two"
        )
        self.assertEqual(result.content_list[1]["page_idx"], 1)
        self.assertTrue(result.content_list[1]["cross_page"])
        self.assertEqual(
            result.content_list[1]["text"], "page two line one page two line two"
        )

    def test_cross_page_marked_lines_are_dropped_when_next_page_has_text(self) -> None:
        middle = {
            "pdf_info": [
                {
                    "page_idx": 0,
                    "page_size": [600, 800],
                    "para_blocks": [
                        {
                            "type": "text",
                            "index": 1,
                            "bbox": [320, 600, 560, 720],
                            "lines": [
                                line("Right column body", [320, 600, 560, 615]),
                                line("and the paragraph ends here", [320, 620, 560, 635]),
                                cross_page_line("continuation at next page top", [50, 260, 300, 275]),
                            ],
                        }
                    ],
                    "discarded_blocks": [],
                },
                {
                    "page_idx": 1,
                    "page_size": [600, 800],
                    "para_blocks": [
                        {
                            "type": "text",
                            "index": 4,
                            "bbox": [50, 260, 300, 290],
                            "lines": [
                                line("continuation at next page top", [50, 260, 300, 275]),
                                line("plus more text.", [50, 278, 300, 290]),
                            ],
                        }
                    ],
                    "discarded_blocks": [],
                },
            ]
        }

        result = MinerUAdapter(AdapterOptions(enrich_styles=False)).adapt(middle)

        self.assertEqual(len(result.content_list), 2)
        self.assertEqual(result.content_list[0]["page_idx"], 0)
        self.assertNotIn("continuation", result.content_list[0]["text"])
        self.assertEqual(result.content_list[1]["page_idx"], 1)
        self.assertIn("continuation at next page top", result.content_list[1]["text"])

    def test_cross_page_marked_lines_skip_figure_only_page(self) -> None:
        middle = {
            "pdf_info": [
                {
                    "page_idx": 0,
                    "page_size": [600, 800],
                    "para_blocks": [
                        {
                            "type": "text",
                            "index": 1,
                            "bbox": [320, 600, 560, 720],
                            "lines": [
                                line("Right column body", [320, 600, 560, 615]),
                                cross_page_line("continuation after a figure page", [50, 60, 300, 75]),
                            ],
                        }
                    ],
                    "discarded_blocks": [],
                },
                {
                    "page_idx": 1,
                    "page_size": [600, 800],
                    "para_blocks": [
                        {
                            "type": "chart",
                            "index": 2,
                            "bbox": [50, 50, 560, 700],
                            "lines": [],
                        }
                    ],
                    "discarded_blocks": [],
                },
                {
                    "page_idx": 2,
                    "page_size": [600, 800],
                    "para_blocks": [
                        {
                            "type": "text",
                            "index": 3,
                            "bbox": [50, 55, 300, 90],
                            "lines": [],
                            "lines_deleted": True,
                        }
                    ],
                    "preproc_blocks": [
                        {
                            "type": "text",
                            "index": 3,
                            "bbox": [50, 55, 300, 90],
                            "lines": [
                                line("continuation after a figure page", [50, 60, 300, 75]),
                                line("and it keeps going.", [50, 77, 300, 90]),
                            ],
                        }
                    ],
                    "discarded_blocks": [],
                },
            ]
        }

        result = MinerUAdapter(AdapterOptions(enrich_styles=False)).adapt(middle)

        continuation = [
            item for item in result.content_list if "continuation after a figure page" in item["text"]
        ]
        self.assertEqual(len(continuation), 1)
        self.assertEqual(continuation[0]["page_idx"], 2)
        self.assertTrue(continuation[0]["cross_page"])

    def test_cross_page_marked_lines_are_split_by_their_real_target_page(self) -> None:
        middle = {
            "pdf_info": [
                {
                    "page_idx": 0,
                    "page_size": [600, 800],
                    "para_blocks": [
                        {
                            "type": "text",
                            "index": 1,
                            "bbox": [50, 700, 560, 720],
                            "lines": [
                                line("page footer", [50, 700, 560, 720]),
                                cross_page_line("next page bottom line", [50, 650, 300, 665]),
                                cross_page_line("two pages later top line", [50, 60, 300, 75]),
                            ],
                        }
                    ],
                    "discarded_blocks": [],
                },
                {
                    "page_idx": 1,
                    "page_size": [600, 800],
                    "para_blocks": [
                        {
                            "type": "text",
                            "index": 2,
                            "bbox": [50, 650, 300, 665],
                            "lines": [],
                            "lines_deleted": True,
                        }
                    ],
                    "preproc_blocks": [
                        {
                            "type": "text",
                            "index": 2,
                            "bbox": [50, 650, 300, 665],
                            "lines": [
                                line("next page bottom line", [50, 650, 300, 665]),
                            ],
                        }
                    ],
                    "discarded_blocks": [],
                },
                {
                    "page_idx": 2,
                    "page_size": [600, 800],
                    "para_blocks": [
                        {
                            "type": "text",
                            "index": 3,
                            "bbox": [50, 60, 300, 75],
                            "lines": [],
                            "lines_deleted": True,
                        }
                    ],
                    "preproc_blocks": [
                        {
                            "type": "text",
                            "index": 3,
                            "bbox": [50, 60, 300, 75],
                            "lines": [
                                line("two pages later top line", [50, 60, 300, 75]),
                            ],
                        }
                    ],
                    "discarded_blocks": [],
                },
            ]
        }

        result = MinerUAdapter(AdapterOptions(enrich_styles=False)).adapt(middle)

        self.assertEqual(result.content_list[0]["page_idx"], 0)
        self.assertEqual(result.content_list[0]["text"], "page footer")
        by_page = {
            item["page_idx"]: item
            for item in result.content_list
            if item.get("cross_page")
        }
        self.assertEqual(by_page[1]["bbox"], [50, 650, 300, 665])
        self.assertEqual(by_page[1]["text"], "next page bottom line")
        self.assertEqual(by_page[2]["bbox"], [50, 60, 300, 75])
        self.assertEqual(by_page[2]["text"], "two pages later top line")

    def test_style_enrichment_contract(self) -> None:
        class Page:
            def get_text(self, *_args, **_kwargs):
                return {
                    "blocks": [
                        {
                            "lines": [
                                {
                                    "spans": [
                                        {
                                            "text": "Bold",
                                            "size": 12.0,
                                            "font": "Times-Bold",
                                            "flags": 16,
                                            "color": 0,
                                            "bbox": [50, 50, 80, 65],
                                        }
                                    ]
                                }
                            ]
                        }
                    ]
                }

        class Document:
            closed = False

            def __len__(self):
                return 1

            def __getitem__(self, _index):
                return Page()

            def close(self):
                self.closed = True

        document = Document()
        fake_fitz = types.SimpleNamespace(
            TEXT_PRESERVE_LIGATURES=1,
            Rect=lambda *coordinates: coordinates,
            open=lambda _path: document,
        )
        with patch.dict("sys.modules", {"fitz": fake_fitz}):
            result = MinerUAdapter().adapt(self.middle, pdf_path="paper.pdf")
        self.assertEqual(result.enriched[1]["rich_spans"][0]["font"], "Times-Bold")
        self.assertEqual(result.enriched[1]["local_md"], "**Bold**")
        self.assertTrue(document.closed)


if __name__ == "__main__":
    unittest.main()
