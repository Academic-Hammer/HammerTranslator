# MinerU Adapter

This package is the boundary between MinerU parsing and the later translation/rendering stages.
It deliberately contains no model loading and no translation API code.

## Input and output

- Input: MinerU `*_middle.json`; optional original PDF.
- `*_content_list.json`: flat, reading-order blocks in PDF point coordinates.
- `*_enriched.json`: the same blocks plus `rich_spans` and `local_md` when a PDF is supplied.

The stable fields used downstream are `type`, `text`, `page_idx`, `bbox`, and
`logical_para_id`. Images, tables, equations, captions, heading levels, original
`lines`, cross-page flags, and discarded layout elements are retained when present.

## Usage

```bash
python -m mineru_adapter path/to/paper_middle.json --pdf path/to/paper_origin.pdf
```

Without PyMuPDF, or when only structural conversion is needed:

```bash
python -m mineru_adapter path/to/paper_middle.json --no-styles
```

Python API:

```python
from mineru_adapter import MinerUAdapter

result = MinerUAdapter().adapt_file(
    "paper_middle.json",
    pdf_path="paper_origin.pdf",
    output_dir="output",
)
```

## Coordinate contract

Coordinates come directly from `middle.json` and remain in PDF points with a
top-left origin. Do not feed MinerU's legacy `content_list.json` into this adapter:
newer MinerU versions normalize that file's coordinates to a 0–1000 range.
