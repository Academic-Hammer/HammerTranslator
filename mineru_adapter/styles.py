"""Optional extraction of PDF-native font spans using PyMuPDF."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from .geometry import valid_bbox


class StyleEnricher:
    def __init__(self, clip_margin: float = 1.0):
        self.clip_margin = clip_margin

    def enrich(self, items: list[dict[str, Any]], pdf_path: str | Path) -> tuple[list[dict[str, Any]], list[str]]:
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("Style enrichment requires PyMuPDF (pip install PyMuPDF)") from exc

        output = deepcopy(items)
        warnings: list[str] = []
        document = fitz.open(str(pdf_path))
        try:
            for item in output:
                bbox = item.get("bbox")
                page_idx = item.get("page_idx")
                if not valid_bbox(bbox):
                    continue
                if not isinstance(page_idx, int) or not 0 <= page_idx < len(document):
                    warnings.append(f"Skipped style extraction for invalid page index {page_idx!r}")
                    continue
                margin = self.clip_margin
                rect = fitz.Rect(bbox[0] - margin, bbox[1] - margin, bbox[2] + margin, bbox[3] + margin)
                raw = document[page_idx].get_text("dict", clip=rect, flags=fitz.TEXT_PRESERVE_LIGATURES)
                spans = self._collect_spans(raw)
                if spans:
                    item["rich_spans"] = spans
                    item["local_md"] = self._markdown(spans)
        finally:
            document.close()
        return output, warnings

    @staticmethod
    def _collect_spans(raw: dict[str, Any]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for block in raw.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = str(span.get("text", "")).replace("\x00", "").replace("\r", "\n")
                    if not text.strip():
                        continue
                    result.append(
                        {
                            "text": text,
                            "size": span.get("size"),
                            "font": span.get("font"),
                            "flags": span.get("flags", 0),
                            "color": span.get("color", 0),
                            "bbox": list(span.get("bbox", [])),
                        }
                    )
        return result

    @staticmethod
    def _markdown(spans: list[dict[str, Any]]) -> str:
        fragments: list[str] = []
        for span in spans:
            text = str(span["text"])
            flags = int(span.get("flags", 0))
            if flags & 16:
                text = f"**{text}**"
            elif flags & 2:
                text = f"*{text}*"
            fragments.append(text)
        return " ".join(fragments).replace("** **", " ")
