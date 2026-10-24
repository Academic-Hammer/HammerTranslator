"""Conversion of MinerU page blocks into the project's flat content list."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .geometry import copy_bbox, intersection_area, union_bbox
from .text import extract_text

TEXT_TYPES = {
    "text",
    "list",
    "index",
    "abstract",
    "vertical_text",
    "phonetic",
    "formula_number",
}
HEADING_TYPES = {"title", "doc_title", "paragraph_title"}
DISCARDED_TYPES = {"header", "footer", "page_number", "aside_text", "page_footnote", "discarded"}


@dataclass(frozen=True)
class ConversionOptions:
    image_dir: str = "images"
    preserve_discarded: bool = True
    split_outside_lines: bool = True


class MiddleJsonConverter:
    def __init__(self, options: ConversionOptions):
        self.options = options
        self.warnings: list[str] = []

    def convert(self, middle_json: dict[str, Any]) -> list[dict[str, Any]]:
        pages = middle_json.get("pdf_info")
        if not isinstance(pages, list):
            raise ValueError("MinerU middle JSON must contain a 'pdf_info' list")

        output: list[dict[str, Any]] = []
        logical_id = 0
        page_source_texts = [self._page_source_text(value) for value in pages]
        page_emitted_texts = [self._page_emitted_text(value) for value in pages]
        for fallback_page_idx, page in enumerate(pages):
            if not isinstance(page, dict):
                self.warnings.append(f"Skipped non-object page at position {fallback_page_idx}")
                continue
            page_idx = page.get("page_idx", fallback_page_idx)
            blocks = self._ordered_blocks(page, page_idx)
            for block in blocks:
                split_blocks = (
                    self._split_outside_lines(
                        block,
                        fallback_page_idx,
                        page_source_texts,
                        page_emitted_texts,
                    )
                    if self.options.split_outside_lines
                    else [block]
                )
                for split_block in split_blocks:
                    page_offset = int(split_block.pop("_continuation_page_offset", 0))
                    item_page_idx = int(page_idx) + page_offset
                    if item_page_idx >= len(pages):
                        self.warnings.append(
                            f"Page {page_idx} contains an apparent cross-page continuation "
                            "after the final PDF page; the source page was retained"
                        )
                        item_page_idx = int(page_idx)
                    items = self._convert_block(split_block, item_page_idx, logical_id)
                    output.extend(item for item in items if self._has_payload(item))
                    if items:
                        logical_id += 1
        return output

    def _ordered_blocks(self, page: dict[str, Any], page_idx: int) -> list[dict[str, Any]]:
        para = [block for block in page.get("para_blocks", []) if isinstance(block, dict)]
        discarded = [block for block in page.get("discarded_blocks", []) if isinstance(block, dict)]
        if not self.options.preserve_discarded:
            discarded = []
        blocks = para + discarded

        def key(block: dict[str, Any]) -> tuple[float, float, float]:
            index = block.get("index")
            if isinstance(index, (int, float)):
                return (0, float(index), 0)
            bbox = copy_bbox(block.get("bbox")) or [0, 0, 0, 0]
            return (1, float(bbox[1]), float(bbox[0]))

        if any("index" not in block for block in blocks):
            self.warnings.append(f"Page {page_idx} contains blocks without an index; bbox order was used as fallback")
        return sorted(blocks, key=key)

    @staticmethod
    def _line_text(line: dict[str, Any]) -> str:
        spans = line.get("spans")
        if not isinstance(spans, list):
            return ""
        return "".join(
            str(span.get("content") or "")
            for span in spans
            if isinstance(span, dict)
        )

    def _all_page_blocks(self, page: Any) -> list[dict[str, Any]]:
        if not isinstance(page, dict):
            return []
        blocks = [block for block in page.get("para_blocks", []) if isinstance(block, dict)]
        blocks.extend(
            block for block in page.get("preproc_blocks", []) if isinstance(block, dict)
        )
        if self.options.preserve_discarded:
            blocks.extend(
                block
                for block in page.get("discarded_blocks", [])
                if isinstance(block, dict)
            )
        return blocks

    def _page_source_text(self, page: Any) -> str:
        """Collect every line MinerU associated with a page, including its preproc copy."""

        return "".join(
            self._line_text(line)
            for block in self._all_page_blocks(page)
            for line in block.get("lines") or []
            if isinstance(line, dict)
        )

    def _page_emitted_text(self, page: Any) -> str:
        """Collect text the converter will emit for a page after line splitting."""

        if not isinstance(page, dict):
            return ""
        blocks = [block for block in page.get("para_blocks", []) if isinstance(block, dict)]
        if self.options.preserve_discarded:
            blocks.extend(
                block
                for block in page.get("discarded_blocks", [])
                if isinstance(block, dict)
            )
        texts: list[str] = []
        for block in blocks:
            bbox = copy_bbox(block.get("bbox"))
            for line in block.get("lines") or []:
                if not isinstance(line, dict):
                    continue
                if bbox is not None and intersection_area(bbox, line.get("bbox")) <= 0:
                    continue
                texts.append(self._line_text(line))
        return "".join(texts)

    @staticmethod
    def _line_is_cross_page(line: dict[str, Any]) -> bool:
        spans = line.get("spans")
        if not isinstance(spans, list):
            return False
        marked = [span for span in spans if isinstance(span, dict)]
        return bool(marked) and any(bool(span.get("cross_page")) for span in marked)

    @staticmethod
    def _lines_overlap_on_page(first: dict[str, Any], second: dict[str, Any]) -> bool:
        """Return whether two lines occupy the same page position.

        MinerU sometimes leaves lines from the next page inside the previous
        page's block bbox.  Those lines overlap the previous page's own lines
        in both axes, which cannot happen for ordinary single-column text.
        """

        first_bbox = copy_bbox(first.get("bbox"))
        second_bbox = copy_bbox(second.get("bbox"))
        if first_bbox is None or second_bbox is None:
            return False
        overlap = intersection_area(first_bbox, second_bbox)
        if overlap <= 0:
            return False
        first_area = (first_bbox[2] - first_bbox[0]) * (first_bbox[3] - first_bbox[1])
        second_area = (second_bbox[2] - second_bbox[0]) * (second_bbox[3] - second_bbox[1])
        smaller = min(first_area, second_area)
        return smaller > 0 and overlap / smaller >= 0.5

    @staticmethod
    def _normalize_text(text: str) -> str:
        return "".join(text.split()).casefold()

    def _split_outside_lines(
        self,
        block: dict[str, Any],
        page_position: int = 0,
        page_source_texts: list[str] | None = None,
        page_emitted_texts: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if block.get("type") not in TEXT_TYPES or not isinstance(block.get("lines"), list):
            return [block]
        bbox = copy_bbox(block.get("bbox"))
        if bbox is None:
            return [block]
        inside: list[dict[str, Any]] = []
        outside: list[dict[str, Any]] = []
        lines = [line for line in block["lines"] if isinstance(line, dict)]
        current_page_lines = [line for line in lines if not self._line_is_cross_page(line)]
        displaced_cross_page_lines: list[dict[str, Any]] = []
        for line in lines:
            if self._line_is_cross_page(line) and any(
                self._lines_overlap_on_page(line, current) for current in current_page_lines
            ):
                displaced_cross_page_lines.append(line)
                continue
            target = inside if intersection_area(bbox, line.get("bbox")) > 0 else outside
            target.append(line)
        if not inside:
            return [block]
        if not outside and not displaced_cross_page_lines:
            return [block]
        first = deepcopy(block)
        first["lines"] = inside
        result = [first]

        cross_page_lines = [
            line
            for line in outside
            if self._line_is_cross_page(line)
        ] + displaced_cross_page_lines
        remaining = [
            line
            for line in outside
            if not self._line_is_cross_page(line)
        ]
        if cross_page_lines:
            source_texts = page_source_texts or []
            emitted_texts = page_emitted_texts or []
            grouped: dict[int, list[dict[str, Any]]] = {}
            for line in cross_page_lines:
                needle = self._normalize_text(self._line_text(line))
                target_position: int | None = None
                if needle:
                    for candidate in range(
                        page_position + 1,
                        min(len(source_texts), page_position + 4),
                    ):
                        if needle in self._normalize_text(source_texts[candidate]):
                            target_position = candidate
                            break
                if target_position is None:
                    target_position = page_position + 1
                grouped.setdefault(target_position, []).append(line)

            for target_position, target_lines in sorted(grouped.items()):
                continuation_text = "".join(
                    self._line_text(line) for line in target_lines
                )
                needle = self._normalize_text(continuation_text)
                already_emitted = (
                    target_position < len(emitted_texts)
                    and needle
                    and needle in self._normalize_text(emitted_texts[target_position])
                )
                if already_emitted:
                    continue
                continuation = deepcopy(block)
                continuation["lines"] = target_lines
                continuation_bbox = union_bbox(line.get("bbox") for line in target_lines)
                if continuation_bbox:
                    continuation["bbox"] = continuation_bbox
                continuation["cross_page"] = True
                continuation["_continuation_page_offset"] = (
                    target_position - page_position
                )
                result.append(continuation)
        if remaining:
            second = deepcopy(block)
            second["lines"] = remaining
            second_bbox = union_bbox(line.get("bbox") for line in remaining)
            if second_bbox:
                second["bbox"] = second_bbox
            if self._looks_like_next_page(bbox, second_bbox):
                second["cross_page"] = True
                second["_continuation_page_offset"] = 1
            else:
                second["cross_column"] = True
            result.append(second)
        return result

    @staticmethod
    def _looks_like_next_page(
        source_bbox: list[float | int],
        continuation_bbox: list[float | int] | None,
    ) -> bool:
        """Distinguish a same-column page reset from a same-page column jump."""

        if continuation_bbox is None or continuation_bbox[1] >= source_bbox[1]:
            return False
        overlap = min(source_bbox[2], continuation_bbox[2]) - max(
            source_bbox[0], continuation_bbox[0]
        )
        narrower_width = min(
            source_bbox[2] - source_bbox[0],
            continuation_bbox[2] - continuation_bbox[0],
        )
        return narrower_width > 0 and overlap / narrower_width >= 0.5

    def _base(self, block: dict[str, Any], page_idx: int, logical_id: int) -> dict[str, Any]:
        result: dict[str, Any] = {"page_idx": page_idx, "logical_para_id": logical_id}
        for field in ("index", "cross_page", "cross_column"):
            if field in block:
                result[field] = deepcopy(block[field])
        bbox = copy_bbox(block.get("bbox"))
        if bbox is not None:
            result["bbox"] = bbox
        if isinstance(block.get("lines"), list):
            result["lines"] = deepcopy(block["lines"])
        return result

    def _convert_block(self, block: dict[str, Any], page_idx: int, logical_id: int) -> list[dict[str, Any]]:
        block_type = str(block.get("type", "text"))
        base = self._base(block, page_idx, logical_id)
        if block_type in TEXT_TYPES | HEADING_TYPES | DISCARDED_TYPES | {"caption", "footnote"}:
            output_type = "text" if block_type in TEXT_TYPES | HEADING_TYPES else block_type
            item = {"type": output_type, "text": extract_text(block), **base}
            if block_type not in {"text", "list", "index"}:
                item["sub_type"] = block_type
            if block_type in HEADING_TYPES or "text_level" in block:
                default_level = 1 if block_type != "paragraph_title" else 2
                level = block.get("text_level", block.get("level", default_level))
                if isinstance(level, int) and level > 0:
                    item["text_level"] = min(level, 4)
            return [item]
        if block_type == "ref_text":
            return [{"type": "reference", "sub_type": block_type, "text": extract_text(block), **base}]
        if block_type in {"interline_equation", "equation"}:
            item = {"type": "equation", "text": extract_text(block), **base}
            item["text_format"] = "latex"
            image_path = self._first_span_value(block, "image_path")
            if image_path:
                item["img_path"] = self._image_path(str(image_path))
            return [item]
        if block_type in {"image", "table"}:
            return self._convert_composite(block, block_type, base)

        text = extract_text(block)
        if text:
            return [{"type": block_type, "text": text, **base}]
        self.warnings.append(f"Skipped unsupported empty block type '{block_type}' on page {page_idx}")
        return []

    def _convert_composite(
        self, block: dict[str, Any], kind: str, base: dict[str, Any]
    ) -> list[dict[str, Any]]:
        children = block.get("blocks", [])
        if not isinstance(children, list):
            children = []
        items: list[dict[str, Any]] = []
        body_created = False
        for child in children:
            if not isinstance(child, dict):
                continue
            child_type = str(child.get("type", ""))
            if child_type in {f"{kind}_caption", f"{kind}_footnote"}:
                item_type = "caption" if child_type.endswith("caption") else "footnote"
                item = {"type": item_type, "text": extract_text(child), **deepcopy(base)}
                child_bbox = copy_bbox(child.get("bbox"))
                if child_bbox is not None:
                    item["bbox"] = child_bbox
                if isinstance(child.get("lines"), list):
                    item["lines"] = deepcopy(child["lines"])
                items.append(item)
            elif child_type == f"{kind}_body":
                item = {"type": kind, "text": "", **deepcopy(base)}
                image_path = self._first_span_value(child, "image_path")
                if image_path:
                    item["img_path"] = self._image_path(str(image_path))
                if kind == "table":
                    html = self._first_span_value(child, "html")
                    if html:
                        item["table_body"] = html
                items.append(item)
                body_created = True
        if not body_created:
            item = {"type": kind, "text": "", **deepcopy(base)}
            image_path = self._first_span_value(block, "image_path")
            if image_path:
                item["img_path"] = self._image_path(str(image_path))
            items.append(item)
        return items

    @staticmethod
    def _first_span_value(block: dict[str, Any], field: str) -> Any:
        nested = block.get("blocks", [])
        children = nested if isinstance(nested, list) else []
        for child in [block, *children]:
            if not isinstance(child, dict):
                continue
            for line in child.get("lines", []):
                if not isinstance(line, dict):
                    continue
                for span in line.get("spans", []):
                    if isinstance(span, dict) and span.get(field):
                        return span[field]
        return None

    def _image_path(self, name: str) -> str:
        if not self.options.image_dir:
            return name
        return f"{self.options.image_dir.rstrip('/')}/{name.lstrip('/')}"

    @staticmethod
    def _has_payload(item: dict[str, Any]) -> bool:
        return item.get("type") in {"image", "table", "equation"} or bool(
            item.get("text") or item.get("img_path") or item.get("table_body")
        )
