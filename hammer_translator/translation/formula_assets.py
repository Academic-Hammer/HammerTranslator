"""Build reusable vector assets for formula-aware layout and Reflow."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, TYPE_CHECKING

import fitz

from .formula import FormulaBinding

if TYPE_CHECKING:
    from .units import TranslationPlan


FORMULA_ASSET_CONTRACT_VERSION = "formula-assets-v1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATHJAX_SCRIPT = PROJECT_ROOT / "formula_runtime" / "mathjax_to_svg.mjs"
DEFAULT_MATHJAX_PACKAGE = PROJECT_ROOT / "formula_runtime" / "node_modules" / "mathjax"


class FormulaAssetError(ValueError):
    """Raised when a generated formula asset is unsafe or malformed."""


@dataclass(frozen=True)
class MathJaxExpression:
    formula_id: str
    latex: str
    display: bool


@dataclass(frozen=True)
class SourceProbe:
    page_idx: int
    requested_bbox: tuple[float, float, float, float]
    clip_bbox: tuple[float, float, float, float]
    text_match: str
    text_span_count: int
    drawing_count: int
    raster_image_count: int
    baseline_from_top_pt: float | None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(value)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _safe_error_code(value: str) -> str:
    code = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return code[:96] or "unknown_error"


class MathJaxSvgRenderer:
    """Offline batch wrapper around the project's pinned MathJax runtime."""

    def __init__(
        self,
        *,
        node_executable: str = "node",
        script_path: Path = DEFAULT_MATHJAX_SCRIPT,
        package_path: Path = DEFAULT_MATHJAX_PACKAGE,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.node_executable = node_executable
        self.script_path = script_path.resolve()
        self.package_path = package_path.resolve()
        self.timeout_seconds = timeout_seconds

    @property
    def available(self) -> bool:
        return self.script_path.is_file() and (self.package_path / "package.json").is_file()

    @property
    def version(self) -> str | None:
        package_json = self.package_path / "package.json"
        if not package_json.is_file():
            return None
        try:
            value = json.loads(package_json.read_text(encoding="utf-8")).get("version")
        except (OSError, json.JSONDecodeError):
            return None
        return str(value) if value else None

    def render_batch(
        self, expressions: Iterable[MathJaxExpression]
    ) -> tuple[dict[str, str], dict[str, str]]:
        batch = list(expressions)
        if not batch:
            return {}, {}
        if not self.available:
            raise FormulaAssetError("mathjax_runtime_unavailable")
        request = {
            "expressions": [
                {
                    "formula_id": expression.formula_id,
                    "latex": expression.latex,
                    "display": expression.display,
                }
                for expression in batch
            ]
        }
        try:
            completed = subprocess.run(
                [self.node_executable, str(self.script_path)],
                input=json.dumps(request, ensure_ascii=False),
                text=True,
                capture_output=True,
                check=False,
                timeout=self.timeout_seconds,
                cwd=self.script_path.parent,
            )
        except FileNotFoundError as exc:
            raise FormulaAssetError("node_runtime_unavailable") from exc
        except subprocess.TimeoutExpired as exc:
            raise FormulaAssetError("mathjax_timeout") from exc
        if completed.returncode != 0:
            raise FormulaAssetError(
                f"mathjax_process_failed_{_safe_error_code(completed.stderr)}"
            )
        try:
            payload = json.loads(completed.stdout)
            results = list(payload["results"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise FormulaAssetError("mathjax_response_invalid") from exc
        rendered: dict[str, str] = {}
        errors: dict[str, str] = {}
        for record in results:
            try:
                formula_id = str(record["formula_id"])
            except (KeyError, TypeError) as exc:
                raise FormulaAssetError("mathjax_response_invalid") from exc
            if isinstance(record.get("svg"), str):
                rendered[formula_id] = record["svg"]
            elif record.get("error") is not None:
                errors[formula_id] = _safe_error_code(str(record["error"]))
            else:
                raise FormulaAssetError("mathjax_response_invalid")
        expected = {expression.formula_id for expression in batch}
        if set(rendered) | set(errors) != expected or set(rendered) & set(errors):
            raise FormulaAssetError("mathjax_response_formula_ids_mismatch")
        return rendered, errors


def _rect_intersects(first: fitz.Rect, second: fitz.Rect) -> bool:
    intersection = first & second
    return not intersection.is_empty and intersection.width > 0 and intersection.height > 0


_LATEX_SYMBOLS = {
    "times": "×",
    "cdot": "·",
    "pm": "±",
    "le": "≤",
    "leq": "≤",
    "ge": "≥",
    "geq": "≥",
    "neq": "≠",
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "theta": "θ",
    "lambda": "λ",
    "mu": "μ",
    "pi": "π",
    "sigma": "σ",
    "phi": "φ",
    "omega": "ω",
}


def _visual_signature(value: str, *, latex: bool) -> str:
    text = unicodedata.normalize("NFKC", value)
    if latex:
        text = re.sub(r"\\(?:mathrm|mathbf|mathit|mathsf|mathtt|operatorname|text|left|right)\b", "", text)
        for command, replacement in _LATEX_SYMBOLS.items():
            text = re.sub(rf"\\{command}\b", replacement, text)
        text = re.sub(r"\\([%_{}#$&])", r"\1", text)
        text = re.sub(r"\\[A-Za-z]+", "", text)
        text = re.sub(r"[{}]", "", text)
    text = text.replace("−", "-").replace("⋅", "·")
    text = text.replace("^", "").replace("_", "")
    return re.sub(r"\s+", "", text).lower()


def _text_lines(page: fitz.Page, rect: fitz.Rect) -> list[tuple[str, fitz.Rect]]:
    records: list[tuple[str, fitz.Rect]] = []
    raw = page.get_text("rawdict", clip=rect)
    for block in raw.get("blocks", []):
        for line in block.get("lines", []):
            characters = [
                str(char.get("c") or "")
                for span in line.get("spans", [])
                for char in span.get("chars", [])
            ]
            text = "".join(characters)
            bbox = line.get("bbox")
            if text.strip() and bbox is not None:
                records.append((text, fitz.Rect(bbox)))
    return records


def _source_probe(page: fitz.Page, binding: FormulaBinding) -> SourceProbe:
    if binding.page_idx is None or binding.source_bbox is None:
        raise FormulaAssetError("source_geometry_missing")
    requested_rect = fitz.Rect(binding.source_bbox)
    rect = fitz.Rect(requested_rect)
    intersection = rect & page.rect
    if (
        intersection.is_empty
        or not math.isclose(intersection.width, rect.width, abs_tol=1.0)
        or not math.isclose(intersection.height, rect.height, abs_tol=1.0)
    ):
        raise FormulaAssetError("source_bbox_outside_pdf_page")

    text_match = "not_checked"
    if binding.kind == "inline":
        expected = _visual_signature(binding.latex, latex=True)
        lines = _text_lines(page, rect)
        observed = _visual_signature("".join(text for text, _ in lines), latex=False)
        if expected and observed == expected:
            text_match = "exact"
        elif expected:
            candidates = [
                line_bbox
                for text, line_bbox in lines
                if _visual_signature(text, latex=False) == expected
            ]
            if len(candidates) != 1:
                raise FormulaAssetError("source_clip_text_mismatch")
            candidate = candidates[0]
            rect = fitz.Rect(
                requested_rect.x0,
                max(requested_rect.y0, candidate.y0),
                requested_rect.x1,
                min(requested_rect.y1, candidate.y1),
            )
            if rect.is_empty or rect.width <= 0 or rect.height <= 0:
                raise FormulaAssetError("source_clip_refinement_empty")
            text_match = "refined"

    text_spans: list[dict[str, Any]] = []
    text_dict = page.get_text("dict", clip=rect)
    for block in text_dict.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if str(span.get("text") or "").strip():
                    text_spans.append(span)

    drawing_count = 0
    for drawing in page.get_drawings():
        drawing_rect = drawing.get("rect")
        if drawing_rect is not None and _rect_intersects(fitz.Rect(drawing_rect), rect):
            drawing_count += 1

    raster_image_count = 0
    for image in page.get_image_info(xrefs=True):
        bbox = image.get("bbox")
        if bbox is not None and _rect_intersects(fitz.Rect(bbox), rect):
            raster_image_count += 1

    if raster_image_count:
        raise FormulaAssetError("source_clip_contains_raster_content")
    if not text_spans and not drawing_count:
        raise FormulaAssetError("source_clip_has_no_vector_content")

    baseline: float | None = None
    candidates = [
        span
        for span in text_spans
        if isinstance(span.get("origin"), (list, tuple)) and len(span["origin"]) == 2
    ]
    if candidates:
        dominant = max(
            candidates,
            key=lambda span: (
                float(span.get("size") or 0.0),
                len(str(span.get("text") or "")),
            ),
        )
        baseline = float(dominant["origin"][1]) - rect.y0
        if not -1.0 <= baseline <= rect.height + 1.0:
            baseline = None
        elif baseline is not None:
            baseline = min(max(baseline, 0.0), rect.height)

    return SourceProbe(
        page_idx=binding.page_idx,
        requested_bbox=tuple(float(value) for value in binding.source_bbox),
        clip_bbox=(rect.x0, rect.y0, rect.x1, rect.y1),
        text_match=text_match,
        text_span_count=len(text_spans),
        drawing_count=drawing_count,
        raster_image_count=raster_image_count,
        baseline_from_top_pt=baseline,
    )


def _source_asset(
    binding: FormulaBinding,
    *,
    probe: SourceProbe,
    source_pdf_sha256: str,
) -> dict[str, Any]:
    width = probe.clip_bbox[2] - probe.clip_bbox[0]
    height = probe.clip_bbox[3] - probe.clip_bbox[1]
    locator = json.dumps(
        {
            "source_pdf_sha256": source_pdf_sha256,
            "page_idx": probe.page_idx,
            "bbox": list(probe.clip_bbox),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "asset_id": f"fa-{binding.formula_id}",
        "formula_id": binding.formula_id,
        "status": "ready",
        "kind": "source_pdf_clip",
        "storage": "source_pdf_reference",
        "relative_path": None,
        "media_type": "application/pdf",
        "sha256": _sha256_text(locator),
        "metrics": {
            "coordinate_system": "pdf_points",
            "width": width,
            "height": height,
            "baseline_from_top": probe.baseline_from_top_pt,
            "depth": (
                height - probe.baseline_from_top_pt
                if probe.baseline_from_top_pt is not None
                else None
            ),
            "aspect_ratio": width / height,
            "view_box": None,
        },
        "source_pdf_clip": {
            "source_pdf_sha256": source_pdf_sha256,
            "page_idx": probe.page_idx,
            "requested_bbox": list(probe.requested_bbox),
            "clip_bbox": list(probe.clip_bbox),
            "text_match": probe.text_match,
            "text_span_count": probe.text_span_count,
            "drawing_count": probe.drawing_count,
            "raster_image_count": probe.raster_image_count,
        },
        "engine": {"name": "pymupdf-source-clip", "version": fitz.VersionBind},
        "fallback_reason": None,
        "failure_code": None,
    }


def _length_ex(value: str | None, *, field: str) -> float:
    match = re.fullmatch(r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))ex\s*", value or "")
    if not match:
        raise FormulaAssetError(f"mathjax_svg_{field}_is_not_ex")
    result = float(match.group(1))
    if not math.isfinite(result) or result <= 0:
        raise FormulaAssetError(f"mathjax_svg_{field}_invalid")
    return result


def _svg_metrics(svg: str) -> dict[str, Any]:
    try:
        root = ET.fromstring(svg)
    except ET.ParseError as exc:
        raise FormulaAssetError("mathjax_svg_xml_invalid") from exc
    if root.tag.split("}")[-1] != "svg":
        raise FormulaAssetError("mathjax_svg_root_invalid")
    for element in root.iter():
        if element.tag.split("}")[-1].lower() == "script":
            raise FormulaAssetError("mathjax_svg_contains_script")
        for key, value in element.attrib.items():
            if key.split("}")[-1] == "href" and not value.startswith("#"):
                raise FormulaAssetError("mathjax_svg_contains_external_reference")

    width = _length_ex(root.get("width"), field="width")
    height = _length_ex(root.get("height"), field="height")
    try:
        view_box = [float(value) for value in str(root.get("viewBox") or "").split()]
    except ValueError as exc:
        raise FormulaAssetError("mathjax_svg_viewbox_invalid") from exc
    if len(view_box) != 4 or view_box[2] <= 0 or view_box[3] <= 0:
        raise FormulaAssetError("mathjax_svg_viewbox_invalid")
    style = str(root.get("style") or "")
    vertical_match = re.search(
        r"vertical-align\s*:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))ex",
        style,
    )
    vertical_align = float(vertical_match.group(1)) if vertical_match else 0.0
    depth = max(0.0, -vertical_align)
    baseline = height - depth
    if not 0.0 <= baseline <= height:
        raise FormulaAssetError("mathjax_svg_baseline_invalid")
    return {
        "coordinate_system": "mathjax_ex",
        "width": width,
        "height": height,
        "baseline_from_top": baseline,
        "depth": depth,
        "aspect_ratio": width / height,
        "view_box": view_box,
    }


def _mathjax_asset(
    binding: FormulaBinding,
    svg: str,
    *,
    asset_root: Path,
    artifact_dir: Path,
    renderer: MathJaxSvgRenderer,
    fallback_reason: str,
) -> dict[str, Any]:
    metrics = _svg_metrics(svg)
    path = asset_root / f"{binding.formula_id}.svg"
    _atomic_write_text(path, svg)
    return {
        "asset_id": f"fa-{binding.formula_id}",
        "formula_id": binding.formula_id,
        "status": "ready",
        "kind": "mathjax_svg",
        "storage": "file",
        "relative_path": path.relative_to(artifact_dir).as_posix(),
        "media_type": "image/svg+xml",
        "sha256": _sha256_file(path),
        "metrics": metrics,
        "source_pdf_clip": None,
        "engine": {"name": "mathjax", "version": renderer.version or "unknown"},
        "fallback_reason": fallback_reason,
        "failure_code": None,
    }


def _failed_asset(
    binding: FormulaBinding,
    *,
    failure_code: str,
    fallback_reason: str | None,
) -> dict[str, Any]:
    return {
        "asset_id": f"fa-{binding.formula_id}",
        "formula_id": binding.formula_id,
        "status": "failed",
        "kind": None,
        "storage": "none",
        "relative_path": None,
        "media_type": None,
        "sha256": None,
        "metrics": None,
        "source_pdf_clip": None,
        "engine": None,
        "fallback_reason": fallback_reason,
        "failure_code": failure_code,
    }


def build_formula_asset_contract(
    plan: "TranslationPlan",
    artifact_path: str | Path,
    *,
    renderer: MathJaxSvgRenderer | None = None,
) -> dict[str, Any]:
    """Resolve exactly one source-PDF or MathJax asset for every formula."""

    target = Path(artifact_path).expanduser().resolve()
    artifact_dir = target.parent
    asset_root = artifact_dir / f"{plan.document.document_id}_formula_assets"
    bindings = [binding for unit in plan.units for binding in unit.formula_bindings]
    renderer = renderer or MathJaxSvgRenderer()
    source_pdf = plan.document.source_pdf.resolve()
    source_pdf_sha256 = _sha256_file(source_pdf)
    records: dict[str, dict[str, Any]] = {}
    pending_mathjax: list[tuple[FormulaBinding, str]] = []

    document = fitz.open(source_pdf)
    try:
        for binding in bindings:
            source_failure: str | None = None
            if binding.asset_preference in {"source_pdf", "source_pdf_then_mathjax"}:
                try:
                    if binding.page_idx is None or not 0 <= binding.page_idx < document.page_count:
                        raise FormulaAssetError("source_page_missing")
                    probe = _source_probe(document[binding.page_idx], binding)
                    records[binding.formula_id] = _source_asset(
                        binding,
                        probe=probe,
                        source_pdf_sha256=source_pdf_sha256,
                    )
                    continue
                except FormulaAssetError as exc:
                    source_failure = _safe_error_code(str(exc))
                    if binding.asset_preference == "source_pdf":
                        records[binding.formula_id] = _failed_asset(
                            binding,
                            failure_code=source_failure,
                            fallback_reason=None,
                        )
                        continue
            reason = source_failure or "source_geometry_unavailable"
            pending_mathjax.append((binding, reason))
    finally:
        document.close()

    if pending_mathjax:
        expressions = [
            MathJaxExpression(
                formula_id=binding.formula_id,
                latex=binding.latex,
                display=binding.kind == "display",
            )
            for binding, _ in pending_mathjax
        ]
        try:
            rendered, render_errors = renderer.render_batch(expressions)
        except FormulaAssetError as exc:
            failure = _safe_error_code(str(exc))
            for binding, reason in pending_mathjax:
                records[binding.formula_id] = _failed_asset(
                    binding,
                    failure_code=failure,
                    fallback_reason=reason,
                )
        else:
            for binding, reason in pending_mathjax:
                if binding.formula_id in render_errors:
                    records[binding.formula_id] = _failed_asset(
                        binding,
                        failure_code=f"mathjax_expression_{render_errors[binding.formula_id]}",
                        fallback_reason=reason,
                    )
                    continue
                try:
                    records[binding.formula_id] = _mathjax_asset(
                        binding,
                        rendered[binding.formula_id],
                        asset_root=asset_root,
                        artifact_dir=artifact_dir,
                        renderer=renderer,
                        fallback_reason=reason,
                    )
                except (FormulaAssetError, OSError) as exc:
                    records[binding.formula_id] = _failed_asset(
                        binding,
                        failure_code=_safe_error_code(str(exc)),
                        fallback_reason=reason,
                    )

    ordered = [records[binding.formula_id] for binding in bindings]
    return {
        "schema_version": FORMULA_ASSET_CONTRACT_VERSION,
        "asset_root": asset_root.relative_to(artifact_dir).as_posix(),
        "source_pdf_sha256": source_pdf_sha256,
        "mathjax": {
            "available": renderer.available,
            "version": renderer.version,
            "script": renderer.script_path.relative_to(PROJECT_ROOT).as_posix(),
        },
        "assets": ordered,
        "stats": {
            "total": len(ordered),
            "ready": sum(record["status"] == "ready" for record in ordered),
            "failed": sum(record["status"] == "failed" for record in ordered),
            "source_pdf_clip": sum(record["kind"] == "source_pdf_clip" for record in ordered),
            "mathjax_svg": sum(record["kind"] == "mathjax_svg" for record in ordered),
            "mathjax_fallback": sum(
                record["kind"] == "mathjax_svg"
                and record["fallback_reason"] != "source_geometry_unavailable"
                for record in ordered
            ),
        },
    }
