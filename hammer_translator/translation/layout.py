"""Shared deterministic text measurement for Translation and Reflow."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import fitz


class LayoutMetricError(ValueError):
    pass


LAYOUT_METRICS_VERSION = "layout-metrics-v2"
SUPPORTED_LAYOUT_METRICS_VERSIONS = {"layout-metrics-v1", LAYOUT_METRICS_VERSION}
MATHJAX_EX_TO_EM = 0.5


TOKEN_PATTERN = re.compile(
    r"\n|(?<!\\)\$(?:\\.|[^$\n])+(?<!\\)\$|\s+|"
    r"[A-Za-z0-9]+(?:[-_/][A-Za-z0-9]+)*|[\u3400-\u4dbf\u4e00-\u9fff]|."
)
DEFAULT_MEASUREMENT_CACHE_MAX_ENTRIES = 50_000


@dataclass(frozen=True)
class FormulaRunSpec:
    """One exact formula occurrence bound to a ready Step 8B asset."""

    formula_id: str
    asset_id: str
    asset_kind: str
    text: str
    start: int
    end: int
    coordinate_system: str
    natural_width: float
    natural_height: float
    natural_baseline_from_top: float
    natural_depth: float

    def shifted(self, offset: int) -> "FormulaRunSpec":
        return replace(self, start=self.start - offset, end=self.end - offset)


@dataclass(frozen=True)
class MixedRun:
    kind: str
    text: str
    x_offset_pt: float
    width_pt: float
    height_pt: float
    baseline_from_top_pt: float
    formula_id: str | None = None
    asset_id: str | None = None
    asset_kind: str | None = None
    asset_scale_to_pt: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MixedLine:
    text: str
    width_pt: float
    height_pt: float
    baseline_from_top_pt: float
    runs: tuple[MixedRun, ...]

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["runs"] = [run.as_dict() for run in self.runs]
        return value


@dataclass(frozen=True)
class LayoutResult:
    fits: bool
    is_overflow: bool
    available_width_pt: float
    available_height_pt: float
    used_width_pt: float
    used_height_pt: float
    overflow_width_pt: float
    overflow_height_pt: float
    line_count: int
    line_height_pt: float
    width_fill_ratio: float
    height_fill_ratio: float
    area_fill_ratio: float
    estimated_cjk_capacity: int
    lines: tuple[str, ...]
    line_widths_pt: tuple[float, ...]
    font_id: str
    font_size_pt: float
    line_height_multiplier: float
    char_spacing_em: float
    mixed_lines: tuple[MixedLine, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["lines"] = list(self.lines)
        value["line_widths_pt"] = list(self.line_widths_pt)
        value["mixed_lines"] = [line.as_dict() for line in self.mixed_lines]
        return value


def bind_formula_run_specs(
    text: str,
    formula_records: list[dict[str, Any]],
    protected_tokens: list[dict[str, Any]],
    asset_records: list[dict[str, Any]],
) -> tuple[FormulaRunSpec, ...]:
    """Bind ordered immutable formula strings to their vector asset metrics.

    Translation protection guarantees exact value preservation and order.  We
    intentionally reject missing, reordered, or failed assets instead of
    measuring a LaTeX source string as ordinary text.
    """

    tokens = {
        str(record.get("placeholder") or ""): record
        for record in protected_tokens
        if record.get("placeholder")
    }
    assets = {
        str(record.get("formula_id") or ""): record
        for record in asset_records
        if record.get("formula_id")
    }
    cursor = 0
    specs: list[FormulaRunSpec] = []
    for formula in formula_records:
        if formula.get("kind") != "inline" or formula.get("render_policy") != "move_with_text":
            continue
        formula_id = str(formula.get("formula_id") or "")
        placeholder = str(formula.get("protected_placeholder") or "")
        token = tokens.get(placeholder)
        if not formula_id or token is None:
            raise LayoutMetricError(f"inline formula {formula_id or '<unknown>'} has no protected token")
        original = str(token.get("original") or "")
        if not original:
            raise LayoutMetricError(f"inline formula {formula_id} has an empty protected value")
        start = text.find(original, cursor)
        if start < 0:
            raise LayoutMetricError(f"inline formula {formula_id} is missing or reordered")
        end = start + len(original)
        asset = assets.get(formula_id)
        if asset is None or asset.get("status") != "ready":
            raise LayoutMetricError(f"inline formula {formula_id} has no ready vector asset")
        metrics = asset.get("metrics")
        if not isinstance(metrics, dict):
            raise LayoutMetricError(f"inline formula {formula_id} has invalid asset metrics")
        try:
            natural_width = float(metrics["width"])
            natural_height = float(metrics["height"])
            natural_baseline = float(metrics["baseline_from_top"])
            natural_depth = float(metrics["depth"])
        except (KeyError, TypeError, ValueError) as exc:
            raise LayoutMetricError(
                f"inline formula {formula_id} has invalid numeric asset metrics"
            ) from exc
        coordinate_system = str(metrics.get("coordinate_system") or "")
        if coordinate_system not in {"pdf_points", "mathjax_ex"}:
            raise LayoutMetricError(
                f"inline formula {formula_id} uses an unsupported coordinate system"
            )
        if (
            natural_width <= 0
            or natural_height <= 0
            or natural_baseline < 0
            or natural_depth < 0
            or not math.isclose(
                natural_baseline + natural_depth,
                natural_height,
                abs_tol=0.02,
            )
        ):
            raise LayoutMetricError(f"inline formula {formula_id} has inconsistent metrics")
        specs.append(
            FormulaRunSpec(
                formula_id=formula_id,
                asset_id=str(asset.get("asset_id") or ""),
                asset_kind=str(asset.get("kind") or ""),
                text=original,
                start=start,
                end=end,
                coordinate_system=coordinate_system,
                natural_width=natural_width,
                natural_height=natural_height,
                natural_baseline_from_top=natural_baseline,
                natural_depth=natural_depth,
            )
        )
        cursor = end
    return tuple(specs)


class LayoutMetrics:
    """Measure with the same bundled font and parameters intended for rendering."""

    def __init__(
        self,
        style_profile: dict[str, Any],
        *,
        measurement_cache_max_entries: int = DEFAULT_MEASUREMENT_CACHE_MAX_ENTRIES,
    ):
        if style_profile.get("schema_version") != "layout-style-v1":
            raise LayoutMetricError("unsupported golden style profile schema")
        if style_profile.get("measurement_engine") not in SUPPORTED_LAYOUT_METRICS_VERSIONS:
            raise LayoutMetricError("golden style profile requires a different measurement engine")
        if measurement_cache_max_entries < 0:
            raise LayoutMetricError("measurement_cache_max_entries cannot be negative")
        self.style_profile = style_profile
        self.fonts = style_profile.get("fonts") or {}
        self.roles = style_profile.get("roles") or {}
        self._font_cache: dict[str, fitz.Font] = {}
        self._measurement_cache_max_entries = int(measurement_cache_max_entries)
        self._glyph_width_cache: dict[tuple[int, str], float] = {}
        self._token_cache: dict[str, tuple[str, ...]] = {}
        self._measurement_cache_stats = {
            "glyph_width_hits": 0,
            "glyph_width_misses": 0,
            "glyph_width_evictions": 0,
            "token_hits": 0,
            "token_misses": 0,
            "token_evictions": 0,
        }

    def measurement_cache_stats(self) -> dict[str, int]:
        """Return local cache diagnostics without exposing cached document text."""

        return {
            **self._measurement_cache_stats,
            "glyph_width_entries": len(self._glyph_width_cache),
            "token_entries": len(self._token_cache),
        }

    def _tokens(self, text: str) -> tuple[str, ...]:
        if self._measurement_cache_max_entries == 0:
            return tuple(TOKEN_PATTERN.findall(text))
        cached = self._token_cache.get(text)
        if cached is not None:
            self._measurement_cache_stats["token_hits"] += 1
            return cached
        self._measurement_cache_stats["token_misses"] += 1
        tokens = tuple(TOKEN_PATTERN.findall(text))
        if len(self._token_cache) >= self._measurement_cache_max_entries:
            self._token_cache.pop(next(iter(self._token_cache)))
            self._measurement_cache_stats["token_evictions"] += 1
        self._token_cache[text] = tokens
        return tokens

    def style_for_role(self, role: str) -> tuple[dict[str, Any], dict[str, Any]]:
        record = self.roles.get(role)
        if not isinstance(record, dict):
            raise LayoutMetricError(f"unknown golden style role: {role}")
        style = record.get("target_style")
        guardrails = record.get("guardrails")
        if not isinstance(style, dict) or not isinstance(guardrails, dict):
            raise LayoutMetricError(f"invalid style profile for role: {role}")
        return dict(style), dict(guardrails)

    def _font(self, font_id: str) -> fitz.Font:
        if font_id in self._font_cache:
            return self._font_cache[font_id]
        record = self.fonts.get(font_id)
        if not isinstance(record, dict):
            raise LayoutMetricError(f"font_id is not registered: {font_id}")
        path = Path(str(record.get("path") or "")).expanduser().resolve()
        if not path.is_file():
            raise LayoutMetricError(f"registered font file is missing: {path}")
        try:
            font = fitz.Font(fontfile=str(path))
        except Exception as exc:
            raise LayoutMetricError(f"cannot load registered font {path}: {exc}") from exc
        self._font_cache[font_id] = font
        return font

    @staticmethod
    def _validate_bbox(bbox: list[float] | tuple[float, ...]) -> tuple[float, float]:
        if len(bbox) != 4:
            raise LayoutMetricError("bbox must contain four coordinates")
        try:
            x0, y0, x1, y1 = (float(value) for value in bbox)
        except (TypeError, ValueError) as exc:
            raise LayoutMetricError("bbox coordinates must be numeric") from exc
        if not all(math.isfinite(value) for value in (x0, y0, x1, y1)):
            raise LayoutMetricError("bbox coordinates must be finite")
        if x1 <= x0 or y1 <= y0:
            raise LayoutMetricError("bbox must satisfy x0 < x1 and y0 < y1")
        return x1 - x0, y1 - y0

    @staticmethod
    def _validate_style(style: dict[str, Any]) -> tuple[str, float, float, float]:
        font_id = str(style.get("font_id") or "")
        try:
            font_size = float(style["font_size_pt"])
            line_height = float(style["line_height_multiplier"])
            char_spacing = float(style["char_spacing_em"])
        except (KeyError, TypeError, ValueError) as exc:
            raise LayoutMetricError("style has invalid numeric fields") from exc
        if not font_id:
            raise LayoutMetricError("style font_id cannot be empty")
        if not 3.0 <= font_size <= 72.0:
            raise LayoutMetricError("font_size_pt must be within 3..72")
        if not 0.8 <= line_height <= 3.0:
            raise LayoutMetricError("line_height_multiplier must be within 0.8..3.0")
        if not -0.2 <= char_spacing <= 0.5:
            raise LayoutMetricError("char_spacing_em must be within -0.2..0.5")
        return font_id, font_size, line_height, char_spacing

    @staticmethod
    def _within(value: float, bounds: dict[str, Any], tolerance: float = 1e-6) -> bool:
        return float(bounds["min"]) - tolerance <= value <= float(bounds["max"]) + tolerance

    def validate_guardrails(
        self,
        role: str,
        style: dict[str, Any],
    ) -> None:
        _, font_size, line_height, char_spacing = self._validate_style(style)
        _, guardrails = self.style_for_role(role)
        checks = (
            (font_size, guardrails["font_size_pt"], "font_size_pt"),
            (line_height, guardrails["line_height_multiplier"], "line_height_multiplier"),
            (char_spacing, guardrails["char_spacing_em"], "char_spacing_em"),
        )
        for value, bounds, name in checks:
            if not self._within(value, bounds):
                raise LayoutMetricError(
                    f"{name}={value} is outside role {role} guardrails "
                    f"[{bounds['min']}, {bounds['max']}]"
                )

    @staticmethod
    def _character_count(text: str) -> int:
        return len(text)

    def _text_width(
        self,
        font: fitz.Font,
        text: str,
        *,
        font_size: float,
        char_spacing_em: float,
    ) -> float:
        if not text:
            return 0.0
        if self._measurement_cache_max_entries == 0:
            base = float(font.text_length(text, fontsize=font_size))
        else:
            # PyMuPDF Font.text_length first sums unit-size character advances,
            # then multiplies once by fontsize.  Preserve that exact operation
            # order while caching the stable unit advances instead of the many
            # one-off line prefixes produced while wrapping CJK text.
            unit_width = 0.0
            for character in text:
                cache_key = (id(font), character)
                width = self._glyph_width_cache.get(cache_key)
                if width is None:
                    width = float(font.text_length(character, fontsize=1.0))
                    self._measurement_cache_stats["glyph_width_misses"] += 1
                    if len(self._glyph_width_cache) >= self._measurement_cache_max_entries:
                        self._glyph_width_cache.pop(next(iter(self._glyph_width_cache)))
                        self._measurement_cache_stats["glyph_width_evictions"] += 1
                    self._glyph_width_cache[cache_key] = width
                else:
                    self._measurement_cache_stats["glyph_width_hits"] += 1
                unit_width += width
            base = unit_width * font_size
        gaps = max(0, self._character_count(text) - 1)
        return max(0.0, base + gaps * char_spacing_em * font_size)

    def _split_long_token(
        self,
        token: str,
        *,
        font: fitz.Font,
        font_size: float,
        char_spacing_em: float,
        available_width: float,
    ) -> list[str]:
        chunks: list[str] = []
        current = ""
        for character in token:
            proposed = current + character
            if current and self._text_width(
                font,
                proposed,
                font_size=font_size,
                char_spacing_em=char_spacing_em,
            ) > available_width:
                chunks.append(current)
                current = character
            else:
                current = proposed
        if current:
            chunks.append(current)
        return chunks or [token]

    def _wrap(
        self,
        text: str,
        *,
        font: fitz.Font,
        font_size: float,
        char_spacing_em: float,
        available_width: float,
    ) -> tuple[list[str], list[float]]:
        if not text:
            return [], []
        lines: list[str] = []
        widths: list[float] = []
        current = ""
        pending_space = False

        def push() -> None:
            nonlocal current, pending_space
            clean = current.rstrip()
            lines.append(clean)
            widths.append(
                self._text_width(
                    font,
                    clean,
                    font_size=font_size,
                    char_spacing_em=char_spacing_em,
                )
            )
            current = ""
            pending_space = False

        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        for token in self._tokens(normalized):
            if token == "\n":
                push()
                continue
            if token.isspace():
                pending_space = bool(current)
                continue
            prefix = " " if pending_space and current else ""
            proposed = current + prefix + token
            proposed_width = self._text_width(
                font,
                proposed,
                font_size=font_size,
                char_spacing_em=char_spacing_em,
            )
            if not current or proposed_width <= available_width:
                current = proposed
                pending_space = False
                continue

            push()
            token_width = self._text_width(
                font,
                token,
                font_size=font_size,
                char_spacing_em=char_spacing_em,
            )
            if token_width <= available_width:
                current = token
                continue
            chunks = self._split_long_token(
                token,
                font=font,
                font_size=font_size,
                char_spacing_em=char_spacing_em,
                available_width=available_width,
            )
            for chunk in chunks[:-1]:
                current = chunk
                push()
            current = chunks[-1]

        if current or not lines:
            push()
        return lines, widths

    def _formula_geometry(
        self,
        spec: FormulaRunSpec,
        *,
        role: str,
        font_size: float,
    ) -> tuple[float, float, float, float]:
        if spec.coordinate_system == "mathjax_ex":
            scale = font_size * MATHJAX_EX_TO_EM
        else:
            role_record = self.roles.get(role) or {}
            source = role_record.get("source_statistics") or {}
            try:
                source_font_size = float(source["font_size_pt"])
            except (KeyError, TypeError, ValueError) as exc:
                raise LayoutMetricError(
                    f"role {role} has no source font size for PDF formula scaling"
                ) from exc
            if source_font_size <= 0:
                raise LayoutMetricError(f"role {role} source font size must be positive")
            scale = font_size / source_font_size
        return (
            spec.natural_width * scale,
            spec.natural_height * scale,
            spec.natural_baseline_from_top * scale,
            scale,
        )

    @staticmethod
    def _merge_text_runs(runs: list[dict[str, Any]], text: str, width: float) -> None:
        if not text:
            return
        if runs and runs[-1]["kind"] == "text":
            runs[-1]["text"] += text
            runs[-1]["width"] += width
        else:
            runs.append({"kind": "text", "text": text, "width": width})

    def _wrap_mixed(
        self,
        text: str,
        *,
        formula_runs: tuple[FormulaRunSpec, ...],
        role: str,
        font: fitz.Font,
        font_size: float,
        line_height_multiplier: float,
        char_spacing_em: float,
        available_width: float,
    ) -> tuple[list[str], list[float], list[MixedLine]]:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        if normalized != text and formula_runs:
            raise LayoutMetricError("formula offsets require normalized newlines")
        ordered = tuple(sorted(formula_runs, key=lambda item: (item.start, item.end)))
        cursor = 0
        atoms: list[tuple[str, str | FormulaRunSpec]] = []
        for spec in ordered:
            if spec.start < cursor or spec.end > len(text) or text[spec.start : spec.end] != spec.text:
                raise LayoutMetricError(f"formula run {spec.formula_id} does not match measured text")
            for token in self._tokens(text[cursor : spec.start]):
                atoms.append(("text", token))
            atoms.append(("formula", spec))
            cursor = spec.end
        for token in self._tokens(text[cursor:]):
            atoms.append(("text", token))

        raw_lines: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        current_width = 0.0
        pending_space = False

        def trim_trailing_space() -> None:
            nonlocal current_width
            if not current or current[-1]["kind"] != "text":
                return
            old = current[-1]["text"]
            new = old.rstrip()
            if new == old:
                return
            old_width = current[-1]["width"]
            new_width = self._text_width(
                font,
                new,
                font_size=font_size,
                char_spacing_em=char_spacing_em,
            )
            current[-1]["text"] = new
            current[-1]["width"] = new_width
            current_width -= old_width - new_width
            if not new:
                current.pop()

        def push() -> None:
            nonlocal current, current_width, pending_space
            trim_trailing_space()
            raw_lines.append(current)
            current = []
            current_width = 0.0
            pending_space = False

        def text_increment(value: str) -> tuple[float, float]:
            if current and current[-1]["kind"] == "text":
                before = current[-1]["text"]
                before_width = current[-1]["width"]
                after_width = self._text_width(
                    font,
                    before + value,
                    font_size=font_size,
                    char_spacing_em=char_spacing_em,
                )
                return after_width - before_width, after_width
            width = self._text_width(
                font,
                value,
                font_size=font_size,
                char_spacing_em=char_spacing_em,
            )
            return width, width

        def append_text(value: str) -> None:
            nonlocal current_width
            increment, merged_width = text_increment(value)
            if current and current[-1]["kind"] == "text":
                current[-1]["text"] += value
                current[-1]["width"] = merged_width
            else:
                current.append({"kind": "text", "text": value, "width": merged_width})
            current_width += increment

        for kind, value in atoms:
            if kind == "text":
                token = str(value)
                if token == "\n":
                    push()
                    continue
                if token.isspace():
                    pending_space = bool(current)
                    continue
                prefix = " " if pending_space and current else ""
                proposed = prefix + token
                increment, _ = text_increment(proposed)
                if current and current_width + increment > available_width:
                    push()
                    proposed = token
                token_width = self._text_width(
                    font,
                    token,
                    font_size=font_size,
                    char_spacing_em=char_spacing_em,
                )
                if not current and token_width > available_width:
                    chunks = self._split_long_token(
                        token,
                        font=font,
                        font_size=font_size,
                        char_spacing_em=char_spacing_em,
                        available_width=available_width,
                    )
                    for chunk in chunks[:-1]:
                        append_text(chunk)
                        push()
                    append_text(chunks[-1])
                else:
                    append_text(proposed)
                pending_space = False
                continue

            spec = value
            assert isinstance(spec, FormulaRunSpec)
            width, height, baseline, scale = self._formula_geometry(
                spec, role=role, font_size=font_size
            )
            if pending_space and current:
                increment, _ = text_increment(" ")
                if current_width + increment + width > available_width:
                    push()
                else:
                    append_text(" ")
            if current and current_width + width > available_width:
                push()
            current.append(
                {
                    "kind": "formula",
                    "text": spec.text,
                    "width": width,
                    "height": height,
                    "baseline": baseline,
                    "formula_id": spec.formula_id,
                    "asset_id": spec.asset_id,
                    "asset_kind": spec.asset_kind,
                    "scale": scale,
                }
            )
            current_width += width
            pending_space = False

        if current or not raw_lines:
            push()

        text_ascent = max(0.0, float(font.ascender) * font_size)
        base_line_height = font_size * line_height_multiplier
        base_baseline = max(0.0, (base_line_height - font_size) / 2) + text_ascent
        base_descent = max(0.0, base_line_height - base_baseline)
        lines: list[str] = []
        widths: list[float] = []
        mixed_lines: list[MixedLine] = []
        for raw in raw_lines:
            formula_baselines = [run["baseline"] for run in raw if run["kind"] == "formula"]
            formula_depths = [
                run["height"] - run["baseline"] for run in raw if run["kind"] == "formula"
            ]
            line_baseline = max([base_baseline, *formula_baselines])
            line_descent = max([base_descent, *formula_depths])
            line_height = line_baseline + line_descent
            x = 0.0
            rendered_runs: list[MixedRun] = []
            for run in raw:
                if run["kind"] == "formula":
                    rendered = MixedRun(
                        kind="formula",
                        text=run["text"],
                        x_offset_pt=round(x, 4),
                        width_pt=round(run["width"], 4),
                        height_pt=round(run["height"], 4),
                        baseline_from_top_pt=round(run["baseline"], 4),
                        formula_id=run["formula_id"],
                        asset_id=run["asset_id"],
                        asset_kind=run["asset_kind"],
                        asset_scale_to_pt=round(run["scale"], 8),
                    )
                else:
                    rendered = MixedRun(
                        kind="text",
                        text=run["text"],
                        x_offset_pt=round(x, 4),
                        width_pt=round(run["width"], 4),
                        height_pt=round(max(font_size, base_baseline), 4),
                        baseline_from_top_pt=round(base_baseline, 4),
                    )
                rendered_runs.append(rendered)
                x += run["width"]
            line_text = "".join(run["text"] for run in raw)
            lines.append(line_text)
            widths.append(x)
            mixed_lines.append(
                MixedLine(
                    text=line_text,
                    width_pt=round(x, 4),
                    height_pt=round(line_height, 4),
                    baseline_from_top_pt=round(line_baseline, 4),
                    runs=tuple(rendered_runs),
                )
            )
        return lines, widths, mixed_lines

    def measure(
        self,
        text: str,
        bbox: list[float] | tuple[float, ...],
        *,
        role: str,
        style: dict[str, Any] | None = None,
        enforce_guardrails: bool = True,
        formula_runs: tuple[FormulaRunSpec, ...] = (),
    ) -> LayoutResult:
        available_width, available_height = self._validate_bbox(bbox)
        golden_style, _ = self.style_for_role(role)
        effective_style = dict(golden_style if style is None else style)
        font_id, font_size, line_height_multiplier, char_spacing_em = self._validate_style(
            effective_style
        )
        if enforce_guardrails:
            self.validate_guardrails(role, effective_style)
        font = self._font(font_id)
        if self.style_profile.get("measurement_engine") == "layout-metrics-v1":
            if formula_runs:
                raise LayoutMetricError("layout-metrics-v1 cannot measure formula runs")
            lines, widths = self._wrap(
                text,
                font=font,
                font_size=font_size,
                char_spacing_em=char_spacing_em,
                available_width=available_width,
            )
            mixed_lines: list[MixedLine] = []
            line_heights = [font_size * line_height_multiplier for _ in lines]
        elif not formula_runs:
            lines, widths = self._wrap(
                text,
                font=font,
                font_size=font_size,
                char_spacing_em=char_spacing_em,
                available_width=available_width,
            )
            base_height = font_size * line_height_multiplier
            text_ascent = max(0.0, float(font.ascender) * font_size)
            baseline = max(0.0, (base_height - font_size) / 2) + text_ascent
            height = max(base_height, baseline + max(0.0, base_height - baseline))
            mixed_lines = [
                MixedLine(
                    text=line,
                    width_pt=round(width, 4),
                    height_pt=round(height, 4),
                    baseline_from_top_pt=round(baseline, 4),
                    runs=(
                        MixedRun(
                            kind="text",
                            text=line,
                            x_offset_pt=0.0,
                            width_pt=round(width, 4),
                            height_pt=round(max(font_size, baseline), 4),
                            baseline_from_top_pt=round(baseline, 4),
                        ),
                    )
                    if line
                    else (),
                )
                for line, width in zip(lines, widths, strict=True)
            ]
            line_heights = [line.height_pt for line in mixed_lines]
        else:
            lines, widths, mixed_lines = self._wrap_mixed(
                text,
                formula_runs=formula_runs,
                role=role,
                font=font,
                font_size=font_size,
                line_height_multiplier=line_height_multiplier,
                char_spacing_em=char_spacing_em,
                available_width=available_width,
            )
            line_heights = [line.height_pt for line in mixed_lines]
        line_height_pt = font_size * line_height_multiplier
        used_width = max(widths, default=0.0)
        used_height = sum(line_heights) if text else 0.0
        overflow_width = max(0.0, used_width - available_width)
        overflow_height = max(0.0, used_height - available_height)
        fits = overflow_width <= 1e-6 and overflow_height <= 1e-6
        width_ratio = used_width / available_width if available_width else 0.0
        height_ratio = used_height / available_height if available_height else 0.0
        used_area = sum(width * height for width, height in zip(widths, line_heights, strict=True))
        available_area = available_width * available_height
        cjk_width = max(font.text_length("汉", fontsize=font_size), 0.001)
        cjk_width += char_spacing_em * font_size
        chars_per_line = max(1, math.floor(available_width / max(cjk_width, 0.001)))
        max_lines = max(0, math.floor(available_height / max(line_height_pt, 0.001)))
        return LayoutResult(
            fits=fits,
            is_overflow=not fits,
            available_width_pt=round(available_width, 4),
            available_height_pt=round(available_height, 4),
            used_width_pt=round(used_width, 4),
            used_height_pt=round(used_height, 4),
            overflow_width_pt=round(overflow_width, 4),
            overflow_height_pt=round(overflow_height, 4),
            line_count=len(lines),
            line_height_pt=round(line_height_pt, 4),
            width_fill_ratio=round(width_ratio, 6),
            height_fill_ratio=round(height_ratio, 6),
            area_fill_ratio=round(used_area / available_area, 6) if available_area else 0.0,
            estimated_cjk_capacity=chars_per_line * max_lines,
            lines=tuple(lines),
            line_widths_pt=tuple(round(width, 4) for width in widths),
            font_id=font_id,
            font_size_pt=font_size,
            line_height_multiplier=line_height_multiplier,
            char_spacing_em=char_spacing_em,
            mixed_lines=tuple(mixed_lines),
        )

    def with_line_height(
        self,
        base: LayoutResult,
        *,
        line_height_multiplier: float,
    ) -> LayoutResult:
        """Recompute vertical geometry while preserving the exact v2 wrapping."""

        if not base.mixed_lines:
            line_height_pt = base.font_size_pt * line_height_multiplier
            used_height = base.line_count * line_height_pt if base.lines else 0.0
            mixed_lines: tuple[MixedLine, ...] = ()
        else:
            font = self._font(base.font_id)
            font_size = base.font_size_pt
            base_height = font_size * line_height_multiplier
            text_ascent = max(0.0, float(font.ascender) * font_size)
            text_baseline = max(0.0, (base_height - font_size) / 2) + text_ascent
            text_descent = max(0.0, base_height - text_baseline)
            rebuilt: list[MixedLine] = []
            for line in base.mixed_lines:
                formula_runs = [run for run in line.runs if run.kind == "formula"]
                baseline = max(
                    [text_baseline, *(run.baseline_from_top_pt for run in formula_runs)]
                )
                descent = max(
                    [
                        text_descent,
                        *(
                            run.height_pt - run.baseline_from_top_pt
                            for run in formula_runs
                        ),
                    ]
                )
                adjusted_runs = tuple(
                    replace(
                        run,
                        height_pt=round(max(font_size, text_baseline), 4),
                        baseline_from_top_pt=round(text_baseline, 4),
                    )
                    if run.kind == "text"
                    else run
                    for run in line.runs
                )
                rebuilt.append(
                    replace(
                        line,
                        height_pt=round(baseline + descent, 4),
                        baseline_from_top_pt=round(baseline, 4),
                        runs=adjusted_runs,
                    )
                )
            mixed_lines = tuple(rebuilt)
            used_height = sum(line.height_pt for line in mixed_lines) if base.lines else 0.0
            line_height_pt = base_height
        overflow_height = max(0.0, used_height - base.available_height_pt)
        fits = base.overflow_width_pt <= 1e-6 and overflow_height <= 1e-6
        available_area = base.available_width_pt * base.available_height_pt
        if mixed_lines:
            used_area = sum(line.width_pt * line.height_pt for line in mixed_lines)
        else:
            used_area = sum(base.line_widths_pt) * line_height_pt
        base_max_lines = max(
            0, math.floor(base.available_height_pt / max(base.line_height_pt, 0.001))
        )
        chars_per_line = (
            max(1, base.estimated_cjk_capacity // base_max_lines)
            if base_max_lines
            else max(1, base.estimated_cjk_capacity)
        )
        max_lines = max(0, math.floor(base.available_height_pt / max(line_height_pt, 0.001)))
        return replace(
            base,
            fits=fits,
            is_overflow=not fits,
            used_height_pt=round(used_height, 4),
            overflow_height_pt=round(overflow_height, 4),
            line_height_pt=round(line_height_pt, 4),
            height_fill_ratio=round(
                used_height / base.available_height_pt if base.available_height_pt else 0.0,
                6,
            ),
            area_fill_ratio=round(used_area / available_area, 6) if available_area else 0.0,
            estimated_cjk_capacity=chars_per_line * max_lines,
            line_height_multiplier=line_height_multiplier,
            mixed_lines=mixed_lines,
        )
