#!/usr/bin/env python3
"""Batch PDF parsing tuned for Apple Silicon Macs.

The script uses MinerU's local source tree, processes one document at a time to
bound unified-memory use, selects MPS for pipeline models, and lets MinerU select
MLX for VLM/hybrid models.  It also runs ``MinerUAdapter`` after parsing so the
standard content-list output uses PDF-point coordinates required downstream.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

ROOT_DIR = Path(__file__).resolve().parent
LOCAL_MINERU_DIR = ROOT_DIR / "MinerU"
DEFAULT_INPUT_DIR = LOCAL_MINERU_DIR / "demo" / "pdfs"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "demo-output-mac"

LOGGER = logging.getLogger("mac-pdf-parser")
PARSE_MANIFEST_SCHEMA_VERSION = "mineru-parse-v1"


@dataclass(frozen=True)
class RuntimeConfig:
    backend: str
    method: str
    language: str
    device: str
    formula: bool
    table: bool
    formula_ch: bool
    image_analysis: bool
    effort: str
    start_page: int
    end_page: int | None
    draw_bboxes: bool
    adapt: bool
    render_threads: int
    window_size: int


def collect_pdfs(source: Path) -> list[tuple[Path, Path]]:
    """Return PDF files and their relative parent directories."""
    if source.is_file():
        if source.suffix.lower() != ".pdf":
            raise ValueError(f"Input file is not a PDF: {source}")
        return [(source.resolve(), Path())]
    if not source.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {source}")
    files = [path for path in source.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf"]
    return [(path.resolve(), path.parent.relative_to(source)) for path in sorted(files)]


def read_mac_hardware() -> dict[str, Any]:
    profile: dict[str, Any] = {
        "system": platform.system(),
        "machine": platform.machine(),
        "macos": platform.mac_ver()[0],
    }
    if platform.system() != "Darwin":
        return profile
    try:
        completed = subprocess.run(
            ["system_profiler", "-json", "SPHardwareDataType", "SPDisplaysDataType"],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
        data = json.loads(completed.stdout)
        hardware = (data.get("SPHardwareDataType") or [{}])[0]
        displays = (data.get("SPDisplaysDataType") or [{}])[0]
        profile.update(
            {
                "model": hardware.get("machine_name"),
                "chip": hardware.get("chip_type"),
                "memory": hardware.get("physical_memory"),
                "cpu_cores": hardware.get("number_processors"),
                "gpu": displays.get("sppci_model"),
                "gpu_cores": displays.get("sppci_cores"),
                "metal": displays.get("spdisplays_metal"),
            }
        )
        if not profile.get("metal") and profile.get("gpu") and profile.get("machine") in {"arm64", "aarch64"}:
            profile["metal"] = "supported (Apple Silicon)"
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, IndexError):
        profile["profile_warning"] = "system_profiler was unavailable"
    return profile


def ensure_local_mineru_path() -> None:
    if not (LOCAL_MINERU_DIR / "mineru" / "__init__.py").is_file():
        raise FileNotFoundError(f"Local MinerU source was not found at {LOCAL_MINERU_DIR}")
    path = str(LOCAL_MINERU_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)


def module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def inspect_accelerators() -> dict[str, Any]:
    result: dict[str, Any] = {
        "mps_built": False,
        "mps_available": False,
        "mlx_available": module_available("mlx"),
        "mlx_vlm_available": module_available("mlx_vlm"),
    }
    if module_available("torch"):
        import torch

        result["torch_version"] = torch.__version__
        mps = getattr(torch.backends, "mps", None)
        result["mps_built"] = bool(mps and mps.is_built())
        result["mps_available"] = bool(mps and mps.is_available())
    return result


def preflight(config: RuntimeConfig) -> tuple[bool, list[str]]:
    ensure_local_mineru_path()
    required = {
        "loguru": "MinerU logging",
        "mineru": "local MinerU package",
        "torch": "pipeline/MPS runtime",
    }
    if config.backend in {"pipeline", "hybrid-engine"}:
        required.update(
            {
                "transformers": "pipeline models",
                "yaml": "MinerU configuration",
                "cv2": "image processing",
                "onnxruntime": "pipeline inference",
            }
        )
    if config.backend in {"vlm-engine", "hybrid-engine"}:
        required.update({"mlx": "Apple GPU runtime", "mlx_vlm": "MinerU VLM on MLX"})
    if config.adapt:
        required["fitz"] = "PDF style extraction"

    messages: list[str] = []
    ok = True
    for module, purpose in required.items():
        available = module_available(module)
        messages.append(f"{'OK' if available else 'MISSING'}  {module:<14} {purpose}")
        ok = ok and available

    accelerators = inspect_accelerators()
    if config.device == "mps" and not accelerators["mps_available"]:
        messages.append("ERROR  MPS was requested but is not available to PyTorch")
        ok = False
    elif config.device == "auto" and module_available("torch"):
        selected = "mps" if accelerators["mps_available"] else "cpu"
        messages.append(f"INFO   automatic pipeline device: {selected}")
    if config.backend in {"vlm-engine", "hybrid-engine"}:
        engine = "mlx" if accelerators["mlx_vlm_available"] else "transformers fallback"
        messages.append(f"INFO   automatic VLM engine: {engine}")
    return ok, messages


def configure_environment(config: RuntimeConfig) -> str:
    """Set conservative Apple-Silicon defaults before MinerU is imported."""
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["MINERU_PDF_RENDER_THREADS"] = str(config.render_threads)
    os.environ["MINERU_PROCESSING_WINDOW_SIZE"] = str(config.window_size)
    os.environ["MINERU_INTRA_OP_NUM_THREADS"] = "4"
    os.environ["MINERU_INTER_OP_NUM_THREADS"] = "2"
    os.environ["MINERU_FORMULA_CH_SUPPORT"] = str(config.formula_ch).lower()

    device = config.device
    if device == "auto":
        accelerators = inspect_accelerators()
        device = "mps" if accelerators["mps_available"] else "cpu"
    os.environ["MINERU_DEVICE_MODE"] = device
    return device


def expected_parse_dir(output_root: Path, stem: str, config: RuntimeConfig) -> Path:
    if config.backend == "pipeline":
        mode = config.method
    elif config.backend == "vlm-engine":
        mode = "vlm"
    else:
        mode = f"hybrid_{config.method}"
    return output_root / stem / mode


def normalize_document_name(value: str) -> str:
    """Return a path-safe output stem without changing the source PDF path."""

    name = Path(value).name
    if name != value or not name or "\x00" in name:
        raise ValueError("document name must be a plain filename")
    stem = Path(name).stem if Path(name).suffix.lower() == ".pdf" else name
    normalized = re.sub(r"[^\w.-]+", "_", stem, flags=re.UNICODE).strip("._")
    if not normalized:
        raise ValueError("document name must contain a usable filename stem")
    return normalized


def adapt_mineru_output(
    parse_dir: Path,
    source_pdf: Path,
    *,
    document_name: str | None = None,
) -> list[str]:
    from mineru_adapter import MinerUAdapter

    stem = document_name or source_pdf.stem
    middle_path = parse_dir / f"{stem}_middle.json"
    if not middle_path.is_file():
        raise FileNotFoundError(f"MinerU did not produce {middle_path}")

    native_content = parse_dir / f"{stem}_content_list.json"
    native_backup = parse_dir / f"{stem}_mineru_content_list.json"
    if native_content.is_file():
        shutil.copy2(native_content, native_backup)

    origin_pdf = parse_dir / f"{stem}_origin.pdf"
    pdf_path = origin_pdf if origin_pdf.is_file() else source_pdf
    result = MinerUAdapter().adapt_file(
        middle_path,
        pdf_path=pdf_path,
        output_dir=parse_dir,
        document_name=stem,
    )
    for warning in result.warnings:
        LOGGER.warning("Adapter: %s", warning)
    return list(result.warnings)


def write_parse_manifest(
    parse_dir: Path,
    source_pdf: Path,
    document_name: str,
    config: RuntimeConfig,
    adapter_warnings: list[str],
) -> Path:
    """Write the stable hand-off record consumed by the unified pipeline."""

    manifest_path = parse_dir / f"{document_name}_parse_manifest.json"
    files = sorted(
        str(path.relative_to(parse_dir)).replace(os.sep, "/")
        for path in parse_dir.rglob("*")
        if path.is_file() and path != manifest_path
    )
    payload = {
        "schema_version": PARSE_MANIFEST_SCHEMA_VERSION,
        "document_name": document_name,
        "source_filename": source_pdf.name,
        "backend": config.backend,
        "method": config.method,
        "language": config.language,
        "adapter_enabled": config.adapt,
        "adapter_warnings": adapter_warnings,
        "files": files,
    }
    temporary = manifest_path.with_name(f".{manifest_path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, manifest_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return manifest_path


def parse_one(
    pdf_path: Path,
    output_root: Path,
    config: RuntimeConfig,
    *,
    document_name: str | None = None,
) -> Path:
    ensure_local_mineru_path()
    from mineru.cli.common import do_parse

    output_name = normalize_document_name(document_name) if document_name else pdf_path.stem
    parse_dir = expected_parse_dir(output_root, output_name, config)
    with pdf_path.open("rb") as handle:
        pdf_bytes = handle.read()
    do_parse(
        output_dir=str(output_root),
        pdf_file_names=[output_name],
        pdf_bytes_list=[pdf_bytes],
        p_lang_list=[config.language],
        backend=config.backend,
        parse_method=config.method,
        formula_enable=config.formula,
        table_enable=config.table,
        f_draw_layout_bbox=config.draw_bboxes,
        f_draw_span_bbox=config.draw_bboxes,
        start_page_id=config.start_page,
        end_page_id=config.end_page,
        image_analysis=config.image_analysis,
        effort=config.effort,
    )
    adapter_warnings: list[str] = []
    if config.adapt:
        adapter_warnings = adapt_mineru_output(
            parse_dir,
            pdf_path,
            document_name=output_name,
        )
    write_parse_manifest(
        parse_dir,
        pdf_path,
        output_name,
        config,
        adapter_warnings,
    )
    return parse_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch PDF parsing for Apple Silicon Macs")
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT_DIR, help="PDF file or directory")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="output root directory")
    parser.add_argument(
        "--document-name",
        help="output filename stem for a single input PDF (used by the unified pipeline)",
    )
    parser.add_argument(
        "--backend",
        choices=["pipeline", "vlm-engine", "hybrid-engine"],
        default="pipeline",
        help="pipeline uses MPS; VLM/hybrid use MLX when installed",
    )
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--method", choices=["auto", "txt", "ocr"], default="auto")
    parser.add_argument("--lang", default="ch")
    parser.add_argument("--formula", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--table", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--formula-ch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image-analysis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--effort", choices=["medium", "high"], default="medium")
    parser.add_argument("--adapt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--draw-bboxes", action="store_true")
    parser.add_argument("--start-page", type=int, default=0)
    parser.add_argument("--end-page", type=int)
    parser.add_argument("--render-threads", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check", action="store_true", help="inspect hardware and dependencies without parsing")
    parser.add_argument("--dry-run", action="store_true", help="show planned work without loading MinerU")
    return parser


def config_from_args(args: argparse.Namespace) -> RuntimeConfig:
    if args.start_page < 0:
        raise ValueError("--start-page must be non-negative")
    if args.end_page is not None and args.end_page < args.start_page:
        raise ValueError("--end-page must not be smaller than --start-page")
    if args.render_threads < 1 or args.window_size < 1:
        raise ValueError("render threads and window size must be positive")
    return RuntimeConfig(
        backend=args.backend,
        method=args.method,
        language=args.lang,
        device=args.device,
        formula=args.formula,
        table=args.table,
        formula_ch=args.formula_ch,
        image_analysis=args.image_analysis,
        effort=args.effort,
        start_page=args.start_page,
        end_page=args.end_page,
        draw_bboxes=args.draw_bboxes,
        adapt=args.adapt,
        render_threads=args.render_threads,
        window_size=args.window_size,
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = build_parser().parse_args(argv)
    try:
        config = config_from_args(args)
    except ValueError as exc:
        LOGGER.error("Invalid configuration: %s", exc)
        return 2

    hardware = read_mac_hardware()
    LOGGER.info("Hardware: %s", json.dumps(hardware, ensure_ascii=False))
    if hardware.get("system") != "Darwin" or hardware.get("machine") not in {"arm64", "aarch64"}:
        LOGGER.error("This entry point is intended for Apple Silicon macOS")
        return 2

    ready, checks = preflight(config)
    for message in checks:
        LOGGER.info(message)
    if args.check:
        return 0 if ready else 2
    if not ready and not args.dry_run:
        LOGGER.error("Dependencies are incomplete. Install requirements-macos.txt in hammertrans first.")
        return 2

    try:
        pdfs = collect_pdfs(args.input.resolve())
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 2
    if not pdfs:
        LOGGER.error("No PDF files found under %s", args.input)
        return 2
    if args.document_name and len(pdfs) != 1:
        LOGGER.error("--document-name is allowed only for one input PDF")
        return 2
    try:
        requested_name = (
            normalize_document_name(args.document_name) if args.document_name else None
        )
    except ValueError as exc:
        LOGGER.error("Invalid document name: %s", exc)
        return 2

    LOGGER.info(
        "Plan: %d PDF(s), backend=%s, device=%s, output=%s",
        len(pdfs),
        config.backend,
        config.device,
        args.output.resolve(),
    )
    if args.dry_run:
        for pdf_path, relative_parent in pdfs:
            LOGGER.info(
                "DRY RUN %s -> %s (document=%s)",
                pdf_path,
                args.output.resolve() / relative_parent,
                requested_name or pdf_path.stem,
            )
        return 0

    selected_device = configure_environment(config)
    LOGGER.info("Selected pipeline device: %s", selected_device)
    success = 0
    failed = 0
    skipped = 0
    for index, (pdf_path, relative_parent) in enumerate(pdfs, start=1):
        output_root = args.output.resolve() / relative_parent
        output_name = requested_name or pdf_path.stem
        parse_dir = expected_parse_dir(output_root, output_name, config)
        middle_path = parse_dir / f"{output_name}_middle.json"
        if args.skip_existing and middle_path.is_file():
            LOGGER.info("[%d/%d] Skip existing: %s", index, len(pdfs), pdf_path)
            skipped += 1
            continue
        LOGGER.info("[%d/%d] Parse: %s", index, len(pdfs), pdf_path)
        started = time.monotonic()
        try:
            result_dir = parse_one(
                pdf_path,
                output_root,
                config,
                document_name=requested_name,
            )
            LOGGER.info("Completed in %.1fs: %s", time.monotonic() - started, result_dir)
            success += 1
        except Exception:
            LOGGER.exception("Failed: %s", pdf_path)
            failed += 1
            if not args.continue_on_error:
                break

    LOGGER.info("Summary: success=%d failed=%d skipped=%d", success, failed, skipped)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
