"""Install the pinned, freely redistributable default fonts on a runtime host."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Sequence

from hammer_translator.font_assets import FontAssetError, required_font_path
from hammer_translator.runtime_paths import runtime_paths
from hammer_translator.translation.config import ConfigError, load_environment_file


class FontInstallError(RuntimeError):
    """A font could not be installed or verified."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bundled_manifest() -> tuple[Path, dict]:
    resource_dir = runtime_paths().resources_root / "fonts"
    path = resource_dir / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FontInstallError(f"cannot read bundled font manifest: {path}") from exc
    if manifest.get("schema_version") != "hammertranslator-font-assets-v1":
        raise FontInstallError("bundled font manifest has an unsupported version")
    return resource_dir, manifest


def _atomic_copy(source: Path, destination: Path) -> None:
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            with source.open("rb") as reader:
                shutil.copyfileobj(reader, handle, length=1024 * 1024)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _download_font(url: str, destination: Path, *, digest: str, size: int) -> None:
    if not url.startswith("https://raw.githubusercontent.com/"):
        raise FontInstallError(f"font source must be an official HTTPS GitHub URL: {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "HammerTranslator/1.0"})
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        received = 0
        checksum = hashlib.sha256()
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                while chunk := response.read(1024 * 1024):
                    received += len(chunk)
                    if received > size:
                        raise FontInstallError(f"download is larger than expected: {destination.name}")
                    checksum.update(chunk)
                    handle.write(chunk)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    try:
        if received != size or checksum.hexdigest() != digest:
            raise FontInstallError(
                f"download verification failed for {destination.name}; no file was installed"
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _copy_verified_font(source: Path, destination: Path, *, digest: str, size: int) -> None:
    if not source.is_file():
        raise FontInstallError(f"font file is missing from offline source: {source}")
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        received = 0
        checksum = hashlib.sha256()
        try:
            with source.open("rb") as reader:
                while chunk := reader.read(1024 * 1024):
                    received += len(chunk)
                    if received > size:
                        raise FontInstallError(f"offline font is too large: {source.name}")
                    checksum.update(chunk)
                    handle.write(chunk)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    try:
        if received != size or checksum.hexdigest() != digest:
            raise FontInstallError(
                f"offline font verification failed for {source.name}; no file was installed"
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def install_default_fonts(
    *,
    repair: bool = False,
    source_dir: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[str]:
    resource_dir, manifest = _bundled_manifest()
    destination_dir = runtime_paths().fonts_dir
    destination_dir.mkdir(parents=True, exist_ok=True)
    bundled_manifest = resource_dir / "manifest.json"
    local_manifest = destination_dir / "manifest.json"
    if local_manifest.is_file() and local_manifest.read_bytes() != bundled_manifest.read_bytes():
        raise FontInstallError(
            f"a different font manifest already exists in {destination_dir}; "
            "choose another HAMMERTRANSLATOR_FONTS_DIR or keep using that custom set"
        )

    results: list[str] = []

    def report(message: str) -> None:
        results.append(message)
        if progress is not None:
            progress(message)

    for filename, record in manifest["fonts"].items():
        if Path(filename).name != filename:
            raise FontInstallError(f"unsafe font filename in manifest: {filename}")
        target = destination_dir / filename
        expected_hash = str(record["sha256"])
        if target.is_file() and _sha256(target) == expected_hash:
            report(f"OK {filename} (already verified)")
            continue
        if target.exists() and not repair:
            raise FontInstallError(
                f"{target} exists but does not match the pinned SHA-256; "
                "use 'hammer-fonts install --repair' to replace this registered file"
            )
        if source_dir is not None:
            report(f"Installing {filename} from {source_dir}")
            _copy_verified_font(
                source_dir / filename,
                target,
                digest=expected_hash,
                size=int(record["size_bytes"]),
            )
        else:
            report(f"Downloading {filename} from its pinned upstream version")
            _download_font(
                str(record["source"]),
                target,
                digest=expected_hash,
                size=int(record["size_bytes"]),
            )
        report(f"OK {filename} (SHA-256 verified)")

    for license_name in sorted({record["license_file"] for record in manifest["fonts"].values()}):
        if Path(license_name).name != license_name:
            raise FontInstallError(f"unsafe license filename in manifest: {license_name}")
        source = resource_dir / license_name
        target = destination_dir / license_name
        if not target.is_file() or target.read_bytes() != source.read_bytes():
            _atomic_copy(source, target)
    if not local_manifest.is_file():
        _atomic_copy(bundled_manifest, local_manifest)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hammer-fonts",
        description="Install and verify the pinned open-source Chinese fonts.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    install = commands.add_parser("install", help="download missing default fonts")
    install.add_argument(
        "--repair",
        action="store_true",
        help="replace only registered font files with invalid hashes",
    )
    install.add_argument(
        "--from-dir",
        type=Path,
        help="install from previously downloaded files when GitHub is unreachable",
    )
    commands.add_parser("check", help="verify the configured font set without network access")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        load_environment_file(runtime_paths().env_file)
        if args.command == "install":
            install_default_fonts(
                repair=args.repair,
                source_dir=args.from_dir,
                progress=lambda message: print(message, flush=True),
            )
        for category in ("title", "heading", "body", "caption"):
            required_font_path(category)
        print(f"Font set ready: {runtime_paths().fonts_dir}")
        return 0
    except (ConfigError, FontAssetError, FontInstallError, OSError, urllib.error.URLError) as exc:
        print(f"font setup failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
