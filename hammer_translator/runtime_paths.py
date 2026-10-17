"""Centralized filesystem locations for source and installed runtimes.

The application keeps operator-owned state outside the Python package while
still preserving the historical paths when it is executed from a source
checkout.  Every location can be overridden explicitly for deployment.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Mapping


APP_HOME_ENV = "HAMMERTRANSLATOR_HOME"
JOBS_ROOT_ENV = "HAMMERTRANSLATOR_JOBS_ROOT"
CONFIG_PATH_ENV = "HAMMERTRANSLATOR_CONFIG"
ENV_FILE_ENV = "HAMMERTRANSLATOR_ENV_FILE"
FONTS_DIR_ENV = "HAMMERTRANSLATOR_FONTS_DIR"
FORMULA_RUNTIME_DIR_ENV = "HAMMERTRANSLATOR_FORMULA_RUNTIME_DIR"
MODEL_CACHE_DIR_ENV = "HAMMERTRANSLATOR_MODEL_CACHE"
TEMP_DIR_ENV = "HAMMERTRANSLATOR_TEMP_DIR"
MINERU_SOURCE_ENV = "HAMMERTRANSLATOR_MINERU_SOURCE"


def _configured_path(environ: Mapping[str, str], name: str, default: Path) -> Path:
    raw = environ.get(name, "").strip()
    return Path(raw).expanduser().resolve() if raw else default.resolve()


def discover_source_root() -> Path | None:
    """Return the active checkout root, if this package is running from one."""

    package_parent = Path(__file__).resolve().parents[1]
    if (package_parent / "pyproject.toml").is_file():
        return package_parent
    return None


def distribution_is_installed() -> bool:
    """Whether child processes can import the packaged application globally."""

    try:
        metadata.version("hammer-translator")
    except metadata.PackageNotFoundError:
        return False
    return True


@dataclass(frozen=True)
class RuntimePaths:
    package_root: Path
    resources_root: Path
    source_root: Path | None
    app_home: Path
    jobs_root: Path
    config_path: Path
    env_file: Path
    fonts_dir: Path
    font_manifest_path: Path
    formula_runtime_dir: Path
    model_cache_dir: Path
    temp_dir: Path
    input_dir: Path
    output_dir: Path
    local_mineru_dir: Path | None


def runtime_paths(environ: Mapping[str, str] | None = None) -> RuntimePaths:
    """Resolve all application paths without creating directories."""

    env = os.environ if environ is None else environ
    package_root = Path(__file__).resolve().parent
    resources_root = package_root / "resources"
    source_root = discover_source_root()
    app_home = _configured_path(
        env,
        APP_HOME_ENV,
        Path.home() / ".hammer-translator",
    )

    source_config = source_root / "translation_config.yaml" if source_root else None
    config_default = (
        source_config
        if source_config is not None and source_config.is_file()
        else resources_root / "translation_config.yaml"
    )
    operator_home_is_explicit = bool(env.get(APP_HOME_ENV, "").strip())
    source_fonts = source_root / "assets" / "fonts" if source_root else None
    fonts_default = (
        source_fonts
        if (
            not operator_home_is_explicit
            and source_fonts is not None
            and source_fonts.is_dir()
        )
        else app_home / "fonts"
    )
    source_formula = source_root / "formula_runtime" if source_root else None
    formula_default = (
        source_formula
        if source_formula is not None and source_formula.is_dir()
        else resources_root / "formula_runtime"
    )
    source_state = (
        source_root
        if source_root is not None and not operator_home_is_explicit
        else app_home
    )

    fonts_dir = _configured_path(env, FONTS_DIR_ENV, fonts_default)
    local_manifest = fonts_dir / "manifest.json"
    manifest = (
        local_manifest
        if local_manifest.is_file()
        else resources_root / "fonts" / "manifest.json"
    )
    return RuntimePaths(
        package_root=package_root,
        resources_root=resources_root,
        source_root=source_root,
        app_home=app_home,
        jobs_root=_configured_path(env, JOBS_ROOT_ENV, source_state / "var" / "jobs"),
        config_path=_configured_path(env, CONFIG_PATH_ENV, config_default),
        env_file=_configured_path(env, ENV_FILE_ENV, source_state / ".env"),
        fonts_dir=fonts_dir,
        font_manifest_path=manifest.resolve(),
        formula_runtime_dir=_configured_path(
            env, FORMULA_RUNTIME_DIR_ENV, formula_default
        ),
        model_cache_dir=_configured_path(
            env, MODEL_CACHE_DIR_ENV, app_home / "models"
        ),
        temp_dir=_configured_path(
            env,
            TEMP_DIR_ENV,
            Path(tempfile.gettempdir()) / "hammer-translator",
        ),
        input_dir=(source_state / "input_pdf").resolve(),
        output_dir=(source_state / "demo-output").resolve(),
        local_mineru_dir=(source_root / "external" / "MinerU") if source_root else None,
    )


def child_process_cwd(job_dir: Path) -> Path:
    """Choose a neutral cwd, falling back to checkout root for uninstalled dev."""

    paths = runtime_paths()
    if distribution_is_installed() or paths.source_root is None:
        return job_dir.resolve()
    return paths.source_root


def portable_resource_label(path: Path) -> str:
    """Describe a runtime resource without assuming it belongs to the checkout."""

    resolved = path.expanduser().resolve()
    paths = runtime_paths()
    roots = (
        (paths.source_root, ""),
        (paths.resources_root, "package"),
        (paths.app_home, "app_home"),
    )
    for root, label in roots:
        if root is None:
            continue
        try:
            relative = resolved.relative_to(root.resolve()).as_posix()
        except ValueError:
            continue
        return relative if not label else f"{label}:{relative}"
    return str(resolved)
