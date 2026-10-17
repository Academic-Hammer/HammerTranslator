"""Safe configuration loading for the translation stage.

This module deliberately contains no HTTP client.  Step 1 only validates local
configuration and environment variables; it cannot make a provider request.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml
from dotenv import dotenv_values


ENV_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
ENV_REFERENCE_RE = re.compile(r"^\$\{([A-Z][A-Z0-9_]*)\}$")
SENSITIVE_KEY_PARTS = ("api_key", "token", "secret", "password")


class ConfigError(ValueError):
    """Raised when translation configuration is unsafe or incomplete."""


@dataclass(frozen=True)
class ProviderConfig:
    """Resolved, non-secret provider configuration."""

    name: str
    base_url: str
    model: str
    api_key_env: str
    credential_configured: bool

    def public_summary(self) -> dict[str, Any]:
        """Return fields safe to print or save in a preflight report."""

        return {
            "name": self.name,
            "base_url": self.base_url,
            "model": self.model,
            "api_key_env": self.api_key_env,
            "credential_configured": self.credential_configured,
            "api_key": "<redacted>",
        }


def _is_sensitive_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return any(part in normalized for part in SENSITIVE_KEY_PARTS)


def _audit_no_inline_secrets(value: Any, path: tuple[str, ...] = ()) -> None:
    """Reject literal credentials anywhere in a YAML document.

    ``api_key_env`` is safe because it names an environment variable rather
    than containing a credential.  Legacy ``api_key: ${NAME}`` references are
    also accepted so old configuration can be migrated without exposing keys.
    """

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            child_path = (*path, key)
            if key.lower().replace("-", "_").endswith("_env"):
                if child and (not isinstance(child, str) or not ENV_NAME_RE.fullmatch(child)):
                    raise ConfigError(
                        f"{'.'.join(child_path)} must be an environment-variable name"
                    )
            elif _is_sensitive_key(key) and child not in (None, ""):
                if not isinstance(child, str) or not ENV_REFERENCE_RE.fullmatch(child):
                    raise ConfigError(
                        f"{'.'.join(child_path)} contains an inline credential; "
                        "use ${ENV_VAR} or api_key_env instead"
                    )
            _audit_no_inline_secrets(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _audit_no_inline_secrets(child, (*path, str(index)))


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML and reject any credential stored directly in the file."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")

    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {config_path}: {exc}") from exc

    if not isinstance(loaded, dict):
        raise ConfigError("translation config must be a YAML mapping")
    _audit_no_inline_secrets(loaded)
    return loaded


def load_environment_file(
    path: str | Path,
    *,
    environ: dict[str, str] | None = None,
    override: bool = False,
) -> tuple[str, ...]:
    """Load a local dotenv file without returning or logging any values.

    Existing process variables win by default, which lets CI, shells, and secret
    managers override the developer-local project file safely.
    """

    env_path = Path(path).expanduser().resolve()
    if not env_path.exists():
        return ()
    if not env_path.is_file():
        raise ConfigError(f"environment file is not a regular file: {env_path}")
    try:
        values = dotenv_values(env_path)
    except (OSError, ValueError) as exc:
        raise ConfigError(f"cannot read environment file: {env_path}") from exc
    target = os.environ if environ is None else environ
    loaded: list[str] = []
    for raw_name, raw_value in values.items():
        name = str(raw_name)
        if not ENV_NAME_RE.fullmatch(name):
            raise ConfigError(
                f"environment file contains an invalid variable name: {name!r}"
            )
        if raw_value is None:
            continue
        if override or name not in target:
            target[name] = str(raw_value)
            loaded.append(name)
    return tuple(loaded)


def resolve_provider(
    config: Mapping[str, Any],
    provider_name: str,
    *,
    environ: Mapping[str, str] | None = None,
    model_override: str | None = None,
) -> ProviderConfig:
    """Resolve one explicitly selected provider without exposing its key."""

    providers = config.get("providers")
    if not isinstance(providers, Mapping) or not providers:
        raise ConfigError("config must define a non-empty providers mapping")

    raw = providers.get(provider_name)
    if not isinstance(raw, Mapping):
        available = ", ".join(sorted(str(name) for name in providers))
        raise ConfigError(
            f"unknown provider {provider_name!r}; available providers: {available or '<none>'}"
        )

    base_url = str(raw.get("base_url") or "").strip()
    model = str(model_override or raw.get("model") or "").strip()
    api_key_env = str(raw.get("api_key_env") or "").strip()
    if not base_url.startswith(("https://", "http://")):
        raise ConfigError(f"providers.{provider_name}.base_url must be an HTTP(S) URL")
    if not model:
        raise ConfigError(f"providers.{provider_name}.model is required")
    if not ENV_NAME_RE.fullmatch(api_key_env):
        raise ConfigError(
            f"providers.{provider_name}.api_key_env must name an environment variable"
        )

    env = os.environ if environ is None else environ
    return ProviderConfig(
        name=provider_name,
        base_url=base_url,
        model=model,
        api_key_env=api_key_env,
        credential_configured=bool(env.get(api_key_env)),
    )
