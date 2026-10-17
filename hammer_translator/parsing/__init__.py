"""Public parsing contracts and the current Apple-Silicon implementation."""

from .cli import (
    PARSE_MANIFEST_SCHEMA_VERSION,
    RuntimeConfig,
    adapt_mineru_output,
    build_parser,
    collect_pdfs,
    config_from_args,
    configure_environment,
    expected_parse_dir,
    main,
    normalize_document_name,
    parse_one,
    preflight,
    read_mac_hardware,
    write_parse_manifest,
)

__all__ = [
    "PARSE_MANIFEST_SCHEMA_VERSION",
    "RuntimeConfig",
    "adapt_mineru_output",
    "build_parser",
    "collect_pdfs",
    "config_from_args",
    "configure_environment",
    "expected_parse_dir",
    "main",
    "normalize_document_name",
    "parse_one",
    "preflight",
    "read_mac_hardware",
    "write_parse_manifest",
]
