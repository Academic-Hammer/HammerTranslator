"""Backward-compatible imports for the former Mac-only parser executor."""

from .mineru_parser import (
    PARSE_MANIFEST_SCHEMA_VERSION,
    PARSER_MODULE,
    MinerUParsingExecutor,
    parse_mode,
    safe_document_stem,
)

MacMinerUParsingExecutor = MinerUParsingExecutor

__all__ = [
    "MacMinerUParsingExecutor",
    "MinerUParsingExecutor",
    "PARSE_MANIFEST_SCHEMA_VERSION",
    "PARSER_MODULE",
    "parse_mode",
    "safe_document_stem",
]
