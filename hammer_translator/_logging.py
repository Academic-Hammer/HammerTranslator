"""Shared logging configuration for CLI entry points."""

from __future__ import annotations

import logging
import sys


def configure_cli_logging() -> None:
    """Set up stderr logging for CLI tools without timestamps.

    Structured JSON output continues to use ``print()`` on stdout so that
    downstream tools can pipe it without parsing log prefixes.
    """

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
