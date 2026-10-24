"""Tests for hammer_translator._hashing."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from hammer_translator._hashing import sha256_file


class Sha256FileTests(unittest.TestCase):
    def test_hashes_empty_file(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False) as handle:
            path = Path(handle.name)
        try:
            expected = hashlib.sha256(b"").hexdigest()
            self.assertEqual(sha256_file(path), expected)
        finally:
            path.unlink(missing_ok=True)

    def test_hashes_known_content(self) -> None:
        content = b"hello hammer translator"
        with tempfile.NamedTemporaryFile(delete=False) as handle:
            handle.write(content)
            path = Path(handle.name)
        try:
            expected = hashlib.sha256(content).hexdigest()
            self.assertEqual(sha256_file(path), expected)
        finally:
            path.unlink(missing_ok=True)

    def test_hashes_large_file_in_chunks(self) -> None:
        content = b"x" * (2 * 1024 * 1024 + 100)
        with tempfile.NamedTemporaryFile(delete=False) as handle:
            handle.write(content)
            path = Path(handle.name)
        try:
            expected = hashlib.sha256(content).hexdigest()
            self.assertEqual(sha256_file(path), expected)
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
