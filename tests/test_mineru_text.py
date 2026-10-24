"""Tests for mineru_adapter.text."""

from __future__ import annotations

import unittest

from mineru_adapter.text import (
    extract_text,
    first_alpha,
    sentence_is_complete,
)


class ExtractTextTests(unittest.TestCase):
    def test_simple_text_block(self) -> None:
        block = {
            "lines": [
                {
                    "spans": [
                        {"type": "text", "content": "Hello world"},
                    ]
                }
            ]
        }
        self.assertEqual(extract_text(block), "Hello world")

    def test_joins_multiple_spans(self) -> None:
        block = {
            "lines": [
                {
                    "spans": [
                        {"type": "text", "content": "Hello"},
                        {"type": "text", "content": "world"},
                    ]
                }
            ]
        }
        result = extract_text(block)
        self.assertIn("Hello", result)
        self.assertIn("world", result)

    def test_inline_equation_wrapped_in_dollars(self) -> None:
        block = {
            "lines": [
                {
                    "spans": [
                        {"type": "inline_equation", "content": "x^2"},
                    ]
                }
            ]
        }
        self.assertEqual(extract_text(block), "$x^2$")

    def test_display_equation_wrapped_in_double_dollars(self) -> None:
        block = {
            "lines": [
                {
                    "spans": [
                        {"type": "interline_equation", "content": "E = mc^2"},
                    ]
                }
            ]
        }
        result = extract_text(block)
        self.assertIn("$$", result)
        self.assertIn("E = mc^2", result)

    def test_nested_blocks(self) -> None:
        block = {
            "blocks": [
                {
                    "lines": [
                        {"spans": [{"type": "text", "content": "inner text"}]}
                    ]
                }
            ]
        }
        self.assertEqual(extract_text(block), "inner text")

    def test_empty_block(self) -> None:
        self.assertEqual(extract_text({}), "")
        self.assertEqual(extract_text({"lines": []}), "")

    def test_hyphenated_line_break_joins_words(self) -> None:
        block = {
            "lines": [
                {"spans": [{"type": "text", "content": "hy-"}]},
                {"spans": [{"type": "text", "content": "phen"}]},
            ]
        }
        result = extract_text(block)
        self.assertEqual(result, "hyphen")


class FirstAlphaTests(unittest.TestCase):
    def test_returns_first_letter(self) -> None:
        self.assertEqual(first_alpha("123abc"), "a")

    def test_returns_empty_for_no_alpha(self) -> None:
        self.assertEqual(first_alpha("12345"), "")

    def test_strips_leading_whitespace(self) -> None:
        self.assertEqual(first_alpha("  hello"), "h")


class SentenceIsCompleteTests(unittest.TestCase):
    def test_english_period(self) -> None:
        self.assertTrue(sentence_is_complete("This is done."))

    def test_chinese_period(self) -> None:
        self.assertTrue(sentence_is_complete("这是结束。"))

    def test_question_mark(self) -> None:
        self.assertTrue(sentence_is_complete("Really?"))

    def test_incomplete_sentence(self) -> None:
        self.assertFalse(sentence_is_complete("still going"))

    def test_empty_string(self) -> None:
        self.assertTrue(sentence_is_complete(""))

    def test_trailing_whitespace_ignored(self) -> None:
        self.assertTrue(sentence_is_complete("Done.  "))


if __name__ == "__main__":
    unittest.main()
