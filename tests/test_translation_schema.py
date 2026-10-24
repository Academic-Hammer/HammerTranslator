"""Tests for hammer_translator.translation.schema."""

from __future__ import annotations

import math
import unittest

from pydantic import ValidationError

from hammer_translator.translation.schema import (
    INPUT_SCHEMA_VERSION,
    InputBlock,
    input_block_json_schema,
)


def _valid_block(**overrides: object) -> dict:
    base = {
        "type": "text",
        "text": "Hello world",
        "page_idx": 0,
        "bbox": [0.0, 0.0, 100.0, 20.0],
        "logical_para_id": 0,
    }
    base.update(overrides)
    return base


class InputBlockTests(unittest.TestCase):
    def test_accepts_valid_block(self) -> None:
        block = InputBlock(**_valid_block())
        self.assertEqual(block.type, "text")
        self.assertEqual(block.page_idx, 0)

    def test_strips_type_whitespace(self) -> None:
        block = InputBlock(**_valid_block(type="  text  "))
        self.assertEqual(block.type, "text")

    def test_rejects_blank_type(self) -> None:
        with self.assertRaises(ValidationError):
            InputBlock(**_valid_block(type="   "))

    def test_rejects_empty_type(self) -> None:
        with self.assertRaises(ValidationError):
            InputBlock(**_valid_block(type=""))

    def test_rejects_negative_page_idx(self) -> None:
        with self.assertRaises(ValidationError):
            InputBlock(**_valid_block(page_idx=-1))

    def test_rejects_inverted_bbox(self) -> None:
        with self.assertRaises(ValidationError):
            InputBlock(**_valid_block(bbox=[100.0, 0.0, 0.0, 20.0]))

    def test_rejects_zero_width_bbox(self) -> None:
        with self.assertRaises(ValidationError):
            InputBlock(**_valid_block(bbox=[10.0, 0.0, 10.0, 20.0]))

    def test_rejects_non_finite_bbox(self) -> None:
        with self.assertRaises(ValidationError):
            InputBlock(**_valid_block(bbox=[0.0, 0.0, float("inf"), 20.0]))

    def test_rejects_nan_bbox(self) -> None:
        with self.assertRaises(ValidationError):
            InputBlock(**_valid_block(bbox=[0.0, 0.0, float("nan"), 20.0]))

    def test_accepts_valid_block_id(self) -> None:
        block = InputBlock(**_valid_block(block_id="para-1.2"))
        self.assertEqual(block.block_id, "para-1.2")

    def test_rejects_invalid_block_id(self) -> None:
        with self.assertRaises(ValidationError):
            InputBlock(**_valid_block(block_id=".starts-with-dot"))

    def test_rejects_block_id_with_spaces(self) -> None:
        with self.assertRaises(ValidationError):
            InputBlock(**_valid_block(block_id="has space"))

    def test_none_block_id_is_allowed(self) -> None:
        block = InputBlock(**_valid_block(block_id=None))
        self.assertIsNone(block.block_id)

    def test_extra_fields_are_preserved(self) -> None:
        block = InputBlock(**_valid_block(custom_field="value"))
        self.assertEqual(block.model_extra["custom_field"], "value")


class JsonSchemaTests(unittest.TestCase):
    def test_schema_has_version_id(self) -> None:
        schema = input_block_json_schema()
        self.assertEqual(schema["$id"], INPUT_SCHEMA_VERSION)

    def test_schema_version_constant(self) -> None:
        self.assertEqual(INPUT_SCHEMA_VERSION, "translation-input-v1")


if __name__ == "__main__":
    unittest.main()
