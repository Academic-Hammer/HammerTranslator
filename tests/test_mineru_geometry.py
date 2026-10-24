"""Tests for mineru_adapter.geometry."""

from __future__ import annotations

import unittest

from mineru_adapter.geometry import copy_bbox, intersection_area, union_bbox, valid_bbox


class ValidBboxTests(unittest.TestCase):
    def test_accepts_four_number_list(self) -> None:
        self.assertTrue(valid_bbox([0, 0, 100, 50]))

    def test_accepts_four_number_tuple(self) -> None:
        self.assertTrue(valid_bbox((10.0, 20.0, 30.0, 40.0)))

    def test_rejects_wrong_length(self) -> None:
        self.assertFalse(valid_bbox([0, 0, 100]))
        self.assertFalse(valid_bbox([0, 0, 100, 50, 200]))

    def test_rejects_inverted_coordinates(self) -> None:
        self.assertFalse(valid_bbox([100, 0, 0, 50]))
        self.assertFalse(valid_bbox([0, 50, 100, 0]))

    def test_rejects_non_numeric(self) -> None:
        self.assertFalse(valid_bbox(["a", "b", "c", "d"]))
        self.assertFalse(valid_bbox([0, 0, None, 50]))

    def test_rejects_string_and_bytes(self) -> None:
        self.assertFalse(valid_bbox("abcd"))
        self.assertFalse(valid_bbox(b"abcd"))

    def test_rejects_non_sequence(self) -> None:
        self.assertFalse(valid_bbox(42))
        self.assertFalse(valid_bbox(None))

    def test_accepts_zero_area_bbox(self) -> None:
        self.assertTrue(valid_bbox([10, 10, 10, 10]))


class CopyBboxTests(unittest.TestCase):
    def test_returns_new_list(self) -> None:
        original = [1, 2, 3, 4]
        copied = copy_bbox(original)
        self.assertEqual(copied, [1, 2, 3, 4])
        self.assertIsNot(copied, original)

    def test_returns_none_for_invalid(self) -> None:
        self.assertIsNone(copy_bbox("invalid"))
        self.assertIsNone(copy_bbox([1, 2, 3]))


class IntersectionAreaTests(unittest.TestCase):
    def test_full_overlap(self) -> None:
        self.assertEqual(intersection_area([0, 0, 10, 10], [0, 0, 10, 10]), 100.0)

    def test_partial_overlap(self) -> None:
        self.assertEqual(intersection_area([0, 0, 10, 10], [5, 5, 15, 15]), 25.0)

    def test_no_overlap(self) -> None:
        self.assertEqual(intersection_area([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)

    def test_invalid_bbox_returns_zero(self) -> None:
        self.assertEqual(intersection_area("bad", [0, 0, 10, 10]), 0.0)


class UnionBboxTests(unittest.TestCase):
    def test_single_box(self) -> None:
        self.assertEqual(union_bbox([[1, 2, 3, 4]]), [1, 2, 3, 4])

    def test_multiple_boxes(self) -> None:
        result = union_bbox([[0, 0, 10, 10], [5, 5, 20, 20]])
        self.assertEqual(result, [0, 0, 20, 20])

    def test_skips_invalid_boxes(self) -> None:
        result = union_bbox([[0, 0, 10, 10], "bad", [5, 5, 15, 15]])
        self.assertEqual(result, [0, 0, 15, 15])

    def test_empty_returns_none(self) -> None:
        self.assertIsNone(union_bbox([]))
        self.assertIsNone(union_bbox(["bad"]))


if __name__ == "__main__":
    unittest.main()
