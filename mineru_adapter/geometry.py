"""Small, dependency-free geometry helpers used by the adapter."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from numbers import Real
from typing import TypeAlias

BBox: TypeAlias = list[float | int]


def valid_bbox(value: object) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        return False
    if not all(isinstance(number, Real) for number in value):
        return False
    x0, y0, x1, y1 = value
    return x0 <= x1 and y0 <= y1


def copy_bbox(value: object) -> BBox | None:
    return list(value) if valid_bbox(value) else None


def intersection_area(first: object, second: object) -> float:
    if not valid_bbox(first) or not valid_bbox(second):
        return 0.0
    first_box = list(first)  # type: ignore[arg-type]
    second_box = list(second)  # type: ignore[arg-type]
    width = min(first_box[2], second_box[2]) - max(first_box[0], second_box[0])
    height = min(first_box[3], second_box[3]) - max(first_box[1], second_box[1])
    return float(max(0, width) * max(0, height))


def union_bbox(values: Iterable[object]) -> BBox | None:
    boxes = [list(value) for value in values if valid_bbox(value)]
    if not boxes:
        return None
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]
