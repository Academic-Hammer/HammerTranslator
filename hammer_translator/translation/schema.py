"""Versioned input models for the Translation Stage."""

from __future__ import annotations

import math
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, field_validator


INPUT_SCHEMA_VERSION = "translation-input-v1"
BLOCK_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")


class InputBlock(BaseModel):
    """Required block fields plus untouched MinerUAdapter extensions."""

    model_config = ConfigDict(extra="allow")

    type: StrictStr = Field(min_length=1)
    text: StrictStr
    page_idx: StrictInt = Field(ge=0)
    bbox: tuple[float, float, float, float]
    logical_para_id: StrictInt = Field(ge=0)
    block_id: StrictStr | None = None

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("type cannot be blank")
        return normalized

    @field_validator("bbox")
    @classmethod
    def validate_bbox(
        cls, value: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        if not all(math.isfinite(coordinate) for coordinate in value):
            raise ValueError("bbox coordinates must be finite")
        x0, y0, x1, y1 = value
        if x1 <= x0 or y1 <= y0:
            raise ValueError("bbox must satisfy x0 < x1 and y0 < y1")
        return value

    @field_validator("block_id")
    @classmethod
    def validate_block_id(cls, value: str | None) -> str | None:
        if value is not None and not BLOCK_ID_RE.fullmatch(value):
            raise ValueError(
                "block_id must begin with an alphanumeric character and contain only "
                "letters, digits, '.', '_', ':' or '-'"
            )
        return value


def input_block_json_schema() -> dict[str, Any]:
    """Expose a JSON-Schema representation for contract tooling."""

    schema = InputBlock.model_json_schema()
    schema["$id"] = INPUT_SCHEMA_VERSION
    return schema
