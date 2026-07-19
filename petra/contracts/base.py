from __future__ import annotations

from datetime import datetime, timedelta
from typing import Annotated, Final

from pydantic import AfterValidator, BaseModel, ConfigDict, Field
from ulid import ULID

SCHEMA_VERSION: Final = "1.0.0"
SHA256_PATTERN = r"^[0-9a-f]{64}$"
ULID_PATTERN = r"^[0-7][0-9A-HJKMNP-TV-Z]{25}$"


def _validate_ulid(value: str) -> str:
    try:
        ULID.from_str(value)
    except ValueError as error:
        raise ValueError("invalid ULID") from error
    return value


def _validate_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("timestamp must use UTC")
    return value


type UlidString = Annotated[
    str,
    Field(pattern=ULID_PATTERN),
    AfterValidator(_validate_ulid),
]
type Sha256 = Annotated[str, Field(pattern=SHA256_PATTERN)]
type UtcDateTime = Annotated[datetime, AfterValidator(_validate_utc)]

type Point2D = tuple[float, float]
type BBox2D = tuple[float, float, float, float]
type Matrix3x3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]


class ContractModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        validate_assignment=True,
    )
