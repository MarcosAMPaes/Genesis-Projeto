from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Annotated, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator
from shapely.geometry import LinearRing, Polygon

from petra.contracts.base import (
    SCHEMA_VERSION,
    BBox2D,
    ContractModel,
    Point2D,
    Sha256,
    UlidString,
)

PositiveFloat = Annotated[float, Field(gt=0)]
Probability = Annotated[float, Field(ge=0, le=1)]

MIN_POLYGON_VERTICES = 3
EXPECTED_MIN_POLYGON_VERTICES = 100
EXPECTED_MAX_POLYGON_VERTICES = 1000
MAX_POLYGON_VERTICES = 5000

type GeometryQualityWarning = Literal[
    "VERTEX_COUNT_BELOW_EXPECTED",
    "VERTEX_COUNT_ABOVE_EXPECTED",
]


def vertex_count_quality_warnings(n_points: int) -> tuple[GeometryQualityWarning, ...]:
    if n_points < EXPECTED_MIN_POLYGON_VERTICES:
        return ("VERTEX_COUNT_BELOW_EXPECTED",)
    if n_points > EXPECTED_MAX_POLYGON_VERTICES:
        return ("VERTEX_COUNT_ABOVE_EXPECTED",)
    return ()


class ModelDescriptor(ContractModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    name: str = Field(min_length=1)
    family: Literal["chroma", "birefnet", "sam2", "sam3"]
    revision: str = Field(min_length=1)
    weights_sha256: Sha256 | None = None
    license_spdx: str = Field(min_length=1)
    license_approved: bool
    supported_devices: tuple[Literal["cpu", "mps", "cuda"], ...] = Field(min_length=1)
    precision: Literal["uint8", "float32", "float16", "bfloat16"]

    @model_validator(mode="after")
    def validate_descriptor(self) -> ModelDescriptor:
        if len(set(self.supported_devices)) != len(self.supported_devices):
            raise ValueError("supported_devices must be unique")
        if self.family != "chroma" and self.weights_sha256 is None:
            raise ValueError("learned models require weights_sha256")
        return self


class AutoPrompt(ContractModel):
    kind: Literal["auto"] = "auto"


class PromptPoint(ContractModel):
    point: Point2D
    label: Literal[0, 1]


class PointsPrompt(ContractModel):
    kind: Literal["points"] = "points"
    points: tuple[PromptPoint, ...] = Field(min_length=1)


class BoxPrompt(ContractModel):
    kind: Literal["box"] = "box"
    box: BBox2D

    @model_validator(mode="after")
    def validate_box(self) -> BoxPrompt:
        x_min, y_min, x_max, y_max = self.box
        if x_max <= x_min or y_max <= y_min:
            raise ValueError("box maximums must exceed minimums")
        return self


class ConceptPrompt(ContractModel):
    kind: Literal["concept"] = "concept"
    concept: str = Field(min_length=1)


type PromptSpec = Annotated[
    AutoPrompt | PointsPrompt | BoxPrompt | ConceptPrompt,
    Field(discriminator="kind"),
]


@dataclass(frozen=True, slots=True)
class MaskPrediction:
    mask: NDArray[np.bool_]
    score: float
    descriptor: ModelDescriptor

    def __post_init__(self) -> None:
        if self.mask.ndim != 2 or self.mask.dtype != np.bool_:
            raise ValueError("mask must be a two-dimensional bool array")
        if not math.isfinite(self.score) or not 0 <= self.score <= 1:
            raise ValueError("score must be finite and in [0, 1]")


class FragmentGeom(ContractModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    fragment_id: UlidString
    session_id: UlidString
    polygon_mm: tuple[Point2D, ...] = Field(
        min_length=MIN_POLYGON_VERTICES + 1,
        max_length=MAX_POLYGON_VERTICES + 1,
    )
    area_mm2: PositiveFloat
    bbox_mm: BBox2D
    n_points: Annotated[
        int,
        Field(ge=MIN_POLYGON_VERTICES, le=MAX_POLYGON_VERTICES),
    ]
    quality_warnings: tuple[GeometryQualityWarning, ...] = ()
    seg_model: str = Field(min_length=1)
    seg_model_revision: str = Field(min_length=1)
    seg_score: Probability
    dp_epsilon_mm: PositiveFloat
    photo_path: str = Field(min_length=1)
    mask_path: str = Field(min_length=1)
    coordinate_frame: Literal["bottom_left_x_right_y_up_mm"]

    @model_validator(mode="after")
    def validate_geometry(self) -> FragmentGeom:
        if self.polygon_mm[0] != self.polygon_mm[-1]:
            raise ValueError("polygon_mm must be closed")
        if self.n_points != len(self.polygon_mm) - 1:
            raise ValueError("n_points must count vertices without the closing duplicate")
        if len(set(self.polygon_mm[:-1])) != self.n_points:
            raise ValueError("polygon_mm cannot contain duplicate vertices")
        expected_warnings = vertex_count_quality_warnings(self.n_points)
        if self.quality_warnings != expected_warnings:
            raise ValueError(
                "quality_warnings must match the polygon vertex count; "
                f"expected {expected_warnings}"
            )

        ring = LinearRing(self.polygon_mm)
        polygon = Polygon(ring)
        if not ring.is_simple or not polygon.is_valid:
            raise ValueError("polygon_mm must be simple and valid")
        if not ring.is_ccw:
            raise ValueError("polygon_mm must use counter-clockwise orientation")
        if not math.isclose(self.area_mm2, polygon.area, rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError("area_mm2 is inconsistent with polygon_mm")
        if any(
            not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-9)
            for actual, expected in zip(self.bbox_mm, polygon.bounds, strict=True)
        ):
            raise ValueError("bbox_mm is inconsistent with polygon_mm")
        return self
