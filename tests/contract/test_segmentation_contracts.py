from __future__ import annotations

import math

import numpy as np
import pytest
from pydantic import TypeAdapter, ValidationError
from shapely.geometry import Polygon

from petra.contracts import (
    FragmentGeom,
    MaskPrediction,
    ModelDescriptor,
    PromptSpec,
    vertex_count_quality_warnings,
)

ULID_FRAGMENT = "01KXY0JVMP1V3TZ9XDMXQQ6GMT"
ULID_SESSION = "01KXY0GVMP1V3TZ9XDMXQQ6GMR"


def regular_polygon(n_points: int = 100) -> list[list[float]]:
    points = [
        [
            100.0 + 50.0 * math.cos(2 * math.pi * index / n_points),
            100.0 + 50.0 * math.sin(2 * math.pi * index / n_points),
        ]
        for index in range(n_points)
    ]
    return [*points, points[0]]


def valid_fragment_data(n_points: int = 100) -> dict[str, object]:
    polygon = regular_polygon(n_points)
    shape = Polygon(polygon)
    return {
        "schema_version": "1.0.0",
        "fragment_id": ULID_FRAGMENT,
        "session_id": ULID_SESSION,
        "polygon_mm": polygon,
        "area_mm2": shape.area,
        "bbox_mm": list(shape.bounds),
        "n_points": n_points,
        "quality_warnings": list(vertex_count_quality_warnings(n_points)),
        "seg_model": "chroma",
        "seg_model_revision": "opencv-5.0.0",
        "seg_score": 1.0,
        "dp_epsilon_mm": 0.5,
        "photo_path": "rectified/session/image.png",
        "mask_path": "masks/session/fragment.png",
        "coordinate_frame": "bottom_left_x_right_y_up_mm",
    }


def chroma_descriptor() -> ModelDescriptor:
    return ModelDescriptor(
        name="chroma",
        family="chroma",
        revision="opencv-5.0.0",
        weights_sha256=None,
        license_spdx="Apache-2.0",
        license_approved=True,
        supported_devices=("cpu",),
        precision="uint8",
    )


@pytest.mark.contract
def test_fragment_geometry_is_closed_valid_ccw_and_consistent() -> None:
    fragment = FragmentGeom.model_validate(valid_fragment_data())
    assert fragment.n_points == 100
    assert fragment.quality_warnings == ()


@pytest.mark.contract
@pytest.mark.parametrize(
    ("n_points", "expected_warning"),
    [
        (3, ("VERTEX_COUNT_BELOW_EXPECTED",)),
        (99, ("VERTEX_COUNT_BELOW_EXPECTED",)),
        (100, ()),
        (1000, ()),
        (1001, ("VERTEX_COUNT_ABOVE_EXPECTED",)),
        (5000, ("VERTEX_COUNT_ABOVE_EXPECTED",)),
    ],
)
def test_fragment_geometry_accepts_vertex_boundaries_with_expected_warning(
    n_points: int, expected_warning: tuple[str, ...]
) -> None:
    fragment = FragmentGeom.model_validate(valid_fragment_data(n_points))
    assert fragment.n_points == n_points
    assert fragment.quality_warnings == expected_warning


@pytest.mark.contract
@pytest.mark.parametrize("n_points", [2, 5001])
def test_fragment_geometry_rejects_hard_vertex_limits(n_points: int) -> None:
    if n_points == 2:
        data = valid_fragment_data(3)
        data.update(
            polygon_mm=[[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
            area_mm2=1.0,
            bbox_mm=[0.0, 0.0, 1.0, 0.0],
            n_points=2,
        )
    else:
        data = valid_fragment_data(n_points)
    with pytest.raises(ValidationError):
        FragmentGeom.model_validate(data)


@pytest.mark.contract
def test_fragment_geometry_rejects_inconsistent_count_and_warning() -> None:
    wrong_count = valid_fragment_data(3)
    wrong_count["n_points"] = 4
    with pytest.raises(ValidationError, match="n_points"):
        FragmentGeom.model_validate(wrong_count)

    missing_warning = valid_fragment_data(3)
    missing_warning["quality_warnings"] = []
    with pytest.raises(ValidationError, match="quality_warnings"):
        FragmentGeom.model_validate(missing_warning)


@pytest.mark.contract
@pytest.mark.parametrize("mutation", ["open", "clockwise", "wrong_area", "unknown"])
def test_fragment_geometry_rejects_contract_violations(mutation: str) -> None:
    data = valid_fragment_data()
    if mutation == "open":
        data["polygon_mm"] = regular_polygon()[:-1]
    elif mutation == "clockwise":
        data["polygon_mm"] = list(reversed(regular_polygon()))
    elif mutation == "wrong_area":
        data["area_mm2"] = 1.0
    else:
        data["coordinate_system"] = "image"
    with pytest.raises(ValidationError):
        FragmentGeom.model_validate(data)


@pytest.mark.contract
@pytest.mark.parametrize(
    "prompt",
    [
        {"kind": "auto"},
        {"kind": "points", "points": [{"point": [10.0, 20.0], "label": 1}]},
        {"kind": "box", "box": [0.0, 0.0, 10.0, 20.0]},
        {"kind": "concept", "concept": "stone fragment"},
    ],
)
def test_prompt_spec_is_discriminated(prompt: dict[str, object]) -> None:
    parsed = TypeAdapter(PromptSpec).validate_python(prompt)
    assert parsed.kind == prompt["kind"]


@pytest.mark.contract
def test_prompt_and_model_descriptor_reject_ambiguous_values() -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(PromptSpec).validate_python({"kind": "box", "box": [1, 1, 0, 0]})
    with pytest.raises(ValidationError, match="weights_sha256"):
        ModelDescriptor(
            name="sam2",
            family="sam2",
            revision="v2.1",
            weights_sha256=None,
            license_spdx="Apache-2.0",
            license_approved=True,
            supported_devices=("cpu",),
            precision="float32",
        )
    descriptor_data = chroma_descriptor().model_dump()
    descriptor_data["supported_devices"] = ["cpu", "cpu"]
    with pytest.raises(ValidationError, match="unique"):
        ModelDescriptor.model_validate(descriptor_data)


@pytest.mark.contract
def test_mask_prediction_requires_binary_two_dimensional_mask() -> None:
    prediction = MaskPrediction(
        mask=np.zeros((8, 8), dtype=np.bool_),
        score=0.9,
        descriptor=chroma_descriptor(),
    )
    assert prediction.mask.shape == (8, 8)
    with pytest.raises(ValueError, match="bool"):
        MaskPrediction(
            mask=np.zeros((8, 8), dtype=np.uint8),
            score=0.9,
            descriptor=chroma_descriptor(),
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        MaskPrediction(
            mask=np.zeros((8, 8), dtype=np.bool_),
            score=math.nan,
            descriptor=chroma_descriptor(),
        )


@pytest.mark.contract
def test_fragment_geometry_rejects_duplicate_vertex_and_wrong_bbox() -> None:
    duplicate = valid_fragment_data()
    polygon = duplicate["polygon_mm"]
    assert isinstance(polygon, list)
    polygon[2] = polygon[1]
    with pytest.raises(ValidationError, match="duplicate"):
        FragmentGeom.model_validate(duplicate)

    wrong_bbox = valid_fragment_data()
    wrong_bbox["bbox_mm"] = [0.0, 0.0, 1.0, 1.0]
    with pytest.raises(ValidationError, match="bbox_mm"):
        FragmentGeom.model_validate(wrong_bbox)
