from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from petra.contracts import FragmentGeom, SessionMeta
from petra.errors import ErrorCode, PetraError
from petra.segmentation.contour import epsilon_px, external_contour, simplify_contour
from petra.segmentation.geometry import (
    extract_fragment_geometry,
    persist_fragment_geom,
    repair_polygon,
)
from petra.segmentation.postprocess import ProcessedMask

pytestmark = pytest.mark.unit


def session_meta() -> SessionMeta:
    return SessionMeta.model_validate(
        {
            "session_id": "01KXY0GVMP1V3TZ9XDMXQQ6GMR",
            "calib_profile_id": "01KXY0FVMP1V3TZ9XDMXQQ6GMQ",
            "source_image": "raw.png",
            "rectified_image": "rectified.png",
            "undistorted": True,
            "scale_mm_px": 0.1,
            "homography": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "aruco_ids": [1, 2, 3, 4],
            "thickness_mm": 20.0,
            "background": "green",
            "residual_check_mm": 0.1,
            "native_gsd_mm_px": 0.1,
            "output_gsd_mm_px": 0.1,
            "resample_ratio": 1.0,
            "rectified_img_size": [2200, 2200],
            "roi_mm": [0.0, 0.0, 220.0, 220.0],
            "reference_plane_height_mm": 0.0,
            "parallax_factor": 0.975,
            "lidar_divergence_pct": 0.2,
            "coordinate_frame": "bottom_left_x_right_y_up_mm",
            "interpolator": "linear",
        }
    )


def serrated_mask(teeth: int = 180) -> np.ndarray:
    center = np.array([1100.0, 1100.0])
    points: list[list[int]] = []
    for index in range(teeth * 2):
        angle = 2 * math.pi * index / (teeth * 2)
        radius = 820.0 if index % 2 == 0 else 790.0
        point = center + radius * np.array([math.cos(angle), math.sin(angle)])
        points.append([round(float(point[0])), round(float(point[1]))])
    mask = np.zeros((2200, 2200), dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(points, dtype=np.int32)], 1)
    return mask.astype(np.bool_)


def triangle_mask() -> np.ndarray:
    mask = np.zeros((2200, 2200), dtype=np.uint8)
    cv2.fillConvexPoly(
        mask,
        np.asarray([[250, 1900], [1100, 250], [1950, 1900]], dtype=np.int32),
        1,
    )
    return mask.astype(np.bool_)


def processed_mask(mask: np.ndarray) -> ProcessedMask:
    metric_scale = 0.1 * 0.975
    return ProcessedMask(
        instance_index=0,
        mask=mask,
        area_mm2=float(np.count_nonzero(mask)) * metric_scale**2,
        source_components=1,
        filled_hole_pixels=0,
        morphology_kernel_px=0,
    )


def test_epsilon_regression_divides_by_metric_scale() -> None:
    assert epsilon_px(0.5, 0.1) == pytest.approx(5.0)
    assert epsilon_px(0.5, 0.08) == pytest.approx(6.25)
    assert epsilon_px(0.5, 2.0) == pytest.approx(0.25)
    assert epsilon_px(0.5, 0.1) != pytest.approx(0.05)


def test_extracts_closed_valid_ccw_metric_polygon_and_persists(tmp_path: Path) -> None:
    mask = serrated_mask()
    extraction = extract_fragment_geometry(
        processed_mask(mask),
        session_meta(),
        seg_model="chroma",
        seg_model_revision="opencv-5.0.0",
        seg_score=1.0,
        photo_path="rectified.png",
        mask_path="mask.png",
        fragment_id="01KXY0JVMP1V3TZ9XDMXQQ6GMT",
    )
    assert 100 <= extraction.simplified_points <= 1000
    assert extraction.raw_points > extraction.simplified_points * 20
    assert 0.99 <= extraction.area_ratio <= 1.01
    assert extraction.hausdorff_mm <= 0.5
    assert extraction.epsilon_px == pytest.approx(0.5 / (0.1 * 0.975))
    assert extraction.fragment.polygon_mm[0] == extraction.fragment.polygon_mm[-1]
    assert extraction.fragment.bbox_mm[0] == pytest.approx(0.0)
    assert extraction.fragment.bbox_mm[1] == pytest.approx(0.0)
    assert extraction.fragment.quality_warnings == ()

    output = tmp_path / "fragment_geom.json"
    persist_fragment_geom(extraction.fragment, output)
    loaded = FragmentGeom.model_validate_json(output.read_text(encoding="utf-8"))
    assert loaded == extraction.fragment


def test_contour_uses_external_chain_none_and_dp_reduces_more_than_95_percent() -> None:
    raw = external_contour(serrated_mask())
    simplified, used = simplify_contour(raw, dp_epsilon_mm=0.5, scale_mm_px=0.1 * 0.975)
    assert used == pytest.approx(0.5 / (0.1 * 0.975))
    assert len(simplified) < len(raw) * 0.05


def test_triangle_is_accepted_with_low_vertex_warning() -> None:
    extraction = extract_fragment_geometry(
        processed_mask(triangle_mask()),
        session_meta(),
        seg_model="chroma",
        seg_model_revision="opencv",
        seg_score=1.0,
        photo_path="photo.png",
        mask_path="mask.png",
    )
    assert extraction.simplified_points == 3
    assert extraction.fragment.quality_warnings == ("VERTEX_COUNT_BELOW_EXPECTED",)
    assert 0.99 <= extraction.area_ratio <= 1.01


@pytest.mark.parametrize("n_points", [2, 5001])
def test_runtime_rejects_hard_vertex_limits(
    monkeypatch: pytest.MonkeyPatch, n_points: int
) -> None:
    angles = np.linspace(0.0, 2.0 * math.pi, n_points, endpoint=False)
    simplified = np.column_stack((1100 + 800 * np.cos(angles), 1100 + 800 * np.sin(angles)))

    def fake_simplify(
        _contour: np.ndarray, *, dp_epsilon_mm: float, scale_mm_px: float
    ) -> tuple[np.ndarray, float]:
        return simplified, dp_epsilon_mm / scale_mm_px

    monkeypatch.setattr("petra.segmentation.geometry.simplify_contour", fake_simplify)
    with pytest.raises(PetraError) as rejected:
        extract_fragment_geometry(
            processed_mask(triangle_mask()),
            session_meta(),
            seg_model="chroma",
            seg_model_revision="opencv",
            seg_score=1.0,
            photo_path="photo.png",
            mask_path="mask.png",
        )
    assert rejected.value.code == ErrorCode.POLYGON_VERTEX_COUNT
    assert rejected.value.details == {"n_points": n_points, "minimum": 3, "maximum": 5000}


def test_contour_input_guards_and_polygon_repair_paths() -> None:
    with pytest.raises(ValueError, match="positive"):
        epsilon_px(0.5, 0.0)
    with pytest.raises(ValueError, match="two-dimensional"):
        external_contour(np.zeros((5, 5, 1), dtype=np.bool_))
    with pytest.raises(ValueError, match="no external"):
        external_contour(np.zeros((5, 5), dtype=np.bool_))
    with pytest.raises(ValueError, match="at least three"):
        simplify_contour(
            np.zeros((2, 2), dtype=np.float64),
            dp_epsilon_mm=0.5,
            scale_mm_px=0.1,
        )

    repaired, changed = repair_polygon(
        np.array([[0.0, 0.0], [2.0, 2.0], [0.0, 2.0], [2.0, 0.0]])
    )
    assert changed
    assert repaired.is_valid

    two_squares_touching_at_a_point = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [3.0, 1.0],
            [2.0, 1.0],
            [2.0, 0.0],
            [0.0, 0.0],
        ]
    )
    with pytest.raises(PetraError) as multipolygon:
        repair_polygon(two_squares_touching_at_a_point)
    assert multipolygon.value.code == ErrorCode.POLYGON_MULTIPOLYGON


def test_area_ratio_gate_uses_mask_area_in_metric_units() -> None:
    mask = serrated_mask()
    wrong_area = processed_mask(mask)
    wrong_area = ProcessedMask(
        instance_index=wrong_area.instance_index,
        mask=wrong_area.mask,
        area_mm2=wrong_area.area_mm2 * 2.0,
        source_components=1,
        filled_hole_pixels=0,
        morphology_kernel_px=0,
    )
    with pytest.raises(PetraError) as rejected:
        extract_fragment_geometry(
            wrong_area,
            session_meta(),
            seg_model="chroma",
            seg_model_revision="opencv",
            seg_score=1.0,
            photo_path="photo.png",
            mask_path="mask.png",
        )
    assert rejected.value.code == ErrorCode.POLYGON_AREA_RATIO
