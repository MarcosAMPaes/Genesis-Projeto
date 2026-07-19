from __future__ import annotations

import numpy as np
import pytest

from petra.calibration.rectify import (
    CharucoBoardConfig,
    CharucoDetection,
    RectifyConfig,
    build_session_meta,
    detect_charuco,
    native_gsd_at,
    plan_rectification,
    rectify_image,
)
from petra.errors import ErrorCode, PetraError

pytestmark = pytest.mark.unit


def synthetic_detection(*, corrupt_last_mm: float = 0.0) -> CharucoDetection:
    x_values, y_values = np.meshgrid(np.linspace(20.0, 180.0, 5), np.linspace(20.0, 130.0, 4))
    points_mm = np.column_stack((x_values.ravel(), y_values.ravel()))
    mm_to_px = np.array(
        [[1.85, 0.12, 210.0], [-0.08, -1.92, 410.0], [0.00035, -0.00022, 1.0]],
        dtype=np.float64,
    )
    homogeneous = np.column_stack((points_mm, np.ones(len(points_mm)))) @ mm_to_px.T
    points_px = homogeneous[:, :2] / homogeneous[:, 2:3]
    points_mm[-1, 0] += corrupt_last_mm
    return CharucoDetection(
        points_px=points_px,
        points_mm=points_mm,
        marker_ids=(1, 2, 3, 4, 5, 6),
    )


def test_generated_charuco_is_detected_with_metric_bottom_left_coordinates() -> None:
    config = CharucoBoardConfig(
        squares_x=8,
        squares_y=6,
        square_length_mm=30.0,
        marker_length_mm=22.0,
        dictionary="DICT_5X5_100",
    )
    image = config.create_board().generateImage((1000, 750), marginSize=20)
    detection = detect_charuco(image, config)
    assert len(set(detection.marker_ids)) >= 4
    assert len(detection.points_px) >= 4
    assert np.all(detection.points_mm[:, 1] > 0)
    assert np.all(detection.points_mm[:, 1] < config.squares_y * config.square_length_mm)


def test_perspective_rectification_preserves_native_gsd_and_axis_orientation() -> None:
    detection = synthetic_detection()
    config = RectifyConfig(roi_mm=(0.0, 0.0, 200.0, 150.0))
    (
        pixel_to_mm,
        pixel_to_raster,
        native_gsd,
        output_gsd,
        ratio,
        residual,
        output_size,
    ) = plan_rectification(detection, config)
    assert output_gsd == pytest.approx(native_gsd)
    assert ratio == pytest.approx(1.0)
    assert residual < 1e-4
    assert output_size == (
        round(200.0 / native_gsd),
        round(150.0 / native_gsd),
    )
    centroid = detection.points_px.mean(axis=0)
    assert native_gsd == pytest.approx(
        native_gsd_at(pixel_to_mm, (float(centroid[0]), float(centroid[1])))
    )

    metric_test = np.array([[20.0, 20.0], [40.0, 20.0], [20.0, 40.0]])
    mm_to_px = np.linalg.inv(pixel_to_mm)
    homogeneous_px = np.column_stack((metric_test, np.ones(3))) @ mm_to_px.T
    pixels = homogeneous_px[:, :2] / homogeneous_px[:, 2:3]
    homogeneous_out = np.column_stack((pixels, np.ones(3))) @ pixel_to_raster.T
    output = homogeneous_out[:, :2] / homogeneous_out[:, 2:3]
    assert output[1, 0] > output[0, 0]
    assert output[2, 1] < output[0, 1]


def test_rectify_warps_to_physical_roi() -> None:
    detection = synthetic_detection()
    config = RectifyConfig(
        roi_mm=(0.0, 0.0, 200.0, 150.0),
        target_gsd_mm_px=0.52,
        max_resample_change_pct=10.0,
    )
    image = np.zeros((500, 700, 3), dtype=np.uint8)
    result = rectify_image(image, detection, config)
    assert result.image_bgr.shape[:2] == (
        result.rectified_img_size[1],
        result.rectified_img_size[0],
    )
    assert result.output_gsd_mm_px == 0.52
    session = build_session_meta(
        result,
        session_id="01KXY0GVMP1V3TZ9XDMXQQ6GMR",
        calib_profile_id="01KXY0FVMP1V3TZ9XDMXQQ6GMQ",
        source_image="raw.png",
        rectified_image="rectified.png",
        thickness_mm=20.0,
        background="verde-fosco",
        reference_plane_height_mm=20.0,
        parallax_factor=1.0,
        lidar_divergence_pct=0.3,
    )
    assert session.native_gsd_mm_px == result.native_gsd_mm_px
    assert session.output_gsd_mm_px == result.output_gsd_mm_px
    assert session.rectified_img_size == result.rectified_img_size


def test_rectification_rejects_occlusion_bad_ids_collinearity_residual_and_gsd() -> None:
    config = RectifyConfig(roi_mm=(0.0, 0.0, 200.0, 150.0))
    base = synthetic_detection()
    duplicate_ids = CharucoDetection(
        points_px=base.points_px,
        points_mm=base.points_mm,
        marker_ids=(1, 1, 2, 3),
    )
    with pytest.raises(PetraError) as ids:
        plan_rectification(duplicate_ids, config)
    assert ids.value.code == ErrorCode.ARUCO_INSUFFICIENT_MARKERS

    collinear = CharucoDetection(
        points_px=np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
        points_mm=np.array([[0.0, 0.0], [2.0, 2.0], [4.0, 4.0], [6.0, 6.0]]),
        marker_ids=(1, 2, 3, 4),
    )
    with pytest.raises(PetraError):
        plan_rectification(collinear, config)

    with pytest.raises(PetraError) as residual:
        plan_rectification(synthetic_detection(corrupt_last_mm=8.0), config)
    assert residual.value.code == ErrorCode.SESSION_RESIDUAL_REJECTED

    mismatched_gsd = RectifyConfig(
        roi_mm=(0.0, 0.0, 200.0, 150.0),
        target_gsd_mm_px=1.0,
    )
    with pytest.raises(PetraError) as gsd:
        plan_rectification(base, mismatched_gsd)
    assert gsd.value.code == ErrorCode.RECTIFY_GSD_MISMATCH

    board_config = CharucoBoardConfig(
        squares_x=8,
        squares_y=6,
        square_length_mm=30.0,
        marker_length_mm=22.0,
        dictionary="DICT_5X5_100",
    )
    blank = np.zeros((500, 500), dtype=np.uint8)
    with pytest.raises(PetraError) as occluded:
        detect_charuco(blank, board_config)
    assert occluded.value.code == ErrorCode.ARUCO_INSUFFICIENT_MARKERS
