from __future__ import annotations

import cv2
import numpy as np
import pytest

from petra.calibration.undistort import Undistorter
from petra.contracts import CalibProfile
from petra.errors import ErrorCode, PetraError

pytestmark = pytest.mark.unit


def profile_data() -> dict[str, object]:
    images = [
        {"path": f"pose-{index}.png", "sha256": f"{index:064x}"} for index in range(20)
    ]
    residuals = [
        {
            "image_sha256": f"{index:064x}",
            "rms_px": 0.1,
            "rvec": [0.0, 0.0, 0.0],
            "tvec": [0.0, 0.0, 800.0],
        }
        for index in range(20)
    ]
    return {
        "id": "01KXY0FVMP1V3TZ9XDMXQQ6GMQ",
        "content_sha256": "f" * 64,
        "device": "device",
        "lens": "lens",
        "K": [[450.0, 0.0, 320.0], [0.0, 455.0, 240.0], [0.0, 0.0, 1.0]],
        "dist": [0.12, -0.08, 0.002, -0.001, 0.02],
        "rms_px": 0.2,
        "img_size": [640, 480],
        "z_mm_lidar": 800.0,
        "created_at": "2026-07-19T15:00:00Z",
        "bench_config_hash": "e" * 64,
        "included_images": images,
        "pose_residuals": residuals,
    }


def test_maps_are_fixed_point_cached_and_capture_is_traceable() -> None:
    profile = CalibProfile.model_validate(profile_data())
    undistorter = Undistorter()
    first = undistorter.maps(profile, (640, 480))
    second = undistorter.maps(profile, (640, 480))
    assert first is second
    assert first.map1.dtype == np.int16
    assert first.map1.shape == (480, 640, 2)
    assert first.map2.dtype == np.uint16

    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.line(image, (0, 240), (639, 240), (255, 255, 255), 2)
    frame = undistorter.apply(image, profile)
    assert frame.image_bgr.shape == image.shape
    assert frame.calib_profile_id == profile.id
    assert frame.undistorted is True


def test_undistort_rejects_resolution_mismatch() -> None:
    profile = CalibProfile.model_validate(profile_data())
    with pytest.raises(PetraError) as mismatch:
        Undistorter().apply(np.zeros((720, 1280, 3), dtype=np.uint8), profile)
    assert mismatch.value.code == ErrorCode.IMAGE_SIZE_MISMATCH


def test_synthetic_distorted_grid_has_subpixel_residual_curvature() -> None:
    profile = CalibProfile.model_validate(profile_data())
    camera_matrix = np.asarray(profile.K, dtype=np.float64)
    distortion = np.asarray(profile.dist, dtype=np.float64)
    ideal_lines: list[np.ndarray] = []
    distorted_lines: list[np.ndarray] = []
    for y_px in (100.0, 170.0, 240.0, 310.0, 380.0):
        x_px = np.linspace(80.0, 560.0, 100)
        ideal = np.column_stack((x_px, np.full_like(x_px, y_px)))
        normalized = np.column_stack(
            (
                (ideal[:, 0] - camera_matrix[0, 2]) / camera_matrix[0, 0],
                (ideal[:, 1] - camera_matrix[1, 2]) / camera_matrix[1, 1],
                np.ones(len(ideal)),
            )
        )
        distorted, _ = cv2.projectPoints(
            normalized,
            np.zeros(3),
            np.zeros(3),
            camera_matrix,
            distortion,
        )
        ideal_lines.append(ideal)
        distorted_lines.append(distorted.reshape(-1, 2))

    undistorter = Undistorter()
    for ideal, distorted in zip(ideal_lines, distorted_lines, strict=True):
        corrected = undistorter.undistort_points(distorted, profile)
        np.testing.assert_allclose(corrected, ideal, atol=0.05)
        assert float(np.ptp(corrected[:, 1])) < 0.1
