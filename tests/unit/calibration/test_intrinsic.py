from __future__ import annotations

import cv2
import numpy as np
import pytest

from petra.calibration.checkerboard import CheckerboardConfig, PoseObservation, object_points
from petra.calibration.intrinsic import calibrate_intrinsics
from petra.errors import ErrorCode, PetraError

pytestmark = pytest.mark.unit


def virtual_camera_observations(*, corrupt_index: int | None = None) -> list[PoseObservation]:
    rng = np.random.default_rng(20260719)
    config = CheckerboardConfig(columns=9, rows=6, square_size_mm=25.0)
    camera_matrix = np.array(
        [[1600.0, 0.0, 960.0], [0.0, 1580.0, 540.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    distortion = np.array([0.02, -0.04, 0.001, -0.0005, 0.01], dtype=np.float64)
    observations: list[PoseObservation] = []
    translations = [
        (-180.0, -100.0),
        (0.0, -110.0),
        (150.0, -90.0),
        (-170.0, 30.0),
        (0.0, 20.0),
        (140.0, 40.0),
        (-150.0, 130.0),
        (0.0, 120.0),
        (130.0, 120.0),
    ]
    for index in range(27):
        tx, ty = translations[index % len(translations)]
        rvec = rng.uniform([-0.28, -0.28, -0.15], [0.28, 0.28, 0.15])
        tvec = np.array([tx, ty, 820.0 + 35.0 * (index % 5)], dtype=np.float64)
        corners, _ = cv2.projectPoints(
            object_points(config), rvec, tvec, camera_matrix, distortion
        )
        corners = corners.reshape(-1, 2)
        corners += rng.normal(0.0, 0.04, size=corners.shape)
        if index == corrupt_index:
            corners += rng.normal(0.0, 8.0, size=corners.shape)
        observations.append(
            PoseObservation(
                source=f"pose-{index:02d}.png",
                image_sha256=f"{index:064x}",
                image_size=(1920, 1080),
                corners_px=corners,
            )
        )
    return observations


def test_virtual_camera_calibration_passes_ta1_synthetic() -> None:
    config = CheckerboardConfig(columns=9, rows=6, square_size_mm=25.0)
    result = calibrate_intrinsics(virtual_camera_observations(), config)
    assert result.accepted
    assert result.rms_px < 0.5
    assert len(result.poses) == 27
    assert result.dist.shape == (5,)
    assert result.K[0, 0] == pytest.approx(1600.0, rel=0.03)
    assert result.K[1, 1] == pytest.approx(1580.0, rel=0.03)


def test_corrupted_pose_is_reported_and_never_silently_removed() -> None:
    config = CheckerboardConfig(columns=9, rows=6, square_size_mm=25.0)
    observations = virtual_camera_observations(corrupt_index=7)
    result = calibrate_intrinsics(observations, config)
    assert len(result.poses) == len(observations)
    assert not result.accepted
    worst = max(result.poses, key=lambda pose: pose.rms_px)
    assert worst.source == "pose-07.png"
    assert worst.rms_px > 5.0

    explicitly_filtered = [pose for pose in observations if pose.source != worst.source]
    recalibrated = calibrate_intrinsics(explicitly_filtered, config)
    assert recalibrated.accepted


def test_calibration_rejects_too_few_or_mixed_resolution_poses() -> None:
    config = CheckerboardConfig(columns=9, rows=6, square_size_mm=25.0)
    observations = virtual_camera_observations()
    with pytest.raises(PetraError) as insufficient:
        calibrate_intrinsics(observations[:19], config)
    assert insufficient.value.code == ErrorCode.CALIB_INSUFFICIENT_POSES

    observations[-1] = PoseObservation(
        source=observations[-1].source,
        image_sha256=observations[-1].image_sha256,
        image_size=(1280, 720),
        corners_px=observations[-1].corners_px,
    )
    with pytest.raises(PetraError) as mismatch:
        calibrate_intrinsics(observations, config)
    assert mismatch.value.code == ErrorCode.IMAGE_SIZE_MISMATCH
