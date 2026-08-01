from __future__ import annotations

import cv2
import numpy as np
import pytest

from petra.calibration.board import CharucoBoardConfig
from petra.calibration.charuco import PoseObservation, board_object_points
from petra.calibration.intrinsic import calibrate_intrinsics
from petra.errors import ErrorCode, PetraError

pytestmark = pytest.mark.unit

CAMERA_MATRIX = np.array(
    [[1600.0, 0.0, 960.0], [0.0, 1580.0, 540.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
DISTORTION = np.array([0.02, -0.04, 0.001, -0.0005, 0.01], dtype=np.float64)


def board_config() -> CharucoBoardConfig:
    return CharucoBoardConfig(
        squares_x=7,
        squares_y=9,
        square_length_mm=38.0,
        marker_length_mm=28.0,
        dictionary="DICT_5X5_100",
    )


def virtual_camera_observations(
    *,
    corrupt_index: int | None = None,
    partial_indices: frozenset[int] = frozenset(),
) -> list[PoseObservation]:
    """Project the real board through a known camera, including partial views."""
    rng = np.random.default_rng(20260801)
    all_points = board_object_points(board_config())
    observations: list[PoseObservation] = []
    translations = [
        (-180.0, -140.0),
        (0.0, -150.0),
        (150.0, -130.0),
        (-170.0, 10.0),
        (0.0, 0.0),
        (140.0, 20.0),
        (-150.0, 150.0),
        (0.0, 160.0),
        (130.0, 150.0),
    ]
    for index in range(27):
        translation_x, translation_y = translations[index % len(translations)]
        rvec = rng.uniform([-0.28, -0.28, -0.15], [0.28, 0.28, 0.15])
        tvec = np.array(
            [translation_x, translation_y, 900.0 + 35.0 * (index % 5)],
            dtype=np.float64,
        )
        visible = np.arange(len(all_points))
        if index in partial_indices:
            visible = visible[: len(all_points) // 2]
        object_points = all_points[visible]
        corners, _ = cv2.projectPoints(object_points, rvec, tvec, CAMERA_MATRIX, DISTORTION)
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
                object_points_mm=object_points,
                corner_ids=tuple(int(value) for value in visible),
            )
        )
    return observations


def test_virtual_camera_calibration_passes_ta1_synthetic() -> None:
    result = calibrate_intrinsics(virtual_camera_observations())
    assert result.accepted
    assert result.rms_px < 0.5
    assert len(result.poses) == 27
    assert result.dist.shape == (5,)
    assert result.K[0, 0] == pytest.approx(1600.0, rel=0.03)
    assert result.K[1, 1] == pytest.approx(1580.0, rel=0.03)
    assert all(pose.corners_used == 48 for pose in result.poses)


def test_partial_board_views_are_accepted_and_reported() -> None:
    partial = frozenset({3, 11, 19})
    result = calibrate_intrinsics(virtual_camera_observations(partial_indices=partial))
    assert result.accepted
    used = {pose.source: pose.corners_used for pose in result.poses}
    assert used["pose-03.png"] == 24
    assert used["pose-04.png"] == 48
    assert result.K[0, 0] == pytest.approx(1600.0, rel=0.03)


def test_corrupted_pose_is_reported_and_never_silently_removed() -> None:
    observations = virtual_camera_observations(corrupt_index=7)
    result = calibrate_intrinsics(observations)
    assert len(result.poses) == len(observations)
    assert not result.accepted
    worst = max(result.poses, key=lambda pose: pose.rms_px)
    assert worst.source == "pose-07.png"
    assert worst.rms_px > 5.0

    explicitly_filtered = [pose for pose in observations if pose.source != worst.source]
    assert calibrate_intrinsics(explicitly_filtered).accepted


def test_calibration_rejects_too_few_or_mixed_resolution_poses() -> None:
    observations = virtual_camera_observations()
    with pytest.raises(PetraError) as insufficient:
        calibrate_intrinsics(observations[:19])
    assert insufficient.value.code == ErrorCode.CALIB_INSUFFICIENT_POSES

    last = observations[-1]
    observations[-1] = PoseObservation(
        source=last.source,
        image_sha256=last.image_sha256,
        image_size=(1280, 720),
        corners_px=last.corners_px,
        object_points_mm=last.object_points_mm,
        corner_ids=last.corner_ids,
    )
    with pytest.raises(PetraError) as mismatch:
        calibrate_intrinsics(observations)
    assert mismatch.value.code == ErrorCode.IMAGE_SIZE_MISMATCH
