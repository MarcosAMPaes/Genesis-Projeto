from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from petra.calibration.checkerboard import CheckerboardConfig, PoseObservation, object_points
from petra.errors import ErrorCode, PetraError


@dataclass(frozen=True, slots=True)
class PoseCalibrationResult:
    source: str
    image_sha256: str
    rms_px: float
    rvec: tuple[float, float, float]
    tvec: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class IntrinsicCalibrationResult:
    K: NDArray[np.float64]
    dist: NDArray[np.float64]
    rms_px: float
    image_size: tuple[int, int]
    poses: tuple[PoseCalibrationResult, ...]

    @property
    def accepted(self) -> bool:
        return len(self.poses) >= 20 and self.rms_px < 0.5


def _vector3(vector: Any) -> tuple[float, float, float]:
    flat = np.asarray(vector, dtype=np.float64).reshape(3)
    return float(flat[0]), float(flat[1]), float(flat[2])


def calibrate_intrinsics(
    observations: list[PoseObservation],
    config: CheckerboardConfig,
) -> IntrinsicCalibrationResult:
    if len(observations) < 20:
        raise PetraError(
            ErrorCode.CALIB_INSUFFICIENT_POSES,
            "at least 20 valid checkerboard poses are required",
            {"valid_poses": len(observations), "required": 20},
        )

    image_sizes = {observation.image_size for observation in observations}
    if len(image_sizes) != 1:
        raise PetraError(
            ErrorCode.IMAGE_SIZE_MISMATCH,
            "all calibration images must use the same resolution",
            {"image_sizes": sorted(image_sizes)},
        )
    image_size = observations[0].image_size
    expected_points = config.rows * config.columns
    for observation in observations:
        if observation.corners_px.shape != (expected_points, 2):
            raise ValueError(
                f"{observation.source} has {len(observation.corners_px)} corners; "
                f"expected {expected_points}"
            )

    board_points = object_points(config).astype(np.float32)
    object_point_sets = [board_points.copy() for _ in observations]
    image_point_sets = [observation.corners_px.astype(np.float32) for observation in observations]
    flags = cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6
    rms, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        object_point_sets,
        image_point_sets,
        image_size,
        None,
        None,
        flags=flags,
    )
    camera_matrix = cast(NDArray[np.float64], np.asarray(camera_matrix, dtype=np.float64))
    distortion = np.asarray(distortion, dtype=np.float64).reshape(-1)[:5]

    pose_results: list[PoseCalibrationResult] = []
    for observation, rvec, tvec in zip(observations, rvecs, tvecs, strict=True):
        projected, _ = cv2.projectPoints(
            board_points,
            rvec,
            tvec,
            camera_matrix,
            distortion,
        )
        difference = projected.reshape(-1, 2) - observation.corners_px
        pose_rms = float(np.sqrt(np.mean(np.sum(np.square(difference), axis=1))))
        pose_results.append(
            PoseCalibrationResult(
                source=observation.source,
                image_sha256=observation.image_sha256,
                rms_px=pose_rms,
                rvec=_vector3(rvec),
                tvec=_vector3(tvec),
            )
        )

    return IntrinsicCalibrationResult(
        K=camera_matrix,
        dist=distortion,
        rms_px=float(rms),
        image_size=image_size,
        poses=tuple(pose_results),
    )
