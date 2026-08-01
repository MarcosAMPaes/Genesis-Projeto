from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from petra.calibration.charuco import MIN_CORNERS_PER_POSE, PoseObservation
from petra.errors import ErrorCode, PetraError

MIN_POSES = 20
MAX_ACCEPTED_RMS_PX = 0.5


@dataclass(frozen=True, slots=True)
class PoseCalibrationResult:
    source: str
    image_sha256: str
    rms_px: float
    corners_used: int
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
        return len(self.poses) >= MIN_POSES and self.rms_px < MAX_ACCEPTED_RMS_PX


def _vector3(vector: Any) -> tuple[float, float, float]:
    flat = np.asarray(vector, dtype=np.float64).reshape(3)
    return float(flat[0]), float(flat[1]), float(flat[2])


def calibrate_intrinsics(observations: list[PoseObservation]) -> IntrinsicCalibrationResult:
    """Zhang intrinsic calibration over ChArUco poses, tolerating partial views."""
    if len(observations) < MIN_POSES:
        raise PetraError(
            ErrorCode.CALIB_INSUFFICIENT_POSES,
            f"at least {MIN_POSES} valid ChArUco poses are required",
            {"valid_poses": len(observations), "required": MIN_POSES},
        )

    image_sizes = {observation.image_size for observation in observations}
    if len(image_sizes) != 1:
        raise PetraError(
            ErrorCode.IMAGE_SIZE_MISMATCH,
            "all calibration images must use the same resolution",
            {"image_sizes": sorted(image_sizes)},
        )
    image_size = observations[0].image_size
    for observation in observations:
        if len(observation.corners_px) < MIN_CORNERS_PER_POSE:
            raise ValueError(
                f"{observation.source} exposes {len(observation.corners_px)} corners; "
                f"at least {MIN_CORNERS_PER_POSE} are required"
            )

    object_point_sets = [
        observation.object_points_mm.astype(np.float32).reshape(-1, 1, 3)
        for observation in observations
    ]
    image_point_sets = [
        observation.corners_px.astype(np.float32).reshape(-1, 1, 2) for observation in observations
    ]
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
            observation.object_points_mm.astype(np.float32),
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
                corners_used=len(observation.corners_px),
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
