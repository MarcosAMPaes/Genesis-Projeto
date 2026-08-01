from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from petra.calibration.board import CharucoBoardConfig

MIN_CORNERS_PER_POSE = 12
"""Minimum interior ChArUco corners a pose must expose to enter the calibration."""


@dataclass(frozen=True, slots=True)
class PoseObservation:
    """A single ChArUco view: pixel corners matched to their metric board points."""

    source: str
    image_sha256: str
    image_size: tuple[int, int]
    corners_px: NDArray[np.float64]
    object_points_mm: NDArray[np.float64]
    corner_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.corners_px.ndim != 2 or self.corners_px.shape[1] != 2:
            raise ValueError("corners_px must have shape (n, 2)")
        if self.object_points_mm.ndim != 2 or self.object_points_mm.shape[1] != 3:
            raise ValueError("object_points_mm must have shape (n, 3)")
        if len(self.corners_px) != len(self.object_points_mm):
            raise ValueError("pixel and metric correspondences must have equal length")
        if len(self.corner_ids) != len(self.corners_px):
            raise ValueError("corner_ids must match the number of correspondences")
        if len(self.corners_px) < MIN_CORNERS_PER_POSE:
            raise ValueError(
                f"a calibration pose needs at least {MIN_CORNERS_PER_POSE} ChArUco corners"
            )
        if len(set(self.corner_ids)) != len(self.corner_ids):
            raise ValueError("corner_ids must be unique within a pose")
        if not np.isfinite(self.corners_px).all() or not np.isfinite(self.object_points_mm).all():
            raise ValueError("correspondences must be finite")


def is_planar_non_collinear(object_points_mm: NDArray[np.float64]) -> bool:
    """Reject degenerate poses whose corners fall on a single board line."""
    planar = object_points_mm[:, :2]
    centred = planar - planar.mean(axis=0)
    return bool(np.linalg.matrix_rank(centred, tol=1e-6) >= 2)


def detect_charuco_pose(
    image_bgr: NDArray[np.uint8],
    config: CharucoBoardConfig,
    *,
    source: str,
    image_sha256: str,
) -> PoseObservation | None:
    """Detect one calibration pose; returns None when the view is unusable."""
    if image_bgr.ndim not in (2, 3):
        raise ValueError("image must be grayscale or BGR")
    gray = image_bgr if image_bgr.ndim == 2 else cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    board = config.create_board()
    detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, _marker_corners, _marker_ids = detector.detectBoard(gray)
    if charuco_corners is None or charuco_ids is None:
        return None
    if len(charuco_ids) < MIN_CORNERS_PER_POSE:
        return None

    object_points, image_points = board.matchImagePoints(charuco_corners, charuco_ids)
    if object_points is None or image_points is None:
        return None
    object_points_mm = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    corners_px = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    if len(object_points_mm) < MIN_CORNERS_PER_POSE:
        return None
    if not is_planar_non_collinear(object_points_mm):
        return None

    height, width = gray.shape
    return PoseObservation(
        source=source,
        image_sha256=image_sha256,
        image_size=(width, height),
        corners_px=corners_px,
        object_points_mm=object_points_mm,
        corner_ids=tuple(int(value) for value in np.asarray(charuco_ids).reshape(-1)),
    )


def board_object_points(config: CharucoBoardConfig) -> NDArray[np.float64]:
    """All interior chessboard corners of the board, in millimetres."""
    corners = config.create_board().getChessboardCorners()
    return cast(NDArray[np.float64], np.asarray(corners, dtype=np.float64).reshape(-1, 3))
