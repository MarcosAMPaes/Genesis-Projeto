from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, cast

import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from petra.contracts import SessionMeta
from petra.errors import ErrorCode, PetraError


class CharucoBoardConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    squares_x: Annotated[int, Field(ge=3)]
    squares_y: Annotated[int, Field(ge=3)]
    square_length_mm: Annotated[float, Field(gt=0)]
    marker_length_mm: Annotated[float, Field(gt=0)]
    dictionary: str = Field(pattern=r"^DICT_[A-Z0-9_]+$")

    @model_validator(mode="after")
    def validate_lengths(self) -> CharucoBoardConfig:
        if self.marker_length_mm >= self.square_length_mm:
            raise ValueError("marker_length_mm must be smaller than square_length_mm")
        return self

    @classmethod
    def from_json(cls, path: Path) -> CharucoBoardConfig:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def create_board(self) -> cv2.aruco.CharucoBoard:
        dictionary_id = getattr(cv2.aruco, self.dictionary, None)
        if not isinstance(dictionary_id, int):
            raise ValueError(f"unknown ArUco dictionary: {self.dictionary}")
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        return cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y),
            self.square_length_mm,
            self.marker_length_mm,
            dictionary,
        )


class RectifyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    gsd_policy: Literal["preserve_native"] = "preserve_native"
    target_gsd_mm_px: Annotated[float, Field(gt=0)] | None = None
    max_resample_change_pct: Annotated[float, Field(gt=0)] = 10.0
    roi_mm: tuple[float, float, Annotated[float, Field(gt=0)], Annotated[float, Field(gt=0)]]
    interpolator: Literal["nearest", "linear", "cubic", "lanczos4"] = "linear"

    @classmethod
    def from_json(cls, path: Path) -> RectifyConfig:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


@dataclass(frozen=True, slots=True)
class CharucoDetection:
    points_px: NDArray[np.float64]
    points_mm: NDArray[np.float64]
    marker_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.points_px.shape != self.points_mm.shape or self.points_px.ndim != 2:
            raise ValueError("pixel and metric correspondences must have equal (n, 2) shape")
        if self.points_px.shape[1] != 2 or len(self.points_px) < 4:
            raise ValueError("at least four 2D correspondences are required")
        if not np.isfinite(self.points_px).all() or not np.isfinite(self.points_mm).all():
            raise ValueError("correspondences must be finite")


@dataclass(frozen=True, slots=True)
class RectificationResult:
    image_bgr: NDArray[np.uint8]
    pixel_to_mm: NDArray[np.float64]
    pixel_to_raster: NDArray[np.float64]
    native_gsd_mm_px: float
    output_gsd_mm_px: float
    resample_ratio: float
    residual_check_mm: float
    rectified_img_size: tuple[int, int]
    marker_ids: tuple[int, ...]
    interpolator: str


def detect_charuco(image_bgr: NDArray[np.uint8], config: CharucoBoardConfig) -> CharucoDetection:
    board = config.create_board()
    detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, _marker_corners, marker_ids = detector.detectBoard(image_bgr)
    unique_marker_ids = (
        tuple(int(value) for value in np.asarray(marker_ids).reshape(-1))
        if marker_ids is not None
        else ()
    )
    if len(set(unique_marker_ids)) < 4:
        raise PetraError(
            ErrorCode.ARUCO_INSUFFICIENT_MARKERS,
            "at least four unique ChArUco markers must be visible",
            {"marker_ids": unique_marker_ids},
        )
    if charuco_corners is None or charuco_ids is None or len(charuco_ids) < 4:
        raise PetraError(
            ErrorCode.ARUCO_INSUFFICIENT_MARKERS,
            "at least four ChArUco corner correspondences are required",
        )
    ids = np.asarray(charuco_ids, dtype=np.int64).reshape(-1)
    board_corners = np.asarray(board.getChessboardCorners(), dtype=np.float64)
    if ids.min() < 0 or ids.max() >= len(board_corners):
        raise ValueError("ChArUco detector returned an out-of-range corner ID")
    points_mm = board_corners[ids, :2]
    board_height = config.squares_y * config.square_length_mm
    points_mm[:, 1] = board_height - points_mm[:, 1]
    return CharucoDetection(
        points_px=np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2),
        points_mm=points_mm,
        marker_ids=unique_marker_ids,
    )


def _project(points: NDArray[np.float64], homography: NDArray[np.float64]) -> NDArray[np.float64]:
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    transformed = homogeneous @ homography.T
    return transformed[:, :2] / transformed[:, 2:3]


def homography_jacobian(
    pixel_to_mm: NDArray[np.float64], point_px: tuple[float, float]
) -> NDArray[np.float64]:
    u, v = point_px
    h = pixel_to_mm
    denominator = h[2, 0] * u + h[2, 1] * v + h[2, 2]
    if math.isclose(float(denominator), 0.0, abs_tol=1e-12):
        raise ValueError("homography is singular at the ChArUco centroid")
    x_numerator = h[0, 0] * u + h[0, 1] * v + h[0, 2]
    y_numerator = h[1, 0] * u + h[1, 1] * v + h[1, 2]
    denominator_squared = denominator * denominator
    return np.array(
        [
            [
                (h[0, 0] * denominator - h[2, 0] * x_numerator) / denominator_squared,
                (h[0, 1] * denominator - h[2, 1] * x_numerator) / denominator_squared,
            ],
            [
                (h[1, 0] * denominator - h[2, 0] * y_numerator) / denominator_squared,
                (h[1, 1] * denominator - h[2, 1] * y_numerator) / denominator_squared,
            ],
        ],
        dtype=np.float64,
    )


def native_gsd_at(pixel_to_mm: NDArray[np.float64], point_px: tuple[float, float]) -> float:
    singular_values = np.linalg.svd(homography_jacobian(pixel_to_mm, point_px), compute_uv=False)
    return float(math.sqrt(float(singular_values[0] * singular_values[1])))


def plan_rectification(
    detection: CharucoDetection, config: RectifyConfig
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, float, float, float, tuple[int, int]]:
    if len(detection.marker_ids) < 4 or len(set(detection.marker_ids)) != len(detection.marker_ids):
        raise PetraError(
            ErrorCode.ARUCO_INSUFFICIENT_MARKERS,
            "at least four unique marker IDs are required",
            {"marker_ids": detection.marker_ids},
        )
    if np.linalg.matrix_rank(detection.points_px - detection.points_px.mean(axis=0)) < 2:
        raise PetraError(
            ErrorCode.ARUCO_INSUFFICIENT_MARKERS,
            "ChArUco correspondences must be non-collinear",
        )
    pixel_to_mm, _ = cv2.findHomography(detection.points_px, detection.points_mm, method=0)
    if pixel_to_mm is None:
        raise PetraError(ErrorCode.ARUCO_INSUFFICIENT_MARKERS, "homography estimation failed")
    pixel_to_mm = cast(NDArray[np.float64], np.asarray(pixel_to_mm, dtype=np.float64))
    residuals = np.linalg.norm(
        _project(detection.points_px, pixel_to_mm) - detection.points_mm,
        axis=1,
    )
    residual_check_mm = float(np.max(residuals))
    if residual_check_mm > 1.0:
        raise PetraError(
            ErrorCode.SESSION_RESIDUAL_REJECTED,
            "ChArUco residual exceeds 1 mm",
            {"residual_check_mm": residual_check_mm},
        )

    centroid = detection.points_px.mean(axis=0)
    native_gsd = native_gsd_at(pixel_to_mm, (float(centroid[0]), float(centroid[1])))
    output_gsd = config.target_gsd_mm_px or native_gsd
    change_pct = abs(output_gsd / native_gsd - 1.0) * 100.0
    if change_pct > config.max_resample_change_pct:
        raise PetraError(
            ErrorCode.RECTIFY_GSD_MISMATCH,
            "requested GSD differs too much from the native scene GSD",
            {"native": native_gsd, "requested": output_gsd, "change_pct": change_pct},
        )

    x_min, y_min, width_mm, height_mm = config.roi_mm
    output_size = (
        max(1, round(width_mm / output_gsd)),
        max(1, round(height_mm / output_gsd)),
    )
    metric_to_raster = np.array(
        [
            [1.0 / output_gsd, 0.0, -x_min / output_gsd],
            [0.0, -1.0 / output_gsd, (y_min + height_mm) / output_gsd],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    pixel_to_raster = metric_to_raster @ pixel_to_mm
    return (
        pixel_to_mm,
        pixel_to_raster,
        native_gsd,
        output_gsd,
        output_gsd / native_gsd,
        residual_check_mm,
        output_size,
    )


def rectify_image(
    image_bgr: NDArray[np.uint8], detection: CharucoDetection, config: RectifyConfig
) -> RectificationResult:
    (
        pixel_to_mm,
        pixel_to_raster,
        native_gsd,
        output_gsd,
        resample_ratio,
        residual_check_mm,
        output_size,
    ) = plan_rectification(detection, config)
    interpolation = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "lanczos4": cv2.INTER_LANCZOS4,
    }[config.interpolator]
    rectified = cv2.warpPerspective(
        image_bgr,
        pixel_to_raster,
        output_size,
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return RectificationResult(
        image_bgr=cast(NDArray[np.uint8], rectified),
        pixel_to_mm=pixel_to_mm,
        pixel_to_raster=pixel_to_raster,
        native_gsd_mm_px=native_gsd,
        output_gsd_mm_px=output_gsd,
        resample_ratio=resample_ratio,
        residual_check_mm=residual_check_mm,
        rectified_img_size=output_size,
        marker_ids=detection.marker_ids,
        interpolator=config.interpolator,
    )


def build_session_meta(
    result: RectificationResult,
    *,
    session_id: str,
    calib_profile_id: str,
    source_image: str,
    rectified_image: str,
    thickness_mm: float,
    background: str,
    reference_plane_height_mm: float,
    parallax_factor: float,
    lidar_divergence_pct: float,
) -> SessionMeta:
    homography = tuple(
        tuple(float(result.pixel_to_mm[row, column]) for column in range(3)) for row in range(3)
    )
    return SessionMeta.model_validate(
        {
            "session_id": session_id,
            "calib_profile_id": calib_profile_id,
            "source_image": source_image,
            "rectified_image": rectified_image,
            "undistorted": True,
            "scale_mm_px": result.output_gsd_mm_px,
            "homography": homography,
            "aruco_ids": result.marker_ids,
            "thickness_mm": thickness_mm,
            "background": background,
            "residual_check_mm": result.residual_check_mm,
            "native_gsd_mm_px": result.native_gsd_mm_px,
            "output_gsd_mm_px": result.output_gsd_mm_px,
            "resample_ratio": result.resample_ratio,
            "rectified_img_size": result.rectified_img_size,
            "reference_plane_height_mm": reference_plane_height_mm,
            "parallax_factor": parallax_factor,
            "lidar_divergence_pct": lidar_divergence_pct,
            "coordinate_frame": "bottom_left_x_right_y_up_mm",
            "interpolator": result.interpolator,
        }
    )
