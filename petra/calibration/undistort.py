from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from petra.contracts import CalibProfile
from petra.errors import ErrorCode, PetraError


@dataclass(frozen=True, slots=True)
class UndistortMaps:
    map1: NDArray[np.int16]
    map2: NDArray[np.uint16]


@dataclass(frozen=True, slots=True)
class UndistortedFrame:
    image_bgr: NDArray[np.uint8]
    calib_profile_id: str
    undistorted: bool = True


class Undistorter:
    def __init__(self) -> None:
        self._cache: dict[tuple[str, tuple[int, int], bytes], UndistortMaps] = {}

    @staticmethod
    def _camera_parameters(
        profile: CalibProfile,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return np.asarray(profile.K, dtype=np.float64), np.asarray(profile.dist, dtype=np.float64)

    @staticmethod
    def _validate_image_size(profile: CalibProfile, image_size: tuple[int, int]) -> None:
        if image_size != profile.img_size:
            raise PetraError(
                ErrorCode.IMAGE_SIZE_MISMATCH,
                "capture resolution differs from calibration profile",
                {"expected": profile.img_size, "actual": image_size},
            )

    def maps(
        self,
        profile: CalibProfile,
        image_size: tuple[int, int],
        output_camera_matrix: NDArray[np.float64] | None = None,
    ) -> UndistortMaps:
        self._validate_image_size(profile, image_size)
        camera_matrix, distortion = self._camera_parameters(profile)
        output_matrix = (
            camera_matrix
            if output_camera_matrix is None
            else np.asarray(output_camera_matrix, dtype=np.float64)
        )
        if output_matrix.shape != (3, 3) or not np.isfinite(output_matrix).all():
            raise ValueError("output_camera_matrix must be a finite 3x3 matrix")
        key = (profile.content_sha256, image_size, output_matrix.tobytes())
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        map1, map2 = cv2.initUndistortRectifyMap(
            camera_matrix,
            distortion,
            None,
            output_matrix,
            image_size,
            cv2.CV_16SC2,
        )
        maps = UndistortMaps(
            map1=cast(NDArray[np.int16], map1),
            map2=cast(NDArray[np.uint16], map2),
        )
        self._cache[key] = maps
        return maps

    def apply(
        self,
        image_bgr: NDArray[np.uint8],
        profile: CalibProfile,
        output_camera_matrix: NDArray[np.float64] | None = None,
    ) -> UndistortedFrame:
        if image_bgr.ndim not in (2, 3):
            raise ValueError("capture must be grayscale or BGR")
        height, width = image_bgr.shape[:2]
        maps = self.maps(profile, (width, height), output_camera_matrix)
        corrected = cv2.remap(
            image_bgr,
            maps.map1,
            maps.map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        return UndistortedFrame(
            image_bgr=cast(NDArray[np.uint8], corrected),
            calib_profile_id=profile.id,
        )

    def undistort_points(
        self,
        points_px: NDArray[np.float64],
        profile: CalibProfile,
    ) -> NDArray[np.float64]:
        if points_px.ndim != 2 or points_px.shape[1] != 2:
            raise ValueError("points_px must have shape (n, 2)")
        camera_matrix, distortion = self._camera_parameters(profile)
        corrected = cv2.undistortPoints(
            points_px.reshape(-1, 1, 2),
            camera_matrix,
            distortion,
            P=camera_matrix,
        )
        return cast(NDArray[np.float64], corrected.reshape(-1, 2))
