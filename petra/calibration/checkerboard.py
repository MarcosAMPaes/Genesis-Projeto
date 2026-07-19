from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field


class CheckerboardConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    columns: Annotated[int, Field(ge=3)]
    rows: Annotated[int, Field(ge=3)]
    square_size_mm: Annotated[float, Field(gt=0)]

    @property
    def pattern_size(self) -> tuple[int, int]:
        return self.columns, self.rows

    @classmethod
    def from_json(cls, path: Path) -> CheckerboardConfig:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


@dataclass(frozen=True, slots=True)
class PoseObservation:
    source: str
    image_sha256: str
    image_size: tuple[int, int]
    corners_px: NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.corners_px.ndim != 2 or self.corners_px.shape[1] != 2:
            raise ValueError("corners_px must have shape (n, 2)")
        if not np.isfinite(self.corners_px).all():
            raise ValueError("corners_px must be finite")


def object_points(config: CheckerboardConfig) -> NDArray[np.float64]:
    points = np.zeros((config.rows * config.columns, 3), dtype=np.float64)
    grid = np.mgrid[0 : config.columns, 0 : config.rows].T.reshape(-1, 2)
    points[:, :2] = grid * config.square_size_mm
    return points


def detect_checkerboard(
    image_bgr: NDArray[np.uint8],
    config: CheckerboardConfig,
    *,
    source: str,
    image_sha256: str,
) -> PoseObservation | None:
    if image_bgr.ndim not in (2, 3):
        raise ValueError("image must be grayscale or BGR")
    gray = image_bgr if image_bgr.ndim == 2 else cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    found, corners = cv2.findChessboardCornersSB(gray, config.pattern_size, flags=flags)
    if not found or corners is None:
        return None
    height, width = gray.shape
    return PoseObservation(
        source=source,
        image_sha256=image_sha256,
        image_size=(width, height),
        corners_px=np.asarray(corners, dtype=np.float64).reshape(-1, 2),
    )
