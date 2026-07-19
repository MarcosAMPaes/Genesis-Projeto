from __future__ import annotations

from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray


def epsilon_px(dp_epsilon_mm: float, scale_mm_px: float) -> float:
    if dp_epsilon_mm <= 0 or scale_mm_px <= 0:
        raise ValueError("epsilon and metric scale must be positive")
    return dp_epsilon_mm / scale_mm_px


def external_contour(mask: NDArray[np.bool_]) -> NDArray[np.float64]:
    if mask.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    contours, _hierarchy = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        raise ValueError("mask has no external contour")
    contour = max(contours, key=cv2.contourArea)
    return cast(NDArray[np.float64], np.asarray(contour, dtype=np.float64).reshape(-1, 2))


def simplify_contour(
    contour_px: NDArray[np.float64],
    *,
    dp_epsilon_mm: float,
    scale_mm_px: float,
) -> tuple[NDArray[np.float64], float]:
    if contour_px.ndim != 2 or contour_px.shape[1] != 2 or len(contour_px) < 3:
        raise ValueError("contour must have shape (n, 2) with at least three points")
    epsilon = epsilon_px(dp_epsilon_mm, scale_mm_px)
    simplified = cv2.approxPolyDP(
        contour_px.astype(np.float32).reshape(-1, 1, 2),
        epsilon=epsilon,
        closed=True,
    )
    return (
        cast(NDArray[np.float64], np.asarray(simplified, dtype=np.float64).reshape(-1, 2)),
        epsilon,
    )
