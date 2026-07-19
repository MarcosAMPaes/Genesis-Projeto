from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree


@dataclass(frozen=True, slots=True)
class MaskMetrics:
    intersection_px: int
    union_px: int
    iou: float
    hausdorff_mm: float


def intersection_over_union(
    prediction: NDArray[np.bool_], truth: NDArray[np.bool_]
) -> tuple[int, int, float]:
    _validate_masks(prediction, truth)
    intersection = int(np.count_nonzero(prediction & truth))
    union = int(np.count_nonzero(prediction | truth))
    return intersection, union, 1.0 if union == 0 else intersection / union


def _boundary_points(mask: NDArray[np.bool_]) -> NDArray[np.float64]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.empty((0, 2), dtype=np.float64)
    return np.concatenate([contour[:, 0, :] for contour in contours]).astype(np.float64, copy=False)


def mask_hausdorff_mm(
    prediction: NDArray[np.bool_], truth: NDArray[np.bool_], *, scale_mm_px: float
) -> float:
    _validate_masks(prediction, truth)
    if not np.isfinite(scale_mm_px) or scale_mm_px <= 0:
        raise ValueError("scale_mm_px must be finite and positive")
    predicted_points = _boundary_points(prediction)
    truth_points = _boundary_points(truth)
    if len(predicted_points) == 0 and len(truth_points) == 0:
        return 0.0
    if len(predicted_points) == 0 or len(truth_points) == 0:
        return float("inf")
    predicted_tree = cKDTree(predicted_points)
    truth_tree = cKDTree(truth_points)
    predicted_to_truth = truth_tree.query(predicted_points, workers=1)[0]
    truth_to_predicted = predicted_tree.query(truth_points, workers=1)[0]
    return float(max(np.max(predicted_to_truth), np.max(truth_to_predicted)) * scale_mm_px)


def measure_masks(
    prediction: NDArray[np.bool_], truth: NDArray[np.bool_], *, scale_mm_px: float
) -> MaskMetrics:
    intersection, union, iou = intersection_over_union(prediction, truth)
    return MaskMetrics(
        intersection_px=intersection,
        union_px=union,
        iou=iou,
        hausdorff_mm=mask_hausdorff_mm(prediction, truth, scale_mm_px=scale_mm_px),
    )


def _validate_masks(prediction: NDArray[np.bool_], truth: NDArray[np.bool_]) -> None:
    if prediction.dtype != np.bool_ or truth.dtype != np.bool_:
        raise ValueError("masks must use bool dtype")
    if prediction.ndim != 2 or truth.ndim != 2 or prediction.shape != truth.shape:
        raise ValueError("masks must be two-dimensional with matching shapes")
