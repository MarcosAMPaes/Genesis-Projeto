from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal, cast

import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from scipy import ndimage

from petra.errors import ErrorCode, PetraError


class PostprocessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    morphology_kernel_px: Literal[0, 3, 5] = 0
    min_area_mm2: Annotated[float, Field(gt=0)] = 2_500.0
    max_area_mm2: Annotated[float, Field(gt=0)] = 1_000_000.0
    large_component_min_fraction: Annotated[float, Field(gt=0, lt=1)] = 0.1


@dataclass(frozen=True, slots=True)
class ProcessedMask:
    instance_index: int
    mask: NDArray[np.bool_]
    area_mm2: float
    source_components: int
    filled_hole_pixels: int
    morphology_kernel_px: int


@dataclass(frozen=True, slots=True)
class MaskRejection:
    instance_index: int
    code: ErrorCode
    message: str
    details: dict[str, object]


@dataclass(frozen=True, slots=True)
class PostprocessResult:
    accepted: tuple[ProcessedMask, ...]
    rejected: tuple[MaskRejection, ...]


def _binary(mask: NDArray[np.bool_] | NDArray[np.uint8]) -> NDArray[np.bool_]:
    if mask.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    return np.asarray(mask > 0, dtype=np.bool_)


def _components(mask: NDArray[np.bool_]) -> tuple[int, NDArray[np.int32], NDArray[np.int64]]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    areas = np.asarray(stats[1:, cv2.CC_STAT_AREA], dtype=np.int64)
    return count - 1, cast(NDArray[np.int32], labels), areas


def _large_component_count(areas: NDArray[np.int64], fraction: float) -> int:
    if not len(areas):
        return 0
    threshold = max(1.0, float(np.max(areas)) * fraction)
    return int(np.count_nonzero(areas >= threshold))


def postprocess_mask(
    mask: NDArray[np.bool_] | NDArray[np.uint8],
    *,
    instance_index: int,
    scale_mm_px: float,
    parallax_factor: float = 1.0,
    marker_mask: NDArray[np.bool_] | NDArray[np.uint8] | None = None,
    contact_ambiguous: bool = False,
    config: PostprocessConfig | None = None,
) -> ProcessedMask:
    selected_config = config or PostprocessConfig()
    if scale_mm_px <= 0 or parallax_factor <= 0:
        raise ValueError("scale_mm_px and parallax_factor must be positive")
    binary = _binary(mask)
    if contact_ambiguous:
        raise PetraError(
            ErrorCode.MASK_CONTACT_AMBIGUOUS,
            "touching pieces could not be separated safely",
        )
    component_count, labels, areas = _components(binary)
    if component_count == 0:
        raise PetraError(ErrorCode.MASK_EMPTY, "segmentation mask is empty")
    if _large_component_count(areas, selected_config.large_component_min_fraction) > 1:
        raise PetraError(
            ErrorCode.MASK_MULTICOMPONENT,
            "one adapter instance contains multiple large components",
            {"component_areas_px": areas.tolist()},
        )
    largest_label = int(np.argmax(areas)) + 1
    largest = labels == largest_label
    filled = cast(NDArray[np.bool_], ndimage.binary_fill_holes(largest))
    filled_hole_pixels = int(np.count_nonzero(filled) - np.count_nonzero(largest))

    kernel_size = selected_config.morphology_kernel_px
    processed = filled
    if kernel_size:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        processed = cv2.morphologyEx(filled.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
    if marker_mask is not None:
        markers = _binary(marker_mask)
        if markers.shape != processed.shape:
            raise ValueError("marker mask must match segmentation mask dimensions")
        processed = processed & ~markers

    post_count, _post_labels, post_areas = _components(processed)
    if post_count == 0:
        raise PetraError(ErrorCode.MASK_EMPTY, "mask became empty after marker exclusion")
    if _large_component_count(post_areas, selected_config.large_component_min_fraction) > 1:
        raise PetraError(
            ErrorCode.MASK_MULTICOMPONENT,
            "marker exclusion split an instance into multiple large components",
            {"component_areas_px": post_areas.tolist()},
        )
    if (
        bool(np.any(processed[0, :]))
        or bool(np.any(processed[-1, :]))
        or bool(np.any(processed[:, 0]))
        or bool(np.any(processed[:, -1]))
    ):
        raise PetraError(ErrorCode.MASK_BORDER_TOUCH, "mask touches the image border")

    metric_scale = scale_mm_px * parallax_factor
    area_mm2 = float(np.count_nonzero(processed)) * metric_scale * metric_scale
    if not selected_config.min_area_mm2 <= area_mm2 <= selected_config.max_area_mm2:
        raise PetraError(
            ErrorCode.MASK_AREA_RANGE,
            "mask area is outside the accepted physical range",
            {
                "area_mm2": area_mm2,
                "min_area_mm2": selected_config.min_area_mm2,
                "max_area_mm2": selected_config.max_area_mm2,
            },
        )
    return ProcessedMask(
        instance_index=instance_index,
        mask=processed,
        area_mm2=area_mm2,
        source_components=component_count,
        filled_hole_pixels=filled_hole_pixels,
        morphology_kernel_px=kernel_size,
    )


def postprocess_instances(
    masks: list[NDArray[np.bool_] | NDArray[np.uint8]],
    *,
    scale_mm_px: float,
    parallax_factor: float = 1.0,
    marker_mask: NDArray[np.bool_] | NDArray[np.uint8] | None = None,
    ambiguous_instances: set[int] | None = None,
    config: PostprocessConfig | None = None,
) -> PostprocessResult:
    accepted: list[ProcessedMask] = []
    rejected: list[MaskRejection] = []
    ambiguous = ambiguous_instances or set()
    for index, mask in enumerate(masks):
        try:
            accepted.append(
                postprocess_mask(
                    mask,
                    instance_index=index,
                    scale_mm_px=scale_mm_px,
                    parallax_factor=parallax_factor,
                    marker_mask=marker_mask,
                    contact_ambiguous=index in ambiguous,
                    config=config,
                )
            )
        except PetraError as error:
            rejected.append(
                MaskRejection(
                    instance_index=index,
                    code=error.code,
                    message=error.message,
                    details=error.details or {},
                )
            )
    return PostprocessResult(accepted=tuple(accepted), rejected=tuple(rejected))
