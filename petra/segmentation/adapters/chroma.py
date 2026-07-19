from __future__ import annotations

from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from petra.contracts import AutoPrompt, MaskPrediction, ModelDescriptor, PromptSpec

BACKGROUND_RANGES: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "green": ((35, 35, 20), (95, 255, 255)),
    "verde-fosco": ((35, 35, 20), (95, 255, 255)),
    "blue": ((90, 35, 20), (140, 255, 255)),
    "azul-fosco": ((90, 35, 20), (140, 255, 255)),
}


class ChromaSegmenter:
    def __init__(
        self,
        descriptor: ModelDescriptor,
        *,
        background: str,
        min_component_px: int = 1,
    ) -> None:
        if descriptor.family != "chroma":
            raise ValueError("ChromaSegmenter requires a chroma descriptor")
        if min_component_px <= 0:
            raise ValueError("min_component_px must be positive")
        if background not in BACKGROUND_RANGES:
            raise ValueError(f"unsupported chroma background: {background}")
        self.descriptor = descriptor
        self.background = background
        self.min_component_px = min_component_px

    def segment(
        self,
        image_rgb: NDArray[np.uint8],
        prompt: PromptSpec,
    ) -> list[MaskPrediction]:
        if not isinstance(prompt, AutoPrompt):
            raise ValueError("chroma backend supports only auto prompt")
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8:
            raise ValueError("image_rgb must be an HxWx3 uint8 array")
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        lower, upper = BACKGROUND_RANGES[self.background]
        background_mask = cv2.inRange(
            hsv,
            np.asarray(lower, dtype=np.uint8),
            np.asarray(upper, dtype=np.uint8),
        )
        foreground = background_mask == 0
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            foreground.astype(np.uint8), connectivity=8
        )
        predictions: list[MaskPrediction] = []
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < self.min_component_px:
                continue
            predictions.append(
                MaskPrediction(
                    mask=cast(NDArray[np.bool_], labels == label),
                    score=1.0,
                    descriptor=self.descriptor,
                )
            )
        predictions.sort(key=lambda item: int(np.count_nonzero(item.mask)), reverse=True)
        return predictions
