from __future__ import annotations

import platform
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from petra.contracts import ConceptPrompt, MaskPrediction, ModelDescriptor, PromptSpec
from petra.errors import ErrorCode, PetraError

UPSTREAM_BLOCKER = "https://github.com/facebookresearch/sam3/issues/526"


class Sam31Runtime(Protocol):
    def predict(
        self, image_rgb: NDArray[np.uint8], concept: str
    ) -> list[tuple[NDArray[np.bool_], float]]: ...


class BlockedSam31Runtime:
    """Fail-closed runtime until the official image loader is reproducible."""

    def predict(
        self, image_rgb: NDArray[np.uint8], concept: str
    ) -> list[tuple[NDArray[np.bool_], float]]:
        del image_rgb, concept
        raise PetraError(
            ErrorCode.MODEL_UNAVAILABLE,
            "SAM 3.1 official checkpoint has no validated public image loading path",
            {"upstream_issue": UPSTREAM_BLOCKER},
        )


class Sam31Segmenter:
    def __init__(
        self,
        descriptor: ModelDescriptor,
        runtime: Sam31Runtime,
        *,
        device: str,
        platform_name: str | None = None,
    ) -> None:
        if descriptor.family != "sam3":
            raise ValueError("Sam31Segmenter requires a sam3 descriptor")
        actual_platform = platform_name or platform.system()
        if actual_platform != "Linux" or device != "cuda":
            raise PetraError(
                ErrorCode.MODEL_UNAVAILABLE,
                "SAM 3.1 requires Linux and CUDA",
                {"platform": actual_platform, "device": device},
            )
        self.descriptor = descriptor
        self.runtime = runtime

    def segment(
        self,
        image_rgb: NDArray[np.uint8],
        prompt: PromptSpec,
    ) -> list[MaskPrediction]:
        if not isinstance(prompt, ConceptPrompt):
            raise ValueError("SAM 3.1 requires a concept prompt")
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8:
            raise ValueError("image_rgb must be an HxWx3 uint8 array")
        raw_predictions = self.runtime.predict(image_rgb, prompt.concept)
        predictions: list[MaskPrediction] = []
        for mask, score in raw_predictions:
            if mask.shape != image_rgb.shape[:2] or mask.dtype != np.bool_:
                raise ValueError("SAM 3.1 runtime returned an invalid mask")
            predictions.append(MaskPrediction(mask=mask, score=score, descriptor=self.descriptor))
        return predictions
