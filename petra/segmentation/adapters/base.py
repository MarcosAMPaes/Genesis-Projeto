from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from petra.contracts import MaskPrediction, ModelDescriptor, PromptSpec


class Segmenter(Protocol):
    descriptor: ModelDescriptor

    def segment(
        self,
        image_rgb: NDArray[np.uint8],
        prompt: PromptSpec,
    ) -> list[MaskPrediction]: ...
