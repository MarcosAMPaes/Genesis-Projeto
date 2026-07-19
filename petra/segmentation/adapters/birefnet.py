from __future__ import annotations

import importlib
from typing import Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from petra.contracts import AutoPrompt, MaskPrediction, ModelDescriptor, PromptSpec


class BiRefNetRuntime(Protocol):
    def predict(self, image_rgb: NDArray[np.uint8]) -> tuple[NDArray[np.bool_], float]: ...


class TransformersBiRefNetRuntime:
    def __init__(self, model_dir: str, *, device: str, threshold: float = 0.5) -> None:
        if not 0 < threshold < 1:
            raise ValueError("threshold must be in (0, 1)")
        self.model_dir = model_dir
        self.device = device
        self.threshold = threshold
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        transformers = importlib.import_module("transformers")
        model_class = transformers.AutoModelForImageSegmentation
        model = model_class.from_pretrained(
            self.model_dir,
            trust_remote_code=True,
            local_files_only=True,
        )
        model.to(self.device)
        model.eval()
        self._model = model
        return model

    def predict(self, image_rgb: NDArray[np.uint8]) -> tuple[NDArray[np.bool_], float]:
        torch = importlib.import_module("torch")
        model = self._load()
        height, width = image_rgb.shape[:2]
        resized = cv2.resize(image_rgb, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        tensor = ((tensor - mean) / std).to(self.device)
        with torch.inference_mode():
            output = model(tensor)
            logits = output[-1] if isinstance(output, (list, tuple)) else output.logits
            probability = torch.sigmoid(logits)
            probability = torch.nn.functional.interpolate(
                probability,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
        probability_np = probability.detach().float().cpu().numpy()
        mask = np.asarray(probability_np >= self.threshold, dtype=np.bool_)
        score = float(np.mean(probability_np[mask])) if np.any(mask) else 0.0
        return mask, score


class BiRefNetSegmenter:
    def __init__(self, descriptor: ModelDescriptor, runtime: BiRefNetRuntime) -> None:
        if descriptor.family != "birefnet":
            raise ValueError("BiRefNetSegmenter requires a birefnet descriptor")
        self.descriptor = descriptor
        self.runtime = runtime

    def segment(
        self,
        image_rgb: NDArray[np.uint8],
        prompt: PromptSpec,
    ) -> list[MaskPrediction]:
        if not isinstance(prompt, AutoPrompt):
            raise ValueError("BiRefNet supports only auto prompt")
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8:
            raise ValueError("image_rgb must be an HxWx3 uint8 array")
        mask, score = self.runtime.predict(image_rgb)
        if mask.shape != image_rgb.shape[:2] or mask.dtype != np.bool_:
            raise ValueError("BiRefNet runtime returned an invalid mask")
        return [
            MaskPrediction(
                mask=mask,
                score=score,
                descriptor=self.descriptor,
            )
        ]
