from __future__ import annotations

import importlib
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from petra.contracts import (
    BoxPrompt,
    MaskPrediction,
    ModelDescriptor,
    PointsPrompt,
    PromptSpec,
)

Sam2Prompt = PointsPrompt | BoxPrompt


class Sam2Runtime(Protocol):
    def predict(
        self, image_rgb: NDArray[np.uint8], prompt: Sam2Prompt
    ) -> tuple[NDArray[np.bool_], float]: ...


class TransformersSam2Runtime:
    """Lazy, local-only SAM 2.1 image runtime using the official HF snapshot."""

    def __init__(self, model_dir: str, *, device: str) -> None:
        self.model_dir = model_dir
        self.device = device
        self._model: Any | None = None
        self._processor: Any | None = None

    def _load(self) -> tuple[Any, Any]:
        if self._model is not None and self._processor is not None:
            return self._model, self._processor
        transformers = importlib.import_module("transformers")
        model = transformers.Sam2Model.from_pretrained(
            self.model_dir,
            local_files_only=True,
        ).to(self.device)
        model.eval()
        processor = transformers.Sam2Processor.from_pretrained(
            self.model_dir,
            local_files_only=True,
        )
        self._model = model
        self._processor = processor
        return model, processor

    def predict(
        self, image_rgb: NDArray[np.uint8], prompt: Sam2Prompt
    ) -> tuple[NDArray[np.bool_], float]:
        torch = importlib.import_module("torch")
        model, processor = self._load()
        prompt_args: dict[str, object]
        if isinstance(prompt, PointsPrompt):
            prompt_args = {
                "input_points": [[[[point.point[0], point.point[1]] for point in prompt.points]]],
                "input_labels": [[[point.label for point in prompt.points]]],
            }
        else:
            prompt_args = {"input_boxes": [[list(prompt.box)]]}
        inputs = processor(
            images=Image.fromarray(image_rgb),
            return_tensors="pt",
            **prompt_args,
        ).to(self.device)
        with torch.inference_mode():
            outputs = model(**inputs)
        masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
        scores = outputs.iou_scores.detach().float().cpu().reshape(-1)
        best_index = int(torch.argmax(scores).item())
        candidate_masks = masks.reshape(-1, image_rgb.shape[0], image_rgb.shape[1])
        mask = np.asarray(candidate_masks[best_index].cpu().numpy() > 0, dtype=np.bool_)
        return mask, float(scores[best_index].item())


class Sam2Segmenter:
    def __init__(self, descriptor: ModelDescriptor, runtime: Sam2Runtime) -> None:
        if descriptor.family != "sam2":
            raise ValueError("Sam2Segmenter requires a sam2 descriptor")
        self.descriptor = descriptor
        self.runtime = runtime

    def segment(
        self,
        image_rgb: NDArray[np.uint8],
        prompt: PromptSpec,
    ) -> list[MaskPrediction]:
        if not isinstance(prompt, (PointsPrompt, BoxPrompt)):
            raise ValueError("SAM 2.1 requires points or box prompt")
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8:
            raise ValueError("image_rgb must be an HxWx3 uint8 array")
        mask, score = self.runtime.predict(image_rgb, prompt)
        if mask.shape != image_rgb.shape[:2] or mask.dtype != np.bool_:
            raise ValueError("SAM 2.1 runtime returned an invalid mask")
        return [MaskPrediction(mask=mask, score=score, descriptor=self.descriptor)]
