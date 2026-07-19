from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from petra.contracts import PointsPrompt, PromptPoint
from petra.segmentation.adapters import Sam2Segmenter, TransformersSam2Runtime
from petra.segmentation.registry import DeviceResolver, ModelRegistry

pytestmark = pytest.mark.model


def test_sam2_official_checkpoint_on_requested_accelerator() -> None:
    model_dir = os.environ.get("PETRA_SAM2_MODEL_DIR")
    device = os.environ.get("PETRA_SAM2_DEVICE")
    if model_dir is None or device not in {"mps", "cuda"}:
        pytest.skip("PETRA_SAM2_MODEL_DIR and PETRA_SAM2_DEVICE=mps|cuda are required")
    registry = ModelRegistry.from_json(Path("config/models/registry.json"))
    registry.verify("sam2.1-hiera-small")
    descriptor = registry.entry("sam2.1-hiera-small").descriptor
    resolved_device = DeviceResolver.resolve(device, descriptor.supported_devices)
    runtime = TransformersSam2Runtime(model_dir, device=resolved_device)
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    prompt = PointsPrompt(points=(PromptPoint(point=(128.0, 128.0), label=1),))
    prediction = Sam2Segmenter(descriptor, runtime).segment(image, prompt)[0]
    assert prediction.mask.shape == image.shape[:2]
