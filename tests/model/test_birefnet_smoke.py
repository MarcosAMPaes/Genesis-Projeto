from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from petra.contracts import AutoPrompt
from petra.segmentation.adapters import BiRefNetSegmenter, TransformersBiRefNetRuntime
from petra.segmentation.registry import ModelRegistry

pytestmark = pytest.mark.model


def test_birefnet_official_checkpoint_with_numpy2() -> None:
    model_dir = os.environ.get("PETRA_BIREFNET_MODEL_DIR")
    if model_dir is None:
        pytest.skip("PETRA_BIREFNET_MODEL_DIR is not configured")
    registry = ModelRegistry.from_json(Path("config/models/registry.json"))
    registry.verify("birefnet-general")
    descriptor = registry.entry("birefnet-general").descriptor
    runtime = TransformersBiRefNetRuntime(model_dir, device="cpu")
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    prediction = BiRefNetSegmenter(descriptor, runtime).segment(image, AutoPrompt())[0]
    assert np.__version__.startswith("2.")
    assert prediction.mask.shape == image.shape[:2]
