from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from petra.segmentation.adapters import (
    BiRefNetSegmenter,
    BlockedSam31Runtime,
    ChromaSegmenter,
    Sam2Segmenter,
    Sam31Segmenter,
    Segmenter,
    TransformersBiRefNetRuntime,
    TransformersSam2Runtime,
)
from petra.segmentation.registry import DeviceResolver, ModelRegistry


@dataclass(frozen=True, slots=True)
class ResolvedSegmenter:
    segmenter: Segmenter
    device: Literal["cpu", "mps", "cuda"]


def resolve_segmenter(
    registry: ModelRegistry,
    backend: str,
    *,
    requested_device: Literal["auto", "cpu", "mps", "cuda"],
    background: str,
) -> ResolvedSegmenter:
    registry.verify(backend)
    entry = registry.entry(backend)
    device = DeviceResolver.resolve(requested_device, entry.descriptor.supported_devices)
    family = entry.descriptor.family
    segmenter: Segmenter
    if family == "chroma":
        segmenter = ChromaSegmenter(entry.descriptor, background=background)
    elif family == "birefnet":
        runtime = TransformersBiRefNetRuntime(
            str(registry.weights_path(backend).parent), device=device
        )
        segmenter = BiRefNetSegmenter(entry.descriptor, runtime)
    elif family == "sam2":
        sam2_runtime = TransformersSam2Runtime(
            str(registry.weights_path(backend).parent), device=device
        )
        segmenter = Sam2Segmenter(entry.descriptor, sam2_runtime)
    else:
        segmenter = Sam31Segmenter(
            entry.descriptor,
            BlockedSam31Runtime(),
            device=device,
        )
    return ResolvedSegmenter(segmenter=segmenter, device=device)
