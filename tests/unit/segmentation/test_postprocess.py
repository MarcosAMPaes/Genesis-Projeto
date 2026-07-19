from __future__ import annotations

import numpy as np
import pytest

from petra.errors import ErrorCode, PetraError
from petra.segmentation.postprocess import (
    PostprocessConfig,
    postprocess_instances,
    postprocess_mask,
)

pytestmark = pytest.mark.unit


def square_mask() -> np.ndarray:
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255
    return mask


def test_fixed_order_keeps_instance_fills_hole_and_removes_marker() -> None:
    mask = square_mask()
    mask[40:45, 40:45] = 0
    mask[5, 5] = 255
    markers = np.zeros_like(mask)
    markers[30:35, 30:35] = 255
    processed = postprocess_mask(
        mask,
        instance_index=0,
        scale_mm_px=1.0,
        marker_mask=markers,
    )
    assert processed.source_components == 2
    assert processed.filled_hole_pixels == 25
    assert not processed.mask[32, 32]
    assert not processed.mask[5, 5]
    assert processed.area_mm2 == pytest.approx(60 * 60 - 25)


@pytest.mark.parametrize(
    ("mask", "scale", "expected"),
    [
        (np.zeros((100, 100), dtype=np.uint8), 1.0, ErrorCode.MASK_EMPTY),
        (
            np.pad(np.ones((60, 60), dtype=np.uint8), ((0, 40), (20, 20))),
            1.0,
            ErrorCode.MASK_BORDER_TOUCH,
        ),
        (np.pad(np.ones((20, 20), dtype=np.uint8), 40), 1.0, ErrorCode.MASK_AREA_RANGE),
        (square_mask(), 20.0, ErrorCode.MASK_AREA_RANGE),
    ],
)
def test_pathological_masks_are_rejected(
    mask: np.ndarray, scale: float, expected: ErrorCode
) -> None:
    with pytest.raises(PetraError) as rejected:
        postprocess_mask(mask, instance_index=0, scale_mm_px=scale)
    assert rejected.value.code == expected


def test_multiple_large_components_and_ambiguous_contact_are_rejected_with_log() -> None:
    multiple = np.zeros((120, 120), dtype=np.uint8)
    multiple[10:60, 10:60] = 255
    multiple[65:115, 65:115] = 255
    result = postprocess_instances(
        [square_mask(), multiple, square_mask()],
        scale_mm_px=1.0,
        ambiguous_instances={2},
    )
    assert len(result.accepted) == 1
    assert [item.code for item in result.rejected] == [
        ErrorCode.MASK_MULTICOMPONENT,
        ErrorCode.MASK_CONTACT_AMBIGUOUS,
    ]


def test_morphology_and_parallax_are_explicit_and_bounded() -> None:
    processed = postprocess_mask(
        square_mask(),
        instance_index=0,
        scale_mm_px=1.0,
        parallax_factor=0.975,
        config=PostprocessConfig(morphology_kernel_px=3),
    )
    assert processed.morphology_kernel_px == 3
    assert processed.area_mm2 == pytest.approx(3600 * 0.975**2)


def test_invalid_shapes_scales_and_marker_effects_are_rejected() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        postprocess_mask(
            np.zeros((10, 10, 3), dtype=np.uint8),
            instance_index=0,
            scale_mm_px=1.0,
        )
    with pytest.raises(ValueError, match="positive"):
        postprocess_mask(square_mask(), instance_index=0, scale_mm_px=0.0)
    with pytest.raises(ValueError, match="dimensions"):
        postprocess_mask(
            square_mask(),
            instance_index=0,
            scale_mm_px=1.0,
            marker_mask=np.zeros((50, 50), dtype=np.uint8),
        )

    remove_all = square_mask()
    with pytest.raises(PetraError) as empty:
        postprocess_mask(
            square_mask(),
            instance_index=0,
            scale_mm_px=1.0,
            marker_mask=remove_all,
        )
    assert empty.value.code == ErrorCode.MASK_EMPTY

    splitting_marker = np.zeros((100, 100), dtype=np.uint8)
    splitting_marker[20:80, 48:52] = 255
    with pytest.raises(PetraError) as split:
        postprocess_mask(
            square_mask(),
            instance_index=0,
            scale_mm_px=1.0,
            marker_mask=splitting_marker,
        )
    assert split.value.code == ErrorCode.MASK_MULTICOMPONENT
