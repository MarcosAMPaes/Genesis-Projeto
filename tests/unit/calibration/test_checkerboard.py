from __future__ import annotations

import cv2
import numpy as np
import pytest

from petra.calibration.checkerboard import CheckerboardConfig, detect_checkerboard, object_points

pytestmark = pytest.mark.unit


def test_object_points_use_certified_square_size() -> None:
    config = CheckerboardConfig(columns=9, rows=6, square_size_mm=25.0)
    points = object_points(config)
    assert points.shape == (54, 3)
    np.testing.assert_array_equal(points[1] - points[0], [25.0, 0.0, 0.0])
    np.testing.assert_array_equal(points[9] - points[0], [0.0, 25.0, 0.0])


def test_find_chessboard_corners_sb_detects_generated_board() -> None:
    config = CheckerboardConfig(columns=9, rows=6, square_size_mm=25.0)
    square_px = 70
    board = np.zeros(((config.rows + 1) * square_px, (config.columns + 1) * square_px), np.uint8)
    for row in range(config.rows + 1):
        for column in range(config.columns + 1):
            if (row + column) % 2 == 0:
                cv2.rectangle(
                    board,
                    (column * square_px, row * square_px),
                    ((column + 1) * square_px, (row + 1) * square_px),
                    255,
                    thickness=-1,
                )
    canvas = np.full((800, 1000), 127, dtype=np.uint8)
    canvas[140 : 140 + board.shape[0], 150 : 150 + board.shape[1]] = board
    observation = detect_checkerboard(
        canvas,
        config,
        source="generated.png",
        image_sha256="a" * 64,
    )
    assert observation is not None
    assert observation.corners_px.shape == (54, 2)
    assert observation.image_size == (1000, 800)
