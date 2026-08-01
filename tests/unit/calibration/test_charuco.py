from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from petra.calibration.board import CharucoBoardConfig
from petra.calibration.charuco import (
    MIN_CORNERS_PER_POSE,
    PoseObservation,
    board_object_points,
    detect_charuco_pose,
    is_planar_non_collinear,
)

pytestmark = pytest.mark.unit

SHIPPED_BOARD = Path("config/boards/charuco-a3-7x9-38mm.json")


def shipped_config() -> CharucoBoardConfig:
    return CharucoBoardConfig.from_json(SHIPPED_BOARD)


def render_board(config: CharucoBoardConfig, *, scale_px_per_mm: float = 4.0) -> np.ndarray:
    board = config.create_board()
    width_mm, height_mm = config.board_size_mm
    size = (round(width_mm * scale_px_per_mm), round(height_mm * scale_px_per_mm))
    image = board.generateImage(size)
    return cv2.copyMakeBorder(image, 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=255)


def test_shipped_board_matches_the_printed_plate() -> None:
    config = shipped_config()
    assert (config.squares_x, config.squares_y) == (7, 9)
    assert config.square_length_mm == pytest.approx(38.0)
    assert config.marker_length_mm == pytest.approx(28.0)
    assert config.dictionary == "DICT_5X5_100"
    assert config.chessboard_corner_count == 48
    assert config.board_size_mm == pytest.approx((266.0, 342.0))
    assert len(board_object_points(config)) == 48


def test_board_config_rejects_marker_not_smaller_than_square(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(
        json.dumps(
            {
                "squares_x": 7,
                "squares_y": 9,
                "square_length_mm": 38.0,
                "marker_length_mm": 38.0,
                "dictionary": "DICT_5X5_100",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="marker_length_mm"):
        CharucoBoardConfig.from_json(path)


def test_detects_full_board_pose_with_all_interior_corners() -> None:
    config = shipped_config()
    observation = detect_charuco_pose(
        render_board(config),
        config,
        source="pose-full.png",
        image_sha256="a" * 64,
    )
    assert observation is not None
    assert len(observation.corners_px) == config.chessboard_corner_count
    assert observation.object_points_mm.shape == (config.chessboard_corner_count, 3)
    assert len(set(observation.corner_ids)) == len(observation.corner_ids)
    assert np.allclose(observation.object_points_mm[:, 2], 0.0)


def test_detects_partial_view_without_requiring_the_whole_board() -> None:
    config = shipped_config()
    image = render_board(config)
    cropped = image[: image.shape[0] * 2 // 3, :]
    observation = detect_charuco_pose(
        cropped,
        config,
        source="pose-partial.png",
        image_sha256="b" * 64,
    )
    assert observation is not None
    assert MIN_CORNERS_PER_POSE <= len(observation.corners_px) < config.chessboard_corner_count


def test_returns_none_when_the_board_is_absent_or_too_small_to_use() -> None:
    config = shipped_config()
    blank = np.full((400, 400, 3), 255, dtype=np.uint8)
    assert detect_charuco_pose(blank, config, source="blank.png", image_sha256="c" * 64) is None

    image = render_board(config)
    sliver = image[: image.shape[0] // 8, :]
    assert detect_charuco_pose(sliver, config, source="sliver.png", image_sha256="d" * 64) is None


def test_collinear_corner_sets_are_rejected_as_degenerate() -> None:
    collinear = np.column_stack(
        (np.arange(20, dtype=np.float64) * 38.0, np.zeros(20), np.zeros(20))
    )
    assert not is_planar_non_collinear(collinear)
    spread = collinear.copy()
    spread[10:, 1] = 38.0
    assert is_planar_non_collinear(spread)


def test_pose_observation_enforces_its_invariants() -> None:
    corners = np.zeros((MIN_CORNERS_PER_POSE, 2), dtype=np.float64)
    objects = np.zeros((MIN_CORNERS_PER_POSE, 3), dtype=np.float64)
    ids = tuple(range(MIN_CORNERS_PER_POSE))

    with pytest.raises(ValueError, match="at least"):
        PoseObservation(
            source="s",
            image_sha256="e" * 64,
            image_size=(10, 10),
            corners_px=corners[:5],
            object_points_mm=objects[:5],
            corner_ids=ids[:5],
        )
    with pytest.raises(ValueError, match="equal length"):
        PoseObservation(
            source="s",
            image_sha256="e" * 64,
            image_size=(10, 10),
            corners_px=corners,
            object_points_mm=objects[:-1],
            corner_ids=ids,
        )
    with pytest.raises(ValueError, match="unique"):
        PoseObservation(
            source="s",
            image_sha256="e" * 64,
            image_size=(10, 10),
            corners_px=corners,
            object_points_mm=objects,
            corner_ids=(0,) * MIN_CORNERS_PER_POSE,
        )
