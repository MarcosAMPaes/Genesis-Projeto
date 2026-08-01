from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from petra.calibration.charuco import MIN_CORNERS_PER_POSE
from petra.calibration.intrinsic import IntrinsicCalibrationResult, PoseCalibrationResult
from petra.calibration.parallax import (
    check_lidar_divergence,
    correct_dimension_mm,
    parallax_factor,
)
from petra.calibration.profile import build_profile, persist_profile
from petra.calibration.rectify import CharucoBoardConfig
from petra.cli import main
from petra.contracts import CalibrationImage, SessionMeta
from petra.errors import ErrorCode, PetraError

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("thickness_mm", [10.0, 20.0, 30.0, 40.0])
def test_reference_table_formula_for_representative_thicknesses(thickness_mm: float) -> None:
    factor = parallax_factor(800.0, thickness_mm, 0.0)
    assert factor == pytest.approx((800.0 - thickness_mm) / 800.0)
    assert correct_dimension_mm(
        100.0,
        z_mm=800.0,
        session_thickness_mm=thickness_mm,
        reference_plane_height_mm=0.0,
    ) == pytest.approx(100.0 * factor)


def test_equal_piece_and_reference_planes_have_unit_factor() -> None:
    assert parallax_factor(800.0, 30.0, 30.0) == pytest.approx(1.0)


def test_piece_override_changes_only_explicit_piece() -> None:
    session = correct_dimension_mm(
        100.0,
        z_mm=800.0,
        session_thickness_mm=20.0,
        reference_plane_height_mm=0.0,
    )
    override = correct_dimension_mm(
        100.0,
        z_mm=800.0,
        session_thickness_mm=20.0,
        reference_plane_height_mm=0.0,
        piece_thickness_override_mm=40.0,
    )
    assert override < session


@given(
    z_mm=st.floats(min_value=100.0, max_value=2000.0, allow_nan=False),
    thickness_mm=st.floats(min_value=0.0, max_value=90.0, allow_nan=False),
)
def test_factor_remains_positive_below_camera(z_mm: float, thickness_mm: float) -> None:
    if thickness_mm >= z_mm:
        return
    assert parallax_factor(z_mm, thickness_mm, 0.0) > 0


@pytest.mark.parametrize(
    ("z_mm", "thickness_mm", "reference_mm"),
    [(0.0, 10.0, 0.0), (800.0, -1.0, 0.0), (800.0, 800.0, 0.0), (800.0, 10.0, 800.0)],
)
def test_invalid_physical_values_are_rejected(
    z_mm: float, thickness_mm: float, reference_mm: float
) -> None:
    with pytest.raises(PetraError) as invalid:
        parallax_factor(z_mm, thickness_mm, reference_mm)
    assert invalid.value.code == ErrorCode.INVALID_PHYSICAL_VALUE


def test_lidar_divergence_warns_without_changing_calibration() -> None:
    within = check_lidar_divergence(800.0, 815.0)
    assert not within.warning
    above = check_lidar_divergence(800.0, 820.0)
    assert above.warning
    assert above.divergence_pct == pytest.approx(2.5)
    assert above.calibrated_z_mm == 800.0


def test_rectify_cli_writes_image_and_complete_session_meta(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    board_config = CharucoBoardConfig(
        squares_x=8,
        squares_y=6,
        square_length_mm=30.0,
        marker_length_mm=22.0,
        dictionary="DICT_5X5_100",
    )
    board_path = tmp_path / "charuco.json"
    board_path.write_text(
        json.dumps(board_config.model_dump(mode="json")),
        encoding="utf-8",
    )
    image_path = tmp_path / "capture.png"
    assert cv2.imwrite(
        str(image_path),
        board_config.create_board().generateImage((800, 600), marginSize=20),
    )
    rectify_config = tmp_path / "rectify.json"
    rectify_config.write_text(
        json.dumps(
            {
                "gsd_policy": "preserve_native",
                "target_gsd_mm_px": None,
                "max_resample_change_pct": 10.0,
                "roi_mm": [0.0, 0.0, 240.0, 180.0],
                "interpolator": "linear",
            }
        ),
        encoding="utf-8",
    )

    poses = tuple(
        PoseCalibrationResult(
            source=f"pose-{index}.png",
            image_sha256=f"{index:064x}",
            rms_px=0.1,
            corners_used=MIN_CORNERS_PER_POSE,
            rvec=(0.0, 0.0, 0.0),
            tvec=(0.0, 0.0, 800.0),
        )
        for index in range(20)
    )
    result = IntrinsicCalibrationResult(
        K=np.array([[700.0, 0.0, 400.0], [0.0, 700.0, 300.0], [0.0, 0.0, 1.0]]),
        dist=np.zeros(5),
        rms_px=0.1,
        image_size=(800, 600),
        poses=poses,
    )
    profile = build_profile(
        result,
        device="device",
        lens="lens",
        z_mm_lidar=800.0,
        bench_config_hash="a" * 64,
        included_images=tuple(
            CalibrationImage(path=pose.source, sha256=pose.image_sha256) for pose in poses
        ),
        profile_id="01KXY0FVMP1V3TZ9XDMXQQ6GMQ",
        created_at=datetime(2026, 7, 19, 15, tzinfo=UTC),
    )
    profile_path = tmp_path / "profile.json"
    persist_profile(profile, profile_path)
    output_path = tmp_path / "rectified.png"
    meta_path = tmp_path / "session_meta.json"

    exit_code = main(
        [
            "calibrate",
            "rectify",
            "--image",
            str(image_path),
            "--profile",
            str(profile_path),
            "--board",
            str(board_path),
            "--config",
            str(rectify_config),
            "--session-id",
            "01KXY0GVMP1V3TZ9XDMXQQ6GMR",
            "--thickness-mm",
            "20",
            "--reference-plane-height-mm",
            "0",
            "--background",
            "verde-fosco",
            "--z-mm-lidar",
            "820",
            "--output",
            str(output_path),
            "--meta",
            str(meta_path),
        ]
    )
    assert exit_code == 0
    assert output_path.exists()
    session = SessionMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))
    assert session.parallax_factor == pytest.approx(0.975)
    assert session.lidar_divergence_pct == pytest.approx(2.5)
    assert "warning" in capsys.readouterr().out
