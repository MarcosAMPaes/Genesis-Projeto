from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from petra.calibration.intrinsic import IntrinsicCalibrationResult, PoseCalibrationResult
from petra.calibration.profile import build_profile, persist_profile
from petra.calibration.rectify import CharucoBoardConfig
from petra.cli import main
from petra.contracts import CalibrationImage, FragmentGeom, SessionMeta

pytestmark = [pytest.mark.integration, pytest.mark.smoke]


def _write_fixture(root: Path) -> tuple[Path, Path, Path]:
    board_path = root / "charuco.json"
    board_path.write_text(
        json.dumps(
            {
                "squares_x": 8,
                "squares_y": 6,
                "square_length_mm": 30.0,
                "marker_length_mm": 22.0,
                "dictionary": "DICT_5X5_100",
            }
        ),
        encoding="utf-8",
    )
    board_config = CharucoBoardConfig.from_json(board_path)
    board = board_config.create_board().generateImage((400, 300), marginSize=0, borderBits=1)
    image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    image[:, :] = (0, 150, 0)
    image[650:950, 50:450] = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)
    fragment_points = [[340, 80], [900, 120], [850, 600], [380, 570]]
    cv2.fillPoly(image, [np.asarray(fragment_points, dtype=np.int32)], (110, 110, 110))
    image_path = root / "raw.png"
    assert cv2.imwrite(str(image_path), image)

    rectify_path = root / "rectify.json"
    rectify_path.write_text(
        json.dumps(
            {
                "gsd_policy": "preserve_native",
                "target_gsd_mm_px": None,
                "max_resample_change_pct": 10.0,
                "roi_mm": [0.0, 0.0, 540.0, 540.0],
                "interpolator": "nearest",
            }
        ),
        encoding="utf-8",
    )
    return image_path, board_path, rectify_path


def _write_profile(root: Path) -> Path:
    poses = tuple(
        PoseCalibrationResult(
            source=f"pose-{index}.png",
            image_sha256=f"{index:064x}",
            rms_px=0.1,
            rvec=(0.0, 0.0, 0.0),
            tvec=(0.0, 0.0, 800.0),
        )
        for index in range(20)
    )
    result = IntrinsicCalibrationResult(
        K=np.array([[1200.0, 0.0, 500.0], [0.0, 1200.0, 500.0], [0.0, 0.0, 1.0]]),
        dist=np.zeros(5, dtype=np.float64),
        rms_px=0.1,
        image_size=(1000, 1000),
        poses=poses,
    )
    included = tuple(CalibrationImage(path=pose.source, sha256=pose.image_sha256) for pose in poses)
    profile = build_profile(
        result,
        device="synthetic-camera",
        lens="fixed",
        z_mm_lidar=800.0,
        bench_config_hash="a" * 64,
        included_images=included,
        profile_id="01KXY0FVMP1V3TZ9XDMXQQ6GMQ",
        created_at=datetime(2026, 7, 19, tzinfo=UTC),
    )
    profile_path = root / "calib_profile.json"
    persist_profile(profile, profile_path)
    return profile_path


def test_frozen_synthetic_session_runs_a_to_b_idempotently(tmp_path: Path) -> None:
    image_path, board_path, rectify_path = _write_fixture(tmp_path)
    profile_path = _write_profile(tmp_path)
    output_dir = tmp_path / "session-output"
    arguments = [
        "process-session",
        "--image",
        str(image_path),
        "--profile",
        str(profile_path),
        "--board",
        str(board_path),
        "--config",
        str(rectify_path),
        "--session-id",
        "01KXY0GVMP1V3TZ9XDMXQQ6GMR",
        "--thickness-mm",
        "20",
        "--reference-plane-height-mm",
        "0",
        "--background",
        "verde-fosco",
        "--backend",
        "chroma",
        "--git-hash",
        "fixture-commit",
        "--output-dir",
        str(output_dir),
    ]
    assert main(arguments) == 0
    first_report = (output_dir / "process-session.json").read_bytes()
    report = json.loads(first_report)
    assert report["acceptance_state"] == "automated-accepted"
    assert report["ts1_status"] == "partial"
    assert len(report["fragments"]) == 1
    SessionMeta.model_validate_json((output_dir / "session_meta.json").read_text())
    geometry = FragmentGeom.model_validate_json(
        (output_dir / report["fragments"][0]).read_text(encoding="utf-8")
    )
    assert geometry.seg_model == "chroma"
    assert geometry.area_mm2 > 25_000
    assert geometry.n_points < 100
    assert geometry.quality_warnings == ("VERTEX_COUNT_BELOW_EXPECTED",)
    assert report["quality_warnings"] == {
        report["fragments"][0]: ["VERTEX_COUNT_BELOW_EXPECTED"]
    }

    assert main(arguments) == 0
    assert (output_dir / "process-session.json").read_bytes() == first_report

    changed = arguments.copy()
    changed[changed.index("20")] = "21"
    assert main(changed) == 3
