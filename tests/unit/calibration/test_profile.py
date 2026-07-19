from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

import petra.calibration.profile as profile_module
from petra.calibration.checkerboard import PoseObservation
from petra.calibration.intrinsic import IntrinsicCalibrationResult, PoseCalibrationResult
from petra.calibration.profile import (
    build_profile,
    canonical_profile_hash,
    create_calibration_artifacts,
    persist_profile,
)
from petra.cli import main
from petra.contracts import CalibProfile, CalibrationImage
from petra.errors import ErrorCode, PetraError

pytestmark = pytest.mark.unit


def accepted_result(*, rms_px: float = 0.2) -> IntrinsicCalibrationResult:
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
    return IntrinsicCalibrationResult(
        K=np.array([[1600.0, 0.0, 960.0], [0.0, 1580.0, 540.0], [0.0, 0.0, 1.0]]),
        dist=np.array([0.01, -0.02, 0.001, 0.0, 0.001]),
        rms_px=rms_px,
        image_size=(1920, 1080),
        poses=poses,
    )


def included_images() -> tuple[CalibrationImage, ...]:
    return tuple(
        CalibrationImage(path=f"pose-{index}.png", sha256=f"{index:064x}") for index in range(20)
    )


def test_build_and_atomically_persist_immutable_profile(tmp_path: Path) -> None:
    profile = build_profile(
        accepted_result(),
        device="iPhone17Pro/main-1x",
        lens="main-1x-fixed-focus",
        z_mm_lidar=800.0,
        bench_config_hash="a" * 64,
        included_images=included_images(),
        profile_id="01KXY0FVMP1V3TZ9XDMXQQ6GMQ",
        created_at=datetime(2026, 7, 19, 15, tzinfo=UTC),
    )
    assert profile.content_sha256 == canonical_profile_hash(profile)

    output = tmp_path / "calib_profile.json"
    persist_profile(profile, output)
    loaded = CalibProfile.model_validate_json(output.read_text(encoding="utf-8"))
    assert loaded == profile
    assert not list(tmp_path.glob("*.tmp"))


def test_profile_rejects_unaccepted_result_and_tampered_hash(
    tmp_path: Path,
) -> None:
    with pytest.raises(PetraError) as rejected:
        build_profile(
            accepted_result(rms_px=0.5),
            device="device",
            lens="lens",
            z_mm_lidar=800.0,
            bench_config_hash="a" * 64,
            included_images=included_images(),
        )
    assert rejected.value.code == ErrorCode.CALIB_RMS_REJECTED

    profile = build_profile(
        accepted_result(),
        device="device",
        lens="lens",
        z_mm_lidar=800.0,
        bench_config_hash="a" * 64,
        included_images=included_images(),
    )
    tampered = profile.model_copy(update={"device": "other-device"})
    with pytest.raises(PetraError) as mismatch:
        persist_profile(tampered, tmp_path / "tampered.json")
    assert mismatch.value.code == ErrorCode.PROFILE_HASH_MISMATCH


def test_cli_always_writes_rejection_report_for_insufficient_poses(
    tmp_path: Path,
) -> None:
    images = tmp_path / "images"
    images.mkdir()
    board = tmp_path / "board.json"
    board.write_text(
        json.dumps({"columns": 9, "rows": 6, "square_size_mm": 25.0}),
        encoding="utf-8",
    )
    bench = tmp_path / "bench.json"
    bench.write_text("{}", encoding="utf-8")
    profile = tmp_path / "calib_profile.json"
    report = tmp_path / "calibration_report.json"

    exit_code = main(
        [
            "calibrate",
            "create",
            "--images",
            str(images),
            "--board",
            str(board),
            "--device",
            "device",
            "--lens",
            "lens",
            "--z-mm-lidar",
            "800",
            "--bench-config",
            str(bench),
            "--output",
            str(profile),
            "--report",
            str(report),
        ]
    )
    assert exit_code == 3
    assert not profile.exists()
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "rejected"
    assert payload["error_code"] == ErrorCode.CALIB_INSUFFICIENT_POSES


def test_artifact_orchestration_tracks_exclusions_failures_and_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    images = tmp_path / "images"
    images.mkdir()
    for index in range(22):
        image = np.full((32, 32, 3), index, dtype=np.uint8)
        assert profile_module.cv2.imwrite(str(images / f"image-{index:02d}.png"), image)
    (images / "invalid.png").write_text("not an image", encoding="utf-8")
    board = tmp_path / "board.json"
    board.write_text(
        json.dumps({"columns": 9, "rows": 6, "square_size_mm": 25.0}),
        encoding="utf-8",
    )
    bench = tmp_path / "bench.json"
    bench.write_text('{"rig":"v1"}', encoding="utf-8")

    def fake_detect(
        image: np.ndarray,
        config: object,
        *,
        source: str,
        image_sha256: str,
    ) -> PoseObservation | None:
        del image, config
        if source.endswith("image-20.png"):
            return None
        return PoseObservation(
            source=source,
            image_sha256=image_sha256,
            image_size=(32, 32),
            corners_px=np.zeros((54, 2), dtype=np.float64),
        )

    def fake_calibrate(
        observations: list[PoseObservation], config: object
    ) -> IntrinsicCalibrationResult:
        del config
        poses = tuple(
            PoseCalibrationResult(
                source=observation.source,
                image_sha256=observation.image_sha256,
                rms_px=0.1,
                rvec=(0.0, 0.0, 0.0),
                tvec=(0.0, 0.0, 800.0),
            )
            for observation in observations
        )
        return IntrinsicCalibrationResult(
            K=np.eye(3, dtype=np.float64),
            dist=np.zeros(5, dtype=np.float64),
            rms_px=0.1,
            image_size=(32, 32),
            poses=poses,
        )

    monkeypatch.setattr(profile_module, "detect_checkerboard", fake_detect)
    monkeypatch.setattr(profile_module, "calibrate_intrinsics", fake_calibrate)
    profile_path = tmp_path / "profile.json"
    report_path = tmp_path / "report.json"
    accepted = create_calibration_artifacts(
        image_dir=images,
        board_path=board,
        device="device",
        lens="lens",
        z_mm_lidar=800.0,
        bench_config_path=bench,
        profile_path=profile_path,
        report_path=report_path,
        excluded_names={"image-21.png"},
    )
    assert accepted
    profile = CalibProfile.model_validate_json(profile_path.read_text(encoding="utf-8"))
    assert canonical_profile_hash(profile) == profile.content_sha256
    assert len(profile.included_images) == 20
    assert len(profile.excluded_images) == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "automated-accepted"
    assert len(report["detection_failures"]) == 2

    with pytest.raises(ValueError, match="not found"):
        create_calibration_artifacts(
            image_dir=images,
            board_path=board,
            device="device",
            lens="lens",
            z_mm_lidar=800.0,
            bench_config_path=bench,
            profile_path=profile_path,
            report_path=report_path,
            excluded_names={"missing.png"},
        )
