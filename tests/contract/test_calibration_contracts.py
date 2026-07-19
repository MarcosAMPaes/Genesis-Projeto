from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from petra.contracts import CalibProfile, SessionMeta

ULID_PROFILE = "01KXY0FVMP1V3TZ9XDMXQQ6GMQ"
ULID_SESSION = "01KXY0GVMP1V3TZ9XDMXQQ6GMR"


def sha(index: int) -> str:
    return f"{index:064x}"


def valid_profile_data() -> dict[str, object]:
    images = [{"path": f"pose-{index}.png", "sha256": sha(index)} for index in range(20)]
    residuals = [
        {
            "image_sha256": sha(index),
            "rms_px": 0.1 + index / 1000,
            "rvec": [0.0, 0.0, 0.0],
            "tvec": [0.0, 0.0, 800.0],
        }
        for index in range(20)
    ]
    return {
        "schema_version": "1.0.0",
        "id": ULID_PROFILE,
        "content_sha256": "f" * 64,
        "device": "iPhone17Pro/main-1x",
        "lens": "main-1x-fixed-focus",
        "K": [[5000.0, 0.0, 2000.0], [0.0, 5000.0, 1500.0], [0.0, 0.0, 1.0]],
        "dist": [0.01, -0.02, 0.0, 0.0, 0.001],
        "rms_px": 0.31,
        "img_size": [4032, 3024],
        "z_mm_lidar": 800.0,
        "created_at": "2026-07-19T15:00:00Z",
        "bench_config_hash": "e" * 64,
        "included_images": images,
        "pose_residuals": residuals,
        "excluded_images": [],
    }


def valid_session_data() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "session_id": ULID_SESSION,
        "calib_profile_id": ULID_PROFILE,
        "source_image": "raw/session/image.png",
        "rectified_image": "rectified/session/image.png",
        "undistorted": True,
        "scale_mm_px": 0.08,
        "homography": [[1.0, 0.0, 0.0], [0.0, -1.0, 600.0], [0.0, 0.0, 1.0]],
        "aruco_ids": [1, 2, 3, 4],
        "thickness_mm": 20.0,
        "background": "verde-fosco",
        "residual_check_mm": 0.42,
        "native_gsd_mm_px": 0.081,
        "output_gsd_mm_px": 0.08,
        "resample_ratio": 0.08 / 0.081,
        "rectified_img_size": [7500, 7500],
        "roi_mm": [0.0, 0.0, 600.0, 600.0],
        "reference_plane_height_mm": 20.0,
        "parallax_factor": 1.0,
        "lidar_divergence_pct": 0.3,
        "coordinate_frame": "bottom_left_x_right_y_up_mm",
        "interpolator": "linear",
    }


@pytest.mark.contract
def test_calib_profile_accepts_exactly_twenty_traced_poses() -> None:
    profile = CalibProfile.model_validate(valid_profile_data())
    assert len(profile.included_images) == 20
    assert profile.created_at == datetime(2026, 7, 19, 15, tzinfo=UTC)


@pytest.mark.contract
@pytest.mark.parametrize(
    ("field", "value"),
    [("rms_px", 0.5), ("included_images", []), ("created_at", "2026-07-19T15:00:00")],
)
def test_calib_profile_rejects_gate_and_provenance_violations(field: str, value: object) -> None:
    data = valid_profile_data()
    data[field] = value
    with pytest.raises(ValidationError):
        CalibProfile.model_validate(data)


@pytest.mark.contract
def test_calib_profile_forbids_unknown_fields() -> None:
    data = valid_profile_data()
    data["silent_outlier_removal"] = True
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CalibProfile.model_validate(data)


@pytest.mark.contract
def test_calib_profile_rejects_invalid_ulid_and_pose_provenance() -> None:
    invalid_id = valid_profile_data()
    invalid_id["id"] = "not-an-ulid"
    with pytest.raises(ValidationError, match=r"ULID|ulid"):
        CalibProfile.model_validate(invalid_id)

    duplicate_image = valid_profile_data()
    images = duplicate_image["included_images"]
    assert isinstance(images, list)
    images[1] = images[0]
    with pytest.raises(ValidationError, match="unique"):
        CalibProfile.model_validate(duplicate_image)

    missing_residual = valid_profile_data()
    residuals = missing_residual["pose_residuals"]
    assert isinstance(residuals, list)
    residuals[-1]["image_sha256"] = sha(0)
    with pytest.raises(ValidationError, match="exactly one"):
        CalibProfile.model_validate(missing_residual)

    included_and_excluded = valid_profile_data()
    included_and_excluded["excluded_images"] = [
        {"path": "also-excluded.png", "sha256": sha(0), "reason": "blur"}
    ]
    with pytest.raises(ValidationError, match="both included and excluded"):
        CalibProfile.model_validate(included_and_excluded)


@pytest.mark.contract
def test_session_meta_enforces_gsd_and_marker_invariants() -> None:
    session = SessionMeta.model_validate(valid_session_data())
    assert session.scale_mm_px == session.output_gsd_mm_px

    duplicate_markers = valid_session_data()
    duplicate_markers["aruco_ids"] = [1, 1, 2, 3]
    with pytest.raises(ValidationError, match="unique"):
        SessionMeta.model_validate(duplicate_markers)

    inconsistent_ratio = valid_session_data()
    inconsistent_ratio["resample_ratio"] = 1.0
    with pytest.raises(ValidationError, match="inconsistent"):
        SessionMeta.model_validate(inconsistent_ratio)

    inconsistent_scale = valid_session_data()
    inconsistent_scale["scale_mm_px"] = 0.09
    with pytest.raises(ValidationError, match="must equal"):
        SessionMeta.model_validate(inconsistent_scale)
