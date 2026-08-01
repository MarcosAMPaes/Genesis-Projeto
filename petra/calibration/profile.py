from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import cv2
import numpy as np
from jsonschema.validators import validator_for
from numpy.typing import NDArray
from pydantic import Field
from ulid import ULID

from petra.calibration.board import CharucoBoardConfig
from petra.calibration.charuco import (
    PoseObservation,
    detect_charuco_pose,
)
from petra.calibration.intrinsic import IntrinsicCalibrationResult, calibrate_intrinsics
from petra.contracts import (
    CalibProfile,
    CalibrationImage,
    ExcludedCalibrationImage,
    PoseResidual,
)
from petra.contracts.base import ContractModel, UtcDateTime
from petra.errors import ErrorCode, PetraError


class CalibrationAttemptReport(ContractModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    status: Literal["automated-accepted", "rejected"]
    created_at: UtcDateTime
    error_code: ErrorCode | None = None
    message: str
    rms_px: float | None = Field(default=None, ge=0)
    valid_poses: int = Field(ge=0)
    included_images: tuple[CalibrationImage, ...]
    excluded_images: tuple[ExcludedCalibrationImage, ...]
    detection_failures: tuple[str, ...]
    pose_residuals: tuple[PoseResidual, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_profile_hash(profile: CalibProfile) -> str:
    content = profile.model_dump(mode="json", exclude={"content_sha256"})
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_profile(
    result: IntrinsicCalibrationResult,
    *,
    device: str,
    lens: str,
    z_mm_lidar: float,
    bench_config_hash: str,
    included_images: tuple[CalibrationImage, ...],
    excluded_images: tuple[ExcludedCalibrationImage, ...] = (),
    profile_id: str | None = None,
    created_at: datetime | None = None,
) -> CalibProfile:
    if not result.accepted:
        raise PetraError(
            ErrorCode.CALIB_RMS_REJECTED,
            "calibration profile cannot be persisted before TA-1 acceptance",
            {"rms_px": result.rms_px, "valid_poses": len(result.poses)},
        )
    pose_residuals = tuple(
        PoseResidual(
            image_sha256=pose.image_sha256,
            rms_px=pose.rms_px,
            rvec=pose.rvec,
            tvec=pose.tvec,
        )
        for pose in result.poses
    )
    profile = CalibProfile(
        id=profile_id or str(ULID()),
        content_sha256="0" * 64,
        device=device,
        lens=lens,
        K=(
            (float(result.K[0, 0]), float(result.K[0, 1]), float(result.K[0, 2])),
            (float(result.K[1, 0]), float(result.K[1, 1]), float(result.K[1, 2])),
            (float(result.K[2, 0]), float(result.K[2, 1]), float(result.K[2, 2])),
        ),
        dist=(
            float(result.dist[0]),
            float(result.dist[1]),
            float(result.dist[2]),
            float(result.dist[3]),
            float(result.dist[4]),
        ),
        rms_px=result.rms_px,
        img_size=result.image_size,
        z_mm_lidar=z_mm_lidar,
        created_at=created_at or datetime.now(UTC),
        bench_config_hash=bench_config_hash,
        included_images=included_images,
        pose_residuals=pose_residuals,
        excluded_images=excluded_images,
    )
    return profile.model_copy(update={"content_sha256": canonical_profile_hash(profile)})


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def persist_profile(profile: CalibProfile, path: Path) -> None:
    verify_profile_hash(profile)
    payload: dict[str, Any] = profile.model_dump(mode="json")
    schema = CalibProfile.model_json_schema(mode="serialization")
    validator_for(schema)(schema).validate(payload)
    _atomic_write(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def verify_profile_hash(profile: CalibProfile) -> None:
    expected_hash = canonical_profile_hash(profile)
    if profile.content_sha256 != expected_hash:
        raise PetraError(
            ErrorCode.PROFILE_HASH_MISMATCH,
            "calibration profile content hash is invalid",
            {"expected": expected_hash, "actual": profile.content_sha256},
        )


def persist_attempt_report(report: CalibrationAttemptReport, path: Path) -> None:
    _atomic_write(path, report.model_dump_json(indent=2) + "\n")


def _report_from_result(
    result: IntrinsicCalibrationResult | None,
    *,
    included_images: tuple[CalibrationImage, ...],
    excluded_images: tuple[ExcludedCalibrationImage, ...],
    detection_failures: tuple[str, ...],
    error: PetraError | None,
) -> CalibrationAttemptReport:
    accepted = result is not None and result.accepted and error is None
    pose_residuals = (
        tuple(
            PoseResidual(
                image_sha256=pose.image_sha256,
                rms_px=pose.rms_px,
                rvec=pose.rvec,
                tvec=pose.tvec,
            )
            for pose in result.poses
        )
        if result is not None
        else ()
    )
    return CalibrationAttemptReport(
        status="automated-accepted" if accepted else "rejected",
        created_at=datetime.now(UTC),
        error_code=error.code if error is not None else None,
        message=(str(error) if error is not None else "TA-1 automated accepted"),
        rms_px=result.rms_px if result is not None else None,
        valid_poses=len(result.poses) if result is not None else len(included_images),
        included_images=included_images,
        excluded_images=excluded_images,
        detection_failures=detection_failures,
        pose_residuals=pose_residuals,
    )


def create_calibration_artifacts(
    *,
    image_dir: Path,
    board_path: Path,
    device: str,
    lens: str,
    z_mm_lidar: float,
    bench_config_path: Path,
    profile_path: Path,
    report_path: Path,
    excluded_names: set[str],
) -> bool:
    config = CharucoBoardConfig.from_json(board_path)
    paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    )
    unknown_exclusions = excluded_names - {path.name for path in paths}
    if unknown_exclusions:
        raise ValueError(f"excluded images not found: {sorted(unknown_exclusions)}")

    observations: list[PoseObservation] = []
    included: list[CalibrationImage] = []
    excluded: list[ExcludedCalibrationImage] = []
    failures: list[str] = []
    for path in paths:
        image_hash = sha256_file(path)
        if path.name in excluded_names:
            excluded.append(
                ExcludedCalibrationImage(path=str(path), sha256=image_hash, reason="CLI exclusion")
            )
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            failures.append(str(path))
            continue
        observation = detect_charuco_pose(
            cast(NDArray[np.uint8], image),
            config,
            source=str(path),
            image_sha256=image_hash,
        )
        if observation is None:
            failures.append(str(path))
            continue
        observations.append(observation)
        included.append(CalibrationImage(path=str(path), sha256=image_hash))

    result: IntrinsicCalibrationResult | None = None
    error: PetraError | None = None
    try:
        result = calibrate_intrinsics(observations)
        if not result.accepted:
            error = PetraError(
                ErrorCode.CALIB_RMS_REJECTED,
                "global calibration RMS must be below 0.5 px",
                {"rms_px": result.rms_px},
            )
    except PetraError as caught:
        error = caught

    report = _report_from_result(
        result,
        included_images=tuple(included),
        excluded_images=tuple(excluded),
        detection_failures=tuple(failures),
        error=error,
    )
    persist_attempt_report(report, report_path)
    if error is not None or result is None:
        return False
    profile = build_profile(
        result,
        device=device,
        lens=lens,
        z_mm_lidar=z_mm_lidar,
        bench_config_hash=sha256_file(bench_config_path),
        included_images=tuple(included),
        excluded_images=tuple(excluded),
    )
    persist_profile(profile, profile_path)
    return True
