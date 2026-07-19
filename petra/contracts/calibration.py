from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import Field, model_validator

from petra.contracts.base import (
    SCHEMA_VERSION,
    ContractModel,
    Matrix3x3,
    Sha256,
    UlidString,
    UtcDateTime,
)

PositiveFloat = Annotated[float, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0)]
ImageSize = tuple[Annotated[int, Field(gt=0)], Annotated[int, Field(gt=0)]]
Distortion5 = tuple[float, float, float, float, float]
Vector3 = tuple[float, float, float]


class CalibrationImage(ContractModel):
    path: str = Field(min_length=1)
    sha256: Sha256


class ExcludedCalibrationImage(CalibrationImage):
    reason: str = Field(min_length=1)


class PoseResidual(ContractModel):
    image_sha256: Sha256
    rms_px: NonNegativeFloat
    rvec: Vector3
    tvec: Vector3


class CalibProfile(ContractModel):
    """Perfil intrinseco aceito; tentativas reprovadas usam relatorio diagnostico."""

    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    id: UlidString
    content_sha256: Sha256
    device: str = Field(min_length=1)
    lens: str = Field(min_length=1)
    K: Matrix3x3
    dist: Distortion5
    rms_px: Annotated[float, Field(ge=0, lt=0.5)]
    img_size: ImageSize
    z_mm_lidar: PositiveFloat
    created_at: UtcDateTime
    bench_config_hash: Sha256
    included_images: tuple[CalibrationImage, ...] = Field(min_length=20)
    pose_residuals: tuple[PoseResidual, ...] = Field(min_length=20)
    excluded_images: tuple[ExcludedCalibrationImage, ...] = ()

    @model_validator(mode="after")
    def validate_pose_provenance(self) -> CalibProfile:
        included = [image.sha256 for image in self.included_images]
        residuals = [pose.image_sha256 for pose in self.pose_residuals]
        excluded = [image.sha256 for image in self.excluded_images]
        if len(set(included)) != len(included):
            raise ValueError("included calibration image hashes must be unique")
        if len(set(excluded)) != len(excluded):
            raise ValueError("excluded calibration image hashes must be unique")
        if set(included) & set(excluded):
            raise ValueError("an image cannot be both included and excluded")
        if set(included) != set(residuals) or len(residuals) != len(included):
            raise ValueError("each included image must have exactly one pose residual")
        return self


class SessionMeta(ContractModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    session_id: UlidString
    calib_profile_id: UlidString
    source_image: str = Field(min_length=1)
    rectified_image: str = Field(min_length=1)
    undistorted: Literal[True]
    scale_mm_px: PositiveFloat
    homography: Matrix3x3
    aruco_ids: tuple[int, ...] = Field(min_length=4)
    thickness_mm: PositiveFloat
    background: str = Field(min_length=1)
    residual_check_mm: Annotated[float, Field(ge=0, le=1.0)]
    native_gsd_mm_px: PositiveFloat
    output_gsd_mm_px: PositiveFloat
    resample_ratio: PositiveFloat = Field(description="output_gsd_mm_px / native_gsd_mm_px")
    rectified_img_size: ImageSize
    reference_plane_height_mm: NonNegativeFloat
    parallax_factor: PositiveFloat
    lidar_divergence_pct: NonNegativeFloat
    coordinate_frame: Literal["bottom_left_x_right_y_up_mm"]
    interpolator: Literal["nearest", "linear", "cubic", "lanczos4"]

    @model_validator(mode="after")
    def validate_session_invariants(self) -> SessionMeta:
        if len(set(self.aruco_ids)) != len(self.aruco_ids):
            raise ValueError("aruco_ids must be unique")
        if not math.isclose(self.scale_mm_px, self.output_gsd_mm_px, rel_tol=1e-9):
            raise ValueError("scale_mm_px must equal output_gsd_mm_px")
        expected_ratio = self.output_gsd_mm_px / self.native_gsd_mm_px
        if not math.isclose(self.resample_ratio, expected_ratio, rel_tol=1e-6):
            raise ValueError("resample_ratio is inconsistent with GSD values")
        return self
