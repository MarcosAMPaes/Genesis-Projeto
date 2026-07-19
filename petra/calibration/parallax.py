from __future__ import annotations

from dataclasses import dataclass

from petra.errors import ErrorCode, PetraError


@dataclass(frozen=True, slots=True)
class LidarCheck:
    calibrated_z_mm: float
    observed_z_mm: float
    divergence_pct: float
    warning: bool


def resolve_thickness_mm(session_thickness_mm: float, piece_override_mm: float | None) -> float:
    thickness = session_thickness_mm if piece_override_mm is None else piece_override_mm
    if thickness < 0:
        raise PetraError(
            ErrorCode.INVALID_PHYSICAL_VALUE,
            "thickness must be expressed in non-negative millimetres",
            {"thickness_mm": thickness},
        )
    return thickness


def parallax_factor(
    z_mm: float,
    thickness_mm: float,
    reference_plane_height_mm: float,
) -> float:
    if z_mm <= 0:
        raise PetraError(
            ErrorCode.INVALID_PHYSICAL_VALUE,
            "camera distance Z must be positive millimetres",
            {"z_mm": z_mm},
        )
    if thickness_mm < 0 or reference_plane_height_mm < 0:
        raise PetraError(
            ErrorCode.INVALID_PHYSICAL_VALUE,
            "plane heights must be non-negative millimetres",
            {
                "thickness_mm": thickness_mm,
                "reference_plane_height_mm": reference_plane_height_mm,
            },
        )
    if thickness_mm >= z_mm or reference_plane_height_mm >= z_mm:
        raise PetraError(
            ErrorCode.INVALID_PHYSICAL_VALUE,
            "piece and reference planes must remain below the camera",
            {
                "z_mm": z_mm,
                "thickness_mm": thickness_mm,
                "reference_plane_height_mm": reference_plane_height_mm,
            },
        )
    return (z_mm - thickness_mm) / (z_mm - reference_plane_height_mm)


def correct_dimension_mm(
    apparent_dimension_mm: float,
    *,
    z_mm: float,
    session_thickness_mm: float,
    reference_plane_height_mm: float,
    piece_thickness_override_mm: float | None = None,
) -> float:
    if apparent_dimension_mm < 0:
        raise PetraError(
            ErrorCode.INVALID_PHYSICAL_VALUE,
            "apparent dimension must be non-negative millimetres",
            {"apparent_dimension_mm": apparent_dimension_mm},
        )
    thickness = resolve_thickness_mm(session_thickness_mm, piece_thickness_override_mm)
    return apparent_dimension_mm * parallax_factor(
        z_mm,
        thickness,
        reference_plane_height_mm,
    )


def check_lidar_divergence(
    calibrated_z_mm: float,
    observed_z_mm: float,
    *,
    warning_threshold_pct: float = 2.0,
) -> LidarCheck:
    if calibrated_z_mm <= 0 or observed_z_mm <= 0 or warning_threshold_pct <= 0:
        raise PetraError(
            ErrorCode.INVALID_PHYSICAL_VALUE,
            "LiDAR distances and warning threshold must be positive",
        )
    divergence_pct = abs(observed_z_mm - calibrated_z_mm) / calibrated_z_mm * 100.0
    return LidarCheck(
        calibrated_z_mm=calibrated_z_mm,
        observed_z_mm=observed_z_mm,
        divergence_pct=divergence_pct,
        warning=divergence_pct > warning_threshold_pct,
    )
