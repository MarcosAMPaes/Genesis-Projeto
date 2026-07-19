from __future__ import annotations

import csv
import json
import platform
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Annotated, Any, Literal

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from petra import __version__
from petra.contracts import CalibProfile, SessionMeta


class DimensionMeasurement(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    piece: str = Field(min_length=1)
    axis: str = Field(min_length=1)
    caliper_mm: Annotated[float, Field(gt=0)]
    system_mm: Annotated[float, Field(gt=0)]
    thickness_mm: Annotated[float, Field(ge=10, le=40)]
    session: str = Field(min_length=1)


class ScaleObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    session: str = Field(min_length=1)
    scale_mm_px: Annotated[float, Field(gt=0)]


class MeasurementResult(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    piece: str
    axis: str
    caliper_mm: float
    system_mm: float
    thickness_mm: float
    session: str
    signed_error_mm: float
    absolute_error_mm: float


class AggregateResult(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    count: int
    mean_signed_error_mm: float
    mean_absolute_error_mm: float
    max_absolute_error_mm: float


class DimensionalValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    schema_version: Literal["1.0.0"] = "1.0.0"
    generated_at: datetime
    acceptance_state: Literal["rejected", "automated-accepted", "physically-validated"]
    gate_g2: Literal["pending", "physically-validated"]
    criteria: dict[str, bool]
    test_records: dict[str, bool]
    overall: AggregateResult
    by_axis: dict[str, AggregateResult]
    by_thickness_mm: dict[str, AggregateResult]
    by_session: dict[str, AggregateResult]
    stability_between_sessions_mm: AggregateResult
    scale_variation_pct: float
    lidar_divergence_pct_by_session: dict[str, float]
    measurements: tuple[MeasurementResult, ...]
    calib_profile_id: str
    git_hash: str
    bench_config_hash: str
    environment: dict[str, str]
    sprint: Literal["S1"] = "S1"
    contractual_period: str


def environment_fingerprint() -> dict[str, str]:
    return {
        "petra": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
    }


def _aggregate(results: list[MeasurementResult]) -> AggregateResult:
    if not results:
        return AggregateResult(
            count=0,
            mean_signed_error_mm=0.0,
            mean_absolute_error_mm=0.0,
            max_absolute_error_mm=0.0,
        )
    return AggregateResult(
        count=len(results),
        mean_signed_error_mm=fmean(result.signed_error_mm for result in results),
        mean_absolute_error_mm=fmean(result.absolute_error_mm for result in results),
        max_absolute_error_mm=max(result.absolute_error_mm for result in results),
    )


def _grouped_aggregates(
    results: list[MeasurementResult], attribute: str
) -> dict[str, AggregateResult]:
    grouped: dict[str, list[MeasurementResult]] = defaultdict(list)
    for result in results:
        grouped[str(getattr(result, attribute))].append(result)
    return {key: _aggregate(values) for key, values in sorted(grouped.items())}


def _stability(results: list[MeasurementResult]) -> AggregateResult:
    grouped: dict[tuple[str, str], list[MeasurementResult]] = defaultdict(list)
    for result in results:
        grouped[(result.piece, result.axis)].append(result)
    differences: list[MeasurementResult] = []
    for (piece, axis), values in grouped.items():
        sessions = {value.session for value in values}
        if len(sessions) < 2:
            continue
        difference = max(value.system_mm for value in values) - min(
            value.system_mm for value in values
        )
        differences.append(
            MeasurementResult(
                piece=piece,
                axis=axis,
                caliper_mm=0.0,
                system_mm=0.0,
                thickness_mm=values[0].thickness_mm,
                session="cross-session",
                signed_error_mm=difference,
                absolute_error_mm=abs(difference),
            )
        )
    return _aggregate(differences)


def validate_dimensions(
    measurements: list[DimensionMeasurement],
    scales: list[ScaleObservation],
    *,
    profile: CalibProfile,
    session_meta: list[SessionMeta],
    git_hash: str,
    contractual_period: str,
    test_records: dict[str, bool],
    physical_evidence: bool,
    generated_at: datetime | None = None,
) -> DimensionalValidationReport:
    results = [
        MeasurementResult(
            **measurement.model_dump(),
            signed_error_mm=measurement.system_mm - measurement.caliper_mm,
            absolute_error_mm=abs(measurement.system_mm - measurement.caliper_mm),
        )
        for measurement in measurements
    ]
    sessions: dict[str, list[MeasurementResult]] = defaultdict(list)
    for result in results:
        sessions[result.session].append(result)

    session_protocol = len(sessions) >= 2
    for session_results in sessions.values():
        pieces: dict[str, set[str]] = defaultdict(set)
        for result in session_results:
            pieces[result.piece].add(result.axis)
        session_protocol = (
            session_protocol
            and len(pieces) >= 10
            and all(len(axes) >= 2 for axes in pieces.values())
        )

    thicknesses = {result.thickness_mm for result in results}
    representative_thickness = (
        bool(thicknesses) and min(thicknesses) <= 10 and max(thicknesses) >= 40
    )
    maximum_error = bool(results) and max(result.absolute_error_mm for result in results) <= 2.0
    scale_values = [scale.scale_mm_px for scale in scales]
    scale_variation_pct = (
        (max(scale_values) - min(scale_values)) / fmean(scale_values) * 100.0
        if scale_values
        else float("inf")
    )
    scale_stability = len({scale.session for scale in scales}) >= 5 and scale_variation_pct < 0.5
    criteria = {
        "ten_pieces_two_axes_two_sessions": session_protocol,
        "representative_thickness_10_40_mm": representative_thickness,
        "max_error_le_2_mm": maximum_error,
        "five_session_scale_variation_lt_0_5_pct": scale_stability,
    }
    automated_pass = all(criteria.values())
    required_tests = all(
        test_records.get(test_id, False) for test_id in ("TA-1", "TA-2", "TA-3", "TA-4")
    )
    physically_validated = automated_pass and physical_evidence and required_tests
    acceptance_state: Literal["rejected", "automated-accepted", "physically-validated"]
    if physically_validated:
        acceptance_state = "physically-validated"
    elif automated_pass:
        acceptance_state = "automated-accepted"
    else:
        acceptance_state = "rejected"

    lidar_by_session = {
        meta.session_id: meta.lidar_divergence_pct
        for meta in sorted(session_meta, key=lambda item: item.session_id)
    }
    return DimensionalValidationReport(
        generated_at=generated_at or datetime.now(UTC),
        acceptance_state=acceptance_state,
        gate_g2="physically-validated" if physically_validated else "pending",
        criteria=criteria,
        test_records=test_records,
        overall=_aggregate(results),
        by_axis=_grouped_aggregates(results, "axis"),
        by_thickness_mm=_grouped_aggregates(results, "thickness_mm"),
        by_session=_grouped_aggregates(results, "session"),
        stability_between_sessions_mm=_stability(results),
        scale_variation_pct=scale_variation_pct,
        lidar_divergence_pct_by_session=lidar_by_session,
        measurements=tuple(results),
        calib_profile_id=profile.id,
        git_hash=git_hash,
        bench_config_hash=profile.bench_config_hash,
        environment=environment_fingerprint(),
        contractual_period=contractual_period,
    )


def _read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"JSON input must contain a list: {path}")
        return payload
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_measurements(path: Path) -> list[DimensionMeasurement]:
    return [DimensionMeasurement.model_validate(record) for record in _read_records(path)]


def load_scales(path: Path) -> list[ScaleObservation]:
    return [ScaleObservation.model_validate(record) for record in _read_records(path)]


def write_dimensional_report(report: DimensionalValidationReport, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dimensional-report.json").write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "dimensional-measurements.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        fieldnames = list(MeasurementResult.model_fields)
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result.model_dump() for result in report.measurements)
    criteria_lines = "\n".join(
        f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in report.criteria.items()
    )
    markdown = (
        "# Relatório de validação dimensional\n\n"
        f"- Estado: `{report.acceptance_state}`\n"
        f"- G2: `{report.gate_g2}`\n"
        f"- Perfil: `{report.calib_profile_id}`\n"
        f"- Git: `{report.git_hash}`\n"
        f"- Erro absoluto médio: {report.overall.mean_absolute_error_mm:.3f} mm\n"
        f"- Erro absoluto máximo: {report.overall.max_absolute_error_mm:.3f} mm\n"
        f"- Variação de escala: {report.scale_variation_pct:.4f}%\n\n"
        "## Critérios\n\n"
        f"{criteria_lines}\n"
    )
    (output_dir / "dimensional-report.md").write_text(markdown, encoding="utf-8")
