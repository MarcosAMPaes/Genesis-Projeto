from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from petra.calibration.validation import (
    DimensionMeasurement,
    ScaleObservation,
    validate_dimensions,
    write_dimensional_report,
)
from petra.contracts import CalibProfile

pytestmark = pytest.mark.unit


def profile() -> CalibProfile:
    images = [{"path": f"pose-{index}.png", "sha256": f"{index:064x}"} for index in range(20)]
    residuals = [
        {
            "image_sha256": f"{index:064x}",
            "rms_px": 0.1,
            "rvec": [0.0, 0.0, 0.0],
            "tvec": [0.0, 0.0, 800.0],
        }
        for index in range(20)
    ]
    return CalibProfile.model_validate(
        {
            "id": "01KXY0FVMP1V3TZ9XDMXQQ6GMQ",
            "content_sha256": "f" * 64,
            "device": "device",
            "lens": "lens",
            "K": [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
            "dist": [0.0] * 5,
            "rms_px": 0.2,
            "img_size": [640, 480],
            "z_mm_lidar": 800.0,
            "created_at": "2026-07-19T15:00:00Z",
            "bench_config_hash": "e" * 64,
            "included_images": images,
            "pose_residuals": residuals,
        }
    )


def measurements() -> list[DimensionMeasurement]:
    values: list[DimensionMeasurement] = []
    thicknesses = (10.0, 20.0, 30.0, 40.0)
    for session_index, session in enumerate(("session-a", "session-b")):
        for piece_index in range(10):
            for axis_index, axis in enumerate(("x", "y")):
                caliper = 100.0 + piece_index * 2 + axis_index
                error = 0.4 + session_index * 0.1
                values.append(
                    DimensionMeasurement(
                        piece=f"piece-{piece_index:02d}",
                        axis=axis,
                        caliper_mm=caliper,
                        system_mm=caliper + error,
                        thickness_mm=thicknesses[piece_index % len(thicknesses)],
                        session=session,
                    )
                )
    return values


def scales() -> list[ScaleObservation]:
    return [
        ScaleObservation(session=f"scale-{index}", scale_mm_px=0.08 + index * 0.00005)
        for index in range(5)
    ]


def test_report_computes_g2_criteria_but_keeps_physical_gate_pending(tmp_path: Path) -> None:
    report = validate_dimensions(
        measurements(),
        scales(),
        profile=profile(),
        session_meta=[],
        git_hash="abc123",
        contractual_period="S1/2026-07",
        test_records={test_id: True for test_id in ("TA-1", "TA-2", "TA-3", "TA-4")},
        physical_evidence=False,
        generated_at=datetime(2026, 7, 19, 15, tzinfo=UTC),
    )
    assert report.acceptance_state == "automated-accepted"
    assert report.gate_g2 == "pending"
    assert report.overall.max_absolute_error_mm == pytest.approx(0.5)
    assert report.scale_variation_pct < 0.5
    assert report.stability_between_sessions_mm.max_absolute_error_mm == pytest.approx(0.1)

    write_dimensional_report(report, tmp_path)
    assert (tmp_path / "dimensional-report.json").exists()
    assert (tmp_path / "dimensional-report.md").exists()
    with (tmp_path / "dimensional-measurements.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 40
    payload = json.loads((tmp_path / "dimensional-report.json").read_text(encoding="utf-8"))
    assert payload["git_hash"] == "abc123"
    assert payload["environment"]["numpy"].startswith("2.")


def test_physical_state_requires_explicit_evidence_and_all_test_records() -> None:
    missing_ta4 = validate_dimensions(
        measurements(),
        scales(),
        profile=profile(),
        session_meta=[],
        git_hash="abc123",
        contractual_period="S1/2026-07",
        test_records={"TA-1": True, "TA-2": True, "TA-3": True},
        physical_evidence=True,
    )
    assert missing_ta4.acceptance_state == "automated-accepted"
    assert missing_ta4.gate_g2 == "pending"

    complete = validate_dimensions(
        measurements(),
        scales(),
        profile=profile(),
        session_meta=[],
        git_hash="abc123",
        contractual_period="S1/2026-07",
        test_records={test_id: True for test_id in ("TA-1", "TA-2", "TA-3", "TA-4")},
        physical_evidence=True,
    )
    assert complete.acceptance_state == "physically-validated"
    assert complete.gate_g2 == "physically-validated"


def test_error_or_scale_gate_failure_rejects_report() -> None:
    bad_measurements = measurements()
    bad_measurements[0] = bad_measurements[0].model_copy(
        update={"system_mm": bad_measurements[0].caliper_mm + 2.1}
    )
    report = validate_dimensions(
        bad_measurements,
        scales()[:4],
        profile=profile(),
        session_meta=[],
        git_hash="abc123",
        contractual_period="S1/2026-07",
        test_records={},
        physical_evidence=False,
    )
    assert report.acceptance_state == "rejected"
    assert report.criteria["max_error_le_2_mm"] is False
    assert report.criteria["five_session_scale_variation_lt_0_5_pct"] is False
