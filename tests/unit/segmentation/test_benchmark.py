from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from petra.cli import main
from petra.segmentation.benchmark import (
    BenchmarkObservation,
    CandidateBenchmarkInput,
    build_benchmark_report,
)
from petra.segmentation.corpus import CorpusManifest
from petra.segmentation.metrics import measure_masks
from petra.segmentation.registry import ModelRegistry

pytestmark = pytest.mark.unit


def corpus(*, frozen: bool = True) -> CorpusManifest:
    samples = []
    for index in range(30):
        samples.append(
            {
                "sample_id": f"sample-{index:02d}",
                "split": "evaluation",
                "storage_backend": "s3",
                "origin": f"s3://petra-validation/sample-{index:02d}",
                "image": {
                    "path": f"images/sample-{index:02d}.png",
                    "sha256": f"{index + 1:064x}",
                    "size_bytes": 100,
                },
                "mask": {
                    "path": f"masks/sample-{index:02d}.png",
                    "sha256": f"{index + 101:064x}",
                    "size_bytes": 50,
                },
                "attributes": {
                    "rock": "granite" if index < 15 else "marble",
                    "color": "light",
                    "veins": "low",
                    "finish": "polished",
                    "background": "verde-fosco",
                    "thickness_mm": 20,
                    "reflection": "low",
                    "translucency": "none",
                },
            }
        )
    return CorpusManifest.model_validate(
        {
            "status": "frozen" if frozen else "draft",
            "lfs_quota_bytes": None,
            "lfs_budget_bytes": None,
            "samples": samples,
        }
    )


def candidate(
    backend: str,
    role: str,
    *,
    iou: float = 0.96,
    hausdorff_mm: float = 1.5,
    cost: int = 1,
) -> CandidateBenchmarkInput:
    union = 10_000
    intersection = round(iou * union)
    observations = [
        BenchmarkObservation(
            sample_id=f"sample-{index:02d}",
            class_name="granite" if index < 15 else "marble",
            intersection_px=intersection,
            union_px=union,
            iou=intersection / union,
            hausdorff_mm=hausdorff_mm,
            intervention_required=role == "promptable",
            prompts_used=1 if role == "promptable" else 0,
        )
        for index in range(30)
    ]
    return CandidateBenchmarkInput.model_validate(
        {
            "backend": backend,
            "role": role,
            "device": "cpu" if backend == "chroma" else "cuda",
            "local_cost_rank": cost,
            "observations": observations,
            "latency_ms": list(range(10, 30)),
            "peak_memory_mb": 128,
        }
    )


def test_mask_metrics_use_metric_symmetric_hausdorff() -> None:
    truth = np.zeros((50, 50), dtype=np.bool_)
    prediction = np.zeros_like(truth)
    truth[10:30, 10:30] = True
    prediction[10:30, 11:31] = True
    metrics = measure_masks(prediction, truth, scale_mm_px=0.5)
    assert metrics.intersection_px == 380
    assert metrics.union_px == 420
    assert metrics.iou == pytest.approx(380 / 420)
    assert metrics.hausdorff_mm == pytest.approx(0.5)
    with pytest.raises(ValueError, match="matching"):
        measure_masks(prediction, truth[:20], scale_mm_px=0.5)


def test_d1_selects_lowest_cost_qualified_automatic_and_promptable_separately() -> None:
    registry = ModelRegistry.from_json(Path("config/models/registry.json"))
    report = build_benchmark_report(
        [
            candidate("birefnet-general", "automatic", cost=2),
            candidate("chroma", "automatic", cost=0),
            candidate("sam2.1-hiera-small", "promptable", cost=1),
            candidate("sam3.1-multiplex", "promptable", cost=0),
        ],
        registry=registry,
        corpus=corpus(),
        corpus_manifest_sha256="a" * 64,
        verified_backends={"birefnet-general", "chroma", "sam2.1-hiera-small", "sam3.1-multiplex"},
        git_hash="abc123",
        hardware="test-host",
        generated_at=datetime(2026, 7, 19, tzinfo=UTC),
    )
    assert report.d1.status == "production-selected"
    assert report.d1.automatic_backend == "chroma"
    assert report.d1.promptable_backend == "sam2.1-hiera-small"
    sam3 = next(item for item in report.candidates if item.backend == "sam3.1-multiplex")
    assert sam3.elimination_reasons == ("license_not_approved",)
    assert report.candidates[0].median_latency_ms_20 == pytest.approx(19.5)


def test_d1_remains_open_on_threshold_draft_corpus_or_unverified_weights() -> None:
    registry = ModelRegistry.from_json(Path("config/models/registry.json"))
    report = build_benchmark_report(
        [candidate("birefnet-general", "automatic", iou=0.95, hausdorff_mm=2.01)],
        registry=registry,
        corpus=corpus(frozen=False),
        corpus_manifest_sha256="b" * 64,
        verified_backends=set(),
        git_hash="abc123",
        hardware="test-host",
    )
    assert report.d1.status == "open"
    assert report.acceptance_state == "automated-accepted"
    assert set(report.candidates[0].elimination_reasons) == {
        "corpus_not_frozen",
        "worst_class_iou_le_0_95",
        "hausdorff_gt_2_mm",
        "weights_not_verified",
    }


def test_benchmark_cli_writes_open_report_for_draft_corpus(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidates.json"
    candidate_path.write_text(
        json.dumps([candidate("chroma", "automatic").model_dump(mode="json")]),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(corpus(frozen=False).model_dump_json(), encoding="utf-8")
    output_dir = tmp_path / "report"
    exit_code = main(
        [
            "segment",
            "benchmark",
            "--input",
            str(candidate_path),
            "--corpus-manifest",
            str(manifest_path),
            "--git-hash",
            "abc123",
            "--hardware",
            "test-host",
            "--output-dir",
            str(output_dir),
        ]
    )
    assert exit_code == 3
    payload = json.loads((output_dir / "segmentation-benchmark.json").read_text(encoding="utf-8"))
    assert payload["d1"]["status"] == "open"
