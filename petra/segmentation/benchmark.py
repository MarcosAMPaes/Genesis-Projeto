from __future__ import annotations

import hashlib
import json
import platform
import statistics
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from petra import __version__
from petra.segmentation.corpus import CorpusManifest
from petra.segmentation.registry import ModelRegistry


class BenchmarkObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    sample_id: str = Field(min_length=1)
    class_name: str = Field(min_length=1)
    intersection_px: Annotated[int, Field(ge=0)]
    union_px: Annotated[int, Field(gt=0)]
    iou: Annotated[float, Field(ge=0, le=1)]
    hausdorff_mm: Annotated[float, Field(ge=0)]
    intervention_required: bool
    prompts_used: Annotated[int, Field(ge=0, le=3)]

    @model_validator(mode="after")
    def validate_iou(self) -> BenchmarkObservation:
        expected = self.intersection_px / self.union_px
        if not np.isclose(self.iou, expected, rtol=0, atol=1e-9):
            raise ValueError("iou must equal intersection_px / union_px")
        return self


class CandidateBenchmarkInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    backend: str = Field(min_length=1)
    role: Literal["automatic", "promptable"]
    device: str = Field(min_length=1)
    local_cost_rank: Annotated[int, Field(ge=0)]
    observations: tuple[BenchmarkObservation, ...] = Field(min_length=1)
    latency_ms: tuple[Annotated[float, Field(gt=0)], ...] = Field(min_length=20)
    peak_memory_mb: Annotated[float, Field(ge=0)]

    @model_validator(mode="after")
    def validate_samples(self) -> CandidateBenchmarkInput:
        ids = [observation.sample_id for observation in self.observations]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate observations must have unique sample_id values")
        return self


class CandidateBenchmarkResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    backend: str
    family: str
    role: Literal["automatic", "promptable"]
    revision: str
    weights_sha256: str | None
    license_spdx: str
    license_approved: bool
    device: str
    local_cost_rank: int
    sample_count: int
    global_iou: float
    iou_by_class: dict[str, float]
    worst_class: str
    worst_class_iou: float
    max_hausdorff_mm: float
    intervention_free_rate: float
    prompts_used: int
    median_latency_ms_20: float
    peak_memory_mb: float
    qualified: bool
    elimination_reasons: tuple[str, ...]


class D1Decision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["open", "production-selected"]
    automatic_backend: str | None
    promptable_backend: str | None
    reasons: tuple[str, ...]


class SegmentationBenchmarkReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: Literal["1.0.0"] = "1.0.0"
    generated_at: datetime
    acceptance_state: Literal["automated-accepted", "production-selected"]
    d1: D1Decision
    corpus_status: Literal["draft", "frozen"]
    corpus_manifest_sha256: str
    git_hash: str
    hardware: str
    environment: dict[str, str]
    candidates: tuple[CandidateBenchmarkResult, ...]
    sprint: Literal["S2"] = "S2"
    tests: tuple[Literal["TB-1", "TB-2", "TB-5"], ...] = ("TB-1", "TB-2", "TB-5")


def _environment() -> dict[str, str]:
    return {
        "petra": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _aggregate_candidate(
    candidate: CandidateBenchmarkInput,
    registry: ModelRegistry,
    *,
    expected_sample_ids: set[str],
    corpus_frozen: bool,
    weights_verified: bool,
) -> CandidateBenchmarkResult:
    descriptor = registry.entry(candidate.backend).descriptor
    intersections: dict[str, int] = defaultdict(int)
    unions: dict[str, int] = defaultdict(int)
    for observation in candidate.observations:
        intersections[observation.class_name] += observation.intersection_px
        unions[observation.class_name] += observation.union_px
    iou_by_class = {name: intersections[name] / unions[name] for name in sorted(intersections)}
    worst_class = min(iou_by_class, key=lambda name: (iou_by_class[name], name))
    worst_class_iou = iou_by_class[worst_class]
    total_intersection = sum(intersections.values())
    total_union = sum(unions.values())
    sample_ids = {observation.sample_id for observation in candidate.observations}
    reasons: list[str] = []
    if not corpus_frozen:
        reasons.append("corpus_not_frozen")
    if sample_ids != expected_sample_ids or len(sample_ids) < 30:
        reasons.append("evaluation_split_incomplete")
    if worst_class_iou <= 0.95:
        reasons.append("worst_class_iou_le_0_95")
    max_hausdorff = max(observation.hausdorff_mm for observation in candidate.observations)
    if max_hausdorff > 2.0:
        reasons.append("hausdorff_gt_2_mm")
    if not descriptor.license_approved:
        reasons.append("license_not_approved")
    if not weights_verified:
        reasons.append("weights_not_verified")
    return CandidateBenchmarkResult(
        backend=candidate.backend,
        family=descriptor.family,
        role=candidate.role,
        revision=descriptor.revision,
        weights_sha256=descriptor.weights_sha256,
        license_spdx=descriptor.license_spdx,
        license_approved=descriptor.license_approved,
        device=candidate.device,
        local_cost_rank=candidate.local_cost_rank,
        sample_count=len(candidate.observations),
        global_iou=total_intersection / total_union,
        iou_by_class=iou_by_class,
        worst_class=worst_class,
        worst_class_iou=worst_class_iou,
        max_hausdorff_mm=max_hausdorff,
        intervention_free_rate=sum(
            not observation.intervention_required for observation in candidate.observations
        )
        / len(candidate.observations),
        prompts_used=sum(observation.prompts_used for observation in candidate.observations),
        median_latency_ms_20=statistics.median(candidate.latency_ms[:20]),
        peak_memory_mb=candidate.peak_memory_mb,
        qualified=not reasons,
        elimination_reasons=tuple(reasons),
    )


def build_benchmark_report(
    candidates: list[CandidateBenchmarkInput],
    *,
    registry: ModelRegistry,
    corpus: CorpusManifest,
    corpus_manifest_sha256: str,
    verified_backends: set[str],
    git_hash: str,
    hardware: str,
    generated_at: datetime | None = None,
) -> SegmentationBenchmarkReport:
    evaluation_ids = {sample.sample_id for sample in corpus.samples if sample.split == "evaluation"}
    results = tuple(
        _aggregate_candidate(
            candidate,
            registry,
            expected_sample_ids=evaluation_ids,
            corpus_frozen=corpus.status == "frozen",
            weights_verified=candidate.backend in verified_backends,
        )
        for candidate in candidates
    )
    qualified_by_role: dict[str, list[CandidateBenchmarkResult]] = defaultdict(list)
    for result in results:
        if result.qualified:
            qualified_by_role[result.role].append(result)

    def choose(role: str) -> str | None:
        choices = qualified_by_role[role]
        if not choices:
            return None
        return min(
            choices,
            key=lambda result: (
                result.local_cost_rank,
                result.median_latency_ms_20,
                result.backend,
            ),
        ).backend

    automatic = choose("automatic")
    promptable = choose("promptable")
    reasons: list[str] = []
    if automatic is None:
        reasons.append("no_qualified_automatic_backend")
    if promptable is None:
        reasons.append("no_qualified_promptable_backend")
    status: Literal["open", "production-selected"] = (
        "production-selected" if not reasons else "open"
    )
    return SegmentationBenchmarkReport(
        generated_at=generated_at or datetime.now(UTC),
        acceptance_state=(
            "production-selected" if status == "production-selected" else "automated-accepted"
        ),
        d1=D1Decision(
            status=status,
            automatic_backend=automatic,
            promptable_backend=promptable,
            reasons=tuple(reasons),
        ),
        corpus_status=corpus.status,
        corpus_manifest_sha256=corpus_manifest_sha256,
        git_hash=git_hash,
        hardware=hardware,
        environment=_environment(),
        candidates=results,
    )


def load_candidate_inputs(path: Path) -> list[CandidateBenchmarkInput]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("benchmark input must be a non-empty JSON list")
    return [CandidateBenchmarkInput.model_validate(item) for item in payload]


def write_benchmark_report(report: SegmentationBenchmarkReport, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "segmentation-benchmark.json").write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = "\n".join(
        f"| {item.backend} | {item.role} | {item.worst_class_iou:.4f} | "
        f"{item.max_hausdorff_mm:.3f} | {'PASS' if item.qualified else 'FAIL'} |"
        for item in report.candidates
    )
    markdown = (
        "# Bake-off de segmentacao D1\n\n"
        f"- Estado: `{report.d1.status}`\n"
        f"- Automatico: `{report.d1.automatic_backend}`\n"
        f"- Promptable: `{report.d1.promptable_backend}`\n"
        f"- Corpus: `{report.corpus_status}` / `{report.corpus_manifest_sha256}`\n"
        f"- Git: `{report.git_hash}`\n"
        f"- Hardware: `{report.hardware}`\n\n"
        "| Backend | Papel | IoU pior classe | Hausdorff max (mm) | Qualificado |\n"
        "|---|---|---:|---:|---|\n"
        f"{rows}\n"
    )
    (output_dir / "segmentation-benchmark.md").write_text(markdown, encoding="utf-8")
