from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Literal, cast

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from ulid import ULID

from petra.calibration.parallax import check_lidar_divergence, parallax_factor
from petra.calibration.profile import sha256_file, verify_profile_hash
from petra.calibration.rectify import (
    CharucoBoardConfig,
    RectificationResult,
    RectifyConfig,
    build_session_meta,
    detect_charuco,
    rectify_image,
)
from petra.calibration.undistort import Undistorter
from petra.calibration.validation import environment_fingerprint
from petra.contracts import CalibProfile, GeometryQualityWarning, PromptSpec
from petra.errors import ErrorCode, PetraError
from petra.segmentation.factory import resolve_segmenter
from petra.segmentation.geometry import extract_fragment_geometry, persist_fragment_geom
from petra.segmentation.postprocess import postprocess_instances
from petra.segmentation.registry import ModelRegistry


class PipelineRejection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    instance_index: int
    code: ErrorCode
    message: str
    details: dict[str, object]


class ProcessSessionReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0.0"] = "1.0.0"
    acceptance_state: Literal["rejected", "automated-accepted"]
    ts1_status: Literal["partial"] = "partial"
    run_fingerprint: str
    session_id: str
    calib_profile_id: str
    backend: str
    backend_revision: str
    device: Literal["cpu", "mps", "cuda"]
    git_hash: str
    environment: dict[str, str]
    lidar_divergence_pct: float
    lidar_warning: bool
    fragments: tuple[str, ...]
    quality_warnings: dict[str, tuple[GeometryQualityWarning, ...]] = Field(
        default_factory=dict
    )
    rejections: tuple[PipelineRejection, ...]
    outputs_sha256: dict[str, str]
    sprint: Literal["S2"] = "S2"


def _canonical_fingerprint(paths: dict[str, Path], values: dict[str, object]) -> str:
    payload = {
        "files": {name: sha256_file(path) for name, path in sorted(paths.items())},
        "values": values,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _marker_mask(
    undistorted_bgr: NDArray[np.uint8],
    board_config: CharucoBoardConfig,
    result: RectificationResult,
) -> NDArray[np.bool_]:
    board = board_config.create_board()
    detector = cv2.aruco.ArucoDetector(board.getDictionary())
    corners, _ids, _rejected = detector.detectMarkers(undistorted_bgr)
    source_mask = np.zeros(undistorted_bgr.shape[:2], dtype=np.uint8)
    for corner in corners:
        polygon = np.rint(np.asarray(corner).reshape(-1, 2)).astype(np.int32)
        cv2.fillConvexPoly(source_mask, polygon, 255)
    rectified = cv2.warpPerspective(
        source_mask,
        result.pixel_to_raster,
        result.rectified_img_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return np.asarray(rectified > 0, dtype=np.bool_)


def _deterministic_fragment_id(
    session_id: str, instance_index: int, mask: NDArray[np.bool_]
) -> str:
    session_bytes = bytes(ULID.from_str(session_id))
    digest = hashlib.sha256()
    digest.update(session_id.encode("ascii"))
    digest.update(instance_index.to_bytes(4, "big", signed=False))
    digest.update(mask.tobytes())
    return str(ULID.from_bytes(session_bytes[:6] + digest.digest()[:10]))


def _verify_existing_outputs(output_dir: Path, report: ProcessSessionReport) -> None:
    for relative_path, expected_hash in report.outputs_sha256.items():
        path = output_dir / relative_path
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise PetraError(
                ErrorCode.SESSION_OUTPUT_CONFLICT,
                "existing process-session output is incomplete or modified",
                {"path": str(path)},
            )


def process_session(
    *,
    image_path: Path,
    profile_path: Path,
    board_path: Path,
    rectify_config_path: Path,
    registry_path: Path,
    output_dir: Path,
    session_id: str,
    thickness_mm: float,
    reference_plane_height_mm: float,
    background: str,
    observed_z_mm_lidar: float | None,
    backend: str,
    requested_device: Literal["auto", "cpu", "mps", "cuda"],
    prompt: PromptSpec,
    git_hash: str,
) -> tuple[ProcessSessionReport, bool]:
    fingerprint = _canonical_fingerprint(
        {
            "image": image_path,
            "profile": profile_path,
            "board": board_path,
            "rectify_config": rectify_config_path,
            "registry": registry_path,
        },
        {
            "session_id": session_id,
            "thickness_mm": thickness_mm,
            "reference_plane_height_mm": reference_plane_height_mm,
            "background": background,
            "observed_z_mm_lidar": observed_z_mm_lidar,
            "backend": backend,
            "requested_device": requested_device,
            "prompt": prompt.model_dump(mode="json"),
            "git_hash": git_hash,
        },
    )
    report_path = output_dir / "process-session.json"
    if output_dir.exists():
        if not report_path.is_file():
            raise PetraError(
                ErrorCode.SESSION_OUTPUT_CONFLICT,
                "output directory already exists without a process-session report",
                {"output_dir": str(output_dir)},
            )
        existing = ProcessSessionReport.model_validate_json(report_path.read_text(encoding="utf-8"))
        if existing.run_fingerprint != fingerprint:
            raise PetraError(
                ErrorCode.SESSION_OUTPUT_CONFLICT,
                "output directory belongs to different process-session inputs",
                {"output_dir": str(output_dir)},
            )
        _verify_existing_outputs(output_dir, existing)
        return existing, True

    profile = CalibProfile.model_validate_json(profile_path.read_text(encoding="utf-8"))
    verify_profile_hash(profile)
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"could not read image: {image_path}")
    source = cast(NDArray[np.uint8], image_bgr)
    undistorted = Undistorter().apply(source, profile)
    board_config = CharucoBoardConfig.from_json(board_path)
    rectify_config = RectifyConfig.from_json(rectify_config_path)
    detection = detect_charuco(undistorted.image_bgr, board_config)
    rectified = rectify_image(undistorted.image_bgr, detection, rectify_config)
    marker_mask = _marker_mask(undistorted.image_bgr, board_config, rectified)
    observed_z = observed_z_mm_lidar if observed_z_mm_lidar is not None else profile.z_mm_lidar
    lidar = check_lidar_divergence(profile.z_mm_lidar, observed_z)
    factor = parallax_factor(
        profile.z_mm_lidar,
        thickness_mm,
        reference_plane_height_mm,
    )
    registry = ModelRegistry.from_json(registry_path)
    resolved = resolve_segmenter(
        registry,
        backend,
        requested_device=requested_device,
        background=background,
    )
    descriptor = registry.entry(backend).descriptor

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}.", dir=output_dir.parent) as tmp:
        stage = Path(tmp)
        rectified_relative = Path("rectified.png")
        session_relative = Path("session_meta.json")
        marker_relative = Path("marker-mask.png")
        final_rectified = output_dir / rectified_relative
        if not cv2.imwrite(str(stage / rectified_relative), rectified.image_bgr):
            raise OSError("could not write rectified image")
        Image.fromarray(marker_mask.astype(np.uint8) * 255).save(stage / marker_relative)
        session = build_session_meta(
            rectified,
            session_id=session_id,
            calib_profile_id=profile.id,
            source_image=str(image_path),
            rectified_image=str(final_rectified),
            thickness_mm=thickness_mm,
            background=background,
            reference_plane_height_mm=reference_plane_height_mm,
            parallax_factor=factor,
            lidar_divergence_pct=lidar.divergence_pct,
        )
        (stage / session_relative).write_text(
            json.dumps(session.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        image_rgb = cast(NDArray[np.uint8], cv2.cvtColor(rectified.image_bgr, cv2.COLOR_BGR2RGB))
        predictions = resolved.segmenter.segment(image_rgb, prompt)
        processed = postprocess_instances(
            [prediction.mask for prediction in predictions],
            scale_mm_px=session.output_gsd_mm_px,
            parallax_factor=session.parallax_factor,
            marker_mask=marker_mask,
        )
        fragments: list[str] = []
        quality_warnings: dict[str, tuple[GeometryQualityWarning, ...]] = {}
        rejections = [
            PipelineRejection(
                instance_index=item.instance_index,
                code=item.code,
                message=item.message,
                details=item.details,
            )
            for item in processed.rejected
        ]
        for item in processed.accepted:
            fragment_id = _deterministic_fragment_id(session_id, item.instance_index, item.mask)
            mask_relative = Path("fragments") / f"{fragment_id}.mask.png"
            geometry_relative = Path("fragments") / f"{fragment_id}.fragment_geom.json"
            (stage / mask_relative).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(item.mask.astype(np.uint8) * 255).save(stage / mask_relative)
            try:
                extraction = extract_fragment_geometry(
                    item,
                    session,
                    seg_model=descriptor.name,
                    seg_model_revision=descriptor.revision,
                    seg_score=predictions[item.instance_index].score,
                    photo_path=str(final_rectified),
                    mask_path=str(output_dir / mask_relative),
                    fragment_id=fragment_id,
                )
                persist_fragment_geom(extraction.fragment, stage / geometry_relative)
                geometry_key = geometry_relative.as_posix()
                fragments.append(geometry_key)
                if extraction.fragment.quality_warnings:
                    quality_warnings[geometry_key] = extraction.fragment.quality_warnings
            except PetraError as error:
                rejections.append(
                    PipelineRejection(
                        instance_index=item.instance_index,
                        code=error.code,
                        message=error.message,
                        details=error.details or {},
                    )
                )
        output_files = sorted(path for path in stage.rglob("*") if path.is_file())
        output_hashes = {
            path.relative_to(stage).as_posix(): sha256_file(path) for path in output_files
        }
        report = ProcessSessionReport(
            acceptance_state="automated-accepted" if fragments else "rejected",
            run_fingerprint=fingerprint,
            session_id=session.session_id,
            calib_profile_id=profile.id,
            backend=descriptor.name,
            backend_revision=descriptor.revision,
            device=resolved.device,
            git_hash=git_hash,
            environment=environment_fingerprint(),
            lidar_divergence_pct=lidar.divergence_pct,
            lidar_warning=lidar.warning,
            fragments=tuple(fragments),
            quality_warnings=quality_warnings,
            rejections=tuple(rejections),
            outputs_sha256=output_hashes,
        )
        (stage / "process-session.json").write_text(
            report.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        os.replace(stage, output_dir)
    return report, False
