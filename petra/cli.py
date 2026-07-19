from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from ulid import ULID

from petra import __version__
from petra.calibration.parallax import check_lidar_divergence, parallax_factor
from petra.calibration.profile import create_calibration_artifacts, verify_profile_hash
from petra.calibration.rectify import (
    CharucoBoardConfig,
    RectifyConfig,
    build_session_meta,
    detect_charuco,
    rectify_image,
)
from petra.calibration.undistort import Undistorter
from petra.calibration.validation import (
    load_measurements,
    load_scales,
    validate_dimensions,
    write_dimensional_report,
)
from petra.contracts import AutoPrompt, CalibProfile, SessionMeta
from petra.errors import ErrorCode, PetraError
from petra.segmentation.adapters import ChromaSegmenter
from petra.segmentation.geometry import extract_fragment_geometry, persist_fragment_geom
from petra.segmentation.postprocess import postprocess_instances
from petra.segmentation.registry import DeviceResolver, ModelRegistry


def _calibrate_create(args: argparse.Namespace) -> int:
    try:
        accepted = create_calibration_artifacts(
            image_dir=Path(args.images),
            board_path=Path(args.board),
            device=args.device,
            lens=args.lens,
            z_mm_lidar=args.z_mm_lidar,
            bench_config_path=Path(args.bench_config),
            profile_path=Path(args.output),
            report_path=Path(args.report),
            excluded_names=set(args.exclude),
        )
    except (OSError, ValueError) as error:
        print(f"invalid input: {error}")
        return 2
    except PetraError as error:
        print(error)
        return 3
    return 0 if accepted else 3


def _calibrate_rectify(args: argparse.Namespace) -> int:
    try:
        profile = CalibProfile.model_validate_json(Path(args.profile).read_text(encoding="utf-8"))
        verify_profile_hash(profile)
        image = cv2.imread(args.image, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"could not read image: {args.image}")
        undistorted = Undistorter().apply(cast(NDArray[np.uint8], image), profile)
        board = CharucoBoardConfig.from_json(Path(args.board))
        config = RectifyConfig.from_json(Path(args.config))
        detection = detect_charuco(undistorted.image_bgr, board)
        result = rectify_image(undistorted.image_bgr, detection, config)
        observed_z = args.z_mm_lidar if args.z_mm_lidar is not None else profile.z_mm_lidar
        lidar = check_lidar_divergence(profile.z_mm_lidar, observed_z)
        factor = parallax_factor(
            profile.z_mm_lidar,
            args.thickness_mm,
            args.reference_plane_height_mm,
        )
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), result.image_bgr):
            print(f"runtime error: could not write image: {output_path}")
            return 4
        session = build_session_meta(
            result,
            session_id=args.session_id or str(ULID()),
            calib_profile_id=profile.id,
            source_image=args.image,
            rectified_image=str(output_path),
            thickness_mm=args.thickness_mm,
            background=args.background,
            reference_plane_height_mm=args.reference_plane_height_mm,
            parallax_factor=factor,
            lidar_divergence_pct=lidar.divergence_pct,
        )
        meta_path = Path(args.meta)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(
            json.dumps(session.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if lidar.warning:
            print(
                f"warning: {lidar.divergence_pct:.3f}% LiDAR divergence; "
                "recalibration review required"
            )
        return 0
    except (OSError, ValueError) as error:
        print(f"invalid input: {error}")
        return 2
    except PetraError as error:
        print(error)
        return 3


def _calibrate_validate_dims(args: argparse.Namespace) -> int:
    try:
        profile = CalibProfile.model_validate_json(Path(args.profile).read_text(encoding="utf-8"))
        verify_profile_hash(profile)
        session_meta = [
            SessionMeta.model_validate_json(Path(path).read_text(encoding="utf-8"))
            for path in args.session_meta
        ]
        test_records = (
            json.loads(Path(args.test_records).read_text(encoding="utf-8"))
            if args.test_records
            else {}
        )
        if not isinstance(test_records, dict):
            raise ValueError("test records must be a JSON object")
        report = validate_dimensions(
            load_measurements(Path(args.input)),
            load_scales(Path(args.scales)),
            profile=profile,
            session_meta=session_meta,
            git_hash=args.git_hash,
            contractual_period=args.contractual_period,
            test_records={str(key): bool(value) for key, value in test_records.items()},
            physical_evidence=args.physical_evidence,
        )
        write_dimensional_report(report, Path(args.output_dir))
        return 0 if report.acceptance_state != "rejected" else 3
    except (OSError, ValueError) as error:
        print(f"invalid input: {error}")
        return 2
    except PetraError as error:
        print(error)
        return 3


def _models_verify(args: argparse.Namespace) -> int:
    try:
        results = ModelRegistry.from_json(Path(args.registry)).verify_all()
        print(json.dumps(results, indent=2, sort_keys=True))
        return 0 if all(value == "verified" for value in results.values()) else 4
    except (OSError, ValueError) as error:
        print(f"invalid input: {error}")
        return 2


def _segment_run(args: argparse.Namespace) -> int:
    try:
        registry = ModelRegistry.from_json(Path(args.registry))
        registry.verify(args.backend)
        entry = registry.entry(args.backend)
        device = DeviceResolver.resolve(args.device, entry.descriptor.supported_devices)
        if entry.descriptor.family != "chroma":
            raise PetraError(
                ErrorCode.MODEL_UNAVAILABLE,
                f"adapter is not installed in this increment: {args.backend}",
            )
        session = SessionMeta.model_validate_json(
            Path(args.session_meta).read_text(encoding="utf-8")
        )
        image_rgb = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.uint8)
        marker_mask = (
            np.asarray(Image.open(args.marker_mask).convert("L"), dtype=np.uint8)
            if args.marker_mask
            else None
        )
        segmenter = ChromaSegmenter(entry.descriptor, background=session.background)
        predictions = segmenter.segment(image_rgb, AutoPrompt())
        processed = postprocess_instances(
            [prediction.mask for prediction in predictions],
            scale_mm_px=session.output_gsd_mm_px,
            parallax_factor=session.parallax_factor,
            marker_mask=marker_mask,
        )
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        accepted: list[str] = []
        geometry_rejections: list[dict[str, object]] = []
        scores = {index: prediction.score for index, prediction in enumerate(predictions)}
        for item in processed.accepted:
            fragment_id = str(ULID())
            mask_path = output_dir / f"{fragment_id}.mask.png"
            Image.fromarray(item.mask.astype(np.uint8) * 255).save(mask_path)
            try:
                extraction = extract_fragment_geometry(
                    item,
                    session,
                    seg_model=entry.descriptor.name,
                    seg_model_revision=entry.descriptor.revision,
                    seg_score=scores[item.instance_index],
                    photo_path=args.image,
                    mask_path=str(mask_path),
                    fragment_id=fragment_id,
                )
                geometry_path = output_dir / f"{fragment_id}.fragment_geom.json"
                persist_fragment_geom(extraction.fragment, geometry_path)
                accepted.append(str(geometry_path))
            except PetraError as error:
                geometry_rejections.append(
                    {
                        "instance_index": item.instance_index,
                        "code": error.code,
                        "message": error.message,
                    }
                )
        rejections = [
            {
                "instance_index": item.instance_index,
                "code": item.code,
                "message": item.message,
                "details": item.details,
            }
            for item in processed.rejected
        ] + geometry_rejections
        run_report = {
            "schema_version": "1.0.0",
            "backend": entry.descriptor.model_dump(mode="json"),
            "device": device,
            "accepted": accepted,
            "rejected": rejections,
        }
        (output_dir / "segmentation-run.json").write_text(
            json.dumps(run_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return 0 if accepted else 3
    except (OSError, ValueError) as error:
        print(f"invalid input: {error}")
        return 2
    except PetraError as error:
        print(error)
        return (
            4
            if error.code
            in {
                ErrorCode.WEIGHTS_MISSING,
                ErrorCode.MODEL_UNAVAILABLE,
                ErrorCode.LICENSE_NOT_APPROVED,
            }
            else 3
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="petra", description="Pipeline Petra Smart")
    parser.add_argument("--version", action="version", version=__version__)
    modules = parser.add_subparsers(dest="module")
    calibrate = modules.add_parser("calibrate", help="calibracao geometrica")
    calibrate_commands = calibrate.add_subparsers(dest="calibrate_command")
    create = calibrate_commands.add_parser("create", help="cria perfil intrinseco")
    create.add_argument("--images", required=True)
    create.add_argument("--board", required=True)
    create.add_argument("--device", required=True)
    create.add_argument("--lens", required=True)
    create.add_argument("--z-mm-lidar", required=True, type=float)
    create.add_argument("--bench-config", required=True)
    create.add_argument("--output", required=True)
    create.add_argument("--report", required=True)
    create.add_argument("--exclude", action="append", default=[])
    create.set_defaults(handler=_calibrate_create)
    rectify = calibrate_commands.add_parser("rectify", help="retifica uma captura")
    rectify.add_argument("--image", required=True)
    rectify.add_argument("--profile", required=True)
    rectify.add_argument("--board", required=True)
    rectify.add_argument("--config", required=True)
    rectify.add_argument("--session-id")
    rectify.add_argument("--thickness-mm", required=True, type=float)
    rectify.add_argument("--reference-plane-height-mm", required=True, type=float)
    rectify.add_argument("--background", required=True)
    rectify.add_argument("--z-mm-lidar", type=float)
    rectify.add_argument("--output", required=True)
    rectify.add_argument("--meta", required=True)
    rectify.set_defaults(handler=_calibrate_rectify)
    validate_dims = calibrate_commands.add_parser(
        "validate-dims", help="valida medidas contra paquimetro"
    )
    validate_dims.add_argument("--input", required=True)
    validate_dims.add_argument("--scales", required=True)
    validate_dims.add_argument("--profile", required=True)
    validate_dims.add_argument("--session-meta", action="append", default=[])
    validate_dims.add_argument("--test-records")
    validate_dims.add_argument("--git-hash", required=True)
    validate_dims.add_argument("--contractual-period", required=True)
    validate_dims.add_argument("--physical-evidence", action="store_true")
    validate_dims.add_argument("--output-dir", required=True)
    validate_dims.set_defaults(handler=_calibrate_validate_dims)
    segment = modules.add_parser("segment", help="segmentacao e geometria")
    segment_commands = segment.add_subparsers(dest="segment_command")
    segment_run = segment_commands.add_parser("run", help="segmenta uma imagem retificada")
    segment_run.add_argument("--image", required=True)
    segment_run.add_argument("--session-meta", required=True)
    segment_run.add_argument("--backend", default="chroma")
    segment_run.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    segment_run.add_argument("--registry", default="config/models/registry.json")
    segment_run.add_argument("--marker-mask")
    segment_run.add_argument("--output-dir", required=True)
    segment_run.set_defaults(handler=_segment_run)
    models = modules.add_parser("models", help="pesos e registro de modelos")
    model_commands = models.add_subparsers(dest="models_command")
    verify = model_commands.add_parser("verify", help="verifica pesos e licencas")
    verify.add_argument("--registry", default="config/models/registry.json")
    verify.set_defaults(handler=_models_verify)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    return int(handler(args))
