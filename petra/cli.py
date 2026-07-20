from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

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
from petra.contracts import (
    AutoPrompt,
    BoxPrompt,
    CalibProfile,
    ConceptPrompt,
    PointsPrompt,
    PromptPoint,
    PromptSpec,
    SessionMeta,
)
from petra.errors import ErrorCode, PetraError
from petra.pipeline import process_session
from petra.segmentation.benchmark import (
    build_benchmark_report,
    file_sha256,
    load_candidate_inputs,
    write_benchmark_report,
)
from petra.segmentation.corpus import CorpusManifest
from petra.segmentation.factory import resolve_segmenter
from petra.segmentation.geometry import extract_fragment_geometry, persist_fragment_geom
from petra.segmentation.postprocess import postprocess_instances
from petra.segmentation.registry import ModelRegistry


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
        registry = ModelRegistry.from_json(Path(args.registry))
        selected = args.model or [entry.descriptor.name for entry in registry.document.models]
        results: dict[str, str] = {}
        for name in selected:
            try:
                registry.verify(name)
                results[name] = "verified"
            except PetraError as error:
                results[name] = str(error)
        print(json.dumps(results, indent=2, sort_keys=True))
        return 0 if all(value == "verified" for value in results.values()) else 4
    except (OSError, ValueError) as error:
        print(f"invalid input: {error}")
        return 2


def _segment_prompt(args: argparse.Namespace, *, default_concept: str | None = None) -> PromptSpec:
    supplied = sum(bool(value) for value in (args.point, args.box, args.concept))
    if supplied > 1:
        raise ValueError("use exactly one prompt family")
    if args.point:
        points: list[PromptPoint] = []
        for x_value, y_value, label_value in args.point:
            if label_value not in {0.0, 1.0}:
                raise ValueError("point label must be 0 or 1")
            points.append(
                PromptPoint(
                    point=(float(x_value), float(y_value)),
                    label=cast(Literal[0, 1], int(label_value)),
                )
            )
        return PointsPrompt(points=tuple(points))
    if args.box:
        return BoxPrompt(
            box=cast(tuple[float, float, float, float], tuple(float(v) for v in args.box))
        )
    if args.concept:
        return ConceptPrompt(concept=str(args.concept))
    if default_concept is not None:
        return ConceptPrompt(concept=default_concept)
    return AutoPrompt()


def _segment_run(args: argparse.Namespace) -> int:
    try:
        registry = ModelRegistry.from_json(Path(args.registry))
        entry = registry.entry(args.backend)
        session = SessionMeta.model_validate_json(
            Path(args.session_meta).read_text(encoding="utf-8")
        )
        resolved = resolve_segmenter(
            registry,
            args.backend,
            requested_device=args.device,
            background=session.background,
        )
        image_rgb = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.uint8)
        marker_mask = (
            np.asarray(Image.open(args.marker_mask).convert("L"), dtype=np.uint8)
            if args.marker_mask
            else None
        )
        default_concept = "stone fragment" if entry.descriptor.family == "sam3" else None
        predictions = resolved.segmenter.segment(
            image_rgb, _segment_prompt(args, default_concept=default_concept)
        )
        processed = postprocess_instances(
            [prediction.mask for prediction in predictions],
            scale_mm_px=session.output_gsd_mm_px,
            parallax_factor=session.parallax_factor,
            marker_mask=marker_mask,
        )
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        accepted: list[str] = []
        quality_warnings: dict[str, list[str]] = {}
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
                geometry_key = str(geometry_path)
                accepted.append(geometry_key)
                if extraction.fragment.quality_warnings:
                    quality_warnings[geometry_key] = list(extraction.fragment.quality_warnings)
                    print(
                        f"warning: {geometry_key}: "
                        f"{', '.join(extraction.fragment.quality_warnings)}"
                    )
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
            "device": resolved.device,
            "accepted": accepted,
            "quality_warnings": quality_warnings,
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


def _segment_benchmark(args: argparse.Namespace) -> int:
    try:
        registry = ModelRegistry.from_json(Path(args.registry))
        candidates = load_candidate_inputs(Path(args.input))
        corpus_path = Path(args.corpus_manifest)
        corpus = CorpusManifest.model_validate_json(corpus_path.read_text(encoding="utf-8"))
        verified_backends: set[str] = set()
        for candidate in candidates:
            try:
                registry.verify(candidate.backend)
                verified_backends.add(candidate.backend)
            except PetraError:
                pass
        report = build_benchmark_report(
            candidates,
            registry=registry,
            corpus=corpus,
            corpus_manifest_sha256=file_sha256(corpus_path),
            verified_backends=verified_backends,
            git_hash=args.git_hash,
            hardware=args.hardware,
        )
        write_benchmark_report(report, Path(args.output_dir))
        return 0 if report.d1.status == "production-selected" else 3
    except (OSError, ValueError) as error:
        print(f"invalid input: {error}")
        return 2
    except PetraError as error:
        print(error)
        return 3


def _process_session(args: argparse.Namespace) -> int:
    try:
        registry = ModelRegistry.from_json(Path(args.registry))
        family = registry.entry(args.backend).descriptor.family
        default_concept = "stone fragment" if family == "sam3" else None
        report, reused = process_session(
            image_path=Path(args.image),
            profile_path=Path(args.profile),
            board_path=Path(args.board),
            rectify_config_path=Path(args.config),
            registry_path=Path(args.registry),
            output_dir=Path(args.output_dir),
            session_id=args.session_id,
            thickness_mm=args.thickness_mm,
            reference_plane_height_mm=args.reference_plane_height_mm,
            background=args.background,
            observed_z_mm_lidar=args.z_mm_lidar,
            backend=args.backend,
            requested_device=args.device,
            prompt=_segment_prompt(args, default_concept=default_concept),
            git_hash=args.git_hash,
        )
        if reused:
            print("process-session: idempotent result reused")
        if report.lidar_warning:
            print(
                f"warning: {report.lidar_divergence_pct:.3f}% LiDAR divergence; "
                "recalibration review required"
            )
        for fragment_path, warnings in report.quality_warnings.items():
            print(f"warning: {fragment_path}: {', '.join(warnings)}")
        return 0 if report.acceptance_state == "automated-accepted" else 3
    except (OSError, ValueError) as error:
        print(f"invalid input: {error}")
        return 2
    except PetraError as error:
        print(error)
        return 3


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
    segment_run.add_argument(
        "--point",
        nargs=3,
        action="append",
        type=float,
        metavar=("X", "Y", "LABEL"),
        help="ponto em pixels e rotulo 0/1; repetivel",
    )
    segment_run.add_argument(
        "--box", nargs=4, type=float, metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX")
    )
    segment_run.add_argument("--concept", help="conceito textual para SAM 3.1")
    segment_run.add_argument("--marker-mask")
    segment_run.add_argument("--output-dir", required=True)
    segment_run.set_defaults(handler=_segment_run)
    segment_benchmark = segment_commands.add_parser(
        "benchmark", help="agrega o bake-off e aplica a regra D1"
    )
    segment_benchmark.add_argument("--input", required=True)
    segment_benchmark.add_argument("--corpus-manifest", default="data/validation/manifest.json")
    segment_benchmark.add_argument("--registry", default="config/models/registry.json")
    segment_benchmark.add_argument("--git-hash", required=True)
    segment_benchmark.add_argument("--hardware", required=True)
    segment_benchmark.add_argument("--output-dir", required=True)
    segment_benchmark.set_defaults(handler=_segment_benchmark)
    models = modules.add_parser("models", help="pesos e registro de modelos")
    model_commands = models.add_subparsers(dest="models_command")
    verify = model_commands.add_parser("verify", help="verifica pesos e licencas")
    verify.add_argument("--registry", default="config/models/registry.json")
    verify.add_argument("--model", action="append")
    verify.set_defaults(handler=_models_verify)
    process = modules.add_parser("process-session", help="executa o fluxo rastreavel A para B")
    process.add_argument("--image", required=True)
    process.add_argument("--profile", required=True)
    process.add_argument("--board", required=True)
    process.add_argument("--config", required=True)
    process.add_argument("--registry", default="config/models/registry.json")
    process.add_argument("--session-id", required=True)
    process.add_argument("--thickness-mm", required=True, type=float)
    process.add_argument("--reference-plane-height-mm", required=True, type=float)
    process.add_argument("--background", required=True)
    process.add_argument("--z-mm-lidar", type=float)
    process.add_argument("--backend", default="chroma")
    process.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    process.add_argument("--point", nargs=3, action="append", type=float)
    process.add_argument("--box", nargs=4, type=float)
    process.add_argument("--concept")
    process.add_argument("--git-hash", required=True)
    process.add_argument("--output-dir", required=True)
    process.set_defaults(handler=_process_session)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    return int(handler(args))
