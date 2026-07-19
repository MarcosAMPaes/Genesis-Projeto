from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray
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
from petra.contracts import CalibProfile
from petra.errors import PetraError


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    return int(handler(args))
