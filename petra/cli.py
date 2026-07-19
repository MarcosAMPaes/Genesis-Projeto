from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from petra import __version__
from petra.calibration.profile import create_calibration_artifacts
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    return int(handler(args))
