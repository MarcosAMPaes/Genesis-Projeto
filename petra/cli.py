from __future__ import annotations

import argparse
from collections.abc import Sequence

from petra import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="petra", description="Pipeline Petra Smart")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_subparsers(dest="module")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.module is None:
        parser.print_help()
    return 0
