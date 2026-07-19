from __future__ import annotations

import os
from pathlib import Path

import pytest

from petra.cli import main

pytestmark = pytest.mark.benchmark


def test_frozen_corpus_bakeoff_closes_d1_only_with_qualified_backends(tmp_path: Path) -> None:
    input_path = os.environ.get("PETRA_BENCHMARK_INPUT")
    git_hash = os.environ.get("PETRA_BENCHMARK_GIT_HASH")
    hardware = os.environ.get("PETRA_BENCHMARK_HARDWARE")
    if not input_path or not git_hash or not hardware:
        pytest.skip("PETRA_BENCHMARK_INPUT/GIT_HASH/HARDWARE are required")
    exit_code = main(
        [
            "segment",
            "benchmark",
            "--input",
            input_path,
            "--git-hash",
            git_hash,
            "--hardware",
            hardware,
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert exit_code == 0, "D1 remains open; inspect segmentation-benchmark.json"
