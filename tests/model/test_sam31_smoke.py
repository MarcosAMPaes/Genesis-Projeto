from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.model


def test_sam31_smoke_remains_blocked_until_l5_and_upstream_loader_are_closed() -> None:
    runtime = json.loads(Path("config/models/sam3.1-runtime.json").read_text(encoding="utf-8"))
    if runtime["status"] == "blocked":
        pytest.skip("SAM 3.1 L5/loader blockers are recorded in runtime manifest")
    pytest.fail("replace this sentinel with the pinned Linux/CUDA smoke before enabling SAM 3.1")
