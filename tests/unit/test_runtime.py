from __future__ import annotations

import cv2
import numpy as np
import pytest

from petra.cli import main

pytestmark = pytest.mark.unit


def test_runtime_uses_numpy_2() -> None:
    assert int(np.__version__.split(".", maxsplit=1)[0]) >= 2


def test_opencv_main_exposes_charuco() -> None:
    assert hasattr(cv2, "aruco")
    assert hasattr(cv2.aruco, "CharucoBoard")
    assert hasattr(cv2.aruco, "CharucoDetector")


def test_cli_help_is_executable(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "Pipeline Petra Smart" in capsys.readouterr().out
