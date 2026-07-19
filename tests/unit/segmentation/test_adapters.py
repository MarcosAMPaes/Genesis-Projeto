from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from petra.cli import main
from petra.contracts import (
    AutoPrompt,
    BoxPrompt,
    ConceptPrompt,
    ModelDescriptor,
    PointsPrompt,
    PromptPoint,
    SessionMeta,
)
from petra.errors import ErrorCode, PetraError
from petra.segmentation.adapters import (
    BiRefNetSegmenter,
    ChromaSegmenter,
    Sam2Segmenter,
    Sam31Segmenter,
)
from petra.segmentation.registry import DeviceResolver, ModelRegistry

pytestmark = pytest.mark.unit


def chroma_descriptor() -> ModelDescriptor:
    return ModelDescriptor(
        name="chroma",
        family="chroma",
        revision="opencv-5.0.0",
        weights_sha256=None,
        license_spdx="Apache-2.0",
        license_approved=True,
        supported_devices=("cpu",),
        precision="uint8",
    )


def session() -> SessionMeta:
    return SessionMeta.model_validate(
        {
            "session_id": "01KXY0GVMP1V3TZ9XDMXQQ6GMR",
            "calib_profile_id": "01KXY0FVMP1V3TZ9XDMXQQ6GMQ",
            "source_image": "raw.png",
            "rectified_image": "rectified.png",
            "undistorted": True,
            "scale_mm_px": 0.1,
            "homography": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "aruco_ids": [1, 2, 3, 4],
            "thickness_mm": 20.0,
            "background": "verde-fosco",
            "residual_check_mm": 0.1,
            "native_gsd_mm_px": 0.1,
            "output_gsd_mm_px": 0.1,
            "resample_ratio": 1.0,
            "rectified_img_size": [2200, 2200],
            "roi_mm": [0.0, 0.0, 220.0, 220.0],
            "reference_plane_height_mm": 0.0,
            "parallax_factor": 0.975,
            "lidar_divergence_pct": 0.2,
            "coordinate_frame": "bottom_left_x_right_y_up_mm",
            "interpolator": "linear",
        }
    )


def test_chroma_separates_multiple_foreground_instances_offline() -> None:
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    image[:, :] = [0, 150, 0]
    image[20:90, 20:100] = [120, 120, 120]
    image[110:190, 180:280] = [80, 80, 80]
    predictions = ChromaSegmenter(
        chroma_descriptor(), background="verde-fosco", min_component_px=100
    ).segment(image, AutoPrompt())
    assert len(predictions) == 2
    assert np.count_nonzero(predictions[0].mask) == 80 * 100
    assert np.count_nonzero(predictions[1].mask) == 70 * 80
    with pytest.raises(ValueError, match="auto"):
        ChromaSegmenter(chroma_descriptor(), background="verde-fosco").segment(
            image, BoxPrompt(box=(0.0, 0.0, 10.0, 10.0))
        )


def test_registry_and_device_resolver_never_fallback_silently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = ModelRegistry.from_json(Path("config/models/registry.json"))
    registry.verify("chroma")
    results = registry.verify_all()
    assert results["chroma"] == "verified"
    assert "WEIGHTS_MISSING" in results["birefnet-general"]
    assert (
        main(
            [
                "models",
                "verify",
                "--registry",
                "config/models/registry.json",
                "--model",
                "chroma",
            ]
        )
        == 0
    )

    missing_registry = tmp_path / "registry.json"
    missing_registry.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "models": [
                    {
                        "descriptor": {
                            "name": "sam2-test",
                            "family": "sam2",
                            "revision": "test",
                            "weights_sha256": "a" * 64,
                            "license_spdx": "Apache-2.0",
                            "license_approved": True,
                            "supported_devices": ["cpu", "cuda"],
                            "precision": "float32",
                        },
                        "weights_path": "missing.pt",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(PetraError) as missing:
        ModelRegistry.from_json(missing_registry).verify("sam2-test")
    assert missing.value.code == ErrorCode.WEIGHTS_MISSING

    monkeypatch.setattr(DeviceResolver, "available_devices", staticmethod(lambda: ("cpu",)))
    with pytest.raises(PetraError) as unavailable:
        DeviceResolver.resolve("cuda", ("cpu", "cuda"))
    assert unavailable.value.code == ErrorCode.MODEL_UNAVAILABLE
    assert DeviceResolver.resolve("auto", ("cpu", "cuda")) == "cpu"


def test_birefnet_adapter_is_local_and_numpy2_compatible_at_contract_boundary() -> None:
    class FakeRuntime:
        def predict(self, image_rgb: np.ndarray) -> tuple[np.ndarray, float]:
            return np.asarray(image_rgb[:, :, 0] > 100, dtype=np.bool_), 0.97

    descriptor = (
        ModelRegistry.from_json(Path("config/models/registry.json"))
        .entry("birefnet-general")
        .descriptor
    )
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[8:24, 8:24, 0] = 255
    prediction = BiRefNetSegmenter(descriptor, FakeRuntime()).segment(image, AutoPrompt())[0]
    assert np.__version__.startswith("2.")
    assert prediction.score == pytest.approx(0.97)
    assert np.count_nonzero(prediction.mask) == 16 * 16


def test_sam2_adapter_supports_point_and_box_without_silent_prompt_conversion() -> None:
    class FakeRuntime:
        def __init__(self) -> None:
            self.prompts: list[PointsPrompt | BoxPrompt] = []

        def predict(
            self, image_rgb: np.ndarray, prompt: PointsPrompt | BoxPrompt
        ) -> tuple[np.ndarray, float]:
            self.prompts.append(prompt)
            return np.ones(image_rgb.shape[:2], dtype=np.bool_), 0.96

    descriptor = (
        ModelRegistry.from_json(Path("config/models/registry.json"))
        .entry("sam2.1-hiera-small")
        .descriptor
    )
    runtime = FakeRuntime()
    segmenter = Sam2Segmenter(descriptor, runtime)
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    point_prompt = PointsPrompt(points=(PromptPoint(point=(8.0, 9.0), label=1),))
    box_prompt = BoxPrompt(box=(2.0, 3.0, 20.0, 21.0))

    assert segmenter.segment(image, point_prompt)[0].mask.shape == (24, 32)
    assert segmenter.segment(image, box_prompt)[0].score == pytest.approx(0.96)
    assert runtime.prompts == [point_prompt, box_prompt]
    with pytest.raises(ValueError, match="points or box"):
        segmenter.segment(image, ConceptPrompt(concept="stone fragment"))


def test_sam31_is_fail_closed_to_linux_cuda_concept_and_pending_license() -> None:
    class FakeRuntime:
        def predict(self, image_rgb: np.ndarray, concept: str) -> list[tuple[np.ndarray, float]]:
            assert concept == "stone fragment"
            mask = np.zeros(image_rgb.shape[:2], dtype=np.bool_)
            mask[3:12, 4:16] = True
            return [(mask, 0.95)]

    registry = ModelRegistry.from_json(Path("config/models/registry.json"))
    descriptor = registry.entry("sam3.1-multiplex").descriptor
    with pytest.raises(PetraError) as license_error:
        registry.verify("sam3.1-multiplex")
    assert license_error.value.code == ErrorCode.LICENSE_NOT_APPROVED
    with pytest.raises(PetraError, match="Linux and CUDA"):
        Sam31Segmenter(descriptor, FakeRuntime(), device="cpu", platform_name="Windows")

    segmenter = Sam31Segmenter(descriptor, FakeRuntime(), device="cuda", platform_name="Linux")
    image = np.zeros((20, 24, 3), dtype=np.uint8)
    prediction = segmenter.segment(image, ConceptPrompt(concept="stone fragment"))[0]
    assert prediction.score == pytest.approx(0.95)
    with pytest.raises(ValueError, match="concept"):
        segmenter.segment(image, AutoPrompt())

    runtime_config = json.loads(Path("config/models/sam3.1-runtime.json").read_text())
    assert runtime_config["status"] == "blocked"
    assert runtime_config["code_revision"] is None
    assert runtime_config["container_image_digest"] is None


def test_segment_run_cli_emits_metric_geometry_with_chroma(tmp_path: Path) -> None:
    image = np.zeros((2200, 2200, 3), dtype=np.uint8)
    image[:, :] = [0, 150, 0]
    points: list[list[int]] = []
    for index in range(360):
        angle = 2 * math.pi * index / 360
        radius = 820 if index % 2 == 0 else 790
        points.append(
            [
                round(1100 + radius * math.cos(angle)),
                round(1100 + radius * math.sin(angle)),
            ]
        )
    cv2.fillPoly(image, [np.asarray(points, dtype=np.int32)], (110, 110, 110))
    image_path = tmp_path / "rectified.png"
    Image.fromarray(image).save(image_path)
    session_path = tmp_path / "session_meta.json"
    session_path.write_text(session().model_dump_json(indent=2), encoding="utf-8")
    output_dir = tmp_path / "output"

    exit_code = main(
        [
            "segment",
            "run",
            "--image",
            str(image_path),
            "--session-meta",
            str(session_path),
            "--backend",
            "chroma",
            "--registry",
            "config/models/registry.json",
            "--output-dir",
            str(output_dir),
        ]
    )
    assert exit_code == 0
    report = json.loads((output_dir / "segmentation-run.json").read_text(encoding="utf-8"))
    assert report["device"] == "cpu"
    assert len(report["accepted"]) == 1
    assert report["rejected"] == []
    assert Path(report["accepted"][0]).exists()
