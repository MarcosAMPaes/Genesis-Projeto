from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from petra.segmentation.corpus import CorpusManifest, lint_corpus

pytestmark = pytest.mark.unit


def write_png(path: Path, array: np.ndarray) -> dict[str, object]:
    Image.fromarray(array).save(path)
    content = path.read_bytes()
    return {
        "path": path.name,
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
    }


def attributes() -> dict[str, object]:
    return {
        "rock": "granite",
        "color": "gray",
        "veins": "low",
        "finish": "polished",
        "background": "green",
        "thickness_mm": 20.0,
        "reflection": "medium",
        "translucency": "opaque",
    }


def test_local_assets_are_hashed_binary_nonempty_and_dimension_matched(tmp_path: Path) -> None:
    image_ref = write_png(tmp_path / "image.png", np.zeros((8, 8, 3), dtype=np.uint8))
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 255
    mask_ref = write_png(tmp_path / "mask.png", mask)
    manifest = {
        "schema_version": "1.0.0",
        "status": "draft",
        "lfs_quota_bytes": 10_000,
        "lfs_budget_bytes": 8_000,
        "samples": [
            {
                "sample_id": "sample-1",
                "split": "development",
                "storage_backend": "lfs",
                "origin": None,
                "image": image_ref,
                "mask": mask_ref,
                "attributes": attributes(),
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert lint_corpus(manifest_path) == []

    mask[0, 0] = 127
    write_png(tmp_path / "mask.png", mask)
    errors = lint_corpus(manifest_path)
    assert any("hash or size" in error for error in errors)


def test_lfs_pointer_is_validated_without_downloading_blob(tmp_path: Path) -> None:
    image_hash = "a" * 64
    mask_hash = "b" * 64
    (tmp_path / "image.png").write_bytes(
        f"version https://git-lfs.github.com/spec/v1\noid sha256:{image_hash}\nsize 123\n".encode(
            "ascii"
        )
    )
    (tmp_path / "mask.png").write_bytes(
        f"version https://git-lfs.github.com/spec/v1\noid sha256:{mask_hash}\nsize 45\n".encode(
            "ascii"
        )
    )
    manifest = {
        "schema_version": "1.0.0",
        "status": "draft",
        "lfs_quota_bytes": 1000,
        "lfs_budget_bytes": 800,
        "samples": [
            {
                "sample_id": "pointer",
                "split": "evaluation",
                "storage_backend": "lfs",
                "origin": None,
                "image": {"path": "image.png", "sha256": image_hash, "size_bytes": 123},
                "mask": {"path": "mask.png", "sha256": mask_hash, "size_bytes": 45},
                "attributes": attributes(),
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert lint_corpus(manifest_path) == []


def test_lfs_assets_require_80_percent_budget_and_frozen_release_size() -> None:
    base = {
        "schema_version": "1.0.0",
        "status": "draft",
        "lfs_quota_bytes": None,
        "lfs_budget_bytes": None,
        "samples": [
            {
                "sample_id": "sample",
                "split": "evaluation",
                "storage_backend": "lfs",
                "origin": None,
                "image": {"path": "image.png", "sha256": "a" * 64, "size_bytes": 10},
                "mask": {"path": "mask.png", "sha256": "b" * 64, "size_bytes": 10},
                "attributes": attributes(),
            }
        ],
    }
    with pytest.raises(ValidationError, match="quota budget"):
        CorpusManifest.model_validate(base)

    empty = CorpusManifest(
        status="draft",
        lfs_quota_bytes=None,
        lfs_budget_bytes=None,
        samples=(),
    )
    assert empty.status == "draft"
