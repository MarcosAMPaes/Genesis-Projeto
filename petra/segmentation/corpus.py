from __future__ import annotations

import hashlib
import re
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator

from petra.contracts.base import Sha256

LFS_POINTER_PATTERN = re.compile(
    rb"^version https://git-lfs.github.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize ([0-9]+)\n?$"
)


class AssetRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1)
    sha256: Sha256
    size_bytes: Annotated[int, Field(gt=0)]

    @model_validator(mode="after")
    def validate_relative_path(self) -> AssetRef:
        path = PurePosixPath(self.path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("asset path must be relative and remain inside data/validation")
        return self


class SampleAttributes(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rock: str = Field(min_length=1)
    color: str = Field(min_length=1)
    veins: str = Field(min_length=1)
    finish: str = Field(min_length=1)
    background: str = Field(min_length=1)
    thickness_mm: Annotated[float, Field(gt=0)]
    reflection: str = Field(min_length=1)
    translucency: str = Field(min_length=1)


class CorpusSample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_id: str = Field(min_length=1)
    split: Literal["development", "evaluation"]
    storage_backend: Literal["lfs", "s3"]
    origin: str | None = None
    image: AssetRef
    mask: AssetRef
    attributes: SampleAttributes

    @model_validator(mode="after")
    def validate_origin(self) -> CorpusSample:
        if self.storage_backend == "s3" and not self.origin:
            raise ValueError("S3 samples require an origin")
        return self


class CorpusManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0.0"] = "1.0.0"
    status: Literal["draft", "frozen"]
    lfs_quota_bytes: Annotated[int, Field(gt=0)] | None
    lfs_budget_bytes: Annotated[int, Field(gt=0)] | None
    samples: tuple[CorpusSample, ...]

    @model_validator(mode="after")
    def validate_budget(self) -> CorpusManifest:
        if (self.lfs_quota_bytes is None) != (self.lfs_budget_bytes is None):
            raise ValueError("LFS quota and budget must be declared together")
        if self.lfs_quota_bytes is not None and self.lfs_budget_bytes is not None:
            expected = int(self.lfs_quota_bytes * 0.8)
            if self.lfs_budget_bytes != expected:
                raise ValueError("lfs_budget_bytes must equal 80% of lfs_quota_bytes")
        if any(sample.storage_backend == "lfs" for sample in self.samples):
            if self.lfs_budget_bytes is None:
                raise ValueError("LFS assets are blocked until quota budget is declared")
            total = sum(
                sample.image.size_bytes + sample.mask.size_bytes
                for sample in self.samples
                if sample.storage_backend == "lfs"
            )
            if total > self.lfs_budget_bytes:
                raise ValueError("LFS corpus exceeds the declared budget")
        return self


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_asset(root: Path, asset: AssetRef, backend: str) -> tuple[bool, str | None]:
    path = root / PurePosixPath(asset.path)
    if not path.is_file():
        if backend == "s3":
            return False, None
        return False, f"missing LFS pointer or asset: {asset.path}"
    content = path.read_bytes()
    pointer = LFS_POINTER_PATTERN.fullmatch(content)
    if pointer is not None:
        pointer_hash = pointer.group(1).decode("ascii")
        pointer_size = int(pointer.group(2))
        if pointer_hash != asset.sha256 or pointer_size != asset.size_bytes:
            return False, f"LFS pointer metadata mismatch: {asset.path}"
        return False, None
    if len(content) != asset.size_bytes or _sha256(path) != asset.sha256:
        return True, f"asset hash or size mismatch: {asset.path}"
    return True, None


def lint_corpus(manifest_path: Path, *, require_frozen: bool = False) -> list[str]:
    manifest = CorpusManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if require_frozen and manifest.status != "frozen":
        errors.append("release corpus must be frozen")
    evaluation_count = sum(sample.split == "evaluation" for sample in manifest.samples)
    if require_frozen and evaluation_count < 30:
        errors.append("frozen evaluation split must contain at least 30 samples")
    ids = [sample.sample_id for sample in manifest.samples]
    if len(ids) != len(set(ids)):
        errors.append("sample_id values must be unique")
    hashes_by_split: dict[str, set[str]] = {"development": set(), "evaluation": set()}
    root = manifest_path.parent
    for sample in manifest.samples:
        if sample.image.sha256 in hashes_by_split["development"] | hashes_by_split["evaluation"]:
            errors.append(f"duplicate image hash across corpus: {sample.sample_id}")
        hashes_by_split[sample.split].add(sample.image.sha256)
        image_local, image_error = _validate_asset(root, sample.image, sample.storage_backend)
        mask_local, mask_error = _validate_asset(root, sample.mask, sample.storage_backend)
        errors.extend(error for error in (image_error, mask_error) if error is not None)
        if image_local and mask_local:
            with Image.open(root / sample.image.path) as image:
                image_size = image.size
            with Image.open(root / sample.mask.path) as mask:
                mask_array = np.asarray(mask.convert("L"), dtype=np.uint8)
                mask_size = mask.size
            values = set(int(value) for value in np.unique(mask_array))
            if image_size != mask_size:
                errors.append(f"image/mask dimensions differ: {sample.sample_id}")
            if not values <= {0, 255}:
                errors.append(f"mask is not binary: {sample.sample_id}")
            if 255 not in values:
                errors.append(f"mask is empty: {sample.sample_id}")
    return errors
