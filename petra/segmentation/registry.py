from __future__ import annotations

import hashlib
import importlib
import importlib.util
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from petra.contracts import ModelDescriptor
from petra.errors import ErrorCode, PetraError


class ModelRegistryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    descriptor: ModelDescriptor
    weights_path: str | None = None


class ModelRegistryDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0.0"] = "1.0.0"
    models: tuple[ModelRegistryEntry, ...] = Field(min_length=1)


class ModelRegistry:
    def __init__(self, document: ModelRegistryDocument, *, base_dir: Path) -> None:
        self.document = document
        self.base_dir = base_dir
        names = [entry.descriptor.name for entry in document.models]
        if len(names) != len(set(names)):
            raise ValueError("model registry names must be unique")
        self._entries = {entry.descriptor.name: entry for entry in document.models}

    @classmethod
    def from_json(cls, path: Path) -> ModelRegistry:
        document = ModelRegistryDocument.model_validate_json(path.read_text(encoding="utf-8"))
        return cls(document, base_dir=path.parent)

    def entry(self, name: str) -> ModelRegistryEntry:
        try:
            return self._entries[name]
        except KeyError as error:
            raise PetraError(
                ErrorCode.MODEL_UNAVAILABLE,
                f"model is not registered: {name}",
            ) from error

    def verify(self, name: str) -> None:
        entry = self.entry(name)
        descriptor = entry.descriptor
        if not descriptor.license_approved:
            raise PetraError(
                ErrorCode.LICENSE_NOT_APPROVED,
                f"model license is not approved: {name}",
            )
        if descriptor.weights_sha256 is None:
            if descriptor.family == "chroma":
                return
            raise PetraError(ErrorCode.WEIGHTS_MISSING, f"weights hash is absent: {name}")
        if entry.weights_path is None:
            raise PetraError(ErrorCode.WEIGHTS_MISSING, f"weights path is absent: {name}")
        weights = (self.base_dir / entry.weights_path).resolve()
        if not weights.is_file():
            raise PetraError(
                ErrorCode.WEIGHTS_MISSING,
                f"weights file is missing: {weights}",
            )
        digest = hashlib.sha256(weights.read_bytes()).hexdigest()
        if digest != descriptor.weights_sha256:
            raise PetraError(
                ErrorCode.WEIGHTS_MISSING,
                f"weights hash mismatch: {name}",
                {"expected": descriptor.weights_sha256, "actual": digest},
            )

    def verify_all(self) -> dict[str, str]:
        results: dict[str, str] = {}
        for name in sorted(self._entries):
            try:
                self.verify(name)
                results[name] = "verified"
            except PetraError as error:
                results[name] = str(error)
        return results


class DeviceResolver:
    @staticmethod
    def available_devices() -> tuple[str, ...]:
        devices = ["cpu"]
        if importlib.util.find_spec("torch") is None:
            return tuple(devices)
        torch = importlib.import_module("torch")

        if torch.backends.mps.is_available():
            devices.append("mps")
        if torch.cuda.is_available():
            devices.append("cuda")
        return tuple(devices)

    @classmethod
    def resolve(
        cls,
        requested: Literal["auto", "cpu", "mps", "cuda"],
        supported: tuple[Literal["cpu", "mps", "cuda"], ...],
    ) -> Literal["cpu", "mps", "cuda"]:
        available = cls.available_devices()
        if requested != "auto":
            if requested not in supported or requested not in available:
                raise PetraError(
                    ErrorCode.MODEL_UNAVAILABLE,
                    f"requested device is unavailable: {requested}",
                    {"available": available, "supported": supported},
                )
            return requested
        for candidate in ("cuda", "mps", "cpu"):
            if candidate in available and candidate in supported:
                return candidate
        raise PetraError(
            ErrorCode.MODEL_UNAVAILABLE,
            "no supported execution device is available",
            {"available": available, "supported": supported},
        )
