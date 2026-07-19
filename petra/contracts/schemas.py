from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import TypeAdapter

from petra.contracts.calibration import CalibProfile, SessionMeta
from petra.contracts.segmentation import FragmentGeom, ModelDescriptor, PromptSpec

SchemaFactory = Callable[[], dict[str, Any]]


def _model_schema(
    model: type[CalibProfile | SessionMeta | FragmentGeom | ModelDescriptor],
) -> SchemaFactory:
    return lambda: model.model_json_schema(mode="serialization")


SCHEMA_FACTORIES: dict[str, SchemaFactory] = {
    "calib_profile.schema.json": _model_schema(CalibProfile),
    "fragment_geom.schema.json": _model_schema(FragmentGeom),
    "model_descriptor.schema.json": _model_schema(ModelDescriptor),
    "prompt_spec.schema.json": lambda: TypeAdapter(PromptSpec).json_schema(mode="serialization"),
    "session_meta.schema.json": _model_schema(SessionMeta),
}
