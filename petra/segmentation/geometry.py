from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.geometry.polygon import orient
from ulid import ULID

from petra.contracts import FragmentGeom, SessionMeta
from petra.errors import ErrorCode, PetraError
from petra.segmentation.contour import external_contour, simplify_contour
from petra.segmentation.postprocess import ProcessedMask


@dataclass(frozen=True, slots=True)
class GeometryExtraction:
    fragment: FragmentGeom
    raw_points: int
    simplified_points: int
    epsilon_px: float
    area_ratio: float
    hausdorff_mm: float
    repaired: bool


def _metric_points(
    points_px: np.ndarray,
    *,
    scale_mm_px: float,
    parallax_factor: float,
    x_origin_px: float,
    y_origin_px: float,
) -> np.ndarray:
    metric_scale = scale_mm_px * parallax_factor
    return np.column_stack(
        (
            (points_px[:, 0] - x_origin_px) * metric_scale,
            (y_origin_px - points_px[:, 1]) * metric_scale,
        )
    )


def repair_polygon(points_mm: np.ndarray) -> tuple[Polygon, bool]:
    polygon = Polygon(points_mm)
    if polygon.is_valid and polygon.exterior.is_simple:
        return polygon, False
    repaired = polygon.buffer(0)
    if isinstance(repaired, MultiPolygon):
        raise PetraError(
            ErrorCode.POLYGON_MULTIPOLYGON,
            "buffer(0) produced a MultiPolygon",
        )
    if (
        not isinstance(repaired, Polygon)
        or not repaired.is_valid
        or not repaired.exterior.is_simple
    ):
        raise PetraError(
            ErrorCode.POLYGON_INVALID,
            "polygon remains invalid after one buffer(0) repair",
        )
    return repaired, True


def extract_fragment_geometry(
    processed: ProcessedMask,
    session: SessionMeta,
    *,
    seg_model: str,
    seg_model_revision: str,
    seg_score: float,
    photo_path: str,
    mask_path: str,
    dp_epsilon_mm: float = 0.5,
    fragment_id: str | None = None,
) -> GeometryExtraction:
    raw_px = external_contour(processed.mask)
    metric_scale = session.output_gsd_mm_px * session.parallax_factor
    simplified_px, used_epsilon_px = simplify_contour(
        raw_px,
        dp_epsilon_mm=dp_epsilon_mm,
        scale_mm_px=metric_scale,
    )
    x_origin_px = float(np.min(raw_px[:, 0]))
    y_origin_px = float(np.max(raw_px[:, 1]))
    raw_mm = _metric_points(
        raw_px,
        scale_mm_px=session.output_gsd_mm_px,
        parallax_factor=session.parallax_factor,
        x_origin_px=x_origin_px,
        y_origin_px=y_origin_px,
    )
    simplified_mm = _metric_points(
        simplified_px,
        scale_mm_px=session.output_gsd_mm_px,
        parallax_factor=session.parallax_factor,
        x_origin_px=x_origin_px,
        y_origin_px=y_origin_px,
    )
    polygon, repaired = repair_polygon(simplified_mm)
    polygon = orient(polygon, sign=1.0)
    if polygon.is_empty or polygon.area <= 0:
        raise PetraError(ErrorCode.POLYGON_INVALID, "polygon has no positive area")

    area_ratio = float(polygon.area / processed.area_mm2)
    if not 0.99 <= area_ratio <= 1.01:
        raise PetraError(
            ErrorCode.POLYGON_AREA_RATIO,
            "polygon area differs from mask area by more than 1%",
            {"area_ratio": area_ratio},
        )
    exterior = list(polygon.exterior.coords)
    n_points = len(exterior) - 1
    if not 100 <= n_points <= 1000:
        raise PetraError(
            ErrorCode.POLYGON_VERTEX_COUNT,
            "simplified polygon must contain 100 to 1000 vertices",
            {"n_points": n_points},
        )
    hausdorff_mm = float(LineString(raw_mm).hausdorff_distance(LineString(exterior)))
    if hausdorff_mm > dp_epsilon_mm + 1e-9:
        raise PetraError(
            ErrorCode.POLYGON_HAUSDORFF,
            "simplified contour exceeds the physical epsilon",
            {"hausdorff_mm": hausdorff_mm, "dp_epsilon_mm": dp_epsilon_mm},
        )
    bbox = tuple(float(value) for value in polygon.bounds)
    fragment = FragmentGeom.model_validate(
        {
            "fragment_id": fragment_id or str(ULID()),
            "session_id": session.session_id,
            "polygon_mm": [[float(x), float(y)] for x, y in exterior],
            "area_mm2": float(polygon.area),
            "bbox_mm": bbox,
            "n_points": n_points,
            "seg_model": seg_model,
            "seg_model_revision": seg_model_revision,
            "seg_score": seg_score,
            "dp_epsilon_mm": dp_epsilon_mm,
            "photo_path": photo_path,
            "mask_path": mask_path,
            "coordinate_frame": "bottom_left_x_right_y_up_mm",
        }
    )
    return GeometryExtraction(
        fragment=fragment,
        raw_points=len(raw_px),
        simplified_points=n_points,
        epsilon_px=used_epsilon_px,
        area_ratio=area_ratio,
        hausdorff_mm=hausdorff_mm,
        repaired=repaired,
    )


def persist_fragment_geom(fragment: FragmentGeom, path: Path) -> None:
    payload = json.dumps(fragment.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
