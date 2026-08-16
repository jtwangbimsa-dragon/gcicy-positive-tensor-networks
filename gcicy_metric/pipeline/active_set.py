"""Portable active-point pools for tail-aware H-metric refinement."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .adapter import GCICYAdapter


ACTIVE_POINT_POOL_SCHEMA_VERSION = 1


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ActivePointPool:
    path: Path
    points: list[Any]
    center_ids: np.ndarray
    radii: np.ndarray
    source_center_log_ratios: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        count = len(self.points)
        for name, values in (
            ("center_ids", self.center_ids),
            ("radii", self.radii),
            ("source_center_log_ratios", self.source_center_log_ratios),
        ):
            if np.asarray(values).shape != (count,):
                raise ValueError(f"active point {name} must have one entry per point")


def save_active_point_pool(
    path: Path,
    adapter: GCICYAdapter,
    points: Sequence[Any],
    *,
    model_seed: int,
    exact_model: bool,
    center_ids: Sequence[int],
    radii: Sequence[float],
    source_center_log_ratios: Sequence[float],
    metadata: dict[str, Any],
) -> str:
    output = path.expanduser().resolve()
    if not points:
        raise ValueError("active point pool cannot be empty")
    count = len(points)
    labels = {
        "center_ids": np.asarray(center_ids, dtype=np.int64),
        "radii": np.asarray(radii, dtype=np.float64),
        "source_center_log_ratios": np.asarray(
            source_center_log_ratios,
            dtype=np.float64,
        ),
    }
    if any(values.shape != (count,) for values in labels.values()):
        raise ValueError("active point labels must have one entry per point")
    if not np.all(np.isfinite(labels["radii"])) or np.min(labels["radii"]) < 0:
        raise ValueError("active point radii must be finite and non-negative")
    if not np.all(np.isfinite(labels["source_center_log_ratios"])):
        raise ValueError("active point source scores must be finite")
    point_payload = adapter.point_storage_payload(points)
    arrays: dict[str, np.ndarray] = {
        "active_point_pool_schema_version": np.asarray(
            ACTIVE_POINT_POOL_SCHEMA_VERSION,
            dtype=np.int64,
        ),
        "pipeline_adapter": np.asarray(adapter.key),
        "model_seed": np.asarray(model_seed, dtype=np.int64),
        "exact_model": np.asarray(exact_model),
        "point_count": np.asarray(count, dtype=np.int64),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        **labels,
    }
    for key, values in point_payload.items():
        if not key or key.startswith("point_"):
            raise ValueError("adapter point-payload keys must be unprefixed names")
        array = np.asarray(values)
        if len(array) != count:
            raise ValueError(f"point payload {key} has the wrong leading dimension")
        arrays[f"point_{key}"] = array
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(output)
    return file_sha256(output)


def load_active_point_pool(
    path: Path,
    adapter: GCICYAdapter,
    model: Any,
    *,
    expected_model_seed: int,
    expected_exact_model: bool,
) -> ActivePointPool:
    source = path.expanduser().resolve()
    with np.load(source, allow_pickle=False) as payload:
        schema_version = int(payload["active_point_pool_schema_version"])
        if schema_version != ACTIVE_POINT_POOL_SCHEMA_VERSION:
            raise ValueError("active point pool schema version is not supported")
        if str(payload["pipeline_adapter"]) != adapter.key:
            raise ValueError("active point pool adapter does not match training adapter")
        if int(payload["model_seed"]) != expected_model_seed:
            raise ValueError("active point pool model seed does not match training model")
        if bool(payload["exact_model"]) != expected_exact_model:
            raise ValueError("active point pool exact-model flag does not match")
        count = int(payload["point_count"])
        point_payload = {
            key.removeprefix("point_"): np.asarray(payload[key])
            for key in payload.files
            if key.startswith("point_") and key != "point_count"
        }
        points = adapter.points_from_storage_payload(model, point_payload)
        if len(points) != count:
            raise ValueError("active point pool reconstructed the wrong point count")
        center_ids = np.asarray(payload["center_ids"], dtype=np.int64)
        radii = np.asarray(payload["radii"], dtype=np.float64)
        source_scores = np.asarray(
            payload["source_center_log_ratios"],
            dtype=np.float64,
        )
        metadata = json.loads(str(payload["metadata_json"]))
    return ActivePointPool(
        path=source,
        points=points,
        center_ids=center_ids,
        radii=radii,
        source_center_log_ratios=source_scores,
        metadata=metadata,
    )
