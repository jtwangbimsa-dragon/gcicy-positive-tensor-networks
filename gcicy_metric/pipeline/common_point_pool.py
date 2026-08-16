"""Immutable common point pools for matched gCICY metric comparisons."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .adapter import GCICYAdapter


COMMON_POINT_POOL_SCHEMA_VERSION = 1


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


@dataclass(frozen=True)
class CommonPointPool:
    path: Path
    points: list[Any]
    importance_log_weights: np.ndarray
    importance_weights: np.ndarray
    sampling_cluster_ids: np.ndarray
    baseline_metrics: np.ndarray
    holomorphic_volume_log_density: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        count = len(self.points)
        for name, values in (
            ("importance_log_weights", self.importance_log_weights),
            ("importance_weights", self.importance_weights),
            ("sampling_cluster_ids", self.sampling_cluster_ids),
            ("baseline_metrics", self.baseline_metrics),
            (
                "holomorphic_volume_log_density",
                self.holomorphic_volume_log_density,
            ),
        ):
            if len(np.asarray(values)) != count:
                raise ValueError(f"common point pool {name} has the wrong length")


def common_point_arrays(
    adapter: GCICYAdapter,
    points: Sequence[Any],
) -> dict[str, np.ndarray]:
    if not points:
        raise ValueError("common point pool cannot be empty")
    count = len(points)
    log_weights = np.asarray(
        [adapter.importance_log_weight(point) for point in points],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(log_weights)):
        raise FloatingPointError("common point pool has non-finite importance weights")
    weights = np.exp(log_weights - float(np.max(log_weights)))
    weights /= float(np.mean(weights))
    baseline_metrics = adapter.baseline_metrics(points)
    baseline_eigenvalues = np.linalg.eigvalsh(baseline_metrics)
    if np.any(baseline_eigenvalues[:, 0] <= 0):
        raise FloatingPointError("common point pool baseline metric is not positive")
    log_omega = np.asarray(
        [adapter.holomorphic_volume_log_density(point) for point in points],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(log_omega)):
        raise FloatingPointError(
            "common point pool has non-finite holomorphic-volume density"
        )
    cluster_ids = np.asarray(
        adapter.sampling_cluster_ids(points),
        dtype=np.int64,
    )
    if cluster_ids.shape != (count,):
        raise ValueError("sampling cluster ids must have one entry per point")
    return {
        "importance_log_weights": log_weights,
        "importance_weights": weights,
        "sampling_cluster_ids": cluster_ids,
        "baseline_metrics": np.asarray(baseline_metrics, dtype=np.complex128),
        "baseline_metric_eigenvalues": np.asarray(
            baseline_eigenvalues,
            dtype=np.float64,
        ),
        "holomorphic_volume_log_density": log_omega,
        "projective_charts": np.asarray(
            [point.projective_chart for point in points],
            dtype=np.int64,
        ),
        "independent_indices": np.asarray(
            [point.independent_indices for point in points],
            dtype=np.int64,
        ),
        "dependent_indices": np.asarray(
            [point.dependent_indices for point in points],
            dtype=np.int64,
        ),
        "affine_coordinates": np.asarray(
            [point.affine_coordinates for point in points],
            dtype=np.complex128,
        ),
        "tangent_basis": np.asarray(
            [point.tangent_basis for point in points],
            dtype=np.complex128,
        ),
        "residue_denominator": np.asarray(
            [point.residue_denominator for point in points],
            dtype=np.complex128,
        ),
        "jacobian_min_singular_value": np.asarray(
            [adapter.point_jacobian_min_singular_value(point) for point in points],
            dtype=np.float64,
        ),
    }


def save_common_point_pool(
    path: Path,
    manifest_path: Path,
    adapter: GCICYAdapter,
    points: Sequence[Any],
    *,
    model_seed: int,
    exact_model: bool,
    split: str,
    sampling_seed: int,
    cluster_size: int,
    sampling_metadata: Mapping[str, Any],
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output = path.expanduser().resolve()
    manifest_output = manifest_path.expanduser().resolve()
    if output.exists() or manifest_output.exists():
        raise FileExistsError("common point pool artifacts are immutable")
    if not split:
        raise ValueError("common point pool split cannot be empty")
    if cluster_size <= 0 or len(points) % cluster_size:
        raise ValueError("point count must contain complete sampling clusters")
    point_payload = adapter.point_storage_payload(points)
    arrays = common_point_arrays(adapter, points)
    metadata = {
        "schema": "gcicy-common-point-pool-v1",
        "schema_version": COMMON_POINT_POOL_SCHEMA_VERSION,
        "adapter": adapter.key,
        "adapter_version": adapter.version,
        "model_seed": int(model_seed),
        "exact_model": bool(exact_model),
        "split": split,
        "point_count": int(len(points)),
        "sampling_seed": int(sampling_seed),
        "cluster_size": int(cluster_size),
        "cluster_count": int(len(points) // cluster_size),
        "sampling": _json_value(dict(sampling_metadata)),
        "extra": _json_value(dict(extra_metadata or {})),
    }
    payload: dict[str, np.ndarray] = {
        "common_point_pool_schema_version": np.asarray(
            COMMON_POINT_POOL_SCHEMA_VERSION,
            dtype=np.int64,
        ),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        **arrays,
    }
    for key, values in point_payload.items():
        if not key or key.startswith("point_"):
            raise ValueError("adapter point-payload keys must be unprefixed names")
        array = np.asarray(values)
        if len(array) != len(points):
            raise ValueError(f"point payload {key} has the wrong leading dimension")
        payload[f"point_{key}"] = array
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(output)
    digest = file_sha256(output)
    array_manifest = {
        key: {
            "shape": list(np.asarray(value).shape),
            "dtype": str(np.asarray(value).dtype),
        }
        for key, value in payload.items()
        if key != "metadata_json"
    }
    manifest = {
        **metadata,
        "pool_path": str(output),
        "pool_sha256": digest,
        "arrays": array_manifest,
    }
    temporary_manifest = manifest_output.with_name(
        f".{manifest_output.name}.tmp"
    )
    temporary_manifest.write_text(
        json.dumps(_json_value(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_output)
    return manifest


def load_common_point_pool(
    path: Path,
    adapter: GCICYAdapter,
    model: Any,
    *,
    expected_model_seed: int,
    expected_exact_model: bool,
    expected_split: str | None = None,
) -> CommonPointPool:
    source = path.expanduser().resolve()
    with np.load(source, allow_pickle=False) as payload:
        schema_version = int(payload["common_point_pool_schema_version"])
        if schema_version != COMMON_POINT_POOL_SCHEMA_VERSION:
            raise ValueError("common point pool schema version is not supported")
        metadata = json.loads(str(payload["metadata_json"]))
        if metadata["adapter"] != adapter.key:
            raise ValueError("common point pool adapter does not match")
        if int(metadata["model_seed"]) != expected_model_seed:
            raise ValueError("common point pool model seed does not match")
        if bool(metadata["exact_model"]) != expected_exact_model:
            raise ValueError("common point pool exact-model flag does not match")
        if expected_split is not None and metadata["split"] != expected_split:
            raise ValueError("common point pool split does not match")
        point_payload = {
            key.removeprefix("point_"): np.asarray(payload[key])
            for key in payload.files
            if key.startswith("point_")
        }
        geometry_payload = {
            key: np.asarray(payload[key])
            for key in (
                "projective_charts",
                "independent_indices",
                "dependent_indices",
                "affine_coordinates",
                "tangent_basis",
                "residue_denominator",
                "jacobian_min_singular_value",
            )
        }
        points = adapter.points_from_common_pool_payload(
            model,
            point_payload,
            geometry_payload,
        )
        count = int(metadata["point_count"])
        if len(points) != count:
            raise ValueError("common point pool reconstructed the wrong point count")
        importance_log_weights = np.asarray(
            payload["importance_log_weights"],
            dtype=np.float64,
        )
        importance_weights = np.asarray(
            payload["importance_weights"],
            dtype=np.float64,
        )
        cluster_ids = np.asarray(
            payload["sampling_cluster_ids"],
            dtype=np.int64,
        )
        baseline_metrics = np.asarray(
            payload["baseline_metrics"],
            dtype=np.complex128,
        )
        log_omega = np.asarray(
            payload["holomorphic_volume_log_density"],
            dtype=np.float64,
        )
        stored_affine = np.asarray(
            payload["affine_coordinates"],
            dtype=np.complex128,
        )
        stored_tangent = np.asarray(
            payload["tangent_basis"],
            dtype=np.complex128,
        )
    reconstructed_affine = np.asarray(
        [point.affine_coordinates for point in points],
        dtype=np.complex128,
    )
    reconstructed_tangent = np.asarray(
        [point.tangent_basis for point in points],
        dtype=np.complex128,
    )
    if not np.allclose(reconstructed_affine, stored_affine, rtol=0, atol=1e-13):
        raise ValueError("common point pool affine-coordinate replay failed")
    if not np.allclose(
        reconstructed_tangent,
        stored_tangent,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("common point pool tangent-basis replay failed")
    return CommonPointPool(
        path=source,
        points=points,
        importance_log_weights=importance_log_weights,
        importance_weights=importance_weights,
        sampling_cluster_ids=cluster_ids,
        baseline_metrics=baseline_metrics,
        holomorphic_volume_log_density=log_omega,
        metadata=metadata,
    )
