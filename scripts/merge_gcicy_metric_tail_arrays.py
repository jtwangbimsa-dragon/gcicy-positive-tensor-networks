#!/usr/bin/env python3
"""Merge point-aligned metric audit arrays from independent blind pools."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


REQUIRED_KEYS = {
    "model_log_eta",
    "importance_weights",
    "sampling_cluster_ids",
    "metric_minimum_eigenvalues",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrays", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar_string(value: np.ndarray, name: str) -> str:
    if value.shape != ():
        raise ValueError(f"{name} must be a scalar string")
    return str(value.item())


def save_npz_atomic(path: Path, **payload: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if len(args.arrays) < 2:
        raise SystemExit("at least two array artifacts are required")

    adapters: set[str] = set()
    schemas: list[str] = []
    log_eta_rows: list[np.ndarray] = []
    weight_rows: list[np.ndarray] = []
    cluster_rows: list[np.ndarray] = []
    eigenvalue_rows: list[np.ndarray] = []
    sources: list[dict[str, str | int]] = []
    cluster_offset = 0

    for input_path in args.arrays:
        path = input_path.expanduser().resolve()
        with np.load(path, allow_pickle=False) as data:
            missing = REQUIRED_KEYS.difference(data.files)
            if missing:
                raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
            log_eta = np.asarray(data["model_log_eta"], dtype=np.float64).reshape(-1)
            weights = np.asarray(
                data["importance_weights"], dtype=np.float64
            ).reshape(-1)
            eigenvalues = np.asarray(
                data["metric_minimum_eigenvalues"], dtype=np.float64
            ).reshape(-1)
            clusters = np.asarray(data["sampling_cluster_ids"]).reshape(-1)
            if not (
                log_eta.shape
                == weights.shape
                == eigenvalues.shape
                == clusters.shape
            ):
                raise ValueError(f"{path} contains misaligned point arrays")
            if (
                log_eta.size == 0
                or not np.all(np.isfinite(log_eta))
                or not np.all(np.isfinite(weights))
                or not np.all(np.isfinite(eigenvalues))
                or np.any(weights < 0.0)
                or float(np.sum(weights)) <= 0.0
            ):
                raise ValueError(f"{path} contains invalid numerical arrays")

            unique_clusters, inverse, counts = np.unique(
                clusters,
                return_inverse=True,
                return_counts=True,
            )
            if unique_clusters.size == 0 or len(set(counts.tolist())) != 1:
                raise ValueError(f"{path} does not contain complete equal-size clusters")
            remapped_clusters = inverse + cluster_offset
            cluster_offset += unique_clusters.size

            adapter = (
                scalar_string(data["adapter"], "adapter")
                if "adapter" in data.files
                else ""
            )
            schema = (
                scalar_string(data["schema"], "schema")
                if "schema" in data.files
                else ""
            )
            adapters.add(adapter)
            schemas.append(schema)
            log_eta_rows.append(log_eta)
            weight_rows.append(weights)
            eigenvalue_rows.append(eigenvalues)
            cluster_rows.append(remapped_clusters)
            sources.append(
                {
                    "path": str(path),
                    "sha256": sha256(path),
                    "point_count": int(log_eta.size),
                    "cluster_count": int(unique_clusters.size),
                    "cluster_size": int(counts[0]),
                }
            )

    if len(adapters) != 1:
        raise ValueError("input artifacts use different adapters")
    output_path = args.out.expanduser().resolve()
    save_npz_atomic(
        output_path,
        schema=np.asarray("gcicy-merged-metric-tail-arrays-v1"),
        adapter=np.asarray(next(iter(adapters))),
        model_log_eta=np.concatenate(log_eta_rows),
        importance_weights=np.concatenate(weight_rows),
        sampling_cluster_ids=np.concatenate(cluster_rows),
        metric_minimum_eigenvalues=np.concatenate(eigenvalue_rows),
        source_schemas=np.asarray(schemas),
        source_artifacts_json=np.asarray(json.dumps(sources, sort_keys=True)),
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "sha256": sha256(output_path),
                "point_count": int(sum(row.size for row in log_eta_rows)),
                "cluster_count": int(cluster_offset),
                "sources": sources,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
