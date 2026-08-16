#!/usr/bin/env python3
"""Evaluate one full-H artifact on an immutable common gCICY point pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import get_adapter, load_common_point_pool  # noqa: E402
from gcicy_metric.pipeline.tail import (  # noqa: E402
    add_metric_geometry_evidence,
    bilateral_tail_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--common-pool", type=Path, required=True)
    parser.add_argument(
        "--common-pool-split",
        choices=("selection", "confirmation", "blind"),
        required=True,
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--arrays-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_npz_atomic(path: Path, **payload: np.ndarray) -> None:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(output)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("batch-size must be positive")
    started = time.perf_counter()
    adapter = get_adapter(args.adapter)
    model = adapter.make_model(args.model_seed, exact=True)
    artifact_path = args.artifact.expanduser().resolve()
    artifact = adapter.load_h_artifact(artifact_path, model)
    pool_path = args.common_pool.expanduser().resolve()
    pool = load_common_point_pool(
        pool_path,
        adapter,
        model,
        expected_model_seed=args.model_seed,
        expected_exact_model=True,
        expected_split=args.common_pool_split,
    )

    log_eta_rows = []
    minimum_eigenvalue_rows = []
    nonpositive_or_nonfinite = 0
    metric_started = time.perf_counter()
    for start in range(0, len(pool.points), args.batch_size):
        stop = min(start + args.batch_size, len(pool.points))
        points = pool.points[start:stop]
        metrics = np.asarray(
            adapter.h_metrics(points, artifact),
            dtype=np.complex128,
        )
        eigenvalues = np.linalg.eigvalsh(metrics)
        minimum = eigenvalues[:, 0]
        valid = np.all(np.isfinite(eigenvalues), axis=1) & (minimum > 0)
        nonpositive_or_nonfinite += int(np.count_nonzero(~valid))
        log_eta = np.full(len(points), np.nan, dtype=np.float64)
        if np.any(valid):
            valid_points = [
                point for point, keep in zip(points, valid, strict=True) if keep
            ]
            log_eta[valid] = adapter.residual_values(
                valid_points,
                metrics[valid],
            )
        log_eta_rows.append(log_eta)
        minimum_eigenvalue_rows.append(minimum)
    metric_seconds = time.perf_counter() - metric_started

    model_log_eta = np.concatenate(log_eta_rows)
    minimum_eigenvalues = np.concatenate(minimum_eigenvalue_rows)
    if nonpositive_or_nonfinite:
        raise FloatingPointError(
            "full-H common-pool evaluation found "
            f"{nonpositive_or_nonfinite} non-positive or non-finite metrics"
        )
    metrics = bilateral_tail_metrics(
        model_log_eta,
        pool.importance_weights,
        pool.sampling_cluster_ids,
    )
    metrics = add_metric_geometry_evidence(metrics, minimum_eigenvalues)

    arrays_path = args.arrays_out.expanduser().resolve()
    save_npz_atomic(
        arrays_path,
        schema=np.asarray("gcicy-common-full-h-point-arrays-v1"),
        adapter=np.asarray(adapter.key),
        common_pool_sha256=np.asarray(sha256(pool_path)),
        model_artifact_sha256=np.asarray(sha256(artifact_path)),
        model_log_eta=model_log_eta,
        importance_weights=pool.importance_weights,
        sampling_cluster_ids=pool.sampling_cluster_ids,
        metric_minimum_eigenvalues=minimum_eigenvalues,
    )
    report = {
        "schema": "gcicy-common-full-h-evaluation-v1",
        "adapter": adapter.key,
        "model_seed": args.model_seed,
        "artifact": str(artifact_path),
        "artifact_sha256": sha256(artifact_path),
        "artifact_degree": list(artifact.degree),
        "common_pool": str(pool_path),
        "common_pool_sha256": sha256(pool_path),
        "common_pool_split": args.common_pool_split,
        "point_count": len(pool.points),
        "cluster_count": int(len(np.unique(pool.sampling_cluster_ids))),
        "arrays": str(arrays_path),
        "arrays_sha256": sha256(arrays_path),
        "metrics": metrics,
        "runtime_seconds": {
            "metric_evaluation": metric_seconds,
            "total": time.perf_counter() - started,
        },
    }
    output = args.out.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps(report, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
