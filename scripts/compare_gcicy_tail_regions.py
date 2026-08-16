#!/usr/bin/env python3
"""Compare frozen tail regions against centers in an active-point pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import (  # noqa: E402
    file_sha256,
    get_adapter,
    load_active_point_pool,
)
from gcicy_metric.pipeline.parallel_sampling import sample_points_parallel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument(
        "--exact-model", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--active-pool", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--points-per-seed", type=int, required=True)
    parser.add_argument("--sampling-workers", type=int, default=1)
    parser.add_argument("--sampling-cluster-size", type=int, default=1)
    parser.add_argument(
        "--sampling-backend", choices=("process", "thread"), default="process"
    )
    parser.add_argument("--ratio-threshold", type=float, default=3.0)
    parser.add_argument("--same-region-distance", type=float, default=0.05)
    parser.add_argument(
        "--distance-thresholds", type=float, nargs="+", default=(0.05, 0.1, 0.25)
    )
    parser.add_argument("--top-points", type=int, default=32)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def weighted_log_mean_exp(values: np.ndarray, weights: np.ndarray) -> float:
    raw = np.asarray(values, dtype=float)
    positive = np.asarray(weights, dtype=float)
    if (
        raw.shape != positive.shape
        or raw.size == 0
        or not np.all(np.isfinite(raw))
        or not np.all(np.isfinite(positive))
        or np.min(positive) <= 0
    ):
        raise ValueError("weighted log-mean-exp inputs must be finite and positive")
    return float(
        np.logaddexp.reduce(np.log(positive) + raw) - np.log(np.sum(positive))
    )


def distance_quantiles(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    probabilities = (0.0, 0.25, 0.5, 0.75, 1.0)
    observed = np.quantile(np.asarray(values, dtype=float), probabilities)
    return {
        label: float(value)
        for label, value in zip(
            ("minimum", "q25", "median", "q75", "maximum"),
            observed,
            strict=True,
        )
    }


def nearest_center(adapter, point: Any, centers: Sequence[Any]) -> tuple[int, float]:
    distances = np.asarray(
        [adapter.point_distance(point, center) for center in centers], dtype=float
    )
    index = int(np.argmin(distances))
    return index, float(distances[index])


def greedy_distinct_centers(
    adapter,
    candidates: Sequence[dict[str, Any]],
    minimum_distance: float,
) -> list[dict[str, Any]]:
    centers: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda row: row["u"], reverse=True):
        if all(
            adapter.point_distance(candidate["point"], center["point"])
            >= minimum_distance
            for center in centers
        ):
            centers.append(candidate)
    return centers


def summarize_artifact(
    adapter,
    artifact_path: Path,
    artifact,
    sampled_batches: Sequence[dict[str, Any]],
    active_centers: Sequence[Any],
    *,
    ratio_threshold: float,
    same_region_distance: float,
    distance_thresholds: Sequence[float],
    top_points: int,
) -> dict[str, Any]:
    evaluated_batches = []
    global_log_numerator = float("-inf")
    global_weight_sum = 0.0
    for batch in sampled_batches:
        points = batch["points"]
        metrics = adapter.h_metrics(points, artifact)
        raw = np.asarray(adapter.residual_values(points, metrics), dtype=float)
        weights = np.asarray(adapter.importance_weights(points), dtype=float)
        log_numerator = float(np.logaddexp.reduce(np.log(weights) + raw))
        global_log_numerator = float(
            np.logaddexp(global_log_numerator, log_numerator)
        )
        global_weight_sum += float(np.sum(weights))
        evaluated_batches.append(
            {
                **batch,
                "raw": raw,
                "weights": weights,
                "seed_log_normalization": weighted_log_mean_exp(raw, weights),
            }
        )
    global_log_normalization = float(
        global_log_numerator - np.log(global_weight_sum)
    )
    positive_log_threshold = float(np.log(ratio_threshold))
    candidates: list[dict[str, Any]] = []
    seed_rows = []
    for batch in evaluated_batches:
        raw = batch["raw"]
        seed_u = raw - batch["seed_log_normalization"]
        seed_top_index = int(np.argmax(seed_u))
        nearest_id, nearest_distance = nearest_center(
            adapter, batch["points"][seed_top_index], active_centers
        )
        seed_rows.append(
            {
                "seed": int(batch["seed"]),
                "point_count": int(len(raw)),
                "seed_log_normalization": float(batch["seed_log_normalization"]),
                "maximum_seed_normalized_ratio": float(np.exp(seed_u[seed_top_index])),
                "maximum_seed_ratio_point_index": seed_top_index,
                "maximum_seed_ratio_nearest_active_center": nearest_id,
                "maximum_seed_ratio_nearest_active_center_distance": nearest_distance,
            }
        )
        global_u = raw - global_log_normalization
        cluster_ids = np.asarray(
            adapter.sampling_cluster_ids(batch["points"]), dtype=np.int64
        )
        for point_index in np.flatnonzero(global_u > positive_log_threshold):
            candidates.append(
                {
                    "point": batch["points"][int(point_index)],
                    "seed": int(batch["seed"]),
                    "point_index": int(point_index),
                    "cluster_id": int(cluster_ids[int(point_index)]),
                    "u": float(global_u[int(point_index)]),
                }
            )
    distinct = greedy_distinct_centers(
        adapter, candidates, minimum_distance=same_region_distance
    )
    old_to_active_distances = []
    old_rows = []
    for row in distinct:
        nearest_id, distance = nearest_center(adapter, row["point"], active_centers)
        old_to_active_distances.append(distance)
        old_rows.append(
            {
                "seed": row["seed"],
                "point_index": row["point_index"],
                "cluster_id": row["cluster_id"],
                "normalized_ratio": float(np.exp(row["u"])),
                "nearest_active_center": nearest_id,
                "nearest_active_center_distance": distance,
            }
        )
    active_to_old_distances = []
    if distinct:
        old_points = [row["point"] for row in distinct]
        for center in active_centers:
            _, distance = nearest_center(adapter, center, old_points)
            active_to_old_distances.append(distance)
    current_metrics = adapter.h_metrics(active_centers, artifact)
    current_raw = np.asarray(
        adapter.residual_values(active_centers, current_metrics), dtype=float
    )
    current_u = current_raw - global_log_normalization
    return {
        "artifact": str(artifact_path.resolve()),
        "artifact_sha256": file_sha256(artifact_path),
        "global_log_normalization_over_replayed_seeds": global_log_normalization,
        "replayed_seeds": seed_rows,
        "global_violating_point_count": int(len(candidates)),
        "global_violating_independent_fibre_count": int(
            len({(row["seed"], row["cluster_id"]) for row in candidates})
        ),
        "global_distinct_tail_center_count": int(len(distinct)),
        "old_distinct_centers": old_rows[:top_points],
        "old_to_active_center_distance_quantiles": distance_quantiles(
            old_to_active_distances
        ),
        "old_centers_within_distance": {
            str(threshold): int(
                np.count_nonzero(np.asarray(old_to_active_distances) < threshold)
            )
            for threshold in distance_thresholds
        },
        "active_to_old_center_distance_quantiles": distance_quantiles(
            active_to_old_distances
        ),
        "active_centers_within_distance": {
            str(threshold): int(
                np.count_nonzero(np.asarray(active_to_old_distances) < threshold)
            )
            for threshold in distance_thresholds
        },
        "active_center_ratios_under_artifact": [
            float(value) for value in np.exp(current_u)
        ],
        "active_centers_above_ratio_threshold": int(
            np.count_nonzero(current_u > positive_log_threshold)
        ),
    }


def main() -> None:
    args = parse_args()
    if (
        args.points_per_seed <= 0
        or args.sampling_workers <= 0
        or args.sampling_cluster_size <= 0
        or args.top_points <= 0
        or args.ratio_threshold <= 1.0
        or args.same_region_distance <= 0
        or not args.seeds
        or len(set(args.seeds)) != len(args.seeds)
        or any(value <= 0 for value in args.distance_thresholds)
    ):
        raise SystemExit("comparison counts, thresholds, and seeds are invalid")
    if args.points_per_seed % args.sampling_cluster_size:
        raise SystemExit("points-per-seed must contain complete sampling clusters")
    started = time.perf_counter()
    adapter = get_adapter(args.adapter)
    model = adapter.make_model(args.model_seed, exact=args.exact_model)
    pool = load_active_point_pool(
        args.active_pool,
        adapter,
        model,
        expected_model_seed=args.model_seed,
        expected_exact_model=args.exact_model,
    )
    active_indices = np.flatnonzero(pool.radii == 0.0)
    active_centers = [pool.points[int(index)] for index in active_indices]
    if not active_centers:
        raise SystemExit("active pool does not contain radius-zero centers")
    sampled_batches = []
    for seed in args.seeds:
        if args.sampling_workers > 1:
            points, shards = sample_points_parallel(
                adapter,
                model_seed=args.model_seed,
                exact_model=args.exact_model,
                count=args.points_per_seed,
                seed=seed,
                workers=args.sampling_workers,
                cluster_size=args.sampling_cluster_size,
                backend=args.sampling_backend,
            )
        else:
            points, diagnostics = adapter.sample_points_with_diagnostics(
                model, args.points_per_seed, seed=seed
            )
            shards = [diagnostics]
        sampled_batches.append({"seed": seed, "points": points, "shards": shards})
        print(f"sampled seed={seed}, points={len(points)}", flush=True)
    artifact_rows = []
    for artifact_path in args.artifacts:
        artifact = adapter.load_h_artifact(artifact_path, model)
        artifact_rows.append(
            summarize_artifact(
                adapter,
                artifact_path,
                artifact,
                sampled_batches,
                active_centers,
                ratio_threshold=args.ratio_threshold,
                same_region_distance=args.same_region_distance,
                distance_thresholds=args.distance_thresholds,
                top_points=args.top_points,
            )
        )
        print(f"evaluated artifact={artifact_path.name}", flush=True)
    output = {
        "schema_version": 1,
        "description": "Frozen-seed Fubini-Study comparison of old and new tail regions.",
        "adapter": adapter.key,
        "model_seed": args.model_seed,
        "exact_model": args.exact_model,
        "active_pool": str(args.active_pool.resolve()),
        "active_pool_sha256": file_sha256(args.active_pool),
        "active_center_count": int(len(active_centers)),
        "seeds": [int(seed) for seed in args.seeds],
        "points_per_seed": int(args.points_per_seed),
        "sampling_workers": int(args.sampling_workers),
        "sampling_cluster_size": int(args.sampling_cluster_size),
        "sampling_backend": args.sampling_backend,
        "ratio_threshold": float(args.ratio_threshold),
        "same_region_distance": float(args.same_region_distance),
        "distance_thresholds": [float(value) for value in args.distance_thresholds],
        "artifacts": artifact_rows,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
