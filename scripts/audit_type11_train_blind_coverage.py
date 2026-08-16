#!/usr/bin/env python3
"""Measure type-(1,1) blind-point coverage by the frozen training cloud."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        type=Path,
        default=ROOT
        / "outputs/cymetric_phi_gcicy_type11_pilot_8k_e20_seed720xx/"
        "train_seed72001_n8192.npz",
    )
    parser.add_argument(
        "--blind",
        type=Path,
        default=ROOT
        / "outputs/cymetric_phi_gcicy_type11_pilot_8k_e20_seed720xx/"
        "blind_seed72003_n8192.npz",
    )
    parser.add_argument(
        "--ratios",
        type=Path,
        default=ROOT
        / "outputs/pipeline/type11_samepoint_mechanism_20260716/"
        "samepoint_ratios.npz",
    )
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "outputs/pipeline/type11_samepoint_mechanism_20260716/"
        "train_blind_coverage.json",
    )
    return parser.parse_args()


def unit_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.complex128)
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    value = spearmanr(left, right).statistic
    return float(value) if np.isfinite(value) else 0.0


def quantiles(values: np.ndarray) -> dict[str, float]:
    probabilities = (0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 1.0)
    observed = np.quantile(values, probabilities)
    return {
        f"q{probability:.3f}": float(value)
        for probability, value in zip(probabilities, observed, strict=True)
    }


def nearest_product_distances(
    train_x: np.ndarray,
    train_y: np.ndarray,
    blind_x: np.ndarray,
    blind_y: np.ndarray,
    *,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(blind_x)
    distances = np.empty(count, dtype=np.float64)
    nearest_indices = np.empty(count, dtype=np.int64)
    nearest_overlaps = np.empty((count, 2), dtype=np.float64)
    train_x_conjugate = train_x.conjugate().T
    train_y_conjugate = train_y.conjugate().T
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        overlap_x = np.clip(
            np.abs(blind_x[start:stop] @ train_x_conjugate), 0.0, 1.0
        )
        overlap_y = np.clip(
            np.abs(blind_y[start:stop] @ train_y_conjugate), 0.0, 1.0
        )
        distance_squared = np.square(np.arccos(overlap_x)) + np.square(
            np.arccos(overlap_y)
        )
        local_indices = np.argmin(distance_squared, axis=1)
        rows = np.arange(stop - start)
        nearest_indices[start:stop] = local_indices
        distances[start:stop] = np.sqrt(distance_squared[rows, local_indices])
        nearest_overlaps[start:stop, 0] = overlap_x[rows, local_indices]
        nearest_overlaps[start:stop, 1] = overlap_y[rows, local_indices]
    return distances, nearest_indices, nearest_overlaps


def nearest_base_distances(
    train_y: np.ndarray,
    blind_y: np.ndarray,
    *,
    chunk_size: int,
) -> np.ndarray:
    output = np.empty(len(blind_y), dtype=np.float64)
    conjugate = train_y.conjugate().T
    for start in range(0, len(blind_y), chunk_size):
        stop = min(start + chunk_size, len(blind_y))
        overlap = np.clip(
            np.abs(blind_y[start:stop] @ conjugate), 0.0, 1.0
        )
        output[start:stop] = np.min(np.arccos(overlap), axis=1)
    return output


def model_coverage_summary(
    ratio: np.ndarray,
    weights: np.ndarray,
    distance: np.ndarray,
    base_distance: np.ndarray,
    nearest_indices: np.ndarray,
    nearest_overlaps: np.ndarray,
    train_cluster_ids: np.ndarray,
    blind_cluster_ids: np.ndarray,
) -> dict[str, Any]:
    ratio = np.asarray(ratio, dtype=np.float64)
    abs_log = np.abs(np.log(ratio))
    top_count = max(1, len(ratio) // 100)
    top_indices = np.argsort(ratio, kind="stable")[-top_count:]
    far_indices = np.argsort(distance, kind="stable")[-top_count:]
    overlap = len(set(top_indices.tolist()) & set(far_indices.tolist()))
    worst = int(np.argmax(ratio))
    worst_distance_percentile = float(np.mean(distance <= distance[worst]))
    bins = np.quantile(distance, (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0))
    coverage_rows = []
    for left, right in zip(bins[:-1], bins[1:], strict=True):
        mask = (distance >= left) & (distance <= right)
        coverage_rows.append(
            {
                "distance_min": float(left),
                "distance_max": float(right),
                "point_count": int(np.sum(mask)),
                "mean_ratio": float(np.mean(ratio[mask])),
                "maximum_ratio": float(np.max(ratio[mask])),
                "weighted_sigma": float(
                    np.sum(weights[mask] * np.abs(1.0 - ratio[mask]))
                    / np.sum(weights[mask])
                ),
            }
        )
    nearest = int(nearest_indices[worst])
    return {
        "spearman_ratio_vs_nearest_product_distance": safe_spearman(
            ratio, distance
        ),
        "spearman_abs_log_ratio_vs_nearest_product_distance": safe_spearman(
            abs_log, distance
        ),
        "spearman_abs_log_ratio_vs_nearest_base_distance": safe_spearman(
            abs_log, base_distance
        ),
        "top_one_percent_overlap_with_farthest_one_percent": overlap,
        "top_one_percent_far_distance_enrichment": overlap / top_count / 0.01,
        "top_ratio_distance_quantiles": quantiles(distance[top_indices]),
        "coverage_bins": coverage_rows,
        "worst_point": {
            "index": worst,
            "cluster_id": int(blind_cluster_ids[worst]),
            "ratio": float(ratio[worst]),
            "nearest_product_distance": float(distance[worst]),
            "nearest_product_distance_percentile": worst_distance_percentile,
            "nearest_base_distance": float(base_distance[worst]),
            "nearest_train_point_index": nearest,
            "nearest_train_cluster_id": int(train_cluster_ids[nearest]),
            "nearest_x_overlap": float(nearest_overlaps[worst, 0]),
            "nearest_y_overlap": float(nearest_overlaps[worst, 1]),
        },
    }


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be positive")
    started = time.perf_counter()
    with np.load(args.train.expanduser().resolve(), allow_pickle=False) as payload:
        train_x = unit_rows(payload["coordinates_x"])
        train_y = unit_rows(payload["coordinates_y"])
        train_cluster_ids = np.asarray(payload["cluster_ids"], dtype=np.int64)
    with np.load(args.blind.expanduser().resolve(), allow_pickle=False) as payload:
        blind_x = unit_rows(payload["coordinates_x"])
        blind_y = unit_rows(payload["coordinates_y"])
        blind_cluster_ids = np.asarray(payload["cluster_ids"], dtype=np.int64)
        weights = np.asarray(payload["importance_weights"], dtype=np.float64)
    with np.load(args.ratios.expanduser().resolve(), allow_pickle=False) as payload:
        ratios = {
            key.removesuffix("_normalized_ratio"): np.asarray(
                payload[key], dtype=np.float64
            )
            for key in payload.files
            if key.endswith("_normalized_ratio")
        }
    if any(len(ratio) != len(blind_x) for ratio in ratios.values()):
        raise ValueError("ratio arrays do not match the blind point count")

    product_distance, nearest_indices, nearest_overlaps = nearest_product_distances(
        train_x,
        train_y,
        blind_x,
        blind_y,
        chunk_size=args.chunk_size,
    )
    train_base_indices = np.asarray(
        [
            np.flatnonzero(train_cluster_ids == cluster_id)[0]
            for cluster_id in np.unique(train_cluster_ids)
        ],
        dtype=np.int64,
    )
    base_distance = nearest_base_distances(
        train_y[train_base_indices],
        blind_y,
        chunk_size=args.chunk_size,
    )
    output = {
        "schema": "type11-train-blind-coverage-v1",
        "train_path": str(args.train.expanduser().resolve()),
        "blind_path": str(args.blind.expanduser().resolve()),
        "ratio_path": str(args.ratios.expanduser().resolve()),
        "train_point_count": int(len(train_x)),
        "train_independent_fibre_count": int(len(np.unique(train_cluster_ids))),
        "blind_point_count": int(len(blind_x)),
        "blind_independent_fibre_count": int(len(np.unique(blind_cluster_ids))),
        "nearest_product_distance": quantiles(product_distance),
        "nearest_base_distance": quantiles(base_distance),
        "models": {
            name: model_coverage_summary(
                ratio,
                weights,
                product_distance,
                base_distance,
                nearest_indices,
                nearest_overlaps,
                train_cluster_ids,
                blind_cluster_ids,
            )
            for name, ratio in ratios.items()
        },
        "runtime_seconds": float(time.perf_counter() - started),
        "interpretation": (
            "Large positive correlation or enrichment would support a coverage "
            "mechanism. Small values mean the optimizer can leave tails even near "
            "the observed training cloud."
        ),
    }
    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "nearest_product_distance": output["nearest_product_distance"],
                "models": {
                    name: {
                        "abs_log_distance_spearman": row[
                            "spearman_abs_log_ratio_vs_nearest_product_distance"
                        ],
                        "far_tail_enrichment": row[
                            "top_one_percent_far_distance_enrichment"
                        ],
                        "worst_point": row["worst_point"],
                    }
                    for name, row in output["models"].items()
                },
                "runtime_seconds": output["runtime_seconds"],
            },
            indent=2,
        )
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
