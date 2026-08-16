#!/usr/bin/env python3
"""Diagnose metric-volume importance-weight tails on one frozen sample seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import get_adapter  # noqa: E402
from gcicy_metric.pipeline.audit import (
    effective_sample_size,
    standard_errors,
)  # noqa: E402
from gcicy_metric.pipeline.parallel_sampling import sample_points_parallel  # noqa: E402
from gcicy_metric.pipeline.spectrum import metric_volume_weights  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument(
        "--exact-model", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--artifacts", type=Path, nargs="+", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--points", type=int, required=True)
    parser.add_argument("--sampling-workers", type=int, default=1)
    parser.add_argument("--sampling-cluster-size", type=int, default=1)
    parser.add_argument(
        "--sampling-backend",
        choices=("process", "thread"),
        default="process",
    )
    parser.add_argument("--top-points", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def quantiles(values: np.ndarray) -> dict[str, float]:
    probabilities = (0.0, 0.5, 0.9, 0.99, 0.999, 0.9999, 1.0)
    observed = np.quantile(np.asarray(values, dtype=float), probabilities)
    return {
        f"q{probability:g}": float(value)
        for probability, value in zip(probabilities, observed, strict=True)
    }


def coordinate_summary(point: Any) -> dict[str, Any]:
    coordinates = getattr(point, "coordinates", ())
    magnitudes = [
        np.abs(np.asarray(factor, dtype=np.complex128)).reshape(-1)
        for factor in coordinates
    ]
    return {
        "projective_chart": [int(value) for value in point.projective_chart],
        "independent_indices": [int(value) for value in point.independent_indices],
        "jacobian_min_singular_value": float(point.jacobian_min_singular_value),
        "minimum_homogeneous_coordinate_magnitude": float(
            min(np.min(factor) for factor in magnitudes)
        ),
        "maximum_homogeneous_coordinate_magnitude": float(
            max(np.max(factor) for factor in magnitudes)
        ),
        "factor_coordinate_magnitudes": [factor.tolist() for factor in magnitudes],
    }


def main() -> None:
    args = parse_args()
    if (
        min(
            args.points,
            args.top_points,
            args.sampling_workers,
            args.sampling_cluster_size,
        )
        <= 0
    ):
        raise SystemExit(
            "points, top-points, workers, and cluster size must be positive"
        )
    adapter = get_adapter(args.adapter)
    model = adapter.make_model(args.model_seed, exact=args.exact_model)
    if args.sampling_workers > 1:
        points, shards = sample_points_parallel(
            adapter,
            model_seed=args.model_seed,
            exact_model=args.exact_model,
            count=args.points,
            seed=args.seed,
            workers=args.sampling_workers,
            cluster_size=args.sampling_cluster_size,
            backend=args.sampling_backend,
        )
        sampling = {
            "mode": "deterministic_parallel_complete_cluster",
            "requested_points": args.points,
            "returned_points": len(points),
            "workers": args.sampling_workers,
            "cluster_size": args.sampling_cluster_size,
            "backend": args.sampling_backend,
            "seed_derivation": "numpy.random.SeedSequence(base_seed).spawn(active_workers)",
            "shards": shards,
        }
    else:
        points, sampling = adapter.sample_points_with_diagnostics(
            model, args.points, seed=args.seed
        )
    cluster_ids = np.asarray(adapter.sampling_cluster_ids(points), dtype=np.int64)
    if cluster_ids.shape != (args.points,):
        raise SystemExit("sampling cluster ids do not match the requested points")
    omega_log_weights = np.asarray(
        [adapter.importance_log_weight(point) for point in points], dtype=float
    )
    omega_shifted = np.exp(omega_log_weights - float(np.max(omega_log_weights)))
    omega_weights = omega_shifted / np.sum(omega_shifted)
    rows = []
    for artifact_path in args.artifacts:
        artifact = adapter.load_h_artifact(artifact_path, model)
        metrics = adapter.h_metrics(points, artifact)
        residuals = np.asarray(adapter.residual_values(points, metrics), dtype=float)
        weights, log_weights = metric_volume_weights(adapter, points, metrics)
        unique_clusters, inverse = np.unique(cluster_ids, return_inverse=True)
        cluster_weights = np.bincount(
            inverse, weights=weights, minlength=len(unique_clusters)
        )
        shifted_ratio = np.exp(residuals - float(np.max(residuals)))
        normalized_ratio = shifted_ratio / float(np.sum(omega_weights * shifted_ratio))
        squared_error_contributions = omega_weights * (1.0 - normalized_ratio) ** 2
        squared_error = float(np.sum(squared_error_contributions))
        ratio_order = np.argsort(normalized_ratio)[::-1]
        error_order = np.argsort(squared_error_contributions)[::-1]
        point_order = np.argsort(weights)[::-1]
        cluster_order = np.argsort(cluster_weights)[::-1]
        top_count = min(args.top_points, len(point_order))

        def metric_point_summary(index: int) -> dict[str, Any]:
            eigenvalues = np.linalg.eigvalsh(
                np.asarray(metrics[int(index)], dtype=np.complex128)
            )
            return {
                "point_index": int(index),
                "cluster_id": int(cluster_ids[index]),
                "normalized_metric_volume_weight": float(weights[index]),
                "normalized_omega_weight": float(omega_weights[index]),
                "normalized_volume_ratio": float(normalized_ratio[index]),
                "squared_error_contribution": float(squared_error_contributions[index]),
                "squared_error_fraction": (
                    float(squared_error_contributions[index] / squared_error)
                    if squared_error > 0
                    else 0.0
                ),
                "log_metric_volume_weight": float(log_weights[index]),
                "log_omega_weight": float(omega_log_weights[index]),
                "log_monge_ampere_ratio": float(residuals[index]),
                "metric_eigenvalues": [float(value) for value in eigenvalues],
                **coordinate_summary(points[int(index)]),
            }

        top_points = []
        for index in point_order[:top_count]:
            row = metric_point_summary(int(index))
            row["normalized_weight"] = row["normalized_metric_volume_weight"]
            top_points.append(row)
        top_cluster_rows = []
        for cluster_index in cluster_order[: min(8, len(cluster_order))]:
            cluster_id = int(unique_clusters[cluster_index])
            member_indices = np.flatnonzero(cluster_ids == cluster_id)
            top_cluster_rows.append(
                {
                    "cluster_id": cluster_id,
                    "normalized_cluster_weight": float(cluster_weights[cluster_index]),
                    "point_indices": [int(value) for value in member_indices],
                    "point_weights": [
                        float(weights[value]) for value in member_indices
                    ],
                    "minimum_jacobian_singular_value": float(
                        min(
                            points[int(value)].jacobian_min_singular_value
                            for value in member_indices
                        )
                    ),
                }
            )
        rows.append(
            {
                "artifact_key": artifact.key,
                "artifact_path": str(artifact_path.expanduser().resolve()),
                "point_effective_sample_size": effective_sample_size(weights),
                "cluster_weight_effective_sample_size": effective_sample_size(
                    cluster_weights
                ),
                "volume_ratio_error_statistics": standard_errors(
                    residuals,
                    omega_weights,
                ),
                "log_metric_volume_weight_quantiles": quantiles(log_weights),
                "log_monge_ampere_ratio_quantiles": quantiles(residuals),
                "volume_ratio_tail": {
                    "maximum_ratio_point": metric_point_summary(int(ratio_order[0])),
                    "maximum_squared_error_point": metric_point_summary(
                        int(error_order[0])
                    ),
                    "squared_error_concentration": {
                        str(count): (
                            float(
                                np.sum(squared_error_contributions[error_order[:count]])
                                / squared_error
                            )
                            if squared_error > 0
                            else 0.0
                        )
                        for count in (1, 4, 16, 64, 256)
                        if count <= len(error_order)
                    },
                },
                "largest_log_weight_minus_median": float(
                    np.max(log_weights) - np.median(log_weights)
                ),
                "point_weight_concentration": {
                    str(count): float(np.sum(weights[point_order[:count]]))
                    for count in (1, 4, 16, 64, 256)
                    if count <= len(weights)
                },
                "cluster_weight_concentration": {
                    str(count): float(np.sum(cluster_weights[cluster_order[:count]]))
                    for count in (1, 4, 16, 64, 256)
                    if count <= len(cluster_weights)
                },
                "top_points": top_points,
                "top_clusters": top_cluster_rows,
            }
        )
    output = {
        "schema_version": 1,
        "description": (
            "Frozen-seed diagnostic of metric-volume importance-weight tails; "
            "this does not alter any spectrum acceptance threshold."
        ),
        "adapter": args.adapter,
        "model_seed": args.model_seed,
        "exact_model": args.exact_model,
        "seed": args.seed,
        "points": args.points,
        "sampling_diagnostics": sampling,
        "omega_weight_summary": {
            "point_effective_sample_size": effective_sample_size(omega_weights),
            "log_weight_quantiles": quantiles(omega_log_weights),
        },
        "artifacts": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    for row in rows:
        print(
            f"{row['artifact_key']}: point ESS="
            f"{row['point_effective_sample_size']:.6g}, cluster ESS="
            f"{row['cluster_weight_effective_sample_size']:.6g}, top point share="
            f"{row['point_weight_concentration']['1']:.6g}"
        )


if __name__ == "__main__":
    main()
