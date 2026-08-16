#!/usr/bin/env python3
"""Run a sampler-matched large-sample metric audit of the bicubic control."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.bicubic_model import (  # noqa: E402
    bicubic_global_h_metrics,
    bicubic_importance_weights,
    bicubic_residual_values,
    make_exact_bicubic_model,
    sample_bicubic_points_with_diagnostics,
)
from gcicy_metric.pipeline import sampling_cluster_statistics  # noqa: E402
from gcicy_metric.pipeline.audit import confidence_interval, standard_errors  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=ROOT / "outputs/bicubic_global_h_metric_k3_k2xk1_gpu.npz",
    )
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--points-per-seed", type=int, default=6144)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--minimum-point-ess", type=float, default=1.0)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs/bicubic_k3_sampler_matched_large_audit.json",
    )
    return parser.parse_args()


def evaluate_seed(job: dict[str, Any]) -> dict[str, Any]:
    artifact = np.load(job["artifact"])
    model_seed = int(artifact["bicubic_model_seed"])
    model = make_exact_bicubic_model(model_seed)
    if not np.allclose(model.coefficients, artifact["bicubic_coefficients"]):
        raise ValueError("bicubic artifact coefficients do not match the model")
    started = time.perf_counter()
    points, sampler = sample_bicubic_points_with_diagnostics(
        model,
        job["points_per_seed"],
        seed=job["seed"],
    )
    weights = bicubic_importance_weights(points)
    normalized_weights = weights / float(np.sum(weights))
    metrics = bicubic_global_h_metrics(
        points,
        artifact["global_section_exponents"],
        artifact["global_h_matrix"],
        normalization=float(artifact["global_section_normalization"]),
    )
    statistics = standard_errors(
        bicubic_residual_values(points, metrics),
        weights,
    )
    cluster_ids = np.repeat(
        np.arange(len(points) // 3, dtype=np.int64),
        3,
    )
    cluster_statistics = sampling_cluster_statistics(
        normalized_weights,
        cluster_ids,
    )
    return {
        "seed": int(job["seed"]),
        **statistics,
        "minimum_metric_eigenvalue": float(
            np.min(np.linalg.eigvalsh(np.asarray(metrics, dtype=np.complex128)))
        ),
        "minimum_jacobian_norm": float(
            min(point.jacobian_norm for point in points)
        ),
        **cluster_statistics,
        "sampler": sampler,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key in (
        "sigma",
        "inverse_sigma",
        "squared_energy",
        "sqrt_squared_energy",
        "weighted_centered_log_ma_rms",
    ):
        values = [float(row[key]) for row in rows]
        output[f"mean_{key}"] = float(np.mean(values))
        output[f"{key}_95_percent_ci"] = confidence_interval(values)
        output[f"maximum_{key}"] = float(np.max(values))
    output.update(
        {
            "minimum_metric_eigenvalue": float(
                min(row["minimum_metric_eigenvalue"] for row in rows)
            ),
            "minimum_jacobian_norm": float(
                min(row["minimum_jacobian_norm"] for row in rows)
            ),
            "minimum_point_effective_sample_size": float(
                min(row["point_effective_sample_size"] for row in rows)
            ),
            "minimum_cluster_weight_effective_sample_size": float(
                min(row["cluster_weight_effective_sample_size"] for row in rows)
            ),
            "attempted_clusters": int(
                sum(row["sampler"]["attempted_clusters"] for row in rows)
            ),
            "accepted_clusters": int(
                sum(row["sampler"]["accepted_clusters"] for row in rows)
            ),
            "rejected_clusters": int(
                sum(row["sampler"]["rejected_clusters"] for row in rows)
            ),
            "minimum_projective_root_separation": float(
                min(
                    row["sampler"]["minimum_projective_root_separation"]
                    for row in rows
                )
            ),
            "maximum_accepted_relative_residual": float(
                max(
                    row["sampler"]["maximum_accepted_relative_residual"]
                    for row in rows
                )
            ),
        }
    )
    return output


def main() -> None:
    args = parse_args()
    if args.points_per_seed <= 0 or args.points_per_seed % 3:
        raise SystemExit("points-per-seed must be a positive multiple of three")
    if args.workers <= 0 or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise SystemExit("workers must be positive and seeds distinct")
    artifact = args.artifact.expanduser().resolve()
    if not artifact.exists():
        raise SystemExit(f"missing artifact: {artifact}")
    for variable in (
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    jobs = [
        {
            "artifact": str(artifact),
            "seed": int(seed),
            "points_per_seed": int(args.points_per_seed),
        }
        for seed in args.seeds
    ]
    started = time.perf_counter()
    rows = []
    with ThreadPoolExecutor(max_workers=min(args.workers, len(jobs))) as executor:
        futures = {executor.submit(evaluate_seed, job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"seed={row['seed']} sigma={row['sigma']:.6e} "
                f"seconds={row['elapsed_seconds']:.1f}",
                flush=True,
            )
    rows.sort(key=lambda row: args.seeds.index(int(row["seed"])))
    summary = aggregate(rows)
    gates = {
        "finite_metric_statistics": all(np.isfinite(row["sigma"]) for row in rows),
        "positive_metric": summary["minimum_metric_eigenvalue"] > 0,
        "point_effective_sample_size": (
            summary["minimum_point_effective_sample_size"]
            >= args.minimum_point_ess
        ),
        "complete_sampling_clusters": all(
            row["sampler"]["all_returned_clusters_complete"] for row in rows
        ),
        "expected_root_count_on_accepted_fibres": all(
            row["sampler"]["accepted_root_count_histogram"]
            == {"3": row["sampler"]["accepted_clusters"]}
            for row in rows
        ),
        "importance_weights_not_used_for_acceptance": all(
            not row["sampler"]["importance_weight_used_for_acceptance"]
            for row in rows
        ),
    }
    output = {
        "schema_version": 1,
        "description": (
            "Sampler-matched ordinary bicubic control using complete three-root "
            "fibres and the same seed-level metric statistics as the gCICY audit."
        ),
        "model_seed": 20260721,
        "artifact": str(artifact),
        "seeds": [int(seed) for seed in args.seeds],
        "points_per_seed": int(args.points_per_seed),
        "total_points": int(len(args.seeds) * args.points_per_seed),
        "workers": int(min(args.workers, len(jobs))),
        "minimum_point_effective_sample_size_required": float(
            args.minimum_point_ess
        ),
        "confidence_interval_convention": (
            "descriptive mean +/- 1.96 sample standard deviation / "
            "sqrt(seed count)"
        ),
        "per_seed": rows,
        "aggregate": summary,
        "gates": gates,
        "success": bool(all(gates.values())),
        "wall_seconds": float(time.perf_counter() - started),
    }
    output_path = (
        args.out.expanduser().resolve()
        if args.out.is_absolute()
        else (ROOT / args.out).resolve()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"success": output["success"], **summary}, indent=2))
    print(f"wrote {output_path}")
    if not output["success"]:
        raise SystemExit("bicubic large-sample audit failed")


if __name__ == "__main__":
    main()
