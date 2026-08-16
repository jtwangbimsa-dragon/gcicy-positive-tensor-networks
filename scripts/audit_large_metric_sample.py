#!/usr/bin/env python3
"""Run a parallel fresh-seed large-sample audit of one gCICY H metric."""

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

from gcicy_metric.pipeline import get_adapter, sampling_cluster_statistics  # noqa: E402
from gcicy_metric.pipeline.audit import confidence_interval, standard_errors  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--points-per-seed", type=int, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--minimum-point-ess", type=float, default=1.0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--inexact-model", action="store_true")
    return parser.parse_args()


def evaluate_seed(job: dict[str, Any]) -> dict[str, Any]:
    adapter = get_adapter(job["adapter"])
    model = adapter.make_model(job["model_seed"], exact=job["exact_model"])
    artifact = adapter.load_h_artifact(Path(job["artifact"]), model)
    started = time.perf_counter()
    points, sampler = adapter.sample_points_with_diagnostics(
        model,
        job["points_per_seed"],
        seed=job["seed"],
    )
    weights = adapter.importance_weights(points)
    normalized_weights = weights / float(np.sum(weights))
    metrics = adapter.h_metrics(points, artifact)
    residuals = adapter.residual_values(points, metrics)
    statistics = standard_errors(residuals, weights)
    cluster_ids = adapter.sampling_cluster_ids(points)
    cluster_statistics = sampling_cluster_statistics(
        normalized_weights,
        cluster_ids,
    )
    metric_eigenvalues = np.linalg.eigvalsh(
        np.asarray(metrics, dtype=np.complex128)
    )
    return {
        "seed": int(job["seed"]),
        **statistics,
        "minimum_metric_eigenvalue": float(np.min(metric_eigenvalues)),
        "minimum_jacobian_singular_value": float(
            min(adapter.point_jacobian_min_singular_value(point) for point in points)
        ),
        **cluster_statistics,
        "sampler": sampler,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def aggregate_metric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
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
            "minimum_jacobian_singular_value": float(
                min(row["minimum_jacobian_singular_value"] for row in rows)
            ),
            "minimum_point_effective_sample_size": float(
                min(row["point_effective_sample_size"] for row in rows)
            ),
            "minimum_cluster_weight_effective_sample_size": float(
                min(row["cluster_weight_effective_sample_size"] for row in rows)
            ),
        }
    )
    return output


def main() -> None:
    args = parse_args()
    if args.points_per_seed <= 0 or args.workers <= 0:
        raise SystemExit("points-per-seed and workers must be positive")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise SystemExit("seeds must be non-empty and distinct")
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
            "adapter": args.adapter,
            "model_seed": int(args.model_seed),
            "exact_model": not args.inexact_model,
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
    aggregate = aggregate_metric_rows(rows)
    gates = {
        "finite_metric_statistics": bool(
            all(
                np.isfinite(row["sigma"])
                and np.isfinite(row["weighted_centered_log_ma_rms"])
                for row in rows
            )
        ),
        "positive_metric": aggregate["minimum_metric_eigenvalue"] > 0,
        "point_effective_sample_size": (
            aggregate["minimum_point_effective_sample_size"]
            >= args.minimum_point_ess
        ),
        "complete_sampling_clusters": all(
            row["sampler"]["all_returned_clusters_complete"] for row in rows
        ),
        "expected_root_count_on_accepted_fibres": all(
            sum(row["sampler"].get("accepted_root_count_histogram", {}).values())
            == row["sampler"]["accepted_clusters"]
            for row in rows
        ),
        "importance_weights_not_used_for_acceptance": all(
            not row["sampler"]["importance_weight_used_for_acceptance"]
            for row in rows
        ),
    }
    adapter = get_adapter(args.adapter)
    model = adapter.make_model(args.model_seed, exact=not args.inexact_model)
    output = {
        "schema_version": 1,
        "description": (
            "Parallel large-sample fresh-seed gCICY metric audit with complete-"
            "fibre and cluster-weight diagnostics."
        ),
        "adapter": adapter.key,
        "configuration": adapter.configuration.to_dict(),
        "model_seed": int(args.model_seed),
        "exact_model": not args.inexact_model,
        "model": adapter.model_metadata(model),
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
        "aggregate": aggregate,
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
    print(json.dumps({"success": output["success"], **aggregate}, indent=2))
    print(f"wrote {output_path}")
    if not output["success"]:
        raise SystemExit("large-sample metric audit failed")


if __name__ == "__main__":
    main()
