#!/usr/bin/env python3
"""Independent audit of the generic-model global H-matrix artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric import (  # noqa: E402
    all_projective_charts,
    generic_baseline_metrics,
    generic_global_h_metric,
    generic_global_h_metrics,
    generic_importance_weights,
    generic_effective_sample_size,
    generic_monge_ampere_log_error,
    generic_point_in_chart,
    generic_residual_stats,
    generic_weighted_residual_stats,
    make_exact_generic_model,
    make_generic_model,
    sample_generic_gcicy_points,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=ROOT / "outputs" / "gcicy_generic_global_h_metric.npz")
    parser.add_argument("--seeds", default="4101,4102,4103,4104,4105,4106,4107,4108")
    parser.add_argument("--points", type=int, default=512)
    parser.add_argument("--chart-points-per-seed", type=int, default=4)
    parser.add_argument("--min-selected", type=float, default=1e-4)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "gcicy_generic_global_h_metric_audit.json")
    return parser.parse_args()


def parse_seeds(text: str) -> list[int]:
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def main() -> None:
    args = parse_args()
    artifact = np.load(args.artifact)
    model_seed = int(artifact["generic_model_seed"])
    exact_model = bool(artifact["exact_integer_coefficients"]) if "exact_integer_coefficients" in artifact.files else False
    model = make_exact_generic_model(model_seed) if exact_model else make_generic_model(model_seed)
    if not np.allclose(model.p1_coefficients, artifact["p1_coefficients"]):
        raise SystemExit("artifact p1 coefficients do not match its model seed")
    if not np.allclose(model.p2_tensor, artifact["p2_tensor"]):
        raise SystemExit("artifact p2 coefficients do not match its model seed")
    exponents = artifact["global_section_exponents"]
    h_matrix = artifact["global_h_matrix"]
    normalization = float(artifact["global_section_normalization"])

    rows = []
    max_chart_ma_error = 0.0
    charts_seen = set()
    min_sampled_jacobian_singular_value = float("inf")
    for seed in parse_seeds(args.seeds):
        points = sample_generic_gcicy_points(model, args.points, seed=seed)
        baseline_metrics = generic_baseline_metrics(points)
        candidate_metrics = generic_global_h_metrics(points, exponents, h_matrix, normalization=normalization)
        baseline = generic_residual_stats(points, baseline_metrics)
        candidate = generic_residual_stats(points, candidate_metrics)
        importance_weights = generic_importance_weights(points)
        baseline_weighted = generic_weighted_residual_stats(points, baseline_metrics, importance_weights)
        candidate_weighted = generic_weighted_residual_stats(points, candidate_metrics, importance_weights)
        passed = (
            np.isfinite(candidate.rms)
            and np.isfinite(candidate_weighted.rms)
            and candidate.min_eigenvalue > 0
            and candidate.rms < baseline.rms
            and candidate_weighted.rms < baseline_weighted.rms
        )
        rows.append(
            {
                "seed": seed,
                "points": args.points,
                "baseline_rms": baseline.rms,
                "candidate_rms": candidate.rms,
                "baseline_weighted_rms": baseline_weighted.rms,
                "candidate_weighted_rms": candidate_weighted.rms,
                "importance_effective_sample_size": generic_effective_sample_size(importance_weights),
                "candidate_max_abs": candidate.max_abs,
                "candidate_min_eigenvalue": candidate.min_eigenvalue,
                "passed": bool(passed),
            }
        )
        min_sampled_jacobian_singular_value = min(
            min_sampled_jacobian_singular_value,
            min(point.jacobian_min_singular_value for point in points),
        )

        for point in points[: args.chart_points_per_seed]:
            reference_metric = generic_global_h_metric(point, exponents, h_matrix, normalization=normalization)
            reference_ma = generic_monge_ampere_log_error(point, reference_metric)
            for chart in all_projective_charts():
                if min(abs(point.x[chart[0]]), abs(point.y[chart[1]]), abs(point.z[chart[2]])) <= args.min_selected:
                    continue
                candidate_point = generic_point_in_chart(model, point.x, point.y, point.z, chart)
                metric = generic_global_h_metric(
                    candidate_point,
                    exponents,
                    h_matrix,
                    normalization=normalization,
                )
                candidate_ma = generic_monge_ampere_log_error(candidate_point, metric)
                max_chart_ma_error = max(max_chart_ma_error, abs(candidate_ma - reference_ma))
                charts_seen.add(chart)

    summary = {
        "description": "Fresh-seed audit of the generic smooth-candidate gCICY global H metric.",
        "artifact": str(args.artifact),
        "model_seed": model_seed,
        "exact_integer_coefficients": exact_model,
        "points_per_seed": args.points,
        "seeds": parse_seeds(args.seeds),
        "all_passed": all(row["passed"] for row in rows),
        "mean_candidate_rms": float(np.mean([row["candidate_rms"] for row in rows])),
        "max_candidate_rms": float(np.max([row["candidate_rms"] for row in rows])),
        "mean_candidate_weighted_rms": float(np.mean([row["candidate_weighted_rms"] for row in rows])),
        "max_candidate_weighted_rms": float(np.max([row["candidate_weighted_rms"] for row in rows])),
        "min_candidate_eigenvalue": float(np.min([row["candidate_min_eigenvalue"] for row in rows])),
        "min_sampled_jacobian_singular_value": min_sampled_jacobian_singular_value,
        "projective_charts_seen": len(charts_seen),
        "max_trained_h_chart_ma_error": float(max_chart_ma_error),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    for row in rows:
        print(
            f"seed {row['seed']}: rms {row['baseline_rms']:.6e} -> {row['candidate_rms']:.6e}, "
            f"weighted {row['baseline_weighted_rms']:.6e} -> {row['candidate_weighted_rms']:.6e}, "
            f"max {row['candidate_max_abs']:.6e}, min eig {row['candidate_min_eigenvalue']:.6e}, "
            f"passed={row['passed']}"
        )
    print(f"trained H projective charts: {len(charts_seen)} / 24")
    print(f"max trained H chart MA error: {max_chart_ma_error:.6e}")
    print(f"minimum sampled Jacobian singular value: {min_sampled_jacobian_singular_value:.6e}")
    print(f"wrote {args.out}")
    if not summary["all_passed"]:
        raise SystemExit("generic global H artifact failed a fresh seed")
    if len(charts_seen) != 24 or max_chart_ma_error > 1e-8:
        raise SystemExit("trained generic H metric failed projective-chart consistency")
    if min_sampled_jacobian_singular_value < 1e-7:
        raise SystemExit("fresh generic samples approached a rank-deficient defining Jacobian")


if __name__ == "__main__":
    main()
