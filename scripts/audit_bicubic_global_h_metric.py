#!/usr/bin/env python3
"""Fresh-seed atlas and residual audit of a bicubic H-metric artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.bicubic_model import (  # noqa: E402
    bicubic_baseline_metrics,
    bicubic_global_h_metric,
    bicubic_global_h_metrics,
    bicubic_holomorphic_volume_log_density,
    bicubic_importance_log_weight,
    bicubic_importance_weights,
    bicubic_point_in_chart,
    bicubic_residual_stats,
    make_exact_bicubic_model,
    sample_bicubic_points,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=ROOT / "outputs" / "bicubic_global_h_metric_k1_weighted.npz",
    )
    parser.add_argument("--seeds", default="9101,9102,9103,9104,9105,9106,9107,9108")
    parser.add_argument("--points", type=int, default=512)
    parser.add_argument("--atlas-points-per-seed", type=int, default=2)
    parser.add_argument("--min-selected", type=float, default=1e-4)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs" / "bicubic_global_h_metric_k1_weighted_audit.json",
    )
    return parser.parse_args()


def parse_seeds(text: str) -> list[int]:
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def ma_error(point, metric) -> float:
    eigenvalues = np.linalg.eigvalsh(metric)
    if eigenvalues[0] <= 0:
        return float("nan")
    return float(np.sum(np.log(eigenvalues)) - bicubic_holomorphic_volume_log_density(point))


def effective_sample_size(weights: np.ndarray) -> float:
    return float(np.sum(weights) ** 2 / np.sum(weights**2))


def main() -> None:
    args = parse_args()
    artifact = np.load(args.artifact)
    model_seed = int(artifact["bicubic_model_seed"])
    model = make_exact_bicubic_model(model_seed)
    if not np.allclose(model.coefficients, artifact["bicubic_coefficients"]):
        raise SystemExit("bicubic artifact coefficients do not match its model seed")
    exponents = artifact["global_section_exponents"]
    h_matrix = artifact["global_h_matrix"]
    normalization = float(artifact["global_section_normalization"])

    rows = []
    projective_charts = set()
    implicit_choices = set()
    max_projective_ma_error = 0.0
    max_implicit_ma_error = 0.0
    max_importance_error = 0.0
    min_jacobian_norm = float("inf")
    for seed in parse_seeds(args.seeds):
        points = sample_bicubic_points(model, args.points, seed=seed)
        baseline_metrics = bicubic_baseline_metrics(points)
        candidate_metrics = bicubic_global_h_metrics(
            points, exponents, h_matrix, normalization=normalization
        )
        importance = bicubic_importance_weights(points)
        baseline = bicubic_residual_stats(points, baseline_metrics)
        candidate = bicubic_residual_stats(points, candidate_metrics)
        baseline_weighted = bicubic_residual_stats(points, baseline_metrics, importance)
        candidate_weighted = bicubic_residual_stats(points, candidate_metrics, importance)
        passed = (
            np.isfinite(candidate.rms)
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
                "importance_effective_sample_size": effective_sample_size(importance),
                "candidate_min_eigenvalue": candidate.min_eigenvalue,
                "passed": bool(passed),
            }
        )
        min_jacobian_norm = min(min_jacobian_norm, min(point.jacobian_norm for point in points))

        for point in points[: args.atlas_points_per_seed]:
            reference_metric = bicubic_global_h_metric(
                point, exponents, h_matrix, normalization=normalization
            )
            reference_ma = ma_error(point, reference_metric)
            reference_importance = bicubic_importance_log_weight(point)
            for x_index in range(3):
                for y_index in range(3):
                    if min(abs(point.x[x_index]), abs(point.y[y_index])) <= args.min_selected:
                        continue
                    chart = (x_index, y_index)
                    candidate_point = bicubic_point_in_chart(model, point.x, point.y, chart)
                    metric = bicubic_global_h_metric(
                        candidate_point, exponents, h_matrix, normalization=normalization
                    )
                    max_projective_ma_error = max(
                        max_projective_ma_error, abs(ma_error(candidate_point, metric) - reference_ma)
                    )
                    max_importance_error = max(
                        max_importance_error,
                        abs(bicubic_importance_log_weight(candidate_point) - reference_importance),
                    )
                    projective_charts.add(chart)
            for dependent in range(4):
                try:
                    candidate_point = bicubic_point_in_chart(
                        model,
                        point.x,
                        point.y,
                        point.projective_chart,
                        dependent_index=dependent,
                    )
                except FloatingPointError:
                    continue
                metric = bicubic_global_h_metric(
                    candidate_point, exponents, h_matrix, normalization=normalization
                )
                max_implicit_ma_error = max(
                    max_implicit_ma_error, abs(ma_error(candidate_point, metric) - reference_ma)
                )
                max_importance_error = max(
                    max_importance_error,
                    abs(bicubic_importance_log_weight(candidate_point) - reference_importance),
                )
                implicit_choices.add(dependent)

    summary = {
        "description": "Fresh-seed weighted and atlas audit of an exact bicubic H metric.",
        "artifact": str(args.artifact),
        "model_seed": model_seed,
        "degree": int(artifact["global_section_degree"]),
        "seeds": parse_seeds(args.seeds),
        "points_per_seed": args.points,
        "all_passed": all(row["passed"] for row in rows),
        "mean_candidate_rms": float(np.mean([row["candidate_rms"] for row in rows])),
        "mean_candidate_weighted_rms": float(
            np.mean([row["candidate_weighted_rms"] for row in rows])
        ),
        "min_candidate_eigenvalue": float(
            np.min([row["candidate_min_eigenvalue"] for row in rows])
        ),
        "minimum_sampled_jacobian_norm": min_jacobian_norm,
        "projective_charts_seen": len(projective_charts),
        "implicit_coordinate_choices_seen": len(implicit_choices),
        "max_projective_ma_error": max_projective_ma_error,
        "max_implicit_ma_error": max_implicit_ma_error,
        "max_importance_log_weight_error": max_importance_error,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print(
            f"seed {row['seed']}: rms {row['baseline_rms']:.6e}->{row['candidate_rms']:.6e}, "
            f"weighted {row['baseline_weighted_rms']:.6e}->{row['candidate_weighted_rms']:.6e}, "
            f"min eig={row['candidate_min_eigenvalue']:.6e}, passed={row['passed']}"
        )
    print(
        f"atlas: projective={len(projective_charts)}/9, implicit={len(implicit_choices)}/4, "
        f"max MA errors={max_projective_ma_error:.3e}/{max_implicit_ma_error:.3e}"
    )
    print(f"wrote {args.out}")
    if not summary["all_passed"]:
        raise SystemExit("bicubic artifact failed fresh-seed residual gates")
    if len(projective_charts) != 9 or len(implicit_choices) != 4:
        raise SystemExit("bicubic audit did not cover the complete atlas")
    if max(max_projective_ma_error, max_implicit_ma_error, max_importance_error) > 1e-8:
        raise SystemExit("bicubic artifact failed atlas consistency")


if __name__ == "__main__":
    main()
