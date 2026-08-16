#!/usr/bin/env python3
"""Check baseline metric and residue consistency on the generic gCICY model."""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric import (  # noqa: E402
    all_projective_charts,
    generic_baseline_metric,
    generic_importance_log_weight,
    generic_monge_ampere_log_error,
    generic_point_in_chart,
    make_generic_model,
    make_exact_generic_model,
    sample_generic_gcicy_points,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-seed", type=int, default=20260711)
    parser.add_argument("--exact-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--points", type=int, default=24)
    parser.add_argument("--seed", type=int, default=3101)
    parser.add_argument("--min-selected", type=float, default=1e-4)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "gcicy_generic_metric_geometry_check.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = make_exact_generic_model(args.model_seed) if args.exact_model else make_generic_model(args.model_seed)
    points = sample_generic_gcicy_points(model, args.points, seed=args.seed)
    max_projective_ma_error = 0.0
    max_implicit_ma_error = 0.0
    max_projective_importance_error = 0.0
    max_implicit_importance_error = 0.0
    min_metric_eigenvalue = float("inf")
    projective_charts = set()
    implicit_charts = set()

    for point in points:
        reference_metric = generic_baseline_metric(point)
        reference_ma = generic_monge_ampere_log_error(point, reference_metric)
        reference_importance = generic_importance_log_weight(point)
        min_metric_eigenvalue = min(min_metric_eigenvalue, float(np.linalg.eigvalsh(reference_metric)[0]))

        for chart in all_projective_charts():
            if min(abs(point.x[chart[0]]), abs(point.y[chart[1]]), abs(point.z[chart[2]])) <= args.min_selected:
                continue
            candidate = generic_point_in_chart(model, point.x, point.y, point.z, chart)
            metric = generic_baseline_metric(candidate)
            candidate_ma = generic_monge_ampere_log_error(candidate, metric)
            max_projective_ma_error = max(max_projective_ma_error, abs(candidate_ma - reference_ma))
            max_projective_importance_error = max(
                max_projective_importance_error,
                abs(generic_importance_log_weight(candidate) - reference_importance),
            )
            min_metric_eigenvalue = min(min_metric_eigenvalue, float(np.linalg.eigvalsh(metric)[0]))
            projective_charts.add(chart)

        equations_point = generic_point_in_chart(
            model,
            point.x,
            point.y,
            point.z,
            point.projective_chart,
        )
        for independent in combinations(range(7), 3):
            try:
                candidate = generic_point_in_chart(
                    model,
                    point.x,
                    point.y,
                    point.z,
                    point.projective_chart,
                    independent=independent,
                )
            except FloatingPointError:
                continue
            metric = generic_baseline_metric(candidate)
            candidate_ma = generic_monge_ampere_log_error(candidate, metric)
            max_implicit_ma_error = max(max_implicit_ma_error, abs(candidate_ma - reference_ma))
            max_implicit_importance_error = max(
                max_implicit_importance_error,
                abs(generic_importance_log_weight(candidate) - reference_importance),
            )
            implicit_charts.add(independent)

    summary = {
        "description": "Generic gCICY baseline metric and Poincare-residue consistency across projective and implicit charts.",
        "model_seed": args.model_seed,
        "exact_integer_coefficients": args.exact_model,
        "sample_seed": args.seed,
        "points": args.points,
        "projective_charts_seen": len(projective_charts),
        "implicit_coordinate_choices_seen": len(implicit_charts),
        "max_projective_monge_ampere_error": float(max_projective_ma_error),
        "max_implicit_monge_ampere_error": float(max_implicit_ma_error),
        "max_projective_importance_log_weight_error": float(max_projective_importance_error),
        "max_implicit_importance_log_weight_error": float(max_implicit_importance_error),
        "minimum_metric_eigenvalue": min_metric_eigenvalue,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"projective charts seen: {len(projective_charts)} / 24")
    print(f"implicit coordinate choices seen: {len(implicit_charts)} / 35")
    print(f"max projective MA error: {max_projective_ma_error:.6e}")
    print(f"max implicit MA error: {max_implicit_ma_error:.6e}")
    print(f"max projective importance-weight error: {max_projective_importance_error:.6e}")
    print(f"max implicit importance-weight error: {max_implicit_importance_error:.6e}")
    print(f"minimum baseline metric eigenvalue: {min_metric_eigenvalue:.6e}")
    if len(projective_charts) != 24 or len(implicit_charts) != 35:
        raise SystemExit("generic metric check did not cover the complete atlas")
    if max_projective_ma_error > 1e-8 or max_implicit_ma_error > 1e-8:
        raise SystemExit("generic metric/residue consistency check failed")
    if max_projective_importance_error > 1e-8 or max_implicit_importance_error > 1e-8:
        raise SystemExit("generic importance-weight consistency check failed")
    if min_metric_eigenvalue <= 0:
        raise SystemExit("generic baseline metric lost positive definiteness")


if __name__ == "__main__":
    main()
