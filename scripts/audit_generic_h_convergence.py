#!/usr/bin/env python3
"""Paired fresh-seed convergence audit for several exact-model H artifacts."""

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


DEFAULT_ARTIFACTS = ",".join(
    [
        str(ROOT / "outputs" / "gcicy_generic_global_h_metric.npz"),
        str(ROOT / "outputs" / "gcicy_generic_global_h_metric_k2_rank40.npz"),
        str(ROOT / "outputs" / "gcicy_generic_global_h_metric_k3_rank64.npz"),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default=DEFAULT_ARTIFACTS)
    parser.add_argument("--seeds", default="6101,6102,6103,6104,6105,6106,6107,6108")
    parser.add_argument("--points", type=int, default=512)
    parser.add_argument("--chart-points-per-seed", type=int, default=1)
    parser.add_argument("--min-selected", type=float, default=1e-4)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs" / "gcicy_generic_h_convergence_audit.json",
    )
    return parser.parse_args()


def parse_ints(text: str) -> list[int]:
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def parse_paths(text: str) -> list[Path]:
    return [Path(value.strip()).expanduser().resolve() for value in text.split(",") if value.strip()]


def load_artifacts(paths: list[Path]):
    loaded = []
    for path in paths:
        if not path.exists():
            raise SystemExit(f"missing artifact: {path}")
        artifact = np.load(path)
        degree = tuple(int(value) for value in artifact["global_section_degree"])
        if degree[0] != degree[1] or degree[1] != degree[2]:
            raise SystemExit(f"artifact does not use degree (k,k,k): {path}")
        loaded.append(
            {
                "path": path,
                "artifact": artifact,
                "degree": degree,
                "k": degree[0],
                "exponents": artifact["global_section_exponents"],
                "h_matrix": artifact["global_h_matrix"],
                "normalization": float(artifact["global_section_normalization"]),
            }
        )
    loaded.sort(key=lambda row: row["k"])
    if len({row["k"] for row in loaded}) != len(loaded):
        raise SystemExit("artifacts must have distinct section degrees")
    return loaded


def confidence_interval(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if len(array) < 2:
        return float(array[0]), float(array[0])
    half_width = 1.96 * float(np.std(array, ddof=1)) / np.sqrt(len(array))
    return float(np.mean(array) - half_width), float(np.mean(array) + half_width)


def main() -> None:
    args = parse_args()
    rows = load_artifacts(parse_paths(args.artifacts))
    seeds = parse_ints(args.seeds)
    first = rows[0]["artifact"]
    model_seed = int(first["generic_model_seed"])
    exact_model = bool(first["exact_integer_coefficients"])
    model = make_exact_generic_model(model_seed) if exact_model else make_generic_model(model_seed)
    for row in rows:
        artifact = row["artifact"]
        if int(artifact["generic_model_seed"]) != model_seed:
            raise SystemExit("all artifacts must use the same model seed")
        if not np.allclose(model.p1_coefficients, artifact["p1_coefficients"]):
            raise SystemExit(f"p1 coefficients do not match: {row['path']}")
        if not np.allclose(model.p2_tensor, artifact["p2_tensor"]):
            raise SystemExit(f"p2 coefficients do not match: {row['path']}")

    by_k: dict[int, list[dict]] = {row["k"]: [] for row in rows}
    seed_rows = []
    min_sampled_jacobian = float("inf")
    chart_errors = {row["k"]: 0.0 for row in rows}
    charts_seen = {row["k"]: set() for row in rows}

    for seed in seeds:
        points = sample_generic_gcicy_points(model, args.points, seed=seed)
        min_sampled_jacobian = min(
            min_sampled_jacobian,
            min(point.jacobian_min_singular_value for point in points),
        )
        baseline_metrics = generic_baseline_metrics(points)
        baseline = generic_residual_stats(points, baseline_metrics)
        importance_weights = generic_importance_weights(points)
        baseline_weighted = generic_weighted_residual_stats(points, baseline_metrics, importance_weights)
        result_row = {
            "seed": seed,
            "baseline_rms": baseline.rms,
            "baseline_weighted_rms": baseline_weighted.rms,
            "importance_effective_sample_size": generic_effective_sample_size(importance_weights),
            "metrics": {},
        }
        for row in rows:
            metrics = generic_global_h_metrics(
                points,
                row["exponents"],
                row["h_matrix"],
                normalization=row["normalization"],
            )
            stats = generic_residual_stats(points, metrics)
            weighted_stats = generic_weighted_residual_stats(points, metrics, importance_weights)
            metric_row = {
                "rms": stats.rms,
                "weighted_rms": weighted_stats.rms,
                "max_abs": stats.max_abs,
                "min_eigenvalue": stats.min_eigenvalue,
                "mean_log_error": stats.mean_log_error,
            }
            result_row["metrics"][str(row["k"])] = metric_row
            by_k[row["k"]].append(metric_row)

        result_row["strictly_decreasing_unweighted"] = all(
            result_row["metrics"][str(left["k"])]["rms"]
            > result_row["metrics"][str(right["k"])]["rms"]
            for left, right in zip(rows, rows[1:])
        )
        result_row["strictly_decreasing_weighted"] = all(
            result_row["metrics"][str(left["k"])]["weighted_rms"]
            > result_row["metrics"][str(right["k"])]["weighted_rms"]
            for left, right in zip(rows, rows[1:])
        )
        seed_rows.append(result_row)

        for point in points[: args.chart_points_per_seed]:
            for row in rows:
                reference_metric = generic_global_h_metric(
                    point,
                    row["exponents"],
                    row["h_matrix"],
                    normalization=row["normalization"],
                )
                reference_ma = generic_monge_ampere_log_error(point, reference_metric)
                for chart in all_projective_charts():
                    if min(abs(point.x[chart[0]]), abs(point.y[chart[1]]), abs(point.z[chart[2]])) <= args.min_selected:
                        continue
                    chart_point = generic_point_in_chart(model, point.x, point.y, point.z, chart)
                    chart_metric = generic_global_h_metric(
                        chart_point,
                        row["exponents"],
                        row["h_matrix"],
                        normalization=row["normalization"],
                    )
                    chart_ma = generic_monge_ampere_log_error(chart_point, chart_metric)
                    chart_errors[row["k"]] = max(chart_errors[row["k"]], abs(chart_ma - reference_ma))
                    charts_seen[row["k"]].add(chart)

    aggregate = []
    for row in rows:
        metric_rows = by_k[row["k"]]
        rms_values = [metric_row["rms"] for metric_row in metric_rows]
        weighted_rms_values = [metric_row["weighted_rms"] for metric_row in metric_rows]
        ci_low, ci_high = confidence_interval(rms_values)
        weighted_ci_low, weighted_ci_high = confidence_interval(weighted_rms_values)
        aggregate.append(
            {
                "k": row["k"],
                "degree": list(row["degree"]),
                "artifact": str(row["path"]),
                "section_count": int(len(row["exponents"])),
                "mean_rms": float(np.mean(rms_values)),
                "std_rms": float(np.std(rms_values, ddof=1)),
                "rms_95_percent_ci": [ci_low, ci_high],
                "max_seed_rms": float(np.max(rms_values)),
                "mean_weighted_rms": float(np.mean(weighted_rms_values)),
                "std_weighted_rms": float(np.std(weighted_rms_values, ddof=1)),
                "weighted_rms_95_percent_ci": [weighted_ci_low, weighted_ci_high],
                "max_seed_weighted_rms": float(np.max(weighted_rms_values)),
                "min_metric_eigenvalue": float(np.min([metric_row["min_eigenvalue"] for metric_row in metric_rows])),
                "projective_charts_seen": len(charts_seen[row["k"]]),
                "max_chart_ma_error": float(chart_errors[row["k"]]),
            }
        )

    ks = np.asarray([row["k"] for row in aggregate], dtype=float)
    rms = np.asarray([row["mean_rms"] for row in aggregate], dtype=float)
    weighted_rms = np.asarray([row["mean_weighted_rms"] for row in aggregate], dtype=float)
    slope = float(np.polyfit(np.log(ks), np.log(rms), 1)[0]) if len(ks) >= 2 else float("nan")
    weighted_slope = (
        float(np.polyfit(np.log(ks), np.log(weighted_rms), 1)[0]) if len(ks) >= 2 else float("nan")
    )
    summary = {
        "description": "Paired fresh-seed convergence audit for exact-model global H metrics.",
        "model_seed": model_seed,
        "exact_integer_coefficients": exact_model,
        "seeds": seeds,
        "points_per_seed": args.points,
        "minimum_sampled_defining_jacobian_singular_value": min_sampled_jacobian,
        "strictly_decreasing_unweighted_on_every_seed": all(
            row["strictly_decreasing_unweighted"] for row in seed_rows
        ),
        "strictly_decreasing_weighted_on_every_seed": all(
            row["strictly_decreasing_weighted"] for row in seed_rows
        ),
        "empirical_unweighted_log_log_slope": slope,
        "empirical_weighted_log_log_slope": weighted_slope,
        "aggregate": aggregate,
        "rows": seed_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for row in aggregate:
        print(
            f"k={row['k']}: sections={row['section_count']}, mean rms={row['mean_rms']:.6e}, "
            f"weighted={row['mean_weighted_rms']:.6e}, "
            f"95% CI=[{row['rms_95_percent_ci'][0]:.6e}, {row['rms_95_percent_ci'][1]:.6e}], "
            f"min eig={row['min_metric_eigenvalue']:.6e}, charts={row['projective_charts_seen']}/24"
        )
    print(
        "strictly decreasing on every seed: "
        f"unweighted={summary['strictly_decreasing_unweighted_on_every_seed']}, "
        f"weighted={summary['strictly_decreasing_weighted_on_every_seed']}"
    )
    print(f"empirical log-log slopes: unweighted={slope:.6f}, weighted={weighted_slope:.6f}")
    print(f"wrote {args.out}")
    if any(row["min_metric_eigenvalue"] <= 0 for row in aggregate):
        raise SystemExit("at least one audited metric is not positive")
    if any(row["projective_charts_seen"] != 24 or row["max_chart_ma_error"] > 1e-8 for row in aggregate):
        raise SystemExit("at least one artifact failed projective-chart consistency")


if __name__ == "__main__":
    main()
