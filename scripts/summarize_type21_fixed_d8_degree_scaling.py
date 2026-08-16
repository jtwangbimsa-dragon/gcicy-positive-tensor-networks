#!/usr/bin/env python3
"""Summarize the registered X21 fixed-D finite-range degree ladder."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


DEGREES = (8, 12, 16, 20, 24)
REPLICATES = (1, 2, 3)
METRICS = (
    "sigma",
    "chi",
    "absolute_log_ratio_q999",
    "absolute_log_ratio_cvar_1pct",
    "maximum_absolute_log_ratio",
    "minimum_metric_eigenvalue",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--degrees",
        type=int,
        nargs="+",
        default=list(DEGREES),
        help="Completed degrees to summarize (default: 8 12 16 20 24).",
    )
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean_sd(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "sample_standard_deviation": statistics.stdev(values),
        "values": values,
    }


def linear_slope(xs: list[float], ys: list[float]) -> float:
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator


def stage_summaries(arm_dir: Path) -> list[dict]:
    rounds = sorted(arm_dir.glob("primary_round*/summary.json"))
    rounds += sorted(arm_dir.glob("precision_round*/summary.json"))
    if rounds:
        return [load(path) for path in rounds]
    return [
        load(arm_dir / "primary_plateau_summary.json"),
        load(arm_dir / "plateau_summary.json"),
    ]


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    degrees = tuple(args.degrees)
    if len(degrees) < 2 or len(set(degrees)) != len(degrees):
        raise SystemExit("--degrees must contain at least two distinct values")
    if tuple(sorted(degrees)) != degrees:
        raise SystemExit("--degrees must be strictly increasing")
    rows: list[dict] = []

    for degree in degrees:
        for replicate in REPLICATES:
            arm_dir = run_dir / f"k{degree}" / f"replicate_{replicate}"
            audit = load(arm_dir / "final_common" / "audit.json")
            bilateral = load(arm_dir / "final_common" / "bilateral.json")
            model_label = f"k{degree}_replicate_{replicate}"
            metrics = bilateral["models"][model_label]["metrics"]
            stages = stage_summaries(arm_dir)
            if any(stage["termination_reason"] != "validation_plateau" for stage in stages):
                raise SystemExit(f"non-plateau stage: k={degree}, replicate={replicate}")
            if metrics["nonpositive_metric_count"] != 0:
                raise SystemExit(f"nonpositive metric: k={degree}, replicate={replicate}")
            model_path = arm_dir / "plateau_model.pt"
            rows.append(
                {
                    "degree": degree,
                    "bond_dimension": 8,
                    "replicate": replicate,
                    "trainable_real_parameter_count": audit[
                        "trainable_real_parameter_count"
                    ],
                    "optimizer_wall_seconds": sum(
                        stage["runtime_seconds"] for stage in stages
                    ),
                    "evaluation_wall_seconds": audit["runtime_seconds"],
                    "maximum_allocated_bytes": max(
                        stage["device_memory"]["maximum_allocated_bytes"]
                        for stage in stages
                    ),
                    "model": str(model_path),
                    "model_sha256": sha256(model_path),
                    "metrics": {metric: metrics[metric] for metric in METRICS},
                }
            )

    by_degree: dict[str, dict] = {}
    for degree in degrees:
        degree_rows = [row for row in rows if row["degree"] == degree]
        parameter_counts = {
            row["trainable_real_parameter_count"] for row in degree_rows
        }
        if len(parameter_counts) != 1:
            raise SystemExit(f"parameter count differs across k={degree} replicates")
        by_degree[str(degree)] = {
            "trainable_real_parameter_count": parameter_counts.pop(),
            "optimizer_wall_seconds": mean_sd(
                [row["optimizer_wall_seconds"] for row in degree_rows]
            ),
            "evaluation_wall_seconds": mean_sd(
                [row["evaluation_wall_seconds"] for row in degree_rows]
            ),
            "maximum_allocated_bytes": mean_sd(
                [row["maximum_allocated_bytes"] for row in degree_rows]
            ),
            "metrics": {
                metric: mean_sd([row["metrics"][metric] for row in degree_rows])
                for metric in METRICS
            },
        }

    adjacent: dict[str, dict] = {}
    for left, right in zip(degrees[:-1], degrees[1:]):
        name = f"k{left}_to_k{right}"
        records = [
            load(run_dir / f"bootstrap_k{left}_to_k{right}_replicate_{rep}.json")
            for rep in REPLICATES
        ]
        adjacent[name] = {}
        for metric in (
            "sigma",
            "chi",
            "absolute_log_ratio_q999",
            "absolute_log_ratio_cvar_1pct",
        ):
            comparisons = [record["comparisons"][metric] for record in records]
            relative = [item["relative_improvement"] for item in comparisons]
            adjacent[name][metric] = {
                "mean_relative_improvement": statistics.mean(relative),
                "sample_standard_deviation_relative_improvement": statistics.stdev(
                    relative
                ),
                "per_replicate_relative_improvement": relative,
                "per_replicate_bootstrap_95pct_confidence_interval": [
                    item["bootstrap_95pct_confidence_interval"]
                    for item in comparisons
                ],
                "all_intervals_strictly_positive": all(
                    item["bootstrap_95pct_confidence_interval"][0] > 0
                    for item in comparisons
                ),
            }

    parameter_values = [
        by_degree[str(degree)]["trainable_real_parameter_count"]
        for degree in degrees
    ]
    sigma_values = [
        by_degree[str(degree)]["metrics"]["sigma"]["mean"] for degree in degrees
    ]
    chi_values = [
        by_degree[str(degree)]["metrics"]["chi"]["mean"] for degree in degrees
    ]
    evaluation_values = [
        by_degree[str(degree)]["evaluation_wall_seconds"]["mean"]
        for degree in degrees
    ]

    payload = {
        "schema": "type21-fixed-d8-finite-range-degree-scaling-summary-v2",
        "claim_boundary": (
            "Finite-range empirical evidence at fixed D=8; no k-to-infinity "
            "or fixed-accuracy asymptotic claim is inferred."
        ),
        "degrees": list(degrees),
        "replicates_per_degree": len(REPLICATES),
        "point_count": rows[0]["metrics"].get("point_count", 196608),
        "all_sampled_metrics_positive": all(
            row["metrics"]["minimum_metric_eigenvalue"] > 0 for row in rows
        ),
        "rows": rows,
        "by_degree": by_degree,
        "adjacent_degree_comparisons": adjacent,
        "finite_range_diagnostics": {
            "parameter_count_is_strictly_increasing": all(
                b > a for a, b in zip(parameter_values[:-1], parameter_values[1:])
            ),
            "mean_sigma_is_monotone_nonincreasing": all(
                b <= a for a, b in zip(sigma_values[:-1], sigma_values[1:])
            ),
            "mean_chi_is_monotone_nonincreasing": all(
                b <= a for a, b in zip(chi_values[:-1], chi_values[1:])
            ),
            "log_parameter_vs_log_degree_slope": linear_slope(
                [math.log(value) for value in degrees],
                [math.log(value) for value in parameter_values],
            ),
            "log_evaluation_time_vs_log_degree_slope": linear_slope(
                [math.log(value) for value in degrees],
                [math.log(value) for value in evaluation_values],
            ),
            "log_mean_sigma_vs_degree_slope": linear_slope(
                list(degrees), [math.log(value) for value in sigma_values]
            ),
            f"sigma_reduction_k{degrees[0]}_to_k{degrees[-1]}": (
                1.0 - sigma_values[-1] / sigma_values[0]
            ),
            f"chi_reduction_k{degrees[0]}_to_k{degrees[-1]}": (
                1.0 - chi_values[-1] / chi_values[0]
            ),
        },
        "final_pool": {
            "path": str(run_dir / "X21_scaling_final_seed86807_n196608.npz"),
            "sha256": sha256(run_dir / "X21_scaling_final_seed86807_n196608.npz"),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
