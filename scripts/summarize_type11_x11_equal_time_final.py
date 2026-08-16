#!/usr/bin/env python3
"""Summarize the X11 equal-time final comparison on one common sample."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


METRICS = (
    "sigma",
    "chi",
    "absolute_log_ratio_q999",
    "absolute_log_ratio_cvar_1pct",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "sample_standard_deviation": float(array.std(ddof=1)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def portable_path(path: Path) -> str:
    try:
        return path.relative_to(Path(__file__).resolve().parents[1]).as_posix()
    except ValueError:
        return path.as_posix()


def main() -> None:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    rows = []

    for replicate in (1, 2, 3):
        directory = root / f"replicate_{replicate}"
        tn = load(directory / "tn.json")
        phi = load(directory / "source_density_phi/report.json")
        full_h = load(directory / "full_h6_equal_time_plateau.json")
        tn_vs_phi = load(directory / "tn_vs_phi_bootstrap.json")
        tn_vs_full_h = load(directory / "tn_vs_full_h6_bootstrap.json")
        full_h_vs_phi = load(directory / "full_h6_vs_phi_bootstrap.json")

        tn_metric = tn["metrics"]["compressed_ma_errors"]
        # All reported final-sample metrics use the normalization recomputed on
        # that sample.  The fixed-training-kappa branch is a training diagnostic
        # and must not be mixed into the final evaluation row.
        phi_metric = phi["blind_test"]["trained_phi_self_normalized"]
        full_h_metric = full_h["metrics"]

        model_values = {
            "positive_TN": {
                key: float(tn_vs_phi["comparisons"][key]["candidate"])
                for key in METRICS
            },
            "source_density_residual_phi": {
                key: float(tn_vs_phi["comparisons"][key]["baseline"])
                for key in METRICS
            },
            "cold_full_H6_equal_time_plateau": {
                key: float(tn_vs_full_h["comparisons"][key]["baseline"])
                for key in METRICS
            },
        }
        model_values["positive_TN"].update(
            maximum_absolute_log_ratio=max(
                abs(math.log(float(tn_metric["normalized_ratio_min"]))),
                abs(math.log(float(tn_metric["normalized_ratio_max"]))),
            ),
            minimum_metric_eigenvalue=float(tn["metrics"]["minimum_metric_eigenvalue"]),
            nonpositive_metric_count=0,
        )
        model_values["source_density_residual_phi"].update(
            maximum_absolute_log_ratio=float(
                phi_metric["abs_log_ratio_weighted_quantiles"]["q1.0000"]
            ),
            minimum_metric_eigenvalue=float(
                phi_metric["min_eigenvalue_weighted_quantiles"]["q0.0000"]
            ),
            nonpositive_metric_count=int(
                phi_metric["nonpositive_or_nonfinite"]["point_count"]
            ),
        )
        model_values["cold_full_H6_equal_time_plateau"].update(
            maximum_absolute_log_ratio=float(
                full_h_metric["maximum_absolute_log_ratio"]
            ),
            minimum_metric_eigenvalue=float(
                full_h_metric["minimum_metric_eigenvalue"]
            ),
            nonpositive_metric_count=int(full_h_metric["nonpositive_metric_count"]),
        )

        rows.append(
            {
                "replicate": replicate,
                "models": model_values,
                "tn_over_phi": tn_vs_phi["comparisons"],
                "tn_over_full_h6": tn_vs_full_h["comparisons"],
                "full_h6_over_phi": full_h_vs_phi["comparisons"],
            }
        )

    model_names = (
        "positive_TN",
        "source_density_residual_phi",
        "cold_full_H6_equal_time_plateau",
    )
    scalar_metrics = METRICS + (
        "maximum_absolute_log_ratio",
        "minimum_metric_eigenvalue",
    )
    aggregate_models = {
        model: {
            metric: aggregate(
                [float(row["models"][model][metric]) for row in rows]
            )
            for metric in scalar_metrics
        }
        for model in model_names
    }

    comparisons = {}
    for label, candidate, baseline, row_key in (
        (
            "tn_over_phi",
            "positive_TN",
            "source_density_residual_phi",
            "tn_over_phi",
        ),
        (
            "tn_over_full_h6",
            "positive_TN",
            "cold_full_H6_equal_time_plateau",
            "tn_over_full_h6",
        ),
        (
            "full_h6_over_phi",
            "cold_full_H6_equal_time_plateau",
            "source_density_residual_phi",
            "full_h6_over_phi",
        ),
    ):
        comparisons[label] = {}
        for metric in METRICS:
            candidate_mean = aggregate_models[candidate][metric]["mean"]
            baseline_mean = aggregate_models[baseline][metric]["mean"]
            intervals = [
                row[row_key][metric]["bootstrap_95pct_confidence_interval"]
                for row in rows
            ]
            comparisons[label][metric] = {
                "relative_improvement_of_means": float(
                    1.0 - candidate_mean / baseline_mean
                ),
                "all_paired_95pct_intervals_strictly_positive": all(
                    float(interval[0]) > 0.0 for interval in intervals
                ),
                "paired_95pct_intervals": intervals,
            }

    output = {
        "schema": "type11-x11-equal-time-final-summary-v1",
        "run_dir": portable_path(root),
        "point_count": 200000,
        "replicate_count": 3,
        "rows": rows,
        "aggregate": aggregate_models,
        "comparisons": comparisons,
        "all_sampled_metrics_positive": all(
            row["models"][model]["minimum_metric_eigenvalue"] > 0.0
            and row["models"][model]["nonpositive_metric_count"] == 0
            for row in rows
            for model in model_names
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
