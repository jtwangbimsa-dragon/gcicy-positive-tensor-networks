#!/usr/bin/env python3
"""Summarize the frozen X11 models on the common 200k final sample."""

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


def main() -> None:
    args = parse_args()
    root = args.run_dir.expanduser().resolve()
    rows = []

    for replicate in (1, 2, 3):
        directory = root / f"replicate_{replicate}"
        tn = load(directory / "tn.json")
        fair_phi = load(directory / "source_density_phi/report.json")
        historical_phi = load(directory / "historical_phi/report.json")
        full_h = load(directory / "cold_full_h6.json")
        tn_vs_fair = load(directory / "tn_vs_source_density_phi_bootstrap.json")
        tn_vs_full_h = load(directory / "tn_vs_cold_full_h6_bootstrap.json")
        fair_vs_historical = load(
            directory / "source_density_vs_historical_phi_bootstrap.json"
        )

        fair_metric = fair_phi["blind_test"]["trained_phi_fixed_training_kappa"]
        historical_metric = historical_phi["blind_test"][
            "trained_phi_fixed_training_kappa"
        ]
        tn_metric = tn["metrics"]["compressed_ma_errors"]
        full_h_metric = full_h["metrics"]

        model_values = {
            "tn": {
                key: float(tn_vs_fair["comparisons"][key]["candidate"])
                for key in METRICS
            },
            "fair_phi": {
                key: float(tn_vs_fair["comparisons"][key]["baseline"])
                for key in METRICS
            },
            "historical_phi": {
                key: float(fair_vs_historical["comparisons"][key]["baseline"])
                for key in METRICS
            },
            "cold_full_h6": {
                key: float(tn_vs_full_h["comparisons"][key]["baseline"])
                for key in METRICS
            },
        }
        model_values["tn"].update(
            maximum_absolute_log_ratio=max(
                abs(math.log(float(tn_metric["normalized_ratio_min"]))),
                abs(math.log(float(tn_metric["normalized_ratio_max"]))),
            ),
            minimum_metric_eigenvalue=float(tn["metrics"]["minimum_metric_eigenvalue"]),
            nonpositive_metric_count=0,
        )
        model_values["fair_phi"].update(
            maximum_absolute_log_ratio=float(
                fair_metric["abs_log_ratio_weighted_quantiles"]["q1.0000"]
            ),
            minimum_metric_eigenvalue=float(
                fair_metric["min_eigenvalue_weighted_quantiles"]["q0.0000"]
            ),
            nonpositive_metric_count=int(
                fair_metric["nonpositive_or_nonfinite"]["point_count"]
            ),
        )
        model_values["historical_phi"].update(
            maximum_absolute_log_ratio=float(
                historical_metric["abs_log_ratio_weighted_quantiles"]["q1.0000"]
            ),
            minimum_metric_eigenvalue=float(
                historical_metric["min_eigenvalue_weighted_quantiles"]["q0.0000"]
            ),
            nonpositive_metric_count=int(
                historical_metric["nonpositive_or_nonfinite"]["point_count"]
            ),
        )
        model_values["cold_full_h6"].update(
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
                "tn_over_fair_phi": tn_vs_fair["comparisons"],
                "tn_over_cold_full_h6": tn_vs_full_h["comparisons"],
                "fair_phi_over_historical_phi": fair_vs_historical["comparisons"],
            }
        )

    model_names = ("tn", "fair_phi", "historical_phi", "cold_full_h6")
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
        ("tn_over_fair_phi", "tn", "fair_phi", "tn_over_fair_phi"),
        ("tn_over_cold_full_h6", "tn", "cold_full_h6", "tn_over_cold_full_h6"),
        (
            "fair_phi_over_historical_phi",
            "fair_phi",
            "historical_phi",
            "fair_phi_over_historical_phi",
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
        "schema": "type11-x11-final-common-holdout-summary-v1",
        "run_dir": str(root),
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
