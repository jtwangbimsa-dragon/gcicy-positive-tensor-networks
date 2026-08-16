#!/usr/bin/env python3
"""Summarize three frozen X21 k=8, D=8 models on one final sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any


METRICS = (
    "sigma",
    "chi",
    "absolute_log_ratio_q999",
    "absolute_log_ratio_cvar_1pct",
    "normalized_ratio_min",
    "normalized_ratio_max",
    "minimum_metric_eigenvalue",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--h4", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sole_metrics(path: Path) -> dict[str, Any]:
    models = load(path)["models"]
    if len(models) != 1:
        raise ValueError(f"expected one model in {path}")
    return next(iter(models.values()))["metrics"]


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    h4 = sole_metrics(args.h4.expanduser().resolve())
    runs = []
    for seed in args.seeds:
        metrics_path = run_dir / f"seed_{seed}_final_bilateral.json"
        bootstrap_path = run_dir / f"h4_vs_seed_{seed}_bootstrap_2000.json"
        metrics = sole_metrics(metrics_path)
        bootstrap = load(bootstrap_path)
        if metrics["nonpositive_metric_count"] != 0:
            raise ValueError(f"seed {seed} has a nonpositive sampled metric")
        comparisons = bootstrap["comparisons"]
        runs.append(
            {
                "seed": seed,
                "metrics": metrics,
                "bootstrap": {
                    name: {
                        "improvement": comparisons[name]["improvement"],
                        "relative_improvement": comparisons[name][
                            "relative_improvement"
                        ],
                        "confidence_interval": comparisons[name][
                            "bootstrap_95pct_confidence_interval"
                        ],
                    }
                    for name in (
                        "sigma",
                        "chi",
                        "absolute_log_ratio_q999",
                        "absolute_log_ratio_cvar_1pct",
                    )
                },
            }
        )

    aggregate = {}
    for name in METRICS:
        values = [float(run["metrics"][name]) for run in runs]
        aggregate[name] = {
            "mean": mean(values),
            "sample_standard_deviation": stdev(values),
            "minimum": min(values),
            "maximum": max(values),
        }

    output = {
        "schema": "type21-d8-three-seed-final-replay-v1",
        "h4_metrics": h4,
        "runs": runs,
        "aggregate": aggregate,
        "checks": {
            "run_count": len(runs),
            "all_sigma_better_than_h4": all(
                run["metrics"]["sigma"] < h4["sigma"] for run in runs
            ),
            "all_chi_better_than_h4": all(
                run["metrics"]["chi"] < h4["chi"] for run in runs
            ),
            "all_sigma_ci_lower_positive": all(
                run["bootstrap"]["sigma"]["confidence_interval"][0] > 0
                for run in runs
            ),
            "all_chi_ci_lower_positive": all(
                run["bootstrap"]["chi"]["confidence_interval"][0] > 0
                for run in runs
            ),
            "all_sampled_metrics_positive": all(
                run["metrics"]["nonpositive_metric_count"] == 0 for run in runs
            ),
        },
    }
    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    temporary.replace(out)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
