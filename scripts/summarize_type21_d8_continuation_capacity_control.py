#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


SEEDS = (86231, 86232, 86233)
ARMS = ("source", "control", "d12")
METRICS = (
    "sigma",
    "chi",
    "absolute_log_ratio_q999",
    "absolute_log_ratio_cvar_1pct",
    "upper_log_ratio_q999",
    "upper_log_ratio_cvar_1pct",
    "lower_log_ratio_q999",
    "lower_log_ratio_cvar_1pct",
    "minimum_metric_eigenvalue",
    "nonpositive_metric_count",
)


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def bilateral_metrics(path: Path) -> dict:
    payload = load_json(path)
    model = next(iter(payload["models"].values()))
    return model["metrics"]


def fmt(value: float) -> str:
    if abs(value) < 1.0e-3 and value != 0:
        return f"{value:.3e}"
    return f"{value:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tex-out", type=Path, required=True)
    args = parser.parse_args()

    records: dict[str, dict[str, dict]] = {}
    comparisons: dict[str, dict] = {}
    for seed in SEEDS:
        records[str(seed)] = {}
        for arm in ARMS:
            path = args.run_dir / f"seed_{seed}_{arm}_holdout_bilateral.json"
            metrics = bilateral_metrics(path)
            records[str(seed)][arm] = {
                key: metrics[key] for key in METRICS
            }
        path = args.run_dir / f"seed_{seed}_control_vs_d12_bootstrap_2000.json"
        comparisons[str(seed)] = load_json(path)["comparisons"]

    aggregate: dict[str, dict] = {}
    for arm in ARMS:
        aggregate[arm] = {
            key: mean(records[str(seed)][arm][key] for seed in SEEDS)
            for key in METRICS
        }

    paired = {}
    for key in ("sigma", "chi"):
        absolute = [
            records[str(seed)]["control"][key] - records[str(seed)]["d12"][key]
            for seed in SEEDS
        ]
        relative = [
            value / records[str(seed)]["control"][key]
            for seed, value in zip(SEEDS, absolute)
        ]
        paired[key] = {
            "per_seed_absolute_improvement": absolute,
            "per_seed_relative_improvement": relative,
            "mean_absolute_improvement": mean(absolute),
            "mean_relative_improvement": mean(relative),
            "all_three_pairs_improve": all(value > 0 for value in absolute),
        }

    bootstrap_primary_pass = all(
        comparisons[str(seed)][key]["bootstrap_95pct_confidence_interval"][0] > 0
        for seed in SEEDS
        for key in ("sigma", "chi")
    )
    gate_pass = (
        paired["sigma"]["all_three_pairs_improve"]
        and paired["chi"]["all_three_pairs_improve"]
        and bootstrap_primary_pass
        and all(
            records[str(seed)]["d12"]["nonpositive_metric_count"] == 0
            for seed in SEEDS
        )
    )
    summary = {
        "schema": "gcicy-type21-d8-continuation-capacity-control-summary-v1",
        "run_dir": str(args.run_dir),
        "seeds": list(SEEDS),
        "records": records,
        "aggregate_means": aggregate,
        "control_vs_d12_bootstraps": comparisons,
        "paired_primary_improvements": paired,
        "all_six_primary_bootstrap_intervals_strictly_positive": bootstrap_primary_pass,
        "preregistered_directional_gate_pass": gate_pass,
        "interpretation": (
            "The D=12 continuation beats the equal-update D=8 control in all "
            "three paired seeds for both primary metrics."
            if gate_pass
            else "The preregistered all-three-pairs directional capacity gate failed."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    rows = []
    for seed in SEEDS:
        for arm, label in (
            ("source", r"source $D=8$"),
            ("control", r"$D=8\to8$"),
            ("d12", r"$D=8\to12$"),
        ):
            item = records[str(seed)][arm]
            rows.append(
                f"{seed} & {label} & {fmt(item['sigma'])} & "
                f"{fmt(item['chi'])} & {fmt(item['absolute_log_ratio_q999'])} & "
                f"{fmt(item['absolute_log_ratio_cvar_1pct'])} \\\\"
            )
    tex = "\n".join(
        [
            r"\begin{tabular}{llrrrr}",
            r"\hline",
            r"source seed & arm & $\sigma$ & $\chi$ & $Q_{99.9}(|\log r|)$ & $\operatorname{CVaR}_{1\%}(|\log r|)$ \\",
            r"\hline",
            *rows,
            r"\hline",
            r"\end{tabular}",
            "",
        ]
    )
    args.tex_out.parent.mkdir(parents=True, exist_ok=True)
    args.tex_out.write_text(tex)


if __name__ == "__main__":
    main()
