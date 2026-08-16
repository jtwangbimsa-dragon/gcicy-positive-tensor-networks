#!/usr/bin/env python3
"""Compare matched per-group and fixed-kappa k=4 checkpoint summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


SUMMARY_METRICS = {
    "sigma": ("mean", "max"),
    "sqrt_squared_energy": ("mean", "max"),
    "positive_log_ratio_q999": ("max",),
    "positive_log_ratio_cvar_1pct": ("max",),
    "normalized_ratio_above_3_weighted_mass": ("max",),
    "normalized_ratio_max": ("max",),
    "min_metric_eigenvalue": ("min",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-group-summary", type=Path, required=True)
    parser.add_argument("--fixed-kappa-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def aggregate(summary: dict[str, Any]) -> dict[str, Any]:
    checkpoints = summary["checkpoint_validation"]
    if not checkpoints:
        raise ValueError("summary has no checkpoint validation rows")
    rows = [entry["candidate"] for entry in checkpoints.values()]
    metrics: dict[str, float] = {}
    for name, reductions in SUMMARY_METRICS.items():
        values = np.asarray([float(row[name]) for row in rows], dtype=float)
        for reduction in reductions:
            metrics[f"{reduction}_{name}"] = float(getattr(np, reduction)(values))
    normalization = summary.get(
        "training_group_normalization",
        {
            "mode": summary.get("request", {}).get(
                "group_normalization_mode",
                "legacy_per_group",
            ),
            "fixed_log_kappa": None,
            "source": "legacy summary without explicit normalization metadata",
        },
    )
    return {
        "normalization": normalization,
        "best_epoch": int(summary["best_epoch"]),
        "checkpoint_seeds": sorted(checkpoints),
        "metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    per_group = aggregate(load(args.per_group_summary))
    fixed_kappa = aggregate(load(args.fixed_kappa_summary))
    if per_group["checkpoint_seeds"] != fixed_kappa["checkpoint_seeds"]:
        raise ValueError("the two summaries do not use the same checkpoint seeds")
    relative_change = {}
    for name, control in per_group["metrics"].items():
        treatment = fixed_kappa["metrics"][name]
        relative_change[name] = (
            float((treatment - control) / abs(control)) if control != 0 else None
        )
    result = {
        "schema_version": 1,
        "description": "Matched k=4 normalization ablation on common checkpoint fibres.",
        "per_group": per_group,
        "fixed_kappa": fixed_kappa,
        "fixed_kappa_relative_change_from_per_group": relative_change,
        "interpretation": (
            "Negative relative changes improve every error metric except "
            "min_metric_eigenvalue, for which a positive change is favorable."
        ),
    }
    text = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
