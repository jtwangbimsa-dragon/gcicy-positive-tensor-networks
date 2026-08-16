#!/usr/bin/env python3
"""Check projective-chart invariance of a trained global-section H metric."""

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
    global_h_metric,
    global_h_metric_in_chart,
    random_parameters,
    selected_coordinate_abs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=ROOT / "outputs" / "gcicy_global_h_metric.npz")
    parser.add_argument("--points", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1501)
    parser.add_argument("--min-selected", type=float, default=1e-4)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "gcicy_global_h_atlas_check.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = np.load(args.artifact)
    exponents = artifact["global_section_exponents"]
    h_matrix = artifact["global_h_matrix"]
    normalization = float(artifact["global_section_normalization"])
    params = random_parameters(args.points, seed=args.seed, scale=0.35)

    rows = []
    for chart in all_projective_charts():
        errors = []
        for point in params:
            if selected_coordinate_abs(point, chart) <= args.min_selected:
                continue
            reference = global_h_metric(point, exponents, h_matrix, normalization=normalization)
            candidate = global_h_metric_in_chart(point, chart, exponents, h_matrix, normalization=normalization)
            errors.append(
                float(np.linalg.norm(candidate - reference) / max(1.0, np.linalg.norm(reference)))
            )
        rows.append(
            {
                "chart": list(chart),
                "usable_points": len(errors),
                "max_relative_metric_error": float(np.max(errors)) if errors else float("nan"),
                "mean_relative_metric_error": float(np.mean(errors)) if errors else float("nan"),
            }
        )

    usable = [row for row in rows if row["usable_points"] > 0]
    summary = {
        "description": "Projective-chart invariance check for a trained global-section H metric.",
        "artifact": str(args.artifact),
        "points": args.points,
        "seed": args.seed,
        "min_selected_coordinate_gate": args.min_selected,
        "charts_with_usable_points": len(usable),
        "num_charts": len(rows),
        "min_usable_points_per_chart": min(row["usable_points"] for row in usable),
        "max_relative_metric_error": max(row["max_relative_metric_error"] for row in usable),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {args.out}")
    print(f"charts with usable points: {summary['charts_with_usable_points']} / {summary['num_charts']}")
    print(f"minimum usable points per chart: {summary['min_usable_points_per_chart']}")
    print(f"max relative metric error: {summary['max_relative_metric_error']:.6e}")
    if summary["charts_with_usable_points"] != summary["num_charts"]:
        raise SystemExit("not all projective charts were tested")
    if summary["max_relative_metric_error"] > 1e-7:
        raise SystemExit("trained global H metric failed projective-chart consistency")


if __name__ == "__main__":
    main()
