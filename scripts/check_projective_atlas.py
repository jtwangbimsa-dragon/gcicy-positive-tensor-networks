#!/usr/bin/env python3
"""Check projective chart consistency for the local gCICY model."""

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
    chart_metric_diagnostic,
    diagnostics_to_dict,
    random_parameters,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--scale", type=float, default=0.35)
    parser.add_argument("--min-selected", type=float, default=1e-4)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "gcicy_projective_atlas_check.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = random_parameters(args.n, seed=args.seed, scale=args.scale)
    diagnostics = [
        chart_metric_diagnostic(params, chart, min_abs_selected=args.min_selected)
        for chart in all_projective_charts()
    ]
    usable = [item for item in diagnostics if item.usable_points > 0]
    max_relative_error = max(item.max_relative_error for item in usable)
    max_abs_error = max(item.max_abs_error for item in usable)
    min_usable_points = min(item.usable_points for item in usable)
    summary = {
        "description": "Consistency check for pullback Fubini-Study metrics across projective affine charts.",
        "sample_points": args.n,
        "seed": args.seed,
        "scale": args.scale,
        "min_selected_coordinate_gate": args.min_selected,
        "num_charts": len(diagnostics),
        "charts_with_usable_points": len(usable),
        "min_usable_points_per_chart": int(min_usable_points),
        "max_abs_metric_error": float(max_abs_error),
        "max_relative_metric_error": float(max_relative_error),
        "diagnostics": [diagnostics_to_dict(item) for item in diagnostics],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {args.out}")
    print(f"charts with usable points: {len(usable)} / {len(diagnostics)}")
    print(f"min usable points per chart: {min_usable_points}")
    print(f"max abs metric error: {max_abs_error:.6e}")
    print(f"max relative metric error: {max_relative_error:.6e}")
    if max_relative_error > 1e-7:
        raise SystemExit("projective chart metric consistency check failed")


if __name__ == "__main__":
    main()
