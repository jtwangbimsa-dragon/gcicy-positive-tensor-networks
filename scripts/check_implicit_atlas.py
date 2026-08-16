#!/usr/bin/env python3
"""Check implicit local-coordinate consistency for the affine gCICY patch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric import (  # noqa: E402
    all_implicit_charts,
    implicit_chart_diagnostic,
    implicit_diagnostic_to_dict,
    random_parameters,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--scale", type=float, default=0.35)
    parser.add_argument("--min-minor", type=float, default=1e-8)
    parser.add_argument("--max-cond", type=float, default=1e8)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "gcicy_implicit_atlas_check.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = random_parameters(args.n, seed=args.seed, scale=args.scale)
    diagnostics = [
        implicit_chart_diagnostic(
            params,
            chart,
            min_abs_minor_det=args.min_minor,
            max_condition_number=args.max_cond,
        )
        for chart in all_implicit_charts()
    ]
    usable = [item for item in diagnostics if item.usable_points > 0]
    max_metric_error = max(item.max_metric_relative_error for item in usable)
    max_ma_error = max(item.max_ma_error for item in usable)
    min_usable = min(item.usable_points for item in usable)
    summary = {
        "description": "Consistency check for implicit local-coordinate charts in the affine gCICY patch.",
        "sample_points": args.n,
        "seed": args.seed,
        "scale": args.scale,
        "min_minor_gate": args.min_minor,
        "max_condition_gate": args.max_cond,
        "num_charts": len(diagnostics),
        "charts_with_usable_points": len(usable),
        "min_usable_points_per_usable_chart": int(min_usable),
        "max_metric_relative_error": float(max_metric_error),
        "max_monge_ampere_error": float(max_ma_error),
        "diagnostics": [implicit_diagnostic_to_dict(item) for item in diagnostics],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {args.out}")
    print(f"charts with usable points: {len(usable)} / {len(diagnostics)}")
    print(f"min usable points per usable chart: {min_usable}")
    print(f"max metric relative error: {max_metric_error:.6e}")
    print(f"max Monge-Ampere error: {max_ma_error:.6e}")
    if max_metric_error > 1e-7 or max_ma_error > 1e-7:
        raise SystemExit("implicit atlas consistency check failed")


if __name__ == "__main__":
    main()
