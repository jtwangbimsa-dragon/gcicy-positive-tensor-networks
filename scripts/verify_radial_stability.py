#!/usr/bin/env python3
"""Check a metric artifact across log-radial r samples in the explicit patch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric import (  # noqa: E402
    artifact_metrics,
    baseline_metrics,
    random_parameters_log_annulus,
    residual_stats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=ROOT / "outputs" / "gcicy_ricci_flat_metric_adam_refined.npz")
    parser.add_argument("--points", type=int, default=256)
    parser.add_argument("--seeds", default="101,202,303,404")
    parser.add_argument("--coordinate-scale", type=float, default=0.3)
    parser.add_argument("--r-min", type=float, default=0.05)
    parser.add_argument("--r-max", type=float, default=1.5)
    parser.add_argument("--min-eigenvalue", type=float, default=1e-5)
    parser.add_argument("--summary", type=Path, default=ROOT / "outputs" / "gcicy_radial_stability.json")
    return parser.parse_args()


def parse_seeds(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    artifact = np.load(args.artifact)
    rows = []
    all_positive = True
    all_improved = True

    for seed in parse_seeds(args.seeds):
        params = random_parameters_log_annulus(
            args.points,
            seed=seed,
            coordinate_scale=args.coordinate_scale,
            r_min=args.r_min,
            r_max=args.r_max,
        )
        base = baseline_metrics(params)
        corrected = artifact_metrics(params, artifact)
        baseline = residual_stats(params, base)
        refined = residual_stats(params, corrected)
        positive = np.isfinite(refined.rms) and refined.min_eigenvalue > args.min_eigenvalue
        improved = positive and refined.rms < baseline.rms
        all_positive = all_positive and positive
        all_improved = all_improved and improved
        rows.append(
            {
                "seed": seed,
                "points": args.points,
                "baseline_rms": baseline.rms,
                "refined_rms": refined.rms,
                "refined_max_abs": refined.max_abs,
                "refined_min_eigenvalue": refined.min_eigenvalue,
                "positive": bool(positive),
                "improved": bool(improved),
            }
        )

    finite_rms = [row["refined_rms"] for row in rows if np.isfinite(row["refined_rms"])]
    summary = {
        "artifact": str(args.artifact),
        "points_per_seed": args.points,
        "coordinate_scale": args.coordinate_scale,
        "r_min": args.r_min,
        "r_max": args.r_max,
        "min_eigenvalue_gate": args.min_eigenvalue,
        "all_positive": bool(all_positive),
        "all_improved": bool(all_improved),
        "mean_refined_rms": float(np.mean(finite_rms)) if finite_rms else float("nan"),
        "max_refined_rms": float(np.max(finite_rms)) if finite_rms else float("nan"),
        "min_refined_eigenvalue": float(np.min([row["refined_min_eigenvalue"] for row in rows])),
        "rows": rows,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    for row in rows:
        print(
            f"seed {row['seed']}: rms {row['baseline_rms']:.6e} -> {row['refined_rms']:.6e}, "
            f"max {row['refined_max_abs']:.6e}, min eig {row['refined_min_eigenvalue']:.6e}, "
            f"positive={row['positive']}, improved={row['improved']}"
        )
    print(f"wrote {args.summary}")
    print(f"all_positive={all_positive}")
    print(f"all_improved={all_improved}")
    if not all_positive:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
