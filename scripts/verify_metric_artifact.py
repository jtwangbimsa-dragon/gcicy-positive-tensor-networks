#!/usr/bin/env python3
"""Verify a saved local gCICY metric artifact on fresh random samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric import artifact_metrics, baseline_metrics, random_parameters, residual_stats  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=ROOT / "outputs" / "gcicy_ricci_flat_metric_adam_refined.npz",
    )
    parser.add_argument("--seeds", default="444,555,666,777,888,999,1111,1222")
    parser.add_argument("--points", type=int, default=256)
    parser.add_argument("--scale", type=float, default=0.35)
    parser.add_argument("--min-eigenvalue", type=float, default=1e-5)
    parser.add_argument("--summary", type=Path, default=ROOT / "outputs" / "gcicy_metric_verification.json")
    return parser.parse_args()


def parse_seeds(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    artifact = np.load(args.artifact)
    rows = []
    all_pass = True

    for seed in parse_seeds(args.seeds):
        params = random_parameters(args.points, seed=seed, scale=args.scale)
        base = baseline_metrics(params)
        corrected = artifact_metrics(params, artifact)
        baseline = residual_stats(params, base)
        refined = residual_stats(params, corrected)
        passed = (
            np.isfinite(refined.rms)
            and refined.min_eigenvalue > args.min_eigenvalue
            and refined.rms < baseline.rms
        )
        all_pass = all_pass and passed
        rows.append(
            {
                "seed": seed,
                "points": args.points,
                "baseline_rms": baseline.rms,
                "refined_rms": refined.rms,
                "refined_max_abs": refined.max_abs,
                "refined_min_eigenvalue": refined.min_eigenvalue,
                "passed": bool(passed),
            }
        )

    finite_rms = [row["refined_rms"] for row in rows if np.isfinite(row["refined_rms"])]
    summary = {
        "artifact": str(args.artifact),
        "points_per_seed": args.points,
        "min_eigenvalue_gate": args.min_eigenvalue,
        "all_passed": bool(all_pass),
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
            f"passed={row['passed']}"
        )
    print(f"wrote {args.summary}")
    print(f"all_passed={all_pass}")
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
