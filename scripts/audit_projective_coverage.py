#!/usr/bin/env python3
"""Audit metric artifacts on broad homogeneous-projective coverage samples."""

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
    random_projective_coverage_points,
    residual_stats,
)


DEFAULT_CANDIDATES = [
    "outputs/gcicy_section_metric_diagonal.npz",
    "outputs/gcicy_global_h_metric.npz",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", nargs="*", default=DEFAULT_CANDIDATES)
    parser.add_argument("--points", type=int, default=512)
    parser.add_argument("--seeds", default="1601,1602,1603,1604")
    parser.add_argument("--min-relative-eigenvalue", type=float, default=1e-12)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "gcicy_projective_coverage_audit.json")
    return parser.parse_args()


def parse_seeds(text: str) -> list[int]:
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def main() -> None:
    args = parse_args()
    samples = [random_projective_coverage_points(args.points, seed=seed)[0] for seed in parse_seeds(args.seeds)]
    candidate_rows = []
    for candidate in args.candidates:
        path = Path(candidate)
        if not path.is_absolute():
            path = ROOT / path
        artifact = np.load(path)
        rows = []
        for seed, params in zip(parse_seeds(args.seeds), samples, strict=True):
            baseline_metric = baseline_metrics(params)
            candidate_metric = artifact_metrics(params, artifact)
            candidate_eigenvalues = np.linalg.eigvalsh(candidate_metric)
            min_relative_eigenvalue = float(
                np.min(candidate_eigenvalues[:, 0] / np.maximum(candidate_eigenvalues[:, -1], 1e-300))
            )
            baseline = residual_stats(params, baseline_metric)
            refined = residual_stats(params, candidate_metric)
            passed = (
                np.isfinite(refined.rms)
                and refined.min_eigenvalue > 0.0
                and min_relative_eigenvalue > args.min_relative_eigenvalue
                and refined.rms < baseline.rms
            )
            rows.append(
                {
                    "seed": seed,
                    "baseline_rms": baseline.rms,
                    "candidate_rms": refined.rms,
                    "candidate_max_abs": refined.max_abs,
                    "candidate_min_eigenvalue": refined.min_eigenvalue,
                    "candidate_min_relative_eigenvalue": min_relative_eigenvalue,
                    "passed": bool(passed),
                }
            )
        finite = [row["candidate_rms"] for row in rows if np.isfinite(row["candidate_rms"])]
        candidate_rows.append(
            {
                "artifact": str(path),
                "passed_all": all(row["passed"] for row in rows),
                "mean_candidate_rms": float(np.mean(finite)) if finite else float("nan"),
                "max_candidate_rms": float(np.max(finite)) if finite else float("nan"),
                "min_candidate_eigenvalue": float(min(row["candidate_min_eigenvalue"] for row in rows)),
                "min_candidate_relative_eigenvalue": float(
                    min(row["candidate_min_relative_eigenvalue"] for row in rows)
                ),
                "rows": rows,
            }
        )

    summary = {
        "description": "Artifact audit on broad projective coverage samples; this sampler is for coverage, not uniform integration.",
        "points_per_seed": args.points,
        "seeds": parse_seeds(args.seeds),
        "min_relative_eigenvalue_gate": args.min_relative_eigenvalue,
        "candidates": candidate_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    for item in candidate_rows:
        print(Path(item["artifact"]).name)
        print(
            f"  pass={item['passed_all']} mean_rms={item['mean_candidate_rms']:.6e} "
            f"max_rms={item['max_candidate_rms']:.6e} min_eig={item['min_candidate_eigenvalue']:.6e} "
            f"min_rel_eig={item['min_candidate_relative_eigenvalue']:.6e}"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
