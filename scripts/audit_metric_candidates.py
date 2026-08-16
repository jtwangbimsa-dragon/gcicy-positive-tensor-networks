#!/usr/bin/env python3
"""Audit metric candidates against local and log-radial validation suites."""

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
    random_parameters,
    random_parameters_log_annulus,
    residual_stats,
)


DEFAULT_CANDIDATES = [
    "outputs/gcicy_ricci_flat_metric_adam_refined.npz",
    "outputs/gcicy_ricci_flat_metric_annulus_large.npz",
    "outputs/gcicy_ricci_flat_metric_joint_local_radial.npz",
    "outputs/gcicy_ricci_flat_metric_joint_guarded.npz",
    "outputs/gcicy_ricci_flat_metric_joint_guarded_adam.npz",
    "outputs/gcicy_section_metric_diagonal.npz",
    "outputs/gcicy_global_h_metric.npz",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", nargs="*", default=DEFAULT_CANDIDATES)
    parser.add_argument("--points", type=int, default=256)
    parser.add_argument("--local-seeds", default="444,555,666,777,888,999,1111,1222")
    parser.add_argument("--radial-seeds", default="101,202,303,404")
    parser.add_argument("--mixed-seeds", default="222:223,333:334,444:445,101:202")
    parser.add_argument("--mixed-local-points", type=int, default=256)
    parser.add_argument("--mixed-radial-points", type=int, default=512)
    parser.add_argument("--min-eigenvalue", type=float, default=1e-5)
    parser.add_argument("--summary", type=Path, default=ROOT / "outputs" / "gcicy_metric_candidate_audit.json")
    return parser.parse_args()


def parse_seeds(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_seed_pairs(text: str) -> list[tuple[int, int]]:
    pairs = []
    for item in text.split(","):
        if not item.strip():
            continue
        left, right = item.split(":")
        pairs.append((int(left.strip()), int(right.strip())))
    return pairs


def evaluate_candidate(path: Path, params_list: list[np.ndarray], min_eigenvalue: float):
    artifact = np.load(path)
    rows = []
    for idx, params in enumerate(params_list):
        base = baseline_metrics(params)
        corrected = artifact_metrics(params, artifact)
        baseline = residual_stats(params, base)
        refined = residual_stats(params, corrected)
        passed = (
            np.isfinite(refined.rms)
            and refined.min_eigenvalue > min_eigenvalue
            and refined.rms < baseline.rms
        )
        rows.append(
            {
                "case": idx,
                "baseline_rms": baseline.rms,
                "refined_rms": refined.rms,
                "refined_min_eigenvalue": refined.min_eigenvalue,
                "passed": bool(passed),
            }
        )
    finite = [row["refined_rms"] for row in rows if np.isfinite(row["refined_rms"])]
    return {
        "passed_all": all(row["passed"] for row in rows),
        "mean_refined_rms": float(np.mean(finite)) if finite else float("nan"),
        "max_refined_rms": float(np.max(finite)) if finite else float("nan"),
        "min_refined_eigenvalue": float(np.min([row["refined_min_eigenvalue"] for row in rows])),
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    local_params = [random_parameters(args.points, seed=seed, scale=0.35) for seed in parse_seeds(args.local_seeds)]
    radial_params = [
        random_parameters_log_annulus(args.points, seed=seed, coordinate_scale=0.3, r_min=0.05, r_max=1.5)
        for seed in parse_seeds(args.radial_seeds)
    ]
    mixed_params = [
        np.vstack(
            [
                random_parameters(args.mixed_local_points, seed=local_seed, scale=0.35),
                random_parameters_log_annulus(
                    args.mixed_radial_points,
                    seed=radial_seed,
                    coordinate_scale=0.3,
                    r_min=0.05,
                    r_max=1.5,
                ),
            ]
        )
        for local_seed, radial_seed in parse_seed_pairs(args.mixed_seeds)
    ]
    candidates = []
    for candidate in args.candidates:
        path = Path(candidate)
        if not path.is_absolute():
            path = ROOT / path
        local = evaluate_candidate(path, local_params, args.min_eigenvalue)
        radial = evaluate_candidate(path, radial_params, args.min_eigenvalue)
        mixed = evaluate_candidate(path, mixed_params, args.min_eigenvalue)
        candidates.append(
            {
                "artifact": str(path),
                "passes_local": local["passed_all"],
                "passes_radial": radial["passed_all"],
                "passes_mixed": mixed["passed_all"],
                "passes_all": bool(local["passed_all"] and radial["passed_all"] and mixed["passed_all"]),
                "local": local,
                "radial": radial,
                "mixed": mixed,
            }
        )

    summary = {
        "description": "Candidate audit over local and log-radial validation suites.",
        "points_per_case": args.points,
        "min_eigenvalue_gate": args.min_eigenvalue,
        "any_candidate_passes_all": any(item["passes_all"] for item in candidates),
        "candidates": candidates,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    for item in candidates:
        print(Path(item["artifact"]).name)
        print(
            f"  local pass={item['passes_local']} mean={item['local']['mean_refined_rms']:.6e} "
            f"min_eig={item['local']['min_refined_eigenvalue']:.6e}"
        )
        print(
            f"  radial pass={item['passes_radial']} mean={item['radial']['mean_refined_rms']:.6e} "
            f"min_eig={item['radial']['min_refined_eigenvalue']:.6e}"
        )
        print(
            f"  mixed pass={item['passes_mixed']} mean={item['mixed']['mean_refined_rms']:.6e} "
            f"min_eig={item['mixed']['min_refined_eigenvalue']:.6e}"
        )
    print(f"wrote {args.summary}")
    print(f"any_candidate_passes_all={summary['any_candidate_passes_all']}")


if __name__ == "__main__":
    main()
