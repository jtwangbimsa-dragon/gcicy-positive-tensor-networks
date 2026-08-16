#!/usr/bin/env python3
"""Fit a conservative metric candidate against local, radial, and mixed gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric import (  # noqa: E402
    apply_correction,
    baseline_metrics,
    basis_hessians,
    linearized_fit,
    random_parameters,
    random_parameters_log_annulus,
    residual_stats,
    residual_values,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--l2", type=float, default=1_000_000.0)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "gcicy_ricci_flat_metric_joint_guarded.npz")
    parser.add_argument("--summary", type=Path, default=ROOT / "outputs" / "gcicy_ricci_flat_joint_guarded_summary.json")
    return parser.parse_args()


def local(n_points: int, seed: int) -> np.ndarray:
    return random_parameters(n_points, seed=seed, scale=0.35)


def radial(n_points: int, seed: int) -> np.ndarray:
    return random_parameters_log_annulus(n_points, seed=seed, coordinate_scale=0.3, r_min=0.05, r_max=1.5)


def mixed(local_seed: int, radial_seed: int) -> np.ndarray:
    return np.vstack([local(256, local_seed), radial(512, radial_seed)])


def evaluate(params: np.ndarray, coefficients: np.ndarray, exponents: np.ndarray):
    base = baseline_metrics(params)
    corrected = apply_correction(base, basis_hessians(params, exponents), coefficients)
    return base, corrected, residual_stats(params, base), residual_stats(params, corrected)


def stats_dict(stats) -> dict[str, float]:
    return {
        "rms": stats.rms,
        "max_abs": stats.max_abs,
        "min_eigenvalue": stats.min_eigenvalue,
        "mean_log_error": stats.mean_log_error,
    }


def main() -> None:
    args = parse_args()
    train = np.vstack([local(1024, 11), radial(2048, 12), mixed(333, 334), mixed(444, 445)])
    validation = np.vstack([mixed(222, 223), mixed(101, 202)])
    export = np.vstack([mixed(222, 223), mixed(333, 334), local(256, 444), radial(256, 101)])

    result = linearized_fit(
        train,
        validation,
        max_degree=args.degree,
        l2=args.l2,
        iterations=args.iterations,
        min_eigenvalue=1e-5,
    )
    base, corrected, baseline_export, corrected_export = evaluate(export, result.coefficients, result.exponents)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        params=export,
        baseline_metrics=base,
        corrected_metrics=corrected,
        correction_metrics=corrected - base,
        coefficients=result.coefficients,
        exponents=result.exponents,
        baseline_log_ma=residual_values(export, base),
        corrected_log_ma=residual_values(export, corrected),
    )
    summary = {
        "description": "Conservative joint local/log-radial/mixed linearized MA fit.",
        "degree": args.degree,
        "basis_size": int(len(result.exponents)),
        "l2": args.l2,
        "iterations": args.iterations,
        "train_points": int(len(train)),
        "validation_points": int(len(validation)),
        "export_points": int(len(export)),
        "baseline_train": stats_dict(result.baseline_train),
        "corrected_train": stats_dict(result.best_train),
        "baseline_validation": stats_dict(result.baseline_validation),
        "corrected_validation": stats_dict(result.best_validation),
        "baseline_export": stats_dict(baseline_export),
        "corrected_export": stats_dict(corrected_export),
        "npz_file": str(args.out),
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {args.out}")
    print(f"wrote {args.summary}")
    print(f"validation MA rms: {result.baseline_validation.rms:.6e} -> {result.best_validation.rms:.6e}")
    print(f"export MA rms: {baseline_export.rms:.6e} -> {corrected_export.rms:.6e}")
    print(f"corrected export min eigenvalue: {corrected_export.min_eigenvalue:.6e}")


if __name__ == "__main__":
    main()
