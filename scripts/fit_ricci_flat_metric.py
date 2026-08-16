#!/usr/bin/env python3
"""Fit and export a local Ricci-flat metric approximation for the gCICY patch."""

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
    embedding,
    linearized_fit,
    random_parameters,
    random_parameters_log_annulus,
    residual_stats,
    residual_values,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=int, default=512, help="number of training points")
    parser.add_argument("--validation", type=int, default=256, help="number of validation points")
    parser.add_argument("--export", type=int, default=256, help="number of exported metric points")
    parser.add_argument("--seed", type=int, default=11, help="training random seed")
    parser.add_argument("--validation-seed", type=int, default=222, help="validation random seed")
    parser.add_argument("--export-seed", type=int, default=333, help="export random seed")
    parser.add_argument("--scale", type=float, default=0.35, help="sampling scale around the chosen patch center")
    parser.add_argument("--sampler", choices=("local", "log-annulus"), default="local")
    parser.add_argument("--coordinate-scale", type=float, default=0.3, help="s,t scale for log-annulus sampling")
    parser.add_argument("--r-min", type=float, default=0.05, help="minimum |r| for log-annulus sampling")
    parser.add_argument("--r-max", type=float, default=1.5, help="maximum |r| for log-annulus sampling")
    parser.add_argument("--degree", type=int, default=3, help="maximum real-polynomial potential degree")
    parser.add_argument("--l2", type=float, default=10.0, help="ridge regularization for each linearized solve")
    parser.add_argument("--iterations", type=int, default=8, help="maximum linearized Monge-Ampere iterations")
    parser.add_argument("--min-eigenvalue", type=float, default=1e-5, help="positivity gate for train and validation")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "gcicy_ricci_flat_metric_approx.npz")
    parser.add_argument("--summary", type=Path, default=ROOT / "outputs" / "gcicy_ricci_flat_fit_summary.json")
    return parser.parse_args()


def stats_dict(stats) -> dict[str, float]:
    return {
        "rms": stats.rms,
        "max_abs": stats.max_abs,
        "min_eigenvalue": stats.min_eigenvalue,
        "mean_log_error": stats.mean_log_error,
    }


def centered(values: np.ndarray) -> np.ndarray:
    return values - float(np.mean(values))


def sample_params(args: argparse.Namespace, n_points: int, seed: int) -> np.ndarray:
    if args.sampler == "local":
        return random_parameters(n_points, seed=seed, scale=args.scale)
    return random_parameters_log_annulus(
        n_points,
        seed=seed,
        coordinate_scale=args.coordinate_scale,
        r_min=args.r_min,
        r_max=args.r_max,
    )


def main() -> None:
    args = parse_args()
    train_params = sample_params(args, args.train, args.seed)
    validation_params = sample_params(args, args.validation, args.validation_seed)
    export_params = sample_params(args, args.export, args.export_seed)

    result = linearized_fit(
        train_params,
        validation_params,
        max_degree=args.degree,
        l2=args.l2,
        iterations=args.iterations,
        min_eigenvalue=args.min_eigenvalue,
    )

    export_base = baseline_metrics(export_params)
    export_basis = basis_hessians(export_params, result.exponents)
    export_corrected = apply_correction(export_base, export_basis, result.coefficients)
    baseline_output = residual_stats(export_params, export_base)
    corrected_output = residual_stats(export_params, export_corrected)
    baseline_log_ma = residual_values(export_params, export_base)
    corrected_log_ma = residual_values(export_params, export_corrected)
    ambient_coords = np.asarray([embedding(point) for point in export_params], dtype=np.complex128)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        params=export_params,
        ambient_coords=ambient_coords,
        baseline_metrics=export_base,
        corrected_metrics=export_corrected,
        correction_metrics=export_corrected - export_base,
        coefficients=result.coefficients,
        exponents=result.exponents,
        baseline_log_ma=baseline_log_ma,
        baseline_centered_log_ma=centered(baseline_log_ma),
        corrected_log_ma=corrected_log_ma,
        corrected_centered_log_ma=centered(corrected_log_ma),
    )

    summary = {
        "description": "Local Ricci-flat approximation by a polynomial Kähler-potential correction.",
        "configuration": [[1, 1, -1, 1], [1, 1, 1, -1], [3, 1, 1, 1]],
        "patch": "x0 = y0 = z0 = 1, local coordinates params = (s, t, r)",
        "method": "linearized Monge-Ampere fit: g = g0 + partial partialbar phi",
        "omega_density": "computed by the complete-intersection residue determinant; here log |Omega|^2 = -4 log |r|",
        "is_exact_ricci_flat": False,
        "is_kahler_local": True,
        "train_points": args.train,
        "validation_points": args.validation,
        "export_points": args.export,
        "seed": args.seed,
        "validation_seed": args.validation_seed,
        "export_seed": args.export_seed,
        "scale": args.scale,
        "sampler": args.sampler,
        "coordinate_scale": args.coordinate_scale,
        "r_min": args.r_min,
        "r_max": args.r_max,
        "degree": args.degree,
        "basis_size": int(len(result.exponents)),
        "l2": args.l2,
        "iterations": args.iterations,
        "min_eigenvalue_gate": args.min_eigenvalue,
        "baseline_train": stats_dict(result.baseline_train),
        "corrected_train": stats_dict(result.best_train),
        "baseline_validation": stats_dict(result.baseline_validation),
        "corrected_validation": stats_dict(result.best_validation),
        "baseline_export": stats_dict(baseline_output),
        "corrected_export": stats_dict(corrected_output),
        "history": result.history,
        "npz_file": str(args.out),
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {args.out}")
    print(f"wrote {args.summary}")
    print(
        "validation MA rms: "
        f"{result.baseline_validation.rms:.6e} -> {result.best_validation.rms:.6e}"
    )
    print(
        "export MA rms: "
        f"{baseline_output.rms:.6e} -> {corrected_output.rms:.6e}"
    )
    print(f"corrected export min eigenvalue: {corrected_output.min_eigenvalue:.6e}")
    print("metric type: local Kahler Monge-Ampere approximation, not an exact certified Ricci-flat metric")


if __name__ == "__main__":
    main()
