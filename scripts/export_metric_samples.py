#!/usr/bin/env python3
"""Export numerical metric samples for the local gCICY patch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric import embedding, equations, induced_metric, metric_diagnostics, random_parameters  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=128, help="number of local patch points to sample")
    parser.add_argument("--seed", type=int, default=11, help="random seed for reproducible sampling")
    parser.add_argument("--scale", type=float, default=0.35, help="sampling scale around the chosen patch center")
    parser.add_argument(
        "--curvature-samples",
        type=int,
        default=3,
        help="number of leading samples for finite-difference scalar curvature diagnostics",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "gcicy_metric_samples.npz")
    parser.add_argument("--summary", type=Path, default=ROOT / "outputs" / "gcicy_metric_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n <= 0:
        raise ValueError("--n must be positive")
    if args.curvature_samples < 0:
        raise ValueError("--curvature-samples must be nonnegative")

    params = random_parameters(args.n, seed=args.seed, scale=args.scale)
    ambient_coords = np.empty((args.n, 7), dtype=np.complex128)
    metrics = np.empty((args.n, 3, 3), dtype=np.complex128)
    residuals = np.empty(args.n, dtype=float)
    eigenvalues = np.empty((args.n, 3), dtype=float)
    logdet = np.empty(args.n, dtype=float)
    scalar_curvature = np.full(args.n, np.nan, dtype=float)

    for idx, point in enumerate(params):
        z = embedding(point)
        g = induced_metric(point)
        eigvals = np.linalg.eigvalsh(g)
        ambient_coords[idx] = z
        metrics[idx] = g
        residuals[idx] = float(np.linalg.norm(equations(z)))
        eigenvalues[idx] = eigvals
        logdet[idx] = float(np.sum(np.log(eigvals)))

    for idx in range(min(args.curvature_samples, args.n)):
        scalar_curvature[idx] = metric_diagnostics(params[idx], include_curvature=True).scalar_curvature

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        params=params,
        ambient_coords=ambient_coords,
        metrics=metrics,
        residuals=residuals,
        eigenvalues=eigenvalues,
        logdet=logdet,
        scalar_curvature=scalar_curvature,
    )

    summary = {
        "description": "Pullback of the product Fubini-Study metric on the local gCICY patch.",
        "configuration": [[1, 1, -1, 1], [1, 1, 1, -1], [3, 1, 1, 1]],
        "patch": "x0 = y0 = z0 = 1, local coordinates params = (s, t, r)",
        "is_ricci_flat": False,
        "n_samples": int(args.n),
        "seed": int(args.seed),
        "scale": float(args.scale),
        "curvature_samples": int(min(args.curvature_samples, args.n)),
        "max_equation_residual": float(np.max(residuals)),
        "min_metric_eigenvalue": float(np.min(eigenvalues)),
        "max_metric_eigenvalue": float(np.max(eigenvalues)),
        "mean_logdet": float(np.mean(logdet)),
        "npz_file": str(args.out),
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {args.out}")
    print(f"wrote {args.summary}")
    print(f"max equation residual: {summary['max_equation_residual']:.3e}")
    print(f"metric eigenvalue range: [{summary['min_metric_eigenvalue']:.6e}, {summary['max_metric_eigenvalue']:.6e}]")
    print("metric type: pullback product Fubini-Study baseline, not Ricci-flat")


if __name__ == "__main__":
    main()
