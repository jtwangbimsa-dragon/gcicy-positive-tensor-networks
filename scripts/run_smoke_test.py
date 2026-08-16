#!/usr/bin/env python3
"""Run a small numerical check for the local gCICY metric project."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric import metric_diagnostics, random_parameters  # noqa: E402


def main() -> None:
    params = random_parameters(5, seed=11)
    print("Local gCICY metric smoke test")
    print("configuration: [[P1, 1, 1, -1, 1], [P1, 1, 1, 1, -1], [P5, 3, 1, 1, 1]]")
    print("metric: pullback of product Fubini-Study metric on P1 x P1 x P5")
    print()

    for idx, point in enumerate(params):
        diag = metric_diagnostics(point, include_curvature=(idx == 0))
        curvature = "not computed"
        if diag.scalar_curvature is not None:
            curvature = f"{diag.scalar_curvature:.6e}"
        print(f"sample {idx}")
        print(f"  params            = {point}")
        print(f"  equation residual = {diag.residual_norm:.3e}")
        print(f"  metric eig range  = [{diag.min_eigenvalue:.6e}, {diag.max_eigenvalue:.6e}]")
        print(f"  log det metric    = {diag.logdet:.6e}")
        print(f"  scalar curvature  = {curvature}")


if __name__ == "__main__":
    main()
