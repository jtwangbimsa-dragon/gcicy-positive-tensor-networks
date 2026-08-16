#!/usr/bin/env python3
"""Validate the global rational-section description of the explicit gCICY."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric import global_diagnostic_to_dict, global_geometry_diagnostic  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intermediate-points", type=int, default=128)
    parser.add_argument("--gcicy-points", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--min-selected", type=float, default=1e-5)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs" / "gcicy_global_geometry_check.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    diagnostic = global_geometry_diagnostic(
        intermediate_points=args.intermediate_points,
        gcicy_points=args.gcicy_points,
        seed=args.seed,
        min_selected=args.min_selected,
    )
    summary = {
        "description": "Global rational-section and patch checks for the original explicit prototype, including its known boundary singularity.",
        "seed": args.seed,
        "min_selected_coordinate_gate": args.min_selected,
        **global_diagnostic_to_dict(diagnostic),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {args.out}")
    print(f"charts with usable points: {diagnostic.charts_with_usable_points} / {diagnostic.charts_tested}")
    print(f"max intermediate residual: {diagnostic.max_intermediate_residual:.6e}")
    print(f"max rational representation error: {diagnostic.max_rational_representation_error:.6e}")
    print(f"max section transition error: {diagnostic.max_section_transition_error:.6e}")
    print(f"max relative gCICY equation residual: {diagnostic.max_gcicy_relative_equation_residual:.6e}")
    print(f"minimum local Jacobian singular value: {diagnostic.min_gcicy_jacobian_singular_value:.6e}")
    print(f"minimum boundary Jacobian singular value: {diagnostic.min_boundary_jacobian_singular_value:.6e}")
    print(f"prototype_is_globally_smooth={diagnostic.prototype_is_globally_smooth}")

    if diagnostic.charts_with_usable_points != diagnostic.charts_tested:
        raise SystemExit("not every projective chart had a usable sampled point")
    if diagnostic.max_intermediate_residual > 1e-9:
        raise SystemExit("intermediate complete-intersection residual is too large")
    if diagnostic.max_rational_representation_error > 1e-9:
        raise SystemExit("rational representatives disagree on the intermediate manifold")
    if diagnostic.max_section_transition_error > 1e-9:
        raise SystemExit("rational section transition check failed")
    if diagnostic.max_gcicy_relative_equation_residual > 1e-9:
        raise SystemExit("local defining equations do not agree across charts")
    if diagnostic.min_gcicy_jacobian_singular_value < 1e-7:
        raise SystemExit("sampled local defining-equation Jacobian lost rank")
    if diagnostic.prototype_is_globally_smooth:
        raise SystemExit("expected the original sparse prototype boundary singularity, but it was not detected")


if __name__ == "__main__":
    main()
