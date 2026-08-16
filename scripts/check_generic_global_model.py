#!/usr/bin/env python3
"""Validate a deterministic generic-coefficient gCICY and its cubic sampler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric import (  # noqa: E402
    Q1_DEGREE,
    Q2_DEGREE,
    affine_coordinates_from_homogeneous,
    evaluate_generic_p1,
    evaluate_generic_p2,
    evaluate_generic_q_sections,
    generic_local_equations_and_jacobian,
    make_generic_model,
    make_exact_generic_model,
    sample_generic_gcicy_points,
    sample_p2_hypersurface,
    section_transition_multiplier,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-seed", type=int, default=20260711)
    parser.add_argument("--exact-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--points", type=int, default=256)
    parser.add_argument("--seed", type=int, default=3001)
    parser.add_argument("--min-selected", type=float, default=1e-5)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "gcicy_generic_global_model_check.json")
    parser.add_argument("--model-out", type=Path, default=ROOT / "outputs" / "gcicy_generic_global_model.npz")
    return parser.parse_args()


def finite_difference_jacobian(model, coords, chart, step=1e-7):
    numerical = np.empty((4, 7), dtype=np.complex128)
    for column in range(7):
        direction = np.zeros(7, dtype=np.complex128)
        direction[column] = step
        plus, _ = generic_local_equations_and_jacobian(model, coords + direction, chart)
        minus, _ = generic_local_equations_and_jacobian(model, coords - direction, chart)
        numerical[:, column] = (plus - minus) / (2.0 * step)
    return numerical


def main() -> None:
    args = parse_args()
    model = make_exact_generic_model(args.model_seed) if args.exact_model else make_generic_model(args.model_seed)
    points = sample_generic_gcicy_points(model, args.points, seed=args.seed)
    max_equation_residual = 0.0
    min_jacobian_singular_value = float("inf")
    max_jacobian_error = 0.0
    charts = set()
    for index, point in enumerate(points):
        equations, jacobian = generic_local_equations_and_jacobian(
            model,
            point.affine_coordinates,
            point.projective_chart,
        )
        max_equation_residual = max(max_equation_residual, float(np.max(np.abs(equations))))
        min_jacobian_singular_value = min(min_jacobian_singular_value, point.jacobian_min_singular_value)
        charts.add(point.projective_chart)
        if index < 8:
            numerical = finite_difference_jacobian(model, point.affine_coordinates, point.projective_chart)
            scale = max(1.0, float(np.linalg.norm(jacobian)), float(np.linalg.norm(numerical)))
            max_jacobian_error = max(max_jacobian_error, float(np.linalg.norm(jacobian - numerical) / scale))

    intermediate = sample_p2_hypersurface(model, 128, seed=args.seed + 1)
    max_representation_error = 0.0
    max_transition_error = 0.0
    transition_charts = set()
    for x, y, z in intermediate:
        tensor = model.p2_tensor
        d0 = np.einsum("jk,j,k->", tensor[0], y, z)
        d1 = np.einsum("jk,j,k->", tensor[1], y, z)
        c0 = np.einsum("ik,i,k->", tensor[:, 0, :], x, z)
        c1 = np.einsum("ik,i,k->", tensor[:, 1, :], x, z)
        if min(abs(x[0]), abs(x[1])) > args.min_selected:
            max_representation_error = max(max_representation_error, abs(d1 / x[0] + d0 / x[1]))
        if min(abs(y[0]), abs(y[1])) > args.min_selected:
            max_representation_error = max(max_representation_error, abs(c1 / y[0] + c0 / y[1]))

        reference_chart = (int(np.argmax(np.abs(x))), int(np.argmax(np.abs(y))), int(np.argmax(np.abs(z))))
        reference_q = evaluate_generic_q_sections(model, x, y, z, reference_chart)
        for x_index in range(2):
            for y_index in range(2):
                for z_index in range(6):
                    chart = (x_index, y_index, z_index)
                    if min(abs(x[x_index]), abs(y[y_index]), abs(z[z_index])) <= args.min_selected:
                        continue
                    transition_charts.add(chart)
                    candidate_q = evaluate_generic_q_sections(model, x, y, z, chart)
                    expected_q1 = section_transition_multiplier(x, y, z, reference_chart, chart, Q1_DEGREE) * reference_q[0]
                    expected_q2 = section_transition_multiplier(x, y, z, reference_chart, chart, Q2_DEGREE) * reference_q[1]
                    scale = max(1.0, abs(candidate_q[0]), abs(candidate_q[1]), abs(expected_q1), abs(expected_q2))
                    max_transition_error = max(
                        max_transition_error,
                        abs(candidate_q[0] - expected_q1) / scale,
                        abs(candidate_q[1] - expected_q2) / scale,
                    )

    max_homogeneous_residual = max(
        max(
            abs(evaluate_generic_p1(model, point.x, point.y, point.z)),
            abs(evaluate_generic_p2(model, point.x, point.y, point.z)),
            *[abs(value) for value in evaluate_generic_q_sections(model, point.x, point.y, point.z, point.projective_chart)],
        )
        for point in points
    )
    summary = {
        "description": "Generic-coefficient gCICY plane-cubic sampling and global patch checks.",
        "model_seed": args.model_seed,
        "exact_integer_coefficients": args.exact_model,
        "sample_seed": args.seed,
        "points": args.points,
        "max_homogeneous_residual": float(max_homogeneous_residual),
        "max_local_equation_residual": max_equation_residual,
        "minimum_sampled_jacobian_singular_value": min_jacobian_singular_value,
        "max_analytic_jacobian_error": max_jacobian_error,
        "sampled_best_charts": len(charts),
        "transition_charts": len(transition_charts),
        "max_rational_representation_error": float(max_representation_error),
        "max_q_section_transition_error": float(max_transition_error),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        args.model_out,
        seed=np.asarray(model.seed),
        p1_exponents=model.p1_exponents,
        p1_coefficients=model.p1_coefficients,
        p2_exponents=model.p2_exponents,
        p2_coefficients=model.p2_coefficients,
        p2_tensor=model.p2_tensor,
    )

    print(f"wrote {args.out}")
    print(f"wrote {args.model_out}")
    print(f"max homogeneous residual: {max_homogeneous_residual:.6e}")
    print(f"max local equation residual: {max_equation_residual:.6e}")
    print(f"minimum sampled Jacobian singular value: {min_jacobian_singular_value:.6e}")
    print(f"max analytic Jacobian error: {max_jacobian_error:.6e}")
    print(f"rational representation error: {max_representation_error:.6e}")
    print(f"q transition error: {max_transition_error:.6e} over {len(transition_charts)} charts")

    if max_homogeneous_residual > 1e-8 or max_equation_residual > 1e-8:
        raise SystemExit("generic gCICY sampler residual is too large")
    if min_jacobian_singular_value < 1e-7:
        raise SystemExit("sampled generic gCICY Jacobian lost rank")
    if max_jacobian_error > 1e-7:
        raise SystemExit("analytic generic-model Jacobian check failed")
    if max_representation_error > 1e-9 or max_transition_error > 1e-9:
        raise SystemExit("generic rational-section consistency check failed")
    if len(transition_charts) != 24:
        raise SystemExit("not all projective charts were covered by transition checks")


if __name__ == "__main__":
    main()
