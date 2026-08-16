#!/usr/bin/env python3
"""Build and validate restricted ambient global-section bases on the gCICY."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric import (  # noqa: E402
    ambient_section_count,
    max_section_transition_error,
    random_parameters,
    reference_section_values_and_jacobian,
    restricted_ambient_basis,
    section_evaluation_matrix,
)


def parse_degree(text: str) -> tuple[int, int, int]:
    values = tuple(int(value.strip()) for value in text.split(","))
    if len(values) != 3 or min(values) < 0:
        raise argparse.ArgumentTypeError("degree must be three non-negative comma-separated integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degrees", nargs="*", type=parse_degree, default=[(1, 1, 1), (1, 1, 2), (2, 2, 2)])
    parser.add_argument("--train-points", type=int, default=512)
    parser.add_argument("--validation-points", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=1e-10)
    parser.add_argument("--min-selected", type=float, default=1e-5)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "gcicy_global_sections_check.json")
    return parser.parse_args()


def derivative_error(point: np.ndarray, exponents: np.ndarray, step: float = 1e-7) -> float:
    values, jacobian = reference_section_values_and_jacobian(point, exponents)
    numerical = np.empty_like(jacobian)
    for column in range(3):
        direction = np.zeros(3, dtype=np.complex128)
        direction[column] = step
        plus, _ = reference_section_values_and_jacobian(point + direction, exponents)
        minus, _ = reference_section_values_and_jacobian(point - direction, exponents)
        numerical[:, column] = (plus - minus) / (2.0 * step)
    scale = max(1.0, float(np.linalg.norm(values)), float(np.linalg.norm(jacobian)), float(np.linalg.norm(numerical)))
    return float(np.linalg.norm(jacobian - numerical) / scale)


def main() -> None:
    args = parse_args()
    train = random_parameters(args.train_points, seed=71, scale=0.35)
    validation = random_parameters(args.validation_points, seed=72, scale=0.35)
    rows = []

    for degree in args.degrees:
        basis = restricted_ambient_basis(train, degree, relative_rank_threshold=args.threshold)
        validation_matrix = section_evaluation_matrix(validation, basis.selected_exponents)
        validation_singular_values = np.linalg.svd(validation_matrix, compute_uv=False)
        validation_rank = int(
            np.sum(validation_singular_values > args.threshold * validation_singular_values[0])
        )
        transition_error, charts_seen = max_section_transition_error(
            validation[:64],
            degree,
            basis.selected_exponents,
            min_selected=args.min_selected,
        )
        jacobian_error = derivative_error(validation[0], basis.selected_exponents)
        row = {
            "degree": list(degree),
            "ambient_section_count": ambient_section_count(degree),
            "restricted_numerical_rank": basis.numerical_rank,
            "validation_rank": validation_rank,
            "largest_singular_value": float(basis.singular_values[0]),
            "smallest_retained_singular_value": float(basis.singular_values[basis.numerical_rank - 1]),
            "largest_discarded_singular_value": (
                float(basis.singular_values[basis.numerical_rank])
                if basis.numerical_rank < len(basis.singular_values)
                else 0.0
            ),
            "charts_seen": charts_seen,
            "max_section_transition_error": transition_error,
            "max_reference_derivative_error": jacobian_error,
            "selected_indices": basis.selected_indices.tolist(),
            "selected_exponents": basis.selected_exponents.tolist(),
        }
        rows.append(row)
        print(
            f"degree={degree}: ambient={row['ambient_section_count']}, "
            f"restricted_rank={basis.numerical_rank}, validation_rank={validation_rank}, "
            f"charts={charts_seen}, transition={transition_error:.3e}, derivative={jacobian_error:.3e}"
        )

    summary = {
        "description": "Restricted ambient global-section basis checks.",
        "train_points": args.train_points,
        "validation_points": args.validation_points,
        "relative_rank_threshold": args.threshold,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")

    if any(row["validation_rank"] != row["restricted_numerical_rank"] for row in rows):
        raise SystemExit("restricted section rank did not reproduce on validation points")
    if any(row["charts_seen"] != 24 for row in rows):
        raise SystemExit("not all projective charts were covered")
    if any(row["max_section_transition_error"] > 1e-10 for row in rows):
        raise SystemExit("global section transition check failed")
    if any(row["max_reference_derivative_error"] > 1e-7 for row in rows):
        raise SystemExit("analytic section derivative check failed")


if __name__ == "__main__":
    main()
