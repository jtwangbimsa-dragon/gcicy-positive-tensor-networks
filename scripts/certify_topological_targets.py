#!/usr/bin/env python3
"""Derive exact polarization and Riemann--Roch targets in the ambient Chow ring."""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
from math import comb
from pathlib import Path
from typing import TypeAlias


ROOT = Path(__file__).resolve().parents[1]
Exponent: TypeAlias = tuple[int, ...]
Polynomial: TypeAlias = dict[Exponent, Fraction]

CASES = (
    {
        "geometry": "X11_m3",
        "ambient_dimensions": (4, 1),
        "configuration_columns": ((1, 3), (4, -1)),
        "polarization": (1, 1),
        "expected": (23, 86),
    },
    {
        "geometry": "X21",
        "ambient_dimensions": (5, 1),
        "configuration_columns": ((1, 1), (2, 2), (3, -1)),
        "polarization": (1, 1),
        "expected": (28, 76),
    },
    {
        "geometry": "X22",
        "ambient_dimensions": (1, 1, 5),
        "configuration_columns": (
            (1, 1, 3),
            (1, 1, 1),
            (-1, 1, 1),
            (1, -1, 1),
        ),
        "polarization": (1, 1, 1),
        "expected": (50, 104),
    },
    {
        "geometry": "X11_m4",
        "ambient_dimensions": (4, 1),
        "configuration_columns": ((1, 4), (4, -2)),
        "polarization": (1, 1),
        "expected": (26, 92),
    },
)


def add(first: Polynomial, second: Polynomial) -> Polynomial:
    result = dict(first)
    for exponent, coefficient in second.items():
        result[exponent] = result.get(exponent, Fraction()) + coefficient
    return {exponent: coefficient for exponent, coefficient in result.items() if coefficient}


def scale(polynomial: Polynomial, coefficient: Fraction) -> Polynomial:
    return {
        exponent: value * coefficient
        for exponent, value in polynomial.items()
        if value * coefficient
    }


def multiply(
    first: Polynomial,
    second: Polynomial,
    dimensions: tuple[int, ...],
) -> Polynomial:
    result: Polynomial = {}
    for first_exponent, first_coefficient in first.items():
        for second_exponent, second_coefficient in second.items():
            exponent = tuple(
                left + right
                for left, right in zip(
                    first_exponent, second_exponent, strict=True
                )
            )
            if any(value > limit for value, limit in zip(exponent, dimensions, strict=True)):
                continue
            result[exponent] = (
                result.get(exponent, Fraction())
                + first_coefficient * second_coefficient
            )
    return {exponent: coefficient for exponent, coefficient in result.items() if coefficient}


def power(
    polynomial: Polynomial,
    exponent: int,
    dimensions: tuple[int, ...],
) -> Polynomial:
    result = {(0,) * len(dimensions): Fraction(1)}
    for _ in range(exponent):
        result = multiply(result, polynomial, dimensions)
    return result


def linear_form(coefficients: tuple[int, ...]) -> Polynomial:
    result: Polynomial = {}
    for index, coefficient in enumerate(coefficients):
        if coefficient:
            exponent = [0] * len(coefficients)
            exponent[index] = 1
            result[tuple(exponent)] = Fraction(coefficient)
    return result


def derive(case: dict[str, object]) -> dict[str, object]:
    dimensions = tuple(int(value) for value in case["ambient_dimensions"])
    columns = tuple(
        tuple(int(value) for value in column)
        for column in case["configuration_columns"]
    )
    polarization = tuple(int(value) for value in case["polarization"])
    one = {(0,) * len(dimensions): Fraction(1)}

    tangent_chern = one
    for factor, dimension in enumerate(dimensions):
        hyperplane = [0] * len(dimensions)
        hyperplane[factor] = 1
        generator = linear_form(tuple(hyperplane))
        ambient_factor: Polynomial = {}
        for exponent in range(dimension + 2):
            ambient_factor = add(
                ambient_factor,
                scale(
                    power(generator, exponent, dimensions),
                    Fraction(comb(dimension + 1, exponent)),
                ),
            )
        tangent_chern = multiply(tangent_chern, ambient_factor, dimensions)

    fundamental_class = one
    maximum_degree = sum(dimensions)
    for column in columns:
        divisor = linear_form(column)
        fundamental_class = multiply(fundamental_class, divisor, dimensions)
        inverse: Polynomial = {}
        for exponent in range(maximum_degree + 1):
            inverse = add(
                inverse,
                scale(
                    power(divisor, exponent, dimensions),
                    Fraction((-1) ** exponent),
                ),
            )
        tangent_chern = multiply(tangent_chern, inverse, dimensions)

    polarization_class = linear_form(polarization)
    c2_class = {
        exponent: coefficient
        for exponent, coefficient in tangent_chern.items()
        if sum(exponent) == 2
    }

    def integrate(polynomial: Polynomial) -> Fraction:
        return multiply(polynomial, fundamental_class, dimensions).get(
            dimensions, Fraction()
        )

    polarization_cube = integrate(power(polarization_class, 3, dimensions))
    c2_polarization = integrate(
        multiply(c2_class, polarization_class, dimensions)
    )
    if polarization_cube.denominator != 1 or c2_polarization.denominator != 1:
        raise ArithmeticError("topological target is not integral")
    l3 = int(polarization_cube)
    c2l = int(c2_polarization)
    expected_l3, expected_c2l = case["expected"]
    passed = (l3, c2l) == (expected_l3, expected_c2l)
    if not passed:
        raise SystemExit(f"topological target mismatch for {case['geometry']}")
    return {
        "geometry": case["geometry"],
        "ambient_dimensions": list(dimensions),
        "configuration_columns": [list(column) for column in columns],
        "polarization": list(polarization),
        "polarization_cube": l3,
        "c2_polarization": c2l,
        "riemann_roch": {
            "formula": f"({l3}*k^3+{c2l // 2}*k)/6",
            "dimensions_k1_to_k4": [
                (l3 * k**3 + (c2l // 2) * k) // 6 for k in range(1, 5)
            ],
        },
        "c2_chow_coefficients": {
            ",".join(str(value) for value in exponent): int(coefficient)
            for exponent, coefficient in sorted(c2_class.items())
        },
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs/topological_target_certificates.json",
    )
    args = parser.parse_args()
    rows = [derive(case) for case in CASES]
    summary = {
        "schema_version": 1,
        "description": (
            "Exact truncated-Chow-ring derivation of polarization cubes, "
            "second-Chern pairings, and threefold Riemann--Roch targets."
        ),
        "method": {
            "fundamental_class": "product of sequential divisor classes",
            "tangent_chern_class": "c(TA) times product (1+Q_a)^(-1)",
            "ambient_relations": "J_i^(n_i+1)=0",
        },
        "certificates": rows,
        "all_certificates_passed": all(row["passed"] for row in rows),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
