"""Non-stabilized type-(2,1) candidate in P5 x P1.

The configuration

    [ P5 | 1 1 |  4 ]
    [ P1 | 1 2 | -1 ]

belongs to the P5 x P1 class scanned in arXiv:1507.03235 and
arXiv:2209.10157.  This module only supplies exact algebraic data for
candidate screening.  It is not a registered metric adapter until complete
smoothness and sampling gates pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .product_projective import (
    affine_monomial_values_and_jacobian,
    factor_offsets,
    factor_sizes,
    product_monomial_exponents,
)


Array = np.ndarray
Chart = tuple[int, int]
FACTOR_DIMENSIONS = (5, 1)
FACTOR_SIZES = factor_sizes(FACTOR_DIMENSIONS)
FACTOR_OFFSETS = factor_offsets(FACTOR_DIMENSIONS)
AMBIENT_COORDINATE_COUNT = sum(FACTOR_SIZES)


@dataclass(frozen=True)
class P5P1Type21CandidateModel:
    seed: int
    a_tensor: Array
    b_tensor: Array
    cubic_exponents: Array
    q_cubic_coefficients: Array
    p1_exponents: Array
    p1_coefficients: Array
    p2_exponents: Array
    p2_coefficients: Array


def _integer_coefficients(
    rng: np.random.Generator,
    count: int,
    coefficient_bound: int,
) -> Array:
    choices = np.concatenate(
        [
            np.arange(-coefficient_bound, 0, dtype=np.int64),
            np.arange(1, coefficient_bound + 1, dtype=np.int64),
        ]
    )
    return rng.choice(choices, size=count).astype(np.complex128)


def _positive_polynomial_data(
    a_tensor: Array,
    b_tensor: Array,
) -> tuple[Array, Array, Array, Array]:
    p1_exponents = []
    p1_coefficients = []
    for y_index in range(2):
        for x_index in range(6):
            row = np.zeros(AMBIENT_COORDINATE_COUNT, dtype=np.int64)
            row[x_index] = 1
            row[6 + y_index] = 1
            p1_exponents.append(row)
            p1_coefficients.append(a_tensor[y_index, x_index])

    p2_exponents = []
    p2_coefficients = []
    for y_power in range(3):
        for x_index in range(6):
            row = np.zeros(AMBIENT_COORDINATE_COUNT, dtype=np.int64)
            row[x_index] = 1
            row[6] = 2 - y_power
            row[7] = y_power
            p2_exponents.append(row)
            p2_coefficients.append(b_tensor[y_power, x_index])
    return (
        np.asarray(p1_exponents, dtype=np.int64),
        np.asarray(p1_coefficients, dtype=np.complex128),
        np.asarray(p2_exponents, dtype=np.int64),
        np.asarray(p2_coefficients, dtype=np.complex128),
    )


def make_p5p1_type21_candidate_model(
    seed: int = 20260801,
    *,
    coefficient_bound: int = 3,
) -> P5P1Type21CandidateModel:
    if coefficient_bound < 1:
        raise ValueError("coefficient_bound must be positive")
    rng = np.random.default_rng(seed)
    a_tensor = _integer_coefficients(rng, 12, coefficient_bound).reshape(2, 6)
    b_tensor = _integer_coefficients(rng, 18, coefficient_bound).reshape(3, 6)
    cubic_exponents = product_monomial_exponents((3,), (5,))
    q_cubic_coefficients = _integer_coefficients(
        rng,
        3 * len(cubic_exponents),
        coefficient_bound,
    ).reshape(3, len(cubic_exponents))
    p1_exponents, p1_coefficients, p2_exponents, p2_coefficients = (
        _positive_polynomial_data(a_tensor, b_tensor)
    )
    return P5P1Type21CandidateModel(
        seed=seed,
        a_tensor=a_tensor,
        b_tensor=b_tensor,
        cubic_exponents=cubic_exponents,
        q_cubic_coefficients=q_cubic_coefficients,
        p1_exponents=p1_exponents,
        p1_coefficients=p1_coefficients,
        p2_exponents=p2_exponents,
        p2_coefficients=p2_coefficients,
    )


def _quartic_product_terms(
    linear_coefficients: Array,
    cubic_exponents: Array,
    cubic_coefficients: Array,
) -> list[tuple[Array, complex]]:
    terms = []
    for x_index, linear_coefficient in enumerate(linear_coefficients):
        for exponents, cubic_coefficient in zip(
            cubic_exponents,
            cubic_coefficients,
            strict=True,
        ):
            row = np.zeros(AMBIENT_COORDINATE_COUNT, dtype=np.int64)
            row[:6] = exponents
            row[x_index] += 1
            terms.append((row, linear_coefficient * cubic_coefficient))
    return terms


def q_polynomial_data(
    model: P5P1Type21CandidateModel,
    chart: Chart,
) -> tuple[Array, Array]:
    """Return a regular local representative of O(4,-1) on M."""

    r_a, r_0, r_1 = model.q_cubic_coefficients
    if chart[1] == 0:
        # y0=1, s=y1: A1*Ra + B1*R1 + B2*(R0+s*R1).
        weighted_terms = (
            (model.a_tensor[1], r_a, 0, 1.0),
            (model.b_tensor[1], r_1, 0, 1.0),
            (model.b_tensor[2], r_0, 0, 1.0),
            (model.b_tensor[2], r_1, 1, 1.0),
        )
        active_y_index = 7
    elif chart[1] == 1:
        # y1=1, r=y0: -A0*Ra-B0*(r*R0+R1)-B1*R0.
        weighted_terms = (
            (model.a_tensor[0], r_a, 0, -1.0),
            (model.b_tensor[0], r_0, 1, -1.0),
            (model.b_tensor[0], r_1, 0, -1.0),
            (model.b_tensor[1], r_0, 0, -1.0),
        )
        active_y_index = 6
    else:
        raise ValueError("the P1 chart index must be zero or one")

    by_exponent: dict[tuple[int, ...], complex] = {}
    for linear, cubic, y_power, multiplier in weighted_terms:
        for base_exponents, coefficient in _quartic_product_terms(
            linear,
            model.cubic_exponents,
            cubic,
        ):
            row = base_exponents.copy()
            row[active_y_index] = y_power
            key = tuple(int(value) for value in row)
            by_exponent[key] = (
                by_exponent.get(key, 0.0 + 0.0j)
                + multiplier * coefficient
            )
    nonzero = [(key, value) for key, value in by_exponent.items() if abs(value) > 0]
    return (
        np.asarray([key for key, _ in nonzero], dtype=np.int64),
        np.asarray([value for _, value in nonzero], dtype=np.complex128),
    )


def local_polynomial_data(
    model: P5P1Type21CandidateModel,
    chart: Chart,
) -> tuple[tuple[Array, Array], tuple[Array, Array], tuple[Array, Array]]:
    return (
        (model.p1_exponents, model.p1_coefficients),
        (model.p2_exponents, model.p2_coefficients),
        q_polynomial_data(model, chart),
    )


def local_equations_and_jacobian(
    model: P5P1Type21CandidateModel,
    chart_coordinates: Array,
    chart: Chart,
) -> tuple[Array, Array]:
    equations = []
    derivatives = []
    for exponents, coefficients in local_polynomial_data(model, chart):
        values, jacobian = affine_monomial_values_and_jacobian(
            chart_coordinates,
            chart,
            exponents,
            FACTOR_DIMENSIONS,
        )
        equations.append(coefficients @ values)
        derivatives.append(coefficients @ jacobian)
    return (
        np.asarray(equations, dtype=np.complex128),
        np.asarray(derivatives, dtype=np.complex128),
    )
