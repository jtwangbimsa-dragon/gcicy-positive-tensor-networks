"""Geometry and metric primitives for a non-stabilized type-(2,1) gCICY.

The configuration is

    [ P5 | 1 2 |  3 ]
    [ P1 | 1 2 | -1 ].

For fixed P1 coordinates the threefold is a complete intersection of type
(1, 2, 3) in P5, so this presentation exhibits a K3 fibration.  The module
supplies the exact algebraic model, a fibrewise linear-section sampler, local
residue data, and global-section metric primitives.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations
from typing import Sequence

import numpy as np
from scipy.linalg import det as scipy_det
from scipy.linalg import null_space, qr

from .global_sections import RestrictedSectionBasis, h_matrix_metric
from .product_projective import (
    affine_coordinates_from_product,
    affine_monomial_values_and_jacobian,
    choose_product_chart,
    evaluate_product_sections_in_chart,
    factor_offsets,
    factor_sizes,
    normalize_product_coordinates,
    product_fubini_study_metric,
    product_monomial_exponents,
    product_multinomial_section_weights,
)
from .simple_patch import fubini_study_metric


Array = np.ndarray
Chart = tuple[int, int]
FACTOR_DIMENSIONS = (5, 1)
FACTOR_SIZES = factor_sizes(FACTOR_DIMENSIONS)
FACTOR_OFFSETS = factor_offsets(FACTOR_DIMENSIONS)
AMBIENT_COORDINATE_COUNT = sum(FACTOR_SIZES)
AMBIENT_AFFINE_DIMENSION = sum(FACTOR_DIMENSIONS)
COMPLEX_DIMENSION = 3
CONSTRAINT_COUNT = 3


@dataclass(frozen=True)
class P5P1Type21Candidate1223Model:
    seed: int
    exact_coefficients: bool
    a_exponents: Array
    a_tensor: Array
    b_exponents: Array
    b_tensor: Array
    r_a_exponents: Array
    r_a_coefficients: Array
    r_b_exponents: Array
    r_b_coefficients: Array
    p1_exponents: Array
    p1_coefficients: Array
    p2_exponents: Array
    p2_coefficients: Array


@dataclass(frozen=True)
class P5P1Type21Point:
    coordinates: tuple[Array, Array]
    projective_chart: Chart
    affine_coordinates: Array
    independent_indices: tuple[int, int, int]
    dependent_indices: tuple[int, int, int]
    tangent_basis: Array
    residue_denominator: complex
    jacobian_min_singular_value: float

    @property
    def x(self) -> Array:
        return self.coordinates[0]

    @property
    def y(self) -> Array:
        return self.coordinates[1]


@dataclass(frozen=True)
class P5P1FibreAttempt:
    """One random-plane fibre attempt before batch-level acceptance."""

    coordinates: tuple[tuple[Array, Array], ...]
    status: str
    resultant_degree: int
    root_count: int
    valid_root_count: int
    minimum_projective_root_separation: float | None
    maximum_relative_residual: float | None


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


def _random_coefficients(rng: np.random.Generator, count: int) -> Array:
    values = rng.normal(size=count) + 1j * rng.normal(size=count)
    return np.asarray(values / np.linalg.norm(values), dtype=np.complex128)


def _positive_polynomial_data(
    a_exponents: Array,
    a_tensor: Array,
    b_exponents: Array,
    b_tensor: Array,
) -> tuple[Array, Array, Array, Array]:
    p1_exponents = []
    p1_coefficients = []
    for y_index in range(2):
        for x_exponents, coefficient in zip(
            a_exponents,
            a_tensor[y_index],
            strict=True,
        ):
            row = np.zeros(AMBIENT_COORDINATE_COUNT, dtype=np.int64)
            row[:6] = x_exponents
            row[6 + y_index] = 1
            p1_exponents.append(row)
            p1_coefficients.append(coefficient)

    p2_exponents = []
    p2_coefficients = []
    for y_power in range(3):
        for x_exponents, coefficient in zip(
            b_exponents,
            b_tensor[y_power],
            strict=True,
        ):
            row = np.zeros(AMBIENT_COORDINATE_COUNT, dtype=np.int64)
            row[:6] = x_exponents
            row[6] = 2 - y_power
            row[7] = y_power
            p2_exponents.append(row)
            p2_coefficients.append(coefficient)
    return (
        np.asarray(p1_exponents, dtype=np.int64),
        np.asarray(p1_coefficients, dtype=np.complex128),
        np.asarray(p2_exponents, dtype=np.int64),
        np.asarray(p2_coefficients, dtype=np.complex128),
    )


def make_p5p1_type21_candidate_1223_model(
    seed: int = 20260802,
    *,
    coefficient_bound: int = 3,
    exact: bool = True,
) -> P5P1Type21Candidate1223Model:
    if coefficient_bound < 1:
        raise ValueError("coefficient_bound must be positive")
    rng = np.random.default_rng(seed)
    a_exponents = product_monomial_exponents((1,), (5,))
    b_exponents = product_monomial_exponents((2,), (5,))
    r_a_exponents = product_monomial_exponents((2,), (5,))
    r_b_exponents = product_monomial_exponents((1,), (5,))
    coefficient_factory = (
        lambda count: _integer_coefficients(rng, count, coefficient_bound)
        if exact
        else _random_coefficients(rng, count)
    )
    a_tensor = coefficient_factory(
        2 * len(a_exponents),
    ).reshape(2, len(a_exponents))
    b_tensor = coefficient_factory(
        3 * len(b_exponents),
    ).reshape(3, len(b_exponents))
    r_a_coefficients = coefficient_factory(
        len(r_a_exponents),
    )
    r_b_coefficients = coefficient_factory(
        2 * len(r_b_exponents),
    ).reshape(2, len(r_b_exponents))
    p1_exponents, p1_coefficients, p2_exponents, p2_coefficients = (
        _positive_polynomial_data(
            a_exponents,
            a_tensor,
            b_exponents,
            b_tensor,
        )
    )
    return P5P1Type21Candidate1223Model(
        seed=seed,
        exact_coefficients=exact,
        a_exponents=a_exponents,
        a_tensor=a_tensor,
        b_exponents=b_exponents,
        b_tensor=b_tensor,
        r_a_exponents=r_a_exponents,
        r_a_coefficients=r_a_coefficients,
        r_b_exponents=r_b_exponents,
        r_b_coefficients=r_b_coefficients,
        p1_exponents=p1_exponents,
        p1_coefficients=p1_coefficients,
        p2_exponents=p2_exponents,
        p2_coefficients=p2_coefficients,
    )


def _product_terms(
    left_exponents: Array,
    left_coefficients: Array,
    right_exponents: Array,
    right_coefficients: Array,
) -> list[tuple[Array, complex]]:
    terms = []
    for left_power, left_coefficient in zip(
        left_exponents,
        left_coefficients,
        strict=True,
    ):
        for right_power, right_coefficient in zip(
            right_exponents,
            right_coefficients,
            strict=True,
        ):
            row = np.zeros(AMBIENT_COORDINATE_COUNT, dtype=np.int64)
            row[:6] = left_power + right_power
            terms.append((row, left_coefficient * right_coefficient))
    return terms


def q_polynomial_data(
    model: P5P1Type21Candidate1223Model,
    chart: Chart,
) -> tuple[Array, Array]:
    """Return one regular local representative of O(3,-1) restricted to M."""

    r_0, r_1 = model.r_b_coefficients
    if chart[1] == 0:
        # y0=1, s=y1: A1*Ra + B1*R1 + B2*(R0+s*R1).
        weighted_terms = (
            (
                model.a_exponents,
                model.a_tensor[1],
                model.r_a_exponents,
                model.r_a_coefficients,
                0,
                1.0,
            ),
            (model.b_exponents, model.b_tensor[1], model.r_b_exponents, r_1, 0, 1.0),
            (model.b_exponents, model.b_tensor[2], model.r_b_exponents, r_0, 0, 1.0),
            (model.b_exponents, model.b_tensor[2], model.r_b_exponents, r_1, 1, 1.0),
        )
        active_y_index = 7
    elif chart[1] == 1:
        # y1=1, r=y0: -A0*Ra - B0*(r*R0+R1) - B1*R0.
        weighted_terms = (
            (
                model.a_exponents,
                model.a_tensor[0],
                model.r_a_exponents,
                model.r_a_coefficients,
                0,
                -1.0,
            ),
            (model.b_exponents, model.b_tensor[0], model.r_b_exponents, r_0, 1, -1.0),
            (model.b_exponents, model.b_tensor[0], model.r_b_exponents, r_1, 0, -1.0),
            (model.b_exponents, model.b_tensor[1], model.r_b_exponents, r_0, 0, -1.0),
        )
        active_y_index = 6
    else:
        raise ValueError("the P1 chart index must be zero or one")

    by_exponent: dict[tuple[int, ...], complex] = {}
    for (
        left_exponents,
        left_coefficients,
        right_exponents,
        right_coefficients,
        y_power,
        multiplier,
    ) in weighted_terms:
        for base_exponents, coefficient in _product_terms(
            left_exponents,
            left_coefficients,
            right_exponents,
            right_coefficients,
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
    model: P5P1Type21Candidate1223Model,
    chart: Chart,
) -> tuple[tuple[Array, Array], tuple[Array, Array], tuple[Array, Array]]:
    return (
        (model.p1_exponents, model.p1_coefficients),
        (model.p2_exponents, model.p2_coefficients),
        q_polynomial_data(model, chart),
    )


def local_equations_jacobian_and_scales(
    model: P5P1Type21Candidate1223Model,
    chart_coordinates: Array,
    chart: Chart,
) -> tuple[Array, Array, Array]:
    equations = []
    derivatives = []
    scales = []
    for exponents, coefficients in local_polynomial_data(model, chart):
        values, jacobian = affine_monomial_values_and_jacobian(
            chart_coordinates,
            chart,
            exponents,
            FACTOR_DIMENSIONS,
        )
        equations.append(coefficients @ values)
        derivatives.append(coefficients @ jacobian)
        scales.append(np.abs(coefficients) @ np.abs(values))
    return (
        np.asarray(equations, dtype=np.complex128),
        np.asarray(derivatives, dtype=np.complex128),
        np.asarray(scales, dtype=float),
    )


def local_equations_and_jacobian(
    model: P5P1Type21Candidate1223Model,
    chart_coordinates: Array,
    chart: Chart,
) -> tuple[Array, Array]:
    equations, jacobian, _ = local_equations_jacobian_and_scales(
        model,
        chart_coordinates,
        chart,
    )
    return equations, jacobian


def _best_implicit_coordinates(
    jacobian: Array,
) -> tuple[tuple[int, int, int], tuple[int, int, int], Array, complex]:
    best = None
    all_indices = set(range(AMBIENT_AFFINE_DIMENSION))
    for dependent in combinations(range(AMBIENT_AFFINE_DIMENSION), CONSTRAINT_COUNT):
        matrix = jacobian[:, dependent]
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        if singular_values[-1] <= 1e-12:
            continue
        condition = float(singular_values[0] / singular_values[-1])
        if best is None or condition < best[0]:
            independent = tuple(sorted(all_indices - set(dependent)))
            best = (condition, dependent, independent, complex(scipy_det(matrix)))
    if best is None:
        raise FloatingPointError("no nonsingular implicit-coordinate minor was found")
    _, dependent, independent, determinant = best
    tangent = np.zeros(
        (AMBIENT_AFFINE_DIMENSION, COMPLEX_DIMENSION),
        dtype=np.complex128,
    )
    tangent[list(independent), :] = np.eye(COMPLEX_DIMENSION, dtype=np.complex128)
    tangent[list(dependent), :] = -np.linalg.solve(
        jacobian[:, dependent],
        jacobian[:, independent],
    )
    return independent, dependent, tangent, determinant


def _fixed_implicit_coordinates(
    jacobian: Array,
    independent: tuple[int, int, int],
) -> tuple[tuple[int, int, int], Array, complex]:
    if (
        len(set(independent)) != COMPLEX_DIMENSION
        or min(independent) < 0
        or max(independent) >= AMBIENT_AFFINE_DIMENSION
    ):
        raise ValueError("independent indices are invalid")
    dependent = tuple(
        index for index in range(AMBIENT_AFFINE_DIMENSION) if index not in independent
    )
    matrix = jacobian[:, dependent]
    determinant = complex(scipy_det(matrix))
    if abs(determinant) < 1e-14:
        raise FloatingPointError("dependent-coordinate Jacobian minor is singular")
    tangent = np.zeros(
        (AMBIENT_AFFINE_DIMENSION, COMPLEX_DIMENSION),
        dtype=np.complex128,
    )
    tangent[list(independent), :] = np.eye(COMPLEX_DIMENSION, dtype=np.complex128)
    tangent[list(dependent), :] = -np.linalg.solve(
        matrix,
        jacobian[:, independent],
    )
    return dependent, tangent, determinant


def p5p1_type21_point_in_chart(
    model: P5P1Type21Candidate1223Model,
    coordinates: Sequence[Array],
    chart: Chart,
    *,
    independent: tuple[int, int, int] | None = None,
) -> P5P1Type21Point:
    affine = affine_coordinates_from_product(
        coordinates,
        chart,
        FACTOR_DIMENSIONS,
    )
    equations, jacobian, scales = local_equations_jacobian_and_scales(
        model,
        affine,
        chart,
    )
    relative_residual = float(
        np.max(np.abs(equations) / np.maximum(1.0, scales))
    )
    if not np.isfinite(relative_residual) or relative_residual > 2e-8:
        raise FloatingPointError(
            "point does not satisfy the P5 x P1 type-(2,1) equations "
            f"(relative residual {relative_residual:.3e})"
        )
    if independent is None:
        independent, dependent, tangent, determinant = _best_implicit_coordinates(
            jacobian
        )
    else:
        dependent, tangent, determinant = _fixed_implicit_coordinates(
            jacobian,
            independent,
        )
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    copied = tuple(
        np.asarray(values, dtype=np.complex128).copy() for values in coordinates
    )
    return P5P1Type21Point(
        coordinates=copied,  # type: ignore[arg-type]
        projective_chart=chart,
        affine_coordinates=affine,
        independent_indices=independent,
        dependent_indices=dependent,
        tangent_basis=tangent,
        residue_denominator=determinant,
        jacobian_min_singular_value=float(singular_values[-1]),
    )


def _random_projective_point(rng: np.random.Generator, size: int) -> Array:
    values = rng.normal(size=size) + 1j * rng.normal(size=size)
    return np.asarray(values / np.linalg.norm(values), dtype=np.complex128)


def _p1_linear_row(model: P5P1Type21Candidate1223Model, y: Array) -> Array:
    rows = np.zeros((2, 6), dtype=np.complex128)
    for y_index in range(2):
        for powers, coefficient in zip(
            model.a_exponents,
            model.a_tensor[y_index],
            strict=True,
        ):
            rows[y_index, int(np.argmax(powers))] = coefficient
    return np.asarray(y @ rows, dtype=np.complex128)


def _restrict_x_plane(
    exponents: Array,
    coefficients: Array,
    plane: Array,
    y: Array,
) -> dict[tuple[int, int, int], complex]:
    output: dict[tuple[int, int, int], complex] = {}
    for powers, coefficient in zip(exponents, coefficients, strict=True):
        scalar = coefficient * np.prod(y ** powers[6:8])
        polynomial: dict[tuple[int, int, int], complex] = {
            (0, 0, 0): complex(scalar)
        }
        for x_index, power in enumerate(powers[:6]):
            for _ in range(int(power)):
                updated: dict[tuple[int, int, int], complex] = {}
                for term_powers, term_coefficient in polynomial.items():
                    for plane_index in range(3):
                        new_powers = list(term_powers)
                        new_powers[plane_index] += 1
                        key = tuple(new_powers)
                        updated[key] = (
                            updated.get(key, 0.0 + 0.0j)
                            + term_coefficient * plane[x_index, plane_index]
                        )
                polynomial = updated
        for term_powers, term_coefficient in polynomial.items():
            output[term_powers] = (
                output.get(term_powers, 0.0 + 0.0j) + term_coefficient
            )
    return output


def _affine_bivariate_coefficients(
    homogeneous: dict[tuple[int, int, int], complex],
    degree: int,
) -> Array:
    output = np.zeros((degree + 1, degree + 1), dtype=np.complex128)
    for (u_power, v_power, _), coefficient in homogeneous.items():
        output[u_power, v_power] += coefficient
    return output


def _polynomial_add(left: Array, right: Array) -> Array:
    size = max(len(left), len(right))
    output = np.zeros(size, dtype=np.complex128)
    output[: len(left)] += left
    output[: len(right)] += right
    return output


def _polynomial_matrix_determinant(matrix: list[list[Array]]) -> Array:
    size = len(matrix)
    output = np.zeros(1, dtype=np.complex128)
    for permutation in permutations(range(size)):
        inversions = sum(
            permutation[left] > permutation[right]
            for left in range(size)
            for right in range(left + 1, size)
        )
        term = np.asarray(
            [-1.0 + 0.0j if inversions % 2 else 1.0 + 0.0j]
        )
        for row, column in enumerate(permutation):
            term = np.polynomial.polynomial.polymul(term, matrix[row][column])
        output = _polynomial_add(output, term)
    return output


def _resultant_in_v(quadratic: Array, cubic: Array) -> Array:
    quadratic_in_v = [quadratic[:, index].copy() for index in range(3)]
    cubic_in_v = [cubic[:, index].copy() for index in range(4)]
    zero = np.zeros(1, dtype=np.complex128)
    sylvester = []
    for shift in range(3):
        row = [zero.copy() for _ in range(5)]
        for index, coefficient in enumerate(reversed(quadratic_in_v)):
            row[shift + index] = coefficient
        sylvester.append(row)
    for shift in range(2):
        row = [zero.copy() for _ in range(5)]
        for index, coefficient in enumerate(reversed(cubic_in_v)):
            row[shift + index] = coefficient
        sylvester.append(row)
    return _polynomial_matrix_determinant(sylvester)


def _evaluate_bivariate(coefficients: Array, u: complex, v: complex) -> complex:
    return complex(
        sum(
            coefficients[u_power, v_power]
            * u**u_power
            * v**v_power
            for u_power in range(coefficients.shape[0])
            for v_power in range(coefficients.shape[1])
        )
    )


def _evaluate_bivariate_derivative(
    coefficients: Array,
    u: complex,
    v: complex,
    axis: int,
) -> complex:
    output = 0.0 + 0.0j
    for u_power in range(coefficients.shape[0]):
        for v_power in range(coefficients.shape[1]):
            power = u_power if axis == 0 else v_power
            if power == 0:
                continue
            output += (
                coefficients[u_power, v_power]
                * power
                * u ** (u_power - (axis == 0))
                * v ** (v_power - (axis == 1))
            )
    return complex(output)


def _bivariate_scale(coefficients: Array, u: complex, v: complex) -> float:
    return float(
        sum(
            abs(coefficients[u_power, v_power])
            * abs(u) ** u_power
            * abs(v) ** v_power
            for u_power in range(coefficients.shape[0])
            for v_power in range(coefficients.shape[1])
        )
    )


def _trim_polynomial(coefficients: Array, relative_tolerance: float = 1e-10) -> Array:
    output = np.asarray(coefficients, dtype=np.complex128)
    scale = max(1.0, float(np.max(np.abs(output))))
    while len(output) > 1 and abs(output[-1]) < relative_tolerance * scale:
        output = output[:-1]
    return output


def _solve_conic_cubic(
    quadratic: Array,
    cubic: Array,
    *,
    residual_tolerance: float,
) -> tuple[list[tuple[complex, complex]], int, int]:
    resultant = _trim_polynomial(_resultant_in_v(quadratic, cubic))
    resultant_degree = int(len(resultant) - 1)
    if resultant_degree != 6:
        return [], resultant_degree, max(0, resultant_degree)
    resultant_roots = np.polynomial.polynomial.polyroots(resultant)
    if not np.all(np.isfinite(resultant_roots)):
        return [], resultant_degree, int(len(resultant_roots))
    roots = []
    for u_initial in resultant_roots:
        quadratic_in_v = _trim_polynomial(
            np.asarray(
                [
                    sum(
                        quadratic[u_power, v_power] * u_initial**u_power
                        for u_power in range(quadratic.shape[0])
                    )
                    for v_power in range(3)
                ],
                dtype=np.complex128,
            )
        )
        if len(quadratic_in_v) <= 1:
            continue
        v_candidates = np.polynomial.polynomial.polyroots(quadratic_in_v)
        v_initial = min(
            v_candidates,
            key=lambda value: abs(_evaluate_bivariate(cubic, u_initial, value)),
        )
        u = complex(u_initial)
        v = complex(v_initial)
        for _ in range(15):
            values = np.asarray(
                [
                    _evaluate_bivariate(quadratic, u, v),
                    _evaluate_bivariate(cubic, u, v),
                ],
                dtype=np.complex128,
            )
            jacobian = np.asarray(
                [
                    [
                        _evaluate_bivariate_derivative(quadratic, u, v, 0),
                        _evaluate_bivariate_derivative(quadratic, u, v, 1),
                    ],
                    [
                        _evaluate_bivariate_derivative(cubic, u, v, 0),
                        _evaluate_bivariate_derivative(cubic, u, v, 1),
                    ],
                ],
                dtype=np.complex128,
            )
            try:
                correction = np.linalg.solve(jacobian, values)
            except np.linalg.LinAlgError:
                break
            u -= correction[0]
            v -= correction[1]
            if np.linalg.norm(correction) < 1e-13:
                break
        residual = max(
            abs(_evaluate_bivariate(quadratic, u, v))
            / max(1.0, _bivariate_scale(quadratic, u, v)),
            abs(_evaluate_bivariate(cubic, u, v))
            / max(1.0, _bivariate_scale(cubic, u, v)),
        )
        duplicate = any(
            abs(u - old_u) + abs(v - old_v)
            <= 1e-7 * (1.0 + abs(u) + abs(v))
            for old_u, old_v in roots
        )
        if np.isfinite(residual) and residual <= residual_tolerance and not duplicate:
            roots.append((u, v))
    return roots, resultant_degree, int(len(resultant_roots))


def _sample_one_fibre(
    model: P5P1Type21Candidate1223Model,
    rng: np.random.Generator,
    *,
    residual_tolerance: float,
) -> P5P1FibreAttempt:
    def rejected(
        status: str,
        *,
        resultant_degree: int = -1,
        root_count: int = 0,
        valid_root_count: int = 0,
        minimum_separation: float | None = None,
        maximum_residual: float | None = None,
    ) -> P5P1FibreAttempt:
        return P5P1FibreAttempt(
            coordinates=(),
            status=status,
            resultant_degree=resultant_degree,
            root_count=root_count,
            valid_root_count=valid_root_count,
            minimum_projective_root_separation=minimum_separation,
            maximum_relative_residual=maximum_residual,
        )

    y = _random_projective_point(rng, 2)
    fibre_basis = null_space(_p1_linear_row(model, y).reshape(1, 6))
    if fibre_basis.shape != (6, 5):
        return rejected("invalid_fibre_basis")
    random_plane = rng.normal(size=(5, 3)) + 1j * rng.normal(size=(5, 3))
    plane_basis, _ = np.linalg.qr(random_plane, mode="reduced")
    plane = fibre_basis @ plane_basis

    y_chart = int(np.argmax(np.abs(y)))
    (y_local,) = normalize_product_coordinates(
        (y,),
        (y_chart,),
        (1,),
    )
    q_exponents, q_coefficients = q_polynomial_data(model, (0, y_chart))
    quadratic = _affine_bivariate_coefficients(
        _restrict_x_plane(
            model.p2_exponents,
            model.p2_coefficients,
            plane,
            y_local,
        ),
        2,
    )
    cubic = _affine_bivariate_coefficients(
        _restrict_x_plane(q_exponents, q_coefficients, plane, y_local),
        3,
    )
    roots, resultant_degree, root_count = _solve_conic_cubic(
        quadratic,
        cubic,
        residual_tolerance=residual_tolerance,
    )
    if resultant_degree != 6:
        return rejected(
            "nonsextic_resultant",
            resultant_degree=resultant_degree,
            root_count=root_count,
            valid_root_count=len(roots),
        )
    if root_count != 6 or len(roots) != 6:
        return rejected(
            "incomplete_conic_cubic_solve",
            resultant_degree=resultant_degree,
            root_count=root_count,
            valid_root_count=len(roots),
        )
    output: list[tuple[Array, Array]] = []
    x_roots: list[Array] = []
    residuals: list[float] = []
    for u, v in roots:
        x = plane @ np.asarray([u, v, 1.0 + 0.0j])
        if np.linalg.norm(x) < 1e-12:
            return rejected(
                "zero_projective_root",
                resultant_degree=resultant_degree,
                root_count=root_count,
                valid_root_count=len(output),
            )
        x /= np.linalg.norm(x)
        coordinates = (x, y.copy())
        chart = choose_product_chart(coordinates)
        affine = affine_coordinates_from_product(
            coordinates,
            chart,
            FACTOR_DIMENSIONS,
        )
        equations, _, scales = local_equations_jacobian_and_scales(
            model,
            affine,
            chart,  # type: ignore[arg-type]
        )
        residual = float(
            np.max(np.abs(equations) / np.maximum(1.0, scales))
        )
        residuals.append(residual)
        if not np.isfinite(residual) or residual > residual_tolerance:
            return rejected(
                "root_residual_failure",
                resultant_degree=resultant_degree,
                root_count=root_count,
                valid_root_count=len(output),
                maximum_residual=float(np.nanmax(residuals)),
            )
        x_roots.append(x.copy())
        output.append(coordinates)
    separations = []
    for left, right in combinations(x_roots, 2):
        overlap = min(1.0, float(abs(np.vdot(left, right))))
        separations.append(float(np.sqrt(max(0.0, 1.0 - overlap**2))))
    return P5P1FibreAttempt(
        coordinates=tuple(output),
        status="accepted",
        resultant_degree=resultant_degree,
        root_count=root_count,
        valid_root_count=len(output),
        minimum_projective_root_separation=float(min(separations)),
        maximum_relative_residual=float(max(residuals)),
    )


def _sample_p5p1_type21_points(
    model: P5P1Type21Candidate1223Model,
    count: int,
    *,
    seed: int,
    residual_tolerance: float,
    max_attempt_factor: int,
) -> tuple[list[P5P1Type21Point], dict[str, object]]:
    if count <= 0:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    accepted_fibres: list[list[P5P1Type21Point]] = []
    attempts = 0
    rejection_reasons: dict[str, int] = {}
    root_count_histogram: dict[str, int] = {}
    valid_root_count_histogram: dict[str, int] = {}
    minimum_root_separation = float("inf")
    maximum_relative_residual = 0.0
    required_fibres = (count + 5) // 6
    maximum_attempts = max_attempt_factor * max(1, required_fibres)
    while len(accepted_fibres) < required_fibres and attempts < maximum_attempts:
        attempt = _sample_one_fibre(
            model,
            rng,
            residual_tolerance=residual_tolerance,
        )
        attempts += 1
        root_key = str(int(attempt.root_count))
        valid_key = str(int(attempt.valid_root_count))
        root_count_histogram[root_key] = root_count_histogram.get(root_key, 0) + 1
        valid_root_count_histogram[valid_key] = (
            valid_root_count_histogram.get(valid_key, 0) + 1
        )
        if attempt.status != "accepted":
            rejection_reasons[attempt.status] = (
                rejection_reasons.get(attempt.status, 0) + 1
            )
            continue
        converted = []
        try:
            for coordinates in attempt.coordinates:
                chart = choose_product_chart(coordinates)
                converted.append(
                    p5p1_type21_point_in_chart(
                        model,
                        coordinates,
                        chart,  # type: ignore[arg-type]
                    )
                )
        except (FloatingPointError, np.linalg.LinAlgError):
            rejection_reasons["implicit_coordinate_failure"] = (
                rejection_reasons.get("implicit_coordinate_failure", 0) + 1
            )
            continue
        if len(converted) != 6:
            rejection_reasons["incomplete_converted_fibre"] = (
                rejection_reasons.get("incomplete_converted_fibre", 0) + 1
            )
            continue
        permutation = rng.permutation(6)
        accepted_fibres.append([converted[int(index)] for index in permutation])
        if attempt.minimum_projective_root_separation is not None:
            minimum_root_separation = min(
                minimum_root_separation,
                attempt.minimum_projective_root_separation,
            )
        if attempt.maximum_relative_residual is not None:
            maximum_relative_residual = max(
                maximum_relative_residual,
                attempt.maximum_relative_residual,
            )
    if len(accepted_fibres) < required_fibres:
        raise RuntimeError(
            f"only sampled {len(accepted_fibres)} complete fibres after "
            f"{attempts} attempts"
        )
    flattened = [point for fibre in accepted_fibres for point in fibre]
    output = flattened[:count]
    truncated_final_fibre = bool(count % 6)
    diagnostics: dict[str, object] = {
        "sampler": "p5p1_random_plane_conic_cubic_all_six_roots",
        "requested_points": int(count),
        "returned_points": int(len(output)),
        "expected_points_per_cluster": 6,
        "attempted_clusters": int(attempts),
        "accepted_clusters": int(len(accepted_fibres)),
        "rejected_clusters": int(attempts - len(accepted_fibres)),
        "cluster_acceptance_rate": float(len(accepted_fibres) / attempts),
        "returned_complete_clusters": int(count // 6),
        "truncated_final_cluster": truncated_final_fibre,
        "all_returned_clusters_complete": not truncated_final_fibre,
        "root_count_histogram": root_count_histogram,
        "valid_root_count_histogram": valid_root_count_histogram,
        "accepted_root_count_histogram": {"6": int(len(accepted_fibres))},
        "rejection_reasons": rejection_reasons,
        "minimum_projective_root_separation": (
            float(minimum_root_separation)
            if np.isfinite(minimum_root_separation)
            else None
        ),
        "maximum_accepted_relative_residual": float(maximum_relative_residual),
        "residual_tolerance": float(residual_tolerance),
        "importance_weight_used_for_acceptance": False,
        "implicit_jacobian_invertibility_required_for_acceptance": True,
    }
    return output, diagnostics


def sample_p5p1_type21_points(
    model: P5P1Type21Candidate1223Model,
    count: int,
    *,
    seed: int,
    residual_tolerance: float = 2e-8,
    max_attempt_factor: int = 10,
) -> list[P5P1Type21Point]:
    points, _ = _sample_p5p1_type21_points(
        model,
        count,
        seed=seed,
        residual_tolerance=residual_tolerance,
        max_attempt_factor=max_attempt_factor,
    )
    return points


def sample_p5p1_type21_points_with_diagnostics(
    model: P5P1Type21Candidate1223Model,
    count: int,
    *,
    seed: int,
    residual_tolerance: float = 2e-8,
    max_attempt_factor: int = 10,
) -> tuple[list[P5P1Type21Point], dict[str, object]]:
    """Sample complete six-root fibres and return JSON-safe diagnostics."""

    return _sample_p5p1_type21_points(
        model,
        count,
        seed=seed,
        residual_tolerance=residual_tolerance,
        max_attempt_factor=max_attempt_factor,
    )


def p5p1_type21_baseline_metric(point: P5P1Type21Point) -> Array:
    ambient = product_fubini_study_metric(
        point.affine_coordinates,
        FACTOR_DIMENSIONS,
    )
    metric = point.tangent_basis.conjugate().T @ ambient @ point.tangent_basis
    return 0.5 * (metric + metric.conjugate().T)


def p5p1_type21_holomorphic_volume_log_density(
    point: P5P1Type21Point,
) -> float:
    if abs(point.residue_denominator) < 1e-14:
        raise FloatingPointError("residue denominator is too small")
    return float(-2.0 * np.log(abs(point.residue_denominator)))


def p5p1_type21_proposal_log_density(point: P5P1Type21Point) -> float:
    """Density of omega_y wedge omega_x^2 / 2! in intrinsic coordinates."""

    x_metric = fubini_study_metric(point.affine_coordinates[:5])
    x_tangent = point.tangent_basis[:5, :]
    pulled_x = x_tangent.conjugate().T @ x_metric @ x_tangent
    y_metric = float(
        np.real(fubini_study_metric(point.affine_coordinates[5:6])[0, 0])
    )
    y_vector = point.tangent_basis[5, :]
    pulled_y = y_metric * np.outer(np.conjugate(y_vector), y_vector)
    pulled_x = 0.5 * (pulled_x + pulled_x.conjugate().T)
    pulled_y = 0.5 * (pulled_y + pulled_y.conjugate().T)
    coefficient = 0.0 + 0.0j
    for permutation in permutations(range(COMPLEX_DIMENSION)):
        inversions = sum(
            permutation[left] > permutation[right]
            for left in range(COMPLEX_DIMENSION)
            for right in range(left + 1, COMPLEX_DIMENSION)
        )
        sign = -1.0 if inversions % 2 else 1.0
        for y_row in range(COMPLEX_DIMENSION):
            term = pulled_y[y_row, permutation[y_row]]
            for x_row in range(COMPLEX_DIMENSION):
                if x_row != y_row:
                    term *= pulled_x[x_row, permutation[x_row]]
            coefficient += sign * term
    density = float(np.real(coefficient))
    if not np.isfinite(density) or density <= 0:
        raise FloatingPointError("proposal density is not positive")
    return float(np.log(density))


def p5p1_type21_importance_log_weight(point: P5P1Type21Point) -> float:
    return (
        p5p1_type21_holomorphic_volume_log_density(point)
        - p5p1_type21_proposal_log_density(point)
    )


def p5p1_type21_monge_ampere_log_error(
    point: P5P1Type21Point,
    metric: Array,
) -> float:
    eigenvalues = np.linalg.eigvalsh(np.asarray(metric, dtype=np.complex128))
    if eigenvalues[0] <= 0:
        raise FloatingPointError("metric is not positive definite")
    return float(
        np.sum(np.log(eigenvalues))
        - p5p1_type21_holomorphic_volume_log_density(point)
    )


def p5p1_type21_section_values_and_jacobian(
    point: P5P1Type21Point,
    exponents: Array,
) -> tuple[Array, Array]:
    values, ambient_jacobian = affine_monomial_values_and_jacobian(
        point.affine_coordinates,
        point.projective_chart,
        exponents,
        FACTOR_DIMENSIONS,
    )
    return values, ambient_jacobian @ point.tangent_basis


def p5p1_type21_restricted_ambient_basis(
    points: Sequence[P5P1Type21Point],
    degree: tuple[int, int],
    *,
    relative_rank_threshold: float = 1e-10,
) -> RestrictedSectionBasis:
    exponents = product_monomial_exponents(degree, FACTOR_DIMENSIONS)
    matrix = np.asarray(
        [
            evaluate_product_sections_in_chart(
                point.coordinates,
                point.projective_chart,
                exponents,
                FACTOR_DIMENSIONS,
            )
            for point in points
        ],
        dtype=np.complex128,
    )
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-14)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values.size == 0 or singular_values[0] <= 0:
        raise FloatingPointError("section evaluation matrix has zero rank")
    numerical_rank = int(
        np.sum(singular_values > relative_rank_threshold * singular_values[0])
    )
    _, _, pivots = qr(matrix, mode="economic", pivoting=True)
    selected_indices = np.asarray(pivots[:numerical_rank], dtype=np.int64)
    return RestrictedSectionBasis(
        degree=degree,  # type: ignore[arg-type]
        ambient_exponents=exponents,
        selected_indices=selected_indices,
        selected_exponents=exponents[selected_indices],
        numerical_rank=numerical_rank,
        singular_values=singular_values,
        relative_rank_threshold=relative_rank_threshold,
    )


def p5p1_type21_restriction_coefficients(
    points: Sequence[P5P1Type21Point],
    basis: RestrictedSectionBasis,
) -> tuple[Array, float]:
    all_values = np.asarray(
        [
            evaluate_product_sections_in_chart(
                point.coordinates,
                point.projective_chart,
                basis.ambient_exponents,
                FACTOR_DIMENSIONS,
            )
            for point in points
        ],
        dtype=np.complex128,
    )
    all_values /= np.maximum(
        np.linalg.norm(all_values, axis=1, keepdims=True),
        1e-14,
    )
    selected_values = all_values[:, basis.selected_indices]
    coefficients, _, _, _ = np.linalg.lstsq(
        selected_values,
        all_values,
        rcond=basis.relative_rank_threshold,
    )
    error = float(
        np.linalg.norm(selected_values @ coefficients - all_values)
        / max(1.0, np.linalg.norm(all_values))
    )
    return coefficients, error


def p5p1_type21_restricted_fubini_study_h_matrix(
    points: Sequence[P5P1Type21Point],
    basis: RestrictedSectionBasis,
) -> tuple[Array, float]:
    coefficients, relation_error = p5p1_type21_restriction_coefficients(
        points,
        basis,
    )
    weights = product_multinomial_section_weights(
        basis.ambient_exponents,
        FACTOR_DIMENSIONS,
    )
    h_matrix = np.conjugate(coefficients) @ (weights[:, None] * coefficients.T)
    h_matrix = 0.5 * (h_matrix + h_matrix.conjugate().T)
    return h_matrix, relation_error


def p5p1_type21_global_h_metric(
    point: P5P1Type21Point,
    exponents: Array,
    h_matrix: Array,
    *,
    normalization: float = 1.0,
) -> Array:
    values, jacobian = p5p1_type21_section_values_and_jacobian(
        point,
        exponents,
    )
    return h_matrix_metric(
        values,
        jacobian,
        h_matrix,
        normalization=normalization,
    )
