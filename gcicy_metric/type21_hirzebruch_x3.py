"""Smooth type-(2,1) control representation of the Hirzebruch gCICY X3.

The underlying smooth threefold is the m=3 member

    [ P4 | 1 |  4 ]
    [ P1 | 3 | -1 ]

of the Hirzebruch sequence in arXiv:1606.07420.  We use the equivalent
type-(2,1) presentation

    [ P4 | 1 0 |  4 ]
    [ P1 | 3 0 | -1 ]
    [ P1 | 0 1 |  1 ].

The second positive equation fixes the last P1 at one point, while the
generalized equation carries one compensating degree in that factor.  This
is intentionally a control example: it tests every type-(2,1) pipeline
boundary while retaining a direct smoothness comparison with a studied
type-(1,1) geometry.
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
    homogeneous_product_from_affine,
    normalize_product_coordinates,
    product_fubini_study_metric,
    product_monomial_exponents,
    product_multinomial_section_weights,
)
from .simple_patch import fubini_study_metric


Array = np.ndarray
Chart = tuple[int, int, int]
FACTOR_DIMENSIONS = (4, 1, 1)
FACTOR_SIZES = factor_sizes(FACTOR_DIMENSIONS)
FACTOR_OFFSETS = factor_offsets(FACTOR_DIMENSIONS)
AMBIENT_COORDINATE_COUNT = sum(FACTOR_SIZES)
AMBIENT_AFFINE_DIMENSION = sum(FACTOR_DIMENSIONS)
COMPLEX_DIMENSION = 3
CONSTRAINT_COUNT = 3


@dataclass(frozen=True)
class HirzebruchType21Model:
    seed: int
    p_tensor: Array
    cubic_exponents: Array
    q_cubic_coefficients: Array
    point_equation: Array
    point_section: Array
    p1_exponents: Array
    p1_coefficients: Array
    p2_exponents: Array
    p2_coefficients: Array


@dataclass(frozen=True)
class HirzebruchType21Point:
    coordinates: tuple[Array, Array, Array]
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

    @property
    def z(self) -> Array:
        return self.coordinates[2]


@dataclass(frozen=True)
class HirzebruchFibreAttempt:
    """One random-line fibre attempt before batch-level acceptance."""

    coordinates: tuple[tuple[Array, Array, Array], ...]
    status: str
    polynomial_degree: int
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
    p_tensor: Array,
    point_equation: Array,
) -> tuple[Array, Array, Array, Array]:
    base_degree = int(p_tensor.shape[0] - 1)
    p1_exponents = []
    p1_coefficients = []
    for y_power in range(base_degree + 1):
        for x_index in range(5):
            row = np.zeros(AMBIENT_COORDINATE_COUNT, dtype=np.int64)
            row[x_index] = 1
            row[5] = base_degree - y_power
            row[6] = y_power
            p1_exponents.append(row)
            p1_coefficients.append(p_tensor[y_power, x_index])
    p2_exponents = []
    for z_index in range(2):
        row = np.zeros(AMBIENT_COORDINATE_COUNT, dtype=np.int64)
        row[7 + z_index] = 1
        p2_exponents.append(row)
    return (
        np.asarray(p1_exponents, dtype=np.int64),
        np.asarray(p1_coefficients, dtype=np.complex128),
        np.asarray(p2_exponents, dtype=np.int64),
        np.asarray(point_equation, dtype=np.complex128),
    )


def make_hirzebruch_type21_model(
    seed: int = 20260731,
    *,
    exact: bool = True,
    coefficient_bound: int = 3,
    base_degree: int = 3,
) -> HirzebruchType21Model:
    if coefficient_bound < 1:
        raise ValueError("coefficient_bound must be positive")
    if base_degree not in (3, 4):
        raise ValueError("the implemented Hirzebruch family degrees are 3 and 4")
    rng = np.random.default_rng(seed)
    cubic_exponents = product_monomial_exponents((3,), (4,))
    draw = (
        (lambda count: _integer_coefficients(rng, count, coefficient_bound))
        if exact
        else (lambda count: _random_coefficients(rng, count))
    )
    p_tensor = draw((base_degree + 1) * 5).reshape(base_degree + 1, 5)
    if base_degree == 3:
        q_cubic_coefficients = draw(3 * len(cubic_exponents)).reshape(
            3,
            len(cubic_exponents),
        )
    else:
        quadratic_exponents = product_monomial_exponents((2,), (4,))
        cubic_index = {
            tuple(int(value) for value in powers): index
            for index, powers in enumerate(cubic_exponents)
        }

        def linear_times_quadratic(linear: Array, quadratic: Array) -> Array:
            output = np.zeros(len(cubic_exponents), dtype=np.complex128)
            for x_index, linear_coefficient in enumerate(linear):
                for powers, quadratic_coefficient in zip(
                    quadratic_exponents,
                    quadratic,
                    strict=True,
                ):
                    target = powers.copy()
                    target[x_index] += 1
                    output[cubic_index[tuple(int(value) for value in target)]] += (
                        linear_coefficient * quadratic_coefficient
                    )
            return output

        # The unique middle Cech obstruction is sum_i g_i C_i.  Koszul
        # syzygies C_i=sum_j g_j R_ij with R_ij=-R_ji kill it identically.
        q_cubic_coefficients = np.zeros(
            (5, len(cubic_exponents)),
            dtype=np.complex128,
        )
        for left in range(5):
            for right in range(left + 1, 5):
                relation = draw(len(quadratic_exponents))
                q_cubic_coefficients[left] += linear_times_quadratic(
                    p_tensor[right],
                    relation,
                )
                q_cubic_coefficients[right] -= linear_times_quadratic(
                    p_tensor[left],
                    relation,
                )
    point_equation = np.asarray([1.0, 1.0], dtype=np.complex128)
    point_section = np.asarray([1.0, -1.0], dtype=np.complex128)
    p1_exponents, p1_coefficients, p2_exponents, p2_coefficients = (
        _positive_polynomial_data(p_tensor, point_equation)
    )
    return HirzebruchType21Model(
        seed=seed,
        p_tensor=p_tensor,
        cubic_exponents=cubic_exponents,
        q_cubic_coefficients=q_cubic_coefficients,
        point_equation=point_equation,
        point_section=point_section,
        p1_exponents=p1_exponents,
        p1_coefficients=p1_coefficients,
        p2_exponents=p2_exponents,
        p2_coefficients=p2_coefficients,
    )


def _evaluate_monomials(values: Array, exponents: Array) -> Array:
    coordinates = np.asarray(values, dtype=np.complex128).reshape(-1)
    powers = np.asarray(exponents, dtype=np.int64)
    return np.prod(coordinates[None, :] ** powers, axis=1)


def _y_monomials(y: Array, degree: int) -> Array:
    y0, y1 = np.asarray(y, dtype=np.complex128)
    return np.asarray(
        [y0 ** (degree - power) * y1**power for power in range(degree + 1)],
        dtype=np.complex128,
    )


def evaluate_p1(model: HirzebruchType21Model, x: Array, y: Array) -> complex:
    row = _y_monomials(y, model.p_tensor.shape[0] - 1) @ model.p_tensor
    return complex(row @ np.asarray(x, dtype=np.complex128))


def evaluate_p2(model: HirzebruchType21Model, z: Array) -> complex:
    return complex(model.point_equation @ np.asarray(z, dtype=np.complex128))


def evaluate_g(model: HirzebruchType21Model, x: Array) -> Array:
    return np.asarray(model.p_tensor @ np.asarray(x, dtype=np.complex128))


def evaluate_q_cubics(model: HirzebruchType21Model, x: Array) -> Array:
    values = _evaluate_monomials(x, model.cubic_exponents)
    return np.asarray(model.q_cubic_coefficients @ values, dtype=np.complex128)


def _quartic_product_terms(
    model: HirzebruchType21Model,
    g_index: int,
    cubic_index: int,
) -> list[tuple[Array, complex]]:
    terms = []
    for x_index in range(5):
        g_coefficient = model.p_tensor[g_index, x_index]
        for exponents, cubic_coefficient in zip(
            model.cubic_exponents,
            model.q_cubic_coefficients[cubic_index],
            strict=True,
        ):
            row = np.zeros(AMBIENT_COORDINATE_COUNT, dtype=np.int64)
            row[:5] = exponents
            row[x_index] += 1
            terms.append((row, g_coefficient * cubic_coefficient))
    return terms


def q_polynomial_data(
    model: HirzebruchType21Model,
    chart: Chart,
) -> tuple[Array, Array]:
    """Expand a regular local representative of O(4,2-m,1)."""

    base_degree = int(model.p_tensor.shape[0] - 1)
    if base_degree == 4:
        target_base_degree = 2 - base_degree
        weighted_terms = []
        if chart[1] == 0:
            active_y_index = 6
            for g_index in range(base_degree + 1):
                for cubic_index in range(2 * base_degree - 3):
                    laurent_power = g_index - (cubic_index + 1)
                    if laurent_power >= 0:
                        weighted_terms.append(
                            (g_index, cubic_index, laurent_power, -1.0)
                        )
        elif chart[1] == 1:
            active_y_index = 5
            for g_index in range(base_degree + 1):
                for cubic_index in range(2 * base_degree - 3):
                    laurent_power = g_index - (cubic_index + 1)
                    if laurent_power <= target_base_degree:
                        weighted_terms.append(
                            (
                                g_index,
                                cubic_index,
                                target_base_degree - laurent_power,
                                1.0,
                            )
                        )
        else:
            raise ValueError("the y chart index must be zero or one")
    elif base_degree != 3:
        raise ValueError("unsupported Hirzebruch base degree")

    if base_degree == 3 and chart[1] == 1:
        weighted_terms = (
            (0, 0, 2, -1.0),
            (0, 1, 1, -1.0),
            (0, 2, 0, -1.0),
            (1, 0, 1, -1.0),
            (1, 1, 0, -1.0),
            (2, 0, 0, -1.0),
        )
        active_y_index = 5
    elif base_degree == 3 and chart[1] == 0:
        weighted_terms = (
            (1, 2, 0, 1.0),
            (2, 1, 0, 1.0),
            (2, 2, 1, 1.0),
            (3, 0, 0, 1.0),
            (3, 1, 1, 1.0),
            (3, 2, 2, 1.0),
        )
        active_y_index = 6
    elif base_degree == 3:
        raise ValueError("the y chart index must be zero or one")

    by_exponent: dict[tuple[int, ...], complex] = {}
    for g_index, cubic_index, y_power, multiplier in weighted_terms:
        for base_exponents, coefficient in _quartic_product_terms(
            model,
            g_index,
            cubic_index,
        ):
            for z_index, z_coefficient in enumerate(model.point_section):
                row = base_exponents.copy()
                row[active_y_index] = y_power
                row[7 + z_index] = 1
                key = tuple(int(value) for value in row)
                by_exponent[key] = (
                    by_exponent.get(key, 0.0 + 0.0j)
                    + multiplier * coefficient * z_coefficient
                )
    nonzero = [(key, value) for key, value in by_exponent.items() if abs(value) > 0]
    return (
        np.asarray([key for key, _ in nonzero], dtype=np.int64),
        np.asarray([value for _, value in nonzero], dtype=np.complex128),
    )


def local_polynomial_data(
    model: HirzebruchType21Model,
    chart: Chart,
) -> tuple[tuple[Array, Array], tuple[Array, Array], tuple[Array, Array]]:
    return (
        (model.p1_exponents, model.p1_coefficients),
        (model.p2_exponents, model.p2_coefficients),
        q_polynomial_data(model, chart),
    )


def local_equations_jacobian_and_scales(
    model: HirzebruchType21Model,
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
    model: HirzebruchType21Model,
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


def hirzebruch_type21_point_in_chart(
    model: HirzebruchType21Model,
    coordinates: Sequence[Array],
    chart: Chart,
    *,
    independent: tuple[int, int, int] | None = None,
) -> HirzebruchType21Point:
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
    if not np.isfinite(relative_residual) or relative_residual > 2e-9:
        raise FloatingPointError(
            "point does not satisfy the type-(2,1) equations "
            f"(relative residual {relative_residual:.3e})"
        )
    if independent is None:
        independent, dependent, tangent, determinant = _best_implicit_coordinates(jacobian)
    else:
        dependent, tangent, determinant = _fixed_implicit_coordinates(
            jacobian,
            independent,
        )
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    copied = tuple(np.asarray(values, dtype=np.complex128).copy() for values in coordinates)
    return HirzebruchType21Point(
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


def _restrict_x_line(
    exponents: Array,
    coefficients: Array,
    x_base: Array,
    x_direction: Array,
    y: Array,
    z: Array,
) -> Array:
    polynomial = np.zeros(5, dtype=np.complex128)
    for powers, coefficient in zip(exponents, coefficients, strict=True):
        scalar = coefficient
        scalar *= np.prod(y ** powers[5:7])
        scalar *= np.prod(z ** powers[7:9])
        term = [1.0 + 0.0j]
        for index, power in enumerate(powers[:5]):
            for _ in range(int(power)):
                product = [0.0 + 0.0j] * (len(term) + 1)
                for degree, value in enumerate(term):
                    product[degree] += value * x_base[index]
                    product[degree + 1] += value * x_direction[index]
                term = product
        polynomial[: len(term)] += scalar * np.asarray(term, dtype=np.complex128)
    return polynomial


def _sample_one_fibre(
    model: HirzebruchType21Model,
    rng: np.random.Generator,
    *,
    residual_tolerance: float,
    root_separation_tolerance: float,
    safe_projection: bool,
) -> HirzebruchFibreAttempt:
    if (
        not np.isfinite(root_separation_tolerance)
        or not 0.0 <= root_separation_tolerance < 1.0
    ):
        raise ValueError("root separation tolerance must lie in [0, 1)")

    def rejected(
        status: str,
        *,
        polynomial_degree: int = -1,
        root_count: int = 0,
        valid_root_count: int = 0,
        minimum_separation: float | None = None,
        maximum_residual: float | None = None,
    ) -> HirzebruchFibreAttempt:
        return HirzebruchFibreAttempt(
            coordinates=(),
            status=status,
            polynomial_degree=polynomial_degree,
            root_count=root_count,
            valid_root_count=valid_root_count,
            minimum_projective_root_separation=minimum_separation,
            maximum_relative_residual=maximum_residual,
        )

    y = _random_projective_point(rng, 2)
    z = np.asarray(
        [model.point_equation[1], -model.point_equation[0]],
        dtype=np.complex128,
    )
    z /= np.linalg.norm(z)
    p_row = _y_monomials(y, model.p_tensor.shape[0] - 1) @ model.p_tensor
    if safe_projection:
        row_norm = float(np.linalg.norm(p_row))
        if not np.isfinite(row_norm) or row_norm < 1e-14:
            return rejected("degenerate_positive_equation")
        normal = np.conjugate(p_row) / row_norm

        def projected_gaussian() -> Array:
            draw = rng.normal(size=5) + 1j * rng.normal(size=5)
            return np.asarray(
                draw - normal * np.vdot(normal, draw),
                dtype=np.complex128,
            )

        x_base = projected_gaussian()
        x_direction = projected_gaussian()
    else:
        fibre_basis = null_space(p_row.reshape(1, 5))
        if fibre_basis.shape != (5, 4):
            return rejected("invalid_fibre_basis")
        x_base = fibre_basis @ (
            rng.normal(size=4) + 1j * rng.normal(size=4)
        )
        x_direction = fibre_basis @ (
            rng.normal(size=4) + 1j * rng.normal(size=4)
        )
    line_matrix = np.column_stack((x_base, x_direction))
    line_basis, line_scale = np.linalg.qr(line_matrix, mode="reduced")
    if (
        line_basis.shape != (5, 2)
        or line_scale.shape != (2, 2)
        or float(np.min(np.abs(np.diag(line_scale)))) < 1e-12
    ):
        return rejected("degenerate_random_line")
    x_base = np.asarray(line_basis[:, 0], dtype=np.complex128)
    x_direction = np.asarray(line_basis[:, 1], dtype=np.complex128)
    chart = (0, int(np.argmax(np.abs(y))), int(np.argmax(np.abs(z))))
    y_local, z_local = normalize_product_coordinates(
        (y, z),
        chart[1:],
        FACTOR_DIMENSIONS[1:],
    )
    q_exponents, q_coefficients = q_polynomial_data(model, chart)
    quartic = _restrict_x_line(
        q_exponents,
        q_coefficients,
        x_base,
        x_direction,
        y_local,
        z_local,
    )
    scale = max(1.0, float(np.max(np.abs(quartic))))
    while len(quartic) > 1 and abs(quartic[-1]) < 1e-12 * scale:
        quartic = quartic[:-1]
    polynomial_degree = int(len(quartic) - 1)
    if polynomial_degree != 4:
        return rejected(
            "nonquartic_restriction",
            polynomial_degree=polynomial_degree,
            root_count=max(0, polynomial_degree),
        )
    roots = np.polynomial.polynomial.polyroots(quartic)
    if len(roots) != 4 or not np.all(np.isfinite(roots)):
        return rejected(
            "invalid_quartic_roots",
            polynomial_degree=polynomial_degree,
            root_count=int(len(roots)),
        )
    output: list[tuple[Array, Array, Array]] = []
    x_roots: list[Array] = []
    residuals: list[float] = []
    for root in roots:
        x = x_base + root * x_direction
        if np.linalg.norm(x) < 1e-12:
            return rejected(
                "zero_projective_root",
                polynomial_degree=polynomial_degree,
                root_count=int(len(roots)),
                valid_root_count=len(output),
            )
        x /= np.linalg.norm(x)
        coordinates = (x, y.copy(), z.copy())
        point_chart = choose_product_chart(coordinates)
        affine = affine_coordinates_from_product(
            coordinates,
            point_chart,
            FACTOR_DIMENSIONS,
        )
        equations, _, scales = local_equations_jacobian_and_scales(
            model,
            affine,
            point_chart,  # type: ignore[arg-type]
        )
        residual = float(
            np.max(np.abs(equations) / np.maximum(1.0, scales))
        )
        residuals.append(residual)
        if not np.isfinite(residual) or residual > residual_tolerance:
            return rejected(
                "root_residual_failure",
                polynomial_degree=polynomial_degree,
                root_count=int(len(roots)),
                valid_root_count=len(output),
                maximum_residual=float(np.nanmax(residuals)),
            )
        x_roots.append(x.copy())
        output.append(coordinates)
    separations = []
    for left, right in combinations(x_roots, 2):
        overlap = min(1.0, float(abs(np.vdot(left, right))))
        separations.append(float(np.sqrt(max(0.0, 1.0 - overlap**2))))
    minimum_separation = float(min(separations))
    if minimum_separation <= root_separation_tolerance:
        return rejected(
            "duplicate_projective_root",
            polynomial_degree=polynomial_degree,
            root_count=int(len(roots)),
            valid_root_count=len(output),
            minimum_separation=minimum_separation,
            maximum_residual=float(max(residuals)),
        )
    return HirzebruchFibreAttempt(
        coordinates=tuple(output),
        status="accepted",
        polynomial_degree=polynomial_degree,
        root_count=int(len(roots)),
        valid_root_count=len(output),
        minimum_projective_root_separation=minimum_separation,
        maximum_relative_residual=float(max(residuals)),
    )


def _sample_hirzebruch_type21_points(
    model: HirzebruchType21Model,
    count: int,
    *,
    seed: int,
    residual_tolerance: float,
    root_separation_tolerance: float,
    max_attempt_factor: int,
    safe_projection: bool,
) -> tuple[list[HirzebruchType21Point], dict[str, object]]:
    if count <= 0:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    accepted_fibres: list[list[HirzebruchType21Point]] = []
    attempts = 0
    rejection_reasons: dict[str, int] = {}
    root_count_histogram: dict[str, int] = {}
    valid_root_count_histogram: dict[str, int] = {}
    minimum_root_separation = float("inf")
    maximum_relative_residual = 0.0
    required_fibres = (count + 3) // 4
    maximum_attempts = max_attempt_factor * max(1, required_fibres)
    while len(accepted_fibres) < required_fibres and attempts < maximum_attempts:
        attempt = _sample_one_fibre(
            model,
            rng,
            residual_tolerance=residual_tolerance,
            root_separation_tolerance=root_separation_tolerance,
            safe_projection=safe_projection,
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
                    hirzebruch_type21_point_in_chart(
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
        if len(converted) != 4:
            rejection_reasons["incomplete_converted_fibre"] = (
                rejection_reasons.get("incomplete_converted_fibre", 0) + 1
            )
            continue
        permutation = rng.permutation(4)
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
    truncated_final_fibre = bool(count % 4)
    diagnostics: dict[str, object] = {
        "sampler": "hirzebruch_random_line_all_quartic_roots",
        "requested_points": int(count),
        "returned_points": int(len(output)),
        "expected_points_per_cluster": 4,
        "attempted_clusters": int(attempts),
        "accepted_clusters": int(len(accepted_fibres)),
        "rejected_clusters": int(attempts - len(accepted_fibres)),
        "cluster_acceptance_rate": float(len(accepted_fibres) / attempts),
        "returned_complete_clusters": int(count // 4),
        "truncated_final_cluster": truncated_final_fibre,
        "all_returned_clusters_complete": not truncated_final_fibre,
        "root_count_histogram": root_count_histogram,
        "valid_root_count_histogram": valid_root_count_histogram,
        "accepted_root_count_histogram": {"4": int(len(accepted_fibres))},
        "rejection_reasons": rejection_reasons,
        "minimum_projective_root_separation": (
            float(minimum_root_separation)
            if np.isfinite(minimum_root_separation)
            else None
        ),
        "maximum_accepted_relative_residual": float(maximum_relative_residual),
        "residual_tolerance": float(residual_tolerance),
        "root_separation_tolerance": float(root_separation_tolerance),
        "safe_projection": bool(safe_projection),
        "importance_weight_used_for_acceptance": False,
        "implicit_jacobian_invertibility_required_for_acceptance": True,
    }
    return output, diagnostics


def sample_hirzebruch_type21_points(
    model: HirzebruchType21Model,
    count: int,
    *,
    seed: int,
    residual_tolerance: float = 2e-9,
    root_separation_tolerance: float = 1e-7,
    max_attempt_factor: int = 20,
    safe_projection: bool = False,
) -> list[HirzebruchType21Point]:
    points, _ = _sample_hirzebruch_type21_points(
        model,
        count,
        seed=seed,
        residual_tolerance=residual_tolerance,
        root_separation_tolerance=root_separation_tolerance,
        max_attempt_factor=max_attempt_factor,
        safe_projection=safe_projection,
    )
    return points


def sample_hirzebruch_type21_points_with_diagnostics(
    model: HirzebruchType21Model,
    count: int,
    *,
    seed: int,
    residual_tolerance: float = 2e-9,
    root_separation_tolerance: float = 1e-7,
    max_attempt_factor: int = 20,
    safe_projection: bool = False,
) -> tuple[list[HirzebruchType21Point], dict[str, object]]:
    """Sample complete quartic fibres and expose rejection/root diagnostics."""

    return _sample_hirzebruch_type21_points(
        model,
        count,
        seed=seed,
        residual_tolerance=residual_tolerance,
        root_separation_tolerance=root_separation_tolerance,
        max_attempt_factor=max_attempt_factor,
        safe_projection=safe_projection,
    )


def hirzebruch_type21_baseline_metric(point: HirzebruchType21Point) -> Array:
    ambient = product_fubini_study_metric(
        point.affine_coordinates,
        FACTOR_DIMENSIONS,
    )
    metric = point.tangent_basis.conjugate().T @ ambient @ point.tangent_basis
    return 0.5 * (metric + metric.conjugate().T)


def hirzebruch_type21_product_fubini_study_distance(
    left: HirzebruchType21Point,
    right: HirzebruchType21Point,
) -> float:
    """Return the product Fubini--Study distance between two points."""

    squared_distance = 0.0
    for left_factor, right_factor in zip(
        left.coordinates,
        right.coordinates,
        strict=True,
    ):
        left_unit = left_factor / np.linalg.norm(left_factor)
        right_unit = right_factor / np.linalg.norm(right_factor)
        overlap = float(np.clip(abs(np.vdot(left_unit, right_unit)), 0.0, 1.0))
        squared_distance += float(np.arccos(overlap) ** 2)
    return float(np.sqrt(squared_distance))


def _damped_dependent_coordinate_newton(
    model: HirzebruchType21Model,
    affine: Array,
    chart: Chart,
    dependent: tuple[int, int, int],
    *,
    residual_tolerance: float,
    maximum_iterations: int,
) -> tuple[Array, int, float]:
    candidate = np.asarray(affine, dtype=np.complex128).copy()
    best_residual = float("inf")
    for iteration in range(maximum_iterations + 1):
        equations, jacobian, scales = local_equations_jacobian_and_scales(
            model,
            candidate,
            chart,
        )
        relative_residual = float(
            np.max(np.abs(equations) / np.maximum(1.0, scales))
        )
        if not np.isfinite(relative_residual):
            raise FloatingPointError("local Newton residual is not finite")
        if relative_residual <= residual_tolerance:
            return candidate, iteration, relative_residual
        if iteration == maximum_iterations:
            break
        matrix = jacobian[:, dependent]
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        if singular_values[-1] <= 1e-12:
            raise FloatingPointError("local Newton dependent Jacobian is singular")
        step = np.linalg.solve(matrix, equations)
        accepted = False
        for damping_power in range(9):
            damping = 2.0 ** (-damping_power)
            trial = candidate.copy()
            trial[list(dependent)] -= damping * step
            trial_equations, _, trial_scales = local_equations_jacobian_and_scales(
                model,
                trial,
                chart,
            )
            trial_residual = float(
                np.max(np.abs(trial_equations) / np.maximum(1.0, trial_scales))
            )
            if np.isfinite(trial_residual) and trial_residual < relative_residual:
                candidate = trial
                best_residual = trial_residual
                accepted = True
                break
        if not accepted:
            raise FloatingPointError("local Newton line search did not decrease residual")
    raise FloatingPointError(
        "local Newton solve did not converge "
        f"(best relative residual {best_residual:.3e})"
    )


def retract_hirzebruch_type21_intrinsic_step(
    model: HirzebruchType21Model,
    center: HirzebruchType21Point,
    intrinsic_step: Array,
    *,
    residual_tolerance: float = 1e-10,
    maximum_newton_iterations: int = 16,
) -> tuple[HirzebruchType21Point, dict[str, object]]:
    """Retract one prescribed intrinsic tangent step back to the threefold."""

    step = np.asarray(intrinsic_step, dtype=np.complex128).reshape(-1)
    if step.shape != (COMPLEX_DIMENSION,) or not np.all(np.isfinite(step)):
        raise ValueError("intrinsic step must contain three finite complex values")
    if residual_tolerance <= 0 or maximum_newton_iterations <= 0:
        raise ValueError("retraction solver tolerances must be positive")
    chart = center.projective_chart
    independent = center.independent_indices
    dependent = center.dependent_indices
    affine = np.asarray(center.affine_coordinates, dtype=np.complex128).copy()
    affine[list(independent)] += step
    affine[list(dependent)] += center.tangent_basis[list(dependent), :] @ step
    solved, iterations, relative_residual = _damped_dependent_coordinate_newton(
        model,
        affine,
        chart,
        dependent,
        residual_tolerance=residual_tolerance,
        maximum_iterations=maximum_newton_iterations,
    )
    coordinates = homogeneous_product_from_affine(
        solved,
        chart,
        FACTOR_DIMENSIONS,
    )
    normalized = tuple(factor / np.linalg.norm(factor) for factor in coordinates)
    stable_chart = choose_product_chart(normalized)
    point = hirzebruch_type21_point_in_chart(
        model,
        normalized,
        stable_chart,  # type: ignore[arg-type]
    )
    distance = hirzebruch_type21_product_fubini_study_distance(center, point)
    if not np.isfinite(distance):
        raise FloatingPointError("retraction distance is not finite")
    return point, {
        "intrinsic_step_norm": float(np.linalg.norm(step)),
        "product_fubini_study_distance": float(distance),
        "relative_equation_residual": float(relative_residual),
        "newton_iterations": int(iterations),
        "source_projective_chart": [int(value) for value in chart],
        "result_projective_chart": [int(value) for value in stable_chart],
    }


def sample_hirzebruch_type21_point_neighborhood(
    model: HirzebruchType21Model,
    center: HirzebruchType21Point,
    count: int,
    *,
    seed: int,
    radius: float,
    residual_tolerance: float = 1e-10,
    maximum_newton_iterations: int = 16,
    maximum_attempt_factor: int = 20,
) -> tuple[list[HirzebruchType21Point], dict[str, object]]:
    """Sample a metric-whitened on-manifold neighborhood of one point."""

    if count <= 0:
        raise ValueError("neighborhood point count must be positive")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("neighborhood radius must be finite and positive")
    if residual_tolerance <= 0 or maximum_newton_iterations <= 0:
        raise ValueError("neighborhood solver tolerances must be positive")
    rng = np.random.default_rng(seed)
    center_metric = hirzebruch_type21_baseline_metric(center)
    eigenvalues, eigenvectors = np.linalg.eigh(center_metric)
    if eigenvalues[0] <= 0:
        raise FloatingPointError("center metric is not positive definite")
    inverse_square_root = (
        eigenvectors * (1.0 / np.sqrt(eigenvalues))[None, :]
    ) @ eigenvectors.conjugate().T
    output: list[HirzebruchType21Point] = []
    distances: list[float] = []
    residuals: list[float] = []
    iteration_counts: list[int] = []
    rejection_reasons: dict[str, int] = {}
    maximum_attempts = maximum_attempt_factor * count
    attempts = 0
    while len(output) < count and attempts < maximum_attempts:
        attempts += 1
        direction = rng.normal(size=COMPLEX_DIMENSION) + 1j * rng.normal(
            size=COMPLEX_DIMENSION
        )
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm <= 1e-14:
            rejection_reasons["degenerate_direction"] = (
                rejection_reasons.get("degenerate_direction", 0) + 1
            )
            continue
        intrinsic_step = radius * (inverse_square_root @ (direction / norm))
        try:
            point, retraction = retract_hirzebruch_type21_intrinsic_step(
                model,
                center,
                intrinsic_step,
                residual_tolerance=residual_tolerance,
                maximum_newton_iterations=maximum_newton_iterations,
            )
            distance = float(retraction["product_fubini_study_distance"])
            if not np.isfinite(distance) or distance <= 0:
                raise FloatingPointError("neighborhood distance is not positive")
        except (FloatingPointError, np.linalg.LinAlgError, ValueError) as exc:
            reason = type(exc).__name__
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
            continue
        output.append(point)
        distances.append(distance)
        residuals.append(float(retraction["relative_equation_residual"]))
        iteration_counts.append(int(retraction["newton_iterations"]))
    if len(output) < count:
        raise RuntimeError(
            f"only sampled {len(output)} of {count} requested neighborhood points "
            f"after {attempts} attempts"
        )
    diagnostics: dict[str, object] = {
        "sampler": "intrinsic_metric_whitened_damped_newton",
        "seed": int(seed),
        "requested_points": int(count),
        "returned_points": int(len(output)),
        "radius": float(radius),
        "attempts": int(attempts),
        "acceptance_rate": float(len(output) / attempts),
        "minimum_product_fubini_study_distance": float(np.min(distances)),
        "mean_product_fubini_study_distance": float(np.mean(distances)),
        "maximum_product_fubini_study_distance": float(np.max(distances)),
        "maximum_relative_equation_residual": float(np.max(residuals)),
        "maximum_newton_iterations_used": int(np.max(iteration_counts)),
        "rejection_reasons": rejection_reasons,
        "independent_monte_carlo_sample": False,
    }
    return output, diagnostics


def hirzebruch_type21_holomorphic_volume_log_density(
    point: HirzebruchType21Point,
) -> float:
    if abs(point.residue_denominator) < 1e-14:
        raise FloatingPointError("residue denominator is too small")
    return float(-2.0 * np.log(abs(point.residue_denominator)))


def hirzebruch_type21_proposal_log_density(
    point: HirzebruchType21Point,
) -> float:
    """Density of omega_y wedge omega_x^2 / 2! in intrinsic coordinates."""

    x_metric = fubini_study_metric(point.affine_coordinates[:4])
    x_tangent = point.tangent_basis[:4, :]
    pulled_x = x_tangent.conjugate().T @ x_metric @ x_tangent
    y_metric = float(np.real(fubini_study_metric(point.affine_coordinates[4:5])[0, 0]))
    y_vector = point.tangent_basis[4, :]
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


def hirzebruch_type21_importance_log_weight(
    point: HirzebruchType21Point,
) -> float:
    return (
        hirzebruch_type21_holomorphic_volume_log_density(point)
        - hirzebruch_type21_proposal_log_density(point)
    )


def hirzebruch_type21_monge_ampere_log_error(
    point: HirzebruchType21Point,
    metric: Array,
) -> float:
    eigenvalues = np.linalg.eigvalsh(np.asarray(metric, dtype=np.complex128))
    if eigenvalues[0] <= 0:
        raise FloatingPointError("metric is not positive definite")
    return float(
        np.sum(np.log(eigenvalues))
        - hirzebruch_type21_holomorphic_volume_log_density(point)
    )


def hirzebruch_type21_section_values_and_jacobian(
    point: HirzebruchType21Point,
    exponents: Array,
) -> tuple[Array, Array]:
    values, ambient_jacobian = affine_monomial_values_and_jacobian(
        point.affine_coordinates,
        point.projective_chart,
        exponents,
        FACTOR_DIMENSIONS,
    )
    return values, ambient_jacobian @ point.tangent_basis


def hirzebruch_type21_restricted_ambient_basis(
    points: Sequence[HirzebruchType21Point],
    degree: tuple[int, ...],
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


def hirzebruch_type21_restriction_coefficients(
    points: Sequence[HirzebruchType21Point],
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
    all_values /= np.maximum(np.linalg.norm(all_values, axis=1, keepdims=True), 1e-14)
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


def hirzebruch_type21_restricted_fubini_study_h_matrix(
    points: Sequence[HirzebruchType21Point],
    basis: RestrictedSectionBasis,
) -> tuple[Array, float]:
    coefficients, relation_error = hirzebruch_type21_restriction_coefficients(
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


def hirzebruch_type21_global_h_metric(
    point: HirzebruchType21Point,
    exponents: Array,
    h_matrix: Array,
    *,
    normalization: float = 1.0,
) -> Array:
    values, jacobian = hirzebruch_type21_section_values_and_jacobian(
        point,
        exponents,
    )
    return h_matrix_metric(
        values,
        jacobian,
        h_matrix,
        normalization=normalization,
    )
