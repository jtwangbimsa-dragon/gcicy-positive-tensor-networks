"""Explicit type-(2,1) gCICY model in P3 x P1 x P1 x P1.

The configuration is the second (9,13) example in Table 13 of
arXiv:1507.03235::

    [ P3 | 2 0 |  2 ]
    [ P1 | 0 3 | -1 ]
    [ P1 | 0 1 |  1 ]
    [ P1 | 1 1 |  0 ]

Write the positive equations as

    p1 = w0 A(x) + w1 B(x),
    p2 = w0 C(y,z) + w1 D(y,z).

Eliminating w gives F = A D - B C.  If

    F = f0 y0^3 + f1 y0^2 y1 + f2 y0 y1^2 + f3 y1^3,

then H0(M, O(2,-1,1,0)) has dimension three.  The local representatives
implemented below are the divided differences F(t)/(y0-t*y1), reduced
modulo F.  This produces regular polynomial representatives on both y
charts without introducing numerical pole cancellations.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from math import factorial
from typing import Sequence

import numpy as np
from scipy.linalg import qr

from .global_sections import RestrictedSectionBasis, h_matrix_metric
from .simple_patch import fubini_study_metric


Array = np.ndarray
Chart = tuple[int, int, int, int]
FACTOR_DIMENSIONS = (3, 1, 1, 1)
FACTOR_SIZES = tuple(value + 1 for value in FACTOR_DIMENSIONS)
FACTOR_OFFSETS = (0, 4, 6, 8)
AMBIENT_COORDINATE_COUNT = sum(FACTOR_SIZES)
AMBIENT_AFFINE_DIMENSION = sum(FACTOR_DIMENSIONS)
COMPLEX_DIMENSION = 3
CONSTRAINT_COUNT = 3


@dataclass(frozen=True)
class Type21Model:
    """Dense coefficients for one explicit member of the type-(2,1) family."""

    seed: int
    x_quadratic_exponents: Array
    a_coefficients: Array
    b_coefficients: Array
    c_tensor: Array
    d_tensor: Array
    q_moments: Array
    p1_exponents: Array
    p1_coefficients: Array
    p2_exponents: Array
    p2_coefficients: Array


@dataclass(frozen=True)
class Type21Point:
    """One homogeneous point with exact implicit-coordinate data."""

    coordinates: tuple[Array, Array, Array, Array]
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

    @property
    def w(self) -> Array:
        return self.coordinates[3]


def _compositions(total: int, length: int) -> list[tuple[int, ...]]:
    if total < 0 or length <= 0:
        raise ValueError("total must be non-negative and length must be positive")
    if length == 1:
        return [(total,)]
    output = []
    for first in range(total + 1):
        for rest in _compositions(total - first, length - 1):
            output.append((first, *rest))
    return output


def product_monomial_exponents(degree: tuple[int, ...]) -> Array:
    """Enumerate ambient monomials for a non-negative four-factor degree."""

    if len(degree) != len(FACTOR_SIZES) or min(degree) < 0:
        raise ValueError("degree must be a non-negative four-entry tuple")
    blocks = [_compositions(value, size) for value, size in zip(degree, FACTOR_SIZES, strict=True)]
    rows = [tuple(entry for block in choice for entry in block) for choice in product(*blocks)]
    return np.asarray(rows, dtype=np.int64)


def _normalized_random_coefficients(rng: np.random.Generator, count: int) -> Array:
    values = rng.normal(size=count) + 1j * rng.normal(size=count)
    return np.asarray(values / np.linalg.norm(values), dtype=np.complex128)


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
    x_exponents: Array,
    a_coefficients: Array,
    b_coefficients: Array,
    c_tensor: Array,
    d_tensor: Array,
) -> tuple[Array, Array, Array, Array]:
    p1_exponents = []
    p1_coefficients = []
    for exponents, coefficient in zip(x_exponents, a_coefficients, strict=True):
        row = np.zeros(AMBIENT_COORDINATE_COUNT, dtype=np.int64)
        row[:4] = exponents
        row[8] = 1
        p1_exponents.append(row)
        p1_coefficients.append(coefficient)
    for exponents, coefficient in zip(x_exponents, b_coefficients, strict=True):
        row = np.zeros(AMBIENT_COORDINATE_COUNT, dtype=np.int64)
        row[:4] = exponents
        row[9] = 1
        p1_exponents.append(row)
        p1_coefficients.append(coefficient)

    p2_exponents = []
    p2_coefficients = []
    for y_power in range(4):
        for z_index in range(2):
            for w_index, tensor in ((0, c_tensor), (1, d_tensor)):
                row = np.zeros(AMBIENT_COORDINATE_COUNT, dtype=np.int64)
                row[4] = 3 - y_power
                row[5] = y_power
                row[6 + z_index] = 1
                row[8 + w_index] = 1
                p2_exponents.append(row)
                p2_coefficients.append(tensor[y_power, z_index])
    return (
        np.asarray(p1_exponents, dtype=np.int64),
        np.asarray(p1_coefficients, dtype=np.complex128),
        np.asarray(p2_exponents, dtype=np.int64),
        np.asarray(p2_coefficients, dtype=np.complex128),
    )


def make_type21_model(
    seed: int = 20260721,
    *,
    exact: bool = True,
    coefficient_bound: int = 3,
) -> Type21Model:
    """Construct deterministic dense coefficients and a generic generalized section."""

    if coefficient_bound < 1:
        raise ValueError("coefficient_bound must be positive")
    rng = np.random.default_rng(seed)
    x_exponents = np.asarray(_compositions(2, 4), dtype=np.int64)
    draw = (
        (lambda count: _integer_coefficients(rng, count, coefficient_bound))
        if exact
        else (lambda count: _normalized_random_coefficients(rng, count))
    )
    a_coefficients = draw(len(x_exponents))
    b_coefficients = draw(len(x_exponents))
    c_tensor = draw(8).reshape(4, 2)
    d_tensor = draw(8).reshape(4, 2)
    q_moments = draw(3)
    p1_exponents, p1_coefficients, p2_exponents, p2_coefficients = _positive_polynomial_data(
        x_exponents,
        a_coefficients,
        b_coefficients,
        c_tensor,
        d_tensor,
    )
    return Type21Model(
        seed=seed,
        x_quadratic_exponents=x_exponents,
        a_coefficients=a_coefficients,
        b_coefficients=b_coefficients,
        c_tensor=c_tensor,
        d_tensor=d_tensor,
        q_moments=q_moments,
        p1_exponents=p1_exponents,
        p1_coefficients=p1_coefficients,
        p2_exponents=p2_exponents,
        p2_coefficients=p2_coefficients,
    )


def _evaluate_monomials(values: Array, exponents: Array) -> Array:
    coordinates = np.asarray(values, dtype=np.complex128).reshape(-1)
    powers = np.asarray(exponents, dtype=np.int64)
    return np.prod(coordinates[None, :] ** powers, axis=1)


def evaluate_ab(model: Type21Model, x: Array) -> tuple[complex, complex]:
    values = _evaluate_monomials(x, model.x_quadratic_exponents)
    return complex(model.a_coefficients @ values), complex(model.b_coefficients @ values)


def _y_cubic_monomials(y: Array) -> Array:
    y0, y1 = np.asarray(y, dtype=np.complex128)
    return np.asarray([y0**3, y0**2 * y1, y0 * y1**2, y1**3], dtype=np.complex128)


def evaluate_cd(model: Type21Model, y: Array, z: Array) -> tuple[complex, complex]:
    y_values = _y_cubic_monomials(y)
    z_values = np.asarray(z, dtype=np.complex128)
    c_value = np.einsum("i,ij,j->", y_values, model.c_tensor, z_values)
    d_value = np.einsum("i,ij,j->", y_values, model.d_tensor, z_values)
    return complex(c_value), complex(d_value)


def determinant_coefficient_vectors(model: Type21Model, x: Array) -> Array:
    """Return f_i(z) rows in F=sum_i f_i(z)y0^(3-i)y1^i."""

    a_value, b_value = evaluate_ab(model, x)
    return np.asarray(
        a_value * model.d_tensor - b_value * model.c_tensor,
        dtype=np.complex128,
    )


def evaluate_positive_equations(
    model: Type21Model,
    x: Array,
    y: Array,
    z: Array,
    w: Array,
) -> tuple[complex, complex]:
    a_value, b_value = evaluate_ab(model, x)
    c_value, d_value = evaluate_cd(model, y, z)
    return (
        complex(w[0] * a_value + w[1] * b_value),
        complex(w[0] * c_value + w[1] * d_value),
    )


def normalize_product_coordinates(
    coordinates: Sequence[Array],
    chart: Chart,
) -> tuple[Array, Array, Array, Array]:
    if len(coordinates) != len(FACTOR_SIZES) or len(chart) != len(FACTOR_SIZES):
        raise ValueError("coordinates and chart must have four factors")
    output = []
    for values, selected, size in zip(coordinates, chart, FACTOR_SIZES, strict=True):
        array = np.asarray(values, dtype=np.complex128).reshape(-1)
        if array.shape != (size,) or not 0 <= selected < size:
            raise ValueError("coordinate or chart shape does not match the ambient product")
        denominator = array[selected]
        if abs(denominator) < 1e-14:
            raise FloatingPointError("requested projective chart is not available")
        output.append(array / denominator)
    return tuple(output)  # type: ignore[return-value]


def affine_coordinates_from_product(
    coordinates: Sequence[Array],
    chart: Chart,
) -> Array:
    normalized = normalize_product_coordinates(coordinates, chart)
    return np.concatenate(
        [np.delete(values, selected) for values, selected in zip(normalized, chart, strict=True)]
    )


def homogeneous_product_from_affine(chart_coordinates: Array, chart: Chart) -> tuple[Array, ...]:
    affine = np.asarray(chart_coordinates, dtype=np.complex128).reshape(-1)
    if affine.shape != (AMBIENT_AFFINE_DIMENSION,):
        raise ValueError("unexpected affine-coordinate shape")
    output = []
    cursor = 0
    for size, selected in zip(FACTOR_SIZES, chart, strict=True):
        values = np.empty(size, dtype=np.complex128)
        values[selected] = 1.0
        mask = np.arange(size) != selected
        values[mask] = affine[cursor : cursor + size - 1]
        cursor += size - 1
        output.append(values)
    return tuple(output)


def choose_product_chart(coordinates: Sequence[Array]) -> Chart:
    return tuple(int(np.argmax(np.abs(values))) for values in coordinates)  # type: ignore[return-value]


def _active_homogeneous_indices(chart: Chart) -> list[int]:
    selected = {
        offset + chart_index
        for offset, chart_index in zip(FACTOR_OFFSETS, chart, strict=True)
    }
    return [index for index in range(AMBIENT_COORDINATE_COUNT) if index not in selected]


def affine_monomial_values_and_jacobian(
    chart_coordinates: Array,
    chart: Chart,
    exponents: Array,
) -> tuple[Array, Array]:
    """Evaluate homogeneous monomials and analytic affine derivatives."""

    coordinates = homogeneous_product_from_affine(chart_coordinates, chart)
    homogeneous = np.concatenate(coordinates)
    powers = np.asarray(exponents, dtype=np.int64)
    if powers.ndim != 2 or powers.shape[1] != AMBIENT_COORDINATE_COUNT:
        raise ValueError("unexpected exponent shape")
    active = _active_homogeneous_indices(chart)
    values = np.prod(homogeneous[None, :] ** powers, axis=1)
    jacobian = np.empty((len(powers), AMBIENT_AFFINE_DIMENSION), dtype=np.complex128)
    for column, homogeneous_index in enumerate(active):
        exponent = powers[:, homogeneous_index]
        reduced = powers.copy()
        reduced[:, homogeneous_index] = np.maximum(reduced[:, homogeneous_index] - 1, 0)
        jacobian[:, column] = exponent * np.prod(
            homogeneous[None, :] ** reduced,
            axis=1,
        )
    return values, jacobian


def _determinant_terms(model: Type21Model, determinant_index: int) -> list[tuple[Array, complex]]:
    """Expand f_i=A*D_i-B*C_i into x,z monomials."""

    terms = []
    for exponents, coefficient in zip(
        model.x_quadratic_exponents,
        model.a_coefficients,
        strict=True,
    ):
        for z_index in range(2):
            row = np.zeros(AMBIENT_COORDINATE_COUNT, dtype=np.int64)
            row[:4] = exponents
            row[6 + z_index] = 1
            terms.append((row, coefficient * model.d_tensor[determinant_index, z_index]))
    for exponents, coefficient in zip(
        model.x_quadratic_exponents,
        model.b_coefficients,
        strict=True,
    ):
        for z_index in range(2):
            row = np.zeros(AMBIENT_COORDINATE_COUNT, dtype=np.int64)
            row[:4] = exponents
            row[6 + z_index] = 1
            terms.append((row, -coefficient * model.c_tensor[determinant_index, z_index]))
    return terms


def q_polynomial_data(model: Type21Model, chart: Chart) -> tuple[Array, Array]:
    """Return the regular polynomial representative of q in one y chart."""

    l0, l1, l2 = model.q_moments
    if chart[1] == 1:
        weighted_terms = (
            (0, 2, -l0),
            (0, 1, -l1),
            (0, 0, -l2),
            (1, 1, -l0),
            (1, 0, -l1),
            (2, 0, -l0),
        )
        active_y_index = 4
    elif chart[1] == 0:
        weighted_terms = (
            (1, 0, l2),
            (2, 0, l1),
            (2, 1, l2),
            (3, 0, l0),
            (3, 1, l1),
            (3, 2, l2),
        )
        active_y_index = 5
    else:
        raise ValueError("the y chart index must be zero or one")

    coefficients_by_exponent: dict[tuple[int, ...], complex] = {}
    for determinant_index, y_power, multiplier in weighted_terms:
        for base_exponents, coefficient in _determinant_terms(model, determinant_index):
            exponents = base_exponents.copy()
            exponents[active_y_index] = y_power
            key = tuple(int(value) for value in exponents)
            coefficients_by_exponent[key] = (
                coefficients_by_exponent.get(key, 0.0 + 0.0j)
                + multiplier * coefficient
            )
    nonzero = [
        (key, value)
        for key, value in coefficients_by_exponent.items()
        if abs(value) > 0
    ]
    return (
        np.asarray([key for key, _ in nonzero], dtype=np.int64),
        np.asarray([value for _, value in nonzero], dtype=np.complex128),
    )


def local_polynomial_data(
    model: Type21Model,
    chart: Chart,
) -> tuple[tuple[Array, Array], tuple[Array, Array], tuple[Array, Array]]:
    """Return exact polynomial data for p1, p2, and the local q representative."""

    return (
        (model.p1_exponents, model.p1_coefficients),
        (model.p2_exponents, model.p2_coefficients),
        q_polynomial_data(model, chart),
    )


def local_equations_jacobian_and_scales(
    model: Type21Model,
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
    model: Type21Model,
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
            best = (condition, dependent, independent, complex(np.linalg.det(matrix)))
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
    determinant = complex(np.linalg.det(matrix))
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


def type21_point_in_chart(
    model: Type21Model,
    coordinates: Sequence[Array],
    chart: Chart,
    *,
    independent: tuple[int, int, int] | None = None,
) -> Type21Point:
    affine = affine_coordinates_from_product(coordinates, chart)
    equations, jacobian, scales = local_equations_jacobian_and_scales(model, affine, chart)
    relative_residual = float(
        np.max(np.abs(equations) / np.maximum(1.0, scales))
    )
    if not np.isfinite(relative_residual) or relative_residual > 2e-9:
        raise FloatingPointError(
            "point does not satisfy the local type-(2,1) equations "
            f"(relative residual {relative_residual:.3e})"
        )
    if independent is None:
        independent, dependent, tangent, determinant = _best_implicit_coordinates(jacobian)
    else:
        dependent, tangent, determinant = _fixed_implicit_coordinates(jacobian, independent)
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    copied = tuple(np.asarray(values, dtype=np.complex128).copy() for values in coordinates)
    return Type21Point(
        coordinates=copied,  # type: ignore[arg-type]
        projective_chart=chart,
        affine_coordinates=affine,
        independent_indices=independent,
        dependent_indices=dependent,
        tangent_basis=tangent,
        residue_denominator=determinant,
        jacobian_min_singular_value=float(singular_values[-1]),
    )


def _right_null_vector(matrix: Array) -> Array:
    _, singular_values, vh = np.linalg.svd(np.asarray(matrix, dtype=np.complex128))
    if singular_values[-1] > 1e-7 * max(1.0, singular_values[0]):
        raise FloatingPointError("matrix does not have a stable projective null vector")
    vector = vh[-1].conjugate()
    norm = float(np.linalg.norm(vector))
    if norm < 1e-14:
        raise FloatingPointError("null vector has zero norm")
    return vector / norm


def _random_projective_point(rng: np.random.Generator, size: int) -> Array:
    values = rng.normal(size=size) + 1j * rng.normal(size=size)
    return np.asarray(values / np.linalg.norm(values), dtype=np.complex128)


def _sample_over_x(
    model: Type21Model,
    rng: np.random.Generator,
    *,
    residual_tolerance: float,
) -> list[tuple[Array, Array, Array, Array]]:
    x = _random_projective_point(rng, 4)
    f_vectors = determinant_coefficient_vectors(model, x)
    l0, l1, l2 = model.q_moments
    q_vectors = np.asarray(
        [
            l2 * f_vectors[1] + l1 * f_vectors[2] + l0 * f_vectors[3],
            l2 * f_vectors[2] + l1 * f_vectors[3],
            l2 * f_vectors[3],
        ],
        dtype=np.complex128,
    )
    determinant = np.polynomial.polynomial.polysub(
        np.polynomial.polynomial.polymul(f_vectors[:, 0], q_vectors[:, 1]),
        np.polynomial.polynomial.polymul(f_vectors[:, 1], q_vectors[:, 0]),
    )
    scale = max(1.0, float(np.max(np.abs(determinant))))
    while len(determinant) > 1 and abs(determinant[-1]) < 1e-11 * scale:
        determinant = determinant[:-1]
    if len(determinant) <= 1:
        return []
    roots = np.polynomial.polynomial.polyroots(determinant)
    output = []
    for root in roots:
        y = np.asarray([1.0 + 0.0j, root], dtype=np.complex128)
        y /= np.linalg.norm(y)
        powers_f = np.asarray([1.0, root, root**2, root**3], dtype=np.complex128)
        powers_q = np.asarray([1.0, root, root**2], dtype=np.complex128)
        f_row = powers_f @ f_vectors
        q_row = powers_q @ q_vectors
        try:
            z = _right_null_vector(np.asarray([f_row, q_row]))
        except FloatingPointError:
            continue
        a_value, b_value = evaluate_ab(model, x)
        c_value, d_value = evaluate_cd(model, y, z)
        try:
            w = _right_null_vector(
                np.asarray([[a_value, b_value], [c_value, d_value]], dtype=np.complex128)
            )
        except FloatingPointError:
            continue
        coordinates = (x.copy(), y, z, w)
        chart = choose_product_chart(coordinates)
        affine = affine_coordinates_from_product(coordinates, chart)
        equations, _, scales = local_equations_jacobian_and_scales(model, affine, chart)
        relative_residual = float(
            np.max(np.abs(equations) / np.maximum(1.0, scales))
        )
        if relative_residual <= residual_tolerance:
            output.append(coordinates)
    return output


def sample_type21_points(
    model: Type21Model,
    count: int,
    *,
    seed: int,
    residual_tolerance: float = 2e-9,
    max_attempt_factor: int = 30,
) -> list[Type21Point]:
    """Sample the two-sheeted projection X -> P3 and reconstruct z and w."""

    if count <= 0:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    raw_points: list[tuple[Array, Array, Array, Array]] = []
    attempts = 0
    while len(raw_points) < count and attempts < max_attempt_factor * count:
        raw_points.extend(
            _sample_over_x(
                model,
                rng,
                residual_tolerance=residual_tolerance,
            )
        )
        attempts += 1
    if len(raw_points) < count:
        raise RuntimeError(
            f"only sampled {len(raw_points)} points after {attempts} projection attempts"
        )
    output = []
    for coordinates in raw_points[:count]:
        chart = choose_product_chart(coordinates)
        try:
            output.append(type21_point_in_chart(model, coordinates, chart))
        except (FloatingPointError, np.linalg.LinAlgError):
            continue
    if len(output) != count:
        raise RuntimeError("some type-(2,1) samples failed implicit-coordinate validation")
    return output


def product_fubini_study_metric(
    chart_coordinates: Array,
    *,
    weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> Array:
    coordinates = np.asarray(chart_coordinates, dtype=np.complex128).reshape(-1)
    if coordinates.shape != (AMBIENT_AFFINE_DIMENSION,):
        raise ValueError("unexpected affine-coordinate shape")
    metric = np.zeros(
        (AMBIENT_AFFINE_DIMENSION, AMBIENT_AFFINE_DIMENSION),
        dtype=np.complex128,
    )
    cursor = 0
    for dimension, weight in zip(FACTOR_DIMENSIONS, weights, strict=True):
        block = slice(cursor, cursor + dimension)
        metric[block, block] = weight * fubini_study_metric(coordinates[block])
        cursor += dimension
    return metric


def type21_baseline_metric(point: Type21Point) -> Array:
    ambient = product_fubini_study_metric(point.affine_coordinates)
    metric = point.tangent_basis.conjugate().T @ ambient @ point.tangent_basis
    return 0.5 * (metric + metric.conjugate().T)


def type21_holomorphic_volume_log_density(point: Type21Point) -> float:
    if abs(point.residue_denominator) < 1e-14:
        raise FloatingPointError("residue denominator is too small")
    return float(-2.0 * np.log(abs(point.residue_denominator)))


def type21_proposal_log_density(point: Type21Point) -> float:
    """Density of the P3 Fubini-Study proposal pulled back to X."""

    x_metric = fubini_study_metric(point.affine_coordinates[:3])
    x_tangent = point.tangent_basis[:3, :]
    metric = x_tangent.conjugate().T @ x_metric @ x_tangent
    eigenvalues = np.linalg.eigvalsh(0.5 * (metric + metric.conjugate().T))
    if eigenvalues[0] <= 0:
        raise FloatingPointError("projection proposal density is not positive")
    return float(np.sum(np.log(eigenvalues)))


def type21_importance_log_weight(point: Type21Point) -> float:
    return (
        type21_holomorphic_volume_log_density(point)
        - type21_proposal_log_density(point)
    )


def type21_monge_ampere_log_error(point: Type21Point, metric: Array) -> float:
    eigenvalues = np.linalg.eigvalsh(np.asarray(metric, dtype=np.complex128))
    if eigenvalues[0] <= 0:
        raise FloatingPointError("metric is not positive definite")
    return float(
        np.sum(np.log(eigenvalues))
        - type21_holomorphic_volume_log_density(point)
    )


def evaluate_product_sections_in_chart(
    coordinates: Sequence[Array],
    chart: Chart,
    exponents: Array,
) -> Array:
    normalized = normalize_product_coordinates(coordinates, chart)
    return _evaluate_monomials(np.concatenate(normalized), exponents)


def type21_section_values_and_jacobian(
    point: Type21Point,
    exponents: Array,
) -> tuple[Array, Array]:
    values, ambient_jacobian = affine_monomial_values_and_jacobian(
        point.affine_coordinates,
        point.projective_chart,
        exponents,
    )
    return values, ambient_jacobian @ point.tangent_basis


def type21_restricted_ambient_basis(
    points: Sequence[Type21Point],
    degree: tuple[int, ...],
    *,
    relative_rank_threshold: float = 1e-10,
) -> RestrictedSectionBasis:
    exponents = product_monomial_exponents(degree)
    matrix = np.asarray(
        [
            evaluate_product_sections_in_chart(
                point.coordinates,
                point.projective_chart,
                exponents,
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


def product_multinomial_section_weights(exponents: Array) -> Array:
    powers = np.asarray(exponents, dtype=np.int64)
    if powers.ndim != 2 or powers.shape[1] != AMBIENT_COORDINATE_COUNT:
        raise ValueError("unexpected exponent shape")
    weights = np.ones(len(powers), dtype=float)
    for offset, size in zip(FACTOR_OFFSETS, FACTOR_SIZES, strict=True):
        block = powers[:, offset : offset + size]
        for row, degree in enumerate(np.sum(block, axis=1)):
            numerator = factorial(int(degree))
            denominator = int(
                np.prod([factorial(int(value)) for value in block[row]])
            )
            weights[row] *= numerator / denominator
    return weights


def type21_restriction_coefficients(
    points: Sequence[Type21Point],
    basis: RestrictedSectionBasis,
) -> tuple[Array, float]:
    all_values = np.asarray(
        [
            evaluate_product_sections_in_chart(
                point.coordinates,
                point.projective_chart,
                basis.ambient_exponents,
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


def type21_restricted_fubini_study_h_matrix(
    points: Sequence[Type21Point],
    basis: RestrictedSectionBasis,
) -> tuple[Array, float]:
    coefficients, relation_error = type21_restriction_coefficients(points, basis)
    weights = product_multinomial_section_weights(basis.ambient_exponents)
    h_matrix = np.conjugate(coefficients) @ (weights[:, None] * coefficients.T)
    h_matrix = 0.5 * (h_matrix + h_matrix.conjugate().T)
    return h_matrix, relation_error


def type21_global_h_metric(
    point: Type21Point,
    exponents: Array,
    h_matrix: Array,
    *,
    normalization: float = 1.0,
) -> Array:
    values, jacobian = type21_section_values_and_jacobian(point, exponents)
    return h_matrix_metric(
        values,
        jacobian,
        h_matrix,
        normalization=normalization,
    )
