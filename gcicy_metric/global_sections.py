"""Global ambient section bases restricted to the explicit gCICY.

For non-negative multi-degree (kx, ky, kz), every homogeneous ambient
monomial of O(kx,ky,kz) restricts to a genuine global section on X.  The
restrictions need not be linearly independent because the defining equations
vanish on X.  This module evaluates the sections in arbitrary projective
charts and uses rank-revealing QR on sampled points to select a stable basis
for the restricted ambient section space.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, factorial

import numpy as np
from scipy.linalg import qr

from .atlas import Chart, affine_chart_coordinates, homogeneous_coordinates
from .global_geometry import homogeneous_from_affine, normalize_homogeneous, section_transition_multiplier
from .simple_patch import Array, embedding, embedding_jacobian
from .simple_patch import holomorphic_jacobian


MultiDegree = tuple[int, int, int]


@dataclass(frozen=True)
class RestrictedSectionBasis:
    """A numerically independent subset of restricted ambient monomials."""

    degree: MultiDegree
    ambient_exponents: Array
    selected_indices: Array
    selected_exponents: Array
    numerical_rank: int
    singular_values: Array
    relative_rank_threshold: float


def _compositions(total: int, length: int) -> list[tuple[int, ...]]:
    if total < 0 or length <= 0:
        raise ValueError("Expected a non-negative total and positive length.")
    if length == 1:
        return [(total,)]
    output: list[tuple[int, ...]] = []
    for first in range(total + 1):
        for rest in _compositions(total - first, length - 1):
            output.append((first, *rest))
    return output


def ambient_section_count(degree: MultiDegree) -> int:
    """Return h0 of the ambient O(kx,ky,kz) for non-negative degree."""

    kx, ky, kz = degree
    if min(degree) < 0:
        raise ValueError("Ambient monomial sections require non-negative degrees.")
    return (kx + 1) * (ky + 1) * comb(kz + 5, 5)


def ambient_monomial_exponents(degree: MultiDegree) -> Array:
    """Return exponent rows ordered as (x0,x1,y0,y1,z0,...,z5)."""

    if min(degree) < 0:
        raise ValueError("Ambient monomial sections require non-negative degrees.")
    x_exponents = _compositions(degree[0], 2)
    y_exponents = _compositions(degree[1], 2)
    z_exponents = _compositions(degree[2], 6)
    rows = [(*x_power, *y_power, *z_power) for x_power in x_exponents for y_power in y_exponents for z_power in z_exponents]
    result = np.asarray(rows, dtype=np.int64)
    if len(result) != ambient_section_count(degree):
        raise RuntimeError("Ambient section enumeration produced the wrong dimension.")
    return result


def evaluate_ambient_monomials(x: Array, y: Array, z: Array, exponents: Array) -> Array:
    """Evaluate homogeneous ambient monomials on one representative."""

    values = np.concatenate(
        [
            np.asarray(x, dtype=np.complex128).reshape(-1),
            np.asarray(y, dtype=np.complex128).reshape(-1),
            np.asarray(z, dtype=np.complex128).reshape(-1),
        ]
    )
    powers = np.asarray(exponents, dtype=np.int64)
    if values.shape != (10,) or powers.ndim != 2 or powers.shape[1] != 10:
        raise ValueError(f"Expected coordinate shape (10,) and exponent shape (n,10), got {values.shape}, {powers.shape}.")
    return np.prod(values[None, :] ** powers, axis=1)


def evaluate_sections_in_chart(x: Array, y: Array, z: Array, chart: Chart, exponents: Array) -> Array:
    """Evaluate local representatives of global ambient sections in a chart."""

    return evaluate_ambient_monomials(*normalize_homogeneous(x, y, z, chart), exponents)


def affine_section_values_and_jacobian(chart_coords: Array, chart: Chart, exponents: Array) -> tuple[Array, Array]:
    """Evaluate local section values and derivatives in seven affine coordinates."""

    coords = np.asarray(chart_coords, dtype=np.complex128).reshape(-1)
    powers = np.asarray(exponents, dtype=np.int64)
    if coords.shape != (7,) or powers.ndim != 2 or powers.shape[1] != 10:
        raise ValueError("Unexpected affine-coordinate or exponent shape.")

    x, y, z = homogeneous_from_affine(coords, chart)
    homogeneous_values = np.concatenate([x, y, z])
    active_indices = [index for index in range(10) if index not in (chart[0], 2 + chart[1], 4 + chart[2])]
    active_powers = powers[:, active_indices]
    values = np.prod(homogeneous_values[None, :] ** powers, axis=1)
    jacobian = np.empty((len(powers), 7), dtype=np.complex128)
    for column, homogeneous_index in enumerate(active_indices):
        exponent = powers[:, homogeneous_index]
        reduced = powers.copy()
        reduced[:, homogeneous_index] = np.maximum(reduced[:, homogeneous_index] - 1, 0)
        derivative = exponent * np.prod(homogeneous_values[None, :] ** reduced, axis=1)
        jacobian[:, column] = derivative
    return values, jacobian


def reference_section_values_and_jacobian(params: Array, exponents: Array) -> tuple[Array, Array]:
    """Evaluate global sections and their (s,t,r) derivatives in the reference patch."""

    point = np.asarray(params, dtype=np.complex128).reshape(-1)
    ambient = embedding(point)
    values, ambient_jacobian = affine_section_values_and_jacobian(ambient, (0, 0, 0), exponents)
    return values, ambient_jacobian @ embedding_jacobian(point)


def section_evaluation_matrix(params: Array, exponents: Array, chart: Chart = (0, 0, 0)) -> Array:
    """Evaluate section columns on a batch of parameterized X points."""

    points = np.asarray(params, dtype=np.complex128)
    matrix = np.asarray(
        [evaluate_sections_in_chart(*homogeneous_coordinates(point), chart, exponents) for point in points],
        dtype=np.complex128,
    )
    row_norms = np.linalg.norm(matrix, axis=1)
    return matrix / np.maximum(row_norms[:, None], 1e-14)


def restricted_ambient_basis(
    params: Array,
    degree: MultiDegree,
    *,
    relative_rank_threshold: float = 1e-10,
) -> RestrictedSectionBasis:
    """Select numerically independent restricted ambient sections by pivoted QR."""

    if not 0 < relative_rank_threshold < 1:
        raise ValueError("relative_rank_threshold must lie between zero and one.")
    exponents = ambient_monomial_exponents(degree)
    matrix = section_evaluation_matrix(params, exponents)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values.size == 0 or singular_values[0] == 0:
        raise FloatingPointError("Section evaluation matrix has zero numerical rank.")
    numerical_rank = int(np.sum(singular_values > relative_rank_threshold * singular_values[0]))
    _, _, pivots = qr(matrix, mode="economic", pivoting=True)
    selected_indices = np.asarray(pivots[:numerical_rank], dtype=np.int64)
    return RestrictedSectionBasis(
        degree=degree,
        ambient_exponents=exponents,
        selected_indices=selected_indices,
        selected_exponents=exponents[selected_indices],
        numerical_rank=numerical_rank,
        singular_values=singular_values,
        relative_rank_threshold=relative_rank_threshold,
    )


def multinomial_section_weights(exponents: Array) -> Array:
    """Return product multinomial weights for an ambient monomial basis."""

    powers = np.asarray(exponents, dtype=np.int64)
    if powers.ndim != 2 or powers.shape[1] != 10:
        raise ValueError("Expected an exponent array with shape (n,10).")
    blocks = (powers[:, :2], powers[:, 2:4], powers[:, 4:])
    weights = np.ones(len(powers), dtype=float)
    for block in blocks:
        degrees = np.sum(block, axis=1)
        for row, degree in enumerate(degrees):
            numerator = factorial(int(degree))
            denominator = int(np.prod([factorial(int(value)) for value in block[row]]))
            weights[row] *= numerator / denominator
    return weights


def restricted_fubini_study_h_matrix(params: Array, basis: RestrictedSectionBasis) -> tuple[Array, float]:
    """Represent the ambient product Fubini-Study potential in a restricted basis."""

    all_values = section_evaluation_matrix(params, basis.ambient_exponents)
    selected_values = all_values[:, basis.selected_indices]
    coefficients, _, _, _ = np.linalg.lstsq(selected_values, all_values, rcond=basis.relative_rank_threshold)
    reconstruction = selected_values @ coefficients
    relation_error = float(np.linalg.norm(reconstruction - all_values) / max(1.0, np.linalg.norm(all_values)))
    weights = multinomial_section_weights(basis.ambient_exponents)
    h_matrix = np.conjugate(coefficients) @ (weights[:, None] * coefficients.T)
    h_matrix = 0.5 * (h_matrix + h_matrix.conjugate().T)
    return h_matrix, relation_error


def h_matrix_metric(values: Array, jacobian: Array, h_matrix: Array, normalization: float = 1.0) -> Array:
    """Evaluate partial-partialbar log(s^dagger H s) from values and derivatives."""

    sections = np.asarray(values, dtype=np.complex128).reshape(-1)
    derivatives = np.asarray(jacobian, dtype=np.complex128)
    h_value = np.asarray(h_matrix, dtype=np.complex128)
    h_value = 0.5 * (h_value + h_value.conjugate().T)
    if derivatives.shape[0] != len(sections) or h_value.shape != (len(sections), len(sections)):
        raise ValueError("Section, derivative, and H-matrix dimensions do not agree.")
    hs = h_value @ sections
    denominator = float(np.real(np.vdot(sections, hs)))
    if denominator <= 0:
        raise FloatingPointError("The section norm s^dagger H s is not positive.")
    h_derivatives = h_value @ derivatives
    first = derivatives.conjugate().T @ h_derivatives / denominator
    holomorphic_gradient = sections.conjugate() @ h_derivatives
    metric = first - np.outer(np.conjugate(holomorphic_gradient), holomorphic_gradient) / denominator**2
    metric *= normalization
    return 0.5 * (metric + metric.conjugate().T)


def global_h_metric(params: Array, exponents: Array, h_matrix: Array, normalization: float = 1.0) -> Array:
    """Evaluate a global-section H-matrix metric in reference local coordinates."""

    values, jacobian = reference_section_values_and_jacobian(params, exponents)
    return h_matrix_metric(values, jacobian, h_matrix, normalization=normalization)


def global_h_metrics(params: Array, exponents: Array, h_matrix: Array, normalization: float = 1.0) -> Array:
    """Evaluate a global-section H-matrix metric on a batch of points."""

    return np.asarray(
        [global_h_metric(point, exponents, h_matrix, normalization=normalization) for point in params],
        dtype=np.complex128,
    )


def global_h_metric_in_chart(
    params: Array,
    chart: Chart,
    exponents: Array,
    h_matrix: Array,
    normalization: float = 1.0,
) -> Array:
    """Evaluate and pull back the same H-matrix metric through any ambient chart."""

    chart_coords = affine_chart_coordinates(params, chart)
    values, ambient_derivatives = affine_section_values_and_jacobian(chart_coords, chart, exponents)
    chart_jacobian = holomorphic_jacobian(lambda point: affine_chart_coordinates(point, chart), params)
    return h_matrix_metric(
        values,
        ambient_derivatives @ chart_jacobian,
        h_matrix,
        normalization=normalization,
    )


def max_section_transition_error(
    params: Array,
    degree: MultiDegree,
    exponents: Array,
    *,
    min_selected: float = 1e-5,
) -> tuple[float, int]:
    """Check common line-bundle transitions of a global section vector."""

    reference_chart: Chart = (0, 0, 0)
    max_error = 0.0
    charts_seen: set[Chart] = set()
    for point in np.asarray(params, dtype=np.complex128):
        x, y, z = homogeneous_coordinates(point)
        reference = evaluate_sections_in_chart(x, y, z, reference_chart, exponents)
        for x_index in range(2):
            for y_index in range(2):
                for z_index in range(6):
                    chart = (x_index, y_index, z_index)
                    if min(abs(x[x_index]), abs(y[y_index]), abs(z[z_index])) <= min_selected:
                        continue
                    charts_seen.add(chart)
                    candidate = evaluate_sections_in_chart(x, y, z, chart, exponents)
                    multiplier = section_transition_multiplier(x, y, z, reference_chart, chart, degree)
                    expected = multiplier * reference
                    scale = max(1.0, float(np.linalg.norm(candidate)), float(np.linalg.norm(expected)))
                    max_error = max(max_error, float(np.linalg.norm(candidate - expected)) / scale)
    return max_error, len(charts_seen)
