"""A reproducible generic smooth-candidate model for the gCICY configuration.

Unlike ``simple_patch``, this module does not force the (1,1,3) equation to be
linear in one ambient coordinate.  It uses generic deterministic coefficient
tensors, solves the three linear equations p2=q1=q2 for a projective P2 fibre,
and samples the remaining plane cubic p1=0 by intersecting it with random
lines.  Local equations and Jacobians are evaluated in arbitrary projective
affine charts.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product

import numpy as np
from scipy.linalg import null_space, qr

from .atlas import Chart, ambient_chart_metric
from .global_geometry import homogeneous_from_affine, normalize_homogeneous
from .global_sections import (
    RestrictedSectionBasis,
    affine_section_values_and_jacobian,
    ambient_monomial_exponents,
    evaluate_ambient_monomials,
    evaluate_sections_in_chart,
    h_matrix_metric,
    multinomial_section_weights,
)
from .ricci_flat_fit import ResidualStats
from .simple_patch import Array


@dataclass(frozen=True)
class GenericGCICYModel:
    """Coefficient data for p1 of degree (1,1,3) and p2 of degree (1,1,1)."""

    seed: int
    p1_exponents: Array
    p1_coefficients: Array
    p2_exponents: Array
    p2_coefficients: Array
    p2_tensor: Array


@dataclass(frozen=True)
class GenericGCICYPoint:
    """One sampled homogeneous point and a well-conditioned local chart."""

    x: Array
    y: Array
    z: Array
    projective_chart: Chart
    affine_coordinates: Array
    independent_indices: tuple[int, int, int]
    dependent_indices: tuple[int, int, int, int]
    tangent_basis: Array
    residue_denominator: complex
    jacobian_min_singular_value: float


@dataclass(frozen=True)
class GenericFibreAttempt:
    """One random-line plane-cubic fibre attempt before batch acceptance."""

    coordinates: tuple[tuple[Array, Array, Array], ...]
    status: str
    polynomial_degree: int
    root_count: int
    valid_root_count: int
    minimum_projective_root_separation: float | None
    maximum_relative_residual: float | None


def _normalized_random_coefficients(rng: np.random.Generator, count: int) -> Array:
    values = rng.normal(size=count) + 1j * rng.normal(size=count)
    return np.asarray(values / np.linalg.norm(values), dtype=np.complex128)


def make_generic_model(seed: int = 20260710) -> GenericGCICYModel:
    """Create deterministic generic coefficients for the target configuration."""

    rng = np.random.default_rng(seed)
    p1_exponents = ambient_monomial_exponents((1, 1, 3))
    p1_coefficients = _normalized_random_coefficients(rng, len(p1_exponents))
    p2_tensor = _normalized_random_coefficients(rng, 24).reshape(2, 2, 6)
    p2_exponents = ambient_monomial_exponents((1, 1, 1))
    p2_coefficients = np.empty(len(p2_exponents), dtype=np.complex128)
    for row, exponent in enumerate(p2_exponents):
        x_index = int(np.argmax(exponent[:2]))
        y_index = int(np.argmax(exponent[2:4]))
        z_index = int(np.argmax(exponent[4:]))
        p2_coefficients[row] = p2_tensor[x_index, y_index, z_index]
    return GenericGCICYModel(
        seed=seed,
        p1_exponents=p1_exponents,
        p1_coefficients=p1_coefficients,
        p2_exponents=p2_exponents,
        p2_coefficients=p2_coefficients,
        p2_tensor=p2_tensor,
    )


def make_exact_generic_model(seed: int = 20260711, coefficient_bound: int = 3) -> GenericGCICYModel:
    """Create a dense small-integer model suitable for exact CAS checks."""

    if coefficient_bound < 1:
        raise ValueError("coefficient_bound must be positive")
    rng = np.random.default_rng(seed)
    choices = np.concatenate(
        [
            np.arange(-coefficient_bound, 0, dtype=np.int64),
            np.arange(1, coefficient_bound + 1, dtype=np.int64),
        ]
    )
    p1_exponents = ambient_monomial_exponents((1, 1, 3))
    p1_coefficients = rng.choice(choices, size=len(p1_exponents)).astype(np.complex128)
    p2_tensor = rng.choice(choices, size=24).reshape(2, 2, 6).astype(np.complex128)
    p2_exponents = ambient_monomial_exponents((1, 1, 1))
    p2_coefficients = np.empty(len(p2_exponents), dtype=np.complex128)
    for row, exponent in enumerate(p2_exponents):
        x_index = int(np.argmax(exponent[:2]))
        y_index = int(np.argmax(exponent[2:4]))
        z_index = int(np.argmax(exponent[4:]))
        p2_coefficients[row] = p2_tensor[x_index, y_index, z_index]
    return GenericGCICYModel(
        seed=seed,
        p1_exponents=p1_exponents,
        p1_coefficients=p1_coefficients,
        p2_exponents=p2_exponents,
        p2_coefficients=p2_coefficients,
        p2_tensor=p2_tensor,
    )


def evaluate_p1(model: GenericGCICYModel, x: Array, y: Array, z: Array) -> complex:
    return complex(model.p1_coefficients @ evaluate_ambient_monomials(x, y, z, model.p1_exponents))


def evaluate_p2(model: GenericGCICYModel, x: Array, y: Array, z: Array) -> complex:
    return complex(np.einsum("ijk,i,j,k->", model.p2_tensor, x, y, z))


def p2_linear_rows(model: GenericGCICYModel, x: Array, y: Array, chart: Chart) -> tuple[Array, Array, Array]:
    """Return z-linear coefficient rows for p2 and the two local q equations."""

    p2_row = np.einsum("ijk,i,j->k", model.p2_tensor, x, y)
    if chart[0] == 0:
        q1_row = np.einsum("jk,j->k", model.p2_tensor[1], y)
    else:
        q1_row = -np.einsum("jk,j->k", model.p2_tensor[0], y)
    if chart[1] == 0:
        q2_row = np.einsum("ik,i->k", model.p2_tensor[:, 1, :], x)
    else:
        q2_row = -np.einsum("ik,i->k", model.p2_tensor[:, 0, :], x)
    return p2_row, q1_row, q2_row


def evaluate_q_sections(
    model: GenericGCICYModel,
    x: Array,
    y: Array,
    z: Array,
    chart: Chart,
) -> tuple[complex, complex]:
    """Evaluate the polynomial local representatives of q1 and q2."""

    x_local, y_local, z_local = normalize_homogeneous(x, y, z, chart)
    _, q1_row, q2_row = p2_linear_rows(model, x_local, y_local, chart)
    return complex(q1_row @ z_local), complex(q2_row @ z_local)


def _q_polynomial_data(model: GenericGCICYModel, chart: Chart, q_index: int) -> tuple[Array, Array]:
    exponents = []
    coefficients = []
    if q_index == 1:
        x_tensor_index = 1 if chart[0] == 0 else 0
        sign = 1.0 if chart[0] == 0 else -1.0
        for y_index, z_index in product(range(2), range(6)):
            exponent = [0] * 10
            exponent[2 + y_index] = 1
            exponent[4 + z_index] = 1
            exponents.append(exponent)
            coefficients.append(sign * model.p2_tensor[x_tensor_index, y_index, z_index])
    elif q_index == 2:
        y_tensor_index = 1 if chart[1] == 0 else 0
        sign = 1.0 if chart[1] == 0 else -1.0
        for x_index, z_index in product(range(2), range(6)):
            exponent = [0] * 10
            exponent[x_index] = 1
            exponent[4 + z_index] = 1
            exponents.append(exponent)
            coefficients.append(sign * model.p2_tensor[x_index, y_tensor_index, z_index])
    else:
        raise ValueError("q_index must be 1 or 2")
    return np.asarray(exponents, dtype=np.int64), np.asarray(coefficients, dtype=np.complex128)


def local_equations_and_jacobian(
    model: GenericGCICYModel,
    chart_coords: Array,
    chart: Chart,
) -> tuple[Array, Array]:
    """Evaluate four local equations and their analytic 4x7 Jacobian."""

    equations, jacobian, _ = _local_equations_jacobian_and_scales(model, chart_coords, chart)
    return equations, jacobian


def _local_equations_jacobian_and_scales(
    model: GenericGCICYModel,
    chart_coords: Array,
    chart: Chart,
) -> tuple[Array, Array, Array]:
    """Also return termwise absolute scales for backward-error checks."""

    rows = []
    derivatives = []
    scales = []
    polynomial_data = [
        (model.p1_exponents, model.p1_coefficients),
        (model.p2_exponents, model.p2_coefficients),
        _q_polynomial_data(model, chart, 1),
        _q_polynomial_data(model, chart, 2),
    ]
    for exponents, coefficients in polynomial_data:
        values, jacobian = affine_section_values_and_jacobian(chart_coords, chart, exponents)
        rows.append(coefficients @ values)
        derivatives.append(coefficients @ jacobian)
        scales.append(np.abs(coefficients) @ np.abs(values))
    return (
        np.asarray(rows, dtype=np.complex128),
        np.asarray(derivatives, dtype=np.complex128),
        np.asarray(scales, dtype=float),
    )


def choose_projective_chart(x: Array, y: Array, z: Array) -> Chart:
    """Choose the largest homogeneous coordinate in each projective factor."""

    return int(np.argmax(np.abs(x))), int(np.argmax(np.abs(y))), int(np.argmax(np.abs(z)))


def affine_coordinates_from_homogeneous(x: Array, y: Array, z: Array, chart: Chart) -> Array:
    x_local, y_local, z_local = normalize_homogeneous(x, y, z, chart)
    return np.concatenate(
        [
            np.delete(x_local, chart[0]),
            np.delete(y_local, chart[1]),
            np.delete(z_local, chart[2]),
        ]
    )


def best_implicit_coordinates(jacobian: Array) -> tuple[tuple[int, int, int], tuple[int, int, int, int], Array, complex]:
    """Choose the best-conditioned 4x4 dependent minor and tangent basis."""

    best = None
    all_indices = set(range(7))
    for dependent in combinations(range(7), 4):
        dependent_matrix = jacobian[:, dependent]
        singular_values = np.linalg.svd(dependent_matrix, compute_uv=False)
        if singular_values[-1] <= 1e-12:
            continue
        condition = float(singular_values[0] / singular_values[-1])
        if best is None or condition < best[0]:
            independent = tuple(sorted(all_indices - set(dependent)))
            best = (condition, dependent, independent, complex(np.linalg.det(dependent_matrix)))
    if best is None:
        raise FloatingPointError("No nonsingular dependent-coordinate minor was found.")
    _, dependent, independent, determinant = best
    tangent = np.zeros((7, 3), dtype=np.complex128)
    tangent[list(independent), :] = np.eye(3, dtype=np.complex128)
    tangent[list(dependent), :] = -np.linalg.solve(jacobian[:, dependent], jacobian[:, independent])
    return independent, dependent, tangent, determinant


def implicit_coordinates(
    jacobian: Array,
    independent: tuple[int, int, int],
) -> tuple[tuple[int, int, int, int], Array, complex]:
    """Build a tangent basis and residue denominator for fixed independent coordinates."""

    if len(set(independent)) != 3 or min(independent) < 0 or max(independent) >= 7:
        raise ValueError("independent must contain three distinct affine-coordinate indices")
    dependent = tuple(index for index in range(7) if index not in independent)
    dependent_matrix = jacobian[:, dependent]
    determinant = complex(np.linalg.det(dependent_matrix))
    if abs(determinant) < 1e-14:
        raise FloatingPointError("The dependent-coordinate Jacobian minor is singular.")
    tangent = np.zeros((7, 3), dtype=np.complex128)
    tangent[list(independent), :] = np.eye(3, dtype=np.complex128)
    tangent[list(dependent), :] = -np.linalg.solve(dependent_matrix, jacobian[:, independent])
    return dependent, tangent, determinant


def generic_point_in_chart(
    model: GenericGCICYModel,
    x: Array,
    y: Array,
    z: Array,
    chart: Chart,
    independent: tuple[int, int, int] | None = None,
) -> GenericGCICYPoint:
    """Construct implicit local-coordinate data for a homogeneous point in a chart."""

    affine = affine_coordinates_from_homogeneous(x, y, z, chart)
    equations, jacobian, equation_scales = _local_equations_jacobian_and_scales(model, affine, chart)
    relative_residual = np.max(np.abs(equations) / np.maximum(1.0, equation_scales))
    if not np.isfinite(relative_residual) or relative_residual > 1e-10:
        raise FloatingPointError(
            "The homogeneous point does not satisfy the local gCICY equations "
            f"(backward relative residual {relative_residual:.3e})."
        )
    if independent is None:
        independent, dependent, tangent, determinant = best_implicit_coordinates(jacobian)
    else:
        dependent, tangent, determinant = implicit_coordinates(jacobian, independent)
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    return GenericGCICYPoint(
        x=np.asarray(x, dtype=np.complex128),
        y=np.asarray(y, dtype=np.complex128),
        z=np.asarray(z, dtype=np.complex128),
        projective_chart=chart,
        affine_coordinates=affine,
        independent_indices=independent,
        dependent_indices=dependent,
        tangent_basis=tangent,
        residue_denominator=determinant,
        jacobian_min_singular_value=float(singular_values[-1]),
    )


def generic_baseline_metric(point: GenericGCICYPoint, weights: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Array:
    """Pull back the ambient product Fubini-Study metric to implicit coordinates."""

    ambient_metric = ambient_chart_metric(point.affine_coordinates, weights=weights)
    metric = point.tangent_basis.conjugate().T @ ambient_metric @ point.tangent_basis
    return 0.5 * (metric + metric.conjugate().T)


def generic_factor_metrics(point: GenericGCICYPoint) -> tuple[Array, Array, Array]:
    """Pull back the three ambient factor Fubini-Study forms separately."""

    output = []
    for factor in range(3):
        weights = tuple(1.0 if index == factor else 0.0 for index in range(3))
        ambient_metric = ambient_chart_metric(point.affine_coordinates, weights=weights)
        metric = point.tangent_basis.conjugate().T @ ambient_metric @ point.tangent_basis
        output.append(0.5 * (metric + metric.conjugate().T))
    return output[0], output[1], output[2]


def generic_holomorphic_volume_log_density(point: GenericGCICYPoint) -> float:
    """Return the local Poincare-residue log density in implicit coordinates."""

    if abs(point.residue_denominator) < 1e-14:
        raise FloatingPointError("The residue denominator is too small.")
    return float(-2.0 * np.log(abs(point.residue_denominator)))


def generic_proposal_log_density(point: GenericGCICYPoint) -> float:
    """Return the local density of the sampler measure omega_x omega_y omega_z.

    The sampler draws Fubini-Study points on both P1 factors and intersects a
    random line with the plane cubic fibre.  Its proposal form is therefore
    ``omega_x wedge omega_y wedge omega_z|_X`` up to a global constant.  The
    expression below evaluates that mixed volume form in arbitrary implicit
    coordinates.
    """

    ambient_metric = ambient_chart_metric(point.affine_coordinates)
    tangent = point.tangent_basis
    x_covector = tangent[0, :]
    y_covector = tangent[1, :]
    vertical_vector = np.cross(x_covector, y_covector)
    z_tangent = tangent[2:, :]
    z_metric = z_tangent.conjugate().T @ ambient_metric[2:, 2:] @ z_tangent
    density = (
        float(np.real(ambient_metric[0, 0]))
        * float(np.real(ambient_metric[1, 1]))
        * float(np.real(np.vdot(vertical_vector, z_metric @ vertical_vector)))
    )
    if not np.isfinite(density) or density <= 0:
        raise FloatingPointError("The sampler proposal density is not positive.")
    return float(np.log(density))


def generic_importance_log_weight(point: GenericGCICYPoint) -> float:
    """Return log[(Omega Omega-bar)/(omega_x omega_y omega_z)] up to a constant."""

    return generic_holomorphic_volume_log_density(point) - generic_proposal_log_density(point)


def generic_importance_weights(points: list[GenericGCICYPoint]) -> Array:
    """Return mean-one importance weights for integration against Omega Omega-bar."""

    log_weights = np.asarray([generic_importance_log_weight(point) for point in points], dtype=float)
    log_weights -= float(np.max(log_weights))
    weights = np.exp(log_weights)
    mean = float(np.mean(weights))
    if not np.isfinite(mean) or mean <= 0:
        raise FloatingPointError("Importance weights could not be normalized.")
    return weights / mean


def generic_effective_sample_size(weights: Array) -> float:
    """Return the standard importance-sampling effective sample size."""

    values = np.asarray(weights, dtype=float).reshape(-1)
    if values.size == 0 or np.any(values < 0) or not np.all(np.isfinite(values)):
        raise ValueError("weights must be a non-empty finite non-negative array")
    denominator = float(np.sum(values**2))
    if denominator <= 0:
        raise ValueError("weights must have positive sum")
    return float(np.sum(values) ** 2 / denominator)


def _positive_hermitian_logdet(metric: Array) -> float:
    """Compute log det for a positive Hermitian metric without ill-conditioned eigenvalues."""

    values = np.asarray(metric, dtype=np.complex128)
    hermitian = 0.5 * (values + values.conj().T)
    try:
        factor = np.linalg.cholesky(hermitian)
    except np.linalg.LinAlgError as exc:
        raise FloatingPointError("Metric is not positive definite.") from exc
    diagonal = np.real(np.diag(factor))
    if np.any(diagonal <= 0) or not np.all(np.isfinite(diagonal)):
        raise FloatingPointError("Metric is not positive definite.")
    return float(2.0 * np.sum(np.log(diagonal)))


def generic_metric_volume_ratio(point: GenericGCICYPoint, metric: Array) -> float:
    """Return the metric volume density divided by the sampler proposal density."""

    eigenvalues = np.linalg.eigvalsh(np.asarray(metric, dtype=np.complex128))
    if eigenvalues[0] <= 0:
        raise FloatingPointError("Metric is not positive definite.")
    return float(
        np.exp(_positive_hermitian_logdet(metric) - generic_proposal_log_density(point))
    )


def generic_line_bundle_slope_ratio(
    point: GenericGCICYPoint,
    metric: Array,
    line_bundle: tuple[float, float, float],
) -> float:
    """Return the local proposal-normalized density for c1(L) J^2/2!."""

    h_metric = np.asarray(metric, dtype=np.complex128)
    volume_ratio = generic_metric_volume_ratio(point, h_metric)
    factor_metrics = generic_factor_metrics(point)
    curvature = sum(weight * factor for weight, factor in zip(line_bundle, factor_metrics, strict=True))
    contraction = float(np.real(np.trace(np.linalg.solve(h_metric, curvature))))
    return volume_ratio * contraction


def generic_monge_ampere_log_error(point: GenericGCICYPoint, metric: Array) -> float:
    """Return log det(g)-log|Omega|^2 in the point's implicit coordinates."""

    eigenvalues = np.linalg.eigvalsh(np.asarray(metric, dtype=np.complex128))
    if eigenvalues[0] <= 0:
        raise FloatingPointError("Metric is not positive definite.")
    return float(
        _positive_hermitian_logdet(metric)
        - generic_holomorphic_volume_log_density(point)
    )


def generic_section_values_and_jacobian(point: GenericGCICYPoint, exponents: Array) -> tuple[Array, Array]:
    """Evaluate global sections and derivatives in a point's implicit coordinates."""

    values, ambient_jacobian = affine_section_values_and_jacobian(
        point.affine_coordinates,
        point.projective_chart,
        exponents,
    )
    return values, ambient_jacobian @ point.tangent_basis


def generic_restricted_ambient_basis(
    points: list[GenericGCICYPoint],
    degree: tuple[int, int, int],
    *,
    relative_rank_threshold: float = 1e-10,
) -> RestrictedSectionBasis:
    """Select a stable restricted ambient section basis on generic-model points."""

    exponents = ambient_monomial_exponents(degree)
    matrix = np.asarray(
        [
            evaluate_sections_in_chart(point.x, point.y, point.z, point.projective_chart, exponents)
            for point in points
        ],
        dtype=np.complex128,
    )
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-14)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
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


def generic_restricted_fubini_study_h_matrix(
    points: list[GenericGCICYPoint],
    basis: RestrictedSectionBasis,
) -> tuple[Array, float]:
    """Represent the ambient product Fubini-Study potential in the generic basis."""

    coefficients, relation_error = generic_restriction_coefficients(points, basis)
    weights = multinomial_section_weights(basis.ambient_exponents)
    h_matrix = np.conjugate(coefficients) @ (weights[:, None] * coefficients.T)
    h_matrix = 0.5 * (h_matrix + h_matrix.conjugate().T)
    return h_matrix, relation_error


def generic_restriction_coefficients(
    points: list[GenericGCICYPoint],
    basis: RestrictedSectionBasis,
) -> tuple[Array, float]:
    """Express every ambient monomial as a linear combination of the selected basis."""

    all_values = np.asarray(
        [
            evaluate_sections_in_chart(
                point.x,
                point.y,
                point.z,
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
    relation_error = float(
        np.linalg.norm(selected_values @ coefficients - all_values) / max(1.0, np.linalg.norm(all_values))
    )
    return coefficients, relation_error


def generic_lift_h_matrix_power(
    points: list[GenericGCICYPoint],
    source_exponents: Array,
    source_h_matrix: Array,
    target_basis: RestrictedSectionBasis,
) -> tuple[Array, float]:
    """Represent a positive section norm raised to an integer power.

    If ``F=s^dagger H s`` uses degree ``d`` monomial sections, this returns
    the target-basis matrix for ``F**power``.  With the target metric
    normalization divided by ``power``, the lifted metric is identical to the
    source metric and is therefore a reliable warm start for k convergence.
    """

    exponents = np.asarray(source_exponents, dtype=np.int64)
    h_matrix = np.asarray(source_h_matrix, dtype=np.complex128)
    if exponents.ndim != 2 or exponents.shape[1] != 10:
        raise ValueError("source_exponents must have shape (n,10)")
    if h_matrix.shape != (len(exponents), len(exponents)):
        raise ValueError("source H-matrix dimension does not match its sections")
    source_degree = (
        int(np.sum(exponents[0, :2])),
        int(np.sum(exponents[0, 2:4])),
        int(np.sum(exponents[0, 4:])),
    )
    if any(
        not np.all(np.sum(exponents[:, block], axis=1) == degree)
        for block, degree in zip((slice(0, 2), slice(2, 4), slice(4, 10)), source_degree, strict=True)
    ):
        raise ValueError("source sections do not share one multi-degree")
    ratios = []
    for target, source in zip(target_basis.degree, source_degree, strict=True):
        if source <= 0 or target % source != 0:
            raise ValueError("target degree must be an integer multiple of the source degree")
        ratios.append(target // source)
    if len(set(ratios)) != 1 or ratios[0] < 1:
        raise ValueError("target degree must use one common positive power")
    power = ratios[0]

    current_exponents = exponents.copy()
    current_h = 0.5 * (h_matrix + h_matrix.conjugate().T)
    for factor in range(2, power + 1):
        degree = tuple(factor * value for value in source_degree)
        ambient_exponents = ambient_monomial_exponents(degree)
        exponent_index = {tuple(row): index for index, row in enumerate(ambient_exponents)}
        multiplication = np.empty((len(current_exponents), len(exponents)), dtype=np.int64)
        for left_index, left in enumerate(current_exponents):
            for right_index, right in enumerate(exponents):
                multiplication[left_index, right_index] = exponent_index[tuple(left + right)]

        lifted = np.zeros((len(ambient_exponents), len(ambient_exponents)), dtype=np.complex128)
        for left_source in range(len(exponents)):
            left_indices = multiplication[:, left_source]
            for right_source in range(len(exponents)):
                right_indices = multiplication[:, right_source]
                lifted[np.ix_(left_indices, right_indices)] += current_h * h_matrix[left_source, right_source]
        current_exponents = ambient_exponents
        current_h = lifted

    target_ambient = np.asarray(target_basis.ambient_exponents, dtype=np.int64)
    if current_exponents.shape != target_ambient.shape or not np.array_equal(current_exponents, target_ambient):
        source_index = {tuple(row): index for index, row in enumerate(current_exponents)}
        order = np.asarray([source_index[tuple(row)] for row in target_ambient], dtype=np.int64)
        current_h = current_h[np.ix_(order, order)]

    coefficients, relation_error = generic_restriction_coefficients(points, target_basis)
    target_h = np.conjugate(coefficients) @ current_h @ coefficients.T
    target_h = 0.5 * (target_h + target_h.conjugate().T)
    eigenvalues = np.linalg.eigvalsh(target_h)
    if eigenvalues[0] <= 0:
        tolerance = 1e-11 * max(1.0, float(eigenvalues[-1]))
        if eigenvalues[0] < -tolerance:
            raise FloatingPointError("lifted H-matrix is not positive definite")
        target_h += (tolerance - eigenvalues[0]) * np.eye(len(target_h))
    return target_h, relation_error


def generic_global_h_metric(
    point: GenericGCICYPoint,
    exponents: Array,
    h_matrix: Array,
    normalization: float = 1.0,
) -> Array:
    values, jacobian = generic_section_values_and_jacobian(point, exponents)
    return h_matrix_metric(values, jacobian, h_matrix, normalization=normalization)


def generic_global_h_metrics(
    points: list[GenericGCICYPoint],
    exponents: Array,
    h_matrix: Array,
    normalization: float = 1.0,
) -> Array:
    return np.asarray(
        [generic_global_h_metric(point, exponents, h_matrix, normalization=normalization) for point in points],
        dtype=np.complex128,
    )


def generic_baseline_metrics(points: list[GenericGCICYPoint]) -> Array:
    return np.asarray([generic_baseline_metric(point) for point in points], dtype=np.complex128)


def generic_residual_values(points: list[GenericGCICYPoint], metrics: Array) -> Array:
    values = np.empty(len(points), dtype=float)
    for index, (point, metric) in enumerate(zip(points, metrics, strict=True)):
        eigenvalues = np.linalg.eigvalsh(metric)
        if eigenvalues[0] <= 0:
            values[index] = np.nan
        else:
            values[index] = float(
                _positive_hermitian_logdet(metric)
                - generic_holomorphic_volume_log_density(point)
            )
    return values


def generic_residual_stats(points: list[GenericGCICYPoint], metrics: Array) -> ResidualStats:
    eigenvalues = np.linalg.eigvalsh(np.asarray(metrics, dtype=np.complex128))
    minimum = float(np.min(eigenvalues))
    values = generic_residual_values(points, metrics)
    if not np.all(np.isfinite(values)):
        return ResidualStats(float("nan"), float("nan"), minimum, float("nan"))
    centered = values - float(np.mean(values))
    return ResidualStats(
        rms=float(np.sqrt(np.mean(centered**2))),
        max_abs=float(np.max(np.abs(centered))),
        min_eigenvalue=minimum,
        mean_log_error=float(np.mean(values)),
    )


def generic_weighted_residual_stats(
    points: list[GenericGCICYPoint],
    metrics: Array,
    weights: Array | None = None,
) -> ResidualStats:
    """Return residual statistics integrated with the holomorphic volume form."""

    eigenvalues = np.linalg.eigvalsh(np.asarray(metrics, dtype=np.complex128))
    minimum = float(np.min(eigenvalues))
    values = generic_residual_values(points, metrics)
    if not np.all(np.isfinite(values)):
        return ResidualStats(float("nan"), float("nan"), minimum, float("nan"))
    importance = generic_importance_weights(points) if weights is None else np.asarray(weights, dtype=float)
    if importance.shape != values.shape or np.any(importance < 0) or not np.all(np.isfinite(importance)):
        raise ValueError("weights must be finite, non-negative, and match the number of points")
    total = float(np.sum(importance))
    if total <= 0:
        raise ValueError("weights must have positive sum")
    mean = float(np.sum(importance * values) / total)
    centered = values - mean
    return ResidualStats(
        rms=float(np.sqrt(np.sum(importance * centered**2) / total)),
        max_abs=float(np.max(np.abs(centered))),
        min_eigenvalue=minimum,
        mean_log_error=mean,
    )


def _restricted_cubic_coefficients(
    model: GenericGCICYModel,
    x: Array,
    y: Array,
    z_base: Array,
    z_direction: Array,
) -> Array:
    coefficients = np.zeros(4, dtype=np.complex128)
    for exponent, coefficient in zip(model.p1_exponents, model.p1_coefficients, strict=True):
        scalar = coefficient * np.prod(x ** exponent[:2]) * np.prod(y ** exponent[2:4])
        polynomial = np.asarray([1.0 + 0.0j])
        for z_index, power in enumerate(exponent[4:]):
            for _ in range(int(power)):
                polynomial = np.convolve(polynomial, [z_base[z_index], z_direction[z_index]])
        coefficients[: len(polynomial)] += scalar * polynomial
    return coefficients


def _sample_one_fibre(
    model: GenericGCICYModel,
    rng: np.random.Generator,
    *,
    residual_tolerance: float,
    root_separation_tolerance: float,
) -> GenericFibreAttempt:
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
    ) -> GenericFibreAttempt:
        return GenericFibreAttempt(
            coordinates=(),
            status=status,
            polynomial_degree=polynomial_degree,
            root_count=root_count,
            valid_root_count=valid_root_count,
            minimum_projective_root_separation=minimum_separation,
            maximum_relative_residual=maximum_residual,
        )

    x = rng.normal(size=2) + 1j * rng.normal(size=2)
    y = rng.normal(size=2) + 1j * rng.normal(size=2)
    x /= np.linalg.norm(x)
    y /= np.linalg.norm(y)
    chart = (int(np.argmax(np.abs(x))), int(np.argmax(np.abs(y))), 0)
    linear_rows = np.asarray(p2_linear_rows(model, x, y, chart))
    fibre_basis = null_space(linear_rows)
    if fibre_basis.shape != (6, 3):
        return rejected("invalid_plane_fibre_basis")

    random_line = rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2))
    line_basis, line_scale = np.linalg.qr(random_line, mode="reduced")
    if (
        line_basis.shape != (3, 2)
        or line_scale.shape != (2, 2)
        or float(np.min(np.abs(np.diag(line_scale)))) < 1e-12
    ):
        return rejected("degenerate_random_line")
    z_base = fibre_basis @ line_basis[:, 0]
    z_direction = fibre_basis @ line_basis[:, 1]
    cubic = _restricted_cubic_coefficients(model, x, y, z_base, z_direction)
    while len(cubic) > 1 and abs(cubic[-1]) < 1e-12 * max(1.0, np.max(np.abs(cubic))):
        cubic = cubic[:-1]
    polynomial_degree = int(len(cubic) - 1)
    if polynomial_degree != 3:
        return rejected(
            "noncubic_restriction",
            polynomial_degree=polynomial_degree,
            root_count=max(0, polynomial_degree),
        )
    roots = np.polynomial.polynomial.polyroots(cubic)
    if len(roots) != 3 or not np.all(np.isfinite(roots)):
        return rejected(
            "invalid_cubic_roots",
            polynomial_degree=polynomial_degree,
            root_count=int(len(roots)),
        )
    output: list[tuple[Array, Array, Array]] = []
    z_roots: list[Array] = []
    residuals: list[float] = []
    for root in roots:
        z = z_base + root * z_direction
        if np.linalg.norm(z) < 1e-12:
            return rejected(
                "zero_projective_root",
                polynomial_degree=polynomial_degree,
                root_count=int(len(roots)),
                valid_root_count=len(output),
            )
        z /= np.linalg.norm(z)
        projective_chart = choose_projective_chart(x, y, z)
        affine = affine_coordinates_from_homogeneous(x, y, z, projective_chart)
        equations, _, scales = _local_equations_jacobian_and_scales(
            model,
            affine,
            projective_chart,
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
        z_roots.append(z.copy())
        output.append((x.copy(), y.copy(), z))
    separations = []
    for left, right in combinations(z_roots, 2):
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
    return GenericFibreAttempt(
        coordinates=tuple(output),
        status="accepted",
        polynomial_degree=polynomial_degree,
        root_count=int(len(roots)),
        valid_root_count=len(output),
        minimum_projective_root_separation=minimum_separation,
        maximum_relative_residual=float(max(residuals)),
    )


def _sample_generic_gcicy_points(
    model: GenericGCICYModel,
    n_points: int,
    *,
    seed: int,
    residual_tolerance: float,
    root_separation_tolerance: float,
    max_attempt_factor: int,
) -> tuple[list[GenericGCICYPoint], dict[str, object]]:
    if n_points <= 0:
        raise ValueError("n_points must be positive")
    rng = np.random.default_rng(seed)
    accepted_fibres: list[list[GenericGCICYPoint]] = []
    attempts = 0
    rejection_reasons: dict[str, int] = {}
    root_count_histogram: dict[str, int] = {}
    valid_root_count_histogram: dict[str, int] = {}
    minimum_root_separation = float("inf")
    maximum_relative_residual = 0.0
    required_fibres = (n_points + 2) // 3
    maximum_attempts = max_attempt_factor * max(1, required_fibres)
    while len(accepted_fibres) < required_fibres and attempts < maximum_attempts:
        attempt = _sample_one_fibre(
            model,
            rng,
            residual_tolerance=residual_tolerance,
            root_separation_tolerance=root_separation_tolerance,
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
            for x, y, z in attempt.coordinates:
                chart = choose_projective_chart(x, y, z)
                converted.append(generic_point_in_chart(model, x, y, z, chart))
        except (FloatingPointError, np.linalg.LinAlgError):
            rejection_reasons["implicit_coordinate_failure"] = (
                rejection_reasons.get("implicit_coordinate_failure", 0) + 1
            )
            continue
        if len(converted) != 3:
            rejection_reasons["incomplete_converted_fibre"] = (
                rejection_reasons.get("incomplete_converted_fibre", 0) + 1
            )
            continue
        permutation = rng.permutation(3)
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
    output = flattened[:n_points]
    truncated_final_fibre = bool(n_points % 3)
    diagnostics: dict[str, object] = {
        "sampler": "p1p1p5_random_line_plane_cubic_all_three_roots",
        "requested_points": int(n_points),
        "returned_points": int(len(output)),
        "expected_points_per_cluster": 3,
        "attempted_clusters": int(attempts),
        "accepted_clusters": int(len(accepted_fibres)),
        "rejected_clusters": int(attempts - len(accepted_fibres)),
        "cluster_acceptance_rate": float(len(accepted_fibres) / attempts),
        "returned_complete_clusters": int(n_points // 3),
        "truncated_final_cluster": truncated_final_fibre,
        "all_returned_clusters_complete": not truncated_final_fibre,
        "root_count_histogram": root_count_histogram,
        "valid_root_count_histogram": valid_root_count_histogram,
        "accepted_root_count_histogram": {"3": int(len(accepted_fibres))},
        "rejection_reasons": rejection_reasons,
        "minimum_projective_root_separation": (
            float(minimum_root_separation)
            if np.isfinite(minimum_root_separation)
            else None
        ),
        "maximum_accepted_relative_residual": float(maximum_relative_residual),
        "residual_tolerance": float(residual_tolerance),
        "root_separation_tolerance": float(root_separation_tolerance),
        "importance_weight_used_for_acceptance": False,
        "implicit_jacobian_invertibility_required_for_acceptance": True,
    }
    return output, diagnostics


def sample_generic_gcicy_points(
    model: GenericGCICYModel,
    n_points: int,
    seed: int = 0,
    *,
    residual_tolerance: float = 1e-8,
    root_separation_tolerance: float = 1e-7,
    max_attempt_factor: int = 20,
) -> list[GenericGCICYPoint]:
    """Sample points by random-line intersections with the plane-cubic fibres."""

    points, _ = _sample_generic_gcicy_points(
        model,
        n_points,
        seed=seed,
        residual_tolerance=residual_tolerance,
        root_separation_tolerance=root_separation_tolerance,
        max_attempt_factor=max_attempt_factor,
    )
    return points


def sample_generic_gcicy_points_with_diagnostics(
    model: GenericGCICYModel,
    n_points: int,
    seed: int = 0,
    *,
    residual_tolerance: float = 1e-8,
    root_separation_tolerance: float = 1e-7,
    max_attempt_factor: int = 20,
) -> tuple[list[GenericGCICYPoint], dict[str, object]]:
    """Sample complete plane-cubic fibres and return JSON-safe diagnostics."""

    return _sample_generic_gcicy_points(
        model,
        n_points,
        seed=seed,
        residual_tolerance=residual_tolerance,
        root_separation_tolerance=root_separation_tolerance,
        max_attempt_factor=max_attempt_factor,
    )


def sample_p2_hypersurface(
    model: GenericGCICYModel,
    n_points: int,
    seed: int = 0,
) -> list[tuple[Array, Array, Array]]:
    """Sample p2=0 points for rational-representative transition checks."""

    rng = np.random.default_rng(seed)
    output = []
    while len(output) < n_points:
        x = rng.normal(size=2) + 1j * rng.normal(size=2)
        y = rng.normal(size=2) + 1j * rng.normal(size=2)
        z = rng.normal(size=6) + 1j * rng.normal(size=6)
        x /= np.linalg.norm(x)
        y /= np.linalg.norm(y)
        row = np.einsum("ijk,i,j->k", model.p2_tensor, x, y)
        solve_index = int(np.argmax(np.abs(row)))
        z[solve_index] = -(
            row @ z - row[solve_index] * z[solve_index]
        ) / row[solve_index]
        z /= np.linalg.norm(z)
        output.append((x, y, z))
    return output
