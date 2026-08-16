"""Exact ordinary bicubic control model in P2 x P2."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations
from math import factorial

import numpy as np
from scipy.linalg import qr

from .ricci_flat_fit import ResidualStats
from .simple_patch import Array, fubini_study_metric
from .global_sections import h_matrix_metric


BicubicChart = tuple[int, int]


@dataclass(frozen=True)
class BicubicModel:
    seed: int
    exponents: Array
    coefficients: Array


@dataclass(frozen=True)
class BicubicPoint:
    x: Array
    y: Array
    projective_chart: BicubicChart
    affine_coordinates: Array
    independent_indices: tuple[int, int, int]
    dependent_index: int
    tangent_basis: Array
    residue_denominator: complex
    jacobian_norm: float


@dataclass(frozen=True)
class BicubicSectionBasis:
    degree: int
    ambient_exponents: Array
    selected_indices: Array
    selected_exponents: Array
    numerical_rank: int
    singular_values: Array
    relative_rank_threshold: float


@dataclass(frozen=True)
class BicubicFibreAttempt:
    coordinates: tuple[tuple[Array, Array], ...]
    status: str
    polynomial_degree: int
    root_count: int
    valid_root_count: int
    minimum_projective_root_separation: float | None
    maximum_relative_residual: float | None


def _compositions(total: int, length: int) -> list[tuple[int, ...]]:
    if length == 1:
        return [(total,)]
    output = []
    for first in range(total + 1):
        for rest in _compositions(total - first, length - 1):
            output.append((first, *rest))
    return output


def bicubic_monomial_exponents(degree: int) -> Array:
    """Return all monomial exponents of O(degree, degree) on P2 x P2."""

    if degree < 0:
        raise ValueError("degree must be non-negative")
    factor = _compositions(degree, 3)
    return np.asarray([(*left, *right) for left in factor for right in factor], dtype=np.int64)


def make_exact_bicubic_model(seed: int = 20260721, coefficient_bound: int = 3) -> BicubicModel:
    """Create a deterministic dense small-integer bicubic hypersurface."""

    if coefficient_bound < 1:
        raise ValueError("coefficient_bound must be positive")
    rng = np.random.default_rng(seed)
    choices = np.concatenate(
        [
            np.arange(-coefficient_bound, 0, dtype=np.int64),
            np.arange(1, coefficient_bound + 1, dtype=np.int64),
        ]
    )
    exponents = bicubic_monomial_exponents(3)
    coefficients = rng.choice(choices, size=len(exponents)).astype(np.complex128)
    return BicubicModel(seed=seed, exponents=exponents, coefficients=coefficients)


def evaluate_bicubic(model: BicubicModel, x: Array, y: Array) -> complex:
    values = np.concatenate([np.asarray(x, dtype=np.complex128), np.asarray(y, dtype=np.complex128)])
    return complex(model.coefficients @ np.prod(values[None, :] ** model.exponents, axis=1))


def normalize_bicubic_homogeneous(x: Array, y: Array, chart: BicubicChart) -> tuple[Array, Array]:
    x_values = np.asarray(x, dtype=np.complex128) / x[chart[0]]
    y_values = np.asarray(y, dtype=np.complex128) / y[chart[1]]
    return x_values, y_values


def bicubic_affine_coordinates(x: Array, y: Array, chart: BicubicChart) -> Array:
    x_local, y_local = normalize_bicubic_homogeneous(x, y, chart)
    return np.concatenate([np.delete(x_local, chart[0]), np.delete(y_local, chart[1])])


def bicubic_homogeneous_from_affine(coords: Array, chart: BicubicChart) -> tuple[Array, Array]:
    values = np.asarray(coords, dtype=np.complex128).reshape(-1)
    if values.shape != (4,):
        raise ValueError("bicubic affine coordinates must have shape (4,)")
    x = np.empty(3, dtype=np.complex128)
    y = np.empty(3, dtype=np.complex128)
    x[chart[0]] = 1.0
    y[chart[1]] = 1.0
    x[np.arange(3) != chart[0]] = values[:2]
    y[np.arange(3) != chart[1]] = values[2:]
    return x, y


def bicubic_section_values_and_ambient_jacobian(
    coords: Array,
    chart: BicubicChart,
    exponents: Array,
) -> tuple[Array, Array]:
    """Evaluate O(k,k) monomials and their four affine derivatives."""

    x, y = bicubic_homogeneous_from_affine(coords, chart)
    homogeneous = np.concatenate([x, y])
    powers = np.asarray(exponents, dtype=np.int64)
    active = [index for index in range(6) if index not in (chart[0], 3 + chart[1])]
    values = np.prod(homogeneous[None, :] ** powers, axis=1)
    jacobian = np.empty((len(powers), 4), dtype=np.complex128)
    for column, homogeneous_index in enumerate(active):
        exponent = powers[:, homogeneous_index]
        reduced = powers.copy()
        reduced[:, homogeneous_index] = np.maximum(reduced[:, homogeneous_index] - 1, 0)
        jacobian[:, column] = exponent * np.prod(homogeneous[None, :] ** reduced, axis=1)
    return values, jacobian


def bicubic_local_equation_and_jacobian(
    model: BicubicModel,
    coords: Array,
    chart: BicubicChart,
) -> tuple[complex, Array]:
    values, jacobian = bicubic_section_values_and_ambient_jacobian(coords, chart, model.exponents)
    return complex(model.coefficients @ values), np.asarray(model.coefficients @ jacobian)


def bicubic_point_in_chart(
    model: BicubicModel,
    x: Array,
    y: Array,
    chart: BicubicChart,
    dependent_index: int | None = None,
) -> BicubicPoint:
    coords = bicubic_affine_coordinates(x, y, chart)
    equation, jacobian = bicubic_local_equation_and_jacobian(model, coords, chart)
    if abs(equation) > 1e-8:
        raise FloatingPointError("point does not satisfy the bicubic equation")
    dependent = int(np.argmax(np.abs(jacobian))) if dependent_index is None else dependent_index
    if not 0 <= dependent < 4 or abs(jacobian[dependent]) < 1e-12:
        raise FloatingPointError("bicubic dependent-coordinate derivative is singular")
    independent = tuple(index for index in range(4) if index != dependent)
    tangent = np.zeros((4, 3), dtype=np.complex128)
    tangent[list(independent), :] = np.eye(3, dtype=np.complex128)
    tangent[dependent, :] = -jacobian[list(independent)] / jacobian[dependent]
    return BicubicPoint(
        x=np.asarray(x, dtype=np.complex128),
        y=np.asarray(y, dtype=np.complex128),
        projective_chart=chart,
        affine_coordinates=coords,
        independent_indices=independent,
        dependent_index=dependent,
        tangent_basis=tangent,
        residue_denominator=complex(jacobian[dependent]),
        jacobian_norm=float(np.linalg.norm(jacobian)),
    )


def _restricted_cubic_coefficients(
    model: BicubicModel,
    x: Array,
    y_base: Array,
    y_direction: Array,
) -> Array:
    coefficients = np.zeros(4, dtype=np.complex128)
    for exponent, coefficient in zip(model.exponents, model.coefficients, strict=True):
        scalar = coefficient * np.prod(x ** exponent[:3])
        polynomial = np.asarray([1.0 + 0.0j])
        for index, power in enumerate(exponent[3:]):
            for _ in range(int(power)):
                polynomial = np.convolve(polynomial, [y_base[index], y_direction[index]])
        coefficients[: len(polynomial)] += scalar * polynomial
    return coefficients


def _sample_one_bicubic_fibre(
    model: BicubicModel,
    rng: np.random.Generator,
    *,
    residual_tolerance: float,
) -> BicubicFibreAttempt:
    def rejected(
        status: str,
        *,
        polynomial_degree: int = -1,
        root_count: int = 0,
        valid_root_count: int = 0,
        minimum_separation: float | None = None,
        maximum_residual: float | None = None,
    ) -> BicubicFibreAttempt:
        return BicubicFibreAttempt(
            coordinates=(),
            status=status,
            polynomial_degree=polynomial_degree,
            root_count=root_count,
            valid_root_count=valid_root_count,
            minimum_projective_root_separation=minimum_separation,
            maximum_relative_residual=maximum_residual,
        )

    x = rng.normal(size=3) + 1j * rng.normal(size=3)
    x /= np.linalg.norm(x)
    frame = rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2))
    line, scale = np.linalg.qr(frame, mode="reduced")
    if (
        line.shape != (3, 2)
        or scale.shape != (2, 2)
        or float(np.min(np.abs(np.diag(scale)))) < 1e-12
    ):
        return rejected("degenerate_random_line")
    y_base = line[:, 0]
    y_direction = line[:, 1]
    cubic = _restricted_cubic_coefficients(model, x, y_base, y_direction)
    coefficient_scale = max(1.0, float(np.max(np.abs(cubic))))
    while len(cubic) > 1 and abs(cubic[-1]) < 1e-12 * coefficient_scale:
        cubic = cubic[:-1]
    degree = int(len(cubic) - 1)
    if degree != 3:
        return rejected(
            "noncubic_restriction",
            polynomial_degree=degree,
            root_count=max(0, degree),
        )
    roots = np.polynomial.polynomial.polyroots(cubic)
    if len(roots) != 3 or not np.all(np.isfinite(roots)):
        return rejected(
            "invalid_cubic_roots",
            polynomial_degree=degree,
            root_count=int(len(roots)),
        )
    coordinates: list[tuple[Array, Array]] = []
    y_roots: list[Array] = []
    residuals: list[float] = []
    for root in roots:
        y = y_base + root * y_direction
        if np.linalg.norm(y) < 1e-12:
            return rejected(
                "zero_projective_root",
                polynomial_degree=degree,
                root_count=int(len(roots)),
                valid_root_count=len(coordinates),
            )
        y /= np.linalg.norm(y)
        homogeneous = np.concatenate([x, y])
        monomials = np.prod(
            homogeneous[None, :] ** model.exponents,
            axis=1,
        )
        residual_scale = max(
            1.0,
            float(np.sum(np.abs(model.coefficients * monomials))),
        )
        residual = float(abs(model.coefficients @ monomials) / residual_scale)
        residuals.append(residual)
        if not np.isfinite(residual) or residual > residual_tolerance:
            return rejected(
                "root_residual_failure",
                polynomial_degree=degree,
                root_count=int(len(roots)),
                valid_root_count=len(coordinates),
                maximum_residual=float(np.nanmax(residuals)),
            )
        y_roots.append(y.copy())
        coordinates.append((x.copy(), y))
    separations = []
    for left, right in combinations(y_roots, 2):
        overlap = min(1.0, float(abs(np.vdot(left, right))))
        separations.append(float(np.sqrt(max(0.0, 1.0 - overlap**2))))
    return BicubicFibreAttempt(
        coordinates=tuple(coordinates),
        status="accepted",
        polynomial_degree=degree,
        root_count=int(len(roots)),
        valid_root_count=len(coordinates),
        minimum_projective_root_separation=float(min(separations)),
        maximum_relative_residual=float(max(residuals)),
    )


def _sample_bicubic_points(
    model: BicubicModel,
    n_points: int,
    *,
    seed: int,
    residual_tolerance: float,
    max_attempt_factor: int,
) -> tuple[list[BicubicPoint], dict[str, object]]:
    if n_points <= 0:
        raise ValueError("n_points must be positive")
    rng = np.random.default_rng(seed)
    required_fibres = (n_points + 2) // 3
    maximum_attempts = max_attempt_factor * max(1, required_fibres)
    accepted_fibres: list[list[BicubicPoint]] = []
    attempts = 0
    rejection_reasons: dict[str, int] = {}
    root_count_histogram: dict[str, int] = {}
    valid_root_count_histogram: dict[str, int] = {}
    minimum_root_separation = float("inf")
    maximum_relative_residual = 0.0
    while len(accepted_fibres) < required_fibres and attempts < maximum_attempts:
        attempt = _sample_one_bicubic_fibre(
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
            for x, y in attempt.coordinates:
                chart = int(np.argmax(np.abs(x))), int(np.argmax(np.abs(y)))
                converted.append(bicubic_point_in_chart(model, x, y, chart))
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
            f"only sampled {len(accepted_fibres)} complete bicubic fibres after "
            f"{attempts} attempts"
        )
    output = [
        point for fibre in accepted_fibres for point in fibre
    ][:n_points]
    truncated_final_fibre = bool(n_points % 3)
    diagnostics: dict[str, object] = {
        "sampler": "bicubic_random_line_all_three_roots",
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
        "importance_weight_used_for_acceptance": False,
        "implicit_jacobian_invertibility_required_for_acceptance": True,
    }
    return output, diagnostics


def sample_bicubic_points(
    model: BicubicModel,
    n_points: int,
    seed: int = 0,
    *,
    residual_tolerance: float = 1e-8,
    max_attempt_factor: int = 20,
) -> list[BicubicPoint]:
    """Sample complete random-line intersections with the cubic fibres."""

    points, _ = _sample_bicubic_points(
        model,
        n_points,
        seed=seed,
        residual_tolerance=residual_tolerance,
        max_attempt_factor=max_attempt_factor,
    )
    return points


def sample_bicubic_points_with_diagnostics(
    model: BicubicModel,
    n_points: int,
    seed: int = 0,
    *,
    residual_tolerance: float = 1e-8,
    max_attempt_factor: int = 20,
) -> tuple[list[BicubicPoint], dict[str, object]]:
    """Sample points and report complete-fibre acceptance diagnostics."""

    return _sample_bicubic_points(
        model,
        n_points,
        seed=seed,
        residual_tolerance=residual_tolerance,
        max_attempt_factor=max_attempt_factor,
    )


def bicubic_factor_metrics(point: BicubicPoint) -> tuple[Array, Array]:
    coords = point.affine_coordinates
    ambient_x = np.zeros((4, 4), dtype=np.complex128)
    ambient_y = np.zeros((4, 4), dtype=np.complex128)
    ambient_x[:2, :2] = fubini_study_metric(coords[:2])
    ambient_y[2:, 2:] = fubini_study_metric(coords[2:])
    tangent = point.tangent_basis
    x_metric = tangent.conjugate().T @ ambient_x @ tangent
    y_metric = tangent.conjugate().T @ ambient_y @ tangent
    return 0.5 * (x_metric + x_metric.conjugate().T), 0.5 * (y_metric + y_metric.conjugate().T)


def bicubic_baseline_metric(point: BicubicPoint) -> Array:
    x_metric, y_metric = bicubic_factor_metrics(point)
    return x_metric + y_metric


def bicubic_baseline_metrics(points: list[BicubicPoint]) -> Array:
    return np.asarray([bicubic_baseline_metric(point) for point in points], dtype=np.complex128)


def bicubic_holomorphic_volume_log_density(point: BicubicPoint) -> float:
    return float(-2.0 * np.log(abs(point.residue_denominator)))


def bicubic_proposal_log_density(point: BicubicPoint) -> float:
    """Return the sampler density omega_x^2/2 wedge omega_y."""

    x_metric, y_metric = bicubic_factor_metrics(point)
    coefficient = 0.0 + 0.0j
    for permutation in permutations(range(3)):
        inversions = sum(
            permutation[left] > permutation[right]
            for left in range(3)
            for right in range(left + 1, 3)
        )
        sign = -1.0 if inversions % 2 else 1.0
        for y_row in range(3):
            term = y_metric[y_row, permutation[y_row]]
            for x_row in range(3):
                if x_row != y_row:
                    term *= x_metric[x_row, permutation[x_row]]
            coefficient += sign * term
    density = float(np.real(coefficient))
    if not np.isfinite(density) or density <= 0:
        raise FloatingPointError("bicubic proposal density is not positive")
    return float(np.log(density))


def bicubic_importance_weights(points: list[BicubicPoint]) -> Array:
    log_weights = np.asarray([bicubic_importance_log_weight(point) for point in points])
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    return weights / np.mean(weights)


def bicubic_importance_log_weight(point: BicubicPoint) -> float:
    return bicubic_holomorphic_volume_log_density(point) - bicubic_proposal_log_density(point)


def bicubic_restricted_basis(
    points: list[BicubicPoint],
    degree: int,
    *,
    relative_rank_threshold: float = 1e-10,
) -> BicubicSectionBasis:
    exponents = bicubic_monomial_exponents(degree)
    matrix = []
    for point in points:
        x, y = normalize_bicubic_homogeneous(point.x, point.y, point.projective_chart)
        matrix.append(np.prod(np.concatenate([x, y])[None, :] ** exponents, axis=1))
    matrix = np.asarray(matrix, dtype=np.complex128)
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-14)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    rank = int(np.sum(singular_values > relative_rank_threshold * singular_values[0]))
    _, _, pivots = qr(matrix, mode="economic", pivoting=True)
    selected = np.asarray(pivots[:rank], dtype=np.int64)
    return BicubicSectionBasis(
        degree=degree,
        ambient_exponents=exponents,
        selected_indices=selected,
        selected_exponents=exponents[selected],
        numerical_rank=rank,
        singular_values=singular_values,
        relative_rank_threshold=relative_rank_threshold,
    )


def bicubic_restriction_coefficients(
    points: list[BicubicPoint],
    basis: BicubicSectionBasis,
) -> tuple[Array, float]:
    matrix = []
    for point in points:
        x, y = normalize_bicubic_homogeneous(point.x, point.y, point.projective_chart)
        matrix.append(np.prod(np.concatenate([x, y])[None, :] ** basis.ambient_exponents, axis=1))
    all_values = np.asarray(matrix, dtype=np.complex128)
    all_values /= np.maximum(np.linalg.norm(all_values, axis=1, keepdims=True), 1e-14)
    selected_values = all_values[:, basis.selected_indices]
    coefficients, _, _, _ = np.linalg.lstsq(
        selected_values, all_values, rcond=basis.relative_rank_threshold
    )
    error = float(
        np.linalg.norm(selected_values @ coefficients - all_values) / max(1.0, np.linalg.norm(all_values))
    )
    return coefficients, error


def _bicubic_multinomial_weights(exponents: Array) -> Array:
    weights = np.ones(len(exponents), dtype=float)
    for row, exponent in enumerate(exponents):
        for block in (exponent[:3], exponent[3:]):
            numerator = factorial(int(np.sum(block)))
            denominator = np.prod([factorial(int(value)) for value in block])
            weights[row] *= numerator / denominator
    return weights


def bicubic_fubini_study_h_matrix(
    points: list[BicubicPoint],
    basis: BicubicSectionBasis,
) -> tuple[Array, float]:
    coefficients, error = bicubic_restriction_coefficients(points, basis)
    weights = _bicubic_multinomial_weights(basis.ambient_exponents)
    h_matrix = np.conjugate(coefficients) @ (weights[:, None] * coefficients.T)
    return 0.5 * (h_matrix + h_matrix.conjugate().T), error


def bicubic_lift_h_matrix_power(
    points: list[BicubicPoint],
    source_exponents: Array,
    source_h_matrix: Array,
    target_basis: BicubicSectionBasis,
) -> tuple[Array, float]:
    """Represent a lower-degree section norm raised to an integer power."""

    exponents = np.asarray(source_exponents, dtype=np.int64)
    h_matrix = np.asarray(source_h_matrix, dtype=np.complex128)
    if exponents.ndim != 2 or exponents.shape[1] != 6:
        raise ValueError("source_exponents must have shape (n,6)")
    if h_matrix.shape != (len(exponents), len(exponents)):
        raise ValueError("source H-matrix dimension does not match its sections")
    source_x_degree = int(np.sum(exponents[0, :3]))
    source_y_degree = int(np.sum(exponents[0, 3:]))
    if source_x_degree != source_y_degree or source_x_degree <= 0:
        raise ValueError("source sections must have one positive O(k,k) degree")
    if not np.all(np.sum(exponents[:, :3], axis=1) == source_x_degree) or not np.all(
        np.sum(exponents[:, 3:], axis=1) == source_y_degree
    ):
        raise ValueError("source sections do not share one degree")
    if target_basis.degree % source_x_degree != 0:
        raise ValueError("target degree must be an integer multiple of source degree")
    power = target_basis.degree // source_x_degree

    current_exponents = exponents.copy()
    current_h = 0.5 * (h_matrix + h_matrix.conjugate().T)
    for factor in range(2, power + 1):
        ambient_exponents = bicubic_monomial_exponents(factor * source_x_degree)
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

    if not np.array_equal(current_exponents, target_basis.ambient_exponents):
        exponent_index = {tuple(row): index for index, row in enumerate(current_exponents)}
        order = np.asarray([exponent_index[tuple(row)] for row in target_basis.ambient_exponents])
        current_h = current_h[np.ix_(order, order)]

    coefficients, error = bicubic_restriction_coefficients(points, target_basis)
    target_h = np.conjugate(coefficients) @ current_h @ coefficients.T
    target_h = 0.5 * (target_h + target_h.conjugate().T)
    eigenvalues = np.linalg.eigvalsh(target_h)
    tolerance = 1e-11 * max(1.0, float(eigenvalues[-1]))
    if eigenvalues[0] < -tolerance:
        raise FloatingPointError("lifted bicubic H-matrix is not positive definite")
    if eigenvalues[0] <= 0:
        target_h += (tolerance - eigenvalues[0]) * np.eye(len(target_h))
    return target_h, error


def bicubic_lift_h_matrix_product(
    points: list[BicubicPoint],
    left_exponents: Array,
    left_h_matrix: Array,
    right_exponents: Array,
    right_h_matrix: Array,
    target_basis: BicubicSectionBasis,
) -> tuple[Array, float]:
    """Represent the product of two section norms at the summed degree."""

    left_exponents = np.asarray(left_exponents, dtype=np.int64)
    right_exponents = np.asarray(right_exponents, dtype=np.int64)
    left_h_matrix = np.asarray(left_h_matrix, dtype=np.complex128)
    right_h_matrix = np.asarray(right_h_matrix, dtype=np.complex128)

    def section_degree(exponents: Array, h_matrix: Array, name: str) -> int:
        if exponents.ndim != 2 or exponents.shape[1] != 6:
            raise ValueError(f"{name}_exponents must have shape (n,6)")
        if h_matrix.shape != (len(exponents), len(exponents)):
            raise ValueError(f"{name} H-matrix dimension does not match its sections")
        x_degrees = np.sum(exponents[:, :3], axis=1)
        y_degrees = np.sum(exponents[:, 3:], axis=1)
        degree = int(x_degrees[0])
        if degree <= 0 or not np.all(x_degrees == degree) or not np.all(y_degrees == degree):
            raise ValueError(f"{name} sections must share one positive O(k,k) degree")
        return degree

    left_degree = section_degree(left_exponents, left_h_matrix, "left")
    right_degree = section_degree(right_exponents, right_h_matrix, "right")
    if left_degree + right_degree != target_basis.degree:
        raise ValueError("target degree must equal the sum of source degrees")

    ambient_exponents = target_basis.ambient_exponents
    exponent_index = {tuple(row): index for index, row in enumerate(ambient_exponents)}
    multiplication = np.empty((len(left_exponents), len(right_exponents)), dtype=np.int64)
    for left_index, left in enumerate(left_exponents):
        for right_index, right in enumerate(right_exponents):
            multiplication[left_index, right_index] = exponent_index[tuple(left + right)]

    ambient_h = np.zeros((len(ambient_exponents), len(ambient_exponents)), dtype=np.complex128)
    for left_row in range(len(left_exponents)):
        ambient_rows = multiplication[left_row]
        for left_column in range(len(left_exponents)):
            ambient_columns = multiplication[left_column]
            ambient_h[np.ix_(ambient_rows, ambient_columns)] += (
                left_h_matrix[left_row, left_column] * right_h_matrix
            )

    coefficients, error = bicubic_restriction_coefficients(points, target_basis)
    target_h = np.conjugate(coefficients) @ ambient_h @ coefficients.T
    target_h = 0.5 * (target_h + target_h.conjugate().T)
    eigenvalues = np.linalg.eigvalsh(target_h)
    tolerance = 1e-11 * max(1.0, float(eigenvalues[-1]))
    if eigenvalues[0] < -tolerance:
        raise FloatingPointError("product bicubic H-matrix is not positive semidefinite")
    if eigenvalues[0] <= 0:
        target_h += (tolerance - eigenvalues[0]) * np.eye(len(target_h))
    return target_h, error


def bicubic_section_values_and_jacobian(point: BicubicPoint, exponents: Array) -> tuple[Array, Array]:
    values, ambient_jacobian = bicubic_section_values_and_ambient_jacobian(
        point.affine_coordinates, point.projective_chart, exponents
    )
    return values, ambient_jacobian @ point.tangent_basis


def bicubic_global_h_metric(
    point: BicubicPoint,
    exponents: Array,
    h_matrix: Array,
    normalization: float = 1.0,
) -> Array:
    values, jacobian = bicubic_section_values_and_jacobian(point, exponents)
    return h_matrix_metric(values, jacobian, h_matrix, normalization=normalization)


def bicubic_global_h_metrics(
    points: list[BicubicPoint],
    exponents: Array,
    h_matrix: Array,
    normalization: float = 1.0,
) -> Array:
    return np.asarray(
        [bicubic_global_h_metric(point, exponents, h_matrix, normalization) for point in points],
        dtype=np.complex128,
    )


def bicubic_residual_values(points: list[BicubicPoint], metrics: Array) -> Array:
    eigenvalues = np.linalg.eigvalsh(np.asarray(metrics, dtype=np.complex128))
    values = np.full(len(points), np.nan, dtype=float)
    positive = eigenvalues[:, 0] > 0
    values[positive] = np.sum(np.log(eigenvalues[positive]), axis=1) - np.asarray(
        [bicubic_holomorphic_volume_log_density(point) for point in points]
    )[positive]
    return values


def bicubic_residual_stats(
    points: list[BicubicPoint],
    metrics: Array,
    weights: Array | None = None,
) -> ResidualStats:
    eigenvalues = np.linalg.eigvalsh(np.asarray(metrics, dtype=np.complex128))
    minimum = float(np.min(eigenvalues))
    values = bicubic_residual_values(points, metrics)
    if not np.all(np.isfinite(values)):
        return ResidualStats(float("nan"), float("nan"), minimum, float("nan"))
    importance = np.ones(len(points)) if weights is None else np.asarray(weights, dtype=float)
    total = float(np.sum(importance))
    mean = float(np.sum(importance * values) / total)
    centered = values - mean
    return ResidualStats(
        rms=float(np.sqrt(np.sum(importance * centered**2) / total)),
        max_abs=float(np.max(np.abs(centered))),
        min_eigenvalue=minimum,
        mean_log_error=mean,
    )
