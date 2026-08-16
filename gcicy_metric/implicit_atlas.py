"""Implicit local-coordinate atlas for the affine gCICY patch."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import warnings

import numpy as np

from .simple_patch import (
    Array,
    ambient_fs_metric,
    embedding,
    equation_jacobian,
    holomorphic_volume_log_density,
    induced_metric,
)


CoordinateChart = tuple[int, int, int]
REFERENCE_CHART: CoordinateChart = (3, 4, 5)


@dataclass(frozen=True)
class ImplicitChartDiagnostic:
    """Consistency diagnostics for one implicit local-coordinate chart."""

    chart: CoordinateChart
    attempted_points: int
    usable_points: int
    min_abs_minor_det: float
    max_condition_number: float
    max_metric_relative_error: float
    max_ma_error: float


def all_implicit_charts() -> list[CoordinateChart]:
    """Return all choices of three independent ambient affine coordinates."""

    return list(combinations(range(7), 3))


def dependent_indices(independent: CoordinateChart) -> tuple[int, int, int, int]:
    """Return complementary dependent ambient coordinate indices."""

    independent_set = set(independent)
    return tuple(idx for idx in range(7) if idx not in independent_set)


def implicit_tangent_basis(ambient_coords: Array, independent: CoordinateChart) -> Array:
    """Return dz/du for an implicit local-coordinate chart."""

    jacobian = equation_jacobian(ambient_coords)
    dependent = dependent_indices(independent)
    f_dependent = jacobian[:, dependent]
    f_independent = jacobian[:, independent]
    solved = -np.linalg.solve(f_dependent, f_independent)
    basis = np.zeros((7, 3), dtype=np.complex128)
    for col, idx in enumerate(independent):
        basis[idx, col] = 1.0
    for row, idx in enumerate(dependent):
        basis[idx, :] = solved[row, :]
    return basis


def implicit_minor_det(ambient_coords: Array, independent: CoordinateChart) -> complex:
    """Return det dF/dz_dependent for an implicit chart."""

    jacobian = equation_jacobian(ambient_coords)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.linalg.det(jacobian[:, dependent_indices(independent)])


def implicit_chart_condition(ambient_coords: Array, independent: CoordinateChart) -> float:
    """Return the condition number of the implicit dependent-coordinate minor."""

    jacobian = equation_jacobian(ambient_coords)
    return float(np.linalg.cond(jacobian[:, dependent_indices(independent)]))


def implicit_induced_metric(ambient_coords: Array, independent: CoordinateChart) -> Array:
    """Pull back the ambient FS metric using an implicit tangent basis."""

    basis = implicit_tangent_basis(ambient_coords, independent)
    metric = basis.conjugate().T @ ambient_fs_metric(ambient_coords) @ basis
    return 0.5 * (metric + metric.conjugate().T)


def implicit_holomorphic_volume_log_density(ambient_coords: Array, independent: CoordinateChart) -> float:
    """Return log |Omega|^2 in the implicit local coordinates."""

    det = implicit_minor_det(ambient_coords, independent)
    if abs(det) < 1e-14:
        raise FloatingPointError("Implicit residue denominator is too close to zero.")
    return float(-2.0 * np.log(abs(det)))


def chart_transition_from_reference(params: Array, independent: CoordinateChart) -> Array:
    """Return du_chart / d(s,t,r) for a chart on the explicit parameter domain."""

    reference_basis = implicit_tangent_basis(embedding(params), REFERENCE_CHART)
    return reference_basis[list(independent), :]


def transformed_metric_from_chart_to_reference(
    ambient_coords: Array,
    independent: CoordinateChart,
    params: Array,
) -> Array:
    """Transform an implicit-chart metric back to reference (s,t,r) coordinates."""

    transition = chart_transition_from_reference(params, independent)
    chart_metric = implicit_induced_metric(ambient_coords, independent)
    return transition.conjugate().T @ chart_metric @ transition


def implicit_ma_log_error(ambient_coords: Array, independent: CoordinateChart) -> float:
    """Return log det(g) - log |Omega|^2 in an implicit chart."""

    metric = implicit_induced_metric(ambient_coords, independent)
    eigvals = np.linalg.eigvalsh(metric)
    if eigvals[0] <= 0:
        raise FloatingPointError(f"Metric is not positive definite: eigenvalues={eigvals}.")
    return float(np.sum(np.log(eigvals)) - implicit_holomorphic_volume_log_density(ambient_coords, independent))


def implicit_chart_diagnostic(
    params: Array,
    independent: CoordinateChart,
    *,
    min_abs_minor_det: float = 1e-8,
    max_condition_number: float = 1e8,
) -> ImplicitChartDiagnostic:
    """Compare implicit-chart baseline metric and MA error against the reference chart."""

    points = np.asarray(params, dtype=np.complex128)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected an array of shape (n, 3), got {points.shape}.")

    metric_errors: list[float] = []
    ma_errors: list[float] = []
    minor_values: list[float] = []
    conditions: list[float] = []
    for point in points:
        ambient = embedding(point)
        minor = abs(implicit_minor_det(ambient, independent))
        condition = implicit_chart_condition(ambient, independent)
        if minor <= min_abs_minor_det or condition >= max_condition_number:
            continue
        minor_values.append(float(minor))
        conditions.append(float(condition))
        reference_metric = induced_metric(point)
        candidate = transformed_metric_from_chart_to_reference(ambient, independent, point)
        metric_error = np.linalg.norm(candidate - reference_metric) / max(1.0, np.linalg.norm(reference_metric))
        metric_errors.append(float(metric_error))
        reference_ma = float(np.sum(np.log(np.linalg.eigvalsh(reference_metric))) - holomorphic_volume_log_density(point))
        ma_errors.append(abs(implicit_ma_log_error(ambient, independent) - reference_ma))

    if not metric_errors:
        return ImplicitChartDiagnostic(
            chart=independent,
            attempted_points=int(points.shape[0]),
            usable_points=0,
            min_abs_minor_det=float("nan"),
            max_condition_number=float("nan"),
            max_metric_relative_error=float("nan"),
            max_ma_error=float("nan"),
        )

    return ImplicitChartDiagnostic(
        chart=independent,
        attempted_points=int(points.shape[0]),
        usable_points=len(metric_errors),
        min_abs_minor_det=float(np.min(minor_values)),
        max_condition_number=float(np.max(conditions)),
        max_metric_relative_error=float(np.max(metric_errors)),
        max_ma_error=float(np.max(ma_errors)),
    )


def implicit_diagnostic_to_dict(diagnostic: ImplicitChartDiagnostic) -> dict[str, float | int | list[int]]:
    """Serialize an implicit chart diagnostic as JSON-compatible data."""

    return {
        "chart": list(diagnostic.chart),
        "attempted_points": diagnostic.attempted_points,
        "usable_points": diagnostic.usable_points,
        "min_abs_minor_det": diagnostic.min_abs_minor_det,
        "max_condition_number": diagnostic.max_condition_number,
        "max_metric_relative_error": diagnostic.max_metric_relative_error,
        "max_ma_error": diagnostic.max_ma_error,
    }
