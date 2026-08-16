"""Projective atlas utilities for the local gCICY model."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from .simple_patch import Array, embedding, fubini_study_metric, holomorphic_jacobian, induced_metric


Chart = tuple[int, int, int]


@dataclass(frozen=True)
class ChartMetricDiagnostic:
    """Metric consistency diagnostics for one projective affine chart."""

    chart: Chart
    attempted_points: int
    usable_points: int
    min_selected_coordinate: float
    max_abs_error: float
    max_relative_error: float
    mean_relative_error: float


def all_projective_charts() -> list[Chart]:
    """Return all affine charts for P1 x P1 x P5."""

    return list(product(range(2), range(2), range(6)))


def homogeneous_coordinates(params: Array) -> tuple[Array, Array, Array]:
    """Return homogeneous coordinates in P1 x P1 x P5 for local params."""

    a, b, w1, w2, w3, w4, w5 = embedding(params)
    x = np.array([1.0, a], dtype=np.complex128)
    y = np.array([1.0, b], dtype=np.complex128)
    z = np.array([1.0, w1, w2, w3, w4, w5], dtype=np.complex128)
    return x, y, z


def affine_factor_coordinates(homogeneous: Array, selected_index: int) -> Array:
    """Affine coordinates for one projective factor on a selected chart."""

    values = np.asarray(homogeneous, dtype=np.complex128).reshape(-1)
    if not 0 <= selected_index < values.size:
        raise ValueError(f"Chart index {selected_index} is out of range for P^{values.size - 1}.")
    selected = values[selected_index]
    if abs(selected) < 1e-14:
        raise FloatingPointError("Selected homogeneous coordinate is too close to zero.")
    return np.delete(values / selected, selected_index)


def selected_coordinate_abs(params: Array, chart: Chart) -> float:
    """Minimum absolute value of the selected homogeneous coordinates."""

    x, y, z = homogeneous_coordinates(params)
    return float(min(abs(x[chart[0]]), abs(y[chart[1]]), abs(z[chart[2]])))


def affine_chart_coordinates(params: Array, chart: Chart = (0, 0, 0)) -> Array:
    """Return affine coordinates in a chosen P1 x P1 x P5 chart."""

    x, y, z = homogeneous_coordinates(params)
    return np.concatenate(
        [
            affine_factor_coordinates(x, chart[0]),
            affine_factor_coordinates(y, chart[1]),
            affine_factor_coordinates(z, chart[2]),
        ]
    )


def ambient_chart_metric(chart_coords: Array, weights: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Array:
    """Product Fubini-Study metric in arbitrary affine chart coordinates."""

    coords = np.asarray(chart_coords, dtype=np.complex128)
    if coords.shape != (7,):
        raise ValueError(f"Expected 7 affine chart coordinates, got {coords.shape}.")
    metric = np.zeros((7, 7), dtype=np.complex128)
    metric[0:1, 0:1] = weights[0] * fubini_study_metric(coords[0:1])
    metric[1:2, 1:2] = weights[1] * fubini_study_metric(coords[1:2])
    metric[2:7, 2:7] = weights[2] * fubini_study_metric(coords[2:7])
    return metric


def induced_metric_in_chart(
    params: Array,
    chart: Chart = (0, 0, 0),
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Array:
    """Pull back the product Fubini-Study metric through an arbitrary chart."""

    coords = affine_chart_coordinates(params, chart)
    metric = ambient_chart_metric(coords, weights=weights)
    jacobian = holomorphic_jacobian(lambda point: affine_chart_coordinates(point, chart), params)
    pulled_back = jacobian.conjugate().T @ metric @ jacobian
    return 0.5 * (pulled_back + pulled_back.conjugate().T)


def chart_metric_diagnostic(
    params: Array,
    chart: Chart,
    *,
    min_abs_selected: float = 1e-5,
) -> ChartMetricDiagnostic:
    """Compare arbitrary-chart and base-chart pullback FS metrics."""

    points = np.asarray(params, dtype=np.complex128)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected an array of shape (n, 3), got {points.shape}.")

    abs_errors: list[float] = []
    relative_errors: list[float] = []
    selected_values: list[float] = []
    for point in points:
        selected = selected_coordinate_abs(point, chart)
        if selected <= min_abs_selected:
            continue
        selected_values.append(selected)
        reference = induced_metric(point)
        candidate = induced_metric_in_chart(point, chart)
        abs_error = float(np.linalg.norm(candidate - reference))
        relative_error = abs_error / max(1.0, float(np.linalg.norm(reference)))
        abs_errors.append(abs_error)
        relative_errors.append(relative_error)

    if not abs_errors:
        return ChartMetricDiagnostic(
            chart=chart,
            attempted_points=int(points.shape[0]),
            usable_points=0,
            min_selected_coordinate=float("nan"),
            max_abs_error=float("nan"),
            max_relative_error=float("nan"),
            mean_relative_error=float("nan"),
        )

    return ChartMetricDiagnostic(
        chart=chart,
        attempted_points=int(points.shape[0]),
        usable_points=len(abs_errors),
        min_selected_coordinate=float(np.min(selected_values)),
        max_abs_error=float(np.max(abs_errors)),
        max_relative_error=float(np.max(relative_errors)),
        mean_relative_error=float(np.mean(relative_errors)),
    )


def diagnostics_to_dict(diagnostic: ChartMetricDiagnostic) -> dict[str, float | int | list[int]]:
    """Serialize a chart diagnostic as plain JSON-compatible data."""

    return {
        "chart": list(diagnostic.chart),
        "attempted_points": diagnostic.attempted_points,
        "usable_points": diagnostic.usable_points,
        "min_selected_coordinate": diagnostic.min_selected_coordinate,
        "max_abs_error": diagnostic.max_abs_error,
        "max_relative_error": diagnostic.max_relative_error,
        "mean_relative_error": diagnostic.mean_relative_error,
    }
