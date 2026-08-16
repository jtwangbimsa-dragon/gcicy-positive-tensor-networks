"""Empirical regularity diagnostics for finite-dimensional H-metrics.

The routines in this module measure local derivatives on the threefold.  They
do not construct a global Lipschitz bound or certify an epsilon-net.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .adapter import GCICYAdapter, HMetricArtifact


Array = np.ndarray


@dataclass(frozen=True)
class EmpiricalTangentGradient:
    """Two-scale central-difference estimate in six real orthonormal directions."""

    center_raw_log_ratio: float
    coarse_step: float
    fine_step: float
    reference_metric_eigenvalues: Array
    coarse_components: Array
    fine_components: Array
    richardson_components: Array
    richardson_error_components: Array
    coarse_distances: Array
    fine_distances: Array
    maximum_retraction_residual: float
    maximum_newton_iterations: int

    @property
    def coarse_norm(self) -> float:
        return float(np.linalg.norm(self.coarse_components))

    @property
    def fine_norm(self) -> float:
        return float(np.linalg.norm(self.fine_components))

    @property
    def richardson_norm(self) -> float:
        return float(np.linalg.norm(self.richardson_components))

    @property
    def richardson_error_norm(self) -> float:
        return float(np.linalg.norm(self.richardson_error_components))

    def to_dict(self) -> dict[str, Any]:
        gradient_scale = max(self.richardson_norm, np.finfo(float).tiny)
        return {
            "interpretation": "empirical_local_derivative_not_global_bound",
            "center_raw_log_ratio": self.center_raw_log_ratio,
            "coarse_step": self.coarse_step,
            "fine_step": self.fine_step,
            "reference_metric_eigenvalues": self.reference_metric_eigenvalues.tolist(),
            "coarse_components": self.coarse_components.tolist(),
            "fine_components": self.fine_components.tolist(),
            "richardson_components": self.richardson_components.tolist(),
            "richardson_error_components": (
                self.richardson_error_components.tolist()
            ),
            "coarse_gradient_norm": self.coarse_norm,
            "fine_gradient_norm": self.fine_norm,
            "richardson_gradient_norm": self.richardson_norm,
            "richardson_error_norm": self.richardson_error_norm,
            "relative_richardson_error": (
                self.richardson_error_norm / gradient_scale
            ),
            "coarse_product_fubini_study_distances": self.coarse_distances.tolist(),
            "fine_product_fubini_study_distances": self.fine_distances.tolist(),
            "coarse_distance_over_step": (
                self.coarse_distances / self.coarse_step
            ).tolist(),
            "fine_distance_over_step": (
                self.fine_distances / self.fine_step
            ).tolist(),
            "maximum_retraction_residual": self.maximum_retraction_residual,
            "maximum_newton_iterations": self.maximum_newton_iterations,
        }


def reference_orthonormal_complex_directions(reference_metric: Array) -> Array:
    """Return complex tangent columns orthonormal for a Hermitian metric."""

    metric = np.asarray(reference_metric, dtype=np.complex128)
    if metric.ndim != 2 or metric.shape[0] != metric.shape[1] or len(metric) == 0:
        raise ValueError("reference metric must be a non-empty square matrix")
    metric = 0.5 * (metric + metric.conjugate().T)
    eigenvalues, eigenvectors = np.linalg.eigh(metric)
    if not np.all(np.isfinite(eigenvalues)) or eigenvalues[0] <= 0:
        raise FloatingPointError("reference metric must be positive definite")
    directions = (
        eigenvectors * (1.0 / np.sqrt(eigenvalues))[None, :]
    ) @ eigenvectors.conjugate().T
    gram = directions.conjugate().T @ metric @ directions
    if float(np.max(np.abs(gram - np.eye(len(metric))))) > 1e-9:
        raise FloatingPointError("failed to construct reference-orthonormal directions")
    return directions


def estimate_h_metric_residual_gradient(
    adapter: GCICYAdapter,
    model: Any,
    point: Any,
    artifact: HMetricArtifact,
    *,
    coarse_step: float = 1e-3,
    residual_tolerance: float = 1e-10,
    maximum_newton_iterations: int = 16,
) -> EmpiricalTangentGradient:
    """Estimate ``|grad log(det(g_H)/|Omega|^2)|`` at one point.

    The norm is measured using the adapter's baseline metric.  Six real
    tangent directions are obtained from three complex orthonormal columns and
    their multiples by ``1j``.  Central differences at ``h`` and ``h/2`` are
    combined with Richardson extrapolation.
    """

    if not np.isfinite(coarse_step) or coarse_step <= 0:
        raise ValueError("coarse step must be finite and positive")
    if residual_tolerance <= 0 or maximum_newton_iterations <= 0:
        raise ValueError("retraction solver controls must be positive")
    reference_metric = np.asarray(adapter.baseline_metric(point), dtype=np.complex128)
    eigenvalues = np.linalg.eigvalsh(reference_metric)
    directions = reference_orthonormal_complex_directions(reference_metric)
    real_directions = [
        phase * directions[:, index]
        for index in range(directions.shape[1])
        for phase in (1.0, 1.0j)
    ]
    fine_step = 0.5 * coarse_step
    retracted_points: list[Any] = []
    diagnostics: list[dict[str, Any]] = []
    locations: dict[tuple[int, str, int], int] = {}
    for direction_index, direction in enumerate(real_directions):
        for scale_name, scale in (("coarse", coarse_step), ("fine", fine_step)):
            for sign in (-1, 1):
                candidate, row = adapter.retract_intrinsic_step(
                    model,
                    point,
                    sign * scale * direction,
                    residual_tolerance=residual_tolerance,
                    maximum_newton_iterations=maximum_newton_iterations,
                )
                locations[(direction_index, scale_name, sign)] = len(retracted_points)
                retracted_points.append(candidate)
                diagnostics.append(row)
    metrics = adapter.h_metrics(retracted_points, artifact)
    residuals = np.asarray(
        adapter.residual_values(retracted_points, metrics),
        dtype=float,
    )
    if not np.all(np.isfinite(residuals)):
        raise FloatingPointError("retracted H-metric residual is not finite")

    coarse_components = np.empty(len(real_directions), dtype=float)
    fine_components = np.empty(len(real_directions), dtype=float)
    coarse_distances = np.empty((len(real_directions), 2), dtype=float)
    fine_distances = np.empty((len(real_directions), 2), dtype=float)
    for direction_index in range(len(real_directions)):
        coarse_minus = locations[(direction_index, "coarse", -1)]
        coarse_plus = locations[(direction_index, "coarse", 1)]
        fine_minus = locations[(direction_index, "fine", -1)]
        fine_plus = locations[(direction_index, "fine", 1)]
        coarse_components[direction_index] = (
            residuals[coarse_plus] - residuals[coarse_minus]
        ) / (2.0 * coarse_step)
        fine_components[direction_index] = (
            residuals[fine_plus] - residuals[fine_minus]
        ) / (2.0 * fine_step)
        coarse_distances[direction_index] = (
            float(diagnostics[coarse_minus]["product_fubini_study_distance"]),
            float(diagnostics[coarse_plus]["product_fubini_study_distance"]),
        )
        fine_distances[direction_index] = (
            float(diagnostics[fine_minus]["product_fubini_study_distance"]),
            float(diagnostics[fine_plus]["product_fubini_study_distance"]),
        )
    richardson = (4.0 * fine_components - coarse_components) / 3.0
    richardson_error = np.abs(fine_components - coarse_components) / 3.0
    center_metric = adapter.h_metric(point, artifact)
    center_residual = adapter.monge_ampere_log_error(point, center_metric)
    return EmpiricalTangentGradient(
        center_raw_log_ratio=float(center_residual),
        coarse_step=float(coarse_step),
        fine_step=float(fine_step),
        reference_metric_eigenvalues=np.asarray(eigenvalues, dtype=float),
        coarse_components=coarse_components,
        fine_components=fine_components,
        richardson_components=richardson,
        richardson_error_components=richardson_error,
        coarse_distances=coarse_distances,
        fine_distances=fine_distances,
        maximum_retraction_residual=float(
            max(float(row["relative_equation_residual"]) for row in diagnostics)
        ),
        maximum_newton_iterations=max(
            int(row["newton_iterations"]) for row in diagnostics
        ),
    )


def finite_distance_residual_slopes(
    adapter: GCICYAdapter,
    centers: list[Any],
    points: list[Any],
    center_ids: Array,
    center_residuals: Array,
    point_residuals: Array,
) -> tuple[Array, Array]:
    """Return ambient-FS secant slopes and center distances for local points."""

    labels = np.asarray(center_ids, dtype=np.int64)
    center_values = np.asarray(center_residuals, dtype=float)
    values = np.asarray(point_residuals, dtype=float)
    if labels.shape != (len(points),) or values.shape != (len(points),):
        raise ValueError("point labels and residuals must match the local points")
    if center_values.shape != (len(centers),):
        raise ValueError("center residuals must match the centers")
    if len(points) == 0 or np.min(labels) < 0 or np.max(labels) >= len(centers):
        raise ValueError("center ids are invalid")
    distances = np.asarray(
        [
            adapter.point_distance(centers[int(center_id)], point)
            for center_id, point in zip(labels, points, strict=True)
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(distances)) or np.min(distances) <= 0:
        raise FloatingPointError("local secant distances must be finite and positive")
    slopes = np.abs(values - center_values[labels]) / distances
    return slopes, distances
