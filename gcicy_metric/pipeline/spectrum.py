"""Metric-dependent scalar-Laplacian Ritz spectra for gCICY adapters."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from .adapter import GCICYAdapter, HMetricArtifact
from .audit import confidence_interval, effective_sample_size


Array = np.ndarray


@dataclass(frozen=True)
class ScalarTrialData:
    """Values and holomorphic gradients of real global scalar functions."""

    values: Array
    holomorphic_gradients: Array
    linear_feature_count: int

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        gradients = np.asarray(self.holomorphic_gradients)
        if values.ndim != 1 or gradients.ndim != 2:
            raise ValueError("scalar trial values and gradients have invalid dimensions")
        if gradients.shape[0] != values.size:
            raise ValueError("scalar trial value and gradient counts differ")
        if not 0 < self.linear_feature_count <= values.size:
            raise ValueError("linear feature count is inconsistent with trial data")
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(gradients)):
            raise ValueError("scalar trial data must be finite")


def _traceless_hermitian_basis(size: int) -> Array:
    if size < 2:
        raise ValueError("a projective factor must have at least two coordinates")
    basis: list[Array] = []
    for left in range(size):
        for right in range(left + 1, size):
            symmetric = np.zeros((size, size), dtype=np.complex128)
            symmetric[left, right] = 1.0 / np.sqrt(2.0)
            symmetric[right, left] = 1.0 / np.sqrt(2.0)
            basis.append(symmetric)

            antisymmetric = np.zeros((size, size), dtype=np.complex128)
            antisymmetric[left, right] = -1j / np.sqrt(2.0)
            antisymmetric[right, left] = 1j / np.sqrt(2.0)
            basis.append(antisymmetric)
    for order in range(1, size):
        diagonal = np.zeros((size, size), dtype=np.complex128)
        scale = np.sqrt(order * (order + 1.0))
        diagonal[np.arange(order), np.arange(order)] = 1.0 / scale
        diagonal[order, order] = -order / scale
        basis.append(diagonal)
    output = np.asarray(basis, dtype=np.complex128)
    if output.shape != (size * size - 1, size, size):
        raise AssertionError("internal Hermitian basis dimension is incorrect")
    return output


@lru_cache(maxsize=None)
def _random_cubic_directions(
    linear_feature_count: int,
    cubic_feature_count: int,
    seed: int,
) -> Array:
    """Return fixed unit directions for metric-independent cubic enrichment."""

    if linear_feature_count <= 0 or cubic_feature_count <= 0:
        raise ValueError("linear and cubic feature counts must be positive")
    rng = np.random.default_rng(seed)
    directions = rng.normal(
        size=(cubic_feature_count, linear_feature_count)
    ).astype(float)
    norms = np.linalg.norm(directions, axis=1)
    if np.any(norms <= 0) or not np.all(np.isfinite(norms)):
        raise FloatingPointError("random cubic directions could not be normalized")
    directions /= norms[:, None]
    directions.setflags(write=False)
    return directions


def product_projective_moment_map_trial_data(
    affine_coordinates: Array,
    projective_chart: Sequence[int],
    tangent_basis: Array,
    factor_dimensions: Sequence[int],
    *,
    level: int,
    cubic_feature_count: int = 256,
    cubic_feature_seed: int = 314159,
) -> ScalarTrialData:
    """Build nested moment-map trial spaces through a cubic enrichment."""

    if level not in (1, 2, 3):
        raise ValueError("only scalar trial levels 1, 2, and 3 are implemented")
    if level == 3 and cubic_feature_count <= 0:
        raise ValueError("level 3 requires a positive cubic feature count")
    dimensions = tuple(int(value) for value in factor_dimensions)
    chart = tuple(int(value) for value in projective_chart)
    affine = np.asarray(affine_coordinates, dtype=np.complex128).reshape(-1)
    tangent = np.asarray(tangent_basis, dtype=np.complex128)
    ambient_dimension = sum(dimensions)
    if len(chart) != len(dimensions):
        raise ValueError("projective chart and factor counts differ")
    if affine.size != ambient_dimension or tangent.shape[0] != ambient_dimension:
        raise ValueError("affine coordinates or tangent basis have the wrong ambient size")

    values: list[float] = []
    gradients: list[Array] = []
    offset = 0
    for dimension, selected in zip(dimensions, chart, strict=True):
        size = dimension + 1
        if selected < 0 or selected >= size:
            raise ValueError("projective chart index is out of range")
        active = [index for index in range(size) if index != selected]
        homogeneous = np.ones(size, dtype=np.complex128)
        homogeneous[selected] = 1.0
        homogeneous[active] = affine[offset : offset + dimension]
        differential = np.zeros((size, tangent.shape[1]), dtype=np.complex128)
        differential[active] = tangent[offset : offset + dimension]
        offset += dimension

        norm_squared = float(np.real(np.vdot(homogeneous, homogeneous)))
        if not np.isfinite(norm_squared) or norm_squared <= 0:
            raise FloatingPointError("homogeneous coordinate norm is not positive")
        norm_gradient = np.conjugate(homogeneous) @ differential
        for generator in _traceless_hermitian_basis(size):
            numerator_complex = np.vdot(homogeneous, generator @ homogeneous)
            numerator = float(np.real(numerator_complex))
            if abs(float(np.imag(numerator_complex))) > 1e-10:
                raise FloatingPointError("Hermitian moment map is not real")
            numerator_gradient = np.conjugate(homogeneous) @ generator @ differential
            values.append(numerator / norm_squared)
            gradients.append(
                (numerator_gradient * norm_squared - numerator * norm_gradient)
                / norm_squared**2
            )

    linear_values = np.asarray(values, dtype=float)
    linear_gradients = np.asarray(gradients, dtype=np.complex128)
    linear_count = linear_values.size
    if level == 1:
        return ScalarTrialData(linear_values, linear_gradients, linear_count)

    left, right = np.triu_indices(linear_count)
    quadratic_values = linear_values[left] * linear_values[right]
    quadratic_gradients = (
        linear_gradients[left] * linear_values[right, None]
        + linear_values[left, None] * linear_gradients[right]
    )
    quadratic = ScalarTrialData(
        values=np.concatenate((linear_values, quadratic_values)),
        holomorphic_gradients=np.concatenate(
            (linear_gradients, quadratic_gradients),
            axis=0,
        ),
        linear_feature_count=linear_count,
    )
    if level == 2:
        return quadratic

    directions = _random_cubic_directions(
        linear_count,
        int(cubic_feature_count),
        int(cubic_feature_seed),
    )
    projected_values = directions @ linear_values
    projected_gradients = directions @ linear_gradients
    cubic_values = projected_values**3
    cubic_gradients = 3.0 * projected_values[:, None] ** 2 * projected_gradients
    return ScalarTrialData(
        values=np.concatenate((quadratic.values, cubic_values)),
        holomorphic_gradients=np.concatenate(
            (quadratic.holomorphic_gradients, cubic_gradients),
            axis=0,
        ),
        linear_feature_count=linear_count,
    )


def metric_volume_weights(
    adapter: GCICYAdapter,
    points: Sequence[Any],
    metrics: Array,
) -> tuple[Array, Array]:
    """Return normalized dV_g/proposal weights and their unshifted logs."""

    residuals = adapter.residual_values(points, metrics)
    log_weights = residuals + np.asarray(
        [adapter.importance_log_weight(point) for point in points],
        dtype=float,
    )
    if not np.all(np.isfinite(log_weights)):
        raise FloatingPointError("metric-volume weights are not finite")
    shifted = np.exp(log_weights - float(np.max(log_weights)))
    total = float(np.sum(shifted))
    if not np.isfinite(total) or total <= 0:
        raise FloatingPointError("metric-volume weights cannot be normalized")
    return shifted / total, log_weights


def sampling_cluster_statistics(weights: Array, cluster_ids: Array) -> dict[str, Any]:
    """Summarize point- and cluster-weight concentration.

    The cluster-weight ESS is a diagnostic for importance-weight concentration
    after summing weights within sampling clusters. It is not an
    observable-specific estimate of the number of independent measurements.
    """

    normalized_weights = np.asarray(weights, dtype=float).reshape(-1)
    identifiers = np.asarray(cluster_ids, dtype=np.int64).reshape(-1)
    if normalized_weights.shape != identifiers.shape or len(identifiers) == 0:
        raise ValueError("sampling cluster ids must match the non-empty weight array")
    if np.any(normalized_weights < 0) or not np.isclose(
        float(np.sum(normalized_weights)),
        1.0,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("sampling cluster statistics require normalized weights")
    _, inverse, counts = np.unique(
        identifiers,
        return_inverse=True,
        return_counts=True,
    )
    cluster_weights = np.bincount(
        inverse,
        weights=normalized_weights,
        minlength=len(counts),
    )
    cluster_weight_ess = effective_sample_size(cluster_weights)
    return {
        "point_effective_sample_size": effective_sample_size(normalized_weights),
        "sampling_cluster_count": int(len(counts)),
        "minimum_points_per_sampling_cluster": int(np.min(counts)),
        "maximum_points_per_sampling_cluster": int(np.max(counts)),
        "cluster_weight_effective_sample_size": cluster_weight_ess,
        # Compatibility alias for audit JSON written before the terminology
        # was made explicit in the manuscript.
        "sampling_cluster_effective_sample_size": cluster_weight_ess,
    }


def _solve_ritz_matrices(
    mass: Array,
    stiffness: Array,
    *,
    eigenvalue_count: int,
    mass_relative_threshold: float,
) -> dict[str, Any]:
    mass_matrix = 0.5 * (np.asarray(mass, dtype=float) + np.asarray(mass, dtype=float).T)
    stiffness_matrix = 0.5 * (
        np.asarray(stiffness, dtype=float) + np.asarray(stiffness, dtype=float).T
    )
    mass_eigenvalues, mass_vectors = np.linalg.eigh(mass_matrix)
    largest_mass = float(mass_eigenvalues[-1])
    if largest_mass <= 0:
        raise FloatingPointError("scalar trial mass matrix has zero rank")
    retained = mass_eigenvalues > mass_relative_threshold * largest_mass
    retained_rank = int(np.sum(retained))
    if retained_rank < eigenvalue_count:
        raise FloatingPointError(
            f"scalar trial rank {retained_rank} is smaller than requested spectrum "
            f"length {eigenvalue_count}"
        )
    whitening = mass_vectors[:, retained] / np.sqrt(
        mass_eigenvalues[retained]
    )[None, :]
    operator = whitening.T @ stiffness_matrix @ whitening
    operator = 0.5 * (operator + operator.T)
    eigenvalues = np.linalg.eigvalsh(operator)
    negative_tolerance = 1e-9 * max(1.0, float(np.max(np.abs(eigenvalues))))
    if float(np.min(eigenvalues)) < -negative_tolerance:
        raise FloatingPointError("Ritz stiffness produced a significant negative eigenvalue")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    discarded_mass = mass_eigenvalues[~retained]
    smallest_retained_mass = float(mass_eigenvalues[retained][0])
    largest_discarded_mass = (
        float(discarded_mass[-1]) if discarded_mass.size else None
    )
    return {
        "eigenvalues": [float(value) for value in eigenvalues[:eigenvalue_count]],
        "spectral_gap": float(eigenvalues[0]),
        "retained_trial_rank": retained_rank,
        "smallest_retained_mass_eigenvalue": smallest_retained_mass,
        "smallest_retained_relative_mass_eigenvalue": (
            smallest_retained_mass / largest_mass
        ),
        "largest_discarded_mass_eigenvalue": largest_discarded_mass,
        "largest_discarded_relative_mass_eigenvalue": (
            largest_discarded_mass / largest_mass
            if largest_discarded_mass is not None
            else None
        ),
        "largest_mass_eigenvalue": largest_mass,
        "mass_condition_number": float(
            largest_mass / smallest_retained_mass
        ),
    }


def _weighted_quantile(values: Array, weights: Array, probability: float) -> float:
    samples = np.asarray(values, dtype=float).reshape(-1)
    normalized_weights = np.asarray(weights, dtype=float).reshape(-1)
    if samples.shape != normalized_weights.shape or not 0.0 <= probability <= 1.0:
        raise ValueError("weighted quantile inputs are inconsistent")
    order = np.argsort(samples)
    sorted_values = samples[order]
    cumulative = np.cumsum(normalized_weights[order])
    cumulative /= cumulative[-1]
    return float(np.interp(probability, cumulative, sorted_values))


def paired_metric_distortion(
    source_metrics: Array,
    target_metrics: Array,
    weights: Array,
) -> dict[str, Any]:
    """Coordinate-invariant pointwise distortion between two Hermitian metrics."""

    source = np.asarray(source_metrics, dtype=np.complex128)
    target = np.asarray(target_metrics, dtype=np.complex128)
    normalized_weights = np.asarray(weights, dtype=float).reshape(-1)
    if source.shape != target.shape or source.ndim != 3 or source.shape[1] != source.shape[2]:
        raise ValueError("paired metric arrays must have equal square matrix shapes")
    if normalized_weights.shape != (len(source),) or np.any(normalized_weights < 0):
        raise ValueError("paired metric weights have an invalid shape or sign")
    total_weight = float(np.sum(normalized_weights))
    if total_weight <= 0 or not np.isfinite(total_weight):
        raise ValueError("paired metric weights must have positive finite sum")
    normalized_weights = normalized_weights / total_weight

    source_eigenvalues = np.linalg.eigvalsh(source)
    target_eigenvalues = np.linalg.eigvalsh(target)
    if np.min(source_eigenvalues) <= 0 or np.min(target_eigenvalues) <= 0:
        raise FloatingPointError("paired metric distortion requires positive metrics")
    source_cholesky = np.linalg.cholesky(source)
    inverse_cholesky = np.linalg.inv(source_cholesky)
    whitened = np.einsum(
        "nia,nab,njb->nij",
        inverse_cholesky,
        target,
        np.conjugate(inverse_cholesky),
        optimize=True,
    )
    whitened = 0.5 * (whitened + np.conjugate(np.swapaxes(whitened, 1, 2)))
    generalized_eigenvalues = np.linalg.eigvalsh(whitened)
    if np.min(generalized_eigenvalues) <= 0:
        raise FloatingPointError("metric-pair generalized eigenvalues are not positive")
    log_stretches = np.log(generalized_eigenvalues)
    log_volume_ratio = np.sum(log_stretches, axis=1)
    mean_log_stretch = np.mean(log_stretches, axis=1, keepdims=True)
    affine_distance = np.sqrt(np.sum(log_stretches**2, axis=1))
    shape_distance = np.sqrt(np.sum((log_stretches - mean_log_stretch) ** 2, axis=1))

    def weighted_mean(values: Array) -> float:
        return float(np.sum(normalized_weights * np.asarray(values, dtype=float)))

    return {
        "mean_affine_invariant_distance": weighted_mean(affine_distance),
        "rms_affine_invariant_distance": float(
            np.sqrt(weighted_mean(affine_distance**2))
        ),
        "median_affine_invariant_distance": _weighted_quantile(
            affine_distance, normalized_weights, 0.5
        ),
        "p95_affine_invariant_distance": _weighted_quantile(
            affine_distance, normalized_weights, 0.95
        ),
        "p99_affine_invariant_distance": _weighted_quantile(
            affine_distance, normalized_weights, 0.99
        ),
        "mean_shape_distance": weighted_mean(shape_distance),
        "rms_shape_distance": float(np.sqrt(weighted_mean(shape_distance**2))),
        "mean_log_volume_ratio": weighted_mean(log_volume_ratio),
        "rms_log_volume_ratio": float(
            np.sqrt(weighted_mean(log_volume_ratio**2))
        ),
        "minimum_generalized_eigenvalue": float(np.min(generalized_eigenvalues)),
        "maximum_generalized_eigenvalue": float(np.max(generalized_eigenvalues)),
        "comparison_effective_sample_size": effective_sample_size(normalized_weights),
    }


def estimate_scalar_laplacian_ritz(
    values: Array,
    holomorphic_gradients: Array,
    metrics: Array,
    weights: Array,
    *,
    eigenvalue_count: int,
    mass_relative_threshold: float = 1e-10,
    mass_threshold_sweep: Sequence[float] = (),
    cluster_ids: Array | None = None,
) -> dict[str, Any]:
    """Estimate positive eigenvalues for Delta=-2 g^(i,jbar) d_i d_jbar."""

    feature_values = np.asarray(values, dtype=float)
    gradients = np.asarray(holomorphic_gradients, dtype=np.complex128)
    metric_values = np.asarray(metrics, dtype=np.complex128)
    normalized_weights = np.asarray(weights, dtype=float).reshape(-1)
    if feature_values.ndim != 2 or gradients.ndim != 3 or metric_values.ndim != 3:
        raise ValueError("Ritz inputs have invalid dimensions")
    point_count, feature_count = feature_values.shape
    if gradients.shape[:2] != (point_count, feature_count):
        raise ValueError("Ritz feature and gradient shapes differ")
    if metric_values.shape != (
        point_count,
        gradients.shape[2],
        gradients.shape[2],
    ):
        raise ValueError("Ritz metric shape is inconsistent with gradients")
    if normalized_weights.shape != (point_count,) or np.any(normalized_weights < 0):
        raise ValueError("Ritz weights have an invalid shape or sign")
    if not np.isclose(float(np.sum(normalized_weights)), 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError("Ritz weights must sum to one")
    if eigenvalue_count <= 0 or mass_relative_threshold <= 0:
        raise ValueError("eigenvalue count and mass threshold must be positive")
    sweep_thresholds = tuple(float(value) for value in mass_threshold_sweep)
    if len(set(sweep_thresholds)) != len(sweep_thresholds) or any(
        not 0.0 < value < 1.0 for value in sweep_thresholds
    ):
        raise ValueError(
            "mass-threshold sweep values must be distinct and lie between zero and one"
        )
    if cluster_ids is None:
        identifiers = np.arange(point_count, dtype=np.int64)
    else:
        identifiers = np.asarray(cluster_ids, dtype=np.int64).reshape(-1)
        if identifiers.shape != (point_count,):
            raise ValueError("sampling cluster ids must match Ritz point count")

    means = normalized_weights @ feature_values
    centered = feature_values - means[None, :]
    mass = centered.T @ (normalized_weights[:, None] * centered)
    mass = 0.5 * (mass + mass.T)

    metric_eigenvalues = np.linalg.eigvalsh(metric_values)
    if np.min(metric_eigenvalues) <= 0:
        raise FloatingPointError("scalar spectrum requires positive-definite metrics")
    inverse_metrics = np.linalg.inv(metric_values)
    cholesky = np.linalg.cholesky(inverse_metrics)
    contracted = np.einsum(
        "nfi,nij->nfj",
        gradients,
        cholesky,
        optimize=True,
    )
    stiffness = 2.0 * np.real(
        np.einsum(
            "nfi,ngi,n->fg",
            contracted,
            np.conjugate(contracted),
            normalized_weights,
            optimize=True,
        )
    )
    stiffness = 0.5 * (stiffness + stiffness.T)

    solved = _solve_ritz_matrices(
        mass,
        stiffness,
        eigenvalue_count=eigenvalue_count,
        mass_relative_threshold=mass_relative_threshold,
    )
    cluster_statistics = sampling_cluster_statistics(
        normalized_weights,
        identifiers,
    )
    output = {
        **solved,
        "raw_feature_count": feature_count,
        "mass_relative_threshold": mass_relative_threshold,
        "minimum_metric_eigenvalue": float(np.min(metric_eigenvalues)),
        "integration_effective_sample_size": cluster_statistics[
            "point_effective_sample_size"
        ],
        "integration_cluster_effective_sample_size": cluster_statistics[
            "cluster_weight_effective_sample_size"
        ],
        "independent_sampling_cluster_count": cluster_statistics[
            "sampling_cluster_count"
        ],
        "minimum_points_per_sampling_cluster": cluster_statistics[
            "minimum_points_per_sampling_cluster"
        ],
        "maximum_points_per_sampling_cluster": cluster_statistics[
            "maximum_points_per_sampling_cluster"
        ],
    }
    if sweep_thresholds:
        output["mass_threshold_sweep"] = [
            {
                "mass_relative_threshold": threshold,
                **_solve_ritz_matrices(
                    mass,
                    stiffness,
                    eigenvalue_count=eigenvalue_count,
                    mass_relative_threshold=threshold,
                ),
            }
            for threshold in sweep_thresholds
        ]
    return output


def cluster_delete_group_jackknife_scalar_ritz(
    values: Array,
    holomorphic_gradients: Array,
    metrics: Array,
    weights: Array,
    cluster_ids: Array,
    *,
    eigenvalue_count: int,
    mass_relative_threshold: float = 1e-10,
    group_count: int = 16,
) -> dict[str, Any]:
    """Delete groups of independent fibres and jackknife low Ritz values."""

    if group_count < 2:
        raise ValueError("cluster jackknife requires at least two groups")
    feature_values = np.asarray(values, dtype=float)
    gradients = np.asarray(holomorphic_gradients, dtype=np.complex128)
    metric_values = np.asarray(metrics, dtype=np.complex128)
    normalized_weights = np.asarray(weights, dtype=float).reshape(-1)
    identifiers = np.asarray(cluster_ids, dtype=np.int64).reshape(-1)
    point_count, feature_count = feature_values.shape
    if gradients.shape[:2] != (point_count, feature_count):
        raise ValueError("cluster jackknife feature and gradient shapes differ")
    if metric_values.shape != (point_count, gradients.shape[2], gradients.shape[2]):
        raise ValueError("cluster jackknife metric shape is inconsistent")
    if normalized_weights.shape != (point_count,) or identifiers.shape != (point_count,):
        raise ValueError("cluster jackknife weights and ids must match points")
    if not np.isclose(float(np.sum(normalized_weights)), 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError("cluster jackknife weights must sum to one")

    unique_clusters, inverse_clusters = np.unique(identifiers, return_inverse=True)
    if len(unique_clusters) < 2 * group_count:
        raise ValueError("cluster jackknife needs at least two clusters per group")
    point_groups = np.arange(len(unique_clusters), dtype=np.int64)[
        inverse_clusters
    ] % group_count

    inverse_metrics = np.linalg.inv(metric_values)
    cholesky = np.linalg.cholesky(inverse_metrics)
    contracted = np.einsum(
        "nfi,nij->nfj",
        gradients,
        cholesky,
        optimize=True,
    )
    total_first = normalized_weights @ feature_values
    total_second = feature_values.T @ (
        normalized_weights[:, None] * feature_values
    )
    total_stiffness = 2.0 * np.real(
        np.einsum(
            "nfi,ngi,n->fg",
            contracted,
            np.conjugate(contracted),
            normalized_weights,
            optimize=True,
        )
    )
    full_mass = total_second - np.outer(total_first, total_first)
    full = _solve_ritz_matrices(
        full_mass,
        total_stiffness,
        eigenvalue_count=eigenvalue_count,
        mass_relative_threshold=mass_relative_threshold,
    )

    leave_group_values = []
    group_point_counts = []
    group_cluster_counts = []
    group_weight_masses = []
    for group in range(group_count):
        selected = point_groups == group
        group_weights = normalized_weights[selected]
        deleted_weight = float(np.sum(group_weights))
        remaining_weight = 1.0 - deleted_weight
        if remaining_weight <= 0:
            raise FloatingPointError("cluster jackknife deleted all integration weight")
        group_first = group_weights @ feature_values[selected]
        group_second = feature_values[selected].T @ (
            group_weights[:, None] * feature_values[selected]
        )
        group_stiffness = 2.0 * np.real(
            np.einsum(
                "nfi,ngi,n->fg",
                contracted[selected],
                np.conjugate(contracted[selected]),
                group_weights,
                optimize=True,
            )
        )
        remaining_first = (total_first - group_first) / remaining_weight
        remaining_second = (total_second - group_second) / remaining_weight
        remaining_mass = remaining_second - np.outer(
            remaining_first,
            remaining_first,
        )
        remaining_stiffness = (
            total_stiffness - group_stiffness
        ) / remaining_weight
        solved = _solve_ritz_matrices(
            remaining_mass,
            remaining_stiffness,
            eigenvalue_count=eigenvalue_count,
            mass_relative_threshold=mass_relative_threshold,
        )
        leave_group_values.append(solved["eigenvalues"])
        group_point_counts.append(int(np.sum(selected)))
        group_cluster_counts.append(int(len(np.unique(inverse_clusters[selected]))))
        group_weight_masses.append(deleted_weight)

    point_estimate = np.asarray(full["eigenvalues"], dtype=float)
    leave_values = np.asarray(leave_group_values, dtype=float)
    pseudo_values = (
        group_count * point_estimate[None, :]
        - (group_count - 1) * leave_values
    )
    bias_corrected = np.mean(pseudo_values, axis=0)
    standard_errors = np.std(pseudo_values, axis=0, ddof=1) / np.sqrt(group_count)
    intervals = np.column_stack(
        (bias_corrected - 1.96 * standard_errors, bias_corrected + 1.96 * standard_errors)
    )
    return {
        "method": "deterministic_delete_group_cluster_jackknife",
        "group_count": int(group_count),
        "independent_sampling_cluster_count": int(len(unique_clusters)),
        "point_estimate": point_estimate.tolist(),
        "bias_corrected_eigenvalues": bias_corrected.tolist(),
        "standard_errors": standard_errors.tolist(),
        "normal_95_percent_intervals": intervals.tolist(),
        "leave_one_group_eigenvalues": leave_values.tolist(),
        "group_point_counts": group_point_counts,
        "group_cluster_counts": group_cluster_counts,
        "group_weight_masses": group_weight_masses,
    }


def _aggregate_mass_threshold_sweep(
    rows: list[dict[str, Any]],
    eigenvalue_count: int,
) -> dict[str, Any]:
    sweeps = [row.get("mass_threshold_sweep") for row in rows]
    if not any(sweeps):
        return {}
    if not all(sweeps):
        raise ValueError("mass-threshold sweep is missing for some spectrum seeds")
    thresholds = [
        float(entry["mass_relative_threshold"]) for entry in sweeps[0]
    ]
    if any(
        [float(entry["mass_relative_threshold"]) for entry in sweep]
        != thresholds
        for sweep in sweeps[1:]
    ):
        raise ValueError("mass-threshold sweep grids differ between seeds")

    primary_thresholds = {
        float(row["mass_relative_threshold"]) for row in rows
    }
    if len(primary_thresholds) != 1:
        raise ValueError("primary mass threshold differs between seeds")
    primary_threshold = primary_thresholds.pop()
    if primary_threshold not in thresholds:
        raise ValueError("mass-threshold sweep must include the primary threshold")

    aggregate_rows = []
    mean_vectors = []
    for index, threshold in enumerate(thresholds):
        seed_entries = [sweep[index] for sweep in sweeps]
        eigenvalue_arrays = np.asarray(
            [entry["eigenvalues"] for entry in seed_entries],
            dtype=float,
        )
        means = np.mean(eigenvalue_arrays, axis=0)
        mean_vectors.append(means)
        discarded = [
            float(entry["largest_discarded_relative_mass_eigenvalue"])
            for entry in seed_entries
            if entry["largest_discarded_relative_mass_eigenvalue"] is not None
        ]
        aggregate_rows.append(
            {
                "mass_relative_threshold": threshold,
                "retained_trial_ranks": sorted(
                    {int(entry["retained_trial_rank"]) for entry in seed_entries}
                ),
                "minimum_smallest_retained_relative_mass_eigenvalue": float(
                    min(
                        entry["smallest_retained_relative_mass_eigenvalue"]
                        for entry in seed_entries
                    )
                ),
                "maximum_largest_discarded_relative_mass_eigenvalue": (
                    float(max(discarded)) if discarded else None
                ),
                "eigenvalues": [
                    {
                        "index": mode + 1,
                        "mean": float(means[mode]),
                        "95_percent_ci": confidence_interval(
                            eigenvalue_arrays[:, mode].tolist()
                        ),
                    }
                    for mode in range(eigenvalue_count)
                ],
            }
        )

    reference_index = thresholds.index(primary_threshold)
    reference = mean_vectors[reference_index]
    maximum_relative_change = 0.0
    for aggregate, means in zip(aggregate_rows, mean_vectors, strict=True):
        relative = (means - reference) / reference
        aggregate["first_three_relative_changes_from_primary"] = [
            float(value) for value in relative[:3]
        ]
        maximum_relative_change = max(
            maximum_relative_change,
            float(np.max(np.abs(relative[:3]))),
        )
    return {
        "mass_threshold_sweep_primary_threshold": primary_threshold,
        "mass_threshold_sweep": aggregate_rows,
        "maximum_absolute_first_three_threshold_relative_change": (
            maximum_relative_change
        ),
    }


def _aggregate_spectra(rows: list[dict[str, Any]], eigenvalue_count: int) -> dict[str, Any]:
    arrays = np.asarray([row["eigenvalues"] for row in rows], dtype=float)
    entries = []
    for index in range(eigenvalue_count):
        values = arrays[:, index].tolist()
        entries.append(
            {
                "index": index + 1,
                "mean": float(np.mean(values)),
                "95_percent_ci": confidence_interval(values),
                "seed_standard_deviation": float(np.std(values, ddof=1))
                if len(values) > 1
                else 0.0,
            }
        )
    summary = {
        "eigenvalues": entries,
        "mean_spectral_gap": entries[0]["mean"],
        "spectral_gap_95_percent_ci": entries[0]["95_percent_ci"],
        "retained_trial_ranks": sorted(
            {int(row["retained_trial_rank"]) for row in rows}
        ),
        "minimum_metric_eigenvalue": float(
            min(row["minimum_metric_eigenvalue"] for row in rows)
        ),
        "minimum_integration_effective_sample_size": float(
            min(row["integration_effective_sample_size"] for row in rows)
        ),
        "minimum_effective_samples_per_retained_direction": float(
            min(
                row["integration_effective_sample_size"]
                / row["retained_trial_rank"]
                for row in rows
            )
        ),
        "minimum_integration_cluster_effective_sample_size": float(
            min(row["integration_cluster_effective_sample_size"] for row in rows)
        ),
        "minimum_effective_clusters_per_retained_direction": float(
            min(
                row["integration_cluster_effective_sample_size"]
                / row["retained_trial_rank"]
                for row in rows
            )
        ),
        "minimum_cluster_weight_ess_per_retained_rank": float(
            min(
                row["integration_cluster_effective_sample_size"]
                / row["retained_trial_rank"]
                for row in rows
            )
        ),
        "minimum_independent_sampling_cluster_count": int(
            min(row["independent_sampling_cluster_count"] for row in rows)
        ),
        "seeds": rows,
    }
    summary.update(_aggregate_mass_threshold_sweep(rows, eigenvalue_count))
    jackknife_rows = [
        row["cluster_delete_group_jackknife"]
        for row in rows
        if "cluster_delete_group_jackknife" in row
    ]
    if jackknife_rows:
        if len(jackknife_rows) != len(rows):
            raise ValueError("cluster jackknife results are missing for some seeds")
        jackknife_count = min(
            len(row["bias_corrected_eigenvalues"]) for row in jackknife_rows
        )
        jackknife_entries = []
        maximum_relative_standard_error = 0.0
        for index in range(jackknife_count):
            corrected = [
                float(row["bias_corrected_eigenvalues"][index])
                for row in jackknife_rows
            ]
            standard_errors = [
                float(row["standard_errors"][index]) for row in jackknife_rows
            ]
            point_estimates = [
                float(row["point_estimate"][index]) for row in jackknife_rows
            ]
            relative_errors = [
                error / max(abs(value), 1e-15)
                for value, error in zip(
                    point_estimates,
                    standard_errors,
                    strict=True,
                )
            ]
            maximum_relative_standard_error = max(
                maximum_relative_standard_error,
                max(relative_errors),
            )
            jackknife_entries.append(
                {
                    "index": index + 1,
                    "mean_bias_corrected_eigenvalue": float(np.mean(corrected)),
                    "bias_corrected_95_percent_ci_across_seeds": confidence_interval(
                        corrected
                    ),
                    "mean_within_seed_standard_error": float(
                        np.mean(standard_errors)
                    ),
                    "maximum_within_seed_standard_error": float(
                        np.max(standard_errors)
                    ),
                    "maximum_within_seed_relative_standard_error": float(
                        np.max(relative_errors)
                    ),
                }
            )
        summary["cluster_delete_group_jackknife"] = {
            "method": "deterministic_delete_group_cluster_jackknife",
            "group_counts": sorted(
                {int(row["group_count"]) for row in jackknife_rows}
            ),
            "minimum_independent_sampling_cluster_count": int(
                min(
                    row["independent_sampling_cluster_count"]
                    for row in jackknife_rows
                )
            ),
            "eigenvalues": jackknife_entries,
            "maximum_within_seed_relative_standard_error": float(
                maximum_relative_standard_error
            ),
        }
    return summary


def _paired_spectrum_comparison(
    source_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    eigenvalue_count: int,
) -> dict[str, Any]:
    source = {
        int(row["seed"]): np.asarray(row["eigenvalues"], dtype=float)
        for row in source_rows
    }
    target = {
        int(row["seed"]): np.asarray(row["eigenvalues"], dtype=float)
        for row in target_rows
    }
    if source.keys() != target.keys():
        raise ValueError("paired spectrum seed sets differ")
    seeds = sorted(source)
    entries = []
    for index in range(eigenvalue_count):
        source_values = np.asarray([source[seed][index] for seed in seeds], dtype=float)
        target_values = np.asarray([target[seed][index] for seed in seeds], dtype=float)
        differences = target_values - source_values
        relative = differences / source_values
        entries.append(
            {
                "index": index + 1,
                "source_mean": float(np.mean(source_values)),
                "target_mean": float(np.mean(target_values)),
                "mean_difference": float(np.mean(differences)),
                "difference_95_percent_ci": confidence_interval(differences.tolist()),
                "mean_relative_change": float(np.mean(relative)),
                "relative_change_95_percent_ci": confidence_interval(relative.tolist()),
            }
        )
    return {
        "seed_count": len(seeds),
        "eigenvalues": entries,
    }


def _aggregate_metric_distortions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate_keys = (
        "mean_affine_invariant_distance",
        "rms_affine_invariant_distance",
        "median_affine_invariant_distance",
        "p95_affine_invariant_distance",
        "p99_affine_invariant_distance",
        "mean_shape_distance",
        "rms_shape_distance",
        "mean_log_volume_ratio",
        "rms_log_volume_ratio",
    )
    output: dict[str, Any] = {"seeds": rows}
    for key in aggregate_keys:
        values = [float(row[key]) for row in rows]
        output[f"mean_seed_{key}"] = float(np.mean(values))
        output[f"{key}_95_percent_ci"] = confidence_interval(values)
    output["minimum_generalized_eigenvalue"] = float(
        min(row["minimum_generalized_eigenvalue"] for row in rows)
    )
    output["maximum_generalized_eigenvalue"] = float(
        max(row["maximum_generalized_eigenvalue"] for row in rows)
    )
    output["minimum_comparison_effective_sample_size"] = float(
        min(row["comparison_effective_sample_size"] for row in rows)
    )
    return output


def merge_scalar_laplacian_audits(
    summaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Merge disjoint one-or-more-seed audits from their raw seed rows."""

    if not summaries:
        raise ValueError("at least one scalar audit is required for merging")
    base = copy.deepcopy(summaries[0])
    common_keys = (
        "schema_version",
        "adapter",
        "model_seed",
        "exact_model",
        "points_per_seed",
        "eigenvalue_count",
        "mass_relative_threshold",
        "mass_threshold_sweep",
        "integration_measure",
        "trial_feature_configuration",
        "minimum_effective_sample_size_required",
        "minimum_ess_per_retained_rank_required",
        "minimum_cluster_ess_per_retained_rank_required",
        "cluster_jackknife_configuration",
    )
    for summary in summaries[1:]:
        for key in common_keys:
            if summary.get(key) != base.get(key):
                raise ValueError(f"scalar audit merge mismatch for {key}")

    seeds = []
    sampling_diagnostics = []
    for summary in summaries:
        seeds.extend(int(seed) for seed in summary["seeds"])
        sampling_diagnostics.extend(summary.get("sampling_diagnostics", []))
    if len(set(seeds)) != len(seeds):
        raise ValueError("scalar audit merge seed sets overlap")
    seeds = sorted(seeds)

    artifact_templates = {row["key"]: row for row in base["artifacts"]}
    artifact_order = [row["key"] for row in base["artifacts"]]
    levels = [
        int(row["level"])
        for row in base["artifacts"][0]["trial_levels"]
    ]
    rows: dict[str, dict[int, list[dict[str, Any]]]] = {
        key: {level: [] for level in levels} for key in artifact_order
    }
    raw_feature_counts: dict[int, int] = {}
    for summary in summaries:
        current = {row["key"]: row for row in summary["artifacts"]}
        if list(current) != artifact_order:
            raise ValueError("scalar audit artifact ordering differs")
        for key in artifact_order:
            current_levels = {
                int(row["level"]): row for row in current[key]["trial_levels"]
            }
            if sorted(current_levels) != levels:
                raise ValueError("scalar audit trial levels differ")
            for level in levels:
                raw_count = int(current_levels[level]["raw_feature_count"])
                raw_feature_counts.setdefault(level, raw_count)
                if raw_feature_counts[level] != raw_count:
                    raise ValueError("scalar audit raw feature counts differ")
                rows[key][level].extend(current_levels[level]["seeds"])

    artifact_summaries = []
    all_ess = []
    all_ess_rank_ratios = []
    all_cluster_ess_rank_ratios = []
    all_cluster_jackknife_relative_errors = []
    stable_ranks = True
    eigenvalue_count = int(base["eigenvalue_count"])
    for key in artifact_order:
        level_summaries = []
        for level in levels:
            aggregate = _aggregate_spectra(rows[key][level], eigenvalue_count)
            all_ess.append(aggregate["minimum_integration_effective_sample_size"])
            all_ess_rank_ratios.append(
                aggregate["minimum_effective_samples_per_retained_direction"]
            )
            all_cluster_ess_rank_ratios.append(
                aggregate["minimum_effective_clusters_per_retained_direction"]
            )
            stable_ranks &= len(aggregate["retained_trial_ranks"]) == 1
            if "cluster_delete_group_jackknife" in aggregate:
                all_cluster_jackknife_relative_errors.append(
                    aggregate["cluster_delete_group_jackknife"][
                        "maximum_within_seed_relative_standard_error"
                    ]
                )
            level_summaries.append(
                {
                    "level": level,
                    "raw_feature_count": raw_feature_counts[level],
                    **aggregate,
                }
            )
        artifact_summary = {
            "key": key,
            "path": artifact_templates[key]["path"],
            "degree": artifact_templates[key]["degree"],
            "trial_levels": level_summaries,
        }
        comparisons = []
        for source_level, target_level in zip(levels[:-1], levels[1:]):
            comparisons.append(
                {
                    "source_level": source_level,
                    "target_level": target_level,
                    **_paired_spectrum_comparison(
                        rows[key][source_level],
                        rows[key][target_level],
                        eigenvalue_count,
                    ),
                }
            )
        if comparisons:
            artifact_summary["trial_level_comparisons"] = comparisons
        if 1 in levels and 2 in levels:
            artifact_summary["linear_to_quadratic_trial_comparison"] = (
                _paired_spectrum_comparison(
                    rows[key][1],
                    rows[key][2],
                    eigenvalue_count,
                )
            )
        if 2 in levels and 3 in levels:
            artifact_summary["quadratic_to_cubic_trial_comparison"] = (
                _paired_spectrum_comparison(
                    rows[key][2],
                    rows[key][3],
                    eigenvalue_count,
                )
            )
        artifact_summaries.append(artifact_summary)

    artifact_pairwise_comparisons = []
    for source_key, target_key in zip(artifact_order[:-1], artifact_order[1:]):
        for level in levels:
            artifact_pairwise_comparisons.append(
                {
                    "source_artifact": source_key,
                    "target_artifact": target_key,
                    "trial_level": level,
                    **_paired_spectrum_comparison(
                        rows[source_key][level],
                        rows[target_key][level],
                        eigenvalue_count,
                    ),
                }
            )

    metric_pairwise_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for summary in summaries:
        for comparison in summary["metric_pairwise_comparisons"]:
            pair = (
                comparison["source_artifact"],
                comparison["target_artifact"],
            )
            metric_pairwise_rows.setdefault(pair, []).extend(comparison["seeds"])
    metric_pairwise_comparisons = [
        {
            "source_artifact": source,
            "target_artifact": target,
            **_aggregate_metric_distortions(metric_pairwise_rows[(source, target)]),
        }
        for source, target in zip(artifact_order[:-1], artifact_order[1:])
    ]

    nested_checks = [
        row
        for summary in summaries
        for row in summary["nested_trial_checks"]
    ]
    finite_positive = all(
        entry["mean"] > 0 and np.isfinite(entry["mean"])
        for artifact in artifact_summaries
        for level in artifact["trial_levels"]
        for entry in level["eigenvalues"]
    )
    gates = {
        "finite_positive_spectra": bool(finite_positive),
        "stable_trial_rank_across_seeds": bool(stable_ranks),
        "nested_ritz_monotonicity": bool(
            not nested_checks or all(row["passed"] for row in nested_checks)
        ),
        "integration_effective_sample_size": bool(
            all(
                value >= base["minimum_effective_sample_size_required"]
                for value in all_ess
            )
        ),
        "effective_samples_per_retained_direction": bool(
            all(
                value >= base["minimum_ess_per_retained_rank_required"]
                for value in all_ess_rank_ratios
            )
        ),
        "sampling_cluster_completeness": bool(
            all(
                row.get("all_returned_clusters_complete", False)
                for row in sampling_diagnostics
            )
        ),
        "cluster_effective_samples_per_retained_direction": bool(
            all(
                value >= base["minimum_cluster_ess_per_retained_rank_required"]
                for value in all_cluster_ess_rank_ratios
            )
        ),
        "finite_positive_metric_pair_distortion": bool(
            all(
                np.isfinite(row["mean_seed_rms_affine_invariant_distance"])
                and row["minimum_generalized_eigenvalue"] > 0
                for row in metric_pairwise_comparisons
            )
        ),
    }
    jackknife_config = base["cluster_jackknife_configuration"]
    if int(jackknife_config["group_count"]):
        gates["finite_cluster_jackknife"] = bool(
            all_cluster_jackknife_relative_errors
            and all(
                np.isfinite(value) and value >= 0
                for value in all_cluster_jackknife_relative_errors
            )
        )
    maximum_jackknife_error = jackknife_config.get(
        "maximum_relative_standard_error"
    )
    if maximum_jackknife_error is not None:
        gates["cluster_jackknife_relative_standard_error"] = bool(
            all_cluster_jackknife_relative_errors
            and all(
                value <= maximum_jackknife_error
                for value in all_cluster_jackknife_relative_errors
            )
        )

    base["seeds"] = seeds
    base["sampling_diagnostics"] = sorted(
        sampling_diagnostics,
        key=lambda row: int(row["seed"]),
    )
    base["artifacts"] = artifact_summaries
    base["artifact_pairwise_comparisons"] = artifact_pairwise_comparisons
    base["metric_pairwise_comparisons"] = metric_pairwise_comparisons
    base["nested_trial_checks"] = nested_checks
    base["gates"] = gates
    base["success"] = bool(all(gates.values()))
    base["timing_seconds"] = {
        "sampling_cpu_sum": float(
            sum(row["timing_seconds"]["sampling"] for row in summaries)
        ),
        "metric_and_spectrum_cpu_sum": float(
            sum(
                row["timing_seconds"]["metric_and_spectrum"]
                for row in summaries
            )
        ),
        "worker_total_cpu_sum": float(
            sum(row["timing_seconds"]["total"] for row in summaries)
        ),
    }
    base.pop("specification", None)
    return base


def run_scalar_laplacian_audit(
    adapter: GCICYAdapter,
    *,
    model_seed: int,
    exact_model: bool,
    artifact_paths: Sequence[Path],
    seeds: Sequence[int],
    points_per_seed: int,
    trial_levels: Sequence[int] = (1, 2),
    eigenvalue_count: int = 12,
    mass_relative_threshold: float = 1e-10,
    mass_threshold_sweep: Sequence[float] = (),
    minimum_effective_sample_size: float = 1.0,
    integration_measure: str = "metric_volume",
    cubic_feature_count: int = 256,
    cubic_feature_seed: int = 314159,
    minimum_ess_per_retained_rank: float = 1.0,
    minimum_cluster_ess_per_retained_rank: float = 1.0,
    cluster_jackknife_groups: int = 0,
    cluster_jackknife_level: int | None = None,
    cluster_jackknife_eigenvalue_count: int = 3,
    maximum_cluster_jackknife_relative_standard_error: float | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    """Run fresh-seed scalar-Laplacian Ritz estimates for saved metrics."""

    started = time.perf_counter()
    levels = tuple(int(level) for level in trial_levels)
    if not levels or any(level not in (1, 2, 3) for level in levels):
        raise ValueError("trial levels must be a non-empty subset of (1, 2, 3)")
    if tuple(sorted(set(levels))) != levels:
        raise ValueError("trial levels must be distinct and increasing")
    if not artifact_paths or not seeds or points_per_seed <= 0:
        raise ValueError("artifacts, seeds, and points per seed must be non-empty")
    if integration_measure not in {"metric_volume", "holomorphic_volume"}:
        raise ValueError(
            "integration_measure must be metric_volume or holomorphic_volume"
        )
    if 3 in levels and cubic_feature_count <= 0:
        raise ValueError("level 3 requires a positive cubic feature count")
    if minimum_ess_per_retained_rank <= 0:
        raise ValueError("minimum ESS per retained rank must be positive")
    sweep_thresholds = tuple(float(value) for value in mass_threshold_sweep)
    if len(set(sweep_thresholds)) != len(sweep_thresholds) or any(
        not 0.0 < value < 1.0 for value in sweep_thresholds
    ):
        raise ValueError(
            "mass-threshold sweep values must be distinct and lie between zero and one"
        )
    if sweep_thresholds and mass_relative_threshold not in sweep_thresholds:
        raise ValueError("mass-threshold sweep must include the primary threshold")
    if minimum_cluster_ess_per_retained_rank <= 0:
        raise ValueError("minimum cluster ESS per retained rank must be positive")
    if cluster_jackknife_groups == 1 or cluster_jackknife_groups < 0:
        raise ValueError("cluster jackknife groups must be zero or at least two")
    selected_jackknife_level = (
        max(levels) if cluster_jackknife_level is None else int(cluster_jackknife_level)
    )
    if cluster_jackknife_groups and selected_jackknife_level not in levels:
        raise ValueError("cluster jackknife level must be an enabled trial level")
    if not 0 < cluster_jackknife_eigenvalue_count <= eigenvalue_count:
        raise ValueError("cluster jackknife eigenvalue count is invalid")
    if (
        maximum_cluster_jackknife_relative_standard_error is not None
        and maximum_cluster_jackknife_relative_standard_error <= 0
    ):
        raise ValueError("cluster jackknife relative-error gate must be positive")

    model = adapter.make_model(model_seed, exact=exact_model)
    artifacts = [adapter.load_h_artifact(Path(path), model) for path in artifact_paths]
    rows: dict[str, dict[int, list[dict[str, Any]]]] = {
        artifact.key: {level: [] for level in levels} for artifact in artifacts
    }
    feature_counts: dict[int, int] = {}
    nested_checks: list[dict[str, Any]] = []
    metric_distortion_rows: dict[tuple[str, str], list[dict[str, Any]]] = {
        (source.key, target.key): []
        for source, target in zip(artifacts[:-1], artifacts[1:])
    }
    sampling_seconds = 0.0
    evaluation_seconds = 0.0
    sampling_diagnostics: list[dict[str, Any]] = []

    for seed in seeds:
        seed_started = time.perf_counter()
        if progress:
            print(
                f"scalar spectrum seed={int(seed)}: sampling {points_per_seed} points",
                flush=True,
            )
        sample_started = time.perf_counter()
        points, diagnostics = adapter.sample_points_with_diagnostics(
            model,
            points_per_seed,
            seed=int(seed),
        )
        cluster_ids = adapter.sampling_cluster_ids(points)
        if np.asarray(cluster_ids).shape != (len(points),):
            raise RuntimeError("adapter sampling cluster ids do not match points")
        sampling_diagnostics.append({"seed": int(seed), **diagnostics})
        sampling_seconds += time.perf_counter() - sample_started
        trial_data = [
            adapter.scalar_laplacian_trial_data(
                point,
                level=max(levels),
                cubic_feature_count=cubic_feature_count,
                cubic_feature_seed=cubic_feature_seed,
            )
            for point in points
        ]
        full_values = np.asarray([item.values for item in trial_data], dtype=float)
        full_gradients = np.asarray(
            [item.holomorphic_gradients for item in trial_data],
            dtype=np.complex128,
        )
        linear_count = trial_data[0].linear_feature_count
        quadratic_count = linear_count + linear_count * (linear_count + 1) // 2
        counts = {
            1: linear_count,
            2: quadratic_count,
            3: full_values.shape[1],
        }
        for level in levels:
            feature_counts.setdefault(level, counts[level])
            if feature_counts[level] != counts[level]:
                raise RuntimeError("scalar trial feature count changed between seeds")

        omega_weights = adapter.importance_weights(points)
        omega_weights = omega_weights / np.sum(omega_weights)
        metric_cache: dict[str, Array] = {}
        for artifact in artifacts:
            if progress:
                print(
                    f"scalar spectrum seed={int(seed)}: evaluating {artifact.key}",
                    flush=True,
                )
            evaluation_started = time.perf_counter()
            metrics = adapter.h_metrics(points, artifact)
            metric_cache[artifact.key] = metrics
            if integration_measure == "holomorphic_volume":
                weights = omega_weights
            else:
                weights, _ = metric_volume_weights(adapter, points, metrics)
            seed_spectra: dict[int, dict[str, Any]] = {}
            for level in levels:
                count = counts[level]
                estimate = estimate_scalar_laplacian_ritz(
                    full_values[:, :count],
                    full_gradients[:, :count],
                    metrics,
                    weights,
                    eigenvalue_count=eigenvalue_count,
                    mass_relative_threshold=mass_relative_threshold,
                    mass_threshold_sweep=sweep_thresholds,
                    cluster_ids=cluster_ids,
                )
                if (
                    cluster_jackknife_groups
                    and level == selected_jackknife_level
                ):
                    estimate["cluster_delete_group_jackknife"] = (
                        cluster_delete_group_jackknife_scalar_ritz(
                            full_values[:, :count],
                            full_gradients[:, :count],
                            metrics,
                            weights,
                            cluster_ids,
                            eigenvalue_count=cluster_jackknife_eigenvalue_count,
                            mass_relative_threshold=mass_relative_threshold,
                            group_count=cluster_jackknife_groups,
                        )
                    )
                row = {"seed": int(seed), **estimate}
                rows[artifact.key][level].append(row)
                seed_spectra[level] = estimate
            for source_level, target_level in zip(levels[:-1], levels[1:]):
                source_values = np.asarray(
                    seed_spectra[source_level]["eigenvalues"],
                    dtype=float,
                )
                target_values = np.asarray(
                    seed_spectra[target_level]["eigenvalues"],
                    dtype=float,
                )
                excess = target_values - source_values
                tolerance = 1e-7 * np.maximum(1.0, np.abs(source_values))
                check = {
                    "artifact": artifact.key,
                    "seed": int(seed),
                    "source_level": source_level,
                    "target_level": target_level,
                    "max_target_minus_source": float(np.max(excess)),
                    "passed": bool(np.all(excess <= tolerance)),
                }
                if source_level == 1 and target_level == 2:
                    check["max_quadratic_minus_linear"] = float(np.max(excess))
                nested_checks.append(
                    check
                )
            evaluation_seconds += time.perf_counter() - evaluation_started

        for source, target in zip(artifacts[:-1], artifacts[1:]):
            distortion = paired_metric_distortion(
                metric_cache[source.key],
                metric_cache[target.key],
                omega_weights,
            )
            metric_distortion_rows[(source.key, target.key)].append(
                {"seed": int(seed), **distortion}
            )
        if progress:
            print(
                f"scalar spectrum seed={int(seed)}: completed in "
                f"{time.perf_counter() - seed_started:.1f} s",
                flush=True,
            )

    artifact_summaries = []
    all_ess = []
    all_ess_rank_ratios = []
    all_cluster_ess = []
    all_cluster_ess_rank_ratios = []
    all_cluster_jackknife_relative_errors = []
    stable_ranks = True
    for artifact in artifacts:
        level_summaries = []
        for level in levels:
            summary = _aggregate_spectra(rows[artifact.key][level], eigenvalue_count)
            all_ess.append(summary["minimum_integration_effective_sample_size"])
            all_ess_rank_ratios.append(
                summary["minimum_effective_samples_per_retained_direction"]
            )
            all_cluster_ess.append(
                summary["minimum_integration_cluster_effective_sample_size"]
            )
            all_cluster_ess_rank_ratios.append(
                summary["minimum_effective_clusters_per_retained_direction"]
            )
            stable_ranks &= len(summary["retained_trial_ranks"]) == 1
            if "cluster_delete_group_jackknife" in summary:
                all_cluster_jackknife_relative_errors.append(
                    summary["cluster_delete_group_jackknife"][
                        "maximum_within_seed_relative_standard_error"
                    ]
                )
            level_summaries.append(
                {
                    "level": level,
                    "raw_feature_count": feature_counts[level],
                    **summary,
                }
            )
        artifact_summary = {
            "key": artifact.key,
            "path": str(artifact.path),
            "degree": list(artifact.degree),
            "trial_levels": level_summaries,
        }
        level_comparisons = []
        for source_level, target_level in zip(levels[:-1], levels[1:]):
            level_comparisons.append(
                {
                    "source_level": source_level,
                    "target_level": target_level,
                    **_paired_spectrum_comparison(
                        rows[artifact.key][source_level],
                        rows[artifact.key][target_level],
                        eigenvalue_count,
                    ),
                }
            )
        if level_comparisons:
            artifact_summary["trial_level_comparisons"] = level_comparisons
        if 1 in levels and 2 in levels:
            artifact_summary["linear_to_quadratic_trial_comparison"] = (
                _paired_spectrum_comparison(
                    rows[artifact.key][1],
                    rows[artifact.key][2],
                    eigenvalue_count,
                )
            )
        if 2 in levels and 3 in levels:
            artifact_summary["quadratic_to_cubic_trial_comparison"] = (
                _paired_spectrum_comparison(
                    rows[artifact.key][2],
                    rows[artifact.key][3],
                    eigenvalue_count,
                )
            )
        artifact_summaries.append(artifact_summary)

    artifact_pairwise_comparisons = []
    for source, target in zip(artifacts[:-1], artifacts[1:]):
        for level in levels:
            artifact_pairwise_comparisons.append(
                {
                    "source_artifact": source.key,
                    "target_artifact": target.key,
                    "trial_level": level,
                    **_paired_spectrum_comparison(
                        rows[source.key][level],
                        rows[target.key][level],
                        eigenvalue_count,
                    ),
                }
            )

    metric_pairwise_comparisons = []
    for source, target in zip(artifacts[:-1], artifacts[1:]):
        metric_pairwise_comparisons.append(
            {
                "source_artifact": source.key,
                "target_artifact": target.key,
                **_aggregate_metric_distortions(
                    metric_distortion_rows[(source.key, target.key)]
                ),
            }
        )

    finite_positive = all(
        entry["mean"] > 0 and np.isfinite(entry["mean"])
        for artifact in artifact_summaries
        for level in artifact["trial_levels"]
        for entry in level["eigenvalues"]
    )
    gates = {
        "finite_positive_spectra": bool(finite_positive),
        "stable_trial_rank_across_seeds": bool(stable_ranks),
        "nested_ritz_monotonicity": bool(
            not nested_checks or all(row["passed"] for row in nested_checks)
        ),
        "integration_effective_sample_size": bool(
            all(value >= minimum_effective_sample_size for value in all_ess)
        ),
        "effective_samples_per_retained_direction": bool(
            all(
                value >= minimum_ess_per_retained_rank
                for value in all_ess_rank_ratios
            )
        ),
        "sampling_cluster_completeness": bool(
            all(
                row.get("all_returned_clusters_complete", False)
                for row in sampling_diagnostics
            )
        ),
        "cluster_effective_samples_per_retained_direction": bool(
            all(
                value >= minimum_cluster_ess_per_retained_rank
                for value in all_cluster_ess_rank_ratios
            )
        ),
        "finite_positive_metric_pair_distortion": bool(
            all(
                np.isfinite(row["mean_seed_rms_affine_invariant_distance"])
                and row["minimum_generalized_eigenvalue"] > 0
                for row in metric_pairwise_comparisons
            )
        ),
    }
    if cluster_jackknife_groups:
        gates["finite_cluster_jackknife"] = bool(
            all_cluster_jackknife_relative_errors
            and all(
                np.isfinite(value) and value >= 0
                for value in all_cluster_jackknife_relative_errors
            )
        )
    if maximum_cluster_jackknife_relative_standard_error is not None:
        gates["cluster_jackknife_relative_standard_error"] = bool(
            all_cluster_jackknife_relative_errors
            and all(
                value <= maximum_cluster_jackknife_relative_standard_error
                for value in all_cluster_jackknife_relative_errors
            )
        )
    return {
        "schema_version": 1,
        "description": (
            "Fresh-seed scalar-Laplacian Rayleigh--Ritz audit using global "
            "product-projective moment-map trial functions."
        ),
        "laplacian_convention": "Delta = -2 g^{i jbar} partial_i partial_jbar",
        "interpretation": (
            (
                "Monte Carlo finite-trial-space Rayleigh--Ritz estimates for the "
                "Laplace--Beltrami operator of the approximate metric. "
                if integration_measure == "metric_volume"
                else
                "Monte Carlo finite-trial-space estimates for the Omega-weighted "
                "divergence operator; this is not the ordinary Laplace--Beltrami "
                "operator of the approximate metric unless its volume is exactly "
                "proportional to the holomorphic target volume. "
            )
            + "Monte Carlo matrix estimates do not inherit a strict variational "
            "upper-bound guarantee. Level 3 is a reproducible metric-independent "
            "random cubic enrichment, not the complete cubic polynomial space."
        ),
        "adapter": adapter.key,
        "configuration": adapter.configuration.to_dict(),
        "model": adapter.model_metadata(model),
        "model_seed": model_seed,
        "exact_model": exact_model,
        "seeds": [int(seed) for seed in seeds],
        "points_per_seed": points_per_seed,
        "eigenvalue_count": eigenvalue_count,
        "mass_relative_threshold": mass_relative_threshold,
        "mass_threshold_sweep": list(sweep_thresholds),
        "integration_measure": integration_measure,
        "trial_feature_configuration": {
            "level_3_kind": "random_rank_one_cubic_enrichment",
            "cubic_feature_count": int(cubic_feature_count),
            "cubic_feature_seed": int(cubic_feature_seed),
            "independent_of_samples_and_metric_artifacts": True,
        },
        "minimum_effective_sample_size_required": minimum_effective_sample_size,
        "minimum_ess_per_retained_rank_required": minimum_ess_per_retained_rank,
        "minimum_cluster_ess_per_retained_rank_required": (
            minimum_cluster_ess_per_retained_rank
        ),
        "cluster_jackknife_configuration": {
            "group_count": int(cluster_jackknife_groups),
            "trial_level": int(selected_jackknife_level),
            "eigenvalue_count": int(cluster_jackknife_eigenvalue_count),
            "maximum_relative_standard_error": (
                maximum_cluster_jackknife_relative_standard_error
            ),
        },
        "sampling_diagnostics": sampling_diagnostics,
        "artifacts": artifact_summaries,
        "artifact_pairwise_comparisons": artifact_pairwise_comparisons,
        "metric_pairwise_comparisons": metric_pairwise_comparisons,
        "nested_trial_checks": nested_checks,
        "gates": gates,
        "success": bool(all(gates.values())),
        "timing_seconds": {
            "sampling": sampling_seconds,
            "metric_and_spectrum": evaluation_seconds,
            "total": time.perf_counter() - started,
        },
    }
