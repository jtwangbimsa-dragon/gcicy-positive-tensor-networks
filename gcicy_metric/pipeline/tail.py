"""Bilateral Monge-Ampere tail statistics and paired cluster bootstrap."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .audit import (
    effective_sample_size,
    normalized_volume_ratios,
    weighted_quantile,
    weighted_upper_tail_mean,
)


COMPARISON_METRICS = (
    "sigma",
    "chi",
    "absolute_log_ratio_q999",
    "absolute_log_ratio_cvar_1pct",
    "upper_log_ratio_q999",
    "upper_log_ratio_cvar_1pct",
    "lower_log_ratio_q999",
    "lower_log_ratio_cvar_1pct",
    "normalized_ratio_above_3_weighted_mass",
    "normalized_ratio_below_one_third_weighted_mass",
)


def _validated_inputs(
    log_eta: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(log_eta, dtype=np.float64).reshape(-1)
    masses = np.asarray(weights, dtype=np.float64).reshape(-1)
    if values.shape != masses.shape or values.size == 0:
        raise ValueError("log_eta and weights must be non-empty aligned arrays")
    if (
        not np.all(np.isfinite(values))
        or not np.all(np.isfinite(masses))
        or np.any(masses < 0.0)
        or float(np.sum(masses)) <= 0.0
    ):
        raise ValueError("log_eta and weights must be finite with positive mass")
    return values, masses


def bilateral_tail_metrics(
    log_eta: np.ndarray,
    weights: np.ndarray,
    cluster_ids: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Compute bulk and bilateral tail diagnostics under importance weights."""

    values, masses = _validated_inputs(log_eta, weights)
    normalized_weights = masses / np.sum(masses)
    ratio, log_ratio = normalized_volume_ratios(values, masses)
    residual = 1.0 - ratio
    absolute_log = np.abs(log_ratio)
    upper_log = np.maximum(log_ratio, 0.0)
    lower_log = np.maximum(-log_ratio, 0.0)
    squared_energy = float(np.sum(normalized_weights * residual**2))

    output: dict[str, float | int] = {
        "point_count": int(values.size),
        "sigma": float(np.sum(normalized_weights * np.abs(residual))),
        "squared_energy": squared_energy,
        "chi": float(np.sqrt(squared_energy)),
        "absolute_log_ratio_q999": weighted_quantile(
            absolute_log, masses, 0.999
        ),
        "absolute_log_ratio_cvar_1pct": weighted_upper_tail_mean(
            absolute_log, masses, 0.01
        ),
        "upper_log_ratio_q999": weighted_quantile(upper_log, masses, 0.999),
        "upper_log_ratio_cvar_1pct": weighted_upper_tail_mean(
            upper_log, masses, 0.01
        ),
        "lower_log_ratio_q999": weighted_quantile(lower_log, masses, 0.999),
        "lower_log_ratio_cvar_1pct": weighted_upper_tail_mean(
            lower_log, masses, 0.01
        ),
        "normalized_ratio_min": float(np.min(ratio)),
        "normalized_ratio_max": float(np.max(ratio)),
        "maximum_absolute_log_ratio": float(np.max(absolute_log)),
        "normalized_ratio_above_3_weighted_mass": float(
            np.sum(normalized_weights * (ratio > 3.0))
        ),
        "normalized_ratio_below_one_third_weighted_mass": float(
            np.sum(normalized_weights * (ratio < (1.0 / 3.0)))
        ),
        "normalized_ratio_above_3_point_count": int(np.count_nonzero(ratio > 3.0)),
        "normalized_ratio_below_one_third_point_count": int(
            np.count_nonzero(ratio < (1.0 / 3.0))
        ),
        "importance_effective_sample_size": effective_sample_size(masses),
    }

    if cluster_ids is not None:
        clusters = np.asarray(cluster_ids).reshape(-1)
        if clusters.shape != values.shape:
            raise ValueError("cluster_ids must align with log_eta")
        _, inverse, counts = np.unique(
            clusters,
            return_inverse=True,
            return_counts=True,
        )
        cluster_masses = np.bincount(inverse, weights=masses)
        output.update(
            {
                "fibre_cluster_count": int(cluster_masses.size),
                "fibre_cluster_effective_sample_size": effective_sample_size(
                    cluster_masses
                ),
                "fibre_cluster_size_min": int(np.min(counts)),
                "fibre_cluster_size_max": int(np.max(counts)),
            }
        )
    return output


def _check_paired_arrays(
    baseline_log_eta: np.ndarray,
    candidate_log_eta: np.ndarray,
    weights: np.ndarray,
    cluster_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    baseline, masses = _validated_inputs(baseline_log_eta, weights)
    candidate, candidate_masses = _validated_inputs(candidate_log_eta, weights)
    clusters = np.asarray(cluster_ids).reshape(-1)
    if candidate.shape != baseline.shape or clusters.shape != baseline.shape:
        raise ValueError("paired arrays and cluster_ids must have equal shape")
    if not np.array_equal(masses, candidate_masses):
        raise ValueError("baseline and candidate must use identical weights")
    unique_clusters, inverse = np.unique(clusters, return_inverse=True)
    if unique_clusters.size < 2:
        raise ValueError("paired cluster bootstrap requires at least two clusters")
    return baseline, candidate, masses, unique_clusters, inverse


def paired_cluster_bootstrap(
    baseline_log_eta: np.ndarray,
    candidate_log_eta: np.ndarray,
    weights: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    replicates: int = 1000,
    seed: int = 20260728,
) -> dict[str, Any]:
    """Bootstrap paired model improvements by resampling complete fibres.

    Positive improvement means that the candidate has a lower error statistic
    than the baseline. Each bootstrap replicate independently renormalizes the
    Monge-Ampere ratio for both models.
    """

    if replicates <= 0:
        raise ValueError("replicates must be positive")
    baseline, candidate, masses, unique_clusters, inverse = _check_paired_arrays(
        baseline_log_eta,
        candidate_log_eta,
        weights,
        cluster_ids,
    )
    baseline_full = bilateral_tail_metrics(baseline, masses)
    candidate_full = bilateral_tail_metrics(candidate, masses)
    estimates = np.asarray(
        [
            float(baseline_full[name]) - float(candidate_full[name])
            for name in COMPARISON_METRICS
        ],
        dtype=np.float64,
    )

    rng = np.random.default_rng(seed)
    cluster_count = unique_clusters.size
    samples = np.empty((replicates, len(COMPARISON_METRICS)), dtype=np.float64)
    for replicate in range(replicates):
        draws = rng.integers(0, cluster_count, size=cluster_count)
        multiplicities = np.bincount(draws, minlength=cluster_count)
        bootstrap_weights = masses * multiplicities[inverse]
        baseline_metrics = bilateral_tail_metrics(baseline, bootstrap_weights)
        candidate_metrics = bilateral_tail_metrics(candidate, bootstrap_weights)
        samples[replicate] = [
            float(baseline_metrics[name]) - float(candidate_metrics[name])
            for name in COMPARISON_METRICS
        ]

    comparisons: dict[
        str,
        dict[str, float | list[float] | str | None],
    ] = {}
    for index, name in enumerate(COMPARISON_METRICS):
        values = samples[:, index]
        baseline_value = float(baseline_full[name])
        estimate = float(estimates[index])
        relative_improvement = (
            estimate / baseline_value if baseline_value != 0.0 else None
        )
        comparisons[name] = {
            "direction": "positive_is_candidate_improvement",
            "baseline": baseline_value,
            "candidate": float(candidate_full[name]),
            "improvement": estimate,
            "relative_improvement": relative_improvement,
            "bootstrap_mean_improvement": float(np.mean(values)),
            "bootstrap_standard_error": float(np.std(values, ddof=1))
            if replicates > 1
            else 0.0,
            "bootstrap_95pct_confidence_interval": [
                float(value) for value in np.quantile(values, [0.025, 0.975])
            ],
            "bootstrap_win_probability": float(np.mean(values > 0.0)),
        }

    return {
        "schema": "gcicy-paired-fibre-cluster-bootstrap-v1",
        "bootstrap_replicates": int(replicates),
        "bootstrap_seed": int(seed),
        "point_count": int(baseline.size),
        "fibre_cluster_count": int(cluster_count),
        "comparison_metrics": list(COMPARISON_METRICS),
        "comparisons": comparisons,
    }


def add_metric_geometry_evidence(
    metrics: dict[str, Any],
    minimum_eigenvalues: np.ndarray | None,
) -> dict[str, Any]:
    """Attach finite-sample positivity diagnostics when eigenvalues are saved."""

    output = dict(metrics)
    if minimum_eigenvalues is None:
        output["metric_geometry_evidence_available"] = False
        output["minimum_metric_eigenvalue"] = None
        output["nonpositive_metric_count"] = None
        return output
    values = np.asarray(minimum_eigenvalues, dtype=np.float64).reshape(-1)
    if values.size != int(output["point_count"]) or not np.all(np.isfinite(values)):
        raise ValueError("minimum_eigenvalues must be finite and point-aligned")
    output["metric_geometry_evidence_available"] = True
    output["minimum_metric_eigenvalue"] = float(np.min(values))
    output["nonpositive_metric_count"] = int(np.count_nonzero(values <= 0.0))
    return output


def comparison_gate(
    comparison: Mapping[str, Any],
    *,
    minimum_relative_improvement: float,
) -> bool:
    """Return whether a lower-is-better paired comparison passes its gate."""

    interval = comparison["bootstrap_95pct_confidence_interval"]
    return bool(
        comparison["relative_improvement"] is not None
        and float(comparison["relative_improvement"])
        >= minimum_relative_improvement
        and float(interval[0]) > 0.0
    )
