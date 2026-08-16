"""Generic sampling, metric, and atlas audit for registered gCICY adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np

from .adapter import GCICYAdapter, HMetricArtifact
from .parallel_sampling import sample_points_parallel


@dataclass(frozen=True)
class AuditThresholds:
    max_atlas_error: float = 1e-8
    min_jacobian_singular_value: float = 1e-7
    min_metric_eigenvalue: float = 0.0
    min_effective_sample_size: float = 1.0
    max_primary_mean_sigma: float | None = None
    max_primary_seed_sigma: float | None = None
    require_mean_sigma_improvement: bool = True
    require_every_seed_sigma_improvement: bool = False
    require_ordered_artifact_sigma_improvement: bool = False

    def __post_init__(self) -> None:
        if self.max_atlas_error <= 0:
            raise ValueError("max_atlas_error must be positive")
        if self.min_jacobian_singular_value < 0:
            raise ValueError("min_jacobian_singular_value cannot be negative")
        if self.min_effective_sample_size <= 0:
            raise ValueError("min_effective_sample_size must be positive")
        if self.max_primary_mean_sigma is not None and self.max_primary_mean_sigma <= 0:
            raise ValueError("max_primary_mean_sigma must be positive when set")
        if self.max_primary_seed_sigma is not None and self.max_primary_seed_sigma <= 0:
            raise ValueError("max_primary_seed_sigma must be positive when set")


@dataclass(frozen=True)
class AuditRequest:
    model_seed: int
    exact_model: bool
    artifact_paths: tuple[Path, ...]
    seeds: tuple[int, ...]
    points_per_seed: int
    atlas_points_per_seed: int = 1
    minimum_selected_coordinate: float = 1e-4
    sampling_workers: int = 1
    sampling_cluster_size: int = 1
    sampling_backend: str = "process"
    thresholds: AuditThresholds = AuditThresholds()

    def __post_init__(self) -> None:
        if not self.artifact_paths:
            raise ValueError("at least one H-metric artifact is required")
        if not self.seeds:
            raise ValueError("at least one audit seed is required")
        if self.points_per_seed <= 0:
            raise ValueError("points_per_seed must be positive")
        if self.atlas_points_per_seed <= 0:
            raise ValueError("atlas_points_per_seed must be positive")
        if self.atlas_points_per_seed > self.points_per_seed:
            raise ValueError("atlas_points_per_seed cannot exceed points_per_seed")
        if self.minimum_selected_coordinate <= 0:
            raise ValueError("minimum_selected_coordinate must be positive")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("audit seeds must be distinct")
        if self.sampling_workers <= 0 or self.sampling_cluster_size <= 0:
            raise ValueError(
                "sampling_workers and sampling_cluster_size must be positive"
            )
        if self.sampling_backend not in {"process", "thread"}:
            raise ValueError("sampling_backend must be process or thread")
        if self.sampling_workers > 1:
            if self.points_per_seed % self.sampling_cluster_size:
                raise ValueError(
                    "parallel audit points_per_seed must be divisible by sampling_cluster_size"
                )
            if (
                self.points_per_seed // self.sampling_cluster_size
                < self.sampling_workers
            ):
                raise ValueError(
                    "parallel audit requires at least one complete cluster per worker"
                )


def confidence_interval(values: list[float]) -> list[float]:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    if len(array) < 2:
        return [mean, mean]
    half_width = 1.96 * float(np.std(array, ddof=1)) / np.sqrt(len(array))
    return [mean - half_width, mean + half_width]


def effective_sample_size(weights: np.ndarray) -> float:
    values = np.asarray(weights, dtype=float).reshape(-1)
    if values.size == 0 or np.any(values < 0) or not np.all(np.isfinite(values)):
        raise ValueError("weights must be a non-empty finite non-negative array")
    denominator = float(np.sum(values**2))
    if denominator <= 0:
        raise ValueError("weights must have positive sum")
    return float(np.sum(values) ** 2 / denominator)


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    probability: float,
) -> float:
    samples = np.asarray(values, dtype=float).reshape(-1)
    masses = np.asarray(weights, dtype=float).reshape(-1)
    if samples.shape != masses.shape or samples.size == 0:
        raise ValueError("weighted quantile inputs must be non-empty and aligned")
    if (
        not np.all(np.isfinite(samples))
        or not np.all(np.isfinite(masses))
        or np.any(masses < 0)
        or np.sum(masses) <= 0
    ):
        raise ValueError("weighted quantile inputs must be finite and non-negative")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("weighted quantile probability must lie in [0, 1]")
    order = np.argsort(samples, kind="stable")
    cumulative = np.cumsum(masses[order])
    threshold = probability * cumulative[-1]
    index = min(
        int(np.searchsorted(cumulative, threshold, side="left")), len(order) - 1
    )
    return float(samples[order[index]])


def weighted_upper_tail_mean(
    values: np.ndarray,
    weights: np.ndarray,
    tail_fraction: float,
) -> float:
    samples = np.asarray(values, dtype=float).reshape(-1)
    masses = np.asarray(weights, dtype=float).reshape(-1)
    if samples.shape != masses.shape or samples.size == 0:
        raise ValueError("weighted tail inputs must be non-empty and aligned")
    if (
        not np.all(np.isfinite(samples))
        or not np.all(np.isfinite(masses))
        or np.any(masses < 0)
        or np.sum(masses) <= 0
    ):
        raise ValueError("weighted tail inputs must be finite and non-negative")
    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail fraction must lie in (0, 1]")
    normalized = masses / np.sum(masses)
    order = np.argsort(samples, kind="stable")[::-1]
    sorted_values = samples[order]
    sorted_masses = normalized[order]
    mass_before = np.cumsum(sorted_masses) - sorted_masses
    selected = np.minimum(
        sorted_masses,
        np.maximum(tail_fraction - mass_before, 0.0),
    )
    return float(np.sum(selected * sorted_values) / tail_fraction)


def normalized_volume_ratios(
    log_eta: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(log_eta, dtype=float)
    masses = np.asarray(weights, dtype=float)
    if values.shape != masses.shape or values.size == 0:
        raise ValueError("log_eta and weights must be non-empty aligned arrays")
    if (
        not np.all(np.isfinite(values))
        or not np.all(np.isfinite(masses))
        or np.any(masses < 0)
        or np.sum(masses) <= 0
    ):
        raise ValueError("log_eta and weights must be finite with positive mass")
    normalized_weights = masses / np.sum(masses)
    maximum = float(np.max(values))
    log_mean = maximum + float(
        np.log(np.sum(normalized_weights * np.exp(values - maximum)))
    )
    log_ratio = values - log_mean
    return np.exp(log_ratio), log_ratio


def standard_errors(log_eta: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    """Literature-standard normalized Monge-Ampere error measures."""

    log_eta = np.asarray(log_eta, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if log_eta.shape != weights.shape or not np.all(np.isfinite(log_eta)):
        raise ValueError("log_eta and weights must be finite arrays of equal shape")
    if np.any(weights < 0) or not np.all(np.isfinite(weights)) or np.sum(weights) <= 0:
        raise ValueError("weights must be finite, non-negative, and have positive sum")
    normalized_weights = weights / np.sum(weights)
    normalized_ratio, log_normalized_ratio = normalized_volume_ratios(
        log_eta,
        weights,
    )
    positive_log_ratio = np.maximum(log_normalized_ratio, 0.0)
    ratio_error = 1.0 - normalized_ratio
    inverse_ratio_error = 1.0 - 1.0 / normalized_ratio
    centered_log_eta = log_eta - float(np.sum(normalized_weights * log_eta))
    energy = float(np.sum(normalized_weights * ratio_error**2))
    metric_volume_weights = normalized_weights * normalized_ratio
    metric_volume_weights /= np.sum(metric_volume_weights)
    return {
        "sigma": float(np.sum(normalized_weights * np.abs(ratio_error))),
        "inverse_sigma": float(
            np.sum(normalized_weights * np.abs(inverse_ratio_error))
        ),
        "squared_energy": energy,
        "sqrt_squared_energy": float(np.sqrt(energy)),
        "weighted_centered_log_ma_rms": float(
            np.sqrt(np.sum(normalized_weights * centered_log_eta**2))
        ),
        "normalized_ratio_min": float(np.min(normalized_ratio)),
        "normalized_ratio_max": float(np.max(normalized_ratio)),
        "positive_log_ratio_q999": weighted_quantile(
            positive_log_ratio,
            weights,
            0.999,
        ),
        "positive_log_ratio_cvar_1pct": weighted_upper_tail_mean(
            positive_log_ratio,
            weights,
            0.01,
        ),
        "normalized_ratio_above_3_weighted_mass": float(
            np.sum(normalized_weights * (normalized_ratio > 3.0))
        ),
        "normalized_ratio_above_3_point_count": int(
            np.count_nonzero(normalized_ratio > 3.0)
        ),
        "importance_effective_sample_size": effective_sample_size(weights),
        "metric_volume_effective_sample_size": effective_sample_size(
            metric_volume_weights
        ),
        "metric_volume_maximum_point_mass": float(np.max(metric_volume_weights)),
    }


def _metric_minimum_eigenvalue(metrics: np.ndarray) -> float:
    values = np.linalg.eigvalsh(np.asarray(metrics, dtype=np.complex128))
    return float(np.min(values))


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"seeds": rows}
    for key in (
        "sigma",
        "inverse_sigma",
        "squared_energy",
        "sqrt_squared_energy",
        "weighted_centered_log_ma_rms",
        "positive_log_ratio_q999",
        "positive_log_ratio_cvar_1pct",
        "normalized_ratio_above_3_weighted_mass",
    ):
        values = [float(row[key]) for row in rows]
        output[f"mean_{key}"] = float(np.mean(values))
        output[f"{key}_95_percent_ci"] = confidence_interval(values)
        output[f"max_seed_{key}"] = float(np.max(values))
    output["min_metric_eigenvalue"] = float(
        np.min([row["min_metric_eigenvalue"] for row in rows])
    )
    output["mean_importance_effective_sample_size"] = float(
        np.mean([row["importance_effective_sample_size"] for row in rows])
    )
    output["min_importance_effective_sample_size"] = float(
        np.min([row["importance_effective_sample_size"] for row in rows])
    )
    output["mean_metric_volume_effective_sample_size"] = float(
        np.mean([row["metric_volume_effective_sample_size"] for row in rows])
    )
    output["min_metric_volume_effective_sample_size"] = float(
        np.min([row["metric_volume_effective_sample_size"] for row in rows])
    )
    output["max_metric_volume_maximum_point_mass"] = float(
        np.max([row["metric_volume_maximum_point_mass"] for row in rows])
    )
    output["min_normalized_ratio"] = float(
        np.min([row["normalized_ratio_min"] for row in rows])
    )
    output["max_normalized_ratio"] = float(
        np.max([row["normalized_ratio_max"] for row in rows])
    )
    return output


def _paired_sigma_summary(
    baseline_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    baseline = {int(row["seed"]): float(row["sigma"]) for row in baseline_rows}
    candidate = {int(row["seed"]): float(row["sigma"]) for row in candidate_rows}
    if baseline.keys() != candidate.keys():
        raise ValueError("baseline and candidate seed sets differ")
    rows = []
    improvements = []
    for seed in sorted(baseline):
        improvement = baseline[seed] - candidate[seed]
        improvements.append(improvement)
        rows.append(
            {
                "seed": seed,
                "baseline_sigma": baseline[seed],
                "candidate_sigma": candidate[seed],
                "absolute_improvement": improvement,
                "relative_improvement": improvement / baseline[seed],
            }
        )
    interval = confidence_interval(improvements)
    mean_baseline = float(np.mean(list(baseline.values())))
    return {
        "seeds": rows,
        "improved_seed_count": int(sum(value > 0 for value in improvements)),
        "seed_count": len(improvements),
        "mean_absolute_improvement": float(np.mean(improvements)),
        "absolute_improvement_95_percent_ci": interval,
        "relative_improvement_against_mean_baseline": float(
            np.mean(improvements) / mean_baseline
        ),
        "relative_improvement_95_percent_ci": [
            interval[0] / mean_baseline,
            interval[1] / mean_baseline,
        ],
    }


def _empty_atlas_state(artifacts: list[HMetricArtifact]) -> dict[str, Any]:
    return {
        "projective_charts_seen": set(),
        "implicit_choices_seen": set(),
        "max_baseline_projective_ma_error": 0.0,
        "max_baseline_implicit_ma_error": 0.0,
        "max_projective_importance_log_weight_error": 0.0,
        "max_implicit_importance_log_weight_error": 0.0,
        "artifact_projective_ma_errors": {artifact.key: 0.0 for artifact in artifacts},
        "artifact_implicit_ma_errors": {artifact.key: 0.0 for artifact in artifacts},
        "max_error_witnesses": {
            "baseline_projective_ma": None,
            "baseline_implicit_ma": None,
            "projective_importance_log_weight": None,
            "implicit_importance_log_weight": None,
            "artifact_projective_ma": {artifact.key: None for artifact in artifacts},
            "artifact_implicit_ma": {artifact.key: None for artifact in artifacts},
        },
        "minimum_atlas_metric_eigenvalue": float("inf"),
    }


def _update_atlas_metric_minimum(state: dict[str, Any], metric: np.ndarray) -> None:
    state["minimum_atlas_metric_eigenvalue"] = min(
        state["minimum_atlas_metric_eigenvalue"],
        float(np.min(np.linalg.eigvalsh(np.asarray(metric, dtype=np.complex128)))),
    )


def _point_witness_metadata(adapter: GCICYAdapter, point: Any) -> dict[str, Any]:
    output: dict[str, Any] = {
        "projective_chart": [
            int(value) for value in adapter.point_projective_chart(point)
        ],
        "jacobian_min_singular_value": float(
            adapter.point_jacobian_min_singular_value(point)
        ),
    }
    if hasattr(point, "independent_indices"):
        output["independent_indices"] = [
            int(value) for value in point.independent_indices
        ]
    if hasattr(point, "dependent_indices"):
        output["dependent_indices"] = [int(value) for value in point.dependent_indices]
    if hasattr(point, "residue_denominator"):
        output["residue_denominator_magnitude"] = float(abs(point.residue_denominator))
    if hasattr(point, "affine_coordinates"):
        magnitudes = np.abs(np.asarray(point.affine_coordinates)).reshape(-1)
        output["affine_coordinate_magnitude_minimum"] = float(np.min(magnitudes))
        output["affine_coordinate_magnitude_maximum"] = float(np.max(magnitudes))
    factors = []
    for attribute in ("x", "y", "z"):
        if hasattr(point, attribute):
            factors.append(
                [
                    float(value)
                    for value in np.abs(np.asarray(getattr(point, attribute))).reshape(
                        -1
                    )
                ]
            )
    if factors:
        output["homogeneous_factor_coordinate_magnitudes"] = factors
    if hasattr(point, "tangent_basis"):
        condition = float(
            np.linalg.cond(np.asarray(point.tangent_basis, dtype=np.complex128))
        )
        output["tangent_basis_condition_number"] = (
            condition if np.isfinite(condition) else None
        )
    return output


def _metric_witness(metric: np.ndarray) -> dict[str, Any]:
    values = np.asarray(metric, dtype=np.complex128)
    hermitian = 0.5 * (values + values.conj().T)
    eigenvalues = np.linalg.eigvalsh(values)
    with np.errstate(divide="ignore", invalid="ignore"):
        sign, slogdet = np.linalg.slogdet(values)
    cholesky = np.linalg.cholesky(hermitian)
    slogdet_value = float(slogdet)
    sign_parts = [float(np.real(sign)), float(np.imag(sign))]
    return {
        "metric_eigenvalues": [float(value) for value in eigenvalues],
        "eigenvalue_logdet": float(np.sum(np.log(eigenvalues))),
        "slogdet_logabsdet": slogdet_value if np.isfinite(slogdet_value) else None,
        "slogdet_phase": sign_parts if np.all(np.isfinite(sign_parts)) else None,
        "cholesky_logdet": float(2.0 * np.sum(np.log(np.real(np.diag(cholesky))))),
        "hermitian_symmetry_max_error": float(np.max(np.abs(values - values.conj().T))),
    }


def _audit_point_atlas(
    adapter: GCICYAdapter,
    model: Any,
    point: Any,
    artifacts: list[HMetricArtifact],
    state: dict[str, Any],
    *,
    minimum_selected_coordinate: float,
    sample_seed: int,
    sample_point_index: int,
) -> None:
    reference_baseline_metric = adapter.baseline_metric(point)
    reference_baseline_ma = adapter.monge_ampere_log_error(
        point, reference_baseline_metric
    )
    reference_importance = adapter.importance_log_weight(point)
    reference_artifact_ma = {}
    reference_artifact_metric = {}
    witness_context = {
        "seed": int(sample_seed),
        "point_index": int(sample_point_index),
        "reference_point": _point_witness_metadata(adapter, point),
    }
    _update_atlas_metric_minimum(state, reference_baseline_metric)
    for artifact in artifacts:
        metric = adapter.h_metric(point, artifact)
        reference_artifact_metric[artifact.key] = metric
        reference_artifact_ma[artifact.key] = adapter.monge_ampere_log_error(
            point, metric
        )
        _update_atlas_metric_minimum(state, metric)

    for chart in adapter.projective_charts():
        chart = tuple(int(value) for value in chart)
        if not adapter.chart_is_available(
            point, chart, minimum=minimum_selected_coordinate
        ):
            continue
        candidate = adapter.rechart_point(model, point, chart)
        candidate_metric = adapter.baseline_metric(candidate)
        candidate_baseline_ma = adapter.monge_ampere_log_error(
            candidate, candidate_metric
        )
        baseline_error = abs(candidate_baseline_ma - reference_baseline_ma)
        if baseline_error > state["max_baseline_projective_ma_error"]:
            state["max_baseline_projective_ma_error"] = baseline_error
            state["max_error_witnesses"]["baseline_projective_ma"] = {
                **witness_context,
                "candidate_point": _point_witness_metadata(adapter, candidate),
                "absolute_error": baseline_error,
                "reference_ma": reference_baseline_ma,
                "candidate_ma": candidate_baseline_ma,
                "reference_metric": _metric_witness(reference_baseline_metric),
                "candidate_metric": _metric_witness(candidate_metric),
            }
        candidate_importance = adapter.importance_log_weight(candidate)
        importance_error = abs(candidate_importance - reference_importance)
        if importance_error > state["max_projective_importance_log_weight_error"]:
            state["max_projective_importance_log_weight_error"] = importance_error
            state["max_error_witnesses"]["projective_importance_log_weight"] = {
                **witness_context,
                "candidate_point": _point_witness_metadata(adapter, candidate),
                "absolute_error": importance_error,
                "reference_log_weight": reference_importance,
                "candidate_log_weight": candidate_importance,
            }
        _update_atlas_metric_minimum(state, candidate_metric)
        for artifact in artifacts:
            metric = adapter.h_metric(candidate, artifact)
            candidate_ma = adapter.monge_ampere_log_error(candidate, metric)
            error = abs(candidate_ma - reference_artifact_ma[artifact.key])
            if error > state["artifact_projective_ma_errors"][artifact.key]:
                state["artifact_projective_ma_errors"][artifact.key] = error
                state["max_error_witnesses"]["artifact_projective_ma"][artifact.key] = {
                    **witness_context,
                    "candidate_point": _point_witness_metadata(adapter, candidate),
                    "absolute_error": error,
                    "reference_ma": reference_artifact_ma[artifact.key],
                    "candidate_ma": candidate_ma,
                    "reference_metric": _metric_witness(
                        reference_artifact_metric[artifact.key]
                    ),
                    "candidate_metric": _metric_witness(metric),
                }
            _update_atlas_metric_minimum(state, metric)
        state["projective_charts_seen"].add(chart)

    reference_chart = adapter.point_projective_chart(point)
    for independent in adapter.implicit_coordinate_choices():
        independent = tuple(int(value) for value in independent)
        try:
            candidate = adapter.rechart_point(
                model,
                point,
                reference_chart,
                independent=independent,
            )
        except (FloatingPointError, np.linalg.LinAlgError):
            continue
        candidate_metric = adapter.baseline_metric(candidate)
        candidate_baseline_ma = adapter.monge_ampere_log_error(
            candidate, candidate_metric
        )
        baseline_error = abs(candidate_baseline_ma - reference_baseline_ma)
        if baseline_error > state["max_baseline_implicit_ma_error"]:
            state["max_baseline_implicit_ma_error"] = baseline_error
            state["max_error_witnesses"]["baseline_implicit_ma"] = {
                **witness_context,
                "candidate_point": _point_witness_metadata(adapter, candidate),
                "absolute_error": baseline_error,
                "reference_ma": reference_baseline_ma,
                "candidate_ma": candidate_baseline_ma,
                "reference_metric": _metric_witness(reference_baseline_metric),
                "candidate_metric": _metric_witness(candidate_metric),
            }
        candidate_importance = adapter.importance_log_weight(candidate)
        importance_error = abs(candidate_importance - reference_importance)
        if importance_error > state["max_implicit_importance_log_weight_error"]:
            state["max_implicit_importance_log_weight_error"] = importance_error
            state["max_error_witnesses"]["implicit_importance_log_weight"] = {
                **witness_context,
                "candidate_point": _point_witness_metadata(adapter, candidate),
                "absolute_error": importance_error,
                "reference_log_weight": reference_importance,
                "candidate_log_weight": candidate_importance,
            }
        _update_atlas_metric_minimum(state, candidate_metric)
        for artifact in artifacts:
            metric = adapter.h_metric(candidate, artifact)
            candidate_ma = adapter.monge_ampere_log_error(candidate, metric)
            error = abs(candidate_ma - reference_artifact_ma[artifact.key])
            if error > state["artifact_implicit_ma_errors"][artifact.key]:
                state["artifact_implicit_ma_errors"][artifact.key] = error
                state["max_error_witnesses"]["artifact_implicit_ma"][artifact.key] = {
                    **witness_context,
                    "candidate_point": _point_witness_metadata(adapter, candidate),
                    "absolute_error": error,
                    "reference_ma": reference_artifact_ma[artifact.key],
                    "candidate_ma": candidate_ma,
                    "reference_metric": _metric_witness(
                        reference_artifact_metric[artifact.key]
                    ),
                    "candidate_metric": _metric_witness(metric),
                }
            _update_atlas_metric_minimum(state, metric)
        state["implicit_choices_seen"].add(independent)


def run_pipeline_audit(adapter: GCICYAdapter, request: AuditRequest) -> dict[str, Any]:
    """Run one reproducible audit without knowing configuration-specific equations."""

    started = time.perf_counter()
    model = adapter.make_model(request.model_seed, exact=request.exact_model)
    artifacts = [
        adapter.load_h_artifact(path, model) for path in request.artifact_paths
    ]
    if len({artifact.key for artifact in artifacts}) != len(artifacts):
        raise ValueError("artifact file stems must be unique")

    baseline_rows: list[dict[str, Any]] = []
    artifact_rows: dict[str, list[dict[str, Any]]] = {
        artifact.key: [] for artifact in artifacts
    }
    atlas = _empty_atlas_state(artifacts)
    minimum_jacobian_singular_value = float("inf")
    sampling_seconds = 0.0
    metric_seconds = 0.0
    atlas_seconds = 0.0
    sampling_shards: dict[str, list[dict[str, int]]] = {}

    for seed in request.seeds:
        sampling_started = time.perf_counter()
        if request.sampling_workers > 1:
            points, shard_metadata = sample_points_parallel(
                adapter,
                model_seed=request.model_seed,
                exact_model=request.exact_model,
                count=request.points_per_seed,
                seed=seed,
                workers=request.sampling_workers,
                cluster_size=request.sampling_cluster_size,
                backend=request.sampling_backend,
            )
            sampling_shards[str(seed)] = shard_metadata
        else:
            points = adapter.sample_points(model, request.points_per_seed, seed=seed)
        sampling_seconds += time.perf_counter() - sampling_started
        if len(points) != request.points_per_seed:
            raise RuntimeError(
                f"adapter returned {len(points)} points, expected {request.points_per_seed}"
            )
        minimum_jacobian_singular_value = min(
            minimum_jacobian_singular_value,
            min(adapter.point_jacobian_min_singular_value(point) for point in points),
        )

        metric_started = time.perf_counter()
        weights = adapter.importance_weights(points)
        baseline_metrics = adapter.baseline_metrics(points)
        baseline_stats = standard_errors(
            adapter.residual_values(points, baseline_metrics), weights
        )
        baseline_rows.append(
            {
                "seed": seed,
                **baseline_stats,
                "min_metric_eigenvalue": _metric_minimum_eigenvalue(baseline_metrics),
            }
        )
        for artifact in artifacts:
            metrics = adapter.h_metrics(points, artifact)
            stats = standard_errors(adapter.residual_values(points, metrics), weights)
            artifact_rows[artifact.key].append(
                {
                    "seed": seed,
                    **stats,
                    "min_metric_eigenvalue": _metric_minimum_eigenvalue(metrics),
                }
            )
        metric_seconds += time.perf_counter() - metric_started

        atlas_started = time.perf_counter()
        for point_index, point in enumerate(points[: request.atlas_points_per_seed]):
            _audit_point_atlas(
                adapter,
                model,
                point,
                artifacts,
                atlas,
                minimum_selected_coordinate=request.minimum_selected_coordinate,
                sample_seed=seed,
                sample_point_index=point_index,
            )
        atlas_seconds += time.perf_counter() - atlas_started

    baseline = _aggregate_rows(baseline_rows)
    artifact_summaries = []
    for artifact in artifacts:
        aggregate = _aggregate_rows(artifact_rows[artifact.key])
        paired = _paired_sigma_summary(baseline_rows, artifact_rows[artifact.key])
        artifact_summaries.append(
            {
                **artifact.to_dict(),
                **aggregate,
                "paired_sigma_improvement": paired,
                "mean_sigma_improved": aggregate["mean_sigma"] < baseline["mean_sigma"],
                "every_seed_sigma_improved": paired["improved_seed_count"]
                == len(request.seeds),
            }
        )

    pairwise_artifact_comparisons = []
    consecutive_artifact_comparisons = []
    for source_index, source in enumerate(artifacts):
        for target_index in range(source_index + 1, len(artifacts)):
            target = artifacts[target_index]
            paired = _paired_sigma_summary(
                artifact_rows[source.key],
                artifact_rows[target.key],
            )
            comparison = {
                "source_artifact": source.key,
                "source_degree": list(source.degree),
                "target_artifact": target.key,
                "target_degree": list(target.degree),
                **paired,
                "mean_sigma_improved": paired["mean_absolute_improvement"] > 0,
                "every_seed_sigma_improved": paired["improved_seed_count"]
                == len(request.seeds),
            }
            pairwise_artifact_comparisons.append(comparison)
            if target_index == source_index + 1:
                consecutive_artifact_comparisons.append(comparison)

    expected_projective = adapter.expected_projective_chart_count()
    expected_implicit = adapter.expected_implicit_coordinate_choice_count()
    atlas_summary = {
        "projective_charts_seen": len(atlas["projective_charts_seen"]),
        "projective_charts_expected": expected_projective,
        "implicit_coordinate_choices_seen": len(atlas["implicit_choices_seen"]),
        "implicit_coordinate_choices_expected": expected_implicit,
        "implicit_coordinate_choices_combinatorial": (
            adapter.configuration.implicit_coordinate_choice_count
        ),
        "max_baseline_projective_ma_error": atlas["max_baseline_projective_ma_error"],
        "max_baseline_implicit_ma_error": atlas["max_baseline_implicit_ma_error"],
        "max_projective_importance_log_weight_error": atlas[
            "max_projective_importance_log_weight_error"
        ],
        "max_implicit_importance_log_weight_error": atlas[
            "max_implicit_importance_log_weight_error"
        ],
        "artifact_projective_ma_errors": atlas["artifact_projective_ma_errors"],
        "artifact_implicit_ma_errors": atlas["artifact_implicit_ma_errors"],
        "max_error_witnesses": atlas["max_error_witnesses"],
        "minimum_atlas_metric_eigenvalue": atlas["minimum_atlas_metric_eigenvalue"],
    }

    thresholds = request.thresholds
    atlas_errors = [
        atlas_summary["max_baseline_projective_ma_error"],
        atlas_summary["max_baseline_implicit_ma_error"],
        atlas_summary["max_projective_importance_log_weight_error"],
        atlas_summary["max_implicit_importance_log_weight_error"],
        *atlas_summary["artifact_projective_ma_errors"].values(),
        *atlas_summary["artifact_implicit_ma_errors"].values(),
    ]
    minimum_metric_eigenvalue = min(
        baseline["min_metric_eigenvalue"],
        atlas_summary["minimum_atlas_metric_eigenvalue"],
        *(summary["min_metric_eigenvalue"] for summary in artifact_summaries),
    )
    primary = artifact_summaries[-1]
    gates = {
        "configuration_valid": True,
        "complete_projective_atlas": atlas_summary["projective_charts_seen"]
        == expected_projective,
        "complete_implicit_atlas": atlas_summary["implicit_coordinate_choices_seen"]
        == expected_implicit,
        "atlas_consistency": max(atlas_errors) <= thresholds.max_atlas_error,
        "jacobian_rank_margin": minimum_jacobian_singular_value
        >= thresholds.min_jacobian_singular_value,
        "metric_positive": minimum_metric_eigenvalue > thresholds.min_metric_eigenvalue,
        "importance_effective_sample_size": baseline[
            "min_importance_effective_sample_size"
        ]
        >= thresholds.min_effective_sample_size,
        "mean_sigma_improved": (
            all(summary["mean_sigma_improved"] for summary in artifact_summaries)
            if thresholds.require_mean_sigma_improvement
            else True
        ),
        "every_seed_sigma_improved": (
            all(summary["every_seed_sigma_improved"] for summary in artifact_summaries)
            if thresholds.require_every_seed_sigma_improvement
            else True
        ),
        "ordered_artifact_sigma_improved": (
            all(
                comparison["mean_sigma_improved"]
                and comparison["every_seed_sigma_improved"]
                for comparison in consecutive_artifact_comparisons
            )
            if thresholds.require_ordered_artifact_sigma_improvement
            else True
        ),
        "primary_mean_sigma_target": (
            primary["mean_sigma"] <= thresholds.max_primary_mean_sigma
            if thresholds.max_primary_mean_sigma is not None
            else True
        ),
        "primary_max_seed_sigma_target": (
            primary["max_seed_sigma"] <= thresholds.max_primary_seed_sigma
            if thresholds.max_primary_seed_sigma is not None
            else True
        ),
    }

    return {
        "schema_version": 1,
        "description": "Configuration-driven gCICY metric and geometry audit.",
        "adapter": {"key": adapter.key, "version": adapter.version},
        "configuration": adapter.configuration.to_dict(),
        "model": adapter.model_metadata(model),
        "request": {
            "model_seed": request.model_seed,
            "exact_model": request.exact_model,
            "seeds": list(request.seeds),
            "points_per_seed": request.points_per_seed,
            "atlas_points_per_seed": request.atlas_points_per_seed,
            "minimum_selected_coordinate": request.minimum_selected_coordinate,
            "sampling_workers": request.sampling_workers,
            "sampling_cluster_size": request.sampling_cluster_size,
            "sampling_backend": request.sampling_backend,
            "thresholds": asdict(thresholds),
        },
        "parallel_sampling": {
            "workers": request.sampling_workers,
            "cluster_size": request.sampling_cluster_size,
            "backend": request.sampling_backend,
            "seed_derivation": "numpy.random.SeedSequence(base_seed).spawn(active_workers)",
            "shards": sampling_shards,
        },
        "definitions": {
            "eta": "det(g) / |Omega|^2 in matching exact local coordinates",
            "sigma": "Omega-weighted mean(abs(1 - eta / mean_Omega(eta)))",
            "inverse_sigma": "Omega-weighted mean(abs(1 - mean_Omega(eta) / eta))",
        },
        "baseline": baseline,
        "artifacts": artifact_summaries,
        "artifact_pairwise_sigma_comparisons": pairwise_artifact_comparisons,
        "atlas": atlas_summary,
        "minimum_sampled_jacobian_singular_value": minimum_jacobian_singular_value,
        "minimum_metric_eigenvalue": minimum_metric_eigenvalue,
        "gates": gates,
        "success": bool(all(gates.values())),
        "runtime_seconds": {
            "sampling": sampling_seconds,
            "metrics": metric_seconds,
            "atlas": atlas_seconds,
            "total": time.perf_counter() - started,
        },
    }
