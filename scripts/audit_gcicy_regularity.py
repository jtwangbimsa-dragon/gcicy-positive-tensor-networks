#!/usr/bin/env python3
"""Audit empirical residual regularity for frozen gCICY H-metrics."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import (  # noqa: E402
    estimate_h_metric_residual_gradient,
    file_sha256,
    finite_distance_residual_slopes,
    get_adapter,
    load_active_point_pool,
    reference_relative_spectrum,
)
from gcicy_metric.pipeline.parallel_sampling import sample_points_parallel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    return parser.parse_args()


def resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def logsumexp(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("log-sum-exp values must be non-empty and finite")
    return float(np.logaddexp.reduce(array))


def weighted_log_mean_exp(values: np.ndarray, log_weights: np.ndarray) -> float:
    raw = np.asarray(values, dtype=float)
    logw = np.asarray(log_weights, dtype=float)
    if raw.shape != logw.shape or raw.size == 0:
        raise ValueError("weighted log-mean-exp inputs must match and be non-empty")
    return logsumexp(logw + raw) - logsumexp(logw)


def weighted_quantile_indices(
    values: np.ndarray,
    log_weights: np.ndarray,
    probabilities: Sequence[float],
) -> list[int]:
    raw = np.asarray(values, dtype=float)
    logw = np.asarray(log_weights, dtype=float)
    requested = np.asarray(probabilities, dtype=float)
    if (
        raw.shape != logw.shape
        or raw.size == 0
        or not np.all(np.isfinite(raw))
        or not np.all(np.isfinite(logw))
        or requested.ndim != 1
        or len(requested) == 0
        or np.min(requested) < 0
        or np.max(requested) > 1
    ):
        raise ValueError("weighted quantile inputs are invalid")
    order = np.argsort(raw, kind="stable")
    weights = np.exp(logw[order] - float(np.max(logw)))
    cumulative = np.cumsum(weights)
    cumulative /= cumulative[-1]
    return [
        int(order[min(np.searchsorted(cumulative, probability), len(order) - 1)])
        for probability in requested
    ]


def quantile_summary(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    probabilities = np.asarray([0.0, 0.5, 0.9, 0.99, 1.0])
    observed = np.quantile(np.asarray(values, dtype=float), probabilities)
    return {
        label: float(value)
        for label, value in zip(
            ("minimum", "median", "q90", "q99", "maximum"),
            observed,
            strict=True,
        )
    }


def selected_coordinate_magnitude(point: Any, chart: Sequence[int]) -> float:
    return float(
        min(
            abs(factor[int(index)])
            for factor, index in zip(point.coordinates, chart, strict=False)
        )
    )


def add_selected_point(
    selected: dict[str, dict[str, Any]],
    key: str,
    *,
    point: Any,
    descriptor: dict[str, Any],
    reason: str,
) -> None:
    if key not in selected:
        selected[key] = {
            "key": key,
            "point": point,
            "descriptor": descriptor,
            "selection_reasons": [],
        }
    selected[key]["selection_reasons"].append(reason)


def summarize_gradient_records(
    records: Sequence[dict[str, Any]],
    artifact_keys: Sequence[str],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for artifact_key in artifact_keys:
        rows = [row for row in records if row["artifact_key"] == artifact_key]
        by_source = {}
        for source in sorted({row["source"] for row in rows}):
            source_rows = [row for row in rows if row["source"] == source]
            by_source[source] = {
                "count": len(source_rows),
                "gradient_norm": quantile_summary(
                    [row["gradient"]["richardson_gradient_norm"] for row in source_rows]
                ),
                "relative_discretization_error": quantile_summary(
                    [row["gradient"]["relative_richardson_error"] for row in source_rows]
                ),
            }
        output[artifact_key] = {
            "count": len(rows),
            "gradient_norm": quantile_summary(
                [row["gradient"]["richardson_gradient_norm"] for row in rows]
            ),
            "relative_discretization_error": quantile_summary(
                [row["gradient"]["relative_richardson_error"] for row in rows]
            ),
            "by_source": by_source,
        }
    if len(artifact_keys) == 2:
        first, second = artifact_keys
        paired: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in records:
            paired[row["point_key"]][row["artifact_key"]] = row
        log_ratios = []
        for values in paired.values():
            if set(values) != {first, second}:
                continue
            numerator = values[second]["gradient"]["richardson_gradient_norm"]
            denominator = values[first]["gradient"]["richardson_gradient_norm"]
            if numerator > 0 and denominator > 0:
                log_ratios.append(float(np.log(numerator / denominator)))
        output["paired_second_over_first_log_gradient_norm_ratio"] = {
            "first": first,
            "second": second,
            "count": len(log_ratios),
            "log_ratio": quantile_summary(log_ratios),
            "ratio": quantile_summary([float(np.exp(value)) for value in log_ratios]),
        }
    return output


def main() -> None:
    args = parse_args()
    spec_path = args.spec.expanduser().resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if int(spec.get("schema_version", -1)) != 1:
        raise SystemExit("regularity specification must use schema_version 1")
    output_path = resolve_path(spec["output"], spec_path.parent)
    if output_path.exists():
        raise FileExistsError("regularity output already exists; refusing to overwrite")
    adapter = get_adapter(str(spec["adapter"]))
    model_seed = int(spec["model"]["seed"])
    exact_model = bool(spec["model"].get("exact", True))
    model = adapter.make_model(model_seed, exact=exact_model)
    artifact_paths = [
        resolve_path(value, spec_path.parent) for value in spec["artifacts"]
    ]
    artifacts = [adapter.load_h_artifact(path, model) for path in artifact_paths]
    artifact_keys = [artifact.key for artifact in artifacts]
    if len(artifacts) == 0 or len(set(artifact_keys)) != len(artifacts):
        raise ValueError("regularity artifacts must be non-empty and uniquely named")
    active_pool_path = resolve_path(spec["active_pool"], spec_path.parent)
    active_pool = load_active_point_pool(
        active_pool_path,
        adapter,
        model,
        expected_model_seed=model_seed,
        expected_exact_model=exact_model,
    )
    started = time.perf_counter()

    basis_spec = spec["reference_basis"]
    basis_seed = int(basis_spec["seed"])
    basis_point_count = int(basis_spec["points"])
    basis_points = adapter.sample_points(model, basis_point_count, seed=basis_seed)
    recomputed_basis = adapter.restricted_section_basis(
        basis_points,
        artifacts[0].degree,
    )
    if any(tuple(artifact.degree) != tuple(artifacts[0].degree) for artifact in artifacts):
        raise ValueError("all regularity artifacts must use the same section degree")
    saved_index_rows = []
    for path in artifact_paths:
        with np.load(path, allow_pickle=False) as payload:
            if "basis_selected_indices" not in payload.files:
                raise ValueError("artifact does not record basis_selected_indices")
            saved_index_rows.append(
                np.asarray(payload["basis_selected_indices"], dtype=np.int64)
            )
    saved_indices = saved_index_rows[0]
    if any(not np.array_equal(row, saved_indices) for row in saved_index_rows[1:]):
        raise ValueError("regularity artifacts use different saved section bases")
    if (
        saved_indices.shape != (artifacts[0].section_count,)
        or len(np.unique(saved_indices)) != len(saved_indices)
        or np.min(saved_indices) < 0
        or np.max(saved_indices) >= len(recomputed_basis.ambient_exponents)
    ):
        raise ValueError("saved section-basis indices are invalid")
    basis_mapping_matches = [
        bool(
            np.array_equal(
                recomputed_basis.ambient_exponents[saved_indices],
                artifact.section_exponents,
            )
        )
        for artifact in artifacts
    ]
    if not all(basis_mapping_matches):
        raise ValueError("saved basis indices do not map to artifact section exponents")
    basis = replace(
        recomputed_basis,
        selected_indices=saved_indices,
        selected_exponents=artifacts[0].section_exponents,
        numerical_rank=len(saved_indices),
    )
    recomputed_overlap = len(
        set(int(value) for value in recomputed_basis.selected_indices)
        & set(int(value) for value in saved_indices)
    )
    reference_h, relation_error = adapter.restricted_fubini_study_h_matrix(
        basis_points,
        basis,
    )
    reference_eigenvalues = np.linalg.eigvalsh(reference_h)
    if reference_eigenvalues[0] <= 0:
        raise FloatingPointError("reference H-matrix is not positive definite")
    spectrum_rows = []
    for path, artifact in zip(artifact_paths, artifacts, strict=True):
        raw_eigenvalues = np.linalg.eigvalsh(artifact.h_matrix)
        relative = reference_relative_spectrum(reference_h, artifact.h_matrix)
        spectrum_rows.append(
            {
                "artifact_key": artifact.key,
                "artifact": str(path),
                "artifact_sha256": file_sha256(path),
                "raw_basis_dependent_minimum_eigenvalue": float(raw_eigenvalues[0]),
                "raw_basis_dependent_maximum_eigenvalue": float(raw_eigenvalues[-1]),
                "raw_basis_dependent_condition_number": float(
                    raw_eigenvalues[-1] / raw_eigenvalues[0]
                ),
                "reference_relative_minimum_eigenvalue": float(relative[0]),
                "reference_relative_maximum_eigenvalue": float(relative[-1]),
                "reference_relative_condition_number": float(relative[-1] / relative[0]),
                "reference_relative_maximum_absolute_log_eigenvalue": float(
                    np.max(np.abs(np.log(relative)))
                ),
                "reference_relative_log_eigenvalue_quantiles": quantile_summary(
                    np.log(relative).tolist()
                ),
            }
        )
    print(
        f"reference basis rank={basis.numerical_rank}, relation_error={relation_error:.3e}",
        flush=True,
    )

    replay = spec["global_replay"]
    batches = replay["batches"]
    sampling_workers = int(replay.get("sampling_workers", 1))
    sampling_cluster_size = int(replay.get("sampling_cluster_size", 1))
    sampling_backend = str(replay.get("sampling_backend", "process"))
    top_per_seed = int(replay["top_per_seed"])
    weighted_quantiles = [float(value) for value in replay["weighted_quantiles"]]
    if (
        not batches
        or top_per_seed <= 0
        or len({int(row["seed"]) for row in batches}) != len(batches)
        or any(int(row["points"]) % sampling_cluster_size for row in batches)
    ):
        raise ValueError("global replay batches or selection controls are invalid")
    selected_points: dict[str, dict[str, Any]] = {}
    global_log_numerators = {key: float("-inf") for key in artifact_keys}
    global_log_weight_sum = float("-inf")
    replay_rows = []
    for batch in batches:
        seed = int(batch["seed"])
        count = int(batch["points"])
        batch_started = time.perf_counter()
        if sampling_workers > 1:
            points, shards = sample_points_parallel(
                adapter,
                model_seed=model_seed,
                exact_model=exact_model,
                count=count,
                seed=seed,
                workers=sampling_workers,
                cluster_size=sampling_cluster_size,
                backend=sampling_backend,
            )
        else:
            points = adapter.sample_points(model, count, seed=seed)
            shards = []
        log_weights = np.asarray(
            [adapter.importance_log_weight(point) for point in points],
            dtype=float,
        )
        batch_log_weight_sum = logsumexp(log_weights)
        global_log_weight_sum = float(
            np.logaddexp(global_log_weight_sum, batch_log_weight_sum)
        )
        cluster_ids = np.asarray(adapter.sampling_cluster_ids(points), dtype=np.int64)
        artifact_batch_rows = []
        for artifact in artifacts:
            raw = adapter.residual_values(points, adapter.h_metrics(points, artifact))
            if not np.all(np.isfinite(raw)):
                raise FloatingPointError(
                    f"artifact {artifact.key}, seed {seed} has non-finite residuals"
                )
            global_log_numerators[artifact.key] = float(
                np.logaddexp(
                    global_log_numerators[artifact.key],
                    logsumexp(log_weights + raw),
                )
            )
            retain = min(top_per_seed, len(points))
            top_indices = np.argpartition(raw, -retain)[-retain:]
            for rank, index in enumerate(
                top_indices[np.argsort(raw[top_indices])[::-1]],
                start=1,
            ):
                point_index = int(index)
                add_selected_point(
                    selected_points,
                    f"global:{seed}:{point_index}",
                    point=points[point_index],
                    descriptor={
                        "source": "global_replay",
                        "seed": seed,
                        "point_index": point_index,
                        "cluster_id": int(cluster_ids[point_index]),
                    },
                    reason=f"{artifact.key}:top_rank_{rank}",
                )
            quantile_indices = weighted_quantile_indices(
                raw,
                log_weights,
                weighted_quantiles,
            )
            for probability, point_index in zip(
                weighted_quantiles,
                quantile_indices,
                strict=True,
            ):
                add_selected_point(
                    selected_points,
                    f"global:{seed}:{point_index}",
                    point=points[point_index],
                    descriptor={
                        "source": "global_replay",
                        "seed": seed,
                        "point_index": point_index,
                        "cluster_id": int(cluster_ids[point_index]),
                    },
                    reason=f"{artifact.key}:weighted_quantile_{probability:g}",
                )
            artifact_batch_rows.append(
                {
                    "artifact_key": artifact.key,
                    "seed_log_normalization": weighted_log_mean_exp(
                        raw,
                        log_weights,
                    ),
                    "maximum_raw_log_ratio": float(np.max(raw)),
                }
            )
        replay_rows.append(
            {
                "seed": seed,
                "point_count": len(points),
                "independent_fibre_count": len(points) // sampling_cluster_size,
                "sampling_shards": shards,
                "artifacts": artifact_batch_rows,
                "runtime_seconds": float(time.perf_counter() - batch_started),
            }
        )
        print(
            f"replayed seed={seed}, points={len(points)}, "
            f"seconds={time.perf_counter() - batch_started:.1f}",
            flush=True,
        )
    global_log_normalizations = {
        key: float(value - global_log_weight_sum)
        for key, value in global_log_numerators.items()
    }

    center_ids = np.asarray(active_pool.center_ids, dtype=np.int64)
    radii = np.asarray(active_pool.radii, dtype=float)
    unique_center_ids = sorted(int(value) for value in np.unique(center_ids))
    center_indices = []
    for center_id in unique_center_ids:
        candidates = np.flatnonzero((center_ids == center_id) & (radii == 0.0))
        if len(candidates) != 1:
            raise ValueError("each active center must have exactly one radius-zero point")
        center_indices.append(int(candidates[0]))
    if unique_center_ids != list(range(len(unique_center_ids))):
        raise ValueError("active center ids must be consecutive from zero")
    centers = [active_pool.points[index] for index in center_indices]
    local_indices = np.flatnonzero(radii > 0)
    local_points = [active_pool.points[int(index)] for index in local_indices]
    local_center_ids = center_ids[local_indices]
    active_residuals: dict[str, np.ndarray] = {}
    active_slopes: dict[str, np.ndarray] = {}
    active_distances = None
    for artifact in artifacts:
        residuals = adapter.residual_values(
            active_pool.points,
            adapter.h_metrics(active_pool.points, artifact),
        )
        active_residuals[artifact.key] = residuals
        slopes, distances = finite_distance_residual_slopes(
            adapter,
            centers,
            local_points,
            local_center_ids,
            residuals[center_indices],
            residuals[local_indices],
        )
        active_slopes[artifact.key] = slopes
        if active_distances is None:
            active_distances = distances
        elif not np.allclose(active_distances, distances, rtol=0.0, atol=1e-14):
            raise RuntimeError("artifact-independent local distances changed")
    assert active_distances is not None
    for center_id, pool_index in zip(
        unique_center_ids,
        center_indices,
        strict=True,
    ):
        add_selected_point(
            selected_points,
            f"active:{pool_index}",
            point=active_pool.points[pool_index],
            descriptor={
                "source": "active_pool",
                "pool_index": pool_index,
                "center_id": center_id,
                "radius": 0.0,
            },
            reason="audited_center",
        )
    local_score = np.max(
        np.stack([active_slopes[key] for key in artifact_keys], axis=0),
        axis=0,
    )
    for center_id in unique_center_ids:
        center_mask = local_center_ids == center_id
        for radius in sorted(float(value) for value in np.unique(radii[center_ids == center_id]) if value > 0):
            group = np.flatnonzero(center_mask & (radii[local_indices] == radius))
            if len(group) == 0:
                continue
            ordered = group[np.argsort(local_score[group], kind="stable")]
            choices = {
                "median_max_across_artifacts_secant_slope": int(ordered[len(ordered) // 2]),
                "maximum_across_artifacts_secant_slope": int(ordered[-1]),
            }
            for reason, local_position in choices.items():
                pool_index = int(local_indices[local_position])
                add_selected_point(
                    selected_points,
                    f"active:{pool_index}",
                    point=active_pool.points[pool_index],
                    descriptor={
                        "source": "active_pool",
                        "pool_index": pool_index,
                        "center_id": center_id,
                        "radius": radius,
                    },
                    reason=reason,
                )
    secant_rows = []
    for artifact in artifacts:
        by_radius = {}
        for radius in sorted(float(value) for value in np.unique(radii[local_indices])):
            mask = radii[local_indices] == radius
            by_radius[str(radius)] = {
                "count": int(np.count_nonzero(mask)),
                "ambient_product_fubini_study_distance": quantile_summary(
                    active_distances[mask].tolist()
                ),
                "absolute_residual_secant_slope": quantile_summary(
                    active_slopes[artifact.key][mask].tolist()
                ),
            }
        secant_rows.append(
            {
                "artifact_key": artifact.key,
                "count": len(local_points),
                "absolute_residual_secant_slope": quantile_summary(
                    active_slopes[artifact.key].tolist()
                ),
                "by_radius": by_radius,
            }
        )

    gradient_spec = spec["gradient"]
    coarse_step = float(gradient_spec["coarse_step"])
    residual_tolerance = float(gradient_spec.get("residual_tolerance", 1e-10))
    maximum_newton_iterations = int(
        gradient_spec.get("maximum_newton_iterations", 16)
    )
    selected_rows = sorted(selected_points.values(), key=lambda row: row["key"])
    gradient_records = []
    point_lookup = {row["key"]: row["point"] for row in selected_rows}
    total_estimates = len(selected_rows) * len(artifacts)
    estimate_index = 0
    for selected in selected_rows:
        for artifact in artifacts:
            estimate_index += 1
            estimate = estimate_h_metric_residual_gradient(
                adapter,
                model,
                selected["point"],
                artifact,
                coarse_step=coarse_step,
                residual_tolerance=residual_tolerance,
                maximum_newton_iterations=maximum_newton_iterations,
            )
            row = {
                "point_key": selected["key"],
                **selected["descriptor"],
                "selection_reasons": sorted(set(selected["selection_reasons"])),
                "artifact_key": artifact.key,
                "global_replay_normalized_log_ratio": float(
                    estimate.center_raw_log_ratio
                    - global_log_normalizations[artifact.key]
                ),
                "gradient": estimate.to_dict(),
            }
            gradient_records.append(row)
            if estimate_index % 20 == 0 or estimate_index == total_estimates:
                print(
                    f"gradient estimates={estimate_index}/{total_estimates}",
                    flush=True,
                )

    cross_patch_count = int(gradient_spec.get("cross_patch_point_count", 0))
    cross_patch_charts = int(gradient_spec.get("cross_patch_charts_per_point", 1))
    minimum_chart_coordinate = float(
        gradient_spec.get("minimum_chart_coordinate", 1e-4)
    )
    ranked_records = sorted(
        gradient_records,
        key=lambda row: row["gradient"]["richardson_gradient_norm"],
        reverse=True,
    )
    cross_patch_rows = []
    artifact_lookup = {artifact.key: artifact for artifact in artifacts}
    for reference in ranked_records[:cross_patch_count]:
        point = point_lookup[reference["point_key"]]
        current_chart = adapter.point_projective_chart(point)
        available = [
            chart
            for chart in adapter.projective_charts()
            if chart != current_chart
            and adapter.chart_is_available(
                point,
                chart,
                minimum=minimum_chart_coordinate,
            )
        ]
        available.sort(key=lambda chart: selected_coordinate_magnitude(point, chart))
        if cross_patch_charts > 1 and len(available) > 1:
            positions = np.linspace(0, len(available) - 1, cross_patch_charts)
            selected_charts = [available[int(round(value))] for value in positions]
        else:
            selected_charts = available[:cross_patch_charts]
        for chart in dict.fromkeys(selected_charts):
            candidate = adapter.rechart_point(model, point, chart)
            estimate = estimate_h_metric_residual_gradient(
                adapter,
                model,
                candidate,
                artifact_lookup[reference["artifact_key"]],
                coarse_step=coarse_step,
                residual_tolerance=residual_tolerance,
                maximum_newton_iterations=maximum_newton_iterations,
            )
            reference_norm = reference["gradient"]["richardson_gradient_norm"]
            cross_patch_rows.append(
                {
                    "point_key": reference["point_key"],
                    "artifact_key": reference["artifact_key"],
                    "reference_chart": list(current_chart),
                    "candidate_chart": list(chart),
                    "candidate_selected_coordinate_magnitude": (
                        selected_coordinate_magnitude(point, chart)
                    ),
                    "reference_gradient_norm": reference_norm,
                    "candidate_gradient_norm": estimate.richardson_norm,
                    "relative_gradient_norm_error": float(
                        abs(estimate.richardson_norm - reference_norm)
                        / max(reference_norm, np.finfo(float).tiny)
                    ),
                    "candidate_gradient": estimate.to_dict(),
                }
            )

    summary = {
        "schema_version": 1,
        "specification": str(spec_path),
        "specification_sha256": file_sha256(spec_path),
        "adapter": adapter.key,
        "model_seed": model_seed,
        "exact_model": exact_model,
        "claim_scope": {
            "classification": "empirical_regularity_diagnostic",
            "certified_global_lipschitz_bound": False,
            "certified_epsilon_net": False,
            "certified_global_supremum_bound": False,
            "distance_warning": (
                "Local secants use ambient product Fubini-Study chord distances; "
                "these are not certified intrinsic covering distances."
            ),
            "allowed_statement": (
                "Two-scale tangent derivatives and finite-distance secants were "
                "measured on the specified disclosed points."
            ),
            "forbidden_statement": (
                "No unobserved residual spike exists anywhere on the threefold."
            ),
        },
        "reference_basis": {
            "seed": basis_seed,
            "point_count": basis_point_count,
            "numerical_rank": int(basis.numerical_rank),
            "saved_basis_mapping_matches_all_artifacts": bool(
                all(basis_mapping_matches)
            ),
            "recomputed_numerical_rank": int(recomputed_basis.numerical_rank),
            "recomputed_qr_exactly_matches_saved_indices": bool(
                np.array_equal(recomputed_basis.selected_indices, saved_indices)
            ),
            "recomputed_qr_saved_index_overlap_count": int(recomputed_overlap),
            "fubini_study_relation_error": float(relation_error),
            "raw_basis_dependent_reference_condition_number": float(
                reference_eigenvalues[-1] / reference_eigenvalues[0]
            ),
        },
        "h_matrix_spectra": spectrum_rows,
        "global_replay": {
            "batches": replay_rows,
            "global_log_normalizations": global_log_normalizations,
            "selected_point_count": int(
                sum(row["descriptor"]["source"] == "global_replay" for row in selected_rows)
            ),
        },
        "active_pool": {
            "path": str(active_pool_path),
            "sha256": file_sha256(active_pool_path),
            "point_count": len(active_pool.points),
            "center_count": len(centers),
            "selected_point_count": int(
                sum(row["descriptor"]["source"] == "active_pool" for row in selected_rows)
            ),
            "secant_slopes": secant_rows,
        },
        "gradient_protocol": {
            "coarse_step": coarse_step,
            "fine_step": coarse_step / 2.0,
            "real_reference_orthonormal_direction_count": 6,
            "central_retractions_per_point": 24,
            "selected_geometric_point_count": len(selected_rows),
            "artifact_point_estimate_count": len(gradient_records),
        },
        "gradient_summary": summarize_gradient_records(
            gradient_records,
            artifact_keys,
        ),
        "gradient_records": gradient_records,
        "cross_patch_gradient_audit": {
            "row_count": len(cross_patch_rows),
            "maximum_relative_gradient_norm_error": float(
                max(
                    (row["relative_gradient_norm_error"] for row in cross_patch_rows),
                    default=0.0,
                )
            ),
            "rows": cross_patch_rows,
        },
        "protected_seed_namespaces_not_accessed": ["706xx", "710xx"],
        "runtime_seconds": float(time.perf_counter() - started),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "output_sha256": file_sha256(output_path),
                "selected_points": len(selected_rows),
                "gradient_estimates": len(gradient_records),
                "maximum_cross_patch_relative_error": summary[
                    "cross_patch_gradient_audit"
                ]["maximum_relative_gradient_norm_error"],
                "runtime_seconds": summary["runtime_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
