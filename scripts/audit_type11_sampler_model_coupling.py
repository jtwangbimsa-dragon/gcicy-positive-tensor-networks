#!/usr/bin/env python3
"""Audit type-(1,1) sampling geometry against model residual tails."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.stats import hypergeom, kstest, spearmanr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import get_adapter  # noqa: E402
from gcicy_metric.simple_patch import fubini_study_metric  # noqa: E402
from gcicy_metric.type21_hirzebruch_x3 import (  # noqa: E402
    _y_monomials,
    hirzebruch_type21_holomorphic_volume_log_density,
    hirzebruch_type21_importance_log_weight,
    hirzebruch_type21_proposal_log_density,
    local_equations_jacobian_and_scales,
)


EXPECTED_X3_OVER_PROPOSAL = 11.0 / 12.0
EXPECTED_TOTAL_OVER_PROPOSAL = 23.0 / 12.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--tail-arrays", type=Path, required=True)
    parser.add_argument(
        "--ratio",
        action="append",
        default=[],
        metavar="NAME=PATH:KEY",
        help="Add a normalized model-ratio array; repeat as needed.",
    )
    parser.add_argument("--model-seed", type=int, default=20260731)
    parser.add_argument("--atlas-points", type=int, default=12)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def parse_ratio_argument(value: str) -> tuple[str, Path, str]:
    if "=" not in value or ":" not in value:
        raise ValueError("--ratio must use NAME=PATH:KEY")
    name, source = value.split("=", 1)
    path_value, key = source.rsplit(":", 1)
    if not name or not key:
        raise ValueError("ratio name and key must be non-empty")
    path = Path(path_value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return name, path, key


def projective_root_separations(
    coordinates_x: np.ndarray,
    cluster_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    unique_clusters = np.unique(cluster_ids)
    cluster_values = np.empty(len(unique_clusters), dtype=np.float64)
    point_values = np.empty(len(cluster_ids), dtype=np.float64)
    for output_index, cluster_id in enumerate(unique_clusters):
        indices = np.flatnonzero(cluster_ids == cluster_id)
        if len(indices) != 4:
            raise ValueError(f"cluster {cluster_id} does not contain four roots")
        roots = coordinates_x[indices]
        roots = roots / np.linalg.norm(roots, axis=1, keepdims=True)
        separations = []
        for left in range(4):
            for right in range(left + 1, 4):
                overlap = min(1.0, float(abs(np.vdot(roots[left], roots[right]))))
                separations.append(math.sqrt(max(0.0, 1.0 - overlap**2)))
        minimum = float(min(separations))
        cluster_values[output_index] = minimum
        point_values[indices] = minimum
    return cluster_values, point_values


def quantiles(values: np.ndarray) -> dict[str, float]:
    probabilities = (0.0, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 1.0)
    observed = np.quantile(np.asarray(values, dtype=np.float64), probabilities)
    return {
        f"q{probability:.3f}": float(value)
        for probability, value in zip(probabilities, observed, strict=True)
    }


def mean_standard_error(values: np.ndarray, expected: float) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    return {
        "sample_count": int(len(values)),
        "mean": mean,
        "standard_error": standard_error,
        "expected": float(expected),
        "difference": mean - expected,
        "z_score": (mean - expected) / standard_error if standard_error > 0 else 0.0,
    }


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    result = spearmanr(left, right, nan_policy="omit").statistic
    return float(result) if np.isfinite(result) else 0.0


def top_tail_enrichment(
    score: np.ndarray,
    diagnostic: np.ndarray,
    *,
    score_upper: bool,
    diagnostic_upper: bool,
    fraction: float = 0.01,
) -> dict[str, float | int]:
    count = max(1, int(math.ceil(fraction * len(score))))
    score_order = np.argsort(score, kind="stable")
    diagnostic_order = np.argsort(diagnostic, kind="stable")
    score_indices = score_order[-count:] if score_upper else score_order[:count]
    diagnostic_indices = (
        diagnostic_order[-count:] if diagnostic_upper else diagnostic_order[:count]
    )
    overlap = len(set(score_indices.tolist()) & set(diagnostic_indices.tolist()))
    observed_fraction = overlap / count
    return {
        "tail_count": count,
        "overlap_count": overlap,
        "overlap_fraction": observed_fraction,
        "enrichment_over_independence": observed_fraction / fraction,
    }


def model_diagnostics(
    ratio: np.ndarray,
    weights: np.ndarray,
    chart_ids: np.ndarray,
    cluster_ids: np.ndarray,
    diagnostics: dict[str, np.ndarray],
) -> dict[str, Any]:
    ratio = np.asarray(ratio, dtype=np.float64)
    if ratio.shape != weights.shape or np.any(~np.isfinite(ratio)) or np.min(ratio) <= 0:
        raise ValueError("normalized model ratios must be finite and positive")
    abs_log = np.abs(np.log(ratio))
    top_positive = np.argsort(ratio, kind="stable")[-max(1, len(ratio) // 100) :]
    top_absolute = np.argsort(abs_log, kind="stable")[-max(1, len(ratio) // 100) :]
    correlations = {
        name: safe_spearman(abs_log, values) for name, values in diagnostics.items()
    }
    chart_rows = []
    for chart_id in np.unique(chart_ids):
        mask = chart_ids == chart_id
        chart_rows.append(
            {
                "chart_id": int(chart_id),
                "point_count": int(np.sum(mask)),
                "mean_ratio": float(np.mean(ratio[mask])),
                "ratio_q99": float(np.quantile(ratio[mask], 0.99)),
                "ratio_above_3_fraction": float(np.mean(ratio[mask] > 3.0)),
                "mean_abs_log_ratio": float(np.mean(abs_log[mask])),
            }
        )
    worst = int(np.argmax(ratio))
    worst_absolute = int(np.argmax(abs_log))
    weight_sum = float(np.sum(weights))
    return {
        "sigma": float(np.sum(weights * np.abs(1.0 - ratio)) / weight_sum),
        "chi_l2": float(
            math.sqrt(np.sum(weights * (1.0 - ratio) ** 2) / weight_sum)
        ),
        "ratio_max": float(np.max(ratio)),
        "ratio_min": float(np.min(ratio)),
        "ratio_above_3_point_count": int(np.sum(ratio > 3.0)),
        "ratio_above_3_weighted_mass": float(np.sum(weights[ratio > 3.0]) / weight_sum),
        "top_positive_one_percent_independent_clusters": int(
            len(np.unique(cluster_ids[top_positive]))
        ),
        "top_absolute_one_percent_independent_clusters": int(
            len(np.unique(cluster_ids[top_absolute]))
        ),
        "spearman_abs_log_residual": correlations,
        "tail_enrichment": {
            "positive_ratio_vs_low_root_separation": top_tail_enrichment(
                ratio,
                diagnostics["root_separation"],
                score_upper=True,
                diagnostic_upper=False,
            ),
            "positive_ratio_vs_low_jacobian_singular_value": top_tail_enrichment(
                ratio,
                diagnostics[
                    "normalized_reduced_jacobian_min_singular_value"
                ],
                score_upper=True,
                diagnostic_upper=False,
            ),
            "positive_ratio_vs_high_importance_weight": top_tail_enrichment(
                ratio,
                diagnostics["log_importance_weight"],
                score_upper=True,
                diagnostic_upper=True,
            ),
        },
        "worst_positive_point": {
            "index": worst,
            "cluster_id": int(cluster_ids[worst]),
            "chart_id": int(chart_ids[worst]),
            "ratio": float(ratio[worst]),
            "importance_weight": float(weights[worst]),
            **{name: float(values[worst]) for name, values in diagnostics.items()},
        },
        "worst_absolute_point": {
            "index": worst_absolute,
            "cluster_id": int(cluster_ids[worst_absolute]),
            "chart_id": int(chart_ids[worst_absolute]),
            "ratio": float(ratio[worst_absolute]),
            "abs_log_ratio": float(abs_log[worst_absolute]),
        },
        "by_chart": chart_rows,
    }


def atlas_invariance(
    adapter: Any,
    model: Any,
    points: list[Any],
    selected_indices: list[int],
) -> dict[str, Any]:
    rows = []
    maximum_weight_spread = 0.0
    maximum_fs_raw_spread = 0.0
    maximum_equation_residual = 0.0
    for point_index in selected_indices:
        point = points[point_index]
        weight_values = []
        fs_raw_values = []
        chart_rows = []
        for chart in adapter.projective_charts():
            if not adapter.chart_is_available(point, chart, minimum=1.0e-5):
                continue
            candidate = adapter.rechart_point(model, point, chart)
            equations, _, scales = local_equations_jacobian_and_scales(
                model,
                candidate.affine_coordinates,
                candidate.projective_chart,
            )
            equation_residual = float(
                np.max(np.abs(equations) / np.maximum(1.0, scales))
            )
            metric = adapter.baseline_metric(candidate)
            eigenvalues = np.linalg.eigvalsh(metric)
            fs_raw = float(
                np.sum(np.log(eigenvalues))
                - hirzebruch_type21_holomorphic_volume_log_density(candidate)
            )
            log_weight = hirzebruch_type21_importance_log_weight(candidate)
            weight_values.append(log_weight)
            fs_raw_values.append(fs_raw)
            maximum_equation_residual = max(maximum_equation_residual, equation_residual)
            chart_rows.append(
                {
                    "chart": [int(value) for value in chart],
                    "log_importance_weight": float(log_weight),
                    "fs_raw_log_ratio": fs_raw,
                    "relative_equation_residual": equation_residual,
                }
            )
        weight_spread = float(np.ptp(weight_values))
        fs_raw_spread = float(np.ptp(fs_raw_values))
        maximum_weight_spread = max(maximum_weight_spread, weight_spread)
        maximum_fs_raw_spread = max(maximum_fs_raw_spread, fs_raw_spread)
        rows.append(
            {
                "point_index": int(point_index),
                "charts_checked": len(chart_rows),
                "log_importance_weight_spread": weight_spread,
                "fs_raw_log_ratio_spread": fs_raw_spread,
                "charts": chart_rows,
            }
        )
    return {
        "points_checked": len(rows),
        "maximum_log_importance_weight_spread": maximum_weight_spread,
        "maximum_fs_raw_log_ratio_spread": maximum_fs_raw_spread,
        "maximum_relative_equation_residual": maximum_equation_residual,
        "points": rows,
    }


def main() -> None:
    args = parse_args()
    if args.atlas_points <= 0:
        raise SystemExit("--atlas-points must be positive")
    point_path = args.points.expanduser().resolve()
    tail_path = args.tail_arrays.expanduser().resolve()
    out_path = args.out.expanduser().resolve()
    if not point_path.exists() or not tail_path.exists():
        raise SystemExit("point and tail array files must exist")

    with np.load(point_path, allow_pickle=False) as payload:
        point_metadata = json.loads(str(payload["metadata_json"]))
        coordinates_x = np.asarray(payload["coordinates_x"], dtype=np.complex128)
        coordinates_y = np.asarray(payload["coordinates_y"], dtype=np.complex128)
        coordinates_z = np.asarray(payload["coordinates_z"], dtype=np.complex128)
        cached_base_metrics = np.asarray(payload["base_metrics"], dtype=np.complex128)
        cached_log_omega = np.asarray(payload["log_omega"], dtype=np.float64)
        cached_weights = np.asarray(payload["importance_weights"], dtype=np.float64)
        cluster_ids = np.asarray(payload["cluster_ids"], dtype=np.int64)
        chart_ids = np.asarray(payload["charts"], dtype=np.int64)
    count = len(cached_weights)
    if count % 4 or len(np.unique(cluster_ids)) * 4 != count:
        raise ValueError("cached points do not contain complete four-root fibres")

    model_ratios: dict[str, np.ndarray] = {}
    ratio_sources: dict[str, dict[str, str]] = {}
    with np.load(tail_path, allow_pickle=False) as payload:
        for name, key in (("phi", "phi_normalized_ratio"), ("old_h", "h_normalized_ratio")):
            model_ratios[name] = np.asarray(payload[key], dtype=np.float64)
            ratio_sources[name] = {"path": str(tail_path), "key": key}
    for raw_value in args.ratio:
        name, path, key = parse_ratio_argument(raw_value)
        if name in model_ratios:
            raise ValueError(f"duplicate ratio name {name}")
        with np.load(path, allow_pickle=False) as payload:
            if key not in payload.files:
                raise KeyError(f"{path} has no array {key}")
            model_ratios[name] = np.asarray(payload[key], dtype=np.float64)
        ratio_sources[name] = {"path": str(path), "key": key}
    if any(values.shape != (count,) for values in model_ratios.values()):
        raise ValueError("model ratio arrays do not match the cached point count")

    started = time.perf_counter()
    adapter = get_adapter("p4p1_type11_hirzebruch_x3")
    model = adapter.make_model(args.model_seed, exact=True)
    points = adapter.points_from_storage_payload(
        model,
        {
            "coordinates_x": coordinates_x,
            "coordinates_y": coordinates_y,
            "coordinates_z": coordinates_z,
        },
    )

    cluster_root_separation, point_root_separation = projective_root_separations(
        coordinates_x, cluster_ids
    )
    unique_clusters = np.unique(cluster_ids)
    first_indices = np.asarray(
        [np.flatnonzero(cluster_ids == cluster_id)[0] for cluster_id in unique_clusters]
    )
    y_moment = np.abs(coordinates_y[first_indices, 1]) ** 2
    y_moment /= np.sum(np.abs(coordinates_y[first_indices]) ** 2, axis=1)
    y_ks = kstest(y_moment, "uniform")

    equation_residuals = np.empty(count, dtype=np.float64)
    stabilized_jacobian_minimum = np.empty(count, dtype=np.float64)
    reduced_jacobian_minimum = np.empty(count, dtype=np.float64)
    positive_equation_row_norm = np.empty(count, dtype=np.float64)
    proposal_log_density = np.empty(count, dtype=np.float64)
    omega_log_density = np.empty(count, dtype=np.float64)
    recomputed_log_weight = np.empty(count, dtype=np.float64)
    x3_over_proposal = np.empty(count, dtype=np.float64)
    total_over_proposal = np.empty(count, dtype=np.float64)

    for index, point in enumerate(points):
        equations, jacobian, scales = local_equations_jacobian_and_scales(
            model, point.affine_coordinates, point.projective_chart
        )
        equation_residuals[index] = np.max(
            np.abs(equations) / np.maximum(1.0, scales)
        )
        stabilized_jacobian_minimum[index] = np.linalg.svd(
            jacobian, compute_uv=False
        )[-1]

        # Remove the fixed auxiliary-z equation and compare the p and q
        # gradients in the inverse Fubini--Study cotangent metric.  Normalizing
        # the two rows makes the smallest singular value insensitive to local
        # equation rescalings and exposes near-dependent direct constraints.
        # local_polynomial_data orders the stabilized equations as (p, z, q).
        # The direct P4 x P1 presentation therefore uses rows zero and two.
        reduced_jacobian = jacobian[[0, 2], :5]
        ambient_metric = np.zeros((5, 5), dtype=np.complex128)
        ambient_metric[:4, :4] = fubini_study_metric(
            point.affine_coordinates[:4]
        )
        ambient_metric[4, 4] = fubini_study_metric(
            point.affine_coordinates[4:5]
        )[0, 0]
        inverse_ambient_metric = np.linalg.inv(ambient_metric)
        covector_gram = (
            reduced_jacobian
            @ inverse_ambient_metric
            @ reduced_jacobian.conjugate().T
        )
        covector_norms = np.sqrt(np.real(np.diag(covector_gram)))
        if np.min(covector_norms) <= 1.0e-14:
            raise FloatingPointError("direct constraint gradient is too small")
        normalized_gram = covector_gram / np.outer(
            covector_norms, covector_norms
        )
        normalized_gram = 0.5 * (
            normalized_gram + normalized_gram.conjugate().T
        )
        reduced_jacobian_minimum[index] = math.sqrt(
            max(0.0, float(np.linalg.eigvalsh(normalized_gram)[0]))
        )
        positive_equation_row_norm[index] = np.linalg.norm(
            _y_monomials(
                point.y,
                model.p_tensor.shape[0] - 1,
            )
            @ model.p_tensor
        )
        proposal_log_density[index] = hirzebruch_type21_proposal_log_density(point)
        omega_log_density[index] = hirzebruch_type21_holomorphic_volume_log_density(point)
        recomputed_log_weight[index] = hirzebruch_type21_importance_log_weight(point)

        x_metric = fubini_study_metric(point.affine_coordinates[:4])
        x_tangent = point.tangent_basis[:4, :]
        pulled_x = x_tangent.conjugate().T @ x_metric @ x_tangent
        pulled_x = 0.5 * (pulled_x + pulled_x.conjugate().T)
        x_eigenvalues = np.linalg.eigvalsh(pulled_x)
        base_eigenvalues = np.linalg.eigvalsh(cached_base_metrics[index])
        if x_eigenvalues[0] <= 0 or base_eigenvalues[0] <= 0:
            raise FloatingPointError("topological moment metric is not positive")
        x3_over_proposal[index] = math.exp(
            float(np.sum(np.log(x_eigenvalues))) - proposal_log_density[index]
        )
        total_over_proposal[index] = math.exp(
            float(np.sum(np.log(base_eigenvalues))) - proposal_log_density[index]
        )

    cached_centered_log_weight = np.log(cached_weights)
    cached_centered_log_weight -= np.mean(cached_centered_log_weight)
    recomputed_centered_log_weight = recomputed_log_weight - np.mean(
        recomputed_log_weight
    )
    log_weight_reproduction_error = np.abs(
        cached_centered_log_weight - recomputed_centered_log_weight
    )
    omega_reproduction_error = np.abs(cached_log_omega - omega_log_density)

    diagnostic_arrays = {
        "root_separation": point_root_separation,
        "normalized_reduced_jacobian_min_singular_value": (
            reduced_jacobian_minimum
        ),
        "positive_equation_row_norm": positive_equation_row_norm,
        "log_importance_weight": np.log(cached_weights),
        "proposal_log_density": proposal_log_density,
        "omega_log_density": omega_log_density,
        "y_moment_coordinate": np.repeat(y_moment, 4),
    }
    models = {
        name: model_diagnostics(
            ratio,
            cached_weights,
            chart_ids,
            cluster_ids,
            diagnostic_arrays,
        )
        for name, ratio in model_ratios.items()
    }

    overlaps = {}
    names = list(model_ratios)
    top_count = max(1, count // 100)
    expected_overlap_count = top_count**2 / count
    for left_index, left_name in enumerate(names):
        left = set(np.argsort(model_ratios[left_name])[-top_count:].tolist())
        for right_name in names[left_index + 1 :]:
            right = set(np.argsort(model_ratios[right_name])[-top_count:].tolist())
            overlap_count = len(left & right)
            overlaps[f"{left_name}__{right_name}"] = {
                "top_one_percent_point_overlap": overlap_count,
                "top_one_percent_overlap_fraction": overlap_count / top_count,
                "independent_expected_overlap_count": expected_overlap_count,
                "overlap_enrichment_over_independence": (
                    overlap_count / expected_overlap_count
                ),
                "hypergeometric_upper_tail_pvalue": float(
                    hypergeom.sf(overlap_count - 1, count, top_count, top_count)
                ),
                "jaccard_fraction": overlap_count / (2 * top_count - overlap_count),
                "spearman_raw_ratio": safe_spearman(
                    model_ratios[left_name], model_ratios[right_name]
                ),
                "spearman_abs_log_ratio": safe_spearman(
                    np.abs(np.log(model_ratios[left_name])),
                    np.abs(np.log(model_ratios[right_name])),
                ),
            }

    selected_atlas_indices = []
    rankings = [np.argsort(ratio)[::-1] for ratio in model_ratios.values()]
    rank = 0
    while len(selected_atlas_indices) < args.atlas_points:
        added = False
        for ranking in rankings:
            if rank >= len(ranking):
                continue
            index = int(ranking[rank])
            if index not in selected_atlas_indices:
                selected_atlas_indices.append(index)
                added = True
                if len(selected_atlas_indices) >= args.atlas_points:
                    break
        if not added and rank >= count:
            break
        rank += 1
    atlas = atlas_invariance(adapter, model, points, selected_atlas_indices)

    topological_moments = {
        "jx3_over_three_jx2jy": mean_standard_error(
            x3_over_proposal, EXPECTED_X3_OVER_PROPOSAL
        ),
        "jx_plus_jy_cubed_over_three_jx2jy": mean_standard_error(
            total_over_proposal, EXPECTED_TOTAL_OVER_PROPOSAL
        ),
        "pointwise_identity_total_minus_x": {
            "expected": 1.0,
            "maximum_absolute_error": float(
                np.max(np.abs((total_over_proposal - x3_over_proposal) - 1.0))
            ),
        },
    }
    gates = {
        "complete_four_root_fibres": bool(len(unique_clusters) * 4 == count),
        "base_y_uniform_ks_p_above_1e_3": bool(float(y_ks.pvalue) > 1.0e-3),
        "minimum_root_separation_above_1e_6": bool(
            np.min(cluster_root_separation) > 1.0e-6
        ),
        "minimum_normalized_reduced_jacobian_singular_value_above_1e_6": bool(
            np.min(reduced_jacobian_minimum) > 1.0e-6
        ),
        "minimum_positive_equation_row_norm_above_1e_8": bool(
            np.min(positive_equation_row_norm) > 1.0e-8
        ),
        "maximum_equation_residual_below_1e_8": bool(
            np.max(equation_residuals) < 1.0e-8
        ),
        "cached_importance_weights_reproduce_below_1e_9": bool(
            np.max(log_weight_reproduction_error) < 1.0e-9
        ),
        "cached_omega_density_reproduces_below_1e_9": bool(
            np.max(omega_reproduction_error) < 1.0e-9
        ),
        "topological_moments_within_four_standard_errors": bool(
            abs(topological_moments["jx3_over_three_jx2jy"]["z_score"]) < 4.0
            and abs(
                topological_moments[
                    "jx_plus_jy_cubed_over_three_jx2jy"
                ]["z_score"]
            )
            < 4.0
        ),
        "atlas_weight_spread_below_1e_8": bool(
            atlas["maximum_log_importance_weight_spread"] < 1.0e-8
        ),
        "atlas_fs_raw_spread_below_1e_8": bool(
            atlas["maximum_fs_raw_log_ratio_spread"] < 1.0e-8
        ),
    }
    output = {
        "schema": "type11-sampler-model-coupling-audit-v2",
        "model_seed": int(args.model_seed),
        "point_cache": {
            "path": str(point_path),
            "sha256": sha256_file(point_path),
            "metadata": point_metadata,
            "point_count": count,
            "independent_fibre_count": int(len(unique_clusters)),
        },
        "tail_arrays": {"path": str(tail_path), "sha256": sha256_file(tail_path)},
        "ratio_sources": ratio_sources,
        "sampler_geometry": {
            "base_y_moment_coordinate": {
                "mean": float(np.mean(y_moment)),
                "expected_mean": 0.5,
                "second_moment": float(np.mean(y_moment**2)),
                "expected_second_moment": 1.0 / 3.0,
                "ks_uniform_statistic": float(y_ks.statistic),
                "ks_uniform_pvalue": float(y_ks.pvalue),
            },
            "projective_root_separation": quantiles(cluster_root_separation),
            "relative_equation_residual": quantiles(equation_residuals),
            "stabilized_jacobian_min_singular_value": quantiles(
                stabilized_jacobian_minimum
            ),
            "normalized_reduced_jacobian_min_singular_value": quantiles(
                reduced_jacobian_minimum
            ),
            "positive_equation_row_norm": quantiles(
                positive_equation_row_norm
            ),
            "importance_weight": {
                "effective_sample_size": float(
                    np.sum(cached_weights) ** 2 / np.sum(cached_weights**2)
                ),
                "quantiles": quantiles(cached_weights),
                "maximum_centered_log_reproduction_error": float(
                    np.max(log_weight_reproduction_error)
                ),
            },
            "omega_density": {
                "maximum_cached_reproduction_error": float(
                    np.max(omega_reproduction_error)
                )
            },
            "topological_moments": topological_moments,
            "atlas_invariance": atlas,
        },
        "models": models,
        "model_tail_overlap": overlaps,
        "gates": gates,
        "basic_sampler_gates_passed": bool(all(gates.values())),
        "runtime_seconds": float(time.perf_counter() - started),
        "claim_limit": (
            "Passing these finite-sample geometry and moment checks rejects several "
            "sampler bugs but does not prove exact sampling or global metric accuracy."
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(json_value(output), indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "basic_sampler_gates_passed": output["basic_sampler_gates_passed"],
                "gates": gates,
                "topological_moments": topological_moments,
                "model_maxima": {
                    name: row["ratio_max"] for name, row in models.items()
                },
                "runtime_seconds": output["runtime_seconds"],
            },
            indent=2,
        )
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
