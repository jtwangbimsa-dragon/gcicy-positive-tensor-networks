#!/usr/bin/env python3
"""Build the paired type-(1,1) gCICY batching/capacity comparison."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.special import logsumexp
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import get_adapter  # noqa: E402
from gcicy_metric.simple_patch import fubini_study_metric  # noqa: E402
from gcicy_metric.type21_hirzebruch_x3 import (  # noqa: E402
    _y_monomials,
    hirzebruch_type21_holomorphic_volume_log_density,
    hirzebruch_type21_proposal_log_density,
    local_equations_jacobian_and_scales,
)


ARTIFACTS = {
    "h_k2_point_weighted": (
        "outputs/pipeline/"
        "p4p1_type11_hirzebruch_k2_samepoints_point_weighted_b64_e20.npz"
    ),
    "h_k2_complete_fibres_point_weighted_local_cpu": (
        "outputs/pipeline/"
        "p4p1_type11_hirzebruch_k2_samepoints_"
        "point_weighted_b64_e20_local_cpu.npz"
    ),
    "h_k2_independent_fibres_point_weighted": (
        "outputs/pipeline/"
        "p4p1_type11_hirzebruch_k2_independent_fibres_"
        "point_weighted_b64_e20_local_cpu.npz"
    ),
    "h_k3_point_weighted": (
        "outputs/pipeline/"
        "p4p1_type11_hirzebruch_k3_samepoints_point_weighted_b64_e20.npz"
    ),
    "h_k4_point_weighted": (
        "outputs/pipeline/"
        "p4p1_type11_hirzebruch_k4_samepoints_point_weighted_b64_e20.npz"
    ),
    "h_k4_cluster_weighted": (
        "outputs/pipeline/"
        "p4p1_type11_hirzebruch_k4_samepoints_cluster_weighted_b64_e20.npz"
    ),
    "h_k4_point_uniform": (
        "outputs/pipeline/"
        "p4p1_type11_hirzebruch_k4_samepoints_point_uniform_b64_e20.npz"
    ),
    "h_k4_cluster_uniform": (
        "outputs/pipeline/"
        "p4p1_type11_hirzebruch_k4_samepoints_cluster_uniform_b64_e20.npz"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--point-cache",
        type=Path,
        default=ROOT
        / "outputs/cymetric_phi_gcicy_type11_pilot_8k_e20_seed720xx/"
        "blind_seed72003_n8192.npz",
    )
    parser.add_argument(
        "--phi-arrays",
        type=Path,
        default=ROOT
        / "outputs/cymetric_phi_gcicy_type11_pilot_8k_e20_seed720xx/"
        "blind_tail_arrays.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/pipeline/type11_samepoint_mechanism_20260716",
    )
    return parser.parse_args()


def normalize_log_ratio(raw: np.ndarray, weights: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    log_kappa = float(logsumexp(np.log(weights) + raw) - np.log(np.sum(weights)))
    return np.exp(raw - log_kappa)


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    value = spearmanr(left, right).statistic
    return float(value) if np.isfinite(value) else 0.0


def cluster_bootstrap_metric_deltas(
    source_ratio: np.ndarray,
    target_ratio: np.ndarray,
    weights: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    seed: int = 20260717,
    replications: int = 4000,
) -> dict[str, Any]:
    """Bootstrap paired metric changes while preserving complete fibres."""

    _, inverse = np.unique(cluster_ids, return_inverse=True)
    cluster_count = int(np.max(inverse)) + 1

    def aggregate(values: np.ndarray) -> np.ndarray:
        return np.bincount(inverse, weights=values, minlength=cluster_count)

    cluster_weights = aggregate(weights)
    source_sigma = aggregate(weights * np.abs(1.0 - source_ratio))
    target_sigma = aggregate(weights * np.abs(1.0 - target_ratio))
    source_l2 = aggregate(weights * np.square(1.0 - source_ratio))
    target_l2 = aggregate(weights * np.square(1.0 - target_ratio))
    source_tail = aggregate(weights * (source_ratio > 3.0))
    target_tail = aggregate(weights * (target_ratio > 3.0))

    rng = np.random.default_rng(seed)
    deltas = {
        "sigma": np.empty(replications, dtype=np.float64),
        "l2": np.empty(replications, dtype=np.float64),
        "ratio_above_3_weighted_mass": np.empty(replications, dtype=np.float64),
    }
    chunk_size = 100
    for start in range(0, replications, chunk_size):
        stop = min(start + chunk_size, replications)
        draws = rng.integers(
            0,
            cluster_count,
            size=(stop - start, cluster_count),
        )
        denominator = np.sum(cluster_weights[draws], axis=1)
        source_sigma_value = np.sum(source_sigma[draws], axis=1) / denominator
        target_sigma_value = np.sum(target_sigma[draws], axis=1) / denominator
        source_l2_value = np.sqrt(np.sum(source_l2[draws], axis=1) / denominator)
        target_l2_value = np.sqrt(np.sum(target_l2[draws], axis=1) / denominator)
        source_tail_value = np.sum(source_tail[draws], axis=1) / denominator
        target_tail_value = np.sum(target_tail[draws], axis=1) / denominator
        deltas["sigma"][start:stop] = target_sigma_value - source_sigma_value
        deltas["l2"][start:stop] = target_l2_value - source_l2_value
        deltas["ratio_above_3_weighted_mass"][start:stop] = (
            target_tail_value - source_tail_value
        )

    weight_sum = float(np.sum(weights))
    observed = {
        "sigma": (
            float(np.sum(weights * np.abs(1.0 - source_ratio)) / weight_sum),
            float(np.sum(weights * np.abs(1.0 - target_ratio)) / weight_sum),
        ),
        "l2": (
            float(
                math.sqrt(
                    np.sum(weights * np.square(1.0 - source_ratio)) / weight_sum
                )
            ),
            float(
                math.sqrt(
                    np.sum(weights * np.square(1.0 - target_ratio)) / weight_sum
                )
            ),
        ),
        "ratio_above_3_weighted_mass": (
            float(np.sum(weights[source_ratio > 3.0]) / weight_sum),
            float(np.sum(weights[target_ratio > 3.0]) / weight_sum),
        ),
    }
    metrics: dict[str, Any] = {}
    for name, (source_value, target_value) in observed.items():
        sample = deltas[name]
        metrics[name] = {
            "source": source_value,
            "target": target_value,
            "target_minus_source": target_value - source_value,
            "relative_improvement": (source_value - target_value) / source_value,
            "bootstrap_target_minus_source_ci95": [
                float(np.quantile(sample, 0.025)),
                float(np.quantile(sample, 0.975)),
            ],
            "bootstrap_probability_target_lower": float(np.mean(sample < 0.0)),
        }
    return {
        "resampling_unit": "complete_four-root_blind_fibre",
        "cluster_count": cluster_count,
        "replications": replications,
        "seed": seed,
        "metrics": metrics,
    }


def ratio_summary(
    ratio: np.ndarray,
    weights: np.ndarray,
    cluster_ids: np.ndarray,
    chart_ids: np.ndarray,
) -> dict[str, Any]:
    ratio = np.asarray(ratio, dtype=np.float64)
    abs_log = np.abs(np.log(ratio))
    weight_sum = float(np.sum(weights))
    worst = int(np.argmax(ratio))
    worst_cluster = int(cluster_ids[worst])
    cluster_indices = np.flatnonzero(cluster_ids == worst_cluster)
    top_count = max(1, len(ratio) // 100)
    top_indices = np.argsort(ratio, kind="stable")[-top_count:]
    return {
        "sigma": float(np.sum(weights * np.abs(1.0 - ratio)) / weight_sum),
        "l2": float(
            math.sqrt(np.sum(weights * np.square(1.0 - ratio)) / weight_sum)
        ),
        "ratio_min": float(np.min(ratio)),
        "ratio_max": float(np.max(ratio)),
        "ratio_q99": float(np.quantile(ratio, 0.99)),
        "ratio_q999": float(np.quantile(ratio, 0.999)),
        "abs_log_q999": float(np.quantile(abs_log, 0.999)),
        "ratio_above_3_count": int(np.sum(ratio > 3.0)),
        "ratio_above_3_weighted_mass": float(
            np.sum(weights[ratio > 3.0]) / weight_sum
        ),
        "top_one_percent_independent_clusters": int(
            len(np.unique(cluster_ids[top_indices]))
        ),
        "worst_point": {
            "index": worst,
            "cluster_id": worst_cluster,
            "chart_id": int(chart_ids[worst]),
            "importance_weight": float(weights[worst]),
            "four_root_cluster_ratios": [
                float(value) for value in ratio[cluster_indices]
            ],
        },
    }


def direct_geometry_diagnostics(
    adapter: Any,
    model: Any,
    coordinates_x: np.ndarray,
    coordinates_y: np.ndarray,
    coordinates_z: np.ndarray,
    cluster_ids: np.ndarray,
    index: int,
) -> dict[str, Any]:
    point = adapter.points_from_storage_payload(
        model,
        {
            "coordinates_x": coordinates_x[index : index + 1],
            "coordinates_y": coordinates_y[index : index + 1],
            "coordinates_z": coordinates_z[index : index + 1],
        },
    )[0]
    equations, jacobian, scales = local_equations_jacobian_and_scales(
        model, point.affine_coordinates, point.projective_chart
    )
    reduced_jacobian = jacobian[[0, 2], :5]
    ambient_metric = np.zeros((5, 5), dtype=np.complex128)
    ambient_metric[:4, :4] = fubini_study_metric(point.affine_coordinates[:4])
    ambient_metric[4, 4] = fubini_study_metric(
        point.affine_coordinates[4:5]
    )[0, 0]
    inverse_metric = np.linalg.inv(ambient_metric)
    gram = reduced_jacobian @ inverse_metric @ reduced_jacobian.conjugate().T
    norms = np.sqrt(np.real(np.diag(gram)))
    normalized_gram = gram / np.outer(norms, norms)
    normalized_gram = 0.5 * (normalized_gram + normalized_gram.conjugate().T)
    direct_minimum = math.sqrt(
        max(0.0, float(np.linalg.eigvalsh(normalized_gram)[0]))
    )
    p_row_norm = float(
        np.linalg.norm(
            _y_monomials(point.y, model.p_tensor.shape[0] - 1)
            @ model.p_tensor
        )
    )

    cluster_indices = np.flatnonzero(cluster_ids == cluster_ids[index])
    unit_roots = coordinates_x[cluster_indices]
    unit_roots = unit_roots / np.linalg.norm(unit_roots, axis=1, keepdims=True)
    root_separations = []
    for left in range(len(unit_roots)):
        for right in range(left + 1, len(unit_roots)):
            overlap = min(
                1.0,
                float(abs(np.vdot(unit_roots[left], unit_roots[right]))),
            )
            root_separations.append(math.sqrt(max(0.0, 1.0 - overlap**2)))
    return {
        "relative_equation_residual": float(
            np.max(np.abs(equations) / np.maximum(1.0, scales))
        ),
        "normalized_reduced_jacobian_min_singular_value": direct_minimum,
        "positive_equation_row_norm": p_row_norm,
        "four_root_minimum_projective_separation": float(min(root_separations)),
        "residue_denominator_abs": float(abs(point.residue_denominator)),
        "proposal_log_density": hirzebruch_type21_proposal_log_density(point),
        "omega_log_density": hirzebruch_type21_holomorphic_volume_log_density(
            point
        ),
    }


def main() -> None:
    args = parse_args()
    point_cache = args.point_cache.expanduser().resolve()
    phi_arrays = args.phi_arrays.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    with np.load(point_cache, allow_pickle=False) as payload:
        weights = np.asarray(payload["importance_weights"], dtype=np.float64)
        cluster_ids = np.asarray(payload["cluster_ids"], dtype=np.int64)
        chart_ids = np.asarray(payload["charts"], dtype=np.int64)
        coordinates_x = np.asarray(payload["coordinates_x"], dtype=np.complex128)
        coordinates_y = np.asarray(payload["coordinates_y"], dtype=np.complex128)
        coordinates_z = np.asarray(payload["coordinates_z"], dtype=np.complex128)

    ratios: dict[str, np.ndarray] = {}
    sources: dict[str, str] = {}
    with np.load(phi_arrays, allow_pickle=False) as payload:
        ratios["fubini_study"] = np.asarray(
            payload["fs_normalized_ratio"], dtype=np.float64
        )
        ratios["phi"] = np.asarray(payload["phi_normalized_ratio"], dtype=np.float64)
        ratios["old_h_reference"] = np.asarray(
            payload["h_normalized_ratio"], dtype=np.float64
        )
    sources["fubini_study"] = str(phi_arrays)
    sources["phi"] = str(phi_arrays)
    sources["old_h_reference"] = str(phi_arrays)

    training_metadata: dict[str, Any] = {}
    for name, relative_path in ARTIFACTS.items():
        artifact_path = (ROOT / relative_path).resolve()
        summary_path = artifact_path.with_name(
            artifact_path.stem + "_summary.json"
        )
        with np.load(artifact_path, allow_pickle=False) as payload:
            artifact_weights = np.asarray(
                payload["importance_weights"], dtype=np.float64
            )
            if not np.allclose(artifact_weights, weights, rtol=0.0, atol=1.0e-7):
                raise ValueError(f"{name} does not use the common blind weights")
            ratios[name] = normalize_log_ratio(
                payload["corrected_log_ma"], artifact_weights
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        optimization = summary["optimization"]
        training_metadata[name] = {
            "artifact": str(artifact_path),
            "summary": str(summary_path),
            "degree": summary["request"]["degree"],
            "restricted_section_count": summary["restricted_section_count"],
            "scale_invariant_h_dimension_upper_bound": optimization[
                "scale_invariant_h_dimension_upper_bound"
            ],
            "optimizer_mode": optimization["group_optimizer_step_mode"],
            "training_sampling_mode": optimization.get(
                "training_sampling_mode", "complete_fibres"
            ),
            "training_sampling_cluster_count": optimization.get(
                "training_sampling_cluster_count",
                summary["request"]["train_points"] // 4,
            ),
            "uniform_loss": bool(
                optimization["point_minibatch_uniform_loss"]
                or optimization["cluster_minibatch_uniform_loss"]
            ),
            "optimizer_steps": optimization["optimizer_step_count"],
            "best_epoch": summary["best_epoch"],
            "relative_h_log_eigenvalue_span": optimization[
                "selected_relative_h_update"
            ]["relative_to_initial_log_eigenvalue_span"],
        }
        observed_sigma = ratio_summary(
            ratios[name], weights, cluster_ids, chart_ids
        )["sigma"]
        if not np.isclose(
            observed_sigma,
            summary["export_candidate"]["sigma"],
            rtol=2.0e-5,
            atol=2.0e-6,
        ):
            raise ValueError(f"{name} export sigma does not reproduce its summary")

    summaries = {
        name: ratio_summary(ratio, weights, cluster_ids, chart_ids)
        for name, ratio in ratios.items()
    }
    adapter = get_adapter("p4p1_type11_hirzebruch_x3")
    model = adapter.make_model(20260731, exact=True)
    geometry_by_index: dict[int, dict[str, Any]] = {}
    for summary in summaries.values():
        index = int(summary["worst_point"]["index"])
        if index not in geometry_by_index:
            geometry_by_index[index] = direct_geometry_diagnostics(
                adapter,
                model,
                coordinates_x,
                coordinates_y,
                coordinates_z,
                cluster_ids,
                index,
            )
        summary["worst_point"]["geometry"] = geometry_by_index[index]
    pairwise: dict[str, Any] = {}
    names = list(ratios)
    top_count = max(1, len(weights) // 100)
    for left_index, left_name in enumerate(names):
        left_ratio = ratios[left_name]
        left_top = set(np.argsort(left_ratio)[-top_count:].tolist())
        for right_name in names[left_index + 1 :]:
            right_ratio = ratios[right_name]
            right_top = set(np.argsort(right_ratio)[-top_count:].tolist())
            pairwise[f"{left_name}__{right_name}"] = {
                "spearman_ratio": safe_spearman(left_ratio, right_ratio),
                "spearman_abs_log_ratio": safe_spearman(
                    np.abs(np.log(left_ratio)), np.abs(np.log(right_ratio))
                ),
                "top_one_percent_point_overlap": len(left_top & right_top),
                "top_one_percent_overlap_fraction": len(left_top & right_top)
                / top_count,
                "same_worst_point": int(np.argmax(left_ratio))
                == int(np.argmax(right_ratio)),
                "same_worst_cluster": int(cluster_ids[np.argmax(left_ratio)])
                == int(cluster_ids[np.argmax(right_ratio)]),
            }

    causal_controls = {
        "k2_independent_fibres_vs_complete_fibres": cluster_bootstrap_metric_deltas(
            ratios["h_k2_complete_fibres_point_weighted_local_cpu"],
            ratios["h_k2_independent_fibres_point_weighted"],
            weights,
            cluster_ids,
        )
    }

    output = {
        "schema": "type11-samepoint-mechanism-comparison-v2",
        "point_cache": str(point_cache),
        "point_count": int(len(weights)),
        "independent_four_root_fibres": int(len(np.unique(cluster_ids))),
        "train_seed": 72001,
        "checkpoint_seed": 72002,
        "blind_seed": 72003,
        "training_points": 8192,
        "epochs": 20,
        "batch_points": 64,
        "optimizer_steps": 2560,
        "sources": sources,
        "training_metadata": training_metadata,
        "model_summaries": summaries,
        "pairwise": pairwise,
        "causal_controls": causal_controls,
        "interpretation_limit": (
            "All maxima are observations on the common 8192-point blind sample, "
            "not deterministic global sup-norm bounds."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    ratio_path = output_dir / "samepoint_ratios.npz"
    np.savez_compressed(
        ratio_path,
        importance_weights=weights,
        cluster_ids=cluster_ids,
        chart_ids=chart_ids,
        **{f"{name}_normalized_ratio": ratio for name, ratio in ratios.items()},
    )
    output["ratio_arrays"] = str(ratio_path)
    report_path = output_dir / "comparison.json"
    report_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    rows = []
    for name, summary in summaries.items():
        metadata = training_metadata.get(name, {})
        rows.append(
            "| {name} | {sections} | {parameters} | {sigma:.4f} | {l2:.4f} | "
            "{maximum:.2f} | {mass:.3%} | {worst} |".format(
                name=name,
                sections=metadata.get("restricted_section_count", "--"),
                parameters=metadata.get(
                    "scale_invariant_h_dimension_upper_bound", "--"
                ),
                sigma=summary["sigma"],
                l2=summary["l2"],
                maximum=summary["ratio_max"],
                mass=summary["ratio_above_3_weighted_mass"],
                worst=summary["worst_point"]["index"],
            )
        )
    markdown = "\n".join(
        [
            "# Type-(1,1) same-point mechanism comparison",
            "",
            "All models below are evaluated on blind seed 72003 with 8192 points.",
            "",
            "| Model | Sections | H parameters | Sigma | L2 | Max r | mass(r>3) | Worst index |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            output["interpretation_limit"],
            "",
        ]
    )
    markdown_path = output_dir / "comparison.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"wrote {ratio_path}")
    print(f"wrote {report_path}")
    print(f"wrote {markdown_path}")


if __name__ == "__main__":
    main()
