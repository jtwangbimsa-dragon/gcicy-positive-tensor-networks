#!/usr/bin/env python3
"""Summarize same-point type-(2,2) tensor-network and PhiFS results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline.risk import (  # noqa: E402
    normalized_positive_weights_numpy,
    weighted_cvar_numpy,
    weighted_log_mean_exp_numpy,
)


PHI_ARMS = (
    "depth3_width32",
    "depth3_width64",
    "depth5_width48",
    "depth3_width128",
)

POINT_IDENTITY_KEYS = (
    "coordinates_x",
    "coordinates_y",
    "coordinates_z",
    "projective_charts",
    "independent_indices",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor-run", type=Path, required=True)
    parser.add_argument("--phi-run", type=Path, required=True)
    parser.add_argument("--tensor-manifest", type=Path, required=True)
    parser.add_argument("--phi-manifest", type=Path, required=True)
    parser.add_argument("--official-quintic-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    probability: float,
) -> float:
    probabilities = normalized_positive_weights_numpy(weights)
    order = np.argsort(values, kind="stable")
    sorted_values = np.asarray(values, dtype=np.float64)[order]
    sorted_weights = probabilities[order]
    midpoint_cdf = np.cumsum(sorted_weights) - 0.5 * sorted_weights
    return float(
        np.interp(
            probability,
            midpoint_cdf,
            sorted_values,
            left=sorted_values[0],
            right=sorted_values[-1],
        )
    )


def self_normalized_metrics(
    raw_log_ratio: np.ndarray,
    weights: np.ndarray,
    *,
    minimum_metric_eigenvalue: float,
) -> dict[str, float]:
    raw = np.asarray(raw_log_ratio, dtype=np.float64)
    probabilities = normalized_positive_weights_numpy(weights)
    if raw.shape != probabilities.shape or not np.all(np.isfinite(raw)):
        raise ValueError("raw log ratios and weights must be aligned and finite")
    normalization = weighted_log_mean_exp_numpy(raw, weights)
    centered = raw - normalization
    ratio = np.exp(np.clip(centered, -745.0, 709.0))
    residual = 1.0 - ratio
    absolute_log = np.abs(centered)
    return {
        "normalization_log_kappa": normalization,
        "sigma": float(np.sum(probabilities * np.abs(residual))),
        "chi": float(np.sqrt(np.sum(probabilities * residual**2))),
        "absolute_log_ratio_q999": weighted_quantile(
            absolute_log,
            weights,
            0.999,
        ),
        "absolute_log_ratio_cvar_1pct": weighted_cvar_numpy(
            absolute_log,
            weights,
            tail_fraction=0.01,
        ).value,
        "normalized_ratio_min": float(np.min(ratio)),
        "normalized_ratio_max": float(np.max(ratio)),
        "weighted_probability_r_above_3": float(
            np.sum(probabilities[ratio > 3.0])
        ),
        "minimum_metric_eigenvalue": float(minimum_metric_eigenvalue),
        "importance_effective_sample_size": float(
            1.0 / np.sum(probabilities**2)
        ),
    }


def load_tensor_arm(
    run_dir: Path,
    label: str,
    *,
    blind_seed: int,
) -> dict[str, Any]:
    summary_path = run_dir / f"{label}_summary.json"
    blind_path = run_dir / f"{label}_blind{blind_seed}_n65532.json"
    arrays_path = run_dir / f"{label}_blind{blind_seed}_n65532_arrays.npz"
    summary = load_json(summary_path)
    blind = load_json(blind_path)
    with np.load(arrays_path, allow_pickle=False) as payload:
        raw = np.asarray(payload["model_log_eta"], dtype=np.float64)
        weights = np.asarray(payload["importance_weights"], dtype=np.float64)
        clusters = np.asarray(payload["sampling_cluster_ids"], dtype=np.int64)
        point_identity = {
            key: np.asarray(payload[key]) for key in POINT_IDENTITY_KEYS
        }
    metrics = self_normalized_metrics(
        raw,
        weights,
        minimum_metric_eigenvalue=blind["metrics"]["minimum_metric_eigenvalue"],
    )
    return {
        "family": "positive_tensor_network",
        "label": label,
        "site_count": int(summary["site_count"]),
        "physical_dictionary_rank": int(summary["physical_dictionary_rank"]),
        "trainable_real_parameter_count": int(
            summary["trainable_real_parameter_count"]
        ),
        "best_validation_epoch": int(summary["best_epoch"]),
        "runtime_seconds": float(summary["runtime_seconds"]),
        "metrics": metrics,
        "weights": weights,
        "clusters": clusters,
        "point_identity": point_identity,
        "artifacts": {
            "summary": str(summary_path),
            "summary_sha256": sha256_file(summary_path),
            "blind": str(blind_path),
            "blind_sha256": sha256_file(blind_path),
            "arrays": str(arrays_path),
            "arrays_sha256": sha256_file(arrays_path),
            "model_sha256": summary["model_sha256"],
        },
    }


def load_phi_arm(run_dir: Path, label: str) -> dict[str, Any]:
    report_path = run_dir / label / "report.json"
    arrays_path = run_dir / label / "blind_tail_arrays.npz"
    report = load_json(report_path)
    with np.load(arrays_path, allow_pickle=False) as payload:
        raw = np.asarray(payload["phi_raw_log_ratio"], dtype=np.float64)
        weights = np.asarray(payload["importance_weights"], dtype=np.float64)
        clusters = np.asarray(payload["cluster_ids"], dtype=np.int64)
        minimum = float(np.min(payload["phi_min_eigenvalue"]))
        point_identity = {
            key: np.asarray(payload[key]) for key in POINT_IDENTITY_KEYS
        }
    metrics = self_normalized_metrics(
        raw,
        weights,
        minimum_metric_eigenvalue=minimum,
    )
    return {
        "family": "cymetric_style_phifs",
        "label": label,
        "hidden_layers": int(report["network"]["hidden_layers"]),
        "hidden_width": int(report["network"]["hidden_width"]),
        "trainable_real_parameter_count": int(report["network"]["parameter_count"]),
        "best_validation_epoch": int(report["training"]["best_epoch"]),
        "runtime_seconds": float(report["timing_seconds"]["optimization"]),
        "maximum_atlas_log_ratio_spread": float(
            report["atlas"]["max_raw_log_ratio_spread"]
        ),
        "preregistered_protocol": report["preregistered_protocol"],
        "metrics": metrics,
        "weights": weights,
        "clusters": clusters,
        "point_identity": point_identity,
        "artifacts": {
            "report": str(report_path),
            "report_sha256": sha256_file(report_path),
            "arrays": str(arrays_path),
            "arrays_sha256": sha256_file(arrays_path),
            "model_sha256": report["files"]["checkpoint_sha256"],
        },
    }


def relative_improvement(reference: float, candidate: float) -> float:
    return float((reference - candidate) / reference)


def public_arm(arm: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in arm.items()
        if key not in {"weights", "clusters", "point_identity"}
    }


def maximum_absolute_difference(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return float("inf")
    return float(np.max(np.abs(left - right))) if left.size else 0.0


def main() -> None:
    args = parse_args()
    tensor_run = args.tensor_run.expanduser().resolve()
    phi_run = args.phi_run.expanduser().resolve()
    tensor_manifest_path = args.tensor_manifest.expanduser().resolve()
    phi_manifest_path = args.phi_manifest.expanduser().resolve()
    official_path = args.official_quintic_report.expanduser().resolve()

    tensor_manifest = load_json(tensor_manifest_path)
    phi_manifest = load_json(phi_manifest_path)
    phi_manifest_hash = sha256_file(phi_manifest_path)
    q22 = load_tensor_arm(
        tensor_run,
        "active_q22_k4_control",
        blind_seed=83703,
    )
    q60 = load_tensor_arm(
        tensor_run,
        "active_q60_k4_capacity",
        blind_seed=83703,
    )
    q60_k6 = load_tensor_arm(
        tensor_run,
        "active_q60_k6_degree_control",
        blind_seed=83713,
    )
    phi = {label: load_phi_arm(phi_run, label) for label in PHI_ARMS}

    phi_protocol_checks = {}
    for label, arm in phi.items():
        protocol = arm["preregistered_protocol"]
        declared_parameters = phi_manifest["arms"][label][
            "trainable_real_parameter_count"
        ]
        phi_protocol_checks[label] = {
            "manifest_sha256_matches": protocol["sha256"] == phi_manifest_hash,
            "registered_arm_matches": protocol["registered_arm"] == label,
            "parameter_count_matches": (
                arm["trainable_real_parameter_count"] == declared_parameters
            ),
        }
    phi_protocol_gate = all(
        all(checks.values()) for checks in phi_protocol_checks.values()
    )

    same_point_arms = {"q22_k4": q22, "q60_k4": q60, **phi}
    reference_weights = q22["weights"]
    reference_clusters = q22["clusters"]
    reference_points = q22["point_identity"]
    same_point_checks = {}
    for label, arm in same_point_arms.items():
        weights = arm["weights"]
        clusters = arm["clusters"]
        if weights.shape != reference_weights.shape:
            raise ValueError(f"{label} weight shape does not match q22")
        same_point_checks[label] = {
            "maximum_absolute_importance_weight_difference": float(
                np.max(np.abs(weights - reference_weights))
            ),
            "cluster_ids_exactly_equal": bool(
                np.array_equal(clusters, reference_clusters)
            ),
            "maximum_absolute_coordinate_difference": max(
                maximum_absolute_difference(
                    arm["point_identity"][key],
                    reference_points[key],
                )
                for key in ("coordinates_x", "coordinates_y", "coordinates_z")
            ),
            "projective_charts_exactly_equal": bool(
                np.array_equal(
                    arm["point_identity"]["projective_charts"],
                    reference_points["projective_charts"],
                )
            ),
            "independent_indices_exactly_equal": bool(
                np.array_equal(
                    arm["point_identity"]["independent_indices"],
                    reference_points["independent_indices"],
                )
            ),
        }
    same_point_gate = all(
        row["maximum_absolute_importance_weight_difference"] <= 1e-12
        and row["cluster_ids_exactly_equal"]
        and row["maximum_absolute_coordinate_difference"] <= 1e-12
        and row["projective_charts_exactly_equal"]
        and row["independent_indices_exactly_equal"]
        for row in same_point_checks.values()
    )

    width32 = phi["depth3_width32"]
    width64 = phi["depth3_width64"]
    width128 = phi["depth3_width128"]
    depth48 = phi["depth5_width48"]
    capacity_gate = {
        "chi": q60["metrics"]["chi"] <= 0.90 * q22["metrics"]["chi"],
        "absolute_log_ratio_q999": (
            q60["metrics"]["absolute_log_ratio_q999"]
            <= 1.05 * q22["metrics"]["absolute_log_ratio_q999"] + 0.02
        ),
        "absolute_log_ratio_cvar_1pct": (
            q60["metrics"]["absolute_log_ratio_cvar_1pct"]
            <= 1.05 * q22["metrics"]["absolute_log_ratio_cvar_1pct"] + 0.02
        ),
        "weighted_probability_r_above_3": (
            q60["metrics"]["weighted_probability_r_above_3"]
            <= q22["metrics"]["weighted_probability_r_above_3"] + 0.002
        ),
        "minimum_metric_eigenvalue": (
            q60["metrics"]["minimum_metric_eigenvalue"] > 0
        ),
    }
    capacity_gate["all"] = all(capacity_gate.values())

    official = load_json(official_path)
    report = {
        "schema": "type22-tensor-network-phifs-same-point-comparison-v1",
        "metric_definition": (
            "Each model is self-normalized on the identical blind pool using the "
            "importance-weighted log-mean-exp of log(det(g)/|Omega|^2)."
        ),
        "protocols": {
            "tensor_manifest": str(tensor_manifest_path),
            "tensor_manifest_sha256": sha256_file(tensor_manifest_path),
            "tensor_manifest_schema": tensor_manifest["schema"],
            "phi_manifest": str(phi_manifest_path),
            "phi_manifest_sha256": phi_manifest_hash,
            "phi_manifest_schema": phi_manifest["schema"],
            "phi_arm_checks": phi_protocol_checks,
            "all_phi_arm_gates_pass": phi_protocol_gate,
        },
        "same_point_validation": {
            "blind_seed": 83703,
            "blind_points": 65532,
            "checks": same_point_checks,
            "all_gates_pass": same_point_gate,
        },
        "same_point_models": {
            label: public_arm(arm) for label, arm in same_point_arms.items()
        },
        "degree_control": {"q60_k6": public_arm(q60_k6)},
        "tensor_dictionary_capacity_gate": capacity_gate,
        "network_relationships": {
            "width_effect": {
                "chi_improvement_width32_to_width64": relative_improvement(
                    width32["metrics"]["chi"],
                    width64["metrics"]["chi"],
                ),
                "chi_improvement_width64_to_width128": relative_improvement(
                    width64["metrics"]["chi"],
                    width128["metrics"]["chi"],
                ),
                "sigma_improvement_width32_to_width64": relative_improvement(
                    width32["metrics"]["sigma"],
                    width64["metrics"]["sigma"],
                ),
                "sigma_improvement_width64_to_width128": relative_improvement(
                    width64["metrics"]["sigma"],
                    width128["metrics"]["sigma"],
                ),
            },
            "near_parameter_matched_depth_effect": {
                "depth3_width64_parameters": width64[
                    "trainable_real_parameter_count"
                ],
                "depth5_width48_parameters": depth48[
                    "trainable_real_parameter_count"
                ],
                "chi_improvement_deeper_relative_to_shallower": relative_improvement(
                    width64["metrics"]["chi"],
                    depth48["metrics"]["chi"],
                ),
                "sigma_improvement_deeper_relative_to_shallower": relative_improvement(
                    width64["metrics"]["sigma"],
                    depth48["metrics"]["sigma"],
                ),
            },
            "near_parameter_matched_ansatz_effect": {
                "phifs_depth3_width128_parameters": width128[
                    "trainable_real_parameter_count"
                ],
                "tensor_q60_k4_parameters": q60["trainable_real_parameter_count"],
                "phifs_to_tensor_parameter_ratio": (
                    width128["trainable_real_parameter_count"]
                    / q60["trainable_real_parameter_count"]
                ),
                "chi_improvement_phi_relative_to_tensor": relative_improvement(
                    q60["metrics"]["chi"],
                    width128["metrics"]["chi"],
                ),
                "sigma_improvement_phi_relative_to_tensor": relative_improvement(
                    q60["metrics"]["sigma"],
                    width128["metrics"]["sigma"],
                ),
            },
        },
        "official_quintic_external_control": {
            "report": str(official_path),
            "report_sha256": sha256_file(official_path),
            "geometry": "Fermat quintic",
            "parameter_count": official["network"]["parameter_count"],
            "hidden_layers": official["network"]["hidden_layers"],
            "hidden_width": official["network"]["hidden_width"],
            "blind_points": official["trained_phi_model"]["n_points"],
            "sigma": official["trained_phi_model"]["sigma_official_formula"],
            "chi": official["trained_phi_model"]["weighted_rms_abs_residual"],
            "normalized_ratio_min": official["trained_phi_model"][
                "ratio_weighted_quantiles"
            ]["q0.0000"],
            "normalized_ratio_max": official["trained_phi_model"][
                "ratio_weighted_quantiles"
            ]["q1.0000"],
            "comparison_boundary": (
                "This is an official ordinary-CICY control, not a same-geometry "
                "or same-sampler comparator."
            ),
        },
        "claim_boundary": phi_manifest["claim_boundary"],
    }
    if not same_point_gate:
        raise SystemExit("same-point coordinate, chart, weight, or cluster gate failed")
    if not phi_protocol_gate:
        raise SystemExit("PhiFS preregistered protocol gate failed")
    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_suffix(out_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(out_path)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
