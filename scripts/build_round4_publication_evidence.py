#!/usr/bin/env python3
"""Build the frozen round-four robustness evidence and manuscript tables."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metric-audit",
        type=Path,
        default=ROOT
        / "outputs/pipeline/p4p1_type11_hirzebruch_k4_training_replicas_audit.json",
    )
    parser.add_argument(
        "--scalar-audit",
        type=Path,
        default=ROOT
        / "outputs/pipeline/p4p1_type11_hirzebruch_scalar_training_replicas_32768.json",
    )
    parser.add_argument(
        "--scalar-stress-audit",
        type=Path,
        default=ROOT
        / "outputs/pipeline/p4p1_type11_hirzebruch_k4_training_replicas_seed28601_stress.json",
    )
    parser.add_argument(
        "--scalar-tail-diagnostic",
        type=Path,
        default=ROOT
        / "outputs/pipeline/p4p1_type11_hirzebruch_scalar_training_replicas_seed28601_weight_tail.json",
    )
    parser.add_argument(
        "--hard-region-training-summary",
        type=Path,
        default=ROOT
        / "outputs/pipeline/p4p1_type11_hirzebruch_k4_hard_region_refined_pilot_summary.json",
    )
    parser.add_argument(
        "--hard-region-discovery-audit",
        type=Path,
        default=ROOT
        / "outputs/pipeline/p4p1_type11_hirzebruch_k4_hard_region_discovery_audit.json",
    )
    parser.add_argument(
        "--hard-region-blind-audit",
        type=Path,
        default=ROOT
        / "outputs/pipeline/p4p1_type11_hirzebruch_k4_hard_region_blind_audit.json",
    )
    parser.add_argument(
        "--hard-region-discovery-scalar",
        type=Path,
        default=ROOT
        / "outputs/pipeline/p4p1_type11_hirzebruch_k4_hard_region_discovery_scalar.json",
    )
    parser.add_argument(
        "--hard-region-parallel-tail-diagnostic",
        type=Path,
        default=ROOT
        / "outputs/pipeline/p4p1_type11_hirzebruch_k4_hard_region_refined_seed68507_parallel_weight_tail.json",
    )
    parser.add_argument(
        "--multiseed-training-summary",
        type=Path,
        default=ROOT
        / "outputs/pipeline/p4p1_type11_hirzebruch_k4_multiseed_tail_refined_gpu_summary.json",
    )
    parser.add_argument(
        "--multiseed-discovery-serial",
        type=Path,
        default=ROOT
        / "outputs/pipeline/p4p1_type11_hirzebruch_k4_multiseed_discovery_serial.json",
    )
    parser.add_argument(
        "--multiseed-discovery-parallel",
        type=Path,
        default=ROOT
        / "outputs/pipeline/p4p1_type11_hirzebruch_k4_multiseed_discovery_parallel.json",
    )
    parser.add_argument(
        "--multiseed-discovery-scalar",
        type=Path,
        default=ROOT
        / "outputs/pipeline/p4p1_type11_hirzebruch_k4_multiseed_discovery_scalar.json",
    )
    parser.add_argument(
        "--multiseed-discovery-parallel-tail",
        type=Path,
        default=ROOT
        / "outputs/pipeline/p4p1_type11_hirzebruch_k4_multiseed_seed68507_parallel_weight_tail.json",
    )
    parser.add_argument(
        "--multiseed-blind-audit",
        type=Path,
        default=ROOT
        / "outputs/pipeline/p4p1_type11_hirzebruch_k4_multiseed_blind_audit.json",
    )
    parser.add_argument(
        "--multiseed-blind-scalar",
        type=Path,
        default=ROOT
        / "outputs/pipeline/p4p1_type11_hirzebruch_k4_multiseed_blind_scalar.json",
    )
    parser.add_argument(
        "--multiseed-blind-tail-diagnostic",
        type=Path,
        default=ROOT
        / "outputs/pipeline/p4p1_type11_hirzebruch_k4_multiseed_blind_seed68705_weight_tail.json",
    )
    parser.add_argument(
        "--l2-v1-training-summary",
        type=Path,
        default=ROOT
        / "outputs/pipeline/p4p1_type11_hirzebruch_k4_l2_tail_refined_gpu_summary.json",
    )
    parser.add_argument(
        "--l2-v1-discovery-audit",
        type=Path,
        default=ROOT
        / "outputs/pipeline/p4p1_type11_hirzebruch_k4_l2_tail_discovery_audit.json",
    )
    parser.add_argument(
        "--l2-v2-training-summary",
        type=Path,
        default=ROOT
        / "outputs/pipeline/p4p1_type11_hirzebruch_k4_l2_tail_refined_v2_gpu_summary.json",
    )
    parser.add_argument(
        "--l2-v2-discovery-audit",
        type=Path,
        default=ROOT
        / "outputs/pipeline/p4p1_type11_hirzebruch_k4_l2_tail_discovery_v2_audit.json",
    )
    parser.add_argument(
        "--l2-v2-process-discovery",
        type=Path,
        default=ROOT
        / "outputs/pipeline/p4p1_type11_hirzebruch_k4_l2_tail_v2_seed68507_process.json",
    )
    parser.add_argument(
        "--l2-v2-blind-audit",
        type=Path,
        default=ROOT
        / "outputs/pipeline/p4p1_type11_hirzebruch_k4_l2_tail_blind_v2_audit.json",
    )
    parser.add_argument(
        "--l2-v3-training-summary",
        type=Path,
        default=ROOT
        / "outputs/pipeline/p4p1_type11_hirzebruch_k4_l2_tail_refined_v3_gpu_summary.json",
    )
    parser.add_argument(
        "--l2-v3-discovery-audit",
        type=Path,
        default=ROOT
        / "outputs/pipeline/p4p1_type11_hirzebruch_k4_l2_tail_discovery_v3_audit.json",
    )
    parser.add_argument(
        "--l2-v3-process-discovery",
        type=Path,
        default=ROOT
        / "outputs/pipeline/p4p1_type11_hirzebruch_k4_l2_tail_v3_seed68507_process.json",
    )
    parser.add_argument(
        "--l2-v3-process-blind",
        type=Path,
        default=ROOT
        / "outputs/pipeline/p4p1_type11_hirzebruch_k4_l2_tail_v3_seed69350_blind_process.json",
    )
    parser.add_argument(
        "--l2-v3-blind-audit",
        type=Path,
        default=ROOT
        / "outputs/pipeline/p4p1_type11_hirzebruch_k4_l2_tail_blind_v3_audit.json",
    )
    parser.add_argument(
        "--l2-v3-blind-scalar",
        type=Path,
        default=ROOT
        / "outputs/pipeline/p4p1_type11_hirzebruch_k4_l2_tail_blind_v3_scalar.json",
    )
    parser.add_argument(
        "--l2-v3-final-scalar",
        type=Path,
        default=ROOT
        / "outputs/pipeline/p4p1_type11_hirzebruch_k4_l2_tail_v3_scalar_65536.json",
    )
    parser.add_argument(
        "--l2-v3-failure-diagnostic",
        type=Path,
        default=ROOT
        / "outputs/pipeline/p4p1_type11_hirzebruch_k4_l2_tail_v3_seed69211_serial_failure_diagnostic.json",
    )
    parser.add_argument(
        "--l2-v4-training-summary",
        type=Path,
        default=ROOT
        / "outputs/pipeline/p4p1_type11_hirzebruch_k4_l2_tail_refined_v4_gpu_summary.json",
    )
    parser.add_argument(
        "--l2-v4-checkpoint-audit",
        type=Path,
        default=ROOT
        / "outputs/pipeline/p4p1_type11_hirzebruch_k4_l2_tail_checkpoint_v4_audit.json",
    )
    parser.add_argument(
        "--l2-v4-discovery-audit",
        type=Path,
        default=ROOT
        / "outputs/pipeline/p4p1_type11_hirzebruch_k4_l2_tail_discovery_v4_audit.json",
    )
    parser.add_argument(
        "--l2-v4-process-discovery",
        type=Path,
        default=ROOT
        / "outputs/pipeline/p4p1_type11_hirzebruch_k4_l2_tail_v4_seed68507_process.json",
    )
    parser.add_argument(
        "--l2-v4-one-time-check",
        type=Path,
        default=ROOT
        / "outputs/pipeline/p4p1_type11_hirzebruch_k4_l2_tail_check_v4_audit.json",
    )
    parser.add_argument(
        "--l2-v4-process-blind",
        type=Path,
        default=ROOT
        / "outputs/pipeline/p4p1_type11_hirzebruch_k4_l2_tail_v4_seed69350_blind_process.json",
    )
    parser.add_argument(
        "--l2-v4-blind-audit",
        type=Path,
        default=ROOT
        / "outputs/pipeline/p4p1_type11_hirzebruch_k4_l2_tail_blind_v4_audit.json",
    )
    parser.add_argument(
        "--l2-v4-blind-scalar",
        type=Path,
        default=ROOT
        / "outputs/pipeline/p4p1_type11_hirzebruch_k4_l2_tail_blind_v4_scalar.json",
    )
    parser.add_argument(
        "--l2-v4-final-scalar",
        type=Path,
        default=ROOT
        / "outputs/pipeline/p4p1_type11_hirzebruch_k4_l2_tail_v4_scalar_65536.json",
    )
    parser.add_argument(
        "--l2-v4-status",
        type=Path,
        default=ROOT
        / "outputs/pipeline/reviewer_round4_l2_tail_resolution_v4/status.txt",
    )
    parser.add_argument(
        "--x22-training-summary",
        type=Path,
        default=ROOT
        / "outputs/pipeline/gcicy_model_20260712_k3_rank64_whitened_gpu_summary.json",
    )
    parser.add_argument(
        "--x22-audit",
        type=Path,
        default=ROOT / "outputs/pipeline/p1p1p5_type22_seed20260712_k1_k3_audit.json",
    )
    parser.add_argument(
        "--x22-failed-atlas-audit",
        type=Path,
        default=ROOT
        / "outputs/pipeline/p1p1p5_type22_seed20260712_k1_k3_audit_atlas_failed.json",
    )
    parser.add_argument(
        "--x22-atlas-witness",
        type=Path,
        default=ROOT / "outputs/pipeline/p1p1p5_type22_seed20260712_atlas_witness.json",
    )
    parser.add_argument(
        "--rank-certificates",
        type=Path,
        default=ROOT / "outputs/section_restriction_rank_certificates.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "pipeline_specs/reviewer_round4_publication_manifest.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs/reviewer_round4_publication_evidence.json",
    )
    parser.add_argument(
        "--tex-out",
        type=Path,
        default=ROOT / "paper/generated/round4_tables.tex",
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=ROOT / "paper/generated/round4_tables.md",
    )
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def load_if_present(path: Path) -> dict[str, Any] | None:
    resolved = path.expanduser().resolve()
    return load(resolved) if resolved.is_file() else None


def parse_status(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise SystemExit(f"missing terminal-v4 status file: {resolved}")
    rows: dict[str, str] = {}
    for raw_line in resolved.read_text(encoding="utf-8").splitlines():
        if not raw_line or "=" not in raw_line:
            continue
        key, value = raw_line.split("=", 1)
        rows[key] = value
    if "overall" not in rows:
        raise SystemExit("terminal-v4 status does not contain a terminal outcome")
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.expanduser().resolve().relative_to(ROOT.resolve()).as_posix()


def workspace_artifact_path(value: str) -> Path:
    supplied = Path(value)
    if not supplied.is_absolute():
        return (ROOT / supplied).resolve()
    try:
        supplied.resolve().relative_to(ROOT.resolve())
        return supplied.resolve()
    except ValueError:
        pass
    candidates = (
        ROOT / "outputs/pipeline" / supplied.name,
        ROOT / "outputs" / supplied.name,
    )
    matches = [path.resolve() for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise SystemExit(f"cannot resolve remote artifact path locally: {value}")
    return matches[0]


def artifact_label(value: str) -> str:
    stem = Path(value).stem
    if "l2_tail_refined_v4" in stem:
        return "L2-tail v4"
    if "l2_tail_refined_v3" in stem:
        return "L2-tail v3"
    if "l2_tail_refined_v2" in stem:
        return "L2-tail v2"
    if "l2_tail_refined" in stem:
        return "L2-tail v1"
    if "multiseed_tail_refined" in stem:
        return "multiseed refined"
    if "hard_region_refined" in stem:
        return "hard-region refined"
    if stem.endswith("replica_a"):
        return "replica A"
    if stem.endswith("replica_b"):
        return "replica B"
    if stem.endswith("tail_refined_gpu"):
        return "reference"
    return "reference"


def metric_rows(data: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    if not bool(data["success"]) or not all(data["gates"].values()):
        raise SystemExit("the common-blind X11 optimizer-replica metric audit failed")
    rows = []
    for artifact in data["artifacts"]:
        rows.append(
            {
                "label": artifact_label(artifact["path"]),
                "artifact": relative(workspace_artifact_path(artifact["path"])),
                "mean_sigma": float(artifact["mean_sigma"]),
                "sigma_95_percent_ci": [
                    float(value) for value in artifact["sigma_95_percent_ci"]
                ],
                "maximum_seed_sigma": float(artifact["max_seed_sigma"]),
                "minimum_point_ess": float(
                    artifact["min_importance_effective_sample_size"]
                ),
                "minimum_metric_eigenvalue": float(artifact["min_metric_eigenvalue"]),
            }
        )
    sigma = np.asarray([row["mean_sigma"] for row in rows], dtype=float)
    summary = {
        "maximum_mean_sigma": float(np.max(sigma)),
        "mean_sigma_cv": float(np.std(sigma, ddof=1) / np.mean(sigma)),
        "mean_sigma_relative_range": float(
            (np.max(sigma) - np.min(sigma)) / np.mean(sigma)
        ),
    }
    if summary["maximum_mean_sigma"] > 0.08 or summary["mean_sigma_cv"] > 0.10:
        raise SystemExit(
            "the preregistered X11 optimizer-replica stability gate failed"
        )
    return rows, summary


def scalar_rows(data: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for artifact in data["artifacts"]:
        levels = [row for row in artifact["trial_levels"] if int(row["level"]) == 2]
        if len(levels) != 1:
            raise SystemExit(
                "each optimizer replica must have exactly one level-2 spectrum"
            )
        level = levels[0]
        rows.append(
            {
                "label": artifact_label(artifact["key"]),
                "artifact_key": artifact["key"],
                "first_three_eigenvalue_means": [
                    float(row["mean"]) for row in level["eigenvalues"][:3]
                ],
                "retained_trial_ranks": [
                    int(value) for value in level["retained_trial_ranks"]
                ],
                "minimum_point_ess": float(
                    level["minimum_integration_effective_sample_size"]
                ),
                "minimum_point_ess_per_rank": float(
                    level["minimum_effective_samples_per_retained_direction"]
                ),
                "minimum_cluster_weight_ess_per_rank": float(
                    level["minimum_cluster_weight_ess_per_retained_rank"]
                ),
            }
        )
    values = np.asarray(
        [row["first_three_eigenvalue_means"] for row in rows], dtype=float
    )
    mode_ranges = (np.max(values, axis=0) - np.min(values, axis=0)) / np.mean(
        values, axis=0
    )
    gate_outcomes = {key: bool(value) for key, value in data["gates"].items()}
    return rows, {
        "registered_success": bool(data["success"]) and all(gate_outcomes.values()),
        "gate_outcomes": gate_outcomes,
        "failed_gates": [key for key, value in gate_outcomes.items() if not value],
        "modewise_relative_ranges": [float(value) for value in mode_ranges],
        "maximum_modewise_relative_range": float(np.max(mode_ranges)),
        "points_per_seed": int(data["points_per_seed"]),
        "seeds": [int(value) for value in data["seeds"]],
    }


def x22_rows(
    training: dict[str, Any], audit: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if int(audit["model"]["seed"]) != 20260712:
        raise SystemExit("the second X22 audit has the wrong coefficient seed")
    rows = []
    for artifact in audit["artifacts"]:
        rows.append(
            {
                "degree": [int(value) for value in artifact["degree"]],
                "artifact": relative(workspace_artifact_path(artifact["path"])),
                "mean_sigma": float(artifact["mean_sigma"]),
                "sigma_95_percent_ci": [
                    float(value) for value in artifact["sigma_95_percent_ci"]
                ],
                "every_seed_sigma_improved": bool(
                    artifact["every_seed_sigma_improved"]
                ),
                "minimum_point_ess": float(
                    artifact["min_importance_effective_sample_size"]
                ),
                "minimum_metric_eigenvalue": float(artifact["min_metric_eigenvalue"]),
            }
        )
    primary = rows[-1]
    if primary["degree"] != [3, 3, 3]:
        raise SystemExit("the second-X22 primary artifact has the wrong degree")
    gate_outcomes = {key: bool(value) for key, value in audit["gates"].items()}
    atlas = audit["atlas"]
    atlas_errors = [
        float(atlas["max_baseline_projective_ma_error"]),
        float(atlas["max_baseline_implicit_ma_error"]),
        float(atlas["max_projective_importance_log_weight_error"]),
        float(atlas["max_implicit_importance_log_weight_error"]),
        *(float(value) for value in atlas["artifact_projective_ma_errors"].values()),
        *(float(value) for value in atlas["artifact_implicit_ma_errors"].values()),
    ]
    return rows, {
        "model_seed": 20260712,
        "training_internal_gates_passed": bool(training["passed_internal_gates"]),
        "strict_audit_success": bool(audit["success"]) and all(gate_outcomes.values()),
        "gate_outcomes": gate_outcomes,
        "failed_gates": [key for key, value in gate_outcomes.items() if not value],
        "maximum_atlas_error": max(atlas_errors),
        "atlas_error_threshold": float(
            audit["request"]["thresholds"]["max_atlas_error"]
        ),
        "ordered_every_seed_improvement": bool(
            audit["gates"]["ordered_artifact_sigma_improved"]
        ),
        "k3_mean_sigma_target_passed": primary["mean_sigma"] <= 0.18,
        "training_rank": int(training["rank"]),
        "training_parameterization": training["parameterization"],
        "audit_seeds": [int(value) for value in audit["request"]["seeds"]],
        "points_per_seed": int(audit["request"]["points_per_seed"]),
    }


def _maximum_atlas_error(data: dict[str, Any]) -> float:
    atlas = data["atlas"]
    return max(
        float(atlas["max_baseline_projective_ma_error"]),
        float(atlas["max_baseline_implicit_ma_error"]),
        float(atlas["max_projective_importance_log_weight_error"]),
        float(atlas["max_implicit_importance_log_weight_error"]),
        *(float(value) for value in atlas["artifact_projective_ma_errors"].values()),
        *(float(value) for value in atlas["artifact_implicit_ma_errors"].values()),
    )


def x22_atlas_resolution_summary(
    failed: dict[str, Any],
    witness_data: dict[str, Any],
    passed: dict[str, Any],
) -> dict[str, Any]:
    if bool(failed["gates"]["atlas_consistency"]):
        raise SystemExit("the frozen X22 pre-fix atlas audit is not a failure")
    if not bool(passed["success"]) or not bool(passed["gates"]["atlas_consistency"]):
        raise SystemExit("the resolved X22 audit did not pass its unchanged atlas gate")
    witnesses = witness_data["atlas"]["max_error_witnesses"]["artifact_projective_ma"]
    populated = [(key, row) for key, row in witnesses.items() if row is not None]
    key, witness = max(populated, key=lambda item: float(item[1]["absolute_error"]))
    if key != "gcicy_model_20260712_k3_rank64_whitened_gpu":
        raise SystemExit("the X22 atlas witness identifies an unexpected artifact")
    reference_metric = witness["reference_metric"]
    candidate_metric = witness["candidate_metric"]
    reference_correction = float(reference_metric["cholesky_logdet"]) - float(
        reference_metric["eigenvalue_logdet"]
    )
    candidate_correction = float(candidate_metric["cholesky_logdet"]) - float(
        candidate_metric["eigenvalue_logdet"]
    )
    predicted_cholesky_error = abs(
        (float(witness["candidate_ma"]) + candidate_correction)
        - (float(witness["reference_ma"]) + reference_correction)
    )
    eigenvalues = [float(value) for value in candidate_metric["metric_eigenvalues"]]
    return {
        "outcome": "resolved_by_stable_positive_hermitian_logdet",
        "threshold_unchanged": True,
        "atlas_error_threshold": float(
            passed["request"]["thresholds"]["max_atlas_error"]
        ),
        "failed_formal_maximum_atlas_error": _maximum_atlas_error(failed),
        "pre_fix_witness_maximum_atlas_error": _maximum_atlas_error(witness_data),
        "predicted_witness_error_with_cholesky_logdet": predicted_cholesky_error,
        "resolved_formal_maximum_atlas_error": _maximum_atlas_error(passed),
        "witness": {
            "artifact_key": key,
            "seed": int(witness["seed"]),
            "point_index": int(witness["point_index"]),
            "reference_chart": witness["reference_point"]["projective_chart"],
            "candidate_chart": witness["candidate_point"]["projective_chart"],
            "candidate_metric_condition_number": max(eigenvalues) / min(eigenvalues),
            "candidate_eigenvalue_logdet": float(candidate_metric["eigenvalue_logdet"]),
            "candidate_slogdet_logabsdet": float(candidate_metric["slogdet_logabsdet"]),
            "candidate_cholesky_logdet": float(candidate_metric["cholesky_logdet"]),
            "jacobian_min_singular_value": float(
                witness["candidate_point"]["jacobian_min_singular_value"]
            ),
            "residue_denominator_magnitude": float(
                witness["candidate_point"]["residue_denominator_magnitude"]
            ),
        },
    }


def metric_audit_snapshot(data: dict[str, Any]) -> dict[str, Any]:
    gate_outcomes = {key: bool(value) for key, value in data["gates"].items()}
    rows = []
    for artifact in data["artifacts"]:
        rows.append(
            {
                "label": artifact_label(artifact["path"]),
                "artifact": relative(workspace_artifact_path(artifact["path"])),
                "mean_sigma": float(artifact["mean_sigma"]),
                "sigma_95_percent_ci": [
                    float(value) for value in artifact["sigma_95_percent_ci"]
                ],
                "maximum_seed_sigma": float(artifact["max_seed_sigma"]),
                "minimum_point_ess": float(
                    artifact["min_importance_effective_sample_size"]
                ),
                "minimum_metric_eigenvalue": float(artifact["min_metric_eigenvalue"]),
            }
        )
    return {
        "success": bool(data["success"]) and all(gate_outcomes.values()),
        "gate_outcomes": gate_outcomes,
        "failed_gates": [key for key, value in gate_outcomes.items() if not value],
        "seeds": [int(value) for value in data["request"]["seeds"]],
        "points_per_seed": int(data["request"]["points_per_seed"]),
        "artifacts": rows,
    }


def final_scalar_snapshot(data: dict[str, Any]) -> dict[str, Any]:
    gate_outcomes = {key: bool(value) for key, value in data["gates"].items()}
    if len(data["artifacts"]) != 1:
        raise SystemExit("the final scalar audit must contain one artifact")
    levels = [
        row for row in data["artifacts"][0]["trial_levels"] if int(row["level"]) == 3
    ]
    if len(levels) != 1:
        raise SystemExit("the final scalar audit needs one level-3 row")
    level = levels[0]
    jackknife = level["cluster_delete_group_jackknife"]
    return {
        "registered_success": bool(data["success"]) and all(gate_outcomes.values()),
        "gate_outcomes": gate_outcomes,
        "failed_gates": [key for key, value in gate_outcomes.items() if not value],
        "seeds": [int(value) for value in data["seeds"]],
        "points_per_seed": int(data["points_per_seed"]),
        "level": 3,
        "retained_trial_ranks": [int(value) for value in level["retained_trial_ranks"]],
        "first_three_eigenvalues": [
            {
                "mean": float(row["mean"]),
                "95_percent_ci": [float(value) for value in row["95_percent_ci"]],
            }
            for row in level["eigenvalues"][:3]
        ],
        "minimum_point_ess": float(level["minimum_integration_effective_sample_size"]),
        "minimum_point_ess_per_rank": float(
            level["minimum_effective_samples_per_retained_direction"]
        ),
        "minimum_cluster_weight_ess_per_rank": float(
            level["minimum_cluster_weight_ess_per_retained_rank"]
        ),
        "maximum_jackknife_relative_standard_error": float(
            jackknife["maximum_within_seed_relative_standard_error"]
        ),
        "jackknife_eigenvalues": jackknife["eigenvalues"],
    }


def first_hard_region_summary(
    training: dict[str, Any],
    discovery: dict[str, Any],
    blind: dict[str, Any],
    discovery_scalar: dict[str, Any],
    parallel_tail: dict[str, Any],
) -> dict[str, Any]:
    discovery_snapshot = metric_audit_snapshot(discovery)
    blind_snapshot = metric_audit_snapshot(blind)
    scalar_artifacts, scalar_summary = scalar_rows(discovery_scalar)
    discovery_primary = discovery_snapshot["artifacts"][-1]
    blind_primary = blind_snapshot["artifacts"][-1]
    scalar_primary = scalar_artifacts[-1]
    tail_rows = []
    for artifact in parallel_tail["artifacts"]:
        top = artifact["top_points"][0]
        tail_rows.append(
            {
                "label": artifact_label(artifact["artifact_key"]),
                "artifact_key": artifact["artifact_key"],
                "point_effective_sample_size": float(
                    artifact["point_effective_sample_size"]
                ),
                "cluster_effective_sample_size": float(
                    artifact["cluster_weight_effective_sample_size"]
                ),
                "dominant_point_index": int(top["point_index"]),
                "dominant_point_weight_share": float(top["normalized_weight"]),
                "dominant_point_log_monge_ampere_ratio": float(
                    top["log_monge_ampere_ratio"]
                ),
            }
        )
    acceptance = {
        "training_internal_gates_passed": bool(training["passed_internal_gates"]),
        "discovery_registered_gates_passed": discovery_snapshot["success"],
        "discovery_seed_sigma_at_most_0_10": discovery_primary["mean_sigma"] <= 0.10,
        "discovery_scalar_registered_gates_passed": scalar_summary[
            "registered_success"
        ],
        "discovery_metric_volume_point_ess_at_least_2000": (
            scalar_primary["minimum_point_ess"] >= 2000.0
        ),
        "blind_registered_gates_passed": blind_snapshot["success"],
        "blind_mean_sigma_at_most_0_08": blind_primary["mean_sigma"] <= 0.08,
        "blind_maximum_seed_sigma_at_most_0_12": (
            blind_primary["maximum_seed_sigma"] <= 0.12
        ),
    }
    return {
        "outcome": "failed_blind_primary_max_seed_sigma_target",
        "training_artifact": relative(workspace_artifact_path(training["artifact"])),
        "discovery": discovery_snapshot,
        "discovery_scalar": {**scalar_summary, "artifacts": scalar_artifacts},
        "blind": blind_snapshot,
        "failed_blind_seed_parallel_weight_tail": {
            "seed": int(parallel_tail["seed"]),
            "points": int(parallel_tail["points"]),
            "dominant_point_is_common_across_artifacts": len(
                {row["dominant_point_index"] for row in tail_rows}
            )
            == 1,
            "artifacts": tail_rows,
        },
        "unexecuted_after_fail_fast": ["blind_scalar", "final_scalar_65536"],
        "acceptance": acceptance,
        "all_adaptive_acceptance_gates_passed": bool(all(acceptance.values())),
    }


def multiseed_refinement_summary(
    training: dict[str, Any],
    discovery_serial: dict[str, Any],
    discovery_parallel: dict[str, Any],
    discovery_scalar: dict[str, Any],
    discovery_parallel_tail: dict[str, Any],
    blind: dict[str, Any],
    blind_scalar: dict[str, Any],
    blind_tail: dict[str, Any],
) -> dict[str, Any]:
    serial_snapshot = metric_audit_snapshot(discovery_serial)
    parallel_snapshot = metric_audit_snapshot(discovery_parallel)
    blind_snapshot = metric_audit_snapshot(blind)
    discovery_scalar_rows, discovery_scalar_summary = scalar_rows(discovery_scalar)
    blind_scalar_rows, blind_scalar_summary = scalar_rows(blind_scalar)
    tail_matches = [
        row
        for row in discovery_parallel_tail["artifacts"]
        if row["artifact_key"] == "p4p1_type11_hirzebruch_k4_multiseed_tail_refined_gpu"
    ]
    if len(tail_matches) != 1:
        raise SystemExit("the multiseed parallel tail diagnostic is incomplete")
    tail = tail_matches[0]
    blind_tail_matches = [
        row
        for row in blind_tail["artifacts"]
        if row["artifact_key"] == "p4p1_type11_hirzebruch_k4_multiseed_tail_refined_gpu"
    ]
    if len(blind_tail_matches) != 1:
        raise SystemExit("the multiseed blind-tail diagnostic is incomplete")
    failed_tail = blind_tail_matches[0]
    serial_primary = serial_snapshot["artifacts"][-1]
    parallel_primary = parallel_snapshot["artifacts"][-1]
    blind_primary = blind_snapshot["artifacts"][-1]
    discovery_scalar_primary = discovery_scalar_rows[-1]
    blind_scalar_primary = blind_scalar_rows[-1]
    acceptance = {
        "training_internal_gates_passed": bool(training["passed_internal_gates"]),
        "serial_discovery_gates_passed": serial_snapshot["success"],
        "serial_discovery_sigma_at_most_0_10": serial_primary["mean_sigma"] <= 0.10,
        "serial_discovery_scalar_gates_passed": discovery_scalar_summary[
            "registered_success"
        ],
        "serial_discovery_metric_volume_ess_at_least_2000": (
            discovery_scalar_primary["minimum_point_ess"] >= 2000.0
        ),
        "parallel_discovery_gates_passed": parallel_snapshot["success"],
        "parallel_discovery_sigma_at_most_0_10": parallel_primary["mean_sigma"] <= 0.10,
        "parallel_discovery_metric_volume_ess_at_least_2000": float(
            tail["point_effective_sample_size"]
        )
        >= 2000.0,
        "blind_metric_gates_passed": blind_snapshot["success"],
        "blind_mean_sigma_at_most_0_08": blind_primary["mean_sigma"] <= 0.08,
        "blind_maximum_seed_sigma_at_most_0_12": (
            blind_primary["maximum_seed_sigma"] <= 0.12
        ),
        "blind_scalar_gates_passed": blind_scalar_summary["registered_success"],
        "blind_metric_volume_ess_at_least_4000": (
            blind_scalar_primary["minimum_point_ess"] >= 4000.0
        ),
        "blind_metric_volume_ess_per_rank_at_least_10": (
            blind_scalar_primary["minimum_point_ess_per_rank"] >= 10.0
        ),
        "blind_metric_volume_cluster_ess_per_rank_at_least_10": (
            blind_scalar_primary["minimum_cluster_weight_ess_per_rank"] >= 10.0
        ),
    }
    return {
        "outcome": "failed_blind_scalar_metric_volume_ess",
        "training_artifact": relative(workspace_artifact_path(training["artifact"])),
        "discovery_serial": serial_snapshot,
        "discovery_parallel": parallel_snapshot,
        "discovery_serial_scalar": {
            **discovery_scalar_summary,
            "artifacts": discovery_scalar_rows,
        },
        "discovery_parallel_tail": {
            "seed": int(discovery_parallel_tail["seed"]),
            "point_effective_sample_size": float(tail["point_effective_sample_size"]),
            "cluster_effective_sample_size": float(
                tail["cluster_weight_effective_sample_size"]
            ),
            "dominant_point_weight_share": float(
                tail["top_points"][0]["normalized_weight"]
            ),
        },
        "blind": blind_snapshot,
        "blind_scalar": {**blind_scalar_summary, "artifacts": blind_scalar_rows},
        "failed_blind_tail": {
            "seed": int(blind_tail["seed"]),
            "omega_weight_point_ess": float(
                blind_tail["omega_weight_summary"]["point_effective_sample_size"]
            ),
            "point_effective_sample_size": float(
                failed_tail["point_effective_sample_size"]
            ),
            "cluster_effective_sample_size": float(
                failed_tail["cluster_weight_effective_sample_size"]
            ),
            "dominant_point_weight_share": float(
                failed_tail["top_points"][0]["normalized_weight"]
            ),
            "dominant_point_log_monge_ampere_ratio": float(
                failed_tail["top_points"][0]["log_monge_ampere_ratio"]
            ),
            "sqrt_squared_energy": (
                float(
                    failed_tail["volume_ratio_error_statistics"]["sqrt_squared_energy"]
                )
                if "volume_ratio_error_statistics" in failed_tail
                else None
            ),
        },
        "unexecuted_after_fail_fast": ["final_scalar_65536"],
        "acceptance": acceptance,
        "all_adaptive_acceptance_gates_passed": bool(all(acceptance.values())),
    }


def tail_audit_artifact_snapshot(
    data: dict[str, Any],
    *,
    artifact_key: str | None = None,
) -> dict[str, Any]:
    matches = [
        row
        for row in data["artifacts"]
        if artifact_key is None or row["artifact_key"] == artifact_key
    ]
    if len(matches) != 1:
        raise SystemExit("metric-volume tail audit has an ambiguous artifact row")
    row = matches[0]
    thresholds = data["thresholds"]
    failed_seeds = [
        int(seed_row["seed"])
        for seed_row in row["seeds"]
        if (
            float(seed_row["point_effective_sample_size"])
            < float(thresholds["minimum_point_effective_sample_size"])
            or float(seed_row["cluster_effective_sample_size"])
            < float(thresholds["minimum_cluster_effective_sample_size"])
            or float(seed_row["top_point_weight_share"])
            > float(thresholds["maximum_top_point_weight_share"])
            or float(seed_row["sqrt_squared_energy"])
            > float(thresholds["maximum_sqrt_squared_energy"])
        )
    ]
    return {
        "artifact_key": row["artifact_key"],
        "artifact": relative(workspace_artifact_path(row["artifact_path"])),
        "success": bool(row["success"]),
        "gates": {key: bool(value) for key, value in row["gates"].items()},
        "mean_sigma": float(row["mean_sigma"]),
        "maximum_seed_sigma": max(
            float(seed_row["sigma"]) for seed_row in row["seeds"]
        ),
        "sigma_95_percent_ci": [float(value) for value in row["sigma_95_percent_ci"]],
        "mean_sqrt_squared_energy": float(row["mean_sqrt_squared_energy"]),
        "maximum_sqrt_squared_energy": float(row["maximum_sqrt_squared_energy"]),
        "minimum_point_effective_sample_size": float(
            row["minimum_point_effective_sample_size"]
        ),
        "minimum_cluster_effective_sample_size": float(
            row["minimum_cluster_effective_sample_size"]
        ),
        "maximum_top_point_weight_share": float(row["maximum_top_point_weight_share"]),
        "maximum_normalized_ratio": float(row["maximum_normalized_ratio"]),
        "thresholds": {key: float(value) for key, value in thresholds.items()},
        "seeds": [int(seed_row["seed"]) for seed_row in row["seeds"]],
        "failed_seeds": failed_seeds,
        "seed_rows": row["seeds"],
    }


def l2_training_snapshot(data: dict[str, Any]) -> dict[str, Any]:
    request = data["request"]
    groupwise = bool(request.get("groupwise_objective", False))
    initial_rows = [row for row in data["history"] if int(row["epoch"]) == 0]
    selected_rows = [
        row for row in data["history"] if int(row["epoch"]) == int(data["best_epoch"])
    ]
    if len(initial_rows) != 1 or len(selected_rows) != 1:
        raise SystemExit(
            "L2 training history does not identify unique initial and selected rows"
        )
    initial = initial_rows[0]
    selected = selected_rows[0]
    selection_rows = (
        data.get("checkpoint_validation", {}) if groupwise else data["checks"]
    )
    selected_check_l2 = [
        float(row["candidate"]["sqrt_squared_energy"])
        for row in selection_rows.values()
    ]
    if not selected_check_l2:
        raise SystemExit("L2 training summary has no checkpoint rows")
    return {
        "artifact": relative(workspace_artifact_path(data["artifact"])),
        "success": bool(data["success"]) and bool(data["passed_internal_gates"]),
        "best_epoch": int(data["best_epoch"]),
        "degree": [int(value) for value in request["degree"]],
        "epochs": int(request["epochs"]),
        "evaluation_interval": int(request["eval_every"]),
        "learning_rate": float(request["learning_rate"]),
        "training_seed": int(request["torch_seed"]),
        "basis_seed": int(request["basis_seed"]),
        "export_seed": int(request["export_seed"]),
        "initial_artifact": relative(
            workspace_artifact_path(request["initial_artifact"])
        ),
        "objective_mode": "groupwise" if groupwise else "pooled",
        "sigma_loss_weight": float(request["sigma_loss_weight"]),
        "volume_ratio_l2_loss_weight": float(
            request.get("group_volume_ratio_l2_loss_weight", 0.0)
            if groupwise
            else request["volume_ratio_l2_loss_weight"]
        ),
        "selection_volume_ratio_l2_weight": float(
            request.get("group_volume_ratio_l2_mean_fraction", 0.0)
            if groupwise
            else request["selection_volume_ratio_l2_weight"]
        ),
        "selection_volume_ratio_l2_max_weight": float(
            1.0 - request.get("group_volume_ratio_l2_mean_fraction", 1.0)
            if groupwise
            else request.get("selection_volume_ratio_l2_max_weight", 0.0)
        ),
        "group_volume_ratio_l2_smooth_max_temperature": (
            float(request["group_volume_ratio_l2_smooth_max_temperature"])
            if groupwise
            else None
        ),
        "maximum_check_volume_ratio_l2_gate": (
            None
            if (
                request.get("maximum_checkpoint_volume_ratio_l2")
                if groupwise
                else request.get("maximum_check_volume_ratio_l2")
            )
            is None
            else float(
                request["maximum_checkpoint_volume_ratio_l2"]
                if groupwise
                else request["maximum_check_volume_ratio_l2"]
            )
        ),
        "maximum_checkpoint_sigma_gate": (
            float(request["maximum_checkpoint_sigma"]) if groupwise else None
        ),
        "checkpoint_policy": request.get("checkpoint_policy"),
        "checkpoint_required_consecutive": (
            int(request["checkpoint_required_consecutive"]) if groupwise else None
        ),
        "initial_mean_check_sigma": float(initial["mean_check_sigma"]),
        "initial_mean_check_volume_ratio_l2": float(
            initial["mean_check_volume_ratio_l2"]
        ),
        "selected_mean_check_sigma": float(selected["mean_check_sigma"]),
        "selected_mean_check_volume_ratio_l2": float(
            selected["mean_check_volume_ratio_l2"]
        ),
        "selected_maximum_check_volume_ratio_l2": max(selected_check_l2),
        "selected_check_count_above_0_75": sum(
            value > 0.75 for value in selected_check_l2
        ),
        "all_selected_check_gates_passed": all(
            bool(row["passed"]) for row in selection_rows.values()
        ),
        "train_batches": request["train_batches"],
        "validation_batches": request["validation_batches"],
        "check_batches": (
            request.get("checkpoint_batches", [])
            if groupwise
            else request["check_batches"]
        ),
        "checkpoint_confirmation": data.get("checkpoint_confirmation"),
        "selected_training_groups": data.get("selected_training_groups", []),
        "export_candidate": data["export_candidate"],
    }


def v3_training_failure_snapshot(
    training: dict[str, Any], diagnostic: dict[str, Any]
) -> dict[str, Any]:
    snapshot = l2_training_snapshot(training)
    if snapshot["success"] or snapshot["best_epoch"] != 0:
        raise SystemExit("the frozen L2-v3 training result is not the recorded failure")
    artifact = workspace_artifact_path(training["artifact"])
    initial_artifact = workspace_artifact_path(training["request"]["initial_artifact"])
    if int(diagnostic["seed"]) != 69211 or int(diagnostic["points"]) != 32768:
        raise SystemExit("the frozen L2-v3 serial diagnostic has the wrong sample")
    rows = {row["artifact_key"]: row for row in diagnostic["artifacts"]}
    expected_keys = {
        "p4p1_type11_hirzebruch_k4_l2_tail_refined_v2_gpu",
        "p4p1_type11_hirzebruch_k4_l2_tail_refined_v3_gpu",
    }
    if set(rows) != expected_keys:
        raise SystemExit("the frozen L2-v3 serial diagnostic is incomplete")
    v2_row = rows["p4p1_type11_hirzebruch_k4_l2_tail_refined_v2_gpu"]
    row = rows["p4p1_type11_hirzebruch_k4_l2_tail_refined_v3_gpu"]
    comparison_pairs = [
        (
            v2_row["volume_ratio_error_statistics"][key],
            row["volume_ratio_error_statistics"][key],
        )
        for key in (
            "sigma",
            "inverse_sigma",
            "squared_energy",
            "sqrt_squared_energy",
            "weighted_centered_log_ma_rms",
            "normalized_ratio_min",
            "normalized_ratio_max",
        )
    ]
    comparison_pairs.extend(
        [
            (
                v2_row["point_effective_sample_size"],
                row["point_effective_sample_size"],
            ),
            (
                v2_row["cluster_weight_effective_sample_size"],
                row["cluster_weight_effective_sample_size"],
            ),
            (
                v2_row["point_weight_concentration"]["1"],
                row["point_weight_concentration"]["1"],
            ),
        ]
    )
    discrepancies = [
        abs(float(left) - float(right)) for left, right in comparison_pairs
    ]
    if not all(
        np.isclose(float(left), float(right), rtol=1e-11, atol=1e-11)
        for left, right in comparison_pairs
    ):
        raise SystemExit("the epoch-zero L2-v3 metric does not reproduce v2")
    errors = row["volume_ratio_error_statistics"]
    return {
        "training": snapshot,
        "outcome": "failed_training_no_eligible_post_initialization_checkpoint",
        "epoch_zero_reexpression_matches_v2_on_exact_serial_sample": True,
        "maximum_v2_v3_diagnostic_absolute_difference": max(discrepancies),
        "artifact_container_sha256_equal": sha256(artifact) == sha256(initial_artifact),
        "exact_serial_failure": {
            "seed": int(diagnostic["seed"]),
            "points": int(diagnostic["points"]),
            "sigma": float(errors["sigma"]),
            "sqrt_squared_energy": float(errors["sqrt_squared_energy"]),
            "point_effective_sample_size": float(row["point_effective_sample_size"]),
            "cluster_effective_sample_size": float(
                row["cluster_weight_effective_sample_size"]
            ),
            "top_point_weight_share": float(row["point_weight_concentration"]["1"]),
            "maximum_normalized_ratio": float(errors["normalized_ratio_max"]),
        },
        "unexecuted_after_fail_fast": [
            "discovery_tail",
            "discovery_process",
            "blind_process",
            "blind_tail",
            "blind_scalar",
            "final_scalar_65536",
        ],
    }


def l2_manifest_consistency(
    resolution: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    protocol = manifest["adaptive_followups"]["l2_volume_ratio_tail_refinement"]
    frozen = protocol["v3_risk_aware_checkpoint"]
    acceptance = protocol["acceptance"]
    training = resolution["v3"]["training"]
    v2_blind = resolution["v2"]["blind_tail"]
    v2_failed_seeds = v2_blind["failed_seeds"]
    blind_tail = resolution["v3"]["blind_tail"]
    blind_scalar = resolution["v3"]["blind_scalar"]
    final_scalar = resolution["v3"]["final_scalar"]
    checks = {
        "v2_failed_seeds_match_manifest": v2_failed_seeds
        == [int(seed) for seed in protocol["v2_blind_result"]["failed_seeds"]],
        "v2_minimum_point_ess_matches_manifest": v2_blind[
            "minimum_point_effective_sample_size"
        ]
        == float(protocol["v2_blind_result"]["minimum_point_ess"]),
        "v2_minimum_cluster_ess_matches_manifest": v2_blind[
            "minimum_cluster_effective_sample_size"
        ]
        == float(protocol["v2_blind_result"]["minimum_cluster_ess"]),
        "v2_maximum_point_share_matches_manifest": v2_blind[
            "maximum_top_point_weight_share"
        ]
        == float(protocol["v2_blind_result"]["maximum_top_point_share"]),
        "v2_maximum_l2_matches_manifest": v2_blind["maximum_sqrt_squared_energy"]
        == float(protocol["v2_blind_result"]["maximum_sqrt_squared_energy"]),
        "torch_seed_matches_manifest": training["training_seed"]
        == int(frozen["torch_seed"]),
        "basis_seed_matches_manifest": training["basis_seed"]
        == int(frozen["basis_seed"]),
        "training_seeds_match_manifest": [
            int(row["seed"]) for row in training["train_batches"]
        ]
        == [int(seed) for seed in frozen["training_seeds"]],
        "validation_seeds_match_manifest": [
            int(row["seed"]) for row in training["validation_batches"]
        ]
        == [int(seed) for seed in frozen["validation_seeds"]],
        "check_seeds_match_manifest": [
            int(row["seed"]) for row in training["check_batches"]
        ]
        == [int(seed) for seed in frozen["check_seeds"]],
        "export_seed_matches_manifest": training["export_seed"]
        == int(frozen["export_seed"]),
        "l2_loss_weight_matches_manifest": training["volume_ratio_l2_loss_weight"]
        == float(frozen["volume_ratio_l2_loss_weight"]),
        "mean_checkpoint_weight_matches_manifest": training[
            "selection_volume_ratio_l2_weight"
        ]
        == float(frozen["mean_check_volume_ratio_l2_weight"]),
        "maximum_checkpoint_weight_matches_manifest": training[
            "selection_volume_ratio_l2_max_weight"
        ]
        == float(frozen["maximum_check_volume_ratio_l2_weight"]),
        "hard_l2_gate_matches_manifest": training["maximum_check_volume_ratio_l2_gate"]
        == float(frozen["hard_per_check_maximum_sqrt_squared_energy"]),
        "discovery_maximum_seed_sigma_gate_matches_manifest": float(
            acceptance["discovery_maximum_seed_sigma"]
        )
        == 0.10,
        "discovery_process_sigma_gate_matches_manifest": float(
            acceptance["discovery_process_sigma"]
        )
        == 0.10,
        "blind_mean_sigma_gate_matches_manifest": float(
            acceptance["blind_tail_mean_sigma"]
        )
        == 0.08,
        "blind_maximum_seed_sigma_gate_matches_manifest": float(
            acceptance["blind_tail_maximum_seed_sigma"]
        )
        == 0.12,
        "blind_process_sigma_gate_matches_manifest": float(
            acceptance["blind_process_sigma"]
        )
        == 0.12,
        "blind_tail_seeds_match_manifest": blind_tail["seeds"]
        == [int(seed) for seed in frozen["new_blind_tail_and_scalar_seeds"]],
        "blind_scalar_seeds_match_manifest": blind_scalar["seeds"]
        == [int(seed) for seed in frozen["new_blind_tail_and_scalar_seeds"]],
        "blind_process_seed_matches_manifest": resolution["v3"]["process_blind"]["seed"]
        == int(frozen["new_blind_process_seed"]),
        "final_scalar_seeds_match_manifest": final_scalar["seeds"]
        == [int(seed) for seed in frozen["final_scalar_seeds"]],
    }
    return {
        "checks": checks,
        "all_checks_passed": bool(all(checks.values())),
    }


def process_tail_artifact_snapshot(
    data: dict[str, Any],
    *,
    artifact_key: str,
    minimum_point_effective_sample_size: float = 4000.0,
    minimum_cluster_effective_sample_size: float = 3040.0,
    maximum_top_point_weight_share: float = 0.01,
    maximum_sqrt_squared_energy: float = 0.75,
    maximum_sigma: float | None = None,
) -> dict[str, Any]:
    matches = [row for row in data["artifacts"] if row["artifact_key"] == artifact_key]
    if len(matches) != 1:
        raise SystemExit("process metric-volume tail audit has an ambiguous artifact")
    row = matches[0]
    errors = row["volume_ratio_error_statistics"]
    gates = {
        "point_effective_sample_size": float(row["point_effective_sample_size"])
        >= minimum_point_effective_sample_size,
        "cluster_effective_sample_size": float(
            row["cluster_weight_effective_sample_size"]
        )
        >= minimum_cluster_effective_sample_size,
        "top_point_weight_share": float(row["point_weight_concentration"]["1"])
        <= maximum_top_point_weight_share,
        "sqrt_squared_energy": float(errors["sqrt_squared_energy"])
        <= maximum_sqrt_squared_energy,
    }
    if maximum_sigma is not None:
        gates["sigma"] = float(errors["sigma"]) <= maximum_sigma
    return {
        "artifact_key": artifact_key,
        "artifact": relative(workspace_artifact_path(row["artifact_path"])),
        "seed": int(data["seed"]),
        "points": int(data["points"]),
        "point_effective_sample_size": float(row["point_effective_sample_size"]),
        "cluster_effective_sample_size": float(
            row["cluster_weight_effective_sample_size"]
        ),
        "top_point_weight_share": float(row["point_weight_concentration"]["1"]),
        "sigma": float(errors["sigma"]),
        "sqrt_squared_energy": float(errors["sqrt_squared_energy"]),
        "maximum_normalized_ratio": float(errors["normalized_ratio_max"]),
        "gates": gates,
        "success": bool(all(gates.values())),
    }


def l2_tail_resolution_summary(
    v1_training: dict[str, Any],
    v1_discovery: dict[str, Any],
    v2_training: dict[str, Any],
    v2_discovery: dict[str, Any],
    v2_process: dict[str, Any],
    v2_blind: dict[str, Any],
    v3_training: dict[str, Any],
    v3_discovery: dict[str, Any],
    v3_process_discovery: dict[str, Any],
    v3_process_blind: dict[str, Any],
    v3_blind: dict[str, Any],
    v3_blind_scalar: dict[str, Any],
    v3_final_scalar: dict[str, Any],
) -> dict[str, Any]:
    v1_training_snapshot = l2_training_snapshot(v1_training)
    v2_training_snapshot = l2_training_snapshot(v2_training)
    v3_training_snapshot = l2_training_snapshot(v3_training)
    v1 = tail_audit_artifact_snapshot(
        v1_discovery,
        artifact_key="p4p1_type11_hirzebruch_k4_l2_tail_refined_gpu",
    )
    v2 = tail_audit_artifact_snapshot(
        v2_discovery,
        artifact_key="p4p1_type11_hirzebruch_k4_l2_tail_refined_v2_gpu",
    )
    v2_blind_snapshot = tail_audit_artifact_snapshot(v2_blind)
    v3 = tail_audit_artifact_snapshot(
        v3_discovery,
        artifact_key="p4p1_type11_hirzebruch_k4_l2_tail_refined_v3_gpu",
    )
    v3_blind_snapshot = tail_audit_artifact_snapshot(v3_blind)
    v2_process_snapshot = process_tail_artifact_snapshot(
        v2_process,
        artifact_key="p4p1_type11_hirzebruch_k4_l2_tail_refined_v2_gpu",
    )
    v3_process_discovery_snapshot = process_tail_artifact_snapshot(
        v3_process_discovery,
        artifact_key="p4p1_type11_hirzebruch_k4_l2_tail_refined_v3_gpu",
    )
    v3_process_blind_snapshot = process_tail_artifact_snapshot(
        v3_process_blind,
        artifact_key="p4p1_type11_hirzebruch_k4_l2_tail_refined_v3_gpu",
    )
    blind_scalar_rows, blind_scalar_summary = scalar_rows(v3_blind_scalar)
    final = final_scalar_snapshot(v3_final_scalar)

    expected_v2_failed_seeds = [69002, 69004, 69007, 69008, 69013, 69015]
    expected_v3_training_seeds = [68913, *expected_v2_failed_seeds]
    expected_v3_discovery_seeds = [28601, 68705, *expected_v3_training_seeds]
    v2_adaptation_seeds = {
        28601,
        68705,
        68507,
        *(int(row["seed"]) for row in v2_training_snapshot["train_batches"]),
        *(int(row["seed"]) for row in v2_training_snapshot["validation_batches"]),
        *(int(row["seed"]) for row in v2_training_snapshot["check_batches"]),
    }
    v3_adaptation_seeds = {
        68507,
        *expected_v3_discovery_seeds,
        *(int(row["seed"]) for row in v3_training_snapshot["train_batches"]),
        *(int(row["seed"]) for row in v3_training_snapshot["validation_batches"]),
        *(int(row["seed"]) for row in v3_training_snapshot["check_batches"]),
    }
    v2_blind_overlap = sorted(set(v2_blind_snapshot["seeds"]) & v2_adaptation_seeds)
    v3_blind_overlap = sorted(set(v3_blind_snapshot["seeds"]) & v3_adaptation_seeds)

    acceptance = {
        "v1_training_passed": v1_training_snapshot["success"],
        "v1_failure_preserved": not v1["success"],
        "v1_failed_only_l2_gate": [
            key for key, value in v1["gates"].items() if not value
        ]
        == ["sqrt_squared_energy"],
        "v2_training_passed": v2_training_snapshot["success"],
        "v2_discovery_seeds_exactly_frozen": v2["seeds"] == [28601, 68705],
        "v2_discovery_passed": v2["success"],
        "v2_process_discovery_seed_exactly_frozen": v2_process_snapshot["seed"]
        == 68507,
        "v2_process_discovery_passed": v2_process_snapshot["success"],
        "v2_blind_seeds_exactly_frozen": v2_blind_snapshot["seeds"]
        == list(range(69001, 69017)),
        "v2_blind_seeds_disjoint_from_adaptation_and_discovery": not v2_blind_overlap,
        "v2_blind_failure_preserved": not v2_blind_snapshot["success"],
        "v2_failed_seed_inventory_exact": v2_blind_snapshot["failed_seeds"]
        == expected_v2_failed_seeds,
        "v3_training_passed": v3_training_snapshot["success"],
        "v3_training_batches_exactly_disclosed_failures": [
            int(row["seed"]) for row in v3_training_snapshot["train_batches"]
        ]
        == expected_v3_training_seeds,
        "v3_check_l2_gate_unchanged": v3_training_snapshot[
            "maximum_check_volume_ratio_l2_gate"
        ]
        == 0.75,
        "v3_all_training_check_gates_passed": v3_training_snapshot[
            "all_selected_check_gates_passed"
        ],
        "v3_selected_maximum_check_l2_at_most_0_75": v3_training_snapshot[
            "selected_maximum_check_volume_ratio_l2"
        ]
        <= 0.75,
        "v3_discovery_seeds_exactly_frozen": v3["seeds"] == expected_v3_discovery_seeds,
        "v3_discovery_passed": v3["success"],
        "v3_discovery_maximum_seed_sigma_at_most_0_10": v3["maximum_seed_sigma"]
        <= 0.10,
        "v3_process_discovery_seed_exactly_frozen": v3_process_discovery_snapshot[
            "seed"
        ]
        == 68507,
        "v3_process_discovery_passed": v3_process_discovery_snapshot["success"],
        "v3_process_discovery_sigma_at_most_0_10": v3_process_discovery_snapshot[
            "sigma"
        ]
        <= 0.10,
        "v3_process_blind_seed_exactly_frozen": v3_process_blind_snapshot["seed"]
        == 69350,
        "v3_process_blind_passed": v3_process_blind_snapshot["success"],
        "v3_process_blind_sigma_at_most_0_12": v3_process_blind_snapshot["sigma"]
        <= 0.12,
        "v3_blind_seeds_exactly_frozen": v3_blind_snapshot["seeds"]
        == list(range(69301, 69317)),
        "v3_blind_seeds_disjoint_from_adaptation_and_discovery": not v3_blind_overlap,
        "v3_blind_tail_passed": v3_blind_snapshot["success"],
        "v3_blind_mean_sigma_at_most_0_08": v3_blind_snapshot["mean_sigma"] <= 0.08,
        "v3_blind_maximum_seed_sigma_at_most_0_12": v3_blind_snapshot[
            "maximum_seed_sigma"
        ]
        <= 0.12,
        "v3_blind_scalar_seeds_match_tail_audit": blind_scalar_summary["seeds"]
        == v3_blind_snapshot["seeds"],
        "v3_blind_scalar_points_per_seed_exactly_32768": blind_scalar_summary[
            "points_per_seed"
        ]
        == 32768,
        "v3_blind_scalar_passed": blind_scalar_summary["registered_success"],
        "v3_final_seeds_exactly_frozen": final["seeds"] == list(range(69401, 69409)),
        "v3_final_points_per_seed_exactly_65536": final["points_per_seed"] == 65536,
        "v3_final_scalar_passed": final["registered_success"],
        "v3_final_point_ess_at_least_8000": final["minimum_point_ess"] >= 8000.0,
        "v3_final_point_ess_per_rank_at_least_12": (
            final["minimum_point_ess_per_rank"] >= 12.0
        ),
        "v3_final_cluster_ess_per_rank_at_least_15": (
            final["minimum_cluster_weight_ess_per_rank"] >= 15.0
        ),
        "v3_final_jackknife_rse_at_most_0_05": (
            final["maximum_jackknife_relative_standard_error"] <= 0.05
        ),
    }
    return {
        "v1": {
            "training": v1_training_snapshot,
            "discovery": v1,
            "outcome": "failed_discovery_sqrt_squared_energy_gate",
        },
        "v2": {
            "training": v2_training_snapshot,
            "discovery": v2,
            "process_discovery": v2_process_snapshot,
            "blind_tail": v2_blind_snapshot,
            "outcome": "failed_blind_tail",
            "unexecuted_after_fail_fast": ["blind_scalar", "final_scalar_65536"],
        },
        "v3": {
            "training": v3_training_snapshot,
            "discovery": v3,
            "process_discovery": v3_process_discovery_snapshot,
            "process_blind": v3_process_blind_snapshot,
            "blind_tail": v3_blind_snapshot,
            "blind_scalar": {
                **blind_scalar_summary,
                "artifacts": blind_scalar_rows,
            },
            "final_scalar": final,
        },
        "v2_blind_seed_overlap_with_adaptation_or_discovery": v2_blind_overlap,
        "v3_blind_seed_overlap_with_adaptation_or_discovery": v3_blind_overlap,
        "acceptance": acceptance,
        "all_final_acceptance_gates_passed": bool(all(acceptance.values())),
    }


def terminal_v4_summary(
    status: dict[str, str],
    training: dict[str, Any] | None,
    checkpoint: dict[str, Any] | None,
    discovery: dict[str, Any] | None,
    process_discovery: dict[str, Any] | None,
    one_time_check: dict[str, Any] | None,
    process_blind: dict[str, Any] | None,
    blind: dict[str, Any] | None,
    blind_scalar: dict[str, Any] | None,
    final_scalar: dict[str, Any] | None,
) -> dict[str, Any]:
    artifact_key = "p4p1_type11_hirzebruch_k4_l2_tail_refined_v4_gpu"

    def started(stage: str) -> bool:
        return f"{stage}_started_utc" in status

    def passed(stage: str) -> bool:
        return status.get(f"{stage}_exit") == "0"

    def available(stage: str, data: dict[str, Any] | None) -> dict[str, Any] | None:
        if passed(stage) and data is None:
            raise SystemExit(f"terminal-v4 stage {stage} passed without its output")
        return data if started(stage) else None

    training = available("training", training)
    checkpoint = available("checkpoint_tail", checkpoint)
    discovery = available("discovery_tail", discovery)
    process_discovery = available("discovery_process", process_discovery)
    one_time_check = available("one_time_check", one_time_check)
    process_blind = available("blind_process", process_blind)
    blind = available("blind_tail", blind)
    blind_scalar = available("blind_scalar", blind_scalar)
    final_scalar = available("final_scalar", final_scalar)

    snapshots: dict[str, Any] = {}
    if training is not None:
        snapshots["training"] = l2_training_snapshot(training)
    if checkpoint is not None:
        snapshots["checkpoint_tail"] = tail_audit_artifact_snapshot(
            checkpoint, artifact_key=artifact_key
        )
    if discovery is not None:
        snapshots["development_tail"] = tail_audit_artifact_snapshot(
            discovery, artifact_key=artifact_key
        )
    if process_discovery is not None:
        snapshots["development_process"] = process_tail_artifact_snapshot(
            process_discovery,
            artifact_key=artifact_key,
            maximum_top_point_weight_share=0.005,
            maximum_sigma=0.10,
        )
    if one_time_check is not None:
        snapshots["one_time_check"] = tail_audit_artifact_snapshot(
            one_time_check, artifact_key=artifact_key
        )
    if process_blind is not None:
        snapshots["blind_process"] = process_tail_artifact_snapshot(
            process_blind,
            artifact_key=artifact_key,
            maximum_top_point_weight_share=0.005,
            maximum_sigma=0.075,
        )
    if blind is not None:
        snapshots["blind_tail"] = tail_audit_artifact_snapshot(
            blind, artifact_key=artifact_key
        )
    if blind_scalar is not None:
        rows, summary = scalar_rows(blind_scalar)
        snapshots["blind_scalar"] = {**summary, "artifacts": rows}
    if final_scalar is not None:
        snapshots["final_scalar"] = final_scalar_snapshot(final_scalar)

    acceptance: dict[str, bool] = {}
    if "training" in snapshots:
        row = snapshots["training"]
        acceptance["training_internal_gates_passed"] = bool(
            passed("training")
            and passed("training_contract")
            and row["success"]
            and row["objective_mode"] == "groupwise"
        )
    if "checkpoint_tail" in snapshots:
        row = snapshots["checkpoint_tail"]
        acceptance["checkpoint_tail_passed"] = bool(
            passed("checkpoint_tail")
            and passed("checkpoint_sigma_gate")
            and row["success"]
            and row["maximum_seed_sigma"] <= 0.075
        )
    if "development_tail" in snapshots:
        row = snapshots["development_tail"]
        acceptance["development_tail_passed"] = bool(
            passed("discovery_tail")
            and passed("discovery_tail_gate")
            and row["success"]
            and row["maximum_seed_sigma"] <= 0.10
        )
    if "development_process" in snapshots:
        acceptance["development_process_passed"] = bool(
            passed("discovery_process")
            and passed("discovery_process_gate")
            and snapshots["development_process"]["success"]
        )
    if "one_time_check" in snapshots:
        row = snapshots["one_time_check"]
        acceptance["one_time_check_passed"] = bool(
            passed("one_time_check")
            and passed("one_time_check_sigma_gate")
            and row["success"]
            and row["maximum_seed_sigma"] <= 0.075
        )
    if "blind_process" in snapshots:
        acceptance["blind_process_passed"] = bool(
            passed("blind_process")
            and passed("blind_process_gate")
            and snapshots["blind_process"]["success"]
        )
    if "blind_tail" in snapshots:
        row = snapshots["blind_tail"]
        acceptance["blind_tail_passed"] = bool(
            passed("blind_tail")
            and passed("blind_tail_quality_gate")
            and row["success"]
            and row["mean_sigma"] <= 0.08
            and row["maximum_seed_sigma"] <= 0.075
        )
    if "blind_scalar" in snapshots:
        acceptance["blind_scalar_passed"] = bool(
            passed("blind_scalar") and snapshots["blind_scalar"]["registered_success"]
        )
    if "final_scalar" in snapshots:
        acceptance["final_scalar_passed"] = bool(
            passed("final_scalar") and snapshots["final_scalar"]["registered_success"]
        )

    core_stages = [
        "training",
        "checkpoint_tail",
        "discovery_tail",
        "discovery_process",
        "one_time_check",
        "blind_process",
        "blind_tail",
        "blind_scalar",
        "final_scalar",
    ]
    full_order = [
        "preflight",
        "groupwise_tests",
        "training",
        "training_contract",
        "provenance",
        "checkpoint_tail",
        "checkpoint_sigma_gate",
        "discovery_tail",
        "discovery_tail_gate",
        "discovery_process",
        "discovery_process_gate",
        "one_time_check",
        "one_time_check_sigma_gate",
        "blind_process",
        "blind_process_gate",
        "blind_tail",
        "blind_tail_quality_gate",
        "blind_scalar",
        "final_scalar",
    ]
    failed = [
        stage for stage in full_order if status.get(f"{stage}_exit") not in (None, "0")
    ]
    fail_fast_respected = True
    if failed:
        failed_index = full_order.index(failed[0])
        fail_fast_respected = not any(
            started(stage) for stage in full_order[failed_index + 1 :]
        )
    return {
        "terminal_outcome": status["overall"],
        "fail_fast_respected": fail_fast_respected,
        "failed_stage": failed[0] if failed else None,
        "unexecuted_stages": [stage for stage in core_stages if not started(stage)],
        **snapshots,
        "acceptance": acceptance,
        "all_terminal_acceptance_gates_passed": bool(
            status["overall"] == "completed"
            and len(acceptance) == len(core_stages)
            and all(acceptance.values())
        ),
    }


def l2_tail_resolution_v4_summary(
    v1_training: dict[str, Any],
    v1_discovery: dict[str, Any],
    v2_training: dict[str, Any],
    v2_discovery: dict[str, Any],
    v2_process: dict[str, Any],
    v2_blind: dict[str, Any],
    v3_training: dict[str, Any],
    v3_failure_diagnostic: dict[str, Any],
    v4_status: dict[str, str],
    v4_training: dict[str, Any] | None,
    v4_checkpoint: dict[str, Any] | None,
    v4_discovery: dict[str, Any] | None,
    v4_process_discovery: dict[str, Any] | None,
    v4_one_time_check: dict[str, Any] | None,
    v4_process_blind: dict[str, Any] | None,
    v4_blind: dict[str, Any] | None,
    v4_blind_scalar: dict[str, Any] | None,
    v4_final_scalar: dict[str, Any] | None,
) -> dict[str, Any]:
    v1_training_snapshot = l2_training_snapshot(v1_training)
    v2_training_snapshot = l2_training_snapshot(v2_training)
    v1 = tail_audit_artifact_snapshot(
        v1_discovery,
        artifact_key="p4p1_type11_hirzebruch_k4_l2_tail_refined_gpu",
    )
    v2 = tail_audit_artifact_snapshot(
        v2_discovery,
        artifact_key="p4p1_type11_hirzebruch_k4_l2_tail_refined_v2_gpu",
    )
    v2_blind_snapshot = tail_audit_artifact_snapshot(v2_blind)
    v2_process_snapshot = process_tail_artifact_snapshot(
        v2_process,
        artifact_key="p4p1_type11_hirzebruch_k4_l2_tail_refined_v2_gpu",
    )
    v3 = v3_training_failure_snapshot(v3_training, v3_failure_diagnostic)
    v4 = terminal_v4_summary(
        v4_status,
        v4_training,
        v4_checkpoint,
        v4_discovery,
        v4_process_discovery,
        v4_one_time_check,
        v4_process_blind,
        v4_blind,
        v4_blind_scalar,
        v4_final_scalar,
    )
    preservation = {
        "v1_failure_preserved": not v1["success"],
        "v2_blind_failure_preserved": not v2_blind_snapshot["success"],
        "v3_training_failure_preserved": not v3["training"]["success"],
        "v3_exact_serial_reexpression_matches_v2": v3[
            "epoch_zero_reexpression_matches_v2_on_exact_serial_sample"
        ],
        "v4_fail_fast_respected": v4["fail_fast_respected"],
    }
    return {
        "v1": {
            "training": v1_training_snapshot,
            "discovery": v1,
            "outcome": "failed_discovery_sqrt_squared_energy_gate",
        },
        "v2": {
            "training": v2_training_snapshot,
            "discovery": v2,
            "process_discovery": v2_process_snapshot,
            "blind_tail": v2_blind_snapshot,
            "outcome": "failed_blind_tail",
            "unexecuted_after_fail_fast": ["blind_scalar", "final_scalar_65536"],
        },
        "v3": v3,
        "v4": v4,
        "preservation_checks": preservation,
        "all_final_acceptance_gates_passed": bool(
            all(preservation.values()) and v4["all_terminal_acceptance_gates_passed"]
        ),
    }


def l2_manifest_consistency_v4(
    resolution: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    protocol = manifest["adaptive_followups"]["l2_volume_ratio_tail_refinement"]
    v3_frozen = protocol["v3_risk_aware_checkpoint"]
    v3_result = protocol["v3_training_result"]
    v4_frozen = protocol["terminal_v4_groupwise_protocol"]
    v3 = resolution["v3"]
    v4 = resolution["v4"]
    serial = v3["exact_serial_failure"]
    checks: dict[str, bool] = {
        "manifest_declares_terminal_v4": bool(v4_frozen["terminal_attempt"]),
        "manifest_forbids_v5": bool(v4_frozen["no_v5_after_failure"]),
        "manifest_forbids_replacement_blind_namespace": bool(
            v4_frozen["no_new_blind_namespace_after_failure"]
        ),
        "manifest_forbids_threshold_relaxation": bool(
            v4_frozen["no_threshold_relaxation_after_freeze"]
        ),
        "v3_failed_at_training": not v3["training"]["success"],
        "v3_selected_epoch_zero": v3["training"]["best_epoch"] == 0,
        "v3_training_seeds_match_manifest": [
            int(row["seed"]) for row in v3["training"]["train_batches"]
        ]
        == [int(seed) for seed in v3_frozen["training_seeds"]],
        "v3_validation_seeds_match_manifest": [
            int(row["seed"]) for row in v3["training"]["validation_batches"]
        ]
        == [int(seed) for seed in v3_frozen["validation_seeds"]],
        "v3_check_seeds_match_manifest": [
            int(row["seed"]) for row in v3["training"]["check_batches"]
        ]
        == [int(seed) for seed in v3_frozen["check_seeds"]],
        "v3_serial_seed_matches_manifest": serial["seed"]
        == int(v3_result["disclosed_exact_serial_failure_seed"]),
        "v3_serial_sigma_matches_manifest": bool(
            np.isclose(serial["sigma"], v3_result["exact_serial_sigma"])
        ),
        "v3_serial_chi_matches_manifest": bool(
            np.isclose(
                serial["sqrt_squared_energy"],
                v3_result["exact_serial_sqrt_squared_energy"],
            )
        ),
        "v3_serial_point_ess_matches_manifest": bool(
            np.isclose(
                serial["point_effective_sample_size"],
                v3_result["exact_serial_point_ess"],
            )
        ),
        "v3_serial_cluster_ess_matches_manifest": bool(
            np.isclose(
                serial["cluster_effective_sample_size"],
                v3_result["exact_serial_cluster_ess"],
            )
        ),
        "v3_serial_top_share_matches_manifest": bool(
            np.isclose(
                serial["top_point_weight_share"],
                v3_result["exact_serial_top_point_share"],
            )
        ),
        "v4_fail_fast_respected": bool(v4["fail_fast_respected"]),
    }
    training = v4.get("training")
    if training is not None:
        checks.update(
            {
                "v4_degree_matches_manifest": training["degree"]
                == [int(value) for value in v4_frozen["degree"]],
                "v4_epochs_match_manifest": training["epochs"]
                == int(v4_frozen["epochs"]),
                "v4_evaluation_interval_matches_manifest": training[
                    "evaluation_interval"
                ]
                == int(v4_frozen["evaluation_interval"]),
                "v4_learning_rate_matches_manifest": training["learning_rate"]
                == float(v4_frozen["learning_rate"]),
                "v4_torch_seed_matches_manifest": training["training_seed"]
                == int(v4_frozen["torch_seed"]),
                "v4_basis_seed_matches_manifest": training["basis_seed"]
                == int(v4_frozen["basis_seed"]),
                "v4_export_seed_matches_manifest": training["export_seed"]
                == int(v4_frozen["export_seed"]),
                "v4_development_groups_match_manifest": [
                    int(row["seed"]) for row in training["train_batches"]
                ]
                == [int(seed) for seed in v4_frozen["development_group_seeds"]],
                "v4_checkpoint_groups_match_manifest": [
                    int(row["seed"]) for row in training["check_batches"]
                ]
                == [int(seed) for seed in v4_frozen["checkpoint_seeds"]],
                "v4_sigma_weight_matches_manifest": training["sigma_loss_weight"]
                == float(v4_frozen["sigma_loss_weight"]),
                "v4_chi_weight_matches_manifest": training[
                    "volume_ratio_l2_loss_weight"
                ]
                == float(v4_frozen["group_volume_ratio_l2_loss_weight"]),
                "v4_chi_mean_fraction_matches_manifest": training[
                    "selection_volume_ratio_l2_weight"
                ]
                == float(v4_frozen["group_volume_ratio_l2_mean_fraction"]),
                "v4_chi_smoothmax_fraction_matches_manifest": training[
                    "selection_volume_ratio_l2_max_weight"
                ]
                == float(v4_frozen["group_volume_ratio_l2_smooth_max_fraction"]),
                "v4_chi_temperature_matches_manifest": training[
                    "group_volume_ratio_l2_smooth_max_temperature"
                ]
                == float(v4_frozen["group_volume_ratio_l2_smooth_max_temperature"]),
                "v4_checkpoint_sigma_gate_matches_manifest": training[
                    "maximum_checkpoint_sigma_gate"
                ]
                == float(v4_frozen["checkpoint_maximum_sigma"]),
                "v4_checkpoint_chi_gate_matches_manifest": training[
                    "maximum_check_volume_ratio_l2_gate"
                ]
                == float(v4_frozen["checkpoint_maximum_sqrt_squared_energy"]),
                "v4_checkpoint_consecutive_count_matches_manifest": training[
                    "checkpoint_required_consecutive"
                ]
                == int(v4_frozen["checkpoint_required_consecutive_passes"]),
            }
        )
    expected_development_audit = [
        28601,
        68507,
        68705,
        *(int(seed) for seed in v4_frozen["development_group_seeds"]),
    ]
    stage_seed_checks = (
        ("checkpoint_tail", "checkpoint_seeds"),
        ("one_time_check", "one_time_check_seeds"),
        ("blind_tail", "blind_tail_and_scalar_seeds"),
    )
    for stage, manifest_key in stage_seed_checks:
        if stage in v4:
            checks[f"v4_{stage}_seeds_match_manifest"] = v4[stage]["seeds"] == [
                int(seed) for seed in v4_frozen[manifest_key]
            ]
    if "development_tail" in v4:
        checks["v4_development_audit_seeds_match_frozen_inventory"] = (
            v4["development_tail"]["seeds"] == expected_development_audit
        )
    if "development_process" in v4:
        checks["v4_development_process_seed_is_frozen"] = (
            v4["development_process"]["seed"] == 68507
        )
    if "blind_process" in v4:
        checks["v4_blind_process_seed_matches_manifest"] = v4["blind_process"][
            "seed"
        ] == int(v4_frozen["blind_process_seed"])
    if "blind_scalar" in v4:
        checks["v4_blind_scalar_seeds_match_manifest"] = v4["blind_scalar"][
            "seeds"
        ] == [int(seed) for seed in v4_frozen["blind_tail_and_scalar_seeds"]]
    if "final_scalar" in v4:
        checks["v4_final_scalar_seeds_match_manifest"] = v4["final_scalar"][
            "seeds"
        ] == [int(seed) for seed in v4_frozen["final_scalar_seeds"]]
    namespaces = {
        "development": set(int(seed) for seed in v4_frozen["development_group_seeds"]),
        "checkpoint": set(int(seed) for seed in v4_frozen["checkpoint_seeds"]),
        "one_time": set(int(seed) for seed in v4_frozen["one_time_check_seeds"]),
        "blind": set(int(seed) for seed in v4_frozen["blind_tail_and_scalar_seeds"]),
        "final": set(int(seed) for seed in v4_frozen["final_scalar_seeds"]),
    }
    checks["v4_stochastic_namespaces_are_pairwise_disjoint"] = all(
        not left_values & right_values
        for index, left_values in enumerate(namespaces.values())
        for right_values in list(namespaces.values())[index + 1 :]
    )
    return {"checks": checks, "all_checks_passed": bool(all(checks.values()))}


def tail_diagnostic_summary(
    tail: dict[str, Any], stress: dict[str, Any]
) -> dict[str, Any]:
    stress_by_key = {
        Path(artifact["path"]).stem: artifact for artifact in stress["artifacts"]
    }
    rows = []
    for artifact in tail["artifacts"]:
        key = artifact["artifact_key"]
        top = artifact["top_points"][0]
        rows.append(
            {
                "label": artifact_label(key),
                "artifact_key": key,
                "metric_volume_point_ess": float(
                    artifact["point_effective_sample_size"]
                ),
                "dominant_point_index": int(top["point_index"]),
                "dominant_cluster_id": int(top["cluster_id"]),
                "dominant_point_weight_share": float(top["normalized_weight"]),
                "dominant_point_log_monge_ampere_ratio": float(
                    top["log_monge_ampere_ratio"]
                ),
                "stress_seed_sigma": float(stress_by_key[key]["mean_sigma"]),
            }
        )
    return {
        "seed": int(tail["seed"]),
        "points": int(tail["points"]),
        "omega_weight_point_ess": float(
            tail["omega_weight_summary"]["point_effective_sample_size"]
        ),
        "dominant_point_is_common_across_artifacts": len(
            {row["dominant_point_index"] for row in rows}
        )
        == 1,
        "artifacts": rows,
    }


def rank_certificate(data: dict[str, Any]) -> dict[str, Any]:
    matches = [
        row for row in data["certificates"] if row["geometry"] == "X22_seed20260712"
    ]
    if len(matches) != 1 or not bool(matches[0]["all_targets_met"]):
        raise SystemExit("the second-X22 exact section-rank certificate is missing")
    degrees = matches[0]["degrees"]
    if len(degrees) != 1:
        raise SystemExit("the second-X22 rank certificate has an unexpected inventory")
    degree = degrees[0]
    if (
        degree["degree"] != [3, 3, 3]
        or int(degree["riemann_roch_target"]) != 251
        or int(degree["finite_field_rank"]) != 251
        or int(degree["reported_artifact_finite_field_rank"]) != 251
        or int(degree["reported_artifact_pivot_minor_determinant_mod_prime"]) == 0
    ):
        raise SystemExit("the second-X22 k=3 section basis is not exactly certified")
    return {
        "geometry": matches[0]["geometry"],
        "model_seed": int(matches[0]["model_seed"]),
        "prime": int(matches[0]["prime"]),
        "degree": degree["degree"],
        "riemann_roch_target": int(degree["riemann_roch_target"]),
        "reported_artifact_rank": int(degree["reported_artifact_finite_field_rank"]),
        "reported_artifact_minor_determinant": int(
            degree["reported_artifact_pivot_minor_determinant_mod_prime"]
        ),
    }


def write_tex(
    path: Path,
    metrics: list[dict[str, Any]],
    metric_summary: dict[str, float],
    spectra: list[dict[str, Any]],
    scalar_summary: dict[str, Any],
    x22: list[dict[str, Any]],
    x22_summary: dict[str, Any],
    first_adaptation: dict[str, Any],
    multiseed: dict[str, Any],
    l2_resolution: dict[str, Any],
) -> None:
    scalar_status = "passed" if scalar_summary["registered_success"] else "failed"
    multiseed_status = (
        "passed" if multiseed["all_adaptive_acceptance_gates_passed"] else "failed"
    )
    l2_status = (
        "passed" if l2_resolution["all_final_acceptance_gates_passed"] else "failed"
    )
    x22_status = "passed" if x22_summary["strict_audit_success"] else "failed"
    lines = [
        "% Generated by scripts/build_round4_publication_evidence.py.",
        "\\noindent\\textbf{Round-four gate status.} "
        f"The registered common-seed scalar-replica diagnostic {scalar_status}; "
        "the first single-seed adaptation failed its blind maximum-seed gate; "
        f"the subsequent multiseed adaptation {multiseed_status} its blind "
        f"scalar gate; the L2 volume-ratio resolution {l2_status}; and the strict "
        f"second-$X_{{22}}$ audit {x22_status}.  Failed registered diagnostics "
        "remain part of the evidence inventory.",
        "",
        "\\begin{table}[H]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Staged $L^2$ volume-ratio adaptations.  Failed attempts are retained: v1 fails discovery, v2 fails its untouched blind tail, and v3 selects no post-initialization checkpoint.  Terminal v4 uses separately normalized, equally weighted development groups and a frozen consecutive-checkpoint rule.}",
        "\\label{tab:round4-l2-training}",
        "\\begin{tabular}{lrrrrrrrr}",
        "\\toprule",
        "attempt & $\\kappa$ & $\\zeta_{\\mu}$ & $\\zeta_{\\max}$ & hard max. & epoch & initial mean $\\chi$ & selected mean $\\chi$ & selected max. $\\chi$ \\\\",
        "\\midrule",
    ]
    training_attempts = [
        ("L2-v1", l2_resolution["v1"]["training"]),
        ("L2-v2", l2_resolution["v2"]["training"]),
        ("L2-v3", l2_resolution["v3"]["training"]),
    ]
    if "training" in l2_resolution["v4"]:
        training_attempts.append(("L2-v4", l2_resolution["v4"]["training"]))
    for label, training in training_attempts:
        hard_gate = training["maximum_check_volume_ratio_l2_gate"]
        hard_gate_text = "--" if hard_gate is None else f"{hard_gate:.2f}"
        lines.append(
            f"{label} & {training['volume_ratio_l2_loss_weight']:.3g} & "
            f"{training['selection_volume_ratio_l2_weight']:.2f} & "
            f"{training['selection_volume_ratio_l2_max_weight']:.2f} & "
            f"{hard_gate_text} & "
            f"{training['best_epoch']} & "
            f"{training['initial_mean_check_volume_ratio_l2']:.4f} & "
            f"{training['selected_mean_check_volume_ratio_l2']:.4f} & "
            f"{training['selected_maximum_check_volume_ratio_l2']:.4f} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
            "\\begin{table}[H]",
            "\\centering",
            "\\small",
            "\\caption{Training-randomness robustness for the principal $X_{11}$ $k=4$ metric.  All three frozen artifacts are evaluated on the same eight blind seeds.}",
            "\\label{tab:round4-training-replicas}",
            "\\begin{tabular}{lrrr}",
            "\\toprule",
            "artifact & $\\sigma$ [95\\% interval] & min. point ESS & min. $\\lambda(g)$ \\\\",
            "\\midrule",
        ]
    )
    for row in metrics:
        low, high = row["sigma_95_percent_ci"]
        lines.append(
            f"{row['label']} & {row['mean_sigma']:.5f} "
            f"[{low:.5f},{high:.5f}] & {row['minimum_point_ess']:.1f} & "
            f"{row['minimum_metric_eigenvalue']:.2e} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
            "\\begin{table}[H]",
            "\\centering",
            "\\small",
            "\\caption{Registered level-2 scalar Ritz diagnostic for the same three $X_{11}$ training replicas at $N=32768$ points per seed.  This diagnostic is retained as a failure because seed 28601 violates the ESS gates.}",
            "\\label{tab:round4-training-replica-spectrum}",
            "\\begin{tabular}{lrrrr}",
            "\\toprule",
            "artifact & $\\lambda_1$ & $\\lambda_2$ & $\\lambda_3$ & min. pESS/rank \\\\",
            "\\midrule",
        ]
    )
    for row in spectra:
        first, second, third = row["first_three_eigenvalue_means"]
        lines.append(
            f"{row['label']} & {first:.5f} & {second:.5f} & {third:.5f} & "
            f"{row['minimum_point_ess_per_rank']:.3f} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
            "\\begin{table}[H]",
            "\\centering",
            "\\small",
            "\\caption{Adaptive $X_{11}$ hard-region refinement.  Attempt 1 used only the diagnosed direct-sampling seed and failed on its blind set.  Attempt 2 trained on that seed and the exact child streams of the failed blind process seed, then used a new blind set.}",
            "\\label{tab:round4-hard-region}",
            "\\begin{tabular}{lllrrr}",
            "\\toprule",
            "attempt & sample & artifact & $\\sigma$ & max. seed $\\sigma$ & min. point ESS \\\\",
            "\\midrule",
        ]
    )
    metric_samples = [
        ("single-seed", "discovery", first_adaptation["discovery"]),
        ("single-seed", "blind", first_adaptation["blind"]),
        ("multiseed", "direct discovery", multiseed["discovery_serial"]),
        ("multiseed", "process discovery", multiseed["discovery_parallel"]),
        ("multiseed", "new blind", multiseed["blind"]),
    ]
    for attempt, sample, snapshot in metric_samples:
        for row in snapshot["artifacts"]:
            lines.append(
                f"{attempt} & {sample} & {row['label']} & {row['mean_sigma']:.5f} & "
                f"{row['maximum_seed_sigma']:.5f} & {row['minimum_point_ess']:.1f} \\\\"
            )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
            "\\begin{table}[H]",
            "\\centering",
            "\\small",
            "\\caption{Metric-volume scalar validation for the adaptive $X_{11}$ metrics.  Failed registered rows remain in the table.  Terminal-v4 rows appear only when the frozen fail-fast protocol reaches the corresponding scalar stage.}",
            "\\label{tab:round4-hard-region-scalar}",
            "\\begin{tabular}{lrrrr}",
            "\\toprule",
            "sample & min. point ESS & min. pESS/rank & min. cwESS/rank & $\\lambda_1$ \\\\",
            "\\midrule",
        ]
    )
    scalar_samples = [
        (
            "single-seed discovery",
            first_adaptation["discovery_scalar"]["artifacts"][-1],
        ),
        (
            "multiseed direct discovery",
            multiseed["discovery_serial_scalar"]["artifacts"][-1],
        ),
        ("multiseed new blind", multiseed["blind_scalar"]["artifacts"][-1]),
    ]
    for sample, row in scalar_samples:
        lines.append(
            f"{sample} & {row['minimum_point_ess']:.1f} & "
            f"{row['minimum_point_ess_per_rank']:.2f} & "
            f"{row['minimum_cluster_weight_ess_per_rank']:.2f} & "
            f"{row['first_three_eigenvalue_means'][0]:.5f} \\\\"
        )
    process_tail = multiseed["discovery_parallel_tail"]
    lines.append(
        f"multiseed process discovery & {process_tail['point_effective_sample_size']:.1f} "
        "& -- & -- & -- \\\\"
    )
    terminal = l2_resolution["v4"]
    if "blind_scalar" in terminal:
        l2_blind = terminal["blind_scalar"]["artifacts"][-1]
        lines.append(
            f"L2-v4 blind 32768 & {l2_blind['minimum_point_ess']:.1f} & "
            f"{l2_blind['minimum_point_ess_per_rank']:.2f} & "
            f"{l2_blind['minimum_cluster_weight_ess_per_rank']:.2f} & "
            f"{l2_blind['first_three_eigenvalue_means'][0]:.5f} \\\\"
        )
    if "final_scalar" in terminal:
        final_scalar = terminal["final_scalar"]
        lines.append(
            f"L2-v4 final 65536 & {final_scalar['minimum_point_ess']:.1f} & "
            f"{final_scalar['minimum_point_ess_per_rank']:.2f} & "
            f"{final_scalar['minimum_cluster_weight_ess_per_rank']:.2f} & "
            f"{final_scalar['first_three_eigenvalues'][0]['mean']:.5f} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
            "\\begin{table}[H]",
            "\\centering",
            "\\small",
            "\\caption{Upper volume-ratio tail audits for the principal $X_{11}$ metric.  The v1 discovery, v2 blind, and exact serial v3 training failures are retained.  Terminal-v4 rows are emitted only for stages actually reached by the frozen fail-fast protocol.}",
            "\\label{tab:round4-l2-tail}",
            "\\begin{tabular}{lrrrr}",
            "\\toprule",
            "stage & min. point ESS & min. cluster ESS & max. point share & max. $\\chi$ \\\\",
            "\\midrule",
        ]
    )
    tail_rows = [
        ("L2-v1 discovery", l2_resolution["v1"]["discovery"]),
        ("L2-v2 discovery", l2_resolution["v2"]["discovery"]),
        ("L2-v2 blind", l2_resolution["v2"]["blind_tail"]),
    ]
    terminal_tail_labels = (
        ("checkpoint_tail", "L2-v4 checkpoint"),
        ("development_tail", "L2-v4 development"),
        ("one_time_check", "L2-v4 one-time check"),
        ("blind_tail", "L2-v4 blind"),
    )
    tail_rows.extend(
        (label, terminal[key]) for key, label in terminal_tail_labels if key in terminal
    )
    for stage, row in tail_rows:
        lines.append(
            f"{stage} & {row['minimum_point_effective_sample_size']:.1f} & "
            f"{row['minimum_cluster_effective_sample_size']:.1f} & "
            f"{row['maximum_top_point_weight_share']:.5f} & "
            f"{row['maximum_sqrt_squared_energy']:.5f} \\\\"
        )
    v3_serial = l2_resolution["v3"]["exact_serial_failure"]
    lines.append(
        f"L2-v3 serial failure & {v3_serial['point_effective_sample_size']:.1f} & "
        f"{v3_serial['cluster_effective_sample_size']:.1f} & "
        f"{v3_serial['top_point_weight_share']:.5f} & "
        f"{v3_serial['sqrt_squared_energy']:.5f} \\\\"
    )
    process_rows = [
        ("L2-v2 process discovery", l2_resolution["v2"]["process_discovery"])
    ]
    if "development_process" in terminal:
        process_rows.append(
            ("L2-v4 process development", terminal["development_process"])
        )
    if "blind_process" in terminal:
        process_rows.append(("L2-v4 process blind", terminal["blind_process"]))
    for stage, process in process_rows:
        lines.append(
            f"{stage} & {process['point_effective_sample_size']:.1f} & "
            f"{process['cluster_effective_sample_size']:.1f} & "
            f"{process['top_point_weight_share']:.5f} & "
            f"{process['sqrt_squared_energy']:.5f} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
            "\\begin{table}[H]",
            "\\centering",
            "\\small",
            "\\caption{Cross-geometry transfer on the independently fixed smooth type-$(2,2)$ coefficient model with seed 20260712.}",
            "\\label{tab:round4-second-x22}",
            "\\begin{tabular}{lrrr}",
            "\\toprule",
            "degree & $\\sigma$ [95\\% interval] & min. point ESS & min. $\\lambda(g)$ \\\\",
            "\\midrule",
        ]
    )
    for row in x22:
        low, high = row["sigma_95_percent_ci"]
        degree = ",".join(str(value) for value in row["degree"])
        lines.append(
            f"$({degree})$ & {row['mean_sigma']:.5f} "
            f"[{low:.5f},{high:.5f}] & {row['minimum_point_ess']:.1f} & "
            f"{row['minimum_metric_eigenvalue']:.2e} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_markdown(
    path: Path,
    metrics: list[dict[str, Any]],
    spectra: list[dict[str, Any]],
    x22: list[dict[str, Any]],
    scalar_summary: dict[str, Any],
    x22_summary: dict[str, Any],
    first_adaptation: dict[str, Any],
    multiseed: dict[str, Any],
    l2_resolution: dict[str, Any],
) -> None:
    lines = [
        "# Round-four publication tables",
        "",
        f"- Registered scalar-replica audit: {'passed' if scalar_summary['registered_success'] else 'failed'}",
        "- First single-seed adaptation: failed its blind maximum-seed gate",
        f"- Multiseed hard-region resolution: {'passed' if multiseed['all_adaptive_acceptance_gates_passed'] else 'failed'}",
        f"- L2 volume-ratio tail resolution: {'passed' if l2_resolution['all_final_acceptance_gates_passed'] else 'failed'}",
        f"- Strict second-X22 audit: {'passed' if x22_summary['strict_audit_success'] else 'failed'}",
        "",
        "## X11 L2-tail training",
        "",
        "| attempt | kappa | mean weight | max weight | hard max gate | epoch | initial mean chi | selected mean chi | selected max chi |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    training_attempts = [
        ("L2-v1", l2_resolution["v1"]["training"]),
        ("L2-v2", l2_resolution["v2"]["training"]),
        ("L2-v3", l2_resolution["v3"]["training"]),
    ]
    if "training" in l2_resolution["v4"]:
        training_attempts.append(("L2-v4", l2_resolution["v4"]["training"]))
    for label, training in training_attempts:
        hard_gate = training["maximum_check_volume_ratio_l2_gate"]
        hard_gate_text = "--" if hard_gate is None else f"{hard_gate:.2f}"
        lines.append(
            f"| {label} | {training['volume_ratio_l2_loss_weight']:.3g} | "
            f"{training['selection_volume_ratio_l2_weight']:.2f} | "
            f"{training['selection_volume_ratio_l2_max_weight']:.2f} | "
            f"{hard_gate_text} | "
            f"{training['best_epoch']} | "
            f"{training['initial_mean_check_volume_ratio_l2']:.6f} | "
            f"{training['selected_mean_check_volume_ratio_l2']:.6f} | "
            f"{training['selected_maximum_check_volume_ratio_l2']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## X11 training replicas",
            "",
            "| artifact | mean sigma | 95% interval | min point ESS |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in metrics:
        low, high = row["sigma_95_percent_ci"]
        lines.append(
            f"| {row['label']} | {row['mean_sigma']:.6f} | "
            f"[{low:.6f}, {high:.6f}] | {row['minimum_point_ess']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## X11 replica spectra",
            "",
            "| artifact | lambda 1 | lambda 2 | lambda 3 | min pESS/rank |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in spectra:
        values = row["first_three_eigenvalue_means"]
        lines.append(
            f"| {row['label']} | {values[0]:.6f} | {values[1]:.6f} | "
            f"{values[2]:.6f} | {row['minimum_point_ess_per_rank']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## X11 adaptive hard-region refinement",
            "",
            "| sample | artifact | mean sigma | max seed sigma | min point ESS |",
            "|---|---|---:|---:|---:|",
        ]
    )
    metric_samples = [
        ("single-seed discovery", first_adaptation["discovery"]),
        ("single-seed blind", first_adaptation["blind"]),
        ("multiseed direct discovery", multiseed["discovery_serial"]),
        ("multiseed process discovery", multiseed["discovery_parallel"]),
        ("multiseed new blind", multiseed["blind"]),
    ]
    for sample, snapshot in metric_samples:
        for row in snapshot["artifacts"]:
            lines.append(
                f"| {sample} | {row['label']} | {row['mean_sigma']:.6f} | "
                f"{row['maximum_seed_sigma']:.6f} | {row['minimum_point_ess']:.1f} |"
            )
    lines.extend(
        [
            "",
            "## X11 refined metric-volume scalar audit",
            "",
            "| sample | min point ESS | min pESS/rank | min cwESS/rank | lambda 1 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    scalar_samples = [
        (
            "single-seed discovery",
            first_adaptation["discovery_scalar"]["artifacts"][-1],
        ),
        (
            "multiseed direct discovery",
            multiseed["discovery_serial_scalar"]["artifacts"][-1],
        ),
        ("multiseed new blind", multiseed["blind_scalar"]["artifacts"][-1]),
    ]
    for sample, row in scalar_samples:
        lines.append(
            f"| {sample} | {row['minimum_point_ess']:.1f} | "
            f"{row['minimum_point_ess_per_rank']:.2f} | "
            f"{row['minimum_cluster_weight_ess_per_rank']:.2f} | "
            f"{row['first_three_eigenvalue_means'][0]:.6f} |"
        )
    process_tail = multiseed["discovery_parallel_tail"]
    lines.append(
        f"| multiseed process discovery | {process_tail['point_effective_sample_size']:.1f} | -- | -- | -- |"
    )
    terminal = l2_resolution["v4"]
    if "blind_scalar" in terminal:
        l2_blind = terminal["blind_scalar"]["artifacts"][-1]
        lines.append(
            f"| L2-v4 blind 32768 | {l2_blind['minimum_point_ess']:.1f} | "
            f"{l2_blind['minimum_point_ess_per_rank']:.2f} | "
            f"{l2_blind['minimum_cluster_weight_ess_per_rank']:.2f} | "
            f"{l2_blind['first_three_eigenvalue_means'][0]:.6f} |"
        )
    if "final_scalar" in terminal:
        final_scalar = terminal["final_scalar"]
        lines.append(
            f"| L2-v4 final 65536 | {final_scalar['minimum_point_ess']:.1f} | "
            f"{final_scalar['minimum_point_ess_per_rank']:.2f} | "
            f"{final_scalar['minimum_cluster_weight_ess_per_rank']:.2f} | "
            f"{final_scalar['first_three_eigenvalues'][0]['mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## X11 L2 volume-ratio tail audits",
            "",
            "| stage | min point ESS | min cluster ESS | max point share | max chi |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    tail_rows = [
        ("L2-v1 discovery", l2_resolution["v1"]["discovery"]),
        ("L2-v2 discovery", l2_resolution["v2"]["discovery"]),
        ("L2-v2 blind", l2_resolution["v2"]["blind_tail"]),
    ]
    for key, label in (
        ("checkpoint_tail", "L2-v4 checkpoint"),
        ("development_tail", "L2-v4 development"),
        ("one_time_check", "L2-v4 one-time check"),
        ("blind_tail", "L2-v4 blind"),
    ):
        if key in terminal:
            tail_rows.append((label, terminal[key]))
    for stage, row in tail_rows:
        lines.append(
            f"| {stage} | {row['minimum_point_effective_sample_size']:.1f} | "
            f"{row['minimum_cluster_effective_sample_size']:.1f} | "
            f"{row['maximum_top_point_weight_share']:.6f} | "
            f"{row['maximum_sqrt_squared_energy']:.6f} |"
        )
    v3_serial = l2_resolution["v3"]["exact_serial_failure"]
    lines.append(
        f"| L2-v3 serial failure | {v3_serial['point_effective_sample_size']:.1f} | "
        f"{v3_serial['cluster_effective_sample_size']:.1f} | "
        f"{v3_serial['top_point_weight_share']:.6f} | "
        f"{v3_serial['sqrt_squared_energy']:.6f} |"
    )
    process_rows = [
        ("L2-v2 process discovery", l2_resolution["v2"]["process_discovery"])
    ]
    if "development_process" in terminal:
        process_rows.append(
            ("L2-v4 process development", terminal["development_process"])
        )
    if "blind_process" in terminal:
        process_rows.append(("L2-v4 process blind", terminal["blind_process"]))
    for stage, process in process_rows:
        lines.append(
            f"| {stage} | {process['point_effective_sample_size']:.1f} | "
            f"{process['cluster_effective_sample_size']:.1f} | "
            f"{process['top_point_weight_share']:.6f} | "
            f"{process['sqrt_squared_energy']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Second X22 geometry",
            "",
            "| degree | mean sigma | 95% interval | min point ESS |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in x22:
        low, high = row["sigma_95_percent_ci"]
        lines.append(
            f"| {tuple(row['degree'])} | {row['mean_sigma']:.6f} | "
            f"[{low:.6f}, {high:.6f}] | {row['minimum_point_ess']:.1f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    metric_data = load(args.metric_audit)
    scalar_data = load(args.scalar_audit)
    scalar_stress_data = load(args.scalar_stress_audit)
    scalar_tail_data = load(args.scalar_tail_diagnostic)
    hard_training_data = load(args.hard_region_training_summary)
    hard_discovery_data = load(args.hard_region_discovery_audit)
    hard_blind_data = load(args.hard_region_blind_audit)
    hard_discovery_scalar_data = load(args.hard_region_discovery_scalar)
    hard_parallel_tail_data = load(args.hard_region_parallel_tail_diagnostic)
    multiseed_training_data = load(args.multiseed_training_summary)
    multiseed_discovery_serial_data = load(args.multiseed_discovery_serial)
    multiseed_discovery_parallel_data = load(args.multiseed_discovery_parallel)
    multiseed_discovery_scalar_data = load(args.multiseed_discovery_scalar)
    multiseed_discovery_parallel_tail_data = load(
        args.multiseed_discovery_parallel_tail
    )
    multiseed_blind_data = load(args.multiseed_blind_audit)
    multiseed_blind_scalar_data = load(args.multiseed_blind_scalar)
    multiseed_blind_tail_data = load(args.multiseed_blind_tail_diagnostic)
    l2_v1_training_data = load(args.l2_v1_training_summary)
    l2_v1_discovery_data = load(args.l2_v1_discovery_audit)
    l2_v2_training_data = load(args.l2_v2_training_summary)
    l2_v2_discovery_data = load(args.l2_v2_discovery_audit)
    l2_v2_process_data = load(args.l2_v2_process_discovery)
    l2_v2_blind_data = load(args.l2_v2_blind_audit)
    l2_v3_training_data = load(args.l2_v3_training_summary)
    l2_v3_failure_diagnostic_data = load(args.l2_v3_failure_diagnostic)
    l2_v4_status_data = parse_status(args.l2_v4_status)
    l2_v4_training_data = load_if_present(args.l2_v4_training_summary)
    l2_v4_checkpoint_data = load_if_present(args.l2_v4_checkpoint_audit)
    l2_v4_discovery_data = load_if_present(args.l2_v4_discovery_audit)
    l2_v4_process_discovery_data = load_if_present(args.l2_v4_process_discovery)
    l2_v4_one_time_check_data = load_if_present(args.l2_v4_one_time_check)
    l2_v4_process_blind_data = load_if_present(args.l2_v4_process_blind)
    l2_v4_blind_data = load_if_present(args.l2_v4_blind_audit)
    l2_v4_blind_scalar_data = load_if_present(args.l2_v4_blind_scalar)
    l2_v4_final_scalar_data = load_if_present(args.l2_v4_final_scalar)
    x22_training_data = load(args.x22_training_summary)
    x22_audit_data = load(args.x22_audit)
    x22_failed_atlas_data = load(args.x22_failed_atlas_audit)
    x22_atlas_witness_data = load(args.x22_atlas_witness)
    rank_data = load(args.rank_certificates)
    manifest_data = load(args.manifest)
    sources = [
        args.metric_audit,
        args.scalar_audit,
        args.scalar_stress_audit,
        args.scalar_tail_diagnostic,
        args.hard_region_training_summary,
        args.hard_region_discovery_audit,
        args.hard_region_blind_audit,
        args.hard_region_discovery_scalar,
        args.hard_region_parallel_tail_diagnostic,
        args.multiseed_training_summary,
        args.multiseed_discovery_serial,
        args.multiseed_discovery_parallel,
        args.multiseed_discovery_scalar,
        args.multiseed_discovery_parallel_tail,
        args.multiseed_blind_audit,
        args.multiseed_blind_scalar,
        args.multiseed_blind_tail_diagnostic,
        args.l2_v1_training_summary,
        args.l2_v1_discovery_audit,
        args.l2_v2_training_summary,
        args.l2_v2_discovery_audit,
        args.l2_v2_process_discovery,
        args.l2_v2_blind_audit,
        args.l2_v3_training_summary,
        args.l2_v3_failure_diagnostic,
        args.l2_v4_status,
        args.x22_training_summary,
        args.x22_audit,
        args.x22_failed_atlas_audit,
        args.x22_atlas_witness,
        args.rank_certificates,
        args.manifest,
    ]
    optional_v4_outputs = (
        args.l2_v4_training_summary,
        args.l2_v4_checkpoint_audit,
        args.l2_v4_discovery_audit,
        args.l2_v4_process_discovery,
        args.l2_v4_one_time_check,
        args.l2_v4_process_blind,
        args.l2_v4_blind_audit,
        args.l2_v4_blind_scalar,
        args.l2_v4_final_scalar,
    )
    sources.extend(path for path in optional_v4_outputs if path.is_file())
    sources.extend(
        ROOT / path
        for path in (
            "gcicy_metric/pipeline/train.py",
            "scripts/train_gcicy_pipeline.py",
            "scripts/audit_metric_volume_weight_tail.py",
            "scripts/run_metric_volume_tail_seed_parallel.py",
            "scripts/run_round4_l2_tail_resolution_remote.sh",
            "scripts/run_round4_l2_tail_resolution_v2_remote.sh",
            "scripts/run_round4_l2_tail_resolution_v3_remote.sh",
            "scripts/run_round4_v3_failure_serial_diagnosis_remote.sh",
            "scripts/run_round4_l2_tail_resolution_v4_remote.sh",
            "pipeline_specs/p4p1_type11_hirzebruch_k4_l2_tail_refine.json",
            "pipeline_specs/p4p1_type11_hirzebruch_k4_l2_tail_discovery_audit.json",
            "pipeline_specs/p4p1_type11_hirzebruch_k4_l2_tail_refine_v2.json",
            "pipeline_specs/p4p1_type11_hirzebruch_k4_l2_tail_discovery_v2_audit.json",
            "pipeline_specs/p4p1_type11_hirzebruch_k4_l2_tail_blind_v2_audit.json",
            "pipeline_specs/p4p1_type11_hirzebruch_k4_l2_tail_refine_v3.json",
            "pipeline_specs/p4p1_type11_hirzebruch_k4_l2_tail_discovery_v3_audit.json",
            "pipeline_specs/p4p1_type11_hirzebruch_k4_l2_tail_blind_v3_audit.json",
            "pipeline_specs/p4p1_type11_hirzebruch_k4_l2_tail_blind_v3_scalar.json",
            "pipeline_specs/p4p1_type11_hirzebruch_k4_l2_tail_v3_scalar_65536.json",
            "pipeline_specs/p4p1_type11_hirzebruch_k4_l2_tail_refine_v4.json",
            "pipeline_specs/p4p1_type11_hirzebruch_k4_l2_tail_checkpoint_v4_audit.json",
            "pipeline_specs/p4p1_type11_hirzebruch_k4_l2_tail_discovery_v4_audit.json",
            "pipeline_specs/p4p1_type11_hirzebruch_k4_l2_tail_check_v4_audit.json",
            "pipeline_specs/p4p1_type11_hirzebruch_k4_l2_tail_blind_v4_audit.json",
            "pipeline_specs/p4p1_type11_hirzebruch_k4_l2_tail_blind_v4_scalar.json",
            "pipeline_specs/p4p1_type11_hirzebruch_k4_l2_tail_v4_scalar_65536.json",
            "tests/test_round4_v4_protocol.py",
        )
    )
    sources.extend(
        workspace_artifact_path(artifact["path"])
        for artifact in metric_data["artifacts"]
    )
    sources.extend(
        workspace_artifact_path(artifact["path"])
        for artifact in x22_audit_data["artifacts"]
    )
    for data in (
        hard_discovery_data,
        hard_blind_data,
        multiseed_discovery_serial_data,
        multiseed_discovery_parallel_data,
        multiseed_blind_data,
    ):
        sources.extend(
            workspace_artifact_path(artifact["path"]) for artifact in data["artifacts"]
        )
    sources.append(workspace_artifact_path(hard_training_data["artifact"]))
    sources.append(workspace_artifact_path(multiseed_training_data["artifact"]))
    sources.append(workspace_artifact_path(l2_v1_training_data["artifact"]))
    sources.append(workspace_artifact_path(l2_v2_training_data["artifact"]))
    sources.append(workspace_artifact_path(l2_v3_training_data["artifact"]))
    if l2_v4_training_data is not None:
        sources.append(workspace_artifact_path(l2_v4_training_data["artifact"]))
    sources.extend(
        [
            ROOT
            / "outputs/pipeline/p4p1_type11_hirzebruch_k4_tail_refined_gpu_summary.json",
            ROOT
            / "outputs/pipeline/p4p1_type11_hirzebruch_k4_tail_refined_replica_a_summary.json",
            ROOT
            / "outputs/pipeline/p4p1_type11_hirzebruch_k4_tail_refined_replica_b_summary.json",
            ROOT / "outputs/gcicy_model_20260712_k1_weighted_summary.json",
            ROOT
            / "outputs/pipeline/p1p1p5_type22_parallel_sampling_smoke_summary.json",
        ]
    )
    scalar_paths = [
        args.scalar_audit,
        args.hard_region_discovery_scalar,
        args.multiseed_discovery_scalar,
        args.multiseed_blind_scalar,
    ]
    scalar_paths.extend(
        path
        for path in (args.l2_v4_blind_scalar, args.l2_v4_final_scalar)
        if path.is_file()
    )
    for scalar_path in scalar_paths:
        worker_dir = scalar_path.parent / f"{scalar_path.stem}_workers"
        sources.extend(sorted(worker_dir.glob("seed_*")))
    tail_paths = [
        args.l2_v1_discovery_audit,
        args.l2_v2_discovery_audit,
        args.l2_v2_blind_audit,
    ]
    tail_paths.extend(
        path
        for path in (
            args.l2_v4_checkpoint_audit,
            args.l2_v4_discovery_audit,
            args.l2_v4_one_time_check,
            args.l2_v4_blind_audit,
        )
        if path.is_file()
    )
    for tail_path in tail_paths:
        worker_dir = tail_path.parent / f"{tail_path.stem}_workers"
        sources.extend(sorted(worker_dir.glob("seed_*")))
    run_dir = ROOT / "outputs/pipeline/reviewer_round4_remote"
    sources.extend(
        run_dir / name
        for name in (
            "status.txt",
            "preflight.log",
            "parallel_sampling_smoke.log",
            "x11_k4_replica_a.log",
            "x11_k4_replica_b.log",
            "x11_k4_replicas_metric_audit.log",
            "x11_k4_replicas_scalar.log",
        )
    )
    hard_run_dir = ROOT / "outputs/pipeline/reviewer_round4_hard_region_followup"
    sources.extend(
        hard_run_dir / name
        for name in (
            "status.txt",
            "discovery_audit.log",
            "discovery_scalar.log",
            "blind_audit.log",
        )
    )
    multiseed_run_dir = ROOT / "outputs/pipeline/reviewer_round4_multiseed_refinement"
    sources.extend(
        multiseed_run_dir / name
        for name in (
            "status.txt",
            "preflight.log",
            "training.log",
            "provenance.log",
            "provenance.json",
            "discovery_serial_metric.log",
            "discovery_serial_scalar.log",
            "discovery_parallel_metric.log",
            "discovery_parallel_tail.log",
            "discovery_parallel_tail_gate.log",
            "blind_metric.log",
            "blind_scalar.log",
        )
    )
    l2_v1_run_dir = ROOT / "outputs/pipeline/reviewer_round4_l2_tail_resolution"
    sources.extend(
        l2_v1_run_dir / name
        for name in (
            "status.txt",
            "preflight.log",
            "training.log",
            "provenance.log",
            "provenance.json",
            "discovery_serial.log",
            "discovery_serial_gate.log",
        )
    )
    l2_v2_run_dir = ROOT / "outputs/pipeline/reviewer_round4_l2_tail_resolution_v2"
    sources.extend(
        l2_v2_run_dir / name
        for name in (
            "status.txt",
            "preflight.log",
            "training.log",
            "provenance.log",
            "provenance.json",
            "discovery_serial.log",
            "discovery_serial_gate.log",
            "discovery_process.log",
            "discovery_process_gate.log",
            "blind_tail.log",
        )
    )
    l2_v3_run_dir = ROOT / "outputs/pipeline/reviewer_round4_l2_tail_resolution_v3"
    sources.extend(sorted(path for path in l2_v3_run_dir.glob("*") if path.is_file()))
    l2_v3_diagnosis_dir = (
        ROOT / "outputs/pipeline/reviewer_round4_v3_failure_serial_diagnosis"
    )
    sources.extend(
        sorted(path for path in l2_v3_diagnosis_dir.glob("*") if path.is_file())
    )
    l2_v4_run_dir = ROOT / "outputs/pipeline/reviewer_round4_l2_tail_resolution_v4"
    sources.extend(sorted(path for path in l2_v4_run_dir.glob("*") if path.is_file()))
    x22_run_dir = ROOT / "outputs/pipeline/reviewer_round4_x22_recovery"
    sources.extend(
        x22_run_dir / name
        for name in (
            "status.txt",
            "x22_seed20260712_k3.log",
            "x22_seed20260712_k1_k3_audit.log",
        )
    )
    x22_witness_dir = ROOT / "outputs/pipeline/reviewer_round4_x22_atlas_witness"
    sources.extend(
        x22_witness_dir / name
        for name in (
            "status.txt",
            "atlas_witness.log",
            "verify_expected_failure.log",
        )
    )
    x22_resolution_dir = ROOT / "outputs/pipeline/reviewer_round4_x22_atlas_resolution"
    sources.extend(
        x22_resolution_dir / name
        for name in (
            "status.txt",
            "formal_audit.log",
            "section_rank_certificates.log",
        )
    )
    pilot_provenance = (
        ROOT / "outputs/pipeline/reviewer_round4_hard_region_pilot/provenance.json"
    )
    sources.append(pilot_provenance)
    master_log = ROOT / "outputs/pipeline/reviewer_round4_remote_master.log"
    if master_log.is_file():
        sources.append(master_log)
    sources = sorted({path.expanduser().resolve() for path in sources})
    missing_sources = [path for path in sources if not path.is_file()]
    if missing_sources:
        raise SystemExit(
            f"round-four source inventory is incomplete: {missing_sources}"
        )

    metrics, metric_summary = metric_rows(metric_data)
    spectra, scalar_summary = scalar_rows(scalar_data)
    tail_summary = tail_diagnostic_summary(scalar_tail_data, scalar_stress_data)
    first_adaptation = first_hard_region_summary(
        hard_training_data,
        hard_discovery_data,
        hard_blind_data,
        hard_discovery_scalar_data,
        hard_parallel_tail_data,
    )
    multiseed = multiseed_refinement_summary(
        multiseed_training_data,
        multiseed_discovery_serial_data,
        multiseed_discovery_parallel_data,
        multiseed_discovery_scalar_data,
        multiseed_discovery_parallel_tail_data,
        multiseed_blind_data,
        multiseed_blind_scalar_data,
        multiseed_blind_tail_data,
    )
    l2_resolution = l2_tail_resolution_v4_summary(
        l2_v1_training_data,
        l2_v1_discovery_data,
        l2_v2_training_data,
        l2_v2_discovery_data,
        l2_v2_process_data,
        l2_v2_blind_data,
        l2_v3_training_data,
        l2_v3_failure_diagnostic_data,
        l2_v4_status_data,
        l2_v4_training_data,
        l2_v4_checkpoint_data,
        l2_v4_discovery_data,
        l2_v4_process_discovery_data,
        l2_v4_one_time_check_data,
        l2_v4_process_blind_data,
        l2_v4_blind_data,
        l2_v4_blind_scalar_data,
        l2_v4_final_scalar_data,
    )
    manifest_consistency = l2_manifest_consistency_v4(l2_resolution, manifest_data)
    if not manifest_consistency["all_checks_passed"]:
        failed_manifest_checks = [
            key for key, passed in manifest_consistency["checks"].items() if not passed
        ]
        raise SystemExit(
            "round-four outputs do not match the frozen manifest: "
            f"{failed_manifest_checks}"
        )
    l2_resolution["frozen_manifest_consistency"] = manifest_consistency
    l2_resolution["preservation_checks"]["frozen_manifest_consistency"] = (
        manifest_consistency["all_checks_passed"]
    )
    l2_resolution["all_final_acceptance_gates_passed"] = bool(
        all(l2_resolution["preservation_checks"].values())
        and l2_resolution["v4"]["all_terminal_acceptance_gates_passed"]
    )
    x22, x22_summary = x22_rows(x22_training_data, x22_audit_data)
    x22_atlas_resolution = x22_atlas_resolution_summary(
        x22_failed_atlas_data,
        x22_atlas_witness_data,
        x22_audit_data,
    )
    certificate = rank_certificate(rank_data)
    output = {
        "schema_version": 2,
        "description": (
            "Frozen round-four optimizer-seed, terminal-tail, and cross-geometry "
            "robustness evidence."
        ),
        "compute_policy": "all scientific recomputation executed on the remote RTX 4090 host",
        "x11_optimizer_replicas": metrics,
        "x11_optimizer_replica_summary": metric_summary,
        "x11_optimizer_replica_scalar_spectra": spectra,
        "x11_optimizer_replica_scalar_summary": scalar_summary,
        "x11_optimizer_replica_scalar_tail_diagnosis": tail_summary,
        "x11_first_hard_region_adaptation": first_adaptation,
        "x11_multiseed_hard_region_resolution": multiseed,
        "x11_l2_volume_ratio_tail_resolution": l2_resolution,
        "x22_second_geometry": x22,
        "x22_second_geometry_summary": x22_summary,
        "x22_second_geometry_atlas_resolution": x22_atlas_resolution,
        "x22_second_geometry_section_rank_certificate": certificate,
        "registered_acceptance_outcome": {
            "x11_maximum_mean_sigma_at_most_0_08": metric_summary["maximum_mean_sigma"]
            <= 0.08,
            "x11_mean_sigma_cv_at_most_0_10": metric_summary["mean_sigma_cv"] <= 0.10,
            "x11_scalar_registered_gates_passed": scalar_summary["registered_success"],
            "x22_strict_registered_audit_passed": x22_summary["strict_audit_success"],
            "x22_k3_mean_sigma_at_most_0_18": x22[-1]["mean_sigma"] <= 0.18,
            "x22_k3_improves_k1_on_every_seed": x22_summary[
                "ordered_every_seed_improvement"
            ],
            "x22_k3_reported_section_basis_exactly_certified": certificate[
                "reported_artifact_rank"
            ]
            == 251,
        },
        "publication_readiness": {
            "x11_optimizer_replica_metric_gates_passed": True,
            "registered_scalar_failure_preserved": not scalar_summary[
                "registered_success"
            ],
            "multiseed_blind_scalar_failure_preserved": not multiseed[
                "all_adaptive_acceptance_gates_passed"
            ],
            "l2_volume_ratio_tail_resolution_passed": l2_resolution[
                "all_final_acceptance_gates_passed"
            ],
            "x22_strict_audit_passed": x22_summary["strict_audit_success"],
            "x22_k3_mean_sigma_at_most_0_18": x22[-1]["mean_sigma"] <= 0.18,
            "x22_k3_improves_k1_on_every_seed": x22_summary[
                "ordered_every_seed_improvement"
            ],
            "x22_k3_reported_section_basis_exactly_certified": certificate[
                "reported_artifact_rank"
            ]
            == 251,
        },
        "source_files": [
            {"path": relative(path), "sha256": sha256(path)} for path in sources
        ],
    }
    output["all_registered_round4_gates_passed"] = bool(
        all(output["registered_acceptance_outcome"].values())
    )
    output["publication_readiness_gates_passed"] = bool(
        all(output["publication_readiness"].values())
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    write_tex(
        args.tex_out,
        metrics,
        metric_summary,
        spectra,
        scalar_summary,
        x22,
        x22_summary,
        first_adaptation,
        multiseed,
        l2_resolution,
    )
    write_markdown(
        args.markdown_out,
        metrics,
        spectra,
        x22,
        scalar_summary,
        x22_summary,
        first_adaptation,
        multiseed,
        l2_resolution,
    )
    print(f"wrote {args.out}")
    print(f"wrote {args.tex_out}")
    print(
        "all_registered_round4_gates_passed="
        f"{output['all_registered_round4_gates_passed']}"
    )
    print(
        "publication_readiness_gates_passed="
        f"{output['publication_readiness_gates_passed']}"
    )


if __name__ == "__main__":
    main()
