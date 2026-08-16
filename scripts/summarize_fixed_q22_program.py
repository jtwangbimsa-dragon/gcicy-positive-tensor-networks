#!/usr/bin/env python3
"""Summarize the preregistered fixed-q22 degree and geometry program."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--type21-manifest", type=Path, required=True)
    parser.add_argument("--type21-run-dir", type=Path, required=True)
    parser.add_argument("--type22-manifest", type=Path, required=True)
    parser.add_argument("--type22-run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_metrics(report: dict) -> dict[str, float]:
    root = report["metrics"]
    ma = root["compressed_ma_errors"]
    return {
        "sigma": float(ma["sigma"]),
        "chi": float(ma["sqrt_squared_energy"]),
        "positive_log_ratio_q999": float(ma["positive_log_ratio_q999"]),
        "positive_log_ratio_cvar_1pct": float(
            ma["positive_log_ratio_cvar_1pct"]
        ),
        "normalized_ratio_min": float(ma["normalized_ratio_min"]),
        "normalized_ratio_max": float(ma["normalized_ratio_max"]),
        "normalized_ratio_above_3_weighted_mass": float(
            ma["normalized_ratio_above_3_weighted_mass"]
        ),
        "minimum_metric_eigenvalue": float(root["minimum_metric_eigenvalue"]),
    }


def arm_paths(run_dir: Path, arm: str, blind_seed: int) -> tuple[Path, Path]:
    return (
        run_dir / f"q22_{arm}_teacher_free_summary.json",
        run_dir / f"q22_{arm}_teacher_free_blind{blind_seed}_n65532.json",
    )


def load_arm(
    manifest: dict,
    run_dir: Path,
    arm: str,
) -> tuple[dict, dict, dict[str, bool]]:
    arm_spec = manifest["arms"][arm]
    common = manifest["common_training"]
    training_path, blind_path = arm_paths(run_dir, arm, arm_spec["blind_seed"])
    training = load_json(training_path)
    blind = load_json(blind_path)
    expected_loss = {
        "potential": 0.0,
        "metric": 0.0,
        "log_energy": common["loss_weights"]["centered_log_energy"],
        "ma": common["loss_weights"]["monge_ampere_e2"],
        "tail": common["loss_weights"]["upper_tail_cvar"],
    }
    checks = {
        "model_hash_matches_blind": training["model_sha256"]
        == blind["model_sha256"],
        "adapter_matches": training["adapter"] == manifest["geometry"]["adapter"]
        and blind["adapter"] == manifest["geometry"]["adapter"],
        "model_seed_matches": int(training["model_seed"])
        == int(manifest["geometry"]["model_seed"])
        and int(blind["model_seed"]) == int(manifest["geometry"]["model_seed"]),
        "source_hash_matches": training["source_artifact_sha256"]
        == manifest["fixed_artifacts"]["source_sha256"],
        "teacher_free": training["teacher_artifact_sha256"] is None
        and training["training_mode"] == "teacher_free_geometric"
        and training["teacher_artifact"] is None
        and blind["teacher_artifact"] is None,
        "architecture_matches": training["architecture"]
        == "shared_local_dictionary"
        and int(training["bond_dimension"])
        == int(manifest["architecture"]["bond_dimension"])
        and int(training["physical_dictionary_rank"])
        == int(manifest["architecture"]["physical_dictionary_rank"])
        and int(training["trainable_real_parameter_count"])
        == int(arm_spec["trainable_real_parameter_count"])
        and int(blind["trainable_real_parameter_count"])
        == int(arm_spec["trainable_real_parameter_count"]),
        "degree_and_seeds_match": int(training["site_count"])
        == int(arm_spec["site_count"])
        and int(blind["site_count"]) == int(arm_spec["site_count"])
        and int(training["train"]["seed"]) == int(arm_spec["train_seed"])
        and int(training["validation"]["seed"])
        == int(arm_spec["validation_seed"])
        and int(blind["seed"]) == int(arm_spec["blind_seed"]),
        "sample_counts_match": int(training["train"]["points"])
        == int(common["train_points"])
        and int(training["validation"]["points"])
        == int(common["validation_points"])
        and int(blind["points"]) == int(arm_spec["blind_points"]),
        "optimizer_protocol_matches": int(training["epochs"])
        == int(common["epochs"])
        and int(training["batch_size"])
        == int(common["global_point_batch_size"])
        and math.isclose(
            float(training["learning_rate"]),
            float(common["learning_rate"]),
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        and math.isclose(
            float(training["gradient_clip_norm"]),
            float(common["gradient_clip_norm"]),
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        and training["loss_weights"] == expected_loss
        and training["termination_reason"] == "completed_requested_epochs",
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"{arm} failed protocol checks: {', '.join(failed)}")
    metrics = audit_metrics(blind)
    row = {
        "site_count": int(arm_spec["site_count"]),
        "target_degree": arm_spec["target_degree"],
        "trainable_real_parameter_count": int(
            arm_spec["trainable_real_parameter_count"]
        ),
        "metrics": metrics,
        "stability_gates": {
            "weighted_probability_r_above_3": metrics[
                "normalized_ratio_above_3_weighted_mass"
            ]
            <= 0.002,
            "minimum_metric_eigenvalue": metrics["minimum_metric_eigenvalue"] > 0,
        },
        "best_epoch": int(training["best_epoch"]),
        "best_checkpoint_at_final_epoch": int(training["best_epoch"])
        == int(training["epochs"]),
        "runtime_seconds": float(training["runtime_seconds"]),
        "model_sha256": training["model_sha256"],
        "artifacts": {
            "training": str(training_path),
            "blind": str(blind_path),
        },
    }
    return row, training, checks


def preservation_gates(previous: dict, current: dict) -> dict[str, bool]:
    old = previous["metrics"]
    new = current["metrics"]
    return {
        "sigma": new["sigma"] <= 1.20 * old["sigma"] + 0.01,
        "chi": new["chi"] <= 1.25 * old["chi"] + 0.02,
        "positive_log_ratio_q999": new["positive_log_ratio_q999"]
        <= 1.25 * old["positive_log_ratio_q999"] + 0.05,
        "positive_log_ratio_cvar_1pct": new[
            "positive_log_ratio_cvar_1pct"
        ]
        <= 1.25 * old["positive_log_ratio_cvar_1pct"] + 0.05,
        "weighted_probability_r_above_3": new[
            "normalized_ratio_above_3_weighted_mass"
        ]
        <= old["normalized_ratio_above_3_weighted_mass"] + 0.002,
    }


def main() -> None:
    args = parse_args()
    type21_manifest_path = args.type21_manifest.expanduser().resolve()
    type22_manifest_path = args.type22_manifest.expanduser().resolve()
    type21_run_dir = args.type21_run_dir.expanduser().resolve()
    type22_run_dir = args.type22_run_dir.expanduser().resolve()
    type21_manifest = load_json(type21_manifest_path)
    type22_manifest = load_json(type22_manifest_path)

    type21_rows = {}
    type21_checks = {}
    for arm in ("k4", "k6", "k8"):
        row, _, checks = load_arm(type21_manifest, type21_run_dir, arm)
        type21_rows[arm] = row
        type21_checks[arm] = checks
    type21_rows["k6"]["preservation_gates_relative_to_k4"] = preservation_gates(
        type21_rows["k4"], type21_rows["k6"]
    )
    type21_rows["k8"]["preservation_gates_relative_to_k6"] = preservation_gates(
        type21_rows["k6"], type21_rows["k8"]
    )
    expansion = load_json(
        type21_run_dir / "q22_k4_rank_expanded_initial_summary.json"
    )
    expansion_checks = {
        "source_hash_matches_manifest": expansion["source_model_sha256"]
        == type21_manifest["fixed_artifacts"]["teacher_free_q12_k4_sha256"],
        "ranks_match": int(expansion["source_rank"]) == 12
        and int(expansion["target_rank"]) == 22,
        "parameter_count_matches": int(expansion["trainable_real_parameter_count"])
        == int(type21_manifest["arms"]["k4"]["trainable_real_parameter_count"]),
        "lossless_core_gate": float(expansion["relative_core_error"])
        <= float(type21_manifest["rank_expansion"]["required_relative_dense_core_error"]),
        "teacher_free": expansion["teacher_artifact_sha256"] is None,
    }
    if not all(expansion_checks.values()):
        raise ValueError("type21 rank-expansion checks failed")

    type22_k4, _, type22_k4_checks = load_arm(
        type22_manifest, type22_run_dir, "k4"
    )
    type22_trigger_passed = bool(
        all(type22_k4["stability_gates"].values())
    )
    type22_k6_training, type22_k6_blind = arm_paths(
        type22_run_dir,
        "k6",
        type22_manifest["arms"]["k6"]["blind_seed"],
    )
    type22_k6_present = type22_k6_training.exists() and type22_k6_blind.exists()
    if type22_trigger_passed and not type22_k6_present:
        raise ValueError("type22 k6 trigger passed but k6 artifacts are missing")
    type22_rows = {"k4": type22_k4}
    type22_checks = {"k4": type22_k4_checks}
    if type22_k6_present:
        type22_k6, _, type22_k6_checks = load_arm(
            type22_manifest, type22_run_dir, "k6"
        )
        type22_k6["preservation_gates_relative_to_k4"] = preservation_gates(
            type22_k4, type22_k6
        )
        type22_rows["k6"] = type22_k6
        type22_checks["k6"] = type22_k6_checks
    else:
        type22_rows["k6"] = {
            "status": "skipped_by_preregistered_trigger",
            "trigger_passed": False,
        }
    initialization = load_json(
        type22_run_dir / "q22_k4_h_aligned_initial_summary.json"
    )
    initialization_checks = {
        "source_hash_matches_manifest": initialization["source_artifact_sha256"]
        == type22_manifest["fixed_artifacts"]["source_sha256"],
        "rank_matches": int(initialization["physical_dictionary_rank"]) == 22,
        "parameter_count_matches": int(initialization["trainable_real_parameter_count"])
        == int(type22_manifest["arms"]["k4"]["trainable_real_parameter_count"]),
        "reference_reconstruction_gate": float(
            initialization["maximum_reference_core_reconstruction_error"]
        )
        <= 1.0e-12,
        "teacher_free": initialization["teacher_artifact_sha256"] is None,
    }
    if not all(initialization_checks.values()):
        raise ValueError("type22 initialization checks failed")

    report = {
        "schema": "fixed-q22-degree-and-geometry-program-summary-v1",
        "type21_manifest": str(type21_manifest_path),
        "type21_manifest_sha256": sha256_file(type21_manifest_path),
        "type22_manifest": str(type22_manifest_path),
        "type22_manifest_sha256": sha256_file(type22_manifest_path),
        "architecture": {"bond_dimension": 5, "physical_dictionary_rank": 22},
        "type21": {
            "rank_expansion_checks": expansion_checks,
            "protocol_checks": type21_checks,
            "arms": type21_rows,
            "claim_boundary": type21_manifest["claim_boundary"],
        },
        "type22": {
            "initialization_checks": initialization_checks,
            "protocol_checks": type22_checks,
            "k6_trigger_passed": type22_trigger_passed,
            "arms": type22_rows,
            "claim_boundary": type22_manifest["claim_boundary"],
        },
    }
    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_suffix(out_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(out_path)
    print(
        "type21="
        + ",".join(
            f"{arm}:chi={type21_rows[arm]['metrics']['chi']:.6f}"
            for arm in ("k4", "k6", "k8")
        )
    )
    print(f"type22_k6_trigger_passed={type22_trigger_passed}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
