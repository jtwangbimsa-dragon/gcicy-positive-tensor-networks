#!/usr/bin/env python3
"""Summarize the preregistered type-(2,1) oracle rank refinement."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


ARMS = ("q20_dense_restart", "q22_dense_restart")
BLIND_SUFFIX = "blind83463_n65532"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--frontier-summary", type=Path, required=True)
    parser.add_argument("--q24-posthoc", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
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


def metrics(report: dict, *, teacher: bool = False) -> dict[str, float]:
    root = report["metrics"]
    ma = root["teacher_ma_errors" if teacher else "compressed_ma_errors"]
    result = {
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
    }
    if not teacher:
        result.update(
            {
                "potential_rms_to_teacher": float(root["potential_rms_to_teacher"]),
                "affine_metric_rms_to_teacher": float(
                    root["affine_metric_rms_to_teacher"]
                ),
                "minimum_metric_eigenvalue": float(
                    root["minimum_metric_eigenvalue"]
                ),
            }
        )
    return result


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    frontier_path = args.frontier_summary.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    manifest = load_json(manifest_path)
    frontier = load_json(frontier_path)
    reports = {
        arm: {
            "initial": load_json(
                run_dir / f"{arm}_initial_{BLIND_SUFFIX}.json"
            ),
            "training": load_json(run_dir / f"{arm}_energy_oracle_summary.json"),
            "oracle": load_json(
                run_dir / f"{arm}_energy_oracle_{BLIND_SUFFIX}.json"
            ),
        }
        for arm in ARMS
    }

    training_spec = manifest["training"]
    blind_spec = manifest["blind"]
    expected_loss = {
        "potential": training_spec["loss_weights"][
            "centered_potential_variance"
        ],
        "metric": training_spec["loss_weights"][
            "relative_metric_log_eigenvalue_mse"
        ],
        "log_energy": training_spec["loss_weights"]["centered_log_energy"],
        "ma": training_spec["loss_weights"]["monge_ampere_e2"],
        "tail": training_spec["loss_weights"]["upper_tail_cvar"],
    }
    blind_reports = [
        reports[arm][stage]
        for arm in ARMS
        for stage in ("initial", "oracle")
    ]
    protocol_checks = {
        "frontier_trigger_q16_failed_q24_passed": (
            not frontier["arms"]["q16_dense_restart"]["strong_oracle_passed"]
            and frontier["arms"]["q24_dense_restart"]["strong_oracle_passed"]
            and int(frontier["promoted_dense_restart_rank"]) == 24
        ),
        "common_fresh_blind_seed": all(
            int(report["seed"]) == int(blind_spec["seed"])
            for report in blind_reports
        ),
        "common_blind_count": all(
            int(report["points"]) == int(blind_spec["points"])
            for report in blind_reports
        ),
        "initial_hashes_match_manifest": all(
            reports[arm]["initial"]["model_sha256"]
            == manifest["arms"][arm]["initial_sha256"]
            for arm in ARMS
        ),
        "trained_hashes_match_blind": all(
            reports[arm]["training"]["model_sha256"]
            == reports[arm]["oracle"]["model_sha256"]
            for arm in ARMS
        ),
        "teacher_hash_matches_manifest": all(
            reports[arm]["training"]["teacher_artifact_sha256"]
            == manifest["fixed_artifacts"]["energy_full_h_sha256"]
            for arm in ARMS
        ),
        "teacher_unchanged_during_training": all(
            reports[arm]["training"]["initialization"]["teacher_transition"]
            == "teacher_unchanged"
            for arm in ARMS
        ),
        "rank_and_parameter_counts_match_manifest": all(
            int(reports[arm]["oracle"]["physical_dictionary_rank"])
            == int(manifest["arms"][arm]["rank"])
            and int(reports[arm]["oracle"]["trainable_real_parameter_count"])
            == int(manifest["arms"][arm]["trainable_real_parameter_count"])
            for arm in ARMS
        ),
        "training_protocol_matches_manifest": all(
            reports[arm]["training"]["loss_weights"] == expected_loss
            and int(reports[arm]["training"]["train"]["seed"])
            == int(training_spec["train_seed"])
            and int(reports[arm]["training"]["train"]["points"])
            == int(training_spec["train_points"])
            and int(reports[arm]["training"]["validation"]["seed"])
            == int(training_spec["validation_seed"])
            and int(reports[arm]["training"]["validation"]["points"])
            == int(training_spec["validation_points"])
            and int(reports[arm]["training"]["epochs"])
            == int(training_spec["epochs"])
            and math.isclose(
                float(reports[arm]["training"]["learning_rate"]),
                float(training_spec["learning_rate"]),
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
            for arm in ARMS
        ),
    }
    failed = [name for name, passed in protocol_checks.items() if not passed]
    if failed:
        raise ValueError("failed protocol checks: " + ", ".join(failed))

    teacher_rows = [metrics(report, teacher=True) for report in blind_reports]
    teacher = teacher_rows[0]
    if any(
        not math.isclose(
            teacher[key], row[key], rel_tol=1.0e-13, abs_tol=1.0e-14
        )
        for row in teacher_rows[1:]
        for key in teacher
    ):
        raise ValueError("energy full-H metrics differ across common blind reports")
    thresholds = {
        "chi": 1.25 * teacher["chi"] + 0.02,
        "positive_log_ratio_q999": 1.50 * teacher["positive_log_ratio_q999"]
        + 0.10,
        "positive_log_ratio_cvar_1pct": 1.50
        * teacher["positive_log_ratio_cvar_1pct"]
        + 0.10,
        "normalized_ratio_above_3_weighted_mass": teacher[
            "normalized_ratio_above_3_weighted_mass"
        ]
        + 0.002,
        "potential_rms_to_teacher": 0.01,
        "affine_metric_rms_to_teacher": 0.03,
    }

    arm_rows = {}
    for arm in ARMS:
        initial = metrics(reports[arm]["initial"])
        trained = metrics(reports[arm]["oracle"])
        gates = {name: trained[name] <= value for name, value in thresholds.items()}
        gates["minimum_metric_eigenvalue"] = trained["minimum_metric_eigenvalue"] > 0
        best_epoch = int(reports[arm]["training"]["best_epoch"])
        epochs = int(reports[arm]["training"]["epochs"])
        arm_rows[arm] = {
            "rank": int(manifest["arms"][arm]["rank"]),
            "trainable_real_parameter_count": int(
                manifest["arms"][arm]["trainable_real_parameter_count"]
            ),
            "k20_extrapolated_parameter_count": int(
                manifest["arms"][arm]["k20_extrapolated_parameter_count"]
            ),
            "retained_squared_frobenius_energy_fraction": float(
                manifest["arms"][arm][
                    "retained_squared_frobenius_energy_fraction"
                ]
            ),
            "initial_metrics": initial,
            "trained_metrics": trained,
            "changes": {key: trained[key] - initial[key] for key in trained},
            "gates": gates,
            "strong_oracle_passed": bool(all(gates.values())),
            "best_epoch": best_epoch,
            "best_checkpoint_at_final_epoch": best_epoch == epochs,
            "artifacts": {
                "initial": str(
                    run_dir / f"{arm}_initial_{BLIND_SUFFIX}.json"
                ),
                "training": str(run_dir / f"{arm}_energy_oracle_summary.json"),
                "oracle": str(
                    run_dir / f"{arm}_energy_oracle_{BLIND_SUFFIX}.json"
                ),
            },
        }

    if arm_rows["q20_dense_restart"]["strong_oracle_passed"]:
        selected_arm = "q20_dense_restart"
        selected_rank = 20
        selected_source = "current_refinement"
    elif arm_rows["q22_dense_restart"]["strong_oracle_passed"]:
        selected_arm = "q22_dense_restart"
        selected_rank = 22
        selected_source = "current_refinement"
    else:
        selected_arm = "q24_dense_restart"
        selected_rank = 24
        selected_source = "previous_frontier"

    posthoc_same_seed_q24 = None
    if args.q24_posthoc is not None:
        posthoc_path = args.q24_posthoc.expanduser().resolve()
        posthoc_report = load_json(posthoc_path)
        q24_training = load_json(
            frontier_path.parent / "q24_dense_restart_energy_oracle_summary.json"
        )
        posthoc_teacher = metrics(posthoc_report, teacher=True)
        posthoc_checks = {
            "same_blind_seed": int(posthoc_report["seed"])
            == int(blind_spec["seed"]),
            "same_blind_count": int(posthoc_report["points"])
            == int(blind_spec["points"]),
            "q24_model_hash_matches_frontier": posthoc_report["model_sha256"]
            == q24_training["model_sha256"],
            "q24_rank_and_parameter_count": (
                int(posthoc_report["physical_dictionary_rank"]) == 24
                and int(posthoc_report["trainable_real_parameter_count"]) == 8688
            ),
            "same_energy_full_h_metrics": all(
                math.isclose(
                    teacher[key],
                    posthoc_teacher[key],
                    rel_tol=1.0e-13,
                    abs_tol=1.0e-14,
                )
                for key in teacher
            ),
        }
        failed_posthoc = [
            name for name, passed in posthoc_checks.items() if not passed
        ]
        if failed_posthoc:
            raise ValueError(
                "failed post-hoc checks: " + ", ".join(failed_posthoc)
            )
        q22_metrics = arm_rows["q22_dense_restart"]["trained_metrics"]
        q24_metrics = metrics(posthoc_report)
        comparison_keys = (
            "sigma",
            "chi",
            "positive_log_ratio_q999",
            "positive_log_ratio_cvar_1pct",
            "potential_rms_to_teacher",
            "affine_metric_rms_to_teacher",
        )
        posthoc_same_seed_q24 = {
            "excluded_from_registered_selection": True,
            "checks": posthoc_checks,
            "artifact": str(posthoc_path),
            "artifact_sha256": sha256_file(posthoc_path),
            "q24_metrics": q24_metrics,
            "q22_minus_q24": {
                key: q22_metrics[key] - q24_metrics[key]
                for key in comparison_keys
            },
            "q22_relative_increase_over_q24": {
                key: (q22_metrics[key] - q24_metrics[key]) / q24_metrics[key]
                for key in comparison_keys
            },
            "parameter_reduction_from_q24_to_q22": 8688 - 7964,
            "parameter_reduction_fraction_from_q24_to_q22": (
                8688 - 7964
            )
            / 8688,
        }

    report = {
        "schema": "positive-tensor-network-energy-oracle-rank-refinement-summary-v1",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "frontier_summary": str(frontier_path),
        "frontier_summary_sha256": sha256_file(frontier_path),
        "protocol_checks": protocol_checks,
        "energy_full_h_blind_metrics": teacher,
        "thresholds": thresholds,
        "arms": arm_rows,
        "selected_arm": selected_arm,
        "selected_rank": selected_rank,
        "selected_source": selected_source,
        "posthoc_same_seed_q24": posthoc_same_seed_q24,
        "selection_rule": manifest["selection_rule"],
        "claim_boundary": manifest["claim_boundary"],
    }
    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_suffix(out_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(out_path)
    print(f"selected_arm={selected_arm}")
    print(f"selected_rank={selected_rank}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
