#!/usr/bin/env python3
"""Apply the registered gates to the type-(2,1) energy-oracle audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


ARM_FILES = {
    "dense_d5": {
        "initial": "dense_d5_initial_blind83443_n65532.json",
        "training": "dense_d5_energy_oracle_summary.json",
        "oracle": "dense_d5_energy_oracle_blind83443_n65532.json",
    },
    "shared_q12_d5": {
        "initial": "q12_d5_initial_blind83443_n65532.json",
        "training": "q12_d5_energy_oracle_summary.json",
        "oracle": "q12_d5_energy_oracle_blind83443_n65532.json",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
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


def selected_metrics(report: dict, *, teacher: bool = False) -> dict[str, float]:
    metric_root = report["metrics"]
    ma = metric_root["teacher_ma_errors" if teacher else "compressed_ma_errors"]
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
                "potential_rms_to_teacher": float(
                    metric_root["potential_rms_to_teacher"]
                ),
                "affine_metric_rms_to_teacher": float(
                    metric_root["affine_metric_rms_to_teacher"]
                ),
                "minimum_metric_eigenvalue": float(
                    metric_root["minimum_metric_eigenvalue"]
                ),
            }
        )
    return result


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    manifest = load_json(manifest_path)
    reports = {
        arm: {
            name: load_json(run_dir / filename)
            for name, filename in filenames.items()
        }
        for arm, filenames in ARM_FILES.items()
    }

    blind_spec = manifest["blind"]
    training_spec = manifest["training"]
    artifact_spec = manifest["artifacts"]
    all_blind = [
        reports[arm][stage]
        for arm in ARM_FILES
        for stage in ("initial", "oracle")
    ]
    common_checks = {
        "blind_seed_matches_manifest": all(
            int(report["seed"]) == int(blind_spec["seed"])
            for report in all_blind
        ),
        "blind_count_matches_manifest": all(
            int(report["points"]) == int(blind_spec["points"])
            for report in all_blind
        ),
        "adapter_matches_manifest": all(
            report["adapter"] == manifest["geometry"]["adapter"]
            for report in all_blind
        ),
        "energy_teacher_hash_matches_manifest": all(
            reports[arm]["training"]["teacher_artifact_sha256"]
            == artifact_spec["energy_full_h_sha256"]
            for arm in ARM_FILES
        ),
        "initial_hashes_match_manifest": (
            reports["dense_d5"]["initial"]["model_sha256"]
            == artifact_spec["dense_teacher_free_initial_sha256"]
            and reports["shared_q12_d5"]["initial"]["model_sha256"]
            == artifact_spec["q12_teacher_free_initial_sha256"]
        ),
        "oracle_model_hashes_match_training": all(
            reports[arm]["oracle"]["model_sha256"]
            == reports[arm]["training"]["model_sha256"]
            for arm in ARM_FILES
        ),
        "teacher_was_introduced": all(
            reports[arm]["training"]["initialization"]["teacher_transition"]
            == training_spec["teacher_transition"]
            for arm in ARM_FILES
        ),
        "pure_oracle_loss_matches_manifest": all(
            reports[arm]["training"]["loss_weights"]
            == {
                "potential": training_spec["loss_weights"][
                    "centered_potential_variance"
                ],
                "metric": training_spec["loss_weights"][
                    "relative_metric_log_eigenvalue_mse"
                ],
                "log_energy": training_spec["loss_weights"][
                    "centered_log_energy"
                ],
                "ma": training_spec["loss_weights"]["monge_ampere_e2"],
                "tail": training_spec["loss_weights"]["upper_tail_cvar"],
            }
            for arm in ARM_FILES
        ),
        "train_validation_protocol_matches_manifest": all(
            int(reports[arm]["training"]["train"]["seed"])
            == int(training_spec["train_seed"])
            and int(reports[arm]["training"]["train"]["points"])
            == int(training_spec["train_points"])
            and int(reports[arm]["training"]["validation"]["seed"])
            == int(training_spec["validation_seed"])
            and int(reports[arm]["training"]["validation"]["points"])
            == int(training_spec["validation_points"])
            and int(reports[arm]["training"]["epochs"])
            == int(training_spec["epochs"])
            for arm in ARM_FILES
        ),
    }
    failed = [name for name, passed in common_checks.items() if not passed]
    if failed:
        raise ValueError("failed protocol checks: " + ", ".join(failed))

    teacher_metrics_by_report = [
        selected_metrics(report, teacher=True) for report in all_blind
    ]
    teacher_metrics = teacher_metrics_by_report[0]
    if any(
        not math.isclose(
            teacher_metrics[key],
            metrics[key],
            rel_tol=1.0e-13,
            abs_tol=1.0e-14,
        )
        for metrics in teacher_metrics_by_report[1:]
        for key in teacher_metrics
    ):
        raise ValueError("energy full-H metrics differ across common blind reports")

    thresholds = {
        "chi": 1.25 * teacher_metrics["chi"] + 0.02,
        "positive_log_ratio_q999": (
            1.50 * teacher_metrics["positive_log_ratio_q999"] + 0.10
        ),
        "positive_log_ratio_cvar_1pct": (
            1.50 * teacher_metrics["positive_log_ratio_cvar_1pct"] + 0.10
        ),
        "normalized_ratio_above_3_weighted_mass": (
            teacher_metrics["normalized_ratio_above_3_weighted_mass"] + 0.002
        ),
        "potential_rms_to_teacher": 0.01,
        "affine_metric_rms_to_teacher": 0.03,
    }

    arm_rows = {}
    for arm, filenames in ARM_FILES.items():
        initial = selected_metrics(reports[arm]["initial"])
        oracle = selected_metrics(reports[arm]["oracle"])
        ma_gates = {
            name: oracle[name] <= thresholds[name]
            for name in (
                "chi",
                "positive_log_ratio_q999",
                "positive_log_ratio_cvar_1pct",
                "normalized_ratio_above_3_weighted_mass",
            )
        }
        ma_gates["minimum_metric_eigenvalue"] = (
            oracle["minimum_metric_eigenvalue"] > 0.0
        )
        fidelity_gates = {
            name: oracle[name] <= thresholds[name]
            for name in (
                "potential_rms_to_teacher",
                "affine_metric_rms_to_teacher",
            )
        }
        ma_passed = bool(all(ma_gates.values()))
        fidelity_passed = bool(all(fidelity_gates.values()))
        arm_rows[arm] = {
            "architecture": manifest["arms"][arm],
            "initial_metrics": initial,
            "oracle_metrics": oracle,
            "teacher_free_to_oracle_changes": {
                key: oracle[key] - initial[key] for key in oracle
            },
            "ma_accuracy_retention_gates": ma_gates,
            "teacher_metric_fidelity_gates": fidelity_gates,
            "ma_accuracy_retention_passed": ma_passed,
            "teacher_metric_fidelity_passed": fidelity_passed,
            "strong_oracle_passed": ma_passed and fidelity_passed,
            "best_epoch": int(reports[arm]["training"]["best_epoch"]),
            "artifacts": {
                name: str(run_dir / filename)
                for name, filename in filenames.items()
            },
            "artifact_sha256": {
                name: sha256_file(run_dir / filename)
                for name, filename in filenames.items()
            },
        }

    report = {
        "schema": "positive-tensor-network-energy-oracle-audit-summary-v1",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "protocol_checks": common_checks,
        "energy_full_h_blind_metrics": teacher_metrics,
        "thresholds": thresholds,
        "arms": arm_rows,
        "constructive_capacity_found": {
            arm: row["strong_oracle_passed"] for arm, row in arm_rows.items()
        },
        "claim_boundary": manifest["claim_boundary"],
    }
    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_suffix(out_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(out_path)
    print(
        "strong_oracle="
        + ",".join(
            f"{arm}:{row['strong_oracle_passed']}" for arm, row in arm_rows.items()
        ),
        flush=True,
    )
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
