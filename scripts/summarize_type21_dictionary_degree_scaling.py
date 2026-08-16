#!/usr/bin/env python3
"""Certify the registered teacher-free q=12 scaling step from k=4 to k=6."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--k4-summary", type=Path, required=True)
    parser.add_argument("--k6-training", type=Path, required=True)
    parser.add_argument("--k6-blind", type=Path, required=True)
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


def selected_k4_candidate(summary: dict, rank: int) -> dict:
    candidates = [row for row in summary["candidates"] if int(row["rank"]) == rank]
    if len(candidates) != 1:
        raise ValueError(f"expected one k=4 candidate with rank {rank}")
    return candidates[0]


def metric_subset(report: dict) -> dict[str, float]:
    ma = report["metrics"]["compressed_ma_errors"]
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
        "minimum_metric_eigenvalue": float(
            report["metrics"]["minimum_metric_eigenvalue"]
        ),
    }


def main() -> None:
    args = parse_args()
    paths = {
        name: value.expanduser().resolve()
        for name, value in {
            "manifest": args.manifest,
            "k4_summary": args.k4_summary,
            "k6_training": args.k6_training,
            "k6_blind": args.k6_blind,
        }.items()
    }
    manifest = load_json(paths["manifest"])
    k4_summary = load_json(paths["k4_summary"])
    k6_training = load_json(paths["k6_training"])
    k6_blind = load_json(paths["k6_blind"])

    geometry = manifest["geometry"]
    architecture = manifest["architecture"]
    blind_spec = manifest["blind"]
    rank = int(geometry["physical_dictionary_rank"])
    k4_candidate = selected_k4_candidate(k4_summary, rank)
    k4_metrics = {key: float(value) for key, value in k4_candidate["metrics"].items()}
    k6_metrics = metric_subset(k6_blind)

    consistency_checks = {
        "registered_k4_rank_was_promoted": (
            int(k4_summary["promoted_rank"]) == rank and k4_candidate["eligible"]
        ),
        "adapter_matches_manifest": (
            k6_training["adapter"] == geometry["adapter"]
            and k6_blind["adapter"] == geometry["adapter"]
        ),
        "model_seed_matches_manifest": (
            int(k6_training["model_seed"]) == int(geometry["model_seed"])
            and int(k6_blind["model_seed"]) == int(geometry["model_seed"])
        ),
        "site_count_matches_manifest": (
            int(k6_training["site_count"]) == int(geometry["site_count"])
            and int(k6_blind["site_count"]) == int(geometry["site_count"])
        ),
        "dictionary_rank_matches_manifest": (
            int(k6_training["physical_dictionary_rank"]) == rank
            and int(k6_blind["physical_dictionary_rank"]) == rank
        ),
        "parameter_count_matches_manifest": (
            int(k6_training["trainable_real_parameter_count"])
            == int(architecture["total_trainable_real_parameter_count"])
            and int(k6_blind["trainable_real_parameter_count"])
            == int(architecture["total_trainable_real_parameter_count"])
        ),
        "training_is_teacher_free": (
            k6_training["training_mode"] == "teacher_free_geometric"
            and k6_training["teacher_artifact"] is None
            and k6_training["teacher_artifact_sha256"] is None
        ),
        "blind_is_teacher_free": (
            k6_blind["teacher_artifact"] is None
            and not bool(blind_spec["same_degree_teacher_loaded"])
        ),
        "blind_seed_and_count_match_manifest": (
            int(k6_blind["seed"]) == int(blind_spec["seed"])
            and int(k6_blind["points"]) == int(blind_spec["points"])
        ),
        "training_and_blind_model_hashes_match": (
            k6_training["model_sha256"] == k6_blind["model_sha256"]
        ),
    }
    failed_checks = [name for name, passed in consistency_checks.items() if not passed]
    if failed_checks:
        raise ValueError("failed consistency checks: " + ", ".join(failed_checks))

    thresholds = {
        "sigma": 1.20 * k4_metrics["sigma"] + 0.01,
        "chi": 1.25 * k4_metrics["chi"] + 0.02,
        "positive_log_ratio_q999": (
            1.25 * k4_metrics["positive_log_ratio_q999"] + 0.05
        ),
        "positive_log_ratio_cvar_1pct": (
            1.25 * k4_metrics["positive_log_ratio_cvar_1pct"] + 0.05
        ),
        "normalized_ratio_above_3_weighted_mass": (
            k4_metrics["normalized_ratio_above_3_weighted_mass"] + 0.002
        ),
    }
    gates = {
        name: k6_metrics[name] <= threshold
        for name, threshold in thresholds.items()
    }
    gates["minimum_metric_eigenvalue"] = (
        k6_metrics["minimum_metric_eigenvalue"] > 0.0
    )
    passed = bool(all(gates.values()))

    compared_metrics = tuple(thresholds)
    relative_changes = {
        name: (k6_metrics[name] / k4_metrics[name] - 1.0)
        if k4_metrics[name] != 0.0
        else None
        for name in compared_metrics
    }
    report = {
        "schema": "positive-tensor-network-teacher-free-degree-scaling-summary-v1",
        "manifest": str(paths["manifest"]),
        "manifest_sha256": sha256_file(paths["manifest"]),
        "consistency_checks": consistency_checks,
        "k4_reference": {
            "summary": str(paths["k4_summary"]),
            "summary_sha256": sha256_file(paths["k4_summary"]),
            "rank": rank,
            "trainable_real_parameter_count": int(
                k4_candidate["trainable_real_parameter_count"]
            ),
            "metrics": k4_metrics,
        },
        "k6_candidate": {
            "training": str(paths["k6_training"]),
            "training_sha256": sha256_file(paths["k6_training"]),
            "blind": str(paths["k6_blind"]),
            "blind_sha256": sha256_file(paths["k6_blind"]),
            "model_sha256": k6_blind["model_sha256"],
            "rank": rank,
            "trainable_real_parameter_count": int(
                k6_blind["trainable_real_parameter_count"]
            ),
            "metrics": k6_metrics,
        },
        "parameter_scaling": {
            "k4_trainable_real_parameter_count": int(
                k4_candidate["trainable_real_parameter_count"]
            ),
            "k6_trainable_real_parameter_count": int(
                k6_blind["trainable_real_parameter_count"]
            ),
            "k4_to_k6_increment": int(k6_blind["trainable_real_parameter_count"])
            - int(k4_candidate["trainable_real_parameter_count"]),
            "registered_k20_extrapolation": int(
                architecture["k20_extrapolated_parameter_count_at_fixed_D_q"]
            ),
        },
        "thresholds": thresholds,
        "gates": gates,
        "relative_changes_k6_over_k4": relative_changes,
        "teacher_free_scaling_step_passed": passed,
        "claim_boundary": manifest["claim_boundary"],
        "next_action": (
            "repeat the frozen architecture at a higher degree and on a second "
            "gCICY geometry; use same-degree solvers only as post-hoc benchmarks"
            if passed
            else "do not promote the k=6 architecture"
        ),
    }
    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_suffix(out_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(out_path)
    print(f"teacher_free_scaling_step_passed={passed}", flush=True)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
