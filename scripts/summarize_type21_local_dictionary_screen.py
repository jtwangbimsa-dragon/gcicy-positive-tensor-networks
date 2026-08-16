#!/usr/bin/env python3
"""Apply the registered promotion gates to the type-(2,1) dictionary screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("fixed", "trainable"),
        default="fixed",
    )
    parser.add_argument("--ranks", default="4,8,12")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    ranks = tuple(int(value) for value in args.ranks.split(",") if value.strip())
    if not ranks or len(set(ranks)) != len(ranks) or min(ranks) <= 0:
        raise SystemExit("ranks must be distinct positive integers")
    run_dir = args.run_dir.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    manifest = load_json(manifest_path)
    registered = tuple(manifest["architecture"]["candidate_ranks"])
    if ranks != registered:
        raise ValueError(f"requested ranks {ranks} differ from registered ranks {registered}")

    blind_seed = 83413 if args.mode == "fixed" else 83423
    dense_path = run_dir / f"dense_teacher_free_blind{blind_seed}_n32766.json"
    dense_report = load_json(dense_path)
    dense = dense_report["metrics"]["compressed_ma_errors"]
    thresholds = {
        "sigma": 1.20 * float(dense["sigma"]) + 0.01,
        "chi": 1.25 * float(dense["sqrt_squared_energy"]) + 0.02,
        "positive_log_ratio_q999": (
            1.25 * float(dense["positive_log_ratio_q999"]) + 0.05
        ),
        "positive_log_ratio_cvar_1pct": (
            1.25 * float(dense["positive_log_ratio_cvar_1pct"]) + 0.05
        ),
        "normalized_ratio_above_3_weighted_mass": (
            float(dense["normalized_ratio_above_3_weighted_mass"]) + 0.002
        ),
    }

    rows = []
    for rank in ranks:
        if args.mode == "fixed":
            stem = f"model_d5_q{rank}_teacher_free"
        else:
            stem = f"model_d5_q{rank}_trainable_dictionary"
        blind_path = run_dir / f"{stem}_blind{blind_seed}_n32766.json"
        training_path = run_dir / f"{stem}_summary.json"
        blind_report = load_json(blind_path)
        training_report = load_json(training_path)
        metrics = blind_report["metrics"]["compressed_ma_errors"]
        gates = {
            "sigma": float(metrics["sigma"]) <= thresholds["sigma"],
            "chi": (
                float(metrics["sqrt_squared_energy"]) <= thresholds["chi"]
            ),
            "positive_log_ratio_q999": (
                float(metrics["positive_log_ratio_q999"])
                <= thresholds["positive_log_ratio_q999"]
            ),
            "positive_log_ratio_cvar_1pct": (
                float(metrics["positive_log_ratio_cvar_1pct"])
                <= thresholds["positive_log_ratio_cvar_1pct"]
            ),
            "normalized_ratio_above_3_weighted_mass": (
                float(metrics["normalized_ratio_above_3_weighted_mass"])
                <= thresholds["normalized_ratio_above_3_weighted_mass"]
            ),
            "minimum_metric_eigenvalue": (
                float(blind_report["metrics"]["minimum_metric_eigenvalue"]) > 0
            ),
        }
        row = {
                "rank": rank,
                "eligible": bool(all(gates.values())),
                "gates": gates,
                "trainable_real_parameter_count": int(
                    blind_report["trainable_real_parameter_count"]
                ),
                "best_epoch": int(training_report["best_epoch"]),
                "metrics": {
                    "sigma": float(metrics["sigma"]),
                    "chi": float(metrics["sqrt_squared_energy"]),
                    "positive_log_ratio_q999": float(
                        metrics["positive_log_ratio_q999"]
                    ),
                    "positive_log_ratio_cvar_1pct": float(
                        metrics["positive_log_ratio_cvar_1pct"]
                    ),
                    "normalized_ratio_min": float(metrics["normalized_ratio_min"]),
                    "normalized_ratio_max": float(metrics["normalized_ratio_max"]),
                    "normalized_ratio_above_3_weighted_mass": float(
                        metrics["normalized_ratio_above_3_weighted_mass"]
                    ),
                    "minimum_metric_eigenvalue": float(
                        blind_report["metrics"]["minimum_metric_eigenvalue"]
                    ),
                },
                "artifacts": {
                    "blind": str(blind_path),
                    "blind_sha256": sha256_file(blind_path),
                    "training": str(training_path),
                    "training_sha256": sha256_file(training_path),
                },
            }
        if args.mode == "fixed":
            compression_path = run_dir / f"model_d5_q{rank}_svd_summary.json"
            compression_report = load_json(compression_path)
            row["retained_squared_frobenius_energy_fraction"] = float(
                compression_report["retained_squared_frobenius_energy_fraction"]
            )
            row["artifacts"].update(
                {
                    "compression": str(compression_path),
                    "compression_sha256": sha256_file(compression_path),
                }
            )
        rows.append(row)

    eligible = [row["rank"] for row in rows if row["eligible"]]
    promoted_rank = min(eligible) if eligible else None
    report = {
        "schema": "positive-tensor-network-local-dictionary-screen-summary-v1",
        "mode": args.mode,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "dense_reference": {
            "artifact": str(dense_path),
            "artifact_sha256": sha256_file(dense_path),
            "trainable_real_parameter_count": int(
                dense_report["trainable_real_parameter_count"]
            ),
            "metrics": dense,
        },
        "thresholds": thresholds,
        "candidates": rows,
        "promoted_rank": promoted_rank,
        "dictionary_screen_passed": promoted_rank is not None,
        "next_action": (
            "run the registered Ritz consistency audit and prepare k=6"
            if promoted_rank is not None
            else (
                "test a jointly trainable shared dictionary; promote no fixed rank"
                if args.mode == "fixed"
                else "promote no dictionary model"
            )
        ),
    }
    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_suffix(out_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(out_path)
    print(f"promoted_rank={promoted_rank}", flush=True)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
