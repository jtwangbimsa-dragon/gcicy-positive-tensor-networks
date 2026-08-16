#!/usr/bin/env python3
"""Summarize one cold-start quintic TN degree/capacity run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


STAGES = (
    "01_q21_b64_lr3e4_e20",
    "03_q25_b64_lr1e4_e10",
    "04_q25_b64_lr3e5_e20",
    "05_q25_b1024_lr3e5_e30",
    "06_q25_b1024_lr1e5_e50",
    "07_q25_n900k_b1024_lr1e5_e5",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--bond-dimension", type=int, required=True)
    parser.add_argument("--site-count", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def blind_metrics(report: dict[str, Any]) -> dict[str, Any]:
    normalized = report["blind_test"]["normalized_volume"]
    return {
        "sigma": float(normalized["sigma_official_formula"]),
        "chi": float(normalized["weighted_rms_abs_residual"]),
        "q999_abs_residual": float(
            normalized["abs_residual_weighted_quantiles"]["q0.9990"]
        ),
        "cvar99_abs_residual": float(
            normalized["abs_residual_weighted_cvar"]["cvar_0.9900"]
        ),
        "maximum_abs_residual": float(
            normalized["abs_residual_weighted_quantiles"]["q1.0000"]
        ),
        "nonpositive_count": int(
            normalized["nonpositive_min_eigenvalue"]["count"]
        ),
    }


def main() -> None:
    args = parse_args()
    if args.bond_dimension < 1:
        raise ValueError("bond dimension must be positive")
    if args.site_count < 2:
        raise ValueError("site count must be at least two")
    run_root = args.run_root.expanduser().resolve()
    rows = []
    for stage in STAGES:
        report_path = run_root / stage / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        configuration = report["configuration"]
        if int(configuration["bond_dimension"]) != args.bond_dimension:
            raise ValueError(f"unexpected bond dimension in {report_path}")
        if int(configuration["site_count"]) != args.site_count:
            raise ValueError(f"unexpected site count in {report_path}")
        source_degree = int(configuration["source_degree"])
        train_points = int(report["common_point_evidence"]["train_points"])
        epochs = int(configuration["epochs"])
        batch_size = int(configuration["batch_size"])
        rows.append(
            {
                "stage": stage,
                "report": str(report_path),
                "report_sha256": sha256_file(report_path),
                "train_points": train_points,
                "q": int(configuration["dictionary_rank"]),
                "D": int(configuration["bond_dimension"]),
                "source_degree": source_degree,
                "site_count": int(configuration["site_count"]),
                "total_degree": source_degree * int(configuration["site_count"]),
                "parameters": int(
                    report["architecture"]["trainable_real_parameter_count"]
                ),
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": float(configuration["learning_rate"]),
                "optimizer_updates": math.ceil(train_points / batch_size) * epochs,
                "point_exposures": train_points * epochs,
                "best_epoch": int(report["training"]["best_epoch"]),
                "training_seconds": float(report["timing_seconds"]["training"]),
                "wall_seconds": float(report["timing_seconds"]["wall_total"]),
                "blind": blind_metrics(report),
            }
        )

    base_rows = rows[:-1]
    summary = {
        "schema": "quintic-tn-cold-capacity-summary-v1",
        "run_root": str(run_root),
        "bond_dimension": args.bond_dimension,
        "site_count": args.site_count,
        "rows": rows,
        "matched_90k_curriculum": {
            "optimizer_updates": sum(row["optimizer_updates"] for row in base_rows),
            "point_exposures": sum(row["point_exposures"] for row in base_rows),
            "training_seconds": sum(row["training_seconds"] for row in base_rows),
            "final": base_rows[-1]["blind"],
        },
        "additional_900k_refinement": {
            "optimizer_updates": rows[-1]["optimizer_updates"],
            "point_exposures": rows[-1]["point_exposures"],
            "training_seconds": rows[-1]["training_seconds"],
            "final": rows[-1]["blind"],
        },
    }
    output_path = args.out.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output_path)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
