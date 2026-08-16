#!/usr/bin/env python3
"""Summarize a staged quintic TN curriculum without conflating timing scopes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


STAGES = (
    "01_q54_d7_b64_lr3e4_e20",
    "03_q64_d7_b64_lr1e4_e10",
    "04_q64_d7_b64_lr3e5_e10",
    "06_q64_d8_b64_lr3e5_e10",
    "07_q64_d8_b1024_lr3e5_e30",
    "08_q64_d8_b1024_lr1e5_e50",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
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
        "minimum_ratio": float(
            normalized["ratio_weighted_quantiles"]["q0.0000"]
        ),
        "maximum_ratio": float(
            normalized["ratio_weighted_quantiles"]["q1.0000"]
        ),
        "nonpositive_count": int(
            normalized["nonpositive_min_eigenvalue"]["count"]
        ),
    }


def main() -> None:
    args = parse_args()
    run_root = args.run_root.expanduser().resolve()
    rows = []
    for stage in STAGES:
        report_path = run_root / stage / "report.json"
        if not report_path.exists():
            raise FileNotFoundError(report_path)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        configuration = report["configuration"]
        train_points = int(report["common_point_evidence"]["train_points"])
        optimizer_updates = math.ceil(
            train_points / int(configuration["batch_size"])
        ) * int(configuration["epochs"])
        row = {
            "stage": stage,
            "report": str(report_path),
            "report_sha256": sha256_file(report_path),
            "q": int(configuration["dictionary_rank"]),
            "D": int(configuration["bond_dimension"]),
            "parameters": int(
                report["architecture"]["trainable_real_parameter_count"]
            ),
            "epochs": int(configuration["epochs"]),
            "batch_size": int(configuration["batch_size"]),
            "learning_rate": float(configuration["learning_rate"]),
            "optimizer_updates": optimizer_updates,
            "point_exposures": train_points * int(configuration["epochs"]),
            "best_epoch": int(report["training"]["best_epoch"]),
            "training_seconds": float(report["timing_seconds"]["training"]),
            "wall_seconds": float(report["timing_seconds"]["wall_total"]),
            "blind": blind_metrics(report),
        }
        rows.append(row)

    final = rows[-1]
    summary = {
        "schema": "quintic-tn-matched-curriculum-summary-v1",
        "run_root": str(run_root),
        "timing_scope": (
            "fresh staged training execution; excludes prior method-development "
            "and hyperparameter-search compute"
        ),
        "rows": rows,
        "totals": {
            "optimizer_updates": sum(row["optimizer_updates"] for row in rows),
            "point_exposures": sum(row["point_exposures"] for row in rows),
            "training_seconds": sum(row["training_seconds"] for row in rows),
            "stage_wall_seconds": sum(row["wall_seconds"] for row in rows),
        },
        "final": {
            "stage": final["stage"],
            "parameters": final["parameters"],
            **final["blind"],
        },
    }
    output_path = args.out.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
