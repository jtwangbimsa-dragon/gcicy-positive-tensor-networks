#!/usr/bin/env python3
"""Summarize the preregistered three-seed X11 TN-versus-phi comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--replicates", default="1,2,3")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate(rows: list[dict], key: str) -> dict:
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "sample_standard_deviation": float(np.std(values, ddof=1)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    replicates = [int(item) for item in args.replicates.split(",")]
    rows: list[dict] = []

    for replicate in replicates:
        replicate_dir = run_dir / f"replicate_{replicate}"
        tn_path = replicate_dir / "tn_final_blind.json"
        phi_path = replicate_dir / "phi_final_blind" / "report.json"
        bootstrap_path = replicate_dir / "tn_vs_phi_final_blind_bootstrap.json"
        tn = load_json(tn_path)
        phi = load_json(phi_path)
        bootstrap = load_json(bootstrap_path)

        tn_metric = tn["metrics"]["compressed_ma_errors"]
        phi_metric = phi["blind_test"]["trained_phi_self_normalized"]
        comparison = bootstrap["comparisons"]
        tn_max_abs_log = max(
            abs(math.log(float(tn_metric["normalized_ratio_min"]))),
            abs(math.log(float(tn_metric["normalized_ratio_max"]))),
        )

        row = {
            "replicate": replicate,
            "tn_sigma": float(comparison["sigma"]["candidate"]),
            "phi_sigma": float(comparison["sigma"]["baseline"]),
            "tn_chi": float(comparison["chi"]["candidate"]),
            "phi_chi": float(comparison["chi"]["baseline"]),
            "tn_absolute_log_ratio_q999": float(
                comparison["absolute_log_ratio_q999"]["candidate"]
            ),
            "phi_absolute_log_ratio_q999": float(
                comparison["absolute_log_ratio_q999"]["baseline"]
            ),
            "tn_absolute_log_ratio_cvar_1pct": float(
                comparison["absolute_log_ratio_cvar_1pct"]["candidate"]
            ),
            "phi_absolute_log_ratio_cvar_1pct": float(
                comparison["absolute_log_ratio_cvar_1pct"]["baseline"]
            ),
            "tn_maximum_absolute_log_ratio": tn_max_abs_log,
            "phi_maximum_absolute_log_ratio": float(
                phi_metric["abs_log_ratio_weighted_quantiles"]["q1.0000"]
            ),
            "tn_minimum_metric_eigenvalue": float(
                tn["metrics"]["minimum_metric_eigenvalue"]
            ),
            "phi_minimum_metric_eigenvalue": float(
                phi_metric["min_eigenvalue_weighted_quantiles"]["q0.0000"]
            ),
            "tn_nonpositive_metric_count": int(
                0 if tn["metrics"]["minimum_metric_eigenvalue"] > 0.0 else -1
            ),
            "phi_nonpositive_metric_count": int(
                phi_metric["nonpositive_or_nonfinite"]["point_count"]
            ),
            "tn_relative_sigma_improvement_over_phi": float(
                comparison["sigma"]["relative_improvement"]
            ),
            "tn_relative_chi_improvement_over_phi": float(
                comparison["chi"]["relative_improvement"]
            ),
            "paired_sigma_improvement_95pct_ci": comparison["sigma"][
                "bootstrap_95pct_confidence_interval"
            ],
            "paired_chi_improvement_95pct_ci": comparison["chi"][
                "bootstrap_95pct_confidence_interval"
            ],
            "tn_artifact": str(tn_path),
            "tn_artifact_sha256": sha256(tn_path),
            "phi_artifact": str(phi_path),
            "phi_artifact_sha256": sha256(phi_path),
            "paired_bootstrap_artifact": str(bootstrap_path),
            "paired_bootstrap_artifact_sha256": sha256(bootstrap_path),
        }
        rows.append(row)

    scalar_keys = [
        "tn_sigma",
        "phi_sigma",
        "tn_chi",
        "phi_chi",
        "tn_absolute_log_ratio_q999",
        "phi_absolute_log_ratio_q999",
        "tn_absolute_log_ratio_cvar_1pct",
        "phi_absolute_log_ratio_cvar_1pct",
        "tn_maximum_absolute_log_ratio",
        "phi_maximum_absolute_log_ratio",
        "tn_minimum_metric_eigenvalue",
        "phi_minimum_metric_eigenvalue",
        "tn_relative_sigma_improvement_over_phi",
        "tn_relative_chi_improvement_over_phi",
    ]
    output = {
        "schema": "type11-matched-tn-phi-three-seed-final-blind-summary-v1",
        "run_dir": str(run_dir),
        "replicate_count": len(rows),
        "final_blind_point_count": 200000,
        "rows": rows,
        "aggregate": {key: aggregate(rows, key) for key in scalar_keys},
        "all_sigma_improvement_intervals_strictly_positive": all(
            row["paired_sigma_improvement_95pct_ci"][0] > 0.0 for row in rows
        ),
        "all_chi_improvement_intervals_strictly_positive": all(
            row["paired_chi_improvement_95pct_ci"][0] > 0.0 for row in rows
        ),
        "all_sampled_metrics_positive": all(
            row["tn_minimum_metric_eigenvalue"] > 0.0
            and row["phi_minimum_metric_eigenvalue"] > 0.0
            for row in rows
        ),
    }
    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
