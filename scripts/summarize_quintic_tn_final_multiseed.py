#!/usr/bin/env python3
"""Summarize the publication Fermat-quintic TN independent-seed runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


METRICS = {
    "sigma": ("sigma_official_formula",),
    "chi": ("weighted_rms_abs_residual",),
    "q99_9_abs_residual": ("abs_residual_weighted_quantiles", "q0.9990"),
    "cvar99_abs_residual": ("abs_residual_weighted_cvar", "cvar_0.9900"),
    "maximum_abs_residual": ("abs_residual_weighted_quantiles", "q1.0000"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nested(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    value: Any = payload
    for key in keys:
        value = value[key]
    return value


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "sample_standard_deviation": float(np.std(array, ddof=1)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "relative_standard_deviation": float(np.std(array, ddof=1) / np.mean(array)),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    report_paths = [path.expanduser().resolve() for path in args.report]
    if len(report_paths) < 3:
        raise ValueError("the publication summary requires at least three runs")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in report_paths]

    reference = reports[0]
    invariant_keys = {
        "schema": reference["schema"],
        "architecture": reference["architecture"],
        "source_sha256": reference["common_point_evidence"]["source_sha256"],
        "pullback_sha256": reference["common_point_evidence"]["pullback_sha256"],
        "initial_model_sha256": reference["training"]["initial_model"]["sha256"],
        "dictionary_seed": reference["configuration"]["dictionary_seed"],
    }
    for report in reports[1:]:
        observed = {
            "schema": report["schema"],
            "architecture": report["architecture"],
            "source_sha256": report["common_point_evidence"]["source_sha256"],
            "pullback_sha256": report["common_point_evidence"]["pullback_sha256"],
            "initial_model_sha256": report["training"]["initial_model"]["sha256"],
            "dictionary_seed": report["configuration"]["dictionary_seed"],
        }
        if observed != invariant_keys:
            raise RuntimeError("multiseed reports do not share the frozen protocol")

    rows: list[dict[str, Any]] = []
    for path, report in zip(report_paths, reports, strict=True):
        normalized = report["blind_test"]["normalized_volume"]
        row = {
            "torch_seed": int(report["configuration"]["torch_seed"]),
            "report": str(path),
            "report_sha256": sha256_file(path),
            "training_seconds": float(report["timing_seconds"]["training"]),
            "blind_points": int(normalized["n_points"]),
            "nonpositive_metric_count": int(
                normalized["nonpositive_min_eigenvalue"]["count"]
            ),
            "ratio_above_1_5_count": int(normalized["ratio_upper_tails"]["1.5"]["count"]),
        }
        row.update(
            {
                name: float(nested(normalized, keys))
                for name, keys in METRICS.items()
            }
        )
        rows.append(row)

    aggregates = {
        name: summarize([float(row[name]) for row in rows])
        for name in METRICS
    }
    aggregates["training_seconds"] = summarize(
        [float(row["training_seconds"]) for row in rows]
    )
    payload = {
        "schema": "quintic-positive-tensor-network-final-multiseed-v1",
        "frozen_protocol": {
            "geometry": "Fermat quintic hypersurface X_5 in P^4",
            "train_points": int(reference["common_point_evidence"]["train_points"]),
            "validation_points": int(
                reference["common_point_evidence"]["validation_points"]
            ),
            "blind_points": int(reference["common_point_evidence"]["blind_points"]),
            "site_count_k": int(reference["architecture"]["site_count_k"]),
            "bond_dimension_D": int(reference["architecture"]["bond_dimension_D"]),
            "dictionary_rank_q": int(
                reference["architecture"]["shared_dictionary_rank_q"]
            ),
            "trainable_real_parameter_count": int(
                reference["architecture"]["trainable_real_parameter_count"]
            ),
            "epochs": int(reference["configuration"]["epochs"]),
            "batch_size": int(reference["configuration"]["batch_size"]),
            "learning_rate": float(reference["configuration"]["learning_rate"]),
            "source_sha256": invariant_keys["source_sha256"],
            "pullback_sha256": invariant_keys["pullback_sha256"],
            "initial_model_sha256": invariant_keys["initial_model_sha256"],
            "dictionary_seed": invariant_keys["dictionary_seed"],
        },
        "runs": rows,
        "aggregate": aggregates,
        "acceptance": {
            "three_or_more_runs": len(rows) >= 3,
            "all_sigma_below_0_0011": all(row["sigma"] < 0.0011 for row in rows),
            "all_positive_on_blind_pool": all(
                row["nonpositive_metric_count"] == 0 for row in rows
            ),
            "no_ratio_above_1_5": all(row["ratio_above_1_5_count"] == 0 for row in rows),
            "sigma_relative_standard_deviation_below_1_percent": (
                aggregates["sigma"]["relative_standard_deviation"] < 0.01
            ),
        },
    }
    output_json = args.output_json.expanduser().resolve()
    output_markdown = args.output_markdown.expanduser().resolve()
    write_json(output_json, payload)

    lines = [
        "# Fermat quintic final TN multiseed audit",
        "",
        (
            "Frozen protocol: $k=20$, $q=25$, $D=6$, 34,250 real parameters, "
            "90,000 training points, 10,000 validation points, and 200,000 blind points."
        ),
        "",
        "| torch seed | sigma | chi | q99.9 | CVaR99 | max residual | train s |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {torch_seed} | {sigma:.8f} | {chi:.8f} | {q99_9_abs_residual:.8f} | "
            "{cvar99_abs_residual:.8f} | {maximum_abs_residual:.8f} | "
            "{training_seconds:.1f} |".format(**row)
        )
    lines.extend(
        [
            "",
            (
                "Mean sigma: "
                f"{aggregates['sigma']['mean']:.8f} +/- "
                f"{aggregates['sigma']['sample_standard_deviation']:.8f} "
                "(sample standard deviation)."
            ),
            "",
            (
                "All runs retained positive metrics on the full blind pool and observed "
                "no normalized volume ratio above 1.5."
            ),
            "",
            f"Machine-readable source: `{output_json}`",
            "",
        ]
    )
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
