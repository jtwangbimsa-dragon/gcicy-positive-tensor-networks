#!/usr/bin/env python3
"""Summarize the exact-point, exact-parameter Fermat quintic comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor-network-report", type=Path, required=True)
    parser.add_argument("--matched-cymetric-report", type=Path, required=True)
    parser.add_argument("--official-cymetric-report", type=Path)
    parser.add_argument("--full-h-report", type=Path)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-markdown", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def metric_row(name: str, parameters: int, stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "trainable_real_parameters": int(parameters),
        "blind_points": int(stats["n_points"]),
        "sigma": float(stats["sigma_official_formula"]),
        "chi": float(stats["weighted_rms_abs_residual"]),
        "residual_q999": float(stats["abs_residual_weighted_quantiles"]["q0.9990"]),
        "residual_cvar99": float(stats["abs_residual_weighted_cvar"]["cvar_0.9900"]),
        "ratio_min": float(stats["ratio_weighted_quantiles"]["q0.0000"]),
        "ratio_max": float(stats["ratio_weighted_quantiles"]["q1.0000"]),
        "nonpositive_metric_count": int(stats["nonpositive_min_eigenvalue"]["count"]),
        "minimum_metric_eigenvalue": float(
            stats["min_eigenvalue_weighted_quantiles"]["q0.0000"]
        ),
    }


def improvement(smaller: float, baseline: float) -> float:
    return 1.0 - smaller / baseline


def main() -> None:
    args = parse_args()
    tn = load(args.tensor_network_report)
    matched = load(args.matched_cymetric_report)
    tn_parameters = int(tn["architecture"]["trainable_real_parameter_count"])
    matched_parameters = int(matched["network"]["parameter_count"])
    if tn_parameters != matched_parameters:
        raise RuntimeError(
            f"parameter mismatch: TN={tn_parameters}, cymetric={matched_parameters}"
        )
    tn_sources = tn["common_point_evidence"]["source_sha256"]
    if tn_sources["dataset"] != matched["data"]["dataset_sha256"]:
        raise RuntimeError("training-data hashes differ")
    if tn_sources["blind_points"] != matched["data"]["blind_points_sha256"]:
        raise RuntimeError("blind-point hashes differ")

    rows = [
        metric_row(
            "positive TN (q=21,D=5,k=9)",
            tn_parameters,
            tn["blind_test"]["normalized_volume"],
        ),
        metric_row(
            "cymetric PhiFS (3x63 GELU)",
            matched_parameters,
            matched["trained_phi_model"],
        ),
    ]
    if any(row["blind_points"] != 200_000 for row in rows):
        raise RuntimeError("primary comparison must use all 200,000 blind points")

    if args.official_cymetric_report is not None:
        official = load(args.official_cymetric_report)
        rows.append(
            metric_row(
                "cymetric PhiFS (3x64 GELU)",
                int(official["network"]["parameter_count"]),
                official["trained_phi_model"],
            )
        )
    if args.full_h_report is not None:
        full_h = load(args.full_h_report)
        rows.append(
            metric_row(
                "full H (degree 4, raw)",
                int(full_h["basis"]["real_hermitian_parameter_count"]),
                full_h["blind"]["our_full_h"],
            )
        )

    tn_row, matched_row = rows[:2]
    primary_improvements = {
        key: improvement(tn_row[key], matched_row[key])
        for key in ("sigma", "chi", "residual_q999", "residual_cvar99")
    }
    payload = {
        "schema": "quintic-parameter-matched-comparison-v1",
        "geometry": "Fermat quintic hypersurface X_5 in P^4",
        "common_point_gate": {
            "parameter_count": tn_parameters,
            "dataset_sha256": tn_sources["dataset"],
            "blind_points_sha256": tn_sources["blind_points"],
            "blind_points": 200_000,
        },
        "rows": rows,
        "tn_fractional_improvement_over_matched_cymetric": primary_improvements,
        "interpretation_limit": (
            "This compares complete algorithms with different loss functions. "
            "A common-loss ablation is needed to isolate ansatz effects."
        ),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    headers = (
        "model",
        "params",
        "sigma",
        "chi",
        "q99.9 residual",
        "CVaR99",
        "r_min",
        "r_max",
        "min eig",
    )
    lines = [
        "# Fermat quintic exact-parameter comparison",
        "",
        "| " + " | ".join(headers) + " |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {name} | {trainable_real_parameters} | {sigma:.6g} | {chi:.6g} | "
            "{residual_q999:.6g} | {residual_cvar99:.6g} | {ratio_min:.6g} | "
            "{ratio_max:.6g} | {minimum_metric_eigenvalue:.6g} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Primary TN improvement over the exactly matched cymetric arm:",
            "",
            *[
                f"- `{key}`: {100.0 * value:.2f}%"
                for key, value in primary_improvements.items()
            ],
            "",
            payload["interpretation_limit"],
        ]
    )
    args.out_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.out_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
