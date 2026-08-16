#!/usr/bin/env python3
"""Summarize the frozen X21 D=12 final-blind opening."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--full-h", type=Path, required=True)
    parser.add_argument("--d8", type=Path, required=True)
    parser.add_argument("--d12", type=Path, required=True)
    parser.add_argument("--d8-vs-d12-bootstrap", type=Path, required=True)
    parser.add_argument("--full-h-vs-d12-bootstrap", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def sole_metrics(report: dict[str, Any]) -> dict[str, Any]:
    models = report["models"]
    if len(models) != 1:
        raise ValueError("bilateral report must contain exactly one model")
    return next(iter(models.values()))["metrics"]


def artifact(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    return {"path": str(resolved), "sha256": sha256(resolved)}


def main() -> None:
    args = parse_args()
    preregistration = load(args.preregistration)
    full_h = sole_metrics(load(args.full_h))
    d8 = sole_metrics(load(args.d8))
    d12 = sole_metrics(load(args.d12))
    d8_bootstrap = load(args.d8_vs_d12_bootstrap)
    h4_bootstrap = load(args.full_h_vs_d12_bootstrap)

    expected_points = int(preregistration["final_blind"]["point_count"])
    for label, metrics in (("full_h4", full_h), ("d8", d8), ("d12", d12)):
        if int(metrics["point_count"]) != expected_points:
            raise ValueError(f"{label} point count does not match preregistration")
        if int(metrics["nonpositive_metric_count"]) != 0:
            raise ValueError(f"{label} has nonpositive sampled metrics")

    sigma_comparison = d8_bootstrap["comparisons"]["sigma"]
    chi_comparison = d8_bootstrap["comparisons"]["chi"]
    output = {
        "schema": "type21-q121-d12-final-blind-summary-v1",
        "preregistration": artifact(args.preregistration),
        "selection": preregistration["selection"],
        "final_blind": preregistration["final_blind"],
        "metrics": {
            "full_h4": full_h,
            "d8_source": d8,
            "d12_selected": d12,
        },
        "paired_comparisons": {
            "d8_source_vs_d12_selected": d8_bootstrap,
            "full_h4_vs_d12_selected": h4_bootstrap,
        },
        "registered_primary_gate": {
            "definition": (
                "D12 must improve both sigma and chi over its D8 source with "
                "positive paired fibre-bootstrap 95% confidence-interval lower bounds"
            ),
            "sigma_pass": bool(
                float(
                    sigma_comparison["bootstrap_95pct_confidence_interval"][0]
                )
                > 0.0
            ),
            "chi_pass": bool(
                float(
                    chi_comparison["bootstrap_95pct_confidence_interval"][0]
                )
                > 0.0
            ),
        },
        "artifacts": {
            "full_h_bilateral": artifact(args.full_h),
            "d8_bilateral": artifact(args.d8),
            "d12_bilateral": artifact(args.d12),
            "d8_vs_d12_bootstrap": artifact(args.d8_vs_d12_bootstrap),
            "full_h_vs_d12_bootstrap": artifact(
                args.full_h_vs_d12_bootstrap
            ),
        },
    }
    output["registered_primary_gate"]["pass"] = bool(
        output["registered_primary_gate"]["sigma_pass"]
        and output["registered_primary_gate"]["chi_pass"]
    )
    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_suffix(out_path.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    temporary.replace(out_path)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
