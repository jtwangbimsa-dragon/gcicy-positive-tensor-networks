#!/usr/bin/env python3
"""Build a publication-facing summary of one paired TN capacity control."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-training", type=Path, required=True)
    parser.add_argument("--candidate-training", type=Path, required=True)
    parser.add_argument("--baseline-confirmation", type=Path, required=True)
    parser.add_argument("--candidate-confirmation", type=Path, required=True)
    parser.add_argument("--baseline-bilateral", type=Path, required=True)
    parser.add_argument("--candidate-bilateral", type=Path, required=True)
    parser.add_argument("--bootstrap", type=Path, required=True)
    parser.add_argument("--equivalence", type=Path, required=True)
    parser.add_argument("--parameter-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    return {"path": str(resolved), "sha256": sha256(resolved)}


def main() -> None:
    args = parse_args()
    baseline_training = load(args.baseline_training)
    candidate_training = load(args.candidate_training)
    baseline_confirmation = load(args.baseline_confirmation)
    candidate_confirmation = load(args.candidate_confirmation)
    baseline_bilateral = next(
        iter(load(args.baseline_bilateral)["models"].values())
    )["metrics"]
    candidate_bilateral = next(
        iter(load(args.candidate_bilateral)["models"].values())
    )["metrics"]
    bootstrap = load(args.bootstrap)
    equivalence = load(args.equivalence)
    parameter_audit = load(args.parameter_audit)
    baseline_metrics = baseline_confirmation["metrics"]["compressed_ma_errors"]
    candidate_metrics = candidate_confirmation["metrics"]["compressed_ma_errors"]

    paired = bootstrap["comparisons"]
    gates = {
        "epoch0_function_and_metric_preserved": bool(equivalence["success"]),
        "all_expected_trainable_tensors_changed": bool(
            parameter_audit["all_expected_trainable_tensors_changed"]
        ),
        "sigma_improved": candidate_bilateral["sigma"] < baseline_bilateral["sigma"],
        "chi_improved": candidate_bilateral["chi"] < baseline_bilateral["chi"],
        "sigma_bootstrap_ci_positive": paired["sigma"][
            "bootstrap_95pct_confidence_interval"
        ][0]
        > 0,
        "chi_bootstrap_ci_positive": paired["chi"][
            "bootstrap_95pct_confidence_interval"
        ][0]
        > 0,
        "upper_cvar_bootstrap_ci_positive": paired[
            "upper_log_ratio_cvar_1pct"
        ]["bootstrap_95pct_confidence_interval"][0]
        > 0,
        "lower_cvar_bootstrap_ci_positive": paired[
            "lower_log_ratio_cvar_1pct"
        ]["bootstrap_95pct_confidence_interval"][0]
        > 0,
        "candidate_metric_positive": candidate_confirmation["metrics"][
            "minimum_metric_eigenvalue"
        ]
        > 0,
    }
    report = {
        "schema": "positive-tn-paired-capacity-control-summary-v1",
        "comparison_class": "same-point paired capacity mechanism",
        "claim_limit": (
            "This compares q121 D5 and D8 continuations from one TN checkpoint; "
            "it is not a dense full-H to TN comparison."
        ),
        "baseline": {
            "bond_dimension": baseline_training["bond_dimension"],
            "trainable_real_parameter_count": baseline_training[
                "trainable_real_parameter_count"
            ],
            "best_epoch": baseline_training["best_epoch"],
            "runtime_seconds": baseline_training["runtime_seconds"],
            "metrics": baseline_bilateral,
            "minimum_metric_eigenvalue": baseline_confirmation["metrics"][
                "minimum_metric_eigenvalue"
            ],
            "normalized_ratio_min": baseline_metrics["normalized_ratio_min"],
            "normalized_ratio_max": baseline_metrics["normalized_ratio_max"],
        },
        "candidate": {
            "bond_dimension": candidate_training["bond_dimension"],
            "trainable_real_parameter_count": candidate_training[
                "trainable_real_parameter_count"
            ],
            "best_epoch": candidate_training["best_epoch"],
            "runtime_seconds": candidate_training["runtime_seconds"],
            "metrics": candidate_bilateral,
            "minimum_metric_eigenvalue": candidate_confirmation["metrics"][
                "minimum_metric_eigenvalue"
            ],
            "normalized_ratio_min": candidate_metrics["normalized_ratio_min"],
            "normalized_ratio_max": candidate_metrics["normalized_ratio_max"],
        },
        "paired_cluster_bootstrap": paired,
        "gates": gates,
        "capacity_control_passed": bool(all(gates.values())),
        "artifacts": {
            name: artifact(getattr(args, name))
            for name in (
                "baseline_training",
                "candidate_training",
                "baseline_confirmation",
                "candidate_confirmation",
                "baseline_bilateral",
                "candidate_bilateral",
                "bootstrap",
                "equivalence",
                "parameter_audit",
            )
        },
    }
    output = args.out.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(f"capacity_control_passed={report['capacity_control_passed']}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
