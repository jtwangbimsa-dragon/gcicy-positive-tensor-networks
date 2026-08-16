#!/usr/bin/env python3
"""Summarize frozen-pool gCICY TN confirmations across optimizer seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


METRICS = (
    "sigma",
    "chi",
    "absolute_log_ratio_q999",
    "absolute_log_ratio_cvar_1pct",
    "upper_log_ratio_q999",
    "upper_log_ratio_cvar_1pct",
    "lower_log_ratio_q999",
    "lower_log_ratio_cvar_1pct",
    "normalized_ratio_min",
    "normalized_ratio_max",
    "minimum_metric_eigenvalue",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--minimum-runs", type=int, default=3)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    sample_standard_deviation = (
        float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    )
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "sample_standard_deviation": sample_standard_deviation,
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "range": float(np.max(array) - np.min(array)),
        "relative_sample_standard_deviation": (
            sample_standard_deviation / float(np.mean(array))
            if float(np.mean(array)) != 0.0
            else 0.0
        ),
    }


def one_model_metrics(bilateral: dict[str, Any]) -> dict[str, Any]:
    models = bilateral["models"]
    if len(models) != 1:
        raise ValueError("each bilateral report must contain exactly one model")
    return next(iter(models.values()))["metrics"]


def main() -> None:
    args = parse_args()
    if args.minimum_runs < 1:
        raise SystemExit("--minimum-runs must be positive")
    run_dir = args.run_dir.expanduser().resolve()
    seed_manifest_path = run_dir / "seed_protocol_manifest.json"
    seed_manifest = (
        read_json(seed_manifest_path) if seed_manifest_path.is_file() else None
    )
    torch_seeds_by_activation = (
        {
            int(row["activation_seed"]): int(row["torch_seed"])
            for row in seed_manifest["runs"]
        }
        if seed_manifest is not None
        else {}
    )
    seed_dirs = sorted(
        path
        for path in run_dir.glob("seed_*")
        if (path / "SEED_FINISHED").is_file()
    )
    if len(seed_dirs) < args.minimum_runs:
        raise RuntimeError(
            f"found {len(seed_dirs)} completed runs; need {args.minimum_runs}"
        )

    rows: list[dict[str, Any]] = []
    invariant_protocol: dict[str, Any] | None = None
    for seed_dir in seed_dirs:
        training_path = seed_dir / "training_summary.json"
        confirmation_path = seed_dir / "confirmation.json"
        bilateral_path = seed_dir / "bilateral_confirmation.json"
        parameter_audit_path = seed_dir / "parameter_change_audit.json"
        required = (
            training_path,
            confirmation_path,
            bilateral_path,
            parameter_audit_path,
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"incomplete seed directory: {missing}")

        training = read_json(training_path)
        confirmation = read_json(confirmation_path)
        parameter_audit = read_json(parameter_audit_path)
        metrics = one_model_metrics(read_json(bilateral_path))
        protocol = {
            "adapter": training["adapter"],
            "training_mode": training["training_mode"],
            "train_points": training["train"]["points"],
            "train_seed": training["train"]["seed"],
            "train_pool_sha256": training["train"]["common_pool_sha256"],
            "selection_points": training["validation"]["points"],
            "selection_seed": training["validation"]["seed"],
            "selection_pool_sha256": training["validation"]["common_pool_sha256"],
            "confirmation_points": confirmation["points"],
            "confirmation_seed": confirmation["seed"],
            "confirmation_pool_sha256": confirmation["common_pool_sha256"],
            "site_count": training["site_count"],
            "bond_dimension": training["bond_dimension"],
            "physical_dictionary_rank": training["physical_dictionary_rank"],
            "trainable_real_parameter_count": training[
                "trainable_real_parameter_count"
            ],
            "epochs": training["epochs"],
            "batch_size": training["batch_size"],
            "learning_rate": training["learning_rate"],
            "precision": training["precision"],
        }
        if invariant_protocol is None:
            invariant_protocol = protocol
        elif protocol != invariant_protocol:
            raise RuntimeError(f"protocol mismatch in {seed_dir}")

        activation_seed = int(seed_dir.name.removeprefix("seed_"))
        row = {
            "activation_seed": activation_seed,
            "torch_seed": (
                int(training["torch_seed"])
                if training.get("torch_seed") is not None
                else torch_seeds_by_activation.get(activation_seed)
            ),
            "best_epoch": int(training["best_epoch"]),
            "training_runtime_seconds": float(training["runtime_seconds"]),
            "model_sha256": confirmation["model_sha256"],
            "confirmation_array_sha256": confirmation["point_arrays"]["sha256"],
            "training_summary_sha256": sha256(training_path),
            "confirmation_sha256": sha256(confirmation_path),
            "bilateral_confirmation_sha256": sha256(bilateral_path),
            "parameter_change_audit_sha256": sha256(parameter_audit_path),
            "all_expected_trainable_tensors_changed": bool(
                parameter_audit["all_expected_trainable_tensors_changed"]
            ),
            "fixed_tensors_unchanged": bool(
                parameter_audit["fixed_tensors_unchanged"]
            ),
            "nonpositive_metric_count": int(metrics["nonpositive_metric_count"]),
        }
        row.update({name: float(metrics[name]) for name in METRICS})
        rows.append(row)

    aggregates = {
        name: aggregate([float(row[name]) for row in rows])
        for name in (*METRICS, "training_runtime_seconds")
    }
    sigma_values = np.asarray([row["sigma"] for row in rows], dtype=np.float64)
    median_sigma = float(np.median(sigma_values))
    median_index = int(np.argmin(np.abs(sigma_values - median_sigma)))
    payload = {
        "schema": "gcicy-tn-multiseed-confirmation-summary-v1",
        "run_directory": str(run_dir),
        "seed_protocol_manifest": (
            {
                "path": str(seed_manifest_path),
                "sha256": sha256(seed_manifest_path),
            }
            if seed_manifest is not None
            else None
        ),
        "frozen_protocol": invariant_protocol,
        "runs": rows,
        "aggregate": aggregates,
        "representative_seeds": {
            "best_sigma_activation_seed": rows[int(np.argmin(sigma_values))][
                "activation_seed"
            ],
            "median_sigma_activation_seed": rows[median_index]["activation_seed"],
            "worst_sigma_activation_seed": rows[int(np.argmax(sigma_values))][
                "activation_seed"
            ],
        },
        "checks": {
            "run_count": len(rows),
            "minimum_run_count_met": len(rows) >= args.minimum_runs,
            "all_metrics_positive_on_confirmation": all(
                row["nonpositive_metric_count"] == 0 for row in rows
            ),
            "all_trainable_tensors_changed": all(
                row["all_expected_trainable_tensors_changed"] for row in rows
            ),
            "all_fixed_tensors_unchanged": all(
                row["fixed_tensors_unchanged"] for row in rows
            ),
        },
    }

    output_json = args.output_json.expanduser().resolve()
    output_markdown = args.output_markdown.expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# gCICY TN multiseed confirmation",
        "",
        "| activation seed | best epoch | sigma | chi | abs q99.9 | abs CVaR99 | min eig | train s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {activation_seed} | {best_epoch} | {sigma:.8f} | {chi:.8f} | "
            "{absolute_log_ratio_q999:.8f} | "
            "{absolute_log_ratio_cvar_1pct:.8f} | "
            "{minimum_metric_eigenvalue:.8f} | "
            "{training_runtime_seconds:.1f} |".format(**row)
        )
    lines.extend(
        [
            "",
            (
                "Median sigma: "
                f"{aggregates['sigma']['median']:.8f}; range "
                f"[{aggregates['sigma']['minimum']:.8f}, "
                f"{aggregates['sigma']['maximum']:.8f}]."
            ),
            "",
            f"Machine-readable source: `{output_json}`",
            "",
        ]
    )
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
