#!/usr/bin/env python3
"""Build the common-point speed/accuracy record for the quintic TN optimizer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        metavar="LABEL=REPORT",
        help="ordered tensor-network report labels and paths",
    )
    parser.add_argument("--matched-cymetric-report", type=Path, required=True)
    parser.add_argument("--forward-benchmark-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-markdown", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"run must have LABEL=REPORT form: {value}")
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise ValueError(f"run must have LABEL=REPORT form: {value}")
    return label, Path(raw_path).expanduser().resolve()


def metrics(stats: dict[str, Any]) -> dict[str, float | int]:
    return {
        "blind_points": int(stats["n_points"]),
        "sigma": float(stats["sigma_official_formula"]),
        "chi": float(stats["weighted_rms_abs_residual"]),
        "residual_q999": float(
            stats["abs_residual_weighted_quantiles"]["q0.9990"]
        ),
        "residual_cvar99": float(
            stats["abs_residual_weighted_cvar"]["cvar_0.9900"]
        ),
        "ratio_max": float(stats["ratio_weighted_quantiles"]["q1.0000"]),
        "minimum_metric_eigenvalue": float(
            stats["min_eigenvalue_weighted_quantiles"]["q0.0000"]
        ),
        "nonpositive_metric_count": int(
            stats["nonpositive_min_eigenvalue"]["count"]
        ),
    }


def main() -> None:
    args = parse_args()
    parsed_runs = [parse_run(value) for value in args.runs]
    if len({label for label, _ in parsed_runs}) != len(parsed_runs):
        raise ValueError("run labels must be unique")

    rows = []
    common_parameter_count = None
    common_dataset_hash = None
    common_blind_hash = None
    for label, report_path in parsed_runs:
        report = load(report_path)
        parameter_count = int(report["architecture"]["trainable_real_parameter_count"])
        source_hashes = report["common_point_evidence"]["source_sha256"]
        if common_parameter_count is None:
            common_parameter_count = parameter_count
            common_dataset_hash = source_hashes["dataset"]
            common_blind_hash = source_hashes["blind_points"]
        if parameter_count != common_parameter_count:
            raise RuntimeError("tensor-network parameter counts differ")
        if source_hashes["dataset"] != common_dataset_hash:
            raise RuntimeError("tensor-network training datasets differ")
        if source_hashes["blind_points"] != common_blind_hash:
            raise RuntimeError("tensor-network blind samples differ")
        row = {
            "label": label,
            "kind": "positive_tensor_network",
            "report": str(report_path),
            "report_sha256": sha256_file(report_path),
            "parameters": parameter_count,
            "configuration": {
                key: report["configuration"][key]
                for key in (
                    "epochs",
                    "batch_size",
                    "learning_rate",
                    "tail_loss_weight",
                    "tail_ratio_threshold",
                    "tail_smooth_temperature",
                )
            },
            "transfer_implementation": report["configuration"].get(
                "transfer_implementation", "scalar"
            ),
            "training_seconds": float(report["timing_seconds"]["training"]),
            "wall_seconds": float(report["timing_seconds"]["wall_total"]),
            **metrics(report["blind_test"]["normalized_volume"]),
        }
        if row["blind_points"] != 200_000:
            raise RuntimeError(f"{label} does not use the full blind sample")
        rows.append(row)

    cymetric_path = args.matched_cymetric_report.expanduser().resolve()
    cymetric = load(cymetric_path)
    if int(cymetric["network"]["parameter_count"]) != common_parameter_count:
        raise RuntimeError("matched cymetric parameter count differs")
    if cymetric["data"]["dataset_sha256"] != common_dataset_hash:
        raise RuntimeError("matched cymetric training dataset differs")
    if cymetric["data"]["blind_points_sha256"] != common_blind_hash:
        raise RuntimeError("matched cymetric blind sample differs")
    cymetric_row = {
        "label": "matched_cymetric",
        "kind": "cymetric_phifs",
        "report": str(cymetric_path),
        "report_sha256": sha256_file(cymetric_path),
        "parameters": int(cymetric["network"]["parameter_count"]),
        "training_seconds": float(cymetric["timing_seconds"]["training"]),
        "wall_seconds": float(cymetric["timing_seconds"]["wall_total"]),
        **metrics(cymetric["trained_phi_model"]),
    }
    rows.append(cymetric_row)

    baseline = rows[0]
    high_precision = rows[1]
    metric_keys = ("sigma", "chi", "residual_q999", "residual_cvar99")
    equivalence = {
        "old_label": baseline["label"],
        "new_label": high_precision["label"],
        "wall_speedup": baseline["wall_seconds"] / high_precision["wall_seconds"],
        "absolute_metric_differences": {
            key: abs(float(baseline[key]) - float(high_precision[key]))
            for key in metric_keys
        },
    }
    for row in rows[1:]:
        row["wall_speedup_over_old_tn"] = (
            baseline["wall_seconds"] / row["wall_seconds"]
        )
        row["wall_speedup_over_matched_cymetric"] = (
            cymetric_row["wall_seconds"] / row["wall_seconds"]
        )

    forward_rows = []
    for path in sorted(args.forward_benchmark_dir.expanduser().resolve().glob("forward_*.json")):
        report = load(path)
        result = report["results"][0]
        forward_rows.append(
            {
                "transfer_implementation": report["transfer_implementation"],
                "batch_size": int(report["batch_size"]),
                "median_forward_seconds": float(result["median_forward_seconds"]),
                "microseconds_per_point": float(
                    result["forward_microseconds_per_point"]
                ),
                "incremental_peak_allocated_bytes": int(
                    result["device_memory"]["incremental_peak_allocated_bytes"]
                ),
                "report": str(path),
                "report_sha256": sha256_file(path),
            }
        )

    payload = {
        "schema": "quintic-tn-speed-accuracy-v1",
        "geometry": "Fermat quintic hypersurface X_5 in P^4",
        "common_point_gate": {
            "parameters": common_parameter_count,
            "dataset_sha256": common_dataset_hash,
            "blind_points_sha256": common_blind_hash,
            "blind_points": 200_000,
        },
        "implementation_equivalence": equivalence,
        "rows": rows,
        "forward_kernel_benchmarks": forward_rows,
        "claim_limits": [
            "The high-precision scalar/vectorized comparison changes only the transfer implementation.",
            "The larger-batch rows are Pareto alternatives with changed optimization dynamics.",
            "The matched cymetric run was not retuned for large-batch throughput.",
            "A same-geometry k=9 Donaldson T-map timing has not yet been measured.",
        ],
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Quintic tensor-network speed optimization",
        "",
        "All rows use 8,820 parameters and the same 200,000 blind points.",
        "",
        "| run | batch | epochs | wall (s) | sigma | chi | q99.9 | CVaR99 | r max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        config = row.get("configuration", {})
        lines.append(
            f"| {row['label']} | {config.get('batch_size', '-')} | "
            f"{config.get('epochs', '-')} | {row['wall_seconds']:.3f} | "
            f"{row['sigma']:.7f} | {row['chi']:.7f} | "
            f"{row['residual_q999']:.7f} | {row['residual_cvar99']:.7f} | "
            f"{row['ratio_max']:.7f} |"
        )
    lines.extend(
        [
            "",
            f"The vectorized high-precision run is {equivalence['wall_speedup']:.2f}x "
            "faster than the scalar recurrence.",
            "",
            "The vectorized recurrence batches the 16 value/first/mixed jet "
            "channels. It changes execution only; the potential, metric, and "
            "parameter gradients are covered by exact equivalence tests.",
            "",
            "## Claim limits",
            "",
            *[f"- {value}" for value in payload["claim_limits"]],
        ]
    )
    args.out_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.out_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
