#!/usr/bin/env python3
"""Summarize descriptive runtime evidence for the matched X11 experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(mean(values)),
        "sample_standard_deviation": float(stdev(values)),
        "minimum": float(min(values)),
        "maximum": float(max(values)),
    }


def format_mean_sd(record: dict[str, float], digits: int = 1) -> str:
    return (
        f"{record['mean']:.{digits}f}"
        f"\\(\\pm\\){record['sample_standard_deviation']:.{digits}f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--phi-memory-report", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-tex", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    source_paths: list[Path] = []
    tn_training: list[dict[str, Any]] = []
    phi_training: list[dict[str, Any]] = []
    tn_blind: list[dict[str, Any]] = []
    phi_blind: list[dict[str, Any]] = []
    for replicate in (1, 2, 3):
        replicate_dir = run_dir / f"replicate_{replicate}"
        paths = {
            "tn_training": replicate_dir / "tn_training_summary.json",
            "phi_training": replicate_dir / "phi" / "report.json",
            "tn_blind": replicate_dir / "tn_final_blind.json",
            "phi_blind": replicate_dir / "phi_final_blind" / "report.json",
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise SystemExit(f"missing runtime artifacts: {missing}")
        source_paths.extend(paths.values())
        tn_training.append(load(paths["tn_training"]))
        phi_training.append(load(paths["phi_training"]))
        tn_blind.append(load(paths["tn_blind"]))
        phi_blind.append(load(paths["phi_blind"]))

    memory_path = args.phi_memory_report.expanduser().resolve()
    if not memory_path.is_file():
        raise SystemExit(f"missing phi memory replay: {memory_path}")
    source_paths.append(memory_path)
    phi_memory_report = load(memory_path)
    phi_memory = phi_memory_report.get("training_device_memory")
    if not phi_memory:
        raise SystemExit("phi memory replay has no training_device_memory field")

    tn_training_seconds = [float(row["runtime_seconds"]) for row in tn_training]
    phi_training_seconds = [
        float(row["timing_seconds"]["optimization"]) for row in phi_training
    ]
    tn_eval_microseconds = [
        1.0e6 * float(row["runtime_seconds"]) / int(row["points"])
        for row in tn_blind
    ]
    phi_eval_microseconds = [
        1.0e6
        * float(row["timing_seconds"]["blind_phi_evaluation"])
        / int(row["configuration"]["blind_points"])
        for row in phi_blind
    ]
    tn_best_epochs = [float(row["best_epoch"]) for row in tn_training]
    phi_best_epochs = [
        float(row["training"]["best_epoch"]) for row in phi_training
    ]
    tn_peak_allocated = max(
        int(row["device_memory"]["maximum_allocated_bytes"])
        for row in tn_training
    )

    output = {
        "schema": "gcicy-x11-matched-runtime-summary-v1",
        "scope": (
            "Descriptive RTX 4090 measurements under the frozen fixed-update "
            "protocol; not a time-matched optimization study."
        ),
        "hardware": tn_training[0]["device"],
        "fixed_updates_per_model": 15360,
        "tn": {
            "trainable_real_parameter_count": int(
                tn_training[0]["trainable_real_parameter_count"]
            ),
            "training_wall_seconds": summary(tn_training_seconds),
            "best_selection_epoch": summary(tn_best_epochs),
            "final_blind_evaluation_microseconds_per_point": summary(
                tn_eval_microseconds
            ),
            "peak_allocated_bytes": tn_peak_allocated,
            "peak_allocated_gib": tn_peak_allocated / 2**30,
            "peak_memory_evidence": "maximum over three original training reports",
        },
        "phi": {
            "trainable_real_parameter_count": int(
                phi_training[0]["network"]["parameter_count"]
            ),
            "training_wall_seconds": summary(phi_training_seconds),
            "best_selection_epoch": summary(phi_best_epochs),
            "final_blind_evaluation_microseconds_per_point": summary(
                phi_eval_microseconds
            ),
            "peak_allocated_bytes": int(phi_memory["maximum_allocated_bytes"]),
            "peak_allocated_gib": int(phi_memory["maximum_allocated_bytes"])
            / 2**30,
            "peak_memory_evidence": (
                "representative deterministic replay of seed 20261321 under "
                "the frozen protocol"
            ),
            "memory_replay_report": str(memory_path),
            "memory_replay_report_sha256": sha256(memory_path),
        },
        "timing_boundary": {
            "tn_training": (
                "runtime_seconds from the original TN training summaries; "
                "includes model and common-pool setup"
            ),
            "phi_training": (
                "optimization seconds from the original residual reports; "
                "excludes dataset preparation"
            ),
            "evaluation": (
                "per-point elapsed time from each implementation's original "
                "200000-point final-blind audit; not a fused-kernel microbenchmark"
            ),
        },
        "sources": {
            str(path): sha256(path)
            for path in source_paths
        },
    }

    tex = "\n".join(
        [
            "\\resizebox{\\textwidth}{!}{%",
            "\\begin{tabular}{lrrrrrr}",
            "\\toprule",
            "architecture & $P$ & updates & training wall time (s) & "
            "peak allocated (GiB) & evaluation ($\\mu$s/point) & "
            "best selection epoch \\\\",
            "\\midrule",
            "positive TN & "
            f"{output['tn']['trainable_real_parameter_count']:,} & "
            f"{output['fixed_updates_per_model']:,} & "
            f"{format_mean_sd(output['tn']['training_wall_seconds'])} & "
            f"{output['tn']['peak_allocated_gib']:.2f} & "
            f"{format_mean_sd(output['tn']['final_blind_evaluation_microseconds_per_point'])} & "
            f"{format_mean_sd(output['tn']['best_selection_epoch'])} \\\\",
            "$K_{H_2}+\\phi$ residual & "
            f"{output['phi']['trainable_real_parameter_count']:,} & "
            f"{output['fixed_updates_per_model']:,} & "
            f"{format_mean_sd(output['phi']['training_wall_seconds'])} & "
            f"{output['phi']['peak_allocated_gib']:.2f} & "
            f"{format_mean_sd(output['phi']['final_blind_evaluation_microseconds_per_point'])} & "
            f"{format_mean_sd(output['phi']['best_selection_epoch'])} \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            "",
        ]
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.output_tex.write_text(tex, encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
