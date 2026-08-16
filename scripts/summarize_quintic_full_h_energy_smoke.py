#!/usr/bin/env python3
"""Summarize the registered degree-eight full-H energy GPU smoke scan."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


STAGES = (
    "01_b64_lr1e4",
    "02_b256_lr1e4",
    "03_b1024_lr1e4",
    "04_b256_lr3e5",
    "05_b256_lr3e4",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_metrics(validation: dict[str, Any]) -> dict[str, float]:
    audit = validation["audit_normalized"]
    return {
        "sigma": float(audit["sigma_importance_weighted"]),
        "chi": float(audit["l2_importance_weighted"]),
        "cvar99": float(
            audit["cvar_0.99_abs_residual_importance_weighted"]
        ),
        "q999_unweighted": float(
            audit["abs_residual_unweighted_quantiles"]["q0.999"]
        ),
        "maximum_unweighted": float(
            audit["abs_residual_unweighted_quantiles"]["maximum"]
        ),
        "h_condition_number": float(validation["h_matrix"]["condition_number"]),
        "minimum_metric_eigenvalue": float(
            validation["minimum_metric_eigenvalue"]
        ),
    }


def main() -> None:
    args = parse_args()
    run_root = args.run_root.expanduser().resolve()
    rows = []
    source_hashes = set()
    pullback_hashes = set()
    for stage in STAGES:
        report_path = run_root / stage / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        basis = report["basis"]
        if (
            int(basis["degree"]) != 8
            or int(basis["section_count"]) != 460
            or int(basis["real_hermitian_parameter_count"]) != 211_600
            or int(basis["registered_real_parameter_count"]) != 211_600
        ):
            raise RuntimeError(f"unexpected full-H basis in {report_path}")
        if report["configuration"]["point_loss"] != "energy_l2":
            raise RuntimeError(f"non-energy objective in {report_path}")
        if report["configuration"]["importance_normalization"] != "global_mean":
            raise RuntimeError(f"non-global importance normalization in {report_path}")
        history = report["training"]["history"]
        best_epoch = int(report["training"]["best_epoch"])
        selected = next(row for row in history if int(row["epoch"]) == best_epoch)
        initial = audit_metrics(history[0]["validation"])
        final = audit_metrics(selected["validation"])
        optimization_seconds = float(report["timing_seconds"]["optimization"])
        sigma_reduction = initial["sigma"] - final["sigma"]
        source_hashes.add(
            json.dumps(report["data"]["source_sha256"], sort_keys=True)
        )
        pullback_hashes.add(
            json.dumps(report["data"]["pullback_sha256"], sort_keys=True)
        )
        rows.append(
            {
                "stage": stage,
                "report": str(report_path),
                "report_sha256": sha256_file(report_path),
                "batch_size": int(report["configuration"]["batch_size"]),
                "learning_rate": float(report["configuration"]["learning_rate"]),
                "train_count": int(report["data"]["train_count"]),
                "validation_count": int(report["data"]["validation_count"]),
                "updates": int(report["training"]["total_updates"]),
                "best_epoch": best_epoch,
                "initial": initial,
                "selected": final,
                "sigma_reduction": sigma_reduction,
                "relative_sigma_reduction": sigma_reduction / initial["sigma"],
                "sigma_reduction_per_optimization_second": (
                    sigma_reduction / optimization_seconds
                    if optimization_seconds > 0
                    else 0.0
                ),
                "timing_seconds": report["timing_seconds"],
                "peak_cuda_memory_reserved_bytes": int(
                    report["hardware"]["peak_cuda_memory_reserved_bytes"]
                ),
            }
        )
    if len(source_hashes) != 1 or len(pullback_hashes) != 1:
        raise RuntimeError("full-H smoke stages did not use identical data artifacts")

    ranking = sorted(
        (row["stage"] for row in rows),
        key=lambda stage: next(
            row["selected"]["sigma"] for row in rows if row["stage"] == stage
        ),
    )
    summary = {
        "schema": "quintic-full-h-energy-smoke-summary-v1",
        "scientific_scope": (
            "GPU configuration selection only; smoke accuracy is not a formal result."
        ),
        "common_data_verified": True,
        "source_sha256": json.loads(next(iter(source_hashes))),
        "pullback_sha256": json.loads(next(iter(pullback_hashes))),
        "rows": rows,
        "ranking_by_selected_validation_sigma": ranking,
    }
    output_path = args.out.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output_path)

    if args.markdown_out is not None:
        lines = [
            "# Full-H energy GPU smoke",
            "",
            "Configuration selection only; these are not formal accuracy results.",
            "",
            "| Stage | Batch | LR | Updates | Selected sigma | Relative reduction | Optimization (s) | Reduction/s | Peak GiB | cond(H) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                "| {stage} | {batch} | {lr:.2e} | {updates} | {sigma:.8g} | "
                "{relative:.3%} | {seconds:.2f} | {rate:.3e} | {memory:.3f} | "
                "{condition:.3e} |".format(
                    stage=row["stage"],
                    batch=row["batch_size"],
                    lr=row["learning_rate"],
                    updates=row["updates"],
                    sigma=row["selected"]["sigma"],
                    relative=row["relative_sigma_reduction"],
                    seconds=row["timing_seconds"]["optimization"],
                    rate=row["sigma_reduction_per_optimization_second"],
                    memory=row["peak_cuda_memory_reserved_bytes"] / 2**30,
                    condition=row["selected"]["h_condition_number"],
                )
            )
        markdown_path = args.markdown_out.expanduser().resolve()
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_markdown = markdown_path.with_suffix(markdown_path.suffix + ".tmp")
        temporary_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
        temporary_markdown.replace(markdown_path)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
