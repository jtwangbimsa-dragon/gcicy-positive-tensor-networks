#!/usr/bin/env python3
"""Run whole-fibre sampler-audit seeds in parallel and merge their rows."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--points-per-seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--worker-dir", type=Path)
    return parser.parse_args()


def merge_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    merged: dict[str, int] = {}
    for row in rows:
        for label, count in row[key].items():
            merged[str(label)] = merged.get(str(label), 0) + int(count)
    return dict(sorted(merged.items()))


def main() -> None:
    args = parse_args()
    if args.points_per_seed <= 0 or not args.seeds:
        raise SystemExit("seeds and points-per-seed must be non-empty")
    if len(set(args.seeds)) != len(args.seeds):
        raise SystemExit("sampler-audit seeds must be distinct")
    output = args.out if args.out.is_absolute() else ROOT / args.out
    worker_dir = (
        args.worker_dir
        if args.worker_dir is not None
        else output.parent / f"{output.stem}_workers"
    )
    worker_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    worker_count = min(
        len(args.seeds),
        args.workers if args.workers > 0 else len(args.seeds),
    )
    environment = os.environ.copy()
    environment.update(
        {
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )

    def run_seed(seed: int) -> tuple[int, Path, float]:
        worker_output = worker_dir / f"seed_{seed}.json"
        log_path = worker_dir / f"seed_{seed}.log"
        command = [
            sys.executable,
            str(ROOT / "scripts" / "audit_sampler_fibres.py"),
            "--adapter",
            args.adapter,
            "--model-seed",
            str(args.model_seed),
            "--seeds",
            str(seed),
            "--points-per-seed",
            str(args.points_per_seed),
            "--out",
            str(worker_output),
        ]
        started = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if completed.returncode:
            raise RuntimeError(f"sampler audit seed {seed} failed; see {log_path}")
        return seed, worker_output, time.perf_counter() - started

    started = time.perf_counter()
    completed_rows = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(run_seed, seed): seed for seed in args.seeds}
        for future in as_completed(futures):
            seed, path, elapsed = future.result()
            print(f"seed={seed} elapsed={elapsed:.1f}s", flush=True)
            completed_rows.append((seed, path))

    completed_rows.sort()
    outputs = [json.loads(path.read_text(encoding="utf-8")) for _, path in completed_rows]
    if not all(row["success"] and len(row["per_seed"]) == 1 for row in outputs):
        raise SystemExit("one or more sampler workers did not pass")
    per_seed = [row["per_seed"][0] for row in outputs]
    aggregates = [row["aggregate"] for row in outputs]
    aggregate = {
        "attempted_clusters": sum(int(row["attempted_clusters"]) for row in aggregates),
        "accepted_clusters": sum(int(row["accepted_clusters"]) for row in aggregates),
        "rejected_clusters": sum(int(row["rejected_clusters"]) for row in aggregates),
        "returned_complete_clusters": sum(
            int(row["returned_complete_clusters"]) for row in aggregates
        ),
        "root_count_histogram": merge_counts(aggregates, "root_count_histogram"),
        "valid_root_count_histogram": merge_counts(
            aggregates, "valid_root_count_histogram"
        ),
        "accepted_root_count_histogram": merge_counts(
            aggregates, "accepted_root_count_histogram"
        ),
        "rejection_reasons": merge_counts(aggregates, "rejection_reasons"),
        "minimum_projective_root_separation": min(
            float(row["minimum_projective_root_separation"]) for row in aggregates
        ),
        "maximum_accepted_relative_residual": max(
            float(row["maximum_accepted_relative_residual"]) for row in aggregates
        ),
        "minimum_point_effective_sample_size": min(
            float(row["minimum_point_effective_sample_size"]) for row in aggregates
        ),
        "minimum_cluster_weight_effective_sample_size": min(
            float(row["minimum_cluster_weight_effective_sample_size"])
            for row in aggregates
        ),
        "maximum_mean_normalized_importance_weight": max(
            float(row["maximum_mean_normalized_importance_weight"])
            for row in aggregates
        ),
        "maximum_top_one_percent_importance_weight_fraction": max(
            float(row["maximum_top_one_percent_importance_weight_fraction"])
            for row in aggregates
        ),
        "minimum_jacobian_singular_value": min(
            float(row["minimum_jacobian_singular_value"]) for row in aggregates
        ),
    }
    gate_names = tuple(outputs[0]["gates"])
    gates = {
        name: all(bool(row["gates"][name]) for row in outputs) for name in gate_names
    }
    merged = {
        **{
            key: value
            for key, value in outputs[0].items()
            if key not in {"seeds", "per_seed", "aggregate", "gates", "success", "timing_seconds"}
        },
        "parallel_workers": worker_count,
        "seeds": [int(seed) for seed in args.seeds],
        "per_seed": per_seed,
        "aggregate": aggregate,
        "gates": gates,
        "success": bool(all(gates.values())),
        "timing_seconds": float(time.perf_counter() - started),
        "worker_outputs": [path.relative_to(ROOT).as_posix() for _, path in completed_rows],
    }
    output.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    if not merged["success"]:
        raise SystemExit("merged sampler audit failed")


if __name__ == "__main__":
    main()
