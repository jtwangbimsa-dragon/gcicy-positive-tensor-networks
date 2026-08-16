#!/usr/bin/env python3
"""Run frozen-seed metric-volume tail audits in parallel and merge their gates."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cpu-list", type=str)
    parser.add_argument("--worker-dir", type=Path)
    parser.add_argument("--allow-failed", action="store_true")
    return parser.parse_args()


def resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def parse_cpu_list(value: str | None) -> list[int]:
    if value is None:
        return []
    output = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            output.extend(range(int(left), int(right) + 1))
        else:
            output.append(int(item))
    if not output or len(set(output)) != len(output) or min(output) < 0:
        raise ValueError("CPU list must contain distinct nonnegative ids")
    return output


def descriptive_interval(values: list[float]) -> list[float]:
    array = np.asarray(values, dtype=float)
    if len(array) < 2:
        return [float(array[0]), float(array[0])]
    half_width = 1.96 * float(np.std(array, ddof=1)) / np.sqrt(len(array))
    mean = float(np.mean(array))
    return [mean - half_width, mean + half_width]


def main() -> None:
    args = parse_args()
    spec_path = args.spec.expanduser().resolve()
    data = json.loads(spec_path.read_text(encoding="utf-8"))
    if int(data.get("schema_version", -1)) != 1:
        raise ValueError("tail-audit specification must use schema_version 1")
    base = spec_path.parent
    model = data["model"]
    audit = data["audit"]
    seeds = [int(seed) for seed in audit["seeds"]]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("tail-audit seeds must be non-empty and distinct")
    points = int(audit["points_per_seed"])
    if points <= 0:
        raise ValueError("points_per_seed must be positive")
    artifacts = [resolve_path(path, base) for path in data["artifacts"]]
    configured_output = data.get(
        "output",
        f"../outputs/pipeline/{spec_path.stem}.json",
    )
    output = (
        args.out.expanduser().resolve()
        if args.out is not None
        else resolve_path(configured_output, base)
    )
    worker_dir = (
        args.worker_dir.expanduser().resolve()
        if args.worker_dir is not None
        else output.parent / f"{output.stem}_workers"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    worker_dir.mkdir(parents=True, exist_ok=True)

    cpu_ids = parse_cpu_list(args.cpu_list)
    requested_workers = args.workers if args.workers > 0 else len(seeds)
    worker_count = min(requested_workers, len(seeds))
    if cpu_ids:
        worker_count = min(worker_count, len(cpu_ids))
    if worker_count <= 0:
        raise ValueError("tail audit needs at least one worker")
    taskset = shutil.which("taskset")
    environment = os.environ.copy()
    environment.update(
        {
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )

    jobs = []
    for index, seed in enumerate(seeds):
        jobs.append(
            {
                "seed": seed,
                "output": worker_dir / f"seed_{seed}.json",
                "log": worker_dir / f"seed_{seed}.log",
                "cpu": cpu_ids[index % len(cpu_ids)] if cpu_ids else None,
            }
        )

    def run_worker(job: dict) -> dict:
        command = [
            sys.executable,
            str(ROOT / "scripts" / "audit_metric_volume_weight_tail.py"),
            "--adapter",
            str(data["adapter"]),
            "--model-seed",
            str(int(model["seed"])),
            "--exact-model" if bool(model.get("exact", True)) else "--no-exact-model",
            "--artifacts",
            *[str(path) for path in artifacts],
            "--seed",
            str(job["seed"]),
            "--points",
            str(points),
            "--sampling-workers",
            str(int(audit.get("sampling_workers_per_seed", 1))),
            "--sampling-cluster-size",
            str(int(audit.get("sampling_cluster_size", 1))),
            "--sampling-backend",
            str(audit.get("sampling_backend", "process")),
            "--top-points",
            str(int(audit.get("top_points", 16))),
            "--out",
            str(job["output"]),
        ]
        if job["cpu"] is not None and taskset is not None:
            command = [taskset, "-c", str(job["cpu"]), *command]
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        elapsed = time.perf_counter() - started
        job["log"].write_text(completed.stdout + completed.stderr, encoding="utf-8")
        if completed.returncode != 0 or not job["output"].exists():
            raise RuntimeError(
                f"metric-volume tail seed {job['seed']} failed; see {job['log']}"
            )
        return {
            **job,
            "elapsed_seconds": elapsed,
            "result": json.loads(job["output"].read_text(encoding="utf-8")),
        }

    wall_started = time.perf_counter()
    completed_jobs = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(run_worker, job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result()
            completed_jobs.append(row)
            print(
                f"seed={row['seed']} cpu={row['cpu']} "
                f"seconds={row['elapsed_seconds']:.1f}",
                flush=True,
            )
    completed_jobs.sort(key=lambda row: seeds.index(int(row["seed"])))
    parallel_wall_seconds = time.perf_counter() - wall_started

    artifact_keys = [
        row["artifact_key"]
        for row in completed_jobs[0]["result"]["artifacts"]
    ]
    summaries = []
    all_gates = []
    minimum_point_ess = float(audit["minimum_point_effective_sample_size"])
    minimum_cluster_ess = float(audit["minimum_cluster_effective_sample_size"])
    maximum_top_share = float(audit["maximum_top_point_weight_share"])
    maximum_l2 = float(audit["maximum_sqrt_squared_energy"])
    for artifact_index, artifact_key in enumerate(artifact_keys):
        seed_rows = []
        for job in completed_jobs:
            artifact = job["result"]["artifacts"][artifact_index]
            if artifact["artifact_key"] != artifact_key:
                raise ValueError("artifact order changed between tail-audit workers")
            errors = artifact["volume_ratio_error_statistics"]
            seed_rows.append(
                {
                    "seed": int(job["seed"]),
                    "point_effective_sample_size": float(
                        artifact["point_effective_sample_size"]
                    ),
                    "cluster_effective_sample_size": float(
                        artifact["cluster_weight_effective_sample_size"]
                    ),
                    "top_point_weight_share": float(
                        artifact["point_weight_concentration"]["1"]
                    ),
                    "sigma": float(errors["sigma"]),
                    "sqrt_squared_energy": float(errors["sqrt_squared_energy"]),
                    "normalized_ratio_max": float(errors["normalized_ratio_max"]),
                    "top_point": artifact["top_points"][0],
                }
            )
        gates = {
            "point_effective_sample_size": all(
                row["point_effective_sample_size"] >= minimum_point_ess
                for row in seed_rows
            ),
            "cluster_effective_sample_size": all(
                row["cluster_effective_sample_size"] >= minimum_cluster_ess
                for row in seed_rows
            ),
            "top_point_weight_share": all(
                row["top_point_weight_share"] <= maximum_top_share
                for row in seed_rows
            ),
            "sqrt_squared_energy": all(
                row["sqrt_squared_energy"] <= maximum_l2 for row in seed_rows
            ),
        }
        all_gates.extend(gates.values())
        sigma_values = [row["sigma"] for row in seed_rows]
        l2_values = [row["sqrt_squared_energy"] for row in seed_rows]
        summaries.append(
            {
                "artifact_key": artifact_key,
                "artifact_path": completed_jobs[0]["result"]["artifacts"][
                    artifact_index
                ]["artifact_path"],
                "mean_sigma": float(np.mean(sigma_values)),
                "sigma_95_percent_ci": descriptive_interval(sigma_values),
                "mean_sqrt_squared_energy": float(np.mean(l2_values)),
                "sqrt_squared_energy_95_percent_ci": descriptive_interval(l2_values),
                "minimum_point_effective_sample_size": min(
                    row["point_effective_sample_size"] for row in seed_rows
                ),
                "minimum_cluster_effective_sample_size": min(
                    row["cluster_effective_sample_size"] for row in seed_rows
                ),
                "maximum_top_point_weight_share": max(
                    row["top_point_weight_share"] for row in seed_rows
                ),
                "maximum_sqrt_squared_energy": max(l2_values),
                "maximum_normalized_ratio": max(
                    row["normalized_ratio_max"] for row in seed_rows
                ),
                "seeds": seed_rows,
                "gates": gates,
                "success": bool(all(gates.values())),
            }
        )

    omega_rows = [
        {
            "seed": int(job["seed"]),
            "point_effective_sample_size": float(
                job["result"]["omega_weight_summary"]["point_effective_sample_size"]
            ),
        }
        for job in completed_jobs
    ]
    result = {
        "schema_version": 1,
        "description": (
            "Frozen-seed fail-fast audit of upper volume-ratio tails and "
            "metric-volume importance-weight concentration."
        ),
        "specification": str(spec_path),
        "adapter": data["adapter"],
        "model": model,
        "seeds": seeds,
        "points_per_seed": points,
        "thresholds": {
            "minimum_point_effective_sample_size": minimum_point_ess,
            "minimum_cluster_effective_sample_size": minimum_cluster_ess,
            "maximum_top_point_weight_share": maximum_top_share,
            "maximum_sqrt_squared_energy": maximum_l2,
        },
        "omega_weight_diagnostics": omega_rows,
        "artifacts": summaries,
        "parallel_execution": {
            "worker_count": worker_count,
            "cpu_list": cpu_ids,
            "parallel_wall_seconds": parallel_wall_seconds,
            "worker_outputs": [str(row["output"]) for row in completed_jobs],
            "worker_logs": [str(row["log"]) for row in completed_jobs],
            "worker_elapsed_seconds": {
                str(row["seed"]): float(row["elapsed_seconds"])
                for row in completed_jobs
            },
        },
        "success": bool(all(all_gates)),
    }
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"success={result['success']}")
    print(f"parallel_wall_seconds={parallel_wall_seconds:.1f}")
    print(f"wrote {output}")
    if not result["success"] and not args.allow_failed:
        raise SystemExit("metric-volume tail audit failed registered gates")


if __name__ == "__main__":
    main()
