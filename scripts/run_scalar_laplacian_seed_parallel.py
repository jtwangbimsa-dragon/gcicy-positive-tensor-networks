#!/usr/bin/env python3
"""Run scalar-Laplacian audit seeds in parallel and merge raw seed rows."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import merge_scalar_laplacian_audits  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--cpu-list",
        type=str,
        help="Optional Linux CPU ids/ranges, for example 2-9 or 0,2,4-6.",
    )
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


def main() -> None:
    args = parse_args()
    spec_path = args.spec.expanduser().resolve()
    data = json.loads(spec_path.read_text(encoding="utf-8"))
    if int(data.get("schema_version", -1)) != 1:
        raise ValueError("scalar spectrum specification must use schema_version 1")
    seeds = [int(seed) for seed in data["spectrum"]["seeds"]]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("parallel scalar audit seeds must be non-empty and distinct")

    base = spec_path.parent
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
    worker_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    cpu_ids = parse_cpu_list(args.cpu_list)
    requested_workers = args.workers if args.workers > 0 else len(seeds)
    worker_count = min(requested_workers, len(seeds))
    if cpu_ids:
        worker_count = min(worker_count, len(cpu_ids))
    if worker_count <= 0:
        raise ValueError("parallel scalar audit needs at least one worker")

    absolute_artifacts = [
        str(resolve_path(path, base)) for path in data["artifacts"]
    ]
    worker_jobs = []
    for index, seed in enumerate(seeds):
        worker_spec = copy.deepcopy(data)
        worker_spec["artifacts"] = absolute_artifacts
        worker_spec["spectrum"]["seeds"] = [seed]
        worker_output = worker_dir / f"seed_{seed}.json"
        worker_spec_path = worker_dir / f"seed_{seed}_spec.json"
        worker_spec["output"] = str(worker_output)
        worker_spec_path.write_text(
            json.dumps(worker_spec, indent=2) + "\n",
            encoding="utf-8",
        )
        worker_jobs.append(
            {
                "index": index,
                "seed": seed,
                "spec": worker_spec_path,
                "output": worker_output,
                "log": worker_dir / f"seed_{seed}.log",
                "cpu": cpu_ids[index % len(cpu_ids)] if cpu_ids else None,
            }
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
    taskset = shutil.which("taskset")

    def run_worker(job: dict) -> dict:
        command = [
            sys.executable,
            str(ROOT / "scripts" / "audit_scalar_laplacian.py"),
            "--spec",
            str(job["spec"]),
            "--out",
            str(job["output"]),
            "--allow-failed",
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
        job["log"].write_text(
            completed.stdout + completed.stderr,
            encoding="utf-8",
        )
        if completed.returncode != 0 or not job["output"].exists():
            raise RuntimeError(
                f"scalar audit seed {job['seed']} failed; see {job['log']}"
            )
        return {
            **job,
            "elapsed_seconds": elapsed,
            "summary": json.loads(job["output"].read_text(encoding="utf-8")),
        }

    wall_started = time.perf_counter()
    completed_jobs = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(run_worker, job): job for job in worker_jobs}
        for future in as_completed(futures):
            result = future.result()
            completed_jobs.append(result)
            print(
                f"seed={result['seed']} cpu={result['cpu']} "
                f"seconds={result['elapsed_seconds']:.1f}",
                flush=True,
            )
    parallel_wall_seconds = time.perf_counter() - wall_started
    completed_jobs.sort(key=lambda row: seeds.index(int(row["seed"])))
    merged = merge_scalar_laplacian_audits(
        [row["summary"] for row in completed_jobs]
    )
    merged["specification"] = str(spec_path)
    merged["parallel_execution"] = {
        "worker_count": int(worker_count),
        "cpu_list": cpu_ids,
        "parallel_wall_seconds": float(parallel_wall_seconds),
        "worker_outputs": [str(row["output"]) for row in completed_jobs],
        "worker_logs": [str(row["log"]) for row in completed_jobs],
        "worker_elapsed_seconds": {
            str(row["seed"]): float(row["elapsed_seconds"])
            for row in completed_jobs
        },
    }
    output.write_text(
        json.dumps(merged, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"success={merged['success']}")
    print(f"parallel_wall_seconds={parallel_wall_seconds:.1f}")
    print(f"wrote {output}")
    if not merged["success"] and not args.allow_failed:
        failed = ", ".join(
            key for key, value in merged["gates"].items() if not value
        )
        raise SystemExit(f"scalar spectrum audit failed gates: {failed}")


if __name__ == "__main__":
    main()
