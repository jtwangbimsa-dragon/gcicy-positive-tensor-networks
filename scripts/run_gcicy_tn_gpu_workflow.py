#!/usr/bin/env python3
"""Validate, run, resume, inspect, and aggregate a post-v1 TN GPU campaign."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline.experiment_workflow import (  # noqa: E402
    WorkflowError,
    bootstrap_lock,
    check_cuda,
    initialize_run_root,
    load_workflow_plan,
    record_cuda_identity,
    run_workflow,
    selection_requires_gpu,
    status_rows,
    summarize_results,
    verify_cuda_requirements,
    verify_frozen_inputs,
    verify_source_guard,
)


def add_plan_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--set",
        dest="variables",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="override one explicit manifest variable",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    validate = subparsers.add_parser(
        "validate", help="verify schema, source, and frozen hashes"
    )
    add_plan_arguments(validate)
    validate.add_argument("--gpu", default="0")
    validate.add_argument("--check-cuda", action="store_true")

    plan = subparsers.add_parser("plan", help="show the expanded immutable DAG")
    add_plan_arguments(plan)
    plan.add_argument("--json", action="store_true")

    run = subparsers.add_parser("run", help="run or resume selected jobs")
    add_plan_arguments(run)
    run.add_argument("--gpu", default="0")
    run.add_argument("--lock-root", type=Path, default=Path("/tmp/gcicy-tn-gpu-locks"))
    run.add_argument("--gpu-lock-timeout-seconds", type=float, default=-1)
    run.add_argument("--heartbeat-seconds", type=float, default=30.0)
    run.add_argument("--phase", action="append", default=[])
    run.add_argument("--only", action="append", default=[])
    run.add_argument("--keep-going", action="store_true")
    run.add_argument("--skip-cuda-check", action="store_true")

    status = subparsers.add_parser("status", help="show durable job states")
    add_plan_arguments(status)
    status.add_argument("--json", action="store_true")

    summarize = subparsers.add_parser(
        "summarize", help="create a content-addressed long-form result snapshot"
    )
    add_plan_arguments(summarize)
    summarize.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def parse_variables(rows: list[str]) -> dict[str, str]:
    variables = {}
    for row in rows:
        if "=" not in row:
            raise WorkflowError(f"--set requires NAME=VALUE, got {row!r}")
        key, value = row.split("=", 1)
        if not key:
            raise WorkflowError("--set variable name may not be empty")
        variables[key] = value
    return variables


def load_plan(args: argparse.Namespace):
    return load_workflow_plan(
        args.manifest,
        repo_root=args.repo_root,
        frozen_root=args.frozen_root,
        run_root=args.run_root,
        variable_overrides=parse_variables(args.variables),
        python_executable=args.python,
    )


def print_plan(plan, *, as_json: bool) -> None:
    rows = [
        {
            "job_id": job.id,
            "phase": job.phase,
            "gpu": job.gpu,
            "needs": list(job.needs),
            "job_sha256": job.digest,
            "command": list(job.command),
        }
        for job in plan.jobs
    ]
    if as_json:
        print(
            json.dumps(
                {
                    "campaign_id": plan.campaign_id,
                    "plan_sha256": plan.digest,
                    "job_count": len(rows),
                    "jobs": rows,
                },
                indent=2,
            )
        )
        return
    print(f"campaign={plan.campaign_id}")
    print(f"plan_sha256={plan.digest}")
    print(f"jobs={len(rows)} phases={','.join(plan.phases)}")
    for row in rows:
        dependencies = ",".join(row["needs"]) or "-"
        print(
            f"{row['job_id']:<54} phase={row['phase']:<20} "
            f"gpu={str(row['gpu']).lower():<5} needs={dependencies}"
        )


def main() -> None:
    args = parse_args()
    try:
        plan = load_plan(args)
        if args.action == "plan":
            print_plan(plan, as_json=args.json)
            return
        if args.action == "validate":
            source = verify_source_guard(plan)
            frozen = verify_frozen_inputs(plan)
            payload = {
                "campaign_id": plan.campaign_id,
                "plan_sha256": plan.digest,
                "source": source,
                "frozen_inputs": frozen,
            }
            if args.check_cuda:
                cuda = check_cuda(args.python, args.gpu)
                verify_cuda_requirements(plan, cuda)
                payload["cuda"] = cuda
            print(json.dumps(payload, indent=2))
            return
        if args.action == "status":
            rows = status_rows(plan)
            if args.json:
                print(json.dumps(rows, indent=2))
            else:
                counts = Counter(row["status"] for row in rows)
                print(f"campaign={plan.campaign_id} plan_sha256={plan.digest}")
                print(
                    " ".join(f"{key}={value}" for key, value in sorted(counts.items()))
                )
                for row in rows:
                    print(
                        f"{row['job_id']:<54} {row['status']:<12} "
                        f"attempts={row['attempts']} phase={row['phase']}"
                    )
            return
        if args.action == "run":
            selected_phases = args.phase or None
            selected_jobs = args.only or None
            gpu_required = selection_requires_gpu(
                plan,
                phases=selected_phases,
                only=selected_jobs,
            )
            if args.skip_cuda_check and gpu_required:
                raise WorkflowError(
                    "--skip-cuda-check is allowed only when the selected job "
                    "closure contains no GPU work"
                )
            with bootstrap_lock(
                plan,
                args.lock_root.expanduser().resolve(),
                timeout_seconds=args.gpu_lock_timeout_seconds,
            ):
                source = verify_source_guard(plan)
                frozen = verify_frozen_inputs(plan)
                initialize_run_root(
                    plan,
                    frozen_inputs=frozen,
                    source_identity=source,
                )
                if gpu_required:
                    cuda = check_cuda(args.python, args.gpu)
                    verify_cuda_requirements(plan, cuda)
                    record_cuda_identity(plan, gpu_id=args.gpu, cuda=cuda)
                    print(
                        f"cuda_device={cuda['device']} torch={cuda['torch']} "
                        f"cuda={cuda['cuda']}",
                        flush=True,
                    )
            counts = run_workflow(
                plan,
                gpu_id=args.gpu,
                lock_root=args.lock_root.expanduser().resolve(),
                gpu_lock_timeout_seconds=args.gpu_lock_timeout_seconds,
                heartbeat_seconds=args.heartbeat_seconds,
                phases=selected_phases,
                only=selected_jobs,
                keep_going=args.keep_going,
            )
            print(json.dumps(counts, indent=2))
            return
        if args.action == "summarize":
            output = summarize_results(plan, allow_incomplete=args.allow_incomplete)
            print(f"wrote {output}")
            return
        raise WorkflowError(f"unsupported action: {args.action}")
    except (OSError, ValueError, WorkflowError, json.JSONDecodeError) as exc:
        raise SystemExit(f"workflow error: {exc}") from exc


if __name__ == "__main__":
    main()
