#!/usr/bin/env python3
"""Audit and recover a return-code-zero job after a JSON-path validator fix.

This is intentionally narrower than a general state editor.  It may only
promote one terminal job whose numerical command returned zero, whose declared
outputs are still present, and whose JSON gates all pass under the current
workflow code.  The previous plan lock and job state are preserved verbatim in
an append-only amendment directory before either live file is changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline.experiment_workflow import (  # noqa: E402
    WorkflowError,
    _output_records,
    _validate_json_gates,
    atomic_write_json,
    campaign_lock,
    job_state_path,
    load_workflow_plan,
    read_job_state,
    utc_now,
    verify_frozen_inputs,
    verify_source_guard,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--expected-old-commit", required=True)
    parser.add_argument(
        "--expected-error-fragment",
        default="missing JSON field",
        help="required substring in the terminal validation error",
    )
    return parser.parse_args()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    run_root = args.run_root.expanduser().resolve()
    plan = load_workflow_plan(
        args.manifest,
        repo_root=repo_root,
        frozen_root=args.frozen_root.expanduser().resolve(),
        run_root=run_root,
        python_executable=args.python,
    )
    if args.job_id not in plan.jobs_by_id:
        raise SystemExit(f"unknown job: {args.job_id}")

    lock_path = run_root / ".workflow" / "plan.lock.json"
    provenance_path = run_root / ".workflow" / "campaign_provenance.json"
    if not lock_path.is_file() or not provenance_path.is_file():
        raise SystemExit("campaign plan lock or provenance is missing")

    try:
        with campaign_lock(plan):
            locked_bytes = lock_path.read_bytes()
            locked = json.loads(locked_bytes)
            if locked.get("plan_sha256") != plan.digest:
                raise WorkflowError("locked plan digest does not match current plan")
            old_source = locked.get("source", {})
            if old_source.get("commit") != args.expected_old_commit:
                raise WorkflowError("locked source is not the expected pre-fix commit")

            new_source = verify_source_guard(plan)
            if new_source == old_source:
                raise WorkflowError("current source is identical to the locked source")
            frozen = verify_frozen_inputs(plan)
            if locked.get("frozen_inputs") != frozen:
                raise WorkflowError("frozen-input identity changed")

            state_path = job_state_path(plan, args.job_id)
            state_bytes = state_path.read_bytes()
            state = read_job_state(plan, args.job_id)
            if state is None or state.get("status") != "failed":
                raise WorkflowError("job is not in the terminal failed state")
            if state.get("returncode") != 0:
                raise WorkflowError("numerical command did not return zero")
            attempts = list(state.get("attempts", []))
            if not attempts or attempts[-1].get("returncode") != 0:
                raise WorkflowError("last recorded numerical attempt did not return zero")
            old_error = str(state.get("validation_error", ""))
            if args.expected_error_fragment not in old_error:
                raise WorkflowError("terminal error is not the expected JSON validation error")

            job = plan.jobs_by_id[args.job_id]
            outputs = _output_records(job)
            gates = _validate_json_gates(job)

            stamp = utc_now().replace(":", "").replace("-", "")
            amendment_dir = run_root / ".workflow" / "amendments" / (
                f"{stamp}-{args.job_id}-json-path-hotfix"
            )
            if amendment_dir.exists():
                raise WorkflowError(f"amendment directory already exists: {amendment_dir}")
            amendment_dir.mkdir(parents=True)
            (amendment_dir / "plan.lock.before.json").write_bytes(locked_bytes)
            (amendment_dir / "job-state.before.json").write_bytes(state_bytes)

            amendment = {
                "schema": "gcicy-experiment-source-amendment-v1",
                "created_utc": utc_now(),
                "campaign_id": plan.campaign_id,
                "plan_sha256": plan.digest,
                "job_id": args.job_id,
                "reason": (
                    "Non-numerical JSON-path parser hotfix: literal-dot quantile "
                    "keys were previously split as nested path components."
                ),
                "old_source": old_source,
                "new_source": new_source,
                "old_validation_error": old_error,
                "old_plan_lock_sha256": sha256_bytes(locked_bytes),
                "old_job_state_sha256": sha256_bytes(state_bytes),
                "output_records": outputs,
                "json_gates": gates,
                "numerical_outputs_modified": False,
            }
            atomic_write_json(amendment_dir / "amendment.json", amendment)

            recovered = {
                **state,
                "status": "succeeded",
                "updated_utc": utc_now(),
                "outputs": outputs,
                "json_gates": gates,
                "validation_error": None,
                "recovered_after_json_path_hotfix": {
                    "amendment": str(amendment_dir / "amendment.json"),
                    "old_validation_error": old_error,
                    "old_job_state_sha256": sha256_bytes(state_bytes),
                },
            }
            atomic_write_json(state_path, recovered)

            amendment_ref = {
                "path": str(amendment_dir / "amendment.json"),
                "sha256": sha256_bytes(
                    (amendment_dir / "amendment.json").read_bytes()
                ),
            }
            locked["source"] = new_source
            locked.setdefault("source_amendments", []).append(amendment_ref)
            atomic_write_json(lock_path, locked)

            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            provenance.setdefault("source_amendments", []).append(amendment_ref)
            atomic_write_json(provenance_path, provenance)

            print(json.dumps(amendment, indent=2))
    except (OSError, ValueError, WorkflowError, json.JSONDecodeError) as exc:
        raise SystemExit(f"recovery error: {exc}") from exc


if __name__ == "__main__":
    main()
