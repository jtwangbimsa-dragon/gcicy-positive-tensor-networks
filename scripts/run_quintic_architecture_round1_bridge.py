#!/usr/bin/env python3
"""Prepare, run, and normalize the fixed quintic architecture Round-1 bridge."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import fcntl
import json
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline.architecture_auto_research import (
    ACTION_SCHEMA,
    AutoResearchError,
    CampaignStore,
    sha256_file,
)  # noqa: E402
from gcicy_metric.pipeline.quintic_architecture_round1_bridge import (  # noqa: E402
    ROUND1_BASELINE_SHA256,
    ROUND1_EDGE,
    ROUND1_NEW_REAL_PARAMETERS,
    ROUND1_SOURCE_DIMENSION,
    ROUND1_STRUCTURAL_MAXIMUM,
    ROUND1_TARGET_DIMENSION,
    normalize_round1_bridge,
    prepare_round1_bridge,
    run_round1_bridge,
)


DEFAULT_CANDIDATE_ID = "edge6-rank39"
_USER_UNIT = re.compile(r"[A-Za-z0-9_.@:-]+\.service")
_WORKFLOW_JOB = re.compile(r"[A-Za-z0-9_.-]+")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GPU_LOCK = re.compile(r"gcicy-tn-gpu-[A-Za-z0-9_.-]+\.lock")


def _baseline(value: str) -> tuple[int, Path]:
    try:
        seed_text, path_text = value.split("=", 1)
        seed = int(seed_text)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError(
            "expected SEED=/path/to/checkpoint.pt"
        ) from error
    if seed <= 0 or not path_text:
        raise argparse.ArgumentTypeError("seed and checkpoint path must be nonempty")
    return seed, Path(path_text)


def _read_json(path: Path):
    try:
        return json.loads(path.expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AutoResearchError(f"cannot read JSON {path}") from error


def _read_object(path: Path) -> dict[str, object]:
    value = _read_json(path)
    if not isinstance(value, dict):
        raise AutoResearchError(f"JSON root must be an object: {path}")
    return value


def _runtime(args: argparse.Namespace) -> dict[str, object]:
    return {
        "device": args.device,
        "threads": args.threads,
        "train_chunk_size": args.train_chunk_size,
        "feature_batch_size": args.feature_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "early_stopping_evaluations": args.early_stopping_evaluations,
    }


def _baselines(args: argparse.Namespace) -> dict[int, Path]:
    baselines = dict(args.baseline_checkpoint)
    if len(baselines) != len(args.baseline_checkpoint):
        raise AutoResearchError("baseline seed was provided more than once")
    return baselines


def _ensure_registered_action(
    store: CampaignStore,
    *,
    candidate_id: str,
) -> dict[str, object]:
    ledger = store.status()
    existing_round = ledger.get("rounds", {}).get("1")
    if isinstance(existing_round, dict):
        candidate = existing_round.get("candidates", {}).get(candidate_id)
        if not isinstance(candidate, dict):
            raise AutoResearchError("Round 1 belongs to another candidate set")
        return _read_object(Path(candidate["action_path"]))
    if ledger.get("current_round") != 0 or ledger.get("champion_candidate_id") != "baseline":
        raise AutoResearchError("Round 1 cannot be bootstrapped from this campaign state")
    action = {
        "schema": ACTION_SCHEMA,
        "candidate_id": candidate_id,
        "round": 1,
        "parent_family_sha256": ledger["champion_family"]["family_sha256"],
        "search_indices_sha256": ledger["search_indices_sha256"],
        "mutation": {
            "kind": "rank",
            "target": "internal-edge",
            "edge": ROUND1_EDGE,
            "source_dimension": ROUND1_SOURCE_DIMENSION,
            "target_dimension": ROUND1_TARGET_DIMENSION,
            "structural_maximum": ROUND1_STRUCTURAL_MAXIMUM,
            "new_output_real_parameters": ROUND1_NEW_REAL_PARAMETERS,
        },
        "pipeline": [{"kind": "local-activate"}, {"kind": "matched-relax"}],
    }
    return store.register_action(action)


def wait_for_user_unit(unit: str, *, poll_seconds: int) -> dict[str, str]:
    """Wait for one registered user service to exit successfully, or fail closed."""

    if not _USER_UNIT.fullmatch(unit):
        raise AutoResearchError("wait-for-user-unit is malformed")
    if poll_seconds <= 0:
        raise AutoResearchError("wait-poll-seconds must be positive")
    properties = (
        "LoadState",
        "ActiveState",
        "SubState",
        "Result",
        "ExecMainStatus",
    )
    while True:
        completed = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                unit,
                *[f"--property={name}" for name in properties],
                "--no-pager",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise AutoResearchError(
                f"cannot inspect prerequisite user service {unit}"
            )
        state = {}
        for line in completed.stdout.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                state[key] = value
        if set(state) != set(properties) or state["LoadState"] != "loaded":
            raise AutoResearchError(
                f"prerequisite user service is not durably inspectable: {unit}"
            )
        active = state["ActiveState"]
        if active in {"active", "activating", "reloading", "deactivating"}:
            print(
                f"waiting for prerequisite {unit}: {active}/{state['SubState']}",
                flush=True,
            )
            time.sleep(poll_seconds)
            continue
        if (
            active == "inactive"
            and state["Result"] == "success"
            and state["ExecMainStatus"] == "0"
        ):
            print(f"prerequisite completed successfully: {unit}", flush=True)
            return state
        raise AutoResearchError(
            f"prerequisite user service did not succeed: {unit} ({state})"
        )


def _require_within(path: Path, root: Path, *, name: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise AutoResearchError(f"{name} escapes the prerequisite run root") from error
    return resolved


def _json_field(value, dotted_path: str):
    components = dotted_path.split(".")
    missing = object()

    def descend(current, offset: int):
        if offset == len(components):
            return current
        if isinstance(current, list):
            try:
                index = int(components[offset])
            except ValueError:
                return missing
            if 0 <= index < len(current):
                return descend(current[index], offset + 1)
            return missing
        if isinstance(current, dict):
            for end in range(offset + 1, len(components) + 1):
                key = ".".join(components[offset:end])
                if key in current:
                    resolved = descend(current[key], end)
                    if resolved is not missing:
                        return resolved
        return missing

    result = descend(value, 0)
    if result is missing:
        raise AutoResearchError(f"prerequisite JSON gate field is missing: {dotted_path}")
    return result


def _verify_gate(
    gate: object,
    *,
    run_root: Path,
    job_id: str,
    output_paths: set[Path],
) -> None:
    required = {"path", "field", "comparison", "target", "observed", "passed"}
    if not isinstance(gate, dict) or set(gate) != required:
        raise AutoResearchError(
            f"prerequisite workflow JSON gate is malformed: {job_id}"
        )
    path = _require_within(
        Path(str(gate["path"])),
        run_root,
        name=f"prerequisite JSON gate {job_id}",
    )
    if not path.is_file():
        raise AutoResearchError(
            f"prerequisite workflow JSON gate file is missing: {job_id}"
        )
    if path not in output_paths:
        raise AutoResearchError(
            f"prerequisite workflow JSON gate is not a hashed output: {job_id}"
        )
    field = str(gate["field"])
    comparison = str(gate["comparison"])
    if comparison not in {"equals", "gt", "ge", "lt", "le"}:
        raise AutoResearchError(
            f"prerequisite workflow JSON comparison is invalid: {job_id}"
        )
    observed = _json_field(_read_json(path), field)
    target = gate["target"]
    try:
        if comparison == "equals":
            passed = observed == target
        elif comparison == "gt":
            passed = observed > target
        elif comparison == "ge":
            passed = observed >= target
        elif comparison == "lt":
            passed = observed < target
        else:
            passed = observed <= target
    except TypeError as error:
        raise AutoResearchError(
            f"prerequisite workflow JSON comparison is malformed: {job_id}"
        ) from error
    if gate["passed"] is not True or observed != gate["observed"] or not passed:
        raise AutoResearchError(
            f"prerequisite workflow JSON gate no longer passes: {job_id}/{field}"
        )


def _verify_completed_workflow(
    run_root: Path,
    *,
    phases: tuple[str, ...],
    expected_campaign_id: str,
    expected_plan_sha256: str,
    expected_job_count: int,
    expected_output_count: int,
    expected_gate_count: int,
    expected_gpu_slot: str,
) -> dict[str, object]:
    workflow_root = run_root / ".workflow"
    locked_plan = _read_object(workflow_root / "plan.lock.json")
    if locked_plan.get("schema") != "gcicy-experiment-plan-lock-v1":
        raise AutoResearchError("prerequisite workflow plan-lock schema is invalid")
    plan_sha256 = str(locked_plan.get("plan_sha256", ""))
    if not _SHA256.fullmatch(plan_sha256):
        raise AutoResearchError("prerequisite workflow plan hash is malformed")
    jobs = locked_plan.get("jobs")
    campaign_id = str(locked_plan.get("campaign_id", ""))
    if not _WORKFLOW_JOB.fullmatch(campaign_id):
        raise AutoResearchError("prerequisite workflow campaign id is invalid")
    if not isinstance(jobs, list) or not jobs:
        raise AutoResearchError("prerequisite workflow has no locked jobs")
    if campaign_id != expected_campaign_id or plan_sha256 != expected_plan_sha256:
        raise AutoResearchError("prerequisite workflow identity differs from expected")
    manifest_sha256 = str(locked_plan.get("manifest_sha256", ""))
    if not _SHA256.fullmatch(manifest_sha256):
        raise AutoResearchError("prerequisite workflow manifest hash is malformed")
    sentinel = _read_object(run_root / ".gcicy-experiment-root")
    if sentinel != {
        "schema": "gcicy-experiment-root-v1",
        "campaign_id": campaign_id,
        "plan_sha256": plan_sha256,
        "manifest_sha256": manifest_sha256,
    }:
        raise AutoResearchError("prerequisite workflow root sentinel is invalid")
    runner_identity = _read_object(workflow_root / "runner.lock")
    if (
        runner_identity.get("campaign_id") != campaign_id
        or runner_identity.get("plan_sha256") != plan_sha256
    ):
        raise AutoResearchError("prerequisite runner-lock identity is invalid")
    cuda_identity = _read_object(workflow_root / "cuda_identity.json")
    if (
        cuda_identity.get("schema") != "gcicy-experiment-cuda-identity-v1"
        or cuda_identity.get("campaign_id") != campaign_id
        or cuda_identity.get("plan_sha256") != plan_sha256
        or cuda_identity.get("gpu_slot") != expected_gpu_slot
    ):
        raise AutoResearchError("prerequisite CUDA identity is invalid")
    by_id = {}
    locked_phases = set()
    for row in jobs:
        if not isinstance(row, dict):
            raise AutoResearchError("prerequisite workflow job row is malformed")
        job_id = str(row.get("id", ""))
        if not _WORKFLOW_JOB.fullmatch(job_id) or job_id in by_id:
            raise AutoResearchError("prerequisite workflow job id is invalid")
        needs = row.get("needs")
        if not isinstance(needs, list) or not all(
            isinstance(value, str) and _WORKFLOW_JOB.fullmatch(value)
            for value in needs
        ):
            raise AutoResearchError(f"prerequisite workflow needs are invalid: {job_id}")
        if not _SHA256.fullmatch(str(row.get("digest", ""))):
            raise AutoResearchError(f"prerequisite workflow digest is invalid: {job_id}")
        phase = str(row.get("phase", ""))
        if not _WORKFLOW_JOB.fullmatch(phase):
            raise AutoResearchError(f"prerequisite workflow phase is invalid: {job_id}")
        locked_phases.add(phase)
        by_id[job_id] = row
    unknown_phases = sorted(set(phases) - locked_phases)
    if unknown_phases:
        raise AutoResearchError(
            f"prerequisite workflow phases are unknown: {unknown_phases}"
        )
    selected = {
        job_id for job_id, row in by_id.items() if str(row.get("phase")) in phases
    }
    if not selected:
        raise AutoResearchError("prerequisite workflow phases select no jobs")
    stack = list(selected)
    while stack:
        job_id = stack.pop()
        for dependency in by_id[job_id]["needs"]:
            if dependency not in by_id:
                raise AutoResearchError(
                    f"prerequisite workflow dependency is missing: {dependency}"
                )
            if dependency not in selected:
                selected.add(dependency)
                stack.append(dependency)
    if len(selected) != expected_job_count:
        raise AutoResearchError(
            "prerequisite workflow selected closure has an unexpected job count"
        )
    verified_outputs = 0
    verified_gates = 0
    all_output_paths: set[Path] = set()
    for job_id in sorted(selected):
        job = by_id[job_id]
        state = _read_object(workflow_root / "jobs" / f"{job_id}.json")
        expected = {
            "schema": "gcicy-experiment-job-state-v1",
            "campaign_id": campaign_id,
            "plan_sha256": plan_sha256,
            "job_id": job_id,
            "job_digest": job["digest"],
            "phase": job["phase"],
            "status": "succeeded",
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise AutoResearchError(
                    f"prerequisite workflow job is not sealed-success: {job_id}/{key}"
                )
        outputs = state.get("outputs")
        gates = state.get("json_gates")
        if not isinstance(outputs, list) or not isinstance(gates, list):
            raise AutoResearchError(
                f"prerequisite workflow job evidence is malformed: {job_id}"
            )
        if not outputs:
            raise AutoResearchError(
                f"prerequisite workflow job has no sealed outputs: {job_id}"
            )
        output_rows = []
        job_output_paths: set[Path] = set()
        for output in outputs:
            if not isinstance(output, dict):
                raise AutoResearchError(
                    f"prerequisite workflow output row is malformed: {job_id}"
                )
            path = _require_within(
                Path(str(output.get("path", ""))),
                run_root,
                name=f"prerequisite output {job_id}",
            )
            try:
                expected_bytes = int(output.get("bytes", -1))
            except (TypeError, ValueError) as error:
                raise AutoResearchError(
                    f"prerequisite workflow output size is malformed: {job_id}"
                ) from error
            if path in job_output_paths or path in all_output_paths:
                raise AutoResearchError(
                    f"prerequisite workflow output path is repeated: {job_id}"
                )
            job_output_paths.add(path)
            all_output_paths.add(path)
            output_rows.append((path, expected_bytes, output.get("sha256")))
        for gate in gates:
            _verify_gate(
                gate,
                run_root=run_root,
                job_id=job_id,
                output_paths=job_output_paths,
            )
            verified_gates += 1
        for path, expected_bytes, expected_sha256 in output_rows:
            if (
                not path.is_file()
                or path.stat().st_size != expected_bytes
                or sha256_file(path) != expected_sha256
            ):
                raise AutoResearchError(
                    f"prerequisite workflow output hash is invalid: {job_id}"
                )
            verified_outputs += 1
    if (
        verified_outputs != expected_output_count
        or verified_gates != expected_gate_count
    ):
        raise AutoResearchError(
            "prerequisite workflow output/gate totals differ from expected"
        )
    return {
        "campaign_id": campaign_id,
        "plan_sha256": plan_sha256,
        "phases": list(phases),
        "verified_jobs": len(selected),
        "verified_outputs": verified_outputs,
        "verified_gates": verified_gates,
    }


@contextmanager
def completed_workflow_guard(
    run_root: Path,
    *,
    phases: tuple[str, ...],
    poll_seconds: int,
    expected_campaign_id: str,
    expected_plan_sha256: str,
    expected_job_count: int,
    expected_output_count: int,
    expected_gate_count: int,
    expected_gpu_slot: str,
):
    """Wait on a durable campaign lock, verify its DAG, and prevent restart."""

    run_root = run_root.expanduser().resolve()
    if "gcicy_metric_k2_20260710" in str(run_root):
        raise AutoResearchError("prerequisite workflow may not be a frozen root")
    if not phases or len(set(phases)) != len(phases):
        raise AutoResearchError("prerequisite workflow phases must be distinct")
    if any(not _WORKFLOW_JOB.fullmatch(phase) for phase in phases):
        raise AutoResearchError("prerequisite workflow phase is malformed")
    if not _WORKFLOW_JOB.fullmatch(expected_campaign_id):
        raise AutoResearchError("expected prerequisite campaign id is malformed")
    if not _SHA256.fullmatch(expected_plan_sha256):
        raise AutoResearchError("expected prerequisite plan hash is malformed")
    if (
        expected_job_count <= 0
        or expected_output_count <= 0
        or expected_gate_count <= 0
        or not expected_gpu_slot.isdigit()
    ):
        raise AutoResearchError(
            "expected prerequisite evidence counts or GPU slot are invalid"
        )
    if poll_seconds <= 0:
        raise AutoResearchError("wait-poll-seconds must be positive")
    lock_path = run_root / ".workflow" / "runner.lock"
    if not lock_path.is_file():
        raise AutoResearchError("prerequisite workflow runner lock is missing")
    handle = lock_path.open("r+", encoding="utf-8")
    try:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                print(
                    f"waiting for prerequisite workflow lock: {run_root}",
                    flush=True,
                )
                time.sleep(poll_seconds)
        report = _verify_completed_workflow(
            run_root,
            phases=phases,
            expected_campaign_id=expected_campaign_id,
            expected_plan_sha256=expected_plan_sha256,
            expected_job_count=expected_job_count,
            expected_output_count=expected_output_count,
            expected_gate_count=expected_gate_count,
            expected_gpu_slot=expected_gpu_slot,
        )
        print(
            "prerequisite workflow completed and verified: "
            f"{report['verified_jobs']} jobs, {report['verified_outputs']} outputs, "
            f"{report['verified_gates']} gates",
            flush=True,
        )
        yield report
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


@contextmanager
def exclusive_gpu_lock(path: Path, *, gpu_slot: str, poll_seconds: int):
    """Hold the same advisory GPU lock used by the preceding workflow."""

    path = path.expanduser().resolve()
    expected_parent = Path("/tmp/gcicy-tn-gpu-locks").resolve()
    if (
        not path.is_file()
        or path.parent != expected_parent
        or not _GPU_LOCK.fullmatch(path.name)
        or path.name != f"gcicy-tn-gpu-{gpu_slot}.lock"
        or poll_seconds <= 0
    ):
        raise AutoResearchError("registered GPU lock is missing or wait is invalid")
    handle = path.open("r+", encoding="utf-8")
    try:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                print(f"waiting for registered GPU lock: {path}", flush=True)
                time.sleep(poll_seconds)
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


@contextmanager
def round1_execution_lock(run_root: Path, *, poll_seconds: int):
    """Serialize all execute calls for one Round 1 campaign."""

    run_root = run_root.expanduser().resolve()
    if not run_root.is_dir() or poll_seconds <= 0:
        raise AutoResearchError("Round 1 campaign root or lock wait is invalid")
    path = run_root / ".round1-execute.lock"
    handle = path.open("a+", encoding="utf-8")
    try:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                print(f"waiting for Round 1 execution lock: {path}", flush=True)
                time.sleep(poll_seconds)
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def require_cuda_idle() -> None:
    """Fail before worker launch if another compute process still owns the GPU."""

    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise AutoResearchError("cannot verify that the CUDA GPU is idle")
    processes = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
    if processes:
        raise AutoResearchError(
            f"refusing to share the CUDA GPU with existing processes: {processes}"
        )
    print("CUDA GPU is idle; starting Round 1 workers", flush=True)


def execute_round1(args: argparse.Namespace) -> dict[str, object]:
    """Idempotently drive the fixed campaign through Round-1 adjudication."""

    store = CampaignStore.initialize(
        args.campaign_run_root,
        _read_object(args.protocol),
    )
    with round1_execution_lock(
        args.campaign_run_root,
        poll_seconds=args.wait_poll_seconds,
    ):
        action = _ensure_registered_action(store, candidate_id=args.candidate_id)
        ledger = store.status()
        candidate = ledger["rounds"]["1"]["candidates"][args.candidate_id]
        if candidate.get("evidence_sha256") is None:
            workflow_expectations = (
                args.expected_workflow_campaign_id,
                args.expected_workflow_plan_sha256,
                args.expected_workflow_job_count,
                args.expected_workflow_output_count,
                args.expected_workflow_gate_count,
                args.expected_workflow_gpu_slot,
            )
            if args.wait_for_user_unit is not None and args.wait_for_workflow_run_root:
                raise AutoResearchError("choose one prerequisite waiting mechanism")
            if args.wait_for_workflow_run_root is None and args.wait_for_workflow_phase:
                raise AutoResearchError("workflow phases require a workflow run root")
            if (
                args.wait_for_workflow_run_root is not None
                and not args.wait_for_workflow_phase
            ):
                raise AutoResearchError("workflow run root requires at least one phase")
            if args.wait_for_workflow_run_root is not None and any(
                value is None for value in workflow_expectations
            ):
                raise AutoResearchError(
                    "workflow handoff requires exact campaign, plan, evidence counts, and GPU identity"
                )
            if args.wait_for_workflow_run_root is None and any(
                value is not None for value in workflow_expectations
            ):
                raise AutoResearchError(
                    "workflow identity arguments require a workflow run root"
                )
            if args.gpu_lock_file is not None and args.expected_workflow_gpu_slot is None:
                raise AutoResearchError("GPU lock requires an expected GPU slot")
            if (
                args.device == "cuda"
                and args.wait_for_workflow_run_root is not None
                and args.gpu_lock_file is None
            ):
                raise AutoResearchError("workflow GPU handoff requires --gpu-lock-file")
            if (
                args.wait_for_workflow_run_root is not None
                and args.gpu_lock_file is not None
            ):
                runner_lock = (
                    args.wait_for_workflow_run_root.expanduser().resolve()
                    / ".workflow"
                    / "runner.lock"
                )
                gpu_lock = args.gpu_lock_file.expanduser().resolve()
                if not runner_lock.is_file() or not gpu_lock.is_file():
                    raise AutoResearchError("workflow or GPU handoff lock is missing")
                if runner_lock.samefile(gpu_lock):
                    raise AutoResearchError(
                        "workflow runner lock and GPU lock must be different files"
                    )
            prepare_round1_bridge(
                campaign_run_root=args.campaign_run_root,
                candidate_id=args.candidate_id,
                output_root=args.output_root,
                baseline_checkpoints=_baselines(args),
                runtime=_runtime(args),
                repository_root=ROOT,
            )
            workflow_guard = (
                completed_workflow_guard(
                    args.wait_for_workflow_run_root,
                    phases=tuple(args.wait_for_workflow_phase),
                    poll_seconds=args.wait_poll_seconds,
                    expected_campaign_id=args.expected_workflow_campaign_id,
                    expected_plan_sha256=args.expected_workflow_plan_sha256,
                    expected_job_count=args.expected_workflow_job_count,
                    expected_output_count=args.expected_workflow_output_count,
                    expected_gate_count=args.expected_workflow_gate_count,
                    expected_gpu_slot=args.expected_workflow_gpu_slot,
                )
                if args.wait_for_workflow_run_root is not None
                else nullcontext(None)
            )
            with workflow_guard:
                if args.wait_for_user_unit is not None:
                    wait_for_user_unit(
                        args.wait_for_user_unit,
                        poll_seconds=args.wait_poll_seconds,
                    )
                gpu_guard = (
                    exclusive_gpu_lock(
                        args.gpu_lock_file,
                        gpu_slot=args.expected_workflow_gpu_slot,
                        poll_seconds=args.wait_poll_seconds,
                    )
                    if args.gpu_lock_file is not None
                    else nullcontext()
                )
                with gpu_guard:
                    latest = store.status()["rounds"]["1"]["candidates"][
                        args.candidate_id
                    ]
                    if latest.get("evidence_sha256") is None:
                        if args.device == "cuda":
                            require_cuda_idle()
                        run_round1_bridge(args.output_root)
                        evidence = normalize_round1_bridge(args.output_root)
                        store.record_search_evidence(evidence)
        adjudication = store.adjudicate_round(1)
        result = {
            "action": action,
            "adjudication": adjudication,
            "ledger": store.status(),
            "baseline_sha256": ROUND1_BASELINE_SHA256,
        }
    return result


def _add_execution_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--campaign-run-root", type=Path, required=True)
    parser.add_argument("--candidate-id", default=DEFAULT_CANDIDATE_ID)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--baseline-checkpoint",
        type=_baseline,
        action="append",
        required=True,
        metavar="SEED=PATH",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--train-chunk-size", type=int, default=512)
    parser.add_argument("--feature-batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--early-stopping-evaluations", type=int, default=6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    _add_execution_arguments(prepare)

    execute = commands.add_parser(
        "execute",
        help=(
            "Initialize/register, run all three seeds, normalize evidence, "
            "and adjudicate Round 1 idempotently."
        ),
    )
    execute.add_argument("--protocol", type=Path, required=True)
    execute.add_argument("--wait-for-user-unit")
    execute.add_argument("--wait-for-workflow-run-root", type=Path)
    execute.add_argument("--wait-for-workflow-phase", action="append", default=[])
    execute.add_argument("--expected-workflow-campaign-id")
    execute.add_argument("--expected-workflow-plan-sha256")
    execute.add_argument("--expected-workflow-job-count", type=int)
    execute.add_argument("--expected-workflow-output-count", type=int)
    execute.add_argument("--expected-workflow-gate-count", type=int)
    execute.add_argument("--expected-workflow-gpu-slot")
    execute.add_argument("--gpu-lock-file", type=Path)
    execute.add_argument("--wait-poll-seconds", type=int, default=30)
    _add_execution_arguments(execute)

    run = commands.add_parser("run")
    run.add_argument("--bridge-root", type=Path, required=True)
    run.add_argument("--seed", type=int)

    normalize = commands.add_parser("normalize")
    normalize.add_argument("--bridge-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        result = prepare_round1_bridge(
            campaign_run_root=args.campaign_run_root,
            candidate_id=args.candidate_id,
            output_root=args.output_root,
            baseline_checkpoints=_baselines(args),
            runtime=_runtime(args),
            repository_root=ROOT,
        )
    elif args.command == "execute":
        result = execute_round1(args)
    elif args.command == "run":
        result = run_round1_bridge(args.bridge_root, selected_seed=args.seed)
    elif args.command == "normalize":
        result = normalize_round1_bridge(args.bridge_root)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except AutoResearchError as error:
        print(f"round1 bridge error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
