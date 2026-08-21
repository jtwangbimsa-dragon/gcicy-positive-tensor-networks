"""Durable, manifest-driven execution for post-release gCICY TN experiments.

The runner in this module is deliberately separate from the numerical training
code.  It treats released artifacts as immutable inputs, constrains every
declared output to a new campaign root, records content hashes, and maintains a
small resumable job state machine.  It uses only the Python standard library so
that validating a campaign does not require a CUDA installation.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import re
import signal
import socket
import subprocess
import sys
import time
from typing import Any, Iterator, Mapping, Sequence


WORKFLOW_SCHEMA = "gcicy-tn-gpu-workflow-v1"
ROOT_SENTINEL = ".gcicy-experiment-root"
VARIABLE_PATTERN = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")
MATRIX_PATTERN = re.compile(r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}")
JOB_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
OUTPUT_FLAGS = {
    "--artifact-out",
    "--arrays-out",
    "--checkpoint",
    "--out",
    "--run-dir",
    "--stage-dir",
    "--summary",
    "--summary-out",
}


class WorkflowError(RuntimeError):
    """Raised when a campaign is unsafe, inconsistent, or incomplete."""


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def digest_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    try:
        directory_fd = os.open(path.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve(strict=False)


def _expand_string(value: str, variables: Mapping[str, str]) -> str:
    previous = None
    expanded = value
    for _ in range(20):
        if expanded == previous:
            break
        previous = expanded

        def replacement(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in variables:
                raise WorkflowError(f"undefined workflow variable: {key}")
            return str(variables[key])

        expanded = VARIABLE_PATTERN.sub(replacement, expanded)
    unresolved = VARIABLE_PATTERN.findall(expanded)
    if unresolved:
        raise WorkflowError(f"recursive or unresolved variables: {unresolved}")
    return expanded


def expand_variables(value: Any, variables: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        return _expand_string(value, variables)
    if isinstance(value, list):
        return [expand_variables(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: expand_variables(item, variables) for key, item in value.items()}
    return value


def _render_matrix_string(value: str, row: Mapping[str, Any]) -> str:
    def replacement(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in row:
            raise WorkflowError(f"unknown matrix axis in template: {key}")
        return str(row[key])

    return MATRIX_PATTERN.sub(replacement, value)


def _render_matrix(value: Any, row: Mapping[str, Any]) -> Any:
    if isinstance(value, str):
        return _render_matrix_string(value, row)
    if isinstance(value, list):
        return [_render_matrix(item, row) for item in value]
    if isinstance(value, dict):
        return {key: _render_matrix(item, row) for key, item in value.items()}
    return value


def expand_job_matrices(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for raw in rows:
        template = dict(raw)
        matrix = template.pop("matrix", None)
        if matrix is None:
            expanded.append(template)
            continue
        if not isinstance(matrix, dict) or not matrix:
            raise WorkflowError("job matrix must be a non-empty object")
        axes = list(matrix)
        values = []
        for axis in axes:
            entries = matrix[axis]
            if not isinstance(entries, list) or not entries:
                raise WorkflowError(f"matrix axis {axis} must be a non-empty list")
            if any(isinstance(entry, (dict, list)) for entry in entries):
                raise WorkflowError(f"matrix axis {axis} values must be scalar")
            values.append(entries)
        for combination in itertools.product(*values):
            factors = dict(zip(axes, combination, strict=True))
            job = _render_matrix(template, factors)
            scientific = dict(job.get("scientific", {}))
            recorded_factors = dict(scientific.get("factors", {}))
            recorded_factors.update(factors)
            scientific["factors"] = recorded_factors
            job["scientific"] = scientific
            expanded.append(job)
    return expanded


@dataclass(frozen=True)
class FrozenInput:
    role: str
    path: Path
    sha256: str
    bytes: int | None


@dataclass(frozen=True)
class ExpectedOutput:
    path: Path
    allow_empty: bool = False


@dataclass(frozen=True)
class Job:
    id: str
    phase: str
    needs: tuple[str, ...]
    command: tuple[str, ...]
    working_directory: Path
    environment: Mapping[str, str]
    gpu: bool
    expected_outputs: tuple[ExpectedOutput, ...]
    json_gates: tuple[Mapping[str, Any], ...]
    retry: Mapping[str, Any]
    resume: Mapping[str, Any] | None
    scientific: Mapping[str, Any]
    result: Mapping[str, Any] | None
    digest: str


@dataclass(frozen=True)
class WorkflowPlan:
    manifest_path: Path
    manifest_sha256: str
    campaign_id: str
    repo_root: Path
    frozen_root: Path
    run_root: Path
    protected_paths: tuple[Path, ...]
    frozen_inputs: tuple[FrozenInput, ...]
    jobs: tuple[Job, ...]
    phases: tuple[str, ...]
    source_guard: Mapping[str, Any]
    environment_requirements: Mapping[str, Any]
    raw_expanded_manifest: Mapping[str, Any]
    digest: str

    @property
    def jobs_by_id(self) -> dict[str, Job]:
        return {job.id: job for job in self.jobs}


def _topological_order(jobs: Sequence[Job]) -> tuple[Job, ...]:
    by_id = {job.id: job for job in jobs}
    indegree = {job.id: 0 for job in jobs}
    children = {job.id: [] for job in jobs}
    for job in jobs:
        for dependency in job.needs:
            if dependency not in by_id:
                raise WorkflowError(f"job {job.id} depends on unknown job {dependency}")
            indegree[job.id] += 1
            children[dependency].append(job.id)
    ready = sorted(job_id for job_id, degree in indegree.items() if degree == 0)
    ordered: list[Job] = []
    while ready:
        job_id = ready.pop(0)
        ordered.append(by_id[job_id])
        for child in sorted(children[job_id]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort()
    if len(ordered) != len(jobs):
        cycle_members = sorted(job_id for job_id, degree in indegree.items() if degree)
        raise WorkflowError(f"workflow dependency cycle: {cycle_members}")
    return tuple(ordered)


def _parse_expected_outputs(rows: Any, *, run_root: Path) -> tuple[ExpectedOutput, ...]:
    if not isinstance(rows, list):
        raise WorkflowError("expected_outputs must be a list")
    outputs = []
    for raw in rows:
        if isinstance(raw, str):
            path_value = raw
            allow_empty = False
        elif isinstance(raw, dict) and set(raw).issubset({"path", "allow_empty"}):
            if "path" not in raw:
                raise WorkflowError("expected output object requires path")
            path_value = str(raw["path"])
            allow_empty = bool(raw.get("allow_empty", False))
        else:
            raise WorkflowError("expected output must be a path or path object")
        path = _resolve_path(path_value, run_root)
        if not _is_within(path, run_root) or path == run_root:
            raise WorkflowError(f"declared output escapes RUN_ROOT: {path}")
        outputs.append(ExpectedOutput(path=path, allow_empty=allow_empty))
    return tuple(outputs)


def _validate_output_flags(command: Sequence[str], run_root: Path) -> None:
    for index, token in enumerate(command):
        flag = token.split("=", 1)[0]
        if flag not in OUTPUT_FLAGS:
            continue
        if "=" in token:
            path_text = token.split("=", 1)[1]
        else:
            if index + 1 >= len(command):
                raise WorkflowError(f"output flag has no value: {token}")
            path_text = command[index + 1]
        path = _resolve_path(path_text, run_root)
        if not _is_within(path, run_root) or path == run_root:
            raise WorkflowError(f"command output flag escapes RUN_ROOT: {flag} {path}")


def load_workflow_plan(
    manifest_path: Path,
    *,
    repo_root: Path,
    frozen_root: Path,
    run_root: Path,
    variable_overrides: Mapping[str, str] | None = None,
    python_executable: str | None = None,
) -> WorkflowPlan:
    manifest_path = manifest_path.expanduser().resolve()
    repo_root = repo_root.expanduser().resolve()
    frozen_root = frozen_root.expanduser().resolve()
    run_root = run_root.expanduser().resolve(strict=False)
    for label, immutable_root in (
        ("REPO_ROOT", repo_root),
        ("FROZEN_ROOT", frozen_root),
    ):
        if _is_within(run_root, immutable_root) or _is_within(immutable_root, run_root):
            raise WorkflowError(
                f"RUN_ROOT must be disjoint from {label}: "
                f"{run_root} / {immutable_root}"
            )
    raw_bytes = manifest_path.read_bytes()
    try:
        raw = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        raise WorkflowError(f"invalid workflow JSON: {exc}") from exc
    if raw.get("schema") != WORKFLOW_SCHEMA:
        raise WorkflowError(f"workflow schema must be {WORKFLOW_SCHEMA}")
    campaign_id = str(raw.get("campaign_id", ""))
    if not JOB_ID_PATTERN.fullmatch(campaign_id):
        raise WorkflowError("campaign_id must use lowercase job-id characters")

    variables: dict[str, str] = {
        "REPO_ROOT": str(repo_root),
        "FROZEN_ROOT": str(frozen_root),
        "RUN_ROOT": str(run_root),
        "PYTHON": python_executable or sys.executable,
    }
    overrides = {
        str(key): str(value) for key, value in (variable_overrides or {}).items()
    }
    variables.update(overrides)
    configured_variables = raw.get("variables", {})
    if not isinstance(configured_variables, dict):
        raise WorkflowError("variables must be an object")
    for key, value in configured_variables.items():
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", str(key)):
            raise WorkflowError(f"invalid variable name: {key}")
        if str(key) not in overrides:
            variables[str(key)] = _expand_string(str(value), variables)

    expanded = expand_variables(raw, variables)
    protected_rows = expanded.get("protected_paths", [])
    if not isinstance(protected_rows, list) or not protected_rows:
        raise WorkflowError("protected_paths must be a non-empty list")
    protected_paths = tuple(
        _resolve_path(str(value), frozen_root) for value in protected_rows
    )
    for protected in protected_paths:
        if _is_within(run_root, protected) or _is_within(protected, run_root):
            raise WorkflowError(
                f"RUN_ROOT and protected path overlap: {run_root} / {protected}"
            )

    frozen_rows = expanded.get("frozen_inputs", [])
    if not isinstance(frozen_rows, list) or not frozen_rows:
        raise WorkflowError("frozen_inputs must be a non-empty list")
    frozen_inputs = []
    roles = set()
    for row in frozen_rows:
        if not isinstance(row, dict):
            raise WorkflowError("frozen input entries must be objects")
        role = str(row.get("role", ""))
        if not role or role in roles:
            raise WorkflowError(f"duplicate or empty frozen input role: {role}")
        roles.add(role)
        expected_sha = str(row.get("sha256", "")).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
            raise WorkflowError(f"frozen input {role} has invalid sha256")
        frozen_inputs.append(
            FrozenInput(
                role=role,
                path=_resolve_path(str(row["path"]), frozen_root),
                sha256=expected_sha,
                bytes=None if row.get("bytes") is None else int(row["bytes"]),
            )
        )

    job_rows = expanded.get("jobs", [])
    template_rows = expanded.get("job_templates", [])
    if not isinstance(job_rows, list) or not isinstance(template_rows, list):
        raise WorkflowError("jobs and job_templates must be lists")
    concrete_rows = [*job_rows, *expand_job_matrices(template_rows)]
    if not concrete_rows:
        raise WorkflowError("workflow must contain at least one job")
    defaults = expanded.get("defaults", {})
    if not isinstance(defaults, dict):
        raise WorkflowError("defaults must be an object")
    default_cwd = _resolve_path(
        str(defaults.get("working_directory", repo_root)),
        repo_root,
    )
    default_environment = defaults.get("environment", {})
    default_retry = defaults.get("retry", {})
    if not isinstance(default_environment, dict) or not isinstance(default_retry, dict):
        raise WorkflowError("default environment and retry must be objects")

    jobs = []
    seen_ids = set()
    output_owners: dict[Path, str] = {}
    for row in concrete_rows:
        if not isinstance(row, dict):
            raise WorkflowError("job entries must be objects")
        job_id = str(row.get("id", ""))
        if not JOB_ID_PATTERN.fullmatch(job_id) or job_id in seen_ids:
            raise WorkflowError(f"duplicate or invalid job id: {job_id}")
        seen_ids.add(job_id)
        phase = str(row.get("phase", "default"))
        command = row.get("command")
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(token, str) and token for token in command)
        ):
            raise WorkflowError(f"job {job_id} command must be a string list")
        command_tuple = tuple(command)
        _validate_output_flags(command_tuple, run_root)
        needs = row.get("needs", [])
        if not isinstance(needs, list) or not all(
            isinstance(item, str) for item in needs
        ):
            raise WorkflowError(f"job {job_id} needs must be a string list")
        cwd = _resolve_path(str(row.get("working_directory", default_cwd)), repo_root)
        if not (_is_within(cwd, repo_root) or _is_within(cwd, run_root)):
            raise WorkflowError(
                f"job {job_id} working directory must be in REPO_ROOT or RUN_ROOT"
            )
        if any(_is_within(cwd, protected) for protected in protected_paths):
            raise WorkflowError(f"job {job_id} working directory is protected: {cwd}")
        environment = {
            str(key): str(value) for key, value in default_environment.items()
        }
        configured_env = row.get("environment", {})
        if not isinstance(configured_env, dict):
            raise WorkflowError(f"job {job_id} environment must be an object")
        environment.update(
            {str(key): str(value) for key, value in configured_env.items()}
        )
        outputs = _parse_expected_outputs(
            row.get("expected_outputs", []), run_root=run_root
        )
        for output in outputs:
            if output.path in output_owners:
                raise WorkflowError(
                    f"jobs {output_owners[output.path]} and {job_id} share output {output.path}"
                )
            output_owners[output.path] = job_id
        retry = dict(default_retry)
        configured_retry = row.get("retry", {})
        if not isinstance(configured_retry, dict):
            raise WorkflowError(f"job {job_id} retry must be an object")
        retry.update(configured_retry)
        max_attempts = int(retry.get("max_attempts", 1))
        if max_attempts <= 0:
            raise WorkflowError(f"job {job_id} max_attempts must be positive")
        retry["max_attempts"] = max_attempts
        resume = row.get("resume")
        if resume is not None:
            if not isinstance(resume, dict) or "checkpoint" not in resume:
                raise WorkflowError(f"job {job_id} resume requires checkpoint")
            resume = dict(resume)
            checkpoint = _resolve_path(str(resume["checkpoint"]), run_root)
            if not _is_within(checkpoint, run_root):
                raise WorkflowError(f"job {job_id} checkpoint escapes RUN_ROOT")
            resume["checkpoint"] = str(checkpoint)
            resume.setdefault("flag", "--resume-checkpoint")
        json_gates = row.get("json_gates", [])
        if not isinstance(json_gates, list):
            raise WorkflowError(f"job {job_id} json_gates must be a list")
        normalized_gates = []
        for gate in json_gates:
            if not isinstance(gate, dict) or "path" not in gate:
                raise WorkflowError(f"job {job_id} JSON gate requires path")
            normalized_gate = dict(gate)
            gate_path = _resolve_path(str(gate["path"]), run_root)
            if not _is_within(gate_path, run_root):
                raise WorkflowError(f"job {job_id} JSON gate escapes RUN_ROOT")
            normalized_gate["path"] = str(gate_path)
            normalized_gates.append(normalized_gate)
        scientific = row.get("scientific", {})
        result = row.get("result")
        if not isinstance(scientific, dict) or (
            result is not None and not isinstance(result, dict)
        ):
            raise WorkflowError(f"job {job_id} scientific/result metadata is invalid")
        if result is not None:
            result = dict(result)
            if "path" not in result:
                raise WorkflowError(f"job {job_id} result requires path")
            result_path = _resolve_path(str(result["path"]), run_root)
            if not _is_within(result_path, run_root):
                raise WorkflowError(f"job {job_id} result path escapes RUN_ROOT")
            result["path"] = str(result_path)
        resources = row.get("resources", {})
        if not isinstance(resources, dict):
            raise WorkflowError(f"job {job_id} resources must be an object")
        normalized = {
            "id": job_id,
            "phase": phase,
            "needs": list(needs),
            "command": list(command_tuple),
            "working_directory": str(cwd),
            "environment": environment,
            "gpu": bool(resources.get("gpu", False)),
            "expected_outputs": [
                {"path": str(item.path), "allow_empty": item.allow_empty}
                for item in outputs
            ],
            "json_gates": normalized_gates,
            "retry": retry,
            "resume": resume,
            "scientific": scientific,
            "result": result,
        }
        jobs.append(
            Job(
                id=job_id,
                phase=phase,
                needs=tuple(needs),
                command=command_tuple,
                working_directory=cwd,
                environment=environment,
                gpu=normalized["gpu"],
                expected_outputs=outputs,
                json_gates=tuple(normalized_gates),
                retry=retry,
                resume=resume,
                scientific=scientific,
                result=result,
                digest=digest_value(normalized),
            )
        )

    ordered_jobs = _topological_order(jobs)
    phases = tuple(dict.fromkeys(job.phase for job in ordered_jobs))
    expanded_for_digest = dict(expanded)
    expanded_for_digest["jobs"] = [
        {"id": job.id, "digest": job.digest} for job in ordered_jobs
    ]
    expanded_for_digest.pop("job_templates", None)
    plan_digest = digest_value(expanded_for_digest)
    return WorkflowPlan(
        manifest_path=manifest_path,
        manifest_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        campaign_id=campaign_id,
        repo_root=repo_root,
        frozen_root=frozen_root,
        run_root=run_root,
        protected_paths=protected_paths,
        frozen_inputs=tuple(frozen_inputs),
        jobs=ordered_jobs,
        phases=phases,
        source_guard=expanded.get("source_guard", {}),
        environment_requirements=expanded.get("environment", {}),
        raw_expanded_manifest=expanded,
        digest=plan_digest,
    )


def verify_frozen_inputs(plan: WorkflowPlan) -> list[dict[str, Any]]:
    verified = []
    for item in plan.frozen_inputs:
        if not item.path.is_file():
            raise WorkflowError(f"missing frozen input {item.role}: {item.path}")
        size = item.path.stat().st_size
        if item.bytes is not None and size != item.bytes:
            raise WorkflowError(
                f"frozen input size mismatch for {item.role}: {size} != {item.bytes}"
            )
        observed = sha256_file(item.path)
        if observed != item.sha256:
            raise WorkflowError(
                f"frozen input hash mismatch for {item.role}: {observed} != {item.sha256}"
            )
        verified.append(
            {
                "role": item.role,
                "path": str(item.path),
                "bytes": size,
                "sha256": observed,
            }
        )
    return verified


def git_identity(repo_root: Path) -> dict[str, Any]:
    def capture(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        commit = capture("rev-parse", "HEAD")
        branch = capture("branch", "--show-current")
        status_porcelain = capture("status", "--porcelain=v1")
        tracked_diff = subprocess.run(
            ["git", "diff", "--binary", "--no-ext-diff", "HEAD", "--"],
            cwd=repo_root,
            check=True,
            capture_output=True,
        ).stdout
        untracked_raw = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=repo_root,
            check=True,
            capture_output=True,
        ).stdout
        worktree_digest = hashlib.sha256()
        worktree_digest.update(len(tracked_diff).to_bytes(8, "big"))
        worktree_digest.update(tracked_diff)
        for relative_bytes in sorted(filter(None, untracked_raw.split(b"\0"))):
            relative = relative_bytes.decode("utf-8", errors="surrogateescape")
            candidate = repo_root / relative
            worktree_digest.update(len(relative_bytes).to_bytes(8, "big"))
            worktree_digest.update(relative_bytes)
            if candidate.is_symlink():
                payload = os.readlink(candidate).encode(
                    "utf-8", errors="surrogateescape"
                )
            else:
                payload = candidate.read_bytes()
            worktree_digest.update(len(payload).to_bytes(8, "big"))
            worktree_digest.update(payload)
        return {
            "commit": commit,
            "branch": branch,
            "status_porcelain": status_porcelain,
            "worktree_sha256": worktree_digest.hexdigest(),
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        raise WorkflowError(f"cannot identify experiment Git tree: {exc}") from exc


def verify_source_guard(plan: WorkflowPlan) -> dict[str, Any]:
    identity = git_identity(plan.repo_root)
    guard = plan.source_guard
    forbidden = {str(value) for value in guard.get("forbidden_branches", [])}
    if identity["branch"] in forbidden:
        raise WorkflowError(
            f"workflow may not run from protected branch {identity['branch']}"
        )
    prefix = guard.get("required_branch_prefix")
    if prefix is not None and not identity["branch"].startswith(str(prefix)):
        raise WorkflowError(
            f"branch {identity['branch']} does not start with required prefix {prefix}"
        )
    base_commit = guard.get("release_base_commit")
    if base_commit:
        completed = subprocess.run(
            ["git", "merge-base", "--is-ancestor", str(base_commit), "HEAD"],
            cwd=plan.repo_root,
            check=False,
        )
        if completed.returncode != 0:
            raise WorkflowError(
                f"release base {base_commit} is not an ancestor of HEAD"
            )
    if bool(guard.get("require_clean", False)) and identity["status_porcelain"]:
        raise WorkflowError("source guard requires a clean Git tree")
    return identity


def initialize_run_root(
    plan: WorkflowPlan,
    *,
    frozen_inputs: Sequence[Mapping[str, Any]],
    source_identity: Mapping[str, Any],
) -> None:
    plan.run_root.mkdir(parents=True, exist_ok=True)
    sentinel = plan.run_root / ROOT_SENTINEL
    identity = {
        "schema": "gcicy-experiment-root-v1",
        "campaign_id": plan.campaign_id,
        "plan_sha256": plan.digest,
        "manifest_sha256": plan.manifest_sha256,
    }
    if sentinel.exists():
        observed = json.loads(sentinel.read_text(encoding="utf-8"))
        if observed != identity:
            raise WorkflowError(
                f"RUN_ROOT belongs to a different campaign or plan: {sentinel}"
            )
    else:
        if any(plan.run_root.iterdir()):
            raise WorkflowError(
                f"refusing non-empty RUN_ROOT without {ROOT_SENTINEL}: {plan.run_root}"
            )
        atomic_write_json(sentinel, identity)
    workflow_dir = plan.run_root / ".workflow"
    workflow_dir.mkdir(parents=True, exist_ok=True)
    locked_plan = workflow_dir / "plan.lock.json"
    plan_payload = {
        "schema": "gcicy-experiment-plan-lock-v1",
        "campaign_id": plan.campaign_id,
        "plan_sha256": plan.digest,
        "manifest": str(plan.manifest_path),
        "manifest_sha256": plan.manifest_sha256,
        "source": dict(source_identity),
        "frozen_inputs": list(frozen_inputs),
        "jobs": [
            {
                "id": job.id,
                "phase": job.phase,
                "needs": list(job.needs),
                "digest": job.digest,
            }
            for job in plan.jobs
        ],
        "created_utc": utc_now(),
    }
    if locked_plan.exists():
        observed = json.loads(locked_plan.read_text(encoding="utf-8"))
        if observed.get("plan_sha256") != plan.digest:
            raise WorkflowError("locked plan digest does not match current plan")
        if observed.get("source") != dict(source_identity):
            raise WorkflowError(
                "locked source identity does not match the current Git worktree; "
                "use a new RUN_ROOT"
            )
        if observed.get("frozen_inputs") != list(frozen_inputs):
            raise WorkflowError(
                "locked frozen-input identity does not match current verification"
            )
    else:
        atomic_write_json(locked_plan, plan_payload)
    provenance = workflow_dir / "campaign_provenance.json"
    if not provenance.exists():
        atomic_write_json(
            provenance,
            {
                "schema": "gcicy-experiment-campaign-provenance-v1",
                "campaign_id": plan.campaign_id,
                "plan_sha256": plan.digest,
                "created_utc": utc_now(),
                "host": socket.gethostname(),
                "platform": platform.platform(),
                "python": sys.version,
                "python_executable": sys.executable,
                "source": dict(source_identity),
                "frozen_inputs": list(frozen_inputs),
                "environment_requirements": dict(plan.environment_requirements),
            },
        )


def _locked_plan_payload(plan: WorkflowPlan) -> dict[str, Any]:
    path = plan.run_root / ".workflow" / "plan.lock.json"
    if not path.is_file():
        raise WorkflowError(f"campaign is not initialized: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WorkflowError(f"corrupt locked plan {path}: {exc}") from exc
    if payload.get("plan_sha256") != plan.digest:
        raise WorkflowError("locked plan digest does not match current plan")
    return payload


def verify_locked_source_identity(plan: WorkflowPlan) -> dict[str, Any] | None:
    """Recheck the exact source snapshot recorded when the run root was made."""

    payload = _locked_plan_payload(plan)
    expected = payload.get("source")
    # Unit-level or legacy callers may provide a synthetic source identity.
    if not isinstance(expected, dict) or "worktree_sha256" not in expected:
        return None
    observed = verify_source_guard(plan)
    if observed != expected:
        raise WorkflowError(
            "current Git source differs from the campaign plan lock; "
            "use the original clean worktree or a new RUN_ROOT"
        )
    return observed


def record_cuda_identity(
    plan: WorkflowPlan, *, gpu_id: str, cuda: Mapping[str, Any]
) -> None:
    """Lock the observed CUDA/PyTorch device identity for this campaign."""

    identity = {
        "schema": "gcicy-experiment-cuda-identity-v1",
        "campaign_id": plan.campaign_id,
        "plan_sha256": plan.digest,
        "gpu_slot": str(gpu_id),
        "torch": cuda.get("torch"),
        "cuda": cuda.get("cuda"),
        "device": cuda.get("device"),
        "capability": cuda.get("capability"),
    }
    path = plan.run_root / ".workflow" / "cuda_identity.json"
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != identity:
            raise WorkflowError(
                "CUDA/PyTorch identity differs from the campaign lock; "
                "use the original environment or a new RUN_ROOT"
            )
    else:
        atomic_write_json(path, identity)


def verify_cuda_requirements(plan: WorkflowPlan, cuda: Mapping[str, Any]) -> None:
    expected_torch = plan.environment_requirements.get("torch")
    observed_torch = str(cuda.get("torch", ""))
    if expected_torch is not None and observed_torch.split("+", 1)[0] != str(
        expected_torch
    ):
        raise WorkflowError(
            f"PyTorch version {observed_torch!r} does not match registered "
            f"version {expected_torch!r}"
        )


def verify_locked_cuda_slot(plan: WorkflowPlan, *, gpu_id: str) -> None:
    path = plan.run_root / ".workflow" / "cuda_identity.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("plan_sha256") != plan.digest or payload.get("gpu_slot") != str(
        gpu_id
    ):
        raise WorkflowError(
            "configured GPU slot differs from the campaign CUDA identity lock"
        )


def job_state_path(plan: WorkflowPlan, job_id: str) -> Path:
    return plan.run_root / ".workflow" / "jobs" / f"{job_id}.json"


def read_job_state(plan: WorkflowPlan, job_id: str) -> dict[str, Any] | None:
    path = job_state_path(plan, job_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WorkflowError(f"corrupt job state {path}: {exc}") from exc


def _output_records(job: Job) -> list[dict[str, Any]]:
    records = []
    for output in job.expected_outputs:
        if not output.path.is_file():
            raise WorkflowError(f"job {job.id} missing output {output.path}")
        size = output.path.stat().st_size
        if size == 0 and not output.allow_empty:
            raise WorkflowError(f"job {job.id} produced empty output {output.path}")
        records.append(
            {
                "path": str(output.path),
                "bytes": size,
                "sha256": sha256_file(output.path),
            }
        )
    return records


def _outputs_match_state(job: Job, state: Mapping[str, Any]) -> bool:
    if state.get("job_digest") != job.digest or state.get("status") != "succeeded":
        return False
    recorded = state.get("outputs", [])
    if len(recorded) != len(job.expected_outputs):
        return False
    by_path = {str(row.get("path")): row for row in recorded}
    for output in job.expected_outputs:
        row = by_path.get(str(output.path))
        if row is None or not output.path.is_file():
            return False
        if output.path.stat().st_size != int(row.get("bytes", -1)):
            return False
        if sha256_file(output.path) != row.get("sha256"):
            return False
    return True


def _json_field(value: Any, dotted_path: str) -> Any:
    current = value
    for component in dotted_path.split("."):
        if isinstance(current, list):
            current = current[int(component)]
        elif isinstance(current, dict) and component in current:
            current = current[component]
        else:
            raise WorkflowError(f"missing JSON field {dotted_path}")
    return current


def _validate_json_gates(job: Job) -> list[dict[str, Any]]:
    reports = []
    for raw in job.json_gates:
        if not isinstance(raw, dict) or "path" not in raw or "field" not in raw:
            raise WorkflowError(f"job {job.id} has malformed JSON gate")
        path = _resolve_path(str(raw["path"]), job.working_directory)
        value = _json_field(
            json.loads(path.read_text(encoding="utf-8")), str(raw["field"])
        )
        comparisons = [key for key in ("equals", "gt", "ge", "lt", "le") if key in raw]
        if len(comparisons) != 1:
            raise WorkflowError(f"job {job.id} JSON gate requires one comparison")
        comparison = comparisons[0]
        target = raw[comparison]
        passed = {
            "equals": value == target,
            "gt": value > target,
            "ge": value >= target,
            "lt": value < target,
            "le": value <= target,
        }[comparison]
        report = {
            "path": str(path),
            "field": raw["field"],
            "comparison": comparison,
            "target": target,
            "observed": value,
            "passed": bool(passed),
        }
        reports.append(report)
        if not passed:
            raise WorkflowError(f"job {job.id} failed JSON gate: {report}")
    return reports


def _selected_job_ids(
    plan: WorkflowPlan,
    *,
    phases: Sequence[str] | None,
    only: Sequence[str] | None,
) -> set[str]:
    by_id = plan.jobs_by_id
    if only:
        unknown = sorted(set(only) - set(by_id))
        if unknown:
            raise WorkflowError(f"unknown selected jobs: {unknown}")
        selected = set(only)
    elif phases:
        unknown_phases = sorted(set(phases) - set(plan.phases))
        if unknown_phases:
            raise WorkflowError(f"unknown selected phases: {unknown_phases}")
        selected = {job.id for job in plan.jobs if job.phase in phases}
    else:
        selected = {job.id for job in plan.jobs}
    stack = list(selected)
    while stack:
        current = stack.pop()
        for dependency in by_id[current].needs:
            if dependency not in selected:
                selected.add(dependency)
                stack.append(dependency)
    return selected


def selection_requires_gpu(
    plan: WorkflowPlan,
    *,
    phases: Sequence[str] | None,
    only: Sequence[str] | None,
) -> bool:
    selected = _selected_job_ids(plan, phases=phases, only=only)
    return any(plan.jobs_by_id[job_id].gpu for job_id in selected)


@contextmanager
def gpu_lock(lock_root: Path, gpu_id: str, *, timeout_seconds: float) -> Iterator[Path]:
    lock_root.mkdir(parents=True, exist_ok=True)
    path = lock_root / f"gcicy-tn-gpu-{gpu_id}.lock"
    handle = path.open("a+", encoding="utf-8")
    started = time.monotonic()
    while True:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if timeout_seconds >= 0 and time.monotonic() - started >= timeout_seconds:
                handle.close()
                raise WorkflowError(f"timed out waiting for GPU lock {path}")
            time.sleep(
                5.0 if timeout_seconds < 0 else min(5.0, max(0.1, timeout_seconds))
            )
    handle.seek(0)
    handle.truncate()
    json.dump(
        {
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "gpu_id": gpu_id,
            "acquired_utc": utc_now(),
        },
        handle,
    )
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
    try:
        yield path
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


@contextmanager
def campaign_lock(plan: WorkflowPlan) -> Iterator[Path]:
    """Prevent two runners from mutating one campaign state tree concurrently."""

    path = plan.run_root / ".workflow" / "runner.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise WorkflowError(
            f"another runner already owns campaign {plan.campaign_id}: {path}"
        ) from exc
    handle.seek(0)
    handle.truncate()
    json.dump(
        {
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "campaign_id": plan.campaign_id,
            "plan_sha256": plan.digest,
            "acquired_utc": utc_now(),
        },
        handle,
    )
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
    try:
        yield path
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


@contextmanager
def bootstrap_lock(
    plan: WorkflowPlan, lock_root: Path, *, timeout_seconds: float
) -> Iterator[Path]:
    """Serialize first-run sentinel, plan, and CUDA identity publication."""

    lock_root.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(str(plan.run_root).encode("utf-8")).hexdigest()[:24]
    path = lock_root / f"gcicy-tn-bootstrap-{key}.lock"
    handle = path.open("a+", encoding="utf-8")
    started = time.monotonic()
    while True:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if timeout_seconds >= 0 and time.monotonic() - started >= timeout_seconds:
                handle.close()
                raise WorkflowError(f"timed out waiting for bootstrap lock {path}")
            time.sleep(
                5.0 if timeout_seconds < 0 else min(5.0, max(0.1, timeout_seconds))
            )
    try:
        yield path
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _retryable(job: Job, returncode: int, log_text: str) -> bool:
    exit_codes = {int(value) for value in job.retry.get("exit_codes", [])}
    signals = {int(value) for value in job.retry.get("signals", [4, 5, 11])}
    patterns = [str(value) for value in job.retry.get("log_patterns", [])]
    direct_signal = -returncode if returncode < 0 else None
    encoded_signal = returncode - 128 if 128 < returncode <= 255 else None
    if (
        returncode in exit_codes
        or direct_signal in signals
        or encoded_signal in signals
    ):
        return True
    return any(pattern in log_text for pattern in patterns)


def _prepare_command(job: Job, previous_state: Mapping[str, Any] | None) -> list[str]:
    command = list(job.command)
    if not previous_state or not job.resume:
        return command
    checkpoint = Path(str(job.resume["checkpoint"]))
    flag = str(job.resume.get("flag", "--resume-checkpoint"))
    if checkpoint.is_file() and flag not in command:
        command.extend([flag, str(checkpoint)])
    return command


def _run_workflow_unlocked(
    plan: WorkflowPlan,
    *,
    gpu_id: str,
    lock_root: Path,
    gpu_lock_timeout_seconds: float = -1,
    heartbeat_seconds: float = 30.0,
    phases: Sequence[str] | None = None,
    only: Sequence[str] | None = None,
    keep_going: bool = False,
) -> dict[str, int]:
    selected = _selected_job_ids(plan, phases=phases, only=only)
    counts = {"succeeded": 0, "skipped": 0, "failed": 0, "blocked": 0}
    for job in plan.jobs:
        if job.id not in selected:
            continue
        verify_locked_source_identity(plan)
        state = read_job_state(plan, job.id)
        if state is not None and state.get("job_digest") != job.digest:
            raise WorkflowError(
                f"job {job.id} state belongs to a different specification"
            )
        if state is not None and _outputs_match_state(job, state):
            counts["skipped"] += 1
            continue
        if state is not None and state.get("status") == "succeeded":
            raise WorkflowError(
                f"job {job.id} succeeded outputs were modified or removed; refusing overwrite"
            )
        if (
            state is not None
            and state.get("status") == "failed"
            and state.get("retryable") is False
        ):
            counts["failed"] += 1
            if not keep_going:
                raise WorkflowError(
                    f"job {job.id} has a terminal failure; see "
                    f"{job_state_path(plan, job.id)}"
                )
            continue
        dependency_states = [
            read_job_state(plan, dependency) for dependency in job.needs
        ]
        if any(
            dependency_state is None
            or not _outputs_match_state(plan.jobs_by_id[dependency], dependency_state)
            for dependency, dependency_state in zip(
                job.needs, dependency_states, strict=True
            )
        ):
            blocked = {
                "schema": "gcicy-experiment-job-state-v1",
                "campaign_id": plan.campaign_id,
                "plan_sha256": plan.digest,
                "job_id": job.id,
                "job_digest": job.digest,
                "status": "blocked",
                "blocked_by": list(job.needs),
                "updated_utc": utc_now(),
                "attempts": [] if state is None else state.get("attempts", []),
            }
            atomic_write_json(job_state_path(plan, job.id), blocked)
            counts["blocked"] += 1
            if not keep_going:
                raise WorkflowError(f"job {job.id} is blocked by dependencies")
            continue
        if state is None and any(
            output.path.exists() for output in job.expected_outputs
        ):
            raise WorkflowError(
                f"job {job.id} has undeclared pre-existing outputs; use a fresh RUN_ROOT"
            )

        attempts = [] if state is None else list(state.get("attempts", []))
        max_attempts = int(job.retry.get("max_attempts", 1))
        if (
            state is not None
            and state.get("status") == "running"
            and len(attempts) >= max_attempts
        ):
            validation_error = None
            output_records: list[dict[str, Any]] = []
            gate_records: list[dict[str, Any]] = []
            try:
                verify_locked_source_identity(plan)
                if not job.expected_outputs:
                    raise WorkflowError(
                        "cannot recover an interrupted job without declared outputs"
                    )
                output_records = _output_records(job)
                gate_records = _validate_json_gates(job)
            except (OSError, ValueError, WorkflowError, json.JSONDecodeError) as exc:
                validation_error = str(exc)
            if validation_error is None:
                recovered_state = {
                    **state,
                    "status": "succeeded",
                    "updated_utc": utc_now(),
                    "outputs": output_records,
                    "json_gates": gate_records,
                    "attempts": attempts,
                    "recovered_after_runner_interruption": True,
                }
                atomic_write_json(job_state_path(plan, job.id), recovered_state)
                counts["succeeded"] += 1
                continue
            exhausted_state = {
                **state,
                "status": "failed",
                "updated_utc": utc_now(),
                "retryable": False,
                "failure_reason": "attempt_budget_exhausted",
                "validation_error": validation_error,
                "attempts": attempts,
            }
            atomic_write_json(job_state_path(plan, job.id), exhausted_state)
            counts["failed"] += 1
            if not keep_going:
                raise WorkflowError(
                    f"job {job.id} exhausted its attempt budget after runner "
                    f"interruption; see {job_state_path(plan, job.id)}"
                )
            continue
        succeeded = False
        while len(attempts) < max_attempts:
            attempt_number = len(attempts) + 1
            command = _prepare_command(job, state)
            log_dir = plan.run_root / ".workflow" / "logs" / job.id
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"attempt-{attempt_number:02d}.log"
            attempt = {
                "attempt": attempt_number,
                "started_utc": utc_now(),
                "command": command,
                "working_directory": str(job.working_directory),
                "log": str(log_path),
                "resume_checkpoint": (
                    None if not job.resume else str(job.resume.get("checkpoint"))
                ),
            }
            attempts.append(attempt)
            running_state = {
                "schema": "gcicy-experiment-job-state-v1",
                "campaign_id": plan.campaign_id,
                "plan_sha256": plan.digest,
                "job_id": job.id,
                "job_digest": job.digest,
                "phase": job.phase,
                "status": "running",
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "gpu_id": gpu_id if job.gpu else None,
                "scientific": dict(job.scientific),
                "updated_utc": utc_now(),
                "attempts": attempts,
            }
            atomic_write_json(job_state_path(plan, job.id), running_state)
            environment = os.environ.copy()
            environment.update(job.environment)
            if job.gpu:
                environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            lock_context = (
                gpu_lock(
                    lock_root,
                    str(gpu_id),
                    timeout_seconds=gpu_lock_timeout_seconds,
                )
                if job.gpu
                else nullcontext()
            )
            returncode = -1
            with lock_context:
                with log_path.open("a", encoding="utf-8", buffering=1) as log_handle:
                    log_handle.write(
                        f"workflow_job_started_utc={utc_now()} job={job.id} "
                        f"attempt={attempt_number}\n"
                    )
                    process = subprocess.Popen(
                        command,
                        cwd=job.working_directory,
                        env=environment,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        text=True,
                        start_new_session=True,
                    )
                    try:
                        while True:
                            try:
                                returncode = process.wait(
                                    timeout=max(0.1, heartbeat_seconds)
                                )
                                break
                            except subprocess.TimeoutExpired:
                                pass
                            running_state["updated_utc"] = utc_now()
                            running_state["child_pid"] = process.pid
                            atomic_write_json(
                                job_state_path(plan, job.id), running_state
                            )
                    except KeyboardInterrupt:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            returncode = process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            returncode = process.wait()
                        raise
                    log_handle.write(
                        f"workflow_job_finished_utc={utc_now()} job={job.id} "
                        f"attempt={attempt_number} returncode={returncode}\n"
                    )
            attempt["finished_utc"] = utc_now()
            attempt["returncode"] = int(returncode)
            validation_error = None
            output_records: list[dict[str, Any]] = []
            gate_records: list[dict[str, Any]] = []
            try:
                verify_locked_source_identity(plan)
            except WorkflowError as exc:
                validation_error = str(exc)
            if returncode == 0 and validation_error is None:
                try:
                    output_records = _output_records(job)
                    gate_records = _validate_json_gates(job)
                except (
                    OSError,
                    ValueError,
                    WorkflowError,
                    json.JSONDecodeError,
                ) as exc:
                    validation_error = str(exc)
            if returncode == 0 and validation_error is None:
                final_state = {
                    **running_state,
                    "status": "succeeded",
                    "updated_utc": utc_now(),
                    "outputs": output_records,
                    "json_gates": gate_records,
                    "attempts": attempts,
                }
                atomic_write_json(job_state_path(plan, job.id), final_state)
                counts["succeeded"] += 1
                succeeded = True
                break
            log_tail = ""
            try:
                log_tail = log_path.read_text(encoding="utf-8", errors="replace")[
                    -8000:
                ]
            except OSError:
                pass
            retryable = validation_error is None and _retryable(
                job, int(returncode), log_tail
            )
            failed_state = {
                **running_state,
                "status": "failed",
                "updated_utc": utc_now(),
                "returncode": int(returncode),
                "validation_error": validation_error,
                "retryable": retryable,
                "log_tail": log_tail,
                "attempts": attempts,
            }
            atomic_write_json(job_state_path(plan, job.id), failed_state)
            state = failed_state
            if not retryable:
                break
            backoff = float(job.retry.get("backoff_seconds", 30.0))
            if backoff > 0:
                time.sleep(backoff)
        if not succeeded:
            if (
                state is not None
                and state.get("retryable") is True
                and len(attempts) >= max_attempts
            ):
                state = {
                    **state,
                    "updated_utc": utc_now(),
                    "retryable": False,
                    "failure_reason": "attempt_budget_exhausted",
                    "attempts": attempts,
                }
                atomic_write_json(job_state_path(plan, job.id), state)
            counts["failed"] += 1
            if not keep_going:
                raise WorkflowError(
                    f"job {job.id} failed; see {job_state_path(plan, job.id)}"
                )
    return counts


def run_workflow(
    plan: WorkflowPlan,
    *,
    gpu_id: str,
    lock_root: Path,
    gpu_lock_timeout_seconds: float = -1,
    heartbeat_seconds: float = 30.0,
    phases: Sequence[str] | None = None,
    only: Sequence[str] | None = None,
    keep_going: bool = False,
) -> dict[str, int]:
    with campaign_lock(plan):
        verify_locked_source_identity(plan)
        verify_locked_cuda_slot(plan, gpu_id=gpu_id)
        return _run_workflow_unlocked(
            plan,
            gpu_id=gpu_id,
            lock_root=lock_root,
            gpu_lock_timeout_seconds=gpu_lock_timeout_seconds,
            heartbeat_seconds=heartbeat_seconds,
            phases=phases,
            only=only,
            keep_going=keep_going,
        )


def status_rows(plan: WorkflowPlan) -> list[dict[str, Any]]:
    rows = []
    for job in plan.jobs:
        state = read_job_state(plan, job.id)
        status = "pending" if state is None else str(state.get("status", "unknown"))
        if (
            state is not None
            and status == "succeeded"
            and not _outputs_match_state(job, state)
        ):
            status = "invalidated"
        rows.append(
            {
                "job_id": job.id,
                "phase": job.phase,
                "status": status,
                "attempts": 0 if state is None else len(state.get("attempts", [])),
                "gpu": job.gpu,
                "needs": list(job.needs),
            }
        )
    return rows


def summarize_results(plan: WorkflowPlan, *, allow_incomplete: bool = False) -> Path:
    _locked_plan_payload(plan)
    rows: list[dict[str, Any]] = []
    missing = []
    for job in plan.jobs:
        if job.result is None:
            continue
        state = read_job_state(plan, job.id)
        if state is None or not _outputs_match_state(job, state):
            if bool(job.result.get("required", True)):
                missing.append(job.id)
            continue
        source_path = _resolve_path(str(job.result["path"]), plan.run_root)
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        factors = dict(job.scientific.get("factors", {}))
        factors.update(job.result.get("factors", {}))
        row: dict[str, Any] = {
            "job_id": job.id,
            "phase": job.phase,
            **factors,
        }
        fields = job.result.get("fields", {})
        if not isinstance(fields, dict):
            raise WorkflowError(f"job {job.id} result fields must be an object")
        for name, dotted_path in fields.items():
            row[str(name)] = _json_field(payload, str(dotted_path))
        rows.append(row)
    if missing and not allow_incomplete:
        raise WorkflowError(f"required result jobs are incomplete: {missing}")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = plan.run_root / "aggregates" / plan.digest / timestamp
    if output_dir.exists():
        raise WorkflowError(f"aggregate output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    summary = {
        "schema": "gcicy-tn-workflow-aggregate-v1",
        "campaign_id": plan.campaign_id,
        "plan_sha256": plan.digest,
        "created_utc": utc_now(),
        "complete": not missing,
        "missing_required_jobs": missing,
        "row_count": len(rows),
        "rows": rows,
    }
    atomic_write_json(output_dir / "summary.json", summary)
    fieldnames = sorted({key for row in rows for key in row})
    csv_path = output_dir / "runs.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    checksums = {
        "summary.json": sha256_file(output_dir / "summary.json"),
        "runs.csv": sha256_file(csv_path),
    }
    atomic_write_json(output_dir / "checksums.json", checksums)
    latest = plan.run_root / "aggregates" / plan.digest / "latest.json"
    atomic_write_json(
        latest,
        {
            "created_utc": summary["created_utc"],
            "directory": str(output_dir),
            "summary_sha256": checksums["summary.json"],
        },
    )
    return output_dir


def check_cuda(python_executable: str, gpu_id: str) -> dict[str, Any]:
    program = (
        "import json, torch; "
        "assert torch.cuda.is_available(), 'CUDA unavailable'; "
        "x=torch.ones(1, device='cuda'); "
        "print(json.dumps({'torch':torch.__version__,"
        "'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0),"
        "'capability':torch.cuda.get_device_capability(0)}))"
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    completed = subprocess.run(
        [python_executable, "-c", program],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise WorkflowError(
            f"CUDA preflight failed on slot {gpu_id}: {completed.stderr.strip()}"
        )
    return json.loads(completed.stdout)
