"""Normalize raw X21 workflow artifacts into baseline provenance certificates.

This module is intentionally read-only with respect to workflow and recovery
roots.  It accepts paths to committed raw artifacts, recomputes every artifact
hash, and emits only the fixed certificate schemas understood by
``x21_auto_research``.  In particular, callers cannot supply metric values,
success booleans, or bare artifact digests.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

from .experiment_workflow import sha256_file
from .x21_auto_research import (
    BASELINE_COMPLETION_SCHEMA,
    FAMILY_SCHEMA,
    RECOVERY_EVIDENCE_SCHEMA,
    X21AutoResearchError,
    digest_value,
    normalize_family,
    validate_baseline_completion_evidence,
    validate_protocol,
    validate_recovery_evidence,
)


RECOVERY_RAW_BUNDLE_SCHEMA = "gcicy-x21-exact-recovery-raw-bundle-v1"
PARENT_BINDINGS_SCHEMA = "gcicy-x21-parent-bindings-v1"
PLAN_LOCK_SCHEMA = "gcicy-experiment-plan-lock-v1"
JOB_STATE_SCHEMA = "gcicy-experiment-job-state-v1"
PLATEAU_STAGE_SCHEMA = "gcicy-tn-plateau-stage-v1"
TRAINING_SUMMARY_SCHEMA = "type11-positive-tensor-network-training-v1"
MODEL_SCHEMA = "type11-positive-tensor-network-v1"
CHECKPOINT_SCHEMA = "type11-positive-tensor-network-checkpoint-v1"

_UNIT_ID = re.compile(r"[A-Za-z0-9_.@:-]+\.service")
_RESUME_ROUND = re.compile(r"\[plateau-stage\]\s+round\s+1/5:\s+resume\b")
_RESUME_EPOCH = re.compile(r"\bresuming at epoch=37\b")
_PLATEAU_PUBLISHED = re.compile(r"\bplateau published from round\s+\d+\b")
_INVOCATION_ID = re.compile(r"[0-9a-f]{32}\Z")
_SYSTEMD_FAILURE = re.compile(
    r"(?:main process exited|failed with result|failed to start)", re.IGNORECASE
)

EXPECTED_K = 20
EXPECTED_D = 14
EXPECTED_PRECISION = "complex64"
EXPECTED_PARAMETER_COUNT = 860_552
EXPECTED_DICTIONARY_RANK = 121
EXPECTED_SOURCE_EPOCH = 36
EXPECTED_NEXT_EPOCH = 37


def _error(message: str) -> X21AutoResearchError:
    return X21AutoResearchError(f"X21 baseline provenance: {message}")


def _read_json(path: Path, *, role: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _error(f"could not read {role} {resolved}: {exc}") from exc
    if not isinstance(value, dict):
        raise _error(f"{role} must be a JSON object")
    # Reading the digest is deliberate even when the enclosing certificate has
    # no field for it: callers cannot substitute an asserted JSON digest.
    sha256_file(resolved)
    return value


def _exact_keys(value: Mapping[str, Any], *, required: set[str], role: str) -> None:
    observed = set(value)
    if observed != required:
        missing = sorted(required - observed)
        extra = sorted(observed - required)
        raise _error(f"{role} fields differ; missing={missing}, extra={extra}")


def _path_from(base: Path, raw: Any, *, role: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise _error(f"{role} must be a non-empty path string")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base / path
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise _error(f"missing {role}: {path}") from exc


def _within(path: Path, root: Path, *, role: str) -> Path:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise _error(f"{role} escapes its registered root: {path}") from exc
    return path


def _integer(value: Any, *, role: str, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise _error(f"{role} must be an integer")
    if minimum is not None and value < minimum:
        raise _error(f"{role} must be at least {minimum}")
    return value


def _load_torch_payload(path: Path, *, role: str) -> dict[str, Any]:
    """Load a trusted, local training artifact on CPU for structural checks."""

    try:
        import torch

        value = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError, TypeError, ModuleNotFoundError) as exc:
        raise _error(f"could not load {role} {path} on CPU: {exc}") from exc
    if not isinstance(value, dict):
        raise _error(f"{role} must contain a dictionary")
    return value


def _protocol(path: Path) -> dict[str, Any]:
    return validate_protocol(_read_json(path, role="X21 auto-research protocol"))


def _contract(protocol: Mapping[str, Any], seed: int) -> dict[str, Any]:
    matches = [
        row
        for row in protocol["baseline_provenance_contracts"]
        if int(row["seed"]) == seed
    ]
    if len(matches) != 1:
        raise _error(f"seed {seed} has no unique baseline provenance contract")
    return dict(matches[0])


def _read_workflow_identity(
    workflow_root: Path, *, contract: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = workflow_root.expanduser().resolve(strict=True)
    plan_path = root / ".workflow" / "plan.lock.json"
    state_path = root / ".workflow" / "jobs" / f"{contract['source_job_id']}.json"
    plan = _read_json(plan_path, role="workflow plan lock")
    state = _read_json(state_path, role="workflow job state")
    if plan.get("schema") != PLAN_LOCK_SCHEMA:
        raise _error("workflow plan lock schema is invalid")
    if state.get("schema") != JOB_STATE_SCHEMA:
        raise _error("workflow job state schema is invalid")

    expected_identity = {
        "campaign_id": contract["source_campaign_id"],
        "plan_sha256": contract["source_plan_sha256"],
    }
    for field, expected in expected_identity.items():
        if plan.get(field) != expected or state.get(field) != expected:
            raise _error(f"workflow {field} does not match the protocol contract")
    if state.get("job_id") != contract["source_job_id"]:
        raise _error("workflow job state identifies another job")
    if state.get("job_digest") != contract["source_job_digest"]:
        raise _error("workflow job state digest does not match the contract")

    rows = plan.get("jobs")
    if not isinstance(rows, list):
        raise _error("workflow plan lock jobs must be a list")
    locked = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("id") == state["job_id"]
    ]
    if len(locked) != 1 or locked[0].get("digest") != state["job_digest"]:
        raise _error("workflow job digest is not bound by the plan lock")
    return plan, state


def _last_command(state: Mapping[str, Any]) -> tuple[list[str], dict[str, Any]]:
    attempts = state.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise _error("workflow job state has no attempts")
    attempt = attempts[-1]
    if not isinstance(attempt, dict):
        raise _error("workflow terminal attempt is malformed")
    command = attempt.get("command")
    if not isinstance(command, list) or not all(
        isinstance(token, str) and token for token in command
    ):
        raise _error("workflow terminal command is malformed")
    return list(command), attempt


def _option(command: Sequence[str], flag: str) -> str:
    positions = [index for index, token in enumerate(command) if token == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise _error(f"workflow command must contain exactly one {flag}")
    return command[positions[0] + 1]


def _validate_job_contract(
    state: Mapping[str, Any], *, seed: int
) -> tuple[list[str], dict[str, Any]]:
    command, attempt = _last_command(state)
    expected_options = {
        "--site-count": str(EXPECTED_K),
        "--bond-dimension": str(EXPECTED_D),
        "--precision": EXPECTED_PRECISION,
        "--torch-seed": str(seed),
        "--max-rounds": "5",
        "--first-kappa-source": "saved_model",
        "--continuation-kappa-source": "saved_model",
    }
    for flag, expected in expected_options.items():
        if _option(command, flag) != expected:
            raise _error(f"workflow command {flag} is not the registered {expected!r}")
    scientific = state.get("scientific")
    if not isinstance(scientific, dict):
        raise _error("workflow job state lacks scientific metadata")
    factors = scientific.get("factors")
    expected_replicate = seed - 8_660_000
    if expected_replicate not in {1, 2, 3}:
        raise _error("registered optimizer seed does not encode replicate 1, 2, or 3")
    if not isinstance(factors, dict) or factors != {
        "k": EXPECTED_K,
        "D": EXPECTED_D,
        "replicate": expected_replicate,
        "trainable_real_parameter_count": EXPECTED_PARAMETER_COUNT,
    }:
        raise _error("workflow job scientific factors are not k20,D14 cores-only")
    if scientific.get("parameter_scope") != "cores-only":
        raise _error("workflow job is not cores-only")
    return command, attempt


def _validate_interrupted_source(
    state: Mapping[str, Any], command: Sequence[str]
) -> None:
    """Bind this v1 recovery to the observed three native SIGSEGV exits."""

    attempts = state.get("attempts")
    if state.get("status") != "failed" or state.get("returncode") != 139:
        raise _error(
            "recovery source job is not the registered terminal native failure"
        )
    if not isinstance(attempts, list) or len(attempts) != 3:
        raise _error("recovery source job must contain its three original attempts")
    for expected_attempt, attempt in enumerate(attempts, start=1):
        if not isinstance(attempt, dict):
            raise _error("recovery source attempt is malformed")
        if (
            attempt.get("attempt") != expected_attempt
            or attempt.get("returncode") != 139
        ):
            raise _error("recovery source attempt sequence/exit code changed")
        if attempt.get("command") != list(command):
            raise _error("recovery source attempts did not use one locked command")


def _validate_model_payload(
    payload: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    seed: int,
    role: str,
) -> None:
    expected = {
        "schema": MODEL_SCHEMA,
        "model_seed": 20260802,
        "sampling_cluster_size": 6,
        "site_count": EXPECTED_K,
        "bond_dimension": EXPECTED_D,
        "precision": EXPECTED_PRECISION,
        "physical_dictionary_rank": EXPECTED_DICTIONARY_RANK,
        "trainable_physical_dictionary": False,
        "physical_dictionary_gauge": "fixed",
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise _error(f"{role}.{field} is not the registered value {value!r}")
    if (
        payload.get("source_artifact_sha256")
        != protocol["data_contract"]["source_artifact_sha256"]
    ):
        raise _error(f"{role} uses another X21 source artifact")
    if (
        payload.get("train_common_pool_sha256")
        != protocol["data_contract"]["train_pool_sha256"]
    ):
        raise _error(f"{role} uses another training pool")
    if (
        payload.get("validation_common_pool_sha256")
        != protocol["data_contract"]["selection_pool_sha256"]
    ):
        raise _error(f"{role} uses another validation pool")


def _checkpoint_payload(
    path: Path,
    *,
    protocol: Mapping[str, Any],
    seed: int,
    initial_model_sha256: str | None,
    role: str,
) -> dict[str, Any]:
    payload = _load_torch_payload(path, role=role)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise _error(f"{role} schema is invalid")
    required_state = {
        "current_model_state_dict",
        "best_state_dict",
        "optimizer_state_dict",
        "permutation_generator_state",
        "cpu_rng_state",
        "cuda_rng_state",
        "history",
        "training_semantics",
        "frozen_input_hashes",
        "epoch",
        "next_epoch",
    }
    missing = sorted(required_state - set(payload))
    if missing:
        raise _error(f"{role} lacks exact resume state: {missing}")
    epoch = _integer(payload["epoch"], role=f"{role}.epoch", minimum=0)
    next_epoch = _integer(payload["next_epoch"], role=f"{role}.next_epoch", minimum=1)
    if next_epoch != epoch + 1:
        raise _error(f"{role} has an inconsistent epoch boundary")
    history = payload["history"]
    if not isinstance(history, list) or not history:
        raise _error(f"{role}.history must be non-empty")
    history_epochs = [
        row.get("epoch") if isinstance(row, dict) else None for row in history
    ]
    if history_epochs != sorted(set(history_epochs)) or history_epochs[-1] != epoch:
        raise _error(f"{role}.history does not end at the checkpoint epoch")

    semantics = payload["training_semantics"]
    if not isinstance(semantics, dict):
        raise _error(f"{role}.training_semantics must be an object")
    model = semantics.get("model")
    optimization = semantics.get("optimization")
    if not isinstance(model, dict) or not isinstance(optimization, dict):
        raise _error(f"{role} lacks model/optimization training semantics")
    for field, value in {
        "site_count": EXPECTED_K,
        "bond_dimension": EXPECTED_D,
        "precision": EXPECTED_PRECISION,
        "physical_dictionary_rank": EXPECTED_DICTIONARY_RANK,
        "trainable_physical_dictionary": False,
    }.items():
        if model.get(field) != value:
            raise _error(f"{role}.training_semantics.model.{field} changed")
    if optimization.get("torch_seed") != seed:
        raise _error(f"{role} uses another optimizer seed")
    if semantics.get("implementation") is None:
        raise _error(f"{role} lacks implementation identity")

    frozen = payload["frozen_input_hashes"]
    if not isinstance(frozen, dict):
        raise _error(f"{role}.frozen_input_hashes must be an object")
    expected_frozen = {
        "source_artifact_sha256": protocol["data_contract"]["source_artifact_sha256"],
        "teacher_artifact_sha256": None,
        "initial_model_sha256": initial_model_sha256,
        "train_common_pool_sha256": protocol["data_contract"]["train_pool_sha256"],
        "validation_common_pool_sha256": protocol["data_contract"][
            "selection_pool_sha256"
        ],
    }
    if frozen != expected_frozen:
        raise _error(f"{role} frozen inputs differ from the protocol and source model")
    return payload


def _safe_stage_member(stage_dir: Path, raw: Any, *, role: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise _error(f"{role} must be a non-empty relative path")
    candidate = Path(raw)
    if candidate.is_absolute():
        raise _error(f"{role} must be relative to the stage directory")
    return _within((stage_dir / candidate).resolve(strict=True), stage_dir, role=role)


def _validate_summary(
    summary: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    seed: int,
    model_sha256: str,
    checkpoint_sha256: str,
    require_resume_from: str | None,
) -> None:
    expected = {
        "schema": TRAINING_SUMMARY_SCHEMA,
        "model_seed": 20260802,
        "sampling_cluster_size": 6,
        "torch_seed": seed,
        "site_count": EXPECTED_K,
        "bond_dimension": EXPECTED_D,
        "precision": EXPECTED_PRECISION,
        "physical_dictionary_rank": EXPECTED_DICTIONARY_RANK,
        "trainable_physical_dictionary": False,
        "physical_dictionary_gauge": "fixed",
        "trainable_real_parameter_count": EXPECTED_PARAMETER_COUNT,
        "termination_reason": "validation_plateau",
        "model_sha256": model_sha256,
    }
    for field, value in expected.items():
        if summary.get(field) != value:
            raise _error(
                f"plateau summary {field} is not the registered value {value!r}"
            )
    if (
        summary.get("source_artifact_sha256")
        != protocol["data_contract"]["source_artifact_sha256"]
    ):
        raise _error("plateau summary uses another X21 source artifact")
    train = summary.get("train")
    validation = summary.get("validation")
    if not isinstance(train, dict) or not isinstance(validation, dict):
        raise _error("plateau summary lacks train/validation identities")
    if (
        train.get("points") != protocol["data_contract"]["train_points"]
        or train.get("common_pool_sha256")
        != protocol["data_contract"]["train_pool_sha256"]
    ):
        raise _error("plateau summary training pool changed")
    if (
        validation.get("points") != protocol["data_contract"]["selection_points"]
        or validation.get("common_pool_sha256")
        != protocol["data_contract"]["selection_pool_sha256"]
    ):
        raise _error("plateau summary validation pool changed")
    checkpoint = summary.get("checkpoint")
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("sha256") != checkpoint_sha256
    ):
        raise _error("plateau summary does not bind the terminal checkpoint")
    if require_resume_from is None:
        if checkpoint.get("resume_kind") not in {"fresh_training", "crash_recovery"}:
            raise _error("plateau summary checkpoint resume kind is invalid")
    else:
        if checkpoint.get("resume_kind") != "crash_recovery":
            raise _error("recovery summary is not marked crash_recovery")
        if checkpoint.get("resumed_from_sha256") != require_resume_from:
            raise _error("recovery summary resumed from another checkpoint")
        if (
            _integer(
                checkpoint.get("last_completed_validation_epoch"),
                role="plateau summary checkpoint epoch",
                minimum=EXPECTED_NEXT_EPOCH,
            )
            < EXPECTED_NEXT_EPOCH
        ):
            raise _error("recovery did not advance beyond epoch 36")


def _validate_stage(
    stage_dir: Path,
    *,
    protocol: Mapping[str, Any],
    seed: int,
    initial_model_sha256: str | None,
    require_resume_from: str | None = None,
) -> dict[str, Any]:
    stage_dir = stage_dir.expanduser().resolve(strict=True)
    stage_path = stage_dir / "stage_summary.json"
    stage = _read_json(stage_path, role="plateau stage summary")
    if stage.get("schema") != PLATEAU_STAGE_SCHEMA:
        raise _error("plateau stage summary schema is invalid")
    if (
        stage.get("status") != "validation_plateau"
        or stage.get("termination_reason") != "validation_plateau"
    ):
        raise _error("plateau stage is not terminal validation_plateau")
    plateau_round = _integer(
        stage.get("plateau_round"), role="plateau round", minimum=1
    )
    rounds = stage.get("rounds")
    if (
        not isinstance(rounds, list)
        or len(rounds) != plateau_round
        or stage.get("rounds_completed") != len(rounds)
    ):
        raise _error("plateau stage round ledger is inconsistent")
    terminal = rounds[-1]
    if not isinstance(terminal, dict) or terminal.get("round") != plateau_round:
        raise _error("plateau stage terminal round is malformed")
    if (
        terminal.get("status") != "validation_plateau"
        or terminal.get("termination_reason") != "validation_plateau"
    ):
        raise _error("plateau stage terminal round is not validation_plateau")

    stage_initial = (
        Path(str(stage.get("initial_model"))).expanduser().resolve(strict=True)
    )
    if (
        sha256_file(stage_initial) != initial_model_sha256
        or stage.get("initial_model_sha256") != initial_model_sha256
    ):
        raise _error("plateau stage initial model is not hash-bound")
    terminal_initial = (
        Path(str(terminal.get("initial_model"))).expanduser().resolve(strict=True)
    )
    terminal_initial_sha256 = sha256_file(terminal_initial)
    if terminal.get("initial_model_sha256") != terminal_initial_sha256:
        raise _error("terminal round initial model is not hash-bound")
    if require_resume_from is not None and plateau_round != 1:
        raise _error("registered exact recovery must resume and terminate in round 1")

    model_path = _safe_stage_member(
        stage_dir, stage.get("plateau_model"), role="plateau model"
    )
    summary_path = _safe_stage_member(
        stage_dir, stage.get("plateau_summary"), role="plateau summary"
    )
    terminal_model = _safe_stage_member(
        stage_dir, terminal.get("model"), role="terminal round model"
    )
    terminal_summary = _safe_stage_member(
        stage_dir, terminal.get("summary"), role="terminal round summary"
    )
    checkpoint_path = _safe_stage_member(
        stage_dir, terminal.get("checkpoint"), role="terminal round checkpoint"
    )

    hashes = {
        "model": sha256_file(model_path),
        "summary": sha256_file(summary_path),
        "stage_summary": sha256_file(stage_path),
        "checkpoint": sha256_file(checkpoint_path),
        "terminal_model": sha256_file(terminal_model),
        "terminal_summary": sha256_file(terminal_summary),
    }
    if (
        stage.get("plateau_model_sha256") != hashes["model"]
        or terminal.get("model_sha256") != hashes["terminal_model"]
    ):
        raise _error("plateau stage model hash record is stale")
    if (
        stage.get("plateau_summary_sha256") != hashes["summary"]
        or terminal.get("summary_sha256") != hashes["terminal_summary"]
    ):
        raise _error("plateau stage summary hash record is stale")
    if terminal.get("checkpoint_sha256") != hashes["checkpoint"]:
        raise _error("plateau stage checkpoint hash record is stale")
    if (
        hashes["model"] != hashes["terminal_model"]
        or hashes["summary"] != hashes["terminal_summary"]
    ):
        raise _error("published plateau artifacts differ from their terminal round")

    model_payload = _load_torch_payload(model_path, role="plateau model")
    _validate_model_payload(
        model_payload, protocol=protocol, seed=seed, role="plateau model"
    )
    checkpoint_payload = _checkpoint_payload(
        checkpoint_path,
        protocol=protocol,
        seed=seed,
        initial_model_sha256=terminal_initial_sha256,
        role="terminal checkpoint",
    )
    summary = _read_json(summary_path, role="plateau training summary")
    _validate_summary(
        summary,
        protocol=protocol,
        seed=seed,
        model_sha256=hashes["model"],
        checkpoint_sha256=hashes["checkpoint"],
        require_resume_from=require_resume_from,
    )
    if summary != _read_json(terminal_summary, role="terminal round training summary"):
        raise _error("published and terminal training summaries differ")
    return {
        "stage_dir": stage_dir,
        "stage": stage,
        "summary": summary,
        "model_payload": model_payload,
        "checkpoint_payload": checkpoint_payload,
        "paths": {
            "model": model_path,
            "summary": summary_path,
            "stage_summary": stage_path,
            "checkpoint": checkpoint_path,
        },
        "hashes": hashes,
    }


def _validate_state_outputs(
    state: Mapping[str, Any], *, stage: Mapping[str, Any], workflow_root: Path
) -> None:
    outputs = state.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 3:
        raise _error("successful workflow endpoint must record exactly three outputs")
    expected_paths = {
        stage["paths"]["model"],
        stage["paths"]["summary"],
        stage["paths"]["stage_summary"],
    }
    observed_paths: set[Path] = set()
    for row in outputs:
        if not isinstance(row, dict):
            raise _error("workflow output record is malformed")
        path = _within(
            Path(str(row.get("path"))).expanduser().resolve(strict=True),
            workflow_root,
            role="workflow output",
        )
        if path in observed_paths:
            raise _error("workflow output records contain a duplicate path")
        observed_paths.add(path)
        digest = sha256_file(path)
        if row.get("sha256") != digest or row.get("bytes") != path.stat().st_size:
            raise _error(f"workflow output record is stale: {path}")
    if observed_paths != expected_paths:
        raise _error("workflow outputs are not the registered plateau endpoint")


def normalize_baseline_completion(
    *, protocol_path: Path, workflow_root: Path, seed: int
) -> dict[str, Any]:
    """Build a completion certificate exclusively from a successful workflow."""

    protocol = _protocol(protocol_path)
    contract = _contract(protocol, seed)
    if contract["kind"] not in {"existing-complete", "fresh-completion"}:
        raise _error(f"seed {seed} is not registered for completion provenance")
    root = workflow_root.expanduser().resolve(strict=True)
    _plan, state = _read_workflow_identity(root, contract=contract)
    command, attempt = _validate_job_contract(state, seed=seed)
    if state.get("status") != "succeeded":
        raise _error("baseline workflow job is not succeeded")
    if attempt.get("returncode") != 0:
        raise _error("baseline workflow terminal native exit code is not zero")
    stage_dir = _within(
        Path(_option(command, "--stage-dir")).expanduser().resolve(strict=True),
        root,
        role="workflow plateau stage",
    )
    initial_model = (
        Path(_option(command, "--initial-model")).expanduser().resolve(strict=True)
    )
    initial_model_sha256 = sha256_file(initial_model)
    stage = _validate_stage(
        stage_dir,
        protocol=protocol,
        seed=seed,
        initial_model_sha256=initial_model_sha256,
    )
    _validate_state_outputs(state, stage=stage, workflow_root=root)
    certificate = {
        "schema": BASELINE_COMPLETION_SCHEMA,
        "seed": seed,
        "source_campaign_id": contract["source_campaign_id"],
        "source_plan_sha256": contract["source_plan_sha256"],
        "source_job_id": contract["source_job_id"],
        "source_job_digest": contract["source_job_digest"],
        "terminal_status": "validation_plateau",
        "native_exit_code": 0,
        "output_model_sha256": stage["hashes"]["model"],
        "output_checkpoint_sha256": stage["hashes"]["checkpoint"],
        "output_summary_sha256": stage["hashes"]["summary"],
    }
    return validate_baseline_completion_evidence(certificate)


def _find_source_checkpoint(stage_dir: Path, *, expected_sha256: str) -> Path:
    matches = []
    for path in sorted(stage_dir.glob("round_*/checkpoint.pt")):
        if path.is_file() and sha256_file(path) == expected_sha256:
            matches.append(path.resolve(strict=True))
    if len(matches) != 1:
        raise _error(
            "source workflow must contain exactly one checkpoint matching the "
            "registered recovery hash"
        )
    return matches[0]


def _read_systemd_show(path: Path) -> dict[str, str]:
    sha256_file(path)
    rows: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise _error(f"could not read systemd show export {path}: {exc}") from exc
    for line in lines:
        if not line or "=" not in line:
            raise _error("systemd show export contains a malformed line")
        key, value = line.split("=", 1)
        if key in rows:
            raise _error(f"systemd show export repeats {key}")
        rows[key] = value
    required = {
        "Id",
        "LoadState",
        "ActiveState",
        "SubState",
        "Result",
        "ExecMainCode",
        "ExecMainStatus",
        "NRestarts",
    }
    if set(rows) != required:
        raise _error(
            f"systemd show fields differ; missing={sorted(required-set(rows))}, "
            f"extra={sorted(set(rows)-required)}"
        )
    if not _UNIT_ID.fullmatch(rows["Id"]):
        raise _error("systemd unit Id is invalid")
    if (rows["ActiveState"], rows["SubState"]) != ("inactive", "dead"):
        raise _error("recovery systemd unit has not reached an inactive terminal state")
    if rows["Result"] != "success":
        raise _error("recovery systemd unit result is not success")
    loaded_exit = rows["LoadState"] == "loaded" and rows["ExecMainCode"] in {
        "1",
        "exited",
    }
    collected_exit = (
        rows["LoadState"] == "not-found"
        and rows["ExecMainCode"] == "0"
        and rows["NRestarts"] == "0"
    )
    if not (loaded_exit or collected_exit) or rows["ExecMainStatus"] != "0":
        raise _error("recovery native process did not exit with status zero")
    restarts = int(rows["NRestarts"])
    if restarts < 0:
        raise _error("systemd NRestarts is negative")
    return rows


def _read_journal_evidence(
    path: Path, *, unit_id: str, collected_transient: bool
) -> str:
    """Read a journal export and authenticate a collected transient unit.

    A ``systemd-run --collect`` unit no longer has live ExecMain metadata after
    collection.  In that case the JSON journal must bind one invocation ID to
    the systemd start/done records, every service stdout row, and the terminal
    systemd resource-accounting record.  Failure records are forbidden.
    """

    try:
        text = path.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeError) as exc:
        raise _error(f"could not read recovery journal {path}: {exc}") from exc
    sha256_file(path)
    if not collected_transient:
        return text

    records: list[dict[str, Any]] = []
    for index, line in enumerate(text.splitlines()):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise _error(
                "collected transient recovery requires a JSON-lines journal export"
            ) from exc
        if not isinstance(row, dict):
            raise _error(f"journal row {index} is not an object")
        records.append(row)
    if not records:
        raise _error("recovery journal export is empty")

    service_rows = [row for row in records if row.get("_SYSTEMD_USER_UNIT") == unit_id]
    systemd_rows = [row for row in records if row.get("USER_UNIT") == unit_id]
    invocation_ids = {
        str(row["_SYSTEMD_INVOCATION_ID"])
        for row in service_rows
        if row.get("_SYSTEMD_INVOCATION_ID") is not None
    } | {
        str(row["USER_INVOCATION_ID"])
        for row in systemd_rows
        if row.get("USER_INVOCATION_ID") is not None
    }
    if len(invocation_ids) != 1 or not _INVOCATION_ID.fullmatch(
        next(iter(invocation_ids), "")
    ):
        raise _error("recovery journal does not bind one systemd invocation")
    invocation_id = next(iter(invocation_ids))
    if not service_rows or any(
        row.get("_SYSTEMD_INVOCATION_ID") != invocation_id for row in service_rows
    ):
        raise _error("recovery stdout rows are not bound to the invocation")

    start = [
        row
        for row in systemd_rows
        if row.get("JOB_TYPE") == "start"
        and row.get("USER_INVOCATION_ID") == invocation_id
        and str(row.get("MESSAGE", "")).startswith("Starting ")
    ]
    started = [
        row
        for row in systemd_rows
        if row.get("JOB_TYPE") == "start"
        and row.get("JOB_RESULT") == "done"
        and row.get("USER_INVOCATION_ID") == invocation_id
        and str(row.get("MESSAGE", "")).startswith("Started ")
    ]
    terminal = [
        row
        for row in systemd_rows
        if row.get("MESSAGE_ID") == "ae8f7b866b0347b9af31fe1c80b127c0"
        and row.get("USER_INVOCATION_ID") == invocation_id
    ]
    if len(start) != 1 or len(started) != 1 or len(terminal) != 1:
        raise _error("recovery journal lacks a complete systemd invocation envelope")
    messages = [str(row.get("MESSAGE", "")) for row in records]
    if any(_SYSTEMD_FAILURE.search(message) for message in messages):
        raise _error("recovery journal contains a systemd failure record")
    try:
        plateau_time = max(
            int(row["__REALTIME_TIMESTAMP"])
            for row in service_rows
            if _PLATEAU_PUBLISHED.search(str(row.get("MESSAGE", "")))
        )
        terminal_time = int(terminal[0]["__REALTIME_TIMESTAMP"])
    except (KeyError, TypeError, ValueError) as exc:
        raise _error("recovery journal timestamps are incomplete") from exc
    if terminal_time < plateau_time:
        raise _error("systemd terminal accounting precedes plateau publication")
    return "\n".join(messages)


def _recovery_bundle(path: Path) -> tuple[dict[str, Any], Path]:
    resolved = path.expanduser().resolve(strict=True)
    value = _read_json(resolved, role="exact recovery raw bundle")
    _exact_keys(
        value,
        required={
            "schema",
            "source_workflow_root",
            "source_model",
            "stage_dir",
            "systemd_show",
            "journal",
        },
        role="exact recovery raw bundle",
    )
    if value["schema"] != RECOVERY_RAW_BUNDLE_SCHEMA:
        raise _error(f"recovery raw bundle schema must be {RECOVERY_RAW_BUNDLE_SCHEMA}")
    return value, resolved.parent


def normalize_exact_recovery(
    *, protocol_path: Path, raw_bundle_path: Path
) -> dict[str, Any]:
    """Build an exact-recovery certificate from source and systemd artifacts."""

    protocol = _protocol(protocol_path)
    seed = int(protocol["recovery_seed"])
    contract = _contract(protocol, seed)
    if contract["kind"] != "exact-recovery":
        raise _error("protocol recovery seed is not registered as exact-recovery")
    if contract["training_semantics_sha256"] != contract["source_job_digest"]:
        raise _error(
            "recovery training-semantics identity is not bound to the source job"
        )
    if contract["frozen_inputs_sha256"] != contract["source_plan_sha256"]:
        raise _error("recovery frozen-input identity is not bound to the source plan")

    bundle, base = _recovery_bundle(raw_bundle_path)
    source_root = _path_from(
        base, bundle["source_workflow_root"], role="source workflow root"
    )
    recovery_model = _path_from(
        base, bundle["source_model"], role="recovery source model"
    )
    recovery_stage = _path_from(
        base, bundle["stage_dir"], role="recovery stage directory"
    )
    systemd_show_path = _path_from(
        base, bundle["systemd_show"], role="systemd show export"
    )
    journal_path = _path_from(base, bundle["journal"], role="systemd journal export")

    _plan, source_state = _read_workflow_identity(source_root, contract=contract)
    source_command, _source_attempt = _validate_job_contract(source_state, seed=seed)
    _validate_interrupted_source(source_state, source_command)
    source_stage = _within(
        Path(_option(source_command, "--stage-dir")).expanduser().resolve(strict=True),
        source_root,
        role="source workflow plateau stage",
    )
    original_source_model = (
        Path(_option(source_command, "--initial-model"))
        .expanduser()
        .resolve(strict=True)
    )
    original_model_sha256 = sha256_file(original_source_model)
    copied_model_sha256 = sha256_file(recovery_model)
    if (
        original_model_sha256 != contract["source_model_sha256"]
        or copied_model_sha256 != original_model_sha256
    ):
        raise _error(
            "recovery source model does not match the registered workflow model"
        )
    original_model_payload = _load_torch_payload(
        original_source_model, role="workflow source model"
    )
    copied_model_payload = _load_torch_payload(
        recovery_model, role="recovery copied source model"
    )
    _validate_model_payload(
        original_model_payload,
        protocol=protocol,
        seed=seed,
        role="workflow source model",
    )
    _validate_model_payload(
        copied_model_payload, protocol=protocol, seed=seed, role="recovery source model"
    )

    source_checkpoint = _find_source_checkpoint(
        source_stage,
        expected_sha256=contract["source_checkpoint_sha256"],
    )
    source_checkpoint_payload = _checkpoint_payload(
        source_checkpoint,
        protocol=protocol,
        seed=seed,
        initial_model_sha256=original_model_sha256,
        role="source checkpoint",
    )
    if (
        source_checkpoint_payload["epoch"] != EXPECTED_SOURCE_EPOCH
        or source_checkpoint_payload["next_epoch"] != EXPECTED_NEXT_EPOCH
    ):
        raise _error("source checkpoint is not the registered epoch 36 -> 37 boundary")
    if contract["source_epoch"] != EXPECTED_SOURCE_EPOCH:
        raise _error("protocol recovery source epoch is not 36")

    systemd = _read_systemd_show(systemd_show_path)
    journal = _read_journal_evidence(
        journal_path,
        unit_id=systemd["Id"],
        collected_transient=systemd["LoadState"] == "not-found",
    )
    for pattern, label in (
        (_RESUME_ROUND, "round 1/5 resume"),
        (_RESUME_EPOCH, "epoch 37 trainer resume"),
        (_PLATEAU_PUBLISHED, "terminal plateau publication"),
    ):
        if pattern.search(journal) is None:
            raise _error(f"recovery journal lacks {label} evidence")

    endpoint = _validate_stage(
        recovery_stage,
        protocol=protocol,
        seed=seed,
        initial_model_sha256=copied_model_sha256,
        require_resume_from=contract["source_checkpoint_sha256"],
    )
    terminal_checkpoint = endpoint["checkpoint_payload"]
    if terminal_checkpoint["epoch"] < EXPECTED_NEXT_EPOCH:
        raise _error("terminal recovery checkpoint did not advance past epoch 36")
    if (
        terminal_checkpoint["training_semantics"]
        != source_checkpoint_payload["training_semantics"]
    ):
        raise _error(
            "recovery implementation/training semantics differ from the source checkpoint"
        )
    if (
        terminal_checkpoint["frozen_input_hashes"]
        != source_checkpoint_payload["frozen_input_hashes"]
    ):
        raise _error("recovery frozen inputs differ from the source checkpoint")
    source_history = source_checkpoint_payload["history"]
    terminal_history = terminal_checkpoint["history"]
    if terminal_history[: len(source_history)] != source_history:
        raise _error(
            "recovery terminal history does not preserve the source checkpoint history"
        )

    certificate = {
        "schema": RECOVERY_EVIDENCE_SCHEMA,
        "campaign_id": contract["recovery_campaign_id"],
        "seed": seed,
        "mode": "exact-checkpoint-resume",
        "source_campaign_id": contract["source_campaign_id"],
        "source_plan_sha256": contract["source_plan_sha256"],
        "source_job_id": contract["source_job_id"],
        "source_job_digest": contract["source_job_digest"],
        "source_model_sha256": original_model_sha256,
        "source_checkpoint_sha256": sha256_file(source_checkpoint),
        "source_epoch": EXPECTED_SOURCE_EPOCH,
        "source_next_epoch": EXPECTED_NEXT_EPOCH,
        "training_semantics_sha256": contract["training_semantics_sha256"],
        "frozen_inputs_sha256": contract["frozen_inputs_sha256"],
        "implementation_match": True,
        "optimizer_state_restored": True,
        "rng_state_restored": True,
        "terminal_status": "validation_plateau",
        "native_exit_code": int(systemd["ExecMainStatus"]),
        "operational_attempt_count": int(systemd["NRestarts"]) + 1,
        "output_model_sha256": endpoint["hashes"]["model"],
        "output_checkpoint_sha256": endpoint["hashes"]["checkpoint"],
        "output_summary_sha256": endpoint["hashes"]["summary"],
    }
    return validate_recovery_evidence(certificate)


def _publish(path: Path, value: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    if destination.exists():
        if destination.read_text(encoding="utf-8") != payload:
            raise _error(
                f"refusing to overwrite a different certificate: {destination}"
            )
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def assemble_baseline_family(
    *,
    protocol_path: Path,
    certificate_paths: Mapping[int, Path],
    model_paths: Mapping[int, Path],
    checkpoint_paths: Mapping[int, Path],
) -> dict[str, Any]:
    """Build a registerable three-seed family from normalized certificates.

    The returned object deliberately retains each inline certificate.  The
    scientific controller validates those bodies against the protocol and
    strips them when it publishes its immutable baseline artifact.
    """

    protocol = _protocol(protocol_path)
    seeds = tuple(int(seed) for seed in protocol["promotion_seeds"])
    expected = set(seeds)
    if set(certificate_paths) != expected:
        raise _error("certificate seed set does not match the protocol")
    if set(model_paths) != expected:
        raise _error("model seed set does not match the protocol")
    if set(checkpoint_paths) != expected:
        raise _error("checkpoint seed set does not match the protocol")

    models: list[dict[str, Any]] = []
    for seed in seeds:
        contract = _contract(protocol, seed)
        certificate = _read_json(
            certificate_paths[seed], role=f"seed {seed} baseline certificate"
        )
        if contract["kind"] == "exact-recovery":
            normalized_certificate = validate_recovery_evidence(certificate)
        else:
            normalized_certificate = validate_baseline_completion_evidence(certificate)
        model = model_paths[seed].expanduser().resolve(strict=True)
        model_sha256 = sha256_file(model)
        checkpoint = checkpoint_paths[seed].expanduser().resolve(strict=True)
        checkpoint_sha256 = sha256_file(checkpoint)
        if normalized_certificate["output_model_sha256"] != model_sha256:
            raise _error(f"seed {seed} certificate does not identify its model file")
        if normalized_certificate["output_checkpoint_sha256"] != checkpoint_sha256:
            raise _error(
                f"seed {seed} certificate does not identify its checkpoint file"
            )
        models.append(
            {
                "seed": seed,
                # FAMILY_SCHEMA predates this provenance normalizer and calls the
                # runnable saved-model identity ``checkpoint_sha256``.  Bind that
                # field to model.pt, never to the optimizer/RNG recovery checkpoint.
                "checkpoint_sha256": model_sha256,
                "provenance": {
                    "kind": contract["kind"],
                    "certificate_sha256": digest_value(normalized_certificate),
                    "certificate": normalized_certificate,
                },
            }
        )

    candidate = {
        "schema": FAMILY_SCHEMA,
        "model_metadata": protocol["expected_baseline_metadata"],
        "models": models,
    }
    normalized_family = normalize_family(
        candidate,
        seeds=seeds,
        context="assembled_baseline_family",
        expected_metadata=protocol["expected_baseline_metadata"],
        recovery_seed=int(protocol["recovery_seed"]),
        require_recovery_certificate=True,
        baseline_provenance_contracts=protocol["baseline_provenance_contracts"],
    )
    return {**candidate, "family_sha256": normalized_family["family_sha256"]}


def assemble_parent_bindings(
    *,
    protocol_path: Path,
    certificate_paths: Mapping[int, Path],
    model_paths: Mapping[int, Path],
    checkpoint_paths: Mapping[int, Path],
) -> dict[str, Any]:
    """Create the exact bridge input after revalidating all three artifacts."""

    family = assemble_baseline_family(
        protocol_path=protocol_path,
        certificate_paths=certificate_paths,
        model_paths=model_paths,
        checkpoint_paths=checkpoint_paths,
    )
    family_by_seed = {int(row["seed"]): row for row in family["models"]}
    return {
        "schema": PARENT_BINDINGS_SCHEMA,
        "models": [
            {
                "seed": seed,
                "model_path": str(model_paths[seed].expanduser().resolve(strict=True)),
                "checkpoint_path": str(
                    checkpoint_paths[seed].expanduser().resolve(strict=True)
                ),
                "certificate_path": str(
                    certificate_paths[seed].expanduser().resolve(strict=True)
                ),
            }
            for seed in sorted(family_by_seed)
        ],
    }


def _seed_path_map(values: Sequence[str], *, role: str) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for raw in values:
        seed_text, separator, path_text = raw.partition("=")
        if not separator or not seed_text.isdecimal() or not path_text:
            raise _error(f"{role} must use SEED=/absolute/path syntax")
        seed = int(seed_text)
        if seed in result:
            raise _error(f"duplicate {role} seed {seed}")
        path = Path(path_text).expanduser()
        if not path.is_absolute():
            raise _error(f"{role} path must be absolute for seed {seed}")
        result[seed] = path
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    completion = commands.add_parser("completion")
    completion.add_argument("--protocol", type=Path, required=True)
    completion.add_argument("--workflow-root", type=Path, required=True)
    completion.add_argument("--seed", type=int, required=True)
    completion.add_argument("--out", type=Path, required=True)
    recovery = commands.add_parser("recovery")
    recovery.add_argument("--protocol", type=Path, required=True)
    recovery.add_argument("--raw-bundle", type=Path, required=True)
    recovery.add_argument("--out", type=Path, required=True)
    family = commands.add_parser("family")
    family.add_argument("--protocol", type=Path, required=True)
    family.add_argument(
        "--certificate", action="append", default=[], metavar="SEED=PATH"
    )
    family.add_argument("--model", action="append", default=[], metavar="SEED=PATH")
    family.add_argument(
        "--checkpoint", action="append", default=[], metavar="SEED=PATH"
    )
    family.add_argument("--out", type=Path, required=True)
    family.add_argument("--bindings-out", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "completion":
            certificate = normalize_baseline_completion(
                protocol_path=args.protocol,
                workflow_root=args.workflow_root,
                seed=args.seed,
            )
        elif args.command == "recovery":
            certificate = normalize_exact_recovery(
                protocol_path=args.protocol,
                raw_bundle_path=args.raw_bundle,
            )
        else:
            certificate_paths = _seed_path_map(args.certificate, role="certificate")
            model_paths = _seed_path_map(args.model, role="model")
            checkpoint_paths = _seed_path_map(args.checkpoint, role="checkpoint")
            certificate = assemble_baseline_family(
                protocol_path=args.protocol,
                certificate_paths=certificate_paths,
                model_paths=model_paths,
                checkpoint_paths=checkpoint_paths,
            )
            if args.bindings_out is not None:
                bindings = assemble_parent_bindings(
                    protocol_path=args.protocol,
                    certificate_paths=certificate_paths,
                    model_paths=model_paths,
                    checkpoint_paths=checkpoint_paths,
                )
                _publish(args.bindings_out, bindings)
        _publish(args.out, certificate)
    except (OSError, ValueError, X21AutoResearchError) as exc:
        raise SystemExit(f"x21 baseline provenance error: {exc}") from exc
    print(json.dumps(certificate, indent=2, sort_keys=True, allow_nan=False))
    return 0


__all__ = [
    "RECOVERY_RAW_BUNDLE_SCHEMA",
    "assemble_baseline_family",
    "assemble_parent_bindings",
    "normalize_baseline_completion",
    "normalize_exact_recovery",
]
