"""Executable, fail-closed bridge for X21 Auto Research round 1.

The bridge deliberately has a narrower authority than
``x21_auto_research.X21CampaignStore``.  It can execute the first registered
non-resource action only: matched, fixed-update continuation of each of the
three k=20, D=14, complex64 parents with learning rates 3e-5 (control) and
1.5e-5 (candidate).

Only immutable train and selection-labelled common pools are exposed to child
processes.  The v1 protocol's 49,152-point ``development_pool_sha256`` is the
historical file named ``X21_confirmation_*`` and is always rejected.  A v1 run
is therefore training-diagnostic only.  A separately initialized v2 controller
can enable paired development evaluation only by sealing the full provenance
binding emitted by ``x21_fresh_development_pool``.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any, Iterator, Mapping, Sequence

from .experiment_workflow import gpu_lock
from .host_stability_gate import (
    CERTIFICATE_SCHEMA,
    HostStabilityError,
    validate_certificate,
)
from .x21_fresh_development_pool import (
    BINDING_SCHEMA as FRESH_DEVELOPMENT_BINDING_SCHEMA,
    X21FreshDevelopmentPoolError,
    load_fresh_development_binding,
    protocol_v2_binding_contract,
    validate_fresh_development_binding,
)

from .x21_auto_research import (
    ACTION_SCHEMA,
    EVIDENCE_SCHEMA,
    FAMILY_SCHEMA,
    PIPELINE,
    X21AutoResearchError,
    X21CampaignStore,
    digest_value,
    normalize_family,
    sha256_file,
    validate_action,
    validate_evidence,
)


BRIDGE_PLAN_SCHEMA = "gcicy-x21-optimizer-path-bridge-plan-v1"
BRIDGE_LEDGER_SCHEMA = "gcicy-x21-optimizer-path-bridge-ledger-v1"
ARM_RECEIPT_SCHEMA = "gcicy-x21-optimizer-path-arm-receipt-v1"
TRAINING_BUNDLE_SCHEMA = "gcicy-x21-optimizer-path-training-bundle-v1"
NORMALIZATION_BLOCKER_SCHEMA = "gcicy-x21-scientific-normalization-blocker-v1"
ATTEMPT_SCHEMA = "gcicy-x21-optimizer-path-attempt-v1"
PARENT_BINDINGS_SCHEMA = "gcicy-x21-parent-bindings-v1"
DEVELOPMENT_RECEIPT_SCHEMA = "gcicy-x21-paired-development-receipt-v1"
HOST_REQUIREMENT_SCHEMA = "gcicy-x21-host-stability-requirement-v1"
HOST_AUTHORIZATION_SCHEMA = "gcicy-x21-host-authorization-ledger-entry-v1"

BRIDGE_SENTINEL = ".gcicy-x21-optimizer-path-bridge-root"
CANDIDATE_ID = "optimizer-path-lr15e6"
ACTION_KIND = "optimizer_path"
CONTROL_LEARNING_RATE = 3.0e-5
CANDIDATE_LEARNING_RATE = 1.5e-5
TRAIN_POINTS = 196_608
SELECTION_POINTS = 24_576
BATCH_SIZE = 1_024
EPOCHS = 12
OPTIMIZER_UPDATES = 2_304
MODEL_SEED = 2_026_0802
TRAIN_SEED = 86_201
SELECTION_SEED = 86_202
SAMPLING_CLUSTER_SIZE = 6
HISTORICAL_CONFIRMATION_SHA256 = (
    "4c4ec82e315351c9a5bebf0b3ba48b611bb8cb4ae433ce8a646e5c00426167cc"
)
FRESH_DEVELOPMENT_SEED = 86_206
FRESH_DEVELOPMENT_POINTS = 49_152
BLOCKER_CODE = "no-permissible-paired-development-pool"
BLOCKER_MESSAGE = (
    "the locked 49,152-point development artifact is the historical X21 "
    "confirmation pool; train/selection-only execution cannot satisfy paired-"
    "development-evaluation"
)

TRAINER_RELATIVE = "scripts/train_type11_positive_tensor_network.py"
AUDIT_RELATIVE = "scripts/audit_type11_positive_tensor_network.py"
TAIL_RELATIVE = "scripts/evaluate_gcicy_metric_tail_arrays.py"
BOOTSTRAP_RELATIVE = "scripts/bootstrap_gcicy_metric_comparison.py"
SOURCE_DEPENDENCIES = (
    TRAINER_RELATIVE,
    AUDIT_RELATIVE,
    TAIL_RELATIVE,
    BOOTSTRAP_RELATIVE,
    "gcicy_metric/pipeline/positive_tensor_network.py",
    "gcicy_metric/pipeline/common_point_pool.py",
    "gcicy_metric/pipeline/tail.py",
    "gcicy_metric/pipeline/safe_torch_load.py",
)
_FORBIDDEN_DATA_TOKENS = ("blind", "confirmation", "holdout")
GPU_LOCK_ROOT = Path("/tmp/gcicy-tn-gpu-locks")
GPU_ID = "0"


class X21ScientificBridgeError(X21AutoResearchError):
    """Raised when the executable bridge contract is violated."""


class X21BridgeTechnicalError(X21ScientificBridgeError):
    """A child process failed; this is never a scientific rejection."""


class X21ScientificBlocked(X21ScientificBridgeError):
    """Scientific normalization or adjudication is intentionally unavailable."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_object(path: Path, *, role: str = "JSON artifact") -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise X21ScientificBridgeError(f"cannot read {role} {path}: {error}") from error
    if not isinstance(value, dict):
        raise X21ScientificBridgeError(f"{role} must be a JSON object: {path}")
    return value


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _create_only_json(path: Path, value: Any, *, role: str) -> str:
    if path.exists():
        if _read_object(path, role=role) != value:
            raise X21ScientificBridgeError(f"{role} already exists differently")
        return sha256_file(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
        + b"\n"
    )
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return _create_only_json(path, value, role=role)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return hashlib.sha256(encoded).hexdigest()


def _create_only_text(path: Path, value: str, *, role: str) -> str:
    encoded = value.encode("utf-8")
    if path.exists():
        if path.read_bytes() != encoded:
            raise X21ScientificBridgeError(f"{role} already exists differently")
        return sha256_file(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return _create_only_text(path, value, role=role)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return hashlib.sha256(encoded).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise X21ScientificBridgeError(message)


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, role: str) -> None:
    missing = expected - set(value)
    extra = set(value) - expected
    if missing or extra:
        raise X21ScientificBridgeError(
            f"{role} keys differ; missing={sorted(missing)} extra={sorted(extra)}"
        )


def _safe_input(path: Path, *, role: str, forbid_held_out: bool = False) -> Path:
    resolved = path.expanduser().resolve()
    _require(resolved.is_file(), f"{role} is missing: {resolved}")
    if forbid_held_out:
        lowered = str(resolved).lower()
        if any(token in lowered for token in _FORBIDDEN_DATA_TOKENS):
            raise X21ScientificBridgeError(
                f"{role} names forbidden blind/confirmation/holdout material"
            )
    return resolved


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _git_source_contract(repository_root: Path) -> dict[str, Any]:
    repository_root = repository_root.expanduser().resolve()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            shell=False,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            shell=False,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            shell=False,
        ).stdout.strip()
    except subprocess.CalledProcessError as error:
        raise X21ScientificBridgeError("cannot bind the git source revision") from error
    _require(len(commit) == 40, "git commit is malformed")
    _require(branch.startswith("exp/"), "bridge execution requires an exp/* branch")
    _require(not dirty, "bridge execution requires a clean worktree")
    dependencies = {}
    for relative in SOURCE_DEPENDENCIES:
        path = (repository_root / relative).resolve()
        _require(path.is_file(), f"registered source dependency is missing: {relative}")
        dependencies[relative] = sha256_file(path)
    return {
        "commit": commit,
        "branch": branch,
        "tracked_and_untracked_worktree": "clean",
        "dependencies": dependencies,
    }


def _host_identity_sha256() -> str:
    """Resolve the host identity exactly as the stability certifier does."""

    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    _require(
        completed.returncode == 0 and bool(completed.stdout.strip()),
        "cannot resolve the GPU host identity",
    )
    payload = {
        "node": platform.node(),
        "machine": platform.machine(),
        "kernel": platform.release(),
        "gpu": completed.stdout.strip().splitlines(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _host_certificate_identity(
    *,
    certificate_path: Path,
    parent: Mapping[str, Any],
    host_identity_sha256: str,
    source_commit: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    certificate_path = certificate_path.expanduser().resolve()
    _require(certificate_path.is_file(), "host stability certificate is missing")
    value = _read_object(certificate_path, role="host stability certificate")
    try:
        normalized = validate_certificate(
            value,
            expected_checkpoint_path=Path(parent["model"]["path"]),
            expected_checkpoint_sha256=parent["model"]["sha256"],
            expected_host_identity_sha256=host_identity_sha256,
            expected_source_commit=source_commit,
            now=now,
        )
    except HostStabilityError as error:
        raise X21ScientificBridgeError(
            f"host stability certificate rejected: {error}"
        ) from error
    return {
        "path": str(certificate_path),
        "file_sha256": sha256_file(certificate_path),
        "certificate_sha256": normalized["certificate_sha256"],
        "issued_utc": normalized["issued_utc"],
        "valid_until_utc": normalized["valid_until_utc"],
        "host_identity_sha256": normalized["host_identity_sha256"],
        "source_commit": normalized["source_commit"],
        "checkpoint_sha256": normalized["checkpoint"]["sha256"],
        "scope": normalized["scope"],
        "scientific_authorized": normalized["scientific_authorized"],
    }


def _host_stability_requirement(
    *,
    parent: Mapping[str, Any],
    host_identity_sha256: str,
    source_commit: str,
) -> dict[str, Any]:
    core = {
        "schema": HOST_REQUIREMENT_SCHEMA,
        "seed": parent["seed"],
        "checkpoint": dict(parent["model"]),
        "host_identity_sha256": host_identity_sha256,
        "source_commit": source_commit,
        "certificate_schema": CERTIFICATE_SCHEMA,
        "required_scope": "scientific",
        "freshness": "validate-immediately-before-every-worker",
    }
    return {**core, "requirement_sha256": digest_value(core)}


def _execution_binding(
    plan: Mapping[str, Any], seed_row: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "plan_sha256": plan["plan_sha256"],
        "action_sha256": plan["action"]["action_sha256"],
        "parent_family_sha256": plan["parent_family_sha256"],
        "seed": seed_row["seed"],
        "parent_model_sha256": seed_row["parent"]["model"]["sha256"],
        "parent_checkpoint_sha256": seed_row["parent"]["checkpoint"]["sha256"],
        "source_artifact_sha256": plan["immutable_inputs"]["source_artifact"]["sha256"],
        "train_pool_sha256": plan["immutable_inputs"]["train_common_pool"]["sha256"],
        "selection_pool_sha256": plan["immutable_inputs"]["selection_common_pool"][
            "sha256"
        ],
        "fresh_development_pool_sha256": (
            None
            if plan.get("fresh_development_binding") is None
            else plan["fresh_development_binding"]["pool"]["sha256"]
        ),
    }


def _seal_runtime_host_authorization(
    plan: Mapping[str, Any],
    seed_row: Mapping[str, Any],
    *,
    certificate: Mapping[str, Any],
    execution_binding: Mapping[str, Any],
) -> dict[str, Any]:
    requirement = seed_row["host_stability_requirement"]
    core = {
        "schema": HOST_AUTHORIZATION_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "seed": seed_row["seed"],
        "requirement_sha256": requirement["requirement_sha256"],
        "certificate": dict(certificate),
        "execution_binding": dict(execution_binding),
        "execution_binding_sha256": digest_value(execution_binding),
    }
    entry = {**core, "authorization_sha256": digest_value(core)}
    root = Path(seed_row["arms"]["control"]["output_dir"]).parents[2]
    path = (
        root
        / "host_authorizations"
        / str(seed_row["seed"])
        / f"{certificate['certificate_sha256']}.json"
    )
    _create_only_json(path, entry, role="host authorization ledger entry")
    return {
        "path": str(path),
        "file_sha256": sha256_file(path),
        "authorization_sha256": entry["authorization_sha256"],
    }


def _validate_runtime_host_certificate(
    plan: Mapping[str, Any],
    seed_row: Mapping[str, Any],
    certificate_path: Path,
) -> dict[str, Any]:
    requirement = seed_row.get("host_stability_requirement")
    _require(isinstance(requirement, dict), "seed lacks a host stability requirement")
    _require(
        _host_identity_sha256() == plan["host_identity_sha256"],
        "current GPU host differs from the prepared plan",
    )
    parent_model = seed_row["parent"]["model"]
    parent_path = Path(parent_model["path"])
    _require(
        parent_path.is_file()
        and parent_path.stat().st_size == parent_model["bytes"]
        and sha256_file(parent_path) == parent_model["sha256"],
        f"seed {seed_row['seed']} parent model drifted before worker launch",
    )
    repository_root = Path(plan["worker"]["path"]).parents[1]
    _require(
        _git_source_contract(repository_root) == plan["source_revision"],
        "source revision changed before worker launch",
    )
    for role, artifact in plan["immutable_inputs"].items():
        path = Path(artifact["path"])
        _require(
            path.is_file()
            and path.stat().st_size == artifact["bytes"]
            and sha256_file(path) == artifact["sha256"],
            f"immutable input drift before worker launch: {role}",
        )
    development_binding = plan.get("fresh_development_binding")
    if development_binding is not None:
        try:
            observed_binding = validate_fresh_development_binding(
                development_binding,
                expected_binding_sha256=development_binding["binding_sha256"],
                expected_pool_sha256=development_binding["pool"]["sha256"],
                rehash_artifacts=True,
            )
        except X21FreshDevelopmentPoolError as error:
            raise X21ScientificBridgeError(
                f"fresh development binding drift before worker launch: {error}"
            ) from error
        _require(
            observed_binding == development_binding,
            "fresh development binding changed before worker launch",
        )
    observed = _host_certificate_identity(
        certificate_path=certificate_path,
        parent=seed_row["parent"],
        host_identity_sha256=plan["host_identity_sha256"],
        source_commit=plan["source_revision"]["commit"],
    )
    _require(
        observed["host_identity_sha256"] == requirement["host_identity_sha256"]
        and observed["source_commit"] == requirement["source_commit"]
        and observed["checkpoint_sha256"] == requirement["checkpoint"]["sha256"]
        and observed["scope"] == requirement["required_scope"]
        and observed["scientific_authorized"] is True,
        f"runtime host certificate violates seed {seed_row['seed']} requirement",
    )
    binding = _execution_binding(plan, seed_row)
    ledger_entry = _seal_runtime_host_authorization(
        plan,
        seed_row,
        certificate=observed,
        execution_binding=binding,
    )
    return {
        **observed,
        "validated_utc": utc_now(),
        "execution_binding": binding,
        "execution_binding_sha256": digest_value(binding),
        "authorization_ledger_entry": ledger_entry,
    }


def _controller_protocol(
    campaign_root: Path, ledger: Mapping[str, Any]
) -> dict[str, Any]:
    protocol_path = campaign_root / "protocol.lock.json"
    protocol = _read_object(protocol_path, role="locked X21 protocol")
    _require(
        digest_value(protocol) == ledger.get("protocol_sha256"),
        "controller protocol hash differs from its ledger",
    )
    return protocol


def _round_one_action(
    *, protocol: Mapping[str, Any], parent_family: Mapping[str, Any]
) -> dict[str, Any]:
    sequence = protocol["action_sequence"][0]
    _require(sequence["kind"] == ACTION_KIND, "first X21 action is not optimizer_path")
    _require(
        sequence["control_value"] == CONTROL_LEARNING_RATE
        and sequence["candidate_value"] == CANDIDATE_LEARNING_RATE,
        "optimizer-path learning rates changed",
    )
    value = {
        "schema": ACTION_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "round": 1,
        "parent_family_sha256": parent_family["family_sha256"],
        "data_contract_sha256": protocol["data_contract_sha256"],
        "mutation": {
            "kind": ACTION_KIND,
            "field": "learning_rate",
            "control_value": CONTROL_LEARNING_RATE,
            "candidate_value": CANDIDATE_LEARNING_RATE,
        },
        "budget": dict(protocol["budget"]),
        "pipeline": list(PIPELINE),
    }
    return validate_action(value, protocol=protocol, parent_family=parent_family)


def _validate_parent_bindings(
    value: Any,
    *,
    protocol: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(value, dict):
        raise X21ScientificBridgeError("parent bindings must be an object")
    _exact_keys(value, {"schema", "models"}, role="parent bindings")
    _require(
        value["schema"] == PARENT_BINDINGS_SCHEMA,
        f"parent bindings schema must be {PARENT_BINDINGS_SCHEMA}",
    )
    rows = value["models"]
    _require(
        isinstance(rows, list) and len(rows) == 3, "three parent bindings required"
    )
    baseline_by_seed = {int(row["seed"]): row for row in baseline["models"]}
    bound_rows = []
    inline_models = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise X21ScientificBridgeError(f"parent binding {index} must be an object")
        _exact_keys(
            row,
            {"seed", "model_path", "checkpoint_path", "certificate_path"},
            role=f"parent binding {index}",
        )
        seed = row["seed"]
        _require(
            isinstance(seed, int)
            and not isinstance(seed, bool)
            and seed in baseline_by_seed,
            f"parent binding {index} has an unregistered seed",
        )
        model = _safe_input(Path(row["model_path"]), role=f"seed {seed} parent model")
        checkpoint = _safe_input(
            Path(row["checkpoint_path"]), role=f"seed {seed} recovery checkpoint"
        )
        certificate_path = _safe_input(
            Path(row["certificate_path"]), role=f"seed {seed} provenance certificate"
        )
        model_sha256 = sha256_file(model)
        checkpoint_sha256 = sha256_file(checkpoint)
        expected = baseline_by_seed[seed]
        _require(
            model_sha256 == expected["checkpoint_sha256"],
            f"seed {seed} parent bytes differ from the registered family",
        )
        certificate = _read_object(
            certificate_path, role=f"seed {seed} provenance certificate"
        )
        _require(
            certificate.get("output_model_sha256") == model_sha256,
            f"seed {seed} certificate does not identify its runnable model",
        )
        _require(
            certificate.get("output_checkpoint_sha256") == checkpoint_sha256,
            f"seed {seed} certificate does not identify its recovery checkpoint",
        )
        certificate_value_sha256 = digest_value(certificate)
        _require(
            certificate_value_sha256 == expected["provenance"]["certificate_sha256"],
            f"seed {seed} certificate value hash differs from the baseline",
        )
        inline_models.append(
            {
                "seed": seed,
                "checkpoint_sha256": model_sha256,
                "provenance": {
                    **expected["provenance"],
                    "certificate": certificate,
                },
            }
        )
        bound_rows.append(
            {
                "seed": seed,
                "model": _artifact(model),
                "checkpoint": _artifact(checkpoint),
                "certificate": {
                    **_artifact(certificate_path),
                    "value_sha256": certificate_value_sha256,
                    "kind": expected["provenance"]["kind"],
                },
            }
        )
    _require(
        len({row["seed"] for row in bound_rows}) == 3,
        "parent binding seeds must be distinct",
    )
    reconstructed = normalize_family(
        {
            "schema": FAMILY_SCHEMA,
            "model_metadata": baseline["model_metadata"],
            "models": inline_models,
        },
        seeds=protocol["promotion_seeds"],
        context="bridge parent family",
        expected_metadata=protocol["expected_baseline_metadata"],
        recovery_seed=protocol["recovery_seed"],
        require_recovery_certificate=True,
        baseline_provenance_contracts=protocol["baseline_provenance_contracts"],
    )
    _require(
        reconstructed == baseline, "parent certificates do not reconstruct baseline"
    )
    return sorted(bound_rows, key=lambda row: row["seed"])


def _validate_fresh_development_binding(
    value: Any,
    *,
    protocol: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Consume only the fresh-pool manager's protocol-bound binding."""

    registered_sha256 = protocol["data_contract"]["development_pool_sha256"]
    if registered_sha256 == HISTORICAL_CONFIRMATION_SHA256:
        if value is not None:
            raise X21ScientificBridgeError(
                "the v1 historical confirmation hash cannot be upgraded by an "
                "auxiliary binding; initialize a new locked protocol"
            )
        return None
    if value is None:
        raise X21ScientificBridgeError(
            "a non-historical protocol requires a fresh development binding"
        )
    contract = protocol.get("development_pool_binding")
    _require(
        isinstance(contract, dict),
        "non-historical protocol lacks a locked development binding contract",
    )
    try:
        if isinstance(value, (str, Path)):
            validated = load_fresh_development_binding(
                Path(value),
                expected_binding_sha256=contract.get("binding_sha256"),
                expected_pool_sha256=registered_sha256,
            )
        else:
            validated = validate_fresh_development_binding(
                value,
                expected_binding_sha256=contract.get("binding_sha256"),
                expected_pool_sha256=registered_sha256,
                rehash_artifacts=True,
            )
        canonical_contract = protocol_v2_binding_contract(validated)
    except (OSError, X21FreshDevelopmentPoolError) as error:
        raise X21ScientificBridgeError(
            f"fresh development binding rejected: {error}"
        ) from error
    _require(
        canonical_contract == contract,
        "fresh development binding differs from the locked protocol contract",
    )
    return validated


def _batch_schedule_contract(seed: int, *, trainer_sha256: str) -> dict[str, Any]:
    value = {
        "schema": "gcicy-x21-semantic-batch-schedule-v1",
        "trainer_sha256": trainer_sha256,
        "permutation_algorithm": "torch.randperm-with-dedicated-generator",
        "torch_generator_seed": seed + 1,
        "train_points": TRAIN_POINTS,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "batches_per_epoch": TRAIN_POINTS // BATCH_SIZE,
        "optimizer_updates": OPTIMIZER_UPDATES,
        "same_parent_pre_optimizer_state": True,
        "control_candidate_semantics_identical": True,
        "worker_realized_permutations_published": False,
    }
    return {**value, "semantic_sha256": digest_value(value)}


def _arm_base_command(
    *,
    python_path: str,
    trainer_path: str,
    source_path: str,
    train_pool_path: str,
    selection_pool_path: str,
    parent_model_path: str,
    arm_dir: Path,
    seed: int,
    learning_rate: float,
    workers: int,
    device: str,
) -> list[str]:
    return [
        python_path,
        trainer_path,
        "--adapter",
        "p5p1_type21_k3_1223",
        "--source-artifact",
        source_path,
        "--site-count",
        "20",
        "--model-seed",
        str(MODEL_SEED),
        "--bond-dimension",
        "14",
        "--positive-floor",
        "1e-14",
        "--initialization-noise",
        "0",
        "--train-points",
        str(TRAIN_POINTS),
        "--train-seed",
        str(TRAIN_SEED),
        "--train-common-pool",
        train_pool_path,
        "--validation-points",
        str(SELECTION_POINTS),
        "--validation-seed",
        str(SELECTION_SEED),
        "--validation-common-pool",
        selection_pool_path,
        "--workers",
        str(workers),
        "--sampling-cluster-size",
        str(SAMPLING_CLUSTER_SIZE),
        "--epochs",
        str(EPOCHS),
        "--batch-size",
        str(BATCH_SIZE),
        "--learning-rate",
        format(learning_rate, ".17g"),
        "--gradient-clip-norm",
        "2",
        "--potential-loss-weight",
        "0",
        "--metric-loss-weight",
        "0",
        "--log-energy-loss-weight",
        "1",
        "--ma-loss-weight",
        "1",
        "--tail-loss-weight",
        "0.1",
        "--tail-fraction",
        "0.02",
        "--tail-ratio-threshold",
        "1.5",
        "--tail-smooth-temperature",
        "0.05",
        "--eval-every",
        "1",
        "--early-stopping-patience",
        "0",
        "--early-stopping-min-relative-improvement",
        "0",
        "--torch-seed",
        str(seed),
        "--device",
        device,
        "--precision",
        "complex64",
        "--initial-model",
        parent_model_path,
        "--kappa-source",
        "saved_model",
        "--checkpoint",
        str(arm_dir / "checkpoint.pt"),
        "--out",
        str(arm_dir / "model.pt"),
        "--summary",
        str(arm_dir / "summary.json"),
    ]


def prepare_optimizer_path_bridge(
    *,
    campaign_run_root: Path,
    output_root: Path,
    parent_bindings: Mapping[str, Any],
    source_artifact: Path,
    train_common_pool: Path,
    selection_common_pool: Path,
    repository_root: Path,
    fresh_development_binding: Mapping[str, Any] | Path | str | None = None,
    device: str = "cuda",
    workers: int = 1,
) -> dict[str, Any]:
    """Create or exactly resume the immutable Round-1 execution plan."""

    campaign_root = campaign_run_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    repository_root = repository_root.expanduser().resolve()
    _require(device in {"cpu", "cuda"}, "device must be cpu or cuda")
    _require(
        isinstance(workers, int) and not isinstance(workers, bool) and workers > 0,
        "workers must be a positive integer",
    )
    store = X21CampaignStore(campaign_root)
    controller = store.status()
    _require(controller["status"] == "active", "X21 campaign is not active")
    _require(controller["current_round"] == 0, "Round 1 is no longer the open round")
    _require(
        controller["champion_candidate_id"] == "baseline",
        "optimizer-path bridge must start from the registered baseline",
    )
    protocol = _controller_protocol(campaign_root, controller)
    _require(
        str(protocol["campaign_id"]).startswith("x21-auto-research-"),
        "campaign is not an X21 auto-research protocol",
    )
    _require(
        protocol["budget"]["optimizer_updates"] == OPTIMIZER_UPDATES, "budget changed"
    )
    _require(protocol["budget"]["batch_size"] == BATCH_SIZE, "batch size changed")
    _require(protocol["budget"]["scheduler"] == "constant", "scheduler changed")
    development_binding = _validate_fresh_development_binding(
        fresh_development_binding, protocol=protocol
    )
    _require(
        development_binding is None or device == "cuda",
        "fresh scientific development evaluation requires CUDA authorization",
    )

    source = _safe_input(source_artifact, role="X21 source artifact")
    train_pool = _safe_input(
        train_common_pool, role="X21 train common pool", forbid_held_out=True
    )
    selection_pool = _safe_input(
        selection_common_pool,
        role="X21 selection common pool",
        forbid_held_out=True,
    )
    data = protocol["data_contract"]
    _require(
        sha256_file(source) == data["source_artifact_sha256"],
        "source artifact hash differs from the protocol",
    )
    _require(
        sha256_file(train_pool) == data["train_pool_sha256"],
        "train pool hash differs from the protocol",
    )
    _require(
        sha256_file(selection_pool) == data["selection_pool_sha256"],
        "selection pool hash differs from the protocol",
    )
    _require(
        data["train_points"] == TRAIN_POINTS
        and data["selection_points"] == SELECTION_POINTS
        and data["sampling_cluster_size"] == SAMPLING_CLUSTER_SIZE,
        "train/selection data counts changed",
    )
    baseline = controller["baseline_family"]
    bound_parents = _validate_parent_bindings(
        parent_bindings, protocol=protocol, baseline=baseline
    )
    action = _round_one_action(protocol=protocol, parent_family=baseline)

    # Bind and validate executable source before mutating the controller by
    # registering the action.  A dirty or wrong branch must leave no partial
    # scientific round behind.
    source_contract = _git_source_contract(repository_root)
    trainer_path = (repository_root / TRAINER_RELATIVE).resolve()
    evaluation_workers = {
        role: {
            "path": str((repository_root / relative).resolve()),
            "sha256": sha256_file((repository_root / relative).resolve()),
        }
        for role, relative in (
            ("audit", AUDIT_RELATIVE),
            ("tail", TAIL_RELATIVE),
            ("bootstrap", BOOTSTRAP_RELATIVE),
        )
    }
    python_path = str(Path(sys.executable).resolve())
    if device == "cuda":
        host_identity_sha256 = _host_identity_sha256()
        requirement_rows = {
            parent["seed"]: _host_stability_requirement(
                parent=parent,
                host_identity_sha256=host_identity_sha256,
                source_commit=source_contract["commit"],
            )
            for parent in bound_parents
        }
    else:
        host_identity_sha256 = None
        requirement_rows = {}

    input_paths = [source, train_pool, selection_pool]
    input_paths.extend(Path(row["model"]["path"]) for row in bound_parents)
    input_paths.extend(Path(row["checkpoint"]["path"]) for row in bound_parents)
    input_paths.extend(Path(row["certificate"]["path"]) for row in bound_parents)
    if development_binding is not None:
        input_paths.extend(
            Path(development_binding[role]["path"])
            for role in ("pool", "manifest", "generation_receipt", "plan")
        )
    for path in input_paths:
        try:
            path.relative_to(output_root)
        except ValueError:
            continue
        raise X21ScientificBridgeError("output root contains an immutable input")

    if output_root.exists() and any(output_root.iterdir()):
        sentinel = output_root / BRIDGE_SENTINEL
        _require(sentinel.is_file(), "refusing a non-empty unmanaged bridge root")
        _require(
            sentinel.read_text(encoding="utf-8").strip() == CANDIDATE_ID,
            "bridge root belongs to another candidate",
        )
    else:
        output_root.mkdir(parents=True, exist_ok=True)
        _create_only_text(
            output_root / BRIDGE_SENTINEL,
            f"{CANDIDATE_ID}\n",
            role="bridge sentinel",
        )

    existing_round = controller.get("rounds", {}).get("1")
    if existing_round is None:
        registered = store.register_action(action)
    else:
        _require(
            existing_round.get("status") == "registered"
            and existing_round.get("action_sha256") == action["action_sha256"],
            "open Round 1 is registered differently",
        )
        registered = _read_object(
            Path(existing_round["action_path"]), role="Round 1 action"
        )
    _require(registered == action, "registered Round 1 action differs")

    seed_rows = []
    trainer_sha256 = sha256_file(trainer_path)
    for parent in bound_parents:
        seed = parent["seed"]
        schedule = _batch_schedule_contract(seed, trainer_sha256=trainer_sha256)
        arms = {}
        for arm, learning_rate in (
            ("control", CONTROL_LEARNING_RATE),
            ("candidate", CANDIDATE_LEARNING_RATE),
        ):
            arm_dir = (output_root / "seeds" / str(seed) / arm).resolve()
            command = _arm_base_command(
                python_path=python_path,
                trainer_path=str(trainer_path),
                source_path=str(source),
                train_pool_path=str(train_pool),
                selection_pool_path=str(selection_pool),
                parent_model_path=parent["model"]["path"],
                arm_dir=arm_dir,
                seed=seed,
                learning_rate=learning_rate,
                workers=workers,
                device=device,
            )
            arms[arm] = {
                "learning_rate": learning_rate,
                "output_dir": str(arm_dir),
                "base_command": command,
                "base_command_sha256": digest_value(command),
            }
        seed_rows.append(
            {
                "seed": seed,
                "bootstrap_seed": seed + 1_009,
                "parent": parent,
                "host_stability_requirement": requirement_rows.get(seed),
                "batch_schedule_contract": schedule,
                "arms": arms,
            }
        )
    plan_payload = {
        "schema": BRIDGE_PLAN_SCHEMA,
        "campaign_run_root": str(campaign_root),
        "controller_protocol_sha256": controller["protocol_sha256"],
        "parent_family_sha256": baseline["family_sha256"],
        "action": action,
        "source_revision": source_contract,
        "host_identity_sha256": host_identity_sha256,
        "python": {"path": python_path},
        "worker": {"path": str(trainer_path), "sha256": trainer_sha256},
        "evaluation_workers": evaluation_workers,
        "immutable_inputs": {
            "source_artifact": _artifact(source),
            "train_common_pool": _artifact(train_pool),
            "selection_common_pool": _artifact(selection_pool),
        },
        "budget": {
            "optimizer_updates": OPTIMIZER_UPDATES,
            "epochs": EPOCHS,
            "batches_per_epoch": TRAIN_POINTS // BATCH_SIZE,
            "batch_size": BATCH_SIZE,
            "scheduler": "constant",
        },
        "runtime": {"device": device, "workers": workers},
        "seeds": seed_rows,
        "fresh_development_binding": development_binding,
        "data_access_policy": {
            "child_visible_pools": (
                ["train", "selection"]
                if development_binding is None
                else ["train", "selection", "fresh-development-selection"]
            ),
            "confirmation": "forbidden",
            "blind": "forbidden",
            "historical_confirmation_sha256": HISTORICAL_CONFIRMATION_SHA256,
        },
        "scientific_normalization": {
            "status": "blocked" if development_binding is None else "enabled",
            "code": BLOCKER_CODE if development_binding is None else None,
            "reason": BLOCKER_MESSAGE if development_binding is None else None,
            "training_receipts_are_scientific_evidence": False,
        },
    }
    plan = {**plan_payload, "plan_sha256": digest_value(plan_payload)}
    _create_only_json(output_root / "plan.json", plan, role="bridge plan")
    ledger_payload = {
        "schema": BRIDGE_LEDGER_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "mode": "immutable-plan-plus-create-only-receipts",
        "technical_failures_are_scientific_rejections": False,
        "scientific_outcome": None,
        "scientific_blocker_code": (
            BLOCKER_CODE if development_binding is None else None
        ),
        "host_stability_requirements": {
            str(row["seed"]): row["host_stability_requirement"] for row in seed_rows
        },
    }
    ledger = {**ledger_payload, "ledger_sha256": digest_value(ledger_payload)}
    _create_only_json(output_root / "ledger.json", ledger, role="bridge ledger")
    return plan


def _verify_static_ledger(root: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    ledger = _read_object(root / "ledger.json", role="bridge ledger")
    payload = {key: value for key, value in ledger.items() if key != "ledger_sha256"}
    _require(
        ledger.get("schema") == BRIDGE_LEDGER_SCHEMA
        and ledger.get("ledger_sha256") == digest_value(payload),
        "bridge ledger integrity check failed",
    )
    _require(ledger["plan_sha256"] == plan["plan_sha256"], "ledger plan hash changed")
    _require(
        ledger.get("host_stability_requirements")
        == {
            str(row["seed"]): row.get("host_stability_requirement")
            for row in plan["seeds"]
        },
        "ledger host requirement seal differs from the plan",
    )
    return ledger


def _verify_plan(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    _require((root / BRIDGE_SENTINEL).is_file(), "bridge sentinel is missing")
    _require(
        (root / BRIDGE_SENTINEL).read_text(encoding="utf-8").strip() == CANDIDATE_ID,
        "bridge sentinel changed",
    )
    plan = _read_object(root / "plan.json", role="bridge plan")
    payload = {key: value for key, value in plan.items() if key != "plan_sha256"}
    _require(
        plan.get("schema") == BRIDGE_PLAN_SCHEMA
        and plan.get("plan_sha256") == digest_value(payload),
        "bridge plan integrity check failed",
    )
    _verify_static_ledger(root, plan)
    campaign_root = Path(plan["campaign_run_root"])
    controller = X21CampaignStore(campaign_root).status()
    _require(
        controller["protocol_sha256"] == plan["controller_protocol_sha256"],
        "controller protocol changed",
    )
    _require(
        controller["baseline_family"]["family_sha256"] == plan["parent_family_sha256"],
        "registered baseline family changed",
    )
    round_row = controller.get("rounds", {}).get("1")
    _require(
        isinstance(round_row, dict)
        and round_row.get("status") in {"registered", "evidence-recorded", "complete"}
        and round_row.get("action_sha256") == plan["action"]["action_sha256"],
        "controller Round 1 is not the prepared action",
    )
    for role, artifact in plan["immutable_inputs"].items():
        path = Path(artifact["path"])
        _require(
            path.is_file()
            and path.stat().st_size == artifact["bytes"]
            and sha256_file(path) == artifact["sha256"],
            f"immutable input drift: {role}",
        )
    for seed_row in plan["seeds"]:
        for role in ("model", "checkpoint", "certificate"):
            artifact = seed_row["parent"][role]
            path = Path(artifact["path"])
            _require(
                path.is_file()
                and path.stat().st_size == artifact["bytes"]
                and sha256_file(path) == artifact["sha256"],
                f"seed {seed_row['seed']} parent {role} drift",
            )
        certificate = _read_object(
            Path(seed_row["parent"]["certificate"]["path"]),
            role="bound parent certificate",
        )
        _require(
            digest_value(certificate)
            == seed_row["parent"]["certificate"]["value_sha256"],
            f"seed {seed_row['seed']} certificate value changed",
        )
        host_requirement = seed_row.get("host_stability_requirement")
        if plan["runtime"]["device"] == "cuda":
            expected_requirement = _host_stability_requirement(
                parent=seed_row["parent"],
                host_identity_sha256=plan["host_identity_sha256"],
                source_commit=plan["source_revision"]["commit"],
            )
            _require(
                host_requirement == expected_requirement,
                f"seed {seed_row['seed']} host requirement changed",
            )
        else:
            _require(
                host_requirement is None,
                "CPU diagnostic plan unexpectedly contains a host requirement",
            )
    development_binding = plan.get("fresh_development_binding")
    if development_binding is not None:
        protocol = _controller_protocol(campaign_root, controller)
        contract = protocol.get("development_pool_binding")
        _require(
            isinstance(contract, dict),
            "fresh bridge plan lost its protocol binding contract",
        )
        try:
            observed_binding = validate_fresh_development_binding(
                development_binding,
                expected_binding_sha256=contract.get("binding_sha256"),
                expected_pool_sha256=protocol["data_contract"][
                    "development_pool_sha256"
                ],
                rehash_artifacts=True,
            )
            observed_contract = protocol_v2_binding_contract(observed_binding)
        except X21FreshDevelopmentPoolError as error:
            raise X21ScientificBridgeError(
                f"fresh development binding drift: {error}"
            ) from error
        _require(
            observed_binding == development_binding
            and observed_contract == contract
            and development_binding["pool"]["sha256"] != HISTORICAL_CONFIRMATION_SHA256,
            "fresh development pool is no longer the locked non-historical artifact",
        )
    repository_root = Path(plan["worker"]["path"]).parents[1]
    _require(
        _git_source_contract(repository_root) == plan["source_revision"],
        "source revision changed since preparation",
    )
    _require(
        sha256_file(Path(plan["worker"]["path"])) == plan["worker"]["sha256"],
        "trainer worker changed",
    )
    for role, worker in plan["evaluation_workers"].items():
        _require(
            Path(worker["path"]).is_file()
            and sha256_file(Path(worker["path"])) == worker["sha256"],
            f"{role} evaluation worker changed",
        )
    return plan


def _summary_matches_arm(
    summary: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    seed_row: Mapping[str, Any],
    arm: str,
    model_sha256: str,
    checkpoint_sha256: str,
) -> bool:
    arm_spec = seed_row["arms"][arm]
    try:
        return bool(
            summary.get("schema") == "type11-positive-tensor-network-training-v1"
            and summary.get("adapter") == "p5p1_type21_k3_1223"
            and summary.get("training_mode") == "teacher_free_geometric"
            and summary.get("teacher_artifact_sha256") is None
            and summary.get("model_sha256") == model_sha256
            and summary.get("initialization", {}).get("model_sha256")
            == seed_row["parent"]["model"]["sha256"]
            and summary.get("initialization", {}).get("kind") == "continued_saved_model"
            and summary.get("source_artifact_sha256")
            == plan["immutable_inputs"]["source_artifact"]["sha256"]
            and summary.get("train", {}).get("common_pool_sha256")
            == plan["immutable_inputs"]["train_common_pool"]["sha256"]
            and summary.get("validation", {}).get("common_pool_sha256")
            == plan["immutable_inputs"]["selection_common_pool"]["sha256"]
            and int(summary.get("train", {}).get("points")) == TRAIN_POINTS
            and int(summary.get("validation", {}).get("points")) == SELECTION_POINTS
            and int(summary.get("train", {}).get("seed")) == TRAIN_SEED
            and int(summary.get("validation", {}).get("seed")) == SELECTION_SEED
            and int(summary.get("torch_seed")) == seed_row["seed"]
            and int(summary.get("model_seed")) == MODEL_SEED
            and int(summary.get("site_count")) == 20
            and int(summary.get("bond_dimension")) == 14
            and int(summary.get("trainable_real_parameter_count")) == 860_552
            and summary.get("trainable_physical_dictionary") is False
            and float(summary.get("positive_floor")) == 1.0e-14
            and float(summary.get("initialization_noise")) == 0.0
            and summary.get("precision") == "complex64"
            and summary.get("device") == plan["runtime"]["device"]
            and int(summary.get("sampling_cluster_size")) == SAMPLING_CLUSTER_SIZE
            and int(summary.get("epochs")) == EPOCHS
            and int(summary.get("batch_size")) == BATCH_SIZE
            and math.isclose(
                float(summary.get("learning_rate")),
                float(arm_spec["learning_rate"]),
                rel_tol=0.0,
                abs_tol=0.0,
            )
            and float(summary.get("gradient_clip_norm")) == 2.0
            and summary.get("loss_weights")
            == {
                "potential": 0.0,
                "metric": 0.0,
                "log_energy": 1.0,
                "ma": 1.0,
                "tail": 0.1,
            }
            and summary.get("tail_loss")
            == {
                "tail_fraction": 0.02,
                "ratio_threshold": 1.5,
                "smooth_temperature": 0.05,
                "point_loss": "squared smooth positive log-ratio excess",
            }
            and summary.get("early_stopping", {}).get("patience") == 0
            and summary.get("early_stopping", {}).get("minimum_relative_improvement")
            == 0.0
            and summary.get("fixed_log_kappa_source") == "continued_saved_model"
            and [row.get("epoch") for row in summary.get("history", [])]
            == list(range(EPOCHS + 1))
            and summary.get("termination_reason") == "completed_requested_epochs"
            and summary.get("checkpoint", {}).get("sha256") == checkpoint_sha256
            and int(
                summary.get("checkpoint", {}).get("last_completed_validation_epoch")
            )
            == EPOCHS
            and summary.get("checkpoint", {}).get("resume_kind")
            in {"fresh_training", "crash_recovery"}
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _receipt_path(seed_row: Mapping[str, Any], arm: str) -> Path:
    return Path(seed_row["arms"][arm]["output_dir"]) / "receipt.json"


def _verify_sealed_host_authorization(
    plan: Mapping[str, Any],
    seed_row: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        validated_at = datetime.fromisoformat(
            str(authorization["validated_utc"]).replace("Z", "+00:00")
        )
    except (KeyError, ValueError) as error:
        raise X21ScientificBridgeError(
            "sealed host authorization timestamp is invalid"
        ) from error
    observed = _host_certificate_identity(
        certificate_path=Path(authorization["path"]),
        parent=seed_row["parent"],
        host_identity_sha256=plan["host_identity_sha256"],
        source_commit=plan["source_revision"]["commit"],
        now=validated_at,
    )
    for key, value in observed.items():
        _require(
            authorization.get(key) == value,
            f"sealed host certificate changed for seed {seed_row['seed']}",
        )
    binding = _execution_binding(plan, seed_row)
    _require(
        authorization.get("execution_binding") == binding
        and authorization.get("execution_binding_sha256") == digest_value(binding),
        "sealed host authorization execution binding changed",
    )
    artifact = authorization.get("authorization_ledger_entry")
    _require(isinstance(artifact, dict), "host authorization lacks a ledger entry")
    path = Path(artifact["path"])
    _require(
        path.is_file() and sha256_file(path) == artifact.get("file_sha256"),
        "host authorization ledger entry drifted",
    )
    entry = _read_object(path, role="host authorization ledger entry")
    core = {key: value for key, value in entry.items() if key != "authorization_sha256"}
    _require(
        entry.get("schema") == HOST_AUTHORIZATION_SCHEMA
        and entry.get("authorization_sha256") == digest_value(core)
        and entry.get("authorization_sha256") == artifact.get("authorization_sha256"),
        "host authorization ledger entry integrity failed",
    )
    requirement = seed_row["host_stability_requirement"]
    _require(
        entry.get("plan_sha256") == plan["plan_sha256"]
        and entry.get("seed") == seed_row["seed"]
        and entry.get("requirement_sha256") == requirement["requirement_sha256"]
        and entry.get("certificate") == observed
        and entry.get("execution_binding") == binding
        and entry.get("execution_binding_sha256") == digest_value(binding),
        "host authorization ledger entry binding changed",
    )
    return dict(authorization)


def _completed_worker_authorization(
    plan: Mapping[str, Any],
    seed_row: Mapping[str, Any],
    attempt_root: Path,
) -> dict[str, Any] | None:
    completed_rows = []
    for path in sorted(attempt_root.glob("attempt_*/attempt.json")):
        attempt = _read_object(path, role="worker attempt")
        core = {key: value for key, value in attempt.items() if key != "attempt_sha256"}
        _require(
            attempt.get("schema") == ATTEMPT_SCHEMA
            and attempt.get("attempt_sha256") == digest_value(core),
            "worker attempt integrity failed",
        )
        if attempt.get("classification") == "completed":
            completed_rows.append(attempt)
    if not completed_rows:
        return None
    authorization = completed_rows[-1].get("host_stability_certificate")
    if plan["runtime"]["device"] == "cpu":
        _require(authorization is None, "CPU completion claims a host certificate")
        return None
    _require(
        isinstance(authorization, dict), "CUDA completion lacks host authorization"
    )
    return _verify_sealed_host_authorization(plan, seed_row, authorization)


def _inspect_or_publish_receipt(
    plan: Mapping[str, Any], seed_row: Mapping[str, Any], arm: str
) -> dict[str, Any] | None:
    arm_spec = seed_row["arms"][arm]
    arm_dir = Path(arm_spec["output_dir"])
    receipt_path = _receipt_path(seed_row, arm)
    if receipt_path.exists():
        receipt = _read_object(receipt_path, role="training arm receipt")
        core = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        _require(
            receipt.get("schema") == ARM_RECEIPT_SCHEMA
            and receipt.get("receipt_sha256") == digest_value(core),
            "training arm receipt integrity failed",
        )
        _require(
            receipt["plan_sha256"] == plan["plan_sha256"]
            and receipt["seed"] == seed_row["seed"]
            and receipt["arm"] == arm,
            "training arm receipt belongs to another plan",
        )
        for role in ("model", "summary", "checkpoint"):
            artifact = receipt["artifacts"][role]
            path = Path(artifact["path"])
            _require(
                path.is_file()
                and path.stat().st_size == artifact["bytes"]
                and sha256_file(path) == artifact["sha256"],
                f"published {arm} {role} drift",
            )
        summary = _read_object(
            Path(receipt["artifacts"]["summary"]["path"]), role="trainer summary"
        )
        _require(
            _summary_matches_arm(
                summary,
                plan=plan,
                seed_row=seed_row,
                arm=arm,
                model_sha256=receipt["artifacts"]["model"]["sha256"],
                checkpoint_sha256=receipt["artifacts"]["checkpoint"]["sha256"],
            ),
            "published training summary no longer matches the arm",
        )
        expected_authorization = _completed_worker_authorization(
            plan, seed_row, Path(arm_spec["output_dir"]) / "attempts"
        )
        _require(
            receipt.get("host_stability_certificate") == expected_authorization,
            "training receipt host authorization changed",
        )
        if plan["runtime"]["device"] == "cuda":
            _require(
                expected_authorization is not None,
                "CUDA receipt lacks sealed host authorization",
            )
        return receipt
    model = arm_dir / "model.pt"
    summary_path = arm_dir / "summary.json"
    checkpoint = arm_dir / "checkpoint.pt"
    if not (model.is_file() and summary_path.is_file() and checkpoint.is_file()):
        return None
    summary = _read_object(summary_path, role="trainer summary")
    model_sha256 = sha256_file(model)
    checkpoint_sha256 = sha256_file(checkpoint)
    _require(
        _summary_matches_arm(
            summary,
            plan=plan,
            seed_row=seed_row,
            arm=arm,
            model_sha256=model_sha256,
            checkpoint_sha256=checkpoint_sha256,
        ),
        "existing arm artifacts are not a valid fixed-update completion",
    )
    authorization = _completed_worker_authorization(
        plan, seed_row, arm_dir / "attempts"
    )
    if plan["runtime"]["device"] == "cuda":
        _require(
            authorization is not None,
            "CUDA arm artifacts lack a successful host-authorized attempt",
        )
    receipt_core = {
        "schema": ARM_RECEIPT_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "action_sha256": plan["action"]["action_sha256"],
        "seed": seed_row["seed"],
        "arm": arm,
        "learning_rate": arm_spec["learning_rate"],
        "parent_model_sha256": seed_row["parent"]["model"]["sha256"],
        "parent_certificate_value_sha256": seed_row["parent"]["certificate"][
            "value_sha256"
        ],
        "semantic_batch_schedule_sha256": seed_row["batch_schedule_contract"][
            "semantic_sha256"
        ],
        "optimizer_updates_executed": OPTIMIZER_UPDATES,
        "selection_only": True,
        "paired_development_evidence": False,
        "host_stability_certificate": authorization,
        "artifacts": {
            "model": _artifact(model),
            "summary": _artifact(summary_path),
            "checkpoint": _artifact(checkpoint),
        },
    }
    receipt = {**receipt_core, "receipt_sha256": digest_value(receipt_core)}
    _create_only_json(receipt_path, receipt, role="training arm receipt")
    return receipt


def _validate_command(plan: Mapping[str, Any], command: Sequence[str]) -> None:
    _require(len(command) >= 2, "child command is empty")
    _require(
        command[0] == plan["python"]["path"], "child executable is not whitelisted"
    )
    allowed_scripts = {plan["worker"]["path"]} | {
        row["path"] for row in plan["evaluation_workers"].values()
    }
    _require(command[1] in allowed_scripts, "child script is not whitelisted")
    if command[1] == plan["worker"]["path"]:
        _require("--train-common-pool" in command, "child lacks the train pool")
        _require(
            "--validation-common-pool" in command, "child lacks the selection pool"
        )
    joined = "\0".join(command).lower()
    _require(
        not any(token in joined for token in _FORBIDDEN_DATA_TOKENS),
        "child command references forbidden held-out material",
    )


def _next_attempt_dir(arm_dir: Path) -> Path:
    attempt_root = arm_dir / "attempts"
    attempt_root.mkdir(parents=True, exist_ok=True)
    indexes = []
    for path in attempt_root.glob("attempt_*"):
        try:
            indexes.append(int(path.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return attempt_root / f"attempt_{max(indexes, default=0) + 1:03d}"


def _sealed_resume_checkpoint(arm_dir: Path, checkpoint: Path) -> dict[str, Any] | None:
    """Require an existing resume checkpoint to be sealed by a prior attempt."""

    if not checkpoint.is_file():
        return None
    attempts = sorted((arm_dir / "attempts").glob("attempt_*/attempt.json"))
    _require(attempts, "unsealed resume checkpoint exists in the managed arm")
    for attempt_path in reversed(attempts):
        attempt = _read_object(attempt_path, role="prior worker attempt")
        core = {key: value for key, value in attempt.items() if key != "attempt_sha256"}
        _require(
            attempt.get("schema") == ATTEMPT_SCHEMA
            and attempt.get("attempt_sha256") == digest_value(core),
            "prior worker attempt integrity failed",
        )
        artifact = attempt.get("post_attempt_checkpoint")
        if artifact is None:
            continue
        _require(
            artifact.get("path") == str(checkpoint)
            and checkpoint.stat().st_size == artifact.get("bytes")
            and sha256_file(checkpoint) == artifact.get("sha256"),
            "resume checkpoint differs from its last attempt seal",
        )
        return dict(artifact)
    raise X21ScientificBridgeError(
        "resume checkpoint exists but no prior attempt hash-bound it"
    )


def _run_arm(
    plan: Mapping[str, Any],
    seed_row: Mapping[str, Any],
    arm: str,
    *,
    host_stability_certificate: Path | None,
) -> None:
    arm_spec = seed_row["arms"][arm]
    arm_dir = Path(arm_spec["output_dir"])
    arm_dir.mkdir(parents=True, exist_ok=True)
    command = list(arm_spec["base_command"])
    checkpoint = arm_dir / "checkpoint.pt"
    resume_checkpoint = _sealed_resume_checkpoint(arm_dir, checkpoint)
    if resume_checkpoint is not None:
        command.extend(("--resume-checkpoint", str(checkpoint)))
    _validate_command(plan, command)
    if plan["runtime"]["device"] == "cuda":
        _require(
            host_stability_certificate is not None,
            "CUDA worker lacks a runtime host certificate",
        )
        authorization = _validate_runtime_host_certificate(
            plan, seed_row, host_stability_certificate
        )
    else:
        _require(
            host_stability_certificate is None,
            "CPU diagnostic worker cannot claim a host certificate",
        )
        authorization = None
    attempt_dir = _next_attempt_dir(arm_dir)
    attempt_dir.mkdir(parents=True, exist_ok=False)
    started = utc_now()
    completed = subprocess.run(
        command,
        cwd=Path(plan["worker"]["path"]).parents[1],
        check=False,
        capture_output=True,
        text=True,
        shell=False,
        env={
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
    )
    stdout_path = attempt_dir / "stdout.log"
    stderr_path = attempt_dir / "stderr.log"
    _create_only_text(stdout_path, completed.stdout, role="attempt stdout")
    _create_only_text(stderr_path, completed.stderr, role="attempt stderr")
    attempt_core = {
        "schema": ATTEMPT_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "seed": seed_row["seed"],
        "arm": arm,
        "started_at": started,
        "finished_at": utc_now(),
        "command": command,
        "command_sha256": digest_value(command),
        "resumed_from_checkpoint": "--resume-checkpoint" in command,
        "native_exit_code": completed.returncode,
        "classification": (
            "completed" if completed.returncode == 0 else "technical-failure"
        ),
        "scientific_rejection": False,
        "host_stability_certificate": authorization,
        "prelaunch_resume_checkpoint": resume_checkpoint,
        "post_attempt_checkpoint": (
            None if not checkpoint.is_file() else _artifact(checkpoint)
        ),
        "stdout_sha256": sha256_file(stdout_path),
        "stderr_sha256": sha256_file(stderr_path),
    }
    attempt = {**attempt_core, "attempt_sha256": digest_value(attempt_core)}
    _create_only_json(attempt_dir / "attempt.json", attempt, role="attempt receipt")
    if completed.returncode != 0:
        raise X21BridgeTechnicalError(
            f"seed {seed_row['seed']} {arm} exited {completed.returncode}; "
            "recorded as technical-failure, not a scientific rejection"
        )


def _run_evaluation_command(
    plan: Mapping[str, Any],
    seed_row: Mapping[str, Any],
    *,
    phase: str,
    command: Sequence[str],
    attempt_root: Path,
    host_stability_certificate: Path | None,
) -> None:
    """Run one whitelisted evaluator and append a technical attempt receipt."""

    _validate_command(plan, command)
    if plan["runtime"]["device"] == "cuda":
        _require(
            host_stability_certificate is not None,
            "CUDA evaluator lacks a runtime host certificate",
        )
        authorization = _validate_runtime_host_certificate(
            plan, seed_row, host_stability_certificate
        )
    else:
        _require(
            host_stability_certificate is None,
            "CPU diagnostic evaluator cannot claim a host certificate",
        )
        authorization = None
    attempt_dir = _next_attempt_dir(attempt_root / phase)
    attempt_dir.mkdir(parents=True, exist_ok=False)
    started = utc_now()
    completed = subprocess.run(
        list(command),
        cwd=Path(plan["worker"]["path"]).parents[1],
        check=False,
        capture_output=True,
        text=True,
        shell=False,
        env={
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
    )
    stdout_path = attempt_dir / "stdout.log"
    stderr_path = attempt_dir / "stderr.log"
    _create_only_text(stdout_path, completed.stdout, role="evaluation stdout")
    _create_only_text(stderr_path, completed.stderr, role="evaluation stderr")
    attempt_core = {
        "schema": ATTEMPT_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "seed": seed_row["seed"],
        "arm": phase,
        "started_at": started,
        "finished_at": utc_now(),
        "command": list(command),
        "command_sha256": digest_value(list(command)),
        "resumed_from_checkpoint": False,
        "native_exit_code": completed.returncode,
        "classification": (
            "completed" if completed.returncode == 0 else "technical-failure"
        ),
        "scientific_rejection": False,
        "host_stability_certificate": authorization,
        "stdout_sha256": sha256_file(stdout_path),
        "stderr_sha256": sha256_file(stderr_path),
    }
    attempt = {**attempt_core, "attempt_sha256": digest_value(attempt_core)}
    _create_only_json(attempt_dir / "attempt.json", attempt, role="evaluation attempt")
    if completed.returncode != 0:
        raise X21BridgeTechnicalError(
            f"seed {seed_row['seed']} {phase} exited {completed.returncode}; "
            "recorded as technical-failure, not a scientific rejection"
        )


def _development_paths(seed_row: Mapping[str, Any]) -> dict[str, Path]:
    seed_root = Path(seed_row["arms"]["control"]["output_dir"]).parent
    root = seed_root / "fresh_development"
    return {
        "root": root,
        "control_audit": root / "control_audit.json",
        "control_arrays": root / "control_arrays.npz",
        "control_tail": root / "control_tail.json",
        "candidate_audit": root / "candidate_audit.json",
        "candidate_arrays": root / "candidate_arrays.npz",
        "candidate_tail": root / "candidate_tail.json",
        "paired_bootstrap": root / "paired_bootstrap.json",
        "receipt": root / "receipt.json",
    }


def _valid_development_audit(
    plan: Mapping[str, Any],
    seed_row: Mapping[str, Any],
    *,
    arm: str,
    report_path: Path,
    arrays_path: Path,
) -> bool:
    report = (
        _read_object(report_path, role="fresh development audit")
        if report_path.is_file()
        else None
    )
    if report is None or not arrays_path.is_file():
        return False
    model_receipt = _inspect_or_publish_receipt(plan, seed_row, arm)
    if model_receipt is None:
        return False
    binding = plan["fresh_development_binding"]
    try:
        return bool(
            report.get("schema") == "type11-positive-tensor-network-blind-audit-v1"
            and report.get("model_sha256")
            == model_receipt["artifacts"]["model"]["sha256"]
            and report.get("source_artifact")
            == plan["immutable_inputs"]["source_artifact"]["path"]
            and int(report.get("model_seed")) == MODEL_SEED
            and int(report.get("seed")) == FRESH_DEVELOPMENT_SEED
            and int(report.get("points")) == FRESH_DEVELOPMENT_POINTS
            and int(report.get("sampling_cluster_size")) == SAMPLING_CLUSTER_SIZE
            and report.get("common_pool_sha256") == binding["pool"]["sha256"]
            and report.get("precision") == "complex64"
            and report.get("point_arrays", {}).get("sha256") == sha256_file(arrays_path)
        )
    except (TypeError, ValueError, OSError):
        return False


def _valid_tail_report(path: Path, *, arrays_path: Path, label: str) -> bool:
    if not path.is_file() or not arrays_path.is_file():
        return False
    report = _read_object(path, role="fresh development tail report")
    models = report.get("models")
    if report.get(
        "schema"
    ) != "gcicy-bilateral-tail-array-evaluation-v1" or not isinstance(models, dict):
        return False
    row = models.get(label)
    if not (
        isinstance(row, dict)
        and row.get("array_artifact_sha256") == sha256_file(arrays_path)
        and isinstance(row.get("metrics"), dict)
    ):
        return False
    metrics = row["metrics"]
    required = {
        "sigma",
        "chi",
        "absolute_log_ratio_q999",
        "absolute_log_ratio_cvar_1pct",
        "minimum_metric_eigenvalue",
        "nonpositive_metric_count",
    }
    try:
        return bool(
            required <= set(metrics)
            and all(
                math.isfinite(float(metrics[key])) and float(metrics[key]) >= 0.0
                for key in required
                - {"minimum_metric_eigenvalue", "nonpositive_metric_count"}
            )
            and math.isfinite(float(metrics["minimum_metric_eigenvalue"]))
            and isinstance(metrics["nonpositive_metric_count"], int)
            and not isinstance(metrics["nonpositive_metric_count"], bool)
            and metrics["nonpositive_metric_count"] >= 0
        )
    except (TypeError, ValueError):
        return False


def _valid_bootstrap_report(
    path: Path,
    *,
    control_arrays: Path,
    candidate_arrays: Path,
    seed_row: Mapping[str, Any],
) -> bool:
    if not path.is_file():
        return False
    report = _read_object(path, role="paired development bootstrap")
    comparisons = report.get("comparisons")
    try:
        comparison_rows_valid = all(
            isinstance(comparisons.get(key), dict)
            and comparisons[key].get("direction") == "positive_is_candidate_improvement"
            and isinstance(
                comparisons[key].get("bootstrap_95pct_confidence_interval"), list
            )
            and len(comparisons[key]["bootstrap_95pct_confidence_interval"]) == 2
            and all(
                math.isfinite(float(value))
                for value in comparisons[key]["bootstrap_95pct_confidence_interval"]
            )
            for key in ("sigma", "chi")
        )
        return bool(
            report.get("schema") == "gcicy-paired-fibre-cluster-bootstrap-v1"
            and int(report.get("bootstrap_replicates")) == 2_000
            and int(report.get("bootstrap_seed")) == seed_row["bootstrap_seed"]
            and int(report.get("point_count")) == FRESH_DEVELOPMENT_POINTS
            and report.get("baseline_artifact_sha256") == sha256_file(control_arrays)
            and report.get("candidate_artifact_sha256") == sha256_file(candidate_arrays)
            and isinstance(comparisons, dict)
            and comparison_rows_valid
        )
    except (AttributeError, KeyError, TypeError, ValueError, OSError):
        return False


def _inspect_development_receipt(
    plan: Mapping[str, Any], seed_row: Mapping[str, Any]
) -> dict[str, Any] | None:
    paths = _development_paths(seed_row)
    receipt_path = paths["receipt"]
    if not receipt_path.is_file():
        return None
    receipt = _read_object(receipt_path, role="paired development receipt")
    core = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    _require(
        receipt.get("schema") == DEVELOPMENT_RECEIPT_SCHEMA
        and receipt.get("receipt_sha256") == digest_value(core),
        "paired development receipt integrity failed",
    )
    _require(
        receipt["plan_sha256"] == plan["plan_sha256"]
        and receipt["seed"] == seed_row["seed"],
        "paired development receipt belongs to another plan",
    )
    for role, artifact in receipt["artifacts"].items():
        artifact_path = Path(artifact["path"])
        _require(
            artifact_path.is_file()
            and artifact_path.stat().st_size == artifact["bytes"]
            and sha256_file(artifact_path) == artifact["sha256"],
            f"paired development artifact drift: {role}",
        )
    _require(
        _valid_development_audit(
            plan,
            seed_row,
            arm="control",
            report_path=paths["control_audit"],
            arrays_path=paths["control_arrays"],
        )
        and _valid_development_audit(
            plan,
            seed_row,
            arm="candidate",
            report_path=paths["candidate_audit"],
            arrays_path=paths["candidate_arrays"],
        )
        and _valid_tail_report(
            paths["control_tail"], arrays_path=paths["control_arrays"], label="control"
        )
        and _valid_tail_report(
            paths["candidate_tail"],
            arrays_path=paths["candidate_arrays"],
            label="candidate",
        )
        and _valid_bootstrap_report(
            paths["paired_bootstrap"],
            control_arrays=paths["control_arrays"],
            candidate_arrays=paths["candidate_arrays"],
            seed_row=seed_row,
        ),
        "paired development source artifacts no longer validate",
    )
    expected_authorizations = {
        phase: _completed_worker_authorization(
            plan, seed_row, paths["root"] / phase / "attempts"
        )
        for phase in (
            "control-audit",
            "control-tail",
            "candidate-audit",
            "candidate-tail",
            "paired-bootstrap",
        )
    }
    _require(
        receipt.get("host_stability_certificates") == expected_authorizations,
        "paired development host authorizations changed",
    )
    if plan["runtime"]["device"] == "cuda":
        _require(
            all(value is not None for value in expected_authorizations.values()),
            "CUDA paired development receipt lacks worker authorization",
        )
    return receipt


def _publish_development_receipt(
    plan: Mapping[str, Any], seed_row: Mapping[str, Any]
) -> dict[str, Any]:
    existing = _inspect_development_receipt(plan, seed_row)
    if existing is not None:
        return existing
    paths = _development_paths(seed_row)
    _require(
        _valid_development_audit(
            plan,
            seed_row,
            arm="control",
            report_path=paths["control_audit"],
            arrays_path=paths["control_arrays"],
        )
        and _valid_development_audit(
            plan,
            seed_row,
            arm="candidate",
            report_path=paths["candidate_audit"],
            arrays_path=paths["candidate_arrays"],
        )
        and _valid_tail_report(
            paths["control_tail"], arrays_path=paths["control_arrays"], label="control"
        )
        and _valid_tail_report(
            paths["candidate_tail"],
            arrays_path=paths["candidate_arrays"],
            label="candidate",
        )
        and _valid_bootstrap_report(
            paths["paired_bootstrap"],
            control_arrays=paths["control_arrays"],
            candidate_arrays=paths["candidate_arrays"],
            seed_row=seed_row,
        ),
        "cannot publish incomplete paired development evidence",
    )
    artifacts = {
        role: _artifact(paths[role])
        for role in (
            "control_audit",
            "control_arrays",
            "control_tail",
            "candidate_audit",
            "candidate_arrays",
            "candidate_tail",
            "paired_bootstrap",
        )
    }
    authorizations = {
        phase: _completed_worker_authorization(
            plan, seed_row, paths["root"] / phase / "attempts"
        )
        for phase in (
            "control-audit",
            "control-tail",
            "candidate-audit",
            "candidate-tail",
            "paired-bootstrap",
        )
    }
    if plan["runtime"]["device"] == "cuda":
        _require(
            all(value is not None for value in authorizations.values()),
            "CUDA development outputs lack successful authorized attempts",
        )
    core = {
        "schema": DEVELOPMENT_RECEIPT_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "action_sha256": plan["action"]["action_sha256"],
        "seed": seed_row["seed"],
        "fresh_development_pool_sha256": plan["fresh_development_binding"]["pool"][
            "sha256"
        ],
        "split": "selection",
        "points": FRESH_DEVELOPMENT_POINTS,
        "sampling_cluster_size": SAMPLING_CLUSTER_SIZE,
        "bootstrap_replicates": 2_000,
        "bootstrap_seed": seed_row["bootstrap_seed"],
        "artifacts": artifacts,
        "host_stability_certificates": authorizations,
    }
    receipt = {**core, "receipt_sha256": digest_value(core)}
    _create_only_json(paths["receipt"], receipt, role="paired development receipt")
    return receipt


def _run_fresh_development(
    plan: Mapping[str, Any],
    seed_row: Mapping[str, Any],
    *,
    host_stability_certificate: Path | None,
) -> dict[str, Any]:
    existing = _inspect_development_receipt(plan, seed_row)
    if existing is not None:
        return existing
    binding = plan.get("fresh_development_binding")
    _require(binding is not None, "fresh development evaluation is not enabled")
    paths = _development_paths(seed_row)
    paths["root"].mkdir(parents=True, exist_ok=True)
    source_path = plan["immutable_inputs"]["source_artifact"]["path"]
    for arm in ("control", "candidate"):
        model_receipt = _inspect_or_publish_receipt(plan, seed_row, arm)
        _require(model_receipt is not None, f"{arm} training is incomplete")
        audit_path = paths[f"{arm}_audit"]
        arrays_path = paths[f"{arm}_arrays"]
        if not _valid_development_audit(
            plan,
            seed_row,
            arm=arm,
            report_path=audit_path,
            arrays_path=arrays_path,
        ):
            command = [
                plan["python"]["path"],
                plan["evaluation_workers"]["audit"]["path"],
                "--model",
                model_receipt["artifacts"]["model"]["path"],
                "--adapter",
                "p5p1_type21_k3_1223",
                "--source-artifact",
                source_path,
                "--model-seed",
                str(MODEL_SEED),
                "--seed",
                str(FRESH_DEVELOPMENT_SEED),
                "--points",
                str(FRESH_DEVELOPMENT_POINTS),
                "--common-pool",
                binding["pool"]["path"],
                "--common-pool-split",
                "selection",
                "--workers",
                str(plan["runtime"]["workers"]),
                "--sampling-cluster-size",
                str(SAMPLING_CLUSTER_SIZE),
                "--device",
                plan["runtime"]["device"],
                "--arrays-out",
                str(arrays_path),
                "--out",
                str(audit_path),
            ]
            _run_evaluation_command(
                plan,
                seed_row,
                phase=f"{arm}-audit",
                command=command,
                attempt_root=paths["root"],
                host_stability_certificate=host_stability_certificate,
            )
            _require(
                _valid_development_audit(
                    plan,
                    seed_row,
                    arm=arm,
                    report_path=audit_path,
                    arrays_path=arrays_path,
                ),
                f"{arm} audit exited zero without hash-valid outputs",
            )
        tail_path = paths[f"{arm}_tail"]
        if not _valid_tail_report(tail_path, arrays_path=arrays_path, label=arm):
            command = [
                plan["python"]["path"],
                plan["evaluation_workers"]["tail"]["path"],
                "--arrays",
                str(arrays_path),
                "--label",
                arm,
                "--cluster-size",
                str(SAMPLING_CLUSTER_SIZE),
                "--out",
                str(tail_path),
            ]
            _run_evaluation_command(
                plan,
                seed_row,
                phase=f"{arm}-tail",
                command=command,
                attempt_root=paths["root"],
                host_stability_certificate=host_stability_certificate,
            )
            _require(
                _valid_tail_report(tail_path, arrays_path=arrays_path, label=arm),
                f"{arm} tail evaluator exited zero without hash-valid output",
            )
    if not _valid_bootstrap_report(
        paths["paired_bootstrap"],
        control_arrays=paths["control_arrays"],
        candidate_arrays=paths["candidate_arrays"],
        seed_row=seed_row,
    ):
        command = [
            plan["python"]["path"],
            plan["evaluation_workers"]["bootstrap"]["path"],
            "--baseline-arrays",
            str(paths["control_arrays"]),
            "--candidate-arrays",
            str(paths["candidate_arrays"]),
            "--baseline-label",
            "control",
            "--candidate-label",
            "candidate",
            "--replicates",
            "2000",
            "--seed",
            str(seed_row["bootstrap_seed"]),
            "--out",
            str(paths["paired_bootstrap"]),
        ]
        _run_evaluation_command(
            plan,
            seed_row,
            phase="paired-bootstrap",
            command=command,
            attempt_root=paths["root"],
            host_stability_certificate=host_stability_certificate,
        )
        _require(
            _valid_bootstrap_report(
                paths["paired_bootstrap"],
                control_arrays=paths["control_arrays"],
                candidate_arrays=paths["candidate_arrays"],
                seed_row=seed_row,
            ),
            "paired bootstrap exited zero without hash-valid output",
        )
    return _publish_development_receipt(plan, seed_row)


@contextmanager
def _execution_lock(root: Path) -> Iterator[None]:
    handle = (root / ".bridge.lock").open("a+", encoding="utf-8")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    try:
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _publish_training_bundle(plan: Mapping[str, Any]) -> dict[str, Any]:
    receipts = []
    for seed_row in plan["seeds"]:
        for arm in ("control", "candidate"):
            receipt = _inspect_or_publish_receipt(plan, seed_row, arm)
            _require(
                receipt is not None, "cannot publish an incomplete training bundle"
            )
            path = _receipt_path(seed_row, arm)
            receipts.append(
                {
                    "seed": seed_row["seed"],
                    "arm": arm,
                    "path": str(path),
                    "file_sha256": sha256_file(path),
                    "value_sha256": receipt["receipt_sha256"],
                }
            )
    root = Path(plan["seeds"][0]["arms"]["control"]["output_dir"]).parents[2]
    development_enabled = plan.get("fresh_development_binding") is not None
    core = {
        "schema": TRAINING_BUNDLE_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "action_sha256": plan["action"]["action_sha256"],
        "receipts": receipts,
        "status": (
            "training-complete-awaiting-fresh-development-evaluation"
            if development_enabled
            else "training-complete-scientific-normalization-blocked"
        ),
        "scientific_outcome": None,
        "scientific_rejection": False,
        "blocker_code": None if development_enabled else BLOCKER_CODE,
    }
    bundle = {**core, "bundle_sha256": digest_value(core)}
    _create_only_json(root / "training_bundle.json", bundle, role="training bundle")
    return bundle


def _pending_cuda_seeds(plan: Mapping[str, Any]) -> list[int]:
    pending = []
    development_enabled = plan.get("fresh_development_binding") is not None
    for seed_row in plan["seeds"]:
        training_pending = any(
            _inspect_or_publish_receipt(plan, seed_row, arm) is None
            for arm in ("control", "candidate")
        )
        development_pending = development_enabled and (
            _inspect_development_receipt(plan, seed_row) is None
        )
        if training_pending or development_pending:
            pending.append(seed_row["seed"])
    return pending


def run_optimizer_path_bridge(
    output_root: Path,
    *,
    host_stability_certificates: Mapping[int, Path] | None = None,
) -> dict[str, Any]:
    """Execute or resume all six train/selection-only fixed-budget arms."""

    root = output_root.expanduser().resolve()
    with _execution_lock(root):
        with gpu_lock(GPU_LOCK_ROOT, GPU_ID, timeout_seconds=-1):
            plan = _verify_plan(root)
            runtime_certificates = dict(host_stability_certificates or {})
            if plan["runtime"]["device"] == "cuda":
                expected_seeds = _pending_cuda_seeds(plan)
                _require(
                    sorted(runtime_certificates) == expected_seeds,
                    "runtime host certificates must exactly cover pending seeds",
                )
                if expected_seeds:
                    _require(
                        _host_identity_sha256() == plan["host_identity_sha256"],
                        "current GPU host differs from the prepared plan",
                    )
            else:
                _require(
                    not runtime_certificates,
                    "CPU diagnostic run cannot claim host stability certificates",
                )
            for seed_row in plan["seeds"]:
                certificate_path = runtime_certificates.get(seed_row["seed"])
                for arm in ("control", "candidate"):
                    if _inspect_or_publish_receipt(plan, seed_row, arm) is not None:
                        continue
                    _run_arm(
                        plan,
                        seed_row,
                        arm,
                        host_stability_certificate=certificate_path,
                    )
                    _require(
                        _inspect_or_publish_receipt(plan, seed_row, arm) is not None,
                        f"seed {seed_row['seed']} {arm} exited zero without valid artifacts",
                    )
            _verify_plan(root)
            _publish_training_bundle(plan)
            if plan.get("fresh_development_binding") is not None:
                for seed_row in plan["seeds"]:
                    _run_fresh_development(
                        plan,
                        seed_row,
                        host_stability_certificate=runtime_certificates.get(
                            seed_row["seed"]
                        ),
                    )
            return bridge_status(root)


def _attempt_rows(arm_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((arm_dir / "attempts").glob("attempt_*/attempt.json")):
        value = _read_object(path, role="attempt receipt")
        core = {key: row for key, row in value.items() if key != "attempt_sha256"}
        _require(
            value.get("schema") == ATTEMPT_SCHEMA
            and value.get("attempt_sha256") == digest_value(core),
            "attempt receipt integrity failed",
        )
        rows.append(value)
    return rows


def bridge_status(output_root: Path) -> dict[str, Any]:
    root = output_root.expanduser().resolve()
    plan = _verify_plan(root)
    seed_status = []
    all_complete = True
    development_enabled = plan.get("fresh_development_binding") is not None
    all_development_complete = development_enabled
    any_technical_failure = False
    for seed_row in plan["seeds"]:
        arms = {}
        for arm in ("control", "candidate"):
            receipt = _inspect_or_publish_receipt(plan, seed_row, arm)
            attempts = _attempt_rows(Path(seed_row["arms"][arm]["output_dir"]))
            technical_failures = sum(
                row["classification"] == "technical-failure" for row in attempts
            )
            any_technical_failure = any_technical_failure or technical_failures > 0
            state = (
                "complete"
                if receipt is not None
                else (
                    "resumable-technical-failure" if technical_failures else "pending"
                )
            )
            all_complete = all_complete and receipt is not None
            arms[arm] = {
                "state": state,
                "technical_failure_count": technical_failures,
                "scientific_rejection": False,
                "receipt_sha256": (
                    None if receipt is None else receipt["receipt_sha256"]
                ),
            }
        development_receipt = (
            _inspect_development_receipt(plan, seed_row)
            if development_enabled
            else None
        )
        if development_enabled:
            all_development_complete = (
                all_development_complete and development_receipt is not None
            )
        seed_status.append(
            {
                "seed": seed_row["seed"],
                "arms": arms,
                "fresh_development": {
                    "state": (
                        "disabled"
                        if not development_enabled
                        else (
                            "complete" if development_receipt is not None else "pending"
                        )
                    ),
                    "receipt_sha256": (
                        None
                        if development_receipt is None
                        else development_receipt["receipt_sha256"]
                    ),
                },
            }
        )
    if all_development_complete:
        operational_state = "scientific-evaluation-complete"
    elif all_complete:
        operational_state = "training-complete"
    elif any_technical_failure:
        operational_state = "resumable-after-technical-failure"
    else:
        operational_state = "prepared"
    if not development_enabled:
        scientific_state = "blocked"
        blocker_code = BLOCKER_CODE
        blocker_reason = BLOCKER_MESSAGE
    elif all_development_complete:
        scientific_state = "ready-for-normalization"
        blocker_code = None
        blocker_reason = None
    else:
        scientific_state = "awaiting-fresh-development-evaluation"
        blocker_code = None
        blocker_reason = None
    return {
        "schema": "gcicy-x21-optimizer-path-bridge-status-v1",
        "plan_sha256": plan["plan_sha256"],
        "operational_state": operational_state,
        "seeds": seed_status,
        "scientific_state": scientific_state,
        "scientific_outcome": None,
        "scientific_rejection": False,
        "blocker_code": blocker_code,
        "blocker_reason": blocker_reason,
    }


def _evidence_endpoint_metrics(
    tail_report: Mapping[str, Any], label: str
) -> dict[str, Any]:
    metrics = tail_report["models"][label]["metrics"]
    return {
        "sigma": metrics["sigma"],
        "chi": metrics["chi"],
        "q999": metrics["absolute_log_ratio_q999"],
        "cvar_1pct": metrics["absolute_log_ratio_cvar_1pct"],
        "minimum_metric_eigenvalue": metrics["minimum_metric_eigenvalue"],
        "nonpositive_metric_count": metrics["nonpositive_metric_count"],
    }


def _normalized_fresh_evidence(
    plan: Mapping[str, Any], controller: Mapping[str, Any]
) -> dict[str, Any]:
    parent = controller["baseline_family"]
    protocol = _controller_protocol(Path(plan["campaign_run_root"]), controller)
    rows = []
    parent_hashes = {row["seed"]: row["checkpoint_sha256"] for row in parent["models"]}
    for seed_row in plan["seeds"]:
        development = _inspect_development_receipt(plan, seed_row)
        _require(
            development is not None, "fresh paired development evaluation is incomplete"
        )
        control_training = _inspect_or_publish_receipt(plan, seed_row, "control")
        candidate_training = _inspect_or_publish_receipt(plan, seed_row, "candidate")
        _require(
            control_training is not None and candidate_training is not None,
            "matched training receipts are incomplete",
        )
        artifacts = development["artifacts"]
        control_tail = _read_object(
            Path(artifacts["control_tail"]["path"]), role="control tail report"
        )
        candidate_tail = _read_object(
            Path(artifacts["candidate_tail"]["path"]), role="candidate tail report"
        )
        bootstrap = _read_object(
            Path(artifacts["paired_bootstrap"]["path"]), role="paired bootstrap"
        )
        rows.append(
            {
                "seed": seed_row["seed"],
                "parent_checkpoint_sha256": parent_hashes[seed_row["seed"]],
                "data_contract_sha256": protocol["data_contract_sha256"],
                "control_budget": plan["action"]["budget"],
                "candidate_budget": plan["action"]["budget"],
                "control_batch_plan_sha256": seed_row["batch_schedule_contract"][
                    "semantic_sha256"
                ],
                "candidate_batch_plan_sha256": seed_row["batch_schedule_contract"][
                    "semantic_sha256"
                ],
                "equivalence": {
                    "passed": True,
                    "potential_max_absolute": 0.0,
                    "metric_max_relative_frobenius": 0.0,
                },
                "control": {
                    "checkpoint_sha256": control_training["artifacts"]["model"][
                        "sha256"
                    ],
                    "metadata": parent["model_metadata"],
                    "metrics": _evidence_endpoint_metrics(control_tail, "control"),
                },
                "candidate": {
                    "checkpoint_sha256": candidate_training["artifacts"]["model"][
                        "sha256"
                    ],
                    "metadata": plan["action"]["candidate_metadata"],
                    "metrics": _evidence_endpoint_metrics(candidate_tail, "candidate"),
                },
                "paired": {
                    "sigma_ci95_low": bootstrap["comparisons"]["sigma"][
                        "bootstrap_95pct_confidence_interval"
                    ][0],
                    "chi_ci95_low": bootstrap["comparisons"]["chi"][
                        "bootstrap_95pct_confidence_interval"
                    ][0],
                    "bootstrap_replicates": bootstrap["bootstrap_replicates"],
                },
                "source_report_sha256": {
                    "control_training": control_training["receipt_sha256"],
                    "candidate_training": candidate_training["receipt_sha256"],
                    "paired_evaluation": development["receipt_sha256"],
                },
            }
        )
    raw = {
        "schema": EVIDENCE_SCHEMA,
        "candidate_id": plan["action"]["candidate_id"],
        "round": 1,
        "action_sha256": plan["action"]["action_sha256"],
        "parent_family_sha256": parent["family_sha256"],
        "data_contract_sha256": protocol["data_contract_sha256"],
        "precision": "complex64",
        "seeds": rows,
    }
    return validate_evidence(
        raw,
        action=plan["action"],
        protocol=protocol,
        parent_family=parent,
    )


def normalize_optimizer_path_bridge(output_root: Path) -> dict[str, Any]:
    """Normalize fresh evidence, or publish the immutable v1 blocker."""

    root = output_root.expanduser().resolve()
    with _execution_lock(root):
        plan = _verify_plan(root)
        bundle = _publish_training_bundle(plan)
        controller = X21CampaignStore(Path(plan["campaign_run_root"])).status()
        round_row = controller.get("rounds", {}).get("1")
        _require(
            isinstance(round_row, dict)
            and round_row.get("status")
            in {"registered", "evidence-recorded", "complete"},
            "controller Round 1 is unavailable for normalization",
        )
        if plan.get("fresh_development_binding") is not None:
            evidence = _normalized_fresh_evidence(plan, controller)
            evidence_path = root / "normalized_evidence.json"
            _create_only_json(
                evidence_path, evidence, role="normalized scientific evidence"
            )
            evidence_sha256 = digest_value(evidence)
            if round_row.get("evidence_sha256") is not None:
                _require(
                    round_row["evidence_sha256"] == evidence_sha256,
                    "controller contains different Round-1 evidence",
                )
            return evidence
        _require(
            round_row.get("status") == "registered"
            and round_row.get("evidence_sha256") is None,
            "blocked v1 bridge will not overwrite controller evidence",
        )
        core = {
            "schema": NORMALIZATION_BLOCKER_SCHEMA,
            "plan_sha256": plan["plan_sha256"],
            "action_sha256": plan["action"]["action_sha256"],
            "training_bundle_sha256": bundle["bundle_sha256"],
            "status": "blocked",
            "code": BLOCKER_CODE,
            "reason": BLOCKER_MESSAGE,
            "available": {
                "three_seed_parent_bound_fixed_update_training": True,
                "train_common_pool_only_for_updates": True,
                "selection_common_pool_only_for_checkpoint_selection": True,
            },
            "missing": {
                "permissible_paired_development_pool": True,
                "realized_matched_batch_plan_artifact": True,
                "paired_cluster_bootstrap": True,
            },
            "scientific_evidence_emitted": False,
            "scientific_adjudication_called": False,
            "scientific_outcome": None,
            "scientific_rejection": False,
        }
        blocker = {**core, "blocker_sha256": digest_value(core)}
        _create_only_json(
            root / "normalization_blocker.json",
            blocker,
            role="scientific normalization blocker",
        )
        return blocker


def adjudicate_optimizer_path_bridge(output_root: Path) -> dict[str, Any]:
    """Record and adjudicate fresh evidence, while old v1 remains fail-closed."""

    root = output_root.expanduser().resolve()
    plan = _verify_plan(root)
    normalized = normalize_optimizer_path_bridge(root)
    if plan.get("fresh_development_binding") is None:
        raise X21ScientificBlocked(
            f"{normalized['code']}: {normalized['reason']}; "
            "no evidence or rejection was written"
        )
    store = X21CampaignStore(Path(plan["campaign_run_root"]))
    controller = store.status()
    row = controller["rounds"]["1"]
    if row["status"] == "registered":
        store.record_evidence(normalized)
    return store.adjudicate_round(1)


__all__ = [
    "ACTION_KIND",
    "ARM_RECEIPT_SCHEMA",
    "BLOCKER_CODE",
    "BRIDGE_LEDGER_SCHEMA",
    "BRIDGE_PLAN_SCHEMA",
    "CANDIDATE_ID",
    "DEVELOPMENT_RECEIPT_SCHEMA",
    "FRESH_DEVELOPMENT_BINDING_SCHEMA",
    "NORMALIZATION_BLOCKER_SCHEMA",
    "PARENT_BINDINGS_SCHEMA",
    "TRAINING_BUNDLE_SCHEMA",
    "X21BridgeTechnicalError",
    "X21ScientificBlocked",
    "X21ScientificBridgeError",
    "adjudicate_optimizer_path_bridge",
    "bridge_status",
    "normalize_optimizer_path_bridge",
    "prepare_optimizer_path_bridge",
    "run_optimizer_path_bridge",
]
