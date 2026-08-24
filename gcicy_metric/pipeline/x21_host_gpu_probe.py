"""Immutable CUDA health probes for the X21 k=20, D=14 parent.

This module is an operational prerequisite for X21 scientific Auto Research,
not a model-selection procedure.  A probe continues one registered cores-only
complex64 plateau model for exactly one full epoch (192 optimizer updates) on
the protocol's immutable train pool and performs the protocol's selection
evaluation.  It never exposes a confirmation, blind, or holdout pool.

Every run is bound to a clean ``exp/*`` source commit, one GPU host identity,
the normalized X21 protocol, the three immutable input hashes, and the parent
model hash.  The managed root is create-only apart from its sealed state
ledger.  Interrupted or failed optimizer execution is never replayed in
place.  Successful runs emit the generic ``gcicy-host-gpu-probe-v1`` receipt
accepted by :mod:`gcicy_metric.pipeline.host_stability_gate`.
"""

from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
from typing import Any, Mapping

import numpy as np

from .experiment_workflow import active_python_executable, gpu_lock
from .host_stability_gate import (
    GPU_PROBE_SCHEMA,
    HostStabilityError,
    normalize_gpu_probe,
)
from .safe_torch_load import safe_torch_load
from .x21_auto_research import (
    X21AutoResearchError,
    digest_value,
    sha256_file,
    validate_protocol,
)


PLAN_SCHEMA = "gcicy-x21-host-gpu-probe-plan-v1"
LEDGER_SCHEMA = "gcicy-x21-host-gpu-probe-ledger-v1"
WORKER_RESULT_SCHEMA = "gcicy-x21-host-gpu-probe-worker-result-v1"
RAW_REPORT_SCHEMA = "gcicy-x21-host-gpu-probe-report-v1"
ROOT_SENTINEL = ".gcicy-x21-host-gpu-probe-root"

ADAPTER = "p5p1_type21_k3_1223"
MODEL_SEED = 2_026_0802
SITE_COUNT = 20
BOND_DIMENSION = 14
PRECISION = "complex64"
PHYSICAL_DICTIONARY_RANK = 121
TRAINABLE_REAL_PARAMETERS = 860_552
TRAIN_POINTS = 196_608
TRAIN_SEED = 86_201
SELECTION_POINTS = 24_576
SELECTION_SEED = 86_202
SAMPLING_CLUSTER_SIZE = 6
BATCH_SIZE = 1_024
EPOCHS = 1
OPTIMIZER_UPDATES = TRAIN_POINTS // BATCH_SIZE

TRAINER_RELATIVE_PATH = "scripts/train_type11_positive_tensor_network.py"
WORKER_RELATIVE_PATH = "scripts/run_x21_host_gpu_probe.py"
SOURCE_RELATIVE_PATHS = (
    "gcicy_metric/pipeline/x21_host_gpu_probe.py",
    "gcicy_metric/pipeline/x21_auto_research.py",
    "gcicy_metric/pipeline/experiment_workflow.py",
    "gcicy_metric/pipeline/host_stability_gate.py",
    "gcicy_metric/pipeline/safe_torch_load.py",
    "gcicy_metric/pipeline/positive_tensor_network.py",
    "gcicy_metric/pipeline/common_point_pool.py",
    "gcicy_metric/pipeline/parallel_sampling.py",
    "gcicy_metric/pipeline/adapter.py",
    "gcicy_metric/pipeline/registry.py",
    "gcicy_metric/pipeline/adapters/__init__.py",
    "gcicy_metric/pipeline/adapters/p5p1_k3_21.py",
    "gcicy_metric/type21_candidate_p5p1_1223.py",
    TRAINER_RELATIVE_PATH,
    WORKER_RELATIVE_PATH,
)

GPU_LOCK_ROOT = Path("/tmp/gcicy-tn-gpu-locks")
GPU_ID = "0"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9_.-]*\Z")
_FORBIDDEN_DATA_TOKENS = ("blind", "confirmation", "holdout", "shadow")


class X21HostGPUProbeError(X21AutoResearchError):
    """Raised when an X21 host probe violates its immutable contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise X21HostGPUProbeError(message)


def _read_object(path: Path, *, role: str = "JSON artifact") -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise X21HostGPUProbeError(f"cannot read {role} {path}: {error}") from error
    if not isinstance(value, dict):
        raise X21HostGPUProbeError(f"{role} must be a JSON object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, role: str) -> None:
    missing = expected - set(value)
    extra = set(value) - expected
    if missing or extra:
        raise X21HostGPUProbeError(
            f"{role} fields differ; missing={sorted(missing)} extra={sorted(extra)}"
        )


def _safe_identifier(value: Any, *, role: str) -> str:
    result = str(value)
    _require(_IDENTIFIER.fullmatch(result) is not None, f"{role} is invalid")
    return result


def _safe_existing_file(
    path: Path, *, role: str, forbid_held_out: bool = False
) -> Path:
    result = path.expanduser().resolve()
    _require(result.is_file(), f"{role} is missing: {result}")
    if forbid_held_out:
        lowered = str(result).lower()
        _require(
            not any(token in lowered for token in _FORBIDDEN_DATA_TOKENS),
            f"{role} references forbidden held-out material",
        )
    return result


def _safe_root(path: Path, *, repository_root: Path) -> Path:
    result = path.expanduser().resolve()
    repository_root = repository_root.expanduser().resolve()
    _require(
        result != repository_root and repository_root not in result.parents,
        "probe output root must be outside the source checkout",
    )
    lowered = str(result).lower()
    _require(
        not any(token in lowered for token in _FORBIDDEN_DATA_TOKENS),
        "probe output root has a forbidden held-out label",
    )
    return result


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _finite(value: Any, *, role: str) -> float:
    if isinstance(value, bool):
        raise X21HostGPUProbeError(f"{role} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise X21HostGPUProbeError(f"{role} must be numeric") from error
    _require(math.isfinite(result), f"{role} must be finite")
    return result


def _positive_integer(value: Any, *, role: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value > 0,
        f"{role} must be a positive integer",
    )
    return int(value)


def _parse_utc(value: Any, *, role: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as error:
        raise X21HostGPUProbeError(f"{role} is not ISO-8601") from error
    _require(parsed.tzinfo is not None, f"{role} has no timezone")
    return parsed.astimezone(timezone.utc)


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _publish_create_only(path: Path, value: Mapping[str, Any], *, role: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            dict(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        _require(_read_object(path, role=role) == dict(value), f"{role} differs")
        return sha256_file(path)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(value), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _source_contract(repository_root: Path) -> dict[str, Any]:
    root = repository_root.expanduser().resolve()
    try:
        commit = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
                shell=False,
            )
            .stdout.strip()
            .lower()
        )
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            shell=False,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            shell=False,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise X21HostGPUProbeError("cannot bind the probe source checkout") from error
    _require(_COMMIT.fullmatch(commit) is not None, "source commit is malformed")
    _require(branch.startswith("exp/"), "host probe requires an exp/* branch")
    _require(not dirty, "host probe requires a completely clean source checkout")
    dependencies = {}
    for relative in SOURCE_RELATIVE_PATHS:
        path = (root / relative).resolve()
        _require(path.is_file(), f"source dependency is missing: {relative}")
        dependencies[relative] = sha256_file(path)
    return {
        "repository_root": str(root),
        "commit": commit,
        "branch": branch,
        "tracked_and_untracked_worktree": "clean",
        "dependencies": dependencies,
    }


def _host_identity_sha256() -> str:
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
    rows = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
    _require(
        completed.returncode == 0 and len(rows) == 1,
        "host probe requires exactly one inspectable CUDA GPU",
    )
    payload = {
        "node": platform.node(),
        "machine": platform.machine(),
        "kernel": platform.release(),
        "gpu": rows,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _runtime_environment() -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - production dependency
        raise X21HostGPUProbeError("PyTorch is unavailable") from error
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    rows = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
    _require(
        completed.returncode == 0 and len(rows) == 1,
        "runtime inventory requires exactly one inspectable CUDA GPU",
    )
    fields = [field.strip() for field in rows[0].split(",")]
    _require(len(fields) == 2, "CUDA runtime inventory is malformed")
    memory_mib = _positive_integer(int(fields[1]), role="GPU memory MiB")
    return {
        "numpy": str(np.__version__),
        "torch": str(torch.__version__),
        "torch_cuda_build": torch.version.cuda,
        "gpu_name": fields[0],
        "gpu_total_memory_bytes": memory_mib * 1024**2,
    }


def _protocol_contract(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = _read_object(path, role="X21 protocol")
    try:
        protocol = validate_protocol(raw)
    except X21AutoResearchError as error:
        raise X21HostGPUProbeError(f"X21 protocol is invalid: {error}") from error
    data = protocol["data_contract"]
    _require(
        data["train_points"] == TRAIN_POINTS
        and data["selection_points"] == SELECTION_POINTS
        and data["sampling_cluster_size"] == SAMPLING_CLUSTER_SIZE,
        "X21 train/selection dimensions differ from the probe contract",
    )
    return protocol, {
        **_artifact(path),
        "normalized_sha256": digest_value(protocol),
        "data_contract_sha256": protocol["data_contract_sha256"],
        "schema": protocol["schema"],
        "campaign_id": protocol["campaign_id"],
    }


def _load_torch_object(path: Path, *, role: str) -> dict[str, Any]:
    try:
        value = safe_torch_load(path, map_location="cpu")
    except (OSError, RuntimeError, TypeError, ValueError, ModuleNotFoundError) as error:
        raise X21HostGPUProbeError(f"cannot load {role} on CPU") from error
    _require(isinstance(value, dict), f"{role} is not a dictionary")
    return value


def _coefficient_state_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    state_dict = payload.get("state_dict")
    _require(isinstance(state_dict, Mapping), "parent model has no state dictionary")
    coefficient_names = tuple(
        f"coefficient_cores.{index}" for index in range(SITE_COUNT)
    )
    present_coefficient_names = {
        str(name) for name in state_dict if str(name).startswith("coefficient_cores.")
    }
    _require(
        present_coefficient_names == set(coefficient_names),
        "parent model coefficient-core inventory differs from k=20",
    )
    expected_shapes = (
        (1, BOND_DIMENSION, PHYSICAL_DICTIONARY_RANK),
        *((BOND_DIMENSION, BOND_DIMENSION, PHYSICAL_DICTIONARY_RANK),)
        * (SITE_COUNT - 2),
        (BOND_DIMENSION, 1, PHYSICAL_DICTIONARY_RANK),
    )
    complex_parameters = 0
    for name, expected_shape in zip(coefficient_names, expected_shapes, strict=True):
        tensor = state_dict[name]
        shape = tuple(int(dimension) for dimension in getattr(tensor, "shape", ()))
        _require(shape == expected_shape, f"parent model {name} has the wrong shape")
        _require(
            str(getattr(tensor, "dtype", "")) == "torch.complex64",
            f"parent model {name} is not complex64",
        )
        complex_parameters += math.prod(shape)

    dictionary_dimension = math.isqrt(PHYSICAL_DICTIONARY_RANK)
    _require(
        dictionary_dimension**2 == PHYSICAL_DICTIONARY_RANK,
        "physical dictionary rank is not square",
    )
    dictionary = state_dict.get("physical_dictionary")
    dictionary_shape = tuple(
        int(dimension) for dimension in getattr(dictionary, "shape", ())
    )
    _require(
        dictionary_shape
        == (
            PHYSICAL_DICTIONARY_RANK,
            dictionary_dimension,
            dictionary_dimension,
        ),
        "parent model physical dictionary has the wrong shape",
    )
    _require(
        str(getattr(dictionary, "dtype", "")) == "torch.complex64",
        "parent model physical dictionary is not complex64",
    )
    trainable_real_parameters = 2 * complex_parameters
    _require(
        trainable_real_parameters == TRAINABLE_REAL_PARAMETERS,
        "parent model structural trainable parameter count differs from 860552",
    )
    return {
        "coefficient_core_count": len(coefficient_names),
        "coefficient_core_shapes": [list(shape) for shape in expected_shapes],
        "physical_dictionary_shape": list(dictionary_shape),
        "trainable_real_parameter_count": trainable_real_parameters,
        "state_precision": PRECISION,
    }


def _parent_contract(path: Path, *, protocol: Mapping[str, Any]) -> dict[str, Any]:
    payload = _load_torch_object(path, role="parent model")
    expected = {
        "schema": "type11-positive-tensor-network-v1",
        "adapter": ADAPTER,
        "model_seed": MODEL_SEED,
        "site_count": SITE_COUNT,
        "bond_dimension": BOND_DIMENSION,
        "architecture": "shared_local_dictionary",
        "physical_dictionary_rank": PHYSICAL_DICTIONARY_RANK,
        "trainable_physical_dictionary": False,
        "physical_dictionary_gauge": "fixed",
        "precision": PRECISION,
    }
    for field, expected_value in expected.items():
        _require(
            payload.get(field) == expected_value,
            f"parent model {field} differs from {expected_value!r}",
        )
    coefficient_state = _coefficient_state_contract(payload)
    data = protocol["data_contract"]
    frozen = {
        "source_artifact_sha256": data["source_artifact_sha256"],
        "train_common_pool_sha256": data["train_pool_sha256"],
        "validation_common_pool_sha256": data["selection_pool_sha256"],
    }
    for field, expected_value in frozen.items():
        _require(
            payload.get(field) == expected_value,
            f"parent model {field} differs from the protocol",
        )
    return {**expected, **coefficient_state, **frozen}


def _runtime(value: Mapping[str, Any]) -> dict[str, Any]:
    _exact_keys(value, {"device", "workers", "teacher_chunk_size"}, role="runtime")
    _require(value["device"] == "cuda", "X21 host probes require CUDA")
    return {
        "device": "cuda",
        "workers": _positive_integer(value["workers"], role="runtime.workers"),
        "teacher_chunk_size": _positive_integer(
            value["teacher_chunk_size"], role="runtime.teacher_chunk_size"
        ),
    }


def _training_command(
    *,
    python_path: str,
    trainer_path: Path,
    source: Path,
    train_pool: Path,
    selection_pool: Path,
    parent_model: Path,
    training_root: Path,
    parent_seed: int,
    runtime: Mapping[str, Any],
) -> list[str]:
    return [
        python_path,
        str(trainer_path),
        "--adapter",
        ADAPTER,
        "--source-artifact",
        str(source),
        "--site-count",
        str(SITE_COUNT),
        "--model-seed",
        str(MODEL_SEED),
        "--bond-dimension",
        str(BOND_DIMENSION),
        "--positive-floor",
        "1e-14",
        "--initialization-noise",
        "0",
        "--train-points",
        str(TRAIN_POINTS),
        "--train-seed",
        str(TRAIN_SEED),
        "--train-common-pool",
        str(train_pool),
        "--validation-points",
        str(SELECTION_POINTS),
        "--validation-seed",
        str(SELECTION_SEED),
        "--validation-common-pool",
        str(selection_pool),
        "--workers",
        str(runtime["workers"]),
        "--sampling-cluster-size",
        str(SAMPLING_CLUSTER_SIZE),
        "--teacher-chunk-size",
        str(runtime["teacher_chunk_size"]),
        "--epochs",
        str(EPOCHS),
        "--batch-size",
        str(BATCH_SIZE),
        "--learning-rate",
        "3e-5",
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
        str(parent_seed),
        "--device",
        "cuda",
        "--precision",
        PRECISION,
        "--initial-model",
        str(parent_model),
        "--kappa-source",
        "saved_model",
        "--checkpoint",
        str(training_root / "checkpoint.pt"),
        "--out",
        str(training_root / "model.pt"),
        "--summary",
        str(training_root / "summary.json"),
    ]


def _seal_ledger(value: dict[str, Any]) -> None:
    payload = {key: row for key, row in value.items() if key != "state_sha256"}
    value["state_sha256"] = digest_value(payload)


def _write_ledger(path: Path, value: dict[str, Any]) -> None:
    _seal_ledger(value)
    _atomic_write_json(path, value)


def _read_ledger(path: Path) -> dict[str, Any]:
    value = _read_object(path, role="probe ledger")
    _exact_keys(
        value,
        {
            "schema",
            "plan_sha256",
            "state",
            "worker_returncode",
            "worker_stdout_sha256",
            "worker_stderr_sha256",
            "worker_result_sha256",
            "raw_report_sha256",
            "receipt_sha256",
            "state_sha256",
        },
        role="probe ledger",
    )
    payload = {key: row for key, row in value.items() if key != "state_sha256"}
    _require(
        value["schema"] == LEDGER_SCHEMA
        and value["state_sha256"] == digest_value(payload),
        "probe ledger integrity check failed",
    )
    return value


def prepare_probe(
    *,
    protocol_path: Path,
    source_artifact: Path,
    train_common_pool: Path,
    selection_common_pool: Path,
    parent_model: Path,
    parent_model_sha256: str,
    parent_seed: int,
    probe_id: str,
    output_root: Path,
    repository_root: Path,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    """Create or exactly resume one immutable X21 host-probe plan."""

    repository_root = repository_root.expanduser().resolve()
    output_root = _safe_root(output_root, repository_root=repository_root)
    probe_id = _safe_identifier(probe_id, role="probe_id")
    protocol_path = _safe_existing_file(protocol_path, role="X21 protocol")
    source = _safe_existing_file(source_artifact, role="X21 source artifact")
    train_pool = _safe_existing_file(
        train_common_pool, role="X21 train pool", forbid_held_out=True
    )
    selection_pool = _safe_existing_file(
        selection_common_pool, role="X21 selection pool", forbid_held_out=True
    )
    parent = _safe_existing_file(parent_model, role="X21 parent model")
    _require(
        _SHA256.fullmatch(str(parent_model_sha256)) is not None,
        "parent model SHA-256 is malformed",
    )
    _require(
        sha256_file(parent) == parent_model_sha256,
        "parent model hash differs from the registered value",
    )
    protocol, protocol_artifact = _protocol_contract(protocol_path)
    _require(parent_seed in protocol["promotion_seeds"], "parent seed is unregistered")
    data = protocol["data_contract"]
    observed_inputs = {
        "source_artifact": _artifact(source),
        "train_common_pool": _artifact(train_pool),
        "selection_common_pool": _artifact(selection_pool),
    }
    expected_hashes = {
        "source_artifact": data["source_artifact_sha256"],
        "train_common_pool": data["train_pool_sha256"],
        "selection_common_pool": data["selection_pool_sha256"],
    }
    for role, expected_hash in expected_hashes.items():
        _require(
            observed_inputs[role]["sha256"] == expected_hash,
            f"{role} hash differs from the protocol",
        )
    for immutable in (protocol_path, source, train_pool, selection_pool, parent):
        _require(
            output_root != immutable and output_root not in immutable.parents,
            "probe output root contains an immutable input",
        )

    normalized_runtime = _runtime(runtime)
    source_contract = _source_contract(repository_root)
    host_identity = _host_identity_sha256()
    parent_contract = _parent_contract(parent, protocol=protocol)
    runtime_environment = _runtime_environment()
    trainer_path = (repository_root / TRAINER_RELATIVE_PATH).resolve()
    worker_path = (repository_root / WORKER_RELATIVE_PATH).resolve()
    _require(trainer_path.is_file(), "registered X21 trainer is missing")
    _require(worker_path.is_file(), "registered X21 probe worker is missing")

    output_root.mkdir(parents=True, exist_ok=True)
    sentinel = output_root / ROOT_SENTINEL
    if sentinel.exists():
        _require(
            sentinel.read_text(encoding="utf-8").strip() == probe_id,
            "probe root belongs to another probe_id",
        )
    elif any(output_root.iterdir()):
        raise X21HostGPUProbeError("refusing a non-empty unmanaged probe root")
    else:
        try:
            descriptor = os.open(sentinel, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError as error:  # pragma: no cover - concurrent prepare
            raise X21HostGPUProbeError(
                "probe root was prepared concurrently"
            ) from error
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(f"{probe_id}\n")
            handle.flush()
            os.fsync(handle.fileno())

    training_root = (output_root / "training").resolve()
    python_path = active_python_executable()
    command = _training_command(
        python_path=python_path,
        trainer_path=trainer_path,
        source=source,
        train_pool=train_pool,
        selection_pool=selection_pool,
        parent_model=parent,
        training_root=training_root,
        parent_seed=parent_seed,
        runtime=normalized_runtime,
    )
    plan_payload = {
        "schema": PLAN_SCHEMA,
        "probe_id": probe_id,
        "protocol": protocol_artifact,
        "parent_seed": parent_seed,
        "parent_model": {
            **_artifact(parent),
            "contract": parent_contract,
        },
        "immutable_inputs": observed_inputs,
        "source_contract": source_contract,
        "host_identity_sha256": host_identity,
        "runtime_environment": runtime_environment,
        "runtime": normalized_runtime,
        "fixed_workload": {
            "site_count": SITE_COUNT,
            "bond_dimension": BOND_DIMENSION,
            "precision": PRECISION,
            "parameter_scope": "cores-only",
            "epochs": EPOCHS,
            "train_points": TRAIN_POINTS,
            "selection_points": SELECTION_POINTS,
            "batch_size": BATCH_SIZE,
            "optimizer_updates": OPTIMIZER_UPDATES,
        },
        "data_access_policy": {
            "allowed_splits": ["train", "selection"],
            "confirmation": "forbidden",
            "blind": "forbidden",
            "holdout": "forbidden",
        },
        "repository_root": str(repository_root),
        "python": {"path": python_path, "version": platform.python_version()},
        "worker": {"path": str(worker_path), "sha256": sha256_file(worker_path)},
        "trainer": {"path": str(trainer_path), "sha256": sha256_file(trainer_path)},
        "training_command": command,
        "training_command_sha256": digest_value(command),
        "artifacts": {
            "worker_result": str((output_root / "worker-result.json").resolve()),
            "raw_report": str((output_root / "raw-report.json").resolve()),
            "receipt": str((output_root / "receipt.json").resolve()),
            "trained_model": str((training_root / "model.pt").resolve()),
            "training_summary": str((training_root / "summary.json").resolve()),
            "training_checkpoint": str((training_root / "checkpoint.pt").resolve()),
        },
    }
    plan = {**plan_payload, "plan_sha256": digest_value(plan_payload)}
    _publish_create_only(output_root / "plan.json", plan, role="probe plan")
    ledger_path = output_root / "ledger.json"
    if ledger_path.exists():
        ledger = _read_ledger(ledger_path)
        _require(ledger["plan_sha256"] == plan["plan_sha256"], "ledger plan differs")
    else:
        ledger = {
            "schema": LEDGER_SCHEMA,
            "plan_sha256": plan["plan_sha256"],
            "state": "prepared",
            "worker_returncode": None,
            "worker_stdout_sha256": None,
            "worker_stderr_sha256": None,
            "worker_result_sha256": None,
            "raw_report_sha256": None,
            "receipt_sha256": None,
        }
        _write_ledger(ledger_path, ledger)
    return plan


def _verify_plan(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = root.expanduser().resolve()
    plan = _read_object(root / "plan.json", role="probe plan")
    _exact_keys(
        plan,
        {
            "schema",
            "probe_id",
            "protocol",
            "parent_seed",
            "parent_model",
            "immutable_inputs",
            "source_contract",
            "host_identity_sha256",
            "runtime_environment",
            "runtime",
            "fixed_workload",
            "data_access_policy",
            "repository_root",
            "python",
            "worker",
            "trainer",
            "training_command",
            "training_command_sha256",
            "artifacts",
            "plan_sha256",
        },
        role="probe plan",
    )
    payload = {key: row for key, row in plan.items() if key != "plan_sha256"}
    _require(
        plan["schema"] == PLAN_SCHEMA and plan["plan_sha256"] == digest_value(payload),
        "probe plan integrity check failed",
    )
    probe_id = _safe_identifier(plan["probe_id"], role="plan.probe_id")
    _require(
        (root / ROOT_SENTINEL).read_text(encoding="utf-8").strip() == probe_id,
        "probe sentinel differs",
    )
    ledger = _read_ledger(root / "ledger.json")
    _require(ledger["plan_sha256"] == plan["plan_sha256"], "ledger plan differs")

    repository_root = Path(plan["repository_root"]).resolve()
    _require(
        _safe_root(root, repository_root=repository_root) == root,
        "probe root moved",
    )
    _require(
        _source_contract(repository_root) == plan["source_contract"],
        "probe source revision or clean-worktree contract drifted",
    )
    _require(
        _host_identity_sha256() == plan["host_identity_sha256"],
        "probe host identity drifted",
    )
    _require(
        _SHA256.fullmatch(str(plan["host_identity_sha256"])) is not None
        and _COMMIT.fullmatch(str(plan["source_contract"].get("commit", "")))
        is not None,
        "probe host/source identity is malformed",
    )

    protocol_path = _safe_existing_file(
        Path(plan["protocol"]["path"]), role="X21 protocol"
    )
    protocol, protocol_artifact = _protocol_contract(protocol_path)
    _require(protocol_artifact == plan["protocol"], "X21 protocol binding drifted")
    _require(plan["parent_seed"] in protocol["promotion_seeds"], "parent seed drifted")
    inputs = plan["immutable_inputs"]
    _exact_keys(
        inputs,
        {"source_artifact", "train_common_pool", "selection_common_pool"},
        role="immutable inputs",
    )
    expected_hashes = {
        "source_artifact": protocol["data_contract"]["source_artifact_sha256"],
        "train_common_pool": protocol["data_contract"]["train_pool_sha256"],
        "selection_common_pool": protocol["data_contract"]["selection_pool_sha256"],
    }
    resolved_inputs = {}
    for role, row in inputs.items():
        path = _safe_existing_file(
            Path(row["path"]),
            role=role,
            forbid_held_out=role != "source_artifact",
        )
        observed = _artifact(path)
        _require(observed == row, f"{role} artifact drifted")
        _require(
            row["sha256"] == expected_hashes[role], f"{role} protocol hash drifted"
        )
        resolved_inputs[role] = path

    parent_row = plan["parent_model"]
    parent_path = _safe_existing_file(Path(parent_row["path"]), role="parent model")
    observed_parent = _artifact(parent_path)
    _require(
        all(parent_row.get(key) == value for key, value in observed_parent.items()),
        "parent model artifact drifted",
    )
    _require(
        _parent_contract(parent_path, protocol=protocol) == parent_row["contract"],
        "parent model contract drifted",
    )
    runtime = _runtime(plan["runtime"])
    _require(runtime == plan["runtime"], "probe runtime is not normalized")
    _require(
        plan["fixed_workload"]
        == {
            "site_count": SITE_COUNT,
            "bond_dimension": BOND_DIMENSION,
            "precision": PRECISION,
            "parameter_scope": "cores-only",
            "epochs": EPOCHS,
            "train_points": TRAIN_POINTS,
            "selection_points": SELECTION_POINTS,
            "batch_size": BATCH_SIZE,
            "optimizer_updates": OPTIMIZER_UPDATES,
        },
        "fixed X21 probe workload drifted",
    )
    _require(
        plan["data_access_policy"]
        == {
            "allowed_splits": ["train", "selection"],
            "confirmation": "forbidden",
            "blind": "forbidden",
            "holdout": "forbidden",
        },
        "probe data access policy drifted",
    )
    _require(
        plan["runtime_environment"] == _runtime_environment(),
        "probe runtime environment drifted",
    )
    python_path = active_python_executable()
    _require(
        plan["python"] == {"path": python_path, "version": platform.python_version()},
        "probe Python runtime drifted",
    )
    worker_path = (repository_root / WORKER_RELATIVE_PATH).resolve()
    trainer_path = (repository_root / TRAINER_RELATIVE_PATH).resolve()
    _require(
        plan["worker"]
        == {"path": str(worker_path), "sha256": sha256_file(worker_path)},
        "probe worker source drifted",
    )
    _require(
        plan["trainer"]
        == {"path": str(trainer_path), "sha256": sha256_file(trainer_path)},
        "X21 trainer source drifted",
    )
    artifacts = plan["artifacts"]
    expected_artifacts = {
        "worker_result": root / "worker-result.json",
        "raw_report": root / "raw-report.json",
        "receipt": root / "receipt.json",
        "trained_model": root / "training" / "model.pt",
        "training_summary": root / "training" / "summary.json",
        "training_checkpoint": root / "training" / "checkpoint.pt",
    }
    _exact_keys(artifacts, set(expected_artifacts), role="probe artifact paths")
    for role, expected_path in expected_artifacts.items():
        _require(
            Path(artifacts[role]) == expected_path.resolve(), f"{role} path drifted"
        )
    command = _training_command(
        python_path=python_path,
        trainer_path=trainer_path,
        source=resolved_inputs["source_artifact"],
        train_pool=resolved_inputs["train_common_pool"],
        selection_pool=resolved_inputs["selection_common_pool"],
        parent_model=parent_path,
        training_root=(root / "training").resolve(),
        parent_seed=plan["parent_seed"],
        runtime=runtime,
    )
    _require(
        plan["training_command"] == command
        and plan["training_command_sha256"] == digest_value(command),
        "sealed X21 trainer command drifted",
    )
    command_text = " ".join(command).lower()
    _require(
        not any(token in command_text for token in _FORBIDDEN_DATA_TOKENS),
        "sealed trainer command exposes forbidden held-out material",
    )
    _require("--train-physical-dictionary" not in command, "probe is not cores-only")
    _require(
        "--resume-checkpoint" not in command, "probe cannot resume optimizer state"
    )
    return plan, ledger


def _assert_parent_holds_gpu_lock() -> None:
    lock_path = (GPU_LOCK_ROOT / f"gcicy-tn-gpu-{GPU_ID}.lock").resolve()
    try:
        owner = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise X21HostGPUProbeError("probe worker cannot read GPU lock owner") from error
    _require(
        isinstance(owner, dict)
        and owner.get("gpu_id") == GPU_ID
        and owner.get("pid") == os.getppid(),
        "probe worker parent does not own workflow GPU 0",
    )
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    raise X21HostGPUProbeError("workflow GPU 0 lock is not held")


def _all_numeric_values_finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_all_numeric_values_finite(row) for row in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_numeric_values_finite(row) for row in value)
    if isinstance(value, np.ndarray):
        return bool(np.all(np.isfinite(value)))
    if isinstance(value, (float, np.floating, complex, np.complexfloating)):
        return bool(np.isfinite(value))
    return True


def _all_tensors_finite(value: Any, torch: Any) -> bool:
    if torch.is_tensor(value):
        return bool(torch.all(torch.isfinite(value.detach())))
    if isinstance(value, Mapping):
        return all(_all_tensors_finite(row, torch) for row in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_tensors_finite(row, torch) for row in value)
    return True


def _validate_training_summary(
    plan: Mapping[str, Any], summary: Mapping[str, Any]
) -> dict[str, Any]:
    inputs = plan["immutable_inputs"]
    expected = {
        "schema": "type11-positive-tensor-network-training-v1",
        "adapter": ADAPTER,
        "training_mode": "teacher_free_geometric",
        "model_seed": MODEL_SEED,
        "torch_seed": plan["parent_seed"],
        "site_count": SITE_COUNT,
        "bond_dimension": BOND_DIMENSION,
        "architecture": "shared_local_dictionary",
        "physical_dictionary_rank": PHYSICAL_DICTIONARY_RANK,
        "trainable_physical_dictionary": False,
        "physical_dictionary_gauge": "fixed",
        "trainable_real_parameter_count": TRAINABLE_REAL_PARAMETERS,
        "positive_floor": 1.0e-14,
        "initialization_noise": 0.0,
        "device": "cuda",
        "precision": PRECISION,
        "sampling_cluster_size": SAMPLING_CLUSTER_SIZE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": 3.0e-5,
        "gradient_clip_norm": 2.0,
        "fixed_log_kappa_source": "continued_saved_model",
        "termination_reason": "completed_requested_epochs",
    }
    for field, expected_value in expected.items():
        _require(
            summary.get(field) == expected_value,
            f"training summary {field} differs from {expected_value!r}",
        )
    model_path = Path(plan["artifacts"]["trained_model"])
    checkpoint_path = Path(plan["artifacts"]["training_checkpoint"])
    _require(
        summary.get("model") == str(model_path)
        and model_path.is_file()
        and summary.get("model_sha256") == sha256_file(model_path),
        "training summary output model identity differs",
    )
    _require(
        summary.get("source_artifact") == inputs["source_artifact"]["path"]
        and summary.get("teacher_artifact") is None
        and summary.get("teacher_artifact_sha256") is None,
        "training summary source/teacher identity differs",
    )
    _require(
        summary.get("source_artifact_sha256") == inputs["source_artifact"]["sha256"],
        "training summary source hash differs",
    )
    _require(
        summary.get("initialization", {}).get("model_sha256")
        == plan["parent_model"]["sha256"],
        "training summary parent hash differs",
    )
    train = summary.get("train")
    selection = summary.get("validation")
    _require(
        isinstance(train, dict) and isinstance(selection, dict),
        "dataset summaries missing",
    )
    _require(
        train.get("points") == TRAIN_POINTS
        and train.get("seed") == TRAIN_SEED
        and train.get("common_pool") == inputs["train_common_pool"]["path"]
        and train.get("common_pool_sha256") == inputs["train_common_pool"]["sha256"],
        "training summary train pool differs",
    )
    _require(
        selection.get("points") == SELECTION_POINTS
        and selection.get("seed") == SELECTION_SEED
        and selection.get("common_pool") == inputs["selection_common_pool"]["path"]
        and selection.get("common_pool_sha256")
        == inputs["selection_common_pool"]["sha256"],
        "training summary selection pool differs",
    )
    history = summary.get("history")
    _require(
        isinstance(history, list)
        and len(history) == EPOCHS + 1
        and [row.get("epoch") for row in history] == [0, 1],
        "training summary does not prove one complete epoch",
    )
    checkpoint = summary.get("checkpoint")
    _require(
        isinstance(checkpoint, dict)
        and checkpoint.get("path") == str(checkpoint_path)
        and checkpoint_path.is_file()
        and checkpoint.get("sha256") == sha256_file(checkpoint_path)
        and checkpoint.get("last_completed_validation_epoch") == EPOCHS
        and checkpoint.get("resume_kind") == "fresh_training",
        "training summary checkpoint boundary is invalid",
    )
    _require(
        summary.get("loss_weights")
        == {
            "potential": 0.0,
            "metric": 0.0,
            "log_energy": 1.0,
            "ma": 1.0,
            "tail": 0.1,
        },
        "training summary loss weights differ",
    )
    _require(
        summary.get("tail_loss", {}).get("tail_fraction") == 0.02
        and summary.get("tail_loss", {}).get("ratio_threshold") == 1.5
        and summary.get("tail_loss", {}).get("smooth_temperature") == 0.05,
        "training summary tail-loss contract differs",
    )
    _require(
        summary.get("early_stopping", {}).get("patience") == 0,
        "training summary unexpectedly enables early stopping",
    )
    memory = summary.get("device_memory")
    _require(
        isinstance(memory, dict)
        and isinstance(memory.get("maximum_allocated_bytes"), int)
        and not isinstance(memory.get("maximum_allocated_bytes"), bool)
        and memory["maximum_allocated_bytes"] > 0,
        "training summary lacks CUDA peak allocation",
    )
    best = summary.get("best_validation")
    _require(isinstance(best, dict), "training summary lacks selection evaluation")
    minimum = _finite(
        best.get("minimum_metric_eigenvalue"), role="minimum metric eigenvalue"
    )
    compressed = best.get("compressed_ma_errors")
    _require(isinstance(compressed, dict), "selection MA metrics are missing")
    metrics = {
        "selection_score": _finite(
            summary.get("best_validation_selection_score"), role="selection score"
        ),
        "log_energy_rms": _finite(
            best.get("fixed_kappa_log_energy_rms"), role="selection log-energy RMS"
        ),
        "chi": _finite(compressed.get("sqrt_squared_energy"), role="selection chi"),
    }
    _require(
        all(value >= 0.0 for value in metrics.values()),
        "selection metrics are negative",
    )
    return {
        "minimum_metric_eigenvalue": minimum,
        "maximum_allocated_bytes": memory["maximum_allocated_bytes"],
        "selection_metrics": metrics,
    }


def execute_cuda_worker(root: Path) -> dict[str, Any]:
    """Run the exact trainer command in the lock-owning parent's child process."""

    root = root.expanduser().resolve()
    plan, _ = _verify_plan(root)
    _assert_parent_holds_gpu_lock()
    _require(
        _runtime_environment() == plan["runtime_environment"],
        "runtime environment drifted inside the GPU lock",
    )
    result_path = Path(plan["artifacts"]["worker_result"])
    if result_path.exists():
        return _validate_worker_result(
            plan, _read_object(result_path, role="worker result")
        )
    training_artifact_paths = [
        Path(plan["artifacts"][role])
        for role in ("trained_model", "training_summary", "training_checkpoint")
    ]
    _require(
        not any(path.exists() for path in training_artifact_paths),
        "ambiguous prior trainer outputs exist without a worker result",
    )
    try:
        import torch
    except ImportError as error:  # pragma: no cover - production dependency
        raise X21HostGPUProbeError("PyTorch is unavailable in probe worker") from error
    _require(torch.cuda.is_available(), "probe worker cannot see CUDA")
    training_artifact_paths[0].parent.mkdir(parents=True, exist_ok=True)
    parent_path = Path(plan["parent_model"]["path"])
    before_parent_sha256 = sha256_file(parent_path)
    started_utc = _utc_now()
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = GPU_ID
    completed = subprocess.run(
        list(plan["training_command"]),
        check=False,
        capture_output=True,
        text=True,
        shell=False,
        env=environment,
    )
    finished_utc = _utc_now()
    _require(completed.returncode == 0, "X21 one-epoch trainer failed")
    _require(
        sha256_file(parent_path)
        == before_parent_sha256
        == plan["parent_model"]["sha256"],
        "X21 parent model was overwritten or drifted",
    )
    _require(
        all(path.is_file() for path in training_artifact_paths),
        "X21 trainer returned success without all artifacts",
    )
    summary_path = Path(plan["artifacts"]["training_summary"])
    summary = _read_object(summary_path, role="training summary")
    summary_values = _validate_training_summary(plan, summary)
    model_path = Path(plan["artifacts"]["trained_model"])
    model_payload = _load_torch_object(model_path, role="trained probe model")
    protocol, _ = _protocol_contract(Path(plan["protocol"]["path"]))
    _require(
        _parent_contract(model_path, protocol=protocol)
        == plan["parent_model"]["contract"],
        "trained probe model changed the registered X21 architecture",
    )
    _require(
        _all_tensors_finite(model_payload, torch),
        "trained probe model contains a non-finite tensor",
    )
    checkpoint_path = Path(plan["artifacts"]["training_checkpoint"])
    checkpoint_payload = _load_torch_object(checkpoint_path, role="probe checkpoint")
    _require(
        checkpoint_payload.get("schema")
        == "type11-positive-tensor-network-checkpoint-v1"
        and checkpoint_payload.get("epoch") == EPOCHS
        and checkpoint_payload.get("next_epoch") == EPOCHS + 1,
        "probe checkpoint does not prove the one-epoch boundary",
    )
    history = checkpoint_payload.get("history")
    _require(
        isinstance(history, list) and [row.get("epoch") for row in history] == [0, 1],
        "probe checkpoint history does not prove one complete epoch",
    )
    semantics = checkpoint_payload.get("training_semantics")
    _require(isinstance(semantics, dict), "probe checkpoint lacks training semantics")
    semantic_model = semantics.get("model")
    semantic_data = semantics.get("data")
    semantic_optimization = semantics.get("optimization")
    _require(
        isinstance(semantic_model, dict)
        and {
            "site_count": semantic_model.get("site_count"),
            "bond_dimension": semantic_model.get("bond_dimension"),
            "precision": semantic_model.get("precision"),
            "trainable_physical_dictionary": semantic_model.get(
                "trainable_physical_dictionary"
            ),
        }
        == {
            "site_count": SITE_COUNT,
            "bond_dimension": BOND_DIMENSION,
            "precision": PRECISION,
            "trainable_physical_dictionary": False,
        },
        "probe checkpoint model semantics differ",
    )
    _require(
        isinstance(semantic_data, dict)
        and {
            "train_points": semantic_data.get("train_points"),
            "train_seed": semantic_data.get("train_seed"),
            "validation_points": semantic_data.get("validation_points"),
            "validation_seed": semantic_data.get("validation_seed"),
            "sampling_cluster_size": semantic_data.get("sampling_cluster_size"),
            "train_uses_common_pool": semantic_data.get("train_uses_common_pool"),
            "validation_uses_common_pool": semantic_data.get(
                "validation_uses_common_pool"
            ),
        }
        == {
            "train_points": TRAIN_POINTS,
            "train_seed": TRAIN_SEED,
            "validation_points": SELECTION_POINTS,
            "validation_seed": SELECTION_SEED,
            "sampling_cluster_size": SAMPLING_CLUSTER_SIZE,
            "train_uses_common_pool": True,
            "validation_uses_common_pool": True,
        },
        "probe checkpoint data semantics differ",
    )
    _require(
        isinstance(semantic_optimization, dict)
        and {
            "epochs": semantic_optimization.get("epochs"),
            "batch_size": semantic_optimization.get("batch_size"),
            "learning_rate": semantic_optimization.get("learning_rate"),
            "gradient_clip_norm": semantic_optimization.get("gradient_clip_norm"),
            "torch_seed": semantic_optimization.get("torch_seed"),
            "kappa_source": semantic_optimization.get("kappa_source"),
        }
        == {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": 3.0e-5,
            "gradient_clip_norm": 2.0,
            "torch_seed": plan["parent_seed"],
            "kappa_source": "saved_model",
        },
        "probe checkpoint optimizer semantics differ",
    )
    _require(
        checkpoint_payload.get("frozen_input_hashes")
        == {
            "source_artifact_sha256": plan["immutable_inputs"]["source_artifact"][
                "sha256"
            ],
            "teacher_artifact_sha256": None,
            "initial_model_sha256": plan["parent_model"]["sha256"],
            "train_common_pool_sha256": plan["immutable_inputs"]["train_common_pool"][
                "sha256"
            ],
            "validation_common_pool_sha256": plan["immutable_inputs"][
                "selection_common_pool"
            ]["sha256"],
        },
        "probe checkpoint frozen inputs differ",
    )
    _require(
        checkpoint_payload.get("training_semantics", {})
        .get("model", {})
        .get("trainable_physical_dictionary")
        is False,
        "probe checkpoint is not cores-only",
    )
    all_finite = bool(
        _all_numeric_values_finite(summary)
        and _all_tensors_finite(checkpoint_payload, torch)
    )
    artifacts = {
        "model": _artifact(model_path),
        "summary": _artifact(summary_path),
        "checkpoint": _artifact(checkpoint_path),
    }
    result = {
        "schema": WORKER_RESULT_SCHEMA,
        "probe_id": plan["probe_id"],
        "plan_sha256": plan["plan_sha256"],
        "source_commit": plan["source_contract"]["commit"],
        "host_identity_sha256": plan["host_identity_sha256"],
        "checkpoint_sha256": plan["parent_model"]["sha256"],
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "device": "cuda",
        "parameter_scope": "cores-only",
        "optimizer_steps_completed": OPTIMIZER_UPDATES,
        "selection_evaluation_completed": True,
        "all_finite": all_finite,
        "minimum_metric_eigenvalue": summary_values["minimum_metric_eigenvalue"],
        "nonpositive_metric_count": (
            0 if summary_values["minimum_metric_eigenvalue"] > 0.0 else 1
        ),
        "maximum_allocated_bytes": summary_values["maximum_allocated_bytes"],
        "selection_metrics": summary_values["selection_metrics"],
        "trainer_returncode": completed.returncode,
        "trainer_stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
        "trainer_stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
        "training_command_sha256": plan["training_command_sha256"],
        "training_artifacts": artifacts,
    }
    validated = _validate_worker_result(plan, result)
    _publish_create_only(result_path, validated, role="worker result")
    return validated


def _validate_worker_result(
    plan: Mapping[str, Any], value: Mapping[str, Any]
) -> dict[str, Any]:
    _exact_keys(
        value,
        {
            "schema",
            "probe_id",
            "plan_sha256",
            "source_commit",
            "host_identity_sha256",
            "checkpoint_sha256",
            "started_utc",
            "finished_utc",
            "device",
            "parameter_scope",
            "optimizer_steps_completed",
            "selection_evaluation_completed",
            "all_finite",
            "minimum_metric_eigenvalue",
            "nonpositive_metric_count",
            "maximum_allocated_bytes",
            "selection_metrics",
            "trainer_returncode",
            "trainer_stdout_sha256",
            "trainer_stderr_sha256",
            "training_command_sha256",
            "training_artifacts",
        },
        role="worker result",
    )
    _require(value["schema"] == WORKER_RESULT_SCHEMA, "worker result schema differs")
    expected_identity = {
        "probe_id": plan["probe_id"],
        "plan_sha256": plan["plan_sha256"],
        "source_commit": plan["source_contract"]["commit"],
        "host_identity_sha256": plan["host_identity_sha256"],
        "checkpoint_sha256": plan["parent_model"]["sha256"],
        "training_command_sha256": plan["training_command_sha256"],
    }
    _require(
        all(value.get(key) == expected for key, expected in expected_identity.items()),
        "worker result identity binding differs",
    )
    started = _parse_utc(value["started_utc"], role="worker.started_utc")
    finished = _parse_utc(value["finished_utc"], role="worker.finished_utc")
    _require(started <= finished, "worker timestamps are reversed")
    _require(
        value["device"] == "cuda" and value["parameter_scope"] == "cores-only",
        "worker did not run the registered CUDA cores-only workload",
    )
    _require(
        value["optimizer_steps_completed"] == OPTIMIZER_UPDATES,
        "worker did not complete the full one-epoch optimizer budget",
    )
    _require(
        value["selection_evaluation_completed"] is True,
        "worker did not complete selection evaluation",
    )
    _require(isinstance(value["all_finite"], bool), "worker all_finite is malformed")
    minimum = _finite(value["minimum_metric_eigenvalue"], role="minimum eigenvalue")
    nonpositive = value["nonpositive_metric_count"]
    memory = value["maximum_allocated_bytes"]
    _require(
        isinstance(nonpositive, int)
        and not isinstance(nonpositive, bool)
        and nonpositive >= 0,
        "worker nonpositive count is malformed",
    )
    _require(
        isinstance(memory, int) and not isinstance(memory, bool) and memory > 0,
        "worker CUDA memory receipt is malformed",
    )
    _require(
        (minimum > 0.0) == (nonpositive == 0),
        "worker positivity summary is internally inconsistent",
    )
    metrics = value["selection_metrics"]
    _exact_keys(
        metrics, {"selection_score", "log_energy_rms", "chi"}, role="selection metrics"
    )
    normalized_metrics = {
        key: _finite(metrics[key], role=f"selection_metrics.{key}")
        for key in ("selection_score", "log_energy_rms", "chi")
    }
    _require(
        all(row >= 0.0 for row in normalized_metrics.values()),
        "selection metrics are negative",
    )
    _require(value["trainer_returncode"] == 0, "worker trainer return code differs")
    for role in ("trainer_stdout_sha256", "trainer_stderr_sha256"):
        _require(
            _SHA256.fullmatch(str(value[role])) is not None, f"{role} is malformed"
        )
    artifacts = value["training_artifacts"]
    _exact_keys(
        artifacts, {"model", "summary", "checkpoint"}, role="training artifacts"
    )
    expected_paths = {
        "model": Path(plan["artifacts"]["trained_model"]),
        "summary": Path(plan["artifacts"]["training_summary"]),
        "checkpoint": Path(plan["artifacts"]["training_checkpoint"]),
    }
    normalized_artifacts = {}
    for role, path in expected_paths.items():
        _require(path.is_file(), f"training {role} is missing")
        observed = _artifact(path)
        _require(artifacts[role] == observed, f"training {role} artifact drifted")
        normalized_artifacts[role] = observed
    return {
        **dict(value),
        "started_utc": started.isoformat().replace("+00:00", "Z"),
        "finished_utc": finished.isoformat().replace("+00:00", "Z"),
        "minimum_metric_eigenvalue": minimum,
        "maximum_allocated_bytes": memory,
        "selection_metrics": normalized_metrics,
        "training_artifacts": normalized_artifacts,
    }


def _worker_argv(plan: Mapping[str, Any], root: Path) -> list[str]:
    return [
        str(plan["python"]["path"]),
        str(plan["worker"]["path"]),
        "_worker",
        "--root",
        str(root.resolve()),
    ]


def _launch_worker(
    plan: Mapping[str, Any], root: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _worker_argv(plan, root),
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )


def _raw_report(plan: Mapping[str, Any], worker: Mapping[str, Any]) -> dict[str, Any]:
    worker_path = Path(plan["artifacts"]["worker_result"])
    return {
        "schema": RAW_REPORT_SCHEMA,
        "probe_id": plan["probe_id"],
        "plan_path": str((worker_path.parent / "plan.json").resolve()),
        "plan_sha256": plan["plan_sha256"],
        "protocol": plan["protocol"],
        "source_contract": plan["source_contract"],
        "host_identity_sha256": plan["host_identity_sha256"],
        "parent_seed": plan["parent_seed"],
        "parent_model": plan["parent_model"],
        "immutable_inputs": plan["immutable_inputs"],
        "fixed_workload": plan["fixed_workload"],
        "data_access_policy": plan["data_access_policy"],
        "training_command_sha256": plan["training_command_sha256"],
        "worker_result_path": str(worker_path),
        "worker_result_sha256": sha256_file(worker_path),
        "training_artifacts": worker["training_artifacts"],
        "started_utc": worker["started_utc"],
        "finished_utc": worker["finished_utc"],
        "optimizer_steps": worker["optimizer_steps_completed"],
        "selection_evaluation_completed": worker["selection_evaluation_completed"],
        "all_finite": worker["all_finite"],
        "minimum_metric_eigenvalue": worker["minimum_metric_eigenvalue"],
        "nonpositive_metric_count": worker["nonpositive_metric_count"],
        "maximum_allocated_bytes": worker["maximum_allocated_bytes"],
        "selection_metrics": worker["selection_metrics"],
    }


def _receipt(plan: Mapping[str, Any], raw: Mapping[str, Any]) -> dict[str, Any]:
    report_path = Path(plan["artifacts"]["raw_report"])
    return {
        "schema": GPU_PROBE_SCHEMA,
        "probe_id": plan["probe_id"],
        "checkpoint_sha256": plan["parent_model"]["sha256"],
        "host_identity_sha256": plan["host_identity_sha256"],
        "source_commit": plan["source_contract"]["commit"],
        "started_utc": raw["started_utc"],
        "finished_utc": raw["finished_utc"],
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "optimizer_steps": raw["optimizer_steps"],
        "selection_evaluation_completed": raw["selection_evaluation_completed"],
        "all_finite": raw["all_finite"],
        "minimum_metric_eigenvalue": raw["minimum_metric_eigenvalue"],
        "maximum_allocated_bytes": raw["maximum_allocated_bytes"],
    }


def _normalize_receipt(
    plan: Mapping[str, Any], receipt: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        return normalize_gpu_probe(
            receipt,
            index=0,
            expected_checkpoint_sha256=plan["parent_model"]["sha256"],
            expected_host_identity_sha256=plan["host_identity_sha256"],
            expected_source_commit=plan["source_contract"]["commit"],
            issued_at=_parse_utc(receipt["finished_utc"], role="receipt.finished_utc"),
        )
    except HostStabilityError as error:
        raise X21HostGPUProbeError(
            f"generic host-gate receipt normalization failed: {error}"
        ) from error


def _finalize(
    plan: Mapping[str, Any], ledger: dict[str, Any], root: Path
) -> dict[str, Any]:
    verified_plan, verified_ledger = _verify_plan(root)
    _require(verified_plan == plan, "probe plan changed before finalization")
    _require(
        verified_ledger["state"] == "worker-complete",
        "probe cannot finalize before worker completion",
    )
    ledger = verified_ledger
    worker_path = Path(plan["artifacts"]["worker_result"])
    worker = _validate_worker_result(
        plan, _read_object(worker_path, role="worker result")
    )
    raw = _raw_report(plan, worker)
    raw_path = Path(plan["artifacts"]["raw_report"])
    _publish_create_only(raw_path, raw, role="raw probe report")
    receipt = _receipt(plan, raw)
    receipt_path = Path(plan["artifacts"]["receipt"])
    _publish_create_only(receipt_path, receipt, role="GPU probe receipt")
    normalized = _normalize_receipt(plan, receipt)
    ledger["state"] = "complete" if normalized["passes"] else "diagnostic-only"
    ledger["worker_result_sha256"] = sha256_file(worker_path)
    ledger["raw_report_sha256"] = sha256_file(raw_path)
    ledger["receipt_sha256"] = sha256_file(receipt_path)
    _write_ledger(root / "ledger.json", ledger)
    return receipt


def _run_with_gpu_lock_held(root: Path) -> dict[str, Any]:
    plan, ledger = _verify_plan(root)
    _require(
        _runtime_environment() == plan["runtime_environment"],
        "runtime environment drifted inside the GPU lock",
    )
    state = ledger["state"]
    if state == "failed":
        raise X21HostGPUProbeError("failed roots are immutable; choose a new probe_id")
    if state == "worker-complete":
        return _finalize(plan, ledger, root)
    if state == "running":
        if Path(plan["artifacts"]["worker_result"]).is_file():
            ledger["state"] = "worker-complete"
            _write_ledger(root / "ledger.json", ledger)
            return _finalize(plan, ledger, root)
        raise X21HostGPUProbeError(
            "probe optimizer execution is ambiguous; choose a new probe_id/root"
        )
    _require(state == "prepared", "probe ledger state is invalid")
    ledger["state"] = "running"
    _write_ledger(root / "ledger.json", ledger)
    completed = _launch_worker(plan, root)
    ledger = _read_ledger(root / "ledger.json")
    ledger["worker_returncode"] = completed.returncode
    ledger["worker_stdout_sha256"] = hashlib.sha256(
        completed.stdout.encode()
    ).hexdigest()
    ledger["worker_stderr_sha256"] = hashlib.sha256(
        completed.stderr.encode()
    ).hexdigest()
    if completed.returncode != 0:
        ledger["state"] = "failed"
        _write_ledger(root / "ledger.json", ledger)
        raise X21HostGPUProbeError(
            "CUDA probe worker failed; choose a new probe_id/root"
        )
    worker_path = Path(plan["artifacts"]["worker_result"])
    _require(worker_path.is_file(), "probe worker returned success without a result")
    _validate_worker_result(plan, _read_object(worker_path, role="worker result"))
    ledger["state"] = "worker-complete"
    ledger["worker_result_sha256"] = sha256_file(worker_path)
    _write_ledger(root / "ledger.json", ledger)
    return _finalize(plan, ledger, root)


def run_probe(root: Path) -> dict[str, Any]:
    """Execute or finalize a prepared probe under the shared workflow GPU lock."""

    root = root.expanduser().resolve()
    _, ledger = _verify_plan(root)
    if ledger["state"] in {"complete", "diagnostic-only"}:
        return verify_probe(root, require_passing=False)["receipt"]
    with gpu_lock(GPU_LOCK_ROOT, GPU_ID, timeout_seconds=-1):
        return _run_with_gpu_lock_held(root)


def verify_probe(root: Path, *, require_passing: bool = True) -> dict[str, Any]:
    """Re-hash every terminal artifact and optionally require a passing receipt."""

    root = root.expanduser().resolve()
    plan, ledger = _verify_plan(root)
    _require(
        ledger["state"] in {"complete", "diagnostic-only"},
        "probe has no terminal receipt",
    )
    worker_path = Path(plan["artifacts"]["worker_result"])
    raw_path = Path(plan["artifacts"]["raw_report"])
    receipt_path = Path(plan["artifacts"]["receipt"])
    _require(
        worker_path.is_file() and raw_path.is_file() and receipt_path.is_file(),
        "terminal probe artifacts are missing",
    )
    worker = _validate_worker_result(
        plan, _read_object(worker_path, role="worker result")
    )
    raw = _read_object(raw_path, role="raw probe report")
    _require(raw == _raw_report(plan, worker), "raw probe report was forged or drifted")
    receipt = _read_object(receipt_path, role="GPU probe receipt")
    _require(receipt == _receipt(plan, raw), "GPU probe receipt was forged or drifted")
    _require(
        ledger["worker_result_sha256"] == sha256_file(worker_path)
        and ledger["raw_report_sha256"] == sha256_file(raw_path)
        and ledger["receipt_sha256"] == sha256_file(receipt_path),
        "probe ledger artifact hashes drifted",
    )
    normalized = _normalize_receipt(plan, receipt)
    expected_state = "complete" if normalized["passes"] else "diagnostic-only"
    _require(ledger["state"] == expected_state, "probe ledger verdict differs")
    if require_passing:
        _require(normalized["passes"], "probe is diagnostic-only")
    return {
        "plan": plan,
        "ledger": ledger,
        "raw_report": raw,
        "receipt": receipt,
        "normalized_receipt": normalized,
    }


def probe_status(root: Path) -> dict[str, Any]:
    plan, ledger = _verify_plan(root.expanduser().resolve())
    result = {
        "probe_id": plan["probe_id"],
        "state": ledger["state"],
        "plan_sha256": plan["plan_sha256"],
        "parent_seed": plan["parent_seed"],
        "parent_model_sha256": plan["parent_model"]["sha256"],
        "optimizer_steps": OPTIMIZER_UPDATES,
        "receipt_path": plan["artifacts"]["receipt"],
    }
    if ledger["state"] in {"complete", "diagnostic-only"}:
        verified = verify_probe(root, require_passing=False)
        result.update(
            {
                "passes": verified["normalized_receipt"]["passes"],
                "maximum_allocated_bytes": verified["receipt"][
                    "maximum_allocated_bytes"
                ],
                "minimum_metric_eigenvalue": verified["receipt"][
                    "minimum_metric_eigenvalue"
                ],
            }
        )
    return result


__all__ = [
    "BATCH_SIZE",
    "LEDGER_SCHEMA",
    "OPTIMIZER_UPDATES",
    "PLAN_SCHEMA",
    "RAW_REPORT_SCHEMA",
    "ROOT_SENTINEL",
    "WORKER_RESULT_SCHEMA",
    "X21HostGPUProbeError",
    "execute_cuda_worker",
    "prepare_probe",
    "probe_status",
    "run_probe",
    "verify_probe",
]
