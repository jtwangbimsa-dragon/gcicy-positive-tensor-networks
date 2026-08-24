"""Create content-addressed CUDA health probes for quintic Auto Research.

The probe is deliberately narrower than a scientific experiment.  It runs a
short, fixed optimizer workload against one registered complex64 parent and
the already-frozen Round-1 fit/selection split.  It never accepts a held-out
data path and it cannot select or promote a model.

Each root contains an immutable plan, a worker result, an independently
rehashable raw report, and a ``gcicy-host-gpu-probe-v1`` receipt consumable by
``host_stability_gate``.  Ambiguous interrupted optimizer execution is never
replayed in place: callers must choose a fresh ``probe_id`` and output root.
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
import sys
from typing import Any, Mapping

import numpy as np

from .architecture_auto_research import (
    AutoResearchError,
    atomic_write_json,
    digest_value,
    sha256_file,
)
from .host_stability_gate import GPU_PROBE_SCHEMA, normalize_gpu_probe
from .experiment_workflow import gpu_lock
from .quintic_architecture_multi_round import validate_catalog
from .quintic_architecture_round1_bridge import canonical_array_value_sha256
from .quintic_paired_auto_research_bridge import (
    PRECISION,
    _validate_round1_data_contract,
)


PLAN_SCHEMA = "gcicy-quintic-host-gpu-probe-plan-v1"
LEDGER_SCHEMA = "gcicy-quintic-host-gpu-probe-ledger-v1"
WORKER_RESULT_SCHEMA = "gcicy-quintic-host-gpu-probe-worker-result-v1"
RAW_REPORT_SCHEMA = "gcicy-quintic-host-gpu-probe-report-v1"
ROOT_SENTINEL = ".gcicy-quintic-host-gpu-probe-root"

DEFAULT_OPTIMIZER_STEPS = 10
MAXIMUM_OPTIMIZER_STEPS = 50
FIT_COUNT = 30_000
SELECTION_COUNT = 5_000
BATCH_SIZE = 1024

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40,64}$")
_FORBIDDEN_PATH_TOKENS = ("blind", "confirmation", "shadow")

SOURCE_RELATIVE_PATHS = (
    "gcicy_metric/pipeline/architecture_auto_research.py",
    "gcicy_metric/pipeline/host_stability_gate.py",
    "gcicy_metric/pipeline/experiment_workflow.py",
    "gcicy_metric/pipeline/positive_multiplication_tree.py",
    "gcicy_metric/pipeline/quintic_architecture_multi_round.py",
    "gcicy_metric/pipeline/quintic_architecture_round1_bridge.py",
    "gcicy_metric/pipeline/quintic_host_gpu_probe.py",
    "gcicy_metric/pipeline/quintic_paired_auto_research_bridge.py",
    "scripts/evaluate_generic_quintic_h4_architecture_arms.py",
    "scripts/refine_generic_quintic_compiled_tree_native_gn.py",
    "scripts/run_quintic_host_gpu_probe.py",
    "scripts/train_generic_quintic_adaptive_tree_rank.py",
    "scripts/train_generic_quintic_compiled_tree.py",
    "scripts/train_quintic_native_power_lift_tree.py",
)
WORKER_RELATIVE_PATH = "scripts/run_quintic_host_gpu_probe.py"
GPU_LOCK_ROOT = Path("/tmp/gcicy-tn-gpu-locks")
GPU_ID = "0"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AutoResearchError(message)


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AutoResearchError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise AutoResearchError(f"JSON root must be an object: {path}")
    return value


def _exact_keys(value: Mapping[str, Any], required: set[str], context: str) -> None:
    if set(value) != required:
        raise AutoResearchError(
            f"{context} fields are not exact; "
            f"missing={sorted(required - set(value))}, "
            f"extra={sorted(set(value) - required)}"
        )


def _safe_identifier(value: Any, *, context: str) -> str:
    result = str(value)
    if not _IDENTIFIER.fullmatch(result):
        raise AutoResearchError(f"{context} is not a safe identifier")
    return result


def _safe_path(value: Any, *, context: str, reject_held_out: bool = True) -> Path:
    path = Path(str(value)).expanduser().resolve()
    if reject_held_out and any(
        token in str(path).lower() for token in _FORBIDDEN_PATH_TOKENS
    ):
        raise AutoResearchError(f"{context} references forbidden held-out material")
    return path


def _positive_int(
    value: Any,
    *,
    context: str,
    minimum: int = 1,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AutoResearchError(f"{context} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise AutoResearchError(f"{context} is outside its registered range")
    return value


def _finite(value: Any, *, context: str) -> float:
    if isinstance(value, bool):
        raise AutoResearchError(f"{context} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise AutoResearchError(f"{context} must be numeric") from error
    if not math.isfinite(result):
        raise AutoResearchError(f"{context} must be finite")
    return result


def _parse_utc(value: Any, *, context: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as error:
        raise AutoResearchError(f"{context} is not an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise AutoResearchError(f"{context} has no timezone")
    return parsed.astimezone(timezone.utc)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _runtime(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "device",
        "optimizer_steps",
        "learning_rate",
        "gradient_clip_norm",
        "threads",
        "train_chunk_size",
        "feature_batch_size",
        "eval_batch_size",
    }
    _exact_keys(value, required, "runtime")
    _require(value["device"] == "cuda", "host GPU probes require CUDA")
    steps = _positive_int(
        value["optimizer_steps"],
        context="runtime.optimizer_steps",
        minimum=DEFAULT_OPTIMIZER_STEPS,
        maximum=MAXIMUM_OPTIMIZER_STEPS,
    )
    learning_rate = _finite(value["learning_rate"], context="runtime.learning_rate")
    gradient_clip = _finite(
        value["gradient_clip_norm"], context="runtime.gradient_clip_norm"
    )
    _require(learning_rate > 0.0, "runtime.learning_rate must be positive")
    _require(gradient_clip > 0.0, "runtime.gradient_clip_norm must be positive")
    return {
        "device": "cuda",
        "optimizer_steps": steps,
        "learning_rate": learning_rate,
        "gradient_clip_norm": gradient_clip,
        **{
            key: _positive_int(value[key], context=f"runtime.{key}")
            for key in (
                "threads",
                "train_chunk_size",
                "feature_batch_size",
                "eval_batch_size",
            )
        },
    }


def _source_contract(repository_root: Path) -> dict[str, Any]:
    """Bind the full clean checkout, including untracked source files."""

    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            shell=False,
        ).stdout.strip()
        commit = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repository_root,
                check=True,
                capture_output=True,
                text=True,
                shell=False,
            )
            .stdout.strip()
            .lower()
        )
    except subprocess.CalledProcessError as error:
        raise AutoResearchError("cannot bind the probe source checkout") from error
    _require(not status, "host GPU probe requires a completely clean source checkout")
    _require(bool(_COMMIT.fullmatch(commit)), "source commit is malformed")
    dependencies = {}
    for relative in SOURCE_RELATIVE_PATHS:
        path = (repository_root / relative).resolve()
        _require(path.is_file(), f"registered probe dependency is missing: {relative}")
        dependencies[relative] = sha256_file(path)
    return {
        "commit": commit,
        "worktree": "clean",
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
        "host GPU probe requires exactly one inspectable CUDA GPU",
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
        raise AutoResearchError("PyTorch is unavailable") from error
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
        "CUDA runtime inventory requires exactly one inspectable GPU",
    )
    fields = [field.strip() for field in rows[0].split(",")]
    _require(len(fields) == 2, "CUDA runtime inventory is malformed")
    memory_mib = _positive_int(int(fields[1]), context="gpu.memory_total_mib")
    return {
        "numpy": str(np.__version__),
        "torch": str(torch.__version__),
        "torch_cuda_build": torch.version.cuda,
        "gpu_name": fields[0],
        "gpu_total_memory_bytes": memory_mib * 1024**2,
    }


def _checkpoint_contract(path: Path) -> dict[str, Any]:
    try:
        import torch

        from scripts.evaluate_generic_quintic_h4_architecture_arms import (
            infer_architecture,
        )

        payload = torch.load(path, map_location="cpu", weights_only=False)
        precision = str(payload["configuration"]["precision"])
        architecture = infer_architecture(payload)
        teacher_free = payload.get("teacher_runtime_dependency") is False
    except (ImportError, KeyError, OSError, TypeError, ValueError) as error:
        raise AutoResearchError("cannot validate the registered parent") from error
    _require(precision == PRECISION, "registered parent is not complex64")
    _require(architecture == "compiled-tree", "registered parent is not compiled-tree")
    _require(teacher_free, "registered parent has a teacher runtime dependency")
    return {
        "precision": precision,
        "architecture": architecture,
        "teacher_runtime_dependency": False,
    }


def _publish_create_only(path: Path, value: Mapping[str, Any], *, context: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False).encode()
        + b"\n"
    )
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        if _read_object(path) != dict(value):
            raise AutoResearchError(f"{context} already differs")
        return
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _seal_ledger(value: dict[str, Any]) -> None:
    payload = {key: row for key, row in value.items() if key != "state_sha256"}
    value["state_sha256"] = digest_value(payload)


def _write_ledger(path: Path, value: dict[str, Any]) -> None:
    _seal_ledger(value)
    atomic_write_json(path, value)


def _read_ledger(path: Path) -> dict[str, Any]:
    value = _read_object(path)
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
        "probe ledger",
    )
    payload = {key: row for key, row in value.items() if key != "state_sha256"}
    _require(
        value.get("schema") == LEDGER_SCHEMA
        and value.get("state_sha256") == digest_value(payload),
        "probe ledger integrity check failed",
    )
    return value


def _batch_prefix_sha256(path: Path, steps: int) -> str:
    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise AutoResearchError("cannot load the registered batch plan") from error
    _require(
        array.shape == (600, BATCH_SIZE) and np.issubdtype(array.dtype, np.integer),
        "registered batch plan has the wrong shape or dtype",
    )
    prefix = np.asarray(array[:steps], dtype=np.int64)
    _require(
        np.all((prefix >= 0) & (prefix < FIT_COUNT))
        and all(len(np.unique(row)) == len(row) for row in prefix),
        "registered batch-plan prefix is invalid",
    )
    return canonical_array_value_sha256(prefix)


def _not_inside(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return True
    return False


def prepare_probe(
    *,
    catalog_path: Path,
    round1_bridge_root: Path,
    parent_checkpoint: Path,
    parent_checkpoint_sha256: str,
    parent_seed: int,
    probe_id: str,
    output_root: Path,
    runtime: Mapping[str, Any],
    repository_root: Path,
) -> dict[str, Any]:
    """Create or exactly resume one immutable probe plan."""

    probe_id = _safe_identifier(probe_id, context="probe_id")
    repository_root = repository_root.expanduser().resolve()
    output_root = _safe_path(output_root, context="output_root")
    _require(
        _not_inside(output_root, repository_root),
        "probe output root must be outside the source checkout",
    )
    catalog_path = _safe_path(catalog_path, context="catalog_path")
    round1_bridge_root = _safe_path(round1_bridge_root, context="round1_bridge_root")
    parent_checkpoint = _safe_path(parent_checkpoint, context="parent_checkpoint")
    _require(catalog_path.is_file(), "registered catalog is missing")
    _require(parent_checkpoint.is_file(), "registered parent checkpoint is missing")
    _require(
        bool(_SHA256.fullmatch(str(parent_checkpoint_sha256))),
        "registered parent checkpoint SHA-256 is malformed",
    )
    _require(
        sha256_file(parent_checkpoint) == parent_checkpoint_sha256,
        "registered parent checkpoint hash differs",
    )
    catalog = validate_catalog(_read_object(catalog_path))
    _require(
        parent_seed in catalog["promotion_seeds"],
        "parent seed is not registered in the catalog",
    )
    round1_data = _validate_round1_data_contract(round1_bridge_root, catalog)
    seed_row = next(row for row in round1_data["seeds"] if row["seed"] == parent_seed)
    runtime_row = _runtime(runtime)
    batch_path = Path(seed_row["batch_plan"]["path"])
    batch_prefix_sha256 = _batch_prefix_sha256(
        batch_path, runtime_row["optimizer_steps"]
    )
    source_contract = _source_contract(repository_root)
    host_identity = _host_identity_sha256()
    checkpoint_contract = _checkpoint_contract(parent_checkpoint)
    runtime_environment = _runtime_environment()
    worker_path = (repository_root / WORKER_RELATIVE_PATH).resolve()
    _require(worker_path.is_file(), "registered probe worker is missing")

    output_root.mkdir(parents=True, exist_ok=True)
    sentinel = output_root / ROOT_SENTINEL
    if sentinel.exists():
        _require(
            sentinel.read_text(encoding="utf-8").strip() == probe_id,
            "probe root belongs to another probe_id",
        )
    elif any(output_root.iterdir()):
        raise AutoResearchError("refusing a non-empty unmanaged probe root")
    else:
        try:
            descriptor = os.open(sentinel, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError as error:  # pragma: no cover - concurrent prepare
            raise AutoResearchError(
                "probe sentinel was created concurrently"
            ) from error
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(f"{probe_id}\n")
            handle.flush()
            os.fsync(handle.fileno())

    plan_payload = {
        "schema": PLAN_SCHEMA,
        "probe_id": probe_id,
        "catalog": {
            "path": str(catalog_path),
            "file_sha256": sha256_file(catalog_path),
            "catalog_sha256": catalog["catalog_sha256"],
        },
        "round1_data": round1_data,
        "parent_seed": parent_seed,
        "parent_checkpoint": {
            "path": str(parent_checkpoint),
            "sha256": parent_checkpoint_sha256,
            "bytes": parent_checkpoint.stat().st_size,
            "contract": checkpoint_contract,
        },
        "source_contract": source_contract,
        "host_identity_sha256": host_identity,
        "runtime": runtime_row,
        "runtime_environment": runtime_environment,
        "batch_prefix_sha256": batch_prefix_sha256,
        "allowed_splits": ["fit", "selection"],
        "repository_root": str(repository_root),
        "worker": {"path": str(worker_path), "sha256": sha256_file(worker_path)},
        "python": {"path": sys.executable, "version": platform.python_version()},
        "worker_result_path": str((output_root / "worker-result.json").resolve()),
        "raw_report_path": str((output_root / "raw-report.json").resolve()),
        "receipt_path": str((output_root / "receipt.json").resolve()),
    }
    plan = {**plan_payload, "plan_sha256": digest_value(plan_payload)}
    _publish_create_only(output_root / "plan.json", plan, context="probe plan")
    ledger_path = output_root / "ledger.json"
    if ledger_path.exists():
        ledger = _read_ledger(ledger_path)
        _require(
            ledger.get("plan_sha256") == plan["plan_sha256"], "ledger plan mismatch"
        )
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
    root = _safe_path(root, context="probe_root")
    plan = _read_object(root / "plan.json")
    _exact_keys(
        plan,
        {
            "schema",
            "probe_id",
            "catalog",
            "round1_data",
            "parent_seed",
            "parent_checkpoint",
            "source_contract",
            "host_identity_sha256",
            "runtime",
            "runtime_environment",
            "batch_prefix_sha256",
            "allowed_splits",
            "repository_root",
            "worker",
            "python",
            "worker_result_path",
            "raw_report_path",
            "receipt_path",
            "plan_sha256",
        },
        "probe plan",
    )
    payload = {key: row for key, row in plan.items() if key != "plan_sha256"}
    _require(
        plan.get("schema") == PLAN_SCHEMA
        and plan.get("plan_sha256") == digest_value(payload),
        "probe plan integrity check failed",
    )
    probe_id = _safe_identifier(plan.get("probe_id"), context="plan.probe_id")
    _require(
        (root / ROOT_SENTINEL).read_text(encoding="utf-8").strip() == probe_id,
        "probe root sentinel differs",
    )
    ledger = _read_ledger(root / "ledger.json")
    _require(ledger.get("plan_sha256") == plan["plan_sha256"], "ledger plan mismatch")

    repository_root = Path(plan["repository_root"]).resolve()
    _require(
        _not_inside(root, repository_root),
        "probe root moved inside the source checkout",
    )
    _require(
        _source_contract(repository_root) == plan["source_contract"],
        "probe source commit or clean-worktree contract drifted",
    )
    _require(
        _host_identity_sha256() == plan["host_identity_sha256"],
        "probe host identity drifted",
    )
    _require(
        bool(_SHA256.fullmatch(str(plan["host_identity_sha256"])))
        and bool(_COMMIT.fullmatch(str(plan["source_contract"].get("commit", "")))),
        "probe host/source identity is malformed",
    )
    worker = plan["worker"]
    worker_path = (repository_root / WORKER_RELATIVE_PATH).resolve()
    _require(
        worker == {"path": str(worker_path), "sha256": sha256_file(worker_path)},
        "probe worker source drifted",
    )
    _require(
        plan["python"]
        == {"path": sys.executable, "version": platform.python_version()},
        "probe Python runtime drifted",
    )
    catalog_path = _safe_path(plan["catalog"]["path"], context="catalog_path")
    _require(
        sha256_file(catalog_path) == plan["catalog"]["file_sha256"],
        "probe catalog file drifted",
    )
    catalog = validate_catalog(_read_object(catalog_path))
    _require(
        catalog["catalog_sha256"] == plan["catalog"]["catalog_sha256"],
        "probe catalog value contract drifted",
    )
    round1_data = _validate_round1_data_contract(
        Path(plan["round1_data"]["root"]), catalog
    )
    _require(round1_data == plan["round1_data"], "probe Round-1 data binding drifted")
    _require(
        plan["allowed_splits"] == ["fit", "selection"],
        "probe split access is not the registered non-held-out pair",
    )
    runtime = _runtime(plan["runtime"])
    _require(runtime == plan["runtime"], "probe runtime is not normalized")
    parent = plan["parent_checkpoint"]
    parent_path = _safe_path(parent["path"], context="parent_checkpoint")
    _require(
        parent_path.is_file()
        and parent["sha256"] == sha256_file(parent_path)
        and parent["bytes"] == parent_path.stat().st_size,
        "registered parent checkpoint drifted",
    )
    _require(
        _checkpoint_contract(parent_path) == parent["contract"],
        "registered parent checkpoint contract drifted",
    )
    _require(
        plan["parent_seed"] in catalog["promotion_seeds"],
        "probe parent seed drifted",
    )
    seed_row = next(
        row for row in round1_data["seeds"] if row["seed"] == plan["parent_seed"]
    )
    observed_prefix = _batch_prefix_sha256(
        Path(seed_row["batch_plan"]["path"]), runtime["optimizer_steps"]
    )
    _require(
        observed_prefix == plan["batch_prefix_sha256"],
        "probe optimizer batch prefix drifted",
    )
    expected_paths = {
        "worker_result_path": root / "worker-result.json",
        "raw_report_path": root / "raw-report.json",
        "receipt_path": root / "receipt.json",
    }
    for key, path in expected_paths.items():
        _require(Path(plan[key]) == path.resolve(), f"probe {key} drifted")
    return plan, ledger


def _all_model_parameters_finite(model: Any, torch: Any) -> bool:
    return all(
        bool(torch.all(torch.isfinite(parameter.detach())))
        for parameter in model.parameters()
    )


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


def _assert_parent_holds_gpu_lock() -> None:
    """Fail before CUDA initialization unless our parent owns workflow GPU 0."""

    lock_path = (GPU_LOCK_ROOT / f"gcicy-tn-gpu-{GPU_ID}.lock").resolve()
    try:
        owner = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AutoResearchError(
            "probe worker cannot read the GPU lock owner"
        ) from error
    _require(
        isinstance(owner, dict)
        and owner.get("gpu_id") == GPU_ID
        and owner.get("pid") == os.getppid(),
        "probe worker parent does not own the registered GPU lock",
    )
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    raise AutoResearchError("registered GPU lock is not held")


def execute_cuda_worker(root: Path) -> dict[str, Any]:
    """Run the real CUDA workload.  This function is only called in a child process."""

    plan, _ = _verify_plan(root)
    _assert_parent_holds_gpu_lock()
    _require(
        _runtime_environment() == plan["runtime_environment"],
        "probe runtime environment drifted inside the GPU lock",
    )
    result_path = Path(plan["worker_result_path"])
    if result_path.exists():
        return _validate_worker_result(plan, _read_object(result_path))

    try:
        import torch

        from scripts.evaluate_generic_quintic_h4_architecture_arms import (
            build_checkpoint_model,
        )
        from scripts.refine_generic_quintic_compiled_tree_native_gn import (
            load_disjoint_splits,
            load_fixed_split_indices,
        )
        from scripts.train_generic_quintic_adaptive_tree_rank import train_arm
        from scripts.train_generic_quintic_compiled_tree import whiten_dataset
        from scripts.train_quintic_native_power_lift_tree import make_dataset
    except ImportError as error:  # pragma: no cover - production dependency
        raise AutoResearchError(
            "probe numerical worker dependencies are unavailable"
        ) from error

    runtime = plan["runtime"]
    _require(torch.cuda.is_available(), "probe worker cannot see CUDA")
    torch.set_num_threads(runtime["threads"])
    torch.manual_seed(plan["parent_seed"])
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started_utc = _utc_now()

    parent_path = Path(plan["parent_checkpoint"]["path"])
    payload = torch.load(parent_path, map_location="cpu", weights_only=False)
    model = build_checkpoint_model(payload, device=device)
    model.train()
    exponents = np.asarray(payload["configuration"]["exponents"], dtype=np.int64)
    whitening = np.asarray(payload["configuration"]["whitening"], dtype=np.complex128)
    roles = plan["round1_data"]["input_roles"]
    arrays, _ = load_disjoint_splits(
        {
            "fit": (
                Path(roles["search-native-points"]["path"]),
                Path(roles["search-native-pullbacks"]["path"]),
                FIT_COUNT,
            ),
            "selection": (
                Path(roles["search-native-points"]["path"]),
                Path(roles["search-native-pullbacks"]["path"]),
                SELECTION_COUNT,
            ),
        },
        seed=next(
            row["data_seed"]
            for row in plan["round1_data"]["seeds"]
            if row["seed"] == plan["parent_seed"]
        ),
        fixed_indices=load_fixed_split_indices(
            Path(plan["round1_data"]["index_manifest"]["path"])
        ),
    )
    datasets = {
        name: whiten_dataset(
            make_dataset(
                rows,
                exponents,
                feature_batch_size=runtime["feature_batch_size"],
                complex_dtype=torch.complex64,
                device=device,
            ),
            whitening,
        )
        for name, rows in arrays.items()
    }
    source_seed = next(
        row
        for row in plan["round1_data"]["seeds"]
        if row["seed"] == plan["parent_seed"]
    )
    full_batch_plan = np.load(source_seed["batch_plan"]["path"], allow_pickle=False)
    batch_plan = np.asarray(
        full_batch_plan[: runtime["optimizer_steps"]], dtype=np.int64
    )
    _require(
        canonical_array_value_sha256(batch_plan) == plan["batch_prefix_sha256"],
        "worker batch prefix differs from the sealed plan",
    )
    trained = train_arm(
        model,
        training=datasets["fit"],
        selection=datasets["selection"],
        batch_plan=batch_plan,
        learning_rate=runtime["learning_rate"],
        eval_every=runtime["optimizer_steps"],
        train_chunk_size=runtime["train_chunk_size"],
        eval_batch_size=runtime["eval_batch_size"],
        gradient_clip_norm=runtime["gradient_clip_norm"],
        maximum_tail_degradation=1.0,
        scheduler="constant",
    )
    torch.cuda.synchronize(device)
    history = trained["history"]
    _require(
        history and int(history[-1]["step"]) == runtime["optimizer_steps"],
        "worker did not complete the registered optimizer steps",
    )
    statistics = trained["best_statistics"]
    minimum = float(statistics["min_eigenvalue_weighted_quantiles"]["q0.0000"])
    nonpositive = int(statistics["nonpositive_min_eigenvalue"]["count"])
    all_finite = bool(
        _all_numeric_values_finite(
            {
                "initial_statistics": trained["initial_statistics"],
                "initial_tail": trained["initial_tail"],
                "best_statistics": trained["best_statistics"],
                "best_tail": trained["best_tail"],
                "history": history,
            }
        )
        and _all_model_parameters_finite(trained["model"], torch)
    )
    maximum_allocated = int(torch.cuda.max_memory_allocated(device))
    finished_utc = _utc_now()
    worker_result = {
        "schema": WORKER_RESULT_SCHEMA,
        "probe_id": plan["probe_id"],
        "plan_sha256": plan["plan_sha256"],
        "source_commit": plan["source_contract"]["commit"],
        "host_identity_sha256": plan["host_identity_sha256"],
        "checkpoint_sha256": plan["parent_checkpoint"]["sha256"],
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "device": "cuda",
        "optimizer_steps_completed": runtime["optimizer_steps"],
        "selection_evaluation_completed": True,
        "all_finite": all_finite,
        "minimum_metric_eigenvalue": minimum,
        "nonpositive_metric_count": nonpositive,
        "maximum_allocated_bytes": maximum_allocated,
        "batch_prefix_sha256": plan["batch_prefix_sha256"],
        "selection_metrics": {
            "sigma": float(statistics["sigma_official_formula"]),
            "chi": float(statistics["weighted_rms_abs_residual"]),
        },
    }
    validated = _validate_worker_result(plan, worker_result)
    _publish_create_only(result_path, validated, context="probe worker result")
    return validated


def _validate_worker_result(
    plan: Mapping[str, Any], value: Mapping[str, Any]
) -> dict[str, Any]:
    required = {
        "schema",
        "probe_id",
        "plan_sha256",
        "source_commit",
        "host_identity_sha256",
        "checkpoint_sha256",
        "started_utc",
        "finished_utc",
        "device",
        "optimizer_steps_completed",
        "selection_evaluation_completed",
        "all_finite",
        "minimum_metric_eigenvalue",
        "nonpositive_metric_count",
        "maximum_allocated_bytes",
        "batch_prefix_sha256",
        "selection_metrics",
    }
    _exact_keys(value, required, "worker result")
    _require(value["schema"] == WORKER_RESULT_SCHEMA, "worker result schema differs")
    expected_identity = {
        "probe_id": plan["probe_id"],
        "plan_sha256": plan["plan_sha256"],
        "source_commit": plan["source_contract"]["commit"],
        "host_identity_sha256": plan["host_identity_sha256"],
        "checkpoint_sha256": plan["parent_checkpoint"]["sha256"],
        "batch_prefix_sha256": plan["batch_prefix_sha256"],
    }
    _require(
        all(value[key] == expected for key, expected in expected_identity.items()),
        "worker result identity binding differs",
    )
    started = _parse_utc(value["started_utc"], context="worker.started_utc")
    finished = _parse_utc(value["finished_utc"], context="worker.finished_utc")
    _require(started <= finished, "worker timestamps are reversed")
    _require(value["device"] == "cuda", "worker did not execute on CUDA")
    steps = _positive_int(
        value["optimizer_steps_completed"],
        context="worker.optimizer_steps_completed",
        minimum=DEFAULT_OPTIMIZER_STEPS,
        maximum=MAXIMUM_OPTIMIZER_STEPS,
    )
    _require(
        steps == plan["runtime"]["optimizer_steps"],
        "worker optimizer-step count differs from the plan",
    )
    _require(
        value["selection_evaluation_completed"] is True,
        "worker did not complete selection evaluation",
    )
    _require(isinstance(value["all_finite"], bool), "worker all_finite is malformed")
    minimum = _finite(
        value["minimum_metric_eigenvalue"],
        context="worker.minimum_metric_eigenvalue",
    )
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
    _exact_keys(metrics, {"sigma", "chi"}, "worker selection metrics")
    normalized_metrics = {
        key: _finite(metrics[key], context=f"worker.selection_metrics.{key}")
        for key in ("sigma", "chi")
    }
    _require(
        all(row > 0.0 for row in normalized_metrics.values()),
        "worker selection metrics are outside their domain",
    )
    return {
        **dict(value),
        "started_utc": started.isoformat().replace("+00:00", "Z"),
        "finished_utc": finished.isoformat().replace("+00:00", "Z"),
        "optimizer_steps_completed": steps,
        "minimum_metric_eigenvalue": minimum,
        "nonpositive_metric_count": nonpositive,
        "maximum_allocated_bytes": memory,
        "selection_metrics": normalized_metrics,
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


def _data_binding(plan: Mapping[str, Any]) -> dict[str, Any]:
    round1 = plan["round1_data"]
    seed_row = next(
        row for row in round1["seeds"] if row["seed"] == plan["parent_seed"]
    )
    return {
        "catalog_file_sha256": plan["catalog"]["file_sha256"],
        "catalog_sha256": plan["catalog"]["catalog_sha256"],
        "round1_plan_file_sha256": round1["plan_file_sha256"],
        "round1_plan_sha256": round1["plan_sha256"],
        "search_indices_sha256": round1["search_indices_sha256"],
        "fixed_indices_file_sha256": round1["index_manifest"]["file_sha256"],
        "fixed_indices_value_set_sha256": round1["index_manifest"]["value_set_sha256"],
        "input_sha256": {
            role: row["sha256"] for role, row in sorted(round1["input_roles"].items())
        },
        "batch_plan_file_sha256": seed_row["batch_plan"]["file_sha256"],
        "batch_plan_value_set_sha256": seed_row["batch_plan"]["value_set_sha256"],
        "batch_prefix_sha256": plan["batch_prefix_sha256"],
        "allowed_splits": ["fit", "selection"],
    }


def _raw_report(plan: Mapping[str, Any], worker: Mapping[str, Any]) -> dict[str, Any]:
    worker_path = Path(plan["worker_result_path"])
    return {
        "schema": RAW_REPORT_SCHEMA,
        "probe_id": plan["probe_id"],
        "plan_path": str(
            (Path(plan["raw_report_path"]).parent / "plan.json").resolve()
        ),
        "plan_sha256": plan["plan_sha256"],
        "source_contract": plan["source_contract"],
        "host_identity_sha256": plan["host_identity_sha256"],
        "parent_checkpoint": plan["parent_checkpoint"],
        "data_binding": _data_binding(plan),
        "worker_result_path": str(worker_path),
        "worker_result_sha256": sha256_file(worker_path),
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
    report_path = Path(plan["raw_report_path"])
    return {
        "schema": GPU_PROBE_SCHEMA,
        "probe_id": plan["probe_id"],
        "checkpoint_sha256": plan["parent_checkpoint"]["sha256"],
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


def _finalize(
    plan: Mapping[str, Any], ledger: dict[str, Any], root: Path
) -> dict[str, Any]:
    verified_plan, verified_ledger = _verify_plan(root)
    _require(verified_plan == plan, "probe plan changed before finalization")
    _require(
        verified_ledger["state"] == "worker-complete",
        "probe cannot finalize before its worker completes",
    )
    ledger = verified_ledger
    worker_path = Path(plan["worker_result_path"])
    _require(worker_path.is_file(), "probe worker result is missing")
    worker = _validate_worker_result(plan, _read_object(worker_path))
    raw = _raw_report(plan, worker)
    raw_path = Path(plan["raw_report_path"])
    _publish_create_only(raw_path, raw, context="probe raw report")
    receipt = _receipt(plan, raw)
    receipt_path = Path(plan["receipt_path"])
    _publish_create_only(receipt_path, receipt, context="probe receipt")
    normalized = normalize_gpu_probe(
        receipt,
        index=0,
        expected_checkpoint_sha256=plan["parent_checkpoint"]["sha256"],
        expected_host_identity_sha256=plan["host_identity_sha256"],
        expected_source_commit=plan["source_contract"]["commit"],
        issued_at=_parse_utc(receipt["finished_utc"], context="receipt.finished_utc"),
    )
    ledger["state"] = "complete" if normalized["passes"] else "diagnostic-only"
    ledger["worker_result_sha256"] = sha256_file(worker_path)
    ledger["raw_report_sha256"] = sha256_file(raw_path)
    ledger["receipt_sha256"] = sha256_file(receipt_path)
    _write_ledger(root / "ledger.json", ledger)
    return receipt


def _run_probe_with_gpu_lock_held(root: Path) -> dict[str, Any]:
    """Revalidate and execute while workflow GPU 0 is exclusively locked."""

    plan, ledger = _verify_plan(root)
    _require(
        _runtime_environment() == plan["runtime_environment"],
        "probe runtime environment drifted inside the GPU lock",
    )
    state = ledger["state"]
    if state == "failed":
        raise AutoResearchError("failed probe roots are immutable; use a new probe_id")
    if state == "worker-complete":
        return _finalize(plan, ledger, root)
    if state == "running":
        if Path(plan["worker_result_path"]).is_file():
            ledger["state"] = "worker-complete"
            _write_ledger(root / "ledger.json", ledger)
            return _finalize(plan, ledger, root)
        raise AutoResearchError(
            "probe optimizer execution is ambiguous; use a new probe_id/root"
        )
    _require(state == "prepared", "probe ledger state is invalid")

    ledger["state"] = "running"
    _write_ledger(root / "ledger.json", ledger)
    completed = _launch_worker(plan, root)
    ledger = _read_ledger(root / "ledger.json")
    ledger["worker_returncode"] = completed.returncode
    ledger["worker_stdout_sha256"] = hashlib.sha256(
        completed.stdout.encode("utf-8")
    ).hexdigest()
    ledger["worker_stderr_sha256"] = hashlib.sha256(
        completed.stderr.encode("utf-8")
    ).hexdigest()
    if completed.returncode != 0:
        ledger["state"] = "failed"
        _write_ledger(root / "ledger.json", ledger)
        raise AutoResearchError("CUDA probe worker failed; use a new probe_id/root")
    _require(
        Path(plan["worker_result_path"]).is_file(),
        "CUDA probe worker returned success without a result",
    )
    _validate_worker_result(plan, _read_object(Path(plan["worker_result_path"])))
    ledger["state"] = "worker-complete"
    ledger["worker_result_sha256"] = sha256_file(Path(plan["worker_result_path"]))
    _write_ledger(root / "ledger.json", ledger)
    return _finalize(plan, ledger, root)


def run_probe(root: Path) -> dict[str, Any]:
    """Execute or safely finalize a prepared probe root under workflow GPU 0."""

    root = _safe_path(root, context="probe_root")
    plan, ledger = _verify_plan(root)
    if ledger["state"] in {"complete", "diagnostic-only"}:
        verified = verify_probe(root, require_passing=False)
        return verified["receipt"]
    _require(
        plan["runtime"]["device"] == "cuda",
        "probe execution is not registered for CUDA",
    )
    with gpu_lock(GPU_LOCK_ROOT, GPU_ID, timeout_seconds=-1):
        return _run_probe_with_gpu_lock_held(root)


def verify_probe(root: Path, *, require_passing: bool = True) -> dict[str, Any]:
    """Rehash every probe artifact and optionally require a passing receipt."""

    root = _safe_path(root, context="probe_root")
    plan, ledger = _verify_plan(root)
    _require(
        ledger["state"] in {"complete", "diagnostic-only"},
        "probe has no terminal receipt",
    )
    worker_path = Path(plan["worker_result_path"])
    raw_path = Path(plan["raw_report_path"])
    receipt_path = Path(plan["receipt_path"])
    _require(
        worker_path.is_file() and raw_path.is_file() and receipt_path.is_file(),
        "probe terminal artifacts are missing",
    )
    worker = _validate_worker_result(plan, _read_object(worker_path))
    raw = _read_object(raw_path)
    _require(raw == _raw_report(plan, worker), "probe raw report was forged or drifted")
    receipt = _read_object(receipt_path)
    _require(receipt == _receipt(plan, raw), "probe receipt was forged or drifted")
    _require(
        ledger["worker_result_sha256"] == sha256_file(worker_path)
        and ledger["raw_report_sha256"] == sha256_file(raw_path)
        and ledger["receipt_sha256"] == sha256_file(receipt_path),
        "probe ledger artifact hashes drifted",
    )
    normalized = normalize_gpu_probe(
        receipt,
        index=0,
        expected_checkpoint_sha256=plan["parent_checkpoint"]["sha256"],
        expected_host_identity_sha256=plan["host_identity_sha256"],
        expected_source_commit=plan["source_contract"]["commit"],
        issued_at=_parse_utc(receipt["finished_utc"], context="receipt.finished_utc"),
    )
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
    plan, ledger = _verify_plan(root)
    result = {
        "probe_id": plan["probe_id"],
        "state": ledger["state"],
        "plan_sha256": plan["plan_sha256"],
        "parent_checkpoint_sha256": plan["parent_checkpoint"]["sha256"],
        "optimizer_steps": plan["runtime"]["optimizer_steps"],
        "receipt_path": plan["receipt_path"],
    }
    if ledger["state"] in {"complete", "diagnostic-only"}:
        verified = verify_probe(root, require_passing=False)
        result["passes"] = verified["normalized_receipt"]["passes"]
        result["maximum_allocated_bytes"] = verified["receipt"][
            "maximum_allocated_bytes"
        ]
        result["minimum_metric_eigenvalue"] = verified["receipt"][
            "minimum_metric_eigenvalue"
        ]
    return result


__all__ = [
    "DEFAULT_OPTIMIZER_STEPS",
    "LEDGER_SCHEMA",
    "PLAN_SCHEMA",
    "RAW_REPORT_SCHEMA",
    "ROOT_SENTINEL",
    "WORKER_RESULT_SCHEMA",
    "execute_cuda_worker",
    "prepare_probe",
    "probe_status",
    "run_probe",
    "verify_probe",
]
