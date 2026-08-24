"""Fail-closed paired Auto Research bridge for quintic Rounds 3 and 4.

The adapter executes exactly two catalogued comparisons:

* Round 3: internal+cosine versus all-parameter+cosine;
* Round 4: all-parameter+cosine versus all-parameter+constant.

Both arms start from the same per-seed parent checkpoint.  The bridge reuses
the immutable Round-1 development indices and matched batch plans, runs 600
updates in complex64 development-only mode, and has no confirmation or blind
entry point.  It never interprets catalog values as commands.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
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
from .host_stability_gate import HostStabilityError, validate_certificate
from .experiment_workflow import gpu_lock
from .quintic_architecture_multi_round import (
    MultiRoundManager,
    validate_catalog,
    validate_execution_handoff,
)
from .quintic_architecture_round1_bridge import (
    BRIDGE_PLAN_SCHEMA as ROUND1_BRIDGE_PLAN_SCHEMA,
    canonical_array_value_sha256,
)


PAIRED_PLAN_SCHEMA = "gcicy-quintic-paired-auto-research-plan-v2"
PAIRED_LEDGER_SCHEMA = "gcicy-quintic-paired-auto-research-ledger-v1"
PAIRED_EVIDENCE_SCHEMA = "gcicy-quintic-paired-auto-research-evidence-v1"
PAIRED_ADJUDICATION_SCHEMA = "gcicy-quintic-paired-auto-research-adjudication-v1"
PAIRED_AUTHORIZATION_SCHEMA = "gcicy-quintic-paired-auto-research-authorization-v1"
PAIRED_ATTEMPT_SEAL_SCHEMA = "gcicy-quintic-paired-auto-research-attempt-seal-v1"
PAIRED_SENTINEL = ".gcicy-quintic-paired-auto-research-root"

REGISTERED_RECIPES = {
    "r3-all-parameter-joint": {
        "round": 3,
        "category": "parameter-scope",
        "required_adapter": "quintic-paired-parameter-scope-v2",
        "control": {"trainable_scope": "internal", "scheduler": "cosine"},
        "candidate": {"trainable_scope": "all", "scheduler": "cosine"},
    },
    "r4-constant-optimizer-path": {
        "round": 4,
        "category": "optimizer-path",
        "required_adapter": "quintic-paired-optimizer-path-v2",
        "control": {"trainable_scope": "all", "scheduler": "cosine"},
        "candidate": {"trainable_scope": "all", "scheduler": "constant"},
    },
}

WORKER_RELATIVE_PATH = "scripts/compare_generic_quintic_tree_joint_relaxation.py"
SOURCE_RELATIVE_PATHS = (
    "gcicy_metric/pipeline/architecture_auto_research.py",
    "gcicy_metric/pipeline/host_stability_gate.py",
    "gcicy_metric/pipeline/positive_multiplication_tree.py",
    "gcicy_metric/pipeline/quintic_architecture_multi_round.py",
    "gcicy_metric/pipeline/quintic_architecture_round1_bridge.py",
    "gcicy_metric/pipeline/quintic_paired_auto_research_bridge.py",
    "scripts/evaluate_generic_quintic_h4_architecture_arms.py",
    "scripts/refine_generic_quintic_compiled_tree_native_gn.py",
    "scripts/run_quintic_paired_auto_research_bridge.py",
    "scripts/train_generic_quintic_adaptive_tree_rank.py",
    "scripts/train_generic_quintic_compiled_tree.py",
    "scripts/train_quintic_full_h_same_points.py",
    "scripts/train_quintic_native_power_lift_tree.py",
)

FIT_COUNT = 30_000
SELECTION_COUNT = 5_000
DEVELOPMENT_EVALUATION_COUNT = 5_000
OPTIMIZER_UPDATES = 600
EVAL_EVERY = 25
PRECISION = "complex64"
GPU_LOCK_ROOT = Path("/tmp/gcicy-tn-gpu-locks")
GPU_ID = "0"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _read_catalog(path: Path) -> dict[str, Any]:
    value = _read_object(path)
    normalized = validate_catalog(
        {key: row for key, row in value.items() if key != "catalog_sha256"}
    )
    _require(normalized == value, "manager catalog lock was modified")
    return normalized


def _safe_path(value: Any, *, name: str) -> Path:
    path = Path(str(value)).expanduser().resolve()
    lowered = str(path).lower()
    if any(token in lowered for token in ("blind", "shadow", "confirmation")):
        raise AutoResearchError(f"{name} references forbidden held-out data")
    return path


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AutoResearchError(f"{name} must be a positive integer")
    return value


def _finite(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise AutoResearchError(f"{name} must be numeric") from error
    if not math.isfinite(result):
        raise AutoResearchError(f"{name} must be finite")
    return result


def _validate_runtime(runtime: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "device",
        "threads",
        "train_chunk_size",
        "feature_batch_size",
        "eval_batch_size",
    }
    if set(runtime) != required:
        raise AutoResearchError("runtime fields are not exact")
    device = str(runtime["device"])
    if device != "cuda":
        raise AutoResearchError(
            "scientific paired execution requires runtime.device=cuda"
        )
    return {
        "device": device,
        **{
            key: _positive_int(runtime[key], name=f"runtime.{key}")
            for key in required - {"device"}
        },
    }


def _runtime_environment(device: str) -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - production environment contract
        raise AutoResearchError("PyTorch is unavailable") from error
    result: dict[str, Any] = {
        "numpy": str(np.__version__),
        "torch": str(torch.__version__),
        "torch_cuda_build": torch.version.cuda,
        "gpu": None,
    }
    if device == "cuda":
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
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
            "paired bridge requires exactly one inspectable CUDA GPU",
        )
        fields = [field.strip() for field in rows[0].split(",")]
        _require(len(fields) == 3, "nvidia-smi output is malformed")
        result["gpu"] = {
            "name": fields[0],
            "driver_version": fields[1],
            "memory_total_mib": _positive_int(
                int(fields[2]), name="gpu.memory_total_mib"
            ),
        }
    return result


def _host_identity_sha256() -> str:
    """Resolve the same host identity used by ``certify_host_stability.py``."""

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


def _source_contract(repository_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            shell=False,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            shell=False,
        ).stdout.strip()
    except subprocess.CalledProcessError as error:
        raise AutoResearchError("cannot bind the git source revision") from error
    _require(len(commit) == 40, "git commit is malformed")
    _require(not status, "paired execution requires a clean tracked git worktree")
    dependencies = {}
    for relative in SOURCE_RELATIVE_PATHS:
        path = (repository_root / relative).resolve()
        _require(path.is_file(), f"registered dependency is missing: {relative}")
        dependencies[relative] = sha256_file(path)
    return {
        "commit": commit,
        "tracked_worktree": "clean",
        "dependencies": dependencies,
    }


def _checkpoint_precision(path: Path) -> str:
    """Read the trusted scientific parent contract on CPU before GPU launch."""

    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        precision = str(payload["configuration"]["precision"])
    except (ImportError, KeyError, TypeError, ValueError, OSError) as error:
        raise AutoResearchError("cannot read parent checkpoint precision") from error
    if precision != PRECISION:
        raise AutoResearchError("paired parent checkpoint is not complex64")
    return precision


def _create_only_json(path: Path, value: Mapping[str, Any], *, name: str) -> None:
    if path.exists():
        if _read_object(path) != dict(value):
            raise AutoResearchError(f"{name} already differs")
        return
    atomic_write_json(path, value)


def _seal_ledger(value: dict[str, Any]) -> None:
    payload = {key: row for key, row in value.items() if key != "state_sha256"}
    value["state_sha256"] = digest_value(payload)


def _write_ledger(path: Path, value: dict[str, Any]) -> None:
    _seal_ledger(value)
    atomic_write_json(path, value)


def _read_ledger(path: Path) -> dict[str, Any]:
    value = _read_object(path)
    observed = value.get("state_sha256")
    payload = {key: row for key, row in value.items() if key != "state_sha256"}
    _require(
        value.get("schema") == PAIRED_LEDGER_SCHEMA
        and observed == digest_value(payload),
        "paired ledger integrity check failed",
    )
    return value


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        artifact = np.load(path, allow_pickle=False)
        return {name: np.asarray(artifact[name]) for name in artifact.files}
    except (OSError, ValueError) as error:
        raise AutoResearchError(f"cannot load fixed index artifact {path}") from error


def _batch_digest(array: np.ndarray) -> str:
    """Match the comparator's path-independent batch-plan digest."""

    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _validate_recipe(catalog: Mapping[str, Any], recipe_id: str) -> dict[str, Any]:
    expected = REGISTERED_RECIPES.get(recipe_id)
    if expected is None:
        raise AutoResearchError("paired bridge recipe is not registered")
    recipe = next(
        (row for row in catalog["recipes"] if row["recipe_id"] == recipe_id), None
    )
    _require(isinstance(recipe, dict), "registered recipe is absent from the catalog")
    _require(
        recipe["round"] == expected["round"]
        and recipe["category"] == expected["category"]
        and recipe["adapter"]["status"] == "available"
        and recipe["adapter"]["required_adapter"] == expected["required_adapter"],
        "catalog recipe does not match this adapter",
    )
    intervention = recipe["intervention"]
    for role in ("control", "candidate"):
        _require(
            {
                "trainable_scope": intervention[role]["trainable_scope"],
                "scheduler": intervention[role]["scheduler"],
            }
            == expected[role],
            f"{recipe_id} {role} policy is not the registered paired policy",
        )
    budget = recipe["budget"]
    _require(
        budget
        == {
            "local_activate_optimizer_updates": 0,
            "matched_relax_optimizer_updates_per_arm": OPTIMIZER_UPDATES,
            "batch_size": 1024,
            "learning_rate": 3.0e-6,
            "gradient_clip_norm": 1.0,
            "fit_examples": FIT_COUNT,
            "selection_examples": SELECTION_COUNT,
            "development_evaluation_examples": DEVELOPMENT_EVALUATION_COUNT,
        },
        "paired recipe budget is not the fixed 600-update contract",
    )
    return dict(recipe)


def _validate_array_manifest(
    manifest: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    *,
    context: str,
) -> None:
    expected_rows = manifest.get("arrays")
    _require(isinstance(expected_rows, dict), f"{context} array manifest is absent")
    _require(set(expected_rows) == set(arrays), f"{context} arrays are not exact")
    for name, array in arrays.items():
        contiguous = np.ascontiguousarray(array)
        row = expected_rows[name]
        _require(
            isinstance(row, dict)
            and row.get("dtype") == contiguous.dtype.str
            and row.get("shape") == list(contiguous.shape)
            and row.get("value_sha256") == canonical_array_value_sha256(contiguous),
            f"{context}.{name} value manifest differs",
        )
    payload = {"schema": manifest.get("schema"), "arrays": expected_rows}
    _require(
        manifest.get("value_set_sha256") == digest_value(payload),
        f"{context} value-set digest differs",
    )


def _validate_round1_data_contract(
    round1_bridge_root: Path,
    catalog: Mapping[str, Any],
) -> dict[str, Any]:
    root = _safe_path(round1_bridge_root, name="round1_bridge_root")
    plan_path = root / "plan.json"
    plan = _read_object(plan_path)
    payload = {key: value for key, value in plan.items() if key != "plan_sha256"}
    _require(
        plan.get("schema") == ROUND1_BRIDGE_PLAN_SCHEMA
        and plan.get("plan_sha256") == digest_value(payload),
        "Round-1 bridge plan integrity check failed",
    )
    data = catalog["data_contract"]
    _require(
        plan.get("protocol_sha256") == data["protocol_sha256"]
        and plan.get("search_indices_sha256") == data["search_indices_sha256"],
        "Round-1 protocol/index hashes differ from the catalog",
    )
    _require(
        plan.get("historical_confirmation") == "absent",
        "Round-1 source plan is not development-only",
    )
    roles = plan.get("input_roles")
    _require(isinstance(roles, dict), "Round-1 input roles are absent")
    _require(
        set(roles)
        == {
            "baseline-checkpoint",
            "baseline-source-report",
            "search-native-points",
            "search-native-pullbacks",
            "development-evaluation-points",
            "development-evaluation-pullbacks",
        },
        "Round-1 input roles are not the exact development-only set",
    )
    for role, expected_sha256 in data["input_sha256"].items():
        row = roles.get(role)
        _require(isinstance(row, dict), f"Round-1 input role is absent: {role}")
        path = _safe_path(row.get("path"), name=f"Round-1 input {role}")
        _require(
            path.is_file()
            and row.get("sha256") == expected_sha256
            and sha256_file(path) == expected_sha256,
            f"Round-1 input hash differs: {role}",
        )
    index = plan.get("index_manifest")
    _require(isinstance(index, dict), "Round-1 fixed-index manifest is absent")
    index_path = _safe_path(index.get("path"), name="Round-1 fixed indices")
    _require(
        index_path.is_file() and sha256_file(index_path) == index.get("file_sha256"),
        "Round-1 fixed-index file drifted",
    )
    index_arrays = _load_npz(index_path)
    _require(
        set(index_arrays) == {"fit", "selection", "development_evaluation"}
        and len(index_arrays["fit"]) == FIT_COUNT
        and len(index_arrays["selection"]) == SELECTION_COUNT
        and len(index_arrays["development_evaluation"]) == DEVELOPMENT_EVALUATION_COUNT,
        "Round-1 fixed-index values/counts differ",
    )
    _require(
        all(
            array.ndim == 1 and np.issubdtype(array.dtype, np.integer)
            for array in index_arrays.values()
        )
        and all(len(np.unique(array)) == len(array) for array in index_arrays.values())
        and np.intersect1d(index_arrays["fit"], index_arrays["selection"]).size == 0,
        "Round-1 fixed-index values are malformed or overlap",
    )
    _validate_array_manifest(index, index_arrays, context="Round-1 fixed indices")

    seed_rows = plan.get("seeds")
    _require(isinstance(seed_rows, list), "Round-1 seed plan is absent")
    by_seed = {int(row.get("seed", -1)): row for row in seed_rows}
    expected_seeds = catalog["promotion_seeds"]
    _require(sorted(by_seed) == expected_seeds, "Round-1 seeds differ from catalog")
    catalog_batches = {row["seed"]: row for row in data["matched_batch_plans"]}
    normalized_seeds = []
    for seed in expected_seeds:
        seed_row = by_seed[seed]
        batch = seed_row.get("batch_plan")
        expected_batch = catalog_batches[seed]
        _require(isinstance(batch, dict), f"Round-1 batch plan is absent for {seed}")
        batch_path = _safe_path(
            batch.get("path"), name=f"Round-1 batch plan for {seed}"
        )
        _require(
            batch_path.is_file()
            and batch.get("seed") == expected_batch["plan_seed"]
            and seed_row.get("batch_plan_seed") == expected_batch["plan_seed"]
            and seed_row.get("optimizer_seed") == seed
            and batch.get("steps") == OPTIMIZER_UPDATES
            and batch.get("batch_size") == 1024
            and batch.get("file_sha256") == expected_batch["file_sha256"]
            and batch.get("value_set_sha256") == expected_batch["value_set_sha256"]
            and sha256_file(batch_path) == expected_batch["file_sha256"],
            f"Round-1 fixed batch-plan contract differs for {seed}",
        )
        array = np.load(batch_path, allow_pickle=False)
        _require(
            array.shape == (OPTIMIZER_UPDATES, 1024)
            and np.issubdtype(array.dtype, np.integer)
            and np.all((array >= 0) & (array < FIT_COUNT))
            and all(len(np.unique(row)) == len(row) for row in array),
            f"Round-1 batch-plan values are invalid for {seed}",
        )
        _validate_array_manifest(
            batch, {"batch_plan": array}, context=f"Round-1 batch plan {seed}"
        )
        normalized_seeds.append(
            {
                "seed": seed,
                "optimizer_seed": int(seed_row["optimizer_seed"]),
                "data_seed": int(seed_row["data_seed"]),
                "batch_plan_seed": int(seed_row["batch_plan_seed"]),
                "batch_plan": dict(batch),
            }
        )
    return {
        "root": str(root),
        "plan_path": str(plan_path),
        "plan_file_sha256": sha256_file(plan_path),
        "plan_sha256": plan["plan_sha256"],
        "search_indices_sha256": plan["search_indices_sha256"],
        "index_manifest": dict(index),
        "input_roles": {
            role: dict(roles[role]) for role in sorted(data["input_sha256"])
        },
        "seeds": normalized_seeds,
    }


def _flag(name: str, value: Any) -> list[str]:
    return [name, str(value)]


def _worker_argv(plan: Mapping[str, Any], seed_row: Mapping[str, Any]) -> list[str]:
    recipe = plan["recipe"]
    intervention = recipe["intervention"]
    budget = recipe["budget"]
    runtime = plan["runtime"]
    roles = plan["round1_data"]["input_roles"]
    return [
        plan["python"]["path"],
        plan["worker"]["path"],
        *_flag("--control-checkpoint", seed_row["parent_checkpoint"]["path"]),
        *_flag("--candidate-checkpoint", seed_row["parent_checkpoint"]["path"]),
        *_flag("--train-points", roles["search-native-points"]["path"]),
        *_flag("--train-pullbacks", roles["search-native-pullbacks"]["path"]),
        *_flag("--selection-points", roles["search-native-points"]["path"]),
        *_flag("--selection-pullbacks", roles["search-native-pullbacks"]["path"]),
        *_flag(
            "--development-evaluation-points",
            roles["development-evaluation-points"]["path"],
        ),
        *_flag(
            "--development-evaluation-pullbacks",
            roles["development-evaluation-pullbacks"]["path"],
        ),
        *_flag("--fixed-indices-file", plan["round1_data"]["index_manifest"]["path"]),
        *_flag("--fixed-batch-plan-file", seed_row["batch_plan"]["path"]),
        *_flag("--output-dir", seed_row["output_dir"]),
        *_flag("--steps", OPTIMIZER_UPDATES),
        *_flag("--learning-rate", budget["learning_rate"]),
        *_flag("--scheduler", "cosine"),
        *_flag("--control-scheduler", intervention["control"]["scheduler"]),
        *_flag("--candidate-scheduler", intervention["candidate"]["scheduler"]),
        *_flag(
            "--control-trainable-scope",
            intervention["control"]["trainable_scope"],
        ),
        *_flag(
            "--candidate-trainable-scope",
            intervention["candidate"]["trainable_scope"],
        ),
        "--require-equal-total-parameters",
        *_flag("--eval-every", EVAL_EVERY),
        *_flag("--train-size", FIT_COUNT),
        *_flag("--selection-size", SELECTION_COUNT),
        *_flag("--development-evaluation-size", DEVELOPMENT_EVALUATION_COUNT),
        *_flag("--stochastic-batch-size", budget["batch_size"]),
        *_flag("--gradient-clip-norm", budget["gradient_clip_norm"]),
        *_flag("--minimum-relative-sigma-gain", 0.0),
        *_flag("--minimum-relative-chi-gain", 0.0),
        *_flag(
            "--maximum-selection-tail-relative-degradation",
            recipe["promotion_gate"]["maximum_tail_relative_degradation"],
        ),
        *_flag("--train-chunk-size", runtime["train_chunk_size"]),
        *_flag("--feature-batch-size", runtime["feature_batch_size"]),
        *_flag("--eval-batch-size", runtime["eval_batch_size"]),
        *_flag("--seed", seed_row["optimizer_seed"]),
        *_flag("--data-seed", seed_row["data_seed"]),
        *_flag("--batch-plan-seed", seed_row["batch_plan_seed"]),
        *_flag("--threads", runtime["threads"]),
        *_flag("--device", runtime["device"]),
        "--development-only",
        "--development-evaluation",
    ]


def prepare_paired_bridge(
    *,
    manager_root: Path,
    execution_handoff_path: Path,
    output_root: Path,
    host_stability_certificates: Mapping[int, Path],
    runtime: Mapping[str, Any],
    repository_root: Path,
) -> dict[str, Any]:
    """Create or exactly resume one immutable three-seed paired plan."""

    output_root = _safe_path(output_root, name="output_root")
    repository_root = repository_root.expanduser().resolve()
    manager_root = _safe_path(manager_root, name="manager_root")
    execution_handoff_path = _safe_path(
        execution_handoff_path, name="execution_handoff_path"
    )
    handoff = validate_execution_handoff(
        manager_root, execution_handoff_path, require_current=True
    )
    manager = MultiRoundManager(manager_root)
    manager_state = manager.state()
    catalog_path = manager.catalog_path
    catalog = _read_catalog(catalog_path)
    recipe_id = handoff["recipe_id"]
    recipe = _validate_recipe(catalog, recipe_id)
    round1_data = _validate_round1_data_contract(
        Path(manager_state["round1_bridge_root"]), catalog
    )
    runtime_row = _validate_runtime(runtime)
    seeds = catalog["promotion_seeds"]
    _require(
        sorted(host_stability_certificates) == seeds,
        "host certificates must cover exactly the three registered seeds",
    )
    parent_rows = [
        {
            "seed": row["seed"],
            "path": row["checkpoint_path"],
            "sha256": row["checkpoint_sha256"],
        }
        for row in handoff["parent_artifacts"]
    ]
    worker_path = (repository_root / WORKER_RELATIVE_PATH).resolve()
    _require(worker_path.is_file(), "registered paired worker is missing")
    source_contract = _source_contract(repository_root)
    host_identity_sha256 = _host_identity_sha256()
    certificate_rows = {
        parent["seed"]: _certificate_identity(
            certificate_path=host_stability_certificates[parent["seed"]],
            parent=parent,
            host_identity_sha256=host_identity_sha256,
            source_commit=source_contract["commit"],
        )
        for parent in parent_rows
    }
    output_root.mkdir(parents=True, exist_ok=True)
    sentinel = output_root / PAIRED_SENTINEL
    if sentinel.exists():
        _require(
            sentinel.read_text(encoding="utf-8").strip() == recipe_id,
            "paired root belongs to another recipe",
        )
    elif any(output_root.iterdir()):
        raise AutoResearchError("refusing a non-empty unmanaged paired root")
    else:
        sentinel.write_text(f"{recipe_id}\n", encoding="utf-8")

    r1_seed_rows = {row["seed"]: row for row in round1_data["seeds"]}
    seed_rows = []
    for parent in parent_rows:
        seed = parent["seed"]
        source_seed = r1_seed_rows[seed]
        seed_rows.append(
            {
                "seed": seed,
                "optimizer_seed": source_seed["optimizer_seed"],
                "data_seed": source_seed["data_seed"],
                "batch_plan_seed": source_seed["batch_plan_seed"],
                "parent_checkpoint": parent,
                "host_stability_certificate": certificate_rows[seed],
                "batch_plan": source_seed["batch_plan"],
                "output_dir": str(
                    (output_root / "seeds" / str(seed) / "paired").resolve()
                ),
            }
        )
    plan_payload = {
        "schema": PAIRED_PLAN_SCHEMA,
        "manager_handoff": {
            "manager_root": str(manager_root),
            "path": str(execution_handoff_path),
            "file_sha256": sha256_file(execution_handoff_path),
            "handoff_sha256": handoff["handoff_sha256"],
            "proposal_path": handoff["proposal"]["path"],
            "proposal_file_sha256": handoff["proposal"]["file_sha256"],
            "proposal_sha256": handoff["proposal"]["proposal_sha256"],
            "parent_family": handoff["parent_family"],
            "parent_lineage": handoff["parent_lineage"],
        },
        "recipe_id": recipe_id,
        "recipe": recipe,
        "catalog": {
            "path": str(catalog_path),
            "file_sha256": sha256_file(catalog_path),
            "catalog_sha256": catalog["catalog_sha256"],
        },
        "round1_data": round1_data,
        "repository_root": str(repository_root),
        "worker": {"path": str(worker_path), "sha256": sha256_file(worker_path)},
        "source_contract": source_contract,
        "host_identity_sha256": host_identity_sha256,
        "runtime": runtime_row,
        "runtime_environment": _runtime_environment(runtime_row["device"]),
        "python": {"path": sys.executable, "version": platform.python_version()},
        "precision": PRECISION,
        "seeds": seed_rows,
        "development_evaluation_policy": (
            "post-training paired evidence only; never optimizer, early-stop, "
            "checkpoint-selection, or worker-winner input"
        ),
        "historical_confirmation": "absent",
        "final_blind": "absent",
    }
    plan = {**plan_payload, "plan_sha256": digest_value(plan_payload)}
    _create_only_json(output_root / "plan.json", plan, name="paired plan")
    ledger_path = output_root / "ledger.json"
    if ledger_path.exists():
        ledger = _read_ledger(ledger_path)
        _require(
            ledger.get("plan_sha256") == plan["plan_sha256"], "ledger plan mismatch"
        )
    else:
        ledger = {
            "schema": PAIRED_LEDGER_SCHEMA,
            "plan_sha256": plan["plan_sha256"],
            "state": "prepared",
            "seeds": {str(seed): {"worker": "pending"} for seed in seeds},
            "evidence": None,
            "adjudication": None,
        }
        _write_ledger(ledger_path, ledger)
    return plan


def _verify_report_artifacts(
    report_path: Path,
    *,
    expected_report_sha256: str | None = None,
) -> dict[str, Any]:
    report = _read_object(report_path)
    _require(
        report.get("schema") == "generic-quintic-tree-joint-relaxation-v1",
        "paired worker report schema is invalid",
    )
    observed = sha256_file(report_path)
    _require(
        expected_report_sha256 in {None, observed},
        "paired worker report differs from its sealed ledger hash",
    )
    results = report.get("results")
    _require(isinstance(results, dict), "paired worker results are absent")
    for role in ("control", "candidate"):
        row = results.get(role)
        _require(isinstance(row, dict), f"paired {role} result is absent")
        checkpoint = _safe_path(row.get("checkpoint"), name=f"{role} checkpoint")
        _require(
            checkpoint.is_file()
            and sha256_file(checkpoint) == row.get("checkpoint_sha256"),
            f"paired {role} checkpoint hash differs",
        )
    return report


def _verify_plan(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = _safe_path(root, name="paired root")
    plan = _read_object(root / "plan.json")
    payload = {key: value for key, value in plan.items() if key != "plan_sha256"}
    _require(
        plan.get("schema") == PAIRED_PLAN_SCHEMA
        and plan.get("plan_sha256") == digest_value(payload),
        "paired plan integrity check failed",
    )
    _require(
        (root / PAIRED_SENTINEL).read_text(encoding="utf-8").strip()
        == plan.get("recipe_id"),
        "paired root sentinel differs",
    )
    ledger = _read_ledger(root / "ledger.json")
    _require(ledger.get("plan_sha256") == plan["plan_sha256"], "ledger plan mismatch")

    manager_binding = plan.get("manager_handoff")
    _require(isinstance(manager_binding, dict), "manager handoff binding is absent")
    handoff_path = Path(str(manager_binding.get("path", ""))).resolve()
    _require(
        handoff_path.is_file()
        and sha256_file(handoff_path) == manager_binding.get("file_sha256"),
        "manager handoff file drifted",
    )
    handoff = validate_execution_handoff(
        Path(manager_binding["manager_root"]), handoff_path, require_current=False
    )
    expected_manager_binding = {
        "manager_root": handoff["manager_root"],
        "path": str(handoff_path),
        "file_sha256": sha256_file(handoff_path),
        "handoff_sha256": handoff["handoff_sha256"],
        "proposal_path": handoff["proposal"]["path"],
        "proposal_file_sha256": handoff["proposal"]["file_sha256"],
        "proposal_sha256": handoff["proposal"]["proposal_sha256"],
        "parent_family": handoff["parent_family"],
        "parent_lineage": handoff["parent_lineage"],
    }
    _require(
        manager_binding == expected_manager_binding
        and handoff["recipe_id"] == plan.get("recipe_id"),
        "paired plan differs from the manager handoff",
    )
    catalog_path = Path(plan["catalog"]["path"])
    _require(
        sha256_file(catalog_path) == plan["catalog"]["file_sha256"],
        "catalog file drifted",
    )
    catalog = _read_catalog(catalog_path)
    _require(
        catalog["catalog_sha256"] == plan["catalog"]["catalog_sha256"]
        and _validate_recipe(catalog, plan["recipe_id"]) == plan["recipe"],
        "catalog/recipe value contract drifted",
    )
    round1 = _validate_round1_data_contract(Path(plan["round1_data"]["root"]), catalog)
    _require(round1 == plan["round1_data"], "Round-1 data contract drifted")
    repository_root = Path(plan["repository_root"])
    _require(
        _source_contract(repository_root) == plan["source_contract"],
        "source revision or dependency hash drifted",
    )
    worker = plan["worker"]
    _require(
        Path(worker["path"]) == (repository_root / WORKER_RELATIVE_PATH).resolve()
        and sha256_file(Path(worker["path"])) == worker["sha256"],
        "paired worker source drifted",
    )
    _require(plan.get("precision") == PRECISION, "paired plan precision drifted")
    expected_seeds = catalog["promotion_seeds"]
    _require(
        [row["seed"] for row in plan["seeds"]] == expected_seeds
        and sorted(int(seed) for seed in ledger["seeds"]) == expected_seeds,
        "paired seed contract drifted",
    )
    _require(
        [
            {
                "seed": row["seed"],
                "path": row["checkpoint_path"],
                "sha256": row["checkpoint_sha256"],
            }
            for row in handoff["parent_artifacts"]
        ]
        == [row["parent_checkpoint"] for row in plan["seeds"]],
        "paired parent lineage differs from the manager handoff",
    )
    for seed_row in plan["seeds"]:
        parent = seed_row["parent_checkpoint"]
        _require(
            sha256_file(Path(parent["path"])) == parent["sha256"],
            f"parent checkpoint drifted for seed {seed_row['seed']}",
        )
        planned_certificate = seed_row.get("host_stability_certificate")
        _require(
            isinstance(planned_certificate, dict),
            f"planned host certificate is absent for seed {seed_row['seed']}",
        )
        try:
            issued = datetime.fromisoformat(
                str(planned_certificate["issued_utc"]).replace("Z", "+00:00")
            )
        except (KeyError, ValueError) as error:
            raise AutoResearchError(
                "planned host certificate timestamp is invalid"
            ) from error
        observed_certificate = _certificate_identity(
            certificate_path=Path(planned_certificate["path"]),
            parent=parent,
            host_identity_sha256=plan["host_identity_sha256"],
            source_commit=plan["source_contract"]["commit"],
            now=issued,
        )
        _require(
            observed_certificate == planned_certificate,
            f"planned host certificate drifted for seed {seed_row['seed']}",
        )
        seed_ledger = ledger["seeds"][str(seed_row["seed"])]
        report_path = Path(seed_row["output_dir"]) / "report.json"
        if seed_ledger.get("worker") == "complete":
            _require(report_path.is_file(), "completed seed report is missing")
            _verify_report_artifacts(
                report_path,
                expected_report_sha256=seed_ledger.get("worker_report_sha256"),
            )
            _verify_sealed_host_certificate(plan, seed_row, seed_ledger)
            _verify_attempt_audit(plan, seed_row, seed_ledger)
    return plan, ledger


def _execution_binding(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "paired_plan_sha256": plan["plan_sha256"],
        "round1_plan_sha256": plan["round1_data"]["plan_sha256"],
        "search_indices_sha256": plan["round1_data"]["search_indices_sha256"],
        "fixed_indices_file_sha256": plan["round1_data"]["index_manifest"][
            "file_sha256"
        ],
        "fixed_indices_value_sha256": plan["round1_data"]["index_manifest"][
            "value_set_sha256"
        ],
        "fixed_batch_plans": [
            {
                "seed": row["seed"],
                "file_sha256": row["batch_plan"]["file_sha256"],
                "value_set_sha256": row["batch_plan"]["value_set_sha256"],
            }
            for row in plan["seeds"]
        ],
    }


def _certificate_identity(
    *,
    certificate_path: Path,
    parent: Mapping[str, Any],
    host_identity_sha256: str,
    source_commit: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    certificate_path = certificate_path.expanduser().resolve()
    _require(certificate_path.is_file(), "host stability certificate is missing")
    value = _read_object(certificate_path)
    try:
        normalized = validate_certificate(
            value,
            expected_checkpoint_path=Path(parent["path"]),
            expected_checkpoint_sha256=parent["sha256"],
            expected_host_identity_sha256=host_identity_sha256,
            expected_source_commit=source_commit,
            now=now,
        )
    except HostStabilityError as error:
        raise AutoResearchError(
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


def _validate_host_certificate(
    plan: Mapping[str, Any],
    seed_row: Mapping[str, Any],
    certificate_path: Path,
) -> dict[str, Any]:
    base = _certificate_identity(
        certificate_path=certificate_path,
        parent=seed_row["parent_checkpoint"],
        host_identity_sha256=plan["host_identity_sha256"],
        source_commit=plan["source_contract"]["commit"],
    )
    execution_binding = _execution_binding(plan)
    return {
        **base,
        "execution_binding": execution_binding,
        "execution_binding_sha256": digest_value(execution_binding),
    }


def _publish_attempt_authorization(
    root: Path,
    *,
    plan: Mapping[str, Any],
    seed_row: Mapping[str, Any],
    certificate_row: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema": PAIRED_AUTHORIZATION_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "seed": seed_row["seed"],
        "parent_checkpoint": seed_row["parent_checkpoint"],
        "host_stability_certificate": dict(certificate_row),
        "decision": "scientific-worker-authorized",
    }
    value = {**payload, "authorization_sha256": digest_value(payload)}
    path = (
        root
        / "authorizations"
        / str(seed_row["seed"])
        / f"{certificate_row['certificate_sha256']}.json"
    )
    _create_only_json(path, value, name="paired attempt authorization")
    return {
        "path": str(path.resolve()),
        "file_sha256": sha256_file(path),
        "authorization_sha256": value["authorization_sha256"],
    }


def _publish_attempt_seal(
    root: Path,
    *,
    plan: Mapping[str, Any],
    seed_row: Mapping[str, Any],
    authorization: Mapping[str, Any],
    report_path: Path,
) -> dict[str, Any]:
    payload = {
        "schema": PAIRED_ATTEMPT_SEAL_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "seed": seed_row["seed"],
        "authorization_sha256": authorization["authorization_sha256"],
        "worker_report_sha256": sha256_file(report_path),
        "status": "complete",
    }
    value = {**payload, "attempt_seal_sha256": digest_value(payload)}
    path = root / "attempt_seals" / f"seed-{seed_row['seed']}.json"
    _create_only_json(path, value, name="paired attempt seal")
    return {
        "path": str(path.resolve()),
        "file_sha256": sha256_file(path),
        "attempt_seal_sha256": value["attempt_seal_sha256"],
    }


def _verify_attempt_audit(
    plan: Mapping[str, Any], seed_row: Mapping[str, Any], seed_ledger: Mapping[str, Any]
) -> None:
    authorization_ref = seed_ledger.get("authorization")
    seal_ref = seed_ledger.get("attempt_seal")
    _require(
        isinstance(authorization_ref, dict) and isinstance(seal_ref, dict),
        "completed paired seed lacks authorization/attempt seal",
    )
    authorization_path = Path(authorization_ref["path"])
    authorization = _read_object(authorization_path)
    _require(
        sha256_file(authorization_path) == authorization_ref["file_sha256"]
        and authorization.get("schema") == PAIRED_AUTHORIZATION_SCHEMA
        and authorization.get("authorization_sha256")
        == digest_value(
            {
                key: value
                for key, value in authorization.items()
                if key != "authorization_sha256"
            }
        )
        == authorization_ref["authorization_sha256"]
        and authorization.get("plan_sha256") == plan["plan_sha256"]
        and authorization.get("seed") == seed_row["seed"]
        and authorization.get("parent_checkpoint") == seed_row["parent_checkpoint"]
        and authorization.get("host_stability_certificate")
        == seed_ledger.get("host_stability_certificate"),
        "paired authorization artifact drifted",
    )
    seal_path = Path(seal_ref["path"])
    seal = _read_object(seal_path)
    report_path = Path(seed_row["output_dir"]) / "report.json"
    _require(
        sha256_file(seal_path) == seal_ref["file_sha256"]
        and seal.get("schema") == PAIRED_ATTEMPT_SEAL_SCHEMA
        and seal.get("attempt_seal_sha256")
        == digest_value(
            {key: value for key, value in seal.items() if key != "attempt_seal_sha256"}
        )
        == seal_ref["attempt_seal_sha256"]
        and seal.get("plan_sha256") == plan["plan_sha256"]
        and seal.get("seed") == seed_row["seed"]
        and seal.get("authorization_sha256") == authorization["authorization_sha256"]
        and seal.get("worker_report_sha256") == sha256_file(report_path)
        and seal.get("status") == "complete",
        "paired attempt seal drifted",
    )


def _verify_sealed_host_certificate(
    plan: Mapping[str, Any],
    seed_row: Mapping[str, Any],
    seed_ledger: Mapping[str, Any],
) -> None:
    sealed = seed_ledger.get("host_stability_certificate")
    _require(isinstance(sealed, dict), "completed seed lacks a host certificate")
    path = Path(sealed["path"])
    _require(
        path.is_file() and sha256_file(path) == sealed.get("file_sha256"),
        "sealed host stability certificate drifted",
    )
    try:
        # Historical verification uses the issuance instant.  Freshness is
        # rechecked against wall-clock time immediately before every new worker.
        issued = datetime.fromisoformat(
            str(sealed["issued_utc"]).replace("Z", "+00:00")
        )
        base = _certificate_identity(
            certificate_path=path,
            parent=seed_row["parent_checkpoint"],
            host_identity_sha256=plan["host_identity_sha256"],
            source_commit=plan["source_contract"]["commit"],
            now=issued,
        )
    except (HostStabilityError, KeyError, ValueError) as error:
        raise AutoResearchError(
            f"sealed host stability certificate is invalid: {error}"
        ) from error
    _require(
        sealed
        == {
            **base,
            "execution_binding": _execution_binding(plan),
            "execution_binding_sha256": digest_value(_execution_binding(plan)),
        },
        "sealed host stability certificate metadata differs",
    )


def _run_paired_bridge_with_gpu_lock_held(
    root: Path,
    *,
    host_stability_certificates: Mapping[int, Path],
    selected_seed: int | None = None,
) -> dict[str, Any]:
    """Run registered workers with ``shell=False`` and skip completed seeds."""

    root = _safe_path(root, name="paired root")
    plan, ledger = _verify_plan(root)
    _require(
        _host_identity_sha256() == plan["host_identity_sha256"],
        "current GPU host identity differs from the prepared plan",
    )
    expected_seeds = [row["seed"] for row in plan["seeds"]]
    _require(
        sorted(host_stability_certificates) == expected_seeds,
        "runtime host certificates must cover exactly the three registered seeds",
    )
    selected = [row for row in plan["seeds"] if selected_seed in {None, row["seed"]}]
    _require(bool(selected), "selected seed is not registered")
    selected_seeds = {row["seed"] for row in selected}
    for seed_row in plan["seeds"]:
        seed = seed_row["seed"]
        report_exists = (Path(seed_row["output_dir"]) / "report.json").is_file()
        if seed in selected_seeds and not report_exists:
            _validate_host_certificate(
                plan, seed_row, host_stability_certificates[seed]
            )
            _require(
                _checkpoint_precision(Path(seed_row["parent_checkpoint"]["path"]))
                == PRECISION,
                f"parent precision authorization failed for seed {seed}",
            )
    ledger["state"] = "running"
    _write_ledger(root / "ledger.json", ledger)
    for seed_row in selected:
        key = str(seed_row["seed"])
        stage = ledger["seeds"][key]
        output_dir = Path(seed_row["output_dir"])
        report_path = output_dir / "report.json"
        if report_path.is_file():
            _verify_report_artifacts(
                report_path,
                expected_report_sha256=stage.get("worker_report_sha256"),
            )
            stage["worker"] = "complete"
            stage["worker_report_sha256"] = sha256_file(report_path)
            _verify_sealed_host_certificate(plan, seed_row, stage)
            _verify_attempt_audit(plan, seed_row, stage)
            _write_ledger(root / "ledger.json", ledger)
            continue
        if output_dir.exists():
            stage["worker"] = "incomplete-output"
            _write_ledger(root / "ledger.json", ledger)
            raise AutoResearchError(
                f"seed {seed_row['seed']} has an incomplete create-only output"
            )
        stage["worker"] = "running"
        certificate_row = _validate_host_certificate(
            plan, seed_row, host_stability_certificates[seed_row["seed"]]
        )
        authorization = _publish_attempt_authorization(
            root,
            plan=plan,
            seed_row=seed_row,
            certificate_row=certificate_row,
        )
        stage["host_stability_certificate"] = certificate_row
        stage["authorization"] = authorization
        _write_ledger(root / "ledger.json", ledger)
        try:
            subprocess.run(
                _worker_argv(plan, seed_row),
                cwd=Path(plan["repository_root"]),
                check=True,
                shell=False,
            )
        except subprocess.CalledProcessError:
            stage["worker"] = "failed"
            _write_ledger(root / "ledger.json", ledger)
            raise
        _require(report_path.is_file(), "paired worker completed without report.json")
        _verify_report_artifacts(report_path)
        stage["worker"] = "complete"
        stage["worker_report_sha256"] = sha256_file(report_path)
        stage["attempt_seal"] = _publish_attempt_seal(
            root,
            plan=plan,
            seed_row=seed_row,
            authorization=authorization,
            report_path=report_path,
        )
        _write_ledger(root / "ledger.json", ledger)
    if all(row.get("worker") == "complete" for row in ledger["seeds"].values()):
        ledger["state"] = "workers-complete"
        _write_ledger(root / "ledger.json", ledger)
    return ledger


def run_paired_bridge(
    root: Path,
    *,
    host_stability_certificates: Mapping[int, Path],
    selected_seed: int | None = None,
) -> dict[str, Any]:
    """Wait for the shared GPU lock, then run and verify registered workers.

    Certificate freshness is checked only after the lock is acquired, so time
    spent behind another workflow cannot turn an old authorization into a
    scientific launch.
    """

    with gpu_lock(GPU_LOCK_ROOT, GPU_ID, timeout_seconds=-1):
        return _run_paired_bridge_with_gpu_lock_held(
            root,
            host_stability_certificates=host_stability_certificates,
            selected_seed=selected_seed,
        )


def _require_configuration(
    observed: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    for key, value in expected.items():
        if observed.get(key) != value:
            raise AutoResearchError(
                f"paired worker configuration mismatch for {key}: "
                f"{observed.get(key)!r} != {value!r}"
            )


def _metric_row(value: Mapping[str, Any]) -> dict[str, Any]:
    statistics_row = value.get("statistics")
    tail = value.get("tail")
    if not isinstance(statistics_row, dict) or not isinstance(tail, dict):
        raise AutoResearchError("development metric row is incomplete")
    try:
        result = {
            "sigma": float(statistics_row["sigma_official_formula"]),
            "chi": float(statistics_row["weighted_rms_abs_residual"]),
            "q999": float(tail["q999"]),
            "cvar_1pct": float(tail["cvar99"]),
            "minimum_metric_eigenvalue": float(
                statistics_row["min_eigenvalue_weighted_quantiles"]["q0.0000"]
            ),
            "nonpositive_metric_count": int(
                statistics_row["nonpositive_min_eigenvalue"]["count"]
            ),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise AutoResearchError("development metric row is malformed") from error
    _require(
        all(math.isfinite(float(row)) for row in result.values())
        and result["sigma"] > 0
        and result["chi"] > 0
        and result["q999"] >= 0
        and result["cvar_1pct"] >= 0,
        "development metrics are non-finite or outside their domain",
    )
    return result


def _expected_input_hashes(plan: Mapping[str, Any]) -> dict[str, Any]:
    roles = plan["round1_data"]["input_roles"]
    return {
        "train_points": roles["search-native-points"]["sha256"],
        "train_pullbacks": roles["search-native-pullbacks"]["sha256"],
        "selection_points": roles["search-native-points"]["sha256"],
        "selection_pullbacks": roles["search-native-pullbacks"]["sha256"],
        "development_evaluation_points": roles["development-evaluation-points"][
            "sha256"
        ],
        "development_evaluation_pullbacks": roles["development-evaluation-pullbacks"][
            "sha256"
        ],
        "confirmation_points": None,
        "confirmation_pullbacks": None,
    }


def _collect_evidence(
    root: Path,
    plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        all(row.get("worker") == "complete" for row in ledger["seeds"].values()),
        "all three paired workers must complete before normalization",
    )
    roles = plan["round1_data"]["input_roles"]
    fixed_indices_path = Path(plan["round1_data"]["index_manifest"]["path"])
    fixed_indices = _load_npz(fixed_indices_path)
    intervention = plan["recipe"]["intervention"]
    budget = plan["recipe"]["budget"]
    rows = []
    for seed_row in plan["seeds"]:
        seed = seed_row["seed"]
        report_path = Path(seed_row["output_dir"]) / "report.json"
        report = _verify_report_artifacts(
            report_path,
            expected_report_sha256=ledger["seeds"][str(seed)].get(
                "worker_report_sha256"
            ),
        )
        configuration = report.get("configuration")
        _require(isinstance(configuration, dict), "worker configuration is absent")
        _require_configuration(
            configuration,
            {
                "control_checkpoint": seed_row["parent_checkpoint"]["path"],
                "candidate_checkpoint": seed_row["parent_checkpoint"]["path"],
                "train_points": roles["search-native-points"]["path"],
                "train_pullbacks": roles["search-native-pullbacks"]["path"],
                "selection_points": roles["search-native-points"]["path"],
                "selection_pullbacks": roles["search-native-pullbacks"]["path"],
                "development_evaluation_points": roles["development-evaluation-points"][
                    "path"
                ],
                "development_evaluation_pullbacks": roles[
                    "development-evaluation-pullbacks"
                ]["path"],
                "confirmation_points": None,
                "confirmation_pullbacks": None,
                "exclude_indices_file": [],
                "fixed_indices_file": str(fixed_indices_path),
                "fixed_batch_plan_file": seed_row["batch_plan"]["path"],
                "output_dir": seed_row["output_dir"],
                "steps": OPTIMIZER_UPDATES,
                "learning_rate": budget["learning_rate"],
                "scheduler": "cosine",
                "control_scheduler": intervention["control"]["scheduler"],
                "candidate_scheduler": intervention["candidate"]["scheduler"],
                "control_trainable_scope": intervention["control"]["trainable_scope"],
                "candidate_trainable_scope": intervention["candidate"][
                    "trainable_scope"
                ],
                "require_equal_total_parameters": True,
                "eval_every": EVAL_EVERY,
                "train_size": FIT_COUNT,
                "selection_size": SELECTION_COUNT,
                "development_evaluation_size": DEVELOPMENT_EVALUATION_COUNT,
                "stochastic_batch_size": budget["batch_size"],
                "gradient_clip_norm": budget["gradient_clip_norm"],
                "minimum_relative_sigma_gain": 0.0,
                "minimum_relative_chi_gain": 0.0,
                "maximum_selection_tail_relative_degradation": plan["recipe"][
                    "promotion_gate"
                ]["maximum_tail_relative_degradation"],
                "train_chunk_size": plan["runtime"]["train_chunk_size"],
                "feature_batch_size": plan["runtime"]["feature_batch_size"],
                "eval_batch_size": plan["runtime"]["eval_batch_size"],
                "seed": seed_row["optimizer_seed"],
                "data_seed": seed_row["data_seed"],
                "batch_plan_seed": seed_row["batch_plan_seed"],
                "threads": plan["runtime"]["threads"],
                "device": plan["runtime"]["device"],
                "development_only": True,
                "development_evaluation": True,
            },
        )
        _require(
            report.get("contract", {}).get("precision") == PRECISION
            and report.get("confirmation") is None
            and report.get("confirmation_passes") is None,
            "paired worker opened held-out data or used the wrong precision",
        )
        _require(
            report.get("training_policies")
            == {
                "control": {
                    "trainable_scope": intervention["control"]["trainable_scope"],
                    "scheduler": intervention["control"]["scheduler"],
                },
                "candidate": {
                    "trainable_scope": intervention["candidate"]["trainable_scope"],
                    "scheduler": intervention["candidate"]["scheduler"],
                },
            },
            "paired worker policy differs from the registered intervention",
        )
        _require(
            report.get("source_checkpoint_sha256")
            == {
                "control": seed_row["parent_checkpoint"]["sha256"],
                "candidate": seed_row["parent_checkpoint"]["sha256"],
            },
            "paired arms did not start from the same parent checkpoint",
        )
        parity = report.get("total_parameter_parity")
        _require(
            isinstance(parity, dict)
            and parity.get("required") is True
            and parity.get("equal") is True
            and parity.get("candidate_minus_control") == 0,
            "paired total-parameter parity audit failed",
        )
        data = report.get("data")
        _require(
            isinstance(data, dict)
            and data.get("input_sha256") == _expected_input_hashes(plan)
            and data.get("fixed_indices_file_sha256")
            == plan["round1_data"]["index_manifest"]["file_sha256"]
            and data.get("data_seed") == seed_row["data_seed"]
            and data.get("counts")
            == {
                "fit": FIT_COUNT,
                "selection": SELECTION_COUNT,
                "development_evaluation": DEVELOPMENT_EVALUATION_COUNT,
            },
            "paired worker data contract differs",
        )
        _require(
            data.get("indices_sha256") == sha256_file(Path(data["indices"])),
            "paired worker index artifact hash differs",
        )
        observed_indices = _load_npz(Path(data["indices"]))
        _require(
            set(observed_indices) == set(fixed_indices)
            and all(
                np.array_equal(observed_indices[name], fixed_indices[name])
                for name in fixed_indices
            ),
            "paired worker indices differ from the fixed Round-1 values",
        )
        batch = report.get("batch_plan")
        _require(isinstance(batch, dict), "paired worker batch-plan audit is absent")
        observed_batch = np.load(Path(batch["path"]), allow_pickle=False)
        expected_batch = np.load(seed_row["batch_plan"]["path"], allow_pickle=False)
        _require(
            np.array_equal(observed_batch, expected_batch)
            and batch.get("sha256") == sha256_file(Path(batch["path"]))
            and batch.get("array_sha256") == _batch_digest(observed_batch)
            and batch.get("same_for_both_arms") is True
            and batch.get("fixed_source_sha256")
            == seed_row["batch_plan"]["file_sha256"]
            and batch.get("seed") == seed_row["batch_plan_seed"]
            and batch.get("steps") == OPTIMIZER_UPDATES
            and batch.get("batch_size") == budget["batch_size"],
            "paired worker batch plan differs from the fixed Round-1 plan",
        )
        results = report["results"]
        control_result = results["control"]
        candidate_result = results["candidate"]
        _require(
            control_result.get("total_real_parameters")
            == candidate_result.get("total_real_parameters")
            == parity.get("control_total_real_parameters")
            and candidate_result.get("new_real_parameters_vs_control") == 0,
            "paired worker parameter accounting differs",
        )
        for role, result in (
            ("control", control_result),
            ("candidate", candidate_result),
        ):
            _require(
                result.get("training_policy")
                == {
                    "trainable_scope": intervention[role]["trainable_scope"],
                    "scheduler": intervention[role]["scheduler"],
                }
                and result.get("fixed_batch_plan_sha256") == batch.get("array_sha256"),
                f"paired {role} policy/batch audit differs",
            )
            audit = result.get("parameter_audit")
            _require(
                isinstance(audit, dict)
                and audit.get("all_frozen_parameters_exactly_unchanged") is True,
                f"paired {role} frozen-parameter audit failed",
            )
        if plan["recipe_id"] == "r3-all-parameter-joint":
            _require(
                0
                < control_result.get("trainable_real_parameters", 0)
                < candidate_result.get("trainable_real_parameters", 0),
                "Round 3 trainable parameter scopes did not differ",
            )
        else:
            _require(
                control_result.get("trainable_real_parameters")
                == candidate_result.get("trainable_real_parameters")
                == control_result.get("total_real_parameters"),
                "Round 4 did not train all parameters in both arms",
            )
        exact_epoch_zero = bool(
            control_result.get("initial_statistics")
            == candidate_result.get("initial_statistics")
            and control_result.get("initial_tail")
            == candidate_result.get("initial_tail")
        )
        _require(exact_epoch_zero, "paired arms differ at epoch zero")
        development = report.get("development_evaluation")
        _require(
            isinstance(development, dict)
            and development.get("selection_or_checkpoint_role") == "none",
            "development evaluation was not isolated from selection",
        )
        paired = development.get("paired_improvement")
        _require(isinstance(paired, dict), "paired development CI is absent")
        try:
            paired_row = {
                "sigma_ci95_low": _finite(
                    paired["sigma"]["ci95_low"], name="sigma_ci95_low"
                ),
                "e2_ci95_low": _finite(paired["e2"]["ci95_low"], name="e2_ci95_low"),
            }
        except KeyError as error:
            raise AutoResearchError("paired development CI is malformed") from error
        rows.append(
            {
                "seed": seed,
                "parent_checkpoint_sha256": seed_row["parent_checkpoint"]["sha256"],
                "exact_epoch_zero_equivalence": exact_epoch_zero,
                "control_policy": report["training_policies"]["control"],
                "candidate_policy": report["training_policies"]["candidate"],
                "control": _metric_row(development["control"]),
                "candidate": _metric_row(development["candidate"]),
                "paired": paired_row,
                "control_checkpoint": {
                    "path": control_result["checkpoint"],
                    "sha256": control_result["checkpoint_sha256"],
                },
                "candidate_checkpoint": {
                    "path": candidate_result["checkpoint"],
                    "sha256": candidate_result["checkpoint_sha256"],
                },
                "fixed_indices_value_sha256": plan["round1_data"]["index_manifest"][
                    "value_set_sha256"
                ],
                "fixed_batch_plan_value_sha256": seed_row["batch_plan"][
                    "value_set_sha256"
                ],
                "source_report": {
                    "path": str(report_path),
                    "sha256": sha256_file(report_path),
                },
                "host_stability_certificate": ledger["seeds"][str(seed)][
                    "host_stability_certificate"
                ],
            }
        )
    return {
        "schema": PAIRED_EVIDENCE_SCHEMA,
        "recipe_id": plan["recipe_id"],
        "round": plan["recipe"]["round"],
        "plan_sha256": plan["plan_sha256"],
        "catalog_sha256": plan["catalog"]["catalog_sha256"],
        "recipe_sha256": plan["recipe"]["recipe_sha256"],
        "precision": PRECISION,
        "development_only": True,
        "historical_confirmation": "absent",
        "final_blind": "absent",
        "search_indices_sha256": plan["round1_data"]["search_indices_sha256"],
        "seeds": rows,
    }


def normalize_paired_bridge(root: Path) -> dict[str, Any]:
    """Validate worker artifacts and create strict paired development evidence."""

    root = _safe_path(root, name="paired root")
    plan, ledger = _verify_plan(root)
    evidence = _collect_evidence(root, plan, ledger)
    path = root / "paired_evidence.json"
    _create_only_json(path, evidence, name="paired evidence")
    ledger["evidence"] = {
        "path": str(path),
        "sha256": sha256_file(path),
        "value_sha256": digest_value(evidence),
    }
    if ledger.get("state") != "adjudicated":
        ledger["state"] = "normalized"
    _write_ledger(root / "ledger.json", ledger)
    return evidence


def _relative_gain(control: float, candidate: float) -> float:
    return (control - candidate) / control


def _relative_degradation(control: float, candidate: float) -> float:
    if control == 0:
        return 0.0 if candidate == 0 else math.inf
    return (candidate - control) / control


def _adjudication_value(
    plan: Mapping[str, Any], evidence: Mapping[str, Any]
) -> dict[str, Any]:
    thresholds = plan["recipe"]["promotion_gate"]
    sigma_gains = []
    chi_gains = []
    improved = 0
    paired_positive = 0
    positivity = True
    tail_safe = True
    equivalence = True
    seed_rows = []
    for row in evidence["seeds"]:
        control = row["control"]
        candidate = row["candidate"]
        sigma_gain = _relative_gain(control["sigma"], candidate["sigma"])
        chi_gain = _relative_gain(control["chi"], candidate["chi"])
        q999_degradation = _relative_degradation(control["q999"], candidate["q999"])
        cvar_degradation = _relative_degradation(
            control["cvar_1pct"], candidate["cvar_1pct"]
        )
        row_positive = bool(
            control["minimum_metric_eigenvalue"] > 0
            and candidate["minimum_metric_eigenvalue"] > 0
            and control["nonpositive_metric_count"] == 0
            and candidate["nonpositive_metric_count"] == 0
        )
        row_tail = bool(
            q999_degradation <= thresholds["maximum_tail_relative_degradation"]
            and cvar_degradation <= thresholds["maximum_tail_relative_degradation"]
        )
        paired_ok = bool(
            row["paired"]["sigma_ci95_low"] > 0 and row["paired"]["e2_ci95_low"] > 0
        )
        if sigma_gain > 0 and chi_gain > 0:
            improved += 1
        if paired_ok:
            paired_positive += 1
        sigma_gains.append(sigma_gain)
        chi_gains.append(chi_gain)
        positivity = positivity and row_positive
        tail_safe = tail_safe and row_tail
        equivalence = equivalence and row["exact_epoch_zero_equivalence"]
        seed_rows.append(
            {
                "seed": row["seed"],
                "relative_sigma_gain": sigma_gain,
                "relative_chi_gain": chi_gain,
                "relative_q999_degradation": q999_degradation,
                "relative_cvar_1pct_degradation": cvar_degradation,
                "positivity_passes": row_positive,
                "tail_passes": row_tail,
                "paired_ci_passes": paired_ok,
                "parent_checkpoint_sha256": row["parent_checkpoint_sha256"],
                "control_checkpoint_sha256": row["control_checkpoint"]["sha256"],
                "candidate_checkpoint_sha256": row["candidate_checkpoint"]["sha256"],
                "source_report_sha256": row["source_report"]["sha256"],
            }
        )
    median_sigma = statistics.median(sigma_gains)
    median_chi = statistics.median(chi_gains)
    fixed_indices = {row["fixed_indices_value_sha256"] for row in evidence["seeds"]}
    gates = {
        "three_registered_seeds": len(seed_rows) == 3,
        "complex64_development_only": bool(
            evidence["precision"] == PRECISION
            and evidence["development_only"] is True
            and evidence["historical_confirmation"] == "absent"
            and evidence["final_blind"] == "absent"
        ),
        "fixed_round1_indices": len(fixed_indices) == 1,
        "fixed_round1_batch_plans": all(
            row["fixed_batch_plan_value_sha256"]
            == plan["seeds"][index]["batch_plan"]["value_set_sha256"]
            for index, row in enumerate(evidence["seeds"])
        ),
        "matched_600_update_budget": (
            plan["recipe"]["budget"]["matched_relax_optimizer_updates_per_arm"]
            == OPTIMIZER_UPDATES
        ),
        "host_stability_authorized": all(
            row["host_stability_certificate"]["scientific_authorized"] is True
            and row["host_stability_certificate"]["scope"] == "scientific"
            and row["host_stability_certificate"]["execution_binding_sha256"]
            == digest_value(_execution_binding(plan))
            for row in evidence["seeds"]
        ),
        "same_parent_exact_epoch_zero": equivalence,
        "equal_total_parameters": True,
        "positivity": positivity,
        "tail": tail_safe,
        "minimum_improved_seeds": improved >= thresholds["minimum_improved_seeds"],
        "minimum_paired_ci_positive_seeds": paired_positive
        >= thresholds["minimum_paired_ci_positive_seeds"],
        "median_sigma_gain": median_sigma
        >= thresholds["minimum_median_relative_sigma_gain"],
        "median_chi_gain": median_chi >= thresholds["minimum_median_relative_chi_gain"],
        "worst_seed_sigma": min(sigma_gains)
        >= -thresholds["maximum_seed_relative_regression"],
        "worst_seed_chi": min(chi_gains)
        >= -thresholds["maximum_seed_relative_regression"],
    }
    promotion_passes = all(gates.values())
    family_role = "candidate" if promotion_passes else "parent"
    family_payload = {
        "models": [
            {
                "seed": row["seed"],
                "checkpoint_sha256": (
                    row["candidate_checkpoint"]["sha256"]
                    if promotion_passes
                    else row["parent_checkpoint_sha256"]
                ),
            }
            for row in evidence["seeds"]
        ]
    }
    return {
        "schema": PAIRED_ADJUDICATION_SCHEMA,
        "recipe_id": plan["recipe_id"],
        "round": plan["recipe"]["round"],
        "plan_sha256": plan["plan_sha256"],
        "evidence_value_sha256": digest_value(evidence),
        "thresholds": thresholds,
        "promotion_passes": promotion_passes,
        "recommended_role": family_role,
        "recommended_family": {
            **family_payload,
            "family_sha256": digest_value(family_payload),
        },
        "gates": gates,
        "median_relative_sigma_gain": median_sigma,
        "median_relative_chi_gain": median_chi,
        "minimum_relative_sigma_gain": min(sigma_gains),
        "minimum_relative_chi_gain": min(chi_gains),
        "improved_seed_count": improved,
        "paired_ci_positive_seed_count": paired_positive,
        "seed_rows": seed_rows,
        "scientific_scope": "development-only; not confirmation or final-blind evidence",
        "created_at": utc_now(),
    }


def adjudicate_paired_bridge(root: Path) -> dict[str, Any]:
    """Apply the catalog gate to normalized three-seed paired evidence."""

    root = _safe_path(root, name="paired root")
    plan, ledger = _verify_plan(root)
    _require(isinstance(ledger.get("evidence"), dict), "paired evidence is absent")
    evidence_path = Path(ledger["evidence"]["path"])
    evidence = _read_object(evidence_path)
    expected = _collect_evidence(root, plan, ledger)
    _require(
        evidence == expected
        and sha256_file(evidence_path) == ledger["evidence"]["sha256"]
        and digest_value(evidence) == ledger["evidence"]["value_sha256"],
        "normalized paired evidence drifted",
    )
    adjudication = _adjudication_value(plan, evidence)
    path = root / "adjudication.json"
    if path.exists():
        existing = _read_object(path)
        # The timestamp is provenance, not part of a rerun-dependent decision.
        comparable = {
            key: value for key, value in existing.items() if key != "created_at"
        }
        expected_comparable = {
            key: value for key, value in adjudication.items() if key != "created_at"
        }
        _require(comparable == expected_comparable, "adjudication already differs")
        adjudication = existing
    else:
        atomic_write_json(path, adjudication)
    ledger["state"] = "adjudicated"
    ledger["adjudication"] = {
        "path": str(path),
        "sha256": sha256_file(path),
        "value_sha256": digest_value(adjudication),
        "promotion_passes": adjudication["promotion_passes"],
    }
    _write_ledger(root / "ledger.json", ledger)
    return adjudication


def paired_bridge_status(root: Path) -> dict[str, Any]:
    """Return a verified compact status; artifact drift raises immediately."""

    root = _safe_path(root, name="paired root")
    plan, ledger = _verify_plan(root)
    evidence: dict[str, Any] | None = None
    if isinstance(ledger.get("evidence"), dict):
        evidence_path = Path(ledger["evidence"]["path"])
        evidence = _read_object(evidence_path)
        _require(
            evidence == _collect_evidence(root, plan, ledger)
            and sha256_file(evidence_path) == ledger["evidence"]["sha256"]
            and digest_value(evidence) == ledger["evidence"]["value_sha256"],
            "paired evidence drifted",
        )
    if isinstance(ledger.get("adjudication"), dict):
        _require(evidence is not None, "adjudication exists without paired evidence")
        path = Path(ledger["adjudication"]["path"])
        value = _read_object(path)
        _require(
            sha256_file(path) == ledger["adjudication"]["sha256"]
            and digest_value(value) == ledger["adjudication"]["value_sha256"],
            "paired adjudication drifted",
        )
        expected = _adjudication_value(plan, evidence)
        _require(
            {key: row for key, row in value.items() if key != "created_at"}
            == {key: row for key, row in expected.items() if key != "created_at"},
            "paired adjudication no longer matches the evidence/catalog gate",
        )
    return {
        "schema": PAIRED_LEDGER_SCHEMA,
        "recipe_id": plan["recipe_id"],
        "round": plan["recipe"]["round"],
        "plan_sha256": plan["plan_sha256"],
        "state": ledger["state"],
        "seeds": ledger["seeds"],
        "evidence": ledger["evidence"],
        "adjudication": ledger["adjudication"],
    }


__all__ = [
    "PAIRED_ADJUDICATION_SCHEMA",
    "PAIRED_ATTEMPT_SEAL_SCHEMA",
    "PAIRED_AUTHORIZATION_SCHEMA",
    "PAIRED_EVIDENCE_SCHEMA",
    "PAIRED_LEDGER_SCHEMA",
    "PAIRED_PLAN_SCHEMA",
    "REGISTERED_RECIPES",
    "adjudicate_paired_bridge",
    "normalize_paired_bridge",
    "paired_bridge_status",
    "prepare_paired_bridge",
    "run_paired_bridge",
]
