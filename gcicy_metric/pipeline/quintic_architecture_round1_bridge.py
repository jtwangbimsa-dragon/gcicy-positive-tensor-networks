"""Executable, fail-closed bridge for quintic architecture Round 1.

The bridge is deliberately narrower than the architecture action language.
Version 1 executes exactly one preregistered experiment: grow internal edge 6
from 25 to 39 (9,800 new real parameters), locally activate the transported
channels, and compare it with a no-growth control under a matched full-model
relaxation.  No action field is ever interpreted as a command.

Search checkpoint selection uses only a 30,000-row fit split and a 5,000-row
selection split from the frozen native-pool indices.  A separate 5,000-row
development-evaluation split is opened only after both arms have finished and
is used only for paired confidence intervals.  Historical confirmation and
final blind data have no bridge entry point.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from .architecture_auto_research import (
    SEARCH_EVIDENCE_SCHEMA,
    AutoResearchError,
    CampaignStore,
    atomic_write_json,
    digest_value,
    sha256_file,
    validate_action,
    validate_protocol,
)


BRIDGE_PLAN_SCHEMA = "gcicy-quintic-architecture-round1-bridge-plan-v1"
BRIDGE_LEDGER_SCHEMA = "gcicy-quintic-architecture-round1-bridge-ledger-v1"
INDEX_VALUE_SCHEMA = "gcicy-quintic-round1-index-values-v1"
BATCH_VALUE_SCHEMA = "gcicy-quintic-round1-batch-values-v1"
BRIDGE_SENTINEL = ".gcicy-quintic-round1-bridge-root"

ROUND1_BASELINE_SHA256 = (
    "1836b868f9b852f0ab29172a83c07caf4c47d2e75f71b9ff588add2d4f96359b"
)
ROUND1_EDGE = 6
ROUND1_SOURCE_DIMENSION = 25
ROUND1_TARGET_DIMENSION = 39
ROUND1_STRUCTURAL_MAXIMUM = 100
ROUND1_NEW_REAL_PARAMETERS = 9_800
ROUND1_TOTAL_ACTION_REAL_PARAMETER_CAP = 10_000
ROUND1_CONTROL_REAL_PARAMETERS = 121_750
ROUND1_CANDIDATE_REAL_PARAMETERS = 131_550
ROUND1_TRAIN_INDEX_COUNT = 35_000
ROUND1_FIT_COUNT = 30_000
ROUND1_SELECTION_COUNT = 5_000
ROUND1_DEVELOPMENT_EVALUATION_COUNT = 5_000

INPUT_ROLES = {
    "baseline-checkpoint",
    "baseline-source-report",
    "search-native-points",
    "search-native-pullbacks",
    "development-evaluation-points",
    "development-evaluation-pullbacks",
}
WORKER_PATHS = {
    "local_activate": "scripts/train_generic_quintic_adaptive_direct_blocks.py",
    "matched_relax": "scripts/compare_generic_quintic_tree_joint_relaxation.py",
}
DEPENDENCY_PATHS = (
    "gcicy_metric/pipeline/positive_multiplication_tree.py",
    "scripts/evaluate_generic_quintic_h4_architecture_arms.py",
    "scripts/refine_generic_quintic_compiled_tree_native_gn.py",
    "scripts/train_generic_quintic_adaptive_tree_rank.py",
    "scripts/train_generic_quintic_compiled_tree.py",
    "scripts/train_quintic_full_h_same_points.py",
    "scripts/train_quintic_native_power_lift_tree.py",
)


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AutoResearchError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise AutoResearchError(f"JSON root must be an object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AutoResearchError(message)


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AutoResearchError(f"{name} must be a positive integer")
    return value


def _finite_positive(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise AutoResearchError(f"{name} must be numeric") from error
    if not math.isfinite(result) or result <= 0:
        raise AutoResearchError(f"{name} must be finite and positive")
    return result


def _finite_nonnegative(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise AutoResearchError(f"{name} must be numeric") from error
    if not math.isfinite(result) or result < 0:
        raise AutoResearchError(f"{name} must be finite and nonnegative")
    return result


def _json_safe_path(value: Any, *, name: str) -> Path:
    path = Path(str(value)).expanduser().resolve()
    lowered = str(path).lower()
    if "blind" in lowered or "shadow" in lowered or "confirmation" in lowered:
        raise AutoResearchError(f"{name} references forbidden held-out data")
    return path


def canonical_array_value_sha256(value: np.ndarray) -> str:
    """Hash array values, shape, and dtype independently of container bytes."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(b"gcicy-canonical-array-v1\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _array_manifest(arrays: Mapping[str, np.ndarray], *, schema: str) -> dict[str, Any]:
    rows = {
        name: {
            "dtype": np.ascontiguousarray(value).dtype.str,
            "shape": list(np.ascontiguousarray(value).shape),
            "value_sha256": canonical_array_value_sha256(value),
        }
        for name, value in sorted(arrays.items())
    }
    payload = {"schema": schema, "arrays": rows}
    return {**payload, "value_set_sha256": digest_value(payload)}


def validate_round1_contract(
    protocol_value: Mapping[str, Any],
    indices: Mapping[str, Any],
    action_value: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the sole v1 action and its frozen data/budget envelope."""

    raw_protocol = {
        key: value
        for key, value in protocol_value.items()
        if key
        not in {
            "search_precision",
            "finalist_replay_precision",
            "shadow_policy",
            "blind_policy",
        }
    }
    protocol = validate_protocol(raw_protocol)
    action = validate_action(
        action_value,
        expected_parent_family_sha256=protocol["baseline_family"]["family_sha256"],
        expected_indices_sha256=str(indices.get("indices_sha256", "")),
    )
    mutation = action["mutation"]
    _require(action["round"] == 1, "bridge v1 accepts Round 1 only")
    _require(
        mutation
        == {
            "kind": "rank",
            "target": "internal-edge",
            "source_dimension": ROUND1_SOURCE_DIMENSION,
            "target_dimension": ROUND1_TARGET_DIMENSION,
            "structural_maximum": ROUND1_STRUCTURAL_MAXIMUM,
            "new_output_real_parameters": ROUND1_NEW_REAL_PARAMETERS,
            "edge": ROUND1_EDGE,
        },
        "bridge v1 accepts only internal edge 6 growth 25->39 with 9,800 new real parameters",
    )
    _require(
        mutation["new_output_real_parameters"]
        <= ROUND1_TOTAL_ACTION_REAL_PARAMETER_CAP,
        "Round 1 exceeds the v1 total-action parameter cap",
    )
    plan = protocol["search_index_plan"]
    partition_seed = _positive_int(
        plan.get("partition_seed"), name="search_index_plan.partition_seed"
    )
    _require(
        plan
        == {
            "seed": plan["seed"],
            "partition_seed": partition_seed,
            "train_population": 100_000,
            "train_count": ROUND1_TRAIN_INDEX_COUNT,
            "evaluation_population": 20_000,
            "evaluation_count": ROUND1_DEVELOPMENT_EVALUATION_COUNT,
            "shared_population": False,
        },
        "Round 1 requires 35,000 native indices and 5,000 separate development-evaluation indices",
    )
    _require(
        len(indices.get("train_indices", [])) == ROUND1_TRAIN_INDEX_COUNT
        and len(indices.get("evaluation_indices", []))
        == ROUND1_DEVELOPMENT_EVALUATION_COUNT,
        "frozen controller index counts do not match Round 1",
    )
    roles = {row["role"] for row in protocol["search_inputs"]}
    _require(roles == INPUT_ROLES, "Round 1 search-input roles are not exact")
    for stage in ("local_activate", "matched_relax"):
        budget = protocol["budgets"][stage]
        _require(
            budget["train_examples"] == ROUND1_FIT_COUNT
            and budget["evaluation_examples"] == ROUND1_SELECTION_COUNT,
            f"{stage} must use a 30,000/5,000 budget",
        )
    _require(
        protocol["budgets"]["local_activate"]["scheduler"] == "cosine",
        "local activation uses the registered cosine path",
    )
    _require(
        protocol["budgets"]["matched_relax"]["scheduler"] == "cosine",
        "matched relaxation uses the registered cosine path",
    )
    _require(
        protocol["budgets"]["local_activate"]["optimizer_updates"] == 1_200,
        "Round 1 local activation has a 1,200-update maximum",
    )
    baseline_rows = protocol["baseline_family"]["models"]
    _require(
        all(
            row["checkpoint_sha256"] == ROUND1_BASELINE_SHA256 for row in baseline_rows
        ),
        "Round 1 is pinned to the leaf-rank10 baseline checkpoint",
    )
    input_rows = {row["role"]: row for row in protocol["search_inputs"]}
    _require(
        input_rows["baseline-checkpoint"]["sha256"] == ROUND1_BASELINE_SHA256,
        "baseline search input is not the pinned checkpoint",
    )
    return protocol, action


def materialize_round1_indices(
    indices: Mapping[str, Any],
    output_path: Path,
    *,
    partition_seed: int,
) -> dict[str, Any]:
    """Shuffle frozen native indices before splitting; preserve dev-eval separately."""

    train = np.asarray(indices.get("train_indices"))
    evaluation = np.asarray(indices.get("evaluation_indices"))
    if train.ndim != 1 or evaluation.ndim != 1:
        raise AutoResearchError("frozen indices must be one-dimensional")
    if not np.issubdtype(train.dtype, np.integer) or not np.issubdtype(
        evaluation.dtype, np.integer
    ):
        raise AutoResearchError("frozen indices must be integer arrays")
    train = np.asarray(train, dtype=np.int64)
    evaluation = np.asarray(evaluation, dtype=np.int64)
    _require(len(train) == ROUND1_TRAIN_INDEX_COUNT, "train index count is not 35,000")
    _require(
        len(evaluation) == ROUND1_DEVELOPMENT_EVALUATION_COUNT,
        "development-evaluation index count is not 5,000",
    )
    _require(len(np.unique(train)) == len(train), "native indices contain duplicates")
    _require(
        len(np.unique(evaluation)) == len(evaluation),
        "development-evaluation indices contain duplicates",
    )
    partition_seed = _positive_int(partition_seed, name="partition_seed")
    permutation = np.random.default_rng(partition_seed).permutation(len(train))
    partitioned_train = train[permutation]
    arrays = {
        "fit": partitioned_train[:ROUND1_FIT_COUNT],
        "selection": partitioned_train[ROUND1_FIT_COUNT:],
        "development_evaluation": evaluation,
    }
    _require(
        np.intersect1d(arrays["fit"], arrays["selection"]).size == 0,
        "fit and selection indices overlap",
    )
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        existing = np.load(output_path, allow_pickle=False)
        if set(existing.files) != set(arrays) or any(
            not np.array_equal(np.asarray(existing[name]), value)
            for name, value in arrays.items()
        ):
            raise AutoResearchError("fixed index artifact already differs")
    else:
        np.savez_compressed(output_path, **arrays)
    manifest = _array_manifest(arrays, schema=INDEX_VALUE_SCHEMA)
    return {
        **manifest,
        "path": str(output_path),
        "file_sha256": sha256_file(output_path),
        "partition_seed": partition_seed,
        "mapping": {
            "fit": "deterministically shuffled native train_indices[0:30000]",
            "selection": (
                "deterministically shuffled native train_indices[30000:35000]"
            ),
            "development_evaluation": (
                "separate development-evaluation pool evaluation_indices[0:5000]"
            ),
        },
    }


def _materialize_batch_plan(
    path: Path,
    *,
    seed: int,
    steps: int,
    batch_size: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    active = min(ROUND1_FIT_COUNT, batch_size)
    expected = np.stack(
        [rng.choice(ROUND1_FIT_COUNT, size=active, replace=False) for _ in range(steps)]
    ).astype(np.int64, copy=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        observed = np.load(path, allow_pickle=False)
        if not np.array_equal(observed, expected):
            raise AutoResearchError("fixed batch-plan artifact already differs")
    else:
        np.save(path, expected, allow_pickle=False)
    manifest = _array_manifest({"batch_plan": expected}, schema=BATCH_VALUE_SCHEMA)
    return {
        **manifest,
        "path": str(path.resolve()),
        "file_sha256": sha256_file(path),
        "seed": seed,
        "steps": steps,
        "batch_size": active,
    }


def _validate_runtime(runtime: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "device",
        "threads",
        "train_chunk_size",
        "feature_batch_size",
        "eval_batch_size",
        "early_stopping_evaluations",
    }
    if set(runtime) != required:
        raise AutoResearchError("runtime fields are not exact")
    device = str(runtime["device"])
    if device not in {"cpu", "cuda"}:
        raise AutoResearchError("runtime.device must be cpu or cuda")
    return {
        "device": device,
        **{
            key: _positive_int(runtime[key], name=f"runtime.{key}")
            for key in required - {"device"}
        },
    }


def _runtime_environment(device: str) -> dict[str, Any]:
    """Capture software and read-only GPU metadata without creating a CUDA context."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - worker environment is torch-based
        raise AutoResearchError("PyTorch is unavailable in the worker environment") from error
    result: dict[str, Any] = {
        "numpy": str(np.__version__),
        "torch": str(torch.__version__),
        "torch_cuda_build": torch.version.cuda,
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
        )
        rows = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
        _require(
            completed.returncode == 0 and len(rows) == 1,
            "Round 1 requires exactly one inspectable CUDA GPU",
        )
        fields = [field.strip() for field in rows[0].split(",")]
        _require(len(fields) == 3, "nvidia-smi environment output is malformed")
        result["gpu"] = {
            "name": fields[0],
            "driver_version": fields[1],
            "memory_total_mib": _positive_int(
                int(fields[2]), name="gpu.memory_total_mib"
            ),
        }
    else:
        result["gpu"] = None
    return result


def _git_source_contract(repository_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as error:
        raise AutoResearchError("cannot bind the git source revision") from error
    _require(len(commit) == 40, "git commit is malformed")
    _require(not status, "Round 1 execution requires a clean tracked git worktree")
    dependencies = {}
    for relative in DEPENDENCY_PATHS:
        path = (repository_root / relative).resolve()
        _require(path.is_file(), f"registered dependency is missing: {relative}")
        dependencies[relative] = sha256_file(path)
    return {"commit": commit, "tracked_worktree": "clean", "dependencies": dependencies}


def _registered_action(
    store: CampaignStore,
    candidate_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    ledger = store.status()
    protocol = store.protocol()
    indices = store.indices()
    round_row = ledger.get("rounds", {}).get("1")
    if not isinstance(round_row, dict) or round_row.get("status") != "open":
        raise AutoResearchError("Round 1 is not open")
    candidate = round_row.get("candidates", {}).get(candidate_id)
    if not isinstance(candidate, dict) or candidate.get("status") != "registered":
        raise AutoResearchError("candidate is not registered in the open Round 1")
    action = _read_object(Path(candidate["action_path"]))
    protocol, action = validate_round1_contract(protocol, indices, action)
    _require(
        action["action_sha256"] == candidate["action_sha256"],
        "registered action hash differs from the ledger",
    )
    return protocol, indices, action


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
    ledger = _read_object(path)
    observed = ledger.get("state_sha256")
    payload = {key: row for key, row in ledger.items() if key != "state_sha256"}
    _require(
        ledger.get("schema") == BRIDGE_LEDGER_SCHEMA
        and observed == digest_value(payload),
        "bridge ledger integrity check failed",
    )
    return ledger


def prepare_round1_bridge(
    *,
    campaign_run_root: Path,
    candidate_id: str,
    output_root: Path,
    baseline_checkpoints: Mapping[int, Path],
    runtime: Mapping[str, Any],
    repository_root: Path,
) -> dict[str, Any]:
    """Create or exactly resume a fully bound three-seed execution plan."""

    output_root = _json_safe_path(output_root, name="output_root")
    repository_root = repository_root.expanduser().resolve()
    store = CampaignStore(campaign_run_root)
    protocol, controller_indices, action = _registered_action(store, candidate_id)
    runtime_row = _validate_runtime(runtime)
    expected_seeds = protocol["promotion_seeds"]
    _require(
        set(baseline_checkpoints) == set(expected_seeds),
        "baseline paths must cover exactly the three promotion seeds",
    )
    inputs = {row["role"]: row for row in protocol["search_inputs"]}
    for role, row in inputs.items():
        path = _json_safe_path(row["path"], name=f"search input {role}")
        _require(path.is_file(), f"search input is missing: {role}")
        _require(sha256_file(path) == row["sha256"], f"search input drift: {role}")
    worker_rows = {}
    for stage, relative in WORKER_PATHS.items():
        path = (repository_root / relative).resolve()
        _require(path.is_file(), f"registered worker is missing: {relative}")
        worker_rows[stage] = {"path": str(path), "sha256": sha256_file(path)}
    source_revision = _git_source_contract(repository_root)
    baseline_rows = []
    for seed in expected_seeds:
        path = baseline_checkpoints[seed].expanduser().resolve()
        _require(path.is_file(), f"baseline checkpoint is missing for seed {seed}")
        observed = sha256_file(path)
        _require(
            observed == ROUND1_BASELINE_SHA256, "baseline checkpoint hash mismatch"
        )
        _require(
            str(path) == inputs["baseline-checkpoint"]["path"],
            "baseline path differs from the locked protocol input",
        )
        baseline_rows.append({"seed": seed, "path": str(path), "sha256": observed})

    output_root.mkdir(parents=True, exist_ok=True)
    sentinel = output_root / BRIDGE_SENTINEL
    if sentinel.exists():
        _require(
            sentinel.read_text(encoding="utf-8").strip() == candidate_id,
            "bridge root belongs to another candidate",
        )
    elif any(output_root.iterdir()):
        raise AutoResearchError("refusing a non-empty unmanaged bridge root")
    else:
        sentinel.write_text(f"{candidate_id}\n", encoding="utf-8")
    index_manifest = materialize_round1_indices(
        controller_indices,
        output_root / "fixed_indices.npz",
        partition_seed=protocol["search_index_plan"]["partition_seed"],
    )
    seed_rows = []
    matched_budget = protocol["budgets"]["matched_relax"]
    for baseline in baseline_rows:
        seed = baseline["seed"]
        batch_seed = int.from_bytes(
            hashlib.sha256(
                f"{protocol['campaign_id']}:matched-relax:{seed}".encode("utf-8")
            ).digest()[:8],
            "big",
        )
        batch = _materialize_batch_plan(
            output_root / "seeds" / str(seed) / "matched_batch_plan.npy",
            seed=batch_seed,
            steps=matched_budget["optimizer_updates"],
            batch_size=matched_budget["batch_size"],
        )
        seed_rows.append(
            {
                "seed": seed,
                "optimizer_seed": seed,
                "data_seed": protocol["search_index_plan"]["seed"],
                "batch_plan_seed": batch_seed,
                "baseline_checkpoint": baseline,
                "batch_plan": batch,
                "local_output_dir": str(
                    (output_root / "seeds" / str(seed) / "local_activate").resolve()
                ),
                "matched_output_dir": str(
                    (output_root / "seeds" / str(seed) / "matched_relax").resolve()
                ),
            }
        )
    source_bundle = {
        "workers": worker_rows,
        "source_revision": source_revision,
        "search_inputs": {role: inputs[role]["sha256"] for role in sorted(inputs)},
        "baseline_checkpoints": baseline_rows,
        "controller_indices_sha256": controller_indices["indices_sha256"],
        "worker_index_values_sha256": index_manifest["value_set_sha256"],
    }
    plan_payload = {
        "schema": BRIDGE_PLAN_SCHEMA,
        "campaign_run_root": str(Path(campaign_run_root).expanduser().resolve()),
        "candidate_id": candidate_id,
        "protocol_sha256": digest_value(protocol),
        "action": action,
        "search_indices_sha256": controller_indices["indices_sha256"],
        "index_manifest": index_manifest,
        "input_roles": {role: inputs[role] for role in sorted(inputs)},
        "workers": worker_rows,
        "source_revision": source_revision,
        "budgets": protocol["budgets"],
        "thresholds": protocol["thresholds"],
        "runtime": runtime_row,
        "python": {"path": sys.executable, "version": platform.python_version()},
        "runtime_environment": _runtime_environment(runtime_row["device"]),
        "seeds": seed_rows,
        "source_bundle_sha256": digest_value(source_bundle),
        "development_evaluation_policy": (
            "post-training paired-CI only; forbidden for optimizer, early stop, "
            "checkpoint selection, or worker winner"
        ),
        "historical_confirmation": "absent",
    }
    plan = {**plan_payload, "plan_sha256": digest_value(plan_payload)}
    plan_path = output_root / "plan.json"
    _create_only_json(plan_path, plan, name="bridge plan")
    ledger_path = output_root / "ledger.json"
    if ledger_path.exists():
        ledger = _read_ledger(ledger_path)
        _require(ledger["plan_sha256"] == plan["plan_sha256"], "ledger plan mismatch")
    else:
        ledger = {
            "schema": BRIDGE_LEDGER_SCHEMA,
            "plan_sha256": plan["plan_sha256"],
            "state": "prepared",
            "seeds": {
                str(seed): {"local_activate": "pending", "matched_relax": "pending"}
                for seed in expected_seeds
            },
            "evidence": None,
        }
        _write_ledger(ledger_path, ledger)
    return plan


def _flag(name: str, value: Any) -> list[str]:
    return [name, str(value)]


def _stage_argv(
    plan: Mapping[str, Any], seed_row: Mapping[str, Any], stage: str
) -> list[str]:
    inputs = plan["input_roles"]
    runtime = plan["runtime"]
    budget = plan["budgets"][stage]
    common_data = [
        *_flag("--train-points", inputs["search-native-points"]["path"]),
        *_flag("--train-pullbacks", inputs["search-native-pullbacks"]["path"]),
        *_flag("--selection-points", inputs["search-native-points"]["path"]),
        *_flag("--selection-pullbacks", inputs["search-native-pullbacks"]["path"]),
        *_flag("--fixed-indices-file", plan["index_manifest"]["path"]),
        *_flag("--train-size", ROUND1_FIT_COUNT),
        *_flag("--selection-size", ROUND1_SELECTION_COUNT),
        *_flag("--seed", seed_row["optimizer_seed"]),
        *_flag("--data-seed", seed_row["data_seed"]),
        *_flag("--stochastic-batch-size", budget["batch_size"]),
        *_flag("--gradient-clip-norm", budget["gradient_clip_norm"]),
        *_flag("--train-chunk-size", runtime["train_chunk_size"]),
        *_flag("--feature-batch-size", runtime["feature_batch_size"]),
        *_flag("--eval-batch-size", runtime["eval_batch_size"]),
        *_flag("--threads", runtime["threads"]),
        *_flag("--device", runtime["device"]),
        *_flag(
            "--maximum-selection-tail-relative-degradation",
            plan["thresholds"]["maximum_search_tail_relative_degradation"],
        ),
        "--development-only",
    ]
    if stage == "local_activate":
        return [
            plan["python"]["path"],
            plan["workers"][stage]["path"],
            *_flag("--initial-checkpoint", seed_row["baseline_checkpoint"]["path"]),
            *common_data,
            *_flag("--output-dir", seed_row["local_output_dir"]),
            *_flag("--target-edge", "6:39"),
            "--orthogonalize-new-outputs",
            "--expansion-only",
            "--defer-block-acceptance",
            *_flag("--real-parameter-limit", ROUND1_TOTAL_ACTION_REAL_PARAMETER_CAP),
            *_flag("--sweeps", 2),
            *_flag("--epochs-per-block", 200),
            *_flag("--selection-eval-every", budget["eval_every"]),
            *_flag(
                "--early-stopping-evaluations",
                runtime["early_stopping_evaluations"],
            ),
            *_flag("--internal-learning-rate", budget["learning_rate"]),
            *_flag("--leaf-learning-rate", budget["learning_rate"]),
        ]
    if stage != "matched_relax":
        raise AutoResearchError(f"unregistered worker stage: {stage}")
    candidate = Path(seed_row["local_output_dir"]) / "development_candidate.pt"
    return [
        plan["python"]["path"],
        plan["workers"][stage]["path"],
        *_flag("--control-checkpoint", seed_row["baseline_checkpoint"]["path"]),
        *_flag("--candidate-checkpoint", candidate),
        *common_data,
        *_flag(
            "--development-evaluation-points",
            inputs["development-evaluation-points"]["path"],
        ),
        *_flag(
            "--development-evaluation-pullbacks",
            inputs["development-evaluation-pullbacks"]["path"],
        ),
        "--development-evaluation",
        *_flag("--development-evaluation-size", ROUND1_DEVELOPMENT_EVALUATION_COUNT),
        *_flag("--fixed-batch-plan-file", seed_row["batch_plan"]["path"]),
        *_flag("--batch-plan-seed", seed_row["batch_plan_seed"]),
        *_flag("--output-dir", seed_row["matched_output_dir"]),
        *_flag("--steps", budget["optimizer_updates"]),
        *_flag("--learning-rate", budget["learning_rate"]),
        *_flag("--scheduler", budget["scheduler"]),
        *_flag("--eval-every", budget["eval_every"]),
        *_flag("--minimum-relative-sigma-gain", 0.0),
        *_flag("--minimum-relative-chi-gain", 0.0),
    ]


def _verify_plan(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    _require((root / BRIDGE_SENTINEL).is_file(), "bridge sentinel is missing")
    plan = _read_object(root / "plan.json")
    payload = {key: value for key, value in plan.items() if key != "plan_sha256"}
    _require(
        plan.get("schema") == BRIDGE_PLAN_SCHEMA
        and plan.get("plan_sha256") == digest_value(payload),
        "bridge plan integrity check failed",
    )
    ledger = _read_ledger(root / "ledger.json")
    _require(ledger["plan_sha256"] == plan["plan_sha256"], "ledger plan mismatch")
    repository_root = Path(plan["workers"]["local_activate"]["path"]).parents[1]
    _require(
        _git_source_contract(repository_root) == plan["source_revision"],
        "git revision or dependency sources drifted after prepare",
    )
    fixed_path = Path(plan["index_manifest"]["path"])
    _require(
        sha256_file(fixed_path) == plan["index_manifest"]["file_sha256"],
        "fixed-index artifact drift",
    )
    for row in plan["workers"].values():
        _require(sha256_file(Path(row["path"])) == row["sha256"], "worker source drift")
    for row in plan["input_roles"].values():
        _require(sha256_file(Path(row["path"])) == row["sha256"], "search input drift")
    for seed_row in plan["seeds"]:
        checkpoint = seed_row["baseline_checkpoint"]
        _require(
            sha256_file(Path(checkpoint["path"])) == checkpoint["sha256"],
            "baseline checkpoint drift",
        )
        batch = seed_row["batch_plan"]
        batch_array = np.load(batch["path"], allow_pickle=False)
        _require(
            sha256_file(Path(batch["path"])) == batch["file_sha256"]
            and canonical_array_value_sha256(batch_array)
            == batch["arrays"]["batch_plan"]["value_sha256"],
            "fixed batch-plan artifact drift",
        )
    return plan, ledger


def run_round1_bridge(
    root: Path, *, selected_seed: int | None = None
) -> dict[str, Any]:
    """Run fixed workers with ``shell=False`` and resume completed stages."""

    root = root.expanduser().resolve()
    plan, ledger = _verify_plan(root)
    seeds = [row for row in plan["seeds"] if selected_seed in {None, row["seed"]}]
    _require(bool(seeds), "selected seed is not registered")
    ledger["state"] = "running"
    _write_ledger(root / "ledger.json", ledger)
    for seed_row in seeds:
        seed_key = str(seed_row["seed"])
        for stage in ("local_activate", "matched_relax"):
            output_key = (
                "local_output_dir"
                if stage == "local_activate"
                else "matched_output_dir"
            )
            stage_dir = Path(seed_row[output_key])
            report_path = stage_dir / "report.json"
            if report_path.is_file():
                report = _read_object(report_path)
                _require(
                    report.get("schema")
                    == (
                        "generic-quintic-adaptive-direct-blocks-report-v1"
                        if stage == "local_activate"
                        else "generic-quintic-tree-joint-relaxation-v1"
                    ),
                    f"{stage} report schema is invalid",
                )
                observed_report_sha256 = sha256_file(report_path)
                sealed_report_sha256 = ledger["seeds"][seed_key].get(
                    f"{stage}_report_sha256"
                )
                _require(
                    sealed_report_sha256 in {None, observed_report_sha256},
                    f"{stage} report differs from its sealed ledger hash",
                )
                ledger["seeds"][seed_key][stage] = "complete"
                ledger["seeds"][seed_key][
                    f"{stage}_report_sha256"
                ] = observed_report_sha256
                _write_ledger(root / "ledger.json", ledger)
                continue
            if stage_dir.exists():
                ledger["seeds"][seed_key][stage] = "incomplete-output"
                _write_ledger(root / "ledger.json", ledger)
                raise AutoResearchError(
                    f"{stage} has an incomplete create-only output directory"
                )
            argv = _stage_argv(plan, seed_row, stage)
            ledger["seeds"][seed_key][stage] = "running"
            _write_ledger(root / "ledger.json", ledger)
            try:
                subprocess.run(
                    argv, cwd=Path(__file__).resolve().parents[2], check=True
                )
            except subprocess.CalledProcessError:
                ledger["seeds"][seed_key][stage] = "failed"
                _write_ledger(root / "ledger.json", ledger)
                raise
            _require(report_path.is_file(), f"{stage} completed without report.json")
            ledger["seeds"][seed_key][stage] = "complete"
            ledger["seeds"][seed_key][f"{stage}_report_sha256"] = sha256_file(
                report_path
            )
            _write_ledger(root / "ledger.json", ledger)
    if all(
        row[stage] == "complete"
        for row in ledger["seeds"].values()
        for stage in ("local_activate", "matched_relax")
    ):
        ledger["state"] = "workers-complete"
        _write_ledger(root / "ledger.json", ledger)
    return ledger


def _load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    artifact = np.load(path, allow_pickle=False)
    return {name: np.asarray(artifact[name]) for name in artifact.files}


def _expect_equal_arrays(path: Path, expected_path: Path, names: set[str]) -> None:
    observed = _load_npz_arrays(path)
    expected = _load_npz_arrays(expected_path)
    _require(set(observed) == names, f"unexpected index arrays in {path}")
    _require(
        all(np.array_equal(observed[name], expected[name]) for name in names),
        f"worker indices differ from the frozen values: {path}",
    )


def _metric_row(row: Mapping[str, Any]) -> dict[str, Any]:
    statistics = row.get("statistics")
    tail = row.get("tail")
    if not isinstance(statistics, dict) or not isinstance(tail, dict):
        raise AutoResearchError("development-evaluation metric row is incomplete")
    try:
        result = {
            "sigma": float(statistics["sigma_official_formula"]),
            "chi": float(statistics["weighted_rms_abs_residual"]),
            "q999": float(tail["q999"]),
            "cvar_1pct": float(tail["cvar99"]),
            "minimum_metric_eigenvalue": float(
                statistics["min_eigenvalue_weighted_quantiles"]["q0.0000"]
            ),
            "nonpositive_metric_count": int(
                statistics["nonpositive_min_eigenvalue"]["count"]
            ),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise AutoResearchError(
            "development-evaluation metrics are malformed"
        ) from error
    if any(not math.isfinite(float(value)) for value in result.values()):
        raise AutoResearchError("development-evaluation metrics are non-finite")
    return result


def _validate_configuration_has_no_confirmation(report: Mapping[str, Any]) -> None:
    configuration = report.get("configuration")
    if not isinstance(configuration, dict):
        raise AutoResearchError("worker configuration is missing")
    for key in ("confirmation_points", "confirmation_pullbacks"):
        if configuration.get(key) is not None:
            raise AutoResearchError(
                "development worker received historical confirmation"
            )


def _require_configuration(
    configuration: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    stage: str,
) -> None:
    for key, value in expected.items():
        if configuration.get(key) != value:
            raise AutoResearchError(
                f"{stage} configuration mismatch for {key}: "
                f"{configuration.get(key)!r} != {value!r}"
            )


def _expected_input_hashes(plan: Mapping[str, Any]) -> dict[str, str]:
    roles = plan["input_roles"]
    native_points = roles["search-native-points"]["sha256"]
    native_pullbacks = roles["search-native-pullbacks"]["sha256"]
    return {
        "train_points": native_points,
        "train_pullbacks": native_pullbacks,
        "selection_points": native_points,
        "selection_pullbacks": native_pullbacks,
    }


def normalize_round1_bridge(root: Path) -> dict[str, Any]:
    """Normalize real worker reports into controller search evidence."""

    root = root.expanduser().resolve()
    plan, ledger = _verify_plan(root)
    _require(
        all(
            row[stage] == "complete"
            for row in ledger["seeds"].values()
            for stage in ("local_activate", "matched_relax")
        ),
        "all three seed/stage workers must complete before normalization",
    )
    fixed_indices = Path(plan["index_manifest"]["path"])
    evidence_rows = []
    for seed_row in plan["seeds"]:
        seed_key = str(seed_row["seed"])
        local_dir = Path(seed_row["local_output_dir"])
        matched_dir = Path(seed_row["matched_output_dir"])
        local_path = local_dir / "report.json"
        matched_path = matched_dir / "report.json"
        local = _read_object(local_path)
        matched = _read_object(matched_path)
        for stage, report_path in (
            ("local_activate", local_path),
            ("matched_relax", matched_path),
        ):
            _require(
                ledger["seeds"][seed_key].get(f"{stage}_report_sha256")
                == sha256_file(report_path),
                f"{stage} report hash differs from the resumable ledger",
            )
        _validate_configuration_has_no_confirmation(local)
        _validate_configuration_has_no_confirmation(matched)
        _require(local.get("confirmation") is None, "local report opened confirmation")
        _require(
            matched.get("confirmation") is None
            and matched.get("confirmation_passes") is None,
            "matched report opened confirmation",
        )
        local_config = local["configuration"]
        matched_config = matched["configuration"]
        local_budget = plan["budgets"]["local_activate"]
        matched_budget = plan["budgets"]["matched_relax"]
        runtime = plan["runtime"]
        roles = plan["input_roles"]
        native_points = roles["search-native-points"]["path"]
        native_pullbacks = roles["search-native-pullbacks"]["path"]
        dev_points = roles["development-evaluation-points"]["path"]
        dev_pullbacks = roles["development-evaluation-pullbacks"]["path"]
        _require_configuration(
            local_config,
            {
                "initial_checkpoint": seed_row["baseline_checkpoint"]["path"],
                "train_points": native_points,
                "train_pullbacks": native_pullbacks,
                "selection_points": native_points,
                "selection_pullbacks": native_pullbacks,
                "fixed_indices_file": str(fixed_indices),
                "output_dir": str(local_dir),
                "target_edge": ["6:39"],
                "rank_activation_scale": 1.0,
                "orthogonalize_new_outputs": True,
                "real_parameter_limit": ROUND1_TOTAL_ACTION_REAL_PARAMETER_CAP,
                "expansion_only": True,
                "defer_block_acceptance": True,
                "development_only": True,
                "sweeps": 2,
                "epochs_per_block": 200,
                "selection_eval_every": local_budget["eval_every"],
                "early_stopping_evaluations": runtime["early_stopping_evaluations"],
                "internal_learning_rate": local_budget["learning_rate"],
                "leaf_learning_rate": local_budget["learning_rate"],
                "train_size": ROUND1_FIT_COUNT,
                "selection_size": ROUND1_SELECTION_COUNT,
                "stochastic_batch_size": local_budget["batch_size"],
                "gradient_clip_norm": local_budget["gradient_clip_norm"],
                "maximum_selection_tail_relative_degradation": plan["thresholds"][
                    "maximum_search_tail_relative_degradation"
                ],
                "train_chunk_size": runtime["train_chunk_size"],
                "feature_batch_size": runtime["feature_batch_size"],
                "eval_batch_size": runtime["eval_batch_size"],
                "seed": seed_row["optimizer_seed"],
                "data_seed": seed_row["data_seed"],
                "threads": runtime["threads"],
                "device": runtime["device"],
            },
            stage="local_activate",
        )
        _require_configuration(
            matched_config,
            {
                "control_checkpoint": seed_row["baseline_checkpoint"]["path"],
                "candidate_checkpoint": str(local_dir / "development_candidate.pt"),
                "train_points": native_points,
                "train_pullbacks": native_pullbacks,
                "selection_points": native_points,
                "selection_pullbacks": native_pullbacks,
                "development_evaluation_points": dev_points,
                "development_evaluation_pullbacks": dev_pullbacks,
                "fixed_indices_file": str(fixed_indices),
                "fixed_batch_plan_file": seed_row["batch_plan"]["path"],
                "output_dir": str(matched_dir),
                "development_only": True,
                "development_evaluation": True,
                "steps": matched_budget["optimizer_updates"],
                "learning_rate": matched_budget["learning_rate"],
                "scheduler": matched_budget["scheduler"],
                "eval_every": matched_budget["eval_every"],
                "train_size": ROUND1_FIT_COUNT,
                "selection_size": ROUND1_SELECTION_COUNT,
                "development_evaluation_size": (ROUND1_DEVELOPMENT_EVALUATION_COUNT),
                "stochastic_batch_size": matched_budget["batch_size"],
                "gradient_clip_norm": matched_budget["gradient_clip_norm"],
                "maximum_selection_tail_relative_degradation": plan["thresholds"][
                    "maximum_search_tail_relative_degradation"
                ],
                "train_chunk_size": runtime["train_chunk_size"],
                "feature_batch_size": runtime["feature_batch_size"],
                "eval_batch_size": runtime["eval_batch_size"],
                "seed": seed_row["optimizer_seed"],
                "data_seed": seed_row["data_seed"],
                "batch_plan_seed": seed_row["batch_plan_seed"],
                "minimum_relative_sigma_gain": 0.0,
                "minimum_relative_chi_gain": 0.0,
                "threads": runtime["threads"],
                "device": runtime["device"],
            },
            stage="matched_relax",
        )
        _require(
            local.get("schema") == "generic-quintic-adaptive-direct-blocks-report-v1"
            and local.get("parent_checkpoint_sha256") == ROUND1_BASELINE_SHA256,
            "local report source contract is invalid",
        )
        _require(
            local_config.get("development_only") is True
            and local_config.get("expansion_only") is True
            and local_config.get("defer_block_acceptance") is True
            and local_config.get("target_edge") == ["6:39"],
            "local worker flags do not match the registered adapter",
        )
        blocks = local.get("blocks")
        histories = local.get("history")
        _require(
            isinstance(blocks, list)
            and len(blocks) == 3
            and isinstance(histories, list)
            and len(histories) <= 6,
            "Round 1 local worker must expose three blocks and at most two sweeps",
        )
        observed_local_updates = 0
        for history in histories:
            rows = history.get("rows") if isinstance(history, dict) else None
            _require(isinstance(rows, list), "local history rows are malformed")
            observed_local_updates += len(rows)
        _require(
            observed_local_updates <= local_budget["optimizer_updates"],
            "local worker exceeded the 1,200-update Round 1 budget",
        )
        expansion = local.get("expansion")
        _require(isinstance(expansion, dict), "local expansion audit is missing")
        source_dimensions = list(expansion.get("source_edge_dimensions", []))
        target_dimensions = list(expansion.get("target_edge_dimensions", []))
        _require(
            len(source_dimensions) == len(target_dimensions) > ROUND1_EDGE
            and source_dimensions[ROUND1_EDGE] == ROUND1_SOURCE_DIMENSION
            and target_dimensions[ROUND1_EDGE] == ROUND1_TARGET_DIMENSION
            and all(
                before == after
                for edge, (before, after) in enumerate(
                    zip(source_dimensions, target_dimensions, strict=True)
                )
                if edge != ROUND1_EDGE
            ),
            "local report did not perform exactly edge 6: 25->39",
        )
        counts = local.get("parameter_count", {})
        _require(
            counts.get("control_trainable_real_parameters")
            == ROUND1_CONTROL_REAL_PARAMETERS
            and counts.get("candidate_trainable_real_parameters")
            == ROUND1_CANDIDATE_REAL_PARAMETERS
            and counts.get("new_real_parameters") == ROUND1_NEW_REAL_PARAMETERS,
            "local parameter accounting is not the exact 9,800-real action",
        )
        audit = local.get("embedding_audit")
        _require(isinstance(audit, dict), "exact embedding audit is missing")
        for key in ("potential_max_absolute", "metric_max_relative_frobenius"):
            _finite_nonnegative(audit[key], name=key)
        equivalence_passes = bool(
            float(audit["potential_max_absolute"])
            <= plan["thresholds"]["equivalence_potential_absolute_tolerance"]
            and float(audit["metric_max_relative_frobenius"])
            <= plan["thresholds"]["equivalence_metric_relative_tolerance"]
        )
        _expect_equal_arrays(
            Path(local["data"]["indices"]), fixed_indices, {"fit", "selection"}
        )
        _expect_equal_arrays(
            Path(matched["data"]["indices"]),
            fixed_indices,
            {"fit", "selection", "development_evaluation"},
        )
        expected_input_hashes = _expected_input_hashes(plan)
        _require(
            local.get("data", {}).get("input_sha256") == expected_input_hashes,
            "local worker input-pool hashes differ from the locked protocol",
        )
        _require(
            matched.get("data", {}).get("input_sha256")
            == {
                **expected_input_hashes,
                "development_evaluation_points": roles["development-evaluation-points"][
                    "sha256"
                ],
                "development_evaluation_pullbacks": roles[
                    "development-evaluation-pullbacks"
                ]["sha256"],
                "confirmation_points": None,
                "confirmation_pullbacks": None,
            },
            "matched worker input-pool hashes differ from the locked protocol",
        )
        for report in (local, matched):
            _require(
                report.get("data", {}).get("data_seed") == seed_row["data_seed"],
                "worker data seed differs from the frozen index plan",
            )
        _require(
            matched.get("schema") == "generic-quintic-tree-joint-relaxation-v1"
            and matched.get("contract", {}).get("precision") == "complex64",
            "matched report precision/schema is invalid",
        )
        _require(
            matched_config.get("development_only") is True
            and matched_config.get("development_evaluation") is True,
            "matched worker did not isolate development evaluation",
        )
        dev = matched.get("development_evaluation")
        _require(
            isinstance(dev, dict) and dev.get("selection_or_checkpoint_role") == "none",
            "development evaluation was not declared post-selection only",
        )
        paired = dev.get("paired_improvement")
        _require(isinstance(paired, dict), "paired development CI is missing")
        try:
            paired_row = {
                "sigma_ci95_low": float(paired["sigma"]["ci95_low"]),
                "e2_ci95_low": float(paired["e2"]["ci95_low"]),
            }
        except (KeyError, TypeError, ValueError) as error:
            raise AutoResearchError("paired development CI is malformed") from error
        _require(
            all(math.isfinite(value) for value in paired_row.values()),
            "paired development CI is non-finite",
        )
        result_rows = matched.get("results", {})
        control_result = result_rows.get("control", {})
        candidate_result = result_rows.get("candidate", {})
        _require(
            control_result.get("total_real_parameters")
            == ROUND1_CONTROL_REAL_PARAMETERS
            and candidate_result.get("total_real_parameters")
            == ROUND1_CANDIDATE_REAL_PARAMETERS
            and candidate_result.get("new_real_parameters_vs_control")
            == ROUND1_NEW_REAL_PARAMETERS,
            "matched report parameter accounting differs from the local worker",
        )
        local_candidate = Path(local["development_candidate"])
        _require(
            sha256_file(local_candidate) == local["development_candidate_sha256"],
            "local candidate checkpoint hash mismatch",
        )
        _require(
            matched.get("source_checkpoint_sha256", {}).get("control")
            == ROUND1_BASELINE_SHA256
            and matched.get("source_checkpoint_sha256", {}).get("candidate")
            == sha256_file(local_candidate),
            "matched worker input checkpoints are not bound to the plan/local report",
        )
        planned_batch = np.load(seed_row["batch_plan"]["path"], allow_pickle=False)
        observed_batch = np.load(matched["batch_plan"]["path"], allow_pickle=False)
        _require(
            np.array_equal(planned_batch, observed_batch)
            and observed_batch.shape
            == (
                matched_budget["optimizer_updates"],
                min(ROUND1_FIT_COUNT, matched_budget["batch_size"]),
            )
            and np.issubdtype(observed_batch.dtype, np.integer)
            and canonical_array_value_sha256(observed_batch)
            == seed_row["batch_plan"]["arrays"]["batch_plan"]["value_sha256"],
            "matched worker batch plan differs from the frozen plan",
        )
        _require(
            matched.get("batch_plan", {}).get("fixed_source_sha256")
            == seed_row["batch_plan"]["file_sha256"]
            and matched.get("batch_plan", {}).get("seed") == seed_row["batch_plan_seed"]
            and matched.get("batch_plan", {}).get("steps")
            == matched_budget["optimizer_updates"]
            and matched.get("batch_plan", {}).get("batch_size")
            == min(ROUND1_FIT_COUNT, matched_budget["batch_size"]),
            "matched report batch-plan metadata differs from the frozen plan",
        )
        for report in (local, matched):
            _require(
                report.get("data", {}).get("fixed_indices_file_sha256")
                == plan["index_manifest"]["file_sha256"],
                "worker did not bind the fixed-index artifact",
            )
        control_checkpoint = Path(control_result["checkpoint"])
        candidate_checkpoint = Path(candidate_result["checkpoint"])
        _require(
            sha256_file(control_checkpoint) == control_result["checkpoint_sha256"]
            and sha256_file(candidate_checkpoint)
            == candidate_result["checkpoint_sha256"],
            "polished checkpoint hash mismatch",
        )
        evidence_rows.append(
            {
                "seed": seed_row["seed"],
                "equivalence": {
                    "passed": equivalence_passes,
                    "potential_max_absolute": float(audit["potential_max_absolute"]),
                    "metric_max_relative_frobenius": float(
                        audit["metric_max_relative_frobenius"]
                    ),
                },
                "local_activate_budget": plan["budgets"]["local_activate"],
                "control_matched_relax_budget": plan["budgets"]["matched_relax"],
                "candidate_matched_relax_budget": plan["budgets"]["matched_relax"],
                "control_matched_relax_batch_plan_sha256": canonical_array_value_sha256(
                    observed_batch
                ),
                "candidate_matched_relax_batch_plan_sha256": canonical_array_value_sha256(
                    observed_batch
                ),
                "control": _metric_row(dev["control"]),
                "candidate": _metric_row(dev["candidate"]),
                "paired": paired_row,
                "control_checkpoint_sha256": control_result["checkpoint_sha256"],
                "candidate_checkpoint_sha256": candidate_result["checkpoint_sha256"],
                "control_trainable_real_parameter_count": ROUND1_CONTROL_REAL_PARAMETERS,
                "candidate_trainable_real_parameter_count": ROUND1_CANDIDATE_REAL_PARAMETERS,
                "data_indices_sha256": plan["index_manifest"]["value_set_sha256"],
                "source_report_sha256": {
                    "local_activate": sha256_file(local_path),
                    "matched_relax": sha256_file(matched_path),
                },
            }
        )
    evidence = {
        "schema": SEARCH_EVIDENCE_SCHEMA,
        "candidate_id": plan["candidate_id"],
        "round": 1,
        "action_sha256": plan["action"]["action_sha256"],
        "parent_family_sha256": plan["action"]["parent_family_sha256"],
        "search_indices_sha256": plan["search_indices_sha256"],
        "precision": "complex64",
        "seeds": evidence_rows,
    }
    evidence_path = root / "search_evidence.json"
    _create_only_json(evidence_path, evidence, name="normalized search evidence")
    ledger["state"] = "normalized"
    ledger["evidence"] = {
        "path": str(evidence_path),
        "sha256": sha256_file(evidence_path),
        "value_sha256": digest_value(evidence),
    }
    _write_ledger(root / "ledger.json", ledger)
    return evidence


__all__ = [
    "BRIDGE_LEDGER_SCHEMA",
    "BRIDGE_PLAN_SCHEMA",
    "ROUND1_BASELINE_SHA256",
    "ROUND1_TOTAL_ACTION_REAL_PARAMETER_CAP",
    "canonical_array_value_sha256",
    "materialize_round1_indices",
    "normalize_round1_bridge",
    "prepare_round1_bridge",
    "run_round1_bridge",
    "validate_round1_contract",
]
