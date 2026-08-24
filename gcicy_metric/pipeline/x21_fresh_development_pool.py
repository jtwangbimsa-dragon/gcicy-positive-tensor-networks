"""Create and lock a fresh X21 development-selection common point pool.

The historical X21 49,152-point artifact was labelled ``confirmation`` and
has already informed model development.  It is therefore never a permissible
input to a new Auto Research campaign.  This module materializes a different,
selection-labelled pool with a fixed sampling contract and publishes enough
provenance for a separate v2 scientific protocol to consume it.

Authority is intentionally narrow:

* preparation creates one immutable plan and pins the clean source commit;
* execution may invoke only ``generate_gcicy_common_point_pool.py`` with the
  exact command stored in that plan, using ``shell=False``;
* completion re-hashes the pool and generator manifest before publishing an
  immutable receipt, binding, and protocol-v2 patch;
* neither preparation nor execution accepts an existing pool as an input.

The caller must provide the frozen-results root.  The managed output root is
rejected if it overlaps either that root or the source repository.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from .parallel_sampling import parallel_sampling_jobs
from .x21_auto_research import digest_value, sha256_file, validate_protocol


PLAN_SCHEMA = "gcicy-x21-fresh-development-pool-plan-v1"
ATTEMPT_SCHEMA = "gcicy-x21-fresh-development-pool-attempt-v1"
RECEIPT_SCHEMA = "gcicy-x21-fresh-development-pool-receipt-v1"
BINDING_SCHEMA = "gcicy-x21-fresh-development-binding-v1"
PROTOCOL_PATCH_SCHEMA = "gcicy-x21-auto-research-protocol-binding-patch-v2"
PROTOCOL_V1_SCHEMA = "gcicy-x21-auto-research-protocol-v1"
PROTOCOL_V2_SCHEMA = "gcicy-x21-auto-research-protocol-v2"

BASE_CAMPAIGN_ID = "x21-auto-research-v1"
TARGET_CAMPAIGN_ID = "x21-auto-research-v2"
ROOT_SENTINEL = ".gcicy-x21-fresh-development-pool-root"
ROOT_SENTINEL_VALUE = "x21-fresh-development-selection-v1"

ADAPTER = "p5p1_type21_k3_1223"
MODEL_SEED = 2_026_0802
SPLIT = "selection"
POINTS = 49_152
SAMPLING_SEED = 86_206
SAMPLING_CLUSTER_SIZE = 6
EXACT_MODEL = True

HISTORICAL_CONFIRMATION_SHA256 = (
    "4c4ec82e315351c9a5bebf0b3ba48b611bb8cb4ae433ce8a646e5c00426167cc"
)

GENERATOR_RELATIVE = "scripts/generate_gcicy_common_point_pool.py"
MANAGER_RELATIVE = "gcicy_metric/pipeline/x21_fresh_development_pool.py"
SOURCE_DEPENDENCIES = (
    MANAGER_RELATIVE,
    GENERATOR_RELATIVE,
    "gcicy_metric/pipeline/common_point_pool.py",
    "gcicy_metric/pipeline/parallel_sampling.py",
    "gcicy_metric/pipeline/adapter.py",
    "gcicy_metric/pipeline/registry.py",
    "gcicy_metric/pipeline/adapters/__init__.py",
    "gcicy_metric/pipeline/adapters/p5p1_k3_21.py",
    "gcicy_metric/type21_candidate_p5p1_1223.py",
)

POOL_FILENAME = "X21_selection_seed86206_n49152.npz"
MANIFEST_FILENAME = "X21_selection_seed86206_n49152.manifest.json"
PLAN_FILENAME = "plan.json"
RECEIPT_FILENAME = "generation_receipt.json"
BINDING_FILENAME = "fresh_development_binding.json"
PROTOCOL_PATCH_FILENAME = "x21_auto_research_protocol_binding_patch_v2.json"
PROTOCOL_V2_FILENAME = "x21_auto_research_v2.lock.json"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_FORBIDDEN_LABELS = ("confirmation", "blind", "holdout")


class X21FreshDevelopmentPoolError(RuntimeError):
    """Raised when fresh-pool materialization violates the locked contract."""


class X21FreshDevelopmentPoolTechnicalError(X21FreshDevelopmentPoolError):
    """Raised for a child-process failure, never a scientific rejection."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise X21FreshDevelopmentPoolError(message)


def _read_object(path: Path, *, role: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise X21FreshDevelopmentPoolError(
            f"cannot read {role} {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise X21FreshDevelopmentPoolError(f"{role} must be a JSON object: {path}")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, role: str) -> None:
    missing = expected - set(value)
    extra = set(value) - expected
    if missing or extra:
        raise X21FreshDevelopmentPoolError(
            f"{role} keys differ; missing={sorted(missing)} extra={sorted(extra)}"
        )


def _sha(value: Any, *, role: str) -> str:
    result = str(value).lower()
    if not _SHA256.fullmatch(result):
        raise X21FreshDevelopmentPoolError(f"{role} must be a lowercase SHA-256")
    return result


def _integer(value: Any, *, role: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise X21FreshDevelopmentPoolError(f"{role} must be an integer >= {minimum}")
    return value


def _atomic_create_json(path: Path, value: Any, *, role: str) -> str:
    """Publish a JSON artifact once, or accept an identical retry."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            value,
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
        if _read_object(path, role=role) != value:
            raise X21FreshDevelopmentPoolError(
                f"{role} already exists with different content"
            )
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


def _atomic_create_text(path: Path, value: str, *, role: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = value.encode("utf-8")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        if path.read_bytes() != encoded:
            raise X21FreshDevelopmentPoolError(
                f"{role} already exists with different content"
            )
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


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _artifact_with_value(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    return {**_artifact(path), "value_sha256": digest_value(value)}


def _paths_overlap(left: Path, right: Path) -> bool:
    left = left.resolve()
    right = right.resolve()
    return left == right or left in right.parents or right in left.parents


def _safe_existing_file(path: Path, *, role: str) -> Path:
    result = path.expanduser().resolve()
    _require(result.is_file(), f"{role} is missing: {result}")
    return result


def _reject_forbidden_path(path: Path, *, role: str) -> None:
    lowered = path.name.lower()
    _require(
        not any(token in lowered for token in _FORBIDDEN_LABELS),
        f"{role} must be explicitly selection-labelled, not held-out material",
    )
    _require("selection" in lowered, f"{role} must contain the selection label")


def _reject_historical_hash(value: Any, *, role: str) -> str:
    result = _sha(value, role=role)
    _require(
        result != HISTORICAL_CONFIRMATION_SHA256,
        f"{role} is the forbidden historical X21 confirmation artifact",
    )
    return result


def _git_source_contract(repository_root: Path) -> dict[str, Any]:
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
        raise X21FreshDevelopmentPoolError(
            "cannot bind the fresh-pool source revision"
        ) from error
    _require(_GIT_COMMIT.fullmatch(commit) is not None, "git commit is malformed")
    _require(
        branch.startswith("exp/"), "fresh-pool generation requires an exp/* branch"
    )
    _require(not dirty, "fresh-pool generation requires a clean worktree")
    files: dict[str, str] = {}
    for relative in SOURCE_DEPENDENCIES:
        source = (root / relative).resolve()
        _require(source.is_file(), f"source dependency is missing: {relative}")
        files[relative] = sha256_file(source)
    return {
        "repository_root": str(root),
        "commit": commit,
        "branch": branch,
        "tracked_and_untracked_worktree": "clean",
        "files": files,
    }


def _base_protocol_contract(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = _read_object(path, role="base X21 protocol")
    try:
        normalized = validate_protocol(raw)
    except Exception as error:
        raise X21FreshDevelopmentPoolError(
            f"base X21 protocol is invalid: {error}"
        ) from error
    _require(normalized["schema"] == PROTOCOL_V1_SCHEMA, "base protocol is not v1")
    _require(
        normalized["campaign_id"] == BASE_CAMPAIGN_ID,
        "base protocol campaign is not x21-auto-research-v1",
    )
    data = normalized["data_contract"]
    _require(
        data["development_pool_sha256"] == HISTORICAL_CONFIRMATION_SHA256,
        "base protocol no longer identifies the historical development artifact",
    )
    _require(
        data["development_points"] == POINTS
        and data["sampling_cluster_size"] == SAMPLING_CLUSTER_SIZE,
        "base protocol development sampling dimensions changed",
    )
    return raw, normalized


def _generation_contract(*, workers: int, backend: str) -> dict[str, Any]:
    _integer(workers, role="workers")
    _require(backend in {"process", "thread"}, "backend must be process or thread")
    return {
        "adapter": ADAPTER,
        "model_seed": MODEL_SEED,
        "exact_model": EXACT_MODEL,
        "split": SPLIT,
        "points": POINTS,
        "seed": SAMPLING_SEED,
        "cluster_size": SAMPLING_CLUSTER_SIZE,
        "workers": workers,
        "backend": backend,
        "root_separation_tolerance": None,
    }


def _generator_command(
    *,
    python_path: Path,
    generator_path: Path,
    pool_path: Path,
    manifest_path: Path,
    generation: Mapping[str, Any],
) -> list[str]:
    return [
        str(python_path),
        str(generator_path),
        "--adapter",
        ADAPTER,
        "--model-seed",
        str(MODEL_SEED),
        "--split",
        SPLIT,
        "--points",
        str(POINTS),
        "--seed",
        str(SAMPLING_SEED),
        "--cluster-size",
        str(SAMPLING_CLUSTER_SIZE),
        "--workers",
        str(generation["workers"]),
        "--backend",
        str(generation["backend"]),
        "--out",
        str(pool_path),
        "--manifest",
        str(manifest_path),
    ]


def prepare_fresh_development_pool(
    *,
    output_root: Path,
    frozen_root: Path,
    base_protocol: Path,
    repository_root: Path,
    python_executable: Path | None = None,
    workers: int = 6,
    backend: str = "process",
) -> dict[str, Any]:
    """Create or exactly resume the immutable fresh-pool generation plan."""

    root = output_root.expanduser().resolve()
    frozen = frozen_root.expanduser().resolve()
    repository = repository_root.expanduser().resolve()
    protocol_path = _safe_existing_file(base_protocol, role="base X21 protocol")
    python_path = _safe_existing_file(
        Path(sys.executable) if python_executable is None else python_executable,
        role="Python executable",
    )
    _require(frozen.is_dir(), f"frozen root is missing: {frozen}")
    _require(repository.is_dir(), f"repository root is missing: {repository}")
    _require(
        not _paths_overlap(root, frozen),
        "managed output root must not overlap the frozen-results root",
    )
    _require(
        not _paths_overlap(root, repository),
        "managed output root must not overlap the source repository",
    )
    for immutable in (protocol_path, python_path):
        try:
            immutable.relative_to(root)
        except ValueError:
            continue
        raise X21FreshDevelopmentPoolError(
            "managed output root contains an immutable preparation input"
        )

    generation = _generation_contract(workers=workers, backend=backend)
    raw_protocol, normalized_protocol = _base_protocol_contract(protocol_path)
    source = _git_source_contract(repository)
    generator_path = (repository / GENERATOR_RELATIVE).resolve()
    _require(generator_path.is_file(), "registered point-pool generator is missing")

    if root.exists() and any(root.iterdir()):
        sentinel = root / ROOT_SENTINEL
        _require(sentinel.is_file(), "refusing a non-empty unmanaged output root")
        _require(
            sentinel.read_text(encoding="utf-8").strip() == ROOT_SENTINEL_VALUE,
            "fresh-pool output root belongs to another workflow",
        )
    else:
        root.mkdir(parents=True, exist_ok=True)
        _atomic_create_text(
            root / ROOT_SENTINEL,
            f"{ROOT_SENTINEL_VALUE}\n",
            role="fresh-pool root sentinel",
        )

    pool_path = (root / POOL_FILENAME).resolve()
    manifest_path = (root / MANIFEST_FILENAME).resolve()
    _reject_forbidden_path(pool_path, role="fresh development pool")
    _reject_forbidden_path(manifest_path, role="fresh development manifest")
    command = _generator_command(
        python_path=python_path,
        generator_path=generator_path,
        pool_path=pool_path,
        manifest_path=manifest_path,
        generation=generation,
    )
    plan_core = {
        "schema": PLAN_SCHEMA,
        "output_root": str(root),
        "frozen_root": str(frozen),
        "generation": generation,
        "source_revision": source,
        "python": {"path": str(python_path)},
        "generator": {
            "path": str(generator_path),
            "sha256": sha256_file(generator_path),
        },
        "base_protocol": {
            **_artifact_with_value(protocol_path, raw_protocol),
            "normalized_sha256": digest_value(normalized_protocol),
            "schema": PROTOCOL_V1_SCHEMA,
            "campaign_id": BASE_CAMPAIGN_ID,
            "historical_development_pool_sha256": (HISTORICAL_CONFIRMATION_SHA256),
        },
        "outputs": {
            "pool": str(pool_path),
            "manifest": str(manifest_path),
            "receipt": str(root / RECEIPT_FILENAME),
            "binding": str(root / BINDING_FILENAME),
            "protocol_patch": str(root / PROTOCOL_PATCH_FILENAME),
            "protocol_v2": str(root / PROTOCOL_V2_FILENAME),
        },
        "command": command,
        "command_sha256": digest_value(command),
        "data_access_policy": {
            "existing_pool_inputs": [],
            "frozen_root": "read-never-write",
            "historical_confirmation_sha256": HISTORICAL_CONFIRMATION_SHA256,
            "confirmation": "forbidden",
            "blind": "forbidden",
            "holdout": "forbidden",
        },
    }
    plan = {**plan_core, "plan_sha256": digest_value(plan_core)}
    _atomic_create_json(root / PLAN_FILENAME, plan, role="fresh-pool plan")
    return plan


def _validate_artifact_row(value: Any, *, role: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise X21FreshDevelopmentPoolError(f"{role} must be an object")
    _exact_keys(value, {"path", "bytes", "sha256"}, role=role)
    path = _safe_existing_file(Path(value["path"]), role=role)
    size = _integer(value["bytes"], role=f"{role}.bytes", minimum=0)
    expected_sha256 = _sha(value["sha256"], role=f"{role}.sha256")
    _require(path.stat().st_size == size, f"{role} byte size drifted")
    _require(sha256_file(path) == expected_sha256, f"{role} hash drifted")
    return {"path": str(path), "bytes": size, "sha256": expected_sha256}


def _validate_value_artifact_row(value: Any, *, role: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise X21FreshDevelopmentPoolError(f"{role} must be an object")
    _exact_keys(value, {"path", "bytes", "sha256", "value_sha256"}, role=role)
    artifact = _validate_artifact_row(
        {key: value[key] for key in ("path", "bytes", "sha256")}, role=role
    )
    parsed = _read_object(Path(artifact["path"]), role=role)
    semantic = _sha(value["value_sha256"], role=f"{role}.value_sha256")
    _require(digest_value(parsed) == semantic, f"{role} semantic hash drifted")
    return {**artifact, "value_sha256": semantic}


def _verify_plan(output_root: Path) -> dict[str, Any]:
    root = output_root.expanduser().resolve()
    sentinel = root / ROOT_SENTINEL
    _require(sentinel.is_file(), "fresh-pool root sentinel is missing")
    _require(
        sentinel.read_text(encoding="utf-8").strip() == ROOT_SENTINEL_VALUE,
        "fresh-pool root sentinel changed",
    )
    plan = _read_object(root / PLAN_FILENAME, role="fresh-pool plan")
    core = {key: value for key, value in plan.items() if key != "plan_sha256"}
    _require(plan.get("schema") == PLAN_SCHEMA, "fresh-pool plan schema changed")
    _require(
        plan.get("plan_sha256") == digest_value(core),
        "fresh-pool plan integrity check failed",
    )
    _require(Path(plan["output_root"]).resolve() == root, "plan output root changed")
    frozen = Path(plan["frozen_root"]).resolve()
    repository = Path(plan["source_revision"]["repository_root"]).resolve()
    _require(frozen.is_dir(), "frozen-results root disappeared")
    _require(repository.is_dir(), "source repository disappeared")
    _require(
        not _paths_overlap(root, frozen),
        "managed output root now overlaps the frozen-results root",
    )
    _require(
        not _paths_overlap(root, repository),
        "managed output root now overlaps the source repository",
    )
    _require(
        plan["generation"]
        == _generation_contract(
            workers=plan["generation"].get("workers"),
            backend=plan["generation"].get("backend"),
        ),
        "fresh-pool generation contract changed",
    )
    source = _git_source_contract(repository)
    _require(source == plan["source_revision"], "source revision changed after prepare")
    generator = (repository / GENERATOR_RELATIVE).resolve()
    _require(
        plan["generator"] == {"path": str(generator), "sha256": sha256_file(generator)},
        "point-pool generator changed after prepare",
    )
    python_path = _safe_existing_file(
        Path(plan["python"]["path"]), role="locked Python executable"
    )
    base_row = plan["base_protocol"]
    base_path = _safe_existing_file(
        Path(base_row["path"]), role="locked base X21 protocol"
    )
    raw_protocol, normalized_protocol = _base_protocol_contract(base_path)
    expected_base = {
        **_artifact_with_value(base_path, raw_protocol),
        "normalized_sha256": digest_value(normalized_protocol),
        "schema": PROTOCOL_V1_SCHEMA,
        "campaign_id": BASE_CAMPAIGN_ID,
        "historical_development_pool_sha256": HISTORICAL_CONFIRMATION_SHA256,
    }
    _require(base_row == expected_base, "base X21 protocol changed after prepare")

    expected_outputs = {
        "pool": str(root / POOL_FILENAME),
        "manifest": str(root / MANIFEST_FILENAME),
        "receipt": str(root / RECEIPT_FILENAME),
        "binding": str(root / BINDING_FILENAME),
        "protocol_patch": str(root / PROTOCOL_PATCH_FILENAME),
        "protocol_v2": str(root / PROTOCOL_V2_FILENAME),
    }
    _require(plan["outputs"] == expected_outputs, "fresh-pool output paths changed")
    for role in ("pool", "manifest"):
        _reject_forbidden_path(Path(expected_outputs[role]), role=role)
    expected_command = _generator_command(
        python_path=python_path,
        generator_path=generator,
        pool_path=Path(expected_outputs["pool"]),
        manifest_path=Path(expected_outputs["manifest"]),
        generation=plan["generation"],
    )
    _require(plan["command"] == expected_command, "generator command changed")
    _require(
        plan["command_sha256"] == digest_value(expected_command),
        "generator command digest changed",
    )
    command_text = "\0".join(expected_command).lower()
    _require(
        not any(token in command_text for token in _FORBIDDEN_LABELS),
        "generator command references forbidden held-out material",
    )
    _require(
        plan["data_access_policy"]
        == {
            "existing_pool_inputs": [],
            "frozen_root": "read-never-write",
            "historical_confirmation_sha256": HISTORICAL_CONFIRMATION_SHA256,
            "confirmation": "forbidden",
            "blind": "forbidden",
            "holdout": "forbidden",
        },
        "fresh-pool data access policy changed",
    )
    return plan


def _npz_metadata(path: Path) -> dict[str, Any]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            schema_version = int(payload["common_point_pool_schema_version"])
            metadata = json.loads(str(payload["metadata_json"]))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise X21FreshDevelopmentPoolError(
            f"fresh development NPZ metadata is invalid: {error}"
        ) from error
    _require(schema_version == 1, "fresh development NPZ schema version changed")
    _require(
        isinstance(metadata, dict), "fresh development NPZ metadata is not an object"
    )
    return metadata


def _validate_generated_artifacts(plan: Mapping[str, Any]) -> dict[str, Any]:
    pool = _safe_existing_file(Path(plan["outputs"]["pool"]), role="generated pool")
    manifest_path = _safe_existing_file(
        Path(plan["outputs"]["manifest"]), role="generated manifest"
    )
    _reject_forbidden_path(pool, role="generated pool")
    _reject_forbidden_path(manifest_path, role="generated manifest")
    pool_sha256 = _reject_historical_hash(
        sha256_file(pool), role="generated development pool SHA-256"
    )
    manifest = _read_object(manifest_path, role="generated common-pool manifest")
    required_manifest = {
        "schema",
        "schema_version",
        "adapter",
        "adapter_version",
        "model_seed",
        "exact_model",
        "split",
        "point_count",
        "sampling_seed",
        "cluster_size",
        "cluster_count",
        "sampling",
        "extra",
        "pool_path",
        "pool_sha256",
        "arrays",
    }
    _exact_keys(manifest, required_manifest, role="generated common-pool manifest")
    generation = plan["generation"]
    expected_metadata = {
        "schema": "gcicy-common-point-pool-v1",
        "schema_version": 1,
        "adapter": ADAPTER,
        "adapter_version": "1",
        "model_seed": MODEL_SEED,
        "exact_model": True,
        "split": SPLIT,
        "point_count": POINTS,
        "sampling_seed": SAMPLING_SEED,
        "cluster_size": SAMPLING_CLUSTER_SIZE,
        "cluster_count": POINTS // SAMPLING_CLUSTER_SIZE,
    }
    for key, expected in expected_metadata.items():
        _require(manifest.get(key) == expected, f"generated manifest {key} changed")
    _require(
        Path(manifest["pool_path"]).resolve() == pool,
        "generated manifest points to another pool",
    )
    _require(
        _reject_historical_hash(
            manifest["pool_sha256"], role="generated manifest pool_sha256"
        )
        == pool_sha256,
        "generated manifest does not hash-bind the pool",
    )
    sampling = manifest["sampling"]
    _require(
        isinstance(sampling, dict), "generated manifest sampling must be an object"
    )
    _require(
        sampling.get("workers") == generation["workers"]
        and sampling.get("backend") == generation["backend"],
        "generated manifest runtime sampling contract changed",
    )
    _require(
        sampling.get("sampling_options") == {},
        "generated manifest sampling options changed",
    )
    seconds = sampling.get("seconds")
    _require(
        isinstance(seconds, (int, float))
        and not isinstance(seconds, bool)
        and math.isfinite(float(seconds))
        and float(seconds) >= 0,
        "generated manifest sampling duration is invalid",
    )
    expected_shards = [
        {"worker": index, "points": count, "seed": seed}
        for index, (count, seed) in enumerate(
            parallel_sampling_jobs(
                POINTS,
                SAMPLING_SEED,
                generation["workers"],
                SAMPLING_CLUSTER_SIZE,
            )
        )
    ]
    _require(
        sampling.get("shards") == expected_shards,
        "generated manifest sampler shard provenance changed",
    )
    arrays = manifest["arrays"]
    _require(isinstance(arrays, dict) and arrays, "generated manifest arrays are empty")
    for required_array in (
        "common_point_pool_schema_version",
        "importance_log_weights",
        "importance_weights",
        "sampling_cluster_ids",
        "baseline_metrics",
        "holomorphic_volume_log_density",
    ):
        _require(required_array in arrays, f"generated manifest lacks {required_array}")
    for name, row in arrays.items():
        _require(isinstance(row, dict), f"generated array {name} metadata is invalid")
        _require(set(row) == {"shape", "dtype"}, f"generated array {name} keys changed")
        shape = row["shape"]
        _require(isinstance(shape, list), f"generated array {name} shape is invalid")
        if name != "common_point_pool_schema_version":
            _require(
                bool(shape) and shape[0] == POINTS,
                f"generated array {name} does not contain {POINTS} rows",
            )
    extra = manifest["extra"]
    _require(isinstance(extra, dict), "generated manifest extra must be an object")
    _require(
        extra.get("git_commit") == plan["source_revision"]["commit"],
        "generated manifest source commit changed",
    )
    _require(
        extra.get("git_status_porcelain") == "",
        "generator observed a dirty source worktree",
    )
    _require(
        Path(str(extra.get("generator", ""))).resolve()
        == Path(plan["generator"]["path"]),
        "generated manifest names another generator",
    )
    npz_metadata = _npz_metadata(pool)
    _exact_keys(
        npz_metadata,
        required_manifest - {"pool_path", "pool_sha256", "arrays"},
        role="fresh development NPZ metadata",
    )
    for key, expected in expected_metadata.items():
        _require(npz_metadata.get(key) == expected, f"NPZ metadata {key} changed")
    _require(
        npz_metadata == {key: manifest[key] for key in npz_metadata},
        "NPZ metadata and generator manifest differ",
    )
    return {
        "pool": _artifact(pool),
        "npz_metadata_sha256": digest_value(npz_metadata),
        "manifest": {
            **_artifact_with_value(manifest_path, manifest),
            "value": manifest,
        },
    }


def _next_attempt_dir(root: Path) -> Path:
    attempts = root / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    indexes: list[int] = []
    for path in attempts.glob("attempt_*"):
        try:
            indexes.append(int(path.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return attempts / f"attempt_{max(indexes, default=0) + 1:03d}"


def _attempt_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((root / "attempts").glob("attempt_*/attempt.json")):
        value = _read_object(path, role="fresh-pool attempt receipt")
        core = {key: row for key, row in value.items() if key != "attempt_sha256"}
        _require(
            value.get("schema") == ATTEMPT_SCHEMA
            and value.get("attempt_sha256") == digest_value(core),
            "fresh-pool attempt receipt integrity failed",
        )
        rows.append(value)
    return rows


def _run_generator(plan: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(plan["output_root"])
    pool = Path(plan["outputs"]["pool"])
    manifest = Path(plan["outputs"]["manifest"])
    _require(
        not pool.exists() and not manifest.exists(),
        "generator outputs already exist without a successful bound completion",
    )
    command = list(plan["command"])
    _require(
        command[0] == plan["python"]["path"], "Python executable is not whitelisted"
    )
    _require(command[1] == plan["generator"]["path"], "generator is not whitelisted")
    attempt_dir = _next_attempt_dir(root)
    attempt_dir.mkdir(parents=True, exist_ok=False)
    started_at = utc_now()
    completed = subprocess.run(
        command,
        cwd=plan["source_revision"]["repository_root"],
        check=False,
        capture_output=True,
        text=True,
        shell=False,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    stdout_path = attempt_dir / "stdout.log"
    stderr_path = attempt_dir / "stderr.log"
    _atomic_create_text(stdout_path, completed.stdout, role="generator stdout")
    _atomic_create_text(stderr_path, completed.stderr, role="generator stderr")
    attempt_core = {
        "schema": ATTEMPT_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "started_at": started_at,
        "finished_at": utc_now(),
        "command": command,
        "command_sha256": digest_value(command),
        "shell": False,
        "native_exit_code": completed.returncode,
        "classification": (
            "native-exit-zero" if completed.returncode == 0 else "technical-failure"
        ),
        "scientific_rejection": False,
        "stdout_sha256": sha256_file(stdout_path),
        "stderr_sha256": sha256_file(stderr_path),
    }
    attempt = {**attempt_core, "attempt_sha256": digest_value(attempt_core)}
    attempt_path = attempt_dir / "attempt.json"
    _atomic_create_json(attempt_path, attempt, role="generator attempt receipt")
    if completed.returncode != 0:
        raise X21FreshDevelopmentPoolTechnicalError(
            f"fresh-pool generator exited {completed.returncode}; this is an "
            "operational failure, not a scientific rejection"
        )
    return {
        **attempt,
        "path": str(attempt_path),
        "file_sha256": sha256_file(attempt_path),
    }


def _last_successful_attempt(root: Path) -> dict[str, Any] | None:
    rows = _attempt_rows(root)
    successful = [row for row in rows if row["native_exit_code"] == 0]
    if not successful:
        return None
    result = successful[-1]
    attempt_path = sorted((root / "attempts").glob("attempt_*/attempt.json"))[
        rows.index(result)
    ]
    return {
        **result,
        "path": str(attempt_path),
        "file_sha256": sha256_file(attempt_path),
    }


def _publish_receipt(
    plan: Mapping[str, Any],
    generated: Mapping[str, Any],
    attempt: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = generated["manifest"]
    manifest_artifact = {
        key: manifest[key] for key in ("path", "bytes", "sha256", "value_sha256")
    }
    receipt_core = {
        "schema": RECEIPT_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "generation": dict(plan["generation"]),
        "source_revision": copy.deepcopy(plan["source_revision"]),
        "command_sha256": plan["command_sha256"],
        "attempt": {
            "path": attempt["path"],
            "file_sha256": attempt["file_sha256"],
            "attempt_sha256": attempt["attempt_sha256"],
            "native_exit_code": 0,
        },
        "pool": dict(generated["pool"]),
        "manifest": manifest_artifact,
        "npz_metadata_sha256": generated["npz_metadata_sha256"],
        "status": "complete-and-hash-locked",
        "historical_confirmation_sha256": HISTORICAL_CONFIRMATION_SHA256,
        "historical_confirmation_excluded": True,
        "confirmation_opened": False,
        "blind_opened": False,
    }
    receipt = {**receipt_core, "receipt_sha256": digest_value(receipt_core)}
    _atomic_create_json(
        Path(plan["outputs"]["receipt"]), receipt, role="fresh-pool generation receipt"
    )
    return receipt


def _binding_core(
    plan: Mapping[str, Any],
    generated: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    receipt_path = Path(plan["outputs"]["receipt"])
    plan_path = Path(plan["output_root"]) / PLAN_FILENAME
    manifest = generated["manifest"]
    return {
        "schema": BINDING_SCHEMA,
        "role": "fresh-development-search-only",
        "generation": {
            key: plan["generation"][key]
            for key in (
                "adapter",
                "model_seed",
                "exact_model",
                "split",
                "points",
                "seed",
                "cluster_size",
            )
        },
        "pool": dict(generated["pool"]),
        "manifest": {
            key: manifest[key] for key in ("path", "bytes", "sha256", "value_sha256")
        },
        "generation_receipt": _artifact_with_value(receipt_path, receipt),
        "plan": _artifact_with_value(plan_path, plan),
        "source_commit": plan["source_revision"]["commit"],
        "historical_exclusion": {
            "forbidden_pool_sha256": HISTORICAL_CONFIRMATION_SHA256,
            "observed_pool_sha256_differs": True,
            "confirmation_opened": False,
            "blind_opened": False,
            "holdout_opened": False,
        },
    }


def _validate_receipt(value: Any, *, plan_sha256: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise X21FreshDevelopmentPoolError("generation receipt must be an object")
    _exact_keys(
        value,
        {
            "schema",
            "plan_sha256",
            "generation",
            "source_revision",
            "command_sha256",
            "attempt",
            "pool",
            "manifest",
            "npz_metadata_sha256",
            "status",
            "historical_confirmation_sha256",
            "historical_confirmation_excluded",
            "confirmation_opened",
            "blind_opened",
            "receipt_sha256",
        },
        role="generation receipt",
    )
    core = {key: row for key, row in value.items() if key != "receipt_sha256"}
    _require(value.get("schema") == RECEIPT_SCHEMA, "generation receipt schema changed")
    _require(
        value.get("receipt_sha256") == digest_value(core),
        "generation receipt integrity check failed",
    )
    _require(value.get("plan_sha256") == plan_sha256, "generation receipt plan changed")
    generation = value.get("generation")
    _require(isinstance(generation, dict), "generation receipt contract is invalid")
    _integer(generation.get("workers"), role="generation receipt workers")
    _require(
        generation.get("backend") in {"process", "thread"},
        "generation receipt backend changed",
    )
    _require(
        generation
        == {
            "adapter": ADAPTER,
            "model_seed": MODEL_SEED,
            "exact_model": True,
            "split": SPLIT,
            "points": POINTS,
            "seed": SAMPLING_SEED,
            "cluster_size": SAMPLING_CLUSTER_SIZE,
            "workers": generation["workers"],
            "backend": generation["backend"],
            "root_separation_tolerance": None,
        },
        "generation receipt sampling contract changed",
    )
    _require(
        value.get("status") == "complete-and-hash-locked"
        and value.get("historical_confirmation_sha256")
        == HISTORICAL_CONFIRMATION_SHA256
        and value.get("historical_confirmation_excluded") is True
        and value.get("confirmation_opened") is False
        and value.get("blind_opened") is False,
        "generation receipt held-out policy changed",
    )
    _reject_historical_hash(value["pool"]["sha256"], role="receipt pool SHA-256")
    attempt = value["attempt"]
    _require(isinstance(attempt, dict), "generation receipt attempt is invalid")
    _exact_keys(
        attempt,
        {
            "path",
            "file_sha256",
            "attempt_sha256",
            "native_exit_code",
        },
        role="generation receipt attempt",
    )
    attempt_path = _safe_existing_file(
        Path(attempt["path"]), role="generation receipt attempt"
    )
    _require(
        sha256_file(attempt_path)
        == _sha(attempt["file_sha256"], role="attempt file SHA-256"),
        "generation attempt file hash drifted",
    )
    attempt_value = _read_object(attempt_path, role="generation receipt attempt")
    attempt_core = {
        key: row for key, row in attempt_value.items() if key != "attempt_sha256"
    }
    _require(
        attempt_value.get("schema") == ATTEMPT_SCHEMA
        and attempt_value.get("attempt_sha256") == digest_value(attempt_core)
        and attempt_value.get("attempt_sha256")
        == _sha(attempt["attempt_sha256"], role="attempt semantic SHA-256")
        and attempt_value.get("native_exit_code") == attempt["native_exit_code"] == 0,
        "generation attempt receipt changed",
    )
    return dict(value)


def validate_fresh_development_binding(
    value: Any,
    *,
    expected_binding_sha256: str | None = None,
    expected_pool_sha256: str | None = None,
    rehash_artifacts: bool = True,
) -> dict[str, Any]:
    """Validate and normalize a fresh development binding for a bridge.

    ``expected_binding_sha256`` is the semantic self-hash recorded by the v2
    protocol, while ``expected_pool_sha256`` is its data-contract hash.  When
    ``rehash_artifacts`` is true (the default), the pool, manifest, receipt,
    and plan are all read again from disk.
    """

    if not isinstance(value, dict):
        raise X21FreshDevelopmentPoolError(
            "fresh development binding must be an object"
        )
    expected_keys = {
        "schema",
        "role",
        "generation",
        "pool",
        "manifest",
        "generation_receipt",
        "plan",
        "source_commit",
        "historical_exclusion",
        "binding_sha256",
    }
    _exact_keys(value, expected_keys, role="fresh development binding")
    core = {key: row for key, row in value.items() if key != "binding_sha256"}
    _require(
        value["schema"] == BINDING_SCHEMA, "fresh development binding schema changed"
    )
    binding_sha256 = _sha(value["binding_sha256"], role="binding_sha256")
    _require(
        binding_sha256 == digest_value(core),
        "fresh development binding integrity failed",
    )
    if expected_binding_sha256 is not None:
        _require(
            binding_sha256
            == _sha(expected_binding_sha256, role="expected binding SHA-256"),
            "fresh development binding differs from the v2 protocol",
        )
    _require(
        value["role"] == "fresh-development-search-only",
        "fresh development binding role changed",
    )
    generation = value["generation"]
    _require(
        generation
        == {
            "adapter": ADAPTER,
            "model_seed": MODEL_SEED,
            "exact_model": True,
            "split": SPLIT,
            "points": POINTS,
            "seed": SAMPLING_SEED,
            "cluster_size": SAMPLING_CLUSTER_SIZE,
        },
        "fresh development generation identity changed",
    )
    if not isinstance(value["pool"], dict):
        raise X21FreshDevelopmentPoolError("fresh development pool artifact is invalid")
    _exact_keys(value["pool"], {"path", "bytes", "sha256"}, role="binding pool")
    pool_sha256 = _reject_historical_hash(
        value["pool"]["sha256"], role="binding pool SHA-256"
    )
    if expected_pool_sha256 is not None:
        _require(
            pool_sha256
            == _reject_historical_hash(
                expected_pool_sha256, role="expected development pool SHA-256"
            ),
            "fresh development pool differs from the v2 data contract",
        )
    _reject_forbidden_path(Path(value["pool"]["path"]), role="binding pool")
    exclusion = value["historical_exclusion"]
    _require(
        exclusion
        == {
            "forbidden_pool_sha256": HISTORICAL_CONFIRMATION_SHA256,
            "observed_pool_sha256_differs": True,
            "confirmation_opened": False,
            "blind_opened": False,
            "holdout_opened": False,
        },
        "fresh development historical-exclusion contract changed",
    )
    source_commit = str(value["source_commit"]).lower()
    _require(
        _GIT_COMMIT.fullmatch(source_commit) is not None,
        "binding source commit is malformed",
    )
    if rehash_artifacts:
        pool = _validate_artifact_row(value["pool"], role="binding pool")
        _require(pool["sha256"] == pool_sha256, "binding pool SHA-256 changed")
        manifest = _validate_value_artifact_row(
            value["manifest"], role="binding manifest"
        )
        receipt_artifact = _validate_value_artifact_row(
            value["generation_receipt"], role="binding generation receipt"
        )
        plan_artifact = _validate_value_artifact_row(value["plan"], role="binding plan")
        plan_value = _read_object(Path(plan_artifact["path"]), role="binding plan")
        plan_core = {
            key: row for key, row in plan_value.items() if key != "plan_sha256"
        }
        _require(
            plan_value.get("schema") == PLAN_SCHEMA
            and plan_value.get("plan_sha256") == digest_value(plan_core),
            "binding plan self-hash is invalid",
        )
        receipt_value = _read_object(
            Path(receipt_artifact["path"]), role="binding generation receipt"
        )
        receipt = _validate_receipt(
            receipt_value, plan_sha256=plan_value["plan_sha256"]
        )
        _require(receipt["pool"] == pool, "binding pool differs from its receipt")
        _require(
            receipt["manifest"] == manifest,
            "binding manifest differs from its receipt",
        )
        manifest_value = _read_object(Path(manifest["path"]), role="binding manifest")
        _require(
            manifest_value.get("pool_sha256") == pool_sha256,
            "binding manifest names another pool",
        )
        _require(
            source_commit == receipt["source_revision"]["commit"],
            "binding source commit differs from its receipt",
        )
    return copy.deepcopy(value)


def load_fresh_development_binding(
    path: Path,
    *,
    expected_binding_sha256: str | None = None,
    expected_pool_sha256: str | None = None,
) -> dict[str, Any]:
    return validate_fresh_development_binding(
        _read_object(path.expanduser().resolve(), role="fresh development binding"),
        expected_binding_sha256=expected_binding_sha256,
        expected_pool_sha256=expected_pool_sha256,
        rehash_artifacts=True,
    )


def protocol_v2_binding_contract(binding: Mapping[str, Any]) -> dict[str, Any]:
    validated = validate_fresh_development_binding(binding, rehash_artifacts=True)
    generation = validated["generation"]
    return {
        "schema": BINDING_SCHEMA,
        "binding_sha256": validated["binding_sha256"],
        "pool_sha256": validated["pool"]["sha256"],
        "manifest_sha256": validated["manifest"]["sha256"],
        "generation_receipt_sha256": validated["generation_receipt"]["sha256"],
        "adapter": ADAPTER,
        "model_seed": MODEL_SEED,
        "exact_model": True,
        "split": SPLIT,
        "points": generation["points"],
        "seed": generation["seed"],
        "sampling_cluster_size": generation["cluster_size"],
        "historical_confirmation_sha256": HISTORICAL_CONFIRMATION_SHA256,
    }


def apply_protocol_v2_binding(
    base_protocol: Mapping[str, Any], binding: Mapping[str, Any]
) -> dict[str, Any]:
    """Return a v2 protocol value without mutating the supplied v1 object."""

    try:
        normalized_base = validate_protocol(dict(base_protocol))
    except Exception as error:
        raise X21FreshDevelopmentPoolError(
            f"invalid base v1 protocol: {error}"
        ) from error
    _require(
        normalized_base["schema"] == PROTOCOL_V1_SCHEMA
        and normalized_base["campaign_id"] == BASE_CAMPAIGN_ID,
        "protocol-v2 binding requires the registered v1 base",
    )
    _require(
        normalized_base["data_contract"]["development_pool_sha256"]
        == HISTORICAL_CONFIRMATION_SHA256,
        "protocol-v2 binding base no longer has the historical development hash",
    )
    validated = validate_fresh_development_binding(binding, rehash_artifacts=True)
    result = copy.deepcopy(dict(base_protocol))
    result["schema"] = PROTOCOL_V2_SCHEMA
    result["campaign_id"] = TARGET_CAMPAIGN_ID
    result["data_contract"]["development_pool_sha256"] = validated["pool"]["sha256"]
    result["development_pool_binding"] = protocol_v2_binding_contract(validated)
    return result


def _publish_protocol_artifacts(
    plan: Mapping[str, Any], binding: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    base_path = Path(plan["base_protocol"]["path"])
    base = _read_object(base_path, role="locked base X21 protocol")
    target_protocol = apply_protocol_v2_binding(base, binding)
    target_path = Path(plan["outputs"]["protocol_v2"])
    _atomic_create_json(target_path, target_protocol, role="locked X21 v2 protocol")
    binding_contract = target_protocol["development_pool_binding"]
    operations = [
        {
            "op": "replace",
            "path": "/schema",
            "expected": PROTOCOL_V1_SCHEMA,
            "value": PROTOCOL_V2_SCHEMA,
        },
        {
            "op": "replace",
            "path": "/campaign_id",
            "expected": BASE_CAMPAIGN_ID,
            "value": TARGET_CAMPAIGN_ID,
        },
        {
            "op": "replace",
            "path": "/data_contract/development_pool_sha256",
            "expected": HISTORICAL_CONFIRMATION_SHA256,
            "value": binding["pool"]["sha256"],
        },
        {
            "op": "add",
            "path": "/development_pool_binding",
            "expected": None,
            "value": binding_contract,
        },
    ]
    binding_path = Path(plan["outputs"]["binding"])
    patch_core = {
        "schema": PROTOCOL_PATCH_SCHEMA,
        "base_protocol": dict(plan["base_protocol"]),
        "target_protocol": _artifact_with_value(target_path, target_protocol),
        "binding": _artifact_with_value(binding_path, binding),
        "operations": operations,
        "forbidden_historical_development_pool_sha256": (
            HISTORICAL_CONFIRMATION_SHA256
        ),
    }
    patch = {**patch_core, "patch_sha256": digest_value(patch_core)}
    _atomic_create_json(
        Path(plan["outputs"]["protocol_patch"]),
        patch,
        role="X21 protocol-v2 binding patch",
    )
    return target_protocol, patch


@contextmanager
def _execution_lock(root: Path) -> Iterator[None]:
    handle = (root / ".materialize.lock").open("a+", encoding="utf-8")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    try:
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def materialize_fresh_development_pool(output_root: Path) -> dict[str, Any]:
    """Execute or resume the fixed generator and publish immutable bindings."""

    root = output_root.expanduser().resolve()
    with _execution_lock(root):
        plan = _verify_plan(root)
        binding_path = Path(plan["outputs"]["binding"])
        if binding_path.is_file():
            binding = load_fresh_development_binding(binding_path)
            target_protocol, patch = _publish_protocol_artifacts(plan, binding)
            return {
                "status": "complete-and-hash-locked",
                "plan": plan,
                "binding": binding,
                "protocol_v2": target_protocol,
                "protocol_patch": patch,
            }

        pool_exists = Path(plan["outputs"]["pool"]).exists()
        manifest_exists = Path(plan["outputs"]["manifest"]).exists()
        successful_attempt = _last_successful_attempt(root)
        if pool_exists != manifest_exists:
            raise X21FreshDevelopmentPoolError(
                "partial immutable generator output exists; use a new managed root"
            )
        if successful_attempt is None:
            _require(
                not pool_exists,
                "unattributed generator outputs exist; use a new managed root",
            )
            successful_attempt = _run_generator(plan)
        else:
            _require(
                pool_exists and manifest_exists,
                "successful generator attempt is missing immutable outputs",
            )

        _verify_plan(root)
        generated = _validate_generated_artifacts(plan)
        receipt = _publish_receipt(plan, generated, successful_attempt)
        binding_core = _binding_core(plan, generated, receipt)
        binding = {**binding_core, "binding_sha256": digest_value(binding_core)}
        _atomic_create_json(binding_path, binding, role="fresh development binding")
        binding = load_fresh_development_binding(binding_path)
        target_protocol, patch = _publish_protocol_artifacts(plan, binding)
        return {
            "status": "complete-and-hash-locked",
            "plan": plan,
            "binding": binding,
            "protocol_v2": target_protocol,
            "protocol_patch": patch,
        }


def fresh_development_pool_status(output_root: Path) -> dict[str, Any]:
    root = output_root.expanduser().resolve()
    plan = _verify_plan(root)
    pool_exists = Path(plan["outputs"]["pool"]).exists()
    manifest_exists = Path(plan["outputs"]["manifest"]).exists()
    binding_path = Path(plan["outputs"]["binding"])
    attempts = _attempt_rows(root)
    if binding_path.is_file():
        binding = load_fresh_development_binding(binding_path)
        patch_path = Path(plan["outputs"]["protocol_patch"])
        target_path = Path(plan["outputs"]["protocol_v2"])
        _require(patch_path.is_file(), "completed binding lacks its protocol-v2 patch")
        _require(
            target_path.is_file(), "completed binding lacks its locked v2 protocol"
        )
        _publish_protocol_artifacts(plan, binding)
        state = "complete-and-hash-locked"
        binding_sha256: str | None = binding["binding_sha256"]
        pool_sha256: str | None = binding["pool"]["sha256"]
    elif pool_exists != manifest_exists:
        state = "partial-output-requires-new-root"
        binding_sha256 = None
        pool_sha256 = None
    elif any(row["native_exit_code"] == 0 for row in attempts) and not pool_exists:
        state = "successful-attempt-missing-output-requires-new-root"
        binding_sha256 = None
        pool_sha256 = None
    elif any(row["native_exit_code"] == 0 for row in attempts):
        state = "generation-complete-binding-pending"
        binding_sha256 = None
        pool_sha256 = None
    elif attempts and pool_exists:
        state = "failed-attempt-output-requires-new-root"
        binding_sha256 = None
        pool_sha256 = None
    elif attempts:
        state = "retryable-technical-failure"
        binding_sha256 = None
        pool_sha256 = None
    else:
        state = "prepared"
        binding_sha256 = None
        pool_sha256 = None
    return {
        "schema": "gcicy-x21-fresh-development-pool-status-v1",
        "status": state,
        "plan_sha256": plan["plan_sha256"],
        "attempt_count": len(attempts),
        "technical_failure_count": sum(
            row["native_exit_code"] != 0 for row in attempts
        ),
        "scientific_rejection": False,
        "binding_sha256": binding_sha256,
        "pool_sha256": pool_sha256,
        "historical_confirmation_excluded": True,
    }


__all__ = [
    "ADAPTER",
    "ATTEMPT_SCHEMA",
    "BINDING_SCHEMA",
    "EXACT_MODEL",
    "HISTORICAL_CONFIRMATION_SHA256",
    "MODEL_SEED",
    "PLAN_SCHEMA",
    "POINTS",
    "PROTOCOL_PATCH_SCHEMA",
    "PROTOCOL_V2_SCHEMA",
    "RECEIPT_SCHEMA",
    "SAMPLING_CLUSTER_SIZE",
    "SAMPLING_SEED",
    "SPLIT",
    "TARGET_CAMPAIGN_ID",
    "X21FreshDevelopmentPoolError",
    "X21FreshDevelopmentPoolTechnicalError",
    "apply_protocol_v2_binding",
    "fresh_development_pool_status",
    "load_fresh_development_binding",
    "materialize_fresh_development_pool",
    "prepare_fresh_development_pool",
    "protocol_v2_binding_contract",
    "validate_fresh_development_binding",
]
