"""Fail-closed control plane for the post-v1 X21 auto-research campaign.

The module deliberately separates operational checkpoint recovery from the
scientific leaderboard.  ``RecoveryStore`` records only provenance and process
status in its own run root.  ``X21CampaignStore`` accepts only three-seed,
fixed-budget, paired scientific evidence and exposes the four preregistered
single-variable actions in order:

``optimizer_path -> dictionary_unfreeze -> bond_growth_d16 -> site_growth_k24``.

This file is a controller.  It never launches a trainer or accepts a command,
environment variable, path, or arbitrary mutation from an action artifact.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterator, Mapping, Sequence


PROTOCOL_SCHEMA = "gcicy-x21-auto-research-protocol-v1"
PROTOCOL_V2_SCHEMA = "gcicy-x21-auto-research-protocol-v2"
FRESH_DEVELOPMENT_BINDING_SCHEMA = "gcicy-x21-fresh-development-binding-v1"
FAMILY_SCHEMA = "gcicy-x21-model-family-v1"
ACTION_SCHEMA = "gcicy-x21-auto-research-action-v1"
EVIDENCE_SCHEMA = "gcicy-x21-scientific-evidence-v1"
ADJUDICATION_SCHEMA = "gcicy-x21-round-adjudication-v1"
RESOURCE_SKIP_SCHEMA = "gcicy-x21-resource-skip-v1"
RESOURCE_CERTIFICATE_SCHEMA = "gcicy-x21-resource-certificate-v1"
LEDGER_SCHEMA = "gcicy-x21-auto-research-ledger-v1"
RECOVERY_EVIDENCE_SCHEMA = "gcicy-x21-exact-recovery-evidence-v1"
BASELINE_COMPLETION_SCHEMA = "gcicy-x21-baseline-completion-certificate-v1"
RECOVERY_LEDGER_SCHEMA = "gcicy-x21-recovery-ledger-v1"

ROOT_SENTINEL = ".gcicy-x21-auto-research-root"
RECOVERY_ROOT_SENTINEL = ".gcicy-x21-recovery-root"
PHYSICAL_DICTIONARY_RANK = 121
HISTORICAL_CONFIRMATION_SHA256 = (
    "4c4ec82e315351c9a5bebf0b3ba48b611bb8cb4ae433ce8a646e5c00426167cc"
)
FRESH_DEVELOPMENT_ADAPTER = "p5p1_type21_k3_1223"
FRESH_DEVELOPMENT_MODEL_SEED = 2_026_0802
FRESH_DEVELOPMENT_SEED = 86_206
FRESH_DEVELOPMENT_POINTS = 49_152
FRESH_DEVELOPMENT_CLUSTER_SIZE = 6
DICTIONARY_REAL_PARAMETERS = 2 * PHYSICAL_DICTIONARY_RANK**2
ACTION_KINDS = (
    "optimizer_path",
    "dictionary_unfreeze",
    "bond_growth_d16",
    "site_growth_k24",
)
PIPELINE = (
    "epoch-zero-equivalence",
    "matched-fixed-budget",
    "paired-development-evaluation",
)
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_RECOVERY_TOKENS = ("recovery", "sigsegv", "segfault", "native_crash")


class X21AutoResearchError(RuntimeError):
    """Raised when an X21 controller artifact violates the locked contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise X21AutoResearchError(
            f"could not read JSON artifact {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise X21AutoResearchError(f"JSON artifact must be an object: {path}")
    return value


def _atomic_write_json(path: Path, value: Any) -> None:
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
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_create_only(path: Path, value: Any, *, context: str) -> str:
    """Create a JSON artifact once, or verify an identical retry."""

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
        existing = _read_json(path)
        if canonical_json_bytes(existing) != canonical_json_bytes(value):
            raise X21AutoResearchError(
                f"{context} already exists with different content"
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


def _exact_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] | None = None,
    context: str,
) -> None:
    optional = optional or set()
    missing = required - set(value)
    extra = set(value) - required - optional
    if missing or extra:
        raise X21AutoResearchError(
            f"{context} keys differ; missing={sorted(missing)} extra={sorted(extra)}"
        )


def _sha(value: Any, *, context: str) -> str:
    result = str(value).lower()
    if not _SHA256.fullmatch(result):
        raise X21AutoResearchError(f"{context} must be a lowercase SHA-256 digest")
    return result


def _integer(value: Any, *, context: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise X21AutoResearchError(f"{context} must be an integer >= {minimum}")
    return value


def _number(value: Any, *, context: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise X21AutoResearchError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise X21AutoResearchError(f"{context} is outside its finite range")
    return result


def _boolean(value: Any, *, context: str) -> bool:
    if not isinstance(value, bool):
        raise X21AutoResearchError(f"{context} must be boolean")
    return value


def _reject_recovery_material(value: Any, *, context: str) -> None:
    """Keep operational-recovery text out of scientific artifacts."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_recovery_material(str(key), context=context)
            _reject_recovery_material(item, context=context)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_recovery_material(item, context=context)
    elif isinstance(value, str):
        lowered = value.lower()
        if any(token in lowered for token in _RECOVERY_TOKENS):
            raise X21AutoResearchError(
                f"{context} contains operational recovery material"
            )


def coefficient_real_parameters(k: int, bond_dimension: int) -> int:
    return (
        2
        * PHYSICAL_DICTIONARY_RANK
        * (2 * bond_dimension + (k - 2) * bond_dimension**2)
    )


def expected_real_parameters(metadata: Mapping[str, Any]) -> int:
    count = coefficient_real_parameters(
        int(metadata["k"]), int(metadata["bond_dimension"])
    )
    if bool(metadata["train_physical_dictionary"]):
        count += DICTIONARY_REAL_PARAMETERS
    return count


def _normalize_metadata(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise X21AutoResearchError(f"{context} must be an object")
    _exact_keys(
        value,
        required={
            "k",
            "bond_dimension",
            "train_physical_dictionary",
            "trainable_real_parameter_count",
            "precision",
        },
        context=context,
    )
    result = {
        "k": _integer(value["k"], context=f"{context}.k", minimum=3),
        "bond_dimension": _integer(
            value["bond_dimension"], context=f"{context}.bond_dimension", minimum=1
        ),
        "train_physical_dictionary": _boolean(
            value["train_physical_dictionary"],
            context=f"{context}.train_physical_dictionary",
        ),
        "trainable_real_parameter_count": _integer(
            value["trainable_real_parameter_count"],
            context=f"{context}.trainable_real_parameter_count",
            minimum=1,
        ),
        "precision": str(value["precision"]),
    }
    if result["precision"] != "complex64":
        raise X21AutoResearchError(f"{context}.precision must be complex64")
    if result["trainable_real_parameter_count"] != expected_real_parameters(result):
        raise X21AutoResearchError(f"{context} parameter count is inconsistent")
    return result


def _normalize_data_contract(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise X21AutoResearchError("data_contract must be an object")
    _exact_keys(
        value,
        required={
            "source_artifact_sha256",
            "train_pool_sha256",
            "selection_pool_sha256",
            "development_pool_sha256",
            "train_points",
            "selection_points",
            "development_points",
            "sampling_cluster_size",
        },
        context="data_contract",
    )
    return {
        "source_artifact_sha256": _sha(
            value["source_artifact_sha256"],
            context="data_contract.source_artifact_sha256",
        ),
        "train_pool_sha256": _sha(
            value["train_pool_sha256"], context="data_contract.train_pool_sha256"
        ),
        "selection_pool_sha256": _sha(
            value["selection_pool_sha256"],
            context="data_contract.selection_pool_sha256",
        ),
        "development_pool_sha256": _sha(
            value["development_pool_sha256"],
            context="data_contract.development_pool_sha256",
        ),
        "train_points": _integer(
            value["train_points"], context="data_contract.train_points", minimum=1
        ),
        "selection_points": _integer(
            value["selection_points"],
            context="data_contract.selection_points",
            minimum=1,
        ),
        "development_points": _integer(
            value["development_points"],
            context="data_contract.development_points",
            minimum=1,
        ),
        "sampling_cluster_size": _integer(
            value["sampling_cluster_size"],
            context="data_contract.sampling_cluster_size",
            minimum=1,
        ),
    }


def _normalize_budget(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise X21AutoResearchError(f"{context} must be an object")
    _exact_keys(
        value,
        required={
            "optimizer_updates",
            "batch_size",
            "gradient_clip_norm",
            "evaluation_examples",
            "bootstrap_replicates",
            "scheduler",
        },
        context=context,
    )
    scheduler = str(value["scheduler"])
    if scheduler not in {"constant", "cosine"}:
        raise X21AutoResearchError(f"{context}.scheduler is not registered")
    return {
        "optimizer_updates": _integer(
            value["optimizer_updates"],
            context=f"{context}.optimizer_updates",
            minimum=1,
        ),
        "batch_size": _integer(
            value["batch_size"], context=f"{context}.batch_size", minimum=1
        ),
        "gradient_clip_norm": _number(
            value["gradient_clip_norm"],
            context=f"{context}.gradient_clip_norm",
            minimum=0.0,
        ),
        "evaluation_examples": _integer(
            value["evaluation_examples"],
            context=f"{context}.evaluation_examples",
            minimum=1,
        ),
        "bootstrap_replicates": _integer(
            value["bootstrap_replicates"],
            context=f"{context}.bootstrap_replicates",
            minimum=1,
        ),
        "scheduler": scheduler,
    }


def _normalize_thresholds(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise X21AutoResearchError("thresholds must be an object")
    required = {
        "minimum_median_relative_sigma_gain",
        "minimum_median_relative_chi_gain",
        "minimum_improved_seeds",
        "minimum_paired_positive_seeds",
        "maximum_seed_relative_regression",
        "maximum_tail_relative_degradation",
        "equivalence_potential_absolute_tolerance",
        "equivalence_metric_relative_tolerance",
    }
    _exact_keys(value, required=required, context="thresholds")
    result = {
        key: _number(value[key], context=f"thresholds.{key}", minimum=0.0)
        for key in required
        if key not in {"minimum_improved_seeds", "minimum_paired_positive_seeds"}
    }
    for key in ("minimum_improved_seeds", "minimum_paired_positive_seeds"):
        result[key] = _integer(value[key], context=f"thresholds.{key}", minimum=1)
        if result[key] > 3:
            raise X21AutoResearchError(f"thresholds.{key} exceeds the seed count")
    return result


def _normalize_action_sequence(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != len(ACTION_KINDS):
        raise X21AutoResearchError("action_sequence must contain exactly four actions")
    normalized = []
    expected_fields = {
        "optimizer_path": ("learning_rate", 3.0e-5, 1.5e-5),
        "dictionary_unfreeze": ("train_physical_dictionary", False, True),
        "bond_growth_d16": ("bond_dimension", 14, 16),
        "site_growth_k24": ("k", 20, 24),
    }
    for index, row in enumerate(value, start=1):
        if not isinstance(row, dict):
            raise X21AutoResearchError("action_sequence rows must be objects")
        _exact_keys(
            row,
            required={"round", "kind", "field", "control_value", "candidate_value"},
            context=f"action_sequence[{index - 1}]",
        )
        kind = str(row["kind"])
        if row["round"] != index or kind != ACTION_KINDS[index - 1]:
            raise X21AutoResearchError("action_sequence order is fixed")
        field, control_value, candidate_value = expected_fields[kind]
        if (
            row["field"] != field
            or row["control_value"] != control_value
            or row["candidate_value"] != candidate_value
        ):
            raise X21AutoResearchError(f"action_sequence contract changed for {kind}")
        normalized.append(dict(row))
    return normalized


def _normalize_baseline_provenance_contracts(
    value: Any, *, seeds: Sequence[int], recovery_seed: int
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != len(seeds):
        raise X21AutoResearchError(
            "baseline_provenance_contracts must cover exactly three seeds"
        )
    normalized = []
    for index, row in enumerate(value):
        context = f"baseline_provenance_contracts[{index}]"
        if not isinstance(row, dict):
            raise X21AutoResearchError(f"{context} must be an object")
        kind = str(row.get("kind"))
        common = {
            "seed",
            "kind",
            "certificate_schema",
            "source_campaign_id",
            "source_plan_sha256",
            "source_job_id",
            "source_job_digest",
        }
        recovery = {
            "recovery_campaign_id",
            "source_model_sha256",
            "source_checkpoint_sha256",
            "source_epoch",
            "training_semantics_sha256",
            "frozen_inputs_sha256",
        }
        _exact_keys(
            row,
            required=common | (recovery if kind == "exact-recovery" else set()),
            context=context,
        )
        seed = _integer(row["seed"], context=f"{context}.seed", minimum=1)
        if kind not in {"existing-complete", "exact-recovery", "fresh-completion"}:
            raise X21AutoResearchError(f"{context}.kind is not registered")
        expected_schema = (
            RECOVERY_EVIDENCE_SCHEMA
            if kind == "exact-recovery"
            else BASELINE_COMPLETION_SCHEMA
        )
        if row["certificate_schema"] != expected_schema:
            raise X21AutoResearchError(f"{context}.certificate_schema is invalid")
        source_campaign_id = str(row["source_campaign_id"])
        source_job_id = str(row["source_job_id"])
        if not _IDENTIFIER.fullmatch(source_campaign_id) or not _IDENTIFIER.fullmatch(
            source_job_id
        ):
            raise X21AutoResearchError(f"{context} source identity is invalid")
        result = {
            "seed": seed,
            "kind": kind,
            "certificate_schema": expected_schema,
            "source_campaign_id": source_campaign_id,
            "source_plan_sha256": _sha(
                row["source_plan_sha256"], context=f"{context}.source_plan_sha256"
            ),
            "source_job_id": source_job_id,
            "source_job_digest": _sha(
                row["source_job_digest"], context=f"{context}.source_job_digest"
            ),
        }
        if kind == "exact-recovery":
            recovery_campaign_id = str(row["recovery_campaign_id"])
            if not _IDENTIFIER.fullmatch(recovery_campaign_id):
                raise X21AutoResearchError(f"{context}.recovery_campaign_id is invalid")
            result.update(
                {
                    "recovery_campaign_id": recovery_campaign_id,
                    "source_model_sha256": _sha(
                        row["source_model_sha256"],
                        context=f"{context}.source_model_sha256",
                    ),
                    "source_checkpoint_sha256": _sha(
                        row["source_checkpoint_sha256"],
                        context=f"{context}.source_checkpoint_sha256",
                    ),
                    "source_epoch": _integer(
                        row["source_epoch"], context=f"{context}.source_epoch"
                    ),
                    "training_semantics_sha256": _sha(
                        row["training_semantics_sha256"],
                        context=f"{context}.training_semantics_sha256",
                    ),
                    "frozen_inputs_sha256": _sha(
                        row["frozen_inputs_sha256"],
                        context=f"{context}.frozen_inputs_sha256",
                    ),
                }
            )
        normalized.append(result)
    normalized.sort(key=lambda row: row["seed"])
    if [row["seed"] for row in normalized] != list(seeds):
        raise X21AutoResearchError(
            "baseline provenance contracts do not match promotion seeds"
        )
    recovered = next(row for row in normalized if row["seed"] == recovery_seed)
    if recovered["kind"] != "exact-recovery" or any(
        row["seed"] != recovery_seed and row["kind"] == "exact-recovery"
        for row in normalized
    ):
        raise X21AutoResearchError(
            "exactly the registered recovery seed must use exact recovery"
        )
    return normalized


def _normalize_fresh_development_binding_contract(value: Any) -> dict[str, Any]:
    """Validate the portable binding summary embedded in a v2 protocol.

    The materialization controller validates and re-hashes the full binding
    artifact before emitting this summary.  Scientific bridges must also load
    that full artifact and match ``binding_sha256`` and ``pool_sha256`` before
    they may emit evidence.
    """

    if not isinstance(value, dict):
        raise X21AutoResearchError("development_pool_binding must be an object")
    _exact_keys(
        value,
        required={
            "schema",
            "binding_sha256",
            "pool_sha256",
            "manifest_sha256",
            "generation_receipt_sha256",
            "adapter",
            "model_seed",
            "exact_model",
            "split",
            "points",
            "seed",
            "sampling_cluster_size",
            "historical_confirmation_sha256",
        },
        context="development_pool_binding",
    )
    if value["schema"] != FRESH_DEVELOPMENT_BINDING_SCHEMA:
        raise X21AutoResearchError(
            "development_pool_binding schema is not the registered fresh binding"
        )
    result = {
        "schema": FRESH_DEVELOPMENT_BINDING_SCHEMA,
        "binding_sha256": _sha(
            value["binding_sha256"],
            context="development_pool_binding.binding_sha256",
        ),
        "pool_sha256": _sha(
            value["pool_sha256"], context="development_pool_binding.pool_sha256"
        ),
        "manifest_sha256": _sha(
            value["manifest_sha256"],
            context="development_pool_binding.manifest_sha256",
        ),
        "generation_receipt_sha256": _sha(
            value["generation_receipt_sha256"],
            context="development_pool_binding.generation_receipt_sha256",
        ),
        "adapter": str(value["adapter"]),
        "model_seed": _integer(
            value["model_seed"], context="development_pool_binding.model_seed"
        ),
        "exact_model": _boolean(
            value["exact_model"], context="development_pool_binding.exact_model"
        ),
        "split": str(value["split"]),
        "points": _integer(
            value["points"], context="development_pool_binding.points", minimum=1
        ),
        "seed": _integer(
            value["seed"], context="development_pool_binding.seed", minimum=1
        ),
        "sampling_cluster_size": _integer(
            value["sampling_cluster_size"],
            context="development_pool_binding.sampling_cluster_size",
            minimum=1,
        ),
        "historical_confirmation_sha256": _sha(
            value["historical_confirmation_sha256"],
            context="development_pool_binding.historical_confirmation_sha256",
        ),
    }
    expected_identity = {
        "adapter": FRESH_DEVELOPMENT_ADAPTER,
        "model_seed": FRESH_DEVELOPMENT_MODEL_SEED,
        "exact_model": True,
        "split": "selection",
        "points": FRESH_DEVELOPMENT_POINTS,
        "seed": FRESH_DEVELOPMENT_SEED,
        "sampling_cluster_size": FRESH_DEVELOPMENT_CLUSTER_SIZE,
        "historical_confirmation_sha256": HISTORICAL_CONFIRMATION_SHA256,
    }
    for key, expected in expected_identity.items():
        if result[key] != expected:
            raise X21AutoResearchError(
                f"development_pool_binding.{key} changed from the fresh-pool contract"
            )
    if result["pool_sha256"] == HISTORICAL_CONFIRMATION_SHA256:
        raise X21AutoResearchError(
            "development_pool_binding reuses the forbidden historical confirmation"
        )
    return result


def validate_protocol(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise X21AutoResearchError("protocol must be an object")
    schema = str(value.get("schema"))
    if schema not in {PROTOCOL_SCHEMA, PROTOCOL_V2_SCHEMA}:
        raise X21AutoResearchError(
            f"protocol schema must be {PROTOCOL_SCHEMA} or {PROTOCOL_V2_SCHEMA}"
        )
    required = {
        "schema",
        "campaign_id",
        "promotion_seeds",
        "recovery_seed",
        "baseline_provenance_contracts",
        "expected_baseline_metadata",
        "data_contract",
        "budget",
        "allowed_memory_batch_sizes",
        "thresholds",
        "action_sequence",
        "maximum_rounds",
    }
    if schema == PROTOCOL_V2_SCHEMA:
        required.add("development_pool_binding")
    _exact_keys(
        value,
        required=required,
        context="protocol",
    )
    campaign_id = str(value["campaign_id"])
    if not _IDENTIFIER.fullmatch(campaign_id):
        raise X21AutoResearchError("campaign_id has invalid characters")
    if schema == PROTOCOL_V2_SCHEMA and campaign_id != "x21-auto-research-v2":
        raise X21AutoResearchError(
            "the v2 protocol campaign_id must be x21-auto-research-v2"
        )
    seeds_raw = value["promotion_seeds"]
    if not isinstance(seeds_raw, list) or len(seeds_raw) != 3:
        raise X21AutoResearchError("exactly three promotion seeds are required")
    seeds = sorted(
        _integer(seed, context="promotion_seeds", minimum=1) for seed in seeds_raw
    )
    if len(set(seeds)) != 3:
        raise X21AutoResearchError("promotion seeds must be distinct")
    recovery_seed = _integer(value["recovery_seed"], context="recovery_seed", minimum=1)
    if recovery_seed not in seeds:
        raise X21AutoResearchError("recovery_seed must be a promotion seed")
    provenance_contracts = _normalize_baseline_provenance_contracts(
        value["baseline_provenance_contracts"],
        seeds=seeds,
        recovery_seed=recovery_seed,
    )
    baseline_metadata = _normalize_metadata(
        value["expected_baseline_metadata"], context="expected_baseline_metadata"
    )
    if baseline_metadata != {
        "k": 20,
        "bond_dimension": 14,
        "train_physical_dictionary": False,
        "trainable_real_parameter_count": coefficient_real_parameters(20, 14),
        "precision": "complex64",
    }:
        raise X21AutoResearchError("the baseline must be cores-only k20,D14 complex64")
    data_contract = _normalize_data_contract(value["data_contract"])
    development_binding = None
    if schema == PROTOCOL_V2_SCHEMA:
        development_binding = _normalize_fresh_development_binding_contract(
            value["development_pool_binding"]
        )
        if (
            data_contract["development_pool_sha256"]
            != development_binding["pool_sha256"]
        ):
            raise X21AutoResearchError(
                "v2 data_contract development hash differs from its fresh binding"
            )
        if (
            data_contract["development_points"] != FRESH_DEVELOPMENT_POINTS
            or data_contract["sampling_cluster_size"] != FRESH_DEVELOPMENT_CLUSTER_SIZE
        ):
            raise X21AutoResearchError(
                "v2 development pool dimensions differ from the fresh binding"
            )
    budget = _normalize_budget(value["budget"], context="budget")
    allowed = value["allowed_memory_batch_sizes"]
    if not isinstance(allowed, list) or not allowed:
        raise X21AutoResearchError("allowed_memory_batch_sizes must be non-empty")
    allowed_batches = [
        _integer(item, context="allowed_memory_batch_sizes", minimum=1)
        for item in allowed
    ]
    if allowed_batches != sorted(set(allowed_batches), reverse=True):
        raise X21AutoResearchError(
            "allowed_memory_batch_sizes must be unique and descending"
        )
    if budget["batch_size"] not in allowed_batches:
        raise X21AutoResearchError("default batch size is not in the memory ladder")
    maximum_rounds = _integer(
        value["maximum_rounds"], context="maximum_rounds", minimum=1
    )
    if maximum_rounds != len(ACTION_KINDS):
        raise X21AutoResearchError("maximum_rounds must be four")
    data_contract_sha256 = (
        digest_value(data_contract)
        if development_binding is None
        else digest_value(
            {
                "data_contract": data_contract,
                "development_pool_binding": development_binding,
            }
        )
    )
    normalized = {
        "schema": schema,
        "campaign_id": campaign_id,
        "promotion_seeds": seeds,
        "recovery_seed": recovery_seed,
        "baseline_provenance_contracts": provenance_contracts,
        "expected_baseline_metadata": baseline_metadata,
        "data_contract": data_contract,
        "data_contract_sha256": data_contract_sha256,
        "budget": budget,
        "allowed_memory_batch_sizes": allowed_batches,
        "thresholds": _normalize_thresholds(value["thresholds"]),
        "action_sequence": _normalize_action_sequence(value["action_sequence"]),
        "maximum_rounds": maximum_rounds,
        "search_precision": "complex64",
        "finalist_replay_precision": "complex128",
        "development_role": (
            "search-only-already-informed-selection"
            if development_binding is None
            else "fresh-search-only-selection"
        ),
        "blind_policy": "absent-requires-separate-freeze-manifest",
    }
    if development_binding is not None:
        normalized["development_pool_binding"] = development_binding
    return normalized


def _normalize_provenance(value: Any, *, context: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise X21AutoResearchError(f"{context} must be an object")
    _exact_keys(
        value,
        required={"kind", "certificate_sha256"},
        optional={"certificate"},
        context=context,
    )
    kind = str(value["kind"])
    if kind not in {
        "existing-complete",
        "exact-recovery",
        "fresh-completion",
        "scientific-action",
    }:
        raise X21AutoResearchError(f"{context}.kind is not registered")
    return {
        "kind": kind,
        "certificate_sha256": _sha(
            value["certificate_sha256"], context=f"{context}.certificate_sha256"
        ),
    }


def _validate_baseline_certificate_contract(
    certificate: Any,
    *,
    contract: Mapping[str, Any],
    seed: int,
    runnable_model_sha256: str,
) -> dict[str, Any]:
    if contract["kind"] == "exact-recovery":
        normalized = validate_recovery_evidence(certificate)
        expected = {
            "campaign_id": contract["recovery_campaign_id"],
            "seed": seed,
            "source_campaign_id": contract["source_campaign_id"],
            "source_plan_sha256": contract["source_plan_sha256"],
            "source_job_id": contract["source_job_id"],
            "source_job_digest": contract["source_job_digest"],
            "source_model_sha256": contract["source_model_sha256"],
            "source_checkpoint_sha256": contract["source_checkpoint_sha256"],
            "source_epoch": contract["source_epoch"],
            "source_next_epoch": contract["source_epoch"] + 1,
            "training_semantics_sha256": contract["training_semantics_sha256"],
            "frozen_inputs_sha256": contract["frozen_inputs_sha256"],
            "output_model_sha256": runnable_model_sha256,
        }
    else:
        normalized = validate_baseline_completion_evidence(certificate)
        expected = {
            "seed": seed,
            "source_campaign_id": contract["source_campaign_id"],
            "source_plan_sha256": contract["source_plan_sha256"],
            "source_job_id": contract["source_job_id"],
            "source_job_digest": contract["source_job_digest"],
            "output_model_sha256": runnable_model_sha256,
        }
    for key, expected_value in expected.items():
        if normalized.get(key) != expected_value:
            raise X21AutoResearchError(
                f"baseline certificate {key} does not match its protocol contract"
            )
    return normalized


def normalize_family(
    value: Any,
    *,
    seeds: Sequence[int],
    context: str,
    expected_metadata: Mapping[str, Any] | None = None,
    recovery_seed: int | None = None,
    require_recovery_certificate: bool = False,
    baseline_provenance_contracts: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise X21AutoResearchError(f"{context} must be an object")
    _exact_keys(
        value,
        required={"schema", "model_metadata", "models"},
        optional={"family_sha256"},
        context=context,
    )
    if value["schema"] != FAMILY_SCHEMA:
        raise X21AutoResearchError(f"{context}.schema must be {FAMILY_SCHEMA}")
    metadata = _normalize_metadata(
        value["model_metadata"], context=f"{context}.model_metadata"
    )
    if expected_metadata is not None and metadata != dict(expected_metadata):
        raise X21AutoResearchError(f"{context} metadata does not match the contract")
    rows = value["models"]
    if not isinstance(rows, list) or len(rows) != len(seeds):
        raise X21AutoResearchError(f"{context} must contain exactly three models")
    normalized_rows = []
    contracts_by_seed = (
        None
        if baseline_provenance_contracts is None
        else {int(row["seed"]): row for row in baseline_provenance_contracts}
    )
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise X21AutoResearchError(f"{context}.models[{index}] must be an object")
        _exact_keys(
            row,
            required={"seed", "checkpoint_sha256", "provenance"},
            context=f"{context}.models[{index}]",
        )
        seed = _integer(
            row["seed"], context=f"{context}.models[{index}].seed", minimum=1
        )
        checkpoint_sha256 = _sha(
            row["checkpoint_sha256"],
            context=f"{context}.models[{index}].checkpoint_sha256",
        )
        raw_provenance = row["provenance"]
        provenance = _normalize_provenance(
            raw_provenance, context=f"{context}.models[{index}].provenance"
        )
        certificate = raw_provenance.get("certificate")
        if certificate is not None:
            if contracts_by_seed is None or seed not in contracts_by_seed:
                raise X21AutoResearchError(
                    "inline provenance certificates are accepted only for a baseline"
                )
            contract = contracts_by_seed[seed]
            if provenance["kind"] != contract["kind"]:
                raise X21AutoResearchError(
                    "baseline provenance kind differs from its protocol contract"
                )
            normalized_certificate = _validate_baseline_certificate_contract(
                certificate,
                contract=contract,
                seed=seed,
                runnable_model_sha256=checkpoint_sha256,
            )
            if digest_value(normalized_certificate) != provenance["certificate_sha256"]:
                raise X21AutoResearchError(
                    "baseline certificate_sha256 is not bound to its certificate"
                )
        elif contracts_by_seed is not None or (
            require_recovery_certificate and provenance["kind"] == "exact-recovery"
        ):
            raise X21AutoResearchError(
                "baseline provenance requires its hash-bound certificate"
            )
        normalized_rows.append(
            {
                "seed": seed,
                "checkpoint_sha256": checkpoint_sha256,
                "provenance": provenance,
            }
        )
    normalized_rows.sort(key=lambda row: row["seed"])
    if [row["seed"] for row in normalized_rows] != list(seeds):
        raise X21AutoResearchError(f"{context} seeds do not match the protocol")
    if len({row["checkpoint_sha256"] for row in normalized_rows}) != len(seeds):
        raise X21AutoResearchError(f"{context} checkpoint hashes must be distinct")
    if contracts_by_seed is not None and len(
        {row["provenance"]["certificate_sha256"] for row in normalized_rows}
    ) != len(seeds):
        raise X21AutoResearchError(
            f"{context} provenance certificate hashes must be distinct"
        )
    if recovery_seed is not None:
        recovered = next(row for row in normalized_rows if row["seed"] == recovery_seed)
        if recovered["provenance"]["kind"] != "exact-recovery":
            raise X21AutoResearchError(
                "the registered recovery seed requires exact-recovery provenance"
            )
        for row in normalized_rows:
            if (
                row["seed"] != recovery_seed
                and row["provenance"]["kind"] == "exact-recovery"
            ):
                raise X21AutoResearchError(
                    "only the registered recovery seed may be recovered"
                )
    core = {
        "schema": FAMILY_SCHEMA,
        "model_metadata": metadata,
        "models": normalized_rows,
    }
    family_sha256 = digest_value(core)
    if "family_sha256" in value and value["family_sha256"] != family_sha256:
        raise X21AutoResearchError(f"{context}.family_sha256 is inconsistent")
    return {**core, "family_sha256": family_sha256}


def _candidate_metadata(parent: Mapping[str, Any], kind: str) -> dict[str, Any]:
    result = dict(parent)
    if kind == "dictionary_unfreeze":
        if result["train_physical_dictionary"]:
            raise X21AutoResearchError("dictionary is already trainable")
        result["train_physical_dictionary"] = True
    elif kind == "bond_growth_d16":
        if result["bond_dimension"] != 14:
            raise X21AutoResearchError("D16 growth requires a D14 champion")
        result["bond_dimension"] = 16
    elif kind == "site_growth_k24":
        if result["k"] != 20:
            raise X21AutoResearchError("k24 growth requires a k20 champion")
        result["k"] = 24
    result["trainable_real_parameter_count"] = expected_real_parameters(result)
    return result


def _normalize_resource_certificate(
    value: Any,
    *,
    protocol: Mapping[str, Any],
    round_index: int,
    kind: str,
    candidate_metadata: Mapping[str, Any],
    required_outcome: str,
    selected_batch_size: int | None,
) -> dict[str, Any]:
    """Bind a resource decision to its normalized certificate contents."""

    if not isinstance(value, dict):
        raise X21AutoResearchError("resource certificate must be an object")
    _exact_keys(
        value,
        required={
            "schema",
            "round",
            "kind",
            "candidate_metadata_sha256",
            "attempted_batch_sizes",
            "selected_batch_size",
            "outcome",
            "preflight_report_sha256",
        },
        context="resource certificate",
    )
    if value["schema"] != RESOURCE_CERTIFICATE_SCHEMA:
        raise X21AutoResearchError(
            f"resource certificate schema must be {RESOURCE_CERTIFICATE_SCHEMA}"
        )
    if value["round"] != round_index or value["kind"] != kind:
        raise X21AutoResearchError("resource certificate identifies another action")
    metadata_sha256 = _sha(
        value["candidate_metadata_sha256"],
        context="resource_certificate.candidate_metadata_sha256",
    )
    if metadata_sha256 != digest_value(candidate_metadata):
        raise X21AutoResearchError("resource certificate model metadata changed")
    attempts_raw = value["attempted_batch_sizes"]
    if not isinstance(attempts_raw, list) or not attempts_raw:
        raise X21AutoResearchError(
            "resource certificate attempted_batch_sizes must be non-empty"
        )
    attempts = [
        _integer(item, context="resource_certificate.attempted_batch_sizes", minimum=1)
        for item in attempts_raw
    ]
    ladder = list(protocol["allowed_memory_batch_sizes"])
    if attempts != ladder[: len(attempts)]:
        raise X21AutoResearchError(
            "resource certificate did not follow the registered memory ladder"
        )
    outcome = str(value["outcome"])
    if outcome != required_outcome:
        raise X21AutoResearchError("resource certificate outcome changed")
    if outcome == "feasible":
        selected = _integer(
            value["selected_batch_size"],
            context="resource_certificate.selected_batch_size",
            minimum=1,
        )
        if selected != selected_batch_size or attempts[-1] != selected:
            raise X21AutoResearchError(
                "resource certificate does not bind the selected batch size"
            )
    elif outcome == "resource-infeasible":
        if value["selected_batch_size"] is not None or attempts != ladder:
            raise X21AutoResearchError(
                "resource-infeasible certificate must exhaust the memory ladder"
            )
        selected = None
    else:
        raise X21AutoResearchError("resource certificate outcome is not registered")
    return {
        "schema": RESOURCE_CERTIFICATE_SCHEMA,
        "round": round_index,
        "kind": kind,
        "candidate_metadata_sha256": metadata_sha256,
        "attempted_batch_sizes": attempts,
        "selected_batch_size": selected,
        "outcome": outcome,
        "preflight_report_sha256": _sha(
            value["preflight_report_sha256"],
            context="resource_certificate.preflight_report_sha256",
        ),
    }


def validate_action(
    value: Any,
    *,
    protocol: Mapping[str, Any],
    parent_family: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise X21AutoResearchError("action must be an object")
    _reject_recovery_material(value, context="scientific action")
    _exact_keys(
        value,
        required={
            "schema",
            "candidate_id",
            "round",
            "parent_family_sha256",
            "data_contract_sha256",
            "mutation",
            "budget",
            "pipeline",
        },
        optional={
            "resource_certificate_sha256",
            "resource_certificate",
            "candidate_metadata",
            "action_sha256",
        },
        context="action",
    )
    if value["schema"] != ACTION_SCHEMA:
        raise X21AutoResearchError(f"action schema must be {ACTION_SCHEMA}")
    candidate_id = str(value["candidate_id"])
    if not _IDENTIFIER.fullmatch(candidate_id):
        raise X21AutoResearchError("candidate_id has invalid characters")
    round_index = _integer(value["round"], context="action.round", minimum=1)
    if round_index > protocol["maximum_rounds"]:
        raise X21AutoResearchError("action round exceeds the protocol")
    sequence = protocol["action_sequence"][round_index - 1]
    if value["parent_family_sha256"] != parent_family["family_sha256"]:
        raise X21AutoResearchError("action parent is not the current champion")
    if value["data_contract_sha256"] != protocol["data_contract_sha256"]:
        raise X21AutoResearchError("action data contract is not locked")
    if list(value["pipeline"]) != list(PIPELINE):
        raise X21AutoResearchError("action pipeline is fixed")
    budget = _normalize_budget(value["budget"], context="action.budget")
    for key, expected in protocol["budget"].items():
        if key != "batch_size" and budget[key] != expected:
            raise X21AutoResearchError(f"action budget changed {key}")
    kind = sequence["kind"]
    batch_size = budget["batch_size"]
    resource_certificate_sha256 = value.get("resource_certificate_sha256")
    resource_certificate_value = value.get("resource_certificate")
    parent_metadata = parent_family["model_metadata"]
    if kind in {"optimizer_path", "dictionary_unfreeze"}:
        if batch_size != protocol["budget"]["batch_size"]:
            raise X21AutoResearchError(
                "non-structural actions use the default batch size"
            )
        if (
            resource_certificate_sha256 is not None
            or resource_certificate_value is not None
        ):
            raise X21AutoResearchError(
                "non-structural action cannot claim a resource certificate"
            )
    else:
        if batch_size not in protocol["allowed_memory_batch_sizes"]:
            raise X21AutoResearchError("structural action batch size is not registered")
        resource_required = kind == "bond_growth_d16" or (
            kind == "site_growth_k24" and parent_metadata["bond_dimension"] == 16
        )
        if resource_required and (
            resource_certificate_sha256 is None or resource_certificate_value is None
        ):
            raise X21AutoResearchError(
                "structural action requires a hash-bound resource certificate"
            )
        if (resource_certificate_sha256 is None) != (
            resource_certificate_value is None
        ):
            raise X21AutoResearchError(
                "resource certificate and certificate_sha256 must be supplied together"
            )
    mutation = value["mutation"]
    if not isinstance(mutation, dict):
        raise X21AutoResearchError("action.mutation must be an object")
    if kind == "optimizer_path":
        _exact_keys(
            mutation,
            required={"kind", "field", "control_value", "candidate_value"},
            context="action.mutation",
        )
        normalized_mutation = {
            "kind": kind,
            "field": "learning_rate",
            "control_value": _number(
                mutation["control_value"],
                context="action.mutation.control_value",
                minimum=0.0,
            ),
            "candidate_value": _number(
                mutation["candidate_value"],
                context="action.mutation.candidate_value",
                minimum=0.0,
            ),
        }
    elif kind == "dictionary_unfreeze":
        _exact_keys(
            mutation,
            required={
                "kind",
                "field",
                "control_value",
                "candidate_value",
                "new_real_parameters",
            },
            context="action.mutation",
        )
        normalized_mutation = {
            "kind": kind,
            "field": "train_physical_dictionary",
            "control_value": _boolean(
                mutation["control_value"], context="action.mutation.control_value"
            ),
            "candidate_value": _boolean(
                mutation["candidate_value"], context="action.mutation.candidate_value"
            ),
            "new_real_parameters": _integer(
                mutation["new_real_parameters"],
                context="action.mutation.new_real_parameters",
                minimum=1,
            ),
        }
        if normalized_mutation["new_real_parameters"] != DICTIONARY_REAL_PARAMETERS:
            raise X21AutoResearchError("dictionary parameter increment changed")
    else:
        _exact_keys(
            mutation,
            required={
                "kind",
                "field",
                "control_value",
                "candidate_value",
                "transport",
                "activation_scale",
                "new_real_parameters",
            },
            context="action.mutation",
        )
        transport = str(mutation["transport"])
        expected_transport = (
            "one-sided-function-preserving"
            if kind == "bond_growth_d16"
            else "repeat-sites-function-preserving"
        )
        normalized_mutation = {
            "kind": kind,
            "field": sequence["field"],
            "control_value": _integer(
                mutation["control_value"],
                context="action.mutation.control_value",
                minimum=1,
            ),
            "candidate_value": _integer(
                mutation["candidate_value"],
                context="action.mutation.candidate_value",
                minimum=1,
            ),
            "transport": transport,
            "activation_scale": _number(
                mutation["activation_scale"],
                context="action.mutation.activation_scale",
                minimum=0.0,
            ),
            "new_real_parameters": _integer(
                mutation["new_real_parameters"],
                context="action.mutation.new_real_parameters",
                minimum=1,
            ),
        }
        if (
            transport != expected_transport
            or normalized_mutation["activation_scale"] != 0.03
        ):
            raise X21AutoResearchError(
                "structural transport is not the registered exact lift"
            )
    if (
        normalized_mutation["kind"] != kind
        or normalized_mutation["field"] != sequence["field"]
        or normalized_mutation["control_value"] != sequence["control_value"]
        or normalized_mutation["candidate_value"] != sequence["candidate_value"]
    ):
        raise X21AutoResearchError(
            "action changes a field outside its single-variable round"
        )
    candidate_metadata = _candidate_metadata(parent_metadata, kind)
    if "candidate_metadata" in value:
        supplied_candidate_metadata = _normalize_metadata(
            value["candidate_metadata"], context="action.candidate_metadata"
        )
        if supplied_candidate_metadata != candidate_metadata:
            raise X21AutoResearchError("action candidate metadata is inconsistent")
    observed_delta = (
        candidate_metadata["trainable_real_parameter_count"]
        - parent_metadata["trainable_real_parameter_count"]
    )
    if "new_real_parameters" in normalized_mutation and (
        normalized_mutation["new_real_parameters"] != observed_delta
    ):
        raise X21AutoResearchError("action parameter increment is inconsistent")
    normalized_resource_certificate = None
    normalized_resource_certificate_sha256 = None
    if resource_certificate_value is not None:
        normalized_resource_certificate = _normalize_resource_certificate(
            resource_certificate_value,
            protocol=protocol,
            round_index=round_index,
            kind=kind,
            candidate_metadata=candidate_metadata,
            required_outcome="feasible",
            selected_batch_size=batch_size,
        )
        normalized_resource_certificate_sha256 = _sha(
            resource_certificate_sha256,
            context="action.resource_certificate_sha256",
        )
        if normalized_resource_certificate_sha256 != digest_value(
            normalized_resource_certificate
        ):
            raise X21AutoResearchError(
                "resource certificate_sha256 is not bound to its certificate"
            )
    normalized = {
        "schema": ACTION_SCHEMA,
        "candidate_id": candidate_id,
        "round": round_index,
        "parent_family_sha256": parent_family["family_sha256"],
        "data_contract_sha256": protocol["data_contract_sha256"],
        "mutation": normalized_mutation,
        "budget": budget,
        "pipeline": list(PIPELINE),
        "candidate_metadata": candidate_metadata,
    }
    if normalized_resource_certificate is not None:
        normalized["resource_certificate"] = normalized_resource_certificate
        normalized["resource_certificate_sha256"] = (
            normalized_resource_certificate_sha256
        )
    action_sha256 = digest_value(normalized)
    if "action_sha256" in value and value["action_sha256"] != action_sha256:
        raise X21AutoResearchError("action_sha256 is inconsistent")
    return {**normalized, "action_sha256": action_sha256}


def normalize_resource_skip(
    value: Any,
    *,
    protocol: Mapping[str, Any],
    parent_family: Mapping[str, Any],
    expected_round: int,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise X21AutoResearchError("resource skip must be an object")
    _exact_keys(
        value,
        required={
            "schema",
            "round",
            "kind",
            "certificate_sha256",
            "certificate",
            "reason",
        },
        context="resource skip",
    )
    if (
        value["schema"] != RESOURCE_SKIP_SCHEMA
        or value["reason"] != "resource-infeasible"
    ):
        raise X21AutoResearchError("resource skip contract is invalid")
    round_index = _integer(value["round"], context="resource_skip.round", minimum=1)
    if round_index != expected_round or round_index > protocol["maximum_rounds"]:
        raise X21AutoResearchError("resource skip is not for the expected round")
    expected_kind = protocol["action_sequence"][round_index - 1]["kind"]
    if expected_kind not in {"bond_growth_d16", "site_growth_k24"}:
        raise X21AutoResearchError("only a structural round may be resource-skipped")
    if value["kind"] != expected_kind:
        raise X21AutoResearchError("resource skip kind changed")
    candidate_metadata = _candidate_metadata(
        parent_family["model_metadata"], expected_kind
    )
    certificate = _normalize_resource_certificate(
        value["certificate"],
        protocol=protocol,
        round_index=round_index,
        kind=expected_kind,
        candidate_metadata=candidate_metadata,
        required_outcome="resource-infeasible",
        selected_batch_size=None,
    )
    certificate_sha256 = _sha(
        value["certificate_sha256"],
        context="resource_skip.certificate_sha256",
    )
    if certificate_sha256 != digest_value(certificate):
        raise X21AutoResearchError(
            "resource skip certificate_sha256 is not bound to its certificate"
        )
    return {
        "schema": RESOURCE_SKIP_SCHEMA,
        "round": round_index,
        "kind": expected_kind,
        "certificate_sha256": certificate_sha256,
        "certificate": certificate,
        "reason": "resource-infeasible",
    }


def _normalize_metrics(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise X21AutoResearchError(f"{context} must be an object")
    _exact_keys(
        value,
        required={
            "sigma",
            "chi",
            "q999",
            "cvar_1pct",
            "minimum_metric_eigenvalue",
            "nonpositive_metric_count",
        },
        context=context,
    )
    return {
        "sigma": _number(value["sigma"], context=f"{context}.sigma", minimum=0.0),
        "chi": _number(value["chi"], context=f"{context}.chi", minimum=0.0),
        "q999": _number(value["q999"], context=f"{context}.q999", minimum=0.0),
        "cvar_1pct": _number(
            value["cvar_1pct"], context=f"{context}.cvar_1pct", minimum=0.0
        ),
        "minimum_metric_eigenvalue": _number(
            value["minimum_metric_eigenvalue"],
            context=f"{context}.minimum_metric_eigenvalue",
        ),
        "nonpositive_metric_count": _integer(
            value["nonpositive_metric_count"],
            context=f"{context}.nonpositive_metric_count",
        ),
    }


def _normalize_endpoint(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise X21AutoResearchError(f"{context} must be an object")
    _exact_keys(
        value,
        required={"checkpoint_sha256", "metadata", "metrics"},
        context=context,
    )
    return {
        "checkpoint_sha256": _sha(
            value["checkpoint_sha256"], context=f"{context}.checkpoint_sha256"
        ),
        "metadata": _normalize_metadata(
            value["metadata"], context=f"{context}.metadata"
        ),
        "metrics": _normalize_metrics(value["metrics"], context=f"{context}.metrics"),
    }


def _normalize_seed_evidence(
    value: Any,
    *,
    index: int,
    action: Mapping[str, Any],
    protocol: Mapping[str, Any],
    parent_family: Mapping[str, Any],
) -> dict[str, Any]:
    context = f"evidence.seeds[{index}]"
    if not isinstance(value, dict):
        raise X21AutoResearchError(f"{context} must be an object")
    _exact_keys(
        value,
        required={
            "seed",
            "parent_checkpoint_sha256",
            "data_contract_sha256",
            "control_budget",
            "candidate_budget",
            "control_batch_plan_sha256",
            "candidate_batch_plan_sha256",
            "equivalence",
            "control",
            "candidate",
            "paired",
            "source_report_sha256",
        },
        context=context,
    )
    seed = _integer(value["seed"], context=f"{context}.seed", minimum=1)
    parent_hash_by_seed = {
        row["seed"]: row["checkpoint_sha256"] for row in parent_family["models"]
    }
    if seed not in parent_hash_by_seed:
        raise X21AutoResearchError(f"{context}.seed is not registered")
    parent_hash = _sha(
        value["parent_checkpoint_sha256"], context=f"{context}.parent_checkpoint_sha256"
    )
    if parent_hash != parent_hash_by_seed[seed]:
        raise X21AutoResearchError(f"{context} uses another parent checkpoint")
    if value["data_contract_sha256"] != protocol["data_contract_sha256"]:
        raise X21AutoResearchError(f"{context} data contract changed")
    control_budget = _normalize_budget(
        value["control_budget"], context=f"{context}.control_budget"
    )
    candidate_budget = _normalize_budget(
        value["candidate_budget"], context=f"{context}.candidate_budget"
    )
    if control_budget != action["budget"] or candidate_budget != action["budget"]:
        raise X21AutoResearchError(f"{context} budgets do not match the action")
    control_plan = _sha(
        value["control_batch_plan_sha256"],
        context=f"{context}.control_batch_plan_sha256",
    )
    candidate_plan = _sha(
        value["candidate_batch_plan_sha256"],
        context=f"{context}.candidate_batch_plan_sha256",
    )
    equivalence = value["equivalence"]
    if not isinstance(equivalence, dict):
        raise X21AutoResearchError(f"{context}.equivalence must be an object")
    _exact_keys(
        equivalence,
        required={"passed", "potential_max_absolute", "metric_max_relative_frobenius"},
        context=f"{context}.equivalence",
    )
    paired = value["paired"]
    if not isinstance(paired, dict):
        raise X21AutoResearchError(f"{context}.paired must be an object")
    _exact_keys(
        paired,
        required={"sigma_ci95_low", "chi_ci95_low", "bootstrap_replicates"},
        context=f"{context}.paired",
    )
    source_reports = value["source_report_sha256"]
    if not isinstance(source_reports, dict):
        raise X21AutoResearchError(f"{context}.source_report_sha256 must be an object")
    _exact_keys(
        source_reports,
        required={"control_training", "candidate_training", "paired_evaluation"},
        context=f"{context}.source_report_sha256",
    )
    control = _normalize_endpoint(value["control"], context=f"{context}.control")
    candidate = _normalize_endpoint(value["candidate"], context=f"{context}.candidate")
    if control["metadata"] != parent_family["model_metadata"]:
        raise X21AutoResearchError(f"{context} control changed model structure")
    if candidate["metadata"] != action["candidate_metadata"]:
        raise X21AutoResearchError(f"{context} candidate changed an unregistered field")
    return {
        "seed": seed,
        "parent_checkpoint_sha256": parent_hash,
        "data_contract_sha256": protocol["data_contract_sha256"],
        "control_budget": control_budget,
        "candidate_budget": candidate_budget,
        "control_batch_plan_sha256": control_plan,
        "candidate_batch_plan_sha256": candidate_plan,
        "equivalence": {
            "passed": _boolean(
                equivalence["passed"], context=f"{context}.equivalence.passed"
            ),
            "potential_max_absolute": _number(
                equivalence["potential_max_absolute"],
                context=f"{context}.equivalence.potential_max_absolute",
                minimum=0.0,
            ),
            "metric_max_relative_frobenius": _number(
                equivalence["metric_max_relative_frobenius"],
                context=f"{context}.equivalence.metric_max_relative_frobenius",
                minimum=0.0,
            ),
        },
        "control": control,
        "candidate": candidate,
        "paired": {
            "sigma_ci95_low": _number(
                paired["sigma_ci95_low"], context=f"{context}.paired.sigma_ci95_low"
            ),
            "chi_ci95_low": _number(
                paired["chi_ci95_low"], context=f"{context}.paired.chi_ci95_low"
            ),
            "bootstrap_replicates": _integer(
                paired["bootstrap_replicates"],
                context=f"{context}.paired.bootstrap_replicates",
                minimum=1,
            ),
        },
        "source_report_sha256": {
            key: _sha(
                source_reports[key], context=f"{context}.source_report_sha256.{key}"
            )
            for key in ("control_training", "candidate_training", "paired_evaluation")
        },
    }


def validate_evidence(
    value: Any,
    *,
    action: Mapping[str, Any],
    protocol: Mapping[str, Any],
    parent_family: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise X21AutoResearchError("scientific evidence must be an object")
    if (
        protocol.get("schema") != PROTOCOL_V2_SCHEMA
        or not isinstance(protocol.get("development_pool_binding"), Mapping)
        or protocol.get("data_contract", {}).get("development_pool_sha256")
        == HISTORICAL_CONFIRMATION_SHA256
    ):
        raise X21AutoResearchError(
            "scientific evidence requires the v2 fresh development binding; "
            "the v1 historical confirmation is forbidden"
        )
    _reject_recovery_material(value, context="scientific evidence")
    _exact_keys(
        value,
        required={
            "schema",
            "candidate_id",
            "round",
            "action_sha256",
            "parent_family_sha256",
            "data_contract_sha256",
            "precision",
            "seeds",
        },
        context="scientific evidence",
    )
    expected = {
        "schema": EVIDENCE_SCHEMA,
        "candidate_id": action["candidate_id"],
        "round": action["round"],
        "action_sha256": action["action_sha256"],
        "parent_family_sha256": parent_family["family_sha256"],
        "data_contract_sha256": protocol["data_contract_sha256"],
        "precision": "complex64",
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise X21AutoResearchError(f"scientific evidence {key} is not locked")
    rows = value["seeds"]
    if not isinstance(rows, list) or len(rows) != 3:
        raise X21AutoResearchError("scientific evidence requires exactly three seeds")
    normalized_rows = [
        _normalize_seed_evidence(
            row,
            index=index,
            action=action,
            protocol=protocol,
            parent_family=parent_family,
        )
        for index, row in enumerate(rows)
    ]
    normalized_rows.sort(key=lambda row: row["seed"])
    if [row["seed"] for row in normalized_rows] != protocol["promotion_seeds"]:
        raise X21AutoResearchError("scientific evidence seed set changed")
    for endpoint in ("control", "candidate"):
        checkpoint_hashes = [
            row[endpoint]["checkpoint_sha256"] for row in normalized_rows
        ]
        if len(set(checkpoint_hashes)) != len(normalized_rows):
            raise X21AutoResearchError(
                f"{endpoint} checkpoint hashes must be distinct across seeds"
            )
    return {**expected, "seeds": normalized_rows}


def _relative_gain(control: float, candidate: float) -> float:
    if control <= 0.0:
        return 0.0 if candidate == control else -math.inf
    return (control - candidate) / control


def _relative_degradation(control: float, candidate: float) -> float:
    if control <= 0.0:
        return 0.0 if candidate == control else math.inf
    return (candidate - control) / control


def adjudicate(
    evidence: Mapping[str, Any],
    *,
    action: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    import statistics

    thresholds = protocol["thresholds"]
    sigma_gains: list[float] = []
    chi_gains: list[float] = []
    rows = []
    improved = 0
    paired_positive = 0
    all_positive = True
    all_tail_safe = True
    all_equivalent = True
    all_batch_plans_match = True
    all_bootstrap_counts_match = True
    for row in evidence["seeds"]:
        control = row["control"]["metrics"]
        candidate = row["candidate"]["metrics"]
        sigma_gain = _relative_gain(control["sigma"], candidate["sigma"])
        chi_gain = _relative_gain(control["chi"], candidate["chi"])
        q999_degradation = _relative_degradation(control["q999"], candidate["q999"])
        cvar_degradation = _relative_degradation(
            control["cvar_1pct"], candidate["cvar_1pct"]
        )
        positive = bool(
            control["minimum_metric_eigenvalue"] > 0.0
            and candidate["minimum_metric_eigenvalue"] > 0.0
            and control["nonpositive_metric_count"] == 0
            and candidate["nonpositive_metric_count"] == 0
        )
        tail_safe = bool(
            q999_degradation <= thresholds["maximum_tail_relative_degradation"]
            and cvar_degradation <= thresholds["maximum_tail_relative_degradation"]
        )
        eq = row["equivalence"]
        equivalent = bool(
            eq["passed"]
            and eq["potential_max_absolute"]
            <= thresholds["equivalence_potential_absolute_tolerance"]
            and eq["metric_max_relative_frobenius"]
            <= thresholds["equivalence_metric_relative_tolerance"]
        )
        paired_ok = bool(
            row["paired"]["sigma_ci95_low"] > 0.0
            and row["paired"]["chi_ci95_low"] > 0.0
        )
        if sigma_gain > 0.0 and chi_gain > 0.0:
            improved += 1
        if paired_ok:
            paired_positive += 1
        sigma_gains.append(sigma_gain)
        chi_gains.append(chi_gain)
        all_positive = all_positive and positive
        all_tail_safe = all_tail_safe and tail_safe
        all_equivalent = all_equivalent and equivalent
        all_batch_plans_match = all_batch_plans_match and (
            row["control_batch_plan_sha256"] == row["candidate_batch_plan_sha256"]
        )
        all_bootstrap_counts_match = all_bootstrap_counts_match and (
            row["paired"]["bootstrap_replicates"]
            == action["budget"]["bootstrap_replicates"]
        )
        rows.append(
            {
                "seed": row["seed"],
                "relative_sigma_gain": sigma_gain,
                "relative_chi_gain": chi_gain,
                "relative_q999_degradation": q999_degradation,
                "relative_cvar_1pct_degradation": cvar_degradation,
                "positivity_passes": positive,
                "tail_passes": tail_safe,
                "equivalence_passes": equivalent,
                "paired_ci_passes": paired_ok,
                "control_checkpoint_sha256": row["control"]["checkpoint_sha256"],
                "candidate_checkpoint_sha256": row["candidate"]["checkpoint_sha256"],
                "source_report_sha256": row["source_report_sha256"],
            }
        )
    median_sigma = statistics.median(sigma_gains)
    median_chi = statistics.median(chi_gains)
    gates = {
        "three_registered_seeds": len(rows) == 3,
        "fixed_data_contract": evidence["data_contract_sha256"]
        == protocol["data_contract_sha256"],
        "complex64_search": evidence["precision"] == "complex64",
        "matched_fixed_budget": all(
            row["control_budget"] == action["budget"]
            and row["candidate_budget"] == action["budget"]
            for row in evidence["seeds"]
        ),
        "matched_batch_plans": all_batch_plans_match,
        "bootstrap_contract": all_bootstrap_counts_match,
        "exact_epoch_zero_equivalence": all_equivalent,
        "positivity": all_positive,
        "tail_safety": all_tail_safe,
        "minimum_improved_seeds": improved >= thresholds["minimum_improved_seeds"],
        "minimum_paired_positive_seeds": paired_positive
        >= thresholds["minimum_paired_positive_seeds"],
        "median_sigma_gain": median_sigma
        >= thresholds["minimum_median_relative_sigma_gain"],
        "median_chi_gain": median_chi >= thresholds["minimum_median_relative_chi_gain"],
        "worst_seed_sigma": min(sigma_gains)
        >= -thresholds["maximum_seed_relative_regression"],
        "worst_seed_chi": min(chi_gains)
        >= -thresholds["maximum_seed_relative_regression"],
    }
    family_core = {
        "schema": FAMILY_SCHEMA,
        "model_metadata": action["candidate_metadata"],
        "models": [
            {
                "seed": row["seed"],
                "checkpoint_sha256": row["candidate"]["checkpoint_sha256"],
                "provenance": {
                    "kind": "scientific-action",
                    "certificate_sha256": action["action_sha256"],
                },
            }
            for row in evidence["seeds"]
        ],
    }
    candidate_family = normalize_family(
        family_core,
        seeds=protocol["promotion_seeds"],
        context="candidate_family",
        expected_metadata=action["candidate_metadata"],
    )
    return {
        "schema": ADJUDICATION_SCHEMA,
        "round": action["round"],
        "candidate_id": action["candidate_id"],
        "action_kind": action["mutation"]["kind"],
        "action_sha256": action["action_sha256"],
        "promotion_passes": all(gates.values()),
        "gates": gates,
        "median_relative_sigma_gain": median_sigma,
        "median_relative_chi_gain": median_chi,
        "minimum_relative_sigma_gain": min(sigma_gains),
        "minimum_relative_chi_gain": min(chi_gains),
        "improved_seed_count": improved,
        "paired_positive_seed_count": paired_positive,
        "seed_rows": rows,
        "candidate_family": candidate_family,
    }


def validate_baseline_completion_evidence(value: Any) -> dict[str, Any]:
    """Validate a non-recovery baseline endpoint certificate."""

    if not isinstance(value, dict):
        raise X21AutoResearchError("baseline completion evidence must be an object")
    _exact_keys(
        value,
        required={
            "schema",
            "seed",
            "source_campaign_id",
            "source_plan_sha256",
            "source_job_id",
            "source_job_digest",
            "terminal_status",
            "native_exit_code",
            "output_model_sha256",
            "output_checkpoint_sha256",
            "output_summary_sha256",
        },
        context="baseline completion evidence",
    )
    if value["schema"] != BASELINE_COMPLETION_SCHEMA:
        raise X21AutoResearchError(
            f"baseline completion schema must be {BASELINE_COMPLETION_SCHEMA}"
        )
    source_campaign_id = str(value["source_campaign_id"])
    source_job_id = str(value["source_job_id"])
    if not _IDENTIFIER.fullmatch(source_campaign_id) or not _IDENTIFIER.fullmatch(
        source_job_id
    ):
        raise X21AutoResearchError("baseline completion source identity is invalid")
    terminal_status = str(value["terminal_status"])
    if terminal_status != "validation_plateau":
        raise X21AutoResearchError("baseline completion did not reach a plateau")
    result = {
        "schema": BASELINE_COMPLETION_SCHEMA,
        "seed": _integer(value["seed"], context="completion.seed", minimum=1),
        "source_campaign_id": source_campaign_id,
        "source_plan_sha256": _sha(
            value["source_plan_sha256"], context="completion.source_plan_sha256"
        ),
        "source_job_id": source_job_id,
        "source_job_digest": _sha(
            value["source_job_digest"], context="completion.source_job_digest"
        ),
        "terminal_status": terminal_status,
        "native_exit_code": _integer(
            value["native_exit_code"], context="completion.native_exit_code"
        ),
        "output_model_sha256": _sha(
            value["output_model_sha256"], context="completion.output_model_sha256"
        ),
        "output_checkpoint_sha256": _sha(
            value["output_checkpoint_sha256"],
            context="completion.output_checkpoint_sha256",
        ),
        "output_summary_sha256": _sha(
            value["output_summary_sha256"],
            context="completion.output_summary_sha256",
        ),
    }
    if result["native_exit_code"] != 0:
        raise X21AutoResearchError("baseline completion did not exit cleanly")
    return result


def validate_recovery_evidence(value: Any) -> dict[str, Any]:
    """Validate operational evidence; metric fields are deliberately impossible."""

    if not isinstance(value, dict):
        raise X21AutoResearchError("recovery evidence must be an object")
    _exact_keys(
        value,
        required={
            "schema",
            "campaign_id",
            "seed",
            "mode",
            "source_campaign_id",
            "source_plan_sha256",
            "source_job_id",
            "source_job_digest",
            "source_model_sha256",
            "source_checkpoint_sha256",
            "source_epoch",
            "source_next_epoch",
            "training_semantics_sha256",
            "frozen_inputs_sha256",
            "implementation_match",
            "optimizer_state_restored",
            "rng_state_restored",
            "terminal_status",
            "native_exit_code",
            "operational_attempt_count",
            "output_model_sha256",
            "output_checkpoint_sha256",
            "output_summary_sha256",
        },
        context="recovery evidence",
    )
    if value["schema"] != RECOVERY_EVIDENCE_SCHEMA:
        raise X21AutoResearchError(
            f"recovery evidence schema must be {RECOVERY_EVIDENCE_SCHEMA}"
        )
    campaign_id = str(value["campaign_id"])
    if not _IDENTIFIER.fullmatch(campaign_id):
        raise X21AutoResearchError("recovery campaign_id has invalid characters")
    if value["mode"] != "exact-checkpoint-resume":
        raise X21AutoResearchError("only exact checkpoint recovery is registered")
    if value["terminal_status"] != "validation_plateau":
        raise X21AutoResearchError("recovery did not reach a terminal plateau")
    source_campaign_id = str(value["source_campaign_id"])
    source_job_id = str(value["source_job_id"])
    if not _IDENTIFIER.fullmatch(source_campaign_id) or not _IDENTIFIER.fullmatch(
        source_job_id
    ):
        raise X21AutoResearchError("recovery source identity is invalid")
    result = {
        "schema": RECOVERY_EVIDENCE_SCHEMA,
        "campaign_id": campaign_id,
        "seed": _integer(value["seed"], context="recovery.seed", minimum=1),
        "mode": "exact-checkpoint-resume",
        "source_campaign_id": source_campaign_id,
        "source_plan_sha256": _sha(
            value["source_plan_sha256"], context="recovery.source_plan_sha256"
        ),
        "source_job_id": source_job_id,
        "source_job_digest": _sha(
            value["source_job_digest"], context="recovery.source_job_digest"
        ),
        "source_model_sha256": _sha(
            value["source_model_sha256"], context="recovery.source_model_sha256"
        ),
        "source_checkpoint_sha256": _sha(
            value["source_checkpoint_sha256"],
            context="recovery.source_checkpoint_sha256",
        ),
        "source_epoch": _integer(
            value["source_epoch"], context="recovery.source_epoch"
        ),
        "source_next_epoch": _integer(
            value["source_next_epoch"], context="recovery.source_next_epoch", minimum=1
        ),
        "training_semantics_sha256": _sha(
            value["training_semantics_sha256"],
            context="recovery.training_semantics_sha256",
        ),
        "frozen_inputs_sha256": _sha(
            value["frozen_inputs_sha256"], context="recovery.frozen_inputs_sha256"
        ),
        "implementation_match": _boolean(
            value["implementation_match"], context="recovery.implementation_match"
        ),
        "optimizer_state_restored": _boolean(
            value["optimizer_state_restored"],
            context="recovery.optimizer_state_restored",
        ),
        "rng_state_restored": _boolean(
            value["rng_state_restored"], context="recovery.rng_state_restored"
        ),
        "terminal_status": "validation_plateau",
        "native_exit_code": _integer(
            value["native_exit_code"], context="recovery.native_exit_code"
        ),
        "operational_attempt_count": _integer(
            value["operational_attempt_count"],
            context="recovery.operational_attempt_count",
            minimum=1,
        ),
        "output_model_sha256": _sha(
            value["output_model_sha256"], context="recovery.output_model_sha256"
        ),
        "output_checkpoint_sha256": _sha(
            value["output_checkpoint_sha256"],
            context="recovery.output_checkpoint_sha256",
        ),
        "output_summary_sha256": _sha(
            value["output_summary_sha256"], context="recovery.output_summary_sha256"
        ),
    }
    if result["source_next_epoch"] != result["source_epoch"] + 1:
        raise X21AutoResearchError("recovery epoch boundary is inconsistent")
    if not all(
        result[key]
        for key in (
            "implementation_match",
            "optimizer_state_restored",
            "rng_state_restored",
        )
    ):
        raise X21AutoResearchError("recovery did not restore exact training state")
    if result["native_exit_code"] != 0:
        raise X21AutoResearchError("recovery terminal process did not exit cleanly")
    return result


def _event(
    revision: int, name: str, previous: str, details: Any = None
) -> dict[str, Any]:
    core = {
        "revision": revision,
        "event": name,
        "previous_event_sha256": previous,
        "utc": utc_now(),
        "details": details,
    }
    return {**core, "event_sha256": digest_value(core)}


def _seal(ledger: dict[str, Any]) -> None:
    ledger["integrity_sha256"] = digest_value(
        {key: value for key, value in ledger.items() if key != "integrity_sha256"}
    )


def _verify_ledger(ledger: Mapping[str, Any], *, schema: str) -> None:
    if ledger.get("schema") != schema:
        raise X21AutoResearchError("ledger schema is invalid")
    observed = ledger.get("integrity_sha256")
    expected = digest_value(
        {key: value for key, value in ledger.items() if key != "integrity_sha256"}
    )
    if observed != expected:
        raise X21AutoResearchError("ledger integrity hash is invalid")
    events = ledger.get("events")
    if not isinstance(events, list) or not events:
        raise X21AutoResearchError("ledger event chain is missing")
    previous = "0" * 64
    for index, row in enumerate(events):
        if not isinstance(row, dict) or row.get("revision") != index:
            raise X21AutoResearchError("ledger event revisions are invalid")
        if row.get("previous_event_sha256") != previous:
            raise X21AutoResearchError("ledger event chain is broken")
        core = {key: value for key, value in row.items() if key != "event_sha256"}
        if row.get("event_sha256") != digest_value(core):
            raise X21AutoResearchError("ledger event hash is invalid")
        previous = row["event_sha256"]
    if ledger.get("revision") != len(events) - 1:
        raise X21AutoResearchError("ledger revision does not match its event chain")


class RecoveryStore:
    """Independent, non-scientific ledger for exact checkpoint recovery."""

    def __init__(self, run_root: Path) -> None:
        self.run_root = run_root.expanduser().resolve(strict=False)
        self.ledger_path = self.run_root / "recovery_ledger.json"
        self.lock_path = self.run_root / ".recovery.lock"

    @classmethod
    def initialize(cls, run_root: Path, campaign_id: str) -> "RecoveryStore":
        store = cls(run_root)
        if not _IDENTIFIER.fullmatch(campaign_id):
            raise X21AutoResearchError("recovery campaign_id has invalid characters")
        if store.run_root.exists() and any(store.run_root.iterdir()):
            if not (store.run_root / RECOVERY_ROOT_SENTINEL).is_file():
                raise X21AutoResearchError(
                    "refusing a non-empty unmanaged recovery root"
                )
            ledger = store.status()
            if ledger["campaign_id"] != campaign_id:
                raise X21AutoResearchError("recovery root belongs to another campaign")
            return store
        store.run_root.mkdir(parents=True, exist_ok=True)
        ledger = {
            "schema": RECOVERY_LEDGER_SCHEMA,
            "campaign_id": campaign_id,
            "revision": 0,
            "certificates": {},
            "events": [_event(0, "recovery_initialized", "0" * 64)],
        }
        _seal(ledger)
        _atomic_write_json(store.ledger_path, ledger)
        (store.run_root / RECOVERY_ROOT_SENTINEL).write_text(
            f"{campaign_id}\n", encoding="utf-8"
        )
        return store

    def _ledger(self) -> dict[str, Any]:
        ledger = _read_json(self.ledger_path)
        _verify_ledger(ledger, schema=RECOVERY_LEDGER_SCHEMA)
        return ledger

    @contextmanager
    def _locked(self) -> Iterator[dict[str, Any]]:
        if not (self.run_root / RECOVERY_ROOT_SENTINEL).is_file():
            raise X21AutoResearchError("recovery root sentinel is missing")
        handle = self.lock_path.open("a+", encoding="utf-8")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            ledger = self._ledger()
            for seed, record in ledger["certificates"].items():
                path = Path(record["path"])
                evidence = validate_recovery_evidence(_read_json(path))
                if (
                    str(evidence["seed"]) != seed
                    or digest_value(evidence) != record["evidence_sha256"]
                    or sha256_file(path) != record["file_sha256"]
                ):
                    raise X21AutoResearchError("recovery certificate was modified")
            yield ledger
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

    def _commit(self, ledger: dict[str, Any], name: str, details: Any = None) -> None:
        ledger["revision"] += 1
        ledger["events"].append(
            _event(
                ledger["revision"],
                name,
                ledger["events"][-1]["event_sha256"],
                details,
            )
        )
        _seal(ledger)
        _atomic_write_json(self.ledger_path, ledger)

    def record(self, value: Any) -> dict[str, Any]:
        evidence = validate_recovery_evidence(value)
        with self._locked() as ledger:
            if evidence["campaign_id"] != ledger["campaign_id"]:
                raise X21AutoResearchError("recovery evidence campaign changed")
            key = str(evidence["seed"])
            path = self.run_root / "certificates" / f"seed_{key}.json"
            file_sha256 = _publish_create_only(
                path, evidence, context="recovery certificate"
            )
            record = {
                "path": str(path),
                "evidence_sha256": digest_value(evidence),
                "file_sha256": file_sha256,
            }
            existing = ledger["certificates"].get(key)
            if existing is not None:
                if existing != record:
                    raise X21AutoResearchError(
                        "recovery seed is already certified differently"
                    )
                return evidence
            ledger["certificates"][key] = record
            self._commit(ledger, "recovery_certified", {"seed": evidence["seed"]})
            return evidence

    def status(self) -> dict[str, Any]:
        with self._locked() as ledger:
            return copy.deepcopy(ledger)


class X21CampaignStore:
    """Durable scientific ledger with a fixed four-action search language."""

    def __init__(self, run_root: Path) -> None:
        self.run_root = run_root.expanduser().resolve(strict=False)
        self.protocol_path = self.run_root / "protocol.lock.json"
        self.ledger_path = self.run_root / "ledger.json"
        self.baseline_path = self.run_root / "baseline_family.json"
        self.lock_path = self.run_root / ".ledger.lock"

    @classmethod
    def initialize(cls, run_root: Path, protocol_value: Any) -> "X21CampaignStore":
        store = cls(run_root)
        protocol = validate_protocol(protocol_value)
        protocol_sha256 = digest_value(protocol)
        if store.run_root.exists() and any(store.run_root.iterdir()):
            if not (store.run_root / ROOT_SENTINEL).is_file():
                raise X21AutoResearchError(
                    "refusing a non-empty unmanaged scientific root"
                )
            if digest_value(_read_json(store.protocol_path)) != protocol_sha256:
                raise X21AutoResearchError(
                    "scientific root belongs to another protocol"
                )
            store.status()
            return store
        store.run_root.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(store.protocol_path, protocol)
        ledger = {
            "schema": LEDGER_SCHEMA,
            "campaign_id": protocol["campaign_id"],
            "protocol_sha256": protocol_sha256,
            "data_contract_sha256": protocol["data_contract_sha256"],
            "status": "awaiting-baseline",
            "revision": 0,
            "current_round": 0,
            "baseline_family": None,
            "champion_family": None,
            "champion_candidate_id": None,
            "promotion_count": 0,
            "rounds": {},
            "resource_skips": {},
            "leaderboard": [],
            "events": [_event(0, "scientific_initialized", "0" * 64)],
        }
        _seal(ledger)
        _atomic_write_json(store.ledger_path, ledger)
        (store.run_root / ROOT_SENTINEL).write_text(
            f"{protocol['campaign_id']}\n", encoding="utf-8"
        )
        return store

    def _ledger(self) -> dict[str, Any]:
        ledger = _read_json(self.ledger_path)
        _verify_ledger(ledger, schema=LEDGER_SCHEMA)
        return ledger

    def _protocol(self, ledger: Mapping[str, Any]) -> dict[str, Any]:
        protocol = _read_json(self.protocol_path)
        if digest_value(protocol) != ledger["protocol_sha256"]:
            raise X21AutoResearchError("locked protocol was modified")
        return protocol

    @contextmanager
    def _locked(self) -> Iterator[dict[str, Any]]:
        if not (self.run_root / ROOT_SENTINEL).is_file():
            raise X21AutoResearchError("scientific root sentinel is missing")
        handle = self.lock_path.open("a+", encoding="utf-8")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            ledger = self._ledger()
            protocol = self._protocol(ledger)
            self._audit_artifacts(ledger, protocol)
            yield ledger
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

    def _audit_artifacts(
        self, ledger: Mapping[str, Any], protocol: Mapping[str, Any]
    ) -> None:
        if ledger["baseline_family"] is not None:
            baseline = normalize_family(
                _read_json(self.baseline_path),
                seeds=protocol["promotion_seeds"],
                context="baseline_family",
                expected_metadata=protocol["expected_baseline_metadata"],
                recovery_seed=protocol["recovery_seed"],
            )
            if baseline != ledger["baseline_family"]:
                raise X21AutoResearchError("baseline family artifact was modified")
        parent_family = ledger["baseline_family"]
        for round_index in range(1, int(ledger["current_round"]) + 2):
            row = ledger["rounds"].get(str(round_index))
            if row is None:
                continue
            if parent_family is None:
                raise X21AutoResearchError("scientific round lacks a parent family")
            if row["status"] == "resource-skipped":
                skip_path = Path(row["skip_path"])
                skip = normalize_resource_skip(
                    _read_json(skip_path),
                    protocol=protocol,
                    parent_family=parent_family,
                    expected_round=round_index,
                )
                if (
                    digest_value(skip) != row["skip_sha256"]
                    or row["certificate_sha256"] != skip["certificate_sha256"]
                ):
                    raise X21AutoResearchError("resource skip artifact was modified")
                continue
            action_path = Path(row["action_path"])
            action = validate_action(
                _read_json(action_path), protocol=protocol, parent_family=parent_family
            )
            if action["action_sha256"] != row["action_sha256"]:
                raise X21AutoResearchError("action artifact was modified")
            evidence = None
            if row["evidence_sha256"] is not None:
                evidence = validate_evidence(
                    _read_json(Path(row["evidence_path"])),
                    action=action,
                    protocol=protocol,
                    parent_family=parent_family,
                )
                if digest_value(evidence) != row["evidence_sha256"]:
                    raise X21AutoResearchError("scientific evidence was modified")
            if row["status"] == "complete":
                if evidence is None:
                    raise X21AutoResearchError("completed round lacks evidence")
                report = _read_json(Path(row["adjudication_path"]))
                expected = adjudicate(evidence, action=action, protocol=protocol)
                if (
                    report != expected
                    or digest_value(report) != row["adjudication_sha256"]
                ):
                    raise X21AutoResearchError(
                        "adjudication artifact is not reproducible"
                    )
                if report["promotion_passes"]:
                    parent_family = report["candidate_family"]

    def _commit(self, ledger: dict[str, Any], name: str, details: Any = None) -> None:
        ledger["revision"] += 1
        ledger["events"].append(
            _event(
                ledger["revision"],
                name,
                ledger["events"][-1]["event_sha256"],
                details,
            )
        )
        _seal(ledger)
        _atomic_write_json(self.ledger_path, ledger)

    def register_baseline(self, value: Any) -> dict[str, Any]:
        with self._locked() as ledger:
            protocol = self._protocol(ledger)
            family = normalize_family(
                value,
                seeds=protocol["promotion_seeds"],
                context="baseline_family",
                expected_metadata=protocol["expected_baseline_metadata"],
                recovery_seed=protocol["recovery_seed"],
                require_recovery_certificate=True,
                baseline_provenance_contracts=protocol["baseline_provenance_contracts"],
            )
            if ledger["baseline_family"] is not None:
                if ledger["baseline_family"] != family:
                    raise X21AutoResearchError(
                        "baseline family is already locked differently"
                    )
                return family
            if ledger["status"] != "awaiting-baseline":
                raise X21AutoResearchError("baseline cannot be registered now")
            _publish_create_only(self.baseline_path, family, context="baseline family")
            ledger["baseline_family"] = family
            ledger["champion_family"] = family
            ledger["champion_candidate_id"] = "baseline"
            ledger["status"] = "active"
            self._commit(
                ledger,
                "baseline_registered",
                {"family_sha256": family["family_sha256"]},
            )
            return family

    def register_action(self, value: Any) -> dict[str, Any]:
        with self._locked() as ledger:
            if ledger["status"] != "active":
                raise X21AutoResearchError("scientific campaign is not active")
            protocol = self._protocol(ledger)
            expected_round = int(ledger["current_round"]) + 1
            if value.get("round") != expected_round:
                raise X21AutoResearchError(f"next open round is {expected_round}")
            action = validate_action(
                value, protocol=protocol, parent_family=ledger["champion_family"]
            )
            key = str(expected_round)
            existing = ledger["rounds"].get(key)
            action_path = (
                self.run_root / "rounds" / f"round_{expected_round:03d}" / "action.json"
            )
            if existing is not None:
                if (
                    existing["action_sha256"] != action["action_sha256"]
                    or _read_json(action_path) != action
                ):
                    raise X21AutoResearchError(
                        "round action is already registered differently"
                    )
                return action
            _publish_create_only(action_path, action, context="scientific action")
            ledger["rounds"][key] = {
                "status": "registered",
                "parent_family_sha256": ledger["champion_family"]["family_sha256"],
                "action_path": str(action_path),
                "action_sha256": action["action_sha256"],
                "evidence_path": None,
                "evidence_sha256": None,
                "adjudication_path": None,
                "adjudication_sha256": None,
            }
            self._commit(
                ledger,
                "action_registered",
                {"round": expected_round, "candidate_id": action["candidate_id"]},
            )
            return action

    def record_evidence(self, value: Any) -> dict[str, Any]:
        with self._locked() as ledger:
            if ledger["status"] != "active":
                raise X21AutoResearchError("scientific campaign is not active")
            round_index = value.get("round") if isinstance(value, dict) else None
            row = ledger["rounds"].get(str(round_index))
            if row is None or row["status"] not in {"registered", "evidence-recorded"}:
                raise X21AutoResearchError("scientific action is not open")
            protocol = self._protocol(ledger)
            action = validate_action(
                _read_json(Path(row["action_path"])),
                protocol=protocol,
                parent_family=ledger["champion_family"],
            )
            evidence = validate_evidence(
                value,
                action=action,
                protocol=protocol,
                parent_family=ledger["champion_family"],
            )
            evidence_path = Path(row["action_path"]).with_name("evidence.json")
            _publish_create_only(evidence_path, evidence, context="scientific evidence")
            evidence_sha256 = digest_value(evidence)
            if row["evidence_sha256"] is not None:
                if row["evidence_sha256"] != evidence_sha256:
                    raise X21AutoResearchError(
                        "scientific evidence is already locked differently"
                    )
                return evidence
            row["evidence_path"] = str(evidence_path)
            row["evidence_sha256"] = evidence_sha256
            row["status"] = "evidence-recorded"
            self._commit(ledger, "scientific_evidence_recorded", {"round": round_index})
            return evidence

    def adjudicate_round(self, round_index: int) -> dict[str, Any]:
        with self._locked() as ledger:
            row = ledger["rounds"].get(str(round_index))
            if row is None:
                raise X21AutoResearchError("round is not registered")
            if row["status"] == "complete":
                return _read_json(Path(row["adjudication_path"]))
            if row["status"] != "evidence-recorded":
                raise X21AutoResearchError("round lacks scientific evidence")
            if round_index != int(ledger["current_round"]) + 1:
                raise X21AutoResearchError("rounds must be adjudicated in order")
            protocol = self._protocol(ledger)
            parent_family = ledger["champion_family"]
            action = validate_action(
                _read_json(Path(row["action_path"])),
                protocol=protocol,
                parent_family=parent_family,
            )
            evidence = validate_evidence(
                _read_json(Path(row["evidence_path"])),
                action=action,
                protocol=protocol,
                parent_family=parent_family,
            )
            report = adjudicate(evidence, action=action, protocol=protocol)
            report_path = Path(row["action_path"]).with_name("adjudication.json")
            _publish_create_only(report_path, report, context="round adjudication")
            row["status"] = "complete"
            row["adjudication_path"] = str(report_path)
            row["adjudication_sha256"] = digest_value(report)
            ledger["current_round"] = round_index
            ledger["leaderboard"].append(
                {
                    "round": round_index,
                    "candidate_id": action["candidate_id"],
                    "action_kind": action["mutation"]["kind"],
                    "action_sha256": action["action_sha256"],
                    "promotion_passes": report["promotion_passes"],
                    "median_relative_sigma_gain": report["median_relative_sigma_gain"],
                    "median_relative_chi_gain": report["median_relative_chi_gain"],
                }
            )
            if report["promotion_passes"]:
                ledger["champion_family"] = report["candidate_family"]
                ledger["champion_candidate_id"] = action["candidate_id"]
                ledger["promotion_count"] += 1
            if round_index == protocol["maximum_rounds"]:
                ledger["status"] = "complete"
            self._commit(
                ledger,
                "round_adjudicated",
                {"round": round_index, "promotion_passes": report["promotion_passes"]},
            )
            return report

    def skip_resource(self, value: Any) -> dict[str, Any]:
        """Advance an infeasible structural round without creating leaderboard data."""
        with self._locked() as ledger:
            if ledger["status"] != "active":
                raise X21AutoResearchError("scientific campaign is not active")
            protocol = self._protocol(ledger)
            round_index = int(ledger["current_round"]) + 1
            normalized = normalize_resource_skip(
                value,
                protocol=protocol,
                parent_family=ledger["champion_family"],
                expected_round=round_index,
            )
            expected_kind = normalized["kind"]
            key = str(round_index)
            if key in ledger["rounds"]:
                raise X21AutoResearchError("cannot skip a registered action")
            path = (
                self.run_root
                / "rounds"
                / f"round_{round_index:03d}"
                / "resource_skip.json"
            )
            _publish_create_only(path, normalized, context="resource skip")
            skip_sha256 = digest_value(normalized)
            ledger["rounds"][key] = {
                "status": "resource-skipped",
                "skip_path": str(path),
                "skip_sha256": skip_sha256,
                "certificate_sha256": normalized["certificate_sha256"],
            }
            ledger["resource_skips"][key] = {
                "kind": expected_kind,
                "certificate_sha256": normalized["certificate_sha256"],
            }
            ledger["current_round"] = round_index
            if round_index == protocol["maximum_rounds"]:
                ledger["status"] = "complete"
            self._commit(ledger, "resource_round_skipped", {"round": round_index})
            return normalized

    def status(self) -> dict[str, Any]:
        with self._locked() as ledger:
            return copy.deepcopy(ledger)


__all__ = [
    "ACTION_SCHEMA",
    "ADJUDICATION_SCHEMA",
    "BASELINE_COMPLETION_SCHEMA",
    "EVIDENCE_SCHEMA",
    "FAMILY_SCHEMA",
    "FRESH_DEVELOPMENT_BINDING_SCHEMA",
    "HISTORICAL_CONFIRMATION_SHA256",
    "PROTOCOL_SCHEMA",
    "PROTOCOL_V2_SCHEMA",
    "RECOVERY_EVIDENCE_SCHEMA",
    "RESOURCE_CERTIFICATE_SCHEMA",
    "RESOURCE_SKIP_SCHEMA",
    "RecoveryStore",
    "X21AutoResearchError",
    "X21CampaignStore",
    "adjudicate",
    "coefficient_real_parameters",
    "digest_value",
    "expected_real_parameters",
    "normalize_family",
    "normalize_resource_skip",
    "validate_action",
    "validate_baseline_completion_evidence",
    "validate_evidence",
    "validate_protocol",
    "validate_recovery_evidence",
]
