"""Constrained, auditable control plane for architecture auto-research.

This module deliberately does not execute model code or arbitrary commands.
It accepts only a small, typed action language whose numerical adapters are
the already-audited quintic topology/rank/root-residual trainers.  Search data
indices are materialized once, every candidate is compared with a shared
no-growth control under an identical relaxation budget, and promotion requires
three paired seeds.  A finalist must be frozen before a zero-update complex128
replay and a single shadow evaluation can be recorded.

There is intentionally no blind-evaluation entry point.  A final blind study
belongs in a separate, post-freeze experiment manifest and run root.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
from typing import Any, Iterator, Mapping, Sequence


PROTOCOL_SCHEMA = "gcicy-quintic-architecture-auto-research-protocol-v1"
ACTION_SCHEMA = "gcicy-quintic-architecture-action-v1"
SEARCH_EVIDENCE_SCHEMA = "gcicy-quintic-architecture-search-evidence-v1"
ROUND_ADJUDICATION_SCHEMA = "gcicy-quintic-architecture-round-adjudication-v1"
REPLAY_SCHEMA = "gcicy-quintic-architecture-c128-replay-v1"
SHADOW_EVIDENCE_SCHEMA = "gcicy-quintic-architecture-shadow-evidence-v1"
SHADOW_CLAIM_SCHEMA = "gcicy-quintic-architecture-shadow-claim-v1"
SHADOW_ADJUDICATION_SCHEMA = "gcicy-quintic-architecture-shadow-adjudication-v1"
LEDGER_SCHEMA = "gcicy-quintic-architecture-auto-research-ledger-v1"
FROZEN_CANDIDATE_SCHEMA = "gcicy-quintic-frozen-architecture-candidate-v1"
INDEX_SCHEMA = "gcicy-quintic-fixed-search-indices-v1"
ROOT_SENTINEL = ".gcicy-architecture-auto-research-root"

SEARCH_PRECISION = "complex64"
FINALIST_REPLAY_PRECISION = "complex128"
REQUIRED_PROMOTION_SEEDS = 3
MAX_LOCAL_NEW_REAL_PARAMETERS = 10_000
ROOT_RESIDUAL_RANKS = frozenset({2, 4, 8, 16, 25})
EXACT_TOPOLOGY_TRANSPORT = "five-leaf-4plus1-to-2plus3"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_FORBIDDEN_BLIND_TOKEN = re.compile(r"blind", re.IGNORECASE)
_FORBIDDEN_SEARCH_TOKEN = re.compile(r"(?:blind|shadow)", re.IGNORECASE)

_BUDGET_KEYS = {
    "optimizer_updates",
    "batch_size",
    "learning_rate",
    "gradient_clip_norm",
    "train_examples",
    "evaluation_examples",
    "eval_every",
    "scheduler",
}
_METRIC_KEYS = {
    "sigma",
    "chi",
    "q999",
    "cvar_1pct",
    "minimum_metric_eigenvalue",
    "nonpositive_metric_count",
}


class AutoResearchError(RuntimeError):
    """Raised when the registered auto-research contract is violated."""


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


def _event_row(
    *,
    revision: int,
    event: str,
    previous_event_sha256: str,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "revision": revision,
        "event": event,
        "utc": utc_now(),
        "previous_event_sha256": previous_event_sha256,
    }
    if details:
        payload.update(copy.deepcopy(dict(details)))
    return {**payload, "event_sha256": digest_value(payload)}


def _seal_ledger(ledger: dict[str, Any]) -> None:
    payload = {key: value for key, value in ledger.items() if key != "state_sha256"}
    ledger["state_sha256"] = digest_value(payload)


def _validate_ledger_integrity(ledger: Mapping[str, Any]) -> None:
    expected_state = ledger.get("state_sha256")
    payload = {key: value for key, value in ledger.items() if key != "state_sha256"}
    if not isinstance(expected_state, str) or digest_value(payload) != expected_state:
        raise AutoResearchError("ledger state hash is invalid")
    events = ledger.get("events")
    revision = ledger.get("revision")
    if not isinstance(events, list) or not isinstance(revision, int):
        raise AutoResearchError("ledger event chain is malformed")
    if len(events) != revision + 1:
        raise AutoResearchError("ledger revision and event count disagree")
    previous = "0" * 64
    for expected_revision, row in enumerate(events):
        if not isinstance(row, dict):
            raise AutoResearchError("ledger event must be an object")
        observed_hash = row.get("event_sha256")
        event_payload = {
            key: value for key, value in row.items() if key != "event_sha256"
        }
        if (
            row.get("revision") != expected_revision
            or row.get("previous_event_sha256") != previous
            or not isinstance(observed_hash, str)
            or digest_value(event_payload) != observed_hash
        ):
            raise AutoResearchError("ledger event hash chain is invalid")
        previous = observed_hash


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AutoResearchError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise AutoResearchError(f"JSON root must be an object: {path}")
    return value


def _publish_create_only_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    """Publish an immutable JSON artifact, or verify an identical retry.

    Ledger transitions intentionally happen after their referenced artifacts
    reach durable storage.  A process may therefore die with the artifact
    present but the ledger still at the preceding revision.  Exact retries
    recover that window; a different payload fails closed instead of
    overwriting scientific evidence.
    """

    normalized = copy.deepcopy(dict(value))
    if path.exists():
        observed = _read_json(path)
        if observed != normalized:
            raise AutoResearchError(f"{context} already contains another payload")
        return observed
    atomic_write_json(path, normalized)
    return normalized


def _read_json_with_digest(
    path: Path,
    *,
    expected_sha256: str,
    context: str,
) -> dict[str, Any]:
    value = _read_json(path)
    if digest_value(value) != expected_sha256:
        raise AutoResearchError(f"{context} hash does not match the durable ledger")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] = frozenset(),
    context: str,
) -> None:
    missing = required - set(value)
    extra = set(value) - required - optional
    if missing or extra:
        raise AutoResearchError(
            f"{context} keys are invalid; missing={sorted(missing)} "
            f"extra={sorted(extra)}"
        )


def _normalized_sha256(value: Any, *, context: str) -> str:
    result = str(value).lower()
    if not _SHA256.fullmatch(result):
        raise AutoResearchError(f"{context} must be a lowercase SHA-256")
    return result


def _finite_float(value: Any, *, context: str) -> float:
    if isinstance(value, bool):
        raise AutoResearchError(f"{context} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise AutoResearchError(f"{context} must be numeric") from error
    if not math.isfinite(result):
        raise AutoResearchError(f"{context} must be finite")
    return result


def _positive_int(value: Any, *, context: str, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AutoResearchError(f"{context} must be an integer")
    result = value
    minimum = 0 if allow_zero else 1
    if result < minimum:
        raise AutoResearchError(f"{context} must be >= {minimum}")
    return result


def _reject_forbidden_data_tokens(
    value: Any,
    *,
    context: str,
    allow_shadow: bool = False,
) -> None:
    """Reject blind material from every search-control payload.

    This is a deliberately coarse guard.  The final blind protocol is kept in
    a separate tool and run root, so this control plane never needs that word
    in an input role, path, action, or evidence payload.
    """

    pattern = _FORBIDDEN_BLIND_TOKEN if allow_shadow else _FORBIDDEN_SEARCH_TOKEN
    if isinstance(value, str) and pattern.search(value):
        raise AutoResearchError(f"{context} references forbidden held-out data")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_forbidden_data_tokens(
                item,
                context=f"{context}[{index}]",
                allow_shadow=allow_shadow,
            )
    elif isinstance(value, dict):
        for key, item in value.items():
            _reject_forbidden_data_tokens(
                key,
                context=f"{context}.key",
                allow_shadow=allow_shadow,
            )
            _reject_forbidden_data_tokens(
                item,
                context=f"{context}.{key}",
                allow_shadow=allow_shadow,
            )


def _normalize_budget(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AutoResearchError(f"{context} must be an object")
    _require_exact_keys(
        value,
        required=set(_BUDGET_KEYS),
        context=context,
    )
    scheduler = str(value["scheduler"])
    if scheduler not in {"constant", "cosine"}:
        raise AutoResearchError(f"{context}.scheduler is not registered")
    result: dict[str, Any] = {
        "optimizer_updates": _positive_int(
            value["optimizer_updates"], context=f"{context}.optimizer_updates"
        ),
        "batch_size": _positive_int(
            value["batch_size"], context=f"{context}.batch_size"
        ),
        "learning_rate": _finite_float(
            value["learning_rate"], context=f"{context}.learning_rate"
        ),
        "gradient_clip_norm": _finite_float(
            value["gradient_clip_norm"],
            context=f"{context}.gradient_clip_norm",
        ),
        "train_examples": _positive_int(
            value["train_examples"], context=f"{context}.train_examples"
        ),
        "evaluation_examples": _positive_int(
            value["evaluation_examples"],
            context=f"{context}.evaluation_examples",
        ),
        "eval_every": _positive_int(
            value["eval_every"], context=f"{context}.eval_every"
        ),
        "scheduler": scheduler,
    }
    if result["learning_rate"] <= 0 or result["gradient_clip_norm"] <= 0:
        raise AutoResearchError(f"{context} learning rate/clip must be positive")
    return result


def _normalize_thresholds(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AutoResearchError("thresholds must be an object")
    defaults: dict[str, Any] = {
        "minimum_median_relative_sigma_gain": 0.002,
        "minimum_median_relative_chi_gain": 0.002,
        "minimum_improved_seeds": 2,
        "minimum_paired_ci_positive_seeds": 3,
        "maximum_seed_relative_regression": 0.0,
        "maximum_search_tail_relative_degradation": 0.005,
        "equivalence_potential_absolute_tolerance": 1.0e-5,
        "equivalence_metric_relative_tolerance": 2.0e-4,
        "maximum_c128_relative_sigma_delta": 0.005,
    }
    if not set(value).issubset(defaults):
        raise AutoResearchError(
            f"unknown threshold fields: {sorted(set(value) - set(defaults))}"
        )
    defaults.update(value)
    for key in (
        "minimum_median_relative_sigma_gain",
        "minimum_median_relative_chi_gain",
        "maximum_seed_relative_regression",
        "maximum_search_tail_relative_degradation",
        "equivalence_potential_absolute_tolerance",
        "equivalence_metric_relative_tolerance",
        "maximum_c128_relative_sigma_delta",
    ):
        defaults[key] = _finite_float(defaults[key], context=f"thresholds.{key}")
        if defaults[key] < 0:
            raise AutoResearchError(f"thresholds.{key} must be nonnegative")
    for key in ("minimum_improved_seeds", "minimum_paired_ci_positive_seeds"):
        defaults[key] = _positive_int(defaults[key], context=f"thresholds.{key}")
        if defaults[key] > REQUIRED_PROMOTION_SEEDS:
            raise AutoResearchError(f"thresholds.{key} exceeds the seed count")
    return defaults


def _normalize_family(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AutoResearchError(f"{context} must be an object")
    _require_exact_keys(
        value,
        required={"models"},
        optional={"family_sha256"},
        context=context,
    )
    rows = value["models"]
    if not isinstance(rows, list) or not rows:
        raise AutoResearchError(f"{context}.models must be a non-empty list")
    normalized = []
    seen = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise AutoResearchError(f"{context}.models[{index}] must be an object")
        _require_exact_keys(
            row,
            required={"seed", "checkpoint_sha256"},
            context=f"{context}.models[{index}]",
        )
        seed = _positive_int(row["seed"], context=f"{context}.models[{index}].seed")
        if seed in seen:
            raise AutoResearchError(f"{context} repeats seed {seed}")
        seen.add(seed)
        normalized.append(
            {
                "seed": seed,
                "checkpoint_sha256": _normalized_sha256(
                    row["checkpoint_sha256"],
                    context=f"{context}.models[{index}].checkpoint_sha256",
                ),
            }
        )
    normalized.sort(key=lambda row: row["seed"])
    family_payload = {"models": normalized}
    family_sha256 = digest_value(family_payload)
    if "family_sha256" in value and value["family_sha256"] != family_sha256:
        raise AutoResearchError(f"{context}.family_sha256 does not match its models")
    return {**family_payload, "family_sha256": family_sha256}


def _normalize_input(value: Any, *, index: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AutoResearchError(f"search_inputs[{index}] must be an object")
    _require_exact_keys(
        value,
        required={"role", "path", "sha256"},
        optional={"bytes"},
        context=f"search_inputs[{index}]",
    )
    role = str(value["role"])
    path = Path(str(value["path"])).expanduser().resolve()
    if not role or not path.is_file():
        raise AutoResearchError(f"search input is missing: {role} / {path}")
    _reject_forbidden_data_tokens(role, context=f"search_inputs[{index}].role")
    _reject_forbidden_data_tokens(
        path.name,
        context=f"search_inputs[{index}].path",
    )
    expected = _normalized_sha256(
        value["sha256"], context=f"search_inputs[{index}].sha256"
    )
    observed = sha256_file(path)
    if observed != expected:
        raise AutoResearchError(f"search input hash mismatch for {role}")
    size = path.stat().st_size
    if value.get("bytes") is not None and int(value["bytes"]) != size:
        raise AutoResearchError(f"search input byte count mismatch for {role}")
    result = {
        "role": role,
        "path": str(path),
        "sha256": observed,
        "bytes": size,
    }
    return result


def _normalize_index_plan(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AutoResearchError("search_index_plan must be an object")
    _require_exact_keys(
        value,
        required={
            "seed",
            "train_population",
            "train_count",
            "evaluation_population",
            "evaluation_count",
            "shared_population",
        },
        context="search_index_plan",
    )
    result = {
        "seed": _positive_int(
            value["seed"], context="search_index_plan.seed", allow_zero=True
        ),
        "train_population": _positive_int(
            value["train_population"], context="search_index_plan.train_population"
        ),
        "train_count": _positive_int(
            value["train_count"], context="search_index_plan.train_count"
        ),
        "evaluation_population": _positive_int(
            value["evaluation_population"],
            context="search_index_plan.evaluation_population",
        ),
        "evaluation_count": _positive_int(
            value["evaluation_count"], context="search_index_plan.evaluation_count"
        ),
        "shared_population": value["shared_population"],
    }
    if not isinstance(result["shared_population"], bool):
        raise AutoResearchError("search_index_plan.shared_population must be boolean")
    if result["train_count"] > result["train_population"]:
        raise AutoResearchError("train_count exceeds its population")
    if result["evaluation_count"] > result["evaluation_population"]:
        raise AutoResearchError("evaluation_count exceeds its population")
    if result["shared_population"]:
        if result["train_population"] != result["evaluation_population"]:
            raise AutoResearchError("shared populations must have equal sizes")
        if (
            result["train_count"] + result["evaluation_count"]
            > result["train_population"]
        ):
            raise AutoResearchError(
                "shared train/evaluation indices cannot be disjoint"
            )
    return result


def validate_protocol(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AutoResearchError("protocol must be an object")
    # Scan the whole raw protocol for forbidden final-blind material.  Shadow
    # is checked specifically on search input roles and basenames below, so an
    # unrelated parent directory named by a test/campaign cannot cause a false
    # positive.
    _reject_forbidden_data_tokens(
        value,
        context="protocol",
        allow_shadow=True,
    )
    _require_exact_keys(
        value,
        required={
            "schema",
            "campaign_id",
            "baseline_family",
            "promotion_seeds",
            "search_inputs",
            "search_index_plan",
            "budgets",
        },
        optional={"thresholds", "maximum_rounds"},
        context="protocol",
    )
    if value["schema"] != PROTOCOL_SCHEMA:
        raise AutoResearchError(f"protocol schema must be {PROTOCOL_SCHEMA}")
    campaign_id = str(value["campaign_id"])
    if not _IDENTIFIER.fullmatch(campaign_id):
        raise AutoResearchError("campaign_id has invalid characters")
    seeds = value["promotion_seeds"]
    if not isinstance(seeds, list) or len(seeds) != REQUIRED_PROMOTION_SEEDS:
        raise AutoResearchError("exactly three promotion seeds are required")
    promotion_seeds = sorted(
        _positive_int(seed, context="promotion_seeds") for seed in seeds
    )
    if len(set(promotion_seeds)) != REQUIRED_PROMOTION_SEEDS:
        raise AutoResearchError("promotion seeds must be distinct")
    baseline = _normalize_family(value["baseline_family"], context="baseline_family")
    if [row["seed"] for row in baseline["models"]] != promotion_seeds:
        raise AutoResearchError("baseline family must contain every promotion seed")
    inputs = value["search_inputs"]
    if not isinstance(inputs, list) or not inputs:
        raise AutoResearchError("search_inputs must be a non-empty list")
    search_inputs = [
        _normalize_input(row, index=index) for index, row in enumerate(inputs)
    ]
    roles = [row["role"] for row in search_inputs]
    if len(roles) != len(set(roles)):
        raise AutoResearchError("search input roles must be unique")
    budgets = value["budgets"]
    if not isinstance(budgets, dict):
        raise AutoResearchError("budgets must be an object")
    _require_exact_keys(
        budgets,
        required={"local_activate", "matched_relax"},
        context="budgets",
    )
    maximum_rounds = _positive_int(
        value.get("maximum_rounds", 8), context="maximum_rounds"
    )
    return {
        "schema": PROTOCOL_SCHEMA,
        "campaign_id": campaign_id,
        "baseline_family": baseline,
        "promotion_seeds": promotion_seeds,
        "search_inputs": search_inputs,
        "search_index_plan": _normalize_index_plan(value["search_index_plan"]),
        "budgets": {
            "local_activate": _normalize_budget(
                budgets["local_activate"], context="budgets.local_activate"
            ),
            "matched_relax": _normalize_budget(
                budgets["matched_relax"], context="budgets.matched_relax"
            ),
        },
        "thresholds": _normalize_thresholds(value.get("thresholds", {})),
        "maximum_rounds": maximum_rounds,
        "search_precision": SEARCH_PRECISION,
        "finalist_replay_precision": FINALIST_REPLAY_PRECISION,
        "shadow_policy": "freeze-then-single-use",
        "blind_policy": "absent-use-separate-manifest-and-run-root",
    }


def _ranked_indices(population: int, count: int, *, seed: int, label: str) -> list[int]:
    def key(index: int) -> bytes:
        return hashlib.sha256(f"{seed}:{label}:{index}".encode("ascii")).digest()

    return sorted(sorted(range(population), key=key)[:count])


def fixed_search_indices(protocol: Mapping[str, Any]) -> dict[str, Any]:
    plan = protocol["search_index_plan"]
    seed = int(plan["seed"])
    if plan["shared_population"]:
        ordered = sorted(
            range(int(plan["train_population"])),
            key=lambda index: hashlib.sha256(
                f"{seed}:shared:{index}".encode("ascii")
            ).digest(),
        )
        train_count = int(plan["train_count"])
        evaluation_count = int(plan["evaluation_count"])
        train = sorted(ordered[:train_count])
        evaluation = sorted(ordered[train_count : train_count + evaluation_count])
    else:
        train = _ranked_indices(
            int(plan["train_population"]),
            int(plan["train_count"]),
            seed=seed,
            label="train",
        )
        evaluation = _ranked_indices(
            int(plan["evaluation_population"]),
            int(plan["evaluation_count"]),
            seed=seed,
            label="evaluation",
        )
    payload = {
        "schema": INDEX_SCHEMA,
        "plan": dict(plan),
        "search_input_set_sha256": digest_value(protocol["search_inputs"]),
        "train_indices": train,
        "evaluation_indices": evaluation,
    }
    return {**payload, "indices_sha256": digest_value(payload)}


def validate_action(
    value: Any,
    *,
    expected_parent_family_sha256: str | None = None,
    expected_indices_sha256: str | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AutoResearchError("action must be an object")
    _reject_forbidden_data_tokens(value, context="action")
    _require_exact_keys(
        value,
        required={
            "schema",
            "candidate_id",
            "round",
            "parent_family_sha256",
            "search_indices_sha256",
            "mutation",
            "pipeline",
        },
        optional={"action_sha256"},
        context="action",
    )
    if value["schema"] != ACTION_SCHEMA:
        raise AutoResearchError(f"action schema must be {ACTION_SCHEMA}")
    candidate_id = str(value["candidate_id"])
    if not _IDENTIFIER.fullmatch(candidate_id):
        raise AutoResearchError("candidate_id has invalid characters")
    round_index = _positive_int(value["round"], context="action.round")
    parent = _normalized_sha256(
        value["parent_family_sha256"], context="action.parent_family_sha256"
    )
    indices = _normalized_sha256(
        value["search_indices_sha256"], context="action.search_indices_sha256"
    )
    if (
        expected_parent_family_sha256 is not None
        and parent != expected_parent_family_sha256
    ):
        raise AutoResearchError("action parent is not the current champion family")
    if expected_indices_sha256 is not None and indices != expected_indices_sha256:
        raise AutoResearchError("action does not use the frozen search indices")
    pipeline = value["pipeline"]
    required_pipeline = [
        {"kind": "local-activate"},
        {"kind": "matched-relax"},
    ]
    if pipeline != required_pipeline:
        raise AutoResearchError(
            "action pipeline must be local-activate followed by matched-relax"
        )
    mutation = value["mutation"]
    if not isinstance(mutation, dict) or "kind" not in mutation:
        raise AutoResearchError("action mutation must be an object with kind")
    kind = str(mutation["kind"])
    if kind == "topology":
        _require_exact_keys(
            mutation,
            required={"kind", "transport"},
            context="action.mutation",
        )
        if mutation["transport"] != EXACT_TOPOLOGY_TRANSPORT:
            raise AutoResearchError("topology transport is not exact/registered")
        normalized_mutation = {
            "kind": kind,
            "transport": EXACT_TOPOLOGY_TRANSPORT,
        }
    elif kind == "rank":
        _require_exact_keys(
            mutation,
            required={
                "kind",
                "target",
                "source_dimension",
                "target_dimension",
                "structural_maximum",
                "new_output_real_parameters",
            },
            optional={"edge"},
            context="action.mutation",
        )
        target = str(mutation["target"])
        if target not in {"internal-edge", "shared-leaf"}:
            raise AutoResearchError("rank target is not registered")
        source = _positive_int(
            mutation["source_dimension"], context="action.mutation.source_dimension"
        )
        target_dimension = _positive_int(
            mutation["target_dimension"], context="action.mutation.target_dimension"
        )
        structural_maximum = _positive_int(
            mutation["structural_maximum"],
            context="action.mutation.structural_maximum",
        )
        new_parameters = _positive_int(
            mutation["new_output_real_parameters"],
            context="action.mutation.new_output_real_parameters",
        )
        if not source < target_dimension <= structural_maximum:
            raise AutoResearchError("rank growth exceeds the nested structural space")
        if new_parameters > MAX_LOCAL_NEW_REAL_PARAMETERS:
            raise AutoResearchError("rank growth exceeds the local parameter cap")
        if target == "shared-leaf" and target_dimension != source + 1:
            raise AutoResearchError("shared-leaf growth is restricted to one row")
        edge: int | None = None
        if target == "internal-edge":
            if "edge" not in mutation:
                raise AutoResearchError("internal-edge rank growth requires an edge")
            edge = _positive_int(
                mutation["edge"], context="action.mutation.edge", allow_zero=True
            )
        elif "edge" in mutation:
            raise AutoResearchError("shared-leaf growth does not accept an edge")
        normalized_mutation = {
            "kind": kind,
            "target": target,
            "source_dimension": source,
            "target_dimension": target_dimension,
            "structural_maximum": structural_maximum,
            "new_output_real_parameters": new_parameters,
        }
        if edge is not None:
            normalized_mutation["edge"] = edge
    elif kind == "root-residual":
        _require_exact_keys(
            mutation,
            required={"kind", "topology", "rank"},
            context="action.mutation",
        )
        rank = _positive_int(mutation["rank"], context="action.mutation.rank")
        if mutation["topology"] != "five-leaf-2plus3":
            raise AutoResearchError("root residual requires the five-leaf 2+3 topology")
        if rank not in ROOT_RESIDUAL_RANKS:
            raise AutoResearchError("root residual rank is not registered")
        normalized_mutation = {
            "kind": kind,
            "topology": "five-leaf-2plus3",
            "rank": rank,
        }
    else:
        raise AutoResearchError(f"mutation kind is not registered: {kind}")
    normalized = {
        "schema": ACTION_SCHEMA,
        "candidate_id": candidate_id,
        "round": round_index,
        "parent_family_sha256": parent,
        "search_indices_sha256": indices,
        "mutation": normalized_mutation,
        "pipeline": required_pipeline,
    }
    action_sha256 = digest_value(normalized)
    if "action_sha256" in value and value["action_sha256"] != action_sha256:
        raise AutoResearchError("action_sha256 does not match the action")
    return {**normalized, "action_sha256": action_sha256}


def _normalize_metrics(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AutoResearchError(f"{context} must be an object")
    _require_exact_keys(value, required=set(_METRIC_KEYS), context=context)
    result = {
        key: _finite_float(value[key], context=f"{context}.{key}")
        for key in _METRIC_KEYS - {"nonpositive_metric_count"}
    }
    result["nonpositive_metric_count"] = _positive_int(
        value["nonpositive_metric_count"],
        context=f"{context}.nonpositive_metric_count",
        allow_zero=True,
    )
    if result["sigma"] <= 0 or result["chi"] <= 0:
        raise AutoResearchError(f"{context} sigma and chi must be positive")
    if result["q999"] < 0 or result["cvar_1pct"] < 0:
        raise AutoResearchError(f"{context} tail metrics must be nonnegative")
    return result


def _normalize_paired(value: Any, *, context: str) -> dict[str, float]:
    if not isinstance(value, dict):
        raise AutoResearchError(f"{context} must be an object")
    _require_exact_keys(
        value,
        required={"sigma_ci95_low", "e2_ci95_low"},
        context=context,
    )
    return {
        key: _finite_float(value[key], context=f"{context}.{key}")
        for key in ("sigma_ci95_low", "e2_ci95_low")
    }


def _normalize_seed_evidence(
    value: Any,
    *,
    index: int,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    context = f"evidence.seeds[{index}]"
    if not isinstance(value, dict):
        raise AutoResearchError(f"{context} must be an object")
    _require_exact_keys(
        value,
        required={
            "seed",
            "equivalence",
            "local_activate_budget",
            "control_matched_relax_budget",
            "candidate_matched_relax_budget",
            "control_matched_relax_batch_plan_sha256",
            "candidate_matched_relax_batch_plan_sha256",
            "control",
            "candidate",
            "paired",
            "control_checkpoint_sha256",
            "candidate_checkpoint_sha256",
            "control_trainable_real_parameter_count",
            "candidate_trainable_real_parameter_count",
        },
        context=context,
    )
    equivalence = value["equivalence"]
    if not isinstance(equivalence, dict):
        raise AutoResearchError(f"{context}.equivalence must be an object")
    _require_exact_keys(
        equivalence,
        required={
            "passed",
            "potential_max_absolute",
            "metric_max_relative_frobenius",
        },
        context=f"{context}.equivalence",
    )
    result = {
        "seed": _positive_int(value["seed"], context=f"{context}.seed"),
        "equivalence": {
            "passed": equivalence["passed"],
            "potential_max_absolute": _finite_float(
                equivalence["potential_max_absolute"],
                context=f"{context}.equivalence.potential_max_absolute",
            ),
            "metric_max_relative_frobenius": _finite_float(
                equivalence["metric_max_relative_frobenius"],
                context=f"{context}.equivalence.metric_max_relative_frobenius",
            ),
        },
        "local_activate_budget": _normalize_budget(
            value["local_activate_budget"], context=f"{context}.local_activate_budget"
        ),
        "control_matched_relax_budget": _normalize_budget(
            value["control_matched_relax_budget"],
            context=f"{context}.control_matched_relax_budget",
        ),
        "candidate_matched_relax_budget": _normalize_budget(
            value["candidate_matched_relax_budget"],
            context=f"{context}.candidate_matched_relax_budget",
        ),
        "control_matched_relax_batch_plan_sha256": _normalized_sha256(
            value["control_matched_relax_batch_plan_sha256"],
            context=f"{context}.control_matched_relax_batch_plan_sha256",
        ),
        "candidate_matched_relax_batch_plan_sha256": _normalized_sha256(
            value["candidate_matched_relax_batch_plan_sha256"],
            context=f"{context}.candidate_matched_relax_batch_plan_sha256",
        ),
        "control": _normalize_metrics(value["control"], context=f"{context}.control"),
        "candidate": _normalize_metrics(
            value["candidate"], context=f"{context}.candidate"
        ),
        "paired": _normalize_paired(value["paired"], context=f"{context}.paired"),
        "control_checkpoint_sha256": _normalized_sha256(
            value["control_checkpoint_sha256"],
            context=f"{context}.control_checkpoint_sha256",
        ),
        "candidate_checkpoint_sha256": _normalized_sha256(
            value["candidate_checkpoint_sha256"],
            context=f"{context}.candidate_checkpoint_sha256",
        ),
        "control_trainable_real_parameter_count": _positive_int(
            value["control_trainable_real_parameter_count"],
            context=f"{context}.control_trainable_real_parameter_count",
        ),
        "candidate_trainable_real_parameter_count": _positive_int(
            value["candidate_trainable_real_parameter_count"],
            context=f"{context}.candidate_trainable_real_parameter_count",
        ),
    }
    if not isinstance(result["equivalence"]["passed"], bool):
        raise AutoResearchError(f"{context}.equivalence.passed must be boolean")
    return result


def validate_search_evidence(
    value: Any,
    *,
    action: Mapping[str, Any],
    protocol: Mapping[str, Any],
    expected_indices_sha256: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AutoResearchError("search evidence must be an object")
    _reject_forbidden_data_tokens(value, context="search evidence")
    _require_exact_keys(
        value,
        required={
            "schema",
            "candidate_id",
            "round",
            "action_sha256",
            "parent_family_sha256",
            "search_indices_sha256",
            "precision",
            "seeds",
        },
        context="search evidence",
    )
    if value["schema"] != SEARCH_EVIDENCE_SCHEMA:
        raise AutoResearchError(
            f"search evidence schema must be {SEARCH_EVIDENCE_SCHEMA}"
        )
    checks = {
        "candidate_id": action["candidate_id"],
        "round": action["round"],
        "action_sha256": action["action_sha256"],
        "parent_family_sha256": action["parent_family_sha256"],
        "search_indices_sha256": expected_indices_sha256,
        "precision": SEARCH_PRECISION,
    }
    for key, expected in checks.items():
        if value[key] != expected:
            raise AutoResearchError(
                f"search evidence {key} does not match the action/protocol"
            )
    rows = value["seeds"]
    if not isinstance(rows, list) or len(rows) != REQUIRED_PROMOTION_SEEDS:
        raise AutoResearchError("search evidence requires exactly three seed rows")
    normalized_rows = [
        _normalize_seed_evidence(row, index=index, protocol=protocol)
        for index, row in enumerate(rows)
    ]
    normalized_rows.sort(key=lambda row: row["seed"])
    if [row["seed"] for row in normalized_rows] != protocol["promotion_seeds"]:
        raise AutoResearchError("search evidence seeds do not match the protocol")
    return {**checks, "schema": SEARCH_EVIDENCE_SCHEMA, "seeds": normalized_rows}


def _relative_gain(control: float, candidate: float) -> float:
    return (control - candidate) / control


def _relative_degradation(control: float, candidate: float) -> float:
    if control == 0:
        return 0.0 if candidate == 0 else math.inf
    return (candidate - control) / control


def adjudicate_candidate(
    evidence: Mapping[str, Any],
    *,
    action: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    thresholds = protocol["thresholds"]
    seed_rows = []
    sigma_gains = []
    chi_gains = []
    improved = 0
    paired_positive = 0
    all_budgets_match = True
    all_batch_plans_match = True
    all_parameter_accounting_matches = True
    all_equivalent = True
    all_positive = True
    all_tail_safe = True
    observed_parameter_deltas: list[int] = []
    for row in evidence["seeds"]:
        control = row["control"]
        candidate = row["candidate"]
        sigma_gain = _relative_gain(control["sigma"], candidate["sigma"])
        chi_gain = _relative_gain(control["chi"], candidate["chi"])
        q999_degradation = _relative_degradation(control["q999"], candidate["q999"])
        cvar_degradation = _relative_degradation(
            control["cvar_1pct"], candidate["cvar_1pct"]
        )
        budget_match = bool(
            row["local_activate_budget"] == protocol["budgets"]["local_activate"]
            and row["control_matched_relax_budget"]
            == protocol["budgets"]["matched_relax"]
            and row["candidate_matched_relax_budget"]
            == protocol["budgets"]["matched_relax"]
        )
        batch_plan_match = bool(
            row["control_matched_relax_batch_plan_sha256"]
            == row["candidate_matched_relax_batch_plan_sha256"]
        )
        parameter_delta = (
            row["candidate_trainable_real_parameter_count"]
            - row["control_trainable_real_parameter_count"]
        )
        added_parameters = max(0, parameter_delta)
        mutation_kind = action["mutation"]["kind"]
        parameter_accounting_match = bool(
            parameter_delta >= 0
            and added_parameters <= MAX_LOCAL_NEW_REAL_PARAMETERS
            and (
                mutation_kind != "rank"
                or parameter_delta == action["mutation"]["new_output_real_parameters"]
            )
            and (mutation_kind != "root-residual" or parameter_delta > 0)
        )
        equivalence = row["equivalence"]
        equivalent = bool(
            equivalence["passed"]
            and equivalence["potential_max_absolute"]
            <= thresholds["equivalence_potential_absolute_tolerance"]
            and equivalence["metric_max_relative_frobenius"]
            <= thresholds["equivalence_metric_relative_tolerance"]
        )
        control_positive = bool(
            control["minimum_metric_eigenvalue"] > 0
            and control["nonpositive_metric_count"] == 0
        )
        candidate_positive = bool(
            candidate["minimum_metric_eigenvalue"] > 0
            and candidate["nonpositive_metric_count"] == 0
        )
        positive = control_positive and candidate_positive
        tail_safe = bool(
            q999_degradation <= thresholds["maximum_search_tail_relative_degradation"]
            and cvar_degradation
            <= thresholds["maximum_search_tail_relative_degradation"]
        )
        paired_ok = bool(
            row["paired"]["sigma_ci95_low"] > 0 and row["paired"]["e2_ci95_low"] > 0
        )
        if sigma_gain > 0 and chi_gain > 0:
            improved += 1
        if paired_ok:
            paired_positive += 1
        all_budgets_match = all_budgets_match and budget_match
        all_batch_plans_match = all_batch_plans_match and batch_plan_match
        all_parameter_accounting_matches = (
            all_parameter_accounting_matches and parameter_accounting_match
        )
        all_equivalent = all_equivalent and equivalent
        all_positive = all_positive and positive
        all_tail_safe = all_tail_safe and tail_safe
        sigma_gains.append(sigma_gain)
        chi_gains.append(chi_gain)
        observed_parameter_deltas.append(parameter_delta)
        seed_rows.append(
            {
                "seed": row["seed"],
                "relative_sigma_gain": sigma_gain,
                "relative_chi_gain": chi_gain,
                "relative_q999_degradation": q999_degradation,
                "relative_cvar_1pct_degradation": cvar_degradation,
                "budget_match": budget_match,
                "batch_plan_match": batch_plan_match,
                "matched_relax_batch_plan_sha256": row[
                    "candidate_matched_relax_batch_plan_sha256"
                ],
                "control_trainable_real_parameter_count": row[
                    "control_trainable_real_parameter_count"
                ],
                "candidate_trainable_real_parameter_count": row[
                    "candidate_trainable_real_parameter_count"
                ],
                "new_output_real_parameters": added_parameters,
                "parameter_accounting_match": parameter_accounting_match,
                "equivalence_passes": equivalent,
                "control_positivity_passes": control_positive,
                "candidate_positivity_passes": candidate_positive,
                "positivity_passes": positive,
                "tail_passes": tail_safe,
                "paired_ci_passes": paired_ok,
                "control_checkpoint_sha256": row["control_checkpoint_sha256"],
                "candidate_checkpoint_sha256": row["candidate_checkpoint_sha256"],
            }
        )
    median_sigma = statistics.median(sigma_gains)
    median_chi = statistics.median(chi_gains)
    consistent_parameter_delta = len(set(observed_parameter_deltas)) == 1
    gates = {
        "three_registered_seeds": len(seed_rows) == REQUIRED_PROMOTION_SEEDS,
        "complex64_search": evidence["precision"] == SEARCH_PRECISION,
        "fixed_search_indices": (
            evidence["search_indices_sha256"] == action["search_indices_sha256"]
        ),
        "matched_budgets": all_budgets_match,
        "matched_batch_plans": all_batch_plans_match,
        "parameter_accounting": (
            all_parameter_accounting_matches and consistent_parameter_delta
        ),
        "exact_transport_equivalence": all_equivalent,
        "positivity": all_positive,
        "tail": all_tail_safe,
        "minimum_improved_seeds": (improved >= thresholds["minimum_improved_seeds"]),
        "minimum_paired_ci_positive_seeds": (
            paired_positive >= thresholds["minimum_paired_ci_positive_seeds"]
        ),
        "median_sigma_gain": (
            median_sigma >= thresholds["minimum_median_relative_sigma_gain"]
        ),
        "median_chi_gain": (
            median_chi >= thresholds["minimum_median_relative_chi_gain"]
        ),
        "worst_seed_sigma": (
            min(sigma_gains) >= -thresholds["maximum_seed_relative_regression"]
        ),
        "worst_seed_chi": (
            min(chi_gains) >= -thresholds["maximum_seed_relative_regression"]
        ),
    }
    candidate_family = _normalize_family(
        {
            "models": [
                {
                    "seed": row["seed"],
                    "checkpoint_sha256": row["candidate_checkpoint_sha256"],
                }
                for row in evidence["seeds"]
            ]
        },
        context="candidate_family",
    )
    control_family = _normalize_family(
        {
            "models": [
                {
                    "seed": row["seed"],
                    "checkpoint_sha256": row["control_checkpoint_sha256"],
                }
                for row in evidence["seeds"]
            ]
        },
        context="control_family",
    )
    return {
        "candidate_id": evidence["candidate_id"],
        "action_sha256": evidence["action_sha256"],
        "promotion_passes": all(gates.values()),
        "gates": gates,
        "median_relative_sigma_gain": median_sigma,
        "median_relative_chi_gain": median_chi,
        "minimum_relative_sigma_gain": min(sigma_gains),
        "minimum_relative_chi_gain": min(chi_gains),
        "improved_seed_count": improved,
        "paired_ci_positive_seed_count": paired_positive,
        "seed_rows": seed_rows,
        "candidate_family": candidate_family,
        "control_family": control_family,
        "new_output_real_parameters": max(0, observed_parameter_deltas[0]),
    }


def adjudicate_round(
    *,
    round_index: int,
    actions: Mapping[str, Mapping[str, Any]],
    evidences: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    if not evidences:
        raise AutoResearchError("round adjudication requires candidate evidence")
    adjudications = []
    control_by_seed: dict[int, set[str]] = {
        seed: set() for seed in protocol["promotion_seeds"]
    }
    batch_plan_by_seed: dict[int, set[str]] = {
        seed: set() for seed in protocol["promotion_seeds"]
    }
    control_evidence_by_seed: dict[int, set[str]] = {
        seed: set() for seed in protocol["promotion_seeds"]
    }
    for evidence in evidences:
        candidate_id = str(evidence.get("candidate_id", ""))
        if candidate_id not in actions:
            raise AutoResearchError(f"unregistered candidate evidence: {candidate_id}")
        action = actions[candidate_id]
        if action["round"] != round_index:
            raise AutoResearchError("candidate evidence belongs to another round")
        adjudication = adjudicate_candidate(
            evidence,
            action=action,
            protocol=protocol,
        )
        adjudications.append(adjudication)
        for row in evidence["seeds"]:
            control_by_seed[row["seed"]].add(row["control_checkpoint_sha256"])
            batch_plan_by_seed[row["seed"]].add(
                row["candidate_matched_relax_batch_plan_sha256"]
            )
            control_evidence_by_seed[row["seed"]].add(
                digest_value(
                    {
                        "checkpoint_sha256": row["control_checkpoint_sha256"],
                        "metrics": row["control"],
                        "matched_relax_budget": row["control_matched_relax_budget"],
                        "matched_relax_batch_plan_sha256": row[
                            "control_matched_relax_batch_plan_sha256"
                        ],
                        "trainable_real_parameter_count": row[
                            "control_trainable_real_parameter_count"
                        ],
                    }
                )
            )
    common_control = all(
        len(values) == 1 for values in control_evidence_by_seed.values()
    )
    common_batch_plan = all(len(values) == 1 for values in batch_plan_by_seed.values())
    if not common_control or not common_batch_plan:
        for row in adjudications:
            row["gates"]["common_no_growth_control"] = common_control
            row["gates"]["common_matched_relax_batch_plan"] = common_batch_plan
            row["promotion_passes"] = False
    else:
        for row in adjudications:
            row["gates"]["common_no_growth_control"] = True
            row["gates"]["common_matched_relax_batch_plan"] = True
    passing = [row for row in adjudications if row["promotion_passes"]]
    winner = None
    if passing:
        winner = sorted(
            passing,
            key=lambda row: (
                -row["median_relative_sigma_gain"],
                -row["median_relative_chi_gain"],
                row["new_output_real_parameters"],
                row["candidate_id"],
            ),
        )[0]
    return {
        "schema": ROUND_ADJUDICATION_SCHEMA,
        "round": round_index,
        "candidate_count": len(adjudications),
        "common_no_growth_control": common_control,
        "common_matched_relax_batch_plan": common_batch_plan,
        "control_checkpoints_by_seed": {
            str(seed): sorted(values) for seed, values in control_by_seed.items()
        },
        "control_evidence_hashes_by_seed": {
            str(seed): sorted(values)
            for seed, values in control_evidence_by_seed.items()
        },
        "batch_plans_by_seed": {
            str(seed): sorted(values) for seed, values in batch_plan_by_seed.items()
        },
        "candidates": sorted(adjudications, key=lambda row: row["candidate_id"]),
        "winner_candidate_id": None if winner is None else winner["candidate_id"],
        "winner_action_sha256": None if winner is None else winner["action_sha256"],
        "winner_family": None if winner is None else winner["candidate_family"],
        "winner_comparator_family": (
            None if winner is None else winner["control_family"]
        ),
        "stop_recommended": winner is None,
    }


def _normalize_evaluation_row(value: Any, *, index: int) -> dict[str, Any]:
    context = f"shadow.comparisons[{index}]"
    if not isinstance(value, dict):
        raise AutoResearchError(f"{context} must be an object")
    _require_exact_keys(
        value,
        required={
            "seed",
            "control_checkpoint_sha256",
            "candidate_checkpoint_sha256",
            "control",
            "candidate",
            "paired",
        },
        context=context,
    )
    return {
        "seed": _positive_int(value["seed"], context=f"{context}.seed"),
        "control_checkpoint_sha256": _normalized_sha256(
            value["control_checkpoint_sha256"],
            context=f"{context}.control_checkpoint_sha256",
        ),
        "candidate_checkpoint_sha256": _normalized_sha256(
            value["candidate_checkpoint_sha256"],
            context=f"{context}.candidate_checkpoint_sha256",
        ),
        "control": _normalize_metrics(value["control"], context=f"{context}.control"),
        "candidate": _normalize_metrics(
            value["candidate"], context=f"{context}.candidate"
        ),
        "paired": _normalize_paired(value["paired"], context=f"{context}.paired"),
    }


def validate_shadow_evidence(
    value: Any,
    *,
    frozen: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AutoResearchError("shadow evidence must be an object")
    _reject_forbidden_data_tokens(
        value,
        context="shadow evidence",
        allow_shadow=True,
    )
    _require_exact_keys(
        value,
        required={
            "schema",
            "frozen_family_sha256",
            "shadow_dataset_sha256",
            "shadow_indices_sha256",
            "evaluator_sha256",
            "precision",
            "comparisons",
        },
        context="shadow evidence",
    )
    if value["schema"] != SHADOW_EVIDENCE_SCHEMA:
        raise AutoResearchError(
            f"shadow evidence schema must be {SHADOW_EVIDENCE_SCHEMA}"
        )
    if value["frozen_family_sha256"] != frozen["champion_family"]["family_sha256"]:
        raise AutoResearchError("shadow evidence is not for the frozen finalist")
    if value["precision"] != SEARCH_PRECISION:
        raise AutoResearchError(
            "shadow evaluation must use the registered complex64 model"
        )
    dataset_sha = _normalized_sha256(
        value["shadow_dataset_sha256"], context="shadow_dataset_sha256"
    )
    indices_sha = _normalized_sha256(
        value["shadow_indices_sha256"], context="shadow_indices_sha256"
    )
    evaluator_sha = _normalized_sha256(
        value["evaluator_sha256"], context="evaluator_sha256"
    )
    rows = value["comparisons"]
    if not isinstance(rows, list) or len(rows) != REQUIRED_PROMOTION_SEEDS:
        raise AutoResearchError("shadow evidence requires exactly three comparisons")
    comparisons = [
        _normalize_evaluation_row(row, index=index) for index, row in enumerate(rows)
    ]
    comparisons.sort(key=lambda row: row["seed"])
    if [row["seed"] for row in comparisons] != protocol["promotion_seeds"]:
        raise AutoResearchError("shadow seeds do not match the protocol")
    champion = {
        row["seed"]: row["checkpoint_sha256"]
        for row in frozen["champion_family"]["models"]
    }
    comparator = {
        row["seed"]: row["checkpoint_sha256"]
        for row in frozen["comparator_family"]["models"]
    }
    for row in comparisons:
        if row["candidate_checkpoint_sha256"] != champion[row["seed"]]:
            raise AutoResearchError("shadow candidate checkpoint was not frozen")
        if row["control_checkpoint_sha256"] != comparator[row["seed"]]:
            raise AutoResearchError("shadow comparator checkpoint was not frozen")
    return {
        "schema": SHADOW_EVIDENCE_SCHEMA,
        "frozen_family_sha256": frozen["champion_family"]["family_sha256"],
        "shadow_dataset_sha256": dataset_sha,
        "shadow_indices_sha256": indices_sha,
        "evaluator_sha256": evaluator_sha,
        "precision": SEARCH_PRECISION,
        "comparisons": comparisons,
    }


def validate_shadow_claim(
    value: Any,
    *,
    frozen: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AutoResearchError("shadow claim must be an object")
    _reject_forbidden_data_tokens(
        value,
        context="shadow claim",
        allow_shadow=True,
    )
    _require_exact_keys(
        value,
        required={
            "schema",
            "frozen_family_sha256",
            "shadow_dataset_sha256",
            "shadow_indices_sha256",
            "evaluator_sha256",
        },
        context="shadow claim",
    )
    if value["schema"] != SHADOW_CLAIM_SCHEMA:
        raise AutoResearchError(f"shadow claim schema must be {SHADOW_CLAIM_SCHEMA}")
    family_sha256 = frozen["champion_family"]["family_sha256"]
    if value["frozen_family_sha256"] != family_sha256:
        raise AutoResearchError("shadow claim is not for the frozen finalist")
    return {
        "schema": SHADOW_CLAIM_SCHEMA,
        "frozen_family_sha256": family_sha256,
        "shadow_dataset_sha256": _normalized_sha256(
            value["shadow_dataset_sha256"], context="shadow_dataset_sha256"
        ),
        "shadow_indices_sha256": _normalized_sha256(
            value["shadow_indices_sha256"], context="shadow_indices_sha256"
        ),
        "evaluator_sha256": _normalized_sha256(
            value["evaluator_sha256"], context="evaluator_sha256"
        ),
    }


def adjudicate_shadow(
    evidence: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    sigma_gains = []
    chi_gains = []
    seed_rows = []
    improved = 0
    paired_positive = 0
    positive = True
    tail_non_degrading = True
    for row in evidence["comparisons"]:
        control = row["control"]
        candidate = row["candidate"]
        sigma_gain = _relative_gain(control["sigma"], candidate["sigma"])
        chi_gain = _relative_gain(control["chi"], candidate["chi"])
        q999_degradation = _relative_degradation(control["q999"], candidate["q999"])
        cvar_degradation = _relative_degradation(
            control["cvar_1pct"], candidate["cvar_1pct"]
        )
        row_positive = bool(
            candidate["minimum_metric_eigenvalue"] > 0
            and candidate["nonpositive_metric_count"] == 0
        )
        row_tail = q999_degradation <= 0 and cvar_degradation <= 0
        row_paired = bool(
            row["paired"]["sigma_ci95_low"] > 0 and row["paired"]["e2_ci95_low"] > 0
        )
        if sigma_gain > 0 and chi_gain > 0:
            improved += 1
        if row_paired:
            paired_positive += 1
        positive = positive and row_positive
        tail_non_degrading = tail_non_degrading and row_tail
        sigma_gains.append(sigma_gain)
        chi_gains.append(chi_gain)
        seed_rows.append(
            {
                "seed": row["seed"],
                "relative_sigma_gain": sigma_gain,
                "relative_chi_gain": chi_gain,
                "relative_q999_degradation": q999_degradation,
                "relative_cvar_1pct_degradation": cvar_degradation,
                "positivity_passes": row_positive,
                "tail_non_degrading": row_tail,
                "paired_ci_passes": row_paired,
            }
        )
    thresholds = protocol["thresholds"]
    gates = {
        "three_registered_seeds": len(seed_rows) == REQUIRED_PROMOTION_SEEDS,
        "complex64_frozen_models": evidence["precision"] == SEARCH_PRECISION,
        "minimum_improved_seeds": improved >= thresholds["minimum_improved_seeds"],
        "minimum_paired_ci_positive_seeds": (
            paired_positive >= thresholds["minimum_paired_ci_positive_seeds"]
        ),
        "positive_metrics": positive,
        "tail_non_degrading": tail_non_degrading,
        "median_sigma_improves": statistics.median(sigma_gains) > 0,
        "median_chi_improves": statistics.median(chi_gains) > 0,
        "worst_seed_sigma": min(sigma_gains)
        >= -thresholds["maximum_seed_relative_regression"],
        "worst_seed_chi": min(chi_gains)
        >= -thresholds["maximum_seed_relative_regression"],
    }
    return {
        "schema": SHADOW_ADJUDICATION_SCHEMA,
        "shadow_passes": all(gates.values()),
        "gates": gates,
        "median_relative_sigma_gain": statistics.median(sigma_gains),
        "median_relative_chi_gain": statistics.median(chi_gains),
        "seed_rows": seed_rows,
        "shadow_dataset_sha256": evidence["shadow_dataset_sha256"],
        "shadow_indices_sha256": evidence["shadow_indices_sha256"],
        "evaluator_sha256": evidence["evaluator_sha256"],
    }


def validate_replay_evidence(
    value: Any,
    *,
    frozen: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AutoResearchError("complex128 replay evidence must be an object")
    _reject_forbidden_data_tokens(value, context="complex128 replay evidence")
    _require_exact_keys(
        value,
        required={
            "schema",
            "frozen_family_sha256",
            "precision",
            "optimizer_updates",
            "replays",
        },
        context="complex128 replay evidence",
    )
    if value["schema"] != REPLAY_SCHEMA:
        raise AutoResearchError(f"replay schema must be {REPLAY_SCHEMA}")
    if value["frozen_family_sha256"] != frozen["champion_family"]["family_sha256"]:
        raise AutoResearchError("complex128 replay is not for the frozen finalist")
    if value["precision"] != FINALIST_REPLAY_PRECISION:
        raise AutoResearchError("finalist replay must use complex128")
    if (
        isinstance(value["optimizer_updates"], bool)
        or not isinstance(value["optimizer_updates"], int)
        or value["optimizer_updates"] != 0
    ):
        raise AutoResearchError("finalist precision replay must have zero updates")
    rows = value["replays"]
    if not isinstance(rows, list) or len(rows) != REQUIRED_PROMOTION_SEEDS:
        raise AutoResearchError("complex128 replay requires exactly three seed rows")
    source_by_seed = {
        row["seed"]: row["checkpoint_sha256"]
        for row in frozen["champion_family"]["models"]
    }
    normalized = []
    for index, row in enumerate(rows):
        context = f"replays[{index}]"
        if not isinstance(row, dict):
            raise AutoResearchError(f"{context} must be an object")
        _require_exact_keys(
            row,
            required={
                "seed",
                "source_checkpoint_sha256",
                "complex128_checkpoint_sha256",
                "relative_sigma_delta",
                "metrics",
            },
            context=context,
        )
        seed = _positive_int(row["seed"], context=f"{context}.seed")
        source = _normalized_sha256(
            row["source_checkpoint_sha256"],
            context=f"{context}.source_checkpoint_sha256",
        )
        if source_by_seed.get(seed) != source:
            raise AutoResearchError("complex128 replay source was not frozen")
        relative_delta = _finite_float(
            row["relative_sigma_delta"], context=f"{context}.relative_sigma_delta"
        )
        metrics = _normalize_metrics(row["metrics"], context=f"{context}.metrics")
        if (
            abs(relative_delta)
            > protocol["thresholds"]["maximum_c128_relative_sigma_delta"]
        ):
            raise AutoResearchError("complex128 replay changes sigma beyond tolerance")
        if (
            metrics["minimum_metric_eigenvalue"] <= 0
            or metrics["nonpositive_metric_count"]
        ):
            raise AutoResearchError("complex128 replay failed positivity")
        normalized.append(
            {
                "seed": seed,
                "source_checkpoint_sha256": source,
                "complex128_checkpoint_sha256": _normalized_sha256(
                    row["complex128_checkpoint_sha256"],
                    context=f"{context}.complex128_checkpoint_sha256",
                ),
                "relative_sigma_delta": relative_delta,
                "metrics": metrics,
            }
        )
    normalized.sort(key=lambda row: row["seed"])
    if [row["seed"] for row in normalized] != protocol["promotion_seeds"]:
        raise AutoResearchError("complex128 replay seeds do not match the protocol")
    return {
        "schema": REPLAY_SCHEMA,
        "frozen_family_sha256": frozen["champion_family"]["family_sha256"],
        "precision": FINALIST_REPLAY_PRECISION,
        "optimizer_updates": 0,
        "replays": normalized,
    }


class CampaignStore:
    """Create-only, resumable ledger for one constrained search campaign."""

    def __init__(self, run_root: Path) -> None:
        self.run_root = run_root.expanduser().resolve(strict=False)
        self.protocol_path = self.run_root / "protocol.lock.json"
        self.indices_path = self.run_root / "search_indices.json"
        self.ledger_path = self.run_root / "ledger.json"
        self.lock_path = self.run_root / ".ledger.lock"

    @classmethod
    def initialize(cls, run_root: Path, protocol_value: Any) -> "CampaignStore":
        store = cls(run_root)
        protocol = validate_protocol(protocol_value)
        indices = fixed_search_indices(protocol)
        protocol_sha256 = digest_value(protocol)
        if store.run_root.exists() and any(store.run_root.iterdir()):
            if not (
                (store.run_root / ROOT_SENTINEL).is_file()
                and store.protocol_path.is_file()
                and store.indices_path.is_file()
                and store.ledger_path.is_file()
            ):
                raise AutoResearchError("refusing a non-empty, unmanaged run root")
            existing_protocol = _read_json(store.protocol_path)
            existing_indices = _read_json(store.indices_path)
            if (
                digest_value(existing_protocol) != protocol_sha256
                or existing_indices != indices
            ):
                raise AutoResearchError("run root belongs to a different protocol")
            ledger = store.status()
            if (
                ledger.get("protocol_sha256") != protocol_sha256
                or ledger.get("search_indices_sha256") != indices["indices_sha256"]
            ):
                raise AutoResearchError("run root ledger does not match the protocol")
            return store
        store.run_root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(store.protocol_path, protocol)
        atomic_write_json(store.indices_path, indices)
        ledger = {
            "schema": LEDGER_SCHEMA,
            "campaign_id": protocol["campaign_id"],
            "protocol_sha256": protocol_sha256,
            "search_indices_sha256": indices["indices_sha256"],
            "status": "active",
            "revision": 0,
            "current_round": 0,
            "champion_family": protocol["baseline_family"],
            "champion_candidate_id": "baseline",
            "champion_comparator_family": None,
            "promotion_count": 0,
            "rounds": {},
            "frozen": None,
            "complex128_replay": None,
            "shadow": None,
            "events": [
                _event_row(
                    revision=0,
                    event="initialized",
                    previous_event_sha256="0" * 64,
                )
            ],
        }
        _seal_ledger(ledger)
        atomic_write_json(store.ledger_path, ledger)
        (store.run_root / ROOT_SENTINEL).write_text(
            f"{protocol['campaign_id']}\n", encoding="utf-8"
        )
        return store

    def protocol(self) -> dict[str, Any]:
        protocol = _read_json(self.protocol_path)
        ledger = self.ledger()
        if digest_value(protocol) != ledger["protocol_sha256"]:
            raise AutoResearchError("locked protocol was modified")
        return protocol

    def indices(self) -> dict[str, Any]:
        indices = _read_json(self.indices_path)
        ledger = self.ledger()
        if indices.get("indices_sha256") != ledger["search_indices_sha256"]:
            raise AutoResearchError("fixed search indices were modified")
        payload = {
            key: value for key, value in indices.items() if key != "indices_sha256"
        }
        if digest_value(payload) != indices["indices_sha256"]:
            raise AutoResearchError("fixed search index digest is invalid")
        protocol = _read_json(self.protocol_path)
        if digest_value(protocol) != ledger["protocol_sha256"]:
            raise AutoResearchError("locked protocol was modified")
        if indices != fixed_search_indices(protocol):
            raise AutoResearchError("fixed search indices do not match the protocol")
        return indices

    def ledger(self) -> dict[str, Any]:
        ledger = _read_json(self.ledger_path)
        if ledger.get("schema") != LEDGER_SCHEMA:
            raise AutoResearchError("ledger schema is invalid")
        _validate_ledger_integrity(ledger)
        return ledger

    def _verify_referenced_artifacts(
        self,
        ledger: Mapping[str, Any],
        protocol: Mapping[str, Any],
    ) -> None:
        """Fail closed if any artifact already committed to the ledger drifted."""

        for round_key, round_row in ledger["rounds"].items():
            round_index = int(round_key)
            evidences: list[dict[str, Any]] = []
            actions: dict[str, dict[str, Any]] = {}
            for candidate_id, candidate in round_row["candidates"].items():
                action = validate_action(
                    _read_json(Path(candidate["action_path"])),
                    expected_parent_family_sha256=round_row["parent_family_sha256"],
                    expected_indices_sha256=ledger["search_indices_sha256"],
                )
                if (
                    action["candidate_id"] != candidate_id
                    or action["action_sha256"] != candidate["action_sha256"]
                ):
                    raise AutoResearchError(
                        "candidate action artifact does not match the durable ledger"
                    )
                actions[candidate_id] = action
                evidence_sha256 = candidate.get("evidence_sha256")
                if evidence_sha256 is not None:
                    evidence_path = Path(candidate["action_path"]).with_name(
                        "evidence.json"
                    )
                    evidence = _read_json_with_digest(
                        evidence_path,
                        expected_sha256=evidence_sha256,
                        context="candidate evidence artifact",
                    )
                    evidences.append(
                        validate_search_evidence(
                            evidence,
                            action=action,
                            protocol=protocol,
                            expected_indices_sha256=ledger["search_indices_sha256"],
                        )
                    )
            if round_row["status"] == "complete":
                report_path = (
                    self.run_root
                    / "rounds"
                    / f"round_{round_index:03d}"
                    / "adjudication.json"
                )
                report = _read_json_with_digest(
                    report_path,
                    expected_sha256=round_row["adjudication_sha256"],
                    context="round adjudication artifact",
                )
                expected_report = adjudicate_round(
                    round_index=round_index,
                    actions=actions,
                    evidences=evidences,
                    protocol=protocol,
                )
                if report != expected_report:
                    raise AutoResearchError(
                        "round adjudication artifact is not reproducible"
                    )

        frozen = ledger.get("frozen")
        if frozen is not None:
            frozen_artifact = _read_json(self.run_root / "frozen_candidate.json")
            unhashed = {
                key: value
                for key, value in frozen_artifact.items()
                if key != "freeze_sha256"
            }
            if frozen_artifact != frozen or frozen_artifact.get(
                "freeze_sha256"
            ) != digest_value(unhashed):
                raise AutoResearchError(
                    "frozen candidate artifact does not match the durable ledger"
                )

        replay = ledger.get("complex128_replay")
        if replay is not None:
            replay_artifact = _read_json_with_digest(
                Path(replay["path"]),
                expected_sha256=replay["evidence_sha256"],
                context="complex128 replay artifact",
            )
            validate_replay_evidence(
                replay_artifact,
                frozen=frozen,
                protocol=protocol,
            )

        shadow = ledger.get("shadow")
        if shadow is not None:
            claim = shadow["claim"]
            claim_artifact = _read_json_with_digest(
                Path(claim["path"]),
                expected_sha256=claim["claim_sha256"],
                context="shadow claim artifact",
            )
            validate_shadow_claim(claim_artifact, frozen=frozen)
            result = shadow.get("result")
            if result is not None:
                evidence = _read_json_with_digest(
                    Path(result["evidence_path"]),
                    expected_sha256=result["evidence_sha256"],
                    context="shadow evidence artifact",
                )
                normalized = validate_shadow_evidence(
                    evidence,
                    frozen=frozen,
                    protocol=protocol,
                )
                if any(
                    normalized[key] != claim_artifact[key]
                    for key in (
                        "shadow_dataset_sha256",
                        "shadow_indices_sha256",
                        "evaluator_sha256",
                    )
                ):
                    raise AutoResearchError(
                        "shadow evidence does not match the durable claim"
                    )
                adjudication = _read_json_with_digest(
                    Path(result["adjudication_path"]),
                    expected_sha256=result["adjudication_sha256"],
                    context="shadow adjudication artifact",
                )
                if adjudication != adjudicate_shadow(normalized, protocol=protocol):
                    raise AutoResearchError(
                        "shadow adjudication artifact is not reproducible"
                    )

    def status(self) -> dict[str, Any]:
        """Return a ledger only after auditing all committed artifacts."""

        with self._locked() as ledger:
            return copy.deepcopy(ledger)

    @contextmanager
    def _locked(self) -> Iterator[dict[str, Any]]:
        if not (self.run_root / ROOT_SENTINEL).is_file():
            raise AutoResearchError("run root sentinel is missing")
        handle = self.lock_path.open("a+", encoding="utf-8")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            ledger = self.ledger()
            protocol = _read_json(self.protocol_path)
            if digest_value(protocol) != ledger["protocol_sha256"]:
                raise AutoResearchError("locked protocol was modified")
            indices = _read_json(self.indices_path)
            payload = {
                key: value for key, value in indices.items() if key != "indices_sha256"
            }
            if (
                indices.get("indices_sha256") != ledger["search_indices_sha256"]
                or digest_value(payload) != indices.get("indices_sha256")
                or indices != fixed_search_indices(protocol)
            ):
                raise AutoResearchError("fixed search indices were modified")
            self._verify_referenced_artifacts(ledger, protocol)
            yield ledger
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

    def _commit(
        self,
        ledger: dict[str, Any],
        *,
        event: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        ledger["revision"] = int(ledger["revision"]) + 1
        row = _event_row(
            revision=ledger["revision"],
            event=event,
            previous_event_sha256=ledger["events"][-1]["event_sha256"],
            details=details,
        )
        ledger["events"].append(row)
        _seal_ledger(ledger)
        atomic_write_json(self.ledger_path, ledger)

    def register_action(self, value: Any) -> dict[str, Any]:
        with self._locked() as ledger:
            if ledger["status"] != "active":
                raise AutoResearchError("cannot register actions after freeze")
            protocol = _read_json(self.protocol_path)
            action = validate_action(
                value,
                expected_parent_family_sha256=ledger["champion_family"][
                    "family_sha256"
                ],
                expected_indices_sha256=ledger["search_indices_sha256"],
            )
            round_index = int(action["round"])
            if round_index > protocol["maximum_rounds"]:
                raise AutoResearchError("action exceeds maximum_rounds")
            expected_round = int(ledger["current_round"]) + 1
            if round_index != expected_round:
                raise AutoResearchError(f"next open round is {expected_round}")
            key = str(round_index)
            round_row = ledger["rounds"].setdefault(
                key,
                {
                    "status": "open",
                    "parent_family_sha256": ledger["champion_family"]["family_sha256"],
                    "candidates": {},
                    "adjudication_sha256": None,
                    "winner_candidate_id": None,
                },
            )
            candidate_id = action["candidate_id"]
            existing = round_row["candidates"].get(candidate_id)
            if existing is not None:
                if existing["action_sha256"] != action["action_sha256"]:
                    raise AutoResearchError(
                        "candidate_id is already registered differently"
                    )
                action_path = Path(existing["action_path"])
                if not action_path.is_file() or _read_json(action_path) != action:
                    raise AutoResearchError(
                        "registered candidate action artifact is missing or modified"
                    )
                return action
            candidate_dir = (
                self.run_root
                / "rounds"
                / f"round_{round_index:03d}"
                / "candidates"
                / candidate_id
            )
            action_path = candidate_dir / "action.json"
            _publish_create_only_json(
                action_path,
                action,
                context="candidate action artifact",
            )
            round_row["candidates"][candidate_id] = {
                "action_sha256": action["action_sha256"],
                "action_path": str(action_path),
                "status": "registered",
                "evidence_sha256": None,
            }
            self._commit(
                ledger,
                event="action_registered",
                details={"round": round_index, "candidate_id": candidate_id},
            )
            return action

    def record_search_evidence(self, value: Any) -> dict[str, Any]:
        """Persist one completed candidate independently of round adjudication."""

        if not isinstance(value, dict):
            raise AutoResearchError("candidate evidence must be an object")
        with self._locked() as ledger:
            if ledger["status"] != "active":
                raise AutoResearchError("cannot record search evidence after freeze")
            candidate_id = str(value.get("candidate_id", ""))
            round_index = value.get("round")
            if isinstance(round_index, bool) or not isinstance(round_index, int):
                raise AutoResearchError("candidate evidence round must be an integer")
            round_row = ledger["rounds"].get(str(round_index))
            if round_row is None or candidate_id not in round_row["candidates"]:
                raise AutoResearchError("candidate evidence is not registered")
            candidate = round_row["candidates"][candidate_id]
            action = validate_action(
                _read_json(Path(candidate["action_path"])),
                expected_parent_family_sha256=round_row["parent_family_sha256"],
                expected_indices_sha256=ledger["search_indices_sha256"],
            )
            normalized = validate_search_evidence(
                value,
                action=action,
                protocol=_read_json(self.protocol_path),
                expected_indices_sha256=ledger["search_indices_sha256"],
            )
            evidence_sha256 = digest_value(normalized)
            evidence_path = Path(candidate["action_path"]).with_name("evidence.json")
            if candidate["evidence_sha256"] is not None:
                if candidate["evidence_sha256"] != evidence_sha256:
                    raise AutoResearchError(
                        "candidate evidence conflicts with the durable ledger"
                    )
                if (
                    not evidence_path.is_file()
                    or _read_json(evidence_path) != normalized
                ):
                    raise AutoResearchError(
                        "candidate evidence artifact is missing or modified"
                    )
                return normalized
            _publish_create_only_json(
                evidence_path,
                normalized,
                context="candidate evidence artifact",
            )
            candidate["evidence_sha256"] = evidence_sha256
            candidate["status"] = "evidence-recorded"
            self._commit(
                ledger,
                event="candidate_evidence_recorded",
                details={"round": round_index, "candidate_id": candidate_id},
            )
            return normalized

    def adjudicate_round(
        self,
        round_index: int,
        evidence_values: Sequence[Any] = (),
    ) -> dict[str, Any]:
        with self._locked() as ledger:
            if ledger["status"] != "active":
                raise AutoResearchError("cannot adjudicate after freeze")
            key = str(round_index)
            if key not in ledger["rounds"]:
                raise AutoResearchError("round is not registered")
            round_row = ledger["rounds"][key]
            report_path = (
                self.run_root
                / "rounds"
                / f"round_{round_index:03d}"
                / "adjudication.json"
            )
            if round_row["status"] == "complete":
                report = _read_json_with_digest(
                    report_path,
                    expected_sha256=round_row["adjudication_sha256"],
                    context="completed round adjudication",
                )
                for candidate in round_row["candidates"].values():
                    action = validate_action(
                        _read_json(Path(candidate["action_path"])),
                        expected_parent_family_sha256=round_row["parent_family_sha256"],
                        expected_indices_sha256=ledger["search_indices_sha256"],
                    )
                    if action["action_sha256"] != candidate["action_sha256"]:
                        raise AutoResearchError(
                            "completed candidate action does not match the ledger"
                        )
                    evidence_sha256 = candidate.get("evidence_sha256")
                    if not isinstance(evidence_sha256, str):
                        raise AutoResearchError(
                            "completed round lacks candidate evidence provenance"
                        )
                    _read_json_with_digest(
                        Path(candidate["action_path"]).with_name("evidence.json"),
                        expected_sha256=evidence_sha256,
                        context="completed candidate evidence",
                    )
                if not evidence_values:
                    return report
                if len(evidence_values) != len(round_row["candidates"]):
                    raise AutoResearchError("completed round evidence set has changed")
                protocol = _read_json(self.protocol_path)
                seen = set()
                for raw in evidence_values:
                    if not isinstance(raw, dict):
                        raise AutoResearchError("candidate evidence must be an object")
                    candidate_id = str(raw.get("candidate_id", ""))
                    if (
                        candidate_id in seen
                        or candidate_id not in round_row["candidates"]
                    ):
                        raise AutoResearchError(
                            "completed round evidence set has changed"
                        )
                    seen.add(candidate_id)
                    candidate = round_row["candidates"][candidate_id]
                    action = validate_action(
                        _read_json(Path(candidate["action_path"])),
                        expected_parent_family_sha256=round_row["parent_family_sha256"],
                        expected_indices_sha256=ledger["search_indices_sha256"],
                    )
                    normalized = validate_search_evidence(
                        raw,
                        action=action,
                        protocol=protocol,
                        expected_indices_sha256=ledger["search_indices_sha256"],
                    )
                    if digest_value(normalized) != candidate["evidence_sha256"]:
                        raise AutoResearchError(
                            "completed round evidence conflicts with the ledger"
                        )
                return report
            protocol = _read_json(self.protocol_path)
            actions: dict[str, dict[str, Any]] = {}
            for candidate_id, candidate in round_row["candidates"].items():
                action = _read_json(Path(candidate["action_path"]))
                actions[candidate_id] = validate_action(
                    action,
                    expected_parent_family_sha256=round_row["parent_family_sha256"],
                    expected_indices_sha256=ledger["search_indices_sha256"],
                )
            evidences = []
            seen = set()
            supplied = {
                str(raw.get("candidate_id", "")): raw
                for raw in evidence_values
                if isinstance(raw, dict)
            }
            if len(supplied) != len(evidence_values):
                raise AutoResearchError("candidate evidence IDs must be unique")
            if supplied and set(supplied) != set(actions):
                raise AutoResearchError(
                    "every registered candidate needs one evidence file"
                )
            for candidate_id in sorted(actions):
                raw = supplied.get(candidate_id)
                if raw is None:
                    evidence_path = Path(
                        round_row["candidates"][candidate_id]["action_path"]
                    ).with_name("evidence.json")
                    if not evidence_path.is_file():
                        raise AutoResearchError(
                            "every registered candidate needs recorded evidence"
                        )
                    raw = _read_json(evidence_path)
                if not isinstance(raw, dict):
                    raise AutoResearchError("candidate evidence must be an object")
                if candidate_id in seen or candidate_id not in actions:
                    raise AutoResearchError(
                        "evidence candidate set does not match the round"
                    )
                seen.add(candidate_id)
                evidence = validate_search_evidence(
                    raw,
                    action=actions[candidate_id],
                    protocol=protocol,
                    expected_indices_sha256=ledger["search_indices_sha256"],
                )
                evidence_path = Path(
                    round_row["candidates"][candidate_id]["action_path"]
                ).with_name("evidence.json")
                _publish_create_only_json(
                    evidence_path,
                    evidence,
                    context="candidate evidence artifact",
                )
                evidence_sha256 = digest_value(evidence)
                candidate = round_row["candidates"][candidate_id]
                recorded_sha256 = candidate.get("evidence_sha256")
                if recorded_sha256 is not None and recorded_sha256 != evidence_sha256:
                    raise AutoResearchError(
                        "candidate evidence conflicts with the durable ledger"
                    )
                candidate["evidence_sha256"] = evidence_sha256
                candidate["status"] = "evidence-recorded"
                evidences.append(evidence)
            report = adjudicate_round(
                round_index=round_index,
                actions=actions,
                evidences=evidences,
                protocol=protocol,
            )
            _publish_create_only_json(
                report_path,
                report,
                context="round adjudication artifact",
            )
            winner = report["winner_candidate_id"]
            for candidate_id, candidate in round_row["candidates"].items():
                candidate["status"] = (
                    "promoted" if candidate_id == winner else "rejected"
                )
            round_row["status"] = "complete"
            round_row["adjudication_sha256"] = digest_value(report)
            round_row["winner_candidate_id"] = winner
            ledger["current_round"] = round_index
            if winner is not None:
                ledger["champion_family"] = report["winner_family"]
                ledger["champion_comparator_family"] = report[
                    "winner_comparator_family"
                ]
                ledger["champion_candidate_id"] = winner
                ledger["promotion_count"] = int(ledger["promotion_count"]) + 1
            self._commit(
                ledger,
                event="round_adjudicated",
                details={"round": round_index, "winner_candidate_id": winner},
            )
            return report

    def freeze(self) -> dict[str, Any]:
        with self._locked() as ledger:
            if ledger["status"] in {
                "frozen",
                "replayed",
                "shadow-claimed",
                "complete",
            }:
                frozen_path = self.run_root / "frozen_candidate.json"
                frozen = _read_json(frozen_path)
                unhashed = {
                    key: item for key, item in frozen.items() if key != "freeze_sha256"
                }
                if frozen != ledger["frozen"] or frozen.get(
                    "freeze_sha256"
                ) != digest_value(unhashed):
                    raise AutoResearchError(
                        "frozen candidate artifact does not match the ledger"
                    )
                return frozen
            if ledger["status"] != "active":
                raise AutoResearchError(
                    "campaign cannot be frozen in its current state"
                )
            if int(ledger["promotion_count"]) < 1:
                raise AutoResearchError("no structural candidate passed promotion")
            if any(row["status"] != "complete" for row in ledger["rounds"].values()):
                raise AutoResearchError("cannot freeze while a round is open")
            if ledger["champion_comparator_family"] is None:
                raise AutoResearchError("frozen finalist lacks its matched comparator")
            frozen_core = {
                "schema": FROZEN_CANDIDATE_SCHEMA,
                "campaign_id": ledger["campaign_id"],
                "protocol_sha256": ledger["protocol_sha256"],
                "search_indices_sha256": ledger["search_indices_sha256"],
                "candidate_id": ledger["champion_candidate_id"],
                "champion_family": ledger["champion_family"],
                "comparator_family": ledger["champion_comparator_family"],
                "frozen_after_round": ledger["current_round"],
            }
            frozen_path = self.run_root / "frozen_candidate.json"
            if frozen_path.exists():
                frozen = _read_json(frozen_path)
                observed_core = {
                    key: value
                    for key, value in frozen.items()
                    if key not in {"frozen_utc", "freeze_sha256"}
                }
                unhashed = {
                    key: value
                    for key, value in frozen.items()
                    if key != "freeze_sha256"
                }
                if observed_core != frozen_core or frozen.get(
                    "freeze_sha256"
                ) != digest_value(unhashed):
                    raise AutoResearchError(
                        "pre-existing frozen candidate conflicts with the ledger"
                    )
            else:
                frozen = {**frozen_core, "frozen_utc": utc_now()}
                frozen["freeze_sha256"] = digest_value(frozen)
                _publish_create_only_json(
                    frozen_path,
                    frozen,
                    context="frozen candidate artifact",
                )
            ledger["frozen"] = frozen
            ledger["status"] = "frozen"
            self._commit(ledger, event="candidate_frozen")
            return frozen

    def record_complex128_replay(self, value: Any) -> dict[str, Any]:
        with self._locked() as ledger:
            if ledger["status"] in {"replayed", "shadow-claimed", "complete"}:
                existing = ledger["complex128_replay"]
                normalized = validate_replay_evidence(
                    value,
                    frozen=ledger["frozen"],
                    protocol=_read_json(self.protocol_path),
                )
                if existing["evidence_sha256"] != digest_value(normalized):
                    raise AutoResearchError("complex128 replay is already consumed")
                return _read_json_with_digest(
                    self.run_root / "complex128_replay.json",
                    expected_sha256=existing["evidence_sha256"],
                    context="complex128 replay artifact",
                )
            if ledger["status"] != "frozen":
                raise AutoResearchError("freeze the finalist before complex128 replay")
            normalized = validate_replay_evidence(
                value,
                frozen=ledger["frozen"],
                protocol=_read_json(self.protocol_path),
            )
            path = self.run_root / "complex128_replay.json"
            _publish_create_only_json(
                path,
                normalized,
                context="complex128 replay artifact",
            )
            ledger["complex128_replay"] = {
                "path": str(path),
                "evidence_sha256": digest_value(normalized),
            }
            ledger["status"] = "replayed"
            self._commit(ledger, event="complex128_replay_recorded")
            return normalized

    def claim_shadow(self, value: Any) -> dict[str, Any]:
        """Durably bind the one shadow pool before any evaluator opens it."""

        with self._locked() as ledger:
            if ledger["status"] in {"shadow-claimed", "complete"}:
                normalized = validate_shadow_claim(
                    value,
                    frozen=ledger["frozen"],
                )
                existing = ledger["shadow"]["claim"]
                if existing["claim_sha256"] != digest_value(normalized):
                    raise AutoResearchError(
                        "the single shadow claim is already consumed"
                    )
                return _read_json_with_digest(
                    self.run_root / "shadow_claim.json",
                    expected_sha256=existing["claim_sha256"],
                    context="shadow claim artifact",
                )
            if ledger["status"] != "replayed":
                raise AutoResearchError(
                    "shadow can be claimed only after freeze and complex128 replay"
                )
            normalized = validate_shadow_claim(value, frozen=ledger["frozen"])
            claim_path = self.run_root / "shadow_claim.json"
            _publish_create_only_json(
                claim_path,
                normalized,
                context="shadow claim artifact",
            )
            ledger["shadow"] = {
                "claim": {
                    "path": str(claim_path),
                    "claim_sha256": digest_value(normalized),
                    "claimed_utc": utc_now(),
                },
                "result": None,
            }
            ledger["status"] = "shadow-claimed"
            self._commit(ledger, event="shadow_claimed")
            return normalized

    def record_shadow(self, value: Any) -> dict[str, Any]:
        with self._locked() as ledger:
            if ledger["status"] == "complete":
                normalized = validate_shadow_evidence(
                    value,
                    frozen=ledger["frozen"],
                    protocol=_read_json(self.protocol_path),
                )
                claim = _read_json_with_digest(
                    self.run_root / "shadow_claim.json",
                    expected_sha256=ledger["shadow"]["claim"]["claim_sha256"],
                    context="shadow claim artifact",
                )
                if (
                    normalized["shadow_dataset_sha256"]
                    != claim["shadow_dataset_sha256"]
                    or normalized["shadow_indices_sha256"]
                    != claim["shadow_indices_sha256"]
                    or normalized["evaluator_sha256"] != claim["evaluator_sha256"]
                ):
                    raise AutoResearchError(
                        "the single shadow evaluation is already consumed; "
                        "new evidence does not match its durable claim"
                    )
                result = ledger["shadow"]["result"]
                if result["evidence_sha256"] != digest_value(normalized):
                    raise AutoResearchError(
                        "the single shadow evaluation is already consumed"
                    )
                _read_json_with_digest(
                    self.run_root / "shadow_evidence.json",
                    expected_sha256=result["evidence_sha256"],
                    context="shadow evidence artifact",
                )
                return _read_json_with_digest(
                    self.run_root / "shadow_adjudication.json",
                    expected_sha256=result["adjudication_sha256"],
                    context="shadow adjudication artifact",
                )
            if ledger["status"] != "shadow-claimed":
                raise AutoResearchError(
                    "claim the single shadow pool before recording its evaluation"
                )
            protocol = _read_json(self.protocol_path)
            normalized = validate_shadow_evidence(
                value,
                frozen=ledger["frozen"],
                protocol=protocol,
            )
            claim = _read_json_with_digest(
                self.run_root / "shadow_claim.json",
                expected_sha256=ledger["shadow"]["claim"]["claim_sha256"],
                context="shadow claim artifact",
            )
            if (
                normalized["shadow_dataset_sha256"] != claim["shadow_dataset_sha256"]
                or normalized["shadow_indices_sha256"] != claim["shadow_indices_sha256"]
                or normalized["evaluator_sha256"] != claim["evaluator_sha256"]
            ):
                raise AutoResearchError(
                    "shadow evidence does not match the durable pre-evaluation claim"
                )
            evidence_path = self.run_root / "shadow_evidence.json"
            report_path = self.run_root / "shadow_adjudication.json"
            _publish_create_only_json(
                evidence_path,
                normalized,
                context="shadow evidence artifact",
            )
            report = adjudicate_shadow(normalized, protocol=protocol)
            _publish_create_only_json(
                report_path,
                report,
                context="shadow adjudication artifact",
            )
            ledger["shadow"]["result"] = {
                "evidence_path": str(evidence_path),
                "evidence_sha256": digest_value(normalized),
                "adjudication_path": str(report_path),
                "adjudication_sha256": digest_value(report),
                "passes": report["shadow_passes"],
                "consumed_utc": utc_now(),
            }
            ledger["status"] = "complete"
            self._commit(
                ledger,
                event="shadow_consumed",
                details={"shadow_passes": report["shadow_passes"]},
            )
            return report


__all__ = [
    "ACTION_SCHEMA",
    "AutoResearchError",
    "CampaignStore",
    "FINALIST_REPLAY_PRECISION",
    "PROTOCOL_SCHEMA",
    "REPLAY_SCHEMA",
    "SEARCH_EVIDENCE_SCHEMA",
    "SEARCH_PRECISION",
    "SHADOW_CLAIM_SCHEMA",
    "SHADOW_EVIDENCE_SCHEMA",
    "adjudicate_candidate",
    "adjudicate_round",
    "adjudicate_shadow",
    "digest_value",
    "fixed_search_indices",
    "sha256_file",
    "validate_action",
    "validate_protocol",
    "validate_shadow_claim",
]
