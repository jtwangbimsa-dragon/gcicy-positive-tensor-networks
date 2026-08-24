"""Durable v2 manager for the staged quintic architecture research queue.

This module deliberately manages experiments rather than pretending to be a
numerical adapter.  It can audit and consume the existing executable Round 1
campaign, distinguish a scientific rejection from a technical failure, bind
the resulting three-seed champion lineage, and materialize the preregistered
next proposal.  Recipes whose numerical adapters do not yet exist remain
explicitly ``blocked-by-adapter`` and cannot be launched through this module.

No search action accepts a command, an environment, or a data path.  The
catalog permits only the fixed development-data contract and contains no
confirmation or blind evaluator entry point.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import re
from typing import Any, Iterator, Mapping

from .architecture_auto_research import (
    AutoResearchError,
    CampaignStore,
    atomic_write_json,
    digest_value,
    sha256_file,
)
from .quintic_architecture_round1_bridge import (
    BRIDGE_LEDGER_SCHEMA,
    BRIDGE_PLAN_SCHEMA,
    BRIDGE_SENTINEL,
)


CATALOG_SCHEMA = "gcicy-quintic-architecture-multi-round-catalog-v2"
MANAGER_SCHEMA = "gcicy-quintic-architecture-multi-round-manager-v2"
ROUND1_DECISION_SCHEMA = "gcicy-quintic-architecture-round1-decision-v2"
QUEUE_SCHEMA = "gcicy-quintic-architecture-research-queue-v2"
PROPOSAL_SCHEMA = "gcicy-quintic-architecture-proposal-v2"
ROUND_RESULT_SCHEMA = "gcicy-quintic-architecture-round-result-v1"
EXECUTION_HANDOFF_SCHEMA = "gcicy-quintic-architecture-execution-handoff-v1"
ROOT_SENTINEL = ".gcicy-quintic-architecture-multi-round-v2-root"

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_RECIPE_KEYS = frozenset(
    {
        "argv",
        "command",
        "cwd",
        "environment",
        "executable",
        "input_path",
        "output_path",
        "script",
        "shell",
    }
)
_TECHNICAL_STAGE_STATES = frozenset({"failed", "incomplete-output"})
_REGISTERED_DATA_CONTRACT_SHA256 = (
    "c5f6498c35d96df8afabd295c95c26a67fb13d7791a45a576612f1e6937d5751"
)
_EXECUTABLE_ADAPTERS = {
    "quintic-paired-parameter-scope-v2": (3, "parameter-scope"),
    "quintic-paired-optimizer-path-v2": (4, "optimizer-path"),
}


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


def _exact_keys(
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
            f"{context} fields are not exact; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )


def _identifier(value: Any, *, context: str) -> str:
    result = str(value)
    if not _IDENTIFIER.fullmatch(result):
        raise AutoResearchError(f"{context} is not a safe identifier")
    return result


def _positive_int(value: Any, *, context: str, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AutoResearchError(f"{context} must be an integer")
    if value < 0 or (value == 0 and not allow_zero):
        raise AutoResearchError(f"{context} is outside its registered range")
    return value


def _positive_float(value: Any, *, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise AutoResearchError(f"{context} must be numeric") from error
    if not 0.0 < result < float("inf"):
        raise AutoResearchError(f"{context} must be finite and positive")
    return result


def _nonnegative_float(value: Any, *, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise AutoResearchError(f"{context} must be numeric") from error
    if not 0.0 <= result < float("inf"):
        raise AutoResearchError(f"{context} must be finite and nonnegative")
    return result


def _sha256(value: Any, *, context: str) -> str:
    result = str(value)
    if not _SHA256.fullmatch(result):
        raise AutoResearchError(f"{context} must be a lowercase SHA-256")
    return result


def _reject_recipe_escape_hatches(value: Any, *, context: str) -> None:
    if isinstance(value, dict):
        forbidden = _FORBIDDEN_RECIPE_KEYS.intersection(value)
        if forbidden:
            raise AutoResearchError(
                f"{context} contains execution/path escape hatches: {sorted(forbidden)}"
            )
        for key, item in value.items():
            _reject_recipe_escape_hatches(item, context=f"{context}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_recipe_escape_hatches(item, context=f"{context}[{index}]")


def _normalize_budget(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AutoResearchError(f"{context} must be an object")
    _exact_keys(
        value,
        required={
            "local_activate_optimizer_updates",
            "matched_relax_optimizer_updates_per_arm",
            "batch_size",
            "learning_rate",
            "gradient_clip_norm",
            "fit_examples",
            "selection_examples",
            "development_evaluation_examples",
        },
        context=context,
    )
    return {
        "local_activate_optimizer_updates": _positive_int(
            value["local_activate_optimizer_updates"],
            context=f"{context}.local_activate_optimizer_updates",
            allow_zero=True,
        ),
        "matched_relax_optimizer_updates_per_arm": _positive_int(
            value["matched_relax_optimizer_updates_per_arm"],
            context=f"{context}.matched_relax_optimizer_updates_per_arm",
        ),
        "batch_size": _positive_int(
            value["batch_size"], context=f"{context}.batch_size"
        ),
        "learning_rate": _positive_float(
            value["learning_rate"], context=f"{context}.learning_rate"
        ),
        "gradient_clip_norm": _positive_float(
            value["gradient_clip_norm"], context=f"{context}.gradient_clip_norm"
        ),
        "fit_examples": _positive_int(
            value["fit_examples"], context=f"{context}.fit_examples"
        ),
        "selection_examples": _positive_int(
            value["selection_examples"], context=f"{context}.selection_examples"
        ),
        "development_evaluation_examples": _positive_int(
            value["development_evaluation_examples"],
            context=f"{context}.development_evaluation_examples",
        ),
    }


def _normalize_gate(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AutoResearchError(f"{context} must be an object")
    required = {
        "minimum_median_relative_sigma_gain",
        "minimum_median_relative_chi_gain",
        "minimum_improved_seeds",
        "minimum_paired_ci_positive_seeds",
        "maximum_seed_relative_regression",
        "maximum_tail_relative_degradation",
        "require_positive_metric",
    }
    _exact_keys(value, required=required, context=context)
    result = {
        "minimum_median_relative_sigma_gain": _nonnegative_float(
            value["minimum_median_relative_sigma_gain"],
            context=f"{context}.minimum_median_relative_sigma_gain",
        ),
        "minimum_median_relative_chi_gain": _nonnegative_float(
            value["minimum_median_relative_chi_gain"],
            context=f"{context}.minimum_median_relative_chi_gain",
        ),
        "minimum_improved_seeds": _positive_int(
            value["minimum_improved_seeds"],
            context=f"{context}.minimum_improved_seeds",
        ),
        "minimum_paired_ci_positive_seeds": _positive_int(
            value["minimum_paired_ci_positive_seeds"],
            context=f"{context}.minimum_paired_ci_positive_seeds",
        ),
        "maximum_seed_relative_regression": _nonnegative_float(
            value["maximum_seed_relative_regression"],
            context=f"{context}.maximum_seed_relative_regression",
        ),
        "maximum_tail_relative_degradation": _nonnegative_float(
            value["maximum_tail_relative_degradation"],
            context=f"{context}.maximum_tail_relative_degradation",
        ),
        "require_positive_metric": value["require_positive_metric"],
    }
    if (
        result["minimum_improved_seeds"] > 3
        or result["minimum_paired_ci_positive_seeds"] > 3
    ):
        raise AutoResearchError(
            f"{context} seed gates exceed the registered seed count"
        )
    if result["require_positive_metric"] is not True:
        raise AutoResearchError(f"{context} must require a positive metric")
    return result


def _normalize_arm(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AutoResearchError(f"{context} must be an object")
    _exact_keys(
        value,
        required={"trainable_scope", "scheduler", "learning_rate"},
        context=context,
    )
    scope = str(value["trainable_scope"])
    scheduler = str(value["scheduler"])
    if scope not in {"internal", "all"}:
        raise AutoResearchError(f"{context}.trainable_scope is not registered")
    if scheduler not in {"cosine", "constant"}:
        raise AutoResearchError(f"{context}.scheduler is not registered")
    return {
        "trainable_scope": scope,
        "scheduler": scheduler,
        "learning_rate": _positive_float(
            value["learning_rate"], context=f"{context}.learning_rate"
        ),
    }


def _normalize_intervention(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AutoResearchError(f"{context} must be an object")
    kind = str(value.get("kind", ""))
    if kind == "rank-growth":
        _exact_keys(
            value,
            required={
                "kind",
                "edge",
                "source_dimension",
                "target_dimension",
                "structural_maximum",
                "new_total_real_parameters",
                "activation_scale",
                "orthogonalize_new_outputs",
            },
            context=context,
        )
        source = _positive_int(
            value["source_dimension"], context=f"{context}.source_dimension"
        )
        target = _positive_int(
            value["target_dimension"], context=f"{context}.target_dimension"
        )
        maximum = _positive_int(
            value["structural_maximum"], context=f"{context}.structural_maximum"
        )
        added = _positive_int(
            value["new_total_real_parameters"],
            context=f"{context}.new_total_real_parameters",
        )
        if not source < target <= maximum or added > 10_000:
            raise AutoResearchError(
                f"{context} violates the nested-rank capacity envelope"
            )
        if value["orthogonalize_new_outputs"] is not True:
            raise AutoResearchError(f"{context} requires orthogonalized new outputs")
        return {
            "kind": kind,
            "edge": _positive_int(
                value["edge"], context=f"{context}.edge", allow_zero=True
            ),
            "source_dimension": source,
            "target_dimension": target,
            "structural_maximum": maximum,
            "new_total_real_parameters": added,
            "activation_scale": _positive_float(
                value["activation_scale"], context=f"{context}.activation_scale"
            ),
            "orthogonalize_new_outputs": True,
        }
    if kind in {"parameter-scope", "optimizer-path"}:
        _exact_keys(
            value,
            required={"kind", "control", "candidate", "new_total_real_parameters"},
            context=context,
        )
        control = _normalize_arm(value["control"], context=f"{context}.control")
        candidate = _normalize_arm(value["candidate"], context=f"{context}.candidate")
        delta = _positive_int(
            value["new_total_real_parameters"],
            context=f"{context}.new_total_real_parameters",
            allow_zero=True,
        )
        if delta != 0:
            raise AutoResearchError(f"{context} must preserve total parameter count")
        differing = {key for key in control if control[key] != candidate[key]}
        expected = {"trainable_scope"} if kind == "parameter-scope" else {"scheduler"}
        if differing != expected:
            raise AutoResearchError(
                f"{context} must change exactly {sorted(expected)}, observed {sorted(differing)}"
            )
        if kind == "parameter-scope" and (
            control["trainable_scope"],
            candidate["trainable_scope"],
        ) != (
            "internal",
            "all",
        ):
            raise AutoResearchError(
                f"{context} must compare internal control with all-parameter candidate"
            )
        if kind == "optimizer-path" and control["trainable_scope"] != "all":
            raise AutoResearchError(
                f"{context} optimizer comparison must train all parameters"
            )
        return {
            "kind": kind,
            "control": control,
            "candidate": candidate,
            "new_total_real_parameters": 0,
        }
    raise AutoResearchError(f"{context}.kind is not registered")


def _normalize_recipe(value: Any, *, index: int) -> dict[str, Any]:
    context = f"recipes[{index}]"
    if not isinstance(value, dict):
        raise AutoResearchError(f"{context} must be an object")
    _reject_recipe_escape_hatches(value, context=context)
    _exact_keys(
        value,
        required={
            "recipe_id",
            "round",
            "category",
            "hypothesis",
            "precondition",
            "intervention",
            "budget",
            "promotion_gate",
            "adapter",
        },
        optional={"recipe_sha256"},
        context=context,
    )
    recipe_id = _identifier(value["recipe_id"], context=f"{context}.recipe_id")
    round_index = _positive_int(value["round"], context=f"{context}.round")
    category = str(value["category"])
    if category not in {
        "progressive-initialization",
        "parameter-scope",
        "optimizer-path",
    }:
        raise AutoResearchError(f"{context}.category is not registered")
    hypothesis = str(value["hypothesis"])
    if not hypothesis or len(hypothesis) > 400:
        raise AutoResearchError(f"{context}.hypothesis is empty or too long")
    precondition = value["precondition"]
    if not isinstance(precondition, dict):
        raise AutoResearchError(f"{context}.precondition must be an object")
    _exact_keys(
        precondition,
        required={"r1_outcome"} if round_index == 2 else {"previous_round"},
        context=f"{context}.precondition",
    )
    if round_index == 2:
        outcome = str(precondition["r1_outcome"])
        if outcome not in {"promoted", "scientific-rejected"}:
            raise AutoResearchError(f"{context}.precondition.r1_outcome is invalid")
        normalized_precondition = {"r1_outcome": outcome}
    else:
        previous = _positive_int(
            precondition["previous_round"],
            context=f"{context}.precondition.previous_round",
        )
        if previous != round_index - 1:
            raise AutoResearchError(f"{context} does not follow the preceding round")
        normalized_precondition = {"previous_round": previous}
    adapter = value["adapter"]
    if not isinstance(adapter, dict):
        raise AutoResearchError(f"{context}.adapter must be an object")
    _exact_keys(
        adapter,
        required={"status", "required_adapter"},
        context=f"{context}.adapter",
    )
    adapter_status = str(adapter["status"])
    adapter_name = _identifier(
        adapter["required_adapter"],
        context=f"{context}.adapter.required_adapter",
    )
    if adapter_status not in {"blocked-by-adapter", "available"}:
        raise AutoResearchError(f"{context}.adapter.status is invalid")
    if adapter_status == "available" and _EXECUTABLE_ADAPTERS.get(adapter_name) != (
        round_index,
        category,
    ):
        raise AutoResearchError(
            f"{context} cannot claim an executable adapter without the registered "
            "paired implementation"
        )
    normalized = {
        "recipe_id": recipe_id,
        "round": round_index,
        "category": category,
        "hypothesis": hypothesis,
        "precondition": normalized_precondition,
        "intervention": _normalize_intervention(
            value["intervention"], context=f"{context}.intervention"
        ),
        "budget": _normalize_budget(value["budget"], context=f"{context}.budget"),
        "promotion_gate": _normalize_gate(
            value["promotion_gate"], context=f"{context}.promotion_gate"
        ),
        "adapter": {
            "status": adapter_status,
            "required_adapter": adapter_name,
        },
    }
    expected_kind = {
        "progressive-initialization": "rank-growth",
        "parameter-scope": "parameter-scope",
        "optimizer-path": "optimizer-path",
    }[category]
    if normalized["intervention"]["kind"] != expected_kind:
        raise AutoResearchError(
            f"{context}.category does not match its intervention kind"
        )
    recipe_sha256 = digest_value(normalized)
    if "recipe_sha256" in value and value["recipe_sha256"] != recipe_sha256:
        raise AutoResearchError(f"{context}.recipe_sha256 does not match the recipe")
    return {**normalized, "recipe_sha256": recipe_sha256}


def validate_catalog(value: Any) -> dict[str, Any]:
    """Validate and canonicalize the closed v2 hypothesis catalog."""

    if not isinstance(value, dict):
        raise AutoResearchError("catalog must be an object")
    _reject_recipe_escape_hatches(value, context="catalog")
    _exact_keys(
        value,
        required={
            "schema",
            "catalog_id",
            "maximum_rounds",
            "promotion_seeds",
            "data_contract",
            "round1_contract",
            "branches",
            "recipes",
        },
        context="catalog",
    )
    if value["schema"] != CATALOG_SCHEMA:
        raise AutoResearchError(f"catalog schema must be {CATALOG_SCHEMA}")
    catalog_id = _identifier(value["catalog_id"], context="catalog.catalog_id")
    maximum_rounds = _positive_int(
        value["maximum_rounds"], context="catalog.maximum_rounds"
    )
    if maximum_rounds != 4:
        raise AutoResearchError("catalog v2 registers exactly four research rounds")
    seeds = value["promotion_seeds"]
    if not isinstance(seeds, list) or len(seeds) != 3:
        raise AutoResearchError("catalog requires exactly three promotion seeds")
    promotion_seeds = sorted(
        _positive_int(seed, context="catalog.promotion_seeds") for seed in seeds
    )
    if len(set(promotion_seeds)) != 3:
        raise AutoResearchError("catalog promotion seeds must be distinct")

    data = value["data_contract"]
    if not isinstance(data, dict):
        raise AutoResearchError("catalog.data_contract must be an object")
    _exact_keys(
        data,
        required={
            "precision",
            "fit_examples",
            "selection_examples",
            "development_evaluation_examples",
            "protocol_sha256",
            "search_indices_sha256",
            "input_sha256",
            "matched_batch_plans",
            "historical_confirmation",
            "final_blind",
        },
        context="catalog.data_contract",
    )
    input_sha256 = data["input_sha256"]
    if not isinstance(input_sha256, dict):
        raise AutoResearchError("catalog.data_contract.input_sha256 must be an object")
    required_input_roles = {
        "search-native-points",
        "search-native-pullbacks",
        "development-evaluation-points",
        "development-evaluation-pullbacks",
    }
    _exact_keys(
        input_sha256,
        required=required_input_roles,
        context="catalog.data_contract.input_sha256",
    )
    raw_batch_plans = data["matched_batch_plans"]
    if not isinstance(raw_batch_plans, list) or len(raw_batch_plans) != 3:
        raise AutoResearchError("catalog requires three fixed matched batch plans")
    matched_batch_plans = []
    for index, row in enumerate(raw_batch_plans):
        if not isinstance(row, dict):
            raise AutoResearchError("catalog matched batch plan must be an object")
        _exact_keys(
            row,
            required={"seed", "plan_seed", "file_sha256", "value_set_sha256"},
            context=f"catalog.data_contract.matched_batch_plans[{index}]",
        )
        matched_batch_plans.append(
            {
                "seed": _positive_int(
                    row["seed"],
                    context=f"data.matched_batch_plans[{index}].seed",
                ),
                "plan_seed": _positive_int(
                    row["plan_seed"],
                    context=f"data.matched_batch_plans[{index}].plan_seed",
                ),
                "file_sha256": _sha256(
                    row["file_sha256"],
                    context=f"data.matched_batch_plans[{index}].file_sha256",
                ),
                "value_set_sha256": _sha256(
                    row["value_set_sha256"],
                    context=f"data.matched_batch_plans[{index}].value_set_sha256",
                ),
            }
        )
    matched_batch_plans.sort(key=lambda row: row["seed"])
    if [row["seed"] for row in matched_batch_plans] != promotion_seeds:
        raise AutoResearchError("matched batch plans do not cover the promotion seeds")
    normalized_data = {
        "precision": str(data["precision"]),
        "fit_examples": _positive_int(
            data["fit_examples"], context="data.fit_examples"
        ),
        "selection_examples": _positive_int(
            data["selection_examples"], context="data.selection_examples"
        ),
        "development_evaluation_examples": _positive_int(
            data["development_evaluation_examples"],
            context="data.development_evaluation_examples",
        ),
        "protocol_sha256": _sha256(
            data["protocol_sha256"], context="data.protocol_sha256"
        ),
        "search_indices_sha256": _sha256(
            data["search_indices_sha256"], context="data.search_indices_sha256"
        ),
        "input_sha256": {
            role: _sha256(input_sha256[role], context=f"data.input_sha256.{role}")
            for role in sorted(required_input_roles)
        },
        "matched_batch_plans": matched_batch_plans,
        "historical_confirmation": str(data["historical_confirmation"]),
        "final_blind": str(data["final_blind"]),
    }
    if (
        normalized_data["precision"] != "complex64"
        or normalized_data["fit_examples"] != 30_000
        or normalized_data["selection_examples"] != 5_000
        or normalized_data["development_evaluation_examples"] != 5_000
        or normalized_data["historical_confirmation"] != "absent"
        or normalized_data["final_blind"] != "absent"
    ):
        raise AutoResearchError(
            "catalog development-data contract is not the registered one"
        )
    if digest_value(normalized_data) != _REGISTERED_DATA_CONTRACT_SHA256:
        raise AutoResearchError("catalog data/index/batch hashes are not registered")

    r1 = value["round1_contract"]
    if not isinstance(r1, dict):
        raise AutoResearchError("catalog.round1_contract must be an object")
    _exact_keys(
        r1,
        required={
            "campaign_id",
            "candidate_id",
            "baseline_checkpoint_sha256",
            "mutation",
            "pipeline",
            "adapter",
        },
        context="catalog.round1_contract",
    )
    mutation = r1["mutation"]
    if not isinstance(mutation, dict):
        raise AutoResearchError("catalog Round 1 mutation must be an object")
    _exact_keys(
        mutation,
        required={
            "kind",
            "target",
            "edge",
            "source_dimension",
            "target_dimension",
            "structural_maximum",
            "new_output_real_parameters",
        },
        context="catalog.round1_contract.mutation",
    )
    normalized_r1_mutation = _normalize_intervention(
        {
            "kind": "rank-growth",
            "edge": mutation.get("edge"),
            "source_dimension": mutation.get("source_dimension"),
            "target_dimension": mutation.get("target_dimension"),
            "structural_maximum": mutation.get("structural_maximum"),
            "new_total_real_parameters": mutation.get("new_output_real_parameters"),
            "activation_scale": 1.0,
            "orthogonalize_new_outputs": True,
        },
        context="catalog.round1_contract.mutation",
    )
    if (
        str(mutation.get("kind")) != "rank"
        or str(mutation.get("target")) != "internal-edge"
    ):
        raise AutoResearchError("catalog Round 1 must be internal-edge rank growth")
    expected_pipeline = [{"kind": "local-activate"}, {"kind": "matched-relax"}]
    if (
        r1["pipeline"] != expected_pipeline
        or r1["adapter"] != "existing-round1-bridge-v1"
    ):
        raise AutoResearchError(
            "catalog Round 1 adapter/pipeline is not the existing bridge"
        )
    normalized_r1 = {
        "campaign_id": _identifier(r1["campaign_id"], context="round1.campaign_id"),
        "candidate_id": _identifier(r1["candidate_id"], context="round1.candidate_id"),
        "baseline_checkpoint_sha256": _sha256(
            r1["baseline_checkpoint_sha256"],
            context="round1.baseline_checkpoint_sha256",
        ),
        "mutation": {
            "kind": "rank",
            "target": "internal-edge",
            "edge": normalized_r1_mutation["edge"],
            "source_dimension": normalized_r1_mutation["source_dimension"],
            "target_dimension": normalized_r1_mutation["target_dimension"],
            "structural_maximum": normalized_r1_mutation["structural_maximum"],
            "new_output_real_parameters": normalized_r1_mutation[
                "new_total_real_parameters"
            ],
        },
        "pipeline": expected_pipeline,
        "adapter": "existing-round1-bridge-v1",
    }

    raw_recipes = value["recipes"]
    if not isinstance(raw_recipes, list) or len(raw_recipes) != 4:
        raise AutoResearchError(
            "catalog v2 requires two Round-2 branches and Rounds 3/4"
        )
    recipes = [
        _normalize_recipe(row, index=index) for index, row in enumerate(raw_recipes)
    ]
    by_id = {row["recipe_id"]: row for row in recipes}
    if len(by_id) != len(recipes):
        raise AutoResearchError("catalog recipe IDs must be unique")

    branches = value["branches"]
    if not isinstance(branches, dict):
        raise AutoResearchError("catalog.branches must be an object")
    _exact_keys(
        branches,
        required={"r1_promoted", "r1_scientific_rejected"},
        context="catalog.branches",
    )
    normalized_branches: dict[str, list[str]] = {}
    for key, expected_outcome in (
        ("r1_promoted", "promoted"),
        ("r1_scientific_rejected", "scientific-rejected"),
    ):
        branch = branches[key]
        if not isinstance(branch, list) or len(branch) != 3:
            raise AutoResearchError(f"catalog.branches.{key} must contain Rounds 2/3/4")
        ids = [_identifier(item, context=f"catalog.branches.{key}") for item in branch]
        if any(recipe_id not in by_id for recipe_id in ids):
            raise AutoResearchError(
                f"catalog.branches.{key} references an unknown recipe"
            )
        rows = [by_id[recipe_id] for recipe_id in ids]
        if [row["round"] for row in rows] != [2, 3, 4]:
            raise AutoResearchError(
                f"catalog.branches.{key} does not cover Rounds 2/3/4"
            )
        if rows[0]["precondition"] != {"r1_outcome": expected_outcome}:
            raise AutoResearchError(
                f"catalog.branches.{key} selects the wrong Round-2 branch"
            )
        normalized_branches[key] = ids

    normalized = {
        "schema": CATALOG_SCHEMA,
        "catalog_id": catalog_id,
        "maximum_rounds": maximum_rounds,
        "promotion_seeds": promotion_seeds,
        "data_contract": normalized_data,
        "round1_contract": normalized_r1,
        "branches": normalized_branches,
        "recipes": sorted(recipes, key=lambda row: (row["round"], row["recipe_id"])),
    }
    return {**normalized, "catalog_sha256": digest_value(normalized)}


def _event(
    *,
    revision: int,
    event: str,
    previous: str,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "revision": revision,
        "event": event,
        "utc": utc_now(),
        "previous_event_sha256": previous,
        "details": dict(details or {}),
    }
    return {**payload, "event_sha256": digest_value(payload)}


def _seal_state(state: dict[str, Any]) -> None:
    payload = {key: value for key, value in state.items() if key != "state_sha256"}
    state["state_sha256"] = digest_value(payload)


def _validate_state(state: Mapping[str, Any]) -> None:
    _require(state.get("schema") == MANAGER_SCHEMA, "manager state schema is invalid")
    observed = state.get("state_sha256")
    payload = {key: value for key, value in state.items() if key != "state_sha256"}
    _require(observed == digest_value(payload), "manager state integrity check failed")
    events = state.get("events")
    _require(
        isinstance(events, list) and bool(events), "manager event chain is missing"
    )
    previous = "0" * 64
    for index, row in enumerate(events):
        _require(isinstance(row, dict), "manager event row is malformed")
        unhashed = {key: value for key, value in row.items() if key != "event_sha256"}
        _require(
            row.get("revision") == index
            and row.get("previous_event_sha256") == previous
            and row.get("event_sha256") == digest_value(unhashed),
            "manager event chain integrity check failed",
        )
        previous = str(row["event_sha256"])
    _require(state.get("revision") == len(events) - 1, "manager revision is invalid")


def _create_or_verify(path: Path, value: Mapping[str, Any], *, context: str) -> None:
    if path.exists():
        if _read_object(path) != dict(value):
            raise AutoResearchError(f"{context} already exists with different content")
        return
    atomic_write_json(path, dict(value))


def _canonical_artifact_family(
    artifacts: Any,
    *,
    seeds: list[int],
    context: str,
    rehash: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate the path-bearing form of one three-seed model family."""

    _require(
        isinstance(artifacts, list) and len(artifacts) == len(seeds),
        f"{context} must contain exactly the registered seeds",
    )
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(artifacts):
        _require(isinstance(raw, dict), f"{context}[{index}] is malformed")
        _exact_keys(
            raw,
            required={
                "seed",
                "checkpoint_path",
                "checkpoint_sha256",
                "role",
                "source_report_path",
                "source_report_sha256",
            },
            context=f"{context}[{index}]",
        )
        seed = _positive_int(raw["seed"], context=f"{context}[{index}].seed")
        checkpoint_path = Path(str(raw["checkpoint_path"])).expanduser().resolve()
        report_path = Path(str(raw["source_report_path"])).expanduser().resolve()
        checkpoint_sha256 = _sha256(
            raw["checkpoint_sha256"],
            context=f"{context}[{index}].checkpoint_sha256",
        )
        report_sha256 = _sha256(
            raw["source_report_sha256"],
            context=f"{context}[{index}].source_report_sha256",
        )
        if rehash:
            _require(
                checkpoint_path.is_file()
                and sha256_file(checkpoint_path) == checkpoint_sha256,
                f"{context} checkpoint drifted for seed {seed}",
            )
            _require(
                report_path.is_file() and sha256_file(report_path) == report_sha256,
                f"{context} source report drifted for seed {seed}",
            )
        role = _identifier(raw["role"], context=f"{context}[{index}].role")
        rows.append(
            {
                "seed": seed,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha256,
                "role": role,
                "source_report_path": str(report_path),
                "source_report_sha256": report_sha256,
            }
        )
    rows.sort(key=lambda row: row["seed"])
    _require(
        [row["seed"] for row in rows] == seeds,
        f"{context} seeds differ from the registered family",
    )
    family_payload = {
        "models": [
            {"seed": row["seed"], "checkpoint_sha256": row["checkpoint_sha256"]}
            for row in rows
        ]
    }
    return rows, {
        **family_payload,
        "family_sha256": digest_value(family_payload),
    }


def _validate_family(value: Any, *, seeds: list[int], context: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{context} is malformed")
    _exact_keys(
        value,
        required={"models", "family_sha256"},
        context=context,
    )
    models = value["models"]
    _require(
        isinstance(models, list) and len(models) == len(seeds),
        f"{context}.models must contain exactly the registered seeds",
    )
    normalized = []
    for index, row in enumerate(models):
        _require(isinstance(row, dict), f"{context}.models[{index}] is malformed")
        _exact_keys(
            row,
            required={"seed", "checkpoint_sha256"},
            context=f"{context}.models[{index}]",
        )
        normalized.append(
            {
                "seed": _positive_int(
                    row["seed"], context=f"{context}.models[{index}].seed"
                ),
                "checkpoint_sha256": _sha256(
                    row["checkpoint_sha256"],
                    context=f"{context}.models[{index}].checkpoint_sha256",
                ),
            }
        )
    normalized.sort(key=lambda row: row["seed"])
    _require(
        [row["seed"] for row in normalized] == seeds,
        f"{context} seeds differ from the registered family",
    )
    payload = {"models": normalized}
    _require(
        value["family_sha256"] == digest_value(payload),
        f"{context} integrity check failed",
    )
    return {**payload, "family_sha256": value["family_sha256"]}


def _source_lineage(
    *, source_round: int, source_kind: str, source_path: Path, source_value_sha256: str
) -> dict[str, Any]:
    source_path = source_path.expanduser().resolve()
    _require(source_path.is_file(), "proposal lineage source is missing")
    return {
        "source_round": source_round,
        "source_kind": source_kind,
        "source_path": str(source_path),
        "source_file_sha256": sha256_file(source_path),
        "source_value_sha256": source_value_sha256,
    }


def _build_proposal(
    *,
    catalog: Mapping[str, Any],
    recipe: Mapping[str, Any],
    parent_family: Mapping[str, Any],
    parent_artifacts: list[dict[str, Any]],
    parent_lineage: Mapping[str, Any],
) -> dict[str, Any]:
    data_binding = {
        key: catalog["data_contract"][key]
        for key in (
            "protocol_sha256",
            "search_indices_sha256",
            "input_sha256",
            "matched_batch_plans",
        )
    }
    payload = {
        "schema": PROPOSAL_SCHEMA,
        "catalog_sha256": catalog["catalog_sha256"],
        "recipe_id": recipe["recipe_id"],
        "recipe_sha256": recipe["recipe_sha256"],
        "round": recipe["round"],
        "parent_family": dict(parent_family),
        "parent_artifacts": list(parent_artifacts),
        "parent_lineage": dict(parent_lineage),
        "data_binding": data_binding,
        "intervention": recipe["intervention"],
        "budget": recipe["budget"],
        "promotion_gate": recipe["promotion_gate"],
        "adapter": recipe["adapter"],
        "execution_authorized": False,
    }
    return {**payload, "proposal_sha256": digest_value(payload)}


def _validate_proposal(
    value: Any,
    *,
    catalog: Mapping[str, Any],
    expected_recipe_id: str | None = None,
    rehash_parents: bool = True,
) -> dict[str, Any]:
    _require(isinstance(value, dict), "manager proposal is malformed")
    _exact_keys(
        value,
        required={
            "schema",
            "catalog_sha256",
            "recipe_id",
            "recipe_sha256",
            "round",
            "parent_family",
            "parent_artifacts",
            "parent_lineage",
            "data_binding",
            "intervention",
            "budget",
            "promotion_gate",
            "adapter",
            "execution_authorized",
            "proposal_sha256",
        },
        context="manager proposal",
    )
    payload = {key: row for key, row in value.items() if key != "proposal_sha256"}
    _require(
        value["schema"] == PROPOSAL_SCHEMA
        and value["proposal_sha256"] == digest_value(payload),
        "manager proposal integrity check failed",
    )
    recipe_id = _identifier(value["recipe_id"], context="proposal.recipe_id")
    _require(
        expected_recipe_id in {None, recipe_id},
        "manager proposal is not for the expected recipe",
    )
    recipe = next(
        (row for row in catalog["recipes"] if row["recipe_id"] == recipe_id), None
    )
    _require(isinstance(recipe, dict), "manager proposal recipe is not catalogued")
    _require(
        value["catalog_sha256"] == catalog["catalog_sha256"]
        and value["recipe_sha256"] == recipe["recipe_sha256"]
        and value["round"] == recipe["round"]
        and value["intervention"] == recipe["intervention"]
        and value["budget"] == recipe["budget"]
        and value["promotion_gate"] == recipe["promotion_gate"]
        and value["adapter"] == recipe["adapter"]
        and value["execution_authorized"] is False,
        "manager proposal differs from the locked recipe",
    )
    expected_data = {
        key: catalog["data_contract"][key]
        for key in (
            "protocol_sha256",
            "search_indices_sha256",
            "input_sha256",
            "matched_batch_plans",
        )
    }
    _require(
        value["data_binding"] == expected_data,
        "manager proposal data binding differs from the catalog",
    )
    artifacts, family = _canonical_artifact_family(
        value["parent_artifacts"],
        seeds=catalog["promotion_seeds"],
        context="proposal.parent_artifacts",
        rehash=rehash_parents,
    )
    _require(
        _validate_family(
            value["parent_family"],
            seeds=catalog["promotion_seeds"],
            context="proposal.parent_family",
        )
        == family,
        "manager proposal parent paths and family hashes differ",
    )
    lineage = value["parent_lineage"]
    _require(isinstance(lineage, dict), "manager proposal lineage is malformed")
    _exact_keys(
        lineage,
        required={
            "source_round",
            "source_kind",
            "source_path",
            "source_file_sha256",
            "source_value_sha256",
        },
        context="proposal.parent_lineage",
    )
    source_round = _positive_int(
        lineage["source_round"], context="proposal.parent_lineage.source_round"
    )
    _require(
        source_round == recipe["round"] - 1,
        "manager proposal lineage does not come from the preceding round",
    )
    expected_kind = "round1-decision" if source_round == 1 else "round-result"
    _require(
        lineage["source_kind"] == expected_kind,
        "manager proposal lineage kind is invalid",
    )
    source_path = Path(str(lineage["source_path"])).expanduser().resolve()
    _require(
        source_path.is_file()
        and sha256_file(source_path) == lineage["source_file_sha256"],
        "manager proposal lineage source drifted",
    )
    source = _read_object(source_path)
    observed_value_sha256 = (
        source.get("result_sha256")
        if expected_kind == "round-result"
        else digest_value(source)
    )
    _require(
        observed_value_sha256 == lineage["source_value_sha256"],
        "manager proposal lineage value drifted",
    )
    return {
        **payload,
        "parent_family": family,
        "parent_artifacts": artifacts,
        "parent_lineage": {
            **lineage,
            "source_path": str(source_path),
        },
        "proposal_sha256": value["proposal_sha256"],
    }


def _file_reference(path: Path, *, role: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    _require(path.is_file(), f"round result source is missing: {role}")
    return {"role": role, "path": str(path), "sha256": sha256_file(path)}


def _validate_file_references(value: Any, *, context: str) -> list[dict[str, Any]]:
    _require(isinstance(value, list) and bool(value), f"{context} is empty")
    rows = []
    roles: set[str] = set()
    for index, raw in enumerate(value):
        _require(isinstance(raw, dict), f"{context}[{index}] is malformed")
        _exact_keys(
            raw,
            required={"role", "path", "sha256"},
            context=f"{context}[{index}]",
        )
        role = _identifier(raw["role"], context=f"{context}[{index}].role")
        _require(role not in roles, f"{context} roles must be unique")
        roles.add(role)
        path = Path(str(raw["path"])).expanduser().resolve()
        observed = _sha256(raw["sha256"], context=f"{context}[{index}].sha256")
        _require(
            path.is_file() and sha256_file(path) == observed,
            f"round result source drifted: {role}",
        )
        rows.append({"role": role, "path": str(path), "sha256": observed})
    return sorted(rows, key=lambda row: row["role"])


def _build_round_result(
    *,
    proposal: Mapping[str, Any],
    outcome: str,
    recommended_role: str,
    recommended_family: Mapping[str, Any],
    recommended_artifacts: list[dict[str, Any]],
    source_kind: str,
    source_root: Path,
    source_artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = {
        "schema": ROUND_RESULT_SCHEMA,
        "round": proposal["round"],
        "recipe_id": proposal["recipe_id"],
        "recipe_sha256": proposal["recipe_sha256"],
        "proposal_sha256": proposal["proposal_sha256"],
        "outcome": outcome,
        "recommended_role": recommended_role,
        "recommended_family": dict(recommended_family),
        "recommended_artifacts": list(recommended_artifacts),
        "source_kind": source_kind,
        "source_root": str(source_root.expanduser().resolve()),
        "source_artifacts": sorted(source_artifacts, key=lambda row: row["role"]),
    }
    return {**payload, "result_sha256": digest_value(payload)}


def _validate_round_result(
    value: Any,
    *,
    proposal: Mapping[str, Any],
    catalog: Mapping[str, Any],
) -> dict[str, Any]:
    _require(isinstance(value, dict), "round result is malformed")
    _exact_keys(
        value,
        required={
            "schema",
            "round",
            "recipe_id",
            "recipe_sha256",
            "proposal_sha256",
            "outcome",
            "recommended_role",
            "recommended_family",
            "recommended_artifacts",
            "source_kind",
            "source_root",
            "source_artifacts",
            "result_sha256",
        },
        context="round result",
    )
    payload = {key: row for key, row in value.items() if key != "result_sha256"}
    _require(
        value["schema"] == ROUND_RESULT_SCHEMA
        and value["result_sha256"] == digest_value(payload),
        "round result integrity check failed",
    )
    _require(
        value["round"] == proposal["round"]
        and value["recipe_id"] == proposal["recipe_id"]
        and value["recipe_sha256"] == proposal["recipe_sha256"]
        and value["proposal_sha256"] == proposal["proposal_sha256"],
        "round result belongs to another proposal",
    )
    outcome = str(value["outcome"])
    _require(
        outcome in {"promoted", "scientific-rejected"},
        "round result outcome is invalid",
    )
    expected_role = "candidate" if outcome == "promoted" else "parent"
    _require(
        value["recommended_role"] == expected_role,
        "round result recommendation contradicts its outcome",
    )
    artifacts, family = _canonical_artifact_family(
        value["recommended_artifacts"],
        seeds=catalog["promotion_seeds"],
        context="round_result.recommended_artifacts",
    )
    _require(
        _validate_family(
            value["recommended_family"],
            seeds=catalog["promotion_seeds"],
            context="round_result.recommended_family",
        )
        == family,
        "round result recommended paths and family differ",
    )
    if outcome == "scientific-rejected":
        _require(
            family == proposal["parent_family"]
            and artifacts == proposal["parent_artifacts"],
            "scientific rejection must retain the exact proposal parent",
        )
    source_kind = str(value["source_kind"])
    expected_source = (
        "historical-auto-research-not-manager-launched"
        if proposal["round"] == 2
        else "paired-bridge"
    )
    _require(source_kind == expected_source, "round result source kind is invalid")
    source_root = Path(str(value["source_root"])).expanduser().resolve()
    _require(source_root.is_dir(), "round result source root is missing")
    source_artifacts = _validate_file_references(
        value["source_artifacts"], context="round_result.source_artifacts"
    )
    return {
        **payload,
        "recommended_family": family,
        "recommended_artifacts": artifacts,
        "source_root": str(source_root),
        "source_artifacts": source_artifacts,
        "result_sha256": value["result_sha256"],
    }


def _read_handoff(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    value = _read_object(path)
    _exact_keys(
        value,
        required={
            "schema",
            "manager_root",
            "catalog_sha256",
            "manager_revision",
            "manager_state_sha256",
            "manager_event_sha256",
            "round",
            "proposal",
            "recipe_id",
            "recipe_sha256",
            "adapter",
            "parent_family",
            "parent_artifacts",
            "parent_lineage",
            "execution_authorized",
            "launch_policy",
            "handoff_sha256",
        },
        context="execution handoff",
    )
    payload = {key: row for key, row in value.items() if key != "handoff_sha256"}
    _require(
        value["schema"] == EXECUTION_HANDOFF_SCHEMA
        and value["handoff_sha256"] == digest_value(payload),
        "execution handoff integrity check failed",
    )
    return value


def _bridge_snapshot(
    bridge_root: Path, *, expected_candidate_id: str, expected_campaign_root: Path
) -> dict[str, Any]:
    sentinel = bridge_root / BRIDGE_SENTINEL
    plan_path = bridge_root / "plan.json"
    ledger_path = bridge_root / "ledger.json"
    present = [path.exists() for path in (sentinel, plan_path, ledger_path)]
    if not any(present):
        return {"phase": "not-prepared", "technical_failures": []}
    if not all(present):
        raise AutoResearchError("Round 1 bridge is partially initialized")
    _require(
        sentinel.read_text(encoding="utf-8").strip() == expected_candidate_id,
        "Round 1 bridge belongs to another candidate",
    )
    plan = _read_object(plan_path)
    plan_payload = {key: value for key, value in plan.items() if key != "plan_sha256"}
    _require(
        plan.get("schema") == BRIDGE_PLAN_SCHEMA
        and plan.get("plan_sha256") == digest_value(plan_payload),
        "Round 1 bridge plan integrity check failed",
    )
    _require(
        Path(str(plan.get("campaign_run_root", ""))).resolve()
        == expected_campaign_root.resolve()
        and plan.get("candidate_id") == expected_candidate_id,
        "Round 1 bridge plan points at another controller",
    )
    ledger = _read_object(ledger_path)
    ledger_payload = {
        key: value for key, value in ledger.items() if key != "state_sha256"
    }
    _require(
        ledger.get("schema") == BRIDGE_LEDGER_SCHEMA
        and ledger.get("state_sha256") == digest_value(ledger_payload)
        and ledger.get("plan_sha256") == plan["plan_sha256"],
        "Round 1 bridge ledger integrity check failed",
    )
    failures = []
    seeds = ledger.get("seeds")
    _require(isinstance(seeds, dict), "Round 1 bridge seed ledger is malformed")
    for seed, row in seeds.items():
        _require(isinstance(row, dict), "Round 1 bridge seed row is malformed")
        for stage in ("local_activate", "matched_relax"):
            status = row.get(stage)
            if status in _TECHNICAL_STAGE_STATES:
                failures.append({"seed": int(seed), "stage": stage, "status": status})
    return {
        "phase": str(ledger.get("state", "unknown")),
        "technical_failures": sorted(
            failures, key=lambda row: (row["seed"], row["stage"])
        ),
        "plan": plan,
        "ledger": ledger,
        "plan_file_sha256": sha256_file(plan_path),
        "ledger_file_sha256": sha256_file(ledger_path),
    }


def _verify_round1_lineage(
    *,
    campaign_root: Path,
    bridge_root: Path,
    controller: Mapping[str, Any],
    protocol: Mapping[str, Any],
    bridge: Mapping[str, Any],
    outcome: str,
    expected_candidate_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    plan = bridge.get("plan")
    bridge_ledger = bridge.get("ledger")
    _require(
        isinstance(plan, dict) and isinstance(bridge_ledger, dict),
        "Round 1 bridge is absent",
    )
    _require(
        bridge_ledger.get("state") == "normalized", "Round 1 bridge is not normalized"
    )
    evidence_path = bridge_root / "search_evidence.json"
    _require(evidence_path.is_file(), "Round 1 normalized evidence is missing")
    evidence = _read_object(evidence_path)
    bridge_evidence = bridge_ledger.get("evidence")
    _require(
        isinstance(bridge_evidence, dict)
        and bridge_evidence.get("sha256") == sha256_file(evidence_path)
        and bridge_evidence.get("value_sha256") == digest_value(evidence),
        "Round 1 bridge evidence integrity check failed",
    )
    round_row = controller["rounds"]["1"]
    candidate_row = round_row["candidates"][expected_candidate_id]
    controller_evidence_path = Path(candidate_row["action_path"]).with_name(
        "evidence.json"
    )
    _require(
        controller_evidence_path.is_file()
        and _read_object(controller_evidence_path) == evidence
        and candidate_row.get("evidence_sha256") == digest_value(evidence),
        "Round 1 controller and bridge evidence differ",
    )
    evidence_by_seed = {int(row["seed"]): row for row in evidence.get("seeds", [])}
    _require(
        sorted(evidence_by_seed) == protocol["promotion_seeds"],
        "Round 1 evidence seed family is incomplete",
    )
    artifacts = []
    for seed_row in plan.get("seeds", []):
        seed = int(seed_row["seed"])
        baseline = seed_row["baseline_checkpoint"]
        baseline_path = Path(baseline["path"])
        _require(
            baseline_path.is_file()
            and sha256_file(baseline_path) == baseline["sha256"],
            f"Round 1 baseline checkpoint drift for seed {seed}",
        )
        matched_path = Path(seed_row["matched_output_dir"]) / "report.json"
        _require(
            matched_path.is_file()
            and sha256_file(matched_path)
            == bridge_ledger["seeds"][str(seed)].get("matched_relax_report_sha256"),
            f"Round 1 matched report drift for seed {seed}",
        )
        matched = _read_object(matched_path)
        candidate_result = matched.get("results", {}).get("candidate", {})
        candidate_path = Path(str(candidate_result.get("checkpoint", "")))
        candidate_hash = str(candidate_result.get("checkpoint_sha256", ""))
        _require(
            candidate_path.is_file()
            and sha256_file(candidate_path) == candidate_hash
            and evidence_by_seed[seed]["candidate_checkpoint_sha256"] == candidate_hash,
            f"Round 1 candidate checkpoint drift for seed {seed}",
        )
        if outcome == "promoted":
            chosen_path, chosen_hash, role = (
                candidate_path,
                candidate_hash,
                "candidate-polished",
            )
        else:
            chosen_path, chosen_hash, role = (
                baseline_path,
                baseline["sha256"],
                "baseline-retained",
            )
        artifacts.append(
            {
                "seed": seed,
                "checkpoint_path": str(chosen_path.resolve()),
                "checkpoint_sha256": chosen_hash,
                "role": role,
                "source_report_path": str(matched_path.resolve()),
                "source_report_sha256": sha256_file(matched_path),
            }
        )
    artifacts.sort(key=lambda row: row["seed"])
    family_payload = {
        "models": [
            {"seed": row["seed"], "checkpoint_sha256": row["checkpoint_sha256"]}
            for row in artifacts
        ]
    }
    resolved_family = {**family_payload, "family_sha256": digest_value(family_payload)}
    _require(
        resolved_family == controller["champion_family"],
        "Round 1 resolved checkpoint paths do not match the controller champion",
    )
    provenance = {
        "campaign_root": str(campaign_root.resolve()),
        "bridge_root": str(bridge_root.resolve()),
        "protocol_sha256": controller["protocol_sha256"],
        "search_indices_sha256": controller["search_indices_sha256"],
        "bridge_plan_sha256": plan["plan_sha256"],
        "bridge_plan_file_sha256": bridge["plan_file_sha256"],
        "bridge_ledger_file_sha256": bridge["ledger_file_sha256"],
        "search_evidence_sha256": sha256_file(evidence_path),
        "search_evidence_value_sha256": digest_value(evidence),
        "input_sha256": {
            row["role"]: row["sha256"]
            for row in protocol["search_inputs"]
            if row["role"]
            in {
                "search-native-points",
                "search-native-pullbacks",
                "development-evaluation-points",
                "development-evaluation-pullbacks",
            }
        },
        "matched_batch_plans": sorted(
            [
                {
                    "seed": int(row["seed"]),
                    "plan_seed": int(row["batch_plan"]["seed"]),
                    "file_sha256": row["batch_plan"]["file_sha256"],
                    "value_set_sha256": row["batch_plan"]["value_set_sha256"],
                }
                for row in plan["seeds"]
            ],
            key=lambda row: row["seed"],
        ),
    }
    return artifacts, provenance


def inspect_round1(
    *, campaign_root: Path, bridge_root: Path, catalog: Mapping[str, Any]
) -> dict[str, Any]:
    """Audit the existing Round 1 campaign and return a normalized observation."""

    campaign_root = campaign_root.expanduser().resolve()
    bridge_root = bridge_root.expanduser().resolve()
    controller_sentinel = campaign_root / ".gcicy-architecture-auto-research-root"
    if not controller_sentinel.is_file():
        return {"state": "waiting", "phase": "controller-not-initialized"}
    store = CampaignStore(campaign_root)
    controller = store.status()
    protocol = store.protocol()
    expected = catalog["round1_contract"]
    data_contract = catalog["data_contract"]
    _require(
        controller["campaign_id"] == expected["campaign_id"]
        and protocol["promotion_seeds"] == catalog["promotion_seeds"],
        "Round 1 controller identity differs from the catalog",
    )
    _require(
        controller["protocol_sha256"] == data_contract["protocol_sha256"]
        and controller["search_indices_sha256"]
        == data_contract["search_indices_sha256"],
        "Round 1 protocol or fixed search indices differ from the catalog",
    )
    observed_inputs = {
        row["role"]: row["sha256"]
        for row in protocol["search_inputs"]
        if row["role"] in data_contract["input_sha256"]
    }
    _require(
        observed_inputs == data_contract["input_sha256"],
        "Round 1 data input hashes differ from the catalog",
    )
    _require(
        all(
            row["checkpoint_sha256"] == expected["baseline_checkpoint_sha256"]
            for row in protocol["baseline_family"]["models"]
        ),
        "Round 1 baseline family differs from the catalog",
    )
    round_row = controller.get("rounds", {}).get("1")
    if round_row is None:
        return {"state": "waiting", "phase": "action-not-registered"}
    candidates = round_row.get("candidates")
    _require(
        isinstance(candidates, dict) and set(candidates) == {expected["candidate_id"]},
        "Round 1 candidate set differs from the catalog",
    )
    action = _read_object(Path(candidates[expected["candidate_id"]]["action_path"]))
    _require(
        action.get("candidate_id") == expected["candidate_id"]
        and action.get("round") == 1
        and action.get("mutation") == expected["mutation"]
        and action.get("pipeline") == expected["pipeline"],
        "Round 1 action differs from the catalog",
    )
    bridge = _bridge_snapshot(
        bridge_root,
        expected_candidate_id=expected["candidate_id"],
        expected_campaign_root=campaign_root,
    )
    if isinstance(bridge.get("plan"), dict) and "seeds" in bridge["plan"]:
        observed_batch_plans = sorted(
            [
                {
                    "seed": int(row["seed"]),
                    "plan_seed": int(row["batch_plan"]["seed"]),
                    "file_sha256": row["batch_plan"]["file_sha256"],
                    "value_set_sha256": row["batch_plan"]["value_set_sha256"],
                }
                for row in bridge["plan"]["seeds"]
            ],
            key=lambda row: row["seed"],
        )
        _require(
            observed_batch_plans == data_contract["matched_batch_plans"],
            "Round 1 matched batch plans differ from the catalog",
        )
    if bridge["technical_failures"]:
        return {
            "state": "technical-failure",
            "phase": bridge["phase"],
            "failures": bridge["technical_failures"],
            "provenance": {
                "campaign_id": controller["campaign_id"],
                "protocol_sha256": controller["protocol_sha256"],
                "bridge_plan_sha256": bridge["plan"]["plan_sha256"],
                "bridge_plan_file_sha256": bridge["plan_file_sha256"],
                "bridge_ledger_file_sha256": bridge["ledger_file_sha256"],
            },
        }
    if round_row.get("status") != "complete":
        state = (
            "running"
            if bridge["phase"] in {"running", "workers-complete", "normalized"}
            else "waiting"
        )
        return {"state": state, "phase": bridge["phase"]}
    winner = round_row.get("winner_candidate_id")
    if winner == expected["candidate_id"]:
        outcome = "promoted"
    elif winner is None:
        outcome = "scientific-rejected"
    else:
        raise AutoResearchError("Round 1 selected an unregistered winner")
    artifacts, provenance = _verify_round1_lineage(
        campaign_root=campaign_root,
        bridge_root=bridge_root,
        controller=controller,
        protocol=protocol,
        bridge=bridge,
        outcome=outcome,
        expected_candidate_id=expected["candidate_id"],
    )
    adjudication_path = campaign_root / "rounds" / "round_001" / "adjudication.json"
    _require(adjudication_path.is_file(), "Round 1 adjudication artifact is missing")
    return {
        "state": "decided",
        "phase": "complete",
        "outcome": outcome,
        "champion_family": controller["champion_family"],
        "champion_artifacts": artifacts,
        "provenance": {
            **provenance,
            "adjudication_path": str(adjudication_path.resolve()),
            "adjudication_sha256": sha256_file(adjudication_path),
        },
    }


def _legacy_rank_result(
    *, bridge_root: Path, proposal: Mapping[str, Any], catalog: Mapping[str, Any]
) -> dict[str, Any]:
    """Normalize the already-completed, independently controlled Round 2."""

    bridge_root = bridge_root.expanduser().resolve()
    plan_path = bridge_root / "plan.json"
    _require(plan_path.is_file(), "legacy Round 2 bridge plan is missing")
    plan = _read_object(plan_path)
    expected_candidate = {
        "r2-edge6-rank39-scale025": "edge6-rank39-scale025",
        "r2-edge6-rank53-progressive": "edge6-rank53-progressive",
    }.get(str(proposal["recipe_id"]))
    _require(expected_candidate is not None, "legacy result is not a Round 2 recipe")
    campaign_root = Path(str(plan.get("campaign_run_root", ""))).resolve()
    bridge = _bridge_snapshot(
        bridge_root,
        expected_candidate_id=expected_candidate,
        expected_campaign_root=campaign_root,
    )
    _require(
        bridge["phase"] == "normalized" and not bridge["technical_failures"],
        "legacy Round 2 bridge is not a normalized scientific result",
    )
    plan = bridge["plan"]
    action = plan.get("action")
    _require(isinstance(action, dict), "legacy Round 2 action is absent")
    recipe = next(
        row for row in catalog["recipes"] if row["recipe_id"] == proposal["recipe_id"]
    )
    intervention = recipe["intervention"]
    expected_mutation = {
        "kind": "rank",
        "target": "internal-edge",
        "source_dimension": intervention["source_dimension"],
        "target_dimension": intervention["target_dimension"],
        "structural_maximum": intervention["structural_maximum"],
        "new_output_real_parameters": intervention["new_total_real_parameters"],
        "edge": intervention["edge"],
    }
    _require(
        action.get("candidate_id") == expected_candidate
        and action.get("round") == 1
        and action.get("parent_family_sha256")
        == proposal["parent_family"]["family_sha256"]
        and action.get("search_indices_sha256")
        == proposal["data_binding"]["search_indices_sha256"]
        and action.get("mutation") == expected_mutation
        and action.get("pipeline")
        == [{"kind": "local-activate"}, {"kind": "matched-relax"}]
        and plan.get("rank_activation_scale") == intervention["activation_scale"],
        "legacy Round 2 action differs from the manager proposal",
    )
    budget = recipe["budget"]
    local_budget = plan.get("budgets", {}).get("local_activate", {})
    matched_budget = plan.get("budgets", {}).get("matched_relax", {})
    _require(
        local_budget.get("optimizer_updates")
        == budget["local_activate_optimizer_updates"]
        and matched_budget
        == {
            "optimizer_updates": budget["matched_relax_optimizer_updates_per_arm"],
            "batch_size": budget["batch_size"],
            "learning_rate": budget["learning_rate"],
            "gradient_clip_norm": budget["gradient_clip_norm"],
            "train_examples": budget["fit_examples"],
            "evaluation_examples": budget["selection_examples"],
            "eval_every": 25,
            "scheduler": "cosine",
        },
        "legacy Round 2 budget differs from the manager proposal",
    )
    thresholds = plan.get("thresholds", {})
    gate = recipe["promotion_gate"]
    _require(
        all(
            thresholds.get(source) == gate[target]
            for source, target in (
                (
                    "minimum_median_relative_sigma_gain",
                    "minimum_median_relative_sigma_gain",
                ),
                (
                    "minimum_median_relative_chi_gain",
                    "minimum_median_relative_chi_gain",
                ),
                ("minimum_improved_seeds", "minimum_improved_seeds"),
                (
                    "minimum_paired_ci_positive_seeds",
                    "minimum_paired_ci_positive_seeds",
                ),
                (
                    "maximum_seed_relative_regression",
                    "maximum_seed_relative_regression",
                ),
                (
                    "maximum_search_tail_relative_degradation",
                    "maximum_tail_relative_degradation",
                ),
            )
        ),
        "legacy Round 2 promotion gate differs from the manager proposal",
    )
    _require(
        plan.get("protocol_sha256") == proposal["data_binding"]["protocol_sha256"]
        and plan.get("search_indices_sha256")
        == proposal["data_binding"]["search_indices_sha256"]
        and plan.get("historical_confirmation") == "absent",
        "legacy Round 2 data identity differs from the manager proposal",
    )
    observed_inputs = {
        role: row["sha256"]
        for role, row in plan.get("input_roles", {}).items()
        if role in proposal["data_binding"]["input_sha256"]
    }
    _require(
        observed_inputs == proposal["data_binding"]["input_sha256"],
        "legacy Round 2 input hashes differ from the manager proposal",
    )
    observed_batches = sorted(
        [
            {
                "seed": int(row["seed"]),
                "plan_seed": int(row["batch_plan"]["seed"]),
                "file_sha256": row["batch_plan"]["file_sha256"],
                "value_set_sha256": row["batch_plan"]["value_set_sha256"],
            }
            for row in plan.get("seeds", [])
        ],
        key=lambda row: row["seed"],
    )
    _require(
        observed_batches == proposal["data_binding"]["matched_batch_plans"],
        "legacy Round 2 batch plans differ from the manager proposal",
    )
    parent_by_seed = {row["seed"]: row for row in proposal["parent_artifacts"]}
    _require(
        sorted(int(row["seed"]) for row in plan.get("seeds", []))
        == catalog["promotion_seeds"],
        "legacy Round 2 seed family is incomplete",
    )
    for seed_row in plan["seeds"]:
        parent = parent_by_seed[int(seed_row["seed"])]
        baseline = seed_row["baseline_checkpoint"]
        _require(
            Path(str(baseline["path"])).resolve()
            == Path(parent["checkpoint_path"]).resolve()
            and baseline["sha256"] == parent["checkpoint_sha256"],
            "legacy Round 2 parent checkpoint bypasses the manager proposal",
        )

    controller = CampaignStore(campaign_root)
    controller_state = controller.status()
    protocol = controller.protocol()
    _require(
        controller_state["protocol_sha256"] == plan["protocol_sha256"]
        and protocol["baseline_family"] == proposal["parent_family"]
        and protocol["promotion_seeds"] == catalog["promotion_seeds"],
        "legacy Round 2 controller is not bound to the manager parent",
    )
    round_row = controller_state.get("rounds", {}).get("1")
    _require(
        isinstance(round_row, dict)
        and round_row.get("status") == "complete"
        and set(round_row.get("candidates", {})) == {expected_candidate},
        "legacy Round 2 controller is not completely adjudicated",
    )
    action_path = Path(round_row["candidates"][expected_candidate]["action_path"])
    _require(
        _read_object(action_path) == action,
        "legacy Round 2 controller action differs from its bridge",
    )
    evidence_path = bridge_root / "search_evidence.json"
    evidence = _read_object(evidence_path)
    bridge_evidence = bridge["ledger"].get("evidence")
    controller_evidence_path = action_path.with_name("evidence.json")
    _require(
        isinstance(bridge_evidence, dict)
        and sha256_file(evidence_path) == bridge_evidence.get("sha256")
        and digest_value(evidence) == bridge_evidence.get("value_sha256")
        and _read_object(controller_evidence_path) == evidence
        and round_row["candidates"][expected_candidate].get("evidence_sha256")
        == digest_value(evidence),
        "legacy Round 2 evidence hash chain is inconsistent",
    )
    adjudication_path = campaign_root / "rounds" / "round_001" / "adjudication.json"
    adjudication = _read_object(adjudication_path)
    _require(
        adjudication.get("schema") == "gcicy-quintic-architecture-round-adjudication-v1"
        and adjudication.get("candidate_count") == 1
        and len(adjudication.get("candidates", [])) == 1
        and adjudication["candidates"][0].get("candidate_id") == expected_candidate
        and round_row.get("adjudication_sha256") == digest_value(adjudication),
        "legacy Round 2 adjudication is malformed",
    )
    candidate = adjudication["candidates"][0]
    promoted = bool(candidate.get("promotion_passes"))
    _require(
        adjudication.get("winner_candidate_id")
        == (expected_candidate if promoted else None)
        and controller_state["champion_family"]
        == (candidate["candidate_family"] if promoted else proposal["parent_family"]),
        "legacy Round 2 recommendation contradicts its strict adjudication",
    )

    evidence_by_seed = {int(row["seed"]): row for row in evidence["seeds"]}
    source_artifacts = [
        _file_reference(plan_path, role="bridge-plan"),
        _file_reference(bridge_root / "ledger.json", role="bridge-ledger"),
        _file_reference(evidence_path, role="bridge-evidence"),
        _file_reference(
            campaign_root / "protocol.lock.json", role="controller-protocol"
        ),
        _file_reference(campaign_root / "ledger.json", role="controller-ledger"),
        _file_reference(action_path, role="controller-action"),
        _file_reference(controller_evidence_path, role="controller-evidence"),
        _file_reference(adjudication_path, role="controller-adjudication"),
    ]
    candidate_artifacts = []
    for seed_row in sorted(plan["seeds"], key=lambda row: int(row["seed"])):
        seed = int(seed_row["seed"])
        local_path = Path(seed_row["local_output_dir"]) / "report.json"
        matched_path = Path(seed_row["matched_output_dir"]) / "report.json"
        _require(
            sha256_file(local_path)
            == bridge["ledger"]["seeds"][str(seed)]["local_activate_report_sha256"]
            == evidence_by_seed[seed]["source_report_sha256"]["local_activate"]
            and sha256_file(matched_path)
            == bridge["ledger"]["seeds"][str(seed)]["matched_relax_report_sha256"]
            == evidence_by_seed[seed]["source_report_sha256"]["matched_relax"],
            f"legacy Round 2 worker report drifted for seed {seed}",
        )
        source_artifacts.extend(
            [
                _file_reference(local_path, role=f"seed-{seed}-local-report"),
                _file_reference(matched_path, role=f"seed-{seed}-matched-report"),
            ]
        )
        matched = _read_object(matched_path)
        candidate_result = matched.get("results", {}).get("candidate", {})
        candidate_path = Path(str(candidate_result.get("checkpoint", ""))).resolve()
        candidate_sha256 = str(candidate_result.get("checkpoint_sha256", ""))
        _require(
            candidate_path.is_file()
            and sha256_file(candidate_path) == candidate_sha256
            and candidate_sha256
            == evidence_by_seed[seed]["candidate_checkpoint_sha256"],
            f"legacy Round 2 candidate checkpoint drifted for seed {seed}",
        )
        candidate_artifacts.append(
            {
                "seed": seed,
                "checkpoint_path": str(candidate_path),
                "checkpoint_sha256": candidate_sha256,
                "role": "candidate-polished",
                "source_report_path": str(matched_path.resolve()),
                "source_report_sha256": sha256_file(matched_path),
            }
        )
    if promoted:
        artifacts, family = _canonical_artifact_family(
            candidate_artifacts,
            seeds=catalog["promotion_seeds"],
            context="legacy Round 2 candidate artifacts",
        )
        _require(
            family == candidate["candidate_family"],
            "legacy Round 2 candidate paths differ from adjudication",
        )
        outcome, role = "promoted", "candidate"
    else:
        artifacts = list(proposal["parent_artifacts"])
        family = dict(proposal["parent_family"])
        outcome, role = "scientific-rejected", "parent"
    return _build_round_result(
        proposal=proposal,
        outcome=outcome,
        recommended_role=role,
        recommended_family=family,
        recommended_artifacts=artifacts,
        source_kind="historical-auto-research-not-manager-launched",
        source_root=bridge_root,
        source_artifacts=source_artifacts,
    )


def _paired_round_result(
    *, bridge_root: Path, proposal: Mapping[str, Any], catalog: Mapping[str, Any]
) -> dict[str, Any]:
    """Normalize an R3/R4 paired bridge after its strict adjudication."""

    from .quintic_paired_auto_research_bridge import (  # local: avoids cycle
        PAIRED_ADJUDICATION_SCHEMA,
        paired_bridge_status,
    )

    bridge_root = bridge_root.expanduser().resolve()
    status = paired_bridge_status(bridge_root)
    _require(
        status.get("state") == "adjudicated"
        and status.get("round") == proposal["round"]
        and status.get("recipe_id") == proposal["recipe_id"],
        "paired bridge is not the adjudicated current manager round",
    )
    plan_path = bridge_root / "plan.json"
    ledger_path = bridge_root / "ledger.json"
    plan = _read_object(plan_path)
    _require(
        plan.get("manager_handoff", {}).get("proposal_sha256")
        == proposal["proposal_sha256"],
        "paired bridge was not prepared from the current manager proposal",
    )
    evidence_path = bridge_root / "paired_evidence.json"
    adjudication_path = bridge_root / "adjudication.json"
    evidence = _read_object(evidence_path)
    adjudication = _read_object(adjudication_path)
    _require(
        adjudication.get("schema") == PAIRED_ADJUDICATION_SCHEMA
        and adjudication.get("plan_sha256") == plan["plan_sha256"]
        and adjudication.get("evidence_value_sha256") == digest_value(evidence)
        and adjudication.get("recommended_family")
        == (
            {
                "models": [
                    {
                        "seed": row["seed"],
                        "checkpoint_sha256": row["candidate_checkpoint"]["sha256"],
                    }
                    for row in evidence["seeds"]
                ],
                "family_sha256": adjudication["recommended_family"]["family_sha256"],
            }
            if adjudication.get("promotion_passes")
            else proposal["parent_family"]
        ),
        "paired recommendation contradicts its evidence or manager parent",
    )
    source_artifacts = [
        _file_reference(plan_path, role="paired-plan"),
        _file_reference(ledger_path, role="paired-ledger"),
        _file_reference(evidence_path, role="paired-evidence"),
        _file_reference(adjudication_path, role="paired-adjudication"),
    ]
    promoted = bool(adjudication["promotion_passes"])
    candidate_artifacts = []
    for row in evidence["seeds"]:
        seed = int(row["seed"])
        report = Path(row["source_report"]["path"])
        _require(
            report.is_file() and sha256_file(report) == row["source_report"]["sha256"],
            f"paired source report drifted for seed {seed}",
        )
        source_artifacts.append(
            _file_reference(report, role=f"seed-{seed}-paired-report")
        )
        checkpoint = Path(row["candidate_checkpoint"]["path"])
        checkpoint_sha256 = row["candidate_checkpoint"]["sha256"]
        _require(
            checkpoint.is_file() and sha256_file(checkpoint) == checkpoint_sha256,
            f"paired candidate checkpoint drifted for seed {seed}",
        )
        candidate_artifacts.append(
            {
                "seed": seed,
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": checkpoint_sha256,
                "role": "candidate-polished",
                "source_report_path": str(report.resolve()),
                "source_report_sha256": sha256_file(report),
            }
        )
    if promoted:
        artifacts, family = _canonical_artifact_family(
            candidate_artifacts,
            seeds=catalog["promotion_seeds"],
            context="paired candidate artifacts",
        )
        _require(
            family == adjudication["recommended_family"],
            "paired candidate paths differ from adjudication",
        )
        outcome, role = "promoted", "candidate"
    else:
        artifacts = list(proposal["parent_artifacts"])
        family = dict(proposal["parent_family"])
        outcome, role = "scientific-rejected", "parent"
    return _build_round_result(
        proposal=proposal,
        outcome=outcome,
        recommended_role=role,
        recommended_family=family,
        recommended_artifacts=artifacts,
        source_kind="paired-bridge",
        source_root=bridge_root,
        source_artifacts=source_artifacts,
    )


class MultiRoundManager:
    """Create-only manager state for the four-round registered catalog."""

    def __init__(self, run_root: Path):
        self.run_root = run_root.expanduser().resolve()
        self.catalog_path = self.run_root / "catalog.lock.json"
        self.state_path = self.run_root / "state.json"
        self.lock_path = self.run_root / ".manager.lock"

    @classmethod
    def initialize(
        cls,
        run_root: Path,
        catalog_value: Mapping[str, Any],
        *,
        round1_campaign_root: Path,
        round1_bridge_root: Path,
    ) -> "MultiRoundManager":
        manager = cls(run_root)
        catalog = validate_catalog(catalog_value)
        campaign_root = round1_campaign_root.expanduser().resolve()
        bridge_root = round1_bridge_root.expanduser().resolve()
        _require(
            len({manager.run_root, campaign_root, bridge_root}) == 3,
            "manager, Round 1 controller, and bridge roots must be distinct",
        )
        if "gcicy_metric_k2_20260710" in str(manager.run_root):
            raise AutoResearchError("manager run root may not be a frozen-result root")
        manager.run_root.mkdir(parents=True, exist_ok=True)
        sentinel = manager.run_root / ROOT_SENTINEL
        if manager.state_path.exists():
            _require(sentinel.is_file(), "manager root sentinel is missing")
            locked = _read_object(manager.catalog_path)
            state = _read_object(manager.state_path)
            _validate_state(state)
            _require(
                locked == catalog
                and state["catalog_sha256"] == catalog["catalog_sha256"]
                and state["round1_campaign_root"] == str(campaign_root)
                and state["round1_bridge_root"] == str(bridge_root),
                "existing manager root belongs to another locked campaign",
            )
            return manager
        allowed_partial = {sentinel, manager.catalog_path}
        unmanaged = [
            path for path in manager.run_root.iterdir() if path not in allowed_partial
        ]
        _require(not unmanaged, "refusing a non-empty unmanaged manager root")
        if sentinel.exists():
            _require(
                sentinel.read_text(encoding="utf-8").strip() == catalog["catalog_id"],
                "manager root sentinel belongs to another catalog",
            )
        else:
            sentinel.write_text(f"{catalog['catalog_id']}\n", encoding="utf-8")
        _create_or_verify(manager.catalog_path, catalog, context="locked catalog")
        initial_event = _event(
            revision=0,
            event="initialized",
            previous="0" * 64,
            details={"catalog_id": catalog["catalog_id"]},
        )
        state = {
            "schema": MANAGER_SCHEMA,
            "catalog_id": catalog["catalog_id"],
            "catalog_sha256": catalog["catalog_sha256"],
            "round1_campaign_root": str(campaign_root),
            "round1_bridge_root": str(bridge_root),
            "status": "waiting-r1",
            "observed_round1_phase": "not-inspected",
            "round1": {
                "outcome": "pending",
                "decision_path": None,
                "decision_sha256": None,
            },
            "selected_recipe_ids": [],
            "next_recipe_id": None,
            "next_proposal_path": None,
            "next_proposal_sha256": None,
            "completed_rounds": {},
            "execution_handoff": None,
            "queue_path": None,
            "queue_sha256": None,
            "blocker": None,
            "revision": 0,
            "events": [initial_event],
        }
        _seal_state(state)
        atomic_write_json(manager.state_path, state)
        return manager

    def catalog(self) -> dict[str, Any]:
        catalog = _read_object(self.catalog_path)
        normalized = validate_catalog(
            {key: value for key, value in catalog.items() if key != "catalog_sha256"}
        )
        _require(normalized == catalog, "locked catalog was modified")
        return catalog

    def state(self) -> dict[str, Any]:
        state = _read_object(self.state_path)
        _validate_state(state)
        _require(
            state["catalog_sha256"] == self.catalog()["catalog_sha256"],
            "manager state and catalog differ",
        )
        self._verify_artifacts(state)
        return state

    def _verify_artifacts(self, state: Mapping[str, Any]) -> None:
        catalog = self.catalog()
        round1 = state["round1"]
        decision = None
        if round1["decision_path"] is not None:
            decision = _read_object(Path(round1["decision_path"]))
            _require(
                digest_value(decision) == round1["decision_sha256"],
                "Round 1 manager decision was modified",
            )
            if decision.get("outcome") in {"promoted", "scientific-rejected"}:
                artifacts = decision.get("champion_artifacts")
                _require(
                    isinstance(artifacts, list) and len(artifacts) == 3,
                    "Round 1 champion artifact family is malformed",
                )
                for row in artifacts:
                    checkpoint_path = Path(str(row.get("checkpoint_path", "")))
                    report_path = Path(str(row.get("source_report_path", "")))
                    _require(
                        checkpoint_path.is_file()
                        and sha256_file(checkpoint_path)
                        == row.get("checkpoint_sha256"),
                        "Round 1 champion checkpoint drifted after synchronization",
                    )
                    _require(
                        report_path.is_file()
                        and sha256_file(report_path) == row.get("source_report_sha256"),
                        "Round 1 source report drifted after synchronization",
                    )
                provenance = decision.get("provenance")
                _require(
                    isinstance(provenance, dict),
                    "Round 1 decision provenance is malformed",
                )
                adjudication_path = Path(str(provenance.get("adjudication_path", "")))
                bridge_root = Path(str(provenance.get("bridge_root", "")))
                plan_path = bridge_root / "plan.json"
                ledger_path = bridge_root / "ledger.json"
                evidence_path = bridge_root / "search_evidence.json"
                _require(
                    adjudication_path.is_file()
                    and sha256_file(adjudication_path)
                    == provenance.get("adjudication_sha256"),
                    "Round 1 adjudication drifted after synchronization",
                )
                _require(
                    plan_path.is_file()
                    and sha256_file(plan_path)
                    == provenance.get("bridge_plan_file_sha256")
                    and _read_object(plan_path).get("plan_sha256")
                    == provenance.get("bridge_plan_sha256"),
                    "Round 1 bridge plan drifted after synchronization",
                )
                _require(
                    ledger_path.is_file()
                    and sha256_file(ledger_path)
                    == provenance.get("bridge_ledger_file_sha256"),
                    "Round 1 bridge ledger drifted after synchronization",
                )
                _require(
                    evidence_path.is_file()
                    and sha256_file(evidence_path)
                    == provenance.get("search_evidence_sha256")
                    and digest_value(_read_object(evidence_path))
                    == provenance.get("search_evidence_value_sha256"),
                    "Round 1 search evidence drifted after synchronization",
                )
        if state["queue_path"] is not None:
            queue = _read_object(Path(state["queue_path"]))
            _require(
                digest_value(queue) == state["queue_sha256"],
                "manager queue was modified",
            )
            proposal = queue.get("resolved_next_proposal")
            _require(
                isinstance(proposal, dict), "manager queue lacks its resolved proposal"
            )
            proposal_path = self.run_root / "rounds" / "round_002" / "proposal.json"
            _require(
                proposal_path.is_file() and _read_object(proposal_path) == proposal,
                "resolved Round 2 proposal was modified",
            )
            proposal = _validate_proposal(
                proposal,
                catalog=catalog,
                expected_recipe_id=queue["selected_recipe_ids"][0],
            )
            _require(
                isinstance(decision, dict)
                and proposal.get("parent_family") == decision.get("champion_family")
                and proposal.get("parent_artifacts")
                == decision.get("champion_artifacts"),
                "resolved Round 2 parent differs from the durable decision",
            )

        completed = state.get("completed_rounds")
        _require(
            isinstance(completed, dict), "manager completed-round map is malformed"
        )
        for round_text, metadata in completed.items():
            _require(
                round_text in {"2", "3", "4"} and isinstance(metadata, dict),
                "manager completed-round metadata is malformed",
            )
            result_path = Path(str(metadata.get("path", ""))).resolve()
            _require(result_path.is_file(), "manager round result is missing")
            result = _read_object(result_path)
            proposal_path = (
                self.run_root
                / "rounds"
                / f"round_{int(round_text):03d}"
                / "proposal.json"
            )
            proposal = _validate_proposal(_read_object(proposal_path), catalog=catalog)
            normalized = _validate_round_result(
                result, proposal=proposal, catalog=catalog
            )
            _require(
                normalized == result
                and metadata
                == {
                    "path": str(result_path),
                    "file_sha256": sha256_file(result_path),
                    "result_sha256": result["result_sha256"],
                    "outcome": result["outcome"],
                },
                "manager round result metadata drifted",
            )

        next_recipe_id = state.get("next_recipe_id")
        next_proposal_path = state.get("next_proposal_path")
        next_proposal_sha256 = state.get("next_proposal_sha256")
        if next_recipe_id is None:
            _require(
                next_proposal_path is None and next_proposal_sha256 is None,
                "manager has a proposal without a next recipe",
            )
        else:
            _require(
                isinstance(next_proposal_path, str)
                and Path(next_proposal_path).is_file(),
                "manager current proposal is missing",
            )
            current = _validate_proposal(
                _read_object(Path(next_proposal_path)),
                catalog=catalog,
                expected_recipe_id=next_recipe_id,
            )
            _require(
                Path(next_proposal_path).resolve()
                == self.run_root
                / "rounds"
                / f"round_{current['round']:03d}"
                / "proposal.json"
                and current["proposal_sha256"] == next_proposal_sha256,
                "manager current proposal metadata drifted",
            )

        handoff_metadata = state.get("execution_handoff")
        if handoff_metadata is not None:
            _require(
                isinstance(handoff_metadata, dict),
                "manager execution handoff metadata is malformed",
            )
            handoff_path = Path(str(handoff_metadata.get("path", ""))).resolve()
            handoff = _read_handoff(handoff_path)
            _require(
                handoff_metadata
                == {
                    "path": str(handoff_path),
                    "file_sha256": sha256_file(handoff_path),
                    "handoff_sha256": handoff["handoff_sha256"],
                    "round": handoff["round"],
                },
                "manager execution handoff metadata drifted",
            )
            _verify_handoff_against_state(
                handoff, handoff_path=handoff_path, manager=self, state=state
            )

    @contextmanager
    def _locked(self) -> Iterator[dict[str, Any]]:
        _require(
            (self.run_root / ROOT_SENTINEL).is_file(),
            "manager root sentinel is missing",
        )
        handle = self.lock_path.open("a+", encoding="utf-8")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            state = self.state()
            yield state
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

    def _commit(
        self, state: dict[str, Any], *, event: str, details: Mapping[str, Any]
    ) -> None:
        revision = int(state["revision"]) + 1
        state["events"].append(
            _event(
                revision=revision,
                event=event,
                previous=state["events"][-1]["event_sha256"],
                details=details,
            )
        )
        state["revision"] = revision
        _seal_state(state)
        atomic_write_json(self.state_path, state)

    def _materialize_decision_and_queue(
        self,
        state: dict[str, Any],
        *,
        observation: Mapping[str, Any],
        catalog: Mapping[str, Any],
    ) -> None:
        outcome = str(observation["outcome"])
        decision = {
            "schema": ROUND1_DECISION_SCHEMA,
            "catalog_sha256": catalog["catalog_sha256"],
            "outcome": outcome,
            "champion_family": observation["champion_family"],
            "champion_artifacts": observation["champion_artifacts"],
            "provenance": observation["provenance"],
        }
        decision_path = self.run_root / "rounds" / "round_001" / "decision.json"
        _create_or_verify(decision_path, decision, context="Round 1 manager decision")
        decision_sha256 = digest_value(decision)
        if state["round1"]["outcome"] != "pending":
            _require(
                state["round1"]["decision_sha256"] == decision_sha256,
                "Round 1 manager decision conflicts with the durable state",
            )
            return
        branch_key = (
            "r1_promoted" if outcome == "promoted" else "r1_scientific_rejected"
        )
        selected = list(catalog["branches"][branch_key])
        recipes = {row["recipe_id"]: row for row in catalog["recipes"]}
        next_recipe = recipes[selected[0]]
        data_binding = {
            "protocol_sha256": observation["provenance"]["protocol_sha256"],
            "search_indices_sha256": observation["provenance"]["search_indices_sha256"],
            "input_sha256": observation["provenance"]["input_sha256"],
            "matched_batch_plans": observation["provenance"]["matched_batch_plans"],
        }
        _require(
            data_binding
            == {
                key: catalog["data_contract"][key]
                for key in (
                    "protocol_sha256",
                    "search_indices_sha256",
                    "input_sha256",
                    "matched_batch_plans",
                )
            },
            "Round 1 data binding differs from the locked catalog",
        )
        decision_lineage = _source_lineage(
            source_round=1,
            source_kind="round1-decision",
            source_path=decision_path,
            source_value_sha256=decision_sha256,
        )
        proposal = _build_proposal(
            catalog=catalog,
            recipe=next_recipe,
            parent_family=observation["champion_family"],
            parent_artifacts=observation["champion_artifacts"],
            parent_lineage=decision_lineage,
        )
        proposal_path = self.run_root / "rounds" / "round_002" / "proposal.json"
        _create_or_verify(proposal_path, proposal, context="resolved Round 2 proposal")
        queue_payload = {
            "schema": QUEUE_SCHEMA,
            "catalog_sha256": catalog["catalog_sha256"],
            "round1_outcome": outcome,
            "selected_recipe_ids": selected,
            "resolved_next_proposal": proposal,
            "later_recipes": [
                {
                    "recipe_id": recipe_id,
                    "recipe_sha256": recipes[recipe_id]["recipe_sha256"],
                    "round": recipes[recipe_id]["round"],
                    "state": "preregistered-awaiting-parent",
                }
                for recipe_id in selected[1:]
            ],
        }
        queue = {**queue_payload, "queue_value_sha256": digest_value(queue_payload)}
        queue_path = self.run_root / "queue.json"
        _create_or_verify(queue_path, queue, context="research queue")
        state["round1"] = {
            "outcome": outcome,
            "decision_path": str(decision_path),
            "decision_sha256": decision_sha256,
        }
        state["selected_recipe_ids"] = selected
        state["next_recipe_id"] = selected[0]
        state["next_proposal_path"] = str(proposal_path.resolve())
        state["next_proposal_sha256"] = proposal["proposal_sha256"]
        state["execution_handoff"] = None
        state["queue_path"] = str(queue_path)
        state["queue_sha256"] = digest_value(queue)
        state["status"] = "blocked-by-adapter"
        state["observed_round1_phase"] = "complete"
        state["blocker"] = {
            "kind": "blocked-by-adapter",
            "recipe_id": selected[0],
            "required_adapter": next_recipe["adapter"]["required_adapter"],
        }
        self._commit(
            state,
            event="round1_scientific_decision_recorded",
            details={"outcome": outcome, "next_recipe_id": selected[0]},
        )

    def _record_technical_failure(
        self,
        state: dict[str, Any],
        *,
        observation: Mapping[str, Any],
        catalog: Mapping[str, Any],
    ) -> None:
        decision = {
            "schema": ROUND1_DECISION_SCHEMA,
            "catalog_sha256": catalog["catalog_sha256"],
            "outcome": "technical-failure",
            "phase": observation["phase"],
            "failures": observation["failures"],
            "provenance": observation["provenance"],
        }
        decision_path = self.run_root / "rounds" / "round_001" / "decision.json"
        _create_or_verify(decision_path, decision, context="Round 1 technical decision")
        decision_sha256 = digest_value(decision)
        if state["round1"]["outcome"] != "pending":
            _require(
                state["round1"]["decision_sha256"] == decision_sha256,
                "Round 1 technical outcome conflicts with the durable state",
            )
            return
        state["round1"] = {
            "outcome": "technical-failure",
            "decision_path": str(decision_path),
            "decision_sha256": decision_sha256,
        }
        state["status"] = "blocked-technical-failure"
        state["observed_round1_phase"] = str(observation["phase"])
        state["next_recipe_id"] = None
        state["blocker"] = {
            "kind": "technical-failure",
            "round": 1,
            "failures": observation["failures"],
        }
        self._commit(
            state,
            event="round1_technical_failure_recorded",
            details={"failures": observation["failures"]},
        )

    def sync_round1(self) -> dict[str, Any]:
        """Audit Round 1 and durably route its scientific/technical outcome."""

        with self._locked() as state:
            catalog = self.catalog()
            observation = inspect_round1(
                campaign_root=Path(state["round1_campaign_root"]),
                bridge_root=Path(state["round1_bridge_root"]),
                catalog=catalog,
            )
            if observation["state"] == "decided":
                self._materialize_decision_and_queue(
                    state, observation=observation, catalog=catalog
                )
            elif observation["state"] == "technical-failure":
                self._record_technical_failure(
                    state, observation=observation, catalog=catalog
                )
            else:
                if state["round1"]["outcome"] != "pending":
                    raise AutoResearchError(
                        "Round 1 source regressed after a durable decision"
                    )
                desired_status = (
                    "r1-running" if observation["state"] == "running" else "waiting-r1"
                )
                phase = str(observation["phase"])
                if (
                    state["status"] != desired_status
                    or state["observed_round1_phase"] != phase
                ):
                    state["status"] = desired_status
                    state["observed_round1_phase"] = phase
                    self._commit(
                        state,
                        event="round1_progress_observed",
                        details={"status": desired_status, "phase": phase},
                    )
            return self.state()

    def _record_result(
        self, *, round_index: int, bridge_root: Path, historical_r2: bool
    ) -> dict[str, Any]:
        with self._locked() as state:
            if state["status"] == "blocked-technical-failure":
                raise AutoResearchError(
                    "cannot continue after a Round 1 technical failure"
                )
            _require(
                (round_index == 2 and historical_r2)
                or (round_index in {3, 4} and not historical_r2),
                "round result import mode is invalid",
            )
            catalog = self.catalog()
            completed = state["completed_rounds"]
            if str(round_index) in completed:
                proposal_path = (
                    self.run_root
                    / "rounds"
                    / f"round_{round_index:03d}"
                    / "proposal.json"
                )
                proposal = _validate_proposal(
                    _read_object(proposal_path), catalog=catalog
                )
                observed = (
                    _legacy_rank_result(
                        bridge_root=bridge_root, proposal=proposal, catalog=catalog
                    )
                    if round_index == 2
                    else _paired_round_result(
                        bridge_root=bridge_root, proposal=proposal, catalog=catalog
                    )
                )
                existing = _read_object(Path(completed[str(round_index)]["path"]))
                _require(
                    observed == existing,
                    "completed manager round differs from the supplied source",
                )
                return self.state()
            _require(
                state["next_recipe_id"] is not None
                and isinstance(state["next_proposal_path"], str),
                "no current manager proposal is available",
            )
            proposal = _validate_proposal(
                _read_object(Path(state["next_proposal_path"])),
                catalog=catalog,
                expected_recipe_id=state["next_recipe_id"],
            )
            _require(
                proposal["round"] == round_index,
                "round result is not for the current manager proposal",
            )
            if historical_r2:
                _require(
                    proposal["recipe_id"] == "r2-edge6-rank39-scale025",
                    "only the preregistered rejected scale-0.25 R2 may be imported",
                )
                result = _legacy_rank_result(
                    bridge_root=bridge_root, proposal=proposal, catalog=catalog
                )
                _require(
                    result["outcome"] == "scientific-rejected"
                    and result["recommended_role"] == "parent",
                    "historical R2 import accepts only the strict rejected outcome",
                )
            else:
                _require(
                    isinstance(state.get("execution_handoff"), dict),
                    "paired result has no manager-issued execution handoff",
                )
                result = _paired_round_result(
                    bridge_root=bridge_root, proposal=proposal, catalog=catalog
                )
            result = _validate_round_result(result, proposal=proposal, catalog=catalog)
            result_path = (
                self.run_root / "rounds" / f"round_{round_index:03d}" / "result.json"
            )
            _create_or_verify(
                result_path, result, context=f"Round {round_index} result"
            )
            state["completed_rounds"][str(round_index)] = {
                "path": str(result_path.resolve()),
                "file_sha256": sha256_file(result_path),
                "result_sha256": result["result_sha256"],
                "outcome": result["outcome"],
            }
            state["execution_handoff"] = None
            later_recipe_id = next(
                (
                    recipe_id
                    for recipe_id in state["selected_recipe_ids"]
                    if next(
                        row["round"]
                        for row in catalog["recipes"]
                        if row["recipe_id"] == recipe_id
                    )
                    == round_index + 1
                ),
                None,
            )
            if later_recipe_id is None:
                state["next_recipe_id"] = None
                state["next_proposal_path"] = None
                state["next_proposal_sha256"] = None
                state["status"] = "complete"
                state["blocker"] = None
            else:
                next_recipe = next(
                    row
                    for row in catalog["recipes"]
                    if row["recipe_id"] == later_recipe_id
                )
                lineage = _source_lineage(
                    source_round=round_index,
                    source_kind="round-result",
                    source_path=result_path,
                    source_value_sha256=result["result_sha256"],
                )
                next_proposal = _build_proposal(
                    catalog=catalog,
                    recipe=next_recipe,
                    parent_family=result["recommended_family"],
                    parent_artifacts=result["recommended_artifacts"],
                    parent_lineage=lineage,
                )
                next_path = (
                    self.run_root
                    / "rounds"
                    / f"round_{next_recipe['round']:03d}"
                    / "proposal.json"
                )
                _create_or_verify(
                    next_path,
                    next_proposal,
                    context=f"Round {next_recipe['round']} proposal",
                )
                state["next_recipe_id"] = later_recipe_id
                state["next_proposal_path"] = str(next_path.resolve())
                state["next_proposal_sha256"] = next_proposal["proposal_sha256"]
                if next_recipe["adapter"]["status"] == "available":
                    state["status"] = "ready-for-handoff"
                    state["blocker"] = None
                else:
                    state["status"] = "blocked-by-adapter"
                    state["blocker"] = {
                        "kind": "blocked-by-adapter",
                        "recipe_id": later_recipe_id,
                        "required_adapter": next_recipe["adapter"]["required_adapter"],
                    }
            event = (
                "historical_round2_imported"
                if historical_r2
                else "paired_round_result_recorded"
            )
            self._commit(
                state,
                event=event,
                details={
                    "round": round_index,
                    "recipe_id": proposal["recipe_id"],
                    "proposal_sha256": proposal["proposal_sha256"],
                    "result_sha256": result["result_sha256"],
                    "outcome": result["outcome"],
                    "source_kind": result["source_kind"],
                },
            )
            return self.state()

    def sync_historical_r2(self, bridge_root: Path) -> dict[str, Any]:
        """Import the sealed strict-rejected R2 without claiming manager launch."""

        return self._record_result(
            round_index=2, bridge_root=bridge_root, historical_r2=True
        )

    def record_round_result(
        self, round_index: int, bridge_root: Path
    ) -> dict[str, Any]:
        """Consume one manager-launched paired R3/R4 adjudication."""

        return self._record_result(
            round_index=round_index, bridge_root=bridge_root, historical_r2=False
        )

    def run_next(self) -> dict[str, Any]:
        """Materialize a hash-bound handoff; never start numerical work."""

        with self._locked() as state:
            if state["status"] == "blocked-technical-failure":
                raise AutoResearchError(
                    "cannot continue after a Round 1 technical failure"
                )
            if state["next_recipe_id"] is None:
                raise AutoResearchError("no next research recipe is resolved")
            catalog = self.catalog()
            recipe = next(
                row
                for row in catalog["recipes"]
                if row["recipe_id"] == state["next_recipe_id"]
            )
            if recipe["adapter"]["status"] != "available":
                raise AutoResearchError(
                    "next recipe is preregistered but blocked by the missing adapter: "
                    f"{recipe['adapter']['required_adapter']}"
                )
            _require(
                recipe["adapter"]["required_adapter"] in _EXECUTABLE_ADAPTERS,
                "next recipe adapter is not executable by this manager",
            )
            if isinstance(state.get("execution_handoff"), dict):
                handoff_path = Path(state["execution_handoff"]["path"])
                return validate_execution_handoff(
                    self.run_root, handoff_path, require_current=True
                )
            proposal_path = Path(str(state["next_proposal_path"])).resolve()
            proposal = _validate_proposal(
                _read_object(proposal_path),
                catalog=catalog,
                expected_recipe_id=recipe["recipe_id"],
            )
            payload = {
                "schema": EXECUTION_HANDOFF_SCHEMA,
                "manager_root": str(self.run_root),
                "catalog_sha256": catalog["catalog_sha256"],
                "manager_revision": state["revision"],
                "manager_state_sha256": state["state_sha256"],
                "manager_event_sha256": state["events"][-1]["event_sha256"],
                "round": proposal["round"],
                "proposal": {
                    "path": str(proposal_path),
                    "file_sha256": sha256_file(proposal_path),
                    "proposal_sha256": proposal["proposal_sha256"],
                },
                "recipe_id": recipe["recipe_id"],
                "recipe_sha256": recipe["recipe_sha256"],
                "adapter": recipe["adapter"],
                "parent_family": proposal["parent_family"],
                "parent_artifacts": proposal["parent_artifacts"],
                "parent_lineage": proposal["parent_lineage"],
                "execution_authorized": True,
                "launch_policy": (
                    "paired-bridge-prepare-only; GPU launch separately requires "
                    "fresh host-stability certificates"
                ),
            }
            handoff = {**payload, "handoff_sha256": digest_value(payload)}
            handoff_path = (
                self.run_root
                / "rounds"
                / f"round_{proposal['round']:03d}"
                / "execution_handoff.json"
            )
            _create_or_verify(
                handoff_path, handoff, context=f"Round {proposal['round']} handoff"
            )
            handoff_metadata = {
                "path": str(handoff_path.resolve()),
                "file_sha256": sha256_file(handoff_path),
                "handoff_sha256": handoff["handoff_sha256"],
                "round": proposal["round"],
            }
            state["execution_handoff"] = handoff_metadata
            state["status"] = "ready-for-paired-prepare"
            state["blocker"] = None
            self._commit(
                state,
                event="execution_handoff_materialized",
                details={
                    **handoff_metadata,
                    "proposal_sha256": proposal["proposal_sha256"],
                    "prior_state_sha256": handoff["manager_state_sha256"],
                },
            )
            return validate_execution_handoff(
                self.run_root, handoff_path, require_current=True
            )


def _verify_handoff_against_state(
    handoff: Mapping[str, Any],
    *,
    handoff_path: Path,
    manager: MultiRoundManager,
    state: Mapping[str, Any],
    require_current: bool = False,
) -> dict[str, Any]:
    catalog = manager.catalog()
    _require(
        handoff["manager_root"] == str(manager.run_root)
        and handoff["catalog_sha256"] == catalog["catalog_sha256"],
        "execution handoff belongs to another manager",
    )
    proposal_ref = handoff["proposal"]
    _require(isinstance(proposal_ref, dict), "execution handoff proposal is malformed")
    _exact_keys(
        proposal_ref,
        required={"path", "file_sha256", "proposal_sha256"},
        context="execution handoff proposal",
    )
    proposal_path = Path(str(proposal_ref["path"])).resolve()
    _require(
        proposal_path.is_file()
        and sha256_file(proposal_path) == proposal_ref["file_sha256"],
        "execution handoff proposal file drifted",
    )
    proposal = _validate_proposal(
        _read_object(proposal_path),
        catalog=catalog,
        expected_recipe_id=handoff["recipe_id"],
    )
    recipe = next(
        row for row in catalog["recipes"] if row["recipe_id"] == handoff["recipe_id"]
    )
    _require(
        proposal_ref["proposal_sha256"] == proposal["proposal_sha256"]
        and handoff["round"] == proposal["round"]
        and handoff["recipe_sha256"] == recipe["recipe_sha256"]
        and handoff["adapter"] == recipe["adapter"]
        and recipe["adapter"]["status"] == "available"
        and handoff["parent_family"] == proposal["parent_family"]
        and handoff["parent_artifacts"] == proposal["parent_artifacts"]
        and handoff["parent_lineage"] == proposal["parent_lineage"]
        and handoff["execution_authorized"] is True
        and handoff["launch_policy"]
        == (
            "paired-bridge-prepare-only; GPU launch separately requires "
            "fresh host-stability certificates"
        ),
        "execution handoff differs from the manager proposal/recipe",
    )
    revision = handoff["manager_revision"]
    _require(
        isinstance(revision, int)
        and 0 <= revision < len(state["events"]) - 0
        and state["events"][revision]["event_sha256"]
        == handoff["manager_event_sha256"],
        "execution handoff manager event is not in the durable chain",
    )
    _require(
        revision + 1 < len(state["events"]),
        "execution handoff has no durable materialization event",
    )
    event = state["events"][revision + 1]
    metadata = {
        "path": str(handoff_path.resolve()),
        "file_sha256": sha256_file(handoff_path),
        "handoff_sha256": handoff["handoff_sha256"],
        "round": handoff["round"],
    }
    _require(
        event["event"] == "execution_handoff_materialized"
        and event["previous_event_sha256"] == handoff["manager_event_sha256"]
        and event["details"]
        == {
            **metadata,
            "proposal_sha256": proposal["proposal_sha256"],
            "prior_state_sha256": handoff["manager_state_sha256"],
        },
        "execution handoff is not confirmed by the manager event chain",
    )
    if require_current:
        _require(
            state.get("next_recipe_id") == handoff["recipe_id"]
            and state.get("next_proposal_sha256") == proposal["proposal_sha256"]
            and state.get("execution_handoff") == metadata
            and state.get("status") == "ready-for-paired-prepare",
            "execution handoff is no longer the current manager round",
        )
    return dict(handoff)


def validate_execution_handoff(
    manager_root: Path, handoff_path: Path, *, require_current: bool = True
) -> dict[str, Any]:
    """Validate a manager-issued R3/R4 handoff without starting a worker."""

    manager = MultiRoundManager(manager_root)
    state = manager.state()
    resolved = handoff_path.expanduser().resolve()
    handoff = _read_handoff(resolved)
    return _verify_handoff_against_state(
        handoff,
        handoff_path=resolved,
        manager=manager,
        state=state,
        require_current=require_current,
    )


__all__ = [
    "CATALOG_SCHEMA",
    "EXECUTION_HANDOFF_SCHEMA",
    "MANAGER_SCHEMA",
    "MultiRoundManager",
    "ROUND_RESULT_SCHEMA",
    "inspect_round1",
    "validate_catalog",
    "validate_execution_handoff",
]
