#!/usr/bin/env python3
"""Create the preregistered X21 D16 promotion decision artifact.

The decision is intentionally a terminal, create-only workflow step.  It reads
the twelve complex128 endpoint JSON files and the primary/precision plateau
stage records directly, verifies their fixed protocol metadata and hashes, and
then evaluates the preregistered paired-seed rule.  No blind data are read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Sequence


SCHEMA = "x21-architecture-capacity-d16-promotion-decision-v1"
CAMPAIGN_ID = "x21-architecture-capacity-v1"
K_VALUES = (20, 24)
D_VALUES = (8, 14)
REPLICATES = (1, 2, 3)
STAGES = ("primary", "precision")
PARAMETER_COUNTS = {
    (20, 8): 282656,
    (24, 8): 344608,
    (20, 14): 860552,
    (24, 14): 1050280,
}
METRIC_FIELDS = (
    "sigma",
    "chi",
    "absolute_log_ratio_q999",
    "absolute_log_ratio_cvar_1pct",
    "minimum_metric_eigenvalue",
    "nonpositive_metric_count",
)


class DecisionError(RuntimeError):
    """Raised when the registered evidence is missing or malformed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _read_json(path: Path, *, role: str) -> tuple[dict[str, Any], str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise DecisionError(f"missing {role}: {resolved}")
    if resolved.stat().st_size <= 0:
        raise DecisionError(f"empty {role}: {resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DecisionError(f"could not read {role} {resolved}: {error}") from error
    if not isinstance(payload, dict):
        raise DecisionError(f"{role} must be a JSON object: {resolved}")
    return payload, sha256_file(resolved)


def _require_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise DecisionError(f"{field} must be a non-empty string")
    return value


def _require_int(value: Any, *, field: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DecisionError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise DecisionError(f"{field} must be >= {minimum}")
    return value


def _require_number(
    value: Any,
    *,
    field: str,
    minimum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DecisionError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise DecisionError(f"{field} must be finite")
    if minimum is not None and result < minimum:
        raise DecisionError(f"{field} must be >= {minimum}")
    return result


def _relative_path(path: Path, run_root: Path, *, role: str) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(run_root))
    except ValueError as error:
        raise DecisionError(f"{role} escapes run root: {resolved}") from error


def _resolve_stage_member(stage_dir: Path, value: Any, *, field: str) -> Path:
    relative = Path(_require_string(value, field=field))
    if relative.is_absolute():
        raise DecisionError(f"{field} must be relative to the stage directory")
    resolved = (stage_dir / relative).resolve()
    try:
        resolved.relative_to(stage_dir)
    except ValueError as error:
        raise DecisionError(f"{field} escapes the stage directory") from error
    return resolved


def _validate_training_summary(
    payload: dict[str, Any],
    *,
    field_prefix: str,
    k: int,
    bond_dimension: int,
    parameter_count: int,
    positive_floor: float,
    train_pool: Path,
    train_pool_sha256: str,
    selection_pool: Path,
    selection_pool_sha256: str,
) -> dict[str, Any]:
    expected_scalars = {
        "schema": "type11-positive-tensor-network-training-v1",
        "site_count": k,
        "bond_dimension": bond_dimension,
        "trainable_real_parameter_count": parameter_count,
        "physical_dictionary_rank": 121,
        "trainable_physical_dictionary": False,
        "physical_dictionary_gauge": "fixed",
        "positive_floor": positive_floor,
        "precision": "complex64",
        "batch_size": 1024,
    }
    for field, expected in expected_scalars.items():
        observed = payload.get(field)
        if observed != expected:
            raise DecisionError(
                f"{field_prefix}.{field} is {observed!r}, expected {expected!r}"
            )

    train = payload.get("train")
    validation = payload.get("validation")
    if not isinstance(train, dict) or not isinstance(validation, dict):
        raise DecisionError(f"{field_prefix} train/validation metadata is missing")
    expected_pools = (
        (
            "train",
            train,
            196608,
            86201,
            train_pool,
            train_pool_sha256,
        ),
        (
            "validation",
            validation,
            24576,
            86202,
            selection_pool,
            selection_pool_sha256,
        ),
    )
    for role, observed, points, seed, pool, pool_sha256 in expected_pools:
        if observed.get("points") != points:
            raise DecisionError(f"{field_prefix}.{role}.points must equal {points}")
        if observed.get("seed") != seed:
            raise DecisionError(f"{field_prefix}.{role}.seed must equal {seed}")
        observed_pool = Path(
            _require_string(observed.get("common_pool"), field=f"{field_prefix}.{role}.common_pool")
        ).expanduser().resolve()
        if observed_pool != pool:
            raise DecisionError(
                f"{field_prefix}.{role}.common_pool is {observed_pool}, expected {pool}"
            )
        if observed.get("common_pool_sha256") != pool_sha256:
            raise DecisionError(
                f"{field_prefix}.{role}.common_pool_sha256 does not match preregistration"
            )

    history = payload.get("history")
    if not isinstance(history, list) or not history:
        raise DecisionError(f"{field_prefix}.history must be a non-empty list")
    epochs: list[int] = []
    for index, row in enumerate(history):
        if not isinstance(row, dict):
            raise DecisionError(f"{field_prefix}.history.{index} must be an object")
        epochs.append(
            _require_int(
                row.get("epoch"),
                field=f"{field_prefix}.history.{index}.epoch",
                minimum=0,
            )
        )
    last_epoch = epochs[-1]
    if epochs != list(range(last_epoch + 1)):
        raise DecisionError(
            f"{field_prefix}.history epochs must be contiguous from zero"
        )
    runtime_seconds = _require_number(
        payload.get("runtime_seconds"),
        field=f"{field_prefix}.runtime_seconds",
        minimum=0.0,
    )
    train_points = _require_int(
        train.get("points"), field=f"{field_prefix}.train.points", minimum=1
    )
    batch_size = _require_int(
        payload.get("batch_size"), field=f"{field_prefix}.batch_size", minimum=1
    )
    optimizer_steps_per_epoch = (train_points + batch_size - 1) // batch_size
    return {
        "actual_training_epochs": last_epoch,
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "actual_optimizer_updates": last_epoch * optimizer_steps_per_epoch,
        "validation_evaluations_including_epoch_zero": len(history),
        "trainer_runtime_seconds": runtime_seconds,
        "termination_reason": _require_string(
            payload.get("termination_reason"),
            field=f"{field_prefix}.termination_reason",
        ),
    }


def _read_stage_exposure(
    *,
    run_root: Path,
    stage: str,
    k: int,
    bond_dimension: int,
    replicate: int,
    parameter_count: int,
    positive_floor: float,
    train_pool: Path,
    train_pool_sha256: str,
    selection_pool: Path,
    selection_pool_sha256: str,
) -> dict[str, Any]:
    stage_dir = (
        run_root
        / "jobs"
        / "grid"
        / f"k{k}_d{bond_dimension}_r{replicate}"
        / stage
    ).resolve()
    stage_summary_path = stage_dir / "stage_summary.json"
    stage_payload, stage_sha256 = _read_json(
        stage_summary_path, role=f"{stage} stage summary"
    )
    if stage_payload.get("schema") != "gcicy-tn-plateau-stage-v1":
        raise DecisionError(f"{stage_summary_path} has the wrong schema")
    if stage_payload.get("status") != "validation_plateau":
        raise DecisionError(f"{stage_summary_path} is not a completed plateau")
    rounds = stage_payload.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        raise DecisionError(f"{stage_summary_path} has no completed rounds")
    rounds_completed = _require_int(
        stage_payload.get("rounds_completed"),
        field=f"{stage}.rounds_completed",
        minimum=1,
    )
    plateau_round = _require_int(
        stage_payload.get("plateau_round"),
        field=f"{stage}.plateau_round",
        minimum=1,
    )
    if rounds_completed != len(rounds) or plateau_round != len(rounds):
        raise DecisionError(f"{stage_summary_path} has inconsistent round counts")

    plateau_summary_path = _resolve_stage_member(
        stage_dir,
        stage_payload.get("plateau_summary"),
        field=f"{stage}.plateau_summary",
    )
    plateau_payload, plateau_sha256 = _read_json(
        plateau_summary_path, role=f"{stage} plateau summary"
    )
    if stage_payload.get("plateau_summary_sha256") != plateau_sha256:
        raise DecisionError(f"{stage_summary_path} plateau summary hash mismatch")

    round_exposures: list[dict[str, Any]] = []
    for expected_round, record in enumerate(rounds, start=1):
        if not isinstance(record, dict):
            raise DecisionError(f"{stage}.rounds.{expected_round - 1} must be an object")
        if record.get("round") != expected_round:
            raise DecisionError(f"{stage} round records must be consecutive")
        summary_path = _resolve_stage_member(
            stage_dir,
            record.get("summary"),
            field=f"{stage}.rounds.{expected_round}.summary",
        )
        summary_payload, summary_sha256 = _read_json(
            summary_path, role=f"{stage} round {expected_round} summary"
        )
        if record.get("summary_sha256") != summary_sha256:
            raise DecisionError(f"{stage} round {expected_round} summary hash mismatch")
        exposure = _validate_training_summary(
            summary_payload,
            field_prefix=f"{stage}.rounds.{expected_round}.summary",
            k=k,
            bond_dimension=bond_dimension,
            parameter_count=parameter_count,
            positive_floor=positive_floor,
            train_pool=train_pool,
            train_pool_sha256=train_pool_sha256,
            selection_pool=selection_pool,
            selection_pool_sha256=selection_pool_sha256,
        )
        if record.get("termination_reason") != exposure["termination_reason"]:
            raise DecisionError(f"{stage} round {expected_round} termination mismatch")
        if expected_round == len(rounds):
            if exposure["termination_reason"] != "validation_plateau":
                raise DecisionError(f"{stage} final round is not a validation plateau")
            if summary_sha256 != plateau_sha256:
                raise DecisionError(f"{stage} plateau summary is not the final round summary")
        elif exposure["termination_reason"] != "completed_requested_epochs":
            raise DecisionError(f"{stage} non-final round has an invalid termination")
        round_exposures.append(
            {
                "round": expected_round,
                "summary": _relative_path(summary_path, run_root, role="round summary"),
                "summary_sha256": summary_sha256,
                **exposure,
            }
        )

    return {
        "stage": stage,
        "stage_summary": _relative_path(
            stage_summary_path, run_root, role="stage summary"
        ),
        "stage_summary_sha256": stage_sha256,
        "plateau_summary": _relative_path(
            plateau_summary_path, run_root, role="plateau summary"
        ),
        "plateau_summary_sha256": plateau_sha256,
        "rounds_completed": rounds_completed,
        "plateau_round": plateau_round,
        "actual_training_epochs": sum(
            row["actual_training_epochs"] for row in round_exposures
        ),
        "actual_optimizer_updates": sum(
            row["actual_optimizer_updates"] for row in round_exposures
        ),
        "validation_evaluations_including_epoch_zero": sum(
            row["validation_evaluations_including_epoch_zero"]
            for row in round_exposures
        ),
        "trainer_runtime_seconds": sum(
            row["trainer_runtime_seconds"] for row in round_exposures
        ),
        "rounds": round_exposures,
    }


def _read_endpoint(
    *, run_root: Path, k: int, bond_dimension: int, replicate: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    label = f"k{k}-d{bond_dimension}-r{replicate}"
    endpoint_path = (
        run_root
        / "jobs"
        / "precision_replay"
        / f"k{k}_d{bond_dimension}_r{replicate}"
        / "tails.json"
    ).resolve()
    payload, endpoint_sha256 = _read_json(endpoint_path, role=f"endpoint {label}")
    if payload.get("schema") != "gcicy-bilateral-tail-array-evaluation-v1":
        raise DecisionError(f"endpoint {label} has the wrong schema")
    models = payload.get("models")
    if not isinstance(models, dict) or set(models) != {label}:
        raise DecisionError(f"endpoint {label} must contain exactly its registered label")
    model = models[label]
    if not isinstance(model, dict) or not isinstance(model.get("metrics"), dict):
        raise DecisionError(f"endpoint {label} metrics are missing")
    raw_metrics = model["metrics"]
    missing = [field for field in METRIC_FIELDS if field not in raw_metrics]
    if missing:
        raise DecisionError(f"endpoint {label} is missing metrics: {missing}")
    metrics = {
        field: _require_number(
            raw_metrics[field],
            field=f"endpoint.{label}.{field}",
            minimum=0.0 if field != "minimum_metric_eigenvalue" else None,
        )
        for field in METRIC_FIELDS
        if field != "nonpositive_metric_count"
    }
    metrics["nonpositive_metric_count"] = _require_int(
        raw_metrics["nonpositive_metric_count"],
        field=f"endpoint.{label}.nonpositive_metric_count",
        minimum=0,
    )
    source = {
        "path": _relative_path(endpoint_path, run_root, role="endpoint"),
        "sha256": endpoint_sha256,
    }
    return metrics, source


def _mean(rows: Sequence[dict[str, Any]], field: str) -> float:
    return sum(float(row["metrics"][field]) for row in rows) / len(rows)


def _ratio(candidate: float, baseline: float, *, field: str) -> float:
    if baseline <= 0.0:
        raise DecisionError(
            f"k24,D8 arithmetic mean for {field} must be positive to define a ratio"
        )
    return candidate / baseline


def build_decision(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.expanduser().resolve()
    if not run_root.is_dir():
        raise DecisionError(f"run root does not exist: {run_root}")
    manifest = args.manifest.expanduser().resolve()
    manifest_payload, manifest_sha256 = _read_json(manifest, role="manifest")
    if manifest_payload.get("campaign_id") != CAMPAIGN_ID:
        raise DecisionError(f"manifest campaign_id must equal {CAMPAIGN_ID}")

    train_pool = args.train_pool.expanduser().resolve()
    selection_pool = args.selection_pool.expanduser().resolve()
    rows: list[dict[str, Any]] = []
    for k in K_VALUES:
        for bond_dimension in D_VALUES:
            parameter_count = PARAMETER_COUNTS[(k, bond_dimension)]
            for replicate in REPLICATES:
                metrics, endpoint_source = _read_endpoint(
                    run_root=run_root,
                    k=k,
                    bond_dimension=bond_dimension,
                    replicate=replicate,
                )
                stages = [
                    _read_stage_exposure(
                        run_root=run_root,
                        stage=stage,
                        k=k,
                        bond_dimension=bond_dimension,
                        replicate=replicate,
                        parameter_count=parameter_count,
                        positive_floor=args.positive_floor,
                        train_pool=train_pool,
                        train_pool_sha256=args.train_pool_sha256,
                        selection_pool=selection_pool,
                        selection_pool_sha256=args.selection_pool_sha256,
                    )
                    for stage in STAGES
                ]
                total_exposure = {
                    "actual_training_epochs": sum(
                        stage["actual_training_epochs"] for stage in stages
                    ),
                    "actual_optimizer_updates": sum(
                        stage["actual_optimizer_updates"] for stage in stages
                    ),
                    "validation_evaluations_including_epoch_zero": sum(
                        stage["validation_evaluations_including_epoch_zero"]
                        for stage in stages
                    ),
                    "trainer_runtime_seconds": sum(
                        stage["trainer_runtime_seconds"] for stage in stages
                    ),
                }
                positive = (
                    metrics["minimum_metric_eigenvalue"] > 0.0
                    and metrics["nonpositive_metric_count"] == 0
                )
                rows.append(
                    {
                        "row_id": f"k{k}-d{bond_dimension}-r{replicate}",
                        "factors": {
                            "k": k,
                            "D": bond_dimension,
                            "replicate": replicate,
                            "trainable_real_parameter_count": parameter_count,
                        },
                        "endpoint": endpoint_source,
                        "metrics": metrics,
                        "positive": positive,
                        "training_exposure": {
                            "stages": stages,
                            "total": total_exposure,
                        },
                    }
                )

    expected_keys = {
        (k, bond_dimension, replicate)
        for k in K_VALUES
        for bond_dimension in D_VALUES
        for replicate in REPLICATES
    }
    observed_keys = {
        (row["factors"]["k"], row["factors"]["D"], row["factors"]["replicate"])
        for row in rows
    }
    if len(rows) != 12 or observed_keys != expected_keys:
        raise DecisionError("the adjudication row set is not the complete 2x2x3 grid")
    rows.sort(
        key=lambda row: (
            row["factors"]["k"],
            row["factors"]["D"],
            row["factors"]["replicate"],
        )
    )

    baseline = [
        row for row in rows if row["factors"]["k"] == 24 and row["factors"]["D"] == 8
    ]
    candidate = [
        row for row in rows if row["factors"]["k"] == 24 and row["factors"]["D"] == 14
    ]
    baseline_by_seed = {row["factors"]["replicate"]: row for row in baseline}
    candidate_by_seed = {row["factors"]["replicate"]: row for row in candidate}
    if set(baseline_by_seed) != set(REPLICATES) or set(candidate_by_seed) != set(REPLICATES):
        raise DecisionError("k24,D8 and k24,D14 must each contain seeds 1,2,3")

    mean_fields = (
        "sigma",
        "chi",
        "absolute_log_ratio_q999",
        "absolute_log_ratio_cvar_1pct",
    )
    baseline_means = {field: _mean(baseline, field) for field in mean_fields}
    candidate_means = {field: _mean(candidate, field) for field in mean_fields}
    ratios = {
        field: _ratio(candidate_means[field], baseline_means[field], field=field)
        for field in mean_fields
    }
    paired_sigma = []
    for replicate in REPLICATES:
        baseline_sigma = float(baseline_by_seed[replicate]["metrics"]["sigma"])
        candidate_sigma = float(candidate_by_seed[replicate]["metrics"]["sigma"])
        paired_sigma.append(
            {
                "replicate": replicate,
                "baseline_k24_d8": baseline_sigma,
                "candidate_k24_d14": candidate_sigma,
                "baseline_minus_candidate": baseline_sigma - candidate_sigma,
                "strict_win": candidate_sigma < baseline_sigma,
                "tie": candidate_sigma == baseline_sigma,
            }
        )
    strict_wins = sum(bool(row["strict_win"]) for row in paired_sigma)
    ties = sum(bool(row["tie"]) for row in paired_sigma)
    all_positive = all(bool(row["positive"]) for row in rows)

    conditions = {
        "complete_twelve_row_grid": True,
        "all_twelve_rows_strictly_positive": all_positive,
        "k24_d14_over_d8_arithmetic_mean_sigma_ratio_le_0p98": (
            ratios["sigma"] <= args.mean_sigma_ratio_max
        ),
        "k24_d14_over_d8_paired_sigma_strict_wins_ge_2": (
            strict_wins >= args.paired_sigma_wins_min
        ),
        "k24_d14_over_d8_arithmetic_mean_chi_ratio_le_1": (
            ratios["chi"] <= args.mean_chi_ratio_max
        ),
        "k24_d14_over_d8_arithmetic_mean_q999_ratio_le_1p05": (
            ratios["absolute_log_ratio_q999"] <= args.mean_q999_ratio_max
        ),
        "k24_d14_over_d8_arithmetic_mean_cvar_ratio_le_1p05": (
            ratios["absolute_log_ratio_cvar_1pct"] <= args.mean_cvar_ratio_max
        ),
    }
    promote = all(conditions.values())
    adjudicator = Path(__file__).resolve()
    return {
        "schema": SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "complete": True,
        "row_count": len(rows),
        "completeness": {
            "expected_k": list(K_VALUES),
            "expected_D": list(D_VALUES),
            "expected_replicates": list(REPLICATES),
            "expected_rows": 12,
            "observed_rows": len(rows),
            "exact_cartesian_closure": observed_keys == expected_keys,
        },
        "estimand": {
            "name": "uniform-validation-plateau-stopping-rule endpoint contrast",
            "definition": (
                "Each cell uses the same registered primary and precision plateau "
                "stopping rules; endpoints are compared after their own rule-triggered "
                "training exposure."
            ),
            "not_claimed": "fixed-update pure architecture-capacity effect",
            "exposure_accounting": (
                "Actual epochs, optimizer updates, validation evaluations, completed "
                "rounds, trainer runtime, and source-summary hashes are recorded per seed."
            ),
        },
        "fixed_protocol": {
            "physical_dictionary_rank": 121,
            "trainable_physical_dictionary": False,
            "physical_dictionary_gauge": "fixed",
            "positive_floor": args.positive_floor,
            "training_precision": "complex64",
            "evaluation_precision": "complex128",
            "train_pool": str(train_pool),
            "train_pool_sha256": args.train_pool_sha256,
            "selection_pool": str(selection_pool),
            "selection_pool_sha256": args.selection_pool_sha256,
            "development_split": "confirmation",
            "blind_inputs_read": False,
        },
        "manifest": {
            "path": str(manifest),
            "sha256": manifest_sha256,
        },
        "adjudicator": {
            "path": str(adjudicator),
            "sha256": sha256_file(adjudicator),
            "create_only": True,
        },
        "rows": rows,
        "rows_sha256": digest_value(rows),
        "all_valid": all_positive,
        "comparison": {
            "baseline": {"k": 24, "D": 8},
            "candidate": {"k": 24, "D": 14},
            "replicates": list(REPLICATES),
            "aggregation": (
                "For sigma, chi, absolute_log_ratio_q999, and "
                "absolute_log_ratio_cvar_1pct, ratio = arithmetic mean across "
                "the three candidate seeds divided by arithmetic mean across "
                "the three baseline seeds."
            ),
            "baseline_arithmetic_means": baseline_means,
            "candidate_arithmetic_means": candidate_means,
            "candidate_over_baseline_arithmetic_mean_ratios": ratios,
            "paired_sigma": paired_sigma,
            "paired_sigma_strict_wins": strict_wins,
            "paired_sigma_ties": ties,
            "tie_policy": "A tie is not a strict win.",
        },
        "preregistered_thresholds": {
            "mean_sigma_ratio_max": args.mean_sigma_ratio_max,
            "paired_sigma_strict_wins_min": args.paired_sigma_wins_min,
            "mean_chi_ratio_max": args.mean_chi_ratio_max,
            "mean_q999_ratio_max": args.mean_q999_ratio_max,
            "mean_cvar_ratio_max": args.mean_cvar_ratio_max,
        },
        "conditions": conditions,
        "rules_evaluated": True,
        "decision": {
            "promote_d16": promote,
            "outcome": "promote" if promote else "do_not_promote",
            "extension_requirement": (
                "Any D16 accuracy run requires a separate hash-frozen manifest and "
                "a fresh RUN_ROOT after this decision artifact is frozen."
            ),
        },
    }


def write_create_only(path: Path, payload: dict[str, Any]) -> str:
    """Create an immutable artifact, or verify an identical crash-window retry."""

    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    try:
        descriptor = os.open(resolved, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        try:
            existing_bytes = resolved.read_bytes()
            existing = json.loads(existing_bytes)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise DecisionError(
                f"existing decision is unreadable and cannot be replaced: {resolved}"
            ) from error
        try:
            identical = canonical_json_bytes(existing) == canonical_json_bytes(payload)
        except (TypeError, ValueError) as error:
            raise DecisionError(
                f"existing decision is not canonical finite JSON: {resolved}"
            ) from error
        if not identical:
            raise DecisionError(
                f"existing decision differs and cannot be replaced: {resolved}"
            )
        return hashlib.sha256(existing_bytes).hexdigest()
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        resolved.unlink(missing_ok=True)
        raise
    return hashlib.sha256(encoded).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-pool", type=Path, required=True)
    parser.add_argument("--train-pool-sha256", required=True)
    parser.add_argument("--selection-pool", type=Path, required=True)
    parser.add_argument("--selection-pool-sha256", required=True)
    parser.add_argument("--positive-floor", type=float, required=True)
    parser.add_argument("--mean-sigma-ratio-max", type=float, required=True)
    parser.add_argument("--paired-sigma-wins-min", type=int, required=True)
    parser.add_argument("--mean-chi-ratio-max", type=float, required=True)
    parser.add_argument("--mean-q999-ratio-max", type=float, required=True)
    parser.add_argument("--mean-cvar-ratio-max", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    for field in (
        "mean_sigma_ratio_max",
        "mean_chi_ratio_max",
        "mean_q999_ratio_max",
        "mean_cvar_ratio_max",
    ):
        value = getattr(args, field)
        if not math.isfinite(value) or value <= 0.0:
            build_parser().error(f"--{field.replace('_', '-')} must be positive and finite")
    if args.paired_sigma_wins_min < 1 or args.paired_sigma_wins_min > len(REPLICATES):
        build_parser().error("--paired-sigma-wins-min must be between 1 and 3")
    if not math.isfinite(args.positive_floor) or args.positive_floor <= 0.0:
        build_parser().error("--positive-floor must be positive and finite")
    for field in ("train_pool_sha256", "selection_pool_sha256"):
        value = getattr(args, field).lower()
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            build_parser().error(f"--{field.replace('_', '-')} must be a SHA-256 hex digest")
        setattr(args, field, value)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        output_path = args.out.expanduser().resolve()
        payload = build_decision(args)
        decision_sha256 = write_create_only(output_path, payload)
    except DecisionError as error:
        print(f"[x21-capacity-decision] error: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "out": str(output_path),
                "sha256": decision_sha256,
                "promote_d16": payload["decision"]["promote_d16"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
