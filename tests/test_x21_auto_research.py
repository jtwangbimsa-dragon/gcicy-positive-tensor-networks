from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest


from gcicy_metric.pipeline.x21_auto_research import (
    ACTION_SCHEMA,
    BASELINE_COMPLETION_SCHEMA,
    EVIDENCE_SCHEMA,
    FAMILY_SCHEMA,
    FRESH_DEVELOPMENT_BINDING_SCHEMA,
    HISTORICAL_CONFIRMATION_SHA256,
    PROTOCOL_V2_SCHEMA,
    RECOVERY_EVIDENCE_SCHEMA,
    RESOURCE_CERTIFICATE_SCHEMA,
    RESOURCE_SKIP_SCHEMA,
    RecoveryStore,
    X21AutoResearchError,
    X21CampaignStore,
    coefficient_real_parameters,
    digest_value,
    validate_action,
    validate_baseline_completion_evidence,
    validate_protocol,
    validate_recovery_evidence,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "experiments" / "protocols" / "x21_auto_research_v1.json"


def protocol_raw() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def protocol_v2_raw() -> dict:
    value = protocol_raw()
    fresh_pool_sha256 = "7" * 64
    value["schema"] = PROTOCOL_V2_SCHEMA
    value["campaign_id"] = "x21-auto-research-v2"
    value["data_contract"]["development_pool_sha256"] = fresh_pool_sha256
    value["development_pool_binding"] = {
        "schema": FRESH_DEVELOPMENT_BINDING_SCHEMA,
        "binding_sha256": "8" * 64,
        "pool_sha256": fresh_pool_sha256,
        "manifest_sha256": "9" * 64,
        "generation_receipt_sha256": "a" * 64,
        "adapter": "p5p1_type21_k3_1223",
        "model_seed": 20260802,
        "exact_model": True,
        "split": "selection",
        "points": 49152,
        "seed": 86206,
        "sampling_cluster_size": 6,
        "historical_confirmation_sha256": HISTORICAL_CONFIRMATION_SHA256,
    }
    return value


def baseline_family() -> dict:
    certificate = validate_recovery_evidence(recovery_evidence())
    completion_1 = validate_baseline_completion_evidence(completion_evidence(8660001))
    completion_3 = validate_baseline_completion_evidence(completion_evidence(8660003))
    return {
        "schema": FAMILY_SCHEMA,
        "model_metadata": {
            "k": 20,
            "bond_dimension": 14,
            "train_physical_dictionary": False,
            "trainable_real_parameter_count": coefficient_real_parameters(20, 14),
            "precision": "complex64",
        },
        "models": [
            {
                "seed": 8660001,
                "checkpoint_sha256": "1" * 64,
                "provenance": {
                    "kind": "existing-complete",
                    "certificate_sha256": digest_value(completion_1),
                    "certificate": completion_1,
                },
            },
            {
                "seed": 8660002,
                "checkpoint_sha256": certificate["output_model_sha256"],
                "provenance": {
                    "kind": "exact-recovery",
                    "certificate_sha256": digest_value(certificate),
                    "certificate": certificate,
                },
            },
            {
                "seed": 8660003,
                "checkpoint_sha256": "3" * 64,
                "provenance": {
                    "kind": "fresh-completion",
                    "certificate_sha256": digest_value(completion_3),
                    "certificate": completion_3,
                },
            },
        ],
    }


def candidate_metadata(parent: dict, kind: str) -> dict:
    result = copy.deepcopy(parent["model_metadata"])
    if kind == "dictionary_unfreeze":
        result["train_physical_dictionary"] = True
    elif kind == "bond_growth_d16":
        result["bond_dimension"] = 16
    elif kind == "site_growth_k24":
        result["k"] = 24
    result["trainable_real_parameter_count"] = coefficient_real_parameters(
        result["k"], result["bond_dimension"]
    ) + (2 * 121**2 if result["train_physical_dictionary"] else 0)
    return result


def resource_certificate_raw(
    protocol: dict,
    parent: dict,
    round_index: int,
    *,
    outcome: str,
    selected_batch_size: int | None,
) -> dict:
    kind = protocol["action_sequence"][round_index - 1]["kind"]
    ladder = protocol["allowed_memory_batch_sizes"]
    if outcome == "resource-infeasible":
        attempts = ladder
    elif selected_batch_size in ladder:
        attempts = ladder[: ladder.index(selected_batch_size) + 1]
    else:
        attempts = [selected_batch_size]
    return {
        "schema": RESOURCE_CERTIFICATE_SCHEMA,
        "round": round_index,
        "kind": kind,
        "candidate_metadata_sha256": digest_value(candidate_metadata(parent, kind)),
        "attempted_batch_sizes": attempts,
        "selected_batch_size": selected_batch_size,
        "outcome": outcome,
        "preflight_report_sha256": "f" * 64,
    }


def action_raw(
    protocol: dict,
    parent: dict,
    round_index: int,
    *,
    batch_size: int | None = None,
    resource: bool = False,
) -> dict:
    sequence = protocol["action_sequence"][round_index - 1]
    kind = sequence["kind"]
    mutation = {
        "kind": kind,
        "field": sequence["field"],
        "control_value": sequence["control_value"],
        "candidate_value": sequence["candidate_value"],
    }
    metadata = parent["model_metadata"]
    if kind == "dictionary_unfreeze":
        mutation["new_real_parameters"] = 2 * 121**2
    elif kind == "bond_growth_d16":
        after = copy.deepcopy(metadata)
        after["bond_dimension"] = 16
        after["trainable_real_parameter_count"] = coefficient_real_parameters(
            after["k"], 16
        ) + (2 * 121**2 if after["train_physical_dictionary"] else 0)
        mutation.update(
            {
                "transport": "one-sided-function-preserving",
                "activation_scale": 0.03,
                "new_real_parameters": (
                    after["trainable_real_parameter_count"]
                    - metadata["trainable_real_parameter_count"]
                ),
            }
        )
    elif kind == "site_growth_k24":
        after = copy.deepcopy(metadata)
        after["k"] = 24
        after["trainable_real_parameter_count"] = coefficient_real_parameters(
            24, after["bond_dimension"]
        ) + (2 * 121**2 if after["train_physical_dictionary"] else 0)
        mutation.update(
            {
                "transport": "repeat-sites-function-preserving",
                "activation_scale": 0.03,
                "new_real_parameters": (
                    after["trainable_real_parameter_count"]
                    - metadata["trainable_real_parameter_count"]
                ),
            }
        )
    budget = copy.deepcopy(protocol["budget"])
    if batch_size is not None:
        budget["batch_size"] = batch_size
    value = {
        "schema": ACTION_SCHEMA,
        "candidate_id": f"round-{round_index}-{kind}",
        "round": round_index,
        "parent_family_sha256": parent["family_sha256"],
        "data_contract_sha256": protocol["data_contract_sha256"],
        "mutation": mutation,
        "budget": budget,
        "pipeline": [
            "epoch-zero-equivalence",
            "matched-fixed-budget",
            "paired-development-evaluation",
        ],
    }
    if resource:
        certificate = resource_certificate_raw(
            protocol,
            parent,
            round_index,
            outcome="feasible",
            selected_batch_size=budget["batch_size"],
        )
        value["resource_certificate"] = certificate
        value["resource_certificate_sha256"] = digest_value(certificate)
    return value


def metrics(scale: float = 1.0) -> dict:
    return {
        "sigma": 0.01 * scale,
        "chi": 0.02 * scale,
        "q999": 0.05,
        "cvar_1pct": 0.04,
        "minimum_metric_eigenvalue": 0.04,
        "nonpositive_metric_count": 0,
    }


def evidence_raw(
    action: dict, parent: dict, protocol: dict, *, scale: float = 0.99
) -> dict:
    rows = []
    parent_hashes = {row["seed"]: row["checkpoint_sha256"] for row in parent["models"]}
    for offset, seed in enumerate(protocol["promotion_seeds"], start=1):
        rows.append(
            {
                "seed": seed,
                "parent_checkpoint_sha256": parent_hashes[seed],
                "data_contract_sha256": protocol["data_contract_sha256"],
                "control_budget": action["budget"],
                "candidate_budget": action["budget"],
                "control_batch_plan_sha256": f"{offset + 3:x}" * 64,
                "candidate_batch_plan_sha256": f"{offset + 3:x}" * 64,
                "equivalence": {
                    "passed": True,
                    "potential_max_absolute": 0.0,
                    "metric_max_relative_frobenius": 0.0,
                },
                "control": {
                    "checkpoint_sha256": f"{offset + 6:x}" * 64,
                    "metadata": parent["model_metadata"],
                    "metrics": metrics(),
                },
                "candidate": {
                    "checkpoint_sha256": f"{offset + 9:x}" * 64,
                    "metadata": action["candidate_metadata"],
                    "metrics": metrics(scale),
                },
                "paired": {
                    "sigma_ci95_low": 1.0e-5,
                    "chi_ci95_low": 1.0e-5,
                    "bootstrap_replicates": action["budget"]["bootstrap_replicates"],
                },
                "source_report_sha256": {
                    "control_training": "a" * 64,
                    "candidate_training": "b" * 64,
                    "paired_evaluation": "c" * 64,
                },
            }
        )
    return {
        "schema": EVIDENCE_SCHEMA,
        "candidate_id": action["candidate_id"],
        "round": action["round"],
        "action_sha256": action["action_sha256"],
        "parent_family_sha256": parent["family_sha256"],
        "data_contract_sha256": protocol["data_contract_sha256"],
        "precision": "complex64",
        "seeds": rows,
    }


def recovery_evidence() -> dict:
    contract = next(
        row
        for row in protocol_raw()["baseline_provenance_contracts"]
        if row["seed"] == 8660002
    )
    return {
        "schema": RECOVERY_EVIDENCE_SCHEMA,
        "campaign_id": contract["recovery_campaign_id"],
        "seed": 8660002,
        "mode": "exact-checkpoint-resume",
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
        "implementation_match": True,
        "optimizer_state_restored": True,
        "rng_state_restored": True,
        "terminal_status": "validation_plateau",
        "native_exit_code": 0,
        "operational_attempt_count": 1,
        "output_model_sha256": "4" * 64,
        "output_checkpoint_sha256": "5" * 64,
        "output_summary_sha256": "6" * 64,
    }


def completion_evidence(seed: int) -> dict:
    contract = next(
        row
        for row in protocol_raw()["baseline_provenance_contracts"]
        if row["seed"] == seed
    )
    checkpoint = "1" * 64 if seed == 8660001 else "3" * 64
    return {
        "schema": BASELINE_COMPLETION_SCHEMA,
        "seed": seed,
        "source_campaign_id": contract["source_campaign_id"],
        "source_plan_sha256": contract["source_plan_sha256"],
        "source_job_id": contract["source_job_id"],
        "source_job_digest": contract["source_job_digest"],
        "terminal_status": "validation_plateau",
        "native_exit_code": 0,
        "output_model_sha256": checkpoint,
        "output_checkpoint_sha256": checkpoint,
        "output_summary_sha256": hashlib.sha256(str(seed).encode()).hexdigest(),
    }


def initialized_campaign(tmp_path: Path) -> tuple[X21CampaignStore, dict, dict]:
    raw = protocol_v2_raw()
    protocol = validate_protocol(raw)
    store = X21CampaignStore.initialize(tmp_path / "science", raw)
    family = store.register_baseline(baseline_family())
    return store, protocol, family


def test_checked_protocol_locks_four_single_variable_rounds():
    protocol = validate_protocol(protocol_raw())
    assert [row["kind"] for row in protocol["action_sequence"]] == [
        "optimizer_path",
        "dictionary_unfreeze",
        "bond_growth_d16",
        "site_growth_k24",
    ]
    assert protocol["budget"]["optimizer_updates"] == 2304
    assert protocol["promotion_seeds"] == [8660001, 8660002, 8660003]
    assert protocol["recovery_seed"] == 8660002

    protocol_v2 = validate_protocol(protocol_v2_raw())
    assert protocol_v2["schema"] == PROTOCOL_V2_SCHEMA
    assert protocol_v2["development_role"] == "fresh-search-only-selection"
    assert (
        protocol_v2["development_pool_binding"]["pool_sha256"]
        == protocol_v2["data_contract"]["development_pool_sha256"]
    )


def test_v2_protocol_rejects_historical_or_mismatched_development_binding():
    historical = protocol_v2_raw()
    historical["data_contract"][
        "development_pool_sha256"
    ] = HISTORICAL_CONFIRMATION_SHA256
    historical["development_pool_binding"][
        "pool_sha256"
    ] = HISTORICAL_CONFIRMATION_SHA256
    with pytest.raises(X21AutoResearchError, match="forbidden historical"):
        validate_protocol(historical)

    mismatch = protocol_v2_raw()
    mismatch["development_pool_binding"]["pool_sha256"] = "6" * 64
    with pytest.raises(X21AutoResearchError, match="differs from its fresh binding"):
        validate_protocol(mismatch)


def test_recovery_store_is_separate_and_cannot_record_metrics(tmp_path):
    recovery_root = tmp_path / "recovery"
    store = RecoveryStore.initialize(
        recovery_root, "x21-k20d14-r2-exact-recovery-20260824"
    )
    evidence = recovery_evidence()
    assert store.record(evidence)["source_epoch"] == 36
    status = store.status()
    assert set(status["certificates"]) == {"8660002"}
    assert not (recovery_root / "leaderboard.json").exists()

    contaminated = {**evidence, "metrics": {"sigma": 0.1}}
    with pytest.raises(X21AutoResearchError):
        validate_recovery_evidence(contaminated)


def test_baseline_requires_recovery_provenance_only_for_registered_seed(tmp_path):
    store = X21CampaignStore.initialize(tmp_path / "science", protocol_raw())
    family = baseline_family()
    family["models"][1]["provenance"]["kind"] = "fresh-completion"
    with pytest.raises(X21AutoResearchError, match="protocol contract"):
        store.register_baseline(family)


def test_baseline_recovery_hash_is_bound_to_certificate_and_checkpoint(tmp_path):
    store = X21CampaignStore.initialize(tmp_path / "science", protocol_raw())
    missing = baseline_family()
    del missing["models"][1]["provenance"]["certificate"]
    with pytest.raises(X21AutoResearchError, match="hash-bound certificate"):
        store.register_baseline(missing)

    wrong_hash = baseline_family()
    wrong_hash["models"][1]["provenance"]["certificate_sha256"] = "0" * 64
    with pytest.raises(X21AutoResearchError, match="not bound"):
        store.register_baseline(wrong_hash)

    wrong_checkpoint = baseline_family()
    wrong_checkpoint["models"][1]["checkpoint_sha256"] = "2" * 64
    with pytest.raises(X21AutoResearchError, match="protocol contract"):
        store.register_baseline(wrong_checkpoint)


def test_baseline_certificates_are_bound_to_registered_jobs(tmp_path):
    store = X21CampaignStore.initialize(tmp_path / "science", protocol_raw())
    unrelated = baseline_family()
    recovery = unrelated["models"][1]["provenance"]["certificate"]
    recovery["campaign_id"] = "unrelated-recovery"
    unrelated["models"][1]["provenance"]["certificate_sha256"] = digest_value(recovery)
    with pytest.raises(X21AutoResearchError, match="protocol contract"):
        store.register_baseline(unrelated)

    wrong_job = baseline_family()
    completion = wrong_job["models"][0]["provenance"]["certificate"]
    completion["source_job_digest"] = "f" * 64
    wrong_job["models"][0]["provenance"]["certificate_sha256"] = digest_value(
        completion
    )
    with pytest.raises(X21AutoResearchError, match="protocol contract"):
        store.register_baseline(wrong_job)


def test_round_one_promotes_only_three_seed_fixed_budget_evidence(tmp_path):
    store, protocol, parent = initialized_campaign(tmp_path)
    action = store.register_action(action_raw(protocol, parent, 1))
    evidence = store.record_evidence(evidence_raw(action, parent, protocol))
    report = store.adjudicate_round(1)
    assert report["promotion_passes"] is True
    assert report["gates"]["matched_fixed_budget"] is True
    status = store.status()
    assert status["promotion_count"] == 1
    assert status["champion_candidate_id"] == action["candidate_id"]
    assert len(status["leaderboard"]) == 1
    assert "recovery" not in json.dumps(status["leaderboard"]).lower()
    assert evidence["precision"] == "complex64"


def test_v1_historical_pool_cannot_record_scientific_evidence(tmp_path):
    raw = protocol_raw()
    protocol = validate_protocol(raw)
    store = X21CampaignStore.initialize(tmp_path / "historical-science", raw)
    parent = store.register_baseline(baseline_family())
    action = store.register_action(action_raw(protocol, parent, 1))
    with pytest.raises(X21AutoResearchError, match="v2 fresh development binding"):
        store.record_evidence(evidence_raw(action, parent, protocol))
    assert store.status()["rounds"]["1"]["evidence_sha256"] is None


def test_scientific_evidence_rejects_operational_recovery_material(tmp_path):
    store, protocol, parent = initialized_campaign(tmp_path)
    action = store.register_action(action_raw(protocol, parent, 1))
    evidence = evidence_raw(action, parent, protocol)
    evidence["recovery_attempt"] = 4
    with pytest.raises(X21AutoResearchError, match="operational recovery"):
        store.record_evidence(evidence)


def test_three_seeds_cannot_reuse_a_candidate_checkpoint(tmp_path):
    store, protocol, parent = initialized_campaign(tmp_path)
    action = store.register_action(action_raw(protocol, parent, 1))
    evidence = evidence_raw(action, parent, protocol)
    shared = evidence["seeds"][0]["candidate"]["checkpoint_sha256"]
    for row in evidence["seeds"]:
        row["candidate"]["checkpoint_sha256"] = shared
    with pytest.raises(X21AutoResearchError, match="distinct across seeds"):
        store.record_evidence(evidence)


def test_wrong_action_order_and_second_mutation_are_rejected(tmp_path):
    store, protocol, parent = initialized_campaign(tmp_path)
    wrong = action_raw(protocol, parent, 2)
    with pytest.raises(X21AutoResearchError, match="next open round"):
        store.register_action(wrong)

    action = action_raw(protocol, parent, 1)
    action["mutation"]["candidate_value"] = 1.0e-5
    with pytest.raises(X21AutoResearchError, match="single-variable"):
        store.register_action(action)


def test_failed_gate_is_recorded_but_does_not_replace_champion(tmp_path):
    store, protocol, parent = initialized_campaign(tmp_path)
    action = store.register_action(action_raw(protocol, parent, 1))
    evidence = evidence_raw(action, parent, protocol, scale=1.01)
    for row in evidence["seeds"]:
        row["paired"]["sigma_ci95_low"] = -1.0e-5
        row["paired"]["chi_ci95_low"] = -1.0e-5
    store.record_evidence(evidence)
    report = store.adjudicate_round(1)
    assert report["promotion_passes"] is False
    status = store.status()
    assert status["champion_family"]["family_sha256"] == parent["family_sha256"]
    assert status["leaderboard"][0]["promotion_passes"] is False


def test_d16_requires_resource_certificate_and_registered_batch(tmp_path):
    store, protocol, parent = initialized_campaign(tmp_path)
    # Complete rounds 1 and 2 as rejected, keeping the baseline structure.
    for round_index in (1, 2):
        action = store.register_action(action_raw(protocol, parent, round_index))
        evidence = evidence_raw(action, parent, protocol, scale=1.01)
        for row in evidence["seeds"]:
            row["paired"]["sigma_ci95_low"] = -1.0e-5
            row["paired"]["chi_ci95_low"] = -1.0e-5
        store.record_evidence(evidence)
        store.adjudicate_round(round_index)

    missing = action_raw(protocol, parent, 3, batch_size=512)
    with pytest.raises(X21AutoResearchError):
        store.register_action(missing)

    invalid = action_raw(protocol, parent, 3, batch_size=500, resource=True)
    with pytest.raises(X21AutoResearchError, match="batch size"):
        store.register_action(invalid)

    valid = store.register_action(
        action_raw(protocol, parent, 3, batch_size=512, resource=True)
    )
    assert valid["budget"]["batch_size"] == 512


def test_resource_hash_must_match_bound_certificate(tmp_path):
    store, protocol, parent = initialized_campaign(tmp_path)
    for round_index in (1, 2):
        action = store.register_action(action_raw(protocol, parent, round_index))
        evidence = evidence_raw(action, parent, protocol, scale=1.01)
        for row in evidence["seeds"]:
            row["paired"]["sigma_ci95_low"] = -1.0e-5
            row["paired"]["chi_ci95_low"] = -1.0e-5
        store.record_evidence(evidence)
        store.adjudicate_round(round_index)
    action = action_raw(protocol, parent, 3, batch_size=512, resource=True)
    action["resource_certificate_sha256"] = "0" * 64
    with pytest.raises(X21AutoResearchError, match="not bound"):
        store.register_action(action)


def test_resource_skip_advances_without_entering_leaderboard(tmp_path):
    store, protocol, parent = initialized_campaign(tmp_path)
    for round_index in (1, 2):
        action = store.register_action(action_raw(protocol, parent, round_index))
        evidence = evidence_raw(action, parent, protocol, scale=1.01)
        for row in evidence["seeds"]:
            row["paired"]["sigma_ci95_low"] = -1.0e-5
            row["paired"]["chi_ci95_low"] = -1.0e-5
        store.record_evidence(evidence)
        store.adjudicate_round(round_index)
    before = len(store.status()["leaderboard"])
    certificate = resource_certificate_raw(
        protocol,
        parent,
        3,
        outcome="resource-infeasible",
        selected_batch_size=None,
    )
    skip = store.skip_resource(
        {
            "schema": RESOURCE_SKIP_SCHEMA,
            "round": 3,
            "kind": "bond_growth_d16",
            "certificate_sha256": digest_value(certificate),
            "certificate": certificate,
            "reason": "resource-infeasible",
        }
    )
    status = store.status()
    assert skip["round"] == 3
    assert status["current_round"] == 3
    assert len(status["leaderboard"]) == before
    # k24 now opens from the unchanged D14 champion without a D16 certificate.
    action = store.register_action(action_raw(protocol, parent, 4))
    assert action["candidate_metadata"]["k"] == 24
    assert action["candidate_metadata"]["bond_dimension"] == 14


def test_status_fails_closed_after_ledger_tampering(tmp_path):
    store, _, _ = initialized_campaign(tmp_path)
    ledger_path = tmp_path / "science" / "ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["promotion_count"] = 99
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    with pytest.raises(X21AutoResearchError, match="integrity"):
        store.status()
