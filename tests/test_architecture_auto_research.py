from __future__ import annotations

import hashlib
import json

import pytest

from gcicy_metric.pipeline.architecture_auto_research import (
    ACTION_SCHEMA,
    AutoResearchError,
    CampaignStore,
    PROTOCOL_SCHEMA,
    REPLAY_SCHEMA,
    SEARCH_EVIDENCE_SCHEMA,
    SHADOW_CLAIM_SCHEMA,
    SHADOW_EVIDENCE_SCHEMA,
    fixed_search_indices,
    validate_action,
    validate_protocol,
)


SEEDS = [31001, 31002, 31003]


def token(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def budget(*, updates: int) -> dict[str, object]:
    return {
        "optimizer_updates": updates,
        "batch_size": 128,
        "learning_rate": 3.0e-6,
        "gradient_clip_norm": 1.0,
        "train_examples": 300,
        "evaluation_examples": 90,
        "eval_every": 10,
        "scheduler": "cosine",
    }


def raw_protocol(tmp_path, *, shared_population: bool = False):
    train = tmp_path / "search_train.bin"
    evaluation = tmp_path / "search_evaluation.bin"
    train.write_bytes(b"fixed-search-train")
    evaluation.write_bytes(b"fixed-search-evaluation")
    return {
        "schema": PROTOCOL_SCHEMA,
        "campaign_id": "quintic-architecture-auto-v1-test",
        "baseline_family": {
            "models": [
                {"seed": seed, "checkpoint_sha256": token(f"baseline-{seed}")}
                for seed in SEEDS
            ]
        },
        "promotion_seeds": SEEDS,
        "search_inputs": [
            {
                "role": "search-train",
                "path": str(train),
                "sha256": token("fixed-search-train"),
            },
            {
                "role": "search-evaluation",
                "path": str(evaluation),
                "sha256": token("fixed-search-evaluation"),
            },
        ],
        "search_index_plan": {
            "seed": 2026082301,
            "train_population": 1000,
            "train_count": 300,
            "evaluation_population": 1000 if shared_population else 400,
            "evaluation_count": 90,
            "shared_population": shared_population,
        },
        "budgets": {
            "local_activate": budget(updates=40),
            "matched_relax": budget(updates=80),
        },
        "thresholds": {
            "minimum_median_relative_sigma_gain": 0.002,
            "minimum_median_relative_chi_gain": 0.002,
            "maximum_search_tail_relative_degradation": 0.005,
        },
        "maximum_rounds": 4,
    }


def rank_action(store: CampaignStore, candidate_id: str, *, gain_rank: int = 35):
    ledger = store.ledger()
    return {
        "schema": ACTION_SCHEMA,
        "candidate_id": candidate_id,
        "round": ledger["current_round"] + 1,
        "parent_family_sha256": ledger["champion_family"]["family_sha256"],
        "search_indices_sha256": ledger["search_indices_sha256"],
        "mutation": {
            "kind": "rank",
            "target": "internal-edge",
            "edge": 6,
            "source_dimension": 25,
            "target_dimension": gain_rank,
            "structural_maximum": 125,
            "new_output_real_parameters": 5000,
        },
        "pipeline": [
            {"kind": "local-activate"},
            {"kind": "matched-relax"},
        ],
    }


def metrics(*, sigma: float, chi: float, tail_scale: float = 1.0):
    return {
        "sigma": sigma,
        "chi": chi,
        "q999": 0.08 * tail_scale,
        "cvar_1pct": 0.06 * tail_scale,
        "minimum_metric_eigenvalue": 0.02,
        "nonpositive_metric_count": 0,
    }


def evidence(
    action,
    protocol,
    *,
    relative_gain: float,
    control_prefix: str = "shared-control",
    budget_mismatch: bool = False,
):
    rows = []
    for seed in SEEDS:
        control_budget = dict(protocol["budgets"]["matched_relax"])
        candidate_budget = dict(control_budget)
        if budget_mismatch:
            candidate_budget["optimizer_updates"] += 1
        rows.append(
            {
                "seed": seed,
                "equivalence": {
                    "passed": True,
                    "potential_max_absolute": 1.0e-7,
                    "metric_max_relative_frobenius": 1.0e-6,
                },
                "local_activate_budget": protocol["budgets"]["local_activate"],
                "control_matched_relax_budget": control_budget,
                "candidate_matched_relax_budget": candidate_budget,
                "control_matched_relax_batch_plan_sha256": token(f"batch-plan-{seed}"),
                "candidate_matched_relax_batch_plan_sha256": token(
                    f"batch-plan-{seed}"
                ),
                "control": metrics(sigma=0.02, chi=0.03),
                "candidate": metrics(
                    sigma=0.02 * (1.0 - relative_gain),
                    chi=0.03 * (1.0 - relative_gain),
                    tail_scale=1.0 - relative_gain,
                ),
                "paired": {"sigma_ci95_low": 1.0e-5, "e2_ci95_low": 1.0e-6},
                "control_checkpoint_sha256": token(f"{control_prefix}-{seed}"),
                "candidate_checkpoint_sha256": token(
                    f"{action['candidate_id']}-{seed}"
                ),
                "control_trainable_real_parameter_count": 100_000,
                "candidate_trainable_real_parameter_count": (
                    100_000
                    + int(action["mutation"].get("new_output_real_parameters", 0))
                ),
            }
        )
    return {
        "schema": SEARCH_EVIDENCE_SCHEMA,
        "candidate_id": action["candidate_id"],
        "round": action["round"],
        "action_sha256": action["action_sha256"],
        "parent_family_sha256": action["parent_family_sha256"],
        "search_indices_sha256": action["search_indices_sha256"],
        "precision": "complex64",
        "seeds": rows,
    }


def initialize(tmp_path, *, shared_population: bool = False):
    run_root = tmp_path / "run"
    store = CampaignStore.initialize(
        run_root,
        raw_protocol(tmp_path, shared_population=shared_population),
    )
    return store, store.protocol()


def promote_one(store: CampaignStore, protocol):
    action = store.register_action(rank_action(store, "rank-edge-6"))
    report = store.adjudicate_round(
        1,
        [evidence(action, protocol, relative_gain=0.01)],
    )
    assert report["winner_candidate_id"] == "rank-edge-6"
    return report


def test_fixed_indices_are_deterministic_and_disjoint_for_shared_pool(tmp_path):
    protocol = validate_protocol(raw_protocol(tmp_path, shared_population=True))
    first = fixed_search_indices(protocol)
    second = fixed_search_indices(protocol)
    assert first == second
    assert first["indices_sha256"] == second["indices_sha256"]
    assert set(first["train_indices"]).isdisjoint(first["evaluation_indices"])
    assert len(first["train_indices"]) == 300
    assert len(first["evaluation_indices"]) == 90


def test_protocol_rejects_any_search_input_named_as_held_out_final_data(tmp_path):
    protocol = raw_protocol(tmp_path)
    forbidden = tmp_path / "blind_points.bin"
    forbidden.write_bytes(b"forbidden")
    protocol["search_inputs"][0] = {
        "role": "search-train",
        "path": str(forbidden),
        "sha256": token("forbidden"),
    }
    with pytest.raises(AutoResearchError, match="forbidden held-out"):
        validate_protocol(protocol)


def test_protocol_rejects_shadow_material_during_search(tmp_path):
    protocol = raw_protocol(tmp_path)
    protocol["search_inputs"][0]["role"] = "shadow-selection"
    with pytest.raises(AutoResearchError, match="forbidden held-out"):
        validate_protocol(protocol)


def test_action_language_has_no_command_escape_hatch(tmp_path):
    store, _ = initialize(tmp_path)
    action = rank_action(store, "rank-edge-6")
    action["command"] = ["python", "unregistered.py"]
    with pytest.raises(AutoResearchError, match="extra=.*command"):
        validate_action(action)


def test_action_restricts_transport_rank_and_fixed_pipeline(tmp_path):
    store, _ = initialize(tmp_path)
    oversized = rank_action(store, "too-large")
    oversized["mutation"]["new_output_real_parameters"] = 10001
    with pytest.raises(AutoResearchError, match="parameter cap"):
        validate_action(oversized)

    wrong_pipeline = rank_action(store, "skip-control")
    wrong_pipeline["pipeline"] = [{"kind": "local-activate"}]
    with pytest.raises(AutoResearchError, match="matched-relax"):
        validate_action(wrong_pipeline)

    topology = rank_action(store, "bad-topology")
    topology["mutation"] = {"kind": "topology", "transport": "agent-invented"}
    with pytest.raises(AutoResearchError, match="not exact/registered"):
        validate_action(topology)


@pytest.mark.parametrize(
    "mutation",
    [
        {"kind": "topology", "transport": "five-leaf-4plus1-to-2plus3"},
        {
            "kind": "rank",
            "target": "shared-leaf",
            "source_dimension": 5,
            "target_dimension": 6,
            "structural_maximum": 10,
            "new_output_real_parameters": 9800,
        },
        {"kind": "root-residual", "topology": "five-leaf-2plus3", "rank": 8},
    ],
)
def test_every_registered_structural_mutation_roundtrips(tmp_path, mutation):
    store, _ = initialize(tmp_path)
    action = rank_action(store, "registered-mutation")
    action["mutation"] = mutation
    normalized = validate_action(action)
    assert normalized["mutation"] == mutation
    assert len(normalized["action_sha256"]) == 64


def test_local_activation_is_a_mandatory_stage_not_an_agent_mutation(tmp_path):
    store, _ = initialize(tmp_path)
    action = rank_action(store, "not-a-mutation")
    action["mutation"] = {"kind": "local-activate"}
    with pytest.raises(AutoResearchError, match="not registered"):
        validate_action(action)


def test_round_adjudication_uses_three_seeds_and_shared_control(tmp_path):
    store, protocol = initialize(tmp_path)
    slower = store.register_action(rank_action(store, "rank-35", gain_rank=35))
    faster = store.register_action(rank_action(store, "rank-45", gain_rank=45))
    report = store.adjudicate_round(
        1,
        [
            evidence(slower, protocol, relative_gain=0.006),
            evidence(faster, protocol, relative_gain=0.012),
        ],
    )
    assert report["common_no_growth_control"] is True
    assert report["winner_candidate_id"] == "rank-45"
    assert all(row["promotion_passes"] for row in report["candidates"])
    assert store.ledger()["champion_candidate_id"] == "rank-45"


def test_budget_mismatch_is_a_scientific_rejection(tmp_path):
    store, protocol = initialize(tmp_path)
    action = store.register_action(rank_action(store, "mismatched-budget"))
    report = store.adjudicate_round(
        1,
        [
            evidence(
                action,
                protocol,
                relative_gain=0.02,
                budget_mismatch=True,
            )
        ],
    )
    candidate = report["candidates"][0]
    assert candidate["gates"]["matched_budgets"] is False
    assert candidate["promotion_passes"] is False
    assert report["winner_candidate_id"] is None


def test_candidate_evidence_can_be_recorded_and_resumed_before_round_gate(tmp_path):
    store, protocol = initialize(tmp_path)
    action = store.register_action(rank_action(store, "resumable-candidate"))
    row = evidence(action, protocol, relative_gain=0.01)
    recorded = store.record_search_evidence(row)
    assert recorded["candidate_id"] == "resumable-candidate"
    assert store.record_search_evidence(row) == recorded

    reopened = CampaignStore(store.run_root)
    report = reopened.adjudicate_round(1)
    assert report["winner_candidate_id"] == "resumable-candidate"
    assert reopened.adjudicate_round(1) == report

    changed = evidence(action, protocol, relative_gain=0.02)
    with pytest.raises(AutoResearchError, match="conflicts"):
        reopened.record_search_evidence(changed)


def test_recorded_candidate_evidence_cannot_change_before_adjudication(tmp_path):
    store, protocol = initialize(tmp_path)
    action = store.register_action(rank_action(store, "tampered-candidate"))
    row = evidence(action, protocol, relative_gain=0.01)
    store.record_search_evidence(row)
    evidence_path = (
        store.run_root
        / "rounds"
        / "round_001"
        / "candidates"
        / "tampered-candidate"
        / "evidence.json"
    )
    changed = json.loads(evidence_path.read_text(encoding="utf-8"))
    changed["seeds"][0]["candidate"]["sigma"] *= 0.5
    evidence_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(AutoResearchError, match="durable ledger"):
        store.adjudicate_round(1)


def test_different_no_growth_controls_block_cross_candidate_promotion(tmp_path):
    store, protocol = initialize(tmp_path)
    first = store.register_action(rank_action(store, "candidate-a", gain_rank=35))
    second = store.register_action(rank_action(store, "candidate-b", gain_rank=45))
    report = store.adjudicate_round(
        1,
        [
            evidence(first, protocol, relative_gain=0.01, control_prefix="control-a"),
            evidence(second, protocol, relative_gain=0.02, control_prefix="control-b"),
        ],
    )
    assert report["common_no_growth_control"] is False
    assert report["winner_candidate_id"] is None
    assert not any(row["promotion_passes"] for row in report["candidates"])


def test_same_control_checkpoint_with_different_metrics_is_not_shared(tmp_path):
    store, protocol = initialize(tmp_path)
    first = store.register_action(rank_action(store, "candidate-a", gain_rank=35))
    second = store.register_action(rank_action(store, "candidate-b", gain_rank=45))
    first_evidence = evidence(first, protocol, relative_gain=0.01)
    second_evidence = evidence(second, protocol, relative_gain=0.02)
    second_evidence["seeds"][0]["control"]["sigma"] += 1.0e-6
    report = store.adjudicate_round(1, [first_evidence, second_evidence])
    assert report["common_no_growth_control"] is False
    assert report["winner_candidate_id"] is None


def test_one_seed_regression_blocks_promotion_even_with_positive_median(tmp_path):
    store, protocol = initialize(tmp_path)
    action = store.register_action(rank_action(store, "one-seed-regression"))
    row = evidence(action, protocol, relative_gain=0.02)
    row["seeds"][0]["candidate"]["sigma"] = 0.020001
    report = store.adjudicate_round(1, [row])
    candidate = report["candidates"][0]
    assert candidate["median_relative_sigma_gain"] > 0
    assert candidate["gates"]["worst_seed_sigma"] is False
    assert report["winner_candidate_id"] is None


def test_control_and_candidate_must_share_the_exact_batch_plan(tmp_path):
    store, protocol = initialize(tmp_path)
    action = store.register_action(rank_action(store, "different-batches"))
    row = evidence(action, protocol, relative_gain=0.02)
    row["seeds"][1]["candidate_matched_relax_batch_plan_sha256"] = token(
        "different-batch-plan"
    )
    report = store.adjudicate_round(1, [row])
    candidate = report["candidates"][0]
    assert candidate["gates"]["matched_batch_plans"] is False
    assert report["winner_candidate_id"] is None


def test_reported_parameter_delta_must_match_the_registered_rank_growth(tmp_path):
    store, protocol = initialize(tmp_path)
    action = store.register_action(rank_action(store, "wrong-parameter-delta"))
    row = evidence(action, protocol, relative_gain=0.02)
    row["seeds"][1]["candidate_trainable_real_parameter_count"] += 1
    report = store.adjudicate_round(1, [row])
    candidate = report["candidates"][0]
    assert candidate["gates"]["parameter_accounting"] is False
    assert report["winner_candidate_id"] is None


def test_structural_candidate_cannot_report_a_negative_parameter_delta(tmp_path):
    store, protocol = initialize(tmp_path)
    raw = rank_action(store, "negative-parameter-delta")
    raw["mutation"] = {
        "kind": "topology",
        "transport": "five-leaf-4plus1-to-2plus3",
    }
    action = store.register_action(raw)
    row = evidence(action, protocol, relative_gain=0.02)
    for seed_row in row["seeds"]:
        seed_row["candidate_trainable_real_parameter_count"] = 99_999
    report = store.adjudicate_round(1, [row])
    assert report["candidates"][0]["gates"]["parameter_accounting"] is False
    assert report["winner_candidate_id"] is None


def test_search_evidence_cannot_use_complex128(tmp_path):
    store, protocol = initialize(tmp_path)
    action = store.register_action(rank_action(store, "wrong-precision"))
    row = evidence(action, protocol, relative_gain=0.01)
    row["precision"] = "complex128"
    with pytest.raises(AutoResearchError, match="precision"):
        store.adjudicate_round(1, [row])


def replay_evidence(frozen):
    return {
        "schema": REPLAY_SCHEMA,
        "frozen_family_sha256": frozen["champion_family"]["family_sha256"],
        "precision": "complex128",
        "optimizer_updates": 0,
        "replays": [
            {
                "seed": row["seed"],
                "source_checkpoint_sha256": row["checkpoint_sha256"],
                "complex128_checkpoint_sha256": token(f"complex128-{row['seed']}"),
                "relative_sigma_delta": 1.0e-5,
                "metrics": metrics(sigma=0.0198, chi=0.0297, tail_scale=0.99),
            }
            for row in frozen["champion_family"]["models"]
        ],
    }


def shadow_evidence(frozen, *, dataset_label="shadow-v1"):
    champion = {
        row["seed"]: row["checkpoint_sha256"]
        for row in frozen["champion_family"]["models"]
    }
    comparator = {
        row["seed"]: row["checkpoint_sha256"]
        for row in frozen["comparator_family"]["models"]
    }
    return {
        "schema": SHADOW_EVIDENCE_SCHEMA,
        "frozen_family_sha256": frozen["champion_family"]["family_sha256"],
        "shadow_dataset_sha256": token(dataset_label),
        "shadow_indices_sha256": token(f"{dataset_label}-indices"),
        "evaluator_sha256": token("registered-shadow-evaluator-v1"),
        "precision": "complex64",
        "comparisons": [
            {
                "seed": seed,
                "control_checkpoint_sha256": comparator[seed],
                "candidate_checkpoint_sha256": champion[seed],
                "control": metrics(sigma=0.0202, chi=0.0302),
                "candidate": metrics(sigma=0.0198, chi=0.0297, tail_scale=0.98),
                "paired": {"sigma_ci95_low": 1.0e-5, "e2_ci95_low": 1.0e-6},
            }
            for seed in SEEDS
        ],
    }


def shadow_claim(frozen, *, dataset_label="shadow-v1"):
    return {
        "schema": SHADOW_CLAIM_SCHEMA,
        "frozen_family_sha256": frozen["champion_family"]["family_sha256"],
        "shadow_dataset_sha256": token(dataset_label),
        "shadow_indices_sha256": token(f"{dataset_label}-indices"),
        "evaluator_sha256": token("registered-shadow-evaluator-v1"),
    }


def test_freeze_replay_and_shadow_form_a_single_use_state_machine(tmp_path):
    store, protocol = initialize(tmp_path)
    promote_one(store, protocol)
    with pytest.raises(AutoResearchError, match="freeze the finalist"):
        store.record_complex128_replay(
            {
                "schema": REPLAY_SCHEMA,
                "frozen_family_sha256": token("not-frozen"),
                "precision": "complex128",
                "optimizer_updates": 0,
                "replays": [],
            }
        )

    frozen = store.freeze()
    with pytest.raises(AutoResearchError, match="claim the single shadow"):
        store.record_shadow(shadow_evidence(frozen))

    replay = replay_evidence(frozen)
    assert store.record_complex128_replay(replay)["optimizer_updates"] == 0
    assert store.record_complex128_replay(replay)["precision"] == "complex128"

    claim = shadow_claim(frozen)
    assert store.claim_shadow(claim) == claim
    assert store.claim_shadow(claim) == claim
    changed_claim = shadow_claim(frozen, dataset_label="shadow-v2")
    with pytest.raises(AutoResearchError, match="claim is already consumed"):
        store.claim_shadow(changed_claim)
    shadow = shadow_evidence(frozen)
    report = store.record_shadow(shadow)
    assert report["shadow_passes"] is True
    assert store.ledger()["status"] == "complete"
    assert store.record_shadow(shadow) == report
    assert store.freeze() == frozen
    assert store.record_complex128_replay(replay) == replay
    assert store.claim_shadow(claim) == claim

    changed = shadow_evidence(frozen, dataset_label="shadow-v2")
    with pytest.raises(AutoResearchError, match="already consumed"):
        store.record_shadow(changed)


def test_freeze_recovers_the_create_only_artifact_after_commit_interruption(
    tmp_path, monkeypatch
):
    store, protocol = initialize(tmp_path)
    promote_one(store, protocol)

    def interrupt_commit(*args, **kwargs):
        raise RuntimeError("simulated interruption after artifact publication")

    monkeypatch.setattr(store, "_commit", interrupt_commit)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        store.freeze()
    orphan = (store.run_root / "frozen_candidate.json").read_bytes()

    resumed = CampaignStore(store.run_root)
    frozen = resumed.freeze()
    assert (store.run_root / "frozen_candidate.json").read_bytes() == orphan
    assert resumed.ledger()["status"] == "frozen"
    assert frozen["freeze_sha256"]


def test_shadow_evidence_must_match_the_claimed_evaluator(tmp_path):
    store, protocol = initialize(tmp_path)
    promote_one(store, protocol)
    frozen = store.freeze()
    store.record_complex128_replay(replay_evidence(frozen))
    store.claim_shadow(shadow_claim(frozen))
    changed = shadow_evidence(frozen)
    changed["evaluator_sha256"] = token("unregistered-evaluator")
    with pytest.raises(AutoResearchError, match="durable pre-evaluation claim"):
        store.record_shadow(changed)


def test_shadow_payload_cannot_be_relabelled_as_final_held_out_data(tmp_path):
    store, protocol = initialize(tmp_path)
    promote_one(store, protocol)
    frozen = store.freeze()
    store.record_complex128_replay(replay_evidence(frozen))
    store.claim_shadow(shadow_claim(frozen))
    payload = shadow_evidence(frozen)
    payload["note"] = "blind-final"
    with pytest.raises(AutoResearchError, match="forbidden held-out"):
        store.record_shadow(payload)


def test_initialization_is_idempotent_but_protocol_drift_is_rejected(tmp_path):
    protocol = raw_protocol(tmp_path)
    store = CampaignStore.initialize(tmp_path / "run", protocol)
    same = CampaignStore.initialize(tmp_path / "run", protocol)
    assert same.ledger() == store.ledger()

    changed = raw_protocol(tmp_path)
    changed["maximum_rounds"] = 3
    with pytest.raises(AutoResearchError, match="different protocol"):
        CampaignStore.initialize(tmp_path / "run", changed)


def test_fixed_index_artifact_tampering_blocks_state_transitions(tmp_path):
    store, _ = initialize(tmp_path)
    indices = json.loads(store.indices_path.read_text(encoding="utf-8"))
    indices["train_indices"][0] += 1
    store.indices_path.write_text(json.dumps(indices), encoding="utf-8")
    with pytest.raises(AutoResearchError, match="fixed search indices"):
        store.register_action(rank_action(store, "must-not-register"))


def test_completed_round_adjudication_artifact_is_hash_verified(tmp_path):
    store, protocol = initialize(tmp_path)
    action = store.register_action(rank_action(store, "winner"))
    store.adjudicate_round(1, [evidence(action, protocol, relative_gain=0.01)])
    report_path = store.run_root / "rounds" / "round_001" / "adjudication.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["winner_candidate_id"] = None
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(AutoResearchError, match="durable ledger"):
        store.adjudicate_round(1)


def test_tampered_ledger_fails_closed(tmp_path):
    store, _ = initialize(tmp_path)
    ledger = json.loads(store.ledger_path.read_text(encoding="utf-8"))
    ledger["champion_candidate_id"] = "tampered"
    store.ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    with pytest.raises(AutoResearchError, match="state hash"):
        store.ledger()
