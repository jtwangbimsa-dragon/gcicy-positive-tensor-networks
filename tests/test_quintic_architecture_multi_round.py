from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from gcicy_metric.pipeline.architecture_auto_research import (
    ACTION_SCHEMA,
    AutoResearchError,
    CampaignStore,
    PROTOCOL_SCHEMA,
    digest_value,
)
from gcicy_metric.pipeline import architecture_auto_research as control
from gcicy_metric.pipeline import quintic_architecture_multi_round as multi


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = (
    ROOT
    / "experiments"
    / "protocols"
    / "generic_quintic_architecture_multi_round_v2.json"
)
SEEDS = [202608231, 202608232, 202608233]


def token(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def raw_catalog() -> dict:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def family(prefix: str) -> dict:
    payload = {
        "models": [
            {"seed": seed, "checkpoint_sha256": token(f"{prefix}-{seed}")}
            for seed in SEEDS
        ]
    }
    return {**payload, "family_sha256": digest_value(payload)}


def decided_observation(tmp_path: Path, outcome: str) -> dict:
    data_contract = multi.validate_catalog(raw_catalog())["data_contract"]
    bridge_root = tmp_path / "bridge"
    campaign_root = tmp_path / "controller"
    bridge_root.mkdir(exist_ok=True)
    campaign_root.mkdir(exist_ok=True)
    artifacts = []
    models = []
    for seed in SEEDS:
        checkpoint = tmp_path / f"champion-{seed}.pt"
        checkpoint.write_bytes(f"champion-{outcome}-{seed}".encode())
        checkpoint_sha256 = multi.sha256_file(checkpoint)
        report = tmp_path / f"report-{seed}.json"
        report.write_text(json.dumps({"seed": seed, "outcome": outcome}))
        models.append({"seed": seed, "checkpoint_sha256": checkpoint_sha256})
        artifacts.append(
            {
                "seed": seed,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha256,
                "role": (
                    "candidate-polished"
                    if outcome == "promoted"
                    else "baseline-retained"
                ),
                "source_report_path": str(report),
                "source_report_sha256": multi.sha256_file(report),
            }
        )
    family_payload = {"models": models}
    champion = {**family_payload, "family_sha256": digest_value(family_payload)}
    plan = {"plan_sha256": token("plan-value")}
    plan_path = bridge_root / "plan.json"
    plan_path.write_text(json.dumps(plan))
    ledger_path = bridge_root / "ledger.json"
    ledger_path.write_text(json.dumps({"state": "normalized"}))
    evidence = {"schema": "test-evidence", "outcome": outcome}
    evidence_path = bridge_root / "search_evidence.json"
    evidence_path.write_text(json.dumps(evidence))
    adjudication_path = tmp_path / "adjudication.json"
    adjudication_path.write_text(json.dumps({"outcome": outcome}))
    return {
        "state": "decided",
        "phase": "complete",
        "outcome": outcome,
        "champion_family": champion,
        "champion_artifacts": artifacts,
        "provenance": {
            "campaign_root": str(campaign_root),
            "bridge_root": str(bridge_root),
            "protocol_sha256": data_contract["protocol_sha256"],
            "search_indices_sha256": data_contract["search_indices_sha256"],
            "bridge_plan_sha256": token("plan-value"),
            "bridge_plan_file_sha256": multi.sha256_file(plan_path),
            "bridge_ledger_file_sha256": multi.sha256_file(ledger_path),
            "search_evidence_sha256": multi.sha256_file(evidence_path),
            "search_evidence_value_sha256": digest_value(evidence),
            "adjudication_path": str(adjudication_path),
            "adjudication_sha256": multi.sha256_file(adjudication_path),
            "input_sha256": data_contract["input_sha256"],
            "matched_batch_plans": data_contract["matched_batch_plans"],
        },
    }


def manager(tmp_path: Path) -> multi.MultiRoundManager:
    return multi.MultiRoundManager.initialize(
        tmp_path / "manager",
        raw_catalog(),
        round1_campaign_root=tmp_path / "controller",
        round1_bridge_root=tmp_path / "bridge",
    )


def _budget(updates: int) -> dict:
    return {
        "optimizer_updates": updates,
        "batch_size": 1024,
        "learning_rate": 3.0e-6,
        "gradient_clip_norm": 1.0,
        "train_examples": 30_000,
        "evaluation_examples": 5_000,
        "eval_every": 25,
        "scheduler": "cosine",
    }


def open_round1_controller(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    catalog = multi.validate_catalog(raw_catalog())
    expected_baseline = catalog["round1_contract"]["baseline_checkpoint_sha256"]
    files = {}
    for role in (
        "baseline-checkpoint",
        "baseline-source-report",
        "search-native-points",
        "search-native-pullbacks",
        "development-evaluation-points",
        "development-evaluation-pullbacks",
    ):
        path = tmp_path / f"{role}.bin"
        path.write_bytes(role.encode("utf-8"))
        files[role] = path
    original = control.sha256_file

    def hashes(path: Path) -> str:
        resolved = Path(path).resolve()
        if resolved == files["baseline-checkpoint"].resolve():
            return expected_baseline
        return original(resolved)

    monkeypatch.setattr(control, "sha256_file", hashes)
    protocol = {
        "schema": PROTOCOL_SCHEMA,
        "campaign_id": catalog["round1_contract"]["campaign_id"],
        "baseline_family": {
            "models": [
                {"seed": seed, "checkpoint_sha256": expected_baseline} for seed in SEEDS
            ]
        },
        "promotion_seeds": SEEDS,
        "search_inputs": [
            {"role": role, "path": str(path), "sha256": hashes(path)}
            for role, path in files.items()
        ],
        "search_index_plan": {
            "seed": 2026082301,
            "partition_seed": 2026082302,
            "train_population": 100_000,
            "train_count": 35_000,
            "evaluation_population": 20_000,
            "evaluation_count": 5_000,
            "shared_population": False,
        },
        "budgets": {
            "local_activate": {
                **_budget(1200),
                "learning_rate": 3.0e-4,
                "eval_every": 10,
            },
            "matched_relax": _budget(600),
        },
        "maximum_rounds": 4,
    }
    campaign_root = tmp_path / "controller"
    store = CampaignStore.initialize(campaign_root, protocol)
    ledger = store.status()
    expected = catalog["round1_contract"]
    store.register_action(
        {
            "schema": ACTION_SCHEMA,
            "candidate_id": expected["candidate_id"],
            "round": 1,
            "parent_family_sha256": ledger["champion_family"]["family_sha256"],
            "search_indices_sha256": ledger["search_indices_sha256"],
            "mutation": expected["mutation"],
            "pipeline": expected["pipeline"],
        }
    )
    bridge_root = tmp_path / "bridge"
    bridge_root.mkdir()
    (bridge_root / multi.BRIDGE_SENTINEL).write_text(
        f"{expected['candidate_id']}\n", encoding="utf-8"
    )
    plan_payload = {
        "schema": multi.BRIDGE_PLAN_SCHEMA,
        "campaign_run_root": str(campaign_root.resolve()),
        "candidate_id": expected["candidate_id"],
    }
    plan = {**plan_payload, "plan_sha256": digest_value(plan_payload)}
    (bridge_root / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    seed_rows = {
        str(seed): {"local_activate": "pending", "matched_relax": "pending"}
        for seed in SEEDS
    }
    seed_rows[str(SEEDS[0])]["local_activate"] = "failed"
    bridge_ledger_payload = {
        "schema": multi.BRIDGE_LEDGER_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "state": "running",
        "seeds": seed_rows,
        "evidence": None,
    }
    bridge_ledger = {
        **bridge_ledger_payload,
        "state_sha256": digest_value(bridge_ledger_payload),
    }
    (bridge_root / "ledger.json").write_text(
        json.dumps(bridge_ledger), encoding="utf-8"
    )
    return campaign_root, bridge_root


def test_checked_in_catalog_is_closed_and_exact() -> None:
    catalog = multi.validate_catalog(raw_catalog())
    assert catalog["maximum_rounds"] == 4
    assert catalog["data_contract"]["historical_confirmation"] == "absent"
    assert catalog["data_contract"]["final_blind"] == "absent"
    assert catalog["branches"]["r1_promoted"][0] == ("r2-edge6-rank53-progressive")
    assert catalog["branches"]["r1_scientific_rejected"][0] == (
        "r2-edge6-rank39-scale025"
    )
    status_by_round = {
        row["round"]: row["adapter"]["status"] for row in catalog["recipes"]
    }
    assert status_by_round[2] == "blocked-by-adapter"
    assert status_by_round[3] == "available"
    assert status_by_round[4] == "available"


def test_catalog_rejects_commands_and_false_adapter_availability() -> None:
    command = raw_catalog()
    command["recipes"][0]["command"] = ["python", "unreviewed.py"]
    with pytest.raises(AutoResearchError, match="escape hatches"):
        multi.validate_catalog(command)

    available = raw_catalog()
    available["recipes"][0]["adapter"]["status"] = "available"
    with pytest.raises(AutoResearchError, match="cannot claim an executable adapter"):
        multi.validate_catalog(available)

    data_drift = raw_catalog()
    data_drift["data_contract"]["search_indices_sha256"] = token("other-indices")
    with pytest.raises(AutoResearchError, match="data/index/batch hashes"):
        multi.validate_catalog(data_drift)


def test_catalog_scope_and_optimizer_recipes_change_exactly_one_axis() -> None:
    scope = raw_catalog()
    scope_recipe = next(
        row for row in scope["recipes"] if row["category"] == "parameter-scope"
    )
    scope_recipe["intervention"]["candidate"]["scheduler"] = "constant"
    with pytest.raises(AutoResearchError, match="must change exactly"):
        multi.validate_catalog(scope)

    optimizer = raw_catalog()
    optimizer_recipe = next(
        row for row in optimizer["recipes"] if row["category"] == "optimizer-path"
    )
    optimizer_recipe["intervention"]["candidate"]["learning_rate"] = 1.0e-6
    with pytest.raises(AutoResearchError, match="must change exactly"):
        multi.validate_catalog(optimizer)


def test_manager_initialization_is_idempotent_and_waits_for_round1(tmp_path) -> None:
    first = manager(tmp_path)
    second = manager(tmp_path)
    assert first.state() == second.state()
    assert first.state()["status"] == "waiting-r1"
    assert first.state()["round1"]["outcome"] == "pending"
    assert multi.inspect_round1(
        campaign_root=tmp_path / "missing-controller",
        bridge_root=tmp_path / "missing-bridge",
        catalog=first.catalog(),
    ) == {"state": "waiting", "phase": "controller-not-initialized"}


def test_manager_initialization_recovers_after_catalog_publish(tmp_path) -> None:
    run_root = tmp_path / "manager"
    run_root.mkdir()
    catalog = multi.validate_catalog(raw_catalog())
    (run_root / multi.ROOT_SENTINEL).write_text(
        f"{catalog['catalog_id']}\n", encoding="utf-8"
    )
    (run_root / "catalog.lock.json").write_text(json.dumps(catalog), encoding="utf-8")
    store = multi.MultiRoundManager.initialize(
        run_root,
        raw_catalog(),
        round1_campaign_root=tmp_path / "controller",
        round1_bridge_root=tmp_path / "bridge",
    )
    assert store.state()["status"] == "waiting-r1"


def test_round1_promotion_selects_progressive_queue_and_blocks_honestly(
    tmp_path, monkeypatch
) -> None:
    store = manager(tmp_path)
    observation = decided_observation(tmp_path, "promoted")
    monkeypatch.setattr(multi, "inspect_round1", lambda **_: observation)
    result = store.sync_round1()
    assert result["round1"]["outcome"] == "promoted"
    assert result["next_recipe_id"] == "r2-edge6-rank53-progressive"
    assert result["status"] == "blocked-by-adapter"
    assert result["blocker"] == {
        "kind": "blocked-by-adapter",
        "recipe_id": "r2-edge6-rank53-progressive",
        "required_adapter": "quintic-progressive-rank-growth-v2",
    }
    proposal = json.loads(
        (tmp_path / "manager" / "rounds" / "round_002" / "proposal.json").read_text()
    )
    assert proposal["parent_family"] == observation["champion_family"]
    assert proposal["intervention"]["source_dimension"] == 39
    assert proposal["intervention"]["target_dimension"] == 53
    expected_data_binding = {
        key: store.catalog()["data_contract"][key]
        for key in (
            "protocol_sha256",
            "search_indices_sha256",
            "input_sha256",
            "matched_batch_plans",
        )
    }
    assert proposal["data_binding"] == expected_data_binding
    assert proposal["execution_authorized"] is False
    revision = result["revision"]
    assert store.sync_round1()["revision"] == revision
    with pytest.raises(AutoResearchError, match="blocked by the missing adapter"):
        store.run_next()


def test_round1_scientific_rejection_is_not_a_technical_failure(
    tmp_path, monkeypatch
) -> None:
    store = manager(tmp_path)
    observation = decided_observation(tmp_path, "scientific-rejected")
    monkeypatch.setattr(multi, "inspect_round1", lambda **_: observation)
    result = store.sync_round1()
    assert result["round1"]["outcome"] == "scientific-rejected"
    assert result["status"] == "blocked-by-adapter"
    assert result["next_recipe_id"] == "r2-edge6-rank39-scale025"
    assert result["blocker"]["kind"] == "blocked-by-adapter"
    queue = json.loads((tmp_path / "manager" / "queue.json").read_text())
    assert queue["round1_outcome"] == "scientific-rejected"
    assert queue["resolved_next_proposal"]["intervention"]["activation_scale"] == 0.25


def test_round1_technical_failure_blocks_without_a_scientific_rejection(
    tmp_path, monkeypatch
) -> None:
    store = manager(tmp_path)
    observation = {
        "state": "technical-failure",
        "phase": "running",
        "failures": [
            {
                "seed": SEEDS[1],
                "stage": "matched_relax",
                "status": "failed",
            }
        ],
        "provenance": {
            "campaign_id": "generic-quintic-edge6-capacity-round1-20260823",
            "protocol_sha256": token("protocol"),
            "bridge_plan_sha256": token("plan-value"),
            "bridge_plan_file_sha256": token("plan-file"),
            "bridge_ledger_file_sha256": token("ledger-file"),
        },
    }
    monkeypatch.setattr(multi, "inspect_round1", lambda **_: observation)
    result = store.sync_round1()
    assert result["round1"]["outcome"] == "technical-failure"
    assert result["status"] == "blocked-technical-failure"
    assert result["selected_recipe_ids"] == []
    assert result["next_recipe_id"] is None
    assert result["queue_path"] is None
    with pytest.raises(AutoResearchError, match="technical failure"):
        store.run_next()


def test_real_round1_inspector_classifies_failed_worker_as_technical(
    tmp_path, monkeypatch
) -> None:
    campaign_root, bridge_root = open_round1_controller(tmp_path, monkeypatch)
    catalog = multi.validate_catalog(raw_catalog())
    controller = CampaignStore(campaign_root)
    status = controller.status()
    protocol = controller.protocol()
    catalog["data_contract"]["protocol_sha256"] = status["protocol_sha256"]
    catalog["data_contract"]["search_indices_sha256"] = status["search_indices_sha256"]
    catalog["data_contract"]["input_sha256"] = {
        row["role"]: row["sha256"]
        for row in protocol["search_inputs"]
        if row["role"] in catalog["data_contract"]["input_sha256"]
    }
    observation = multi.inspect_round1(
        campaign_root=campaign_root,
        bridge_root=bridge_root,
        catalog=catalog,
    )
    assert observation["state"] == "technical-failure"
    assert observation["failures"] == [
        {
            "seed": SEEDS[0],
            "stage": "local_activate",
            "status": "failed",
        }
    ]


def test_waiting_and_running_round1_are_progress_not_decisions(
    tmp_path, monkeypatch
) -> None:
    store = manager(tmp_path)
    observations = iter(
        [
            {"state": "waiting", "phase": "prepared"},
            {"state": "running", "phase": "running"},
        ]
    )
    monkeypatch.setattr(multi, "inspect_round1", lambda **_: next(observations))
    waiting = store.sync_round1()
    running = store.sync_round1()
    assert waiting["status"] == "waiting-r1"
    assert running["status"] == "r1-running"
    assert running["round1"]["outcome"] == "pending"
    assert running["queue_path"] is None


def test_manager_detects_state_and_proposal_tampering(tmp_path, monkeypatch) -> None:
    store = manager(tmp_path)
    monkeypatch.setattr(
        multi,
        "inspect_round1",
        lambda **_: decided_observation(tmp_path, "promoted"),
    )
    store.sync_round1()
    state_path = tmp_path / "manager" / "state.json"
    state = json.loads(state_path.read_text())
    state["status"] = "complete"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(AutoResearchError, match="state integrity"):
        store.state()


def test_manager_rehashes_parent_checkpoint_and_report_on_every_status(
    tmp_path, monkeypatch
) -> None:
    store = manager(tmp_path)
    observation = decided_observation(tmp_path, "promoted")
    monkeypatch.setattr(multi, "inspect_round1", lambda **_: observation)
    store.sync_round1()
    Path(observation["champion_artifacts"][0]["checkpoint_path"]).write_bytes(
        b"drifted"
    )
    with pytest.raises(AutoResearchError, match="checkpoint drifted"):
        store.state()

    Path(observation["champion_artifacts"][0]["checkpoint_path"]).write_bytes(
        f"champion-promoted-{SEEDS[0]}".encode()
    )
    Path(observation["champion_artifacts"][1]["source_report_path"]).write_text(
        "drifted"
    )
    with pytest.raises(AutoResearchError, match="source report drifted"):
        store.state()


def test_existing_manager_rejects_a_different_round1_root(tmp_path) -> None:
    manager(tmp_path)
    with pytest.raises(AutoResearchError, match="another locked campaign"):
        multi.MultiRoundManager.initialize(
            tmp_path / "manager",
            raw_catalog(),
            round1_campaign_root=tmp_path / "another-controller",
            round1_bridge_root=tmp_path / "bridge",
        )


def test_normalized_catalog_round_trip_rejects_hash_drift() -> None:
    catalog = multi.validate_catalog(raw_catalog())
    normalized_raw = copy.deepcopy(catalog)
    normalized_raw.pop("catalog_sha256")
    assert multi.validate_catalog(normalized_raw) == catalog
    normalized_raw["recipes"][0]["recipe_sha256"] = token("wrong")
    with pytest.raises(AutoResearchError, match="recipe_sha256"):
        multi.validate_catalog(normalized_raw)


def test_historical_r2_rejection_advances_to_real_r3_and_r4_handoffs(
    tmp_path, monkeypatch
) -> None:
    store = manager(tmp_path)
    observation = decided_observation(tmp_path, "scientific-rejected")
    monkeypatch.setattr(multi, "inspect_round1", lambda **_: observation)
    store.sync_round1()

    legacy_root = tmp_path / "legacy-r2"
    legacy_root.mkdir()
    legacy_source = legacy_root / "source.json"
    legacy_source.write_text(json.dumps({"strict": "rejected"}), encoding="utf-8")

    def legacy_result(*, bridge_root, proposal, catalog):
        return multi._build_round_result(
            proposal=proposal,
            outcome="scientific-rejected",
            recommended_role="parent",
            recommended_family=proposal["parent_family"],
            recommended_artifacts=proposal["parent_artifacts"],
            source_kind="historical-auto-research-not-manager-launched",
            source_root=legacy_root,
            source_artifacts=[multi._file_reference(legacy_source, role="source")],
        )

    monkeypatch.setattr(multi, "_legacy_rank_result", legacy_result)
    after_r2 = store.sync_historical_r2(legacy_root)
    assert after_r2["next_recipe_id"] == "r3-all-parameter-joint"
    assert after_r2["completed_rounds"]["2"]["outcome"] == "scientific-rejected"
    assert after_r2["events"][-1]["details"]["source_kind"] == (
        "historical-auto-research-not-manager-launched"
    )
    r3_handoff = store.run_next()
    assert r3_handoff["round"] == 3
    assert r3_handoff["execution_authorized"] is True
    assert (
        multi.validate_execution_handoff(
            store.run_root,
            Path(r3_handoff["proposal"]["path"]).with_name("execution_handoff.json"),
        )
        == r3_handoff
    )

    paired_root = tmp_path / "paired-r3"
    paired_root.mkdir()
    paired_source = paired_root / "source.json"
    paired_source.write_text(json.dumps({"strict": "rejected"}), encoding="utf-8")

    def paired_result(*, bridge_root, proposal, catalog):
        return multi._build_round_result(
            proposal=proposal,
            outcome="scientific-rejected",
            recommended_role="parent",
            recommended_family=proposal["parent_family"],
            recommended_artifacts=proposal["parent_artifacts"],
            source_kind="paired-bridge",
            source_root=paired_root,
            source_artifacts=[multi._file_reference(paired_source, role="source")],
        )

    monkeypatch.setattr(multi, "_paired_round_result", paired_result)
    after_r3 = store.record_round_result(3, paired_root)
    assert after_r3["next_recipe_id"] == "r4-constant-optimizer-path"
    r4_handoff = store.run_next()
    assert r4_handoff["round"] == 4
    assert r4_handoff["parent_lineage"]["source_round"] == 3
    assert r4_handoff["parent_family"] == r3_handoff["parent_family"]


def test_historical_r2_import_cannot_turn_a_rejection_into_a_promotion(
    tmp_path, monkeypatch
) -> None:
    store = manager(tmp_path)
    observation = decided_observation(tmp_path, "scientific-rejected")
    monkeypatch.setattr(multi, "inspect_round1", lambda **_: observation)
    store.sync_round1()
    legacy_root = tmp_path / "legacy-r2"
    legacy_root.mkdir()
    source = legacy_root / "source.json"
    source.write_text("{}", encoding="utf-8")

    def false_promotion(*, bridge_root, proposal, catalog):
        return multi._build_round_result(
            proposal=proposal,
            outcome="promoted",
            recommended_role="candidate",
            recommended_family=proposal["parent_family"],
            recommended_artifacts=proposal["parent_artifacts"],
            source_kind="historical-auto-research-not-manager-launched",
            source_root=legacy_root,
            source_artifacts=[multi._file_reference(source, role="source")],
        )

    monkeypatch.setattr(multi, "_legacy_rank_result", false_promotion)
    with pytest.raises(AutoResearchError, match="strict rejected outcome"):
        store.sync_historical_r2(legacy_root)
