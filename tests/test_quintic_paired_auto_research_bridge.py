from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from gcicy_metric.pipeline.architecture_auto_research import (
    AutoResearchError,
    atomic_write_json,
    digest_value,
)
from gcicy_metric.pipeline import host_stability_gate as host_gate
from gcicy_metric.pipeline import quintic_architecture_multi_round as multi
from gcicy_metric.pipeline import quintic_paired_auto_research_bridge as paired
from gcicy_metric.pipeline.quintic_architecture_round1_bridge import (
    BRIDGE_PLAN_SCHEMA,
    _materialize_batch_plan,
    canonical_array_value_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = (
    ROOT
    / "experiments"
    / "protocols"
    / "generic_quintic_architecture_multi_round_v2.json"
)
SOURCE_COMMIT = "a" * 40
HOST_IDENTITY = hashlib.sha256(b"paired-test-host").hexdigest()


def _manifest(arrays: dict[str, np.ndarray], schema: str) -> dict:
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


def _environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    input_paths = {}
    for role in catalog["data_contract"]["input_sha256"]:
        path = input_root / f"{role}.bin"
        path.write_bytes(role.encode())
        input_paths[role] = path
    for role in ("baseline-checkpoint", "baseline-source-report"):
        path = input_root / f"{role}.bin"
        path.write_bytes(role.encode())
        input_paths[role] = path

    original_sha256 = paired.sha256_file
    pinned = {
        path.resolve(): catalog["data_contract"]["input_sha256"][role]
        for role, path in input_paths.items()
        if role in catalog["data_contract"]["input_sha256"]
    }

    def test_sha256(path: Path) -> str:
        resolved = Path(path).expanduser().resolve()
        return pinned.get(resolved, original_sha256(resolved))

    input_hashes = {
        role: (
            catalog["data_contract"]["input_sha256"][role]
            if role in catalog["data_contract"]["input_sha256"]
            else test_sha256(path)
        )
        for role, path in input_paths.items()
    }

    monkeypatch.setattr(paired, "sha256_file", test_sha256)
    monkeypatch.setattr(
        paired,
        "_source_contract",
        lambda _: {
            "commit": SOURCE_COMMIT,
            "tracked_worktree": "clean",
            "dependencies": {},
        },
    )
    monkeypatch.setattr(paired, "_host_identity_sha256", lambda: HOST_IDENTITY)
    monkeypatch.setattr(paired, "_checkpoint_precision", lambda _: "complex64")
    monkeypatch.setattr(
        paired,
        "_runtime_environment",
        lambda device: {
            "numpy": np.__version__,
            "torch": "test",
            "torch_cuda_build": None,
            "gpu": None,
        },
    )

    round1_root = tmp_path / "round1"
    round1_root.mkdir()
    index_arrays = {
        "fit": np.arange(30_000, dtype=np.int64),
        "selection": np.arange(30_000, 35_000, dtype=np.int64),
        "development_evaluation": np.arange(5_000, dtype=np.int64),
    }
    index_path = round1_root / "fixed_indices.npz"
    np.savez_compressed(index_path, **index_arrays)
    index_manifest = {
        **_manifest(index_arrays, "gcicy-quintic-round1-index-values-v1"),
        "path": str(index_path.resolve()),
        "file_sha256": test_sha256(index_path),
        "partition_seed": 2026082302,
        "mapping": {},
    }
    batch_catalog = {
        row["seed"]: row for row in catalog["data_contract"]["matched_batch_plans"]
    }
    seed_rows = []
    for seed in catalog["promotion_seeds"]:
        registered = batch_catalog[seed]
        batch = _materialize_batch_plan(
            round1_root / "seeds" / str(seed) / "matched_batch_plan.npy",
            seed=registered["plan_seed"],
            steps=600,
            batch_size=1024,
        )
        assert batch["file_sha256"] == registered["file_sha256"]
        assert batch["value_set_sha256"] == registered["value_set_sha256"]
        seed_rows.append(
            {
                "seed": seed,
                "optimizer_seed": seed,
                "data_seed": 2026082301,
                "batch_plan_seed": registered["plan_seed"],
                "batch_plan": batch,
            }
        )
    r1_payload = {
        "schema": BRIDGE_PLAN_SCHEMA,
        "protocol_sha256": catalog["data_contract"]["protocol_sha256"],
        "search_indices_sha256": catalog["data_contract"]["search_indices_sha256"],
        "historical_confirmation": "absent",
        "input_roles": {
            role: {
                "role": role,
                "path": str(path.resolve()),
                "sha256": input_hashes[role],
            }
            for role, path in input_paths.items()
        },
        "index_manifest": index_manifest,
        "seeds": seed_rows,
    }
    r1_plan = {**r1_payload, "plan_sha256": digest_value(r1_payload)}
    atomic_write_json(round1_root / "plan.json", r1_plan)
    parents = {}
    for seed in catalog["promotion_seeds"]:
        parent = tmp_path / f"parent-{seed}.pt"
        parent.write_bytes(f"complex64 parent {seed}".encode())
        parents[seed] = parent
    return {
        "catalog": catalog,
        "round1_root": round1_root,
        "parents": parents,
        "runtime": {
            "device": "cuda",
            "threads": 2,
            "train_chunk_size": 128,
            "feature_batch_size": 128,
            "eval_batch_size": 64,
        },
    }


def _prepare(env: dict, tmp_path: Path, recipe_id: str, name: str = "paired") -> dict:
    if "certificates" not in env:
        env["certificates"] = _certificates(env, tmp_path / "host-certificates")
    manager_root, handoff_path = _manager_handoff(env, tmp_path, recipe_id)
    return paired.prepare_paired_bridge(
        manager_root=manager_root,
        execution_handoff_path=handoff_path,
        output_root=tmp_path / name,
        host_stability_certificates=env["certificates"],
        runtime=env["runtime"],
        repository_root=ROOT,
    )


def _manager_handoff(env: dict, tmp_path: Path, recipe_id: str) -> tuple[Path, Path]:
    cached = env.setdefault("manager_handoffs", {})
    if recipe_id in cached:
        return cached[recipe_id]
    manager_root = tmp_path / f"manager-{recipe_id}"
    provenance = tmp_path / f"manager-provenance-{recipe_id}"
    provenance.mkdir()
    source_artifacts = []
    models = []
    for seed, parent in sorted(env["parents"].items()):
        report = provenance / f"parent-{seed}.json"
        atomic_write_json(report, {"seed": seed, "kind": "test-parent"})
        checkpoint_sha256 = multi.sha256_file(parent)
        models.append({"seed": seed, "checkpoint_sha256": checkpoint_sha256})
        source_artifacts.append(
            {
                "seed": seed,
                "checkpoint_path": str(parent.resolve()),
                "checkpoint_sha256": checkpoint_sha256,
                "role": "baseline-retained",
                "source_report_path": str(report.resolve()),
                "source_report_sha256": multi.sha256_file(report),
            }
        )
    family_payload = {"models": models}
    family = {**family_payload, "family_sha256": digest_value(family_payload)}
    bridge_plan = provenance / "plan.json"
    bridge_ledger = provenance / "ledger.json"
    evidence = provenance / "search_evidence.json"
    adjudication = provenance / "round1-adjudication.json"
    atomic_write_json(bridge_plan, {"plan_sha256": "1" * 64})
    atomic_write_json(bridge_ledger, {"state": "normalized"})
    atomic_write_json(evidence, {"schema": "test-round1-evidence"})
    atomic_write_json(adjudication, {"outcome": "scientific-rejected"})
    data = multi.validate_catalog(env["catalog"])["data_contract"]
    observation = {
        "state": "decided",
        "phase": "complete",
        "outcome": "scientific-rejected",
        "champion_family": family,
        "champion_artifacts": source_artifacts,
        "provenance": {
            "campaign_root": str(provenance),
            "bridge_root": str(provenance),
            "protocol_sha256": data["protocol_sha256"],
            "search_indices_sha256": data["search_indices_sha256"],
            "bridge_plan_sha256": "1" * 64,
            "bridge_plan_file_sha256": multi.sha256_file(bridge_plan),
            "bridge_ledger_file_sha256": multi.sha256_file(bridge_ledger),
            "search_evidence_sha256": multi.sha256_file(evidence),
            "search_evidence_value_sha256": digest_value(
                json.loads(evidence.read_text())
            ),
            "adjudication_path": str(adjudication),
            "adjudication_sha256": multi.sha256_file(adjudication),
            "input_sha256": data["input_sha256"],
            "matched_batch_plans": data["matched_batch_plans"],
        },
    }
    manager = multi.MultiRoundManager.initialize(
        manager_root,
        env["catalog"],
        round1_campaign_root=tmp_path / f"controller-{recipe_id}",
        round1_bridge_root=env["round1_root"],
    )
    original_inspector = multi.inspect_round1
    multi.inspect_round1 = lambda **_: observation
    try:
        manager.sync_round1()
    finally:
        multi.inspect_round1 = original_inspector
    legacy_root = provenance / "legacy-r2"
    legacy_root.mkdir()
    legacy_source = legacy_root / "source.json"
    atomic_write_json(legacy_source, {"outcome": "strict-rejected"})
    original_legacy = multi._legacy_rank_result

    def fake_legacy(*, bridge_root, proposal, catalog):
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

    multi._legacy_rank_result = fake_legacy
    try:
        manager.sync_historical_r2(legacy_root)
    finally:
        multi._legacy_rank_result = original_legacy
    if recipe_id == "r4-constant-optimizer-path":
        manager.run_next()
        paired_root = provenance / "paired-r3"
        paired_root.mkdir()
        paired_source = paired_root / "source.json"
        atomic_write_json(paired_source, {"outcome": "strict-rejected"})
        original_paired = multi._paired_round_result

        def fake_paired(*, bridge_root, proposal, catalog):
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

        multi._paired_round_result = fake_paired
        try:
            manager.record_round_result(3, paired_root)
        finally:
            multi._paired_round_result = original_paired
    handoff = manager.run_next()
    result = (
        manager_root,
        Path(handoff["proposal"]["path"]).with_name("execution_handoff.json"),
    )
    cached[recipe_id] = result
    return result


def _certificate(
    env: dict, tmp_path: Path, seed: int, *, diagnostic: bool = False
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    issued = datetime.now(timezone.utc).replace(microsecond=0)
    parent = env["parents"][seed]
    checkpoint_sha256 = host_gate.sha256_file(parent)
    process_rows = [
        {
            "attempt": index,
            "returncode": 0,
            "stdout_sha256": hashlib.sha256(str(index).encode()).hexdigest(),
            "checkpoint_sha256": checkpoint_sha256,
            "host_identity_sha256": HOST_IDENTITY,
            "source_commit": SOURCE_COMMIT,
        }
        for index in range(1, 21)
    ]
    probes = []
    for index in range(1, 4):
        report = tmp_path / f"host-probe-{seed}-{index}.json"
        report.write_text(f"probe {index}\n", encoding="utf-8")
        probes.append(
            {
                "schema": host_gate.GPU_PROBE_SCHEMA,
                "probe_id": f"probe-{index}",
                "checkpoint_sha256": checkpoint_sha256,
                "host_identity_sha256": HOST_IDENTITY,
                "source_commit": SOURCE_COMMIT,
                "started_utc": (issued - timedelta(minutes=2)).isoformat(),
                "finished_utc": (issued - timedelta(minutes=1)).isoformat(),
                "report_path": str(report.resolve()),
                "report_sha256": host_gate.sha256_file(report),
                "optimizer_steps": 10,
                "selection_evaluation_completed": True,
                "all_finite": True,
                "minimum_metric_eigenvalue": 0.01,
                "maximum_allocated_bytes": 10 * 1024**3,
            }
        )
    value = host_gate.build_certificate(
        checkpoint_path=parent,
        expected_checkpoint_sha256=checkpoint_sha256,
        host_identity_sha256=HOST_IDENTITY,
        source_commit=SOURCE_COMMIT,
        kernel_log="kernel: segfault\n" if diagnostic else "quiet\n",
        fresh_process_results=process_rows,
        gpu_probe_values=probes,
        issued_at=issued,
    )
    prefix = "diagnostic-certificate" if diagnostic else "certificate"
    path = tmp_path / f"{prefix}-{seed}.json"
    host_gate.publish_certificate(path, value)
    return path


def _certificates(
    env: dict, root: Path, *, diagnostic: bool = False
) -> dict[int, Path]:
    return {
        seed: _certificate(env, root, seed, diagnostic=diagnostic)
        for seed in sorted(env["parents"])
    }


def _argv_map(argv: list[str]) -> dict[str, object]:
    result: dict[str, object] = {}
    index = 2
    flags = {
        "--require-equal-total-parameters",
        "--development-only",
        "--development-evaluation",
    }
    while index < len(argv):
        key = argv[index]
        if key in flags:
            result[key] = True
            index += 1
        else:
            result[key] = argv[index + 1]
            index += 2
    return result


def _completed_stage(
    root: Path,
    plan: dict,
    seed_row: dict,
    certificate_path: Path,
    report_path: Path,
) -> dict:
    certificate = paired._validate_host_certificate(plan, seed_row, certificate_path)
    authorization = paired._publish_attempt_authorization(
        root,
        plan=plan,
        seed_row=seed_row,
        certificate_row=certificate,
    )
    return {
        "worker": "complete",
        "worker_report_sha256": paired.sha256_file(report_path),
        "host_stability_certificate": certificate,
        "authorization": authorization,
        "attempt_seal": paired._publish_attempt_seal(
            root,
            plan=plan,
            seed_row=seed_row,
            authorization=authorization,
            report_path=report_path,
        ),
    }


def test_prepare_builds_only_the_two_registered_same_parent_commands(
    tmp_path, monkeypatch
):
    env = _environment(tmp_path, monkeypatch)
    r3 = _prepare(env, tmp_path, "r3-all-parameter-joint", "r3")
    r4 = _prepare(env, tmp_path, "r4-constant-optimizer-path", "r4")
    r3_args = _argv_map(paired._worker_argv(r3, r3["seeds"][0]))
    r4_args = _argv_map(paired._worker_argv(r4, r4["seeds"][0]))
    assert r3["manager_handoff"]["proposal_sha256"]
    assert r4["manager_handoff"]["proposal_sha256"]
    assert (
        r3["manager_handoff"]["parent_family"] == r4["manager_handoff"]["parent_family"]
    )
    assert len({row["parent_checkpoint"]["sha256"] for row in r3["seeds"]}) == 3
    assert all(
        row["host_stability_certificate"]["checkpoint_sha256"]
        == row["parent_checkpoint"]["sha256"]
        for row in r3["seeds"]
    )
    assert r3_args["--control-checkpoint"] == r3_args["--candidate-checkpoint"]
    assert r3_args["--control-trainable-scope"] == "internal"
    assert r3_args["--candidate-trainable-scope"] == "all"
    assert (
        r3_args["--control-scheduler"] == r3_args["--candidate-scheduler"] == "cosine"
    )
    assert (
        r4_args["--control-trainable-scope"]
        == r4_args["--candidate-trainable-scope"]
        == "all"
    )
    assert r4_args["--control-scheduler"] == "cosine"
    assert r4_args["--candidate-scheduler"] == "constant"
    for row in (r3_args, r4_args):
        assert row["--steps"] == "600"
        assert row["--development-only"] is True
        assert row["--development-evaluation"] is True
        assert row["--require-equal-total-parameters"] is True
        assert not any(
            "confirmation" in str(value).lower() or "blind" in str(value).lower()
            for value in row.values()
        )


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        ("manager", "state integrity"),
        ("proposal", "proposal file drifted"),
        ("parent", "checkpoint drifted"),
    ),
)
def test_prepare_rejects_manager_proposal_or_parent_tampering(
    tmp_path, monkeypatch, tamper, message
):
    env = _environment(tmp_path, monkeypatch)
    manager_root, handoff_path = _manager_handoff(
        env, tmp_path, "r3-all-parameter-joint"
    )
    certificates = _certificates(env, tmp_path / "certificates")
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    if tamper == "manager":
        state_path = manager_root / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["status"] = "tampered"
        atomic_write_json(state_path, state)
    elif tamper == "proposal":
        proposal_path = Path(handoff["proposal"]["path"])
        proposal_path.write_text(
            proposal_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
        )
    else:
        next(iter(env["parents"].values())).write_bytes(b"tampered parent")
    with pytest.raises(AutoResearchError, match=message):
        paired.prepare_paired_bridge(
            manager_root=manager_root,
            execution_handoff_path=handoff_path,
            output_root=tmp_path / "must-not-prepare",
            host_stability_certificates=certificates,
            runtime=env["runtime"],
            repository_root=ROOT,
        )


def test_runtime_certificate_can_be_renewed_after_prepare(tmp_path, monkeypatch):
    env = _environment(tmp_path, monkeypatch)
    plan = _prepare(env, tmp_path, "r3-all-parameter-joint")
    renewed = _certificates(env, tmp_path / "renewed-certificates")
    first = plan["seeds"][0]
    assert Path(renewed[first["seed"]]).resolve() != Path(
        first["host_stability_certificate"]["path"]
    )

    @contextmanager
    def fake_gpu_lock(root, gpu_id, *, timeout_seconds):
        yield root / f"gcicy-tn-gpu-{gpu_id}.lock"

    def fake_run(argv, **kwargs):
        values = _argv_map(argv)
        seed_row = next(
            row for row in plan["seeds"] if row["output_dir"] == values["--output-dir"]
        )
        _write_worker_report(plan, seed_row)
        return object()

    monkeypatch.setattr(paired, "gpu_lock", fake_gpu_lock)
    monkeypatch.setattr(paired.subprocess, "run", fake_run)
    ledger = paired.run_paired_bridge(
        tmp_path / "paired",
        host_stability_certificates=renewed,
        selected_seed=first["seed"],
    )
    stage = ledger["seeds"][str(first["seed"])]
    assert stage["worker"] == "complete"
    assert stage["host_stability_certificate"]["path"] == str(
        Path(renewed[first["seed"]]).resolve()
    )
    assert stage["authorization"]["authorization_sha256"]
    assert stage["attempt_seal"]["attempt_seal_sha256"]


def test_scientific_worker_cannot_start_without_an_authorized_fresh_certificate(
    tmp_path, monkeypatch
):
    env = _environment(tmp_path, monkeypatch)
    plan = _prepare(env, tmp_path, "r3-all-parameter-joint")
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("worker must not start")

    monkeypatch.setattr(paired.subprocess, "run", forbidden)
    incomplete = dict(env["certificates"])
    incomplete.pop(next(iter(incomplete)))
    with pytest.raises(AutoResearchError, match="cover exactly"):
        paired.run_paired_bridge(
            tmp_path / "paired",
            host_stability_certificates=incomplete,
        )
    assert called is False

    swapped = dict(env["certificates"])
    first_seed, second_seed = sorted(swapped)[:2]
    swapped[first_seed], swapped[second_seed] = (
        swapped[second_seed],
        swapped[first_seed],
    )
    with pytest.raises(AutoResearchError, match="certificate rejected"):
        paired.run_paired_bridge(
            tmp_path / "paired", host_stability_certificates=swapped
        )
    assert called is False

    # Restore the source/host helpers indirectly affected by subprocess mocking.
    monkeypatch.setattr(paired, "_source_contract", lambda _: plan["source_contract"])
    monkeypatch.setattr(paired, "_host_identity_sha256", lambda: HOST_IDENTITY)
    diagnostic = _certificates(env, tmp_path / "diagnostic", diagnostic=True)
    manager_root, handoff_path = _manager_handoff(
        env, tmp_path, "r3-all-parameter-joint"
    )
    with pytest.raises(AutoResearchError, match="diagnostic work only"):
        paired.prepare_paired_bridge(
            manager_root=manager_root,
            execution_handoff_path=handoff_path,
            output_root=tmp_path / "diagnostic-plan",
            host_stability_certificates=diagnostic,
            runtime=env["runtime"],
            repository_root=ROOT,
        )
    assert called is False


def test_run_resumes_by_seed_and_uses_shell_false(tmp_path, monkeypatch):
    env = _environment(tmp_path, monkeypatch)
    plan = _prepare(env, tmp_path, "r3-all-parameter-joint")
    root = tmp_path / "paired"
    ledger = paired._read_ledger(root / "ledger.json")
    first = plan["seeds"][0]
    first_report = Path(first["output_dir"]) / "report.json"
    first_report.parent.mkdir(parents=True)
    first_report.write_text("{}\n", encoding="utf-8")
    ledger["seeds"][str(first["seed"])] = _completed_stage(
        root,
        plan,
        first,
        env["certificates"][first["seed"]],
        first_report,
    )
    paired._write_ledger(root / "ledger.json", ledger)
    monkeypatch.setattr(paired, "_verify_report_artifacts", lambda *args, **kwargs: {})
    calls = []
    lock_events = []

    @contextmanager
    def fake_gpu_lock(root, gpu_id, *, timeout_seconds):
        lock_events.append(("acquired", root, gpu_id, timeout_seconds))
        yield root / f"gcicy-tn-gpu-{gpu_id}.lock"
        lock_events.append(("released", root, gpu_id, timeout_seconds))

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        values = _argv_map(argv)
        report = Path(values["--output-dir"]) / "report.json"
        report.parent.mkdir(parents=True)
        report.write_text("{}\n", encoding="utf-8")
        return object()

    monkeypatch.setattr(paired.subprocess, "run", fake_run)
    monkeypatch.setattr(paired, "gpu_lock", fake_gpu_lock)
    result = paired.run_paired_bridge(
        root, host_stability_certificates=env["certificates"]
    )
    assert len(calls) == 2
    assert all(call[1]["shell"] is False and call[1]["check"] is True for call in calls)
    assert all(row["worker"] == "complete" for row in result["seeds"].values())
    assert [row[0] for row in lock_events] == ["acquired", "released"]
    assert lock_events[0][1:] == (
        paired.GPU_LOCK_ROOT,
        paired.GPU_ID,
        -1,
    )


def _evaluation(sigma: float, chi: float, q999: float, cvar: float) -> dict:
    return {
        "statistics": {
            "sigma_official_formula": sigma,
            "weighted_rms_abs_residual": chi,
            "min_eigenvalue_weighted_quantiles": {"q0.0000": 0.01},
            "nonpositive_min_eigenvalue": {"count": 0},
        },
        "tail": {"q999": q999, "cvar99": cvar},
    }


def _write_worker_report(plan: dict, seed_row: dict, *, gain: float = 0.005) -> Path:
    output = Path(seed_row["output_dir"])
    output.mkdir(parents=True)
    indices_path = output / "data_indices.npz"
    fixed = np.load(plan["round1_data"]["index_manifest"]["path"], allow_pickle=False)
    np.savez_compressed(indices_path, **{name: fixed[name] for name in fixed.files})
    batch_array = np.load(seed_row["batch_plan"]["path"], allow_pickle=False)
    batch_path = output / "batch_plan.npy"
    np.save(batch_path, batch_array, allow_pickle=False)
    checkpoints = {}
    for role in ("control", "candidate"):
        path = output / f"{role}_polished.pt"
        path.write_bytes(f"{role}-{seed_row['seed']}".encode())
        checkpoints[role] = {"path": path, "sha256": paired.sha256_file(path)}
    intervention = plan["recipe"]["intervention"]
    budget = plan["recipe"]["budget"]
    roles = plan["round1_data"]["input_roles"]
    policies = {
        role: {
            "trainable_scope": intervention[role]["trainable_scope"],
            "scheduler": intervention[role]["scheduler"],
        }
        for role in ("control", "candidate")
    }
    total = 131_550
    internal = 121_750
    initial_statistics = {"sigma_official_formula": 1.1}
    initial_tail = {"q999": 1.2}
    results = {}
    for role in ("control", "candidate"):
        trainable = total if policies[role]["trainable_scope"] == "all" else internal
        results[role] = {
            "checkpoint": str(checkpoints[role]["path"]),
            "checkpoint_sha256": checkpoints[role]["sha256"],
            "total_real_parameters": total,
            "trainable_real_parameters": trainable,
            "new_real_parameters_vs_control": 0,
            "training_policy": policies[role],
            "fixed_batch_plan_sha256": paired._batch_digest(batch_array),
            "initial_statistics": initial_statistics,
            "initial_tail": initial_tail,
            "parameter_audit": {"all_frozen_parameters_exactly_unchanged": True},
        }
    configuration = {
        "control_checkpoint": seed_row["parent_checkpoint"]["path"],
        "candidate_checkpoint": seed_row["parent_checkpoint"]["path"],
        "train_points": roles["search-native-points"]["path"],
        "train_pullbacks": roles["search-native-pullbacks"]["path"],
        "selection_points": roles["search-native-points"]["path"],
        "selection_pullbacks": roles["search-native-pullbacks"]["path"],
        "development_evaluation_points": roles["development-evaluation-points"]["path"],
        "development_evaluation_pullbacks": roles["development-evaluation-pullbacks"][
            "path"
        ],
        "confirmation_points": None,
        "confirmation_pullbacks": None,
        "exclude_indices_file": [],
        "fixed_indices_file": plan["round1_data"]["index_manifest"]["path"],
        "fixed_batch_plan_file": seed_row["batch_plan"]["path"],
        "output_dir": seed_row["output_dir"],
        "steps": 600,
        "learning_rate": budget["learning_rate"],
        "scheduler": "cosine",
        "control_scheduler": intervention["control"]["scheduler"],
        "candidate_scheduler": intervention["candidate"]["scheduler"],
        "control_trainable_scope": intervention["control"]["trainable_scope"],
        "candidate_trainable_scope": intervention["candidate"]["trainable_scope"],
        "require_equal_total_parameters": True,
        "eval_every": 25,
        "train_size": 30_000,
        "selection_size": 5_000,
        "development_evaluation_size": 5_000,
        "stochastic_batch_size": 1024,
        "gradient_clip_norm": 1.0,
        "minimum_relative_sigma_gain": 0.0,
        "minimum_relative_chi_gain": 0.0,
        "maximum_selection_tail_relative_degradation": plan["recipe"]["promotion_gate"][
            "maximum_tail_relative_degradation"
        ],
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
    }
    report = {
        "schema": "generic-quintic-tree-joint-relaxation-v1",
        "contract": {"precision": "complex64"},
        "configuration": configuration,
        "training_policies": policies,
        "total_parameter_parity": {
            "required": True,
            "equal": True,
            "control_total_real_parameters": total,
            "candidate_total_real_parameters": total,
            "candidate_minus_control": 0,
        },
        "source_checkpoint_sha256": {
            "control": seed_row["parent_checkpoint"]["sha256"],
            "candidate": seed_row["parent_checkpoint"]["sha256"],
        },
        "data": {
            "indices": str(indices_path),
            "indices_sha256": paired.sha256_file(indices_path),
            "fixed_indices_file": plan["round1_data"]["index_manifest"]["path"],
            "fixed_indices_file_sha256": plan["round1_data"]["index_manifest"][
                "file_sha256"
            ],
            "input_sha256": paired._expected_input_hashes(plan),
            "counts": {
                "fit": 30_000,
                "selection": 5_000,
                "development_evaluation": 5_000,
            },
            "data_seed": seed_row["data_seed"],
        },
        "batch_plan": {
            "path": str(batch_path),
            "sha256": paired.sha256_file(batch_path),
            "array_sha256": paired._batch_digest(batch_array),
            "same_for_both_arms": True,
            "fixed_source_sha256": seed_row["batch_plan"]["file_sha256"],
            "seed": seed_row["batch_plan_seed"],
            "steps": 600,
            "batch_size": 1024,
        },
        "results": results,
        "development_evaluation": {
            "control": _evaluation(1.0, 1.0, 1.0, 1.0),
            "candidate": _evaluation(1.0 - gain, 1.0 - gain, 1.001, 1.001),
            "paired_improvement": {
                "sigma": {"ci95_low": 0.001},
                "e2": {"ci95_low": 0.001},
            },
            "passes": True,
            "selection_or_checkpoint_role": "none",
        },
        "confirmation": None,
        "confirmation_passes": None,
    }
    path = output / "report.json"
    atomic_write_json(path, report)
    return path


def test_normalizer_and_catalog_adjudication_are_strict_and_rehash_outputs(
    tmp_path, monkeypatch
):
    env = _environment(tmp_path, monkeypatch)
    plan = _prepare(env, tmp_path, "r3-all-parameter-joint")
    root = tmp_path / "paired"
    ledger = paired._read_ledger(root / "ledger.json")
    for seed_row in plan["seeds"]:
        report = _write_worker_report(plan, seed_row)
        ledger["seeds"][str(seed_row["seed"])] = _completed_stage(
            root,
            plan,
            seed_row,
            env["certificates"][seed_row["seed"]],
            report,
        )
    ledger["state"] = "workers-complete"
    paired._write_ledger(root / "ledger.json", ledger)

    evidence = paired.normalize_paired_bridge(root)
    assert len(evidence["seeds"]) == 3
    adjudication = paired.adjudicate_paired_bridge(root)
    assert adjudication["promotion_passes"] is True
    assert adjudication["recommended_role"] == "candidate"
    assert all(adjudication["gates"].values())
    assert paired.paired_bridge_status(root)["state"] == "adjudicated"

    first_report = Path(evidence["seeds"][0]["source_report"]["path"])
    first_report.write_text(first_report.read_text() + " ", encoding="utf-8")
    with pytest.raises(AutoResearchError, match="report differs"):
        paired.paired_bridge_status(root)


def test_scientific_rejection_rolls_back_to_the_parent_family(tmp_path, monkeypatch):
    env = _environment(tmp_path, monkeypatch)
    plan = _prepare(env, tmp_path, "r3-all-parameter-joint")
    root = tmp_path / "paired"
    ledger = paired._read_ledger(root / "ledger.json")
    for seed_row in plan["seeds"]:
        report = _write_worker_report(plan, seed_row, gain=-0.001)
        ledger["seeds"][str(seed_row["seed"])] = _completed_stage(
            root,
            plan,
            seed_row,
            env["certificates"][seed_row["seed"]],
            report,
        )
    ledger["state"] = "workers-complete"
    paired._write_ledger(root / "ledger.json", ledger)

    paired.normalize_paired_bridge(root)
    adjudication = paired.adjudicate_paired_bridge(root)
    assert adjudication["promotion_passes"] is False
    assert adjudication["recommended_role"] == "parent"
    assert [
        row["checkpoint_sha256"] for row in adjudication["recommended_family"]["models"]
    ] == [row["parent_checkpoint"]["sha256"] for row in plan["seeds"]]
