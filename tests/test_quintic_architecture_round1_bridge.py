from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

from gcicy_metric.pipeline import architecture_auto_research as control
from gcicy_metric.pipeline.architecture_auto_research import (
    ACTION_SCHEMA,
    PROTOCOL_SCHEMA,
    AutoResearchError,
    CampaignStore,
    fixed_search_indices,
    validate_search_evidence,
    validate_protocol,
)
from gcicy_metric.pipeline import quintic_architecture_round1_bridge as bridge


SEEDS = [202608231, 202608232, 202608233]


def _budget(*, updates: int, batch: int, learning_rate: float, every: int):
    return {
        "optimizer_updates": updates,
        "batch_size": batch,
        "learning_rate": learning_rate,
        "gradient_clip_norm": 1.0,
        "train_examples": 30_000,
        "evaluation_examples": 5_000,
        "eval_every": every,
        "scheduler": "cosine",
    }


def _raw_protocol(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    paths = {}
    for role, name in {
        "baseline-checkpoint": "baseline.pt",
        "baseline-source-report": "baseline_source.json",
        "search-native-points": "native_points.npz",
        "search-native-pullbacks": "native_pullbacks.npy",
        "development-evaluation-points": "dev_eval_points.npz",
        "development-evaluation-pullbacks": "dev_eval_pullbacks.npy",
    }.items():
        path = tmp_path / name
        path.write_bytes(role.encode("utf-8"))
        paths[role] = path

    original_sha256 = control.sha256_file

    def pinned_sha256(path: Path) -> str:
        path = Path(path).resolve()
        if path == paths["baseline-checkpoint"].resolve():
            return bridge.ROUND1_BASELINE_SHA256
        return original_sha256(path)

    monkeypatch.setattr(control, "sha256_file", pinned_sha256)
    rows = []
    for role, path in paths.items():
        rows.append(
            {
                "role": role,
                "path": str(path),
                "sha256": pinned_sha256(path),
            }
        )
    return (
        {
            "schema": PROTOCOL_SCHEMA,
            "campaign_id": "round1-edge6-test",
            "baseline_family": {
                "models": [
                    {
                        "seed": seed,
                        "checkpoint_sha256": bridge.ROUND1_BASELINE_SHA256,
                    }
                    for seed in SEEDS
                ]
            },
            "promotion_seeds": SEEDS,
            "search_inputs": rows,
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
                "local_activate": _budget(
                    updates=1_200,
                    batch=1_024,
                    learning_rate=3.0e-4,
                    every=10,
                ),
                "matched_relax": _budget(
                    updates=600,
                    batch=1_024,
                    learning_rate=3.0e-6,
                    every=25,
                ),
            },
            "maximum_rounds": 4,
        },
        paths,
        pinned_sha256,
    )


def _action(protocol, indices, *, mutation=None):
    return {
        "schema": ACTION_SCHEMA,
        "candidate_id": "edge6-rank39",
        "round": 1,
        "parent_family_sha256": protocol["baseline_family"]["family_sha256"],
        "search_indices_sha256": indices["indices_sha256"],
        "mutation": mutation
        or {
            "kind": "rank",
            "target": "internal-edge",
            "edge": 6,
            "source_dimension": 25,
            "target_dimension": 39,
            "structural_maximum": 100,
            "new_output_real_parameters": 9_800,
        },
        "pipeline": [{"kind": "local-activate"}, {"kind": "matched-relax"}],
    }


def _contract(tmp_path, monkeypatch):
    raw, paths, pinned_sha256 = _raw_protocol(tmp_path, monkeypatch)
    protocol = validate_protocol(raw)
    indices = fixed_search_indices(protocol)
    action = _action(protocol, indices)
    return raw, protocol, indices, action, paths, pinned_sha256


def test_round1_accepts_only_the_exact_9800_real_edge6_action(tmp_path, monkeypatch):
    raw, protocol, indices, action, _, _ = _contract(tmp_path, monkeypatch)
    normalized_protocol, normalized_action = bridge.validate_round1_contract(
        raw, indices, action
    )
    assert normalized_protocol["budgets"]["local_activate"]["optimizer_updates"] == 1200
    assert normalized_action["mutation"]["target_dimension"] == 39

    for mutation in (
        {"kind": "topology", "transport": "five-leaf-4plus1-to-2plus3"},
        {"kind": "root-residual", "topology": "five-leaf-2plus3", "rank": 8},
        {
            "kind": "rank",
            "target": "internal-edge",
            "edge": 6,
            "source_dimension": 25,
            "target_dimension": 75,
            "structural_maximum": 100,
            "new_output_real_parameters": 9_800,
        },
    ):
        with pytest.raises(AutoResearchError, match="only internal edge 6"):
            bridge.validate_round1_contract(
                raw, indices, _action(protocol, indices, mutation=mutation)
            )


def test_frozen_indices_are_deterministically_shuffled_before_split_and_hashed(
    tmp_path,
):
    indices = {
        "train_indices": list(range(40_000, 75_000)),
        "evaluation_indices": list(range(8_000, 13_000)),
    }
    path = tmp_path / "fixed_indices.npz"
    partition_seed = 2026082302
    manifest = bridge.materialize_round1_indices(
        indices, path, partition_seed=partition_seed
    )
    artifact = np.load(path, allow_pickle=False)
    shuffled = np.arange(40_000, 75_000)[
        np.random.default_rng(partition_seed).permutation(35_000)
    ]
    assert np.array_equal(artifact["fit"], shuffled[:30_000])
    assert np.array_equal(artifact["selection"], shuffled[30_000:])
    assert artifact["fit"].max() > 70_000
    assert artifact["selection"].min() < 45_000
    assert np.array_equal(artifact["development_evaluation"], np.arange(8_000, 13_000))
    assert manifest["arrays"]["fit"]["value_sha256"] == (
        bridge.canonical_array_value_sha256(artifact["fit"])
    )
    assert len(manifest["value_set_sha256"]) == 64
    assert manifest["partition_seed"] == partition_seed
    assert (
        bridge.materialize_round1_indices(
            indices, path, partition_seed=partition_seed
        )
        == manifest
    )


def test_runtime_contract_is_present_and_rejects_extra_fields():
    runtime = {
        "device": "cuda",
        "threads": 6,
        "train_chunk_size": 512,
        "feature_batch_size": 512,
        "eval_batch_size": 256,
        "early_stopping_evaluations": 6,
    }
    assert bridge._validate_runtime(runtime) == runtime
    with pytest.raises(AutoResearchError, match="fields are not exact"):
        bridge._validate_runtime({**runtime, "command": "arbitrary.py"})


def test_prepared_plan_has_three_seeds_separate_rngs_and_fixed_worker_argv(
    tmp_path, monkeypatch
):
    raw, protocol, indices, action, paths, pinned_sha256 = _contract(
        tmp_path, monkeypatch
    )
    run_root = tmp_path / "campaign"
    store = CampaignStore.initialize(run_root, raw)
    store.register_action(action)
    monkeypatch.setattr(bridge, "sha256_file", pinned_sha256)
    monkeypatch.setattr(
        bridge,
        "_git_source_contract",
        lambda _: {"commit": "0" * 40, "tracked_worktree": "clean", "dependencies": {}},
    )
    runtime = {
        "device": "cpu",
        "threads": 2,
        "train_chunk_size": 128,
        "feature_batch_size": 128,
        "eval_batch_size": 64,
        "early_stopping_evaluations": 6,
    }
    plan = bridge.prepare_round1_bridge(
        campaign_run_root=run_root,
        candidate_id="edge6-rank39",
        output_root=tmp_path / "bridge-run",
        baseline_checkpoints={seed: paths["baseline-checkpoint"] for seed in SEEDS},
        runtime=runtime,
        repository_root=Path(__file__).resolve().parents[1],
    )
    assert plan["runtime"] == runtime
    assert plan["rank_activation_scale"] == 1.0
    assert plan["runtime_environment"]["gpu"] is None
    assert plan["runtime_environment"]["numpy"] == np.__version__
    assert plan["runtime_environment"]["torch"]
    assert [row["optimizer_seed"] for row in plan["seeds"]] == SEEDS
    assert len({row["data_seed"] for row in plan["seeds"]}) == 1
    assert len({row["batch_plan_seed"] for row in plan["seeds"]}) == 3
    for row in plan["seeds"]:
        local = bridge._stage_argv(plan, row, "local_activate")
        matched = bridge._stage_argv(plan, row, "matched_relax")
        assert "--sweeps" in local and local[local.index("--sweeps") + 1] == "2"
        assert "--epochs-per-block" in local
        assert local[local.index("--epochs-per-block") + 1] == "200"
        assert local[local.index("--rank-activation-scale") + 1] == "1.0"
        assert "--scheduler" in matched
        assert matched[matched.index("--scheduler") + 1] == "cosine"
        assert "--development-evaluation" in matched
        assert not any("confirmation" in value.lower() for value in local + matched)
        assert local[0] == sys.executable and matched[0] == sys.executable
    scaled_local = bridge._stage_argv(
        {**plan, "rank_activation_scale": 0.25}, plan["seeds"][0], "local_activate"
    )
    assert scaled_local[scaled_local.index("--rank-activation-scale") + 1] == "0.25"
    with pytest.raises(AutoResearchError, match="not preregistered"):
        bridge._rank_activation_scale(1.01)


def test_adaptive_development_mode_rejects_confirmation_paths(tmp_path, monkeypatch):
    from scripts import train_generic_quintic_adaptive_direct_blocks as worker

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "worker",
            "--initial-checkpoint",
            str(tmp_path / "parent.pt"),
            "--train-points",
            str(tmp_path / "train.npz"),
            "--train-pullbacks",
            str(tmp_path / "train.npy"),
            "--selection-points",
            str(tmp_path / "selection.npz"),
            "--selection-pullbacks",
            str(tmp_path / "selection.npy"),
            "--confirmation-points",
            str(tmp_path / "historical.npz"),
            "--confirmation-pullbacks",
            str(tmp_path / "historical.npy"),
            "--output-dir",
            str(tmp_path / "output"),
            "--development-only",
        ],
    )
    with pytest.raises(ValueError, match="must not receive confirmation"):
        worker.validate_args(worker.parse_args())


@pytest.mark.parametrize(
    ("fixed", "message"),
    [
        ({"fit": np.array([0, 1])}, "missing split selection"),
        (
            {"fit": np.array([0, 0]), "selection": np.array([2])},
            "contain duplicates",
        ),
        (
            {"fit": np.array([0, 1]), "selection": np.array([20])},
            "outside their source pool",
        ),
    ],
)
def test_fixed_split_loader_rejects_missing_duplicate_and_out_of_range(
    tmp_path, fixed, message
):
    from scripts.refine_generic_quintic_compiled_tree_native_gn import (
        load_disjoint_splits,
    )

    points = tmp_path / "points.npz"
    pullbacks = tmp_path / "pullbacks.npy"
    np.savez(
        points,
        X=np.zeros((8, 10), dtype=np.float32),
        weights=np.ones(8),
        omega_squared=np.ones(8),
    )
    np.save(pullbacks, np.zeros((8, 3, 5), dtype=np.complex64))
    with pytest.raises(ValueError, match=message):
        load_disjoint_splits(
            {
                "fit": (points, pullbacks, 2),
                "selection": (points, pullbacks, 1),
            },
            seed=1,
            fixed_indices=fixed,
        )


def test_development_evaluation_cannot_change_joint_worker_winner():
    from scripts.compare_generic_quintic_tree_joint_relaxation import (
        adjudication_winner,
    )

    # Even a development-evaluation report whose own diagnostic gate passes is
    # absent from the winner interface.  Only selection (or real confirmation
    # outside development-only mode) can choose the serialized checkpoint.
    development_evaluation_passes = True
    assert development_evaluation_passes
    assert (
        adjudication_winner(
            selection_passes=False,
            confirmation_passes=None,
        )
        == "control"
    )


def test_one_command_bootstrap_registers_only_the_fixed_action(
    tmp_path, monkeypatch
):
    from scripts import run_quintic_architecture_round1_bridge as command

    raw, _, _, _, _, _ = _contract(tmp_path, monkeypatch)
    store = CampaignStore.initialize(tmp_path / "campaign", raw)
    first = command._ensure_registered_action(
        store,
        candidate_id="edge6-rank39",
    )
    second = command._ensure_registered_action(
        store,
        candidate_id="edge6-rank39",
    )
    assert first == second
    assert first["mutation"] == {
        "kind": "rank",
        "target": "internal-edge",
        "edge": 6,
        "source_dimension": 25,
        "target_dimension": 39,
        "structural_maximum": 100,
        "new_output_real_parameters": 9_800,
    }


def test_wait_for_user_unit_requires_a_proven_success(monkeypatch):
    from types import SimpleNamespace

    from scripts import run_quintic_architecture_round1_bridge as command

    outputs = iter(
        [
            (
                "LoadState=loaded\nActiveState=active\nSubState=running\n"
                "Result=success\nExecMainStatus=0\n"
            ),
            (
                "LoadState=loaded\nActiveState=inactive\nSubState=dead\n"
                "Result=success\nExecMainStatus=0\n"
            ),
        ]
    )
    monkeypatch.setattr(
        command.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout=next(outputs)
        ),
    )
    sleeps = []
    monkeypatch.setattr(command.time, "sleep", sleeps.append)
    state = command.wait_for_user_unit("capacity.service", poll_seconds=17)
    assert state["Result"] == "success"
    assert sleeps == [17]

    monkeypatch.setattr(
        command.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=(
                "LoadState=loaded\nActiveState=failed\nSubState=failed\n"
                "Result=exit-code\nExecMainStatus=1\n"
            ),
        ),
    )
    with pytest.raises(AutoResearchError, match="did not succeed"):
        command.wait_for_user_unit("capacity.service", poll_seconds=17)


def test_cuda_idle_guard_rejects_an_existing_compute_process(monkeypatch):
    from types import SimpleNamespace

    from scripts import run_quintic_architecture_round1_bridge as command

    monkeypatch.setattr(
        command.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=""),
    )
    command.require_cuda_idle()
    monkeypatch.setattr(
        command.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout="1234, python, 15609\n"
        ),
    )
    with pytest.raises(AutoResearchError, match="refusing to share"):
        command.require_cuda_idle()


def test_workflow_guard_verifies_selected_closure_and_output_hashes(tmp_path):
    from scripts import run_quintic_architecture_round1_bridge as command

    run_root = tmp_path / "predecessor"
    workflow = run_root / ".workflow"
    jobs = workflow / "jobs"
    jobs.mkdir(parents=True)
    prepare_output = run_root / "prepared.bin"
    prepare_output.write_bytes(b"sealed preparation")
    output = run_root / "result.bin"
    output.write_bytes(b"sealed predecessor result")
    gate_file = run_root / "gate.json"
    _write_json(gate_file, {"result": {"q0.0000": 2.0}})
    plan_sha256 = "a" * 64
    manifest_sha256 = "d" * 64
    rows = [
        {
            "id": "prepare",
            "phase": "preparation",
            "needs": [],
            "digest": "b" * 64,
        },
        {
            "id": "finish",
            "phase": "promotion-decision",
            "needs": ["prepare"],
            "digest": "c" * 64,
        },
    ]
    _write_json(
        workflow / "plan.lock.json",
        {
            "schema": "gcicy-experiment-plan-lock-v1",
            "campaign_id": "predecessor-campaign",
            "plan_sha256": plan_sha256,
            "manifest_sha256": manifest_sha256,
            "jobs": rows,
        },
    )
    _write_json(
        run_root / ".gcicy-experiment-root",
        {
            "schema": "gcicy-experiment-root-v1",
            "campaign_id": "predecessor-campaign",
            "plan_sha256": plan_sha256,
            "manifest_sha256": manifest_sha256,
        },
    )
    _write_json(
        workflow / "runner.lock",
        {
            "campaign_id": "predecessor-campaign",
            "plan_sha256": plan_sha256,
        },
    )
    _write_json(
        workflow / "cuda_identity.json",
        {
            "schema": "gcicy-experiment-cuda-identity-v1",
            "campaign_id": "predecessor-campaign",
            "plan_sha256": plan_sha256,
            "gpu_slot": "0",
        },
    )
    for row in rows:
        _write_json(
            jobs / f"{row['id']}.json",
            {
                "schema": "gcicy-experiment-job-state-v1",
                "campaign_id": "predecessor-campaign",
                "plan_sha256": plan_sha256,
                "job_id": row["id"],
                "job_digest": row["digest"],
                "phase": row["phase"],
                "status": "succeeded",
                "outputs": (
                    [
                        {
                            "path": str(output),
                            "bytes": output.stat().st_size,
                            "sha256": bridge.sha256_file(output),
                        },
                        {
                            "path": str(gate_file),
                            "bytes": gate_file.stat().st_size,
                            "sha256": bridge.sha256_file(gate_file),
                        },
                    ]
                    if row["id"] == "finish"
                    else [
                        {
                            "path": str(prepare_output),
                            "bytes": prepare_output.stat().st_size,
                            "sha256": bridge.sha256_file(prepare_output),
                        }
                    ]
                ),
                "json_gates": (
                    [
                        {
                            "path": str(gate_file),
                            "field": "result.q0.0000",
                            "comparison": "ge",
                            "target": 1.0,
                            "observed": 2.0,
                            "passed": True,
                        }
                    ]
                    if row["id"] == "finish"
                    else []
                ),
            },
        )
    with command.completed_workflow_guard(
        run_root,
        phases=("promotion-decision",),
        poll_seconds=1,
        expected_campaign_id="predecessor-campaign",
        expected_plan_sha256=plan_sha256,
        expected_job_count=2,
        expected_output_count=3,
        expected_gate_count=1,
        expected_gpu_slot="0",
    ) as report:
        assert report["verified_jobs"] == 2
        assert report["verified_outputs"] == 3

    with pytest.raises(AutoResearchError, match="phases are unknown"):
        with command.completed_workflow_guard(
            run_root,
            phases=("promotion-decision", "typo-phase"),
            poll_seconds=1,
            expected_campaign_id="predecessor-campaign",
            expected_plan_sha256=plan_sha256,
            expected_job_count=2,
            expected_output_count=3,
            expected_gate_count=1,
            expected_gpu_slot="0",
        ):
            pass

    _write_json(gate_file, {"result": {"q0.0000": 0.5}})
    with pytest.raises(AutoResearchError, match="gate no longer passes"):
        with command.completed_workflow_guard(
            run_root,
            phases=("promotion-decision",),
            poll_seconds=1,
            expected_campaign_id="predecessor-campaign",
            expected_plan_sha256=plan_sha256,
            expected_job_count=2,
            expected_output_count=3,
            expected_gate_count=1,
            expected_gpu_slot="0",
        ):
            pass

    _write_json(gate_file, {"result": {"q0.0000": 2.0}})
    output.write_bytes(b"tampered")
    with pytest.raises(AutoResearchError, match="output hash is invalid"):
        with command.completed_workflow_guard(
            run_root,
            phases=("promotion-decision",),
            poll_seconds=1,
            expected_campaign_id="predecessor-campaign",
            expected_plan_sha256=plan_sha256,
            expected_job_count=2,
            expected_output_count=3,
            expected_gate_count=1,
            expected_gpu_slot="0",
        ):
            pass


def test_execute_preflights_before_waiting_and_skips_wait_after_evidence(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from scripts import run_quintic_architecture_round1_bridge as command

    events = []

    class Store:
        evidence_sha256 = None

        def status(self):
            return {
                "rounds": {
                    "1": {
                        "candidates": {
                            "edge6-rank39": {
                                "evidence_sha256": self.evidence_sha256
                            }
                        }
                    }
                }
            }

        def record_search_evidence(self, evidence):
            events.append("record")
            self.evidence_sha256 = "1" * 64

        def adjudicate_round(self, round_number):
            events.append("adjudicate")
            return {"round": round_number}

    store = Store()
    monkeypatch.setattr(
        command.CampaignStore, "initialize", lambda *args, **kwargs: store
    )
    monkeypatch.setattr(command, "_read_object", lambda _: {})
    monkeypatch.setattr(command, "_ensure_registered_action", lambda *a, **k: {})
    monkeypatch.setattr(
        command,
        "prepare_round1_bridge",
        lambda **kwargs: events.append("prepare"),
    )
    monkeypatch.setattr(
        command,
        "wait_for_user_unit",
        lambda *args, **kwargs: events.append("wait"),
    )
    monkeypatch.setattr(
        command, "require_cuda_idle", lambda: events.append("gpu-idle")
    )
    monkeypatch.setattr(
        command, "run_round1_bridge", lambda *args: events.append("run")
    )
    monkeypatch.setattr(
        command,
        "normalize_round1_bridge",
        lambda *args: events.append("normalize") or {},
    )
    args = SimpleNamespace(
        campaign_run_root=tmp_path / "campaign",
        protocol=tmp_path / "protocol.json",
        candidate_id="edge6-rank39",
        output_root=tmp_path / "workers",
        baseline_checkpoint=[(seed, tmp_path / "baseline.pt") for seed in SEEDS],
        device="cuda",
        threads=6,
        train_chunk_size=512,
        feature_batch_size=512,
        eval_batch_size=256,
        early_stopping_evaluations=6,
        wait_for_user_unit="capacity.service",
        wait_for_workflow_run_root=None,
        wait_for_workflow_phase=[],
        expected_workflow_campaign_id=None,
        expected_workflow_plan_sha256=None,
        expected_workflow_job_count=None,
        expected_workflow_output_count=None,
        expected_workflow_gate_count=None,
        expected_workflow_gpu_slot=None,
        gpu_lock_file=None,
        wait_poll_seconds=30,
    )
    args.campaign_run_root.mkdir()
    command.execute_round1(args)
    assert events == [
        "prepare",
        "wait",
        "gpu-idle",
        "run",
        "normalize",
        "record",
        "adjudicate",
    ]

    events.clear()
    command.execute_round1(args)
    assert events == ["adjudicate"]


def test_completed_report_hash_cannot_be_resealed_after_drift(tmp_path, monkeypatch):
    local_dir = tmp_path / "local"
    matched_dir = tmp_path / "matched"
    _write_json(
        local_dir / "report.json",
        {"schema": "generic-quintic-adaptive-direct-blocks-report-v1"},
    )
    _write_json(
        matched_dir / "report.json",
        {"schema": "generic-quintic-tree-joint-relaxation-v1"},
    )
    plan = {
        "seeds": [
            {
                "seed": 1,
                "local_output_dir": str(local_dir),
                "matched_output_dir": str(matched_dir),
            }
        ]
    }
    ledger = {
        "state": "workers-complete",
        "seeds": {
            "1": {
                "local_activate": "complete",
                "local_activate_report_sha256": "0" * 64,
                "matched_relax": "complete",
                "matched_relax_report_sha256": bridge.sha256_file(
                    matched_dir / "report.json"
                ),
            }
        },
    }
    monkeypatch.setattr(bridge, "_verify_plan", lambda _: (plan, ledger))
    monkeypatch.setattr(bridge, "_write_ledger", lambda *args: None)
    with pytest.raises(AutoResearchError, match="sealed ledger hash"):
        bridge.run_round1_bridge(tmp_path)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _statistics(sigma: float, chi: float):
    return {
        "sigma_official_formula": sigma,
        "weighted_rms_abs_residual": chi,
        "min_eigenvalue_weighted_quantiles": {"q0.0000": 0.02},
        "nonpositive_min_eigenvalue": {"count": 0},
    }


def _synthetic_normalization_case(tmp_path, monkeypatch):
    raw, protocol, indices, raw_action, _, _ = _contract(tmp_path, monkeypatch)
    action = control.validate_action(raw_action)
    root = tmp_path / "normalization"
    root.mkdir()
    fixed_path = root / "fixed_indices.npz"
    index_manifest = bridge.materialize_round1_indices(
        indices,
        fixed_path,
        partition_seed=protocol["search_index_plan"]["partition_seed"],
    )
    batch = np.tile(np.arange(1_024, dtype=np.int64), (600, 1))
    batch_source = root / "matched_batch_plan.npy"
    np.save(batch_source, batch, allow_pickle=False)
    batch_manifest = bridge._array_manifest(
        {"batch_plan": batch}, schema=bridge.BATCH_VALUE_SCHEMA
    )
    batch_row = {
        **batch_manifest,
        "path": str(batch_source),
        "file_sha256": bridge.sha256_file(batch_source),
        "seed": 91,
        "steps": 600,
        "batch_size": 1_024,
    }
    roles = {row["role"]: row for row in protocol["search_inputs"]}
    runtime = {
        "device": "cpu",
        "threads": 2,
        "train_chunk_size": 128,
        "feature_batch_size": 128,
        "eval_batch_size": 64,
        "early_stopping_evaluations": 6,
    }
    plan = {
        "candidate_id": "edge6-rank39",
        "action": action,
        "search_indices_sha256": indices["indices_sha256"],
        "index_manifest": index_manifest,
        "input_roles": roles,
        "budgets": protocol["budgets"],
        "thresholds": protocol["thresholds"],
        "runtime": runtime,
        "seeds": [],
    }
    ledger = {
        "schema": bridge.BRIDGE_LEDGER_SCHEMA,
        "plan_sha256": "0" * 64,
        "state": "workers-complete",
        "seeds": {},
        "evidence": None,
    }
    source_dimensions = [10, 10, 10, 10, 10, 25, 25, 25]
    target_dimensions = list(source_dimensions)
    target_dimensions[6] = 39
    expected_input_hashes = bridge._expected_input_hashes(plan)
    for seed in SEEDS:
        seed_root = root / str(seed)
        local_dir = seed_root / "local_activate"
        matched_dir = seed_root / "matched_relax"
        local_dir.mkdir(parents=True)
        matched_dir.mkdir(parents=True)
        local_indices = local_dir / "data_indices.npz"
        fixed = np.load(fixed_path, allow_pickle=False)
        np.savez_compressed(
            local_indices,
            fit=fixed["fit"],
            selection=fixed["selection"],
        )
        matched_indices = matched_dir / "data_indices.npz"
        np.savez_compressed(
            matched_indices,
            fit=fixed["fit"],
            selection=fixed["selection"],
            development_evaluation=fixed["development_evaluation"],
        )
        local_candidate = local_dir / "development_candidate.pt"
        local_candidate.write_bytes(f"local-{seed}".encode())
        control_checkpoint = matched_dir / "control_polished.pt"
        candidate_checkpoint = matched_dir / "candidate_polished.pt"
        control_checkpoint.write_bytes(f"control-{seed}".encode())
        candidate_checkpoint.write_bytes(f"candidate-{seed}".encode())
        observed_batch_path = matched_dir / "batch_plan.npy"
        np.save(observed_batch_path, batch, allow_pickle=False)
        local_config = {
            "initial_checkpoint": roles["baseline-checkpoint"]["path"],
            "train_points": roles["search-native-points"]["path"],
            "train_pullbacks": roles["search-native-pullbacks"]["path"],
            "selection_points": roles["search-native-points"]["path"],
            "selection_pullbacks": roles["search-native-pullbacks"]["path"],
            "confirmation_points": None,
            "confirmation_pullbacks": None,
            "fixed_indices_file": str(fixed_path),
            "output_dir": str(local_dir),
            "target_edge": ["6:39"],
            "rank_activation_scale": 1.0,
            "orthogonalize_new_outputs": True,
            "real_parameter_limit": 10_000,
            "expansion_only": True,
            "defer_block_acceptance": True,
            "development_only": True,
            "sweeps": 2,
            "epochs_per_block": 200,
            "selection_eval_every": 10,
            "early_stopping_evaluations": 6,
            "internal_learning_rate": 3.0e-4,
            "leaf_learning_rate": 3.0e-4,
            "train_size": 30_000,
            "selection_size": 5_000,
            "stochastic_batch_size": 1_024,
            "gradient_clip_norm": 1.0,
            "maximum_selection_tail_relative_degradation": 0.005,
            "train_chunk_size": 128,
            "feature_batch_size": 128,
            "eval_batch_size": 64,
            "seed": seed,
            "data_seed": 2026082301,
            "threads": 2,
            "device": "cpu",
        }
        local_report = {
            "schema": "generic-quintic-adaptive-direct-blocks-report-v1",
            "parent_checkpoint_sha256": bridge.ROUND1_BASELINE_SHA256,
            "configuration": local_config,
            "blocks": [{"name": f"block-{index}"} for index in range(3)],
            "history": [{"rows": [{} for _ in range(200)]} for _ in range(6)],
            "expansion": {
                "source_edge_dimensions": source_dimensions,
                "target_edge_dimensions": target_dimensions,
            },
            "parameter_count": {
                "control_trainable_real_parameters": 121_750,
                "candidate_trainable_real_parameters": 131_550,
                "new_real_parameters": 9_800,
            },
            "embedding_audit": {
                "potential_max_absolute": 1.0e-7,
                "metric_max_relative_frobenius": 1.0e-6,
            },
            "confirmation": None,
            "data": {
                "indices": str(local_indices),
                "fixed_indices_file_sha256": index_manifest["file_sha256"],
                "data_seed": 2026082301,
                "input_sha256": expected_input_hashes,
            },
            "development_candidate": str(local_candidate),
            "development_candidate_sha256": bridge.sha256_file(local_candidate),
        }
        local_report_path = local_dir / "report.json"
        _write_json(local_report_path, local_report)
        matched_config = {
            "control_checkpoint": roles["baseline-checkpoint"]["path"],
            "candidate_checkpoint": str(local_candidate),
            "train_points": roles["search-native-points"]["path"],
            "train_pullbacks": roles["search-native-pullbacks"]["path"],
            "selection_points": roles["search-native-points"]["path"],
            "selection_pullbacks": roles["search-native-pullbacks"]["path"],
            "development_evaluation_points": roles["development-evaluation-points"][
                "path"
            ],
            "development_evaluation_pullbacks": roles[
                "development-evaluation-pullbacks"
            ]["path"],
            "confirmation_points": None,
            "confirmation_pullbacks": None,
            "fixed_indices_file": str(fixed_path),
            "fixed_batch_plan_file": str(batch_source),
            "output_dir": str(matched_dir),
            "development_only": True,
            "development_evaluation": True,
            "steps": 600,
            "learning_rate": 3.0e-6,
            "scheduler": "cosine",
            "eval_every": 25,
            "train_size": 30_000,
            "selection_size": 5_000,
            "development_evaluation_size": 5_000,
            "stochastic_batch_size": 1_024,
            "gradient_clip_norm": 1.0,
            "maximum_selection_tail_relative_degradation": 0.005,
            "train_chunk_size": 128,
            "feature_batch_size": 128,
            "eval_batch_size": 64,
            "seed": seed,
            "data_seed": 2026082301,
            "batch_plan_seed": 91,
            "minimum_relative_sigma_gain": 0.0,
            "minimum_relative_chi_gain": 0.0,
            "threads": 2,
            "device": "cpu",
        }
        matched_report = {
            "schema": "generic-quintic-tree-joint-relaxation-v1",
            "contract": {"precision": "complex64"},
            "configuration": matched_config,
            "source_checkpoint_sha256": {
                "control": bridge.ROUND1_BASELINE_SHA256,
                "candidate": bridge.sha256_file(local_candidate),
            },
            "data": {
                "indices": str(matched_indices),
                "fixed_indices_file_sha256": index_manifest["file_sha256"],
                "data_seed": 2026082301,
                "input_sha256": {
                    **expected_input_hashes,
                    "development_evaluation_points": roles[
                        "development-evaluation-points"
                    ]["sha256"],
                    "development_evaluation_pullbacks": roles[
                        "development-evaluation-pullbacks"
                    ]["sha256"],
                    "confirmation_points": None,
                    "confirmation_pullbacks": None,
                },
            },
            "batch_plan": {
                "path": str(observed_batch_path),
                "fixed_source_sha256": batch_row["file_sha256"],
                "seed": 91,
                "steps": 600,
                "batch_size": 1_024,
            },
            "results": {
                "control": {
                    "checkpoint": str(control_checkpoint),
                    "checkpoint_sha256": bridge.sha256_file(control_checkpoint),
                    "total_real_parameters": 121_750,
                },
                "candidate": {
                    "checkpoint": str(candidate_checkpoint),
                    "checkpoint_sha256": bridge.sha256_file(candidate_checkpoint),
                    "total_real_parameters": 131_550,
                    "new_real_parameters_vs_control": 9_800,
                },
            },
            "development_evaluation": {
                "control": {
                    "statistics": _statistics(0.020, 0.030),
                    "tail": {"q999": 0.08, "cvar99": 0.06},
                },
                "candidate": {
                    "statistics": _statistics(0.019, 0.029),
                    "tail": {"q999": 0.079, "cvar99": 0.059},
                },
                "paired_improvement": {
                    "sigma": {"ci95_low": 1.0e-5},
                    "e2": {"ci95_low": 1.0e-6},
                },
                "passes": True,
                "selection_or_checkpoint_role": "none",
            },
            "confirmation": None,
            "confirmation_passes": None,
            "winner": "candidate",
        }
        matched_report_path = matched_dir / "report.json"
        _write_json(matched_report_path, matched_report)
        plan["seeds"].append(
            {
                "seed": seed,
                "optimizer_seed": seed,
                "data_seed": 2026082301,
                "batch_plan_seed": 91,
                "baseline_checkpoint": {
                    "path": roles["baseline-checkpoint"]["path"],
                    "sha256": bridge.ROUND1_BASELINE_SHA256,
                },
                "batch_plan": batch_row,
                "local_output_dir": str(local_dir),
                "matched_output_dir": str(matched_dir),
            }
        )
        ledger["seeds"][str(seed)] = {
            "local_activate": "complete",
            "matched_relax": "complete",
            "local_activate_report_sha256": bridge.sha256_file(local_report_path),
            "matched_relax_report_sha256": bridge.sha256_file(matched_report_path),
        }
    monkeypatch.setattr(bridge, "_verify_plan", lambda _: (plan, ledger))
    return root, plan, ledger, protocol, action


def test_synthetic_normalization_is_accepted_by_controller(tmp_path, monkeypatch):
    root, plan, _, protocol, action = _synthetic_normalization_case(
        tmp_path, monkeypatch
    )
    evidence = bridge.normalize_round1_bridge(root)
    normalized = validate_search_evidence(
        evidence,
        action=action,
        protocol=protocol,
        expected_indices_sha256=plan["search_indices_sha256"],
    )
    assert len(normalized["seeds"]) == 3
    assert all(row["equivalence"]["passed"] for row in normalized["seeds"])


@pytest.mark.parametrize("tamper", ["indices", "batch", "config", "report-hash"])
def test_normalization_fails_closed_on_bound_artifact_drift(
    tmp_path, monkeypatch, tamper
):
    root, plan, ledger, _, _ = _synthetic_normalization_case(tmp_path, monkeypatch)
    seed_row = plan["seeds"][0]
    local_dir = Path(seed_row["local_output_dir"])
    matched_dir = Path(seed_row["matched_output_dir"])
    if tamper == "indices":
        path = local_dir / "data_indices.npz"
        arrays = np.load(path, allow_pickle=False)
        fit = np.asarray(arrays["fit"]).copy()
        fit[0] += 1
        np.savez_compressed(path, fit=fit, selection=arrays["selection"])
        pattern = "indices differ"
    elif tamper == "batch":
        path = matched_dir / "batch_plan.npy"
        values = np.load(path, allow_pickle=False)
        values[0, 0] = 2_000
        np.save(path, values, allow_pickle=False)
        pattern = "batch plan differs"
    elif tamper == "config":
        path = local_dir / "report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        report["configuration"]["data_seed"] += 1
        _write_json(path, report)
        ledger["seeds"][str(seed_row["seed"])]["local_activate_report_sha256"] = (
            bridge.sha256_file(path)
        )
        pattern = "configuration mismatch for data_seed"
    else:
        path = local_dir / "report.json"
        path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
        pattern = "report hash differs"
    with pytest.raises(AutoResearchError, match=pattern):
        bridge.normalize_round1_bridge(root)
