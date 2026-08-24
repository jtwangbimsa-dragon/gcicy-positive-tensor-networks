from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest
import numpy as np

import gcicy_metric.pipeline.x21_scientific_bridge as bridge
import gcicy_metric.pipeline.x21_fresh_development_pool as fresh
from gcicy_metric.pipeline.x21_auto_research import (
    BASELINE_COMPLETION_SCHEMA,
    FAMILY_SCHEMA,
    RECOVERY_EVIDENCE_SCHEMA,
    X21CampaignStore,
    coefficient_real_parameters,
    digest_value,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "experiments" / "protocols" / "x21_auto_research_v1.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _contract(protocol: dict, seed: int) -> dict:
    return next(
        row for row in protocol["baseline_provenance_contracts"] if row["seed"] == seed
    )


def _completion(
    protocol: dict, seed: int, model_sha256: str, checkpoint_sha256: str
) -> dict:
    contract = _contract(protocol, seed)
    return {
        "schema": BASELINE_COMPLETION_SCHEMA,
        "seed": seed,
        "source_campaign_id": contract["source_campaign_id"],
        "source_plan_sha256": contract["source_plan_sha256"],
        "source_job_id": contract["source_job_id"],
        "source_job_digest": contract["source_job_digest"],
        "terminal_status": "validation_plateau",
        "native_exit_code": 0,
        "output_model_sha256": model_sha256,
        "output_checkpoint_sha256": checkpoint_sha256,
        "output_summary_sha256": hashlib.sha256(f"summary:{seed}".encode()).hexdigest(),
    }


def _recovery(protocol: dict, model_sha256: str, checkpoint_sha256: str) -> dict:
    contract = _contract(protocol, 8660002)
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
        "output_model_sha256": model_sha256,
        "output_checkpoint_sha256": checkpoint_sha256,
        "output_summary_sha256": hashlib.sha256(b"recovery-summary").hexdigest(),
    }


@pytest.fixture
def prepared_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    inputs = tmp_path / "immutable"
    inputs.mkdir()
    source = inputs / "source.npz"
    train = inputs / "train_pool.npz"
    selection = inputs / "selection_pool.npz"
    source.write_bytes(b"source")
    train.write_bytes(b"train")
    selection.write_bytes(b"selection")
    protocol["data_contract"]["source_artifact_sha256"] = _sha(source)
    protocol["data_contract"]["train_pool_sha256"] = _sha(train)
    protocol["data_contract"]["selection_pool_sha256"] = _sha(selection)

    parent_rows = []
    family_rows = []
    for seed, token, kind in (
        (8660001, b"parent-one", "existing-complete"),
        (8660002, b"parent-two", "exact-recovery"),
        (8660003, b"parent-three", "fresh-completion"),
    ):
        model = inputs / f"parent_{seed}.pt"
        model.write_bytes(token)
        checkpoint = inputs / f"checkpoint_{seed}.pt"
        checkpoint.write_bytes(b"optimizer-rng:" + token)
        model_sha256 = _sha(model)
        checkpoint_sha256 = _sha(checkpoint)
        certificate = (
            _recovery(protocol, model_sha256, checkpoint_sha256)
            if seed == 8660002
            else _completion(protocol, seed, model_sha256, checkpoint_sha256)
        )
        certificate_path = inputs / f"certificate_{seed}.json"
        _write_json(certificate_path, certificate)
        certificate_value_sha256 = digest_value(certificate)
        parent_rows.append(
            {
                "seed": seed,
                "model_path": str(model),
                "checkpoint_path": str(checkpoint),
                "certificate_path": str(certificate_path),
            }
        )
        family_rows.append(
            {
                "seed": seed,
                "checkpoint_sha256": model_sha256,
                "provenance": {
                    "kind": kind,
                    "certificate_sha256": certificate_value_sha256,
                    "certificate": certificate,
                },
            }
        )
    campaign_root = tmp_path / "campaign"
    store = X21CampaignStore.initialize(campaign_root, protocol)
    store.register_baseline(
        {
            "schema": FAMILY_SCHEMA,
            "model_metadata": {
                "k": 20,
                "bond_dimension": 14,
                "train_physical_dictionary": False,
                "trainable_real_parameter_count": coefficient_real_parameters(20, 14),
                "precision": "complex64",
            },
            "models": family_rows,
        }
    )
    source_contract = {
        "commit": "a" * 40,
        "branch": "exp/test-x21-bridge",
        "tracked_and_untracked_worktree": "clean",
        "dependencies": {
            relative: _sha(ROOT / relative) for relative in bridge.SOURCE_DEPENDENCIES
        },
    }
    monkeypatch.setattr(bridge, "_git_source_contract", lambda _root: source_contract)
    return {
        "campaign_root": campaign_root,
        "store": store,
        "source": source,
        "train": train,
        "selection": selection,
        "bindings": {"schema": bridge.PARENT_BINDINGS_SCHEMA, "models": parent_rows},
        "output_root": tmp_path / "bridge",
    }


def _prepare(inputs: dict) -> dict:
    return bridge.prepare_optimizer_path_bridge(
        campaign_run_root=inputs["campaign_root"],
        output_root=inputs["output_root"],
        parent_bindings=inputs["bindings"],
        source_artifact=inputs["source"],
        train_common_pool=inputs["train"],
        selection_common_pool=inputs["selection"],
        repository_root=ROOT,
        device="cpu",
        workers=1,
    )


def _value_after(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def _successful_worker(command: list[str], **kwargs) -> SimpleNamespace:
    assert kwargs["shell"] is False
    assert command[1] == str((ROOT / bridge.TRAINER_RELATIVE).resolve())
    joined = "\0".join(command).lower()
    assert "confirmation" not in joined and "blind" not in joined
    model = Path(_value_after(command, "--out"))
    summary_path = Path(_value_after(command, "--summary"))
    checkpoint = Path(_value_after(command, "--checkpoint"))
    model.parent.mkdir(parents=True, exist_ok=True)
    seed = int(_value_after(command, "--torch-seed"))
    learning_rate = float(_value_after(command, "--learning-rate"))
    model.write_bytes(f"model:{seed}:{learning_rate}".encode())
    checkpoint.write_bytes(f"checkpoint:{seed}:{learning_rate}".encode())
    summary = {
        "schema": "type11-positive-tensor-network-training-v1",
        "adapter": "p5p1_type21_k3_1223",
        "training_mode": "teacher_free_geometric",
        "teacher_artifact_sha256": None,
        "model_sha256": _sha(model),
        "initialization": {
            "kind": "continued_saved_model",
            "model_sha256": _sha(Path(_value_after(command, "--initial-model"))),
        },
        "source_artifact_sha256": _sha(
            Path(_value_after(command, "--source-artifact"))
        ),
        "train": {
            "points": bridge.TRAIN_POINTS,
            "seed": bridge.TRAIN_SEED,
            "common_pool_sha256": _sha(
                Path(_value_after(command, "--train-common-pool"))
            ),
        },
        "validation": {
            "points": bridge.SELECTION_POINTS,
            "seed": bridge.SELECTION_SEED,
            "common_pool_sha256": _sha(
                Path(_value_after(command, "--validation-common-pool"))
            ),
        },
        "torch_seed": seed,
        "model_seed": bridge.MODEL_SEED,
        "site_count": 20,
        "bond_dimension": 14,
        "trainable_real_parameter_count": 860552,
        "trainable_physical_dictionary": False,
        "positive_floor": 1.0e-14,
        "initialization_noise": 0.0,
        "device": _value_after(command, "--device"),
        "precision": "complex64",
        "sampling_cluster_size": bridge.SAMPLING_CLUSTER_SIZE,
        "epochs": bridge.EPOCHS,
        "batch_size": bridge.BATCH_SIZE,
        "learning_rate": learning_rate,
        "gradient_clip_norm": 2.0,
        "loss_weights": {
            "potential": 0.0,
            "metric": 0.0,
            "log_energy": 1.0,
            "ma": 1.0,
            "tail": 0.1,
        },
        "tail_loss": {
            "tail_fraction": 0.02,
            "ratio_threshold": 1.5,
            "smooth_temperature": 0.05,
            "point_loss": "squared smooth positive log-ratio excess",
        },
        "early_stopping": {"patience": 0, "minimum_relative_improvement": 0.0},
        "fixed_log_kappa_source": "continued_saved_model",
        "history": [{"epoch": epoch} for epoch in range(bridge.EPOCHS + 1)],
        "termination_reason": "completed_requested_epochs",
        "checkpoint": {
            "sha256": _sha(checkpoint),
            "last_completed_validation_epoch": bridge.EPOCHS,
            "resume_kind": (
                "crash_recovery"
                if "--resume-checkpoint" in command
                else "fresh_training"
            ),
        },
    }
    _write_json(summary_path, summary)
    return SimpleNamespace(returncode=0, stdout="completed\n", stderr="")


@pytest.fixture
def fresh_prepared_inputs(
    prepared_inputs: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict:
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    fresh_root = tmp_path / "fresh_selection_manager"
    fresh_source = {
        "repository_root": str(ROOT),
        "commit": "a" * 40,
        "branch": "exp/test-x21-bridge",
        "tracked_and_untracked_worktree": "clean",
        "files": {relative: "b" * 64 for relative in fresh.SOURCE_DEPENDENCIES},
    }
    monkeypatch.setattr(
        fresh, "_git_source_contract", lambda _root: copy.deepcopy(fresh_source)
    )
    fresh_plan = fresh.prepare_fresh_development_pool(
        output_root=fresh_root,
        frozen_root=frozen,
        base_protocol=PROTOCOL_PATH,
        repository_root=ROOT,
        workers=2,
        backend="thread",
    )

    def generate_selection_pool(command: list[str], **kwargs) -> SimpleNamespace:
        assert kwargs["shell"] is False
        pool_path = Path(command[command.index("--out") + 1])
        manifest_path = Path(command[command.index("--manifest") + 1])
        generation = fresh_plan["generation"]
        metadata = {
            "schema": "gcicy-common-point-pool-v1",
            "schema_version": 1,
            "adapter": fresh.ADAPTER,
            "adapter_version": "1",
            "model_seed": fresh.MODEL_SEED,
            "exact_model": True,
            "split": fresh.SPLIT,
            "point_count": fresh.POINTS,
            "sampling_seed": fresh.SAMPLING_SEED,
            "cluster_size": fresh.SAMPLING_CLUSTER_SIZE,
            "cluster_count": fresh.POINTS // fresh.SAMPLING_CLUSTER_SIZE,
            "sampling": {
                "workers": generation["workers"],
                "backend": generation["backend"],
                "sampling_options": {},
                "shards": [
                    {"worker": index, "points": count, "seed": seed}
                    for index, (count, seed) in enumerate(
                        fresh.parallel_sampling_jobs(
                            fresh.POINTS,
                            fresh.SAMPLING_SEED,
                            generation["workers"],
                            fresh.SAMPLING_CLUSTER_SIZE,
                        )
                    )
                ],
                "seconds": 0.01,
            },
            "extra": {
                "git_commit": fresh_plan["source_revision"]["commit"],
                "git_status_porcelain": "",
                "generator": fresh_plan["generator"]["path"],
            },
        }
        np.savez_compressed(
            pool_path,
            common_point_pool_schema_version=np.asarray(1, dtype=np.int64),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        arrays = {
            "common_point_pool_schema_version": {"shape": [], "dtype": "int64"},
            **{
                name: {"shape": [fresh.POINTS], "dtype": "float64"}
                for name in (
                    "importance_log_weights",
                    "importance_weights",
                    "sampling_cluster_ids",
                    "baseline_metrics",
                    "holomorphic_volume_log_density",
                )
            },
        }
        manifest = {
            **metadata,
            "pool_path": str(pool_path.resolve()),
            "pool_sha256": _sha(pool_path),
            "arrays": arrays,
        }
        _write_json(manifest_path, manifest)
        return SimpleNamespace(returncode=0, stdout="completed\n", stderr="")

    monkeypatch.setattr(fresh.subprocess, "run", generate_selection_pool)
    materialized = fresh.materialize_fresh_development_pool(fresh_root)
    binding = materialized["binding"]
    protocol = materialized["protocol_v2"]
    protocol["data_contract"]["source_artifact_sha256"] = _sha(
        prepared_inputs["source"]
    )
    protocol["data_contract"]["train_pool_sha256"] = _sha(prepared_inputs["train"])
    protocol["data_contract"]["selection_pool_sha256"] = _sha(
        prepared_inputs["selection"]
    )
    family_rows = []
    for row in prepared_inputs["bindings"]["models"]:
        seed = row["seed"]
        certificate = json.loads(
            Path(row["certificate_path"]).read_text(encoding="utf-8")
        )
        family_rows.append(
            {
                "seed": seed,
                "checkpoint_sha256": _sha(Path(row["model_path"])),
                "provenance": {
                    "kind": (
                        "exact-recovery"
                        if seed == 8660002
                        else (
                            "existing-complete"
                            if seed == 8660001
                            else "fresh-completion"
                        )
                    ),
                    "certificate_sha256": digest_value(certificate),
                    "certificate": certificate,
                },
            }
        )
    campaign_root = tmp_path / "fresh_campaign"
    store = X21CampaignStore.initialize(campaign_root, protocol)
    store.register_baseline(
        {
            "schema": FAMILY_SCHEMA,
            "model_metadata": {
                "k": 20,
                "bond_dimension": 14,
                "train_physical_dictionary": False,
                "trainable_real_parameter_count": coefficient_real_parameters(20, 14),
                "precision": "complex64",
            },
            "models": family_rows,
        }
    )
    host_identity_sha256 = "c" * 64
    host_certificates = {}
    for row in prepared_inputs["bindings"]["models"]:
        seed = row["seed"]
        certificate_path = tmp_path / "host_certificates" / f"seed_{seed}.json"
        _write_json(
            certificate_path,
            {
                "seed": seed,
                "checkpoint_sha256": _sha(Path(row["model_path"])),
                "host_identity_sha256": host_identity_sha256,
                "source_commit": "a" * 40,
                "authorized": True,
            },
        )
        host_certificates[seed] = certificate_path

    monkeypatch.setattr(bridge, "_host_identity_sha256", lambda: host_identity_sha256)

    def host_certificate_identity(
        *,
        certificate_path: Path,
        parent: dict,
        host_identity_sha256: str,
        source_commit: str,
        now=None,
    ) -> dict:
        resolved = certificate_path.resolve()
        assert resolved.is_file()
        certificate_value = json.loads(resolved.read_text(encoding="utf-8"))
        if certificate_value["seed"] != parent["seed"]:
            raise bridge.X21ScientificBridgeError(
                "host stability certificate rejected: checkpoint seed changed"
            )
        if certificate_value["checkpoint_sha256"] != parent["model"]["sha256"]:
            raise bridge.X21ScientificBridgeError(
                "host stability certificate rejected: checkpoint binding changed"
            )
        if certificate_value["host_identity_sha256"] != host_identity_sha256:
            raise bridge.X21ScientificBridgeError(
                "host stability certificate rejected: host binding changed"
            )
        if certificate_value["source_commit"] != source_commit:
            raise bridge.X21ScientificBridgeError(
                "host stability certificate rejected: source binding changed"
            )
        return {
            "path": str(resolved),
            "file_sha256": _sha(resolved),
            "certificate_sha256": hashlib.sha256(
                f"host:{resolved}".encode()
            ).hexdigest(),
            "issued_utc": "2026-08-24T00:00:00+00:00",
            "valid_until_utc": "2026-08-24T00:30:00+00:00",
            "host_identity_sha256": host_identity_sha256,
            "source_commit": source_commit,
            "checkpoint_sha256": parent["model"]["sha256"],
            "scope": "scientific",
            "scientific_authorized": True,
        }

    monkeypatch.setattr(bridge, "_host_certificate_identity", host_certificate_identity)
    return {
        **prepared_inputs,
        "campaign_root": campaign_root,
        "store": store,
        "output_root": tmp_path / "fresh_bridge",
        "fresh_binding": binding,
        "host_certificates": host_certificates,
    }


def _prepare_fresh(inputs: dict) -> dict:
    return bridge.prepare_optimizer_path_bridge(
        campaign_run_root=inputs["campaign_root"],
        output_root=inputs["output_root"],
        parent_bindings=inputs["bindings"],
        source_artifact=inputs["source"],
        train_common_pool=inputs["train"],
        selection_common_pool=inputs["selection"],
        repository_root=ROOT,
        fresh_development_binding=inputs["fresh_binding"],
        device="cuda",
        workers=1,
    )


def _successful_full_worker(command: list[str], **kwargs) -> SimpleNamespace:
    script = Path(command[1]).name
    if script == Path(bridge.TRAINER_RELATIVE).name:
        return _successful_worker(command, **kwargs)
    assert kwargs["shell"] is False
    joined = "\0".join(command).lower()
    assert "confirmation" not in joined and "blind" not in joined
    if script == Path(bridge.AUDIT_RELATIVE).name:
        arrays_path = Path(_value_after(command, "--arrays-out"))
        report_path = Path(_value_after(command, "--out"))
        arrays_path.parent.mkdir(parents=True, exist_ok=True)
        with arrays_path.open("wb") as handle:
            np.savez_compressed(handle, marker=np.asarray([1], dtype=np.int64))
        report = {
            "schema": "type11-positive-tensor-network-blind-audit-v1",
            "model_sha256": _sha(Path(_value_after(command, "--model"))),
            "source_artifact": str(
                Path(_value_after(command, "--source-artifact")).resolve()
            ),
            "model_seed": int(_value_after(command, "--model-seed")),
            "seed": int(_value_after(command, "--seed")),
            "points": int(_value_after(command, "--points")),
            "sampling_cluster_size": int(
                _value_after(command, "--sampling-cluster-size")
            ),
            "common_pool_sha256": _sha(Path(_value_after(command, "--common-pool"))),
            "precision": "complex64",
            "point_arrays": {"sha256": _sha(arrays_path)},
        }
        _write_json(report_path, report)
    elif script == Path(bridge.TAIL_RELATIVE).name:
        arrays_path = Path(_value_after(command, "--arrays"))
        label = _value_after(command, "--label")
        scale = 1.0 if label == "control" else 0.99
        report = {
            "schema": "gcicy-bilateral-tail-array-evaluation-v1",
            "models": {
                label: {
                    "array_artifact_sha256": _sha(arrays_path),
                    "metrics": {
                        "sigma": 0.01 * scale,
                        "chi": 0.02 * scale,
                        "absolute_log_ratio_q999": 0.05,
                        "absolute_log_ratio_cvar_1pct": 0.04,
                        "minimum_metric_eigenvalue": 0.01,
                        "nonpositive_metric_count": 0,
                    },
                }
            },
        }
        _write_json(Path(_value_after(command, "--out")), report)
    elif script == Path(bridge.BOOTSTRAP_RELATIVE).name:
        baseline = Path(_value_after(command, "--baseline-arrays"))
        candidate = Path(_value_after(command, "--candidate-arrays"))
        comparison = {
            "direction": "positive_is_candidate_improvement",
            "bootstrap_95pct_confidence_interval": [1.0e-5, 2.0e-5],
        }
        report = {
            "schema": "gcicy-paired-fibre-cluster-bootstrap-v1",
            "bootstrap_replicates": int(_value_after(command, "--replicates")),
            "bootstrap_seed": int(_value_after(command, "--seed")),
            "point_count": bridge.FRESH_DEVELOPMENT_POINTS,
            "baseline_artifact_sha256": _sha(baseline),
            "candidate_artifact_sha256": _sha(candidate),
            "comparisons": {"sigma": comparison, "chi": comparison},
        }
        _write_json(Path(_value_after(command, "--out")), report)
    else:  # pragma: no cover - whitelist and dispatch must stay synchronized.
        raise AssertionError(f"unexpected worker: {script}")
    return SimpleNamespace(returncode=0, stdout="completed\n", stderr="")


def test_prepare_selects_first_action_and_binds_each_parent_certificate(
    prepared_inputs,
):
    plan = _prepare(prepared_inputs)
    assert plan["action"]["mutation"] == {
        "kind": "optimizer_path",
        "field": "learning_rate",
        "control_value": 3.0e-5,
        "candidate_value": 1.5e-5,
    }
    assert plan["budget"]["optimizer_updates"] == 2304
    assert {row["seed"] for row in plan["seeds"]} == {8660001, 8660002, 8660003}
    assert (
        len({row["parent"]["certificate"]["value_sha256"] for row in plan["seeds"]})
        == 3
    )
    for row in plan["seeds"]:
        assert row["parent"]["model"]["sha256"] != row["parent"]["checkpoint"]["sha256"]
        assert (
            _value_after(row["arms"]["control"]["base_command"], "--initial-model")
            == row["parent"]["model"]["path"]
        )
        assert (
            row["arms"]["control"]["base_command"]
            != row["arms"]["candidate"]["base_command"]
        )
        assert (
            row["batch_schedule_contract"]["optimizer_updates"]
            == bridge.OPTIMIZER_UPDATES
        )
    assert _prepare(prepared_inputs) == plan
    status = prepared_inputs["store"].status()
    assert status["rounds"]["1"]["status"] == "registered"
    assert status["leaderboard"] == []


def test_prepare_rejects_confirmation_named_pool_before_execution(prepared_inputs):
    confirmation = prepared_inputs["train"].with_name("X21_confirmation_pool.npz")
    confirmation.write_bytes(prepared_inputs["train"].read_bytes())
    with pytest.raises(bridge.X21ScientificBridgeError, match="forbidden"):
        bridge.prepare_optimizer_path_bridge(
            campaign_run_root=prepared_inputs["campaign_root"],
            output_root=prepared_inputs["output_root"],
            parent_bindings=prepared_inputs["bindings"],
            source_artifact=prepared_inputs["source"],
            train_common_pool=confirmation,
            selection_common_pool=prepared_inputs["selection"],
            repository_root=ROOT,
            device="cpu",
        )


def test_parent_certificate_drift_is_rejected(prepared_inputs):
    certificate = Path(prepared_inputs["bindings"]["models"][0]["certificate_path"])
    value = json.loads(certificate.read_text(encoding="utf-8"))
    value["output_summary_sha256"] = "f" * 64
    _write_json(certificate, value)
    with pytest.raises(bridge.X21ScientificBridgeError, match="certificate value hash"):
        _prepare(prepared_inputs)


def test_parent_checkpoint_drift_is_rejected(prepared_inputs):
    checkpoint = Path(prepared_inputs["bindings"]["models"][0]["checkpoint_path"])
    checkpoint.write_bytes(b"drifted-recovery-state")
    with pytest.raises(
        bridge.X21ScientificBridgeError,
        match="certificate does not identify its recovery checkpoint",
    ):
        _prepare(prepared_inputs)


def test_technical_failure_is_resumable_and_never_scientific_rejection(
    prepared_inputs, monkeypatch: pytest.MonkeyPatch
):
    _prepare(prepared_inputs)
    calls = 0

    def fail_once(command, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            assert kwargs["shell"] is False
            checkpoint = Path(_value_after(command, "--checkpoint"))
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b"sealed-partial-checkpoint")
            return SimpleNamespace(returncode=11, stdout="", stderr="native fault\n")
        if calls == 2:
            assert _value_after(command, "--resume-checkpoint") == _value_after(
                command, "--checkpoint"
            )
        return _successful_worker(command, **kwargs)

    monkeypatch.setattr(bridge.subprocess, "run", fail_once)
    with pytest.raises(
        bridge.X21BridgeTechnicalError, match="not a scientific rejection"
    ):
        bridge.run_optimizer_path_bridge(prepared_inputs["output_root"])
    status = bridge.bridge_status(prepared_inputs["output_root"])
    assert status["operational_state"] == "resumable-after-technical-failure"
    assert status["scientific_outcome"] is None
    assert status["scientific_rejection"] is False
    controller = prepared_inputs["store"].status()
    assert controller["rounds"]["1"]["status"] == "registered"
    assert controller["rounds"]["1"]["evidence_sha256"] is None

    completed = bridge.run_optimizer_path_bridge(prepared_inputs["output_root"])
    assert completed["operational_state"] == "training-complete"
    assert all(
        arm["state"] == "complete"
        for seed in completed["seeds"]
        for arm in seed["arms"].values()
    )
    assert calls == 7


def test_normalize_and_adjudicate_fail_closed_without_confirmation(
    prepared_inputs, monkeypatch: pytest.MonkeyPatch
):
    _prepare(prepared_inputs)
    monkeypatch.setattr(bridge.subprocess, "run", _successful_worker)
    bridge.run_optimizer_path_bridge(prepared_inputs["output_root"])
    blocker = bridge.normalize_optimizer_path_bridge(prepared_inputs["output_root"])
    assert blocker["code"] == bridge.BLOCKER_CODE
    assert blocker["scientific_evidence_emitted"] is False
    assert blocker["scientific_outcome"] is None
    assert blocker["scientific_rejection"] is False
    with pytest.raises(bridge.X21ScientificBlocked, match="no evidence or rejection"):
        bridge.adjudicate_optimizer_path_bridge(prepared_inputs["output_root"])
    controller = prepared_inputs["store"].status()
    assert controller["rounds"]["1"]["status"] == "registered"
    assert controller["rounds"]["1"]["evidence_sha256"] is None
    assert controller["leaderboard"] == []


def test_receipt_artifact_drift_is_detected(
    prepared_inputs, monkeypatch: pytest.MonkeyPatch
):
    plan = _prepare(prepared_inputs)
    monkeypatch.setattr(bridge.subprocess, "run", _successful_worker)
    bridge.run_optimizer_path_bridge(prepared_inputs["output_root"])
    model = Path(plan["seeds"][0]["arms"]["control"]["output_dir"]) / "model.pt"
    model.write_bytes(b"tampered")
    with pytest.raises(bridge.X21ScientificBridgeError, match="drift"):
        bridge.bridge_status(prepared_inputs["output_root"])


def test_historical_protocol_cannot_be_overridden_with_fresh_binding(prepared_inputs):
    fake = {
        "schema": bridge.FRESH_DEVELOPMENT_BINDING_SCHEMA,
        "path": str(prepared_inputs["selection"]),
        "sha256": _sha(prepared_inputs["selection"]),
        "split": "selection",
        "seed": bridge.FRESH_DEVELOPMENT_SEED,
        "points": bridge.FRESH_DEVELOPMENT_POINTS,
        "sampling_cluster_size": bridge.SAMPLING_CLUSTER_SIZE,
        "generation_receipt_path": str(
            prepared_inputs["bindings"]["models"][0]["certificate_path"]
        ),
        "generation_receipt_sha256": _sha(
            Path(prepared_inputs["bindings"]["models"][0]["certificate_path"])
        ),
    }
    with pytest.raises(bridge.X21ScientificBridgeError, match="cannot be upgraded"):
        bridge.prepare_optimizer_path_bridge(
            campaign_run_root=prepared_inputs["campaign_root"],
            output_root=prepared_inputs["output_root"],
            parent_bindings=prepared_inputs["bindings"],
            source_artifact=prepared_inputs["source"],
            train_common_pool=prepared_inputs["train"],
            selection_common_pool=prepared_inputs["selection"],
            repository_root=ROOT,
            fresh_development_binding=fake,
            device="cpu",
        )


def test_cuda_run_requires_exact_pending_seed_certificate_mapping_and_binding(
    fresh_prepared_inputs,
):
    plan = _prepare_fresh(fresh_prepared_inputs)
    requirements = {
        row["seed"]: row["host_stability_requirement"] for row in plan["seeds"]
    }
    assert all(
        row["checkpoint"]["sha256"]
        == next(
            seed_row["parent"]["model"]["sha256"]
            for seed_row in plan["seeds"]
            if seed_row["seed"] == seed
        )
        for seed, row in requirements.items()
    )
    ledger = json.loads(
        (fresh_prepared_inputs["output_root"] / "ledger.json").read_text(
            encoding="utf-8"
        )
    )
    assert {
        int(seed): row for seed, row in ledger["host_stability_requirements"].items()
    } == requirements

    incomplete = dict(fresh_prepared_inputs["host_certificates"])
    incomplete.pop(8660003)
    with pytest.raises(bridge.X21ScientificBridgeError, match="pending seeds"):
        bridge.run_optimizer_path_bridge(
            fresh_prepared_inputs["output_root"],
            host_stability_certificates=incomplete,
        )

    swapped = dict(fresh_prepared_inputs["host_certificates"])
    swapped[8660001], swapped[8660002] = swapped[8660002], swapped[8660001]
    with pytest.raises(bridge.X21ScientificBridgeError, match="checkpoint seed"):
        bridge.run_optimizer_path_bridge(
            fresh_prepared_inputs["output_root"],
            host_stability_certificates=swapped,
        )
    controller = fresh_prepared_inputs["store"].status()
    assert controller["rounds"]["1"]["status"] == "registered"
    assert controller["rounds"]["1"]["evidence_sha256"] is None


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("checkpoint_sha256", "d" * 64, "checkpoint binding"),
        ("host_identity_sha256", "e" * 64, "host binding"),
        ("source_commit", "f" * 40, "source binding"),
    ),
)
def test_runtime_certificate_wrong_parent_host_or_source_is_rejected(
    fresh_prepared_inputs,
    field: str,
    value: str,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
):
    _prepare_fresh(fresh_prepared_inputs)
    certificates = dict(fresh_prepared_inputs["host_certificates"])
    wrong = certificates[8660001].with_name(f"wrong_{field}.json")
    payload = json.loads(certificates[8660001].read_text(encoding="utf-8"))
    payload[field] = value
    _write_json(wrong, payload)
    certificates[8660001] = wrong

    def forbidden_worker(*_args, **_kwargs):
        raise AssertionError("bad authorization must stop before child launch")

    monkeypatch.setattr(bridge.subprocess, "run", forbidden_worker)
    with pytest.raises(bridge.X21ScientificBridgeError, match=message):
        bridge.run_optimizer_path_bridge(
            fresh_prepared_inputs["output_root"],
            host_stability_certificates=certificates,
        )
    assert not (fresh_prepared_inputs["output_root"] / "host_authorizations").exists()


def test_fresh_selection_protocol_runs_paired_evidence_and_adjudicates(
    fresh_prepared_inputs, monkeypatch: pytest.MonkeyPatch
):
    plan = _prepare_fresh(fresh_prepared_inputs)
    assert plan["fresh_development_binding"]["generation"]["split"] == "selection"
    assert (
        plan["fresh_development_binding"]["pool"]["sha256"]
        != bridge.HISTORICAL_CONFIRMATION_SHA256
    )
    runtime_certificates = {}
    for seed, source in fresh_prepared_inputs["host_certificates"].items():
        renewed = source.with_name(f"renewed_{source.name}")
        renewed.write_bytes(source.read_bytes())
        runtime_certificates[seed] = renewed

    lock_state = {"held": False}
    runtime_validations = 0
    worker_calls = 0
    certificate_identity = bridge._host_certificate_identity

    @contextmanager
    def locked_gpu(lock_root: Path, gpu_id: str, *, timeout_seconds: float):
        assert lock_root == bridge.GPU_LOCK_ROOT
        assert gpu_id == bridge.GPU_ID
        assert timeout_seconds == -1
        assert lock_state["held"] is False
        lock_state["held"] = True
        try:
            yield lock_root / "gcicy-tn-gpu-0.lock"
        finally:
            lock_state["held"] = False

    def checked_certificate_identity(**kwargs):
        nonlocal runtime_validations
        if kwargs.get("now") is None:
            assert lock_state["held"] is True
            runtime_validations += 1
        return certificate_identity(**kwargs)

    def checked_worker(command: list[str], **kwargs) -> SimpleNamespace:
        nonlocal worker_calls
        assert lock_state["held"] is True
        worker_calls += 1
        return _successful_full_worker(command, **kwargs)

    monkeypatch.setattr(bridge, "gpu_lock", locked_gpu)
    monkeypatch.setattr(
        bridge, "_host_certificate_identity", checked_certificate_identity
    )
    monkeypatch.setattr(bridge.subprocess, "run", checked_worker)
    status = bridge.run_optimizer_path_bridge(
        fresh_prepared_inputs["output_root"],
        host_stability_certificates=runtime_certificates,
    )
    assert lock_state["held"] is False
    assert worker_calls == 21
    assert runtime_validations == worker_calls
    assert status["operational_state"] == "scientific-evaluation-complete"
    assert status["scientific_state"] == "ready-for-normalization"
    authorization_files = sorted(
        (fresh_prepared_inputs["output_root"] / "host_authorizations").glob("*/*.json")
    )
    assert len(authorization_files) == 3
    authorization_hashes = {path: _sha(path) for path in authorization_files}
    replay = bridge.run_optimizer_path_bridge(
        fresh_prepared_inputs["output_root"], host_stability_certificates={}
    )
    assert replay == status
    replay_barrier = threading.Barrier(2)

    def concurrent_replay() -> dict:
        replay_barrier.wait()
        return bridge.run_optimizer_path_bridge(
            fresh_prepared_inputs["output_root"], host_stability_certificates={}
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        concurrent_results = list(
            executor.map(lambda _index: concurrent_replay(), range(2))
        )
    assert concurrent_results == [status, status]
    assert worker_calls == 21
    assert {path: _sha(path) for path in authorization_files} == authorization_hashes
    assert all(
        row["fresh_development"]["state"] == "complete" for row in status["seeds"]
    )
    evidence = bridge.normalize_optimizer_path_bridge(
        fresh_prepared_inputs["output_root"]
    )
    assert evidence["schema"] == "gcicy-x21-scientific-evidence-v1"
    assert len(evidence["seeds"]) == 3
    assert all(
        row["control_batch_plan_sha256"] == row["candidate_batch_plan_sha256"]
        for row in evidence["seeds"]
    )
    report = bridge.adjudicate_optimizer_path_bridge(
        fresh_prepared_inputs["output_root"]
    )
    assert report["promotion_passes"] is True
    controller = fresh_prepared_inputs["store"].status()
    assert controller["rounds"]["1"]["status"] == "complete"
    assert controller["leaderboard"][0]["promotion_passes"] is True
    assert (
        bridge.adjudicate_optimizer_path_bridge(fresh_prepared_inputs["output_root"])
        == report
    )


def test_expired_certificate_can_resume_with_new_append_only_authorization(
    fresh_prepared_inputs, monkeypatch: pytest.MonkeyPatch
):
    _prepare_fresh(fresh_prepared_inputs)
    certificate_identity = bridge._host_certificate_identity
    expired_paths: set[Path] = set()

    def expire_at_runtime(**kwargs):
        if (
            kwargs.get("now") is None
            and Path(kwargs["certificate_path"]).resolve() in expired_paths
        ):
            raise bridge.X21ScientificBridgeError(
                "host stability certificate rejected: certificate has expired"
            )
        return certificate_identity(**kwargs)

    @contextmanager
    def locked_gpu(_root: Path, _gpu_id: str, *, timeout_seconds: float):
        assert timeout_seconds == -1
        yield Path("/tmp/gcicy-tn-gpu-locks/gcicy-tn-gpu-0.lock")

    worker_calls = 0

    def fail_once_then_complete(command: list[str], **kwargs) -> SimpleNamespace:
        nonlocal worker_calls
        worker_calls += 1
        if worker_calls == 1:
            checkpoint = Path(_value_after(command, "--checkpoint"))
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b"renewable-sealed-checkpoint")
            return SimpleNamespace(returncode=11, stdout="", stderr="native fault\n")
        return _successful_full_worker(command, **kwargs)

    monkeypatch.setattr(bridge, "gpu_lock", locked_gpu)
    monkeypatch.setattr(bridge, "_host_certificate_identity", expire_at_runtime)
    monkeypatch.setattr(bridge.subprocess, "run", fail_once_then_complete)
    with pytest.raises(bridge.X21BridgeTechnicalError, match="not a scientific"):
        bridge.run_optimizer_path_bridge(
            fresh_prepared_inputs["output_root"],
            host_stability_certificates=fresh_prepared_inputs["host_certificates"],
        )
    old_entries = sorted(
        (fresh_prepared_inputs["output_root"] / "host_authorizations").glob("*/*.json")
    )
    assert len(old_entries) == 1
    old_hashes = {path: _sha(path) for path in old_entries}
    expired_paths.update(
        path.resolve() for path in fresh_prepared_inputs["host_certificates"].values()
    )
    with pytest.raises(bridge.X21ScientificBridgeError, match="expired"):
        bridge.run_optimizer_path_bridge(
            fresh_prepared_inputs["output_root"],
            host_stability_certificates=fresh_prepared_inputs["host_certificates"],
        )
    assert worker_calls == 1
    expired_status = bridge.bridge_status(fresh_prepared_inputs["output_root"])
    assert expired_status["scientific_outcome"] is None
    assert expired_status["scientific_rejection"] is False
    assert (
        fresh_prepared_inputs["store"].status()["rounds"]["1"]["evidence_sha256"]
        is None
    )

    renewed = {}
    for seed, source in fresh_prepared_inputs["host_certificates"].items():
        path = source.with_name(f"post_expiry_{source.name}")
        path.write_bytes(source.read_bytes())
        renewed[seed] = path
    status = bridge.run_optimizer_path_bridge(
        fresh_prepared_inputs["output_root"],
        host_stability_certificates=renewed,
    )
    assert status["operational_state"] == "scientific-evaluation-complete"
    assert worker_calls == 22
    assert {path: _sha(path) for path in old_entries} == old_hashes
    assert (
        len(
            list(
                (fresh_prepared_inputs["output_root"] / "host_authorizations").glob(
                    "*/*.json"
                )
            )
        )
        == 4
    )
    assert status["scientific_outcome"] is None
    assert status["scientific_rejection"] is False
    controller = fresh_prepared_inputs["store"].status()
    assert controller["rounds"]["1"]["status"] == "registered"
    assert controller["rounds"]["1"]["evidence_sha256"] is None
