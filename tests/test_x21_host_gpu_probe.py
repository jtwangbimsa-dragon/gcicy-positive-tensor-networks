from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from gcicy_metric.pipeline import host_stability_gate as host_gate
from gcicy_metric.pipeline import x21_host_gpu_probe as probe
from gcicy_metric.pipeline.x21_host_gpu_probe import X21HostGPUProbeError


ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "b" * 40
HOST_IDENTITY = hashlib.sha256(b"x21-probe-test-host").hexdigest()
PARENT_SEED = 8_660_001


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


@pytest.fixture
def environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    protocol_path = tmp_path / "x21-protocol.json"
    protocol_path.write_text("{}\n", encoding="utf-8")
    source = tmp_path / "X21_source.npz"
    train_pool = tmp_path / "X21_train_seed86201_n196608.npz"
    selection_pool = tmp_path / "X21_selection_seed86202_n24576.npz"
    parent = tmp_path / "x21-k20-d14-parent.pt"
    for path, contents in (
        (source, b"source"),
        (train_pool, b"train"),
        (selection_pool, b"selection"),
        (parent, b"parent"),
    ):
        path.write_bytes(contents)

    protocol = {
        "schema": "gcicy-x21-auto-research-protocol-v1",
        "campaign_id": "x21-auto-research-v1",
        "promotion_seeds": [8_660_001, 8_660_002, 8_660_003],
        "data_contract": {
            "source_artifact_sha256": host_gate.sha256_file(source),
            "train_pool_sha256": host_gate.sha256_file(train_pool),
            "selection_pool_sha256": host_gate.sha256_file(selection_pool),
            "development_pool_sha256": _sha("unused-development"),
            "train_points": probe.TRAIN_POINTS,
            "selection_points": probe.SELECTION_POINTS,
            "development_points": 49_152,
            "sampling_cluster_size": probe.SAMPLING_CLUSTER_SIZE,
        },
        "data_contract_sha256": _sha("data-contract"),
    }
    protocol_artifact = {
        "path": str(protocol_path.resolve()),
        "bytes": protocol_path.stat().st_size,
        "sha256": host_gate.sha256_file(protocol_path),
        "normalized_sha256": probe.digest_value(protocol),
        "data_contract_sha256": protocol["data_contract_sha256"],
        "schema": protocol["schema"],
        "campaign_id": protocol["campaign_id"],
    }
    source_contract = {
        "repository_root": str(ROOT),
        "commit": SOURCE_COMMIT,
        "branch": "exp/test-x21-probe",
        "tracked_and_untracked_worktree": "clean",
        "dependencies": {"x21_host_gpu_probe.py": _sha("source")},
    }
    runtime_environment = {
        "numpy": np.__version__,
        "torch": "test",
        "torch_cuda_build": "test",
        "gpu_name": "mock-gpu",
        "gpu_total_memory_bytes": 24 * 1024**3,
    }
    parent_contract = {
        "schema": "type11-positive-tensor-network-v1",
        "adapter": probe.ADAPTER,
        "model_seed": probe.MODEL_SEED,
        "site_count": probe.SITE_COUNT,
        "bond_dimension": probe.BOND_DIMENSION,
        "architecture": "shared_local_dictionary",
        "physical_dictionary_rank": probe.PHYSICAL_DICTIONARY_RANK,
        "trainable_physical_dictionary": False,
        "physical_dictionary_gauge": "fixed",
        "trainable_real_parameter_count": probe.TRAINABLE_REAL_PARAMETERS,
        "precision": probe.PRECISION,
        "source_artifact_sha256": host_gate.sha256_file(source),
        "train_common_pool_sha256": host_gate.sha256_file(train_pool),
        "validation_common_pool_sha256": host_gate.sha256_file(selection_pool),
    }
    monkeypatch.setattr(
        probe,
        "_protocol_contract",
        lambda _path: (protocol, protocol_artifact),
    )
    monkeypatch.setattr(probe, "_source_contract", lambda _root: source_contract)
    monkeypatch.setattr(probe, "_host_identity_sha256", lambda: HOST_IDENTITY)
    monkeypatch.setattr(probe, "_runtime_environment", lambda: runtime_environment)
    monkeypatch.setattr(
        probe, "_parent_contract", lambda _path, protocol: parent_contract
    )
    lock_state = {"held": False, "acquisitions": 0}

    @contextmanager
    def mock_gpu_lock(lock_root: Path, gpu_id: str, *, timeout_seconds: float):
        assert lock_root == Path("/tmp/gcicy-tn-gpu-locks")
        assert gpu_id == "0"
        assert timeout_seconds == -1
        assert lock_state["held"] is False
        lock_state["held"] = True
        lock_state["acquisitions"] += 1
        try:
            yield lock_root / "gcicy-tn-gpu-0.lock"
        finally:
            lock_state["held"] = False

    monkeypatch.setattr(probe, "gpu_lock", mock_gpu_lock)
    return {
        "protocol_path": protocol_path,
        "protocol": protocol,
        "protocol_artifact": protocol_artifact,
        "source": source,
        "train_pool": train_pool,
        "selection_pool": selection_pool,
        "parent": parent,
        "parent_contract": parent_contract,
        "source_contract": source_contract,
        "runtime_environment": runtime_environment,
        "lock_state": lock_state,
        "runtime": {"device": "cuda", "workers": 1, "teacher_chunk_size": 512},
    }


def _prepare(environment: dict, tmp_path: Path, probe_id: str) -> dict:
    return probe.prepare_probe(
        protocol_path=environment["protocol_path"],
        source_artifact=environment["source"],
        train_common_pool=environment["train_pool"],
        selection_common_pool=environment["selection_pool"],
        parent_model=environment["parent"],
        parent_model_sha256=host_gate.sha256_file(environment["parent"]),
        parent_seed=PARENT_SEED,
        probe_id=probe_id,
        output_root=tmp_path / probe_id,
        repository_root=ROOT,
        runtime=environment["runtime"],
    )


def _worker_result(plan: dict, **overrides) -> dict:
    root = Path(plan["artifacts"]["worker_result"]).parent
    training = root / "training"
    training.mkdir(parents=True, exist_ok=True)
    files = {
        "model": training / "model.pt",
        "summary": training / "summary.json",
        "checkpoint": training / "checkpoint.pt",
    }
    for role, path in files.items():
        if not path.exists():
            path.write_bytes(f"mock-{role}".encode())
    finished = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(seconds=1)
    started = finished - timedelta(seconds=30)
    value = {
        "schema": probe.WORKER_RESULT_SCHEMA,
        "probe_id": plan["probe_id"],
        "plan_sha256": plan["plan_sha256"],
        "source_commit": plan["source_contract"]["commit"],
        "host_identity_sha256": plan["host_identity_sha256"],
        "checkpoint_sha256": plan["parent_model"]["sha256"],
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "device": "cuda",
        "parameter_scope": "cores-only",
        "optimizer_steps_completed": probe.OPTIMIZER_UPDATES,
        "selection_evaluation_completed": True,
        "all_finite": True,
        "minimum_metric_eigenvalue": 0.01,
        "nonpositive_metric_count": 0,
        "maximum_allocated_bytes": 8 * 1024**3,
        "selection_metrics": {
            "selection_score": 0.1,
            "log_energy_rms": 0.2,
            "chi": 0.3,
        },
        "trainer_returncode": 0,
        "trainer_stdout_sha256": _sha("trainer-stdout"),
        "trainer_stderr_sha256": _sha("trainer-stderr"),
        "training_command_sha256": plan["training_command_sha256"],
        "training_artifacts": {
            role: probe._artifact(path) for role, path in files.items()
        },
    }
    value.update(overrides)
    return value


def _install_worker(
    monkeypatch: pytest.MonkeyPatch, environment: dict, **overrides
) -> None:
    def fake_worker(plan: dict, root: Path) -> subprocess.CompletedProcess[str]:
        assert environment["lock_state"]["held"] is True
        result = _worker_result(plan, **overrides)
        probe._publish_create_only(
            Path(plan["artifacts"]["worker_result"]), result, role="mock worker result"
        )
        return subprocess.CompletedProcess(
            probe._worker_argv(plan, root), 0, "ok\n", ""
        )

    monkeypatch.setattr(probe, "_launch_worker", fake_worker)


def _summary(plan: dict) -> dict:
    model_path = Path(plan["artifacts"]["trained_model"])
    checkpoint_path = Path(plan["artifacts"]["training_checkpoint"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if not model_path.exists():
        model_path.write_bytes(b"mock-trained-model")
    if not checkpoint_path.exists():
        checkpoint_path.write_bytes(b"mock-training-checkpoint")
    return {
        "schema": "type11-positive-tensor-network-training-v1",
        "adapter": probe.ADAPTER,
        "training_mode": "teacher_free_geometric",
        "model": str(model_path),
        "model_sha256": host_gate.sha256_file(model_path),
        "source_artifact": plan["immutable_inputs"]["source_artifact"]["path"],
        "source_artifact_sha256": plan["immutable_inputs"]["source_artifact"]["sha256"],
        "teacher_artifact": None,
        "teacher_artifact_sha256": None,
        "model_seed": probe.MODEL_SEED,
        "torch_seed": plan["parent_seed"],
        "site_count": probe.SITE_COUNT,
        "bond_dimension": probe.BOND_DIMENSION,
        "architecture": "shared_local_dictionary",
        "physical_dictionary_rank": probe.PHYSICAL_DICTIONARY_RANK,
        "trainable_physical_dictionary": False,
        "physical_dictionary_gauge": "fixed",
        "trainable_real_parameter_count": probe.TRAINABLE_REAL_PARAMETERS,
        "positive_floor": 1.0e-14,
        "initialization_noise": 0.0,
        "device": "cuda",
        "precision": probe.PRECISION,
        "sampling_cluster_size": probe.SAMPLING_CLUSTER_SIZE,
        "epochs": 1,
        "batch_size": probe.BATCH_SIZE,
        "learning_rate": 3.0e-5,
        "gradient_clip_norm": 2.0,
        "fixed_log_kappa_source": "continued_saved_model",
        "termination_reason": "completed_requested_epochs",
        "initialization": {"model_sha256": plan["parent_model"]["sha256"]},
        "train": {
            "points": probe.TRAIN_POINTS,
            "seed": probe.TRAIN_SEED,
            "common_pool": plan["immutable_inputs"]["train_common_pool"]["path"],
            "common_pool_sha256": plan["immutable_inputs"]["train_common_pool"][
                "sha256"
            ],
        },
        "validation": {
            "points": probe.SELECTION_POINTS,
            "seed": probe.SELECTION_SEED,
            "common_pool": plan["immutable_inputs"]["selection_common_pool"]["path"],
            "common_pool_sha256": plan["immutable_inputs"]["selection_common_pool"][
                "sha256"
            ],
        },
        "history": [{"epoch": 0}, {"epoch": 1}],
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": host_gate.sha256_file(checkpoint_path),
            "last_completed_validation_epoch": 1,
            "resume_kind": "fresh_training",
        },
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
        },
        "early_stopping": {"patience": 0},
        "device_memory": {
            "maximum_allocated_bytes": 8 * 1024**3,
            "maximum_reserved_bytes": 9 * 1024**3,
        },
        "best_validation_selection_score": 0.1,
        "best_validation": {
            "minimum_metric_eigenvalue": 0.01,
            "fixed_kappa_log_energy_rms": 0.2,
            "compressed_ma_errors": {"sqrt_squared_energy": 0.3},
        },
    }


def test_prepare_binds_protocol_parent_full_epoch_and_nonheldout_pools(
    environment, tmp_path
):
    plan = _prepare(environment, tmp_path, "x21-probe-a")
    assert plan["parent_model"]["sha256"] == host_gate.sha256_file(
        environment["parent"]
    )
    assert plan["protocol"] == environment["protocol_artifact"]
    assert plan["fixed_workload"]["optimizer_updates"] == 192
    assert plan["fixed_workload"]["parameter_scope"] == "cores-only"
    assert plan["data_access_policy"]["allowed_splits"] == ["train", "selection"]
    command = plan["training_command"]
    assert command[0] == str(Path(sys.executable).resolve())
    assert command[1].endswith("scripts/train_type11_positive_tensor_network.py")
    assert command[command.index("--epochs") + 1] == "1"
    assert command[command.index("--batch-size") + 1] == "1024"
    assert command[command.index("--initial-model") + 1] == str(
        environment["parent"].resolve()
    )
    assert "--train-physical-dictionary" not in command
    assert "--resume-checkpoint" not in command
    assert not any(
        token in " ".join(command).lower()
        for token in ("blind", "confirmation", "holdout", "shadow")
    )
    assert Path(command[command.index("--out") + 1]) != environment["parent"]


def test_three_passing_receipts_are_host_certificate_compatible(
    environment, tmp_path, monkeypatch
):
    _install_worker(monkeypatch, environment)
    receipts = []
    for index in range(1, 4):
        root = tmp_path / f"x21-probe-{index}"
        _prepare(environment, tmp_path, f"x21-probe-{index}")
        receipt = probe.run_probe(root)
        verified = probe.verify_probe(root)
        assert verified["receipt"] == receipt
        assert verified["normalized_receipt"]["passes"] is True
        assert receipt["optimizer_steps"] == 192
        assert probe.probe_status(root)["passes"] is True
        receipts.append(receipt)

    process_rows = [
        {
            "attempt": index,
            "returncode": 0,
            "stdout_sha256": _sha(f"fresh-{index}"),
            "checkpoint_sha256": receipts[0]["checkpoint_sha256"],
            "host_identity_sha256": HOST_IDENTITY,
            "source_commit": SOURCE_COMMIT,
        }
        for index in range(1, 21)
    ]
    issued = datetime.now(timezone.utc).replace(microsecond=0)
    certificate = host_gate.build_certificate(
        checkpoint_path=environment["parent"],
        expected_checkpoint_sha256=receipts[0]["checkpoint_sha256"],
        host_identity_sha256=HOST_IDENTITY,
        source_commit=SOURCE_COMMIT,
        kernel_log="quiet\n",
        fresh_process_results=process_rows,
        gpu_probe_values=receipts,
        issued_at=issued,
    )
    assert certificate["scientific_authorized"] is True
    assert environment["lock_state"]["acquisitions"] == 3


def test_parent_and_pool_drift_fail_before_worker(environment, tmp_path, monkeypatch):
    _prepare(environment, tmp_path, "x21-probe-drift")
    environment["parent"].write_bytes(b"changed-parent")
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("worker must not run")

    monkeypatch.setattr(probe, "_launch_worker", forbidden)
    with pytest.raises(X21HostGPUProbeError, match="parent model artifact drifted"):
        probe.run_probe(tmp_path / "x21-probe-drift")
    assert called is False


def test_source_parent_and_protocol_are_revalidated_inside_gpu_lock(
    environment, tmp_path, monkeypatch
):
    root = tmp_path / "x21-probe-lock-audit"
    _prepare(environment, tmp_path, "x21-probe-lock-audit")
    source_checks = []
    parent_checks = []
    protocol_checks = []

    def checked_source(_root):
        source_checks.append(environment["lock_state"]["held"])
        return environment["source_contract"]

    def checked_parent(_path, *, protocol):
        parent_checks.append(environment["lock_state"]["held"])
        return environment["parent_contract"]

    def checked_protocol(_path):
        protocol_checks.append(environment["lock_state"]["held"])
        return environment["protocol"], environment["protocol_artifact"]

    monkeypatch.setattr(probe, "_source_contract", checked_source)
    monkeypatch.setattr(probe, "_parent_contract", checked_parent)
    monkeypatch.setattr(probe, "_protocol_contract", checked_protocol)
    _install_worker(monkeypatch, environment)
    probe.run_probe(root)
    assert False in source_checks and True in source_checks
    assert False in parent_checks and True in parent_checks
    assert False in protocol_checks and True in protocol_checks


def test_tampered_raw_report_is_rejected(environment, tmp_path, monkeypatch):
    _install_worker(monkeypatch, environment)
    root = tmp_path / "x21-probe-forgery"
    _prepare(environment, tmp_path, "x21-probe-forgery")
    probe.run_probe(root)
    raw_path = root / "raw-report.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw["minimum_metric_eigenvalue"] = 9.0
    raw_path.write_text(json.dumps(raw, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(X21HostGPUProbeError, match="forged or drifted"):
        probe.verify_probe(root)


def test_nonpositive_probe_is_diagnostic_only(environment, tmp_path, monkeypatch):
    _install_worker(
        monkeypatch,
        environment,
        minimum_metric_eigenvalue=-0.01,
        nonpositive_metric_count=1,
    )
    root = tmp_path / "x21-probe-nonpositive"
    _prepare(environment, tmp_path, "x21-probe-nonpositive")
    receipt = probe.run_probe(root)
    assert receipt["minimum_metric_eigenvalue"] < 0
    verified = probe.verify_probe(root, require_passing=False)
    assert verified["ledger"]["state"] == "diagnostic-only"
    with pytest.raises(X21HostGPUProbeError, match="diagnostic-only"):
        probe.verify_probe(root)


def test_short_and_ambiguous_execution_fail_closed(environment, tmp_path, monkeypatch):
    _install_worker(monkeypatch, environment, optimizer_steps_completed=191)
    root = tmp_path / "x21-probe-short"
    _prepare(environment, tmp_path, "x21-probe-short")
    with pytest.raises(X21HostGPUProbeError, match="full one-epoch"):
        probe.run_probe(root)
    assert not (root / "receipt.json").exists()

    ambiguous = tmp_path / "x21-probe-ambiguous"
    _prepare(environment, tmp_path, "x21-probe-ambiguous")
    ledger = probe._read_ledger(ambiguous / "ledger.json")
    ledger["state"] = "running"
    probe._write_ledger(ambiguous / "ledger.json", ledger)
    with pytest.raises(X21HostGPUProbeError, match="ambiguous"):
        probe.run_probe(ambiguous)


def test_summary_validator_requires_one_epoch_selection_and_cores_only(
    environment, tmp_path
):
    plan = _prepare(environment, tmp_path, "x21-probe-summary")
    summary = _summary(plan)
    normalized = probe._validate_training_summary(plan, summary)
    assert normalized["minimum_metric_eigenvalue"] == 0.01
    assert normalized["maximum_allocated_bytes"] == 8 * 1024**3
    summary["trainable_physical_dictionary"] = True
    with pytest.raises(X21HostGPUProbeError, match="trainable_physical_dictionary"):
        probe._validate_training_summary(plan, summary)


def test_cuda_worker_invokes_exact_trainer_without_shell_and_validates_checkpoint(
    environment, tmp_path, monkeypatch
):
    plan = _prepare(environment, tmp_path, "x21-probe-worker")
    root = tmp_path / "x21-probe-worker"
    monkeypatch.setattr(probe, "_assert_parent_holds_gpu_lock", lambda: None)
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    calls = []

    checkpoint_payload = {
        "schema": "type11-positive-tensor-network-checkpoint-v1",
        "epoch": 1,
        "next_epoch": 2,
        "history": [{"epoch": 0}, {"epoch": 1}],
        "training_semantics": {
            "model": {
                "site_count": probe.SITE_COUNT,
                "bond_dimension": probe.BOND_DIMENSION,
                "precision": probe.PRECISION,
                "trainable_physical_dictionary": False,
            },
            "data": {
                "train_points": probe.TRAIN_POINTS,
                "train_seed": probe.TRAIN_SEED,
                "validation_points": probe.SELECTION_POINTS,
                "validation_seed": probe.SELECTION_SEED,
                "sampling_cluster_size": probe.SAMPLING_CLUSTER_SIZE,
                "train_uses_common_pool": True,
                "validation_uses_common_pool": True,
            },
            "optimization": {
                "epochs": 1,
                "batch_size": probe.BATCH_SIZE,
                "learning_rate": 3.0e-5,
                "gradient_clip_norm": 2.0,
                "torch_seed": PARENT_SEED,
                "kappa_source": "saved_model",
            },
        },
        "frozen_input_hashes": {
            "source_artifact_sha256": plan["immutable_inputs"]["source_artifact"][
                "sha256"
            ],
            "teacher_artifact_sha256": None,
            "initial_model_sha256": plan["parent_model"]["sha256"],
            "train_common_pool_sha256": plan["immutable_inputs"]["train_common_pool"][
                "sha256"
            ],
            "validation_common_pool_sha256": plan["immutable_inputs"][
                "selection_common_pool"
            ]["sha256"],
        },
    }

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        assert command == plan["training_command"]
        assert kwargs["shell"] is False
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0"
        summary = _summary(plan)
        Path(plan["artifacts"]["training_summary"]).write_text(
            json.dumps(summary) + "\n", encoding="utf-8"
        )
        return subprocess.CompletedProcess(command, 0, "trained\n", "")

    def fake_load(_path: Path, *, role: str):
        if role == "probe checkpoint":
            return checkpoint_payload
        return {"finite": True}

    monkeypatch.setattr(probe.subprocess, "run", fake_run)
    monkeypatch.setattr(probe, "_load_torch_object", fake_load)
    result = probe.execute_cuda_worker(root)
    assert result["optimizer_steps_completed"] == 192
    assert result["selection_evaluation_completed"] is True
    assert result["parameter_scope"] == "cores-only"
    assert result["all_finite"] is True
    assert len(calls) == 1


def test_launchers_use_shell_false(environment, tmp_path, monkeypatch):
    plan = _prepare(environment, tmp_path, "x21-probe-shell")
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args[0], 0, "", "")

    monkeypatch.setattr(probe.subprocess, "run", fake_run)
    probe._launch_worker(plan, tmp_path / "x21-probe-shell")
    assert calls[0][1]["shell"] is False
    assert isinstance(calls[0][0][0], list)


def test_runtime_environment_can_be_unit_tested_with_mock_torch(monkeypatch):
    fake_torch = SimpleNamespace(
        __version__="mock-torch", version=SimpleNamespace(cuda="mock-cuda")
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        probe.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, "mock-gpu, 24576\n", ""
        ),
    )
    assert probe._runtime_environment() == {
        "numpy": np.__version__,
        "torch": "mock-torch",
        "torch_cuda_build": "mock-cuda",
        "gpu_name": "mock-gpu",
        "gpu_total_memory_bytes": 24576 * 1024**2,
    }


def test_worker_refuses_to_initialize_without_parent_owned_gpu_lock(
    tmp_path, monkeypatch
):
    lock_root = tmp_path / "locks"
    lock_root.mkdir()
    lock_path = lock_root / "gcicy-tn-gpu-0.lock"
    monkeypatch.setattr(probe, "GPU_LOCK_ROOT", lock_root)
    handle = lock_path.open("a+", encoding="utf-8")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    handle.write(json.dumps({"pid": os.getppid(), "gpu_id": "0"}) + "\n")
    handle.flush()
    probe._assert_parent_holds_gpu_lock()
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    handle.close()
    with pytest.raises(X21HostGPUProbeError, match="not held"):
        probe._assert_parent_holds_gpu_lock()
