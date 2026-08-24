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

from gcicy_metric.pipeline.architecture_auto_research import AutoResearchError
from gcicy_metric.pipeline import host_stability_gate as host_gate
from gcicy_metric.pipeline import quintic_host_gpu_probe as probe


ROOT = Path(__file__).resolve().parents[1]
CATALOG = (
    ROOT
    / "experiments"
    / "protocols"
    / "generic_quintic_architecture_multi_round_v2.json"
)
SOURCE_COMMIT = "a" * 40
HOST_IDENTITY = hashlib.sha256(b"quintic-probe-test-host").hexdigest()
SEED = 202608231


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _round1_data(tmp_path: Path) -> dict:
    root = tmp_path / "round1"
    root.mkdir()
    batch_path = root / "matched-batch.npy"
    batch = np.tile(np.arange(1024, dtype=np.int64), (600, 1))
    np.save(batch_path, batch, allow_pickle=False)
    index_path = root / "fixed-indices.npz"
    np.savez_compressed(
        index_path,
        fit=np.arange(30_000, dtype=np.int64),
        selection=np.arange(30_000, 35_000, dtype=np.int64),
        development_evaluation=np.arange(5_000, dtype=np.int64),
    )
    roles = {}
    for name in (
        "search-native-points",
        "search-native-pullbacks",
        "development-evaluation-points",
        "development-evaluation-pullbacks",
    ):
        path = root / f"{name}.bin"
        path.write_bytes(name.encode())
        roles[name] = {
            "path": str(path.resolve()),
            "sha256": host_gate.sha256_file(path),
        }
    plan_path = root / "plan.json"
    plan_path.write_text("{}\n", encoding="utf-8")
    seeds = []
    for seed in (202608231, 202608232, 202608233):
        seeds.append(
            {
                "seed": seed,
                "optimizer_seed": seed,
                "data_seed": 2026082301,
                "batch_plan_seed": seed + 17,
                "batch_plan": {
                    "path": str(batch_path.resolve()),
                    "file_sha256": host_gate.sha256_file(batch_path),
                    "value_set_sha256": _sha("batch-values"),
                },
            }
        )
    return {
        "root": str(root.resolve()),
        "plan_path": str(plan_path.resolve()),
        "plan_file_sha256": host_gate.sha256_file(plan_path),
        "plan_sha256": _sha("round1-plan"),
        "search_indices_sha256": _sha("round1-indices"),
        "index_manifest": {
            "path": str(index_path.resolve()),
            "file_sha256": host_gate.sha256_file(index_path),
            "value_set_sha256": _sha("index-values"),
        },
        "input_roles": roles,
        "seeds": seeds,
    }


@pytest.fixture
def environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    parent = tmp_path / "registered-parent.pt"
    parent.write_bytes(b"registered complex64 compiled-tree parent")
    round1 = _round1_data(tmp_path)
    source = {
        "commit": SOURCE_COMMIT,
        "worktree": "clean",
        "dependencies": {"probe.py": _sha("probe-source")},
    }
    monkeypatch.setattr(probe, "_source_contract", lambda _: source)
    monkeypatch.setattr(probe, "_host_identity_sha256", lambda: HOST_IDENTITY)
    monkeypatch.setattr(
        probe,
        "_runtime_environment",
        lambda: {
            "numpy": np.__version__,
            "torch": "test",
            "torch_cuda_build": "test",
            "gpu_name": "mock-gpu",
            "gpu_total_memory_bytes": 24 * 1024**3,
        },
    )
    monkeypatch.setattr(
        probe,
        "_checkpoint_contract",
        lambda _: {
            "precision": "complex64",
            "architecture": "compiled-tree",
            "teacher_runtime_dependency": False,
        },
    )
    monkeypatch.setattr(
        probe,
        "_validate_round1_data_contract",
        lambda *_: round1,
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
        "parent": parent,
        "round1": round1,
        "source": source,
        "lock_state": lock_state,
        "runtime": {
            "device": "cuda",
            "optimizer_steps": 10,
            "learning_rate": 3.0e-6,
            "gradient_clip_norm": 1.0,
            "threads": 2,
            "train_chunk_size": 128,
            "feature_batch_size": 128,
            "eval_batch_size": 64,
        },
    }


def _prepare(environment: dict, tmp_path: Path, probe_id: str) -> dict:
    return probe.prepare_probe(
        catalog_path=CATALOG,
        round1_bridge_root=Path(environment["round1"]["root"]),
        parent_checkpoint=environment["parent"],
        parent_checkpoint_sha256=host_gate.sha256_file(environment["parent"]),
        parent_seed=SEED,
        probe_id=probe_id,
        output_root=tmp_path / probe_id,
        runtime=environment["runtime"],
        repository_root=ROOT,
    )


def _worker_result(plan: dict, **overrides) -> dict:
    finished = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(seconds=1)
    started = finished - timedelta(seconds=30)
    value = {
        "schema": probe.WORKER_RESULT_SCHEMA,
        "probe_id": plan["probe_id"],
        "plan_sha256": plan["plan_sha256"],
        "source_commit": plan["source_contract"]["commit"],
        "host_identity_sha256": plan["host_identity_sha256"],
        "checkpoint_sha256": plan["parent_checkpoint"]["sha256"],
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "device": "cuda",
        "optimizer_steps_completed": plan["runtime"]["optimizer_steps"],
        "selection_evaluation_completed": True,
        "all_finite": True,
        "minimum_metric_eigenvalue": 0.01,
        "nonpositive_metric_count": 0,
        "maximum_allocated_bytes": 8 * 1024**3,
        "batch_prefix_sha256": plan["batch_prefix_sha256"],
        "selection_metrics": {"sigma": 0.12, "chi": 0.34},
    }
    value.update(overrides)
    return value


def _install_worker(
    monkeypatch: pytest.MonkeyPatch, environment: dict, **overrides
) -> None:
    def fake_worker(plan: dict, root: Path) -> subprocess.CompletedProcess[str]:
        assert environment["lock_state"]["held"] is True
        value = _worker_result(plan, **overrides)
        probe._publish_create_only(
            Path(plan["worker_result_path"]), value, context="mock worker result"
        )
        return subprocess.CompletedProcess(
            probe._worker_argv(plan, root), 0, "ok\n", ""
        )

    monkeypatch.setattr(probe, "_launch_worker", fake_worker)


def test_prepare_binds_parent_clean_source_host_and_frozen_nonheldout_data(
    environment, tmp_path
):
    plan = _prepare(environment, tmp_path, "probe-a")
    assert plan["parent_checkpoint"]["path"] == str(environment["parent"].resolve())
    assert plan["parent_checkpoint"]["sha256"] == host_gate.sha256_file(
        environment["parent"]
    )
    assert plan["source_contract"] == environment["source"]
    assert plan["host_identity_sha256"] == HOST_IDENTITY
    assert plan["allowed_splits"] == ["fit", "selection"]
    assert plan["runtime"]["device"] == "cuda"
    assert plan["runtime"]["optimizer_steps"] == 10
    argv = probe._worker_argv(plan, tmp_path / "probe-a")
    assert argv[2:] == ["_worker", "--root", str((tmp_path / "probe-a").resolve())]
    assert not any(
        token in " ".join(argv).lower() for token in ("blind", "confirmation", "shadow")
    )


def test_passing_probe_rehashes_and_is_host_certificate_compatible(
    environment, tmp_path, monkeypatch
):
    _install_worker(monkeypatch, environment)
    receipts = []
    for index in range(1, 4):
        root = tmp_path / f"probe-{index}"
        _prepare(environment, tmp_path, f"probe-{index}")
        receipt = probe.run_probe(root)
        verified = probe.verify_probe(root)
        assert verified["normalized_receipt"]["passes"] is True
        assert verified["receipt"] == receipt
        assert Path(receipt["report_path"]).is_file()
        assert receipt["report_sha256"] == host_gate.sha256_file(
            Path(receipt["report_path"])
        )
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
    host_gate.validate_certificate(
        certificate,
        expected_checkpoint_path=environment["parent"],
        expected_checkpoint_sha256=receipts[0]["checkpoint_sha256"],
        expected_host_identity_sha256=HOST_IDENTITY,
        expected_source_commit=SOURCE_COMMIT,
        now=issued,
    )


def test_parent_and_round1_binding_drift_fail_before_worker(
    environment, tmp_path, monkeypatch
):
    _prepare(environment, tmp_path, "probe-drift")
    environment["parent"].write_bytes(b"changed checkpoint")
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("worker must not start")

    monkeypatch.setattr(probe, "_launch_worker", forbidden)
    with pytest.raises(AutoResearchError, match="checkpoint drifted"):
        probe.run_probe(tmp_path / "probe-drift")
    assert called is False


def test_source_and_parent_are_revalidated_inside_the_shared_gpu_lock(
    environment, tmp_path, monkeypatch
):
    root = tmp_path / "probe-lock-audit"
    _prepare(environment, tmp_path, "probe-lock-audit")
    source_checks = []
    parent_checks = []

    def checked_source(_root):
        source_checks.append(environment["lock_state"]["held"])
        return environment["source"]

    def checked_parent(_path):
        parent_checks.append(environment["lock_state"]["held"])
        return {
            "precision": "complex64",
            "architecture": "compiled-tree",
            "teacher_runtime_dependency": False,
        }

    monkeypatch.setattr(probe, "_source_contract", checked_source)
    monkeypatch.setattr(probe, "_checkpoint_contract", checked_parent)
    _install_worker(monkeypatch, environment)
    probe.run_probe(root)
    assert False in source_checks and True in source_checks
    assert False in parent_checks and True in parent_checks


def test_tampered_raw_report_and_coherently_rehashed_receipt_are_rejected(
    environment, tmp_path, monkeypatch
):
    _install_worker(monkeypatch, environment)
    root = tmp_path / "probe-forgery"
    _prepare(environment, tmp_path, "probe-forgery")
    probe.run_probe(root)

    raw_path = root / "raw-report.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    original_raw = dict(raw)
    raw["minimum_metric_eigenvalue"] = 9.0
    raw_path.write_text(json.dumps(raw, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(AutoResearchError, match="raw report was forged"):
        probe.verify_probe(root)

    raw_path.write_text(
        json.dumps(original_raw, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt_path = root / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["minimum_metric_eigenvalue"] = 9.0
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
    )
    ledger = probe._read_ledger(root / "ledger.json")
    ledger["raw_report_sha256"] = host_gate.sha256_file(raw_path)
    ledger["receipt_sha256"] = host_gate.sha256_file(receipt_path)
    probe._write_ledger(root / "ledger.json", ledger)
    with pytest.raises(AutoResearchError, match="receipt was forged"):
        probe.verify_probe(root)


def test_nonpositive_probe_is_preserved_as_diagnostic_only(
    environment, tmp_path, monkeypatch
):
    _install_worker(
        monkeypatch,
        environment,
        minimum_metric_eigenvalue=-0.01,
        nonpositive_metric_count=2,
    )
    root = tmp_path / "probe-nonpositive"
    _prepare(environment, tmp_path, "probe-nonpositive")
    receipt = probe.run_probe(root)
    assert receipt["minimum_metric_eigenvalue"] < 0
    diagnostic = probe.verify_probe(root, require_passing=False)
    assert diagnostic["ledger"]["state"] == "diagnostic-only"
    assert diagnostic["normalized_receipt"]["passes"] is False
    with pytest.raises(AutoResearchError, match="diagnostic-only"):
        probe.verify_probe(root)


def test_short_or_ambiguous_worker_execution_fails_closed(
    environment, tmp_path, monkeypatch
):
    _install_worker(monkeypatch, environment, optimizer_steps_completed=9)
    root = tmp_path / "probe-short"
    _prepare(environment, tmp_path, "probe-short")
    with pytest.raises(AutoResearchError, match="outside its registered range"):
        probe.run_probe(root)
    assert not (root / "receipt.json").exists()
    with pytest.raises(AutoResearchError):
        probe.run_probe(root)

    ambiguous = tmp_path / "probe-ambiguous"
    _prepare(environment, tmp_path, "probe-ambiguous")
    ledger = probe._read_ledger(ambiguous / "ledger.json")
    ledger["state"] = "running"
    probe._write_ledger(ambiguous / "ledger.json", ledger)
    with pytest.raises(AutoResearchError, match="ambiguous"):
        probe.run_probe(ambiguous)


def test_runtime_environment_can_be_unit_tested_with_mock_torch(monkeypatch):
    fake_torch = SimpleNamespace(
        __version__="mock-torch",
        version=SimpleNamespace(cuda="mock-cuda"),
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
    with pytest.raises(AutoResearchError, match="not held"):
        probe._assert_parent_holds_gpu_lock()
