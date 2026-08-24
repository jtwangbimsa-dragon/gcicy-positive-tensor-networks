from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

import gcicy_metric.pipeline.x21_fresh_development_pool as fresh
from gcicy_metric.pipeline.x21_auto_research import digest_value, validate_protocol


ROOT = Path(__file__).resolve().parents[1]
BASE_PROTOCOL = ROOT / "experiments" / "protocols" / "x21_auto_research_v1.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_contract() -> dict:
    return {
        "repository_root": str(ROOT),
        "commit": "a" * 40,
        "branch": "exp/test-fresh-x21-development",
        "tracked_and_untracked_worktree": "clean",
        "files": {relative: "b" * 64 for relative in fresh.SOURCE_DEPENDENCIES},
    }


@pytest.fixture
def prepared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    output = tmp_path / "fresh-output"
    source = _source_contract()
    monkeypatch.setattr(
        fresh, "_git_source_contract", lambda _root: copy.deepcopy(source)
    )
    plan = fresh.prepare_fresh_development_pool(
        output_root=output,
        frozen_root=frozen,
        base_protocol=BASE_PROTOCOL,
        repository_root=ROOT,
        python_executable=Path(sys.executable),
        workers=2,
        backend="thread",
    )
    return {"root": output, "frozen": frozen, "source": source, "plan": plan}


def _manifest_metadata(plan: dict) -> dict:
    generation = plan["generation"]
    return {
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
                {
                    "worker": index,
                    "points": count,
                    "seed": seed,
                }
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
            "git_commit": plan["source_revision"]["commit"],
            "git_status_porcelain": "",
            "generator": plan["generator"]["path"],
        },
    }


def _write_mock_outputs(command: list[str], plan: dict) -> None:
    pool_path = Path(command[command.index("--out") + 1])
    manifest_path = Path(command[command.index("--manifest") + 1])
    metadata = _manifest_metadata(plan)
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pool_path,
        common_point_pool_schema_version=np.asarray(1, dtype=np.int64),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    array_names = (
        "importance_log_weights",
        "importance_weights",
        "sampling_cluster_ids",
        "baseline_metrics",
        "holomorphic_volume_log_density",
    )
    arrays = {
        "common_point_pool_schema_version": {"shape": [], "dtype": "int64"},
        **{name: {"shape": [fresh.POINTS], "dtype": "float64"} for name in array_names},
    }
    manifest = {
        **metadata,
        "pool_path": str(pool_path.resolve()),
        "pool_sha256": _sha(pool_path),
        "arrays": arrays,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _successful_child(plan: dict):
    def run(command: list[str], **kwargs) -> SimpleNamespace:
        assert kwargs["shell"] is False
        assert kwargs["check"] is False
        assert Path(command[1]) == ROOT / fresh.GENERATOR_RELATIVE
        assert command == plan["command"]
        joined = "\0".join(command).lower()
        assert "confirmation" not in joined
        assert "blind" not in joined
        assert "holdout" not in joined
        _write_mock_outputs(command, plan)
        return SimpleNamespace(returncode=0, stdout="wrote selection pool\n", stderr="")

    return run


def test_prepare_locks_exact_selection_contract_and_is_idempotent(prepared: dict):
    plan = prepared["plan"]
    assert plan["generation"] == {
        "adapter": "p5p1_type21_k3_1223",
        "model_seed": 20260802,
        "exact_model": True,
        "split": "selection",
        "points": 49152,
        "seed": 86206,
        "cluster_size": 6,
        "workers": 2,
        "backend": "thread",
        "root_separation_tolerance": None,
    }
    assert plan["source_revision"]["commit"] == "a" * 40
    assert plan["generator"]["sha256"] == _sha(ROOT / fresh.GENERATOR_RELATIVE)
    assert plan["command"][0] == str(Path(sys.executable).resolve())
    assert plan["command"][1] == str((ROOT / fresh.GENERATOR_RELATIVE).resolve())
    assert Path(plan["outputs"]["pool"]).name == fresh.POOL_FILENAME
    assert plan["data_access_policy"]["existing_pool_inputs"] == []
    assert fresh.HISTORICAL_CONFIRMATION_SHA256 not in plan["command"]

    again = fresh.prepare_fresh_development_pool(
        output_root=prepared["root"],
        frozen_root=prepared["frozen"],
        base_protocol=BASE_PROTOCOL,
        repository_root=ROOT,
        python_executable=Path(sys.executable),
        workers=2,
        backend="thread",
    )
    assert again == plan
    with pytest.raises(fresh.X21FreshDevelopmentPoolError, match="different content"):
        fresh.prepare_fresh_development_pool(
            output_root=prepared["root"],
            frozen_root=prepared["frozen"],
            base_protocol=BASE_PROTOCOL,
            repository_root=ROOT,
            python_executable=Path(sys.executable),
            workers=3,
            backend="thread",
        )


def test_prepare_rejects_frozen_and_source_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(fresh, "_git_source_contract", lambda _root: _source_contract())
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    with pytest.raises(fresh.X21FreshDevelopmentPoolError, match="frozen-results"):
        fresh.prepare_fresh_development_pool(
            output_root=frozen / "new-data",
            frozen_root=frozen,
            base_protocol=BASE_PROTOCOL,
            repository_root=ROOT,
        )
    with pytest.raises(fresh.X21FreshDevelopmentPoolError, match="source repository"):
        fresh.prepare_fresh_development_pool(
            output_root=ROOT / "unsafe-generated-data",
            frozen_root=frozen,
            base_protocol=BASE_PROTOCOL,
            repository_root=ROOT,
        )


def test_mock_materialization_publishes_binding_and_protocol_v2(
    prepared: dict, monkeypatch: pytest.MonkeyPatch
):
    base_before = BASE_PROTOCOL.read_bytes()
    monkeypatch.setattr(fresh.subprocess, "run", _successful_child(prepared["plan"]))
    result = fresh.materialize_fresh_development_pool(prepared["root"])
    binding = result["binding"]
    assert result["status"] == "complete-and-hash-locked"
    assert binding["generation"]["seed"] == 86206
    assert binding["generation"]["split"] == "selection"
    assert binding["pool"]["sha256"] != fresh.HISTORICAL_CONFIRMATION_SHA256
    assert binding["historical_exclusion"]["confirmation_opened"] is False
    assert (
        fresh.validate_fresh_development_binding(
            binding,
            expected_binding_sha256=binding["binding_sha256"],
            expected_pool_sha256=binding["pool"]["sha256"],
        )
        == binding
    )

    protocol = result["protocol_v2"]
    assert protocol["schema"] == fresh.PROTOCOL_V2_SCHEMA
    assert protocol["campaign_id"] == fresh.TARGET_CAMPAIGN_ID
    assert (
        protocol["data_contract"]["development_pool_sha256"]
        == binding["pool"]["sha256"]
    )
    assert (
        protocol["development_pool_binding"]["binding_sha256"]
        == binding["binding_sha256"]
    )
    normalized_protocol = validate_protocol(protocol)
    assert normalized_protocol["development_role"] == "fresh-search-only-selection"
    assert (
        normalized_protocol["development_pool_binding"]["pool_sha256"]
        == binding["pool"]["sha256"]
    )
    patch = result["protocol_patch"]
    replacement = next(
        row
        for row in patch["operations"]
        if row["path"] == "/data_contract/development_pool_sha256"
    )
    assert replacement["expected"] == fresh.HISTORICAL_CONFIRMATION_SHA256
    assert replacement["value"] == binding["pool"]["sha256"]
    assert BASE_PROTOCOL.read_bytes() == base_before

    status = fresh.fresh_development_pool_status(prepared["root"])
    assert status["status"] == "complete-and-hash-locked"
    assert status["binding_sha256"] == binding["binding_sha256"]
    assert (
        fresh.materialize_fresh_development_pool(prepared["root"])["binding"] == binding
    )


def test_old_registered_hash_is_hard_rejected(
    prepared: dict, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(fresh.subprocess, "run", _successful_child(prepared["plan"]))
    binding = fresh.materialize_fresh_development_pool(prepared["root"])["binding"]
    forged = copy.deepcopy(binding)
    forged["pool"]["sha256"] = fresh.HISTORICAL_CONFIRMATION_SHA256
    forged["binding_sha256"] = digest_value(
        {key: value for key, value in forged.items() if key != "binding_sha256"}
    )
    with pytest.raises(
        fresh.X21FreshDevelopmentPoolError, match="forbidden historical"
    ):
        fresh.validate_fresh_development_binding(forged, rehash_artifacts=False)


def test_binding_rehash_detects_pool_drift(
    prepared: dict, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(fresh.subprocess, "run", _successful_child(prepared["plan"]))
    binding = fresh.materialize_fresh_development_pool(prepared["root"])["binding"]
    Path(binding["pool"]["path"]).write_bytes(b"tampered")
    with pytest.raises(fresh.X21FreshDevelopmentPoolError, match="size drifted"):
        fresh.validate_fresh_development_binding(binding)


def test_technical_failure_is_retryable_and_not_scientific_rejection(
    prepared: dict, monkeypatch: pytest.MonkeyPatch
):
    calls = 0

    def fail_once(command: list[str], **kwargs) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        assert kwargs["shell"] is False
        return SimpleNamespace(returncode=17, stdout="", stderr="native failure\n")

    monkeypatch.setattr(fresh.subprocess, "run", fail_once)
    with pytest.raises(
        fresh.X21FreshDevelopmentPoolTechnicalError, match="operational"
    ):
        fresh.materialize_fresh_development_pool(prepared["root"])
    status = fresh.fresh_development_pool_status(prepared["root"])
    assert status["status"] == "retryable-technical-failure"
    assert status["technical_failure_count"] == 1
    assert status["scientific_rejection"] is False
    assert not (prepared["root"] / fresh.BINDING_FILENAME).exists()
    assert calls == 1


def test_source_drift_stops_before_child_execution(
    prepared: dict, monkeypatch: pytest.MonkeyPatch
):
    changed = copy.deepcopy(prepared["source"])
    changed["commit"] = "c" * 40
    monkeypatch.setattr(fresh, "_git_source_contract", lambda _root: changed)

    def forbidden_run(*_args, **_kwargs):
        raise AssertionError("child must not run after source drift")

    monkeypatch.setattr(fresh.subprocess, "run", forbidden_run)
    with pytest.raises(fresh.X21FreshDevelopmentPoolError, match="source revision"):
        fresh.materialize_fresh_development_pool(prepared["root"])
