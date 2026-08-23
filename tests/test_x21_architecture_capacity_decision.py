from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments" / "manifests" / "x21_architecture_capacity_v1.json"
SCRIPT = ROOT / "scripts" / "decide_x21_architecture_capacity_d16.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("x21_capacity_decision", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


decision = _load_module()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _training_summary(
    *,
    k: int,
    D: int,
    parameter_count: int,
    train_pool: Path,
    train_sha256: str,
    selection_pool: Path,
    selection_sha256: str,
    termination_reason: str = "validation_plateau",
    epochs: int = 2,
    runtime_seconds: float = 3.5,
) -> dict:
    return {
        "schema": "type11-positive-tensor-network-training-v1",
        "site_count": k,
        "bond_dimension": D,
        "trainable_real_parameter_count": parameter_count,
        "physical_dictionary_rank": 121,
        "trainable_physical_dictionary": False,
        "physical_dictionary_gauge": "fixed",
        "positive_floor": 1.0e-14,
        "precision": "complex64",
        "batch_size": 1024,
        "train": {
            "points": 196608,
            "seed": 86201,
            "common_pool": str(train_pool.resolve()),
            "common_pool_sha256": train_sha256,
        },
        "validation": {
            "points": 24576,
            "seed": 86202,
            "common_pool": str(selection_pool.resolve()),
            "common_pool_sha256": selection_sha256,
        },
        "history": [{"epoch": epoch} for epoch in range(epochs + 1)],
        "termination_reason": termination_reason,
        "runtime_seconds": runtime_seconds,
    }


def _write_stage(
    *,
    run_root: Path,
    stage: str,
    k: int,
    D: int,
    replicate: int,
    parameter_count: int,
    train_pool: Path,
    train_sha256: str,
    selection_pool: Path,
    selection_sha256: str,
) -> None:
    stage_dir = run_root / "jobs" / "grid" / f"k{k}_d{D}_r{replicate}" / stage
    summary = _training_summary(
        k=k,
        D=D,
        parameter_count=parameter_count,
        train_pool=train_pool,
        train_sha256=train_sha256,
        selection_pool=selection_pool,
        selection_sha256=selection_sha256,
        runtime_seconds=3.0 + 0.25 * replicate,
    )
    round_summary = stage_dir / "round_001" / "summary.json"
    plateau_summary = stage_dir / "plateau_summary.json"
    _write_json(round_summary, summary)
    _write_json(plateau_summary, summary)
    summary_sha256 = _sha256(round_summary)
    assert summary_sha256 == _sha256(plateau_summary)
    _write_json(
        stage_dir / "stage_summary.json",
        {
            "schema": "gcicy-tn-plateau-stage-v1",
            "status": "validation_plateau",
            "termination_reason": "validation_plateau",
            "plateau_round": 1,
            "rounds_completed": 1,
            "max_rounds": 5,
            "plateau_summary": "plateau_summary.json",
            "plateau_summary_sha256": summary_sha256,
            "rounds": [
                {
                    "round": 1,
                    "status": "validation_plateau",
                    "termination_reason": "validation_plateau",
                    "summary": "round_001/summary.json",
                    "summary_sha256": summary_sha256,
                }
            ],
        },
    )


def _metrics(*, k: int, D: int, replicate: int) -> dict:
    sigma = 1.2
    if k == 24 and D == 8:
        sigma = 1.0
    elif k == 24 and D == 14:
        sigma = {1: 0.90, 2: 0.99, 3: 1.0}[replicate]
    return {
        "sigma": sigma,
        "chi": 1.0 if D == 8 else 0.95,
        "absolute_log_ratio_q999": 2.0 if D == 8 else 2.05,
        "absolute_log_ratio_cvar_1pct": 1.5 if D == 8 else 1.55,
        "minimum_metric_eigenvalue": 0.25,
        "nonpositive_metric_count": 0,
    }


def _write_endpoint(
    run_root: Path,
    *,
    k: int,
    D: int,
    replicate: int,
    metrics: dict | None = None,
) -> Path:
    label = f"k{k}-d{D}-r{replicate}"
    path = (
        run_root
        / "jobs"
        / "precision_replay"
        / f"k{k}_d{D}_r{replicate}"
        / "tails.json"
    )
    _write_json(
        path,
        {
            "schema": "gcicy-bilateral-tail-array-evaluation-v1",
            "models": {label: {"metrics": metrics or _metrics(k=k, D=D, replicate=replicate)}},
        },
    )
    return path


def _campaign(tmp_path: Path) -> tuple[Path, list[str]]:
    run_root = tmp_path / "run"
    run_root.mkdir()
    train_pool = tmp_path / "frozen" / "train.npz"
    selection_pool = tmp_path / "frozen" / "selection.npz"
    train_pool.parent.mkdir()
    train_pool.write_bytes(b"registered train pool")
    selection_pool.write_bytes(b"registered selection pool")
    train_sha256 = _sha256(train_pool)
    selection_sha256 = _sha256(selection_pool)

    for k in decision.K_VALUES:
        for D in decision.D_VALUES:
            parameter_count = decision.PARAMETER_COUNTS[(k, D)]
            for replicate in decision.REPLICATES:
                _write_endpoint(run_root, k=k, D=D, replicate=replicate)
                for stage in decision.STAGES:
                    _write_stage(
                        run_root=run_root,
                        stage=stage,
                        k=k,
                        D=D,
                        replicate=replicate,
                        parameter_count=parameter_count,
                        train_pool=train_pool,
                        train_sha256=train_sha256,
                        selection_pool=selection_pool,
                        selection_sha256=selection_sha256,
                    )

    arguments = [
        "--run-root",
        str(run_root),
        "--manifest",
        str(MANIFEST),
        "--train-pool",
        str(train_pool),
        "--train-pool-sha256",
        train_sha256,
        "--selection-pool",
        str(selection_pool),
        "--selection-pool-sha256",
        selection_sha256,
        "--positive-floor",
        "1e-14",
        "--mean-sigma-ratio-max",
        "0.98",
        "--paired-sigma-wins-min",
        "2",
        "--mean-chi-ratio-max",
        "1.0",
        "--mean-q999-ratio-max",
        "1.05",
        "--mean-cvar-ratio-max",
        "1.05",
    ]
    return run_root, arguments


def test_decision_is_complete_deterministic_and_records_exposure(tmp_path):
    _, arguments = _campaign(tmp_path)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    assert decision.main([*arguments, "--out", str(first)]) == 0
    assert decision.main([*arguments, "--out", str(second)]) == 0
    assert first.read_bytes() == second.read_bytes()

    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["schema"] == decision.SCHEMA
    assert payload["complete"] is True
    assert payload["row_count"] == 12
    assert payload["completeness"]["exact_cartesian_closure"] is True
    assert payload["rows_sha256"] == decision.digest_value(payload["rows"])
    assert payload["decision"]["promote_d16"] is True
    assert payload["comparison"]["paired_sigma_strict_wins"] == 2
    assert payload["comparison"]["paired_sigma_ties"] == 1
    assert "arithmetic mean" in payload["comparison"]["aggregation"]
    assert payload["estimand"]["not_claimed"].startswith("fixed-update")

    ratio = payload["comparison"]["candidate_over_baseline_arithmetic_mean_ratios"]
    assert ratio["sigma"] == (0.90 + 0.99 + 1.0) / 3.0
    assert ratio["absolute_log_ratio_q999"] == 2.05 / 2.0
    assert ratio["absolute_log_ratio_cvar_1pct"] == 1.55 / 1.5
    for row in payload["rows"]:
        exposure = row["training_exposure"]
        assert exposure["total"]["actual_training_epochs"] == 4
        assert exposure["total"]["actual_optimizer_updates"] == 768
        assert exposure["total"]["validation_evaluations_including_epoch_zero"] == 6
        assert len(exposure["stages"]) == 2
        assert all(stage["stage_summary_sha256"] for stage in exposure["stages"])


def test_ties_are_not_wins_and_nonpositive_is_a_terminal_rejection(tmp_path):
    run_root, arguments = _campaign(tmp_path)
    for replicate in decision.REPLICATES:
        tied = _metrics(k=24, D=14, replicate=replicate)
        tied["sigma"] = 1.0
        _write_endpoint(run_root, k=24, D=14, replicate=replicate, metrics=tied)
    invalid = _metrics(k=20, D=8, replicate=1)
    invalid["minimum_metric_eigenvalue"] = -1.0e-8
    invalid["nonpositive_metric_count"] = 1
    _write_endpoint(run_root, k=20, D=8, replicate=1, metrics=invalid)

    output = tmp_path / "rejection.json"
    assert decision.main([*arguments, "--out", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["all_valid"] is False
    assert payload["comparison"]["paired_sigma_strict_wins"] == 0
    assert payload["comparison"]["paired_sigma_ties"] == 3
    assert payload["decision"] == {
        "promote_d16": False,
        "outcome": "do_not_promote",
        "extension_requirement": payload["decision"]["extension_requirement"],
    }


def test_missing_or_hash_inconsistent_evidence_fails_without_output(tmp_path):
    run_root, arguments = _campaign(tmp_path)
    missing = (
        run_root
        / "jobs"
        / "precision_replay"
        / "k24_d14_r3"
        / "tails.json"
    )
    missing.unlink()
    output = tmp_path / "missing.json"
    assert decision.main([*arguments, "--out", str(output)]) == 1
    assert not output.exists()

    _write_endpoint(run_root, k=24, D=14, replicate=3)
    stage_summary = (
        run_root
        / "jobs"
        / "grid"
        / "k24_d14_r3"
        / "precision"
        / "stage_summary.json"
    )
    payload = json.loads(stage_summary.read_text(encoding="utf-8"))
    payload["rounds"][0]["summary_sha256"] = "0" * 64
    _write_json(stage_summary, payload)
    assert decision.main([*arguments, "--out", str(output)]) == 1
    assert not output.exists()


def test_crash_window_retry_verifies_identical_output_without_overwrite(tmp_path):
    _, arguments = _campaign(tmp_path)
    output = tmp_path / "decision.json"
    assert decision.main([*arguments, "--out", str(output)]) == 0
    original = output.read_bytes()
    assert decision.main([*arguments, "--out", str(output)]) == 0
    assert output.read_bytes() == original

    same_value = json.loads(output.read_text(encoding="utf-8"))
    output.write_text(json.dumps(same_value, sort_keys=False), encoding="utf-8")
    reformatted = output.read_bytes()
    assert reformatted != original
    assert decision.main([*arguments, "--out", str(output)]) == 0
    assert output.read_bytes() == reformatted

    changed = json.loads(output.read_text(encoding="utf-8"))
    changed["decision"]["outcome"] = "tampered"
    _write_json(output, changed)
    tampered = output.read_bytes()
    assert decision.main([*arguments, "--out", str(output)]) == 1
    assert output.read_bytes() == tampered
