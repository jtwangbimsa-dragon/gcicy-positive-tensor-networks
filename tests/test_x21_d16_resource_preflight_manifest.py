from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT
    / "experiments"
    / "manifests"
    / "x21_k24_d16_resource_preflight_v1.json"
)
BASE_MANIFEST = (
    ROOT
    / "experiments"
    / "manifests"
    / "x21_architecture_capacity_v1.json"
)


def _load_module(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


workflow = _load_module(
    "x21_d16_preflight_workflow",
    "gcicy_metric/pipeline/experiment_workflow.py",
)
certificate = _load_module(
    "x21_d16_preflight_certificate",
    "scripts/certify_x21_resource_preflight.py",
)


def _plan(tmp_path: Path):
    base_run_root = tmp_path / "runs" / "x21-architecture-capacity-v1"
    run_root = tmp_path / "runs" / "x21-k24-d16-resource-preflight-v1"
    return workflow.load_workflow_plan(
        MANIFEST,
        repo_root=ROOT,
        frozen_root=tmp_path / "frozen",
        run_root=run_root,
        variable_overrides={"BASE_CAPACITY_RUN_ROOT": str(base_run_root)},
        python_executable=sys.executable,
    )


def _option(command: tuple[str, ...] | list[str], flag: str) -> str:
    index = command.index(flag)
    return command[index + 1]


def test_manifest_is_resource_only_and_d14_gated(tmp_path):
    plan = _plan(tmp_path)
    assert len(plan.jobs) == 10
    assert Counter(job.phase for job in plan.jobs) == {
        "d14-prerequisite": 1,
        "prepare": 3,
        "initialize-d16": 4,
        "resource-preflight-d16": 1,
        "resource-certification": 1,
    }
    assert sum(job.result is not None for job in plan.jobs) == 1

    prerequisite = plan.jobs_by_id["certify-d14-prerequisite"]
    assert Path(_option(prerequisite.command, "--source-run-root")) != plan.run_root
    assert _option(prerequisite.command, "--source-campaign-id") == (
        "x21-architecture-capacity-v1"
    )
    assert Path(_option(prerequisite.command, "--expected-manifest")) == BASE_MANIFEST
    assert _option(prerequisite.command, "--job-id") == (
        "resource-preflight-k24-d14"
    )
    assert _option(prerequisite.command, "--bond-dimension") == "14"
    assert _option(prerequisite.command, "--allocated-max-bytes") == "21474836480"
    assert _option(prerequisite.command, "--reserved-max-bytes") == "23085449216"
    assert plan.jobs_by_id["prepare-q121-k4"].needs == (prerequisite.id,)

    def dependencies(job_id: str) -> set[str]:
        seen: set[str] = set()
        pending = list(plan.jobs_by_id[job_id].needs)
        while pending:
            dependency = pending.pop()
            if dependency in seen:
                continue
            seen.add(dependency)
            pending.extend(plan.jobs_by_id[dependency].needs)
        return seen

    for job in plan.jobs:
        if job.id != prerequisite.id:
            assert prerequisite.id in dependencies(job.id)

    raw = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert raw["scientific_policy"]["accuracy_jobs"] == 0
    assert raw["scientific_policy"]["target"] == {
        "k": 24,
        "D": 16,
        "physical_dictionary_rank": 121,
        "trainable_real_parameter_count": 1370688,
        "precision": "complex64",
        "batch_size": 1024,
        "epochs": 1,
    }
    serialized = json.dumps(raw).lower()
    assert "precision-replay" not in serialized
    assert "development_pool" not in serialized
    assert "evaluate_gcicy_metric_tail_arrays.py" not in serialized
    assert "audit_type11_positive_tensor_network.py" not in serialized


def test_d16_preflight_has_exact_full_batch_resource_and_validity_gates(tmp_path):
    plan = _plan(tmp_path)
    job = plan.jobs_by_id["resource-preflight-k24-d16"]
    assert Path(job.command[1]).name == "train_type11_positive_tensor_network.py"
    assert _option(job.command, "--site-count") == "24"
    assert _option(job.command, "--bond-dimension") == "16"
    assert _option(job.command, "--precision") == "complex64"
    assert _option(job.command, "--epochs") == "1"
    assert _option(job.command, "--batch-size") == "1024"
    assert _option(job.command, "--train-points") == "196608"
    assert _option(job.command, "--validation-points") == "24576"
    assert "--train-physical-dictionary" not in job.command
    assert job.resume is not None

    gate_rows = {(gate["field"], gate.get("equals"), gate.get("gt"), gate.get("le")) for gate in job.json_gates}
    assert {
        ("site_count", 24, None, None),
        ("bond_dimension", 16, None, None),
        ("trainable_real_parameter_count", 1370688, None, None),
        ("physical_dictionary_rank", 121, None, None),
        ("trainable_physical_dictionary", False, None, None),
        ("physical_dictionary_gauge", "fixed", None, None),
        ("positive_floor", 1.0e-14, None, None),
        ("best_validation.minimum_metric_eigenvalue", None, 0, None),
        ("runtime_seconds", None, 0, None),
        ("device_memory.maximum_allocated_bytes", None, None, 24159191040),
        ("device_memory.maximum_reserved_bytes", None, None, 25232932864),
    } <= gate_rows
    assert 1370688 == 2 * 121 * (2 * 16 + (24 - 2) * 16**2)

    certification = plan.jobs_by_id["certify-d16-resource-preflight"]
    assert certification.needs == (job.id,)
    assert _option(certification.command, "--allocated-max-bytes") == "24159191040"
    assert _option(certification.command, "--reserved-max-bytes") == "25232932864"
    assert certification.scientific["accuracy_job"] is False
    assert certification.result is not None
    assert set(certification.result["fields"]) == {
        "exit_code",
        "maximum_allocated_bytes",
        "maximum_reserved_bytes",
        "runtime_seconds",
        "minimum_metric_eigenvalue",
        "trainable_real_parameter_count",
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _certificate_fixture(tmp_path: Path) -> tuple[list[str], Path, Path, Path]:
    run_root = tmp_path / "source-run"
    summary_path = run_root / "resource_preflight" / "k24_d16" / "summary.json"
    train_pool = tmp_path / "train.npz"
    selection_pool = tmp_path / "selection.npz"
    train_pool.write_bytes(b"train")
    selection_pool.write_bytes(b"selection")
    train_hash = _sha256(train_pool)
    selection_hash = _sha256(selection_pool)
    summary = {
        "schema": "type11-positive-tensor-network-training-v1",
        "termination_reason": "completed_requested_epochs",
        "site_count": 24,
        "bond_dimension": 16,
        "trainable_real_parameter_count": 1370688,
        "precision": "complex64",
        "physical_dictionary_rank": 121,
        "trainable_physical_dictionary": False,
        "physical_dictionary_gauge": "fixed",
        "positive_floor": 1.0e-14,
        "epochs": 1,
        "batch_size": 1024,
        "train": {
            "points": 196608,
            "seed": 86201,
            "common_pool": str(train_pool.resolve()),
            "common_pool_sha256": train_hash,
        },
        "validation": {
            "points": 24576,
            "seed": 86202,
            "common_pool": str(selection_pool.resolve()),
            "common_pool_sha256": selection_hash,
        },
        "device_memory": {
            "maximum_allocated_bytes": 20 * 2**30,
            "maximum_reserved_bytes": 21 * 2**30,
        },
        "runtime_seconds": 9.5,
        "best_validation": {"minimum_metric_eigenvalue": 0.125},
        "teacher_artifact": None,
    }
    _write_json(summary_path, summary)
    plan_sha256 = "a" * 64
    manifest_sha256 = _sha256(MANIFEST)
    _write_json(
        run_root / ".gcicy-experiment-root",
        {
            "campaign_id": "x21-k24-d16-resource-preflight-v1",
            "plan_sha256": plan_sha256,
            "manifest_sha256": manifest_sha256,
        },
    )
    _write_json(
        run_root / ".workflow" / "plan.lock.json",
        {
            "campaign_id": "x21-k24-d16-resource-preflight-v1",
            "plan_sha256": plan_sha256,
            "manifest_sha256": manifest_sha256,
        },
    )
    state_path = run_root / ".workflow" / "jobs" / "resource-preflight-k24-d16.json"
    _write_json(
        state_path,
        {
            "campaign_id": "x21-k24-d16-resource-preflight-v1",
            "plan_sha256": plan_sha256,
            "job_id": "resource-preflight-k24-d16",
            "status": "succeeded",
            "attempts": [{"returncode": 0}],
            "json_gates": [{"passed": True}],
            "outputs": [
                {"path": str(summary_path.resolve()), "sha256": _sha256(summary_path)}
            ],
        },
    )
    arguments = [
        "--source-run-root", str(run_root),
        "--source-campaign-id", "x21-k24-d16-resource-preflight-v1",
        "--expected-manifest", str(MANIFEST),
        "--job-id", "resource-preflight-k24-d16",
        "--source-summary", str(summary_path),
        "--site-count", "24",
        "--bond-dimension", "16",
        "--parameter-count", "1370688",
        "--positive-floor", "1e-14",
        "--train-pool", str(train_pool),
        "--train-pool-sha256", train_hash,
        "--selection-pool", str(selection_pool),
        "--selection-pool-sha256", selection_hash,
        "--allocated-max-bytes", "24159191040",
        "--reserved-max-bytes", "25232932864",
    ]
    return arguments, summary_path, state_path, run_root


def test_certificate_is_finite_hash_bound_and_idempotent_after_crash_window(tmp_path):
    arguments, _, _, _ = _certificate_fixture(tmp_path)
    output = tmp_path / "certificate.json"
    assert certificate.main([*arguments, "--out", str(output)]) == 0
    original = output.read_bytes()
    assert certificate.main([*arguments, "--out", str(output)]) == 0
    assert output.read_bytes() == original
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    assert payload["exit_code"] == 0
    assert payload["all_summary_numbers_finite"] is True
    assert payload["strictly_positive_validation_metric"] is True
    assert payload["maximum_allocated_bytes"] == 20 * 2**30
    assert payload["maximum_reserved_bytes"] == 21 * 2**30
    assert payload["runtime_seconds"] == 9.5
    assert payload["inputs"]["job_state"]["sha256"]
    assert payload["inputs"]["summary"]["sha256"]

    output.write_text(json.dumps(payload, sort_keys=False), encoding="utf-8")
    reformatted = output.read_bytes()
    assert reformatted != original
    assert certificate.main([*arguments, "--out", str(output)]) == 0
    assert output.read_bytes() == reformatted

    payload["status"] = "tampered"
    _write_json(output, payload)
    tampered = output.read_bytes()
    assert certificate.main([*arguments, "--out", str(output)]) == 1
    assert output.read_bytes() == tampered


def test_certificate_rejects_nonfinite_summary_without_output(tmp_path):
    arguments, summary_path, state_path, _ = _certificate_fixture(tmp_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["runtime_seconds"] = float("nan")
    _write_json(summary_path, summary)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["outputs"][0]["sha256"] = _sha256(summary_path)
    _write_json(state_path, state)
    output = tmp_path / "nonfinite.json"
    assert certificate.main([*arguments, "--out", str(output)]) == 1
    assert not output.exists()
