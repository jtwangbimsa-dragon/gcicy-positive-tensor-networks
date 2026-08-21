from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from scripts import run_gcicy_tn_study_arm as study_arm


def test_nested_signal_exit_code_is_preserved_for_workflow_retry() -> None:
    assert study_arm.child_failure_exit_code(-11) == 139
    assert study_arm.child_failure_exit_code(139) == 139
    assert study_arm.child_failure_exit_code(1) == 1


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _base_cli(tmp_path: Path) -> tuple[list[str], dict[str, Path]]:
    paths = {
        "initial": tmp_path / "initial.pt",
        "source": tmp_path / "source.npz",
        "train": tmp_path / "train_pool.npz",
        "selection": tmp_path / "selection_pool.npz",
        "development": tmp_path / "development_pool.npz",
        "arm": tmp_path / "arm",
    }
    for role, path in paths.items():
        if role != "arm":
            path.write_bytes(f"immutable-{role}".encode())
    cli = [
        "--initial-model",
        str(paths["initial"]),
        "--arm-dir",
        str(paths["arm"]),
        "--target-k",
        "8",
        "--target-d",
        "12",
        "--stage",
        "primary,0.0002,0.005,4,10,2",
        "--stage",
        "gentle,0.00005,0.001,6,12,3",
        "--train-physical-dictionary",
        "--adapter",
        "test_adapter",
        "--model-seed",
        "101",
        "--source-artifact",
        str(paths["source"]),
        "--sampling-cluster-size",
        "4",
        "--train-common-pool",
        str(paths["train"]),
        "--selection-common-pool",
        str(paths["selection"]),
        "--development-common-pool",
        str(paths["development"]),
        "--train-points",
        "8",
        "--selection-points",
        "8",
        "--development-points",
        "8",
        "--equivalence-points",
        "8",
        "--train-seed",
        "201",
        "--selection-seed",
        "202",
        "--development-seed",
        "203",
        "--equivalence-seed",
        "204",
        "--torch-seed",
        "205",
        "--transfer-seed",
        "206",
        "--batch-size",
        "4",
        "--device",
        "cuda",
        "--precision",
        "complex128",
        "--python",
        "python-for-test",
    ]
    return cli, paths


def test_complete_arm_orders_transfers_stages_audit_and_reuses_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli, paths = _base_cli(tmp_path)
    args = study_arm.parse_args(cli)
    source_sha256 = study_arm.sha256_file(paths["source"])
    monkeypatch.setattr(
        study_arm,
        "read_model_metadata",
        lambda path: study_arm.ModelMetadata(
            schema=study_arm.MODEL_SCHEMA,
            site_count=4,
            bond_dimension=8,
            architecture="shared_local_dictionary",
            trainable_physical_dictionary=False,
            precision="complex128",
            positive_floor=1.0e-4,
            source_artifact_sha256=source_sha256,
        ),
    )

    commands: list[list[str]] = []
    plateau_index = 0

    def fake_run(command: list[str], *, check: bool) -> subprocess.CompletedProcess:
        nonlocal plateau_index
        assert check is False
        commands.append(command)
        script = Path(command[1]).name
        if script == "resize_positive_tensor_network_sites.py":
            source = Path(_option(command, "--model"))
            model = Path(_option(command, "--out"))
            summary = Path(_option(command, "--summary"))
            model.parent.mkdir(parents=True, exist_ok=True)
            model.write_bytes(b"site-transfer-model")
            _write_json(
                summary,
                {
                    "schema": "positive-tensor-network-site-transfer-v1",
                    "source_model_sha256": study_arm.sha256_file(source),
                    "model_sha256": study_arm.sha256_file(model),
                    "source_site_count": 4,
                    "target_site_count": 8,
                    "bond_dimension": 8,
                    "bridge_one_sided_activation_scale": 0.03,
                    "bridge_noise_seed": 206,
                },
            )
        elif script == "expand_positive_tensor_network_bond.py":
            source = Path(_option(command, "--model"))
            model = Path(_option(command, "--out"))
            summary = Path(_option(command, "--summary"))
            model.parent.mkdir(parents=True, exist_ok=True)
            model.write_bytes(b"bond-transfer-model")
            _write_json(
                summary,
                {
                    "schema": "positive-tensor-network-bond-expansion-v1",
                    "source_model_sha256": study_arm.sha256_file(source),
                    "expanded_model_sha256": study_arm.sha256_file(model),
                    "source_bond_dimension": 8,
                    "target_bond_dimension": 12,
                    "initialization_noise": 0.0,
                    "one_sided_activation_scale": 0.03,
                    "function_preserving_by_construction": True,
                },
            )
        elif script == "audit_positive_tensor_network_equivalence.py":
            model_a = Path(_option(command, "--model-a"))
            model_b = Path(_option(command, "--model-b"))
            out = Path(_option(command, "--out"))
            _write_json(
                out,
                {
                    "schema": "positive-tensor-network-common-point-equivalence-v1",
                    "model_a_sha256": study_arm.sha256_file(model_a),
                    "model_b_sha256": study_arm.sha256_file(model_b),
                    "source_artifact_sha256": source_sha256,
                    "adapter": "test_adapter",
                    "model_seed": 101,
                    "seed": 204,
                    "points": 8,
                    "sampling_cluster_size": 4,
                    "absolute_tolerance": 1.0e-6,
                    "relative_metric_tolerance": 1.0e-6,
                    "success": True,
                },
            )
        elif script == "run_gcicy_tn_plateau_stage.py":
            plateau_index += 1
            initial_model = Path(_option(command, "--initial-model"))
            stage_dir = Path(_option(command, "--stage-dir"))
            model = stage_dir / "plateau_model.pt"
            trainer_summary = stage_dir / "plateau_summary.json"
            stage_summary = stage_dir / "stage_summary.json"
            model.parent.mkdir(parents=True, exist_ok=True)
            model.write_bytes(f"plateau-model-{plateau_index}".encode())
            _write_json(
                trainer_summary,
                {
                    "termination_reason": "validation_plateau",
                    "model_sha256": study_arm.sha256_file(model),
                },
            )
            separator = command.index("--")
            trainer_args = command[separator + 1 :]
            _write_json(
                stage_summary,
                {
                    "schema": "gcicy-tn-plateau-stage-v1",
                    "status": "validation_plateau",
                    "termination_reason": "validation_plateau",
                    "initial_model_sha256": study_arm.sha256_file(initial_model),
                    "max_rounds": int(_option(command, "--max-rounds")),
                    "first_kappa_source": _option(command, "--first-kappa-source"),
                    "continuation_kappa_source": _option(
                        command, "--continuation-kappa-source"
                    ),
                    "trainer_sha256": study_arm.sha256_file(study_arm.TRAINER),
                    "trainer_args": trainer_args,
                    "plateau_model_sha256": study_arm.sha256_file(model),
                    "plateau_summary_sha256": study_arm.sha256_file(trainer_summary),
                },
            )
        elif script == "audit_type11_positive_tensor_network.py":
            model = Path(_option(command, "--model"))
            arrays = Path(_option(command, "--arrays-out"))
            out = Path(_option(command, "--out"))
            arrays.parent.mkdir(parents=True, exist_ok=True)
            arrays.write_bytes(b"mock-development-arrays")
            _write_json(
                out,
                {
                    "schema": "type11-positive-tensor-network-blind-audit-v1",
                    "adapter": "test_adapter",
                    "model_sha256": study_arm.sha256_file(model),
                    "model_seed": 101,
                    "seed": 203,
                    "points": 8,
                    "sampling_cluster_size": 4,
                    "common_pool_sha256": study_arm.sha256_file(paths["development"]),
                    "point_arrays": {"sha256": study_arm.sha256_file(arrays)},
                    "metrics": {"validation_selection_score": 0.125},
                },
            )
        elif script == "evaluate_gcicy_metric_tail_arrays.py":
            arrays = Path(_option(command, "--arrays"))
            out = Path(_option(command, "--out"))
            _write_json(
                out,
                {
                    "schema": "gcicy-bilateral-tail-array-evaluation-v1",
                    "models": {
                        "final_model": {
                            "array_artifact_sha256": study_arm.sha256_file(arrays),
                            "metrics": {"absolute_q999": 0.25},
                        }
                    },
                },
            )
        else:
            raise AssertionError(f"unexpected child script: {script}")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(study_arm.subprocess, "run", fake_run)
    assert study_arm.run_arm(args) == 0

    assert [Path(command[1]).name for command in commands] == [
        "resize_positive_tensor_network_sites.py",
        "audit_positive_tensor_network_equivalence.py",
        "expand_positive_tensor_network_bond.py",
        "audit_positive_tensor_network_equivalence.py",
        "run_gcicy_tn_plateau_stage.py",
        "run_gcicy_tn_plateau_stage.py",
        "audit_type11_positive_tensor_network.py",
        "evaluate_gcicy_metric_tail_arrays.py",
    ]
    plateau_commands = [
        command
        for command in commands
        if Path(command[1]).name == "run_gcicy_tn_plateau_stage.py"
    ]
    assert all("--train-physical-dictionary" in command for command in plateau_commands)
    assert _option(plateau_commands[0], "--learning-rate") == "0.0002"
    assert _option(plateau_commands[1], "--learning-rate") == "5e-05"
    assert _option(plateau_commands[0], "--first-kappa-source") == "initial_model"
    assert _option(plateau_commands[1], "--first-kappa-source") == "saved_model"
    assert all(
        _option(command, "--continuation-kappa-source") == "saved_model"
        for command in plateau_commands
    )

    final_model = paths["arm"] / "final_model.pt"
    arm_summary = paths["arm"] / "arm_summary.json"
    final_summary = paths["arm"] / "final_summary.json"
    assert final_model.read_bytes() == b"plateau-model-2"
    final_payload = json.loads(final_summary.read_text(encoding="utf-8"))
    assert final_payload["status"] == "complete"
    assert final_payload["stage_names"] == ["primary", "gentle"]
    assert final_payload["final_model_sha256"] == study_arm.sha256_file(final_model)
    assert final_payload["arm_summary_sha256"] == study_arm.sha256_file(arm_summary)

    def unexpected_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
        raise AssertionError("hash-valid arm substep was rerun")

    monkeypatch.setattr(study_arm.subprocess, "run", unexpected_run)
    assert study_arm.run_arm(args) == 0


def test_failed_plateau_retains_checkpoint_and_does_not_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli, paths = _base_cli(tmp_path)
    first_stage = cli.index("--stage")
    second_stage = cli.index("--stage", first_stage + 1)
    del cli[second_stage : second_stage + 2]
    cli[cli.index("--target-k") + 1] = "4"
    cli[cli.index("--target-d") + 1] = "8"
    args = study_arm.parse_args(cli)
    source_sha256 = study_arm.sha256_file(paths["source"])
    monkeypatch.setattr(
        study_arm,
        "read_model_metadata",
        lambda path: study_arm.ModelMetadata(
            schema=study_arm.MODEL_SCHEMA,
            site_count=4,
            bond_dimension=8,
            architecture="shared_local_dictionary",
            trainable_physical_dictionary=False,
            precision="complex128",
            positive_floor=1.0e-4,
            source_artifact_sha256=source_sha256,
        ),
    )
    checkpoint: Path | None = None

    def failing_run(command: list[str], *, check: bool) -> subprocess.CompletedProcess:
        nonlocal checkpoint
        assert Path(command[1]).name == "run_gcicy_tn_plateau_stage.py"
        stage_dir = Path(_option(command, "--stage-dir"))
        checkpoint = stage_dir / "round_001" / "checkpoint.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"recoverable-checkpoint")
        return subprocess.CompletedProcess(command, 9)

    monkeypatch.setattr(study_arm.subprocess, "run", failing_run)
    assert study_arm.main(cli) != 0

    assert (
        checkpoint is not None and checkpoint.read_bytes() == b"recoverable-checkpoint"
    )
    assert not (paths["arm"] / "final_model.pt").exists()
    assert not (paths["arm"] / "arm_summary.json").exists()
    assert not (paths["arm"] / "final_summary.json").exists()
