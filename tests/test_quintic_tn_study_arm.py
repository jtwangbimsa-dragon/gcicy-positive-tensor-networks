from __future__ import annotations

import json
import itertools
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_quintic_tn_study_arm as arm
from scripts import audit_quintic_positive_tensor_network_equivalence as equivalence


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _normalized() -> dict:
    return {
        "sigma_official_formula": 0.1,
        "weighted_rms_abs_residual": 0.2,
        "abs_residual_weighted_quantiles": {"q0.9990": 0.3, "q1.0000": 0.4},
        "abs_residual_weighted_cvar": {"cvar_0.9900": 0.35},
        "min_eigenvalue_weighted_quantiles": {"q0.0000": 1e-5},
        "nonpositive_min_eigenvalue": {"count": 0},
    }


def _metadata(
    path: Path,
    *,
    expected_source_degree: int | None = None,
) -> arm.ModelMetadata:
    assert expected_source_degree == 1
    text = str(path)
    if "01_sites_40" in text:
        sites, bond, precision, inherited = 40, 6, "complex128", None
    elif "02_bond_10" in text:
        sites, bond, precision, inherited = 40, 10, "complex64", 6
    elif "03_bond_14" in text or "stages" in text:
        sites, bond, precision, inherited = 40, 14, "complex64", 10
    else:
        sites, bond, precision, inherited = 20, 6, "complex128", None
    return arm.ModelMetadata(
        source_degree=1,
        site_count=sites,
        bond_dimension=bond,
        dictionary_rank=21,
        output_dimension=5,
        precision=precision,
        positive_floor=1e-4,
        transfer_implementation="vectorized",
        fermat_phase_charge_multiplicity=0,
        fermat_s5_orbit_tying=False,
        fermat_two_site_blocking=False,
        inherited_bond_dimension=inherited,
    )


def _real_shape_payload(*, include_source_degree: bool) -> dict:
    payload = {
        "schema": arm.MODEL_SCHEMA,
        "architecture": "shared_local_dictionary",
        "site_count": 20,
        "bond_dimension": 6,
        "physical_dictionary_rank": 25,
        "precision": "complex64",
        "positive_floor": 1e-4,
        "target_normalization": 1.0 / (20 * 3.141592653589793),
        "trainable_physical_dictionary": True,
        "transfer_implementation": "vectorized",
        "state_dict": {
            "reference_h": torch.eye(5, dtype=torch.complex64),
            "physical_dictionary": torch.zeros((25, 5, 5), dtype=torch.complex64),
            "coefficient_cores.0": torch.zeros((1, 6, 25), dtype=torch.complex64),
            **{
                f"coefficient_cores.{index}": torch.zeros(
                    (6, 6, 25), dtype=torch.complex64
                )
                for index in range(1, 19)
            },
            "coefficient_cores.19": torch.zeros((6, 1, 25), dtype=torch.complex64),
        },
    }
    if include_source_degree:
        payload["source_degree"] = 1
        payload["output_dimension"] = 5
    return payload


def test_legacy_anchor_metadata_requires_and_validates_explicit_source_degree(
    tmp_path: Path,
) -> None:
    model = tmp_path / "legacy_anchor.pt"
    torch.save(_real_shape_payload(include_source_degree=False), model)

    metadata = arm.load_model_metadata(model, expected_source_degree=1)
    assert metadata.source_degree == 1
    assert metadata.site_count == 20
    assert metadata.bond_dimension == 6
    assert metadata.dictionary_rank == 25
    assert metadata.output_dimension == 5

    with pytest.raises(arm.ArmError, match="explicit --source-degree is required"):
        arm.load_model_metadata(model)
    with pytest.raises(arm.ArmError, match="source dimension disagrees"):
        arm.load_model_metadata(model, expected_source_degree=2)


def test_current_metadata_is_not_overridden_by_explicit_source_degree(
    tmp_path: Path,
) -> None:
    model = tmp_path / "current.pt"
    torch.save(_real_shape_payload(include_source_degree=True), model)

    metadata = arm.load_model_metadata(model, expected_source_degree=1)
    assert metadata.output_dimension == 5
    with pytest.raises(arm.ArmError, match="saved source_degree does not match"):
        arm.load_model_metadata(model, expected_source_degree=2)


def test_legacy_adapter_is_create_only_hash_bound_and_audit_ready(
    tmp_path: Path,
) -> None:
    source = tmp_path / "legacy.pt"
    torch.save(_real_shape_payload(include_source_degree=False), source)
    source_sha = arm.sha256_file(source)
    source_payload = torch.load(source, map_location="cpu", weights_only=False)

    adapted, summary = arm.prepare_legacy_initial_model(
        source,
        tmp_path / "arm",
        expected_source_degree=1,
    )
    assert summary is not None
    assert arm.sha256_file(source) == source_sha
    assert summary == json.loads(
        (tmp_path / "arm" / "legacy_initial_adapter" / "summary.json").read_text()
    )
    assert summary["source_model_sha256"] == source_sha
    assert summary["adapted_model_sha256"] == arm.sha256_file(adapted)
    assert summary["optimizer_updates"] == 0
    assert summary["state_dict_exact_copy"] is True
    assert summary["added_metadata"] == {
        "source_degree": 1,
        "output_dimension": 5,
    }
    assert not any("/" in str(value) for value in summary.values())

    adapted_payload = torch.load(adapted, map_location="cpu", weights_only=False)
    assert adapted_payload["source_degree"] == 1
    assert adapted_payload["output_dimension"] == 5
    assert "total_degree" not in adapted_payload
    assert "source_model" not in adapted_payload["legacy_metadata_adapter"]
    assert adapted_payload["legacy_metadata_adapter"]["optimizer_updates"] == 0
    assert adapted_payload["state_dict"].keys() == source_payload["state_dict"].keys()
    for name, source_tensor in source_payload["state_dict"].items():
        assert torch.equal(adapted_payload["state_dict"][name], source_tensor)

    resized_payload = dict(adapted_payload)
    resized_payload["site_count"] = 40
    equivalence.validate_model_pair(
        adapted_payload,
        resized_payload,
        allow_different_site_counts=True,
        allow_different_precisions=False,
    )

    reused, reused_summary = arm.prepare_legacy_initial_model(
        source,
        tmp_path / "arm",
        expected_source_degree=1,
    )
    assert reused == adapted
    assert arm.sha256_file(reused) == summary["adapted_model_sha256"]
    assert reused_summary == summary


def test_legacy_adapter_hash_is_location_independent_and_rejects_conflicts(
    tmp_path: Path,
) -> None:
    source_a = tmp_path / "host-a" / "legacy.pt"
    source_b = tmp_path / "host-b" / "legacy-copy.pt"
    source_a.parent.mkdir()
    source_b.parent.mkdir()
    torch.save(_real_shape_payload(include_source_degree=False), source_a)
    source_b.write_bytes(source_a.read_bytes())

    model_a, summary_a = arm.prepare_legacy_initial_model(
        source_a, tmp_path / "run-a", expected_source_degree=1
    )
    model_b, summary_b = arm.prepare_legacy_initial_model(
        source_b, tmp_path / "run-b", expected_source_degree=1
    )
    assert arm.sha256_file(model_a) == arm.sha256_file(model_b)
    assert summary_a == summary_b

    summary_path = tmp_path / "run-a" / "legacy_initial_adapter" / "summary.json"
    summary_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(arm.ArmError, match="summary conflict"):
        arm.prepare_legacy_initial_model(
            source_a, tmp_path / "run-a", expected_source_degree=1
        )


def _cli(tmp_path: Path) -> tuple[list[str], dict[str, Path]]:
    paths = {
        "initial": tmp_path / "initial.pt",
        "arm": tmp_path / "arm",
        "source": tmp_path / "source",
        "pullbacks": tmp_path / "pullbacks",
    }
    torch.save(_real_shape_payload(include_source_degree=True), paths["initial"])
    for relative in ("training_data/dataset.npz", "training_data/basis.pickle"):
        path = paths["source"] / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    for name in (
        "report.json",
        "train_pullbacks.npy",
        "validation_pullbacks.npy",
        "validation_official_fs_metrics.npy",
    ):
        path = paths["pullbacks"] / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    cli = [
        "--initial-model",
        str(paths["initial"]),
        "--arm-dir",
        str(paths["arm"]),
        "--source-run-dir",
        str(paths["source"]),
        "--pullbacks-dir",
        str(paths["pullbacks"]),
        "--target-k",
        "40",
        "--target-d",
        "14",
        "--skip-blind-audit",
        "--source-degree",
        "1",
        "--positive-floor",
        "1e-4",
        "--initialization-noise",
        "0",
        "--fixed-log-kappa",
        "-4.0",
        "--transfer-step",
        "sites:40",
        "--transfer-step",
        "bond:10",
        "--transfer-step",
        "bond:14",
        "--stage",
        "channels,new_channels,0.001,0.01,1,1,3",
        "--stage",
        "cores,cores,0.0005,0.01,1,1,3",
        "--stage",
        "joint,joint,0.0001,0.01,1,1,3",
        "--batch-size",
        "1024",
        "--device",
        "cuda",
        "--precision",
        "complex64",
    ]
    return cli, paths


def test_ordered_transfers_each_audited_then_scoped_stages_and_atomic_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli, paths = _cli(tmp_path)
    args = arm.parse_args(cli)
    monkeypatch.setattr(arm, "load_model_metadata", _metadata)
    labels: list[str] = []

    def fake_child(command: list[str], *, label: str) -> None:
        labels.append(label)
        script = Path(command[1]).name
        if script == arm.RESIZE.name:
            source = Path(_option(command, "--model"))
            model, summary = Path(_option(command, "--out")), Path(
                _option(command, "--summary")
            )
            model.parent.mkdir(parents=True, exist_ok=True)
            model.write_bytes(b"sites40")
            _write_json(
                summary,
                {
                    "schema": "positive-tensor-network-site-transfer-v1",
                    "source_model_sha256": arm.sha256_file(source),
                    "target_site_count": 40,
                    "model_sha256": arm.sha256_file(model),
                },
            )
        elif script == arm.EXPAND.name:
            source = Path(_option(command, "--model"))
            model, summary = Path(_option(command, "--out")), Path(
                _option(command, "--summary")
            )
            target = int(_option(command, "--bond-dimension"))
            model.parent.mkdir(parents=True, exist_ok=True)
            model.write_bytes(f"bond{target}".encode())
            _write_json(
                summary,
                {
                    "schema": "positive-tensor-network-bond-expansion-v1",
                    "source_model_sha256": arm.sha256_file(source),
                    "target_bond_dimension": target,
                    "expanded_model_sha256": arm.sha256_file(model),
                },
            )
        elif script == arm.AUDIT.name:
            model_a, model_b = Path(_option(command, "--model-a")), Path(
                _option(command, "--model-b")
            )
            _write_json(
                Path(_option(command, "--out")),
                {
                    "schema": "quintic-positive-tensor-network-common-point-equivalence-v1",
                    "success": True,
                    "model_a_sha256": arm.sha256_file(model_a),
                    "model_b_sha256": arm.sha256_file(model_b),
                },
            )
        elif script == arm.PLATEAU.name:
            stage_dir = Path(_option(command, "--stage-dir"))
            scope = _option(command, "--parameter-scope")
            input_model = Path(_option(command, "--initial-model"))
            model, report = (
                stage_dir / "plateau_model.pt",
                stage_dir / "plateau_report.json",
            )
            model.parent.mkdir(parents=True, exist_ok=True)
            model.write_bytes(f"stage-{scope}".encode())
            _write_json(
                report,
                {
                    "schema": arm.REPORT_SCHEMA,
                    "termination_reason": "completed_requested_epochs",
                    "evaluation_scope": "development_only",
                    "training": {
                        "best_selection_score": 1.0,
                        "final_validation": {"normalized_volume": _normalized()},
                    },
                    "artifacts": {"model_sha256": arm.sha256_file(model)},
                    "environment": {"device_memory": {"maximum_allocated_bytes": 123}},
                },
            )
            separator = command.index("--")
            trainer_args = command[separator + 1 :]
            assert _option(trainer_args, "--fixed-log-kappa") == "-4.0"
            assert _option(trainer_args, "--source-degree") == "1"
            inherited = (
                int(_option(command, "--inherited-bond-dimension"))
                if "--inherited-bond-dimension" in command[:separator]
                else 0
            )
            _write_json(
                stage_dir / "stage_summary.json",
                {
                    "schema": "quintic-tn-plateau-stage-v1",
                    "status": "validation_plateau",
                    "initial_model_sha256": arm.sha256_file(input_model),
                    "parameter_scope": scope,
                    "trainer_args": trainer_args,
                    "max_rounds": int(_option(command, "--max-rounds")),
                    "plateau_patience_rounds": int(
                        _option(command, "--plateau-patience-rounds")
                    ),
                    "min_relative_gain": float(_option(command, "--min-relative-gain")),
                    "inherited_bond_dimension": inherited,
                    "accept_round_cap": True,
                    "plateau_model_sha256": arm.sha256_file(model),
                    "plateau_report_sha256": arm.sha256_file(report),
                    "selected_round": 1,
                    "selected_best_selection_score": 1.0,
                },
            )
        else:
            raise AssertionError(script)

    monkeypatch.setattr(arm, "_run_child", fake_child)
    assert arm.run_arm(args, python_executable="python-test") == 0
    assert labels[:6] == [
        "transfer 1 (sites:40)",
        "equivalence audit after transfer 1",
        "transfer 2 (bond:10)",
        "equivalence audit after transfer 2",
        "transfer 3 (bond:14)",
        "equivalence audit after transfer 3",
    ]
    assert labels[6:] == [
        "plateau stage channels",
        "plateau stage cores",
        "plateau stage joint",
    ]
    result = json.loads((paths["arm"] / "final_summary.json").read_text())
    assert result["target"] == {"k": 40, "D": 14}
    assert result["stage_scopes"] == ["new_channels", "cores", "joint"]
    assert result["evaluation_scope"] == "development_only"
    assert result["development_metrics"]["nonpositive_metric_count"] == 0
    assert result["development_metrics"]["minimum_metric_eigenvalue"] > 0
    assert result["arm_summary_sha256"] == arm.sha256_file(
        paths["arm"] / "arm_summary.json"
    )

    labels.clear()
    assert arm.run_arm(args, python_executable="python-test") == 0
    assert labels == []


def test_cli_requires_explicit_blind_skip_and_preserves_step_order(
    tmp_path: Path,
) -> None:
    cli, _ = _cli(tmp_path)
    without_skip = [token for token in cli if token != "--skip-blind-audit"]
    with pytest.raises(SystemExit):
        arm.parse_args(without_skip)
    args = arm.parse_args(cli)
    assert [(step.kind, step.target) for step in args.transfer_step] == [
        ("sites", 40),
        ("bond", 10),
        ("bond", 14),
    ]
    assert arm.child_failure_exit_code(-11) == 139


def test_every_wrapper_or_trainer_command_in_registered_manifest_parses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import train_quintic_positive_tensor_network_same_points as trainer

    manifest_path = (
        arm.ROOT
        / "experiments"
        / "manifests"
        / "quintic_kd_progressive_joint_paths_v1.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parsed = 0
    for entry in [*manifest.get("jobs", []), *manifest.get("job_templates", [])]:
        command = entry.get("command", [])
        if len(command) < 2 or Path(command[1]).name not in {
            Path(arm.__file__).name,
            Path(trainer.__file__).name,
        }:
            continue
        matrix = entry.get("matrix", {})
        names = list(matrix)
        rows = itertools.product(*(matrix[name] for name in names)) if names else [()]
        for row in rows:
            replacements = dict(zip(names, row))
            expanded = []
            for token in command[2:]:
                for name, value in replacements.items():
                    token = token.replace("{{" + name + "}}", str(value))
                expanded.append(token)
            if Path(command[1]).name == Path(arm.__file__).name:
                arm.parse_args(expanded)
            else:
                monkeypatch.setattr(sys, "argv", [command[1], *expanded])
                trainer.parse_args()
            parsed += 1
    assert parsed > 21
