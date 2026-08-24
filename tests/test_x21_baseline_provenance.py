from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest
import torch

from gcicy_metric.pipeline.experiment_workflow import sha256_file
from gcicy_metric.pipeline.x21_auto_research import (
    BASELINE_COMPLETION_SCHEMA,
    RECOVERY_EVIDENCE_SCHEMA,
    X21CampaignStore,
    X21AutoResearchError,
    validate_baseline_completion_evidence,
    validate_recovery_evidence,
)
from gcicy_metric.pipeline.x21_baseline_provenance import (
    RECOVERY_RAW_BUNDLE_SCHEMA,
    assemble_baseline_family,
    assemble_parent_bindings,
    main as provenance_main,
    normalize_baseline_completion,
    normalize_exact_recovery,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments" / "protocols" / "x21_auto_research_v1.json"


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _model_payload(protocol: dict) -> dict:
    return {
        "schema": "type11-positive-tensor-network-v1",
        "model_seed": 20260802,
        "sampling_cluster_size": 6,
        "site_count": 20,
        "bond_dimension": 14,
        "precision": "complex64",
        "physical_dictionary_rank": 121,
        "trainable_physical_dictionary": False,
        "physical_dictionary_gauge": "fixed",
        "source_artifact_sha256": protocol["data_contract"]["source_artifact_sha256"],
        "train_common_pool_sha256": protocol["data_contract"]["train_pool_sha256"],
        "validation_common_pool_sha256": protocol["data_contract"][
            "selection_pool_sha256"
        ],
        "state_dict": {"core": torch.ones(1, dtype=torch.complex64)},
    }


def _semantics(seed: int) -> dict:
    return {
        "model": {
            "site_count": 20,
            "bond_dimension": 14,
            "precision": "complex64",
            "physical_dictionary_rank": 121,
            "trainable_physical_dictionary": False,
        },
        "optimization": {"torch_seed": seed},
        "implementation": {"trainer_sha256": "f" * 64, "device": "cpu-test"},
    }


def _checkpoint(
    path: Path,
    *,
    protocol: dict,
    seed: int,
    initial_model_sha256: str,
    epoch: int,
    history: list[dict] | None = None,
) -> dict:
    payload = {
        "schema": "type11-positive-tensor-network-checkpoint-v1",
        "epoch": epoch,
        "next_epoch": epoch + 1,
        "current_model_state_dict": {"core": torch.ones(1)},
        "best_state_dict": {"core": torch.ones(1)},
        "optimizer_state_dict": {"state": {}},
        "permutation_generator_state": torch.tensor([1, 2, 3]),
        "cpu_rng_state": torch.tensor([4, 5, 6]),
        "cuda_rng_state": None,
        "history": history or [{"epoch": 0}, {"epoch": epoch}],
        "training_semantics": _semantics(seed),
        "frozen_input_hashes": {
            "source_artifact_sha256": protocol["data_contract"][
                "source_artifact_sha256"
            ],
            "teacher_artifact_sha256": None,
            "initial_model_sha256": initial_model_sha256,
            "train_common_pool_sha256": protocol["data_contract"]["train_pool_sha256"],
            "validation_common_pool_sha256": protocol["data_contract"][
                "selection_pool_sha256"
            ],
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload


def _stage(
    stage_dir: Path,
    *,
    protocol: dict,
    seed: int,
    initial_model: Path,
    epoch: int,
    resumed_from: str | None = None,
    history: list[dict] | None = None,
) -> dict:
    initial_model_sha256 = sha256_file(initial_model)
    round_dir = stage_dir / "round_001"
    round_dir.mkdir(parents=True, exist_ok=True)
    terminal_model = round_dir / "model.pt"
    torch.save(_model_payload(protocol), terminal_model)
    terminal_checkpoint = round_dir / "checkpoint.pt"
    _checkpoint(
        terminal_checkpoint,
        protocol=protocol,
        seed=seed,
        initial_model_sha256=initial_model_sha256,
        epoch=epoch,
        history=history,
    )
    model_sha = sha256_file(terminal_model)
    checkpoint_sha = sha256_file(terminal_checkpoint)
    terminal_summary = round_dir / "summary.json"
    summary = {
        "schema": "type11-positive-tensor-network-training-v1",
        "model_seed": 20260802,
        "sampling_cluster_size": 6,
        "torch_seed": seed,
        "site_count": 20,
        "bond_dimension": 14,
        "precision": "complex64",
        "physical_dictionary_rank": 121,
        "trainable_physical_dictionary": False,
        "physical_dictionary_gauge": "fixed",
        "trainable_real_parameter_count": 860552,
        "termination_reason": "validation_plateau",
        "model_sha256": model_sha,
        "source_artifact_sha256": protocol["data_contract"]["source_artifact_sha256"],
        "train": {
            "points": protocol["data_contract"]["train_points"],
            "common_pool_sha256": protocol["data_contract"]["train_pool_sha256"],
        },
        "validation": {
            "points": protocol["data_contract"]["selection_points"],
            "common_pool_sha256": protocol["data_contract"]["selection_pool_sha256"],
        },
        "checkpoint": {
            "sha256": checkpoint_sha,
            "last_completed_validation_epoch": epoch,
            "resume_kind": "crash_recovery" if resumed_from else "fresh_training",
            "resumed_from_sha256": resumed_from,
        },
    }
    _json(terminal_summary, summary)
    summary_sha = sha256_file(terminal_summary)
    plateau_model = stage_dir / "plateau_model.pt"
    plateau_summary = stage_dir / "plateau_summary.json"
    shutil.copyfile(terminal_model, plateau_model)
    shutil.copyfile(terminal_summary, plateau_summary)
    stage_summary = stage_dir / "stage_summary.json"
    _json(
        stage_summary,
        {
            "schema": "gcicy-tn-plateau-stage-v1",
            "status": "validation_plateau",
            "termination_reason": "validation_plateau",
            "plateau_round": 1,
            "rounds_completed": 1,
            "max_rounds": 5,
            "initial_model": str(initial_model.resolve()),
            "initial_model_sha256": initial_model_sha256,
            "plateau_model": "plateau_model.pt",
            "plateau_model_sha256": model_sha,
            "plateau_summary": "plateau_summary.json",
            "plateau_summary_sha256": summary_sha,
            "rounds": [
                {
                    "round": 1,
                    "status": "validation_plateau",
                    "termination_reason": "validation_plateau",
                    "initial_model": str(initial_model.resolve()),
                    "initial_model_sha256": initial_model_sha256,
                    "model": "round_001/model.pt",
                    "model_sha256": model_sha,
                    "summary": "round_001/summary.json",
                    "summary_sha256": summary_sha,
                    "checkpoint": "round_001/checkpoint.pt",
                    "checkpoint_sha256": checkpoint_sha,
                }
            ],
        },
    )
    return {
        "model": plateau_model,
        "summary": plateau_summary,
        "stage_summary": stage_summary,
        "checkpoint": terminal_checkpoint,
    }


def _command(*, stage: Path, initial_model: Path, seed: int) -> list[str]:
    return [
        "python",
        "scripts/run_gcicy_tn_plateau_stage.py",
        "--initial-model",
        str(initial_model.resolve()),
        "--stage-dir",
        str(stage.resolve()),
        "--max-rounds",
        "5",
        "--first-kappa-source",
        "saved_model",
        "--continuation-kappa-source",
        "saved_model",
        "--",
        "--site-count",
        "20",
        "--bond-dimension",
        "14",
        "--precision",
        "complex64",
        "--torch-seed",
        str(seed),
    ]


def _workflow(
    root: Path,
    *,
    contract: dict,
    seed: int,
    initial_model: Path,
    stage: Path,
    status: str,
    outputs: dict | None = None,
    returncode: int = 139,
    attempt_count: int = 1,
) -> None:
    _json(
        root / ".workflow" / "plan.lock.json",
        {
            "schema": "gcicy-experiment-plan-lock-v1",
            "campaign_id": contract["source_campaign_id"],
            "plan_sha256": contract["source_plan_sha256"],
            "jobs": [
                {
                    "id": contract["source_job_id"],
                    "phase": "capacity-grid",
                    "needs": [],
                    "digest": contract["source_job_digest"],
                }
            ],
        },
    )
    state = {
        "schema": "gcicy-experiment-job-state-v1",
        "campaign_id": contract["source_campaign_id"],
        "plan_sha256": contract["source_plan_sha256"],
        "job_id": contract["source_job_id"],
        "job_digest": contract["source_job_digest"],
        "status": status,
        "scientific": {
            "parameter_scope": "cores-only",
            "factors": {
                "k": 20,
                "D": 14,
                "replicate": seed - 8_660_000,
                "trainable_real_parameter_count": 860552,
            },
        },
        "attempts": [
            {
                "attempt": attempt,
                "command": _command(
                    stage=stage, initial_model=initial_model, seed=seed
                ),
                "returncode": returncode,
            }
            for attempt in range(1, attempt_count + 1)
        ],
    }
    if status == "failed":
        state["returncode"] = returncode
    if outputs is not None:
        state["outputs"] = [
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in (
                outputs["model"],
                outputs["summary"],
                outputs["stage_summary"],
            )
        ]
    _json(root / ".workflow" / "jobs" / f"{contract['source_job_id']}.json", state)


def _raw_protocol(tmp_path: Path) -> tuple[dict, Path]:
    value = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    path = tmp_path / "protocol.json"
    return value, path


def test_completion_normalizes_real_workflow_artifacts(tmp_path: Path) -> None:
    protocol, protocol_path = _raw_protocol(tmp_path)
    contract = protocol["baseline_provenance_contracts"][0]
    contract["source_plan_sha256"] = "a" * 64
    contract["source_job_digest"] = "b" * 64
    root = tmp_path / "workflow"
    initial_model = (
        root / "jobs" / "grid" / "k20_d14_r1" / "primary" / "plateau_model.pt"
    )
    initial_model.parent.mkdir(parents=True)
    torch.save(_model_payload(protocol), initial_model)
    stage_dir = root / "jobs" / "grid" / "k20_d14_r1" / "precision"
    endpoint = _stage(
        stage_dir,
        protocol=protocol,
        seed=8660001,
        initial_model=initial_model,
        epoch=83,
    )
    _workflow(
        root,
        contract=contract,
        seed=8660001,
        initial_model=initial_model,
        stage=stage_dir,
        status="succeeded",
        outputs=endpoint,
        returncode=0,
    )
    _json(protocol_path, protocol)

    certificate = normalize_baseline_completion(
        protocol_path=protocol_path,
        workflow_root=root,
        seed=8660001,
    )

    assert certificate["schema"] == BASELINE_COMPLETION_SCHEMA
    assert certificate["output_model_sha256"] == sha256_file(endpoint["model"])
    assert certificate["output_checkpoint_sha256"] == sha256_file(
        endpoint["checkpoint"]
    )
    assert validate_baseline_completion_evidence(certificate) == certificate


def test_completion_rehashes_outputs_and_rejects_tampering(tmp_path: Path) -> None:
    protocol, protocol_path = _raw_protocol(tmp_path)
    contract = protocol["baseline_provenance_contracts"][0]
    contract["source_plan_sha256"] = "a" * 64
    contract["source_job_digest"] = "b" * 64
    root = tmp_path / "workflow"
    initial_model = (
        root / "jobs" / "grid" / "k20_d14_r1" / "primary" / "plateau_model.pt"
    )
    initial_model.parent.mkdir(parents=True)
    torch.save(_model_payload(protocol), initial_model)
    stage_dir = root / "jobs" / "grid" / "k20_d14_r1" / "precision"
    endpoint = _stage(
        stage_dir,
        protocol=protocol,
        seed=8660001,
        initial_model=initial_model,
        epoch=83,
    )
    _workflow(
        root,
        contract=contract,
        seed=8660001,
        initial_model=initial_model,
        stage=stage_dir,
        status="succeeded",
        outputs=endpoint,
        returncode=0,
    )
    _json(protocol_path, protocol)
    endpoint["model"].write_bytes(b"tampered")

    with pytest.raises(X21AutoResearchError, match="model hash record is stale"):
        normalize_baseline_completion(
            protocol_path=protocol_path,
            workflow_root=root,
            seed=8660001,
        )


def _recovery_fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    protocol, protocol_path = _raw_protocol(tmp_path)
    contract = protocol["baseline_provenance_contracts"][1]
    contract["source_plan_sha256"] = "c" * 64
    contract["source_job_digest"] = "d" * 64
    contract["training_semantics_sha256"] = contract["source_job_digest"]
    contract["frozen_inputs_sha256"] = contract["source_plan_sha256"]
    source_root = tmp_path / "source-workflow"
    initial_model = (
        source_root / "jobs" / "grid" / "k20_d14_r2" / "primary" / "plateau_model.pt"
    )
    initial_model.parent.mkdir(parents=True)
    torch.save(_model_payload(protocol), initial_model)
    source_stage = source_root / "jobs" / "grid" / "k20_d14_r2" / "precision"
    source_checkpoint = source_stage / "round_001" / "checkpoint.pt"
    source_payload = _checkpoint(
        source_checkpoint,
        protocol=protocol,
        seed=8660002,
        initial_model_sha256=sha256_file(initial_model),
        epoch=36,
        history=[{"epoch": 0}, {"epoch": 36}],
    )
    contract["source_model_sha256"] = sha256_file(initial_model)
    contract["source_checkpoint_sha256"] = sha256_file(source_checkpoint)
    _workflow(
        source_root,
        contract=contract,
        seed=8660002,
        initial_model=initial_model,
        stage=source_stage,
        status="failed",
        attempt_count=3,
    )

    recovery_root = tmp_path / "recovery"
    copied_model = recovery_root / "input" / "primary_model.pt"
    copied_model.parent.mkdir(parents=True)
    shutil.copyfile(initial_model, copied_model)
    recovery_stage = recovery_root / "stage"
    endpoint = _stage(
        recovery_stage,
        protocol=protocol,
        seed=8660002,
        initial_model=copied_model,
        epoch=83,
        resumed_from=contract["source_checkpoint_sha256"],
        history=[*source_payload["history"], {"epoch": 37}, {"epoch": 83}],
    )
    show = recovery_root / "systemd-show.txt"
    show.write_text(
        "\n".join(
            [
                "Id=x21-recovery-test.service",
                "LoadState=loaded",
                "ActiveState=inactive",
                "SubState=dead",
                "Result=success",
                "ExecMainCode=1",
                "ExecMainStatus=0",
                "NRestarts=0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    journal = recovery_root / "journal.log"
    journal.write_text(
        "[plateau-stage] round 1/5: resume\n"
        f"resuming at epoch=37 from {source_checkpoint}\n"
        "checkpointed epoch=83\n"
        f"[plateau-stage] plateau published from round 1: {recovery_stage}\n",
        encoding="utf-8",
    )
    bundle_path = recovery_root / "raw-bundle.json"
    _json(
        bundle_path,
        {
            "schema": RECOVERY_RAW_BUNDLE_SCHEMA,
            "source_workflow_root": str(source_root),
            "source_model": "input/primary_model.pt",
            "stage_dir": "stage",
            "systemd_show": "systemd-show.txt",
            "journal": "journal.log",
        },
    )
    _json(protocol_path, protocol)
    return protocol_path, bundle_path, endpoint


def test_recovery_normalizes_source_checkpoint_and_systemd_artifacts(
    tmp_path: Path,
) -> None:
    protocol_path, bundle_path, endpoint = _recovery_fixture(tmp_path)

    certificate = normalize_exact_recovery(
        protocol_path=protocol_path,
        raw_bundle_path=bundle_path,
    )

    assert certificate["schema"] == RECOVERY_EVIDENCE_SCHEMA
    assert certificate["source_epoch"] == 36
    assert certificate["source_next_epoch"] == 37
    assert certificate["native_exit_code"] == 0
    assert certificate["operational_attempt_count"] == 1
    assert certificate["output_model_sha256"] == sha256_file(endpoint["model"])
    assert validate_recovery_evidence(certificate) == certificate


def test_recovery_accepts_only_invocation_bound_collected_transient_journal(
    tmp_path: Path,
) -> None:
    protocol_path, bundle_path, _endpoint = _recovery_fixture(tmp_path)
    root = bundle_path.parent
    unit = "x21-recovery-test.service"
    invocation = "a" * 32
    (root / "systemd-show.txt").write_text(
        "\n".join(
            [
                f"Id={unit}",
                "LoadState=not-found",
                "ActiveState=inactive",
                "SubState=dead",
                "Result=success",
                "ExecMainCode=0",
                "ExecMainStatus=0",
                "NRestarts=0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    records = [
        {
            "USER_UNIT": unit,
            "USER_INVOCATION_ID": invocation,
            "JOB_TYPE": "start",
            "MESSAGE": f"Starting {unit}",
            "__REALTIME_TIMESTAMP": "1",
        },
        {
            "USER_UNIT": unit,
            "USER_INVOCATION_ID": invocation,
            "JOB_TYPE": "start",
            "JOB_RESULT": "done",
            "MESSAGE": f"Started {unit}",
            "__REALTIME_TIMESTAMP": "2",
        },
        {
            "_SYSTEMD_USER_UNIT": unit,
            "_SYSTEMD_INVOCATION_ID": invocation,
            "MESSAGE": "[plateau-stage] round 1/5: resume",
            "__REALTIME_TIMESTAMP": "3",
        },
        {
            "_SYSTEMD_USER_UNIT": unit,
            "_SYSTEMD_INVOCATION_ID": invocation,
            "MESSAGE": "resuming at epoch=37 from checkpoint.pt",
            "__REALTIME_TIMESTAMP": "4",
        },
        {
            "_SYSTEMD_USER_UNIT": unit,
            "_SYSTEMD_INVOCATION_ID": invocation,
            "MESSAGE": "[plateau-stage] plateau published from round 1: stage",
            "__REALTIME_TIMESTAMP": "5",
        },
        {
            "USER_UNIT": unit,
            "USER_INVOCATION_ID": invocation,
            "MESSAGE_ID": "ae8f7b866b0347b9af31fe1c80b127c0",
            "MESSAGE": f"{unit}: Consumed CPU time",
            "__REALTIME_TIMESTAMP": "6",
        },
    ]
    journal_path = root / "journal.log"
    journal_path.write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
    )
    certificate = normalize_exact_recovery(
        protocol_path=protocol_path, raw_bundle_path=bundle_path
    )
    assert certificate["native_exit_code"] == 0

    records[3]["_SYSTEMD_INVOCATION_ID"] = "b" * 32
    journal_path.write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
    )
    with pytest.raises(X21AutoResearchError, match="one systemd invocation"):
        normalize_exact_recovery(
            protocol_path=protocol_path, raw_bundle_path=bundle_path
        )


def test_recovery_bundle_rejects_free_hashes_and_failed_native_exit(
    tmp_path: Path,
) -> None:
    protocol_path, bundle_path, _endpoint = _recovery_fixture(tmp_path)
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle["claimed_checkpoint_sha256"] = hashlib.sha256(b"not evidence").hexdigest()
    _json(bundle_path, bundle)
    with pytest.raises(X21AutoResearchError, match="fields differ"):
        normalize_exact_recovery(
            protocol_path=protocol_path,
            raw_bundle_path=bundle_path,
        )

    del bundle["claimed_checkpoint_sha256"]
    _json(bundle_path, bundle)
    show = bundle_path.parent / "systemd-show.txt"
    show.write_text(
        show.read_text(encoding="utf-8").replace(
            "ExecMainStatus=0", "ExecMainStatus=139"
        ),
        encoding="utf-8",
    )
    with pytest.raises(X21AutoResearchError, match="did not exit with status zero"):
        normalize_exact_recovery(
            protocol_path=protocol_path,
            raw_bundle_path=bundle_path,
        )


def test_assemble_baseline_family_rehashes_checkpoints_and_is_registerable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    certificate_paths: dict[int, Path] = {}
    model_paths: dict[int, Path] = {}
    checkpoint_paths: dict[int, Path] = {}
    for contract in protocol["baseline_provenance_contracts"]:
        seed = int(contract["seed"])
        model = tmp_path / f"model-{seed}.pt"
        torch.save(_model_payload(protocol), model)
        model_sha256 = sha256_file(model)
        checkpoint = tmp_path / f"checkpoint-{seed}.pt"
        torch.save(
            {
                "schema": "type11-positive-tensor-network-checkpoint-v1",
                "seed": seed,
                "optimizer_state_dict": {"state": {}},
            },
            checkpoint,
        )
        checkpoint_sha256 = sha256_file(checkpoint)
        assert model_sha256 != checkpoint_sha256
        if contract["kind"] == "exact-recovery":
            certificate = {
                "schema": RECOVERY_EVIDENCE_SCHEMA,
                "campaign_id": contract["recovery_campaign_id"],
                "seed": seed,
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
                "output_summary_sha256": "6" * 64,
            }
        else:
            certificate = {
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
                "output_summary_sha256": hashlib.sha256(
                    f"summary-{seed}".encode()
                ).hexdigest(),
            }
        certificate_path = tmp_path / f"certificate-{seed}.json"
        _json(certificate_path, certificate)
        certificate_paths[seed] = certificate_path
        model_paths[seed] = model
        checkpoint_paths[seed] = checkpoint

    family = assemble_baseline_family(
        protocol_path=PROTOCOL,
        certificate_paths=certificate_paths,
        model_paths=model_paths,
        checkpoint_paths=checkpoint_paths,
    )
    assert [row["seed"] for row in family["models"]] == protocol["promotion_seeds"]
    assert all("certificate" in row["provenance"] for row in family["models"])
    assert {row["seed"]: row["checkpoint_sha256"] for row in family["models"]} == {
        seed: sha256_file(path) for seed, path in model_paths.items()
    }

    bindings = assemble_parent_bindings(
        protocol_path=PROTOCOL,
        certificate_paths=certificate_paths,
        model_paths=model_paths,
        checkpoint_paths=checkpoint_paths,
    )
    assert bindings["schema"] == "gcicy-x21-parent-bindings-v1"
    for row in bindings["models"]:
        assert Path(row["model_path"]) == model_paths[row["seed"]].resolve()
        assert Path(row["checkpoint_path"]) == checkpoint_paths[row["seed"]].resolve()
        assert torch.load(row["model_path"], weights_only=False)["schema"] == (
            "type11-positive-tensor-network-v1"
        )
        assert torch.load(row["checkpoint_path"], weights_only=False)["schema"] == (
            "type11-positive-tensor-network-checkpoint-v1"
        )

    family_out = tmp_path / "family.json"
    bindings_out = tmp_path / "parent-bindings.json"
    argv = ["family", "--protocol", str(PROTOCOL)]
    for seed in protocol["promotion_seeds"]:
        argv.extend(["--certificate", f"{seed}={certificate_paths[seed]}"])
        argv.extend(["--model", f"{seed}={model_paths[seed]}"])
        argv.extend(["--checkpoint", f"{seed}={checkpoint_paths[seed]}"])
    argv.extend(["--out", str(family_out), "--bindings-out", str(bindings_out)])
    assert provenance_main(argv) == 0
    capsys.readouterr()
    assert json.loads(family_out.read_text(encoding="utf-8")) == family
    assert json.loads(bindings_out.read_text(encoding="utf-8")) == bindings

    store = X21CampaignStore.initialize(tmp_path / "campaign", protocol)
    registered = store.register_baseline(family)
    assert registered["family_sha256"] == family["family_sha256"]
    assert all("certificate" not in row["provenance"] for row in registered["models"])

    checkpoint_paths[8660001] = tmp_path / "wrong-checkpoint.pt"
    checkpoint_paths[8660001].write_bytes(b"wrong")
    with pytest.raises(X21AutoResearchError, match="does not identify"):
        assemble_baseline_family(
            protocol_path=PROTOCOL,
            certificate_paths=certificate_paths,
            model_paths=model_paths,
            checkpoint_paths=checkpoint_paths,
        )
