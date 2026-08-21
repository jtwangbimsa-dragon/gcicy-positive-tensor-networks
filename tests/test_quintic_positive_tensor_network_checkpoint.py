import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np


torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_quintic_positive_tensor_network_same_points import (  # noqa: E402
    TRAINING_CHECKPOINT_SCHEMA,
    build_training_checkpoint,
    load_source_report_for_evaluation,
    merge_device_memory,
    model_payload,
    parameter_scope_from_args,
    parse_args,
    restore_training_checkpoint_state,
    resumed_training_is_complete,
    save_training_checkpoint_atomic,
    validate_checkpoint_paths,
    validate_training_checkpoint,
)
import scripts.train_quintic_positive_tensor_network_same_points as trainer  # noqa: E402


def test_checkpoint_and_development_only_cli_contract(monkeypatch) -> None:
    required = [
        "trainer",
        "--source-run-dir",
        "source",
        "--output-dir",
        "output",
    ]
    monkeypatch.setattr(sys, "argv", required)
    defaults = parse_args()
    assert defaults.checkpoint is None
    assert defaults.resume_checkpoint is None
    assert defaults.skip_blind_audit is False

    monkeypatch.setattr(
        sys,
        "argv",
        [
            *required,
            "--checkpoint",
            "next.pt",
            "--resume-checkpoint",
            "previous.pt",
            "--skip-blind-audit",
        ],
    )
    args = parse_args()
    assert args.checkpoint == Path("next.pt")
    assert args.resume_checkpoint == Path("previous.pt")
    assert args.skip_blind_audit is True


@pytest.mark.parametrize(
    ("freeze_inherited", "freeze_dictionary", "expected"),
    [
        (3, True, "new_channels"),
        (0, True, "cores"),
        (0, False, "joint"),
    ],
)
def test_parameter_scope_contract(
    freeze_inherited: int,
    freeze_dictionary: bool,
    expected: str,
) -> None:
    args = SimpleNamespace(
        freeze_inherited_bond_dimension=freeze_inherited,
        freeze_physical_dictionary=freeze_dictionary,
    )
    assert parameter_scope_from_args(args) == expected


def test_development_scope_does_not_read_historical_source_report(
    tmp_path,
    monkeypatch,
) -> None:
    report = tmp_path / "report.json"

    def unexpected_read(*_args, **_kwargs):
        raise AssertionError("development-only evaluation must not read source report")

    monkeypatch.setattr(Path, "read_text", unexpected_read)
    assert (
        load_source_report_for_evaluation(
            report,
            skip_blind_audit=True,
        )
        == {}
    )


def make_checkpoint():
    torch.manual_seed(4101)
    model = torch.nn.Linear(3, 2, dtype=torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.025)
    loss = model(torch.ones(4, 3, dtype=torch.float64)).square().sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(4102)
    semantics = {
        "model": {"site_count": 24, "bond_dimension": 14},
        "optimization": {"learning_rate": 0.025, "epochs": 8},
    }
    hashes = {
        "dataset_sha256": "a" * 64,
        "initial_model_sha256": "b" * 64,
        "blind_points_sha256": None,
    }
    payload = build_training_checkpoint(
        model=model,
        optimizer=optimizer,
        permutation_generator=generator,
        device=torch.device("cpu"),
        epoch=2,
        next_epoch=3,
        history=[
            {"epoch": 0, "selection_score": 2.0},
            {"epoch": 2, "selection_score": 1.0},
        ],
        best_state=model.state_dict(),
        best_score=1.0,
        best_epoch=2,
        optimizer_steps=17,
        fixed_log_kappa=0.125,
        fixed_log_kappa_source="fubini_study_metric_on_selected_training_pool",
        distillation_evidence={"epochs": 3},
        timing_seconds={
            "distillation": 2.0,
            "training": 12.5,
            "wall_total": 18.0,
        },
        device_memory=None,
        training_semantics=semantics,
        frozen_input_hashes=hashes,
    )
    return model, optimizer, generator, semantics, hashes, payload


def test_checkpoint_schema_and_strict_validation() -> None:
    _, _, _, semantics, hashes, payload = make_checkpoint()
    assert payload["schema"] == TRAINING_CHECKPOINT_SCHEMA
    assert payload["optimizer_steps"] == 17
    assert payload["distillation_evidence"] == {"epochs": 3}
    validate_training_checkpoint(
        payload,
        training_semantics=semantics,
        frozen_input_hashes=hashes,
    )

    changed_semantics = copy.deepcopy(semantics)
    changed_semantics["optimization"]["learning_rate"] = 0.05
    with pytest.raises(ValueError, match=r"optimization\.learning_rate"):
        validate_training_checkpoint(
            payload,
            training_semantics=changed_semantics,
            frozen_input_hashes=hashes,
        )

    changed_hashes = copy.deepcopy(hashes)
    changed_hashes["dataset_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="frozen input hash mismatch"):
        validate_training_checkpoint(
            payload,
            training_semantics=semantics,
            frozen_input_hashes=changed_hashes,
        )


def test_checkpoint_rejects_ambiguous_boundary_and_bad_timing() -> None:
    _, _, _, semantics, hashes, payload = make_checkpoint()
    payload["next_epoch"] = payload["epoch"]
    with pytest.raises(ValueError, match=r"next_epoch must equal epoch \+ 1"):
        validate_training_checkpoint(
            payload,
            training_semantics=semantics,
            frozen_input_hashes=hashes,
        )

    _, _, _, semantics, hashes, payload = make_checkpoint()
    payload["timing_seconds"]["training"] = float("nan")
    with pytest.raises(ValueError, match="timing_seconds"):
        validate_training_checkpoint(
            payload,
            training_semantics=semantics,
            frozen_input_hashes=hashes,
        )


def test_restore_recovers_model_adam_and_both_cpu_rng_streams() -> None:
    model, optimizer, generator, semantics, hashes, payload = make_checkpoint()
    validate_training_checkpoint(
        payload,
        training_semantics=semantics,
        frozen_input_hashes=hashes,
    )
    expected_global_random = torch.rand(5)
    expected_permutation = torch.randperm(12, generator=generator)
    expected_model = {
        key: value.clone() for key, value in payload["current_model_state_dict"].items()
    }
    expected_optimizer_steps = sorted(
        int(state["step"])
        for state in payload["optimizer_state_dict"]["state"].values()
    )

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(100.0)
    optimizer.state.clear()
    torch.manual_seed(9999)
    generator.manual_seed(9998)
    restore_training_checkpoint_state(
        payload,
        model=model,
        optimizer=optimizer,
        permutation_generator=generator,
        device=torch.device("cpu"),
    )

    for key, value in model.state_dict().items():
        assert torch.equal(value, expected_model[key])
    restored_optimizer_steps = sorted(
        int(state["step"]) for state in optimizer.state_dict()["state"].values()
    )
    assert restored_optimizer_steps == expected_optimizer_steps
    assert torch.equal(torch.rand(5), expected_global_random)
    assert torch.equal(torch.randperm(12, generator=generator), expected_permutation)


def test_cpu_checkpoint_never_reads_cuda_rng(monkeypatch) -> None:
    def unexpected_cuda_rng_access(*_args, **_kwargs):
        raise AssertionError("CPU checkpoint must not inspect CUDA RNG")

    monkeypatch.setattr(torch.cuda, "get_rng_state", unexpected_cuda_rng_access)
    *_, payload = make_checkpoint()
    assert payload["cuda_rng_state"] is None
    assert payload["cuda_rng_device"] is None


def test_resumed_tiny_adam_trajectory_matches_uninterrupted() -> None:
    inputs = torch.arange(24, dtype=torch.float64).reshape(8, 3) / 10.0
    targets = torch.linspace(-0.4, 0.7, 8, dtype=torch.float64).reshape(-1, 1)

    def fresh_state():
        torch.manual_seed(8801)
        model = torch.nn.Linear(3, 1, dtype=torch.float64)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(8802)
        return model, optimizer, generator

    def train_epoch(model, optimizer, generator) -> None:
        permutation = torch.randperm(len(inputs), generator=generator)
        for start in range(0, len(inputs), 2):
            indices = permutation[start : start + 2]
            jitter = 1.0e-4 * torch.rand((), dtype=torch.float64)
            loss = (model(inputs[indices]) - targets[indices] + jitter).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    continuous_model, continuous_optimizer, continuous_generator = fresh_state()
    for _ in range(4):
        train_epoch(continuous_model, continuous_optimizer, continuous_generator)
    expected_cpu_random = torch.rand(4)
    expected_permutation = torch.randperm(8, generator=continuous_generator)

    interrupted_model, interrupted_optimizer, interrupted_generator = fresh_state()
    for _ in range(2):
        train_epoch(interrupted_model, interrupted_optimizer, interrupted_generator)
    payload = build_training_checkpoint(
        model=interrupted_model,
        optimizer=interrupted_optimizer,
        permutation_generator=interrupted_generator,
        device=torch.device("cpu"),
        epoch=2,
        next_epoch=3,
        history=[{"epoch": 0}, {"epoch": 2}],
        best_state=interrupted_model.state_dict(),
        best_score=1.0,
        best_epoch=2,
        optimizer_steps=8,
        fixed_log_kappa=0.0,
        fixed_log_kappa_source="test",
        distillation_evidence=None,
        timing_seconds={
            "distillation": 0.0,
            "training": 1.0,
            "wall_total": 1.5,
        },
        device_memory=None,
        training_semantics={"trajectory": "tiny-adam"},
        frozen_input_hashes={"data": "fixed"},
    )

    resumed_model, resumed_optimizer, resumed_generator = fresh_state()
    restore_training_checkpoint_state(
        payload,
        model=resumed_model,
        optimizer=resumed_optimizer,
        permutation_generator=resumed_generator,
        device=torch.device("cpu"),
    )
    for _ in range(payload["next_epoch"], 5):
        train_epoch(resumed_model, resumed_optimizer, resumed_generator)

    for expected, actual in zip(
        continuous_model.parameters(), resumed_model.parameters(), strict=True
    ):
        assert torch.equal(actual, expected)
    for parameter_id, expected_state in continuous_optimizer.state_dict()[
        "state"
    ].items():
        actual_state = resumed_optimizer.state_dict()["state"][parameter_id]
        for key, expected in expected_state.items():
            actual = actual_state[key]
            if torch.is_tensor(expected):
                assert torch.equal(actual, expected)
            else:
                assert actual == expected
    assert torch.equal(torch.rand(4), expected_cpu_random)
    assert torch.equal(
        torch.randperm(8, generator=resumed_generator), expected_permutation
    )


def test_terminal_boundary_and_cross_process_memory_contract() -> None:
    *_, payload = make_checkpoint()
    assert resumed_training_is_complete(payload, requested_epochs=2)
    assert not resumed_training_is_complete(payload, requested_epochs=8)
    assert merge_device_memory(
        {"maximum_allocated_bytes": 900, "maximum_reserved_bytes": 1200},
        {"maximum_allocated_bytes": 1000, "maximum_reserved_bytes": 1100},
    ) == {
        "maximum_allocated_bytes": 1000,
        "maximum_reserved_bytes": 1200,
    }


def test_checkpoint_paths_cannot_overwrite_final_or_frozen_inputs(tmp_path) -> None:
    source = (tmp_path / "source.npz").resolve()
    with pytest.raises(ValueError, match="checkpoint path"):
        validate_checkpoint_paths(
            checkpoint_path=source,
            resume_checkpoint_path=None,
            model_path=(tmp_path / "model.pt").resolve(),
            report_path=(tmp_path / "report.json").resolve(),
            frozen_input_paths={source},
        )


def test_atomic_checkpoint_replaces_target_and_preserves_it_on_failure(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "training.pt"
    path.write_bytes(b"old")
    save_training_checkpoint_atomic(path, {"generation": 1})
    assert torch.load(path, weights_only=False) == {"generation": 1}

    complete = path.read_bytes()

    def fail_after_partial_write(_payload, temporary) -> None:
        Path(temporary).write_bytes(b"partial")
        raise OSError("simulated write failure")

    monkeypatch.setattr(torch, "save", fail_after_partial_write)
    with pytest.raises(OSError, match="simulated write failure"):
        save_training_checkpoint_atomic(path, {"generation": 2})
    assert path.read_bytes() == complete
    assert list(tmp_path.glob(".training.pt.*.tmp")) == []


def test_continuation_model_preserves_bond_expansion_metadata() -> None:
    args = SimpleNamespace(
        source_degree=1,
        site_count=6,
        bond_dimension=8,
        dictionary_rank=25,
        fermat_phase_charge_multiplicity=0,
        fermat_s5_orbit_tying=False,
        fermat_two_site_blocking=False,
        transfer_implementation="vectorized",
        freeze_physical_dictionary=True,
        positive_floor=1.0e-4,
        precision="complex64",
    )
    model = SimpleNamespace(output_dimension=5, architecture="shared_local_dictionary")
    expansion = {"source_bond_dimension": 4, "target_bond_dimension": 8}
    payload = model_payload(
        model,
        {"weight": torch.ones(1)},
        args=args,
        fixed_log_kappa=0.1,
        source_hashes={"dataset": "a"},
        initial_model_evidence={"bond_expansion": expansion},
    )
    assert payload["bond_expansion"] == expansion
    assert payload["bond_expansion"] is not expansion


def test_development_only_terminal_resume_publishes_without_blind_inputs(
    tmp_path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    pullbacks = tmp_path / "pullbacks"
    training_data = source / "training_data"
    training_data.mkdir(parents=True)
    pullbacks.mkdir()
    dataset = training_data / "dataset.npz"
    basis = training_data / "basis.pickle"
    train_pullbacks = pullbacks / "train_pullbacks.npy"
    validation_pullbacks = pullbacks / "validation_pullbacks.npy"
    official_metrics = pullbacks / "validation_official_fs_metrics.npy"
    np.savez(
        dataset,
        X_train=np.zeros((1, 10), dtype=np.float32),
        y_train=np.ones(1, dtype=np.float64),
        X_val=np.zeros((1, 10), dtype=np.float32),
        y_val=np.ones(1, dtype=np.float64),
    )
    basis.write_bytes(b"fixed basis")
    np.save(train_pullbacks, np.zeros((1, 3, 4), dtype=np.complex64))
    np.save(validation_pullbacks, np.zeros((1, 3, 4), dtype=np.complex64))
    np.save(official_metrics, np.zeros((1, 3, 3), dtype=np.complex64))
    pullback_report = {
        "source_sha256": {
            "dataset": trainer.sha256_file(dataset),
            "basis": trainer.sha256_file(basis),
            "blind_points": "historical-blind-must-not-be-read",
        },
        "output_sha256": {
            "train": trainer.sha256_file(train_pullbacks),
            "validation": trainer.sha256_file(validation_pullbacks),
            "validation_official_fs_metrics": trainer.sha256_file(official_metrics),
            "blind": "historical-blind-must-not-be-read",
        },
    }
    (pullbacks / "report.json").write_text(
        json.dumps(pullback_report),
        encoding="utf-8",
    )
    # These files deliberately do not exist.  A development-only run must not
    # require or open any of them.
    assert not (source / "report.json").exists()
    assert not (source / "blind_points.npz").exists()
    assert not (pullbacks / "blind_pullbacks.npy").exists()

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.physical_dictionary = torch.nn.Parameter(
                torch.ones((1, 1, 1), dtype=torch.complex64)
            )
            self.coefficient_cores = torch.nn.ParameterList(
                [torch.nn.Parameter(torch.ones((1, 1, 1), dtype=torch.complex64))]
            )
            self.output_dimension = 5
            self.architecture = "shared_local_dictionary"
            self.trainable_real_parameter_count = 4

        def orthonormalize_physical_dictionary_(self):
            return self

    monkeypatch.setattr(
        trainer,
        "PositiveTensorNetworkMetric",
        lambda *_args, **_kwargs: TinyModel(),
    )
    monkeypatch.setattr(
        trainer,
        "anchored_orthonormal_physical_dictionary",
        lambda *_args, **_kwargs: np.ones((21, 5, 5), dtype=np.complex128),
    )
    monkeypatch.setattr(
        trainer,
        "tensor_split",
        lambda x, *_args, **_kwargs: {"count": len(x)},
    )
    monkeypatch.setattr(
        trainer,
        "exact_fs_reference_model",
        lambda *_args, **_kwargs: torch.nn.Linear(1, 1),
    )
    monkeypatch.setattr(
        trainer,
        "fs_reproduction_check",
        lambda *_args, **_kwargs: {"status": "passed"},
    )
    evaluation_calls = []

    def fake_evaluate(*_args, **_kwargs):
        evaluation_calls.append(1)
        return (
            {
                "selection_score": 1.0,
                "normalized_volume": {
                    "sigma_official_formula": 0.1,
                    "weighted_rms_abs_residual": 0.2,
                    "ratio_weighted_quantiles": {"q1.0000": 1.1},
                },
            },
            np.zeros(1),
            np.ones(1),
            np.ones(1),
        )

    monkeypatch.setattr(trainer, "evaluate_split", fake_evaluate)

    checkpoint = tmp_path / "checkpoint.pt"

    def run(output: Path, *, resume: bool) -> dict:
        argv = [
            "trainer",
            "--source-run-dir",
            str(source),
            "--pullbacks-dir",
            str(pullbacks),
            "--output-dir",
            str(output),
            "--epochs",
            "0",
            "--fixed-log-kappa",
            "0.0",
            "--checkpoint",
            str(checkpoint),
            "--skip-blind-audit",
            "--device",
            "cpu",
        ]
        if resume:
            argv.extend(("--resume-checkpoint", str(checkpoint)))
        monkeypatch.setattr(sys, "argv", argv)
        trainer.main()
        return json.loads((output / "report.json").read_text(encoding="utf-8"))

    first = run(tmp_path / "first", resume=False)
    assert len(evaluation_calls) == 2  # epoch-0 selection plus final publication
    assert first["evaluation_scope"] == "development_only"
    assert first["termination_reason"] == "completed_requested_epochs"
    assert first["blind_test"]["status"] == "skipped"
    assert first["comparators"] == {
        "official_cymetric_network": None,
        "official_cymetric_blind_test": None,
    }
    assert first["artifacts"]["blind_arrays"] is None
    assert not (tmp_path / "first" / "blind_test_tail_arrays.npz").exists()

    evaluation_calls.clear()
    resumed = run(tmp_path / "resumed", resume=True)
    assert len(evaluation_calls) == 1  # no duplicate epoch-0 selection evaluation
    assert resumed["training"]["checkpoint"]["terminal_boundary_reused"] is True
    assert resumed["training"]["history"] == first["training"]["history"]
    assert not (tmp_path / "resumed" / "blind_test_tail_arrays.npz").exists()
