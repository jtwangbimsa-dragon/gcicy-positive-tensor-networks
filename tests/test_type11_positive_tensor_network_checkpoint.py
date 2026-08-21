import copy
import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

from scripts.train_type11_positive_tensor_network import (  # noqa: E402
    build_training_checkpoint,
    merge_device_memory,
    parse_args,
    restore_training_checkpoint_state,
    resumed_training_termination,
    save_training_checkpoint_atomic,
    validate_checkpoint_paths,
    validate_training_checkpoint,
)


def test_checkpoint_cli_is_optional_and_accepts_resume_paths(monkeypatch) -> None:
    required = [
        "trainer",
        "--source-artifact",
        "source.npz",
        "--bond-dimension",
        "4",
        "--out",
        "model.pt",
        "--summary",
        "summary.json",
    ]
    monkeypatch.setattr(sys, "argv", required)
    default_args = parse_args()
    assert default_args.checkpoint is None
    assert default_args.resume_checkpoint is None

    monkeypatch.setattr(
        sys,
        "argv",
        [
            *required,
            "--checkpoint",
            "checkpoint.pt",
            "--resume-checkpoint",
            "resume.pt",
        ],
    )
    checkpoint_args = parse_args()
    assert checkpoint_args.checkpoint == Path("checkpoint.pt")
    assert checkpoint_args.resume_checkpoint == Path("resume.pt")


def make_checkpoint():
    torch.manual_seed(1701)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.025)
    loss = model(torch.ones(4, 3)).square().sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(1702)
    semantics = {
        "model": {"bond_dimension": 4, "precision": "complex128"},
        "optimization": {"learning_rate": 0.025, "epochs": 8},
    }
    hashes = {
        "source_artifact_sha256": "a" * 64,
        "teacher_artifact_sha256": None,
    }
    payload = build_training_checkpoint(
        model=model,
        optimizer=optimizer,
        permutation_generator=generator,
        epoch=2,
        next_epoch=3,
        history=[
            {"epoch": 0, "validation_selection_score": 2.0},
            {"epoch": 2, "validation_selection_score": 1.0},
        ],
        best_state=model.state_dict(),
        best_score=1.0,
        best_epoch=2,
        material_best_score=1.0,
        evaluations_since_material_improvement=0,
        fixed_log_kappa=torch.tensor(0.125),
        fixed_log_kappa_source="initial_model_training_pool",
        runtime_seconds=12.5,
        device_memory=None,
        training_semantics=semantics,
        frozen_input_hashes=hashes,
    )
    return model, optimizer, generator, semantics, hashes, payload


def test_checkpoint_validation_rejects_semantic_and_input_hash_mismatches() -> None:
    _, _, _, semantics, hashes, payload = make_checkpoint()

    validate_training_checkpoint(
        payload,
        training_semantics=semantics,
        frozen_input_hashes=hashes,
    )

    changed_semantics = copy.deepcopy(semantics)
    changed_semantics["optimization"]["learning_rate"] = 0.05
    with pytest.raises(
        ValueError,
        match=r"optimization\.learning_rate",
    ):
        validate_training_checkpoint(
            payload,
            training_semantics=changed_semantics,
            frozen_input_hashes=hashes,
        )

    changed_hashes = copy.deepcopy(hashes)
    changed_hashes["source_artifact_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="frozen input hash mismatch"):
        validate_training_checkpoint(
            payload,
            training_semantics=semantics,
            frozen_input_hashes=changed_hashes,
        )


def test_checkpoint_validation_requires_an_unambiguous_next_epoch() -> None:
    _, _, _, semantics, hashes, payload = make_checkpoint()
    payload["next_epoch"] = payload["epoch"]

    with pytest.raises(ValueError, match=r"next_epoch must equal epoch \+ 1"):
        validate_training_checkpoint(
            payload,
            training_semantics=semantics,
            frozen_input_hashes=hashes,
        )


def test_checkpoint_paths_cannot_overwrite_resume_or_frozen_inputs(tmp_path) -> None:
    resume = tmp_path / "resume.pt"
    with pytest.raises(ValueError, match="resume checkpoint path"):
        validate_checkpoint_paths(
            checkpoint_path=tmp_path / "next.pt",
            resume_checkpoint_path=resume,
            output_path=resume,
            summary_path=tmp_path / "summary.json",
            frozen_input_paths={tmp_path / "source.npz"},
        )

    source = tmp_path / "source.npz"
    with pytest.raises(ValueError, match="checkpoint path"):
        validate_checkpoint_paths(
            checkpoint_path=source,
            resume_checkpoint_path=None,
            output_path=tmp_path / "model.pt",
            summary_path=tmp_path / "summary.json",
            frozen_input_paths={source},
        )


def test_device_memory_merge_keeps_cross_process_peaks() -> None:
    previous = {
        "maximum_allocated_bytes": 900,
        "maximum_reserved_bytes": 1200,
    }
    current = {
        "maximum_allocated_bytes": 1000,
        "maximum_reserved_bytes": 1100,
    }

    assert merge_device_memory(None, None) is None
    assert merge_device_memory(previous, current) == {
        "maximum_allocated_bytes": 1000,
        "maximum_reserved_bytes": 1200,
    }


def test_terminal_resume_boundary_can_be_transferred_without_new_epoch(
    tmp_path,
) -> None:
    _, _, _, semantics, hashes, payload = make_checkpoint()

    assert (
        resumed_training_termination(
            payload,
            requested_epochs=2,
            early_stopping_patience=0,
        )
        == "completed_requested_epochs"
    )
    target = tmp_path / "transferred.pt"
    save_training_checkpoint_atomic(target, payload)
    transferred = torch.load(target, map_location="cpu", weights_only=False)
    validate_training_checkpoint(
        transferred,
        training_semantics=semantics,
        frozen_input_hashes=hashes,
    )
    assert transferred["next_epoch"] == 3

    payload["evaluations_since_material_improvement"] = 4
    assert (
        resumed_training_termination(
            payload,
            requested_epochs=8,
            early_stopping_patience=4,
        )
        == "validation_plateau"
    )


def test_restore_recovers_model_optimizer_and_cpu_rng_streams() -> None:
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
    )

    for key, value in model.state_dict().items():
        assert torch.equal(value, expected_model[key])
    restored_optimizer_steps = sorted(
        int(state["step"]) for state in optimizer.state_dict()["state"].values()
    )
    assert restored_optimizer_steps == expected_optimizer_steps
    assert torch.equal(torch.rand(5), expected_global_random)
    assert torch.equal(torch.randperm(12, generator=generator), expected_permutation)


def test_cpu_generator_checkpoint_does_not_touch_cuda_rng(monkeypatch) -> None:
    def unexpected_cuda_rng_access(*_args, **_kwargs):
        raise AssertionError("CPU training must not inspect CUDA RNG state")

    monkeypatch.setattr(torch.cuda, "get_rng_state", unexpected_cuda_rng_access)
    *_, payload = make_checkpoint()

    assert payload["cuda_rng_state"] is None
    assert payload["cuda_rng_device"] is None


def test_resumed_tiny_training_matches_uninterrupted_trajectory() -> None:
    inputs = torch.arange(24, dtype=torch.float64).reshape(8, 3) / 10.0
    targets = torch.linspace(-0.4, 0.7, 8, dtype=torch.float64).reshape(-1, 1)

    def fresh_training_state():
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

    continuous_model, continuous_optimizer, continuous_generator = (
        fresh_training_state()
    )
    for _ in range(4):
        train_epoch(continuous_model, continuous_optimizer, continuous_generator)
    expected_cpu_random = torch.rand(4)
    expected_permutation = torch.randperm(8, generator=continuous_generator)

    interrupted_model, interrupted_optimizer, interrupted_generator = (
        fresh_training_state()
    )
    for _ in range(2):
        train_epoch(interrupted_model, interrupted_optimizer, interrupted_generator)
    semantics = {"trajectory": "tiny-adam"}
    hashes = {"data": "fixed"}
    payload = build_training_checkpoint(
        model=interrupted_model,
        optimizer=interrupted_optimizer,
        permutation_generator=interrupted_generator,
        epoch=2,
        next_epoch=3,
        history=[{"epoch": 0}, {"epoch": 2}],
        best_state=interrupted_model.state_dict(),
        best_score=1.0,
        best_epoch=2,
        material_best_score=1.0,
        evaluations_since_material_improvement=0,
        fixed_log_kappa=torch.tensor(0.0),
        fixed_log_kappa_source="test",
        runtime_seconds=1.0,
        device_memory=None,
        training_semantics=semantics,
        frozen_input_hashes=hashes,
    )

    resumed_model, resumed_optimizer, resumed_generator = fresh_training_state()
    restore_training_checkpoint_state(
        payload,
        model=resumed_model,
        optimizer=resumed_optimizer,
        permutation_generator=resumed_generator,
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


def test_atomic_checkpoint_save_replaces_target_and_removes_temporary(
    tmp_path,
) -> None:
    path = tmp_path / "training.pt"
    path.write_bytes(b"old checkpoint")

    save_training_checkpoint_atomic(path, {"generation": 1})
    assert torch.load(path, weights_only=False) == {"generation": 1}
    assert not path.with_suffix(".pt.tmp").exists()

    save_training_checkpoint_atomic(path, {"generation": 2})
    assert torch.load(path, weights_only=False) == {"generation": 2}
    assert not path.with_suffix(".pt.tmp").exists()


def test_atomic_checkpoint_failure_preserves_target_and_cleans_temporary(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "training.pt"
    path.write_bytes(b"complete checkpoint")

    def fail_after_partial_write(_payload, temporary) -> None:
        Path(temporary).write_bytes(b"partial")
        raise OSError("simulated write failure")

    monkeypatch.setattr(torch, "save", fail_after_partial_write)
    with pytest.raises(OSError, match="simulated write failure"):
        save_training_checkpoint_atomic(path, {"generation": 2})

    assert path.read_bytes() == b"complete checkpoint"
    assert list(tmp_path.glob(".training.pt.*.tmp")) == []
