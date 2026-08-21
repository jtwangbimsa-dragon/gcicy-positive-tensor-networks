from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import run_gcicy_tn_plateau_stage as plateau_stage


def test_child_signal_exit_code_is_preserved_for_outer_retry() -> None:
    assert plateau_stage.child_failure_exit_code(-11) == 139
    assert plateau_stage.child_failure_exit_code(-4) == 132
    assert plateau_stage.child_failure_exit_code(2) == 2


def _write_terminal_round(
    model_path: Path,
    summary_path: Path,
    *,
    model_bytes: bytes,
    termination_reason: str,
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(model_bytes)
    summary_path.write_text(
        json.dumps(
            {
                "termination_reason": termination_reason,
                "model_sha256": plateau_stage.sha256_file(model_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def test_rounds_resume_checkpoint_and_publish_only_after_plateau(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage_dir = tmp_path / "stage"
    initial_model = tmp_path / "initial.pt"
    initial_model.write_bytes(b"initial-model")
    fake_trainer = tmp_path / "fake_trainer.py"
    fake_trainer.write_text("# fake trainer used through a subprocess mock\n")

    # A mismatched terminal summary must not suppress round 2.  Its checkpoint
    # makes that round a resume rather than a fresh invocation.
    round_two = stage_dir / "round_002"
    round_two.mkdir(parents=True)
    (round_two / "model.pt").write_bytes(b"stale-model")
    (round_two / "summary.json").write_text(
        json.dumps(
            {
                "termination_reason": "validation_plateau",
                "model_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    (round_two / "checkpoint.pt").write_bytes(b"round-two-checkpoint-before")

    args, trainer_args = plateau_stage.parse_cli(
        [
            "--initial-model",
            str(initial_model),
            "--stage-dir",
            str(stage_dir),
            "--max-rounds",
            "3",
            "--first-kappa-source",
            "initial_model",
            "--continuation-kappa-source",
            "saved_model",
            "--",
            "--source-artifact",
            "source.npz",
            "--bond-dimension",
            "8",
            "--device",
            "cuda",
        ]
    )
    commands: list[list[str]] = []

    def fake_run(command: list[str], *, check: bool) -> subprocess.CompletedProcess:
        assert check is False
        commands.append(command)
        model_path = Path(_option_value(command, "--out"))
        summary_path = Path(_option_value(command, "--summary"))
        checkpoint_path = Path(_option_value(command, "--checkpoint"))
        if len(commands) == 1:
            assert not (stage_dir / "plateau_model.pt").exists()
            _write_terminal_round(
                model_path,
                summary_path,
                model_bytes=b"round-one-model",
                termination_reason="completed_requested_epochs",
            )
            checkpoint_path.write_bytes(b"round-one-checkpoint")
        else:
            assert not (stage_dir / "plateau_model.pt").exists()
            _write_terminal_round(
                model_path,
                summary_path,
                model_bytes=b"round-two-model",
                termination_reason="validation_plateau",
            )
            checkpoint_path.write_bytes(b"round-two-checkpoint-after")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(plateau_stage.subprocess, "run", fake_run)
    result = plateau_stage.run_stage(
        args,
        trainer_args,
        trainer=fake_trainer,
        python_executable="python-for-test",
    )

    assert result == 0
    assert len(commands) == 2
    assert commands[0][:2] == ["python-for-test", str(fake_trainer.resolve())]
    assert commands[0][2:8] == [
        "--source-artifact",
        "source.npz",
        "--bond-dimension",
        "8",
        "--device",
        "cuda",
    ]
    assert _option_value(commands[0], "--initial-model") == str(initial_model.resolve())
    assert _option_value(commands[0], "--kappa-source") == "initial_model"
    assert "--resume-checkpoint" not in commands[0]

    assert _option_value(commands[1], "--initial-model") == str(
        (stage_dir / "round_001" / "model.pt").resolve()
    )
    assert _option_value(commands[1], "--kappa-source") == "saved_model"
    assert _option_value(commands[1], "--resume-checkpoint") == str(
        (round_two / "checkpoint.pt").resolve()
    )

    plateau_model = stage_dir / "plateau_model.pt"
    plateau_summary = stage_dir / "plateau_summary.json"
    stage_summary_path = stage_dir / "stage_summary.json"
    assert plateau_model.read_bytes() == b"round-two-model"
    assert plateau_summary.read_bytes() == (round_two / "summary.json").read_bytes()
    stage_summary = json.loads(stage_summary_path.read_text(encoding="utf-8"))
    assert stage_summary["status"] == "validation_plateau"
    assert stage_summary["plateau_round"] == 2
    assert [row["status"] for row in stage_summary["rounds"]] == [
        "completed_non_plateau",
        "validation_plateau",
    ]
    assert stage_summary["plateau_model_sha256"] == plateau_stage.sha256_file(
        plateau_model
    )
    assert stage_summary["plateau_summary_sha256"] == plateau_stage.sha256_file(
        plateau_summary
    )
    assert stage_summary["rounds"][1]["checkpoint_sha256"] == (
        plateau_stage.sha256_file(round_two / "checkpoint.pt")
    )


def test_hash_valid_completed_round_is_not_rerun_and_exhaustion_is_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage_dir = tmp_path / "stage"
    round_one = stage_dir / "round_001"
    _write_terminal_round(
        round_one / "model.pt",
        round_one / "summary.json",
        model_bytes=b"already-complete",
        termination_reason="completed_requested_epochs",
    )
    fake_trainer = tmp_path / "fake_trainer.py"
    fake_trainer.write_text("# not called\n")
    args, trainer_args = plateau_stage.parse_cli(
        ["--stage-dir", str(stage_dir), "--max-rounds", "1", "--", "--epochs", "2"]
    )

    def unexpected_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
        raise AssertionError("completed round was rerun")

    monkeypatch.setattr(plateau_stage.subprocess, "run", unexpected_run)
    result = plateau_stage.run_stage(args, trainer_args, trainer=fake_trainer)

    assert result != 0
    assert not (stage_dir / "plateau_model.pt").exists()
    assert not (stage_dir / "plateau_summary.json").exists()
    assert not (stage_dir / "stage_summary.json").exists()


def test_nonfinite_terminal_round_is_not_continued(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage_dir = tmp_path / "stage"
    round_one = stage_dir / "round_001"
    _write_terminal_round(
        round_one / "model.pt",
        round_one / "summary.json",
        model_bytes=b"nonfinite-model",
        termination_reason="nonfinite_training_loss",
    )
    fake_trainer = tmp_path / "fake_trainer.py"
    fake_trainer.write_text("# not called\n")
    args, trainer_args = plateau_stage.parse_cli(
        ["--stage-dir", str(stage_dir), "--max-rounds", "2", "--", "--epochs", "2"]
    )

    def unexpected_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
        raise AssertionError("nonfinite terminal round was continued")

    monkeypatch.setattr(plateau_stage.subprocess, "run", unexpected_run)
    with pytest.raises(plateau_stage.StageError, match="non-continuable"):
        plateau_stage.run_stage(args, trainer_args, trainer=fake_trainer)

    assert not (stage_dir / "plateau_model.pt").exists()
    assert not (stage_dir / "stage_summary.json").exists()


@pytest.mark.parametrize(
    "reserved_argument",
    [
        "--out=forbidden.pt",
        "--summary=forbidden.json",
        "--checkpoint=forbidden.pt",
        "--resume-checkpoint=forbidden.pt",
        "--initial-model=forbidden.pt",
        "--kappa-source=auto",
    ],
)
def test_cli_rejects_wrapper_managed_passthrough_options_in_subprocess(
    tmp_path: Path,
    reserved_argument: str,
) -> None:
    script = Path(plateau_stage.__file__).resolve()
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--stage-dir",
            str(tmp_path / "stage"),
            "--",
            reserved_argument,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "wrapper-managed option" in result.stderr
    assert not (tmp_path / "stage").exists()
