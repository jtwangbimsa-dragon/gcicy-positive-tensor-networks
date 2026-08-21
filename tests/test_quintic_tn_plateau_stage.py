from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_quintic_tn_plateau_stage as plateau


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def _write_round(command: list[str], score: float, content: bytes) -> None:
    output = Path(_option(command, "--output-dir"))
    model = output / "best_tensor_network.pt"
    report = output / "report.json"
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(content)
    report.write_text(
        json.dumps(
            {
                "schema": plateau.REPORT_SCHEMA,
                "termination_reason": "completed_requested_epochs",
                "evaluation_scope": "development_only",
                "training": {
                    "parameter_scope": "joint",
                    "best_selection_score": score,
                },
                "artifacts": {"model_sha256": plateau.sha256_file(model)},
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_cold_stage_plateaus_across_complete_invocations_and_reuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = tmp_path / "trainer.py"
    trainer.write_text("# fake\n", encoding="utf-8")
    stage_dir = tmp_path / "stage"
    args, trainer_args = plateau.parse_cli(
        [
            "--stage-dir",
            str(stage_dir),
            "--parameter-scope",
            "joint",
            "--max-rounds",
            "4",
            "--min-relative-gain",
            "0.05",
            "--plateau-patience-rounds",
            "1",
            "--",
            "--epochs",
            "1",
        ]
    )
    commands: list[list[str]] = []
    scores = [10.0, 9.0, 9.1]

    def fake_run(command: list[str], *, check: bool) -> subprocess.CompletedProcess:
        assert check is False
        commands.append(command)
        _write_round(command, scores[len(commands) - 1], f"m{len(commands)}".encode())
        Path(_option(command, "--checkpoint")).write_bytes(b"checkpoint")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(plateau.subprocess, "run", fake_run)
    assert (
        plateau.run_stage(
            args, trainer_args, trainer=trainer, python_executable="python-test"
        )
        == 0
    )
    assert len(commands) == 3
    assert "--initial-model" not in commands[0]
    assert "--initial-model" in commands[1]
    assert (stage_dir / "plateau_model.pt").read_bytes() == b"m2"
    summary = json.loads((stage_dir / "stage_summary.json").read_text())
    assert summary["plateau_trigger_round"] == 3
    assert summary["selected_round"] == 2

    monkeypatch.setattr(
        plateau.subprocess,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("reran stage")),
    )
    assert plateau.run_stage(args, trainer_args, trainer=trainer) == 0


def test_new_channels_requires_initial_model_and_nested_signal_maps() -> None:
    with pytest.raises(SystemExit):
        plateau.parse_cli(
            [
                "--stage-dir",
                "stage",
                "--parameter-scope",
                "new_channels",
                "--inherited-bond-dimension",
                "10",
            ]
        )
    assert plateau.child_failure_exit_code(-15) == 143
    assert plateau.child_failure_exit_code(137) == 137


def test_registered_single_round_cap_publishes_best(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = tmp_path / "trainer.py"
    trainer.write_text("# fake\n", encoding="utf-8")
    stage_dir = tmp_path / "fixed"
    args, trainer_args = plateau.parse_cli(
        [
            "--stage-dir",
            str(stage_dir),
            "--parameter-scope",
            "joint",
            "--max-rounds",
            "1",
            "--accept-round-cap",
            "--",
            "--epochs",
            "50",
        ]
    )

    def fake_run(command: list[str], *, check: bool) -> subprocess.CompletedProcess:
        _write_round(command, 3.0, b"fixed-budget")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(plateau.subprocess, "run", fake_run)
    assert plateau.run_stage(args, trainer_args, trainer=trainer) == 0
    summary = json.loads((stage_dir / "stage_summary.json").read_text())
    assert summary["status"] == "completed_registered_round_cap"
    assert summary["termination_reason"] == "completed_registered_round_cap"
    assert summary["rounds_completed"] == 1


@pytest.mark.parametrize(
    "option",
    [
        "--output-dir=x",
        "--initial-model=x",
        "--checkpoint=x",
        "--resume-checkpoint=x",
        "--skip-blind-audit",
        "--freeze-physical-dictionary",
    ],
)
def test_managed_trainer_options_are_rejected(option: str) -> None:
    with pytest.raises(SystemExit):
        plateau.parse_cli(
            ["--stage-dir", "stage", "--parameter-scope", "joint", "--", option]
        )
