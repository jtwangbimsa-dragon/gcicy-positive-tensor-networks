#!/usr/bin/env python3
"""Run restartable gCICY tensor-network training rounds to a plateau.

All trainer-owned outputs live below an independent round directory.  The
stage-level artifacts are published only after a completed round reports the
``validation_plateau`` termination reason.  ``stage_summary.json`` is replaced
last, so it acts as the commit marker for the three-file publication.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "scripts" / "train_type11_positive_tensor_network.py"

KAPPA_SOURCES = ("auto", "initial_model", "saved_model", "teacher")
CONTINUABLE_TERMINATION_REASONS = frozenset({"completed_requested_epochs"})
RESERVED_TRAINER_OPTIONS = frozenset(
    {
        "--out",
        "--summary",
        "--checkpoint",
        "--resume-checkpoint",
        "--initial-model",
        "--kappa-source",
    }
)


class StageError(RuntimeError):
    """A user-facing stage orchestration failure."""

    def __init__(self, message: str, *, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def child_failure_exit_code(returncode: int) -> int:
    """Preserve child signals with the conventional shell 128+signal code."""

    if returncode < 0:
        return min(255, 128 + (-returncode))
    return returncode if 1 <= returncode <= 255 else 1


@dataclass(frozen=True)
class CompletedRound:
    """Validated terminal artifacts for one trainer invocation."""

    termination_reason: str
    model_sha256: str
    summary_sha256: str
    summary_payload: dict


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _option_name(token: str) -> str:
    return token.split("=", 1)[0]


def _reject_reserved_trainer_options(
    parser: argparse.ArgumentParser,
    trainer_args: Sequence[str],
) -> None:
    conflicts = sorted(
        {
            _option_name(token)
            for token in trainer_args
            if _option_name(token) in RESERVED_TRAINER_OPTIONS
        }
    )
    if conflicts:
        parser.error(
            "trainer passthrough may not set wrapper-managed option(s): "
            + ", ".join(conflicts)
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
        epilog=(
            "Unrecognized options are passed unchanged to the trainer.  A '--' "
            "separator is recommended before trainer options."
        ),
    )
    parser.add_argument(
        "--initial-model",
        type=Path,
        help="optional model used to initialize the first round",
    )
    parser.add_argument(
        "--stage-dir",
        type=Path,
        required=True,
        help="independent output directory for this plateau stage",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=5,
        help="maximum completed training rounds before failing (default: 5)",
    )
    parser.add_argument(
        "--first-kappa-source",
        "--first-round-kappa-source",
        dest="first_kappa_source",
        choices=KAPPA_SOURCES,
        default="initial_model",
        help="trainer kappa source for round 1 (default: initial_model)",
    )
    parser.add_argument(
        "--continuation-kappa-source",
        choices=KAPPA_SOURCES,
        default="saved_model",
        help="trainer kappa source after round 1 (default: saved_model)",
    )
    return parser


def parse_cli(
    argv: Sequence[str] | None = None,
) -> tuple[argparse.Namespace, list[str]]:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if "--" in tokens:
        separator = tokens.index("--")
        wrapper_tokens = tokens[:separator]
        explicit_trainer_args = tokens[separator + 1 :]
    else:
        wrapper_tokens = tokens
        explicit_trainer_args = []

    parser = build_parser()
    args, implicit_trainer_args = parser.parse_known_args(wrapper_tokens)
    trainer_args = [*implicit_trainer_args, *explicit_trainer_args]
    _reject_reserved_trainer_options(parser, trainer_args)
    if args.max_rounds <= 0:
        parser.error("--max-rounds must be positive")
    return args, trainer_args


def inspect_completed_round(
    model_path: Path, summary_path: Path
) -> CompletedRound | None:
    """Return a round only when its model and terminal summary agree by hash."""

    if not model_path.is_file() or not summary_path.is_file():
        return None
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    termination_reason = payload.get("termination_reason")
    expected_model_sha256 = payload.get("model_sha256")
    if not isinstance(termination_reason, str) or not termination_reason.strip():
        return None
    if not isinstance(expected_model_sha256, str) or not expected_model_sha256:
        return None
    try:
        model_sha256 = sha256_file(model_path)
        summary_sha256 = sha256_file(summary_path)
    except OSError:
        return None
    if expected_model_sha256.lower() != model_sha256:
        return None
    return CompletedRound(
        termination_reason=termination_reason,
        model_sha256=model_sha256,
        summary_sha256=summary_sha256,
        summary_payload=payload,
    )


def _prepare_atomic_copy(source: Path, destination: Path) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _prepare_atomic_json(payload: dict, destination: Path) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _publish_plateau(
    *,
    stage_dir: Path,
    model_path: Path,
    summary_path: Path,
    stage_summary: dict,
) -> None:
    plateau_model = stage_dir / "plateau_model.pt"
    plateau_summary = stage_dir / "plateau_summary.json"
    stage_summary_path = stage_dir / "stage_summary.json"
    temporary_paths: list[Path] = []
    try:
        temporary_paths.append(_prepare_atomic_copy(model_path, plateau_model))
        temporary_paths.append(_prepare_atomic_copy(summary_path, plateau_summary))
        temporary_paths.append(_prepare_atomic_json(stage_summary, stage_summary_path))
        os.replace(temporary_paths[0], plateau_model)
        os.replace(temporary_paths[1], plateau_summary)
        # Replacing the manifest last makes it the publication commit marker.
        os.replace(temporary_paths[2], stage_summary_path)
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)


def _require_input_model(path: Path | None, *, round_number: int) -> str | None:
    if path is None:
        return None
    if not path.is_file():
        raise StageError(f"round {round_number} initial model does not exist: {path}")
    return sha256_file(path)


def _round_record(
    *,
    stage_dir: Path,
    round_number: int,
    initial_model: Path | None,
    initial_model_sha256: str | None,
    kappa_source: str,
    model_path: Path,
    summary_path: Path,
    checkpoint_path: Path,
    completed: CompletedRound,
) -> dict:
    checkpoint_sha256 = (
        sha256_file(checkpoint_path) if checkpoint_path.is_file() else None
    )
    return {
        "round": round_number,
        "status": (
            "validation_plateau"
            if completed.termination_reason == "validation_plateau"
            else "completed_non_plateau"
        ),
        "termination_reason": completed.termination_reason,
        "kappa_source": kappa_source,
        "initial_model": None if initial_model is None else str(initial_model),
        "initial_model_sha256": initial_model_sha256,
        "model": str(model_path.relative_to(stage_dir)),
        "model_sha256": completed.model_sha256,
        "summary": str(summary_path.relative_to(stage_dir)),
        "summary_sha256": completed.summary_sha256,
        "checkpoint": str(checkpoint_path.relative_to(stage_dir)),
        "checkpoint_sha256": checkpoint_sha256,
    }


def run_stage(
    args: argparse.Namespace,
    trainer_args: Sequence[str],
    *,
    trainer: Path = TRAINER,
    python_executable: str = sys.executable,
) -> int:
    stage_dir = args.stage_dir.expanduser().resolve()
    stage_dir.mkdir(parents=True, exist_ok=True)
    trainer = trainer.expanduser().resolve()
    if not trainer.is_file():
        raise StageError(f"trainer does not exist: {trainer}")

    initial_model = (
        None
        if args.initial_model is None
        else args.initial_model.expanduser().resolve()
    )
    initial_model_sha256 = _require_input_model(initial_model, round_number=1)
    stage_initial_model = initial_model
    stage_initial_model_sha256 = initial_model_sha256
    records: list[dict] = []

    for round_number in range(1, args.max_rounds + 1):
        round_dir = stage_dir / f"round_{round_number:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        model_path = round_dir / "model.pt"
        summary_path = round_dir / "summary.json"
        checkpoint_path = round_dir / "checkpoint.pt"
        if initial_model in {model_path, summary_path, checkpoint_path}:
            raise StageError(
                f"round {round_number} initial model conflicts with a managed output"
            )

        kappa_source = (
            args.first_kappa_source
            if round_number == 1
            else args.continuation_kappa_source
        )
        completed = inspect_completed_round(model_path, summary_path)
        if completed is None:
            resume_checkpoint = checkpoint_path.is_file()
            command = [python_executable, str(trainer), *trainer_args]
            if initial_model is not None:
                command.extend(("--initial-model", str(initial_model)))
            command.extend(
                (
                    "--kappa-source",
                    kappa_source,
                    "--out",
                    str(model_path),
                    "--summary",
                    str(summary_path),
                    "--checkpoint",
                    str(checkpoint_path),
                )
            )
            if resume_checkpoint:
                command.extend(("--resume-checkpoint", str(checkpoint_path)))
            mode = "resume" if resume_checkpoint else "fresh"
            print(
                f"[plateau-stage] round {round_number}/{args.max_rounds}: {mode}",
                flush=True,
            )
            try:
                result = subprocess.run(command, check=False)
            except OSError as error:
                raise StageError(
                    f"could not start trainer for round {round_number}: {error}"
                ) from error
            if result.returncode != 0:
                raise StageError(
                    f"trainer failed in round {round_number} with exit code "
                    f"{result.returncode}; checkpoint retained at {checkpoint_path}",
                    exit_code=child_failure_exit_code(result.returncode),
                )
            completed = inspect_completed_round(model_path, summary_path)
            if completed is None:
                raise StageError(
                    f"trainer round {round_number} exited successfully without a "
                    "hash-valid terminal model and summary"
                )
        else:
            print(
                f"[plateau-stage] round {round_number}/{args.max_rounds}: "
                "reuse completed artifacts",
                flush=True,
            )

        record = _round_record(
            stage_dir=stage_dir,
            round_number=round_number,
            initial_model=initial_model,
            initial_model_sha256=initial_model_sha256,
            kappa_source=kappa_source,
            model_path=model_path,
            summary_path=summary_path,
            checkpoint_path=checkpoint_path,
            completed=completed,
        )
        records.append(record)
        if completed.termination_reason == "validation_plateau":
            plateau_model_sha256 = completed.model_sha256
            plateau_summary_sha256 = completed.summary_sha256
            stage_summary = {
                "schema": "gcicy-tn-plateau-stage-v1",
                "status": "validation_plateau",
                "termination_reason": "validation_plateau",
                "plateau_round": round_number,
                "rounds_completed": len(records),
                "max_rounds": args.max_rounds,
                "first_kappa_source": args.first_kappa_source,
                "continuation_kappa_source": args.continuation_kappa_source,
                "initial_model": (
                    None if stage_initial_model is None else str(stage_initial_model)
                ),
                "initial_model_sha256": stage_initial_model_sha256,
                "trainer": str(trainer),
                "trainer_sha256": sha256_file(trainer),
                "trainer_args": list(trainer_args),
                "plateau_model": "plateau_model.pt",
                "plateau_model_sha256": plateau_model_sha256,
                "plateau_summary": "plateau_summary.json",
                "plateau_summary_sha256": plateau_summary_sha256,
                "rounds": records,
            }
            _publish_plateau(
                stage_dir=stage_dir,
                model_path=model_path,
                summary_path=summary_path,
                stage_summary=stage_summary,
            )
            print(
                f"[plateau-stage] plateau published from round {round_number}: "
                f"{stage_dir}",
                flush=True,
            )
            return 0

        if completed.termination_reason not in CONTINUABLE_TERMINATION_REASONS:
            raise StageError(
                f"trainer round {round_number} ended with non-continuable "
                f"termination reason {completed.termination_reason!r}; retained "
                f"artifacts at {round_dir}"
            )

        initial_model = model_path
        initial_model_sha256 = completed.model_sha256

    print(
        f"[plateau-stage] no validation plateau after {args.max_rounds} round(s); "
        "stage artifacts were not published",
        file=sys.stderr,
        flush=True,
    )
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    args, trainer_args = parse_cli(argv)
    try:
        return run_stage(args, trainer_args)
    except StageError as error:
        print(f"[plateau-stage] error: {error}", file=sys.stderr, flush=True)
        return error.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
