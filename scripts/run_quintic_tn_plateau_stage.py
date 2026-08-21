#!/usr/bin/env python3
"""Run restartable quintic positive-TN Adam rounds until validation plateaus."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "scripts" / "train_quintic_positive_tensor_network_same_points.py"
SCHEMA = "quintic-tn-plateau-stage-v1"
REPORT_SCHEMA = "quintic-positive-tensor-network-same-points-v1"
SCOPES = ("new_channels", "cores", "joint")
RESERVED_TRAINER_OPTIONS = frozenset(
    {
        "--output-dir",
        "--initial-model",
        "--checkpoint",
        "--resume-checkpoint",
        "--skip-blind-audit",
        "--freeze-inherited-bond-dimension",
        "--freeze-physical-dictionary",
    }
)


class StageError(RuntimeError):
    def __init__(self, message: str, *, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


@dataclass(frozen=True)
class CompletedRound:
    model_sha256: str
    report_sha256: str
    score: float
    report: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _prepared_copy(source: Path, destination: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        return temporary
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def child_failure_exit_code(returncode: int) -> int:
    if returncode < 0:
        return min(255, 128 + (-returncode))
    return returncode if 1 <= returncode <= 255 else 1


def _option_name(token: str) -> str:
    return token.split("=", 1)[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
        epilog="Put '--' before arguments passed to the trainer.",
    )
    parser.add_argument(
        "--initial-model",
        type=Path,
        help="optional for a genuinely cold first round",
    )
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--min-relative-gain", type=float, default=1e-3)
    parser.add_argument("--plateau-patience-rounds", type=int, default=2)
    parser.add_argument(
        "--accept-round-cap",
        action="store_true",
        help=(
            "publish the best completed round when max-rounds is reached; "
            "used for registered fixed-budget stages"
        ),
    )
    parser.add_argument("--parameter-scope", choices=SCOPES, required=True)
    parser.add_argument(
        "--inherited-bond-dimension",
        type=int,
        default=0,
        help="required only for new_channels",
    )
    return parser


def parse_cli(
    argv: Sequence[str] | None = None,
) -> tuple[argparse.Namespace, list[str]]:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if "--" in tokens:
        separator = tokens.index("--")
        wrapper_tokens = tokens[:separator]
        explicit_trainer = tokens[separator + 1 :]
    else:
        wrapper_tokens = tokens
        explicit_trainer = []
    parser = build_parser()
    args, implicit_trainer = parser.parse_known_args(wrapper_tokens)
    trainer_args = [*implicit_trainer, *explicit_trainer]
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
    if args.max_rounds <= 0:
        parser.error("--max-rounds must be positive")
    if args.plateau_patience_rounds <= 0:
        parser.error("--plateau-patience-rounds must be positive")
    if not math.isfinite(args.min_relative_gain) or args.min_relative_gain < 0:
        parser.error("--min-relative-gain must be finite and non-negative")
    if args.parameter_scope == "new_channels":
        if args.inherited_bond_dimension <= 0:
            parser.error("new_channels requires --inherited-bond-dimension > 0")
        if args.initial_model is None:
            parser.error("new_channels requires --initial-model")
    elif args.inherited_bond_dimension:
        parser.error("--inherited-bond-dimension is valid only for new_channels")
    return args, trainer_args


def _nested(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def inspect_completed_round(
    model_path: Path,
    report_path: Path,
    *,
    expected_scope: str,
) -> CompletedRound | None:
    if not model_path.is_file() or not report_path.is_file():
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        model_sha = sha256_file(model_path)
        report_sha = sha256_file(report_path)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(report, dict) or report.get("schema") != REPORT_SCHEMA:
        return None
    if report.get("termination_reason") != "completed_requested_epochs":
        return None
    if report.get("evaluation_scope") != "development_only":
        return None
    if _nested(report, "training", "parameter_scope") != expected_scope:
        return None
    if _nested(report, "artifacts", "model_sha256") != model_sha:
        return None
    score = _nested(report, "training", "best_selection_score")
    if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
        return None
    return CompletedRound(model_sha, report_sha, float(score), report)


def _valid_publication(stage_dir: Path, config_sha256: str) -> bool:
    summary_path = stage_dir / "stage_summary.json"
    model_path = stage_dir / "plateau_model.pt"
    report_path = stage_dir / "plateau_report.json"
    if not all(path.is_file() for path in (summary_path, model_path, report_path)):
        return False
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        return bool(
            isinstance(payload, dict)
            and payload.get("schema") == SCHEMA
            and payload.get("status")
            in {"validation_plateau", "completed_registered_round_cap"}
            and payload.get("configuration_sha256") == config_sha256
            and payload.get("plateau_model_sha256") == sha256_file(model_path)
            and payload.get("plateau_report_sha256") == sha256_file(report_path)
        )
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False


def _publish(
    stage_dir: Path,
    source_model: Path,
    source_report: Path,
    summary: dict[str, Any],
) -> None:
    model_path = stage_dir / "plateau_model.pt"
    report_path = stage_dir / "plateau_report.json"
    summary_path = stage_dir / "stage_summary.json"
    temporaries = [
        _prepared_copy(source_model, model_path),
        _prepared_copy(source_report, report_path),
    ]
    try:
        os.replace(temporaries[0], model_path)
        os.replace(temporaries[1], report_path)
        # The summary is the publication commit marker and is always last.
        _atomic_json(summary_path, summary)
    finally:
        for path in temporaries:
            path.unlink(missing_ok=True)


def _scope_arguments(args: argparse.Namespace) -> list[str]:
    if args.parameter_scope == "new_channels":
        return [
            "--freeze-inherited-bond-dimension",
            str(args.inherited_bond_dimension),
            "--freeze-physical-dictionary",
        ]
    if args.parameter_scope == "cores":
        return ["--freeze-physical-dictionary"]
    return []


def run_stage(
    args: argparse.Namespace,
    trainer_args: Sequence[str],
    *,
    trainer: Path = TRAINER,
    python_executable: str = sys.executable,
) -> int:
    stage_dir = args.stage_dir.expanduser().resolve()
    stage_dir.mkdir(parents=True, exist_ok=True)
    initial_model = (
        None
        if args.initial_model is None
        else args.initial_model.expanduser().resolve()
    )
    trainer = trainer.expanduser().resolve()
    if initial_model is not None and not initial_model.is_file():
        raise StageError(f"initial model does not exist: {initial_model}")
    if not trainer.is_file():
        raise StageError(f"trainer does not exist: {trainer}")
    trainer_sha = sha256_file(trainer)
    initial_sha = None if initial_model is None else sha256_file(initial_model)
    configuration = {
        "schema": SCHEMA,
        "initial_model_sha256": initial_sha,
        "trainer_sha256": trainer_sha,
        "trainer_args": list(trainer_args),
        "max_rounds": int(args.max_rounds),
        "min_relative_gain": float(args.min_relative_gain),
        "plateau_patience_rounds": int(args.plateau_patience_rounds),
        "accept_round_cap": bool(args.accept_round_cap),
        "parameter_scope": args.parameter_scope,
        "inherited_bond_dimension": int(args.inherited_bond_dimension),
    }
    config_sha = canonical_sha256(configuration)
    if _valid_publication(stage_dir, config_sha):
        print(f"[quintic-plateau] reuse published stage {stage_dir}", flush=True)
        return 0

    current_model = initial_model
    current_sha = initial_sha
    best: tuple[int, CompletedRound, Path, Path] | None = None
    stale_rounds = 0
    records: list[dict[str, Any]] = []

    for round_number in range(1, args.max_rounds + 1):
        request = {
            "schema": "quintic-tn-plateau-round-request-v1",
            "stage_configuration_sha256": config_sha,
            "round": round_number,
            "input_model_sha256": current_sha,
            "trainer_sha256": trainer_sha,
            "trainer_args": list(trainer_args),
            "parameter_scope": args.parameter_scope,
            "inherited_bond_dimension": int(args.inherited_bond_dimension),
        }
        request_sha = canonical_sha256(request)
        round_dir = stage_dir / f"round_{round_number:03d}_{request_sha[:16]}"
        output_dir = round_dir / "trainer"
        checkpoint = round_dir / "checkpoint.pt"
        request_path = round_dir / "request.json"
        model_path = output_dir / "best_tensor_network.pt"
        report_path = output_dir / "report.json"
        round_dir.mkdir(parents=True, exist_ok=True)
        if request_path.is_file():
            try:
                existing_request = json.loads(request_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise StageError(
                    f"invalid round request marker: {request_path}"
                ) from error
            if existing_request != request:
                raise StageError(f"round request digest collision at {round_dir}")
        else:
            _atomic_json(request_path, request)

        completed = inspect_completed_round(
            model_path, report_path, expected_scope=args.parameter_scope
        )
        if completed is None:
            command = [
                python_executable,
                str(trainer),
                *trainer_args,
                "--output-dir",
                str(output_dir),
                "--checkpoint",
                str(checkpoint),
                "--skip-blind-audit",
                *_scope_arguments(args),
            ]
            if current_model is not None:
                command.extend(("--initial-model", str(current_model)))
            resume = checkpoint.is_file()
            if resume:
                command.extend(("--resume-checkpoint", str(checkpoint)))
            print(
                f"[quintic-plateau] round {round_number}/{args.max_rounds}: "
                f"{'resume' if resume else 'fresh'}",
                flush=True,
            )
            try:
                result = subprocess.run(command, check=False)
            except OSError as error:
                raise StageError(f"could not start trainer: {error}") from error
            if result.returncode != 0:
                raise StageError(
                    f"trainer failed in round {round_number} with exit code "
                    f"{result.returncode}; checkpoint retained at {checkpoint}",
                    exit_code=child_failure_exit_code(result.returncode),
                )
            completed = inspect_completed_round(
                model_path, report_path, expected_scope=args.parameter_scope
            )
            if completed is None:
                raise StageError(
                    "trainer exited successfully without a hash-valid "
                    "development-only terminal result"
                )
        else:
            print(f"[quintic-plateau] reuse completed round {round_number}", flush=True)

        improved = False
        if best is None:
            improved = True
        else:
            gain = best[1].score - completed.score
            threshold = abs(best[1].score) * float(args.min_relative_gain)
            improved = gain > threshold
        if improved:
            best = (round_number, completed, model_path, report_path)
            stale_rounds = 0
        else:
            stale_rounds += 1
        records.append(
            {
                "round": round_number,
                "request": str(request_path.relative_to(stage_dir)),
                "request_sha256": request_sha,
                "input_model": None if current_model is None else str(current_model),
                "input_model_sha256": current_sha,
                "model": str(model_path.relative_to(stage_dir)),
                "model_sha256": completed.model_sha256,
                "report": str(report_path.relative_to(stage_dir)),
                "report_sha256": completed.report_sha256,
                "checkpoint": str(checkpoint.relative_to(stage_dir)),
                "checkpoint_sha256": (
                    sha256_file(checkpoint) if checkpoint.is_file() else None
                ),
                "best_selection_score": completed.score,
                "material_improvement": improved,
                "stale_rounds": stale_rounds,
            }
        )
        plateau_reached = stale_rounds >= args.plateau_patience_rounds
        cap_accepted = round_number == args.max_rounds and args.accept_round_cap
        if plateau_reached or cap_accepted:
            assert best is not None
            selected_round, selected, selected_model, selected_report = best
            termination_reason = (
                "validation_plateau"
                if plateau_reached
                else "completed_registered_round_cap"
            )
            summary = {
                **configuration,
                "status": termination_reason,
                "termination_reason": termination_reason,
                "configuration_sha256": config_sha,
                "initial_model": (
                    None if initial_model is None else str(initial_model)
                ),
                "trainer": str(trainer),
                "plateau_trigger_round": round_number,
                "selected_round": selected_round,
                "rounds_completed": len(records),
                "selected_best_selection_score": selected.score,
                "plateau_model": "plateau_model.pt",
                "plateau_model_sha256": selected.model_sha256,
                "plateau_report": "plateau_report.json",
                "plateau_report_sha256": selected.report_sha256,
                "rounds": records,
            }
            _publish(stage_dir, selected_model, selected_report, summary)
            print(
                f"[quintic-plateau] {termination_reason} at round {round_number}; "
                f"selected round {selected_round}",
                flush=True,
            )
            return 0
        current_model = model_path
        current_sha = completed.model_sha256

    print(
        f"[quintic-plateau] no plateau after {args.max_rounds} round(s); "
        "no stage publication",
        file=sys.stderr,
        flush=True,
    )
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    args, trainer_args = parse_cli(argv)
    try:
        return run_stage(args, trainer_args)
    except StageError as error:
        print(f"quintic plateau stage failed: {error}", file=sys.stderr)
        return error.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
