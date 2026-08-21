#!/usr/bin/env python3
"""Run one restartable k/D/optimization study arm from a saved gCICY TN.

The wrapper never mutates its initial model or common point pools.  Every
intermediate step is content-addressed by its inputs, parameters, and child
script hashes.  ``final_summary.json`` is replaced last and is therefore the
commit marker for the published ``final_model.pt``/``arm_summary.json`` pair.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SITE_TRANSFER = SCRIPTS / "resize_positive_tensor_network_sites.py"
BOND_TRANSFER = SCRIPTS / "expand_positive_tensor_network_bond.py"
EQUIVALENCE_AUDIT = SCRIPTS / "audit_positive_tensor_network_equivalence.py"
PLATEAU_STAGE = SCRIPTS / "run_gcicy_tn_plateau_stage.py"
TRAINER = SCRIPTS / "train_type11_positive_tensor_network.py"
BLIND_AUDIT = SCRIPTS / "audit_type11_positive_tensor_network.py"
TAIL_EVALUATOR = SCRIPTS / "evaluate_gcicy_metric_tail_arrays.py"

MODEL_SCHEMA = "type11-positive-tensor-network-v1"
SCRIPT_PATHS = {
    "site_transfer": SITE_TRANSFER,
    "bond_transfer": BOND_TRANSFER,
    "equivalence_audit": EQUIVALENCE_AUDIT,
    "plateau_stage": PLATEAU_STAGE,
    "trainer": TRAINER,
    "blind_audit": BLIND_AUDIT,
    "tail_evaluator": TAIL_EVALUATOR,
}
STAGE_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


class ArmError(RuntimeError):
    """A user-facing study-arm orchestration failure."""

    def __init__(self, message: str, *, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def child_failure_exit_code(returncode: int) -> int:
    """Preserve nested signals with the conventional shell 128+signal code."""

    if returncode < 0:
        return min(255, 128 + (-returncode))
    return returncode if 1 <= returncode <= 255 else 1


@dataclass(frozen=True)
class StageSpec:
    name: str
    learning_rate: float
    min_relative_gain: float
    patience: int
    epochs: int
    max_rounds: int


@dataclass(frozen=True)
class ModelMetadata:
    schema: str
    site_count: int
    bond_dimension: int
    architecture: str
    trainable_physical_dictionary: bool
    precision: str
    positive_floor: float
    source_artifact_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _same_number(left: Any, right: float) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=0.0)
    except (TypeError, ValueError):
        return False


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _require_file(path: Path, role: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ArmError(f"{role} does not exist: {resolved}")
    return resolved


def read_model_metadata(path: Path) -> ModelMetadata:
    """Read only the small architecture metadata from a saved CPU artifact."""

    try:
        import torch
    except ImportError as error:
        raise ArmError("PyTorch is required to inspect the initial TN model") from error
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ArmError(f"could not read initial TN model {path}: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema") != MODEL_SCHEMA:
        raise ArmError(f"initial model is not a {MODEL_SCHEMA} artifact: {path}")
    try:
        metadata = ModelMetadata(
            schema=str(payload["schema"]),
            site_count=int(payload["site_count"]),
            bond_dimension=int(payload["bond_dimension"]),
            architecture=str(payload.get("architecture", "dense_local_cores")),
            trainable_physical_dictionary=bool(
                payload.get("trainable_physical_dictionary", False)
            ),
            precision=str(payload["precision"]),
            positive_floor=float(payload["positive_floor"]),
            source_artifact_sha256=str(payload["source_artifact_sha256"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ArmError(f"initial model has incomplete TN metadata: {path}") from error
    if metadata.site_count < 3 or metadata.bond_dimension < 1:
        raise ArmError("initial model has invalid site or bond dimensions")
    return metadata


def parse_stage(value: str) -> StageSpec:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError(
            "--stage requires name,lr,min_gain,patience,epochs,max_rounds"
        )
    name = parts[0]
    if not STAGE_NAME_PATTERN.fullmatch(name):
        raise argparse.ArgumentTypeError(
            "stage name must contain only letters, digits, '.', '_', or '-'"
        )
    try:
        learning_rate = float(parts[1])
        min_relative_gain = float(parts[2])
        patience = int(parts[3])
        epochs = int(parts[4])
        max_rounds = int(parts[5])
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"invalid numeric stage field: {value}"
        ) from error
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise argparse.ArgumentTypeError("stage learning rate must be positive")
    if not math.isfinite(min_relative_gain) or min_relative_gain < 0.0:
        raise argparse.ArgumentTypeError("stage min_gain must be non-negative")
    if patience <= 0 or epochs <= 0 or max_rounds <= 0:
        raise argparse.ArgumentTypeError(
            "stage patience, epochs, and max_rounds must be positive"
        )
    return StageSpec(
        name=name,
        learning_rate=learning_rate,
        min_relative_gain=min_relative_gain,
        patience=patience,
        epochs=epochs,
        max_rounds=max_rounds,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--initial-model", type=Path, required=True)
    parser.add_argument("--arm-dir", type=Path, required=True)
    parser.add_argument("--target-k", type=int, required=True)
    parser.add_argument(
        "--target-d", "--target-D", dest="target_d", type=int, required=True
    )
    parser.add_argument(
        "--stage",
        action="append",
        type=parse_stage,
        required=True,
        metavar="NAME,LR,MIN_GAIN,PATIENCE,EPOCHS,MAX_ROUNDS",
    )
    parser.add_argument("--train-physical-dictionary", action="store_true")

    parser.add_argument("--adapter", default="p4p1_type11_hirzebruch_x3")
    parser.add_argument("--model-seed", type=int, default=20260731)
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--teacher-artifact", type=Path)
    parser.add_argument("--sampling-cluster-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--teacher-chunk-size", type=int, default=512)

    parser.add_argument("--train-common-pool", type=Path, required=True)
    parser.add_argument(
        "--selection-common-pool",
        "--validation-common-pool",
        dest="selection_common_pool",
        type=Path,
        required=True,
    )
    parser.add_argument("--development-common-pool", type=Path, required=True)
    parser.add_argument(
        "--development-pool-split",
        choices=("train", "selection", "confirmation", "blind"),
        default="confirmation",
    )
    parser.add_argument("--train-points", type=int, default=8192)
    parser.add_argument("--selection-points", type=int, default=4096)
    parser.add_argument("--development-points", type=int, default=8192)
    parser.add_argument("--equivalence-points", type=int, default=8192)
    parser.add_argument("--train-seed", type=int, default=72201)
    parser.add_argument("--selection-seed", type=int, default=72202)
    parser.add_argument("--development-seed", type=int, default=72203)
    parser.add_argument("--equivalence-seed", type=int, default=72961)
    parser.add_argument("--torch-seed", type=int, default=20260719)
    parser.add_argument("--transfer-seed", type=int, default=20260746)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument(
        "--positive-floor",
        type=float,
        help="must match the initial model; defaults to its saved positive floor",
    )
    parser.add_argument("--initialization-noise", type=float, default=0.0)
    parser.add_argument("--potential-loss-weight", type=float, default=0.0)
    parser.add_argument("--metric-loss-weight", type=float, default=0.0)
    parser.add_argument("--log-energy-loss-weight", type=float, default=1.0)
    parser.add_argument("--ma-loss-weight", type=float, default=1.0)
    parser.add_argument("--tail-loss-weight", type=float, default=0.1)
    parser.add_argument("--tail-fraction", type=float, default=0.01)
    parser.add_argument("--tail-ratio-threshold", type=float, default=1.5)
    parser.add_argument("--tail-smooth-temperature", type=float, default=0.05)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument(
        "--first-kappa-source",
        choices=("auto", "initial_model", "saved_model", "teacher"),
        default="initial_model",
    )
    parser.add_argument(
        "--continuation-kappa-source",
        choices=("auto", "initial_model", "saved_model", "teacher"),
        default="saved_model",
    )

    parser.add_argument("--site-activation-scale", type=float, default=0.03)
    parser.add_argument("--bond-activation-scale", type=float, default=0.03)
    parser.add_argument("--equivalence-chunk-size", type=int, default=512)
    parser.add_argument("--equivalence-absolute-tolerance", type=float, default=1.0e-6)
    parser.add_argument(
        "--equivalence-relative-metric-tolerance", type=float, default=1.0e-6
    )
    parser.add_argument(
        "--precision",
        choices=("preserve", "complex64", "complex128"),
        default="preserve",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.target_k < 3 or args.target_d < 1:
        parser.error("--target-k must be at least 3 and --target-d must be positive")
    if len({stage.name for stage in args.stage}) != len(args.stage):
        parser.error("stage names must be unique within an arm")
    positive_integers = {
        "sampling-cluster-size": args.sampling_cluster_size,
        "workers": args.workers,
        "teacher-chunk-size": args.teacher_chunk_size,
        "train-points": args.train_points,
        "selection-points": args.selection_points,
        "development-points": args.development_points,
        "equivalence-points": args.equivalence_points,
        "batch-size": args.batch_size,
        "eval-every": args.eval_every,
        "equivalence-chunk-size": args.equivalence_chunk_size,
    }
    invalid = [name for name, value in positive_integers.items() if value <= 0]
    if invalid:
        parser.error("these options must be positive: " + ", ".join(invalid))
    clustered_counts = {
        "train-points": args.train_points,
        "selection-points": args.selection_points,
        "development-points": args.development_points,
        "equivalence-points": args.equivalence_points,
    }
    indivisible = [
        name
        for name, count in clustered_counts.items()
        if count % args.sampling_cluster_size
    ]
    if indivisible:
        parser.error(
            "point counts must be divisible by --sampling-cluster-size: "
            + ", ".join(indivisible)
        )
    nonnegative = {
        "initialization-noise": args.initialization_noise,
        "site-activation-scale": args.site_activation_scale,
        "bond-activation-scale": args.bond_activation_scale,
        "potential-loss-weight": args.potential_loss_weight,
        "metric-loss-weight": args.metric_loss_weight,
        "log-energy-loss-weight": args.log_energy_loss_weight,
        "ma-loss-weight": args.ma_loss_weight,
        "tail-loss-weight": args.tail_loss_weight,
    }
    invalid = [
        name
        for name, value in nonnegative.items()
        if not math.isfinite(value) or value < 0.0
    ]
    if invalid:
        parser.error(
            "these options must be finite and non-negative: " + ", ".join(invalid)
        )
    if args.gradient_clip_norm <= 0.0 or not math.isfinite(args.gradient_clip_norm):
        parser.error("--gradient-clip-norm must be positive and finite")
    if args.positive_floor is not None and (
        args.positive_floor <= 0.0 or not math.isfinite(args.positive_floor)
    ):
        parser.error("--positive-floor must be positive and finite")
    if args.teacher_artifact is None and (
        args.potential_loss_weight or args.metric_loss_weight
    ):
        parser.error(
            "teacher-free stages require zero potential and metric loss weights"
        )
    if not 0.0 < args.tail_fraction <= 1.0:
        parser.error("--tail-fraction must be in (0, 1]")
    if args.tail_ratio_threshold <= 0.0 or args.tail_smooth_temperature <= 0.0:
        parser.error("tail threshold and smooth temperature must be positive")
    if args.equivalence_absolute_tolerance <= 0.0:
        parser.error("equivalence absolute tolerance must be positive")
    if args.equivalence_relative_metric_tolerance <= 0.0:
        parser.error("equivalence relative metric tolerance must be positive")
    return args


def _run_command(command: list[str], label: str) -> None:
    print(f"[study-arm] {label}", flush=True)
    try:
        result = subprocess.run(command, check=False)
    except OSError as error:
        raise ArmError(f"could not start {label}: {error}") from error
    if result.returncode != 0:
        raise ArmError(
            f"{label} failed with exit code {result.returncode}",
            exit_code=child_failure_exit_code(result.returncode),
        )


def _script_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for role, path in SCRIPT_PATHS.items():
        if not path.is_file():
            raise ArmError(f"required {role} script does not exist: {path}")
        hashes[role] = sha256_file(path)
    return hashes


def _valid_site_transfer(
    model: Path,
    summary: Path,
    *,
    source_sha256: str,
    source_k: int,
    target_k: int,
    bond_dimension: int,
    activation_scale: float,
    seed: int,
) -> bool:
    payload = _read_json(summary)
    if payload is None or not model.is_file():
        return False
    try:
        return (
            payload.get("schema") == "positive-tensor-network-site-transfer-v1"
            and payload.get("source_model_sha256") == source_sha256
            and payload.get("model_sha256") == sha256_file(model)
            and int(payload.get("source_site_count")) == source_k
            and int(payload.get("target_site_count")) == target_k
            and int(payload.get("bond_dimension")) == bond_dimension
            and _same_number(
                payload.get("bridge_one_sided_activation_scale"), activation_scale
            )
            and int(payload.get("bridge_noise_seed")) == seed
        )
    except (OSError, TypeError, ValueError):
        return False


def _valid_bond_transfer(
    model: Path,
    summary: Path,
    *,
    source_sha256: str,
    source_d: int,
    target_d: int,
    activation_scale: float,
) -> bool:
    payload = _read_json(summary)
    if payload is None or not model.is_file():
        return False
    try:
        return (
            payload.get("schema") == "positive-tensor-network-bond-expansion-v1"
            and payload.get("source_model_sha256") == source_sha256
            and payload.get("expanded_model_sha256") == sha256_file(model)
            and int(payload.get("source_bond_dimension")) == source_d
            and int(payload.get("target_bond_dimension")) == target_d
            and _same_number(payload.get("initialization_noise"), 0.0)
            and _same_number(
                payload.get("one_sided_activation_scale"), activation_scale
            )
            and payload.get("function_preserving_by_construction") is True
        )
    except (OSError, TypeError, ValueError):
        return False


def _valid_equivalence(
    report_path: Path,
    *,
    model_a_sha256: str,
    model_b_sha256: str,
    source_sha256: str,
    args: argparse.Namespace,
) -> bool:
    payload = _read_json(report_path)
    if payload is None:
        return False
    try:
        return (
            payload.get("schema")
            == "positive-tensor-network-common-point-equivalence-v1"
            and payload.get("model_a_sha256") == model_a_sha256
            and payload.get("model_b_sha256") == model_b_sha256
            and payload.get("source_artifact_sha256") == source_sha256
            and payload.get("adapter") == args.adapter
            and int(payload.get("model_seed")) == args.model_seed
            and int(payload.get("seed")) == args.equivalence_seed
            and int(payload.get("points")) == args.equivalence_points
            and int(payload.get("sampling_cluster_size")) == args.sampling_cluster_size
            and _same_number(
                payload.get("absolute_tolerance"),
                args.equivalence_absolute_tolerance,
            )
            and _same_number(
                payload.get("relative_metric_tolerance"),
                args.equivalence_relative_metric_tolerance,
            )
            and payload.get("success") is True
        )
    except (TypeError, ValueError):
        return False


def _equivalence_command(
    args: argparse.Namespace,
    *,
    model_a: Path,
    model_b: Path,
    out: Path,
    allow_different_site_counts: bool,
) -> list[str]:
    command = [
        args.python_executable,
        str(EQUIVALENCE_AUDIT),
        "--adapter",
        args.adapter,
        "--model-a",
        str(model_a),
        "--model-b",
        str(model_b),
        "--source-artifact",
        str(args.source_artifact),
        "--model-seed",
        str(args.model_seed),
        "--sampling-cluster-size",
        str(args.sampling_cluster_size),
        "--seed",
        str(args.equivalence_seed),
        "--points",
        str(args.equivalence_points),
        "--workers",
        str(args.workers),
        "--chunk-size",
        str(args.equivalence_chunk_size),
        "--absolute-tolerance",
        str(args.equivalence_absolute_tolerance),
        "--relative-metric-tolerance",
        str(args.equivalence_relative_metric_tolerance),
        "--device",
        args.device,
        "--out",
        str(out),
    ]
    if allow_different_site_counts:
        command.append("--allow-different-site-counts")
    return command


def _ensure_equivalence(
    args: argparse.Namespace,
    *,
    model_a: Path,
    model_b: Path,
    out: Path,
    source_sha256: str,
    allow_different_site_counts: bool,
    label: str,
) -> dict[str, Any]:
    model_a_sha256 = sha256_file(model_a)
    model_b_sha256 = sha256_file(model_b)
    command = _equivalence_command(
        args,
        model_a=model_a,
        model_b=model_b,
        out=out,
        allow_different_site_counts=allow_different_site_counts,
    )
    reused = _valid_equivalence(
        out,
        model_a_sha256=model_a_sha256,
        model_b_sha256=model_b_sha256,
        source_sha256=source_sha256,
        args=args,
    )
    if not reused:
        out.parent.mkdir(parents=True, exist_ok=True)
        _run_command(command, label)
        if not _valid_equivalence(
            out,
            model_a_sha256=model_a_sha256,
            model_b_sha256=model_b_sha256,
            source_sha256=source_sha256,
            args=args,
        ):
            raise ArmError(f"{label} did not produce a hash-valid success report")
    else:
        print(f"[study-arm] reuse {label}", flush=True)
    return {
        "status": "reused" if reused else "completed",
        "report": str(out),
        "report_sha256": sha256_file(out),
        "model_a_sha256": model_a_sha256,
        "model_b_sha256": model_b_sha256,
        "command": command,
    }


def _trainer_args(args: argparse.Namespace, stage: StageSpec) -> list[str]:
    trainer_args = [
        "--adapter",
        args.adapter,
        "--source-artifact",
        str(args.source_artifact),
        "--site-count",
        str(args.target_k),
        "--model-seed",
        str(args.model_seed),
        "--bond-dimension",
        str(args.target_d),
        "--positive-floor",
        str(args.positive_floor),
        "--initialization-noise",
        str(args.initialization_noise),
        "--train-points",
        str(args.train_points),
        "--train-seed",
        str(args.train_seed),
        "--train-common-pool",
        str(args.train_common_pool),
        "--validation-points",
        str(args.selection_points),
        "--validation-seed",
        str(args.selection_seed),
        "--validation-common-pool",
        str(args.selection_common_pool),
        "--workers",
        str(args.workers),
        "--sampling-cluster-size",
        str(args.sampling_cluster_size),
        "--teacher-chunk-size",
        str(args.teacher_chunk_size),
        "--epochs",
        str(stage.epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(stage.learning_rate),
        "--gradient-clip-norm",
        str(args.gradient_clip_norm),
        "--potential-loss-weight",
        str(args.potential_loss_weight),
        "--metric-loss-weight",
        str(args.metric_loss_weight),
        "--log-energy-loss-weight",
        str(args.log_energy_loss_weight),
        "--ma-loss-weight",
        str(args.ma_loss_weight),
        "--tail-loss-weight",
        str(args.tail_loss_weight),
        "--tail-fraction",
        str(args.tail_fraction),
        "--tail-ratio-threshold",
        str(args.tail_ratio_threshold),
        "--tail-smooth-temperature",
        str(args.tail_smooth_temperature),
        "--eval-every",
        str(args.eval_every),
        "--early-stopping-patience",
        str(stage.patience),
        "--early-stopping-min-relative-improvement",
        str(stage.min_relative_gain),
        "--torch-seed",
        str(args.torch_seed),
        "--device",
        args.device,
        "--precision",
        args.precision,
    ]
    if args.teacher_artifact is not None:
        trainer_args.extend(("--teacher-artifact", str(args.teacher_artifact)))
    if args.train_physical_dictionary:
        trainer_args.append("--train-physical-dictionary")
    return trainer_args


def _valid_plateau_stage(
    stage_dir: Path,
    *,
    initial_model_sha256: str,
    stage: StageSpec,
    args: argparse.Namespace,
    first_kappa_source: str,
    trainer_args: list[str],
    script_hashes: dict[str, str],
) -> dict[str, Any] | None:
    summary_path = stage_dir / "stage_summary.json"
    model_path = stage_dir / "plateau_model.pt"
    trainer_summary_path = stage_dir / "plateau_summary.json"
    payload = _read_json(summary_path)
    trainer_summary = _read_json(trainer_summary_path)
    if payload is None or trainer_summary is None or not model_path.is_file():
        return None
    try:
        model_sha256 = sha256_file(model_path)
        trainer_summary_sha256 = sha256_file(trainer_summary_path)
        valid = (
            payload.get("schema") == "gcicy-tn-plateau-stage-v1"
            and payload.get("status") == "validation_plateau"
            and payload.get("termination_reason") == "validation_plateau"
            and payload.get("initial_model_sha256") == initial_model_sha256
            and int(payload.get("max_rounds")) == stage.max_rounds
            and payload.get("first_kappa_source") == first_kappa_source
            and payload.get("continuation_kappa_source")
            == args.continuation_kappa_source
            and payload.get("trainer_sha256") == script_hashes["trainer"]
            and payload.get("trainer_args") == trainer_args
            and payload.get("plateau_model_sha256") == model_sha256
            and payload.get("plateau_summary_sha256") == trainer_summary_sha256
            and trainer_summary.get("termination_reason") == "validation_plateau"
            and trainer_summary.get("model_sha256") == model_sha256
        )
    except (OSError, TypeError, ValueError):
        return None
    if not valid:
        return None
    return {
        "stage_summary_payload": payload,
        "model": model_path,
        "model_sha256": model_sha256,
        "trainer_summary": trainer_summary_path,
        "trainer_summary_sha256": trainer_summary_sha256,
        "stage_summary": summary_path,
        "stage_summary_sha256": sha256_file(summary_path),
    }


def _valid_blind_audit(
    report_path: Path,
    arrays_path: Path,
    *,
    model_sha256: str,
    development_pool_sha256: str,
    args: argparse.Namespace,
) -> bool:
    payload = _read_json(report_path)
    if payload is None or not arrays_path.is_file():
        return False
    arrays = payload.get("point_arrays")
    if not isinstance(arrays, dict):
        return False
    try:
        return (
            payload.get("schema") == "type11-positive-tensor-network-blind-audit-v1"
            and payload.get("adapter") == args.adapter
            and payload.get("model_sha256") == model_sha256
            and int(payload.get("model_seed")) == args.model_seed
            and int(payload.get("seed")) == args.development_seed
            and int(payload.get("points")) == args.development_points
            and int(payload.get("sampling_cluster_size")) == args.sampling_cluster_size
            and payload.get("common_pool_sha256") == development_pool_sha256
            and arrays.get("sha256") == sha256_file(arrays_path)
        )
    except (OSError, TypeError, ValueError):
        return False


def _valid_tail_summary(path: Path, *, arrays_sha256: str, label: str) -> bool:
    payload = _read_json(path)
    if (
        payload is None
        or payload.get("schema") != "gcicy-bilateral-tail-array-evaluation-v1"
    ):
        return False
    models = payload.get("models")
    if not isinstance(models, dict) or set(models) != {label}:
        return False
    record = models[label]
    return (
        isinstance(record, dict)
        and record.get("array_artifact_sha256") == arrays_sha256
        and isinstance(record.get("metrics"), dict)
    )


def _prepare_atomic_copy(source: Path, destination: Path) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
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


def _prepare_atomic_json(payload: dict[str, Any], destination: Path) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
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


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _published_is_valid(
    arm_dir: Path,
    *,
    arm_config_sha256: str,
    final_model_sha256: str,
    audit_sha256: str,
    arrays_sha256: str,
    tail_sha256: str,
) -> bool:
    final_model = arm_dir / "final_model.pt"
    arm_summary_path = arm_dir / "arm_summary.json"
    final_summary_path = arm_dir / "final_summary.json"
    final_summary = _read_json(final_summary_path)
    arm_summary = _read_json(arm_summary_path)
    if final_summary is None or arm_summary is None or not final_model.is_file():
        return False
    try:
        return (
            final_summary.get("schema") == "gcicy-tn-study-arm-final-v1"
            and final_summary.get("status") == "complete"
            and final_summary.get("arm_config_sha256") == arm_config_sha256
            and final_summary.get("final_model_sha256") == final_model_sha256
            and sha256_file(final_model) == final_model_sha256
            and final_summary.get("arm_summary_sha256") == sha256_file(arm_summary_path)
            and final_summary.get("audit_sha256") == audit_sha256
            and final_summary.get("arrays_sha256") == arrays_sha256
            and final_summary.get("tail_summary_sha256") == tail_sha256
            and arm_summary.get("schema") == "gcicy-tn-study-arm-v1"
            and arm_summary.get("status") == "complete"
            and arm_summary.get("arm_config_sha256") == arm_config_sha256
            and arm_summary.get("final_model", {}).get("sha256") == final_model_sha256
        )
    except (OSError, AttributeError):
        return False


def _publish(
    arm_dir: Path,
    *,
    source_model: Path,
    arm_summary: dict[str, Any],
    final_summary_base: dict[str, Any],
) -> None:
    final_model = arm_dir / "final_model.pt"
    arm_summary_path = arm_dir / "arm_summary.json"
    final_summary_path = arm_dir / "final_summary.json"
    temporary_paths: list[Path] = []
    try:
        model_temporary = _prepare_atomic_copy(source_model, final_model)
        temporary_paths.append(model_temporary)
        arm_summary_temporary = _prepare_atomic_json(arm_summary, arm_summary_path)
        temporary_paths.append(arm_summary_temporary)
        final_summary = dict(final_summary_base)
        final_summary.update(
            {
                "final_model_sha256": sha256_file(model_temporary),
                "arm_summary_sha256": sha256_file(arm_summary_temporary),
            }
        )
        final_summary_temporary = _prepare_atomic_json(
            final_summary, final_summary_path
        )
        temporary_paths.append(final_summary_temporary)
        os.replace(model_temporary, final_model)
        os.replace(arm_summary_temporary, arm_summary_path)
        # The final summary is the publication commit marker.
        os.replace(final_summary_temporary, final_summary_path)
        _fsync_directory(arm_dir)
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)


def _arm_config(
    args: argparse.Namespace,
    *,
    inputs: dict[str, dict[str, Any]],
    metadata: ModelMetadata,
    script_hashes: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema": "gcicy-tn-study-arm-config-v1",
        "initial_model_metadata": asdict(metadata),
        "inputs": inputs,
        "target": {"k": args.target_k, "D": args.target_d},
        "stages": [asdict(stage) for stage in args.stage],
        "train_physical_dictionary": args.train_physical_dictionary,
        "geometry": {
            "adapter": args.adapter,
            "model_seed": args.model_seed,
            "sampling_cluster_size": args.sampling_cluster_size,
        },
        "pools": {
            "train_points": args.train_points,
            "train_seed": args.train_seed,
            "selection_points": args.selection_points,
            "selection_seed": args.selection_seed,
            "development_points": args.development_points,
            "development_seed": args.development_seed,
            "development_pool_split": args.development_pool_split,
            "equivalence_points": args.equivalence_points,
            "equivalence_seed": args.equivalence_seed,
        },
        "optimization": {
            "batch_size": args.batch_size,
            "gradient_clip_norm": args.gradient_clip_norm,
            "positive_floor": args.positive_floor,
            "initialization_noise": args.initialization_noise,
            "potential_loss_weight": args.potential_loss_weight,
            "metric_loss_weight": args.metric_loss_weight,
            "log_energy_loss_weight": args.log_energy_loss_weight,
            "ma_loss_weight": args.ma_loss_weight,
            "tail_loss_weight": args.tail_loss_weight,
            "tail_fraction": args.tail_fraction,
            "tail_ratio_threshold": args.tail_ratio_threshold,
            "tail_smooth_temperature": args.tail_smooth_temperature,
            "eval_every": args.eval_every,
            "torch_seed": args.torch_seed,
            "precision": args.precision,
            "device": args.device,
        },
        "transfer": {
            "site_activation_scale": args.site_activation_scale,
            "bond_activation_scale": args.bond_activation_scale,
            "transfer_seed": args.transfer_seed,
            "equivalence_absolute_tolerance": args.equivalence_absolute_tolerance,
            "equivalence_relative_metric_tolerance": (
                args.equivalence_relative_metric_tolerance
            ),
            "equivalence_chunk_size": args.equivalence_chunk_size,
        },
        "runtime": {
            "workers": args.workers,
            "teacher_chunk_size": args.teacher_chunk_size,
            "python": args.python_executable,
            "first_kappa_source": args.first_kappa_source,
            "continuation_kappa_source": args.continuation_kappa_source,
        },
        "script_sha256": script_hashes,
    }


def run_arm(args: argparse.Namespace) -> int:
    initial_model = _require_file(args.initial_model, "initial model")
    source_artifact = _require_file(args.source_artifact, "source artifact")
    train_pool = _require_file(args.train_common_pool, "train common pool")
    selection_pool = _require_file(args.selection_common_pool, "selection common pool")
    development_pool = _require_file(
        args.development_common_pool, "development common pool"
    )
    teacher_artifact = (
        None
        if args.teacher_artifact is None
        else _require_file(args.teacher_artifact, "teacher artifact")
    )
    args.initial_model = initial_model
    args.source_artifact = source_artifact
    args.train_common_pool = train_pool
    args.selection_common_pool = selection_pool
    args.development_common_pool = development_pool
    args.teacher_artifact = teacher_artifact

    arm_dir = args.arm_dir.expanduser().resolve()
    managed_publications = {
        arm_dir / "final_model.pt",
        arm_dir / "arm_summary.json",
        arm_dir / "final_summary.json",
    }
    if initial_model in managed_publications:
        raise ArmError("initial model conflicts with a managed arm publication")
    arm_dir.mkdir(parents=True, exist_ok=True)
    work_dir = arm_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    metadata = read_model_metadata(initial_model)
    source_sha256 = sha256_file(source_artifact)
    if metadata.source_artifact_sha256 != source_sha256:
        raise ArmError(
            "initial model source artifact hash does not match --source-artifact"
        )
    if metadata.precision not in {"complex64", "complex128"}:
        raise ArmError(f"initial model has unsupported precision: {metadata.precision}")
    if args.precision == "preserve":
        args.precision = metadata.precision
    elif args.precision != metadata.precision:
        raise ArmError(
            "requested precision differs from the saved initial model; explicit "
            "continuation requires matching precision"
        )
    if args.positive_floor is None:
        args.positive_floor = metadata.positive_floor
    elif not math.isclose(
        args.positive_floor,
        metadata.positive_floor,
        rel_tol=1.0e-12,
        abs_tol=0.0,
    ):
        raise ArmError(
            "requested positive floor differs from the saved initial model and "
            "would confound the targeted arm"
        )
    if (
        args.train_physical_dictionary
        and metadata.architecture != "shared_local_dictionary"
    ):
        raise ArmError(
            "--train-physical-dictionary requires a shared-local-dictionary model"
        )
    if not args.train_physical_dictionary and metadata.trainable_physical_dictionary:
        raise ArmError(
            "initial model already marks its physical dictionary trainable; pass "
            "--train-physical-dictionary so the requested parameter scope is explicit"
        )
    if args.target_k < metadata.site_count:
        raise ArmError(
            f"target k={args.target_k} is smaller than initial k={metadata.site_count}"
        )
    if args.target_k > metadata.site_count:
        if metadata.architecture != "shared_local_dictionary":
            raise ArmError(
                "function-preserving site repeat requires a shared-local-dictionary model"
            )
        repeat_count, remainder = divmod(args.target_k, metadata.site_count)
        if remainder or repeat_count < 2:
            raise ArmError(
                "function-preserving site repeat requires target k to be an "
                "integer multiple of the initial k"
            )
    if args.target_d < metadata.bond_dimension:
        raise ArmError(
            f"target D={args.target_d} is smaller than initial D={metadata.bond_dimension}"
        )

    inputs = {
        "initial_model": _artifact(initial_model),
        "source_artifact": _artifact(source_artifact),
        "train_common_pool": _artifact(train_pool),
        "selection_common_pool": _artifact(selection_pool),
        "development_common_pool": _artifact(development_pool),
    }
    if teacher_artifact is not None:
        inputs["teacher_artifact"] = _artifact(teacher_artifact)
    script_hashes = _script_hashes()
    arm_config = _arm_config(
        args,
        inputs=inputs,
        metadata=metadata,
        script_hashes=script_hashes,
    )
    arm_config_sha256 = canonical_sha256(arm_config)

    current_model = initial_model
    current_sha256 = inputs["initial_model"]["sha256"]
    current_k = metadata.site_count
    current_d = metadata.bond_dimension
    transfer_records: list[dict[str, Any]] = []

    if args.target_k == current_k:
        transfer_records.append(
            {
                "kind": "site_repeat",
                "status": "skipped_already_at_target",
                "source_k": current_k,
                "target_k": args.target_k,
                "model_sha256": current_sha256,
            }
        )
    else:
        config = {
            "kind": "site_repeat",
            "source_model_sha256": current_sha256,
            "source_k": current_k,
            "target_k": args.target_k,
            "bond_dimension": current_d,
            "activation_scale": args.site_activation_scale,
            "seed": args.transfer_seed,
            "site_transfer_script_sha256": script_hashes["site_transfer"],
        }
        digest = canonical_sha256(config)
        step_dir = work_dir / "transfers" / f"site_{digest[:16]}"
        model_out = step_dir / "model.pt"
        summary_out = step_dir / "transfer_summary.json"
        command = [
            args.python_executable,
            str(SITE_TRANSFER),
            "--model",
            str(current_model),
            "--target-site-count",
            str(args.target_k),
            "--transfer-rule",
            "repeat",
            "--bridge-one-sided-activation-scale",
            str(args.site_activation_scale),
            "--bridge-noise-seed",
            str(args.transfer_seed),
            "--out",
            str(model_out),
            "--summary",
            str(summary_out),
        ]
        reused = _valid_site_transfer(
            model_out,
            summary_out,
            source_sha256=current_sha256,
            source_k=current_k,
            target_k=args.target_k,
            bond_dimension=current_d,
            activation_scale=args.site_activation_scale,
            seed=args.transfer_seed,
        )
        if not reused:
            step_dir.mkdir(parents=True, exist_ok=True)
            _run_command(command, f"site repeat k={current_k}->{args.target_k}")
            if not _valid_site_transfer(
                model_out,
                summary_out,
                source_sha256=current_sha256,
                source_k=current_k,
                target_k=args.target_k,
                bond_dimension=current_d,
                activation_scale=args.site_activation_scale,
                seed=args.transfer_seed,
            ):
                raise ArmError("site repeat did not produce hash-valid artifacts")
        else:
            print("[study-arm] reuse site repeat", flush=True)
        previous_model = current_model
        previous_sha256 = current_sha256
        current_model = model_out
        current_sha256 = sha256_file(model_out)
        equivalence_config = {
            "model_a_sha256": previous_sha256,
            "model_b_sha256": current_sha256,
            "source_artifact_sha256": source_sha256,
            "adapter": args.adapter,
            "model_seed": args.model_seed,
            "sampling_cluster_size": args.sampling_cluster_size,
            "seed": args.equivalence_seed,
            "points": args.equivalence_points,
            "chunk_size": args.equivalence_chunk_size,
            "absolute_tolerance": args.equivalence_absolute_tolerance,
            "relative_metric_tolerance": (args.equivalence_relative_metric_tolerance),
            "device": args.device,
            "script_sha256": script_hashes["equivalence_audit"],
        }
        equivalence_digest = canonical_sha256(equivalence_config)
        audit_path = step_dir / f"equivalence_{equivalence_digest[:16]}.json"
        equivalence = _ensure_equivalence(
            args,
            model_a=previous_model,
            model_b=current_model,
            out=audit_path,
            source_sha256=source_sha256,
            allow_different_site_counts=True,
            label="site-transfer equivalence audit",
        )
        transfer_records.append(
            {
                "kind": "site_repeat",
                "status": "reused" if reused else "completed",
                "config_sha256": digest,
                "source_model_sha256": previous_sha256,
                "model": str(model_out),
                "model_sha256": current_sha256,
                "summary": str(summary_out),
                "summary_sha256": sha256_file(summary_out),
                "command": command,
                "equivalence": equivalence,
            }
        )
        current_k = args.target_k

    if args.target_d == current_d:
        transfer_records.append(
            {
                "kind": "bond_expansion",
                "status": "skipped_already_at_target",
                "source_D": current_d,
                "target_D": args.target_d,
                "model_sha256": current_sha256,
            }
        )
    else:
        config = {
            "kind": "bond_expansion",
            "source_model_sha256": current_sha256,
            "source_D": current_d,
            "target_D": args.target_d,
            "activation_scale": args.bond_activation_scale,
            "seed": args.transfer_seed,
            "bond_transfer_script_sha256": script_hashes["bond_transfer"],
        }
        digest = canonical_sha256(config)
        step_dir = work_dir / "transfers" / f"bond_{digest[:16]}"
        model_out = step_dir / "model.pt"
        summary_out = step_dir / "transfer_summary.json"
        command = [
            args.python_executable,
            str(BOND_TRANSFER),
            "--model",
            str(current_model),
            "--bond-dimension",
            str(args.target_d),
            "--initialization-noise",
            "0",
            "--one-sided-activation-scale",
            str(args.bond_activation_scale),
            "--seed",
            str(args.transfer_seed),
            "--precision",
            "preserve",
            "--out",
            str(model_out),
            "--summary",
            str(summary_out),
        ]
        reused = _valid_bond_transfer(
            model_out,
            summary_out,
            source_sha256=current_sha256,
            source_d=current_d,
            target_d=args.target_d,
            activation_scale=args.bond_activation_scale,
        )
        if not reused:
            step_dir.mkdir(parents=True, exist_ok=True)
            _run_command(command, f"bond expansion D={current_d}->{args.target_d}")
            if not _valid_bond_transfer(
                model_out,
                summary_out,
                source_sha256=current_sha256,
                source_d=current_d,
                target_d=args.target_d,
                activation_scale=args.bond_activation_scale,
            ):
                raise ArmError("bond expansion did not produce hash-valid artifacts")
        else:
            print("[study-arm] reuse bond expansion", flush=True)
        previous_model = current_model
        previous_sha256 = current_sha256
        current_model = model_out
        current_sha256 = sha256_file(model_out)
        equivalence_config = {
            "model_a_sha256": previous_sha256,
            "model_b_sha256": current_sha256,
            "source_artifact_sha256": source_sha256,
            "adapter": args.adapter,
            "model_seed": args.model_seed,
            "sampling_cluster_size": args.sampling_cluster_size,
            "seed": args.equivalence_seed,
            "points": args.equivalence_points,
            "chunk_size": args.equivalence_chunk_size,
            "absolute_tolerance": args.equivalence_absolute_tolerance,
            "relative_metric_tolerance": (args.equivalence_relative_metric_tolerance),
            "device": args.device,
            "script_sha256": script_hashes["equivalence_audit"],
        }
        equivalence_digest = canonical_sha256(equivalence_config)
        audit_path = step_dir / f"equivalence_{equivalence_digest[:16]}.json"
        equivalence = _ensure_equivalence(
            args,
            model_a=previous_model,
            model_b=current_model,
            out=audit_path,
            source_sha256=source_sha256,
            allow_different_site_counts=False,
            label="bond-transfer equivalence audit",
        )
        transfer_records.append(
            {
                "kind": "bond_expansion",
                "status": "reused" if reused else "completed",
                "config_sha256": digest,
                "source_model_sha256": previous_sha256,
                "model": str(model_out),
                "model_sha256": current_sha256,
                "summary": str(summary_out),
                "summary_sha256": sha256_file(summary_out),
                "command": command,
                "equivalence": equivalence,
            }
        )
        current_d = args.target_d

    stage_records: list[dict[str, Any]] = []
    for index, stage in enumerate(args.stage, start=1):
        stage_first_kappa_source = (
            args.first_kappa_source if index == 1 else args.continuation_kappa_source
        )
        trainer_args = _trainer_args(args, stage)
        stage_config = {
            "kind": "plateau_stage",
            "index": index,
            "spec": asdict(stage),
            "initial_model_sha256": current_sha256,
            "trainer_args": trainer_args,
            "first_kappa_source": stage_first_kappa_source,
            "continuation_kappa_source": args.continuation_kappa_source,
            "plateau_script_sha256": script_hashes["plateau_stage"],
            "trainer_sha256": script_hashes["trainer"],
        }
        digest = canonical_sha256(stage_config)
        stage_dir = work_dir / "stages" / f"{index:02d}_{stage.name}_{digest[:16]}"
        command = [
            args.python_executable,
            str(PLATEAU_STAGE),
            "--initial-model",
            str(current_model),
            "--stage-dir",
            str(stage_dir),
            "--max-rounds",
            str(stage.max_rounds),
            "--first-kappa-source",
            stage_first_kappa_source,
            "--continuation-kappa-source",
            args.continuation_kappa_source,
            "--",
            *trainer_args,
        ]
        inspected = _valid_plateau_stage(
            stage_dir,
            initial_model_sha256=current_sha256,
            stage=stage,
            args=args,
            first_kappa_source=stage_first_kappa_source,
            trainer_args=trainer_args,
            script_hashes=script_hashes,
        )
        reused = inspected is not None
        if inspected is None:
            stage_dir.mkdir(parents=True, exist_ok=True)
            _run_command(command, f"training stage {index}: {stage.name}")
            inspected = _valid_plateau_stage(
                stage_dir,
                initial_model_sha256=current_sha256,
                stage=stage,
                args=args,
                first_kappa_source=stage_first_kappa_source,
                trainer_args=trainer_args,
                script_hashes=script_hashes,
            )
            if inspected is None:
                raise ArmError(
                    f"training stage {stage.name} did not publish a hash-valid plateau"
                )
        else:
            print(f"[study-arm] reuse training stage {stage.name}", flush=True)
        input_sha256 = current_sha256
        current_model = inspected["model"]
        current_sha256 = inspected["model_sha256"]
        stage_records.append(
            {
                "index": index,
                "name": stage.name,
                "status": "reused" if reused else "completed",
                "config_sha256": digest,
                "spec": asdict(stage),
                "input_model_sha256": input_sha256,
                "plateau_model": str(current_model),
                "plateau_model_sha256": current_sha256,
                "plateau_summary": str(inspected["trainer_summary"]),
                "plateau_summary_sha256": inspected["trainer_summary_sha256"],
                "stage_summary": str(inspected["stage_summary"]),
                "stage_summary_sha256": inspected["stage_summary_sha256"],
                "command": command,
            }
        )

    audit_config = {
        "kind": "final_development_audit",
        "model_sha256": current_sha256,
        "source_artifact_sha256": source_sha256,
        "development_pool_sha256": inputs["development_common_pool"]["sha256"],
        "adapter": args.adapter,
        "model_seed": args.model_seed,
        "seed": args.development_seed,
        "points": args.development_points,
        "sampling_cluster_size": args.sampling_cluster_size,
        "pool_split": args.development_pool_split,
        "device": args.device,
        "workers": args.workers,
        "teacher_artifact_sha256": (
            None if teacher_artifact is None else inputs["teacher_artifact"]["sha256"]
        ),
        "blind_audit_script_sha256": script_hashes["blind_audit"],
        "tail_evaluator_script_sha256": script_hashes["tail_evaluator"],
    }
    audit_digest = canonical_sha256(audit_config)
    audit_dir = work_dir / "final_audit" / audit_digest[:16]
    audit_report = audit_dir / "audit.json"
    arrays_path = audit_dir / "arrays.npz"
    tail_summary_path = audit_dir / "tail_summary.json"
    audit_command = [
        args.python_executable,
        str(BLIND_AUDIT),
        "--model",
        str(current_model),
        "--adapter",
        args.adapter,
        "--source-artifact",
        str(source_artifact),
        "--model-seed",
        str(args.model_seed),
        "--seed",
        str(args.development_seed),
        "--points",
        str(args.development_points),
        "--common-pool",
        str(development_pool),
        "--common-pool-split",
        args.development_pool_split,
        "--workers",
        str(args.workers),
        "--sampling-cluster-size",
        str(args.sampling_cluster_size),
        "--device",
        args.device,
        "--arrays-out",
        str(arrays_path),
        "--out",
        str(audit_report),
    ]
    if teacher_artifact is not None:
        audit_command.extend(("--teacher-artifact", str(teacher_artifact)))
    audit_reused = _valid_blind_audit(
        audit_report,
        arrays_path,
        model_sha256=current_sha256,
        development_pool_sha256=inputs["development_common_pool"]["sha256"],
        args=args,
    )
    if not audit_reused:
        audit_dir.mkdir(parents=True, exist_ok=True)
        _run_command(audit_command, "final immutable-pool audit")
        if not _valid_blind_audit(
            audit_report,
            arrays_path,
            model_sha256=current_sha256,
            development_pool_sha256=inputs["development_common_pool"]["sha256"],
            args=args,
        ):
            raise ArmError("final audit did not produce hash-valid report and arrays")
    else:
        print("[study-arm] reuse final immutable-pool audit", flush=True)

    arrays_sha256 = sha256_file(arrays_path)
    tail_label = "final_model"
    tail_command = [
        args.python_executable,
        str(TAIL_EVALUATOR),
        "--arrays",
        str(arrays_path),
        "--label",
        tail_label,
        "--cluster-size",
        str(args.sampling_cluster_size),
        "--out",
        str(tail_summary_path),
    ]
    tail_reused = _valid_tail_summary(
        tail_summary_path, arrays_sha256=arrays_sha256, label=tail_label
    )
    if not tail_reused:
        _run_command(tail_command, "bilateral tail evaluation")
        if not _valid_tail_summary(
            tail_summary_path, arrays_sha256=arrays_sha256, label=tail_label
        ):
            raise ArmError("tail evaluation did not produce a hash-valid summary")
    else:
        print("[study-arm] reuse bilateral tail evaluation", flush=True)

    # Detect accidental mutation of every declared immutable input before publish.
    for role, record in inputs.items():
        path = Path(record["path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise ArmError(f"immutable input changed while arm was running: {role}")

    audit_payload = _read_json(audit_report)
    tail_payload = _read_json(tail_summary_path)
    if audit_payload is None or tail_payload is None:
        raise ArmError("validated final reports became unreadable before publish")
    audit_sha256 = sha256_file(audit_report)
    tail_sha256 = sha256_file(tail_summary_path)
    final_model_record = {
        "source_path": str(current_model),
        "published_path": str(arm_dir / "final_model.pt"),
        "sha256": current_sha256,
    }
    final_audit_record = {
        "status": "reused" if audit_reused else "completed",
        "config_sha256": audit_digest,
        "report": str(audit_report),
        "report_sha256": audit_sha256,
        "arrays": str(arrays_path),
        "arrays_sha256": arrays_sha256,
        "tail_summary": str(tail_summary_path),
        "tail_summary_sha256": tail_sha256,
        "tail_status": "reused" if tail_reused else "completed",
        "audit_command": audit_command,
        "tail_command": tail_command,
    }
    arm_summary = {
        "schema": "gcicy-tn-study-arm-v1",
        "status": "complete",
        "arm_config_sha256": arm_config_sha256,
        "config": arm_config,
        "immutable_inputs": inputs,
        "transfers": transfer_records,
        "training_stages": stage_records,
        "final_model": final_model_record,
        "final_audit": final_audit_record,
    }
    final_summary_base = {
        "schema": "gcicy-tn-study-arm-final-v1",
        "status": "complete",
        "arm_config_sha256": arm_config_sha256,
        "target_k": args.target_k,
        "target_D": args.target_d,
        "train_physical_dictionary": args.train_physical_dictionary,
        "stage_names": [stage.name for stage in args.stage],
        "audit_sha256": audit_sha256,
        "arrays_sha256": arrays_sha256,
        "tail_summary_sha256": tail_sha256,
        "development_common_pool_sha256": inputs["development_common_pool"]["sha256"],
        "audit_metrics": audit_payload.get("metrics"),
        "tail_metrics": tail_payload["models"][tail_label]["metrics"],
    }
    if _published_is_valid(
        arm_dir,
        arm_config_sha256=arm_config_sha256,
        final_model_sha256=current_sha256,
        audit_sha256=audit_sha256,
        arrays_sha256=arrays_sha256,
        tail_sha256=tail_sha256,
    ):
        print("[study-arm] reuse published arm", flush=True)
        return 0
    _publish(
        arm_dir,
        source_model=current_model,
        arm_summary=arm_summary,
        final_summary_base=final_summary_base,
    )
    if not _published_is_valid(
        arm_dir,
        arm_config_sha256=arm_config_sha256,
        final_model_sha256=current_sha256,
        audit_sha256=audit_sha256,
        arrays_sha256=arrays_sha256,
        tail_sha256=tail_sha256,
    ):
        raise ArmError("atomic arm publication failed hash validation")
    print(f"[study-arm] published {arm_dir}", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run_arm(args)
    except (ArmError, OSError, ValueError) as error:
        print(f"[study-arm] error: {error}", file=sys.stderr, flush=True)
        return error.exit_code if isinstance(error, ArmError) else 1


if __name__ == "__main__":
    raise SystemExit(main())
