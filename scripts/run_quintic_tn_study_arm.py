#!/usr/bin/env python3
"""Run one restartable, development-only quintic positive-TN study arm."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
RESIZE = ROOT / "scripts" / "resize_positive_tensor_network_sites.py"
EXPAND = ROOT / "scripts" / "expand_positive_tensor_network_bond.py"
AUDIT = ROOT / "scripts" / "audit_quintic_positive_tensor_network_equivalence.py"
PLATEAU = ROOT / "scripts" / "run_quintic_tn_plateau_stage.py"
TRAINER = ROOT / "scripts" / "train_quintic_positive_tensor_network_same_points.py"

RESULT_SCHEMA = "quintic-tn-study-arm-result-v1"
SUMMARY_SCHEMA = "quintic-tn-study-arm-summary-v1"
MODEL_SCHEMA = "quintic-positive-tensor-network-v1"
REPORT_SCHEMA = "quintic-positive-tensor-network-same-points-v1"
SCOPE_ORDER = {"new_channels": 0, "cores": 1, "joint": 2}


class ArmError(RuntimeError):
    def __init__(self, message: str, *, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


@dataclass(frozen=True)
class TransferStep:
    kind: str
    target: int


@dataclass(frozen=True)
class StageSpec:
    name: str
    scope: str
    learning_rate: float
    min_relative_gain: float
    patience_rounds: int
    epochs: int
    max_rounds: int


@dataclass(frozen=True)
class ModelMetadata:
    source_degree: int
    site_count: int
    bond_dimension: int
    dictionary_rank: int
    output_dimension: int
    precision: str
    positive_floor: float
    transfer_implementation: str
    fermat_phase_charge_multiplicity: int
    fermat_s5_orbit_tying: bool
    fermat_two_site_blocking: bool
    inherited_bond_dimension: int | None


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


def child_failure_exit_code(returncode: int) -> int:
    if returncode < 0:
        return min(255, 128 + (-returncode))
    return returncode if 1 <= returncode <= 255 else 1


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
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


def parse_transfer_step(text: str) -> TransferStep:
    try:
        kind, target_text = text.split(":", 1)
        target = int(target_text)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError(
            "transfer step must be sites:N or bond:N"
        ) from error
    if kind not in {"sites", "bond"} or target <= 0:
        raise argparse.ArgumentTypeError("transfer step must be sites:N or bond:N")
    return TransferStep(kind, target)


def parse_stage(text: str) -> StageSpec:
    fields = text.split(",")
    if len(fields) != 7:
        raise argparse.ArgumentTypeError(
            "stage must be NAME,SCOPE,LR,MIN_GAIN,PATIENCE,EPOCHS,MAX_ROUNDS"
        )
    name, scope, lr_text, gain_text, patience_text, epochs_text, rounds_text = fields
    if not name or any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for character in name
    ):
        raise argparse.ArgumentTypeError(
            "stage NAME may contain only letters, digits, _ and -"
        )
    if scope not in SCOPE_ORDER:
        raise argparse.ArgumentTypeError(
            "stage SCOPE must be new_channels, cores or joint"
        )
    try:
        lr = float(lr_text)
        gain = float(gain_text)
        patience = int(patience_text)
        epochs = int(epochs_text)
        rounds = int(rounds_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("stage numeric fields are invalid") from error
    if not math.isfinite(lr) or lr <= 0:
        raise argparse.ArgumentTypeError("stage LR must be finite and positive")
    if not math.isfinite(gain) or gain < 0:
        raise argparse.ArgumentTypeError(
            "stage MIN_GAIN must be finite and non-negative"
        )
    if min(patience, epochs, rounds) <= 0:
        raise argparse.ArgumentTypeError(
            "stage PATIENCE, EPOCHS and MAX_ROUNDS must be positive"
        )
    return StageSpec(name, scope, lr, gain, patience, epochs, rounds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--initial-model", type=Path, required=True)
    parser.add_argument("--arm-dir", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--pullbacks-dir", type=Path, required=True)
    parser.add_argument("--target-k", type=int, required=True)
    parser.add_argument("--target-d", type=int, required=True)
    parser.add_argument("--source-degree", type=int, choices=(1, 2), required=True)
    parser.add_argument(
        "--skip-blind-audit",
        action="store_true",
        required=True,
        help="required protocol marker; every fit remains development-only",
    )
    parser.add_argument(
        "--transfer-step",
        type=parse_transfer_step,
        action="append",
        default=[],
        help="ordered and repeatable: sites:N or bond:N",
    )
    parser.add_argument("--stage", type=parse_stage, action="append", required=True)
    parser.add_argument(
        "--site-transfer-rule", choices=("repeat", "interpolate"), default="repeat"
    )
    parser.add_argument(
        "--site-bridge-one-sided-activation-scale", type=float, default=1e-3
    )
    parser.add_argument("--transfer-seed", type=int, default=20260746)
    parser.add_argument("--site-bridge-seed", type=int)
    parser.add_argument("--initialization-noise", type=float, required=True)
    parser.add_argument("--bond-initialization-noise", type=float)
    parser.add_argument("--bond-one-sided-activation-scale", type=float, default=1e-3)
    parser.add_argument("--bond-seed", type=int)
    parser.add_argument(
        "--precision", choices=("complex64", "complex128"), default="complex64"
    )
    parser.add_argument(
        "--audit-points",
        "--equivalence-points",
        dest="audit_points",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--audit-chunk-size",
        "--equivalence-chunk-size",
        dest="audit_chunk_size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--audit-absolute-potential-tolerance",
        "--equivalence-absolute-tolerance",
        dest="audit_absolute_potential_tolerance",
        type=float,
        default=2e-5,
    )
    parser.add_argument("--audit-absolute-metric-tolerance", type=float)
    parser.add_argument(
        "--audit-relative-metric-tolerance",
        "--equivalence-relative-metric-tolerance",
        dest="audit_relative_metric_tolerance",
        type=float,
        default=2e-4,
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--validation-limit", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--positive-floor", type=float, required=True)
    parser.add_argument("--fixed-log-kappa", type=float, required=True)
    parser.add_argument("--log-energy-loss-weight", type=float, default=1.0)
    parser.add_argument("--ma-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--ma-loss-kind", choices=("squared", "absolute"), default="squared"
    )
    parser.add_argument("--tail-loss-weight", type=float, default=0.1)
    parser.add_argument("--tail-fraction", type=float, default=0.02)
    parser.add_argument(
        "--tail-loss-kind",
        choices=("upper_threshold", "absolute_ratio"),
        default="upper_threshold",
    )
    parser.add_argument("--tail-ratio-threshold", type=float, default=1.5)
    parser.add_argument("--tail-smooth-temperature", type=float, default=0.05)
    parser.add_argument(
        "--checkpoint-score-kind", choices=("energy", "sigma"), default="energy"
    )
    parser.add_argument(
        "--training-logdet-method", choices=("cholesky", "eigvalsh"), default="eigvalsh"
    )
    parser.add_argument("--orthonormalize-every", type=int, default=1)
    parser.add_argument("--full-epoch-gradient", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--torch-seed", type=int, default=202607192)
    parser.add_argument("--dictionary-seed", type=int, default=202607191)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.site_bridge_seed is None:
        args.site_bridge_seed = args.transfer_seed
    if args.bond_seed is None:
        args.bond_seed = args.transfer_seed
    if args.bond_initialization_noise is None:
        args.bond_initialization_noise = args.initialization_noise
    if args.audit_absolute_metric_tolerance is None:
        args.audit_absolute_metric_tolerance = args.audit_absolute_potential_tolerance
    for name in (
        "target_k",
        "target_d",
        "audit_points",
        "audit_chunk_size",
        "batch_size",
        "eval_batch_size",
        "eval_every",
        "orthonormalize_every",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("train_limit", "validation_limit"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    nonnegative = (
        "site_bridge_one_sided_activation_scale",
        "initialization_noise",
        "bond_initialization_noise",
        "positive_floor",
        "bond_one_sided_activation_scale",
        "audit_absolute_potential_tolerance",
        "audit_absolute_metric_tolerance",
        "audit_relative_metric_tolerance",
    )
    for name in nonnegative:
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    if args.bond_initialization_noise and args.bond_one_sided_activation_scale:
        parser.error("bond noise and one-sided activation are mutually exclusive")
    if args.positive_floor <= 0:
        parser.error("--positive-floor must be positive")
    if (
        args.site_transfer_rule != "repeat"
        and args.site_bridge_one_sided_activation_scale
    ):
        parser.error("site bridge activation requires --site-transfer-rule repeat")
    if not math.isfinite(args.fixed_log_kappa):
        parser.error("--fixed-log-kappa must be finite")
    names = [stage.name for stage in args.stage]
    if len(names) != len(set(names)):
        parser.error("stage names must be unique")
    orders = [SCOPE_ORDER[stage.scope] for stage in args.stage]
    if orders != sorted(orders):
        parser.error("stage scopes must progress new_channels -> cores -> joint")
    return args


def load_model_metadata(path: Path) -> ModelMetadata:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != MODEL_SCHEMA:
        raise ArmError(f"not a quintic positive-TN artifact: {path}")
    if payload.get("architecture") != "shared_local_dictionary":
        raise ArmError("study arms require shared-local-dictionary models")
    state = payload.get("state_dict")
    if not isinstance(state, dict) or "physical_dictionary" not in state:
        raise ArmError("model has no saved shared physical dictionary")
    dictionary_rank = int(
        payload.get("physical_dictionary_rank", state["physical_dictionary"].shape[0])
    )
    source_degree = int(payload["source_degree"])
    source_section_count = 5 if source_degree == 1 else 15
    dictionary_shape = tuple(state["physical_dictionary"].shape)
    inferred_output = int(dictionary_shape[1] // source_section_count)
    expansion = payload.get("bond_expansion")
    inherited = None
    if isinstance(expansion, dict):
        value = expansion.get("source_bond_dimension")
        if isinstance(value, int) and value > 0:
            inherited = value
    return ModelMetadata(
        source_degree=source_degree,
        site_count=int(payload["site_count"]),
        bond_dimension=int(payload["bond_dimension"]),
        dictionary_rank=dictionary_rank,
        output_dimension=int(payload.get("output_dimension", inferred_output)),
        precision=str(payload["precision"]),
        positive_floor=float(payload["positive_floor"]),
        transfer_implementation=str(
            payload.get("transfer_implementation", "vectorized")
        ),
        fermat_phase_charge_multiplicity=int(
            payload.get("fermat_phase_charge_multiplicity", 0)
        ),
        fermat_s5_orbit_tying=bool(payload.get("fermat_s5_orbit_tying", False)),
        fermat_two_site_blocking=bool(payload.get("fermat_two_site_blocking", False)),
        inherited_bond_dimension=inherited,
    )


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def frozen_input_paths(source_dir: Path, pullbacks_dir: Path) -> tuple[Path, ...]:
    return (
        source_dir / "training_data" / "dataset.npz",
        source_dir / "training_data" / "basis.pickle",
        pullbacks_dir / "report.json",
        pullbacks_dir / "train_pullbacks.npy",
        pullbacks_dir / "validation_pullbacks.npy",
        pullbacks_dir / "validation_official_fs_metrics.npy",
    )


def capture_hashes(paths: Iterable[Path]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            raise ArmError(f"required development input does not exist: {path}")
        result[str(path)] = sha256_file(path)
    return result


def verify_hashes(expected: dict[str, str]) -> None:
    for text, digest in expected.items():
        path = Path(text)
        if not path.is_file() or sha256_file(path) != digest:
            raise ArmError(f"immutable development input changed during arm: {path}")


def resolve_transfer_steps(
    requested: Sequence[TransferStep],
    metadata: ModelMetadata,
    target_k: int,
    target_d: int,
) -> list[TransferStep]:
    if target_k % metadata.source_degree:
        raise ArmError("--target-k must be divisible by the model source degree")
    target_sites = target_k // metadata.source_degree
    steps = list(requested)
    if not steps:
        if metadata.site_count != target_sites:
            steps.append(TransferStep("sites", target_sites))
        if metadata.bond_dimension != target_d:
            steps.append(TransferStep("bond", target_d))
    sites = metadata.site_count
    bond = metadata.bond_dimension
    for step in steps:
        if step.kind == "sites":
            if step.target <= sites:
                raise ArmError("site transfer steps must strictly grow the site count")
            sites = step.target
        else:
            if step.target <= bond:
                raise ArmError("bond transfer steps must strictly grow bond dimension")
            bond = step.target
    if sites != target_sites or bond != target_d:
        raise ArmError(
            f"transfer chain ends at sites={sites},D={bond}; expected "
            f"sites={target_sites},D={target_d}"
        )
    return steps


def _json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _run_child(command: Sequence[str], *, label: str) -> None:
    try:
        result = subprocess.run(list(command), check=False)
    except OSError as error:
        raise ArmError(f"could not start {label}: {error}") from error
    if result.returncode != 0:
        raise ArmError(
            f"{label} failed with exit code {result.returncode}",
            exit_code=child_failure_exit_code(result.returncode),
        )


def _valid_transfer(
    step: TransferStep, source_sha: str, model_path: Path, summary_path: Path
) -> bool:
    if not model_path.is_file() or not summary_path.is_file():
        return False
    payload = _json(summary_path)
    if payload is None or payload.get("source_model_sha256") != source_sha:
        return False
    model_sha = sha256_file(model_path)
    if step.kind == "sites":
        return bool(
            payload.get("schema") == "positive-tensor-network-site-transfer-v1"
            and payload.get("target_site_count") == step.target
            and payload.get("model_sha256") == model_sha
        )
    return bool(
        payload.get("schema") == "positive-tensor-network-bond-expansion-v1"
        and payload.get("target_bond_dimension") == step.target
        and payload.get("expanded_model_sha256") == model_sha
    )


def _valid_audit(path: Path, source_sha: str, target_sha: str) -> bool:
    payload = _json(path)
    return bool(
        payload is not None
        and payload.get("schema")
        == "quintic-positive-tensor-network-common-point-equivalence-v1"
        and payload.get("success") is True
        and payload.get("model_a_sha256") == source_sha
        and payload.get("model_b_sha256") == target_sha
    )


def _trainer_args(
    args: argparse.Namespace, metadata: ModelMetadata, stage: StageSpec
) -> list[str]:
    result = [
        "--source-run-dir",
        str(args.source_run_dir.expanduser().resolve()),
        "--pullbacks-dir",
        str(args.pullbacks_dir.expanduser().resolve()),
        "--source-degree",
        str(metadata.source_degree),
        "--site-count",
        str(metadata.site_count),
        "--bond-dimension",
        str(metadata.bond_dimension),
        "--dictionary-rank",
        str(metadata.dictionary_rank),
        "--output-dimension",
        str(metadata.output_dimension),
        "--precision",
        args.precision,
        "--positive-floor",
        repr(args.positive_floor),
        "--initialization-noise",
        repr(args.initialization_noise),
        "--fixed-log-kappa",
        repr(args.fixed_log_kappa),
        "--transfer-implementation",
        metadata.transfer_implementation,
        "--epochs",
        str(stage.epochs),
        "--batch-size",
        str(args.batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--eval-every",
        str(args.eval_every),
        "--learning-rate",
        repr(stage.learning_rate),
        "--gradient-clip-norm",
        repr(args.gradient_clip_norm),
        "--train-limit",
        str(args.train_limit),
        "--validation-limit",
        str(args.validation_limit),
        "--log-energy-loss-weight",
        repr(args.log_energy_loss_weight),
        "--ma-loss-weight",
        repr(args.ma_loss_weight),
        "--ma-loss-kind",
        args.ma_loss_kind,
        "--tail-loss-weight",
        repr(args.tail_loss_weight),
        "--tail-fraction",
        repr(args.tail_fraction),
        "--tail-loss-kind",
        args.tail_loss_kind,
        "--tail-ratio-threshold",
        repr(args.tail_ratio_threshold),
        "--tail-smooth-temperature",
        repr(args.tail_smooth_temperature),
        "--checkpoint-score-kind",
        args.checkpoint_score_kind,
        "--training-logdet-method",
        args.training_logdet_method,
        "--orthonormalize-every",
        str(args.orthonormalize_every),
        "--device",
        args.device,
        "--torch-seed",
        str(args.torch_seed),
        "--dictionary-seed",
        str(args.dictionary_seed),
    ]
    if args.full_epoch_gradient:
        result.append("--full-epoch-gradient")
    if metadata.fermat_phase_charge_multiplicity:
        result.extend(
            (
                "--fermat-phase-charge-multiplicity",
                str(metadata.fermat_phase_charge_multiplicity),
            )
        )
    if metadata.fermat_s5_orbit_tying:
        result.append("--fermat-s5-orbit-tying")
    if metadata.fermat_two_site_blocking:
        result.append("--fermat-two-site-blocking")
    return result


def _valid_stage(
    stage_dir: Path,
    *,
    input_sha: str,
    stage: StageSpec,
    trainer_args: Sequence[str],
    inherited: int,
) -> tuple[Path, Path, dict[str, Any]] | None:
    summary_path = stage_dir / "stage_summary.json"
    model_path = stage_dir / "plateau_model.pt"
    report_path = stage_dir / "plateau_report.json"
    payload = _json(summary_path)
    if payload is None or not model_path.is_file() or not report_path.is_file():
        return None
    if not (
        payload.get("schema") == "quintic-tn-plateau-stage-v1"
        and payload.get("status")
        in {"validation_plateau", "completed_registered_round_cap"}
        and payload.get("initial_model_sha256") == input_sha
        and payload.get("parameter_scope") == stage.scope
        and payload.get("trainer_args") == list(trainer_args)
        and payload.get("max_rounds") == stage.max_rounds
        and payload.get("plateau_patience_rounds") == stage.patience_rounds
        and payload.get("min_relative_gain") == stage.min_relative_gain
        and payload.get("inherited_bond_dimension") == inherited
        and payload.get("accept_round_cap") is True
        and payload.get("plateau_model_sha256") == sha256_file(model_path)
        and payload.get("plateau_report_sha256") == sha256_file(report_path)
    ):
        return None
    report = _json(report_path)
    if not (
        report is not None
        and report.get("schema") == REPORT_SCHEMA
        and report.get("evaluation_scope") == "development_only"
        and report.get("termination_reason") == "completed_requested_epochs"
        and report.get("artifacts", {}).get("model_sha256") == sha256_file(model_path)
    ):
        return None
    return model_path, report_path, payload


def extract_development_metrics(report: dict[str, Any]) -> dict[str, float | int]:
    """Extract the registered selection metrics and reject schema drift."""

    try:
        normalized = report["training"]["final_validation"]["normalized_volume"]
        values: dict[str, float | int] = {
            "sigma_official_formula": float(normalized["sigma_official_formula"]),
            "weighted_rms_abs_residual": float(normalized["weighted_rms_abs_residual"]),
            "q0.9990": float(normalized["abs_residual_weighted_quantiles"]["q0.9990"]),
            "cvar_0.9900": float(
                normalized["abs_residual_weighted_cvar"]["cvar_0.9900"]
            ),
            "q1.0000": float(normalized["abs_residual_weighted_quantiles"]["q1.0000"]),
            "minimum_metric_eigenvalue": float(
                normalized["min_eigenvalue_weighted_quantiles"]["q0.0000"]
            ),
            "nonpositive_metric_count": int(
                normalized["nonpositive_min_eigenvalue"]["count"]
            ),
        }
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ArmError(
            "final development report lacks the registered validation metrics"
        ) from error
    for name, value in values.items():
        if name == "nonpositive_metric_count":
            raw = normalized["nonpositive_min_eigenvalue"]["count"]
            if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
                raise ArmError("development nonpositive metric count is invalid")
        elif not math.isfinite(float(value)):
            raise ArmError(f"development metric {name} is nonfinite")
    return values


def _valid_publication(arm_dir: Path, config_sha: str) -> bool:
    marker = arm_dir / "final_summary.json"
    model = arm_dir / "final_model.pt"
    report = arm_dir / "final_training_report.json"
    arm_summary = arm_dir / "arm_summary.json"
    payload = _json(marker)
    if payload is None or not all(
        path.is_file() for path in (model, report, arm_summary)
    ):
        return False
    report_payload = _json(report)
    if report_payload is None:
        return False
    try:
        development_metrics = extract_development_metrics(report_payload)
    except ArmError:
        return False
    return bool(
        payload.get("schema") == RESULT_SCHEMA
        and payload.get("status") == "complete"
        and payload.get("arm_config_sha256") == config_sha
        and payload.get("final_model_sha256") == sha256_file(model)
        and payload.get("final_training_report_sha256") == sha256_file(report)
        and payload.get("arm_summary_sha256") == sha256_file(arm_summary)
        and payload.get("development_metrics") == development_metrics
    )


def _publish_final(
    arm_dir: Path,
    model_source: Path,
    report_source: Path,
    arm_summary: dict[str, Any],
    result: dict[str, Any],
) -> None:
    model_path = arm_dir / "final_model.pt"
    report_path = arm_dir / "final_training_report.json"
    model_temp = _prepared_copy(model_source, model_path)
    report_temp = _prepared_copy(report_source, report_path)
    try:
        os.replace(model_temp, model_path)
        os.replace(report_temp, report_path)
        write_json_atomic(arm_dir / "arm_summary.json", arm_summary)
        result["arm_summary_sha256"] = sha256_file(arm_dir / "arm_summary.json")
        # Commit marker last; readers reject any mixed prior publication by hash.
        write_json_atomic(arm_dir / "final_summary.json", result)
    finally:
        model_temp.unlink(missing_ok=True)
        report_temp.unlink(missing_ok=True)


def run_arm(
    args: argparse.Namespace,
    *,
    python_executable: str = sys.executable,
) -> int:
    initial_model = args.initial_model.expanduser().resolve()
    source_dir = args.source_run_dir.expanduser().resolve()
    pullbacks_dir = args.pullbacks_dir.expanduser().resolve()
    arm_dir = args.arm_dir.expanduser().resolve()
    if not initial_model.is_file():
        raise ArmError(f"initial model does not exist: {initial_model}")
    if _is_within(arm_dir, source_dir) or _is_within(arm_dir, pullbacks_dir):
        raise ArmError("--arm-dir must be independent of frozen source/pullback trees")
    for script in (RESIZE, EXPAND, AUDIT, PLATEAU, TRAINER):
        if not script.is_file():
            raise ArmError(f"required script does not exist: {script}")
    arm_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_model_metadata(initial_model)
    if metadata.source_degree != args.source_degree:
        raise ArmError("--source-degree does not match the registered initial model")
    if not math.isclose(
        metadata.positive_floor, args.positive_floor, rel_tol=0.0, abs_tol=0.0
    ):
        raise ArmError("--positive-floor does not match the registered initial model")
    initial_sha = sha256_file(initial_model)
    input_hashes = capture_hashes(frozen_input_paths(source_dir, pullbacks_dir))
    steps = resolve_transfer_steps(
        args.transfer_step, metadata, args.target_k, args.target_d
    )
    if (
        not any(step.kind == "bond" for step in steps)
        and metadata.precision != args.precision
    ):
        raise ArmError(
            "precision conversion requires a bond step or a pre-cast initial model"
        )
    script_hashes = {
        script.name: sha256_file(script)
        for script in (RESIZE, EXPAND, AUDIT, PLATEAU, TRAINER)
    }
    configuration = {
        "schema": SUMMARY_SCHEMA,
        "initial_model_sha256": initial_sha,
        "initial_metadata": asdict(metadata),
        "source_run_dir": str(source_dir),
        "pullbacks_dir": str(pullbacks_dir),
        "input_sha256": input_hashes,
        "target": {"k": args.target_k, "D": args.target_d},
        "transfer_steps": [asdict(step) for step in steps],
        "stages": [asdict(stage) for stage in args.stage],
        "precision": args.precision,
        "site_transfer_rule": args.site_transfer_rule,
        "site_bridge_one_sided_activation_scale": args.site_bridge_one_sided_activation_scale,
        "site_bridge_seed": args.site_bridge_seed,
        "bond_initialization_noise": args.bond_initialization_noise,
        "bond_one_sided_activation_scale": args.bond_one_sided_activation_scale,
        "bond_seed": args.bond_seed,
        "audit": {
            "points": args.audit_points,
            "chunk_size": args.audit_chunk_size,
            "absolute_potential_tolerance": args.audit_absolute_potential_tolerance,
            "absolute_metric_tolerance": args.audit_absolute_metric_tolerance,
            "relative_metric_tolerance": args.audit_relative_metric_tolerance,
        },
        "training_options": {
            key: value
            for key, value in vars(args).items()
            if key
            not in {
                "initial_model",
                "arm_dir",
                "source_run_dir",
                "pullbacks_dir",
                "transfer_step",
                "stage",
            }
        },
        "script_sha256": script_hashes,
    }
    config_sha = canonical_sha256(configuration)
    if _valid_publication(arm_dir, config_sha):
        print(f"[quintic-study-arm] reuse complete arm {arm_dir}", flush=True)
        return 0

    current_model = initial_model
    current_sha = initial_sha
    transfer_records: list[dict[str, Any]] = []
    for index, step in enumerate(steps, 1):
        before = metadata
        request = {
            "step": asdict(step),
            "source_model_sha256": current_sha,
            "script_sha256": script_hashes[
                RESIZE.name if step.kind == "sites" else EXPAND.name
            ],
            "precision": args.precision if step.kind == "bond" else before.precision,
            "site_transfer_rule": args.site_transfer_rule,
            "site_bridge_one_sided_activation_scale": args.site_bridge_one_sided_activation_scale,
            "site_bridge_seed": args.site_bridge_seed,
            "bond_initialization_noise": args.bond_initialization_noise,
            "bond_one_sided_activation_scale": args.bond_one_sided_activation_scale,
            "bond_seed": args.bond_seed,
        }
        request_sha = canonical_sha256(request)
        transfer_dir = (
            arm_dir
            / "transfers"
            / f"{index:02d}_{step.kind}_{step.target}_{request_sha[:16]}"
        )
        transfer_dir.mkdir(parents=True, exist_ok=True)
        output_model = transfer_dir / "model.pt"
        summary_path = transfer_dir / "summary.json"
        audit_path = transfer_dir / "equivalence.json"
        if not _valid_transfer(step, current_sha, output_model, summary_path):
            if step.kind == "sites":
                command = [
                    python_executable,
                    str(RESIZE),
                    "--model",
                    str(current_model),
                    "--target-site-count",
                    str(step.target),
                    "--transfer-rule",
                    args.site_transfer_rule,
                    "--bridge-one-sided-activation-scale",
                    repr(args.site_bridge_one_sided_activation_scale),
                    "--bridge-noise-seed",
                    str(args.site_bridge_seed),
                    "--out",
                    str(output_model),
                    "--summary",
                    str(summary_path),
                ]
            else:
                command = [
                    python_executable,
                    str(EXPAND),
                    "--model",
                    str(current_model),
                    "--bond-dimension",
                    str(step.target),
                    "--initialization-noise",
                    repr(args.bond_initialization_noise),
                    "--one-sided-activation-scale",
                    repr(args.bond_one_sided_activation_scale),
                    "--seed",
                    str(args.bond_seed),
                    "--precision",
                    args.precision,
                    "--out",
                    str(output_model),
                    "--summary",
                    str(summary_path),
                ]
            _run_child(command, label=f"transfer {index} ({step.kind}:{step.target})")
        if not _valid_transfer(step, current_sha, output_model, summary_path):
            raise ArmError(f"transfer {index} did not publish hash-valid artifacts")
        verify_hashes(input_hashes)
        if sha256_file(current_model) != current_sha:
            raise ArmError("transfer modified its source model")
        output_sha = sha256_file(output_model)
        metadata = load_model_metadata(output_model)
        allow_sites = before.site_count != metadata.site_count
        allow_precision = before.precision != metadata.precision
        if not _valid_audit(audit_path, current_sha, output_sha):
            command = [
                python_executable,
                str(AUDIT),
                "--model-a",
                str(current_model),
                "--model-b",
                str(output_model),
                "--source-run-dir",
                str(source_dir),
                "--pullbacks-dir",
                str(pullbacks_dir),
                "--points",
                str(args.audit_points),
                "--chunk-size",
                str(args.audit_chunk_size),
                "--absolute-potential-tolerance",
                repr(args.audit_absolute_potential_tolerance),
                "--absolute-metric-tolerance",
                repr(args.audit_absolute_metric_tolerance),
                "--relative-metric-tolerance",
                repr(args.audit_relative_metric_tolerance),
                "--device",
                "cpu",
                "--out",
                str(audit_path),
            ]
            if allow_sites:
                command.append("--allow-different-site-counts")
            if allow_precision:
                command.append("--allow-different-precisions")
            _run_child(command, label=f"equivalence audit after transfer {index}")
        if not _valid_audit(audit_path, current_sha, output_sha):
            raise ArmError(f"transfer {index} equivalence audit is not hash-valid")
        verify_hashes(input_hashes)
        transfer_records.append(
            {
                "index": index,
                "kind": step.kind,
                "target": step.target,
                "request_sha256": request_sha,
                "source_model": str(current_model),
                "source_model_sha256": current_sha,
                "model": str(output_model),
                "model_sha256": output_sha,
                "summary": str(summary_path),
                "summary_sha256": sha256_file(summary_path),
                "equivalence": str(audit_path),
                "equivalence_sha256": sha256_file(audit_path),
                "metadata": asdict(metadata),
            }
        )
        current_model, current_sha = output_model, output_sha

    target_sites = args.target_k // metadata.source_degree
    if metadata.site_count != target_sites or metadata.bond_dimension != args.target_d:
        raise ArmError(
            "transferred model metadata does not match the requested endpoint"
        )
    if metadata.precision != args.precision:
        raise ArmError("transferred model precision does not match --precision")
    if (
        any(stage.scope == "new_channels" for stage in args.stage)
        and metadata.inherited_bond_dimension is None
    ):
        raise ArmError("new_channels stage requires bond-expansion provenance")

    stage_records: list[dict[str, Any]] = []
    current_report: Path | None = None
    for index, stage in enumerate(args.stage, 1):
        inherited = (
            metadata.inherited_bond_dimension if stage.scope == "new_channels" else 0
        )
        inherited = int(inherited or 0)
        trainer_args = _trainer_args(args, metadata, stage)
        stage_request = {
            "stage": asdict(stage),
            "input_model_sha256": current_sha,
            "trainer_args": trainer_args,
            "inherited_bond_dimension": inherited,
            "plateau_sha256": script_hashes[PLATEAU.name],
            "trainer_sha256": script_hashes[TRAINER.name],
        }
        stage_sha = canonical_sha256(stage_request)
        stage_dir = arm_dir / "stages" / f"{index:02d}_{stage.name}_{stage_sha[:16]}"
        completed = _valid_stage(
            stage_dir,
            input_sha=current_sha,
            stage=stage,
            trainer_args=trainer_args,
            inherited=inherited,
        )
        if completed is None:
            command = [
                python_executable,
                str(PLATEAU),
                "--initial-model",
                str(current_model),
                "--stage-dir",
                str(stage_dir),
                "--max-rounds",
                str(stage.max_rounds),
                "--min-relative-gain",
                repr(stage.min_relative_gain),
                "--plateau-patience-rounds",
                str(stage.patience_rounds),
                "--parameter-scope",
                stage.scope,
                "--accept-round-cap",
            ]
            if inherited:
                command.extend(("--inherited-bond-dimension", str(inherited)))
            command.extend(("--", *trainer_args))
            _run_child(command, label=f"plateau stage {stage.name}")
            completed = _valid_stage(
                stage_dir,
                input_sha=current_sha,
                stage=stage,
                trainer_args=trainer_args,
                inherited=inherited,
            )
        if completed is None:
            raise ArmError(f"stage {stage.name} did not publish hash-valid artifacts")
        verify_hashes(input_hashes)
        if sha256_file(current_model) != current_sha:
            raise ArmError(f"stage {stage.name} modified its input model")
        model_path, report_path, stage_summary = completed
        current_model = model_path
        current_sha = sha256_file(model_path)
        current_report = report_path
        metadata = load_model_metadata(model_path)
        stage_records.append(
            {
                "index": index,
                "name": stage.name,
                "scope": stage.scope,
                "request_sha256": stage_sha,
                "stage_dir": str(stage_dir),
                "stage_summary": str(stage_dir / "stage_summary.json"),
                "stage_summary_sha256": sha256_file(stage_dir / "stage_summary.json"),
                "model": str(model_path),
                "model_sha256": current_sha,
                "report": str(report_path),
                "report_sha256": sha256_file(report_path),
                "selected_round": stage_summary.get("selected_round"),
                "best_selection_score": stage_summary.get(
                    "selected_best_selection_score"
                ),
                "termination_reason": stage_summary.get("termination_reason"),
            }
        )

    assert current_report is not None
    verify_hashes(input_hashes)
    final_report = _json(current_report)
    if final_report is None:
        raise ArmError("final training report is invalid JSON")
    final_score = final_report.get("training", {}).get("best_selection_score")
    if not isinstance(final_score, (int, float)) or not math.isfinite(
        float(final_score)
    ):
        raise ArmError("final training report has no finite selection score")
    device_memory = final_report.get("environment", {}).get("device_memory")
    development_metrics = extract_development_metrics(final_report)
    final_model_sha = current_sha
    final_report_sha = sha256_file(current_report)
    arm_summary = {
        **configuration,
        "status": "complete",
        "arm_config_sha256": config_sha,
        "initial_model": str(initial_model),
        "transfers": transfer_records,
        "stages_completed": stage_records,
        "final_model_sha256": final_model_sha,
        "final_training_report_sha256": final_report_sha,
        "final_selection_score": float(final_score),
        "evaluation_scope": "development_only",
        "development_metrics": development_metrics,
    }
    result = {
        "schema": RESULT_SCHEMA,
        "status": "complete",
        "arm_config_sha256": config_sha,
        "target": {"k": args.target_k, "D": args.target_d},
        "stage_names": [stage.name for stage in args.stage],
        "stage_scopes": [stage.scope for stage in args.stage],
        "final_model": "final_model.pt",
        "final_model_sha256": final_model_sha,
        "arm_summary": "arm_summary.json",
        "arm_summary_sha256": None,
        "final_training_report": "final_training_report.json",
        "final_training_report_sha256": final_report_sha,
        "final_selection_score": float(final_score),
        "evaluation_scope": "development_only",
        "device_memory": device_memory,
        "development_metrics": development_metrics,
    }
    _publish_final(arm_dir, current_model, current_report, arm_summary, result)
    print(f"[quintic-study-arm] published complete arm {arm_dir}", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run_arm(args)
    except ArmError as error:
        print(f"quintic study arm failed: {error}", file=sys.stderr)
        return error.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
