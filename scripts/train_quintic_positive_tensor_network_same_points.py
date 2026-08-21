#!/usr/bin/env python3
"""Train a positive tensor-network metric on the exact cymetric quintic points."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import (  # noqa: E402
    PositiveTensorNetworkMetric,
    anchored_orthonormal_physical_dictionary,
    canonical_matrix_unit_dictionary,
    phase_charge_active_complex_parameter_count,
    phase_charge_core_masks,
    phase_s5_core_orbit_labels,
    phase_s5_two_site_orbit_labels,
    positive_tensor_network_from_artifact_payload,
    rectangular_reference_factor,
)
from gcicy_metric.pipeline.risk import (  # noqa: E402
    smooth_upper_log_ratio_excess_torch,
    weighted_cvar_torch,
)

try:  # Support both direct CLI execution and package-style test imports.
    from scripts.train_quintic_full_h_same_points import (  # noqa: E402
        cvar,
        ratio_statistics,
        sha256_file,
        write_json,
    )
except ModuleNotFoundError:
    from train_quintic_full_h_same_points import (  # noqa: E402
        cvar,
        ratio_statistics,
        sha256_file,
        write_json,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument(
        "--blind-reference-run-dir",
        type=Path,
        help=(
            "optional run containing the frozen blind_points.npz and "
            "blind_test_tail_arrays.npz; training data and the comparator "
            "report remain sourced from --source-run-dir"
        ),
    )
    parser.add_argument("--pullbacks-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-model", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help=(
            "optional crash-recovery checkpoint, atomically replaced at every "
            "completed validation boundary"
        ),
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help=(
            "resume the current/best model, Adam state, RNG streams, history, "
            "timing, and memory counters from a compatible checkpoint"
        ),
    )
    parser.add_argument(
        "--skip-blind-audit",
        action="store_true",
        help=(
            "publish a development-only result without reading the frozen blind "
            "inputs or writing blind_test_tail_arrays.npz"
        ),
    )
    parser.add_argument(
        "--distillation-teacher-model",
        type=Path,
        help=(
            "optional unconstrained high-accuracy TN whose F-level Fermat "
            "Reynolds projection is used to pretrain this model"
        ),
    )
    parser.add_argument("--distillation-epochs", type=int, default=0)
    parser.add_argument("--distillation-group-samples", type=int, default=16)
    parser.add_argument("--distillation-batch-size", type=int, default=0)
    parser.add_argument("--distillation-learning-rate", type=float, default=3.0e-4)
    parser.add_argument(
        "--freeze-inherited-bond-dimension",
        type=int,
        default=0,
        help=(
            "during nested bond growth, freeze every coefficient-core entry "
            "whose left and right bond indices both lie in the inherited block; "
            "entries touching a newly added bond state remain trainable"
        ),
    )
    parser.add_argument(
        "--freeze-physical-dictionary",
        action="store_true",
        help="hold the shared local operator basis fixed during optimization",
    )
    parser.add_argument(
        "--fermat-phase-charge-multiplicity",
        type=int,
        default=0,
        help=(
            "enforce exact (Z/5Z)^4 phase symmetry with charge-conserving "
            "virtual sectors; zero disables the hard constraint"
        ),
    )
    parser.add_argument(
        "--fermat-s5-orbit-tying",
        action="store_true",
        help=(
            "tie every phase-allowed coefficient across its exact S5 orbit; "
            "requires hard Fermat phase sectors"
        ),
    )
    parser.add_argument(
        "--fermat-two-site-blocking",
        action="store_true",
        help=(
            "replace every even adjacent pair by one complete hard-symmetry "
            "two-site orbit tensor"
        ),
    )
    parser.add_argument(
        "--source-degree",
        type=int,
        choices=(1, 2),
        default=1,
        help="degree of the local Veronese source section vector",
    )
    parser.add_argument("--site-count", type=int, default=9)
    parser.add_argument("--bond-dimension", type=int, default=5)
    parser.add_argument("--dictionary-rank", type=int, default=21)
    parser.add_argument(
        "--output-dimension",
        type=int,
        default=0,
        help=(
            "local purification output dimension; zero uses the source-section "
            "dimension"
        ),
    )
    parser.add_argument("--expected-parameter-count", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--test-batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--positive-floor", type=float, default=1.0e-4)
    parser.add_argument("--initialization-noise", type=float, default=1.0e-2)
    parser.add_argument("--log-energy-loss-weight", type=float, default=1.0)
    parser.add_argument("--ma-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--ma-loss-kind",
        choices=("squared", "absolute"),
        default="squared",
        help=(
            "squared uses (r-1)^2; absolute directly minimizes the weighted "
            "L1 Monge-Ampere residual"
        ),
    )
    parser.add_argument("--tail-loss-weight", type=float, default=0.1)
    parser.add_argument(
        "--fermat-symmetry-loss-weight",
        type=float,
        default=0.0,
        help=(
            "pair each training batch with a random Fermat phase/permutation "
            "image and penalize the squared difference of their log volume ratios"
        ),
    )
    parser.add_argument("--tail-fraction", type=float, default=0.02)
    parser.add_argument(
        "--tail-loss-kind",
        choices=("upper_threshold", "absolute_ratio"),
        default="upper_threshold",
        help=(
            "upper_threshold reproduces the historical one-sided spike loss; "
            "absolute_ratio applies CVaR directly to (r-1)^2"
        ),
    )
    parser.add_argument("--tail-ratio-threshold", type=float, default=1.5)
    parser.add_argument("--tail-smooth-temperature", type=float, default=0.05)
    parser.add_argument("--checkpoint-log-energy-weight", type=float, default=1.0)
    parser.add_argument("--checkpoint-ma-weight", type=float, default=1.0)
    parser.add_argument("--checkpoint-tail-weight", type=float, default=0.1)
    parser.add_argument(
        "--checkpoint-score-kind",
        choices=("energy", "sigma"),
        default="energy",
        help="select checkpoints by the historical energy sum or validation sigma",
    )
    parser.add_argument(
        "--full-epoch-gradient",
        action="store_true",
        help=(
            "accumulate the globally normalized gradient over every training "
            "point and take one optimizer step per epoch"
        ),
    )
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument(
        "--fixed-log-kappa",
        type=float,
        default=None,
        help=(
            "registered log volume normalization; when omitted it is estimated "
            "from the FS metric on the selected training pool"
        ),
    )
    parser.add_argument("--validation-limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--orthonormalize-every", type=int, default=1)
    parser.add_argument(
        "--training-logdet-method",
        choices=("cholesky", "eigvalsh"),
        default="eigvalsh",
    )
    parser.add_argument("--dictionary-seed", type=int, default=202607191)
    parser.add_argument("--torch-seed", type=int, default=202607192)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--precision", choices=("complex64", "complex128"), default="complex64"
    )
    parser.add_argument(
        "--transfer-implementation",
        choices=("scalar", "vectorized"),
        default="vectorized",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_integers = (
        args.source_degree,
        args.site_count,
        args.bond_dimension,
        args.dictionary_rank,
        args.batch_size,
        args.eval_every,
        args.eval_batch_size,
        args.test_batch_size,
        args.orthonormalize_every,
    )
    if min(positive_integers) <= 0 or args.epochs < 0:
        raise ValueError("invalid positive integer training argument")
    if args.distillation_epochs < 0:
        raise ValueError("distillation epochs must be non-negative")
    if args.distillation_epochs and args.distillation_teacher_model is None:
        raise ValueError("distillation epochs require --distillation-teacher-model")
    if args.distillation_group_samples <= 0:
        raise ValueError("distillation group samples must be positive")
    if args.distillation_batch_size < 0:
        raise ValueError("distillation batch size must be non-negative")
    if args.distillation_learning_rate <= 0:
        raise ValueError("distillation learning rate must be positive")
    if args.distillation_epochs and args.source_degree != 1:
        raise ValueError("Fermat Reynolds distillation requires an O(1) source")
    if not 0 <= args.freeze_inherited_bond_dimension < args.bond_dimension:
        raise ValueError(
            "freeze-inherited-bond-dimension must lie in [0, bond_dimension)"
        )
    if args.freeze_inherited_bond_dimension and args.initial_model is None:
        raise ValueError("inherited bond freezing requires --initial-model")
    if args.freeze_inherited_bond_dimension and not args.freeze_physical_dictionary:
        raise ValueError(
            "inherited bond freezing requires --freeze-physical-dictionary so "
            "the inherited subnetwork remains fixed"
        )
    source_count = math.comb(args.source_degree + 4, 4)
    output_dimension = args.output_dimension or source_count
    if output_dimension < source_count:
        raise ValueError(
            "output dimension must be at least the source-section dimension"
        )
    if args.dictionary_rank > output_dimension * source_count:
        raise ValueError("dictionary rank exceeds the local rectangular-map dimension")
    if args.learning_rate <= 0 or args.gradient_clip_norm <= 0:
        raise ValueError("learning rate and gradient clipping must be positive")
    if args.positive_floor < 0 or args.initialization_noise < 0:
        raise ValueError("positive floor and initialization noise must be non-negative")
    if args.fixed_log_kappa is not None and not math.isfinite(args.fixed_log_kappa):
        raise ValueError("fixed log kappa must be finite")
    if not 0 < args.tail_fraction <= 1:
        raise ValueError("tail fraction must lie in (0, 1]")
    if (
        args.tail_loss_kind == "upper_threshold" and args.tail_ratio_threshold <= 1
    ) or args.tail_smooth_temperature <= 0:
        raise ValueError("invalid upper-tail objective configuration")
    loss_weights = (
        args.log_energy_loss_weight,
        args.ma_loss_weight,
        args.tail_loss_weight,
    )
    if min(loss_weights) < 0 or not any(loss_weights):
        raise ValueError("loss weights must be non-negative with one enabled")
    if args.full_epoch_gradient and args.tail_loss_weight:
        raise ValueError(
            "full-epoch gradients require zero tail-loss weight because exact "
            "CVaR is not additive across minibatches"
        )
    if args.fermat_symmetry_loss_weight < 0:
        raise ValueError("Fermat symmetry loss weight must be non-negative")
    if args.fermat_symmetry_loss_weight and args.source_degree != 1:
        raise ValueError(
            "Fermat symmetry pairing is currently implemented for O(1) sources"
        )
    if args.fermat_phase_charge_multiplicity < 0:
        raise ValueError("Fermat phase-charge multiplicity must be non-negative")
    if args.fermat_phase_charge_multiplicity:
        expected_bond_dimension = 21 * args.fermat_phase_charge_multiplicity
        if args.source_degree != 1:
            raise ValueError("hard Fermat phase symmetry requires an O(1) source")
        if output_dimension != 5 or args.dictionary_rank != 25:
            raise ValueError(
                "hard Fermat phase symmetry requires output dimension 5 and "
                "the complete 25-element matrix-unit dictionary"
            )
        if args.bond_dimension != expected_bond_dimension:
            raise ValueError(
                "hard Fermat phase symmetry requires bond dimension "
                f"21*multiplicity={expected_bond_dimension}"
            )
        if not args.freeze_physical_dictionary:
            raise ValueError(
                "hard Fermat phase symmetry requires --freeze-physical-dictionary"
            )
        if args.freeze_inherited_bond_dimension:
            raise ValueError(
                "hard phase-charge masks cannot be combined with inherited-bond "
                "freezing"
            )
        if args.fermat_symmetry_loss_weight:
            raise ValueError(
                "hard phase symmetry makes the soft Fermat symmetry loss redundant"
            )
    if args.fermat_s5_orbit_tying and not args.fermat_phase_charge_multiplicity:
        raise ValueError("S5 orbit tying requires hard Fermat phase sectors")
    if args.fermat_two_site_blocking:
        if not args.fermat_s5_orbit_tying:
            raise ValueError("two-site blocking requires hard phase and S5 symmetry")
        if args.site_count < 4 or args.site_count % 2:
            raise ValueError(
                "full two-site blocking requires an even site count of at least four"
            )


TRAINING_CHECKPOINT_SCHEMA = "quintic-positive-tensor-network-checkpoint-v1"


def parameter_scope_from_args(args: argparse.Namespace) -> str:
    """Name the active parameter family using the workflow's stable vocabulary."""

    if args.freeze_inherited_bond_dimension:
        return "new_channels"
    if args.freeze_physical_dictionary:
        return "cores"
    return "joint"


def load_source_report_for_evaluation(
    path: Path,
    *,
    skip_blind_audit: bool,
) -> dict[str, Any]:
    """Do not expose historical comparator results to development-only runs."""

    if skip_blind_audit:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_python_tree(path: Path) -> str:
    """Hash Python implementation sources together with their relative paths."""

    digest = hashlib.sha256()
    for source in sorted(path.rglob("*.py")):
        relative = source.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(source.stat().st_size.to_bytes(8, "big"))
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _metadata_differences(
    expected: Any,
    observed: Any,
    *,
    prefix: str = "",
) -> list[str]:
    """Return deterministic leaf-level differences for checkpoint diagnostics."""

    if type(expected) is not type(observed):
        return [
            f"{prefix}: expected {expected!r} ({type(expected).__name__}), got "
            f"{observed!r} ({type(observed).__name__})"
        ]
    if isinstance(expected, dict) and isinstance(observed, dict):
        differences = []
        for key in sorted(set(expected) | set(observed)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in expected:
                differences.append(f"{path}: unexpected")
            elif key not in observed:
                differences.append(f"{path}: missing")
            else:
                differences.extend(
                    _metadata_differences(expected[key], observed[key], prefix=path)
                )
        return differences
    if isinstance(expected, (list, tuple)):
        if len(expected) != len(observed):
            return [f"{prefix}: expected length {len(expected)}, got {len(observed)}"]
        differences = []
        for index, (expected_item, observed_item) in enumerate(
            zip(expected, observed, strict=True)
        ):
            differences.extend(
                _metadata_differences(
                    expected_item,
                    observed_item,
                    prefix=f"{prefix}[{index}]",
                )
            )
        return differences
    if expected != observed:
        return [f"{prefix}: expected {expected!r}, got {observed!r}"]
    return []


def validate_training_checkpoint(
    payload: dict,
    *,
    training_semantics: dict,
    frozen_input_hashes: dict,
) -> None:
    """Reject incomplete checkpoints and any semantic or input-hash drift."""

    if not isinstance(payload, dict):
        raise ValueError("resume checkpoint payload must be a dictionary")
    if payload.get("schema") != TRAINING_CHECKPOINT_SCHEMA:
        raise ValueError("unrecognized quintic tensor-network training checkpoint")
    required = {
        "training_semantics",
        "frozen_input_hashes",
        "epoch",
        "next_epoch",
        "current_model_state_dict",
        "optimizer_state_dict",
        "permutation_generator_state",
        "cpu_rng_state",
        "cuda_rng_state",
        "cuda_rng_device",
        "history",
        "best_state_dict",
        "best_score",
        "best_epoch",
        "optimizer_steps",
        "fixed_log_kappa",
        "fixed_log_kappa_source",
        "distillation_evidence",
        "timing_seconds",
        "device_memory",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"resume checkpoint lacks fields: {missing}")

    semantic_differences = _metadata_differences(
        training_semantics,
        payload["training_semantics"],
    )
    if semantic_differences:
        raise ValueError(
            "resume checkpoint training semantics mismatch: "
            + "; ".join(semantic_differences)
        )
    hash_differences = _metadata_differences(
        frozen_input_hashes,
        payload["frozen_input_hashes"],
    )
    if hash_differences:
        raise ValueError(
            "resume checkpoint frozen input hash mismatch: "
            + "; ".join(hash_differences)
        )

    epoch = payload["epoch"]
    next_epoch = payload["next_epoch"]
    best_epoch = payload["best_epoch"]
    optimizer_steps = payload["optimizer_steps"]
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
        raise ValueError("resume checkpoint epoch must be a non-negative integer")
    if (
        not isinstance(next_epoch, int)
        or isinstance(next_epoch, bool)
        or next_epoch != epoch + 1
    ):
        raise ValueError("resume checkpoint next_epoch must equal epoch + 1")
    if (
        not isinstance(best_epoch, int)
        or isinstance(best_epoch, bool)
        or not 0 <= best_epoch <= epoch
    ):
        raise ValueError("resume checkpoint best_epoch is outside its epoch range")
    if (
        not isinstance(optimizer_steps, int)
        or isinstance(optimizer_steps, bool)
        or optimizer_steps < 0
    ):
        raise ValueError("resume checkpoint optimizer_steps must be non-negative")

    history = payload["history"]
    if not isinstance(history, list) or not history:
        raise ValueError("resume checkpoint history must be a non-empty list")
    history_epochs = [
        row.get("epoch") if isinstance(row, dict) else None for row in history
    ]
    if (
        any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in history_epochs
        )
        or history_epochs != sorted(set(history_epochs))
        or history_epochs[-1] != epoch
        or best_epoch not in history_epochs
    ):
        raise ValueError(
            "resume checkpoint history epochs must be unique, increasing, end at "
            "epoch, and contain best_epoch"
        )
    if not isinstance(payload["current_model_state_dict"], dict):
        raise ValueError("resume checkpoint current model state must be a dictionary")
    if not isinstance(payload["best_state_dict"], dict):
        raise ValueError("resume checkpoint best model state must be a dictionary")
    if not isinstance(payload["optimizer_state_dict"], dict):
        raise ValueError("resume checkpoint optimizer state must be a dictionary")
    for name in ("permutation_generator_state", "cpu_rng_state"):
        value = payload[name]
        if not torch.is_tensor(value) or value.ndim != 1:
            raise ValueError(
                f"resume checkpoint {name} must be a one-dimensional tensor"
            )
    cuda_rng_state = payload["cuda_rng_state"]
    cuda_rng_device = payload["cuda_rng_device"]
    if (cuda_rng_state is None) != (cuda_rng_device is None):
        raise ValueError(
            "resume checkpoint CUDA RNG state/device must be both set or null"
        )
    if cuda_rng_state is not None and (
        not torch.is_tensor(cuda_rng_state) or cuda_rng_state.ndim != 1
    ):
        raise ValueError("resume checkpoint CUDA RNG state must be one-dimensional")
    if cuda_rng_device is not None and not isinstance(cuda_rng_device, str):
        raise ValueError("resume checkpoint CUDA RNG device must be a string")
    if not np.isfinite(float(payload["fixed_log_kappa"])):
        raise ValueError("resume checkpoint fixed_log_kappa must be finite")
    if not isinstance(payload["fixed_log_kappa_source"], str):
        raise ValueError("resume checkpoint fixed_log_kappa_source must be a string")
    best_score = float(payload["best_score"])
    if np.isnan(best_score):
        raise ValueError("resume checkpoint best_score must not be NaN")

    timing = payload["timing_seconds"]
    expected_timing_keys = {"distillation", "training", "wall_total"}
    if not isinstance(timing, dict) or set(timing) != expected_timing_keys:
        raise ValueError("resume checkpoint timing_seconds has invalid fields")
    if any(
        not np.isfinite(float(value)) or float(value) < 0 for value in timing.values()
    ):
        raise ValueError(
            "resume checkpoint timing_seconds must be finite and non-negative"
        )
    device_memory = payload["device_memory"]
    if device_memory is not None:
        expected_memory_keys = {
            "maximum_allocated_bytes",
            "maximum_reserved_bytes",
        }
        if (
            not isinstance(device_memory, dict)
            or set(device_memory) != expected_memory_keys
        ):
            raise ValueError("resume checkpoint device_memory has invalid fields")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in device_memory.values()
        ):
            raise ValueError("resume checkpoint device_memory must be non-negative")


def validate_checkpoint_paths(
    *,
    checkpoint_path: Path,
    resume_checkpoint_path: Path | None,
    model_path: Path,
    report_path: Path,
    frozen_input_paths: set[Path],
) -> None:
    """Prevent recovery writes from overwriting final or immutable inputs."""

    if model_path == report_path:
        raise ValueError("final model and report paths must differ")
    if model_path in frozen_input_paths or report_path in frozen_input_paths:
        raise ValueError("final outputs must differ from frozen inputs")
    if checkpoint_path in frozen_input_paths | {model_path, report_path}:
        raise ValueError(
            "checkpoint path must differ from final outputs and frozen inputs"
        )
    if resume_checkpoint_path in frozen_input_paths | {model_path, report_path}:
        raise ValueError(
            "resume checkpoint path must differ from final outputs and frozen inputs"
        )


def current_device_memory(device: torch.device) -> dict[str, int] | None:
    """Return this process's CUDA peak counters, or ``None`` for CPU."""

    if device.type != "cuda":
        return None
    return {
        "maximum_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "maximum_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def merge_device_memory(
    previous: dict[str, int] | None,
    current: dict[str, int] | None,
) -> dict[str, int] | None:
    """Take per-counter maxima over every process in a resumed trajectory."""

    if previous is None:
        return None if current is None else copy.deepcopy(current)
    if current is None:
        return copy.deepcopy(previous)
    return {
        key: max(int(previous[key]), int(current[key]))
        for key in ("maximum_allocated_bytes", "maximum_reserved_bytes")
    }


def _clone_to_cpu(value: Any) -> Any:
    """Recursively detach tensor-bearing optimizer/model state onto CPU."""

    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def _actual_cuda_device(device: torch.device) -> str | None:
    """Resolve ``cuda`` to the concrete logical device whose RNG is in use."""

    if device.type != "cuda":
        return None
    index = torch.cuda.current_device() if device.index is None else device.index
    return f"cuda:{index}"


def build_training_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    permutation_generator: torch.Generator,
    device: torch.device,
    epoch: int,
    next_epoch: int,
    history: list[dict[str, Any]],
    best_state: dict[str, torch.Tensor],
    best_score: float,
    best_epoch: int,
    optimizer_steps: int,
    fixed_log_kappa: float,
    fixed_log_kappa_source: str,
    distillation_evidence: dict[str, Any] | None,
    timing_seconds: dict[str, float],
    device_memory: dict[str, int] | None,
    training_semantics: dict,
    frozen_input_hashes: dict,
) -> dict[str, Any]:
    """Snapshot every mutable value at a completed validation boundary."""

    if next_epoch != epoch + 1:
        raise ValueError("next_epoch must identify the epoch after the checkpoint")
    generator_device = torch.device(permutation_generator.device)
    if generator_device.type != device.type or (
        device.type == "cuda"
        and _actual_cuda_device(generator_device) != _actual_cuda_device(device)
    ):
        raise ValueError("permutation generator device does not match training device")
    actual_cuda_device = _actual_cuda_device(device)
    cuda_rng_state = None
    if actual_cuda_device is not None:
        cuda_rng_state = (
            torch.cuda.get_rng_state(torch.device(actual_cuda_device))
            .detach()
            .cpu()
            .clone()
        )
    payload = {
        "schema": TRAINING_CHECKPOINT_SCHEMA,
        "training_semantics": copy.deepcopy(training_semantics),
        "frozen_input_hashes": copy.deepcopy(frozen_input_hashes),
        "epoch": int(epoch),
        "next_epoch": int(next_epoch),
        "current_model_state_dict": _clone_to_cpu(model.state_dict()),
        "optimizer_state_dict": _clone_to_cpu(optimizer.state_dict()),
        "permutation_generator_state": (
            permutation_generator.get_state().detach().cpu().clone()
        ),
        "cpu_rng_state": torch.get_rng_state().detach().cpu().clone(),
        "cuda_rng_state": cuda_rng_state,
        "cuda_rng_device": actual_cuda_device,
        "history": copy.deepcopy(history),
        "best_state_dict": _clone_to_cpu(best_state),
        "best_score": float(best_score),
        "best_epoch": int(best_epoch),
        "optimizer_steps": int(optimizer_steps),
        "fixed_log_kappa": float(fixed_log_kappa),
        "fixed_log_kappa_source": str(fixed_log_kappa_source),
        "distillation_evidence": copy.deepcopy(distillation_evidence),
        "timing_seconds": {key: float(value) for key, value in timing_seconds.items()},
        "device_memory": copy.deepcopy(device_memory),
    }
    validate_training_checkpoint(
        payload,
        training_semantics=training_semantics,
        frozen_input_hashes=frozen_input_hashes,
    )
    return payload


def save_training_checkpoint_atomic(path: Path, payload: dict) -> None:
    """Atomically replace a checkpoint without exposing a partial torch file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def restore_training_checkpoint_state(
    payload: dict,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    permutation_generator: torch.Generator,
    device: torch.device,
) -> None:
    """Restore model, Adam, shuffle generator, and global Torch RNG streams."""

    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "resume checkpoint requires CUDA RNG state but CUDA is unavailable"
        )
    expected_cuda_device = _actual_cuda_device(device)
    if payload["cuda_rng_device"] != expected_cuda_device:
        raise ValueError(
            "resume checkpoint CUDA RNG device does not match the actual device"
        )
    generator_device = torch.device(permutation_generator.device)
    if generator_device.type != device.type or (
        device.type == "cuda"
        and _actual_cuda_device(generator_device) != expected_cuda_device
    ):
        raise ValueError("permutation generator device does not match training device")
    model.load_state_dict(payload["current_model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    permutation_generator.set_state(
        payload["permutation_generator_state"].detach().cpu()
    )
    torch.set_rng_state(payload["cpu_rng_state"].detach().cpu())
    if expected_cuda_device is not None:
        torch.cuda.set_rng_state(
            payload["cuda_rng_state"].detach().cpu(),
            device=torch.device(expected_cuda_device),
        )


def resumed_training_is_complete(payload: dict, *, requested_epochs: int) -> bool:
    """Return whether a validated boundary already satisfies the request."""

    return int(payload["next_epoch"]) > int(requested_epochs)


def inherited_bond_gradient_mask(parameter, inherited_bond_dimension: int):
    """Keep only entries that touch a newly introduced virtual-bond state."""

    if parameter.ndim != 3:
        raise ValueError("coefficient-core parameters must have three axes")
    inherited = int(inherited_bond_dimension)
    if inherited <= 0:
        return parameter.new_ones(parameter.shape)
    mask = parameter.new_ones(parameter.shape)
    old_left = min(inherited, parameter.shape[0])
    old_right = min(inherited, parameter.shape[1])
    mask[:old_left, :old_right, :] = 0
    return mask


def fermat_phase_charge_gradient_masks(
    model: torch.nn.Module,
    multiplicity: int,
) -> list[torch.Tensor]:
    """Materialize exact charge-conservation masks on model devices."""

    masks = phase_charge_core_masks(model.site_count, multiplicity)
    if len(masks) != len(model.coefficient_cores):
        raise ValueError("phase mask and coefficient-core counts disagree")
    result = []
    for coefficient, mask in zip(model.coefficient_cores, masks, strict=True):
        if tuple(mask.shape) != tuple(coefficient.shape):
            raise ValueError(
                "phase mask shape does not match coefficient core: "
                f"{mask.shape} != {tuple(coefficient.shape)}"
            )
        result.append(
            torch.as_tensor(mask, dtype=coefficient.dtype, device=coefficient.device)
        )
    return result


def project_coefficient_cores_to_masks_(
    model: torch.nn.Module,
    masks: list[torch.Tensor],
) -> None:
    """Set every structurally forbidden coefficient to exact zero."""

    with torch.no_grad():
        for coefficient, mask in zip(
            model.coefficient_cores,
            masks,
            strict=True,
        ):
            coefficient.mul_(mask)


def fermat_s5_orbit_label_tensors(
    model: torch.nn.Module,
    multiplicity: int,
) -> list[torch.Tensor]:
    """Move integer S5 orbit labels to the coefficient-core devices."""

    labels = phase_s5_core_orbit_labels(model.site_count, multiplicity)
    result = []
    for coefficient, core_labels in zip(
        model.coefficient_cores,
        labels,
        strict=True,
    ):
        if tuple(core_labels.shape) != tuple(coefficient.shape):
            raise ValueError("S5 orbit labels do not match coefficient-core shape")
        result.append(
            torch.as_tensor(
                core_labels,
                dtype=torch.int64,
                device=coefficient.device,
            )
        )
    return result


def project_tensor_to_s5_orbits(
    tensor: torch.Tensor,
    orbit_labels: torch.Tensor,
) -> torch.Tensor:
    """Orthogonally project one coefficient tensor onto its orbit ties."""

    if tensor.shape != orbit_labels.shape:
        raise ValueError("tensor and S5 orbit-label shapes disagree")
    flat_tensor = tensor.reshape(-1)
    flat_labels = orbit_labels.reshape(-1)
    valid = flat_labels >= 0
    selected_labels = flat_labels[valid]
    if selected_labels.numel() == 0:
        return torch.zeros_like(tensor)
    orbit_count = int(torch.max(selected_labels).item()) + 1
    sums = torch.zeros(
        orbit_count,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    sums = sums.index_add(0, selected_labels, flat_tensor[valid])
    counts = torch.bincount(
        selected_labels,
        minlength=orbit_count,
    ).to(dtype=tensor.real.dtype)
    means = sums / counts
    projected = torch.zeros_like(flat_tensor)
    projected[valid] = means[selected_labels]
    return projected.reshape_as(tensor)


def project_coefficient_cores_to_s5_orbits_(
    model: torch.nn.Module,
    orbit_labels: list[torch.Tensor],
) -> None:
    """Enforce all phase masks and S5 parameter equalities exactly."""

    with torch.no_grad():
        for coefficient, labels in zip(
            model.coefficient_cores,
            orbit_labels,
            strict=True,
        ):
            coefficient.copy_(project_tensor_to_s5_orbits(coefficient, labels))


def apply_fermat_action_torch(
    values: torch.Tensor,
    derivatives: torch.Tensor,
    permutation: torch.Tensor,
    phase_exponents: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one projective Fermat phase/permutation action to an O(1) jet."""

    if values.ndim != 2 or values.shape[1] != 5:
        raise ValueError("Fermat O(1) values must have shape (batch,5)")
    if derivatives.ndim != 3 or derivatives.shape[:2] != values.shape:
        raise ValueError("Fermat O(1) derivatives must have shape (batch,5,d)")
    if permutation.shape != (5,) or phase_exponents.shape != (5,):
        raise ValueError("Fermat action data must contain five coordinates")
    real_dtype = values.real.dtype
    angles = (
        2.0 * math.pi * phase_exponents.to(dtype=real_dtype, device=values.device) / 5.0
    )
    phases = torch.polar(torch.ones_like(angles), angles).to(dtype=values.dtype)
    permutation = permutation.to(dtype=torch.long, device=values.device)
    return (
        values[:, permutation] * phases[None, :],
        derivatives[:, permutation, :] * phases[None, :, None],
    )


def random_fermat_action_torch(
    values: torch.Tensor,
    derivatives: torch.Tensor,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one element of S5 semidirect (Z5)^4 and transport a batch."""

    permutation = torch.randperm(
        5,
        generator=generator,
        device=values.device,
    )
    exponents = torch.randint(
        0,
        5,
        (5,),
        generator=generator,
        device=values.device,
    )
    exponents = torch.remainder(exponents - exponents[-1], 5)
    return apply_fermat_action_torch(
        values,
        derivatives,
        permutation,
        exponents,
    )


def fixed_fermat_actions_torch(
    count: int,
    *,
    generator: torch.Generator,
    complex_dtype: torch.dtype,
    device: torch.device,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Draw one reproducible set of group actions, including the identity."""

    actions: list[tuple[torch.Tensor, torch.Tensor]] = [
        (
            torch.arange(5, dtype=torch.long, device=device),
            torch.ones(5, dtype=complex_dtype, device=device),
        )
    ]
    real_dtype = torch.float32 if complex_dtype == torch.complex64 else torch.float64
    for _ in range(1, int(count)):
        permutation = torch.randperm(5, generator=generator, device=device)
        exponents = torch.randint(
            0,
            5,
            (5,),
            generator=generator,
            device=device,
        )
        exponents = torch.remainder(exponents - exponents[-1], 5)
        angles = 2.0 * math.pi * exponents.to(dtype=real_dtype) / 5.0
        phases = torch.polar(torch.ones_like(angles), angles).to(complex_dtype)
        actions.append((permutation, phases))
    return tuple(actions)


def reynolds_teacher_log_feature_norm(
    teacher: torch.nn.Module,
    values: torch.Tensor,
    actions: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    *,
    chunk_size: int,
) -> torch.Tensor:
    """Compute log of the arithmetic F-level Reynolds average."""

    targets = torch.empty(
        len(values),
        dtype=values.real.dtype,
        device=values.device,
    )
    teacher.eval()
    with torch.no_grad():
        for start in range(0, len(values), chunk_size):
            stop = min(start + chunk_size, len(values))
            block = values[start:stop]
            accumulated = None
            for permutation, phases in actions:
                transformed = block[:, permutation] * phases[None, :]
                row = teacher.log_feature_norm(transformed)
                accumulated = (
                    row if accumulated is None else torch.logaddexp(accumulated, row)
                )
            targets[start:stop] = accumulated - math.log(len(actions))
    return targets


def centered_weighted_log_feature_rms(
    model: torch.nn.Module,
    values: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    *,
    chunk_size: int,
) -> float:
    """Evaluate the gauge-invariant weighted log-F distillation residual."""

    rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), chunk_size):
            stop = min(start + chunk_size, len(values))
            rows.append(model.log_feature_norm(values[start:stop]))
    residual = torch.cat(rows) - targets
    normalized_weights = weights / torch.sum(weights)
    residual = residual - torch.sum(normalized_weights * residual)
    rms = torch.sqrt(torch.sum(normalized_weights * torch.square(residual)))
    return float(rms.detach().cpu())


def veronese_source_features_numpy(
    complex_points: np.ndarray,
    section_derivatives: np.ndarray,
    *,
    source_degree: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate normalized O(1) or O(2) Veronese sections and first jets."""

    points = np.asarray(complex_points)
    derivatives = np.asarray(section_derivatives)
    if points.ndim != 2 or points.shape[1] != 5:
        raise ValueError("quintic points must have shape (points, 5)")
    if derivatives.shape != (len(points), 5, 3):
        raise ValueError("quintic section derivatives must have shape (points, 5, 3)")
    if source_degree == 1:
        return points, derivatives
    if source_degree != 2:
        raise ValueError("only O(1) and O(2) source sections are implemented")

    values = []
    jets = []
    for left in range(5):
        for right in range(left, 5):
            normalization = 1.0 if left == right else math.sqrt(2.0)
            values.append(normalization * points[:, left] * points[:, right])
            jets.append(
                normalization
                * (
                    derivatives[:, left, :] * points[:, right, None]
                    + points[:, left, None] * derivatives[:, right, :]
                )
            )
    return np.stack(values, axis=1), np.stack(jets, axis=1)


def source_features(
    x_values: np.ndarray,
    pullbacks: np.ndarray,
    *,
    source_degree: int,
    complex_dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_coordinates = x_values.shape[1] // 2
    if n_coordinates != 5:
        raise ValueError("the Fermat quintic input must have five complex coordinates")
    complex_points = x_values[:, :5] + 1j * x_values[:, 5:]
    pullback_array = np.asarray(pullbacks)
    if pullback_array.shape[1:] != (3, 5):
        raise ValueError("quintic pullbacks must have shape (points, 3, 5)")
    source_values, source_derivatives = veronese_source_features_numpy(
        complex_points,
        np.transpose(pullback_array, (0, 2, 1)),
        source_degree=source_degree,
    )
    values = torch.tensor(source_values, dtype=complex_dtype, device=device)
    derivatives = torch.tensor(
        source_derivatives,
        dtype=complex_dtype,
        device=device,
    )
    return values, derivatives


def tensor_split(
    x_values: np.ndarray,
    pullbacks: np.ndarray,
    labels: np.ndarray,
    *,
    source_degree: int,
    complex_dtype: torch.dtype,
    real_dtype: torch.dtype,
    device: torch.device,
) -> dict[str, Any]:
    values, derivatives = source_features(
        x_values,
        pullbacks,
        source_degree=source_degree,
        complex_dtype=complex_dtype,
        device=device,
    )
    weights_numpy = np.asarray(labels[:, 0], dtype=np.float64)
    omega_numpy = np.asarray(labels[:, 1], dtype=np.float64)
    if not (
        np.all(np.isfinite(weights_numpy))
        and np.all(weights_numpy > 0)
        and np.all(np.isfinite(omega_numpy))
        and np.all(omega_numpy > 0)
    ):
        raise ValueError("weights and holomorphic volume labels must be positive")
    weights_numpy = weights_numpy / np.sum(weights_numpy)
    return {
        "count": len(x_values),
        "values": values,
        "derivatives": derivatives,
        "weights": torch.tensor(weights_numpy, dtype=real_dtype, device=device),
        "weights_numpy": weights_numpy,
        "log_omega": torch.log(
            torch.tensor(omega_numpy, dtype=real_dtype, device=device)
        ),
    }


def evaluate_raw(
    model: torch.nn.Module,
    dataset: dict[str, Any],
    *,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    raw_rows: list[np.ndarray] = []
    minimum_rows: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, dataset["count"], chunk_size):
            stop = min(start + chunk_size, dataset["count"])
            _, metric = model.potential_and_metric(
                dataset["values"][start:stop],
                dataset["derivatives"][start:stop],
            )
            eigenvalues = torch.linalg.eigvalsh(metric)
            if not bool(torch.all(torch.isfinite(eigenvalues))):
                raise FloatingPointError(
                    "nonfinite metric eigenvalue during evaluation"
                )
            if not bool(torch.all(eigenvalues > 0)):
                raise FloatingPointError("nonpositive tensor-network metric")
            raw = torch.sum(torch.log(eigenvalues), dim=1)
            raw -= dataset["log_omega"][start:stop]
            raw_rows.append(raw.detach().cpu().numpy().astype(np.float64))
            minimum_rows.append(
                torch.min(eigenvalues, dim=1)
                .values.detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
    return np.concatenate(raw_rows), np.concatenate(minimum_rows)


def weighted_log_mean_exp(raw: np.ndarray, weights: np.ndarray) -> float:
    maximum = float(np.max(raw))
    return maximum + math.log(float(np.sum(weights * np.exp(raw - maximum))))


def tail_point_losses_torch(
    log_ratio: torch.Tensor,
    ratio: torch.Tensor,
    *,
    kind: str,
    ratio_threshold: float,
    smooth_temperature: float,
) -> torch.Tensor:
    """Return the per-point loss whose upper weighted tail is optimized."""

    if kind == "absolute_ratio":
        return torch.square(ratio - 1.0)
    if kind != "upper_threshold":
        raise ValueError(f"unsupported tail loss kind: {kind}")
    upper_excess = smooth_upper_log_ratio_excess_torch(
        log_ratio,
        ratio_threshold=ratio_threshold,
        smooth_temperature=smooth_temperature,
    )
    return torch.square(upper_excess)


def ma_point_losses_torch(ratio: torch.Tensor, *, kind: str) -> torch.Tensor:
    """Return the per-point Monge-Ampere loss for the requested norm."""

    residual = ratio - 1.0
    if kind == "squared":
        return torch.square(residual)
    if kind == "absolute":
        return torch.abs(residual)
    raise ValueError(f"unsupported Monge-Ampere loss kind: {kind}")


def fixed_kappa_statistics(
    raw: np.ndarray,
    weights: np.ndarray,
    fixed_log_kappa: float,
    args: argparse.Namespace,
) -> dict[str, float]:
    log_ratio = raw - fixed_log_kappa
    ratio = np.exp(np.clip(log_ratio, -20.0, 20.0))
    tail_loss_kind = getattr(args, "tail_loss_kind", "upper_threshold")
    if tail_loss_kind == "absolute_ratio":
        tail_point_losses = np.square(ratio - 1.0)
    elif tail_loss_kind == "upper_threshold":
        smooth_excess = args.tail_smooth_temperature * np.logaddexp(
            0.0,
            (log_ratio - math.log(args.tail_ratio_threshold))
            / args.tail_smooth_temperature,
        )
        tail_point_losses = np.square(smooth_excess)
    else:
        raise ValueError(f"unsupported tail loss kind: {tail_loss_kind}")
    log_energy = float(np.sum(weights * np.square(log_ratio)))
    ma_energy = float(np.sum(weights * np.square(ratio - 1.0)))
    tail_energy = cvar(
        tail_point_losses,
        weights,
        1.0 - args.tail_fraction,
    )
    score = (
        args.checkpoint_log_energy_weight * log_energy
        + args.checkpoint_ma_weight * ma_energy
        + args.checkpoint_tail_weight * tail_energy
    )
    return {
        "fixed_kappa_log_energy": log_energy,
        "fixed_kappa_log_energy_rms": math.sqrt(log_energy),
        "fixed_kappa_ma_energy": ma_energy,
        "fixed_kappa_sqrt_ma_energy": math.sqrt(ma_energy),
        "fixed_kappa_tail_cvar": tail_energy,
        "selection_score": score,
    }


def evaluate_split(
    model: torch.nn.Module,
    dataset: dict[str, Any],
    *,
    fixed_log_kappa: float,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    raw, minimum = evaluate_raw(model, dataset, chunk_size=args.eval_batch_size)
    normalized, ratio = ratio_statistics(raw, dataset["weights_numpy"], minimum)
    fixed = fixed_kappa_statistics(raw, dataset["weights_numpy"], fixed_log_kappa, args)
    if getattr(args, "checkpoint_score_kind", "energy") == "sigma":
        fixed["selection_score"] = normalized["sigma_official_formula"]
    return {"normalized_volume": normalized, **fixed}, raw, ratio, minimum


def training_log_volume(metric: torch.Tensor, method: str) -> torch.Tensor:
    if method == "cholesky":
        factor, info = torch.linalg.cholesky_ex(metric, check_errors=False)
        torch._assert_async(torch.all(info == 0), "nonpositive training metric")
        diagonal = torch.real(torch.diagonal(factor, dim1=-2, dim2=-1))
        return 2.0 * torch.sum(torch.log(diagonal), dim=1)
    eigenvalues = torch.linalg.eigvalsh(metric)
    torch._assert_async(torch.all(eigenvalues > 0), "nonpositive training metric")
    return torch.sum(torch.log(eigenvalues), dim=1)


def fs_reproduction_check(
    reference_model: torch.nn.Module,
    validation: dict[str, Any],
    official_metrics_path: Path,
) -> dict[str, float]:
    sample_count = min(512, validation["count"])
    with torch.no_grad():
        _, metric = reference_model.potential_and_metric(
            validation["values"][:sample_count],
            validation["derivatives"][:sample_count],
        )
    native = metric.detach().cpu().numpy().astype(np.complex128)
    # PositiveTensorNetworkMetric stores (anti-holomorphic, holomorphic)
    # indices, while cymetric persists (holomorphic, anti-holomorphic).
    computed = np.conj(native)
    official = np.asarray(
        np.load(official_metrics_path, mmap_mode="r")[:sample_count],
        dtype=np.complex128,
    )
    fitted_scale = float(
        np.real(np.vdot(official, computed)) / np.real(np.vdot(official, official))
    )
    relative = np.linalg.norm(
        computed - fitted_scale * official, axis=(1, 2)
    ) / np.maximum(np.linalg.norm(computed, axis=(1, 2)), 1.0e-15)
    result = {
        "tensor_network_native_index_order": "(anti-holomorphic, holomorphic)",
        "official_cymetric_index_order": "(holomorphic, anti-holomorphic)",
        "comparison_transform": "complex conjugation (equivalently matrix transpose)",
        "fitted_tensor_network_over_official_scale": fitted_scale,
        "median_relative_tensor_error_after_scale": float(np.median(relative)),
        "maximum_relative_tensor_error_after_scale": float(np.max(relative)),
        "maximum_determinant_relative_error": float(
            np.max(
                np.abs(np.linalg.det(native) - np.linalg.det(official))
                / np.maximum(np.abs(np.linalg.det(official)), 1.0e-30)
            )
        ),
    }
    if abs(fitted_scale - 1.0) > 2.0e-4 or np.max(relative) > 2.0e-4:
        raise RuntimeError(f"tensor-network/official FS mismatch: {result}")
    return result


def model_payload(
    model: torch.nn.Module,
    state: dict[str, torch.Tensor],
    *,
    args: argparse.Namespace,
    fixed_log_kappa: float,
    source_hashes: dict[str, str],
    initial_model_evidence: dict[str, Any] | None,
    fixed_log_kappa_source: str | None = None,
) -> dict[str, Any]:
    total_degree = args.source_degree * args.site_count
    payload = {
        "schema": "quintic-positive-tensor-network-v1",
        "geometry": "Fermat quintic hypersurface X_5 in P^4",
        "state_dict": state,
        "source_degree": args.source_degree,
        "total_degree": total_degree,
        "site_count": args.site_count,
        "bond_dimension": args.bond_dimension,
        "physical_dictionary_rank": args.dictionary_rank,
        "output_dimension": model.output_dimension,
        "architecture": model.architecture,
        "fermat_phase_charge_multiplicity": (args.fermat_phase_charge_multiplicity),
        "fermat_s5_orbit_tying": args.fermat_s5_orbit_tying,
        "fermat_two_site_blocking": args.fermat_two_site_blocking,
        "fermat_two_site_block_starts": (
            list(range(0, args.site_count, 2)) if args.fermat_two_site_blocking else []
        ),
        "fermat_block_compact_storage": args.fermat_two_site_blocking,
        "hard_symmetry": (
            "fermat_phase_and_s5_invariant_degree_two_blocks"
            if args.fermat_two_site_blocking
            else (
                "fermat_phase_charge_conservation_and_s5_orbit_tying"
                if args.fermat_s5_orbit_tying
                else (
                    "fermat_phase_charge_conservation"
                    if args.fermat_phase_charge_multiplicity
                    else None
                )
            )
        ),
        "transfer_implementation": args.transfer_implementation,
        "trainable_physical_dictionary": not args.freeze_physical_dictionary,
        "physical_dictionary_gauge": (
            "fixed_row_orthonormal_basis"
            if args.freeze_physical_dictionary
            else "row_orthonormalized_after_each_step"
        ),
        "target_normalization": 1.0 / (math.pi * total_degree),
        "positive_floor": args.positive_floor,
        "precision": args.precision,
        "fixed_log_kappa": fixed_log_kappa,
        "fixed_log_kappa_source": (
            fixed_log_kappa_source or "fubini_study_metric_on_fixed_training_pool"
        ),
        "source_sha256": source_hashes,
        "continuation_initial_model": initial_model_evidence,
    }
    if initial_model_evidence is not None and isinstance(
        initial_model_evidence.get("bond_expansion"), dict
    ):
        # Preserve the exact inherited block across multi-round new-channel stages.
        payload["bond_expansion"] = copy.deepcopy(
            initial_model_evidence["bond_expansion"]
        )
    return payload


def exact_fs_reference_model(
    reference_h: np.ndarray,
    *,
    site_count: int,
    bond_dimension: int,
    total_degree: int,
    positive_floor: float,
    transfer_implementation: str,
    seed: int,
    complex_dtype: torch.dtype,
    device: torch.device | str,
) -> PositiveTensorNetworkMetric:
    """Build an exact FS tensor power independent of any learned dictionary."""

    model = PositiveTensorNetworkMetric(
        reference_h,
        site_count=site_count,
        bond_dimension=bond_dimension,
        target_normalization=1.0 / (math.pi * total_degree),
        positive_floor=positive_floor,
        initialization_noise=0.0,
        physical_dictionary=None,
        transfer_implementation=transfer_implementation,
        seed=seed,
        dtype=complex_dtype,
        device=device,
    )
    model.requires_grad_(False)
    return model


def main() -> None:
    args = parse_args()
    validate_args(args)
    started = time.perf_counter()
    source_dir = args.source_run_dir.expanduser().resolve()
    blind_source_dir = (
        args.blind_reference_run_dir.expanduser().resolve()
        if args.blind_reference_run_dir is not None
        else source_dir
    )
    pullbacks_dir = (
        args.pullbacks_dir.expanduser().resolve()
        if args.pullbacks_dir is not None
        else source_dir / "full_h_common_geometry"
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    model_path = output_dir / "best_tensor_network.pt"
    report_path = output_dir / "report.json"
    arrays_path = output_dir / "blind_test_tail_arrays.npz"
    write_json(status_path, {"state": "running", "phase": "initializing"})

    try:
        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        complex_dtype = (
            torch.complex64 if args.precision == "complex64" else torch.complex128
        )
        real_dtype = (
            torch.float32 if complex_dtype == torch.complex64 else torch.float64
        )
        torch.manual_seed(args.torch_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.torch_seed)
            torch.cuda.reset_peak_memory_stats(device)

        dataset_path = source_dir / "training_data" / "dataset.npz"
        basis_path = source_dir / "training_data" / "basis.pickle"
        blind_path = blind_source_dir / "blind_points.npz"
        cymetric_arrays_path = blind_source_dir / "blind_test_tail_arrays.npz"
        source_report_path = source_dir / "report.json"
        pullback_report_path = pullbacks_dir / "report.json"
        train_pullbacks_path = pullbacks_dir / "train_pullbacks.npy"
        validation_pullbacks_path = pullbacks_dir / "validation_pullbacks.npy"
        blind_pullbacks_path = pullbacks_dir / "blind_pullbacks.npy"
        official_fs_metrics_path = pullbacks_dir / "validation_official_fs_metrics.npy"
        required = [
            dataset_path,
            basis_path,
            pullback_report_path,
            train_pullbacks_path,
            validation_pullbacks_path,
            official_fs_metrics_path,
        ]
        if not args.skip_blind_audit:
            required.extend(
                (
                    source_report_path,
                    blind_path,
                    cymetric_arrays_path,
                    blind_pullbacks_path,
                )
            )
        for path in required:
            if not path.exists():
                raise FileNotFoundError(path)

        source_hashes = {
            "dataset": sha256_file(dataset_path),
            "basis": sha256_file(basis_path),
        }
        if not args.skip_blind_audit:
            source_hashes.update(
                {
                    "blind_points": sha256_file(blind_path),
                    "cymetric_tail": sha256_file(cymetric_arrays_path),
                }
            )
        pullback_report = json.loads(pullback_report_path.read_text(encoding="utf-8"))
        registered_source_hashes = pullback_report.get("source_sha256")
        if not isinstance(registered_source_hashes, dict) or any(
            registered_source_hashes.get(name) != digest
            for name, digest in source_hashes.items()
        ):
            raise RuntimeError("pullbacks do not match the fixed source arrays")
        pullback_paths = {
            "train": train_pullbacks_path,
            "validation": validation_pullbacks_path,
            "validation_official_fs_metrics": official_fs_metrics_path,
        }
        if not args.skip_blind_audit:
            pullback_paths["blind"] = blind_pullbacks_path
        pullback_hashes = {
            name: sha256_file(path) for name, path in pullback_paths.items()
        }
        registered_pullback_hashes = pullback_report.get("output_sha256")
        if not isinstance(registered_pullback_hashes, dict) or any(
            registered_pullback_hashes.get(name) != digest
            for name, digest in pullback_hashes.items()
        ):
            raise RuntimeError("pullback hashes do not match their registered report")

        data = np.load(dataset_path, allow_pickle=False)
        train_x = np.asarray(data["X_train"], dtype=np.float32)
        train_y = np.asarray(data["y_train"], dtype=np.float64)
        validation_x = np.asarray(data["X_val"], dtype=np.float32)
        validation_y = np.asarray(data["y_val"], dtype=np.float64)
        train_pullbacks = np.load(train_pullbacks_path, mmap_mode="r")
        validation_pullbacks = np.load(validation_pullbacks_path, mmap_mode="r")
        if len(train_x) != len(train_pullbacks):
            raise RuntimeError("training point and pullback counts disagree")
        if len(validation_x) != len(validation_pullbacks):
            raise RuntimeError("validation point and pullback counts disagree")
        if args.train_limit:
            train_x, train_y = train_x[: args.train_limit], train_y[: args.train_limit]
            train_pullbacks = train_pullbacks[: args.train_limit]
        if args.validation_limit:
            validation_x = validation_x[: args.validation_limit]
            validation_y = validation_y[: args.validation_limit]
            validation_pullbacks = validation_pullbacks[: args.validation_limit]

        write_json(status_path, {"state": "running", "phase": "loading_fixed_data"})
        train = tensor_split(
            train_x,
            train_pullbacks,
            train_y,
            source_degree=args.source_degree,
            complex_dtype=complex_dtype,
            real_dtype=real_dtype,
            device=device,
        )
        validation = tensor_split(
            validation_x,
            validation_pullbacks,
            validation_y,
            source_degree=args.source_degree,
            complex_dtype=complex_dtype,
            real_dtype=real_dtype,
            device=device,
        )

        source_section_count = math.comb(args.source_degree + 4, 4)
        output_dimension = args.output_dimension or source_section_count
        total_degree = args.source_degree * args.site_count
        reference_h = np.eye(source_section_count, dtype=np.complex128)
        target_normalization = 1.0 / (math.pi * total_degree)
        initial_model_payload = None
        initial_model_path = None
        initial_model_evidence = None
        if args.initial_model is None:
            if args.fermat_phase_charge_multiplicity:
                dictionary = canonical_matrix_unit_dictionary(5, 5)
            else:
                dictionary = anchored_orthonormal_physical_dictionary(
                    rectangular_reference_factor(reference_h, output_dimension),
                    args.dictionary_rank,
                    seed=args.dictionary_seed,
                )
        else:
            initial_model_path = args.initial_model.expanduser().resolve()
            if not initial_model_path.exists():
                raise FileNotFoundError(initial_model_path)
            initial_model_payload = torch.load(
                initial_model_path,
                map_location="cpu",
                weights_only=False,
            )
            expected_metadata = {
                "schema": "quintic-positive-tensor-network-v1",
                "architecture": "shared_local_dictionary",
                "source_degree": args.source_degree,
                "site_count": args.site_count,
                "bond_dimension": args.bond_dimension,
                "physical_dictionary_rank": args.dictionary_rank,
                "output_dimension": output_dimension,
                "precision": args.precision,
                "fermat_phase_charge_multiplicity": (
                    args.fermat_phase_charge_multiplicity
                ),
                "fermat_s5_orbit_tying": args.fermat_s5_orbit_tying,
                "fermat_two_site_blocking": args.fermat_two_site_blocking,
            }
            observed_metadata = {
                key: (
                    initial_model_payload.get(key, 1)
                    if key == "source_degree"
                    else (
                        initial_model_payload.get(key, source_section_count)
                        if key == "output_dimension"
                        else (
                            initial_model_payload.get(key, 0)
                            if key == "fermat_phase_charge_multiplicity"
                            else (
                                initial_model_payload.get(key, False)
                                if key
                                in {
                                    "fermat_s5_orbit_tying",
                                    "fermat_two_site_blocking",
                                }
                                else initial_model_payload.get(key)
                            )
                        )
                    )
                )
                for key in expected_metadata
            }
            if observed_metadata != expected_metadata:
                raise ValueError(
                    "initial model metadata mismatch: "
                    f"expected {expected_metadata}, got {observed_metadata}"
                )
            saved_normalization = float(
                initial_model_payload.get("target_normalization", float("nan"))
            )
            if not np.isclose(
                saved_normalization,
                target_normalization,
                rtol=1.0e-12,
                atol=1.0e-14,
            ):
                raise ValueError("initial model target normalization mismatch")
            saved_floor = float(
                initial_model_payload.get("positive_floor", float("nan"))
            )
            if not np.isclose(
                saved_floor,
                args.positive_floor,
                rtol=0.0,
                atol=1.0e-15,
            ):
                raise ValueError("initial model positive floor mismatch")
            initial_state = initial_model_payload.get("state_dict")
            if not isinstance(initial_state, dict):
                raise ValueError("initial model has no state dictionary")
            saved_reference = initial_state.get("reference_h")
            dictionary_value = initial_state.get("physical_dictionary")
            if saved_reference is None or dictionary_value is None:
                raise ValueError("initial model lacks reference or dictionary tensors")
            saved_reference_numpy = np.asarray(
                saved_reference.detach().cpu(), dtype=np.complex128
            )
            if not np.allclose(
                saved_reference_numpy,
                reference_h,
                rtol=1.0e-12,
                atol=1.0e-12,
            ):
                raise ValueError("initial model reference H mismatch")
            dictionary = np.asarray(
                dictionary_value.detach().cpu(), dtype=np.complex128
            )
            if dictionary.shape != (
                args.dictionary_rank,
                output_dimension,
                source_section_count,
            ):
                raise ValueError("initial model physical dictionary shape mismatch")
            if args.fermat_phase_charge_multiplicity and not np.array_equal(
                dictionary,
                canonical_matrix_unit_dictionary(5, 5),
            ):
                raise ValueError(
                    "hard Fermat phase symmetry requires the canonical matrix units"
                )
            if args.freeze_inherited_bond_dimension:
                expansion = initial_model_payload.get("bond_expansion")
                if not isinstance(expansion, dict):
                    raise ValueError(
                        "inherited bond freezing requires a bond-expanded artifact"
                    )
                observed_inherited = int(expansion.get("source_bond_dimension", -1))
                if observed_inherited != args.freeze_inherited_bond_dimension:
                    raise ValueError(
                        "bond expansion source dimension does not match "
                        "--freeze-inherited-bond-dimension"
                    )
            initial_model_evidence = {
                "path": str(initial_model_path),
                "sha256": sha256_file(initial_model_path),
                **observed_metadata,
            }
            if isinstance(initial_model_payload.get("bond_expansion"), dict):
                initial_model_evidence["bond_expansion"] = copy.deepcopy(
                    initial_model_payload["bond_expansion"]
                )
        model = PositiveTensorNetworkMetric(
            reference_h,
            site_count=args.site_count,
            bond_dimension=args.bond_dimension,
            target_normalization=target_normalization,
            output_dimension=output_dimension,
            positive_floor=args.positive_floor,
            initialization_noise=args.initialization_noise,
            physical_dictionary=dictionary,
            trainable_physical_dictionary=(
                False if args.fermat_phase_charge_multiplicity else True
            ),
            transfer_implementation=args.transfer_implementation,
            seed=args.torch_seed,
            dtype=complex_dtype,
            device=device,
        )
        blocked_pair_starts = (
            tuple(range(0, args.site_count, 2)) if args.fermat_two_site_blocking else ()
        )
        blocked_projection_errors = ()
        if blocked_pair_starts:
            blocked_labels = tuple(
                phase_s5_two_site_orbit_labels(
                    left_boundary=start == 0,
                    right_boundary=start + 1 == args.site_count - 1,
                    multiplicity=args.fermat_phase_charge_multiplicity,
                )
                for start in blocked_pair_starts
            )
            blocked_projection_errors = model.block_two_site_coefficient_orbits_(
                blocked_pair_starts,
                blocked_labels,
                drop_covered_coefficient_cores=True,
            )
        if initial_model_payload is not None:
            model.load_state_dict(initial_model_payload["state_dict"], strict=True)
        parameter_count = int(model.trainable_real_parameter_count)
        if (
            args.expected_parameter_count
            and parameter_count != args.expected_parameter_count
        ):
            raise RuntimeError(
                f"parameter-count gate failed: {parameter_count} != "
                f"{args.expected_parameter_count}"
            )
        model.physical_dictionary.requires_grad_(not args.freeze_physical_dictionary)
        coefficient_gradient_masks = []
        coefficient_orbit_labels = []
        gradient_hook_handles = []
        if args.freeze_inherited_bond_dimension:
            for coefficient in model.coefficient_cores:
                mask = inherited_bond_gradient_mask(
                    coefficient,
                    args.freeze_inherited_bond_dimension,
                )
                coefficient_gradient_masks.append(mask)
                gradient_hook_handles.append(
                    coefficient.register_hook(
                        lambda gradient, fixed_mask=mask: gradient * fixed_mask
                    )
                )
        elif (
            args.fermat_phase_charge_multiplicity and not args.fermat_two_site_blocking
        ):
            coefficient_gradient_masks = fermat_phase_charge_gradient_masks(
                model,
                args.fermat_phase_charge_multiplicity,
            )
            project_coefficient_cores_to_masks_(
                model,
                coefficient_gradient_masks,
            )
            if args.fermat_s5_orbit_tying:
                coefficient_orbit_labels = fermat_s5_orbit_label_tensors(
                    model,
                    args.fermat_phase_charge_multiplicity,
                )
                project_coefficient_cores_to_s5_orbits_(
                    model,
                    coefficient_orbit_labels,
                )
                for coefficient, labels in zip(
                    model.coefficient_cores,
                    coefficient_orbit_labels,
                    strict=True,
                ):
                    gradient_hook_handles.append(
                        coefficient.register_hook(
                            lambda gradient, fixed_labels=labels: (
                                project_tensor_to_s5_orbits(
                                    gradient,
                                    fixed_labels,
                                )
                            )
                        )
                    )
            else:
                for coefficient, mask in zip(
                    model.coefficient_cores,
                    coefficient_gradient_masks,
                    strict=True,
                ):
                    gradient_hook_handles.append(
                        coefficient.register_hook(
                            lambda gradient, fixed_mask=mask: gradient * fixed_mask
                        )
                    )
        active_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        if args.fermat_two_site_blocking:
            active_parameter_count = sum(
                parameter.numel() * (2 if parameter.is_complex() else 1)
                for parameter in model.blocked_two_site_orbit_parameters
            )
        elif coefficient_orbit_labels:
            active_parameter_count = 2 * sum(
                int(torch.max(labels).item()) + 1 for labels in coefficient_orbit_labels
            )
        elif coefficient_gradient_masks:
            active_parameter_count = int(
                sum(
                    torch.count_nonzero(mask).item()
                    * (2 if coefficient.is_complex() else 1)
                    for coefficient, mask in zip(
                        model.coefficient_cores,
                        coefficient_gradient_masks,
                        strict=True,
                    )
                )
            )
        else:
            active_parameter_count = sum(
                parameter.numel() * (2 if parameter.is_complex() else 1)
                for parameter in active_parameters
            )

        teacher_path = (
            None
            if not args.distillation_epochs
            else args.distillation_teacher_model.expanduser().resolve()
        )
        if teacher_path is not None and not teacher_path.is_file():
            raise FileNotFoundError(teacher_path)

        frozen_input_paths = {path.expanduser().resolve() for path in required}
        frozen_input_hashes = {
            "dataset_sha256": source_hashes["dataset"],
            "basis_sha256": source_hashes["basis"],
            "source_report_sha256": (
                None if args.skip_blind_audit else sha256_file(source_report_path)
            ),
            "pullback_report_sha256": sha256_file(pullback_report_path),
            "train_pullbacks_sha256": pullback_hashes["train"],
            "validation_pullbacks_sha256": pullback_hashes["validation"],
            "validation_official_fs_metrics_sha256": pullback_hashes[
                "validation_official_fs_metrics"
            ],
            "blind_points_sha256": source_hashes.get("blind_points"),
            "cymetric_tail_sha256": source_hashes.get("cymetric_tail"),
            "blind_pullbacks_sha256": pullback_hashes.get("blind"),
            "initial_model_sha256": (
                None if initial_model_path is None else sha256_file(initial_model_path)
            ),
            "distillation_teacher_model_sha256": (
                None if teacher_path is None else sha256_file(teacher_path)
            ),
        }
        if initial_model_path is not None:
            frozen_input_paths.add(initial_model_path)
        if teacher_path is not None:
            frozen_input_paths.add(teacher_path)

        parameter_scope = parameter_scope_from_args(args)
        training_semantics = {
            "schema": "quintic-positive-tensor-network-training-semantics-v1",
            "model": {
                "source_degree": args.source_degree,
                "site_count": args.site_count,
                "bond_dimension": args.bond_dimension,
                "dictionary_rank": args.dictionary_rank,
                "output_dimension": output_dimension,
                "expected_parameter_count": args.expected_parameter_count,
                "positive_floor": args.positive_floor,
                "initialization_noise": args.initialization_noise,
                "dictionary_seed": args.dictionary_seed,
                "torch_seed": args.torch_seed,
                "precision": args.precision,
                "transfer_implementation": args.transfer_implementation,
                "fermat_phase_charge_multiplicity": (
                    args.fermat_phase_charge_multiplicity
                ),
                "fermat_s5_orbit_tying": args.fermat_s5_orbit_tying,
                "fermat_two_site_blocking": args.fermat_two_site_blocking,
            },
            "parameters": {
                "scope": parameter_scope,
                "freeze_physical_dictionary": args.freeze_physical_dictionary,
                "freeze_inherited_bond_dimension": (
                    args.freeze_inherited_bond_dimension
                ),
                "orthonormalize_every": args.orthonormalize_every,
                "active_real_parameter_count": active_parameter_count,
            },
            "distillation": {
                "enabled": bool(args.distillation_epochs),
                "epochs": args.distillation_epochs,
                "group_samples": args.distillation_group_samples,
                "batch_size": args.distillation_batch_size,
                "learning_rate": args.distillation_learning_rate,
            },
            "optimization": {
                "optimizer": "torch.optim.Adam",
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "eval_every": args.eval_every,
                "eval_batch_size": args.eval_batch_size,
                "learning_rate": args.learning_rate,
                "gradient_clip_norm": args.gradient_clip_norm,
                "log_energy_loss_weight": args.log_energy_loss_weight,
                "ma_loss_weight": args.ma_loss_weight,
                "ma_loss_kind": args.ma_loss_kind,
                "tail_loss_weight": args.tail_loss_weight,
                "tail_fraction": args.tail_fraction,
                "tail_loss_kind": args.tail_loss_kind,
                "tail_ratio_threshold": args.tail_ratio_threshold,
                "tail_smooth_temperature": args.tail_smooth_temperature,
                "fermat_symmetry_loss_weight": args.fermat_symmetry_loss_weight,
                "checkpoint_log_energy_weight": (args.checkpoint_log_energy_weight),
                "checkpoint_ma_weight": args.checkpoint_ma_weight,
                "checkpoint_tail_weight": args.checkpoint_tail_weight,
                "checkpoint_score_kind": args.checkpoint_score_kind,
                "full_epoch_gradient": args.full_epoch_gradient,
                "training_logdet_method": args.training_logdet_method,
                "fixed_log_kappa_requested": args.fixed_log_kappa,
            },
            "data": {
                "train_limit": args.train_limit,
                "validation_limit": args.validation_limit,
                "test_limit": args.test_limit,
                "test_batch_size": args.test_batch_size,
            },
            "evaluation": {
                "scope": (
                    "development_only"
                    if args.skip_blind_audit
                    else "development_and_blind"
                ),
                "skip_blind_audit": args.skip_blind_audit,
            },
            "device": str(device),
            "implementation": {
                "trainer_sha256": sha256_file(Path(__file__).resolve()),
                "support_script_sha256": sha256_file(
                    ROOT / "scripts" / "train_quintic_full_h_same_points.py"
                ),
                "gcicy_metric_python_sha256": sha256_python_tree(ROOT / "gcicy_metric"),
                "python_version": sys.version,
                "numpy_version": np.__version__,
                "torch_version": str(torch.__version__),
            },
        }

        resume_checkpoint_path = (
            None
            if args.resume_checkpoint is None
            else args.resume_checkpoint.expanduser().resolve()
        )
        if resume_checkpoint_path is not None and not resume_checkpoint_path.is_file():
            raise FileNotFoundError(resume_checkpoint_path)
        checkpoint_path = (
            args.checkpoint.expanduser().resolve()
            if args.checkpoint is not None
            else resume_checkpoint_path
        )
        if checkpoint_path is not None:
            validate_checkpoint_paths(
                checkpoint_path=checkpoint_path,
                resume_checkpoint_path=resume_checkpoint_path,
                model_path=model_path,
                report_path=report_path,
                frozen_input_paths=frozen_input_paths,
            )

        resume_payload = None
        resume_checkpoint_sha256 = None
        if resume_checkpoint_path is not None:
            resume_checkpoint_sha256 = sha256_file(resume_checkpoint_path)
            resume_payload = torch.load(
                resume_checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            validate_training_checkpoint(
                resume_payload,
                training_semantics=training_semantics,
                frozen_input_hashes=frozen_input_hashes,
            )

        distillation_evidence = (
            None
            if resume_payload is None
            else copy.deepcopy(resume_payload["distillation_evidence"])
        )
        distillation_seconds = (
            0.0
            if resume_payload is None
            else float(resume_payload["timing_seconds"]["distillation"])
        )
        if args.distillation_epochs and resume_payload is None:
            teacher_payload = torch.load(
                teacher_path,
                map_location="cpu",
                weights_only=False,
            )
            teacher_metadata = {
                "source_degree": int(teacher_payload.get("source_degree", 1)),
                "site_count": int(teacher_payload.get("site_count", -1)),
                "total_degree": int(teacher_payload.get("total_degree", -1)),
            }
            expected_teacher_metadata = {
                "source_degree": args.source_degree,
                "site_count": args.site_count,
                "total_degree": total_degree,
            }
            if teacher_metadata != expected_teacher_metadata:
                raise ValueError(
                    "distillation teacher degree mismatch: "
                    f"expected {expected_teacher_metadata}, got {teacher_metadata}"
                )
            teacher = positive_tensor_network_from_artifact_payload(
                reference_h,
                teacher_payload,
                device=device,
                trainable_physical_dictionary=False,
            )
            teacher.to(device=device, dtype=complex_dtype)
            teacher.requires_grad_(False)
            teacher.eval()

            distillation_batch_size = args.distillation_batch_size or args.batch_size
            action_generator = torch.Generator(device=device)
            action_generator.manual_seed(args.torch_seed + 1009)
            actions = fixed_fermat_actions_torch(
                args.distillation_group_samples,
                generator=action_generator,
                complex_dtype=complex_dtype,
                device=device,
            )
            write_json(
                status_path,
                {
                    "state": "running",
                    "phase": "reynolds_teacher_targets",
                    "teacher": str(teacher_path),
                    "group_samples": len(actions),
                },
            )
            target_chunk_size = max(
                distillation_batch_size,
                args.eval_batch_size,
            )
            train_teacher_targets = reynolds_teacher_log_feature_norm(
                teacher,
                train["values"],
                actions,
                chunk_size=target_chunk_size,
            )
            validation_teacher_targets = reynolds_teacher_log_feature_norm(
                teacher,
                validation["values"],
                actions,
                chunk_size=target_chunk_size,
            )
            del teacher

            distillation_optimizer = torch.optim.Adam(
                active_parameters,
                lr=args.distillation_learning_rate,
            )
            distillation_generator = torch.Generator(device=device)
            distillation_generator.manual_seed(args.torch_seed + 1013)
            distillation_history = []
            distillation_steps = 0
            distillation_best_epoch = 0
            distillation_best_rms = float("inf")
            distillation_best_state = None
            distillation_started = time.perf_counter()
            for distillation_epoch in range(args.distillation_epochs + 1):
                validation_rms = centered_weighted_log_feature_rms(
                    model,
                    validation["values"],
                    validation_teacher_targets,
                    validation["weights"],
                    chunk_size=distillation_batch_size,
                )
                row = {
                    "epoch": distillation_epoch,
                    "validation_centered_log_feature_rms": validation_rms,
                }
                distillation_history.append(row)
                accepted = validation_rms < distillation_best_rms
                if accepted:
                    distillation_best_epoch = distillation_epoch
                    distillation_best_rms = validation_rms
                    distillation_best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                write_json(
                    output_dir / "distillation_history.json",
                    {
                        "rows": distillation_history,
                        "best_epoch": distillation_best_epoch,
                        "best_validation_centered_log_feature_rms": (
                            distillation_best_rms
                        ),
                    },
                )
                torch.save(
                    {
                        "epoch": distillation_epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": distillation_optimizer.state_dict(),
                        "best_state_dict": distillation_best_state,
                        "best_epoch": distillation_best_epoch,
                        "best_validation_centered_log_feature_rms": (
                            distillation_best_rms
                        ),
                        "optimizer_steps": distillation_steps,
                    },
                    output_dir / "distillation_checkpoint.pt",
                )
                write_json(
                    status_path,
                    {
                        "state": "running",
                        "phase": "distillation",
                        "epoch": distillation_epoch,
                        "best_epoch": distillation_best_epoch,
                        "validation_centered_log_feature_rms": validation_rms,
                    },
                )
                print(
                    f"distill_epoch={distillation_epoch} "
                    f"validation_logF_rms={validation_rms:.6e} "
                    f"accepted={accepted}",
                    flush=True,
                )
                if distillation_epoch == args.distillation_epochs:
                    break

                model.train()
                permutation = torch.randperm(
                    train["count"],
                    generator=distillation_generator,
                    device=device,
                )
                for start in range(
                    0,
                    train["count"],
                    distillation_batch_size,
                ):
                    indices = permutation[start : start + distillation_batch_size]
                    batch_weights = train["weights"][indices]
                    batch_weights = batch_weights / torch.sum(batch_weights)
                    distillation_optimizer.zero_grad(set_to_none=True)
                    residual = (
                        model.log_feature_norm(train["values"][indices])
                        - train_teacher_targets[indices]
                    )
                    residual = residual - torch.sum(batch_weights * residual)
                    loss = torch.sum(batch_weights * torch.square(residual))
                    torch._assert_async(
                        torch.isfinite(loss),
                        "nonfinite distillation objective",
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        active_parameters,
                        args.gradient_clip_norm,
                    )
                    distillation_optimizer.step()
                    if args.fermat_s5_orbit_tying and not args.fermat_two_site_blocking:
                        project_coefficient_cores_to_s5_orbits_(
                            model,
                            coefficient_orbit_labels,
                        )
                    elif (
                        args.fermat_phase_charge_multiplicity
                        and not args.fermat_two_site_blocking
                    ):
                        project_coefficient_cores_to_masks_(
                            model,
                            coefficient_gradient_masks,
                        )
                    distillation_steps += 1
            if distillation_best_state is None:
                raise RuntimeError("distillation produced no finite checkpoint")
            model.load_state_dict(distillation_best_state)
            distillation_seconds = time.perf_counter() - distillation_started
            distillation_evidence = {
                "teacher_model": str(teacher_path),
                "teacher_model_sha256": sha256_file(teacher_path),
                "target": "F-level arithmetic Fermat Reynolds projection",
                "group_samples": len(actions),
                "batch_size": distillation_batch_size,
                "learning_rate": args.distillation_learning_rate,
                "epochs": args.distillation_epochs,
                "optimizer_steps": distillation_steps,
                "best_epoch": distillation_best_epoch,
                "best_validation_centered_log_feature_rms": distillation_best_rms,
                "history": distillation_history,
            }
            del train_teacher_targets, validation_teacher_targets

        reference_model = exact_fs_reference_model(
            reference_h,
            site_count=args.site_count,
            bond_dimension=1,
            total_degree=total_degree,
            positive_floor=args.positive_floor,
            transfer_implementation=args.transfer_implementation,
            seed=args.torch_seed,
            complex_dtype=complex_dtype,
            device=device,
        )
        fs_check = fs_reproduction_check(
            reference_model, validation, official_fs_metrics_path
        )
        expected_fixed_log_kappa_source = (
            "fubini_study_metric_on_selected_training_pool"
            if args.fixed_log_kappa is None
            else "registered_command_line_value"
        )
        if resume_payload is not None:
            if (
                resume_payload["fixed_log_kappa_source"]
                != expected_fixed_log_kappa_source
            ):
                raise ValueError(
                    "resume checkpoint fixed_log_kappa source does not match "
                    "the request"
                )
            fixed_log_kappa = float(resume_payload["fixed_log_kappa"])
            fixed_log_kappa_source = str(resume_payload["fixed_log_kappa_source"])
        elif args.fixed_log_kappa is None:
            fs_train_raw, _ = evaluate_raw(
                reference_model, train, chunk_size=args.eval_batch_size
            )
            fixed_log_kappa = weighted_log_mean_exp(
                fs_train_raw, train["weights_numpy"]
            )
            fixed_log_kappa_source = "fubini_study_metric_on_selected_training_pool"
            del fs_train_raw
        else:
            fixed_log_kappa = float(args.fixed_log_kappa)
            fixed_log_kappa_source = "registered_command_line_value"
        del reference_model

        optimizer = torch.optim.Adam(active_parameters, lr=args.learning_rate)
        generator = torch.Generator(device=device)
        generator.manual_seed(args.torch_seed + 1)
        best_state = _clone_to_cpu(model.state_dict())
        best_score = float("inf")
        best_epoch = 0
        history: list[dict[str, Any]] = []
        optimizer_steps = 0
        starting_epoch = 1
        last_checkpoint_epoch = None
        accumulated_training_seconds = 0.0
        accumulated_wall_seconds = 0.0
        accumulated_device_memory = None
        if resume_payload is not None:
            restore_training_checkpoint_state(
                resume_payload,
                model=model,
                optimizer=optimizer,
                permutation_generator=generator,
                device=device,
            )
            history = copy.deepcopy(resume_payload["history"])
            best_state = _clone_to_cpu(resume_payload["best_state_dict"])
            best_score = float(resume_payload["best_score"])
            best_epoch = int(resume_payload["best_epoch"])
            optimizer_steps = int(resume_payload["optimizer_steps"])
            starting_epoch = int(resume_payload["next_epoch"])
            last_checkpoint_epoch = int(resume_payload["epoch"])
            accumulated_training_seconds = float(
                resume_payload["timing_seconds"]["training"]
            )
            accumulated_wall_seconds = float(
                resume_payload["timing_seconds"]["wall_total"]
            )
            accumulated_device_memory = copy.deepcopy(resume_payload["device_memory"])
            print(
                f"resuming at epoch={starting_epoch} from " f"{resume_checkpoint_path}",
                flush=True,
            )
        training_started = time.perf_counter()

        def train_one_epoch() -> None:
            nonlocal optimizer_steps
            model.train()
            permutation = torch.randperm(
                train["count"], generator=generator, device=device
            )
            if args.full_epoch_gradient:
                optimizer.zero_grad(set_to_none=True)
            for start in range(0, train["count"], args.batch_size):
                indices = permutation[start : start + args.batch_size]
                batch_weights = train["weights"][indices]
                if not args.full_epoch_gradient:
                    batch_weights = batch_weights / torch.sum(batch_weights)
                    optimizer.zero_grad(set_to_none=True)
                _, metric = model.potential_and_metric(
                    train["values"][indices], train["derivatives"][indices]
                )
                raw = training_log_volume(metric, args.training_logdet_method)
                raw -= train["log_omega"][indices]
                log_ratio = raw - fixed_log_kappa
                ratio = torch.exp(torch.clamp(log_ratio, -20.0, 20.0))
                log_energy = torch.sum(batch_weights * torch.square(log_ratio))
                ma_energy = torch.sum(
                    batch_weights * ma_point_losses_torch(ratio, kind=args.ma_loss_kind)
                )
                symmetry_energy = torch.zeros((), dtype=real_dtype, device=device)
                if args.fermat_symmetry_loss_weight:
                    transformed_values, transformed_derivatives = (
                        random_fermat_action_torch(
                            train["values"][indices],
                            train["derivatives"][indices],
                            generator=generator,
                        )
                    )
                    _, transformed_metric = model.potential_and_metric(
                        transformed_values,
                        transformed_derivatives,
                    )
                    transformed_raw = training_log_volume(
                        transformed_metric,
                        args.training_logdet_method,
                    )
                    transformed_raw -= train["log_omega"][indices]
                    transformed_log_ratio = transformed_raw - fixed_log_kappa
                    transformed_ratio = torch.exp(
                        torch.clamp(transformed_log_ratio, -20.0, 20.0)
                    )
                    transformed_log_energy = torch.sum(
                        batch_weights * torch.square(transformed_log_ratio)
                    )
                    transformed_ma_energy = torch.sum(
                        batch_weights
                        * ma_point_losses_torch(
                            transformed_ratio,
                            kind=args.ma_loss_kind,
                        )
                    )
                    log_energy = 0.5 * (log_energy + transformed_log_energy)
                    ma_energy = 0.5 * (ma_energy + transformed_ma_energy)
                    symmetry_energy = torch.sum(
                        batch_weights * torch.square(transformed_log_ratio - log_ratio)
                    )
                if args.tail_loss_weight:
                    tail_point_losses = tail_point_losses_torch(
                        log_ratio,
                        ratio,
                        kind=args.tail_loss_kind,
                        ratio_threshold=args.tail_ratio_threshold,
                        smooth_temperature=args.tail_smooth_temperature,
                    )
                    tail_energy = weighted_cvar_torch(
                        tail_point_losses,
                        batch_weights,
                        tail_fraction=args.tail_fraction,
                    )
                else:
                    tail_energy = torch.zeros((), dtype=real_dtype, device=device)
                loss = (
                    args.log_energy_loss_weight * log_energy
                    + args.ma_loss_weight * ma_energy
                    + args.tail_loss_weight * tail_energy
                    + args.fermat_symmetry_loss_weight * symmetry_energy
                )
                torch._assert_async(
                    torch.isfinite(loss), "nonfinite training objective"
                )
                loss.backward()
                if not args.full_epoch_gradient:
                    torch.nn.utils.clip_grad_norm_(
                        active_parameters, args.gradient_clip_norm
                    )
                    optimizer.step()
                    if args.fermat_s5_orbit_tying and not args.fermat_two_site_blocking:
                        project_coefficient_cores_to_s5_orbits_(
                            model,
                            coefficient_orbit_labels,
                        )
                    elif (
                        args.fermat_phase_charge_multiplicity
                        and not args.fermat_two_site_blocking
                    ):
                        project_coefficient_cores_to_masks_(
                            model,
                            coefficient_gradient_masks,
                        )
                    optimizer_steps += 1
                    if (
                        not args.freeze_physical_dictionary
                        and optimizer_steps % args.orthonormalize_every == 0
                    ):
                        model.orthonormalize_physical_dictionary_()
            if args.full_epoch_gradient:
                torch.nn.utils.clip_grad_norm_(
                    active_parameters, args.gradient_clip_norm
                )
                optimizer.step()
                if args.fermat_s5_orbit_tying and not args.fermat_two_site_blocking:
                    project_coefficient_cores_to_s5_orbits_(
                        model,
                        coefficient_orbit_labels,
                    )
                elif (
                    args.fermat_phase_charge_multiplicity
                    and not args.fermat_two_site_blocking
                ):
                    project_coefficient_cores_to_masks_(
                        model,
                        coefficient_gradient_masks,
                    )
                optimizer_steps += 1
            if (
                not args.freeze_physical_dictionary
                and optimizer_steps % args.orthonormalize_every == 0
            ):
                model.orthonormalize_physical_dictionary_()

        def evaluate_and_record(epoch: int) -> None:
            nonlocal best_epoch
            nonlocal best_score
            nonlocal best_state

            validation_row, _, _, _ = evaluate_split(
                model,
                validation,
                fixed_log_kappa=fixed_log_kappa,
                args=args,
            )
            row = {"epoch": epoch, **validation_row}
            history.append(row)
            score = float(row["selection_score"])
            accepted = np.isfinite(score) and score < best_score
            if accepted:
                best_score = score
                best_epoch = epoch
                best_state = _clone_to_cpu(model.state_dict())
            write_json(output_dir / "training_history.json", {"rows": history})
            write_json(
                status_path,
                {
                    "state": "running",
                    "phase": "training",
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "latest": row,
                },
            )
            normalized = row["normalized_volume"]
            print(
                f"epoch={epoch} score={score:.6e} "
                f"sigma={normalized['sigma_official_formula']:.6e} "
                f"chi={normalized['weighted_rms_abs_residual']:.6e} "
                f"rmax={normalized['ratio_weighted_quantiles']['q1.0000']:.6e} "
                f"accepted={accepted}",
                flush=True,
            )

        def accumulated_timing() -> dict[str, float]:
            now = time.perf_counter()
            return {
                "distillation": float(distillation_seconds),
                "training": (accumulated_training_seconds + now - training_started),
                "wall_total": accumulated_wall_seconds + now - started,
            }

        def save_checkpoint(epoch: int) -> None:
            nonlocal last_checkpoint_epoch

            if checkpoint_path is None:
                return
            checkpoint_payload = build_training_checkpoint(
                model=model,
                optimizer=optimizer,
                permutation_generator=generator,
                device=device,
                epoch=epoch,
                next_epoch=epoch + 1,
                history=history,
                best_state=best_state,
                best_score=best_score,
                best_epoch=best_epoch,
                optimizer_steps=optimizer_steps,
                fixed_log_kappa=fixed_log_kappa,
                fixed_log_kappa_source=fixed_log_kappa_source,
                distillation_evidence=distillation_evidence,
                timing_seconds=accumulated_timing(),
                device_memory=merge_device_memory(
                    accumulated_device_memory,
                    current_device_memory(device),
                ),
                training_semantics=training_semantics,
                frozen_input_hashes=frozen_input_hashes,
            )
            save_training_checkpoint_atomic(checkpoint_path, checkpoint_payload)
            last_checkpoint_epoch = epoch
            print(f"checkpointed epoch={epoch} to {checkpoint_path}", flush=True)

        if resume_payload is None:
            evaluate_and_record(0)
            save_checkpoint(0)
        else:
            # Refresh a distinct checkpoint target immediately.  If the supplied
            # boundary is terminal, the loop below is empty and publication starts.
            save_checkpoint(int(resume_payload["epoch"]))

        for epoch in range(starting_epoch, args.epochs + 1):
            train_one_epoch()
            if epoch % args.eval_every == 0 or epoch == args.epochs:
                evaluate_and_record(epoch)
                save_checkpoint(epoch)

        training_seconds = (
            accumulated_training_seconds + time.perf_counter() - training_started
        )
        model.load_state_dict(best_state)
        final_validation, _, _, _ = evaluate_split(
            model,
            validation,
            fixed_log_kappa=fixed_log_kappa,
            args=args,
        )

        evaluation_scope = (
            "development_only" if args.skip_blind_audit else "development_and_blind"
        )
        blind_seconds = 0.0
        blind_count = None
        blind_arrays_sha256 = None
        blind_test_report: dict[str, Any] = {
            "status": "skipped",
            "reason": "--skip-blind-audit",
        }
        if not args.skip_blind_audit:
            write_json(status_path, {"state": "running", "phase": "blind_audit"})
            blind = np.load(blind_path, allow_pickle=False)
            blind_x = np.asarray(blind["X"], dtype=np.float32)
            blind_weights = np.asarray(blind["weights"], dtype=np.float64)
            blind_omega = np.asarray(blind["omega_squared"], dtype=np.float64)
            blind_pullbacks = np.load(blind_pullbacks_path, mmap_mode="r")
            if len(blind_x) != len(blind_pullbacks):
                raise RuntimeError("blind point and pullback counts disagree")
            if args.test_limit:
                blind_x = blind_x[: args.test_limit]
                blind_weights = blind_weights[: args.test_limit]
                blind_omega = blind_omega[: args.test_limit]
                blind_pullbacks = blind_pullbacks[: args.test_limit]
            blind_raw = np.empty(len(blind_x), dtype=np.float64)
            blind_minimum = np.empty(len(blind_x), dtype=np.float64)
            blind_started = time.perf_counter()
            for start in range(0, len(blind_x), args.test_batch_size):
                stop = min(start + args.test_batch_size, len(blind_x))
                values, derivatives = source_features(
                    blind_x[start:stop],
                    blind_pullbacks[start:stop],
                    source_degree=args.source_degree,
                    complex_dtype=complex_dtype,
                    device=device,
                )
                block = {
                    "count": stop - start,
                    "values": values,
                    "derivatives": derivatives,
                    "log_omega": torch.log(
                        torch.tensor(
                            blind_omega[start:stop],
                            dtype=real_dtype,
                            device=device,
                        )
                    ),
                }
                raw, minimum = evaluate_raw(
                    model, block, chunk_size=args.test_batch_size
                )
                blind_raw[start:stop] = raw
                blind_minimum[start:stop] = minimum
            blind_seconds = time.perf_counter() - blind_started
            blind_count = len(blind_x)
            blind_weights_normalized = blind_weights / np.sum(blind_weights)
            blind_statistics, blind_ratio = ratio_statistics(
                blind_raw, blind_weights_normalized, blind_minimum
            )
            blind_fixed = fixed_kappa_statistics(
                blind_raw, blind_weights_normalized, fixed_log_kappa, args
            )
            np.savez_compressed(
                arrays_path,
                raw_log_volume_ratio=blind_raw,
                normalized_ratio=blind_ratio,
                min_eigenvalue=blind_minimum,
                weights=blind_weights,
                omega_squared=blind_omega,
            )
            blind_arrays_sha256 = sha256_file(arrays_path)
            blind_test_report = {
                "status": "complete",
                "normalized_volume": blind_statistics,
                **blind_fixed,
            }

        torch.save(
            model_payload(
                model,
                best_state,
                args=args,
                fixed_log_kappa=fixed_log_kappa,
                source_hashes=source_hashes,
                initial_model_evidence=initial_model_evidence,
                fixed_log_kappa_source=fixed_log_kappa_source,
            ),
            model_path,
        )

        source_report = load_source_report_for_evaluation(
            source_report_path,
            skip_blind_audit=args.skip_blind_audit,
        )
        ambient_section_count = math.comb(total_degree + 4, 4)
        relation_section_count = (
            math.comb(total_degree - 1, 4) if total_degree >= 5 else 0
        )
        degree_k_section_count = ambient_section_count - relation_section_count
        device_memory = merge_device_memory(
            accumulated_device_memory,
            current_device_memory(device),
        )
        final_wall_seconds = accumulated_wall_seconds + time.perf_counter() - started
        report = {
            "schema": "quintic-positive-tensor-network-same-points-v1",
            "evaluation_scope": evaluation_scope,
            "termination_reason": "completed_requested_epochs",
            "scientific_scope": {
                "geometry": "Fermat quintic hypersurface X_5 in P^4",
                "equation": "z0^5 + z1^5 + z2^5 + z3^5 + z4^5 = 0",
                "ansatz": (
                    "positive purified open-boundary tensor network acting on "
                    f"the O({args.source_degree}) Veronese section vector at "
                    "every site"
                ),
                "kahler_potential": (
                    "K=(pi*k)^(-1) log(||B_theta "
                    "s_source^tensor(site_count)||^2 + reference floor), "
                    f"k={total_degree}"
                ),
                "claim_limit": (
                    "No blind sample was read; this development-only result cannot "
                    "support blind-tail claims."
                    if args.skip_blind_audit
                    else (
                        "The common blind sample measures observed tails; it is not "
                        "a deterministic global sup-norm certificate."
                    )
                ),
            },
            "configuration": vars(args),
            "architecture": {
                "source_degree": args.source_degree,
                "source_section_count": source_section_count,
                "purification_output_dimension_R": output_dimension,
                "site_count_m": args.site_count,
                "site_count_k": total_degree,
                "total_degree_k": total_degree,
                "bond_dimension_D": args.bond_dimension,
                "shared_dictionary_rank_q": args.dictionary_rank,
                "local_rectangular_map_dimension": (
                    output_dimension * source_section_count
                ),
                "trainable_real_parameter_count": parameter_count,
                "active_real_parameter_count": active_parameter_count,
                "parameter_formula": (
                    "2*[22*mu+(site_count/2-2)*69*mu^2] for disjoint "
                    "degree-two Fermat-invariant blocks"
                    if args.fermat_two_site_blocking
                    else (
                        "twice the number of S5 orbits of phase-allowed complex "
                        "coefficients; dense tied storage is temporary"
                        if args.fermat_s5_orbit_tying
                        else (
                            "2*[50*mu+(site_count-2)*265*mu^2] active "
                            "coefficients; dense masked storage is temporary"
                            if args.fermat_phase_charge_multiplicity
                            else "2*q*(R*d+2*D+(m-2)*D^2)"
                        )
                    )
                ),
                "hard_fermat_symmetry": {
                    "phase_subgroup": bool(args.fermat_phase_charge_multiplicity),
                    "phase_charge_multiplicity": (
                        args.fermat_phase_charge_multiplicity
                    ),
                    "phase_virtual_sector_count": (
                        21 * args.fermat_phase_charge_multiplicity
                        if args.fermat_phase_charge_multiplicity
                        else 0
                    ),
                    "enforcement": (
                        "exact charge-conserving coefficient masks"
                        if args.fermat_phase_charge_multiplicity
                        else None
                    ),
                    "permutation_subgroup": args.fermat_s5_orbit_tying,
                    "permutation_enforcement": (
                        "exact coefficient orbit tying with projected gradients"
                        if args.fermat_s5_orbit_tying
                        else None
                    ),
                    "two_site_blocking": args.fermat_two_site_blocking,
                    "two_site_block_starts": list(blocked_pair_starts),
                    "two_site_block_projection_relative_errors": [
                        float(value) for value in blocked_projection_errors
                    ],
                },
                "ambient_degree_k_monomial_count": ambient_section_count,
                "hypersurface_relation_count": relation_section_count,
                "equivalent_degree_k_section_count": degree_k_section_count,
                "dense_full_h_real_parameter_count": degree_k_section_count**2,
                "target_normalization": target_normalization,
                "positive_floor": args.positive_floor,
                "trainable_dictionary": not args.freeze_physical_dictionary,
                "frozen_inherited_bond_dimension": (
                    args.freeze_inherited_bond_dimension
                ),
            },
            "common_point_evidence": {
                "source_run_dir": str(source_dir),
                "blind_reference_run_dir": (
                    None if args.skip_blind_audit else str(blind_source_dir)
                ),
                "source_sha256": source_hashes,
                "pullback_sha256": pullback_hashes,
                "source_report_sha256": (
                    None if args.skip_blind_audit else sha256_file(source_report_path)
                ),
                "tensor_network_blind_arrays_sha256": blind_arrays_sha256,
                "train_points": train["count"],
                "validation_points": validation["count"],
                "blind_points": blind_count,
                "fs_reproduction": fs_check,
            },
            "training": {
                "parameter_scope": parameter_scope,
                "best_epoch": best_epoch,
                "best_selection_score": best_score,
                "fixed_log_kappa": fixed_log_kappa,
                "fixed_log_kappa_source": fixed_log_kappa_source,
                "loss_weights": {
                    "log_energy": args.log_energy_loss_weight,
                    "ma": args.ma_loss_weight,
                    "tail_cvar": args.tail_loss_weight,
                    "fermat_symmetry": args.fermat_symmetry_loss_weight,
                },
                "ma_loss_kind": args.ma_loss_kind,
                "tail_loss_kind": args.tail_loss_kind,
                "checkpoint_score_kind": args.checkpoint_score_kind,
                "full_epoch_gradient": args.full_epoch_gradient,
                "initial_model": initial_model_evidence,
                "distillation": distillation_evidence,
                "inherited_bond_growth": {
                    "frozen_inherited_bond_dimension": (
                        args.freeze_inherited_bond_dimension
                    ),
                    "active_new_channel_real_parameter_count": (
                        active_parameter_count
                        if args.freeze_inherited_bond_dimension
                        else None
                    ),
                    "physical_dictionary_frozen": args.freeze_physical_dictionary,
                },
                "final_validation": final_validation,
                "history": history,
                "optimizer_steps": optimizer_steps,
                "checkpoint": {
                    "enabled": checkpoint_path is not None,
                    "path": (None if checkpoint_path is None else str(checkpoint_path)),
                    "sha256": (
                        None
                        if checkpoint_path is None
                        else sha256_file(checkpoint_path)
                    ),
                    "last_completed_validation_epoch": last_checkpoint_epoch,
                    "resumed": resume_checkpoint_path is not None,
                    "resumed_from": (
                        None
                        if resume_checkpoint_path is None
                        else str(resume_checkpoint_path)
                    ),
                    "resumed_from_sha256": resume_checkpoint_sha256,
                    "terminal_boundary_reused": (
                        False
                        if resume_payload is None
                        else resumed_training_is_complete(
                            resume_payload,
                            requested_epochs=args.epochs,
                        )
                    ),
                },
            },
            "blind_test": blind_test_report,
            "comparators": {
                "official_cymetric_network": (
                    None if args.skip_blind_audit else source_report.get("network")
                ),
                "official_cymetric_blind_test": (
                    None
                    if args.skip_blind_audit
                    else source_report.get("trained_phi_model")
                ),
            },
            "artifacts": {
                "model": str(model_path),
                "model_sha256": sha256_file(model_path),
                "blind_arrays": (None if args.skip_blind_audit else str(arrays_path)),
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "device": str(device),
                "device_memory": device_memory,
            },
            "timing_seconds": {
                "distillation": distillation_seconds,
                "training": training_seconds,
                "blind_audit": blind_seconds,
                "wall_total": final_wall_seconds,
            },
        }
        write_json(report_path, report)
        write_json(
            status_path,
            {
                "state": "complete",
                "phase": "complete",
                "report": str(report_path),
                "report_sha256": sha256_file(report_path),
                "wall_seconds": final_wall_seconds,
            },
        )
        print(
            json.dumps(
                (
                    report["blind_test"]
                    if not args.skip_blind_audit
                    else report["training"]["final_validation"]
                ),
                indent=2,
            ),
            flush=True,
        )
    except Exception as error:
        write_json(
            status_path,
            {
                "state": "failed",
                "phase": "exception",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "wall_seconds": time.perf_counter() - started,
            },
        )
        raise


if __name__ == "__main__":
    main()
