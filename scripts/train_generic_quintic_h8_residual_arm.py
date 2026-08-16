#!/usr/bin/env python3
"""Train one matched k=16 residual TN arm from the frozen generic H8 metric."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.generic_quintic import (  # noqa: E402
    GENERIC_QUINTIC_COEFFICIENTS,
    GENERIC_QUINTIC_EXPONENTS,
)
from gcicy_metric.pipeline.exact_lift_tree import (  # noqa: E402
    build_exact_power_lift_tree,
    expand_exact_power_lift_bonds,
)
from gcicy_metric.pipeline.hypersurface_section_ring import (  # noqa: E402
    HypersurfaceSectionRing,
)
from gcicy_metric.pipeline.positive_multiplication_tree import (  # noqa: E402
    PositiveMultiplicationTreeMetric,
    expand_multiplication_tree_bonds,
)
from gcicy_metric.pipeline.residual_power_lift import (  # noqa: E402
    ExactPowerLiftResidualMetric,
    calibrate_residual_scale,
    dense_power_moments,
)
from scripts.refine_generic_quintic_compiled_tree_native_gn import (  # noqa: E402
    load_disjoint_splits,
    load_excluded_indices,
)
from scripts.train_quintic_full_h_same_points import (  # noqa: E402
    ratio_statistics,
    sha256_file,
    write_json,
)
from scripts.train_quintic_native_power_lift_tree import (  # noqa: E402
    make_dataset,
    normalized_native_e2_raw_gradient,
    paired_improvement,
    tail_guard,
    tail_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architecture",
        choices=("direct-chain", "multiplication-tree"),
        required=True,
    )
    parser.add_argument(
        "--training-stage",
        choices=("all", "root"),
        default="all",
        help=(
            "Train every residual tensor or only the complete tree root "
            "coefficient block. The amplitude follows the warmup schedule."
        ),
    )
    parser.add_argument("--baseline-h8", type=Path, required=True)
    parser.add_argument("--residual-h4", type=Path, required=True)
    parser.add_argument("--train-points", type=Path, required=True)
    parser.add_argument("--train-pullbacks", type=Path, required=True)
    parser.add_argument("--selection-points", type=Path, required=True)
    parser.add_argument("--selection-pullbacks", type=Path, required=True)
    parser.add_argument("--confirmation-points", type=Path, required=True)
    parser.add_argument("--confirmation-pullbacks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--exclude-indices-file",
        type=Path,
        nargs="*",
        default=(),
    )
    parser.add_argument("--direct-bond-dimension", type=int, default=4)
    parser.add_argument(
        "--tree-leaf-edge-dimension",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--tree-internal-edge-dimension",
        type=int,
        default=64,
    )
    parser.add_argument("--rank-activation-scale", type=float, default=0.03)
    parser.add_argument(
        "--effective-residual-weight",
        type=float,
        default=1.0e-3,
        help="Initial residual/base norm ratio after geometric-mean calibration.",
    )
    parser.add_argument("--maximum-steps", type=int, default=4_000)
    parser.add_argument("--minimum-steps", type=int, default=1_000)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=3.0e-6)
    parser.add_argument("--adam-epsilon", type=float, default=1.0e-12)
    parser.add_argument(
        "--amplitude-warmup-steps",
        type=int,
        default=200,
        help=(
            "Keep the residual amplitude fixed while the residual tensors "
            "first learn a useful shape."
        ),
    )
    parser.add_argument(
        "--amplitude-learning-rate-multiplier",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--leaf-learning-rate-multiplier",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--internal-learning-rate-multiplier",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--root-learning-rate-multiplier",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--preserve-residual-tensor-norms-during-warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Remove the pure scale escape direction during shape warmup. "
            "The constraint is released after the warmup."
        ),
    )
    parser.add_argument("--training-batch-size", type=int, default=2_048)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--patience-evaluations", type=int, default=10)
    parser.add_argument(
        "--minimum-relative-improvement",
        type=float,
        default=1.0e-4,
    )
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--train-size", type=int, default=100_000)
    parser.add_argument("--selection-size", type=int, default=20_000)
    parser.add_argument("--confirmation-size", type=int, default=50_000)
    parser.add_argument("--calibration-size", type=int, default=4_096)
    parser.add_argument("--feature-batch-size", type=int, default=512)
    parser.add_argument("--train-chunk-size", type=int, default=128)
    parser.add_argument("--eval-chunk-size", type=int, default=256)
    parser.add_argument(
        "--maximum-selection-tail-relative-degradation",
        type=float,
        default=0.005,
    )
    parser.add_argument(
        "--maximum-confirmation-tail-relative-degradation",
        type=float,
        default=0.0,
    )
    parser.add_argument("--seed", type=int, default=202607484)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--precision",
        choices=("complex64", "complex128"),
        default="complex64",
    )
    parser.add_argument(
        "--allow-baseline-training-data-reuse",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Permit continuation on points already used to fit the H8 baseline.",
    )
    return parser.parse_args()


def generic_ring() -> HypersurfaceSectionRing:
    return HypersurfaceSectionRing(
        GENERIC_QUINTIC_EXPONENTS,
        GENERIC_QUINTIC_COEFFICIENTS,
        pivot_coordinate=0,
    )


def load_h_artifact(
    path: Path,
    *,
    expected_degree: int,
    expected_exponents: np.ndarray,
) -> tuple[np.ndarray, float]:
    payload = np.load(path.expanduser().resolve(), allow_pickle=False)
    degree = int(payload["degree"])
    exponents = np.asarray(payload["exponents"], dtype=np.int64)
    matrix = np.asarray(payload["global_h_matrix"], dtype=np.complex128)
    normalization = float(payload["normalization"])
    if degree != expected_degree:
        raise ValueError(f"expected a degree-{expected_degree} H artifact")
    if not np.array_equal(exponents, expected_exponents):
        raise ValueError("H artifact uses a different quotient basis")
    matrix = 0.5 * (matrix + matrix.conj().T)
    eigenvalues = np.linalg.eigvalsh(matrix)
    if eigenvalues[0] <= 0 or not np.all(np.isfinite(eigenvalues)):
        raise ValueError("H artifact is not positive definite")
    if normalization <= 0 or not np.isfinite(normalization):
        raise ValueError("H artifact normalization is invalid")
    return matrix, normalization


def build_residual_model(
    args: argparse.Namespace,
    h4: np.ndarray,
    normalization4: float,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.nn.Module:
    if args.architecture == "direct-chain":
        rank_one = build_exact_power_lift_tree(
            h4,
            source_normalization=normalization4,
            power=4,
            bond_dimension=1,
            positive_floor=0.0,
            precision=args.precision,
            device=device,
        )
        if args.direct_bond_dimension == 1:
            return rank_one
        return expand_exact_power_lift_bonds(
            rank_one,
            args.direct_bond_dimension,
            relative_activation_scale=args.rank_activation_scale,
            seed=args.seed + 101,
        )

    rank_one = PositiveMultiplicationTreeMetric(
        h4,
        leaf_count=4,
        bond_dimension=1,
        source_normalization=normalization4,
        positive_floor=0.0,
        shared_leaf=True,
        dtype=dtype,
        device=device,
    )
    if (
        args.tree_leaf_edge_dimension == 1
        and args.tree_internal_edge_dimension == 1
    ):
        return rank_one
    target_edges = (
        (args.tree_leaf_edge_dimension,) * 4
        + (args.tree_internal_edge_dimension,) * 2
    )
    return expand_multiplication_tree_bonds(
        rank_one,
        target_edges,
        relative_activation_scale=args.rank_activation_scale,
        seed=args.seed + 202,
    )


def make_dual_dataset(
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    exponents4: np.ndarray,
    exponents8: np.ndarray,
    *,
    feature_batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, Any]:
    residual = make_dataset(
        arrays,
        exponents4,
        feature_batch_size=feature_batch_size,
        complex_dtype=dtype,
        device=device,
    )
    baseline = make_dataset(
        arrays,
        exponents8,
        feature_batch_size=feature_batch_size,
        complex_dtype=dtype,
        device=device,
    )
    if residual["count"] != baseline["count"]:
        raise RuntimeError("degree-4 and degree-8 feature rows are not aligned")
    return {
        "count": residual["count"],
        "residual_values": residual["values"],
        "residual_derivatives": residual["derivatives"],
        "baseline_values": baseline["values"],
        "baseline_derivatives": baseline["derivatives"],
        "weights": residual["weights"],
        "weights_numpy": residual["weights_numpy"],
        "log_omega": residual["log_omega"],
    }


def dataset_subset(
    dataset: dict[str, Any],
    indices: np.ndarray,
) -> dict[str, Any]:
    selected = torch.as_tensor(
        indices,
        dtype=torch.int64,
        device=dataset["weights"].device,
    )
    result = {
        "count": len(indices),
        "residual_values": dataset["residual_values"].index_select(0, selected),
        "residual_derivatives": dataset["residual_derivatives"].index_select(
            0, selected
        ),
        "weights": dataset["weights"].index_select(0, selected),
        "weights_numpy": dataset["weights_numpy"][indices],
        "log_omega": dataset["log_omega"].index_select(0, selected),
    }
    for key in ("baseline_values", "baseline_derivatives"):
        if key in dataset:
            result[key] = dataset[key].index_select(0, selected)
    for key in ("baseline_norm", "baseline_gradient", "baseline_mixed"):
        if key in dataset:
            result[key] = dataset[key].index_select(0, selected)
    return result


def cache_baseline_moments(
    model: ExactPowerLiftResidualMetric,
    dataset: dict[str, Any],
    *,
    chunk_size: int,
) -> None:
    norm_rows: list[torch.Tensor] = []
    gradient_rows: list[torch.Tensor] = []
    mixed_rows: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, dataset["count"], chunk_size):
            stop = min(start + chunk_size, dataset["count"])
            moments = dense_power_moments(
                model.baseline_h,
                dataset["baseline_values"][start:stop],
                dataset["baseline_derivatives"][start:stop],
                power=model.baseline_power,
            )
            norm_rows.append(moments.norm)
            gradient_rows.append(moments.holomorphic_gradient)
            mixed_rows.append(moments.mixed_hessian)
    dataset["baseline_norm"] = torch.cat(norm_rows)
    dataset["baseline_gradient"] = torch.cat(gradient_rows)
    dataset["baseline_mixed"] = torch.cat(mixed_rows)
    del dataset["baseline_values"]
    del dataset["baseline_derivatives"]


def raw_log_volume(
    model: ExactPowerLiftResidualMetric,
    dataset: dict[str, Any],
    *,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw_rows: list[torch.Tensor] = []
    minimum_rows: list[torch.Tensor] = []
    for start in range(0, dataset["count"], chunk_size):
        stop = min(start + chunk_size, dataset["count"])
        if "baseline_norm" in dataset:
            _, metric = model.potential_and_metric_from_cached_baseline(
                dataset["residual_values"][start:stop],
                dataset["residual_derivatives"][start:stop],
                dataset["baseline_norm"][start:stop],
                dataset["baseline_gradient"][start:stop],
                dataset["baseline_mixed"][start:stop],
            )
        else:
            _, metric = model.potential_and_metric(
                dataset["residual_values"][start:stop],
                dataset["residual_derivatives"][start:stop],
                dataset["baseline_values"][start:stop],
                dataset["baseline_derivatives"][start:stop],
            )
        eigenvalues = torch.linalg.eigvalsh(metric)
        if not bool(torch.all(torch.isfinite(eigenvalues))):
            raise FloatingPointError("residual metric has nonfinite eigenvalues")
        if not bool(torch.all(eigenvalues > 0)):
            raise FloatingPointError("residual metric is not positive")
        raw_rows.append(
            torch.sum(torch.log(eigenvalues), dim=1)
            - dataset["log_omega"][start:stop]
        )
        minimum_rows.append(torch.min(eigenvalues, dim=1).values)
    return torch.cat(raw_rows), torch.cat(minimum_rows)


def streaming_native_e2_backward(
    model: ExactPowerLiftResidualMetric,
    dataset: dict[str, Any],
    *,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        detached_raw, _ = raw_log_volume(
            model,
            dataset,
            chunk_size=chunk_size,
        )
        energy, ratio, raw_gradient = normalized_native_e2_raw_gradient(
            detached_raw,
            dataset["weights"],
        )
    for start in range(0, dataset["count"], chunk_size):
        stop = min(start + chunk_size, dataset["count"])
        chunk = {
            key: (
                value[start:stop]
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in dataset.items()
            if key != "weights_numpy"
        }
        chunk["count"] = stop - start
        chunk_raw, _ = raw_log_volume(
            model,
            chunk,
            chunk_size=stop - start,
        )
        torch.sum(raw_gradient[start:stop] * chunk_raw).backward()
    return energy, ratio


def evaluate(
    model: ExactPowerLiftResidualMetric,
    dataset: dict[str, Any],
    *,
    chunk_size: int,
) -> tuple[dict[str, Any], np.ndarray, dict[str, float]]:
    model.eval()
    with torch.no_grad():
        raw, minimum = raw_log_volume(
            model,
            dataset,
            chunk_size=chunk_size,
        )
    statistics, ratio = ratio_statistics(
        raw.cpu().numpy(),
        dataset["weights_numpy"],
        minimum.cpu().numpy(),
    )
    return statistics, ratio, tail_summary(statistics)


def state_to_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def gradient_audit(model: torch.nn.Module) -> dict[str, Any]:
    rows = {}
    total_squared = 0.0
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        norm = (
            0.0
            if gradient is None
            else float(torch.linalg.vector_norm(gradient).detach().cpu())
        )
        rows[name] = {
            "gradient_norm": norm,
            "nonzero": bool(norm > 0),
        }
        total_squared += norm * norm
    return {
        "total_gradient_norm": math.sqrt(total_squared),
        "all_parameter_tensors_nonzero": all(
            row["nonzero"] for row in rows.values()
        ),
        "all_trainable_parameter_tensors_nonzero": all(
            rows[name]["nonzero"]
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ),
        "parameters": rows,
    }


def changed_parameter_report(
    model: torch.nn.Module,
    initial: dict[str, torch.Tensor],
) -> dict[str, Any]:
    changed_real = 0
    total_real = 0
    rows = {}
    for name, parameter in model.named_parameters():
        before = initial[name].to(parameter.device)
        difference = torch.abs(parameter.detach() - before)
        changed = int(torch.count_nonzero(difference).cpu())
        multiplier = 2 if parameter.is_complex() else 1
        changed_real += multiplier * changed
        total_real += multiplier * parameter.numel()
        rows[name] = {
            "shape": list(parameter.shape),
            "real_parameters": multiplier * parameter.numel(),
            "changed_real_coordinates_upper_bound": multiplier * changed,
            "maximum_absolute_change": float(torch.max(difference).cpu()),
            "requires_grad": bool(parameter.requires_grad),
        }
    return {
        "changed_real_coordinates_upper_bound": changed_real,
        "total_real_parameters": total_real,
        "changed_fraction_upper_bound": changed_real / total_real,
        "parameters": rows,
    }


def model_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def residual_tensor_norms(
    model: ExactPowerLiftResidualMetric,
) -> dict[str, torch.Tensor]:
    return {
        name: torch.linalg.vector_norm(parameter.detach()).clone()
        for name, parameter in model.residual_model.named_parameters()
    }


def restore_residual_tensor_norms_(
    model: ExactPowerLiftResidualMetric,
    target_norms: dict[str, torch.Tensor],
) -> None:
    with torch.no_grad():
        for name, parameter in model.residual_model.named_parameters():
            current = torch.linalg.vector_norm(parameter)
            target = target_norms[name].to(parameter.device)
            if bool(torch.isfinite(current) & (current > 0)):
                parameter.mul_(target / current)


def optimizer_groups(
    model: ExactPowerLiftResidualMetric,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []

    def add_group(
        name: str,
        parameters: list[torch.nn.Parameter],
        learning_rate: float,
    ) -> None:
        trainable = [
            parameter for parameter in parameters if parameter.requires_grad
        ]
        if trainable:
            groups.append(
                {
                    "params": trainable,
                    "lr": learning_rate,
                    "group_name": name,
                }
            )

    add_group(
        "residual_amplitude",
        [model.residual_amplitude],
        args.learning_rate * args.amplitude_learning_rate_multiplier,
    )
    if args.architecture == "multiplication-tree":
        residual = model.residual_model
        leaf_parameters = (
            [residual.shared_leaf_tensor]
            if residual.shared_leaf
            else list(residual.leaf_tensors)
        )
        add_group(
            "tree_leaves",
            leaf_parameters,
            args.learning_rate * args.leaf_learning_rate_multiplier,
        )
        add_group(
            "tree_internal",
            list(residual.internal_tensors[:-1]),
            args.learning_rate * args.internal_learning_rate_multiplier,
        )
        add_group(
            "tree_root",
            [residual.internal_tensors[-1]],
            args.learning_rate * args.root_learning_rate_multiplier,
        )
    else:
        add_group(
            "direct_chain",
            list(model.residual_model.parameters()),
            args.learning_rate,
        )
    grouped = {
        id(parameter)
        for group in groups
        for parameter in group["params"]
    }
    expected = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }
    if grouped != expected:
        raise RuntimeError("optimizer parameter groups are incomplete")
    return groups


def learning_rates(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    return {
        str(group["group_name"]): float(group["lr"])
        for group in optimizer.param_groups
    }


def real_parameter_count(
    model: torch.nn.Module,
    *,
    trainable_only: bool,
) -> int:
    return int(
        sum(
            parameter.numel() * (2 if parameter.is_complex() else 1)
            for parameter in model.parameters()
            if not trainable_only or parameter.requires_grad
        )
    )


def baseline_data_reuse_audit(
    baseline_h8: Path,
    train_points: Path,
    train_pullbacks: Path,
) -> dict[str, Any]:
    report_path = baseline_h8.parent / "report.json"
    result: dict[str, Any] = {
        "baseline_report": str(report_path),
        "report_found": report_path.exists(),
        "reuses_baseline_training_points": False,
        "reuses_baseline_training_pullbacks": False,
    }
    if not report_path.exists():
        return result
    report = json.loads(report_path.read_text(encoding="utf-8"))
    configuration = report.get("configuration", {})

    def resolve_report_path(value: Any) -> Path | None:
        if not value:
            return None
        candidate = Path(str(value)).expanduser()
        return (
            candidate.resolve()
            if candidate.is_absolute()
            else (ROOT / candidate).resolve()
        )

    old_points = resolve_report_path(configuration.get("train_points"))
    old_pullbacks = resolve_report_path(
        configuration.get("train_pullbacks")
    )
    new_points = train_points.expanduser().resolve()
    new_pullbacks = train_pullbacks.expanduser().resolve()
    result.update(
        {
            "baseline_training_points": (
                None if old_points is None else str(old_points)
            ),
            "baseline_training_pullbacks": (
                None if old_pullbacks is None else str(old_pullbacks)
            ),
            "reuses_baseline_training_points": old_points == new_points,
            "reuses_baseline_training_pullbacks": (
                old_pullbacks == new_pullbacks
            ),
        }
    )
    return result


def main() -> None:
    args = parse_args()
    positive = (
        args.direct_bond_dimension,
        args.tree_leaf_edge_dimension,
        args.tree_internal_edge_dimension,
        args.rank_activation_scale,
        args.effective_residual_weight,
        args.maximum_steps,
        args.minimum_steps,
        args.learning_rate,
        args.minimum_learning_rate,
        args.adam_epsilon,
        args.amplitude_learning_rate_multiplier,
        args.leaf_learning_rate_multiplier,
        args.internal_learning_rate_multiplier,
        args.root_learning_rate_multiplier,
        args.training_batch_size,
        args.eval_every,
        args.patience_evaluations,
        args.gradient_clip_norm,
        args.train_size,
        args.selection_size,
        args.confirmation_size,
        args.calibration_size,
        args.feature_batch_size,
        args.train_chunk_size,
        args.eval_chunk_size,
        args.threads,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("all residual-arm sizes and rates must be positive")
    if args.amplitude_warmup_steps < 0:
        raise ValueError("amplitude warmup steps cannot be negative")
    if args.minimum_steps > args.maximum_steps:
        raise ValueError("minimum steps cannot exceed maximum steps")
    if args.training_stage == "root" and args.architecture != "multiplication-tree":
        raise ValueError("root-only training requires a multiplication tree")
    if not 0 <= args.minimum_relative_improvement < 1:
        raise ValueError("minimum relative improvement must lie in [0,1)")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed + 1)
    device = torch.device(args.device)
    dtype = (
        torch.complex64
        if args.precision == "complex64"
        else torch.complex128
    )
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite a residual-arm run")
    output_dir.mkdir(parents=True)
    status_path = output_dir / "status.json"
    write_json(status_path, {"state": "running", "phase": "initializing"})
    started = time.perf_counter()

    try:
        ring = generic_ring()
        exponents4 = ring.quotient_exponents(4)
        exponents8 = ring.quotient_exponents(8)
        h8_path = args.baseline_h8.expanduser().resolve()
        h4_path = args.residual_h4.expanduser().resolve()
        data_reuse_audit = baseline_data_reuse_audit(
            h8_path,
            args.train_points,
            args.train_pullbacks,
        )
        if (
            not args.allow_baseline_training_data_reuse
            and (
                data_reuse_audit["reuses_baseline_training_points"]
                or data_reuse_audit["reuses_baseline_training_pullbacks"]
            )
        ):
            raise RuntimeError(
                "continuation training data were already used to fit H8; "
                "supply a fresh native pool"
            )
        h8, normalization8 = load_h_artifact(
            h8_path,
            expected_degree=8,
            expected_exponents=exponents8,
        )
        h4, normalization4 = load_h_artifact(
            h4_path,
            expected_degree=4,
            expected_exponents=exponents4,
        )
        residual_model = build_residual_model(
            args,
            h4,
            normalization4,
            dtype=dtype,
            device=device,
        )
        model = ExactPowerLiftResidualMetric(
            h8,
            residual_model,
            baseline_power=2,
            source_normalization=normalization8,
            residual_amplitude=0.0,
            dtype=dtype,
            device=device,
        )
        if args.training_stage == "root":
            model.residual_model.set_trainable_stage_("root")
            model.residual_amplitude.requires_grad_(True)
        total_parameter_count = real_parameter_count(
            model,
            trainable_only=False,
        )
        active_parameter_count = real_parameter_count(
            model,
            trainable_only=True,
        )
        trainable_names = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        if (
            args.training_stage == "all"
            and len(trainable_names) != len(list(model.named_parameters()))
        ):
            raise RuntimeError("every residual-arm parameter must be trainable")

        exclusions = load_excluded_indices(args.exclude_indices_file)
        arrays, indices = load_disjoint_splits(
            {
                "fit": (
                    args.train_points,
                    args.train_pullbacks,
                    args.train_size,
                ),
                "selection": (
                    args.selection_points,
                    args.selection_pullbacks,
                    args.selection_size,
                ),
            },
            seed=args.seed,
            exclusions=exclusions,
        )
        np.savez_compressed(output_dir / "data_indices.npz", **indices)
        write_json(status_path, {"state": "running", "phase": "features"})
        training = make_dual_dataset(
            arrays["fit"],
            exponents4,
            exponents8,
            feature_batch_size=args.feature_batch_size,
            dtype=dtype,
            device=device,
        )
        selection = make_dual_dataset(
            arrays["selection"],
            exponents4,
            exponents8,
            feature_batch_size=args.feature_batch_size,
            dtype=dtype,
            device=device,
        )

        calibration_count = min(args.calibration_size, training["count"])
        calibration = dataset_subset(
            training,
            np.arange(calibration_count, dtype=np.int64),
        )
        scale = calibrate_residual_scale(
            model,
            calibration["residual_values"],
            calibration["residual_derivatives"],
            calibration["baseline_values"],
            calibration["baseline_derivatives"],
            calibration["weights"],
        )
        model.set_residual_scale_(scale)
        cache_baseline_moments(
            model,
            training,
            chunk_size=args.eval_chunk_size,
        )
        cache_baseline_moments(
            model,
            selection,
            chunk_size=args.eval_chunk_size,
        )
        cache_baseline_moments(
            model,
            calibration,
            chunk_size=args.eval_chunk_size,
        )
        model.set_residual_amplitude_(0.0)
        baseline_selection, baseline_ratio, baseline_tail = evaluate(
            model,
            selection,
            chunk_size=args.eval_chunk_size,
        )
        exact_state = state_to_cpu(model)
        model.set_residual_amplitude_(
            math.sqrt(args.effective_residual_weight)
        )
        activated_selection, _, activated_tail = evaluate(
            model,
            selection,
            chunk_size=args.eval_chunk_size,
        )
        activated_state = state_to_cpu(model)
        initial_gradient_batch = dataset_subset(
            calibration,
            np.arange(
                min(256, calibration["count"]),
                dtype=np.int64,
            ),
        )
        model.zero_grad(set_to_none=True)
        streaming_native_e2_backward(
            model,
            initial_gradient_batch,
            chunk_size=min(
                args.train_chunk_size,
                initial_gradient_batch["count"],
            ),
        )
        initial_gradient_audit = gradient_audit(model)
        model.zero_grad(set_to_none=True)

        best_state = exact_state
        best_step = 0
        best_chi = float(
            baseline_selection["weighted_rms_abs_residual"]
        )
        material_reference = best_chi
        stale_evaluations = 0
        history: list[dict[str, Any]] = [
            {
                "step": 0,
                "state": "exact_h8_power_baseline",
                "sigma": baseline_selection["sigma_official_formula"],
                "chi": best_chi,
                **baseline_tail,
            },
            {
                "step": 0,
                "state": "activated_residual",
                "sigma": activated_selection["sigma_official_formula"],
                "chi": activated_selection["weighted_rms_abs_residual"],
                "residual_coefficient": float(
                    model.residual_weight.detach().cpu()
                ),
                "effective_residual_ratio": float(
                    torch.square(model.residual_amplitude).detach().cpu()
                ),
                **activated_tail,
            },
        ]
        write_json(output_dir / "history.json", {"rows": history})
        print(
            f"architecture={args.architecture} "
            f"total_real_parameters={total_parameter_count} "
            f"active_real_parameters={active_parameter_count} "
            f"baseline_sigma={history[0]['sigma']:.8e} "
            f"activated_sigma={history[1]['sigma']:.8e}",
            flush=True,
        )

        optimizer = torch.optim.Adam(
            optimizer_groups(model, args),
            eps=args.adam_epsilon,
        )
        minimum_learning_rate_factor = (
            args.minimum_learning_rate / args.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda current_step: (
                minimum_learning_rate_factor
                + 0.5
                * (1.0 - minimum_learning_rate_factor)
                * (
                    1.0
                    + math.cos(
                        math.pi
                        * min(current_step, args.maximum_steps)
                        / args.maximum_steps
                    )
                )
            ),
        )
        warmup_tensor_norms = residual_tensor_norms(model)
        write_json(status_path, {"state": "running", "phase": "training"})
        completed_steps = 0
        stopped_early = False
        training_compute_seconds = 0.0
        selection_evaluation_seconds = 0.0
        training_loop_started = time.perf_counter()
        for step in range(1, args.maximum_steps + 1):
            step_started = time.perf_counter()
            batch_indices = rng.choice(
                training["count"],
                size=min(args.training_batch_size, training["count"]),
                replace=False,
            )
            active = dataset_subset(training, batch_indices)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            energy, _ = streaming_native_e2_backward(
                model,
                active,
                chunk_size=args.train_chunk_size,
            )
            amplitude_frozen = step <= args.amplitude_warmup_steps
            if amplitude_frozen:
                model.residual_amplitude.grad = None
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.gradient_clip_norm,
            )
            optimizer.step()
            if (
                amplitude_frozen
                and args.preserve_residual_tensor_norms_during_warmup
            ):
                restore_residual_tensor_norms_(model, warmup_tensor_norms)
            scheduler.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            training_compute_seconds += time.perf_counter() - step_started
            completed_steps = step
            if step % args.eval_every and step != args.maximum_steps:
                continue

            evaluation_started = time.perf_counter()
            candidate, _, candidate_tail = evaluate(
                model,
                selection,
                chunk_size=args.eval_chunk_size,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            selection_evaluation_seconds += (
                time.perf_counter() - evaluation_started
            )
            sigma = float(candidate["sigma_official_formula"])
            chi = float(candidate["weighted_rms_abs_residual"])
            passes_gate = bool(
                sigma < baseline_selection["sigma_official_formula"]
                and chi < best_chi
                and tail_guard(
                    candidate_tail,
                    baseline_tail,
                    relative_degradation=(
                        args.maximum_selection_tail_relative_degradation
                    ),
                )
            )
            row = {
                "step": step,
                "learning_rates": learning_rates(optimizer),
                "amplitude_frozen": amplitude_frozen,
                "train_e2": float(energy.cpu()),
                "gradient_norm": float(gradient_norm),
                "residual_coefficient": float(
                    model.residual_weight.detach().cpu()
                ),
                "effective_residual_ratio": float(
                    torch.square(model.residual_amplitude).detach().cpu()
                ),
                "sigma": sigma,
                "chi": chi,
                "passes_selection_gate": passes_gate,
                "training_compute_seconds": training_compute_seconds,
                "selection_evaluation_seconds": (
                    selection_evaluation_seconds
                ),
                "updates_per_second": (
                    step / max(training_compute_seconds, np.finfo(float).tiny)
                ),
                "training_points_per_second": (
                    step
                    * min(args.training_batch_size, training["count"])
                    / max(training_compute_seconds, np.finfo(float).tiny)
                ),
                **candidate_tail,
            }
            history.append(row)
            if passes_gate:
                best_chi = chi
                best_step = step
                best_state = state_to_cpu(model)
            if passes_gate and chi <= material_reference * (
                1.0 - args.minimum_relative_improvement
            ):
                material_reference = chi
                stale_evaluations = 0
            else:
                stale_evaluations += 1
            write_json(output_dir / "history.json", {"rows": history})
            write_json(
                status_path,
                {
                    "state": "running",
                    "phase": "training",
                    "step": step,
                    "best_step": best_step,
                    "best_chi": best_chi,
                    "stale_evaluations": stale_evaluations,
                    "updates_per_second": row["updates_per_second"],
                },
            )
            print(
                f"step={step} val_sigma={sigma:.8e} val_chi={chi:.8e} "
                f"gate={passes_gate} best_step={best_step} "
                f"updates_per_second={row['updates_per_second']:.3f}",
                flush=True,
            )
            if (
                step >= max(
                    args.minimum_steps,
                    args.amplitude_warmup_steps,
                )
                and stale_evaluations >= args.patience_evaluations
            ):
                stopped_early = True
                break
        training_loop_seconds = time.perf_counter() - training_loop_started

        model.load_state_dict(best_state)
        best_selection, best_selection_ratio, best_selection_tail = evaluate(
            model,
            selection,
            chunk_size=args.eval_chunk_size,
        )
        selection_weights_numpy = np.asarray(
            selection["weights_numpy"],
            dtype=np.float64,
        ).copy()
        changes = changed_parameter_report(model, activated_state)

        del selection
        del training
        del calibration
        if device.type == "cuda":
            torch.cuda.empty_cache()

        write_json(status_path, {"state": "running", "phase": "confirmation"})
        confirmation_arrays, confirmation_indices = load_disjoint_splits(
            {
                "confirmation": (
                    args.confirmation_points,
                    args.confirmation_pullbacks,
                    args.confirmation_size,
                )
            },
            seed=args.seed + 10_000,
            exclusions=exclusions,
        )
        with np.load(output_dir / "data_indices.npz") as existing:
            index_payload = {
                name: np.asarray(existing[name], dtype=np.int64)
                for name in existing.files
            }
        index_payload["confirmation"] = confirmation_indices["confirmation"]
        np.savez_compressed(output_dir / "data_indices.npz", **index_payload)
        confirmation = make_dual_dataset(
            confirmation_arrays["confirmation"],
            exponents4,
            exponents8,
            feature_batch_size=args.feature_batch_size,
            dtype=dtype,
            device=device,
        )
        cache_baseline_moments(
            model,
            confirmation,
            chunk_size=args.eval_chunk_size,
        )
        candidate_confirmation, candidate_ratio, candidate_tail = evaluate(
            model,
            confirmation,
            chunk_size=args.eval_chunk_size,
        )
        selected_amplitude = float(model.residual_amplitude.detach().cpu())
        model.set_residual_amplitude_(0.0)
        baseline_confirmation, confirmation_baseline_ratio, confirmation_baseline_tail = (
            evaluate(
                model,
                confirmation,
                chunk_size=args.eval_chunk_size,
            )
        )
        model.set_residual_amplitude_(selected_amplitude)
        confirmation_paired = paired_improvement(
            confirmation_baseline_ratio,
            candidate_ratio,
            confirmation["weights_numpy"],
        )
        confirmation_accepted = bool(
            best_step > 0
            and candidate_confirmation["sigma_official_formula"]
            < baseline_confirmation["sigma_official_formula"]
            and candidate_confirmation["weighted_rms_abs_residual"]
            < baseline_confirmation["weighted_rms_abs_residual"]
            and confirmation_paired["e2"]["ci95_low"] > 0
            and confirmation_paired["sigma"]["ci95_low"] > 0
            and tail_guard(
                candidate_tail,
                confirmation_baseline_tail,
                relative_degradation=(
                    args.maximum_confirmation_tail_relative_degradation
                ),
            )
        )

        final_state = state_to_cpu(model)
        checkpoint = {
            "schema": "generic-quintic-h8-k16-residual-arm-v1",
            "architecture": args.architecture,
            "training_stage": args.training_stage,
            "state_dict": final_state,
            "baseline_h8": h8,
            "residual_h4": h4,
            "normalization8": normalization8,
            "normalization4": normalization4,
            "baseline_power": 2,
            "residual_power": 4,
            "direct_bond_dimension": args.direct_bond_dimension,
            "tree_leaf_edge_dimension": args.tree_leaf_edge_dimension,
            "tree_internal_edge_dimension": (
                args.tree_internal_edge_dimension
            ),
            "rank_activation_scale": args.rank_activation_scale,
            "precision": args.precision,
            "seed": args.seed,
            "teacher_runtime_dependency": False,
            "native_objective_only_after_initialization": True,
        }
        checkpoint_path = output_dir / "checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)
        runtime = time.perf_counter() - started
        report = {
            "schema": "generic-quintic-h8-k16-residual-arm-v1",
            "configuration": {
                **vars(args),
                "baseline_h8": str(h8_path),
                "residual_h4": str(h4_path),
                "output_dir": str(output_dir),
            },
            "model": {
                "architecture": args.architecture,
                "target_degree": 16,
                "total_real_parameter_count": total_parameter_count,
                "active_real_parameter_count": active_parameter_count,
                "trainable_parameter_names": trainable_names,
                "residual_scale": scale,
                "initial_effective_residual_weight": (
                    args.effective_residual_weight
                ),
                "final_residual_weight": float(
                    model.residual_weight.detach().cpu()
                ),
                "final_effective_residual_ratio": float(
                    torch.square(model.residual_amplitude).detach().cpu()
                ),
                "training_stage": args.training_stage,
                "all_parameters_jointly_trainable": (
                    args.training_stage == "all"
                ),
                "amplitude_shape_warmup": {
                    "steps": args.amplitude_warmup_steps,
                    "preserved_residual_tensor_norms": (
                        args.preserve_residual_tensor_norms_during_warmup
                    ),
                },
                "initial_gradient_audit": initial_gradient_audit,
                "change_audit": changes,
            },
            "data_counts": {
                "fit": args.train_size,
                "selection": args.selection_size,
                "confirmation": args.confirmation_size,
            },
            "selection": {
                "exact_h8_power_baseline": baseline_selection,
                "activated_initial": activated_selection,
                "best_step": best_step,
                "best_candidate": best_selection,
                "paired_improvement": paired_improvement(
                    baseline_ratio,
                    best_selection_ratio,
                    selection_weights_numpy,
                ),
            },
            "confirmation": {
                "baseline": baseline_confirmation,
                "candidate": candidate_confirmation,
                "paired_improvement": confirmation_paired,
                "accepted": confirmation_accepted,
            },
            "training": {
                "completed_steps": completed_steps,
                "stopped_early": stopped_early,
                "runtime_seconds": runtime,
                "training_loop_seconds": training_loop_seconds,
                "training_compute_seconds": training_compute_seconds,
                "selection_evaluation_seconds": (
                    selection_evaluation_seconds
                ),
                "updates_per_training_compute_second": (
                    completed_steps
                    / max(training_compute_seconds, np.finfo(float).tiny)
                ),
                "training_points_per_compute_second": (
                    completed_steps
                    * min(args.training_batch_size, args.train_size)
                    / max(training_compute_seconds, np.finfo(float).tiny)
                ),
            },
            "sources": {
                "baseline_h8_sha256": sha256_file(h8_path),
                "residual_h4_sha256": sha256_file(h4_path),
                "baseline_data_reuse_audit": data_reuse_audit,
            },
            "artifacts": {
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "state_sha256": model_hash(final_state),
            },
            "interpretation": {
                "teacher_use": (
                    "H8 and H4 define the exact starting coordinates only; "
                    "all updates use native normalized E2"
                ),
                "blind_status": "not opened",
            },
        }
        write_json(output_dir / "report.json", report)
        write_json(
            status_path,
            {
                "state": "completed",
                "best_step": best_step,
                "confirmation_accepted": confirmation_accepted,
                "runtime_seconds": runtime,
            },
        )
        print(json.dumps(report, indent=2, default=str), flush=True)
    except Exception as error:
        write_json(
            status_path,
            {
                "state": "failed",
                "error": repr(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    main()
