#!/usr/bin/env python3
"""Train a nested O(1)+O(2) positive TN on common Fermat-quintic points."""

from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import (  # noqa: E402
    PositiveTensorNetworkDirectSumMetric,
    PositiveTensorNetworkMetric,
    TensorNetworkMoments,
    anchored_orthonormal_physical_dictionary,
    canonical_matrix_unit_dictionary,
    positive_tensor_network_from_artifact_payload,
    rectangular_reference_factor,
)
from gcicy_metric.pipeline.risk import (  # noqa: E402
    smooth_upper_log_ratio_excess_torch,
    weighted_cvar_torch,
)
from scripts.train_quintic_positive_tensor_network_same_points import (  # noqa: E402
    evaluate_raw,
    exact_fs_reference_model,
    fixed_kappa_statistics,
    fs_reproduction_check,
    ratio_statistics,
    sha256_file,
    source_features,
    tail_point_losses_torch,
    tensor_split,
    weighted_log_mean_exp,
    write_json,
)


SCHEMA = "quintic-positive-tensor-network-direct-sum-v1"
TOTAL_DEGREE = 30
BASE_SOURCE_DEGREE = 1
BASE_SITE_COUNT = 30
DEFAULT_RESIDUAL_SOURCE_DEGREE = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--blind-reference-run-dir", type=Path)
    parser.add_argument("--pullbacks-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--initial-model", type=Path)
    parser.add_argument(
        "--residual-source-degree",
        type=int,
        choices=(1, 2),
        default=DEFAULT_RESIDUAL_SOURCE_DEGREE,
    )
    parser.add_argument(
        "--residual-output-dimension",
        type=int,
        default=0,
        help=(
            "Residual purification output dimension; zero uses the source-section "
            "dimension. Use 25 for the exact rectangular O(2) local-map space."
        ),
    )
    parser.add_argument("--residual-bond-dimension", type=int, default=4)
    parser.add_argument("--residual-dictionary-rank", type=int, default=64)
    parser.add_argument(
        "--freeze-residual-dictionary",
        action="store_true",
        help="Use a fixed orthonormal local-operator basis.",
    )
    parser.add_argument("--expected-parameter-count", type=int, default=148098)
    parser.add_argument("--initial-residual-fraction", type=float, default=1.0e-3)
    parser.add_argument("--calibration-points", type=int, default=8192)
    parser.add_argument("--freeze-base", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--test-batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--residual-initialization-noise", type=float, default=1.0e-2)
    parser.add_argument("--log-energy-loss-weight", type=float, default=1.0)
    parser.add_argument("--ma-loss-weight", type=float, default=1.0)
    parser.add_argument("--tail-loss-weight", type=float, default=0.1)
    parser.add_argument("--tail-fraction", type=float, default=0.02)
    parser.add_argument(
        "--tail-loss-kind",
        choices=("upper_threshold", "absolute_ratio"),
        default="upper_threshold",
    )
    parser.add_argument("--tail-ratio-threshold", type=float, default=1.5)
    parser.add_argument("--tail-smooth-temperature", type=float, default=0.05)
    parser.add_argument("--checkpoint-log-energy-weight", type=float, default=1.0)
    parser.add_argument("--checkpoint-ma-weight", type=float, default=1.0)
    parser.add_argument("--checkpoint-tail-weight", type=float, default=0.1)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--validation-limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument(
        "--skip-blind-audit",
        action="store_true",
        help="skip blind evaluation for intermediate continuation stages",
    )
    parser.add_argument("--dictionary-seed", type=int, default=202607211)
    parser.add_argument("--torch-seed", type=int, default=202607212)
    parser.add_argument("--orthonormalize-every", type=int, default=1)
    parser.add_argument(
        "--training-logdet-method",
        choices=("cholesky", "eigvalsh"),
        default="cholesky",
    )
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
    if (args.base_model is None) == (args.initial_model is None):
        raise ValueError("provide exactly one of --base-model and --initial-model")
    positive = (
        args.residual_bond_dimension,
        args.residual_dictionary_rank,
        args.batch_size,
        args.eval_every,
        args.eval_batch_size,
        args.test_batch_size,
        args.calibration_points,
        args.orthonormalize_every,
    )
    if min(positive) <= 0 or args.epochs < 0:
        raise ValueError("invalid integer training argument")
    residual_sections = math.comb(4 + args.residual_source_degree, 4)
    residual_output = args.residual_output_dimension or residual_sections
    if TOTAL_DEGREE % args.residual_source_degree:
        raise ValueError("residual source degree must divide the total degree")
    if residual_output < residual_sections:
        raise ValueError("residual output dimension cannot be smaller than the source")
    if args.residual_dictionary_rank > residual_output * residual_sections:
        raise ValueError("residual dictionary rank exceeds the local operator space")
    if not 0 < args.initial_residual_fraction < 1:
        raise ValueError("initial residual fraction must lie in (0,1)")
    if args.learning_rate <= 0 or args.gradient_clip_norm <= 0:
        raise ValueError("learning rate and gradient clipping must be positive")
    if args.residual_initialization_noise < 0:
        raise ValueError("residual initialization noise must be non-negative")
    if not 0 < args.tail_fraction <= 1:
        raise ValueError("tail fraction must lie in (0,1]")


def dual_tensor_split(
    x_values: np.ndarray,
    pullbacks: np.ndarray,
    labels: np.ndarray,
    *,
    complex_dtype: torch.dtype,
    real_dtype: torch.dtype,
    device: torch.device,
    residual_source_degree: int = DEFAULT_RESIDUAL_SOURCE_DEGREE,
) -> dict[str, Any]:
    primary = tensor_split(
        x_values,
        pullbacks,
        labels,
        source_degree=BASE_SOURCE_DEGREE,
        complex_dtype=complex_dtype,
        real_dtype=real_dtype,
        device=device,
    )
    residual_values, residual_derivatives = source_features(
        x_values,
        pullbacks,
        source_degree=residual_source_degree,
        complex_dtype=complex_dtype,
        device=device,
    )
    primary["residual_values"] = residual_values
    primary["residual_derivatives"] = residual_derivatives
    return primary


def direct_sum_inputs(
    dataset: dict[str, Any],
    start: int | torch.Tensor,
    stop: int | None = None,
):
    index = slice(start, stop) if isinstance(start, int) else start
    return (
        (dataset["values"][index], dataset["derivatives"][index]),
        (
            dataset["residual_values"][index],
            dataset["residual_derivatives"][index],
        ),
    )


def cache_frozen_base_moments(
    model: torch.nn.Module,
    dataset: dict[str, Any],
    *,
    chunk_size: int,
) -> dict[str, torch.Tensor]:
    """Evaluate a frozen base branch once on a fixed point split."""

    rows: dict[str, list[torch.Tensor]] = {
        "norm": [],
        "holomorphic_gradient": [],
        "mixed_hessian": [],
        "correction": [],
    }
    model.branches[0].eval()
    with torch.no_grad():
        for start in range(0, dataset["count"], chunk_size):
            stop = min(start + chunk_size, dataset["count"])
            moments, correction = model.branches[0]._homogeneously_normalized_feature_moments(
                dataset["values"][start:stop],
                dataset["derivatives"][start:stop],
            )
            rows["norm"].append(moments.norm)
            rows["holomorphic_gradient"].append(moments.holomorphic_gradient)
            rows["mixed_hessian"].append(moments.mixed_hessian)
            rows["correction"].append(correction)
    return {name: torch.cat(values, dim=0) for name, values in rows.items()}


def metric_with_frozen_base_cache(
    model: torch.nn.Module,
    dataset: dict[str, Any],
    index: slice | torch.Tensor,
    cache: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Combine cached base moments with a live residual branch exactly."""

    residual_moments, residual_correction = (
        model.branches[1]._homogeneously_normalized_feature_moments(
            dataset["residual_values"][index],
            dataset["residual_derivatives"][index],
        )
    )
    base_correction = cache["correction"][index]
    base_scale = model.branch_weights[0]
    residual_scale = model.branch_weights[1] * torch.exp(
        residual_correction - base_correction
    )
    combined = TensorNetworkMoments(
        norm=(
            base_scale * cache["norm"][index]
            + residual_scale * residual_moments.norm
        ),
        holomorphic_gradient=(
            base_scale * cache["holomorphic_gradient"][index]
            + residual_scale[:, None] * residual_moments.holomorphic_gradient
        ),
        mixed_hessian=(
            base_scale * cache["mixed_hessian"][index]
            + residual_scale[:, None, None] * residual_moments.mixed_hessian
        ),
    )
    return model._metric_from_moments(combined)


def evaluate_raw_direct_sum(
    model: torch.nn.Module,
    dataset: dict[str, Any],
    *,
    chunk_size: int,
    base_cache: dict[str, torch.Tensor] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    raw_rows: list[np.ndarray] = []
    minimum_rows: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, dataset["count"], chunk_size):
            stop = min(start + chunk_size, dataset["count"])
            index = slice(start, stop)
            if base_cache is None:
                _, metric = model.potential_and_metric(
                    direct_sum_inputs(dataset, start, stop)
                )
            else:
                metric = metric_with_frozen_base_cache(
                    model,
                    dataset,
                    index,
                    base_cache,
                )
            eigenvalues = torch.linalg.eigvalsh(metric)
            if not bool(torch.all(torch.isfinite(eigenvalues) & (eigenvalues > 0))):
                raise FloatingPointError("nonpositive direct-sum metric during evaluation")
            raw = torch.sum(torch.log(eigenvalues), dim=1)
            raw -= dataset["log_omega"][start:stop]
            raw_rows.append(raw.detach().cpu().numpy().astype(np.float64))
            minimum_rows.append(
                torch.min(eigenvalues, dim=1).values.detach().cpu().numpy().astype(np.float64)
            )
    return np.concatenate(raw_rows), np.concatenate(minimum_rows)


def branch_fraction_statistics(
    model: torch.nn.Module,
    dataset: dict[str, Any],
    *,
    chunk_size: int,
    base_cache: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    rows: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, dataset["count"], chunk_size):
            stop = min(start + chunk_size, dataset["count"])
            if base_cache is None:
                fractions = model.branch_norm_fractions(
                    direct_sum_inputs(dataset, start, stop)
                )
            else:
                index = slice(start, stop)
                residual_moments, residual_correction = (
                    model.branches[1]._homogeneously_normalized_feature_moments(
                        dataset["residual_values"][index],
                        dataset["residual_derivatives"][index],
                    )
                )
                residual_scale = model.branch_weights[1] * torch.exp(
                    residual_correction - base_cache["correction"][index]
                )
                base_norm = model.branch_weights[0] * base_cache["norm"][index]
                residual_norm = residual_scale * residual_moments.norm
                total_norm = base_norm + residual_norm
                fractions = torch.stack(
                    (base_norm / total_norm, residual_norm / total_norm),
                    dim=1,
                )
            rows.append(fractions.detach().cpu().numpy().astype(np.float64))
    values = np.concatenate(rows, axis=0)
    probabilities = (0.0, 0.5, 0.9, 0.99, 0.999, 1.0)
    return {
        "n_points": int(len(values)),
        "weighted_means": np.average(
            values,
            axis=0,
            weights=dataset["weights_numpy"],
        ).tolist(),
        "unweighted_quantiles": {
            f"branch_{index}": {
                f"q{probability:.4f}": float(quantile)
                for probability, quantile in zip(
                    probabilities,
                    np.quantile(values[:, index], probabilities),
                    strict=True,
                )
            }
            for index in range(values.shape[1])
        },
    }


def audit_blind(
    model: torch.nn.Module,
    *,
    blind_path: Path,
    blind_pullbacks_path: Path,
    output_dir: Path,
    fixed_log_kappa: float,
    args: argparse.Namespace,
    complex_dtype: torch.dtype,
    real_dtype: torch.dtype,
    device: torch.device,
) -> dict[str, Any]:
    blind = np.load(blind_path, allow_pickle=False)
    blind_x = np.asarray(blind["X"], dtype=np.float32)
    blind_weights = np.asarray(blind["weights"], dtype=np.float64)
    blind_omega = np.asarray(blind["omega_squared"], dtype=np.float64)
    blind_pullbacks = np.load(blind_pullbacks_path, mmap_mode="r")
    if args.test_limit:
        blind_x = blind_x[: args.test_limit]
        blind_weights = blind_weights[: args.test_limit]
        blind_omega = blind_omega[: args.test_limit]
        blind_pullbacks = blind_pullbacks[: args.test_limit]

    blind_raw = np.empty(len(blind_x), dtype=np.float64)
    blind_minimum = np.empty(len(blind_x), dtype=np.float64)
    started = time.perf_counter()
    model.eval()
    for start in range(0, len(blind_x), args.test_batch_size):
        stop = min(start + args.test_batch_size, len(blind_x))
        primary_values, primary_derivatives = source_features(
            blind_x[start:stop],
            blind_pullbacks[start:stop],
            source_degree=BASE_SOURCE_DEGREE,
            complex_dtype=complex_dtype,
            device=device,
        )
        residual_values, residual_derivatives = source_features(
            blind_x[start:stop],
            blind_pullbacks[start:stop],
            source_degree=args.residual_source_degree,
            complex_dtype=complex_dtype,
            device=device,
        )
        with torch.no_grad():
            _, metric = model.potential_and_metric(
                (
                    (primary_values, primary_derivatives),
                    (residual_values, residual_derivatives),
                )
            )
            eigenvalues = torch.linalg.eigvalsh(metric)
            if not bool(torch.all(torch.isfinite(eigenvalues) & (eigenvalues > 0))):
                raise FloatingPointError("nonpositive blind direct-sum metric")
            raw = torch.sum(torch.log(eigenvalues), dim=1)
            raw -= torch.log(
                torch.tensor(
                    blind_omega[start:stop],
                    dtype=real_dtype,
                    device=device,
                )
            )
        blind_raw[start:stop] = raw.detach().cpu().numpy().astype(np.float64)
        blind_minimum[start:stop] = (
            torch.min(eigenvalues, dim=1).values.detach().cpu().numpy().astype(np.float64)
        )

    weights_normalized = blind_weights / np.sum(blind_weights)
    normalized, ratio = ratio_statistics(
        blind_raw,
        weights_normalized,
        blind_minimum,
    )
    fixed = fixed_kappa_statistics(
        blind_raw,
        weights_normalized,
        fixed_log_kappa,
        args,
    )
    arrays_path = output_dir / "blind_test_tail_arrays.npz"
    np.savez_compressed(
        arrays_path,
        raw_log_volume_ratio=blind_raw,
        normalized_ratio=ratio,
        min_eigenvalue=blind_minimum,
        weights=blind_weights,
        omega_squared=blind_omega,
    )
    return {
        "n_points": int(len(blind_x)),
        "statistics": {"normalized_volume": normalized, **fixed},
        "arrays_path": arrays_path,
        "seconds": time.perf_counter() - started,
    }


def evaluate_split(
    model: torch.nn.Module,
    dataset: dict[str, Any],
    *,
    fixed_log_kappa: float,
    args: argparse.Namespace,
    base_cache: dict[str, torch.Tensor] | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    raw, minimum = evaluate_raw_direct_sum(
        model,
        dataset,
        chunk_size=args.eval_batch_size,
        base_cache=base_cache,
    )
    normalized, ratio = ratio_statistics(raw, dataset["weights_numpy"], minimum)
    fixed = fixed_kappa_statistics(
        raw,
        dataset["weights_numpy"],
        fixed_log_kappa,
        args,
    )
    return {"normalized_volume": normalized, **fixed}, raw, ratio, minimum


def state_on_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def make_branch_from_nested_state(
    state: dict[str, torch.Tensor],
    configuration: dict[str, Any],
    *,
    branch_index: int,
    complex_dtype: torch.dtype,
    device: torch.device,
    transfer_implementation: str,
) -> torch.nn.Module:
    prefix = f"branches.{branch_index}."
    dictionary_tensor = state.get(prefix + "physical_dictionary")
    if dictionary_tensor is None:
        raise ValueError("nested artifact lacks a branch dictionary")
    dictionary = np.asarray(dictionary_tensor.detach().cpu(), dtype=np.complex128)
    section_count = int(configuration["section_count"])
    output_dimension = int(configuration.get("output_dimension", dictionary.shape[1]))
    if dictionary.shape[1:] != (output_dimension, section_count):
        raise ValueError("nested artifact dictionary shape mismatch")
    model = PositiveTensorNetworkMetric(
        np.eye(section_count, dtype=np.complex128),
        site_count=int(configuration["site_count"]),
        bond_dimension=int(configuration["bond_dimension"]),
        target_normalization=1.0 / (math.pi * TOTAL_DEGREE),
        output_dimension=output_dimension,
        positive_floor=float(configuration["positive_floor"]),
        initialization_noise=0.0,
        physical_dictionary=dictionary,
        trainable_physical_dictionary=bool(
            configuration.get("trainable_physical_dictionary", True)
        ),
        transfer_implementation=transfer_implementation,
        seed=0,
        dtype=complex_dtype,
        device=device,
    )
    return model


def build_model(
    args: argparse.Namespace,
    *,
    complex_dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    if args.initial_model is not None:
        path = args.initial_model.expanduser().resolve()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema") != SCHEMA:
            raise ValueError("initial model is not a nested direct-sum artifact")
        branch_configurations = payload.get("branch_configurations")
        if not isinstance(branch_configurations, list) or len(branch_configurations) != 2:
            raise ValueError("initial model has invalid branch metadata")
        state = payload.get("state_dict")
        if not isinstance(state, dict):
            raise ValueError("initial nested model has no state dictionary")
        branches = [
            make_branch_from_nested_state(
                state,
                configuration,
                branch_index=index,
                complex_dtype=complex_dtype,
                device=device,
                transfer_implementation=args.transfer_implementation,
            )
            for index, configuration in enumerate(branch_configurations)
        ]
        saved_weights = state.get("branch_weights")
        if saved_weights is None:
            raise ValueError("initial nested model has no branch weights")
        model = PositiveTensorNetworkDirectSumMetric(
            branches,
            branch_total_degrees=(TOTAL_DEGREE, TOTAL_DEGREE),
            branch_weights=np.asarray(
                saved_weights.detach().cpu(), dtype=np.float64
            ),
        )
        model.load_state_dict(state, strict=True)
        evidence = {
            "kind": "nested_continuation",
            "path": str(path),
            "sha256": sha256_file(path),
            "branch_configurations": branch_configurations,
        }
        return model, evidence

    path = args.base_model.expanduser().resolve()
    base_payload = torch.load(path, map_location="cpu", weights_only=False)
    required_base = {
        "schema": "quintic-positive-tensor-network-v1",
        "source_degree": BASE_SOURCE_DEGREE,
        "site_count": BASE_SITE_COUNT,
        "total_degree": TOTAL_DEGREE,
        "precision": args.precision,
    }
    observed_base = {key: base_payload.get(key) for key in required_base}
    if observed_base != required_base:
        raise ValueError(
            f"base model metadata mismatch: expected {required_base}, got {observed_base}"
        )
    base = positive_tensor_network_from_artifact_payload(
        np.eye(5, dtype=np.complex128),
        base_payload,
        device=device,
        trainable_physical_dictionary=True,
    )
    base.transfer_implementation = args.transfer_implementation
    residual_section_count = math.comb(4 + args.residual_source_degree, 4)
    residual_output_dimension = (
        args.residual_output_dimension or residual_section_count
    )
    residual_site_count = TOTAL_DEGREE // args.residual_source_degree
    residual_reference = np.eye(residual_section_count, dtype=np.complex128)
    residual_reference_factor = rectangular_reference_factor(
        residual_reference,
        residual_output_dimension,
    )
    residual_local_dimension = (
        residual_output_dimension * residual_section_count
    )
    if (
        args.freeze_residual_dictionary
        and args.residual_dictionary_rank == residual_local_dimension
    ):
        residual_dictionary = canonical_matrix_unit_dictionary(
            residual_output_dimension,
            residual_section_count,
        )
    else:
        residual_dictionary = anchored_orthonormal_physical_dictionary(
            residual_reference_factor,
            args.residual_dictionary_rank,
            seed=args.dictionary_seed,
        )
    residual = PositiveTensorNetworkMetric(
        residual_reference,
        site_count=residual_site_count,
        bond_dimension=args.residual_bond_dimension,
        target_normalization=1.0 / (math.pi * TOTAL_DEGREE),
        output_dimension=residual_output_dimension,
        positive_floor=0.0,
        initialization_noise=args.residual_initialization_noise,
        physical_dictionary=residual_dictionary,
        trainable_physical_dictionary=not args.freeze_residual_dictionary,
        transfer_implementation=args.transfer_implementation,
        seed=args.torch_seed,
        dtype=complex_dtype,
        device=device,
    )
    model = PositiveTensorNetworkDirectSumMetric(
        (base, residual),
        branch_total_degrees=(TOTAL_DEGREE, TOTAL_DEGREE),
        branch_weights=(1.0, 0.0),
    )
    evidence = {
        "kind": "embedded_base_plus_fresh_residual",
        "path": str(path),
        "sha256": sha256_file(path),
        "base_schema": base_payload["schema"],
    }
    return model, evidence


def calibrate_residual_weight(
    model: torch.nn.Module,
    train: dict[str, Any],
    *,
    target_fraction: float,
    calibration_points: int,
) -> dict[str, float]:
    count = min(calibration_points, train["count"])
    model.set_branch_weights_((1.0, 1.0))
    model.eval()
    with torch.no_grad():
        _, _, aligned = model._aligned_branch_moments(
            direct_sum_inputs(train, 0, count)
        )
        ratio = aligned[1].norm / aligned[0].norm
    median_ratio = float(torch.median(ratio).detach().cpu())
    if not np.isfinite(median_ratio) or median_ratio <= 0:
        raise FloatingPointError("residual/base norm calibration failed")
    residual_weight = target_fraction / ((1.0 - target_fraction) * median_ratio)
    model.set_branch_weights_((1.0, residual_weight))
    with torch.no_grad():
        fractions = model.branch_norm_fractions(direct_sum_inputs(train, 0, count))
    residual_fractions = fractions[:, 1].detach().cpu().numpy()
    return {
        "calibration_points": count,
        "unweighted_median_residual_over_base_norm": median_ratio,
        "target_median_residual_fraction": target_fraction,
        "selected_residual_weight": residual_weight,
        "observed_median_residual_fraction": float(np.median(residual_fractions)),
        "observed_q99_residual_fraction": float(np.quantile(residual_fractions, 0.99)),
        "observed_maximum_residual_fraction": float(np.max(residual_fractions)),
    }


def zero_weight_nesting_check(
    model: torch.nn.Module,
    dataset: dict[str, Any],
    *,
    point_count: int,
    complex_dtype: torch.dtype,
) -> dict[str, float | int]:
    """Certify that disabling the residual branch reproduces the base pointwise."""

    count = min(point_count, dataset["count"])
    model.set_branch_weights_((1.0, 0.0))
    model.eval()
    with torch.no_grad():
        base_potential, base_metric = model.branches[0].potential_and_metric(
            dataset["values"][:count],
            dataset["derivatives"][:count],
        )
        nested_potential, nested_metric = model.potential_and_metric(
            direct_sum_inputs(dataset, 0, count)
        )
    potential_error = float(
        torch.max(torch.abs(nested_potential - base_potential)).detach().cpu()
    )
    metric_error = float(
        torch.max(torch.abs(nested_metric - base_metric)).detach().cpu()
    )
    tolerance = 2.0e-5 if complex_dtype == torch.complex64 else 1.0e-11
    if potential_error > tolerance or metric_error > tolerance:
        raise RuntimeError(
            "zero-weight nesting gate failed: "
            f"potential={potential_error:.3e}, metric={metric_error:.3e}, "
            f"tolerance={tolerance:.3e}"
        )
    return {
        "points": count,
        "maximum_absolute_potential_error": potential_error,
        "maximum_absolute_metric_entry_error": metric_error,
        "tolerance": tolerance,
    }


def training_log_volume(metric: torch.Tensor, method: str) -> torch.Tensor:
    if method == "cholesky":
        factor, info = torch.linalg.cholesky_ex(metric, check_errors=False)
        torch._assert_async(torch.all(info == 0), "nonpositive training metric")
        diagonal = torch.real(torch.diagonal(factor, dim1=-2, dim2=-1))
        return 2.0 * torch.sum(torch.log(diagonal), dim=1)
    eigenvalues = torch.linalg.eigvalsh(metric)
    torch._assert_async(torch.all(eigenvalues > 0), "nonpositive training metric")
    return torch.sum(torch.log(eigenvalues), dim=1)


def branch_configuration(branch: torch.nn.Module, source_degree: int) -> dict[str, Any]:
    return {
        "source_degree": source_degree,
        "section_count": int(branch.section_count),
        "output_dimension": int(branch.output_dimension),
        "site_count": int(branch.site_count),
        "total_degree": source_degree * int(branch.site_count),
        "bond_dimension": int(branch.bond_dimension),
        "dictionary_rank": int(branch.physical_dictionary_rank),
        "trainable_physical_dictionary": bool(
            branch.trainable_physical_dictionary
        ),
        "positive_floor": float(branch.positive_floor),
        "real_parameters": int(branch.trainable_real_parameter_count),
    }


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
        real_dtype = torch.float32 if complex_dtype == torch.complex64 else torch.float64
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
        required = (
            dataset_path,
            basis_path,
            blind_path,
            cymetric_arrays_path,
            source_report_path,
            pullback_report_path,
            train_pullbacks_path,
            validation_pullbacks_path,
            blind_pullbacks_path,
            official_fs_metrics_path,
        )
        for path in required:
            if not path.exists():
                raise FileNotFoundError(path)

        source_hashes = {
            "dataset": sha256_file(dataset_path),
            "basis": sha256_file(basis_path),
            "blind_points": sha256_file(blind_path),
            "cymetric_tail": sha256_file(cymetric_arrays_path),
        }
        pullback_paths = {
            "train": train_pullbacks_path,
            "validation": validation_pullbacks_path,
            "blind": blind_pullbacks_path,
            "validation_official_fs_metrics": official_fs_metrics_path,
        }
        pullback_hashes = {name: sha256_file(path) for name, path in pullback_paths.items()}
        pullback_report = json.loads(pullback_report_path.read_text(encoding="utf-8"))
        if pullback_report.get("source_sha256") != source_hashes:
            raise RuntimeError("pullbacks do not match the fixed source arrays")
        if pullback_report.get("output_sha256") != pullback_hashes:
            raise RuntimeError("pullback hashes do not match their report")

        data = np.load(dataset_path, allow_pickle=False)
        train_x = np.asarray(data["X_train"], dtype=np.float32)
        train_y = np.asarray(data["y_train"], dtype=np.float64)
        validation_x = np.asarray(data["X_val"], dtype=np.float32)
        validation_y = np.asarray(data["y_val"], dtype=np.float64)
        train_pullbacks = np.load(train_pullbacks_path, mmap_mode="r")
        validation_pullbacks = np.load(validation_pullbacks_path, mmap_mode="r")
        if args.train_limit:
            train_x, train_y = train_x[: args.train_limit], train_y[: args.train_limit]
            train_pullbacks = train_pullbacks[: args.train_limit]
        if args.validation_limit:
            validation_x = validation_x[: args.validation_limit]
            validation_y = validation_y[: args.validation_limit]
            validation_pullbacks = validation_pullbacks[: args.validation_limit]

        write_json(status_path, {"state": "running", "phase": "loading_fixed_data"})
        train = dual_tensor_split(
            train_x,
            train_pullbacks,
            train_y,
            complex_dtype=complex_dtype,
            real_dtype=real_dtype,
            device=device,
            residual_source_degree=args.residual_source_degree,
        )
        validation = dual_tensor_split(
            validation_x,
            validation_pullbacks,
            validation_y,
            complex_dtype=complex_dtype,
            real_dtype=real_dtype,
            device=device,
            residual_source_degree=args.residual_source_degree,
        )

        model, initial_model_evidence = build_model(
            args,
            complex_dtype=complex_dtype,
            device=device,
        )
        parameter_count = int(model.trainable_real_parameter_count)
        if args.expected_parameter_count and parameter_count != args.expected_parameter_count:
            raise RuntimeError(
                f"parameter-count gate failed: {parameter_count} != "
                f"{args.expected_parameter_count}"
            )
        model.branches[0].requires_grad_(not args.freeze_base)
        model.branches[1].requires_grad_(True)
        active_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        active_parameter_count = sum(
            parameter.numel() * (2 if parameter.is_complex() else 1)
            for parameter in active_parameters
        )

        reference_model = exact_fs_reference_model(
            np.eye(5, dtype=np.complex128),
            site_count=BASE_SITE_COUNT,
            bond_dimension=1,
            total_degree=TOTAL_DEGREE,
            positive_floor=float(model.branches[0].positive_floor),
            transfer_implementation=args.transfer_implementation,
            seed=args.torch_seed,
            complex_dtype=complex_dtype,
            device=device,
        )
        fs_check = fs_reproduction_check(reference_model, validation, official_fs_metrics_path)
        fs_train_raw, _ = evaluate_raw(
            reference_model,
            train,
            chunk_size=args.eval_batch_size,
        )
        fixed_log_kappa = weighted_log_mean_exp(fs_train_raw, train["weights_numpy"])
        del reference_model, fs_train_raw

        base_caches = None
        base_cache_bytes = 0
        if args.freeze_base:
            write_json(
                status_path,
                {"state": "running", "phase": "caching_frozen_base_moments"},
            )
            base_caches = {
                "train": cache_frozen_base_moments(
                    model,
                    train,
                    chunk_size=args.eval_batch_size,
                ),
                "validation": cache_frozen_base_moments(
                    model,
                    validation,
                    chunk_size=args.eval_batch_size,
                ),
            }
            base_cache_bytes = sum(
                value.numel() * value.element_size()
                for cache in base_caches.values()
                for value in cache.values()
            )

        initialization = None
        nesting_check = None
        history: list[dict[str, Any]] = []
        if args.initial_model is None:
            nesting_check = zero_weight_nesting_check(
                model,
                validation,
                point_count=args.calibration_points,
                complex_dtype=complex_dtype,
            )
            baseline_row, _, _, _ = evaluate_split(
                model,
                validation,
                fixed_log_kappa=fixed_log_kappa,
                args=args,
                base_cache=(
                    None if base_caches is None else base_caches["validation"]
                ),
            )
            history.append({"epoch": -1, "kind": "embedded_base", **baseline_row})
            best_score = float(baseline_row["selection_score"])
            best_epoch = -1
            best_state = state_on_cpu(model)
            initialization = calibrate_residual_weight(
                model,
                train,
                target_fraction=args.initial_residual_fraction,
                calibration_points=args.calibration_points,
            )
        else:
            best_score = float("inf")
            best_epoch = 0
            best_state = state_on_cpu(model)

        initial_row, _, _, _ = evaluate_split(
            model,
            validation,
            fixed_log_kappa=fixed_log_kappa,
            args=args,
            base_cache=(
                None if base_caches is None else base_caches["validation"]
            ),
        )
        initial_score = float(initial_row["selection_score"])
        accepted = np.isfinite(initial_score) and initial_score < best_score
        if accepted:
            best_score = initial_score
            best_epoch = 0
            best_state = state_on_cpu(model)
        history.append({"epoch": 0, "kind": "trainable_initialization", **initial_row})
        print(
            f"epoch=0 score={initial_score:.6e} "
            f"sigma={initial_row['normalized_volume']['sigma_official_formula']:.6e} "
            f"chi={initial_row['normalized_volume']['weighted_rms_abs_residual']:.6e} "
            f"accepted={accepted}",
            flush=True,
        )

        optimizer = torch.optim.Adam(active_parameters, lr=args.learning_rate)
        generator = torch.Generator(device=device)
        generator.manual_seed(args.torch_seed + 1)
        optimizer_seconds = 0.0
        optimizer_steps = 0
        training_started = time.perf_counter()

        for epoch in range(1, args.epochs + 1):
            model.train()
            permutation = torch.randperm(train["count"], generator=generator, device=device)
            epoch_started = time.perf_counter()
            for start in range(0, train["count"], args.batch_size):
                indices = permutation[start : start + args.batch_size]
                batch_weights = train["weights"][indices]
                batch_weights = batch_weights / torch.sum(batch_weights)
                optimizer.zero_grad(set_to_none=True)
                if base_caches is None:
                    _, metric = model.potential_and_metric(
                        direct_sum_inputs(train, indices)
                    )
                else:
                    metric = metric_with_frozen_base_cache(
                        model,
                        train,
                        indices,
                        base_caches["train"],
                    )
                raw = training_log_volume(metric, args.training_logdet_method)
                raw -= train["log_omega"][indices]
                log_ratio = raw - fixed_log_kappa
                ratio = torch.exp(torch.clamp(log_ratio, -20.0, 20.0))
                log_energy = torch.sum(batch_weights * torch.square(log_ratio))
                ma_energy = torch.sum(batch_weights * torch.square(ratio - 1.0))
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
                loss = (
                    args.log_energy_loss_weight * log_energy
                    + args.ma_loss_weight * ma_energy
                    + args.tail_loss_weight * tail_energy
                )
                torch._assert_async(torch.isfinite(loss), "nonfinite training objective")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(active_parameters, args.gradient_clip_norm)
                optimizer.step()
                optimizer_steps += 1
                if optimizer_steps % args.orthonormalize_every == 0:
                    model.orthonormalize_physical_dictionaries_()
            if optimizer_steps % args.orthonormalize_every:
                model.orthonormalize_physical_dictionaries_()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            optimizer_seconds += time.perf_counter() - epoch_started

            if epoch % args.eval_every == 0 or epoch == args.epochs:
                validation_row, _, _, _ = evaluate_split(
                    model,
                    validation,
                    fixed_log_kappa=fixed_log_kappa,
                    args=args,
                    base_cache=(
                        None
                        if base_caches is None
                        else base_caches["validation"]
                    ),
                )
                score = float(validation_row["selection_score"])
                accepted = np.isfinite(score) and score < best_score
                if accepted:
                    best_score = score
                    best_epoch = epoch
                    best_state = state_on_cpu(model)
                row = {"epoch": epoch, "kind": "trained", **validation_row}
                history.append(row)
                write_json(output_dir / "training_history.json", {"rows": history})
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": state_on_cpu(model),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_state_dict": best_state,
                        "best_epoch": best_epoch,
                        "best_score": best_score,
                        "fixed_log_kappa": fixed_log_kappa,
                    },
                    output_dir / "training_checkpoint.pt",
                )
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
                normalized = validation_row["normalized_volume"]
                print(
                    f"epoch={epoch} score={score:.6e} "
                    f"sigma={normalized['sigma_official_formula']:.6e} "
                    f"chi={normalized['weighted_rms_abs_residual']:.6e} "
                    f"rmax={normalized['ratio_weighted_quantiles']['q1.0000']:.6e} "
                    f"accepted={accepted}",
                    flush=True,
                )

        training_seconds = time.perf_counter() - training_started
        model.load_state_dict(best_state, strict=True)
        final_validation, _, _, _ = evaluate_split(
            model,
            validation,
            fixed_log_kappa=fixed_log_kappa,
            args=args,
            base_cache=(
                None if base_caches is None else base_caches["validation"]
            ),
        )

        branch_fractions = branch_fraction_statistics(
            model,
            validation,
            chunk_size=args.eval_batch_size,
            base_cache=(
                None if base_caches is None else base_caches["validation"]
            ),
        )
        blind_result = None
        if not args.skip_blind_audit:
            write_json(status_path, {"state": "running", "phase": "blind_audit"})
            blind_result = audit_blind(
                model,
                blind_path=blind_path,
                blind_pullbacks_path=blind_pullbacks_path,
                output_dir=output_dir,
                fixed_log_kappa=fixed_log_kappa,
                args=args,
                complex_dtype=complex_dtype,
                real_dtype=real_dtype,
                device=device,
            )
        branch_configurations = [
            branch_configuration(model.branches[0], BASE_SOURCE_DEGREE),
            branch_configuration(model.branches[1], args.residual_source_degree),
        ]
        model_path = output_dir / "best_tensor_network_direct_sum.pt"
        model_state = state_on_cpu(model)
        torch.save(
            {
                "schema": SCHEMA,
                "geometry": "Fermat quintic hypersurface X_5 in P^4",
                "state_dict": model_state,
                "total_degree": TOTAL_DEGREE,
                "branch_configurations": branch_configurations,
                "precision": args.precision,
                "transfer_implementation": args.transfer_implementation,
                "fixed_log_kappa": fixed_log_kappa,
                "source_sha256": source_hashes,
                "continuation_initial_model": initial_model_evidence,
            },
            model_path,
        )
        source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
        device_memory = None
        if device.type == "cuda":
            device_memory = {
                "maximum_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "maximum_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
        blind_test = None if blind_result is None else blind_result["statistics"]
        blind_points = 0 if blind_result is None else blind_result["n_points"]
        blind_seconds = 0.0 if blind_result is None else blind_result["seconds"]
        artifacts = {
            "model": str(model_path),
            "model_sha256": sha256_file(model_path),
        }
        if blind_result is not None:
            arrays_path = blind_result["arrays_path"]
            artifacts.update(
                {
                    "blind_arrays": str(arrays_path),
                    "blind_arrays_sha256": sha256_file(arrays_path),
                }
            )
        report = {
            "schema": "quintic-positive-tensor-network-direct-sum-report-v1",
            "scientific_scope": {
                "geometry": "Fermat quintic hypersurface X_5 in P^4",
                "ansatz": (
                    "positive direct sum of degree-matched purified tensor "
                    "networks with source degrees 1 and "
                    f"{args.residual_source_degree}"
                ),
                "kahler_potential": "K=(30*pi)^(-1) log(F_base + lambda*F_residual)",
                "nestedness": "lambda=0 reproduces the supplied O(1) base pointwise",
                "claim_limit": "finite blind testing is not a deterministic sup-norm certificate",
            },
            "configuration": vars(args),
            "architecture": {
                "total_degree": TOTAL_DEGREE,
                "branch_configurations": branch_configurations,
                "branch_weights": [float(value) for value in model.branch_weights.cpu()],
                "total_real_parameter_count": parameter_count,
                "active_real_parameter_count": active_parameter_count,
                "parameter_scaling": "O(k) at fixed branch q and D",
            },
            "common_point_evidence": {
                "source_run_dir": str(source_dir),
                "blind_reference_run_dir": str(blind_source_dir),
                "source_sha256": source_hashes,
                "pullback_sha256": pullback_hashes,
                "source_report_sha256": sha256_file(source_report_path),
                "train_points": train["count"],
                "validation_points": validation["count"],
                "blind_points": blind_points,
                "blind_audit_skipped": bool(args.skip_blind_audit),
                "fs_reproduction": fs_check,
            },
            "training": {
                "best_epoch": best_epoch,
                "best_selection_score": best_score,
                "fixed_log_kappa": fixed_log_kappa,
                "fixed_log_kappa_source": "fubini_study_metric_on_fixed_training_pool",
                "initial_model": initial_model_evidence,
                "residual_weight_calibration": initialization,
                "zero_weight_nesting_check": nesting_check,
                "frozen_base_moment_cache": {
                    "enabled": base_caches is not None,
                    "bytes": base_cache_bytes,
                    "scope": "fixed train and validation points only",
                },
                "final_validation": final_validation,
                "validation_branch_fractions": branch_fractions,
                "history": history,
                "optimizer_steps": optimizer_steps,
            },
            "blind_test": blind_test,
            "comparators": {
                "official_cymetric_network": source_report.get("network"),
                "official_cymetric_blind_test": source_report.get("trained_phi_model"),
            },
            "artifacts": artifacts,
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "device": str(device),
                "device_memory": device_memory,
            },
            "timing_seconds": {
                "optimizer_epochs": optimizer_seconds,
                "training_with_validation_and_checkpoints": training_seconds,
                "blind_audit": blind_seconds,
                "wall_total": time.perf_counter() - started,
            },
        }
        report_path = output_dir / "report.json"
        write_json(report_path, report)
        write_json(
            status_path,
            {
                "state": "complete",
                "phase": "complete",
                "report": str(report_path),
                "report_sha256": sha256_file(report_path),
                "wall_seconds": time.perf_counter() - started,
            },
        )
        evaluation_scope = "blind" if blind_test is not None else "validation"
        final_normalized = (
            blind_test["normalized_volume"]
            if blind_test is not None
            else final_validation["normalized_volume"]
        )
        print(
            f"finished={output_dir} best_epoch={best_epoch} "
            f"evaluation={evaluation_scope} "
            f"sigma={final_normalized['sigma_official_formula']:.9e} "
            f"chi={final_normalized['weighted_rms_abs_residual']:.9e} "
            f"training_seconds={training_seconds:.3f}",
            flush=True,
        )
    except Exception as exc:
        write_json(
            status_path,
            {
                "state": "failed",
                "phase": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "wall_seconds": time.perf_counter() - started,
            },
        )
        raise


if __name__ == "__main__":
    main()
