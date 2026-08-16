#!/usr/bin/env python3
"""Fit a small native-MA coefficient matrix inside a stable Schmidt subspace."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable

import numpy as np
import torch
from torch.func import functional_call, vjp


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import (  # noqa: E402
    positive_tensor_network_from_artifact_payload,
)
from scripts.refine_quintic_hard_symmetry_channel_native_ma import (  # noqa: E402
    evaluate_raw,
    model_summary,
    weighted_log_mean_exp,
)
from scripts.refine_quintic_hard_symmetry_supercore_native_ma import (  # noqa: E402
    tail_nonworse,
)
from scripts.refine_quintic_tn_rank_growth_native_ma_als import (  # noqa: E402
    expanded_model,
    maximum_relative_metric_difference,
    normal_gradient_schmidt_analysis,
    orbit_augment_dataset,
)
from scripts.scan_quintic_tn_all_bonds_native_ma import (  # noqa: E402
    normalize_pair_supercore_scale_,
)
from scripts.train_quintic_full_h_same_points import (  # noqa: E402
    sha256_file,
    write_json,
)
from scripts.train_quintic_positive_tensor_network_same_points import (  # noqa: E402
    fixed_fermat_actions_torch,
    tensor_split,
    training_log_volume,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--pullbacks-dir", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--bond-index", type=int, default=18)
    parser.add_argument("--subspace-rank", type=int, default=8)
    parser.add_argument("--maximum-coefficient-rank", type=int, default=4)
    parser.add_argument("--subspace-fit-train-start", type=int, default=410_000)
    parser.add_argument("--subspace-fit-size", type=int, default=512)
    parser.add_argument(
        "--subspace-selection-validation-start",
        type=int,
        default=30_000,
    )
    parser.add_argument("--subspace-selection-size", type=int, default=512)
    parser.add_argument("--coefficient-fit-train-start", type=int, default=420_000)
    parser.add_argument("--coefficient-fit-size", type=int, default=512)
    parser.add_argument(
        "--coefficient-selection-validation-start",
        type=int,
        default=35_000,
    )
    parser.add_argument("--coefficient-selection-size", type=int, default=512)
    parser.add_argument("--gate-validation-start", type=int, default=36_000)
    parser.add_argument("--gate-size", type=int, default=2048)
    parser.add_argument("--tail-validation-start", type=int, default=40_000)
    parser.add_argument("--tail-size", type=int, default=20_000)
    parser.add_argument("--subspace-group-samples", type=int, default=2)
    parser.add_argument("--optimization-group-samples", type=int, default=4)
    parser.add_argument("--operator-chunk-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--nonlinear-shortlist", type=int, default=6)
    parser.add_argument("--null-draws", type=int, default=10_000)
    parser.add_argument("--null-quantile", type=float, default=0.99)
    parser.add_argument(
        "--ridge-factors",
        type=float,
        nargs="+",
        default=(10.0, 3.0, 1.0, 0.3, 0.1, 0.03, 0.01),
    )
    parser.add_argument(
        "--step-scales",
        type=float,
        nargs="+",
        default=(0.125, 0.25, 0.5, 1.0),
    )
    parser.add_argument("--seed", type=int, default=202607278)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.subspace_rank,
        args.maximum_coefficient_rank,
        args.subspace_fit_size,
        args.subspace_selection_size,
        args.coefficient_fit_size,
        args.coefficient_selection_size,
        args.gate_size,
        args.tail_size,
        args.subspace_group_samples,
        args.optimization_group_samples,
        args.operator_chunk_size,
        args.eval_batch_size,
        args.nonlinear_shortlist,
        args.null_draws,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("rank, sample, batch, and draw counts must be positive")
    starts = (
        args.subspace_fit_train_start,
        args.subspace_selection_validation_start,
        args.coefficient_fit_train_start,
        args.coefficient_selection_validation_start,
        args.gate_validation_start,
        args.tail_validation_start,
        args.bond_index,
    )
    if any(value < 0 for value in starts):
        raise ValueError("dataset starts and bond index cannot be negative")
    if args.maximum_coefficient_rank > args.subspace_rank:
        raise ValueError("coefficient rank cannot exceed discovery subspace rank")
    if not 0 < args.null_quantile < 1:
        raise ValueError("null quantile must lie in (0, 1)")
    if any(not np.isfinite(value) or value <= 0 for value in args.ridge_factors):
        raise ValueError("ridge factors must be finite and positive")
    if any(
        not np.isfinite(value) or not 0 < value <= 1
        for value in args.step_scales
    ):
        raise ValueError("step scales must lie in (0, 1]")


def consensus_principal_basis(
    fit_basis: torch.Tensor,
    selection_basis: torch.Tensor,
    rank: int,
) -> tuple[torch.Tensor, list[float]]:
    """Align two subspaces by principal vectors and return their mean basis."""

    if fit_basis.ndim != 2 or selection_basis.ndim != 2:
        raise ValueError("consensus inputs must be matrices")
    if fit_basis.shape[0] != selection_basis.shape[0]:
        raise ValueError("consensus subspaces have different ambient dimensions")
    stop = min(int(rank), int(fit_basis.shape[1]), int(selection_basis.shape[1]))
    if stop <= 0:
        raise ValueError("consensus rank must be positive")
    fit = fit_basis[:, :stop]
    selection = selection_basis[:, :stop]
    overlap = torch.conj(torch.transpose(fit, 0, 1)) @ selection
    left_rotation, singular, right_adjoint = torch.linalg.svd(
        overlap,
        full_matrices=False,
    )
    right_rotation = torch.conj(torch.transpose(right_adjoint, 0, 1))
    aligned_fit = fit @ left_rotation
    aligned_selection = selection @ right_rotation
    consensus, _ = torch.linalg.qr(
        aligned_fit + aligned_selection,
        mode="reduced",
    )
    return consensus, [
        float(value)
        for value in torch.square(singular).real.detach().cpu()
    ]


def random_subspace_ordered_null(
    ambient_dimension: int,
    rank: int,
    *,
    draws: int,
    quantile: float,
    seed: int,
) -> dict[str, Any]:
    """Monte Carlo null for ordered squared canonical correlations."""

    if ambient_dimension < rank or rank <= 0 or draws <= 0:
        raise ValueError("invalid random-subspace dimensions")
    rng = np.random.default_rng(seed)
    rows = np.empty((draws, rank), dtype=np.float64)
    means = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        raw = (
            rng.standard_normal((ambient_dimension, rank))
            + 1j * rng.standard_normal((ambient_dimension, rank))
        ) / math.sqrt(2.0)
        basis = np.linalg.qr(raw, mode="reduced")[0]
        squared = np.linalg.svd(basis[:rank], compute_uv=False) ** 2
        rows[draw] = squared
        means[draw] = np.mean(squared)
    return {
        "ambient_dimension": int(ambient_dimension),
        "rank": int(rank),
        "draws": int(draws),
        "quantile": float(quantile),
        "expected_mean": float(np.mean(means)),
        "mean_quantile": float(np.quantile(means, quantile)),
        "ordered_quantiles": [
            float(value) for value in np.quantile(rows, quantile, axis=0)
        ],
    }


def significant_prefix(
    observed: list[float],
    thresholds: list[float],
) -> int:
    if len(observed) != len(thresholds):
        raise ValueError("observed and null spectra have different lengths")
    count = 0
    for value, threshold in zip(observed, thresholds, strict=True):
        if value <= threshold:
            break
        count += 1
    return count


def real_coefficients_to_supercore(
    parameter: torch.Tensor,
    baseline_core: torch.Tensor,
    left_basis: torch.Tensor,
    right_basis: torch.Tensor,
) -> torch.Tensor:
    """Map real coordinates of C to Theta0 + U C V-adjoint."""

    left_rank = int(left_basis.shape[1])
    right_rank = int(right_basis.shape[1])
    complex_count = left_rank * right_rank
    if parameter.ndim != 1 or parameter.numel() != 2 * complex_count:
        raise ValueError("reduced coefficient vector has the wrong shape")
    coefficient = torch.complex(
        parameter[:complex_count],
        parameter[complex_count:],
    ).reshape(left_rank, right_rank)
    update_matrix = (
        left_basis
        @ coefficient
        @ torch.conj(torch.transpose(right_basis, 0, 1))
    )
    left_bond, right_bond, first_physical, second_physical = baseline_core.shape
    expected = (
        left_bond * first_physical,
        second_physical * right_bond,
    )
    if update_matrix.shape != expected:
        raise ValueError("reduced update does not match the active supercore")
    update_core = update_matrix.reshape(
        left_bond,
        first_physical,
        second_physical,
        right_bond,
    ).permute(0, 3, 1, 2)
    return baseline_core + update_core


def reduced_raw_function_factory(
    model: torch.nn.Module,
    baseline_core: torch.Tensor,
    left_basis: torch.Tensor,
    right_basis: torch.Tensor,
    dataset: dict[str, Any],
) -> Callable[[slice], Callable[[torch.Tensor], torch.Tensor]]:
    def factory(selection: slice) -> Callable[[torch.Tensor], torch.Tensor]:
        values = dataset["values"][selection]
        derivatives = dataset["derivatives"][selection]
        log_omega = dataset["log_omega"][selection]

        def raw(parameter: torch.Tensor) -> torch.Tensor:
            core = real_coefficients_to_supercore(
                parameter,
                baseline_core,
                left_basis,
                right_basis,
            )
            metric = functional_call(
                model,
                {"two_site_core": core},
                (values, derivatives),
                strict=False,
            )
            return training_log_volume(metric, "cholesky") - log_omega

        return raw

    return factory


def explicit_native_residual_jacobian(
    parameter: torch.Tensor,
    dataset: dict[str, Any],
    raw_factory: Callable[[slice], Callable[[torch.Tensor], torch.Tensor]],
    *,
    chunk_size: int,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Build the exact small native-E2 Jacobian in real C coordinates."""

    selections = tuple(
        slice(start, min(start + chunk_size, dataset["count"]))
        for start in range(0, dataset["count"], chunk_size)
    )
    raw_rows = []
    jacobian_rows = []
    for index, selection in enumerate(selections, start=1):
        function = raw_factory(selection)
        raw, pullback = vjp(function, parameter)
        identity = torch.eye(
            raw.numel(),
            dtype=raw.dtype,
            device=raw.device,
        )
        rows = [pullback(identity[row])[0] for row in range(raw.numel())]
        raw_rows.append(raw.detach())
        jacobian_rows.append(torch.stack(rows).detach())
        if progress is not None:
            progress(index, len(selections))
    raw = torch.cat(raw_rows)
    raw_jacobian = torch.cat(jacobian_rows)
    weights = dataset["weights"]
    log_kappa = weighted_log_mean_exp(raw, weights)
    ratio = torch.exp(raw - log_kappa)
    probability = weights * ratio
    normalization_jacobian = probability @ raw_jacobian
    residual = torch.sqrt(weights) * (ratio - 1.0)
    jacobian = (
        torch.sqrt(weights)[:, None]
        * ratio[:, None]
        * (raw_jacobian - normalization_jacobian[None, :])
    )
    return residual, jacobian, {
        "native_e2": float(torch.dot(residual, residual)),
        "log_kappa": float(log_kappa),
        "jacobian_frobenius_norm": float(torch.linalg.matrix_norm(jacobian)),
    }


def set_reduced_parameter_(
    model: torch.nn.Module,
    parameter: torch.Tensor,
    baseline_core: torch.Tensor,
    left_basis: torch.Tensor,
    right_basis: torch.Tensor,
) -> None:
    core = real_coefficients_to_supercore(
        parameter,
        baseline_core,
        left_basis,
        right_basis,
    )
    with torch.no_grad():
        model.two_site_core.copy_(core)


def normalized_ratio(
    model: torch.nn.Module,
    dataset: dict[str, Any],
    *,
    batch_size: int,
) -> np.ndarray:
    raw, _ = evaluate_raw(model, dataset, batch_size=batch_size)
    weights = np.asarray(dataset["weights_numpy"], dtype=np.float64)
    maximum = float(np.max(raw))
    log_kappa = maximum + math.log(
        float(np.sum(weights * np.exp(raw - maximum)))
    )
    return np.exp(raw - log_kappa)


def paired_improvement(
    baseline_ratio: np.ndarray,
    candidate_ratio: np.ndarray,
    weights: np.ndarray,
    *,
    power: int,
) -> dict[str, float]:
    if power == 1:
        difference = np.abs(baseline_ratio - 1.0) - np.abs(
            candidate_ratio - 1.0
        )
    elif power == 2:
        difference = np.square(baseline_ratio - 1.0) - np.square(
            candidate_ratio - 1.0
        )
    else:
        raise ValueError("paired improvement supports L1 or squared residual")
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / np.sum(weights)
    mean = float(np.sum(weights * difference))
    effective_count = float(1.0 / np.sum(np.square(weights)))
    variance = float(np.sum(weights * np.square(difference - mean)))
    if effective_count > 1:
        variance *= effective_count / (effective_count - 1.0)
    standard_error = math.sqrt(max(variance, 0.0) / effective_count)
    return {
        "mean": mean,
        "standard_error": standard_error,
        "ci95_lower": mean - 1.96 * standard_error,
        "ci95_upper": mean + 1.96 * standard_error,
        "effective_count": effective_count,
    }


def strict_gate(
    candidate: dict[str, dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    paired: dict[str, dict[str, dict[str, float]]],
) -> bool:
    return bool(
        candidate["base"]["e2"] < baseline["base"]["e2"]
        and candidate["objective"]["e2"] < baseline["objective"]["e2"]
        and candidate["base"]["sigma"] <= baseline["base"]["sigma"]
        and candidate["objective"]["sigma"] <= baseline["objective"]["sigma"]
        and candidate["base"]["minimum_metric_eigenvalue"] > 0
        and candidate["objective"]["minimum_metric_eigenvalue"] > 0
        and paired["base"]["e2"]["ci95_lower"] > 0
        and paired["objective"]["e2"]["ci95_lower"] > 0
        and tail_nonworse(candidate["base"], baseline["base"])
        and tail_nonworse(candidate["objective"], baseline["objective"])
    )


def release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    validate_args(args)
    started = time.perf_counter()
    output_model = args.output_model.expanduser().resolve()
    output_report = args.output_report.expanduser().resolve()
    status_path = output_report.with_suffix(output_report.suffix + ".status.json")
    indices_path = output_report.with_suffix(output_report.suffix + ".indices.npz")
    for path in (output_model, output_report, status_path, indices_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite artifact: {path}")
    output_report.parent.mkdir(parents=True, exist_ok=True)
    write_json(status_path, {"state": "running", "phase": "loading"})

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    complex_dtype = torch.complex128
    real_dtype = torch.float64

    model_path = args.model.expanduser().resolve()
    source_dir = args.source_run_dir.expanduser().resolve()
    pullbacks_dir = args.pullbacks_dir.expanduser().resolve()
    dataset_path = source_dir / "training_data" / "dataset.npz"
    train_pullbacks_path = pullbacks_dir / "train_pullbacks.npy"
    validation_pullbacks_path = pullbacks_dir / "validation_pullbacks.npy"
    for path in (
        model_path,
        dataset_path,
        train_pullbacks_path,
        validation_pullbacks_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "quintic-positive-tensor-network-v1":
        raise ValueError("stable Schmidt fitting requires a quintic TN artifact")
    if payload.get("architecture") != "shared_local_dictionary":
        raise ValueError("stable Schmidt fitting requires a shared dictionary")
    source_degree = int(payload.get("source_degree", 1))
    if source_degree != 1:
        raise ValueError("Fermat orbit augmentation requires an O(1) source")
    source_bond_dimension = int(payload["bond_dimension"])
    reference_h = np.asarray(
        payload["state_dict"]["reference_h"].detach().cpu(),
        dtype=np.complex128,
    )
    source_model = positive_tensor_network_from_artifact_payload(
        reference_h,
        payload,
        device=device,
        trainable_physical_dictionary=False,
    ).to(device=device, dtype=complex_dtype)
    source_model.requires_grad_(False).eval()
    if args.bond_index + 1 >= source_model.site_count:
        raise ValueError("selected bond lies outside the chain")

    data = np.load(dataset_path, allow_pickle=False)
    train_pullbacks = np.load(train_pullbacks_path, mmap_mode="r")
    validation_pullbacks = np.load(validation_pullbacks_path, mmap_mode="r")

    windows = {
        "subspace_fit": (
            "train",
            args.subspace_fit_train_start,
            args.subspace_fit_size,
        ),
        "subspace_selection": (
            "validation",
            args.subspace_selection_validation_start,
            args.subspace_selection_size,
        ),
        "coefficient_fit": (
            "train",
            args.coefficient_fit_train_start,
            args.coefficient_fit_size,
        ),
        "coefficient_selection": (
            "validation",
            args.coefficient_selection_validation_start,
            args.coefficient_selection_size,
        ),
        "gate": ("validation", args.gate_validation_start, args.gate_size),
        "tail": ("validation", args.tail_validation_start, args.tail_size),
    }
    indices: dict[str, np.ndarray] = {}
    for name, (domain, start, count) in windows.items():
        available = len(data["X_train"] if domain == "train" else data["X_val"])
        stop = start + count
        if stop > available:
            raise ValueError(f"{name} window exceeds {domain} data")
        indices[name] = np.arange(start, stop, dtype=np.int64)
    np.savez_compressed(indices_path, **indices)

    def make_split(name: str) -> dict[str, Any]:
        domain = windows[name][0]
        selected = indices[name]
        if domain == "train":
            x = data["X_train"]
            y = data["y_train"]
            pullbacks = train_pullbacks
        else:
            x = data["X_val"]
            y = data["y_val"]
            pullbacks = validation_pullbacks
        return tensor_split(
            np.asarray(x[selected], dtype=np.float32),
            np.asarray(pullbacks[selected]),
            np.asarray(y[selected], dtype=np.float64),
            source_degree=source_degree,
            complex_dtype=complex_dtype,
            real_dtype=real_dtype,
            device=device,
        )

    canonical = copy.deepcopy(source_model)
    canonical.mixed_canonicalize_coefficient_pair_(args.bond_index)
    pair_scale = normalize_pair_supercore_scale_(canonical, args.bond_index)

    write_json(status_path, {"state": "running", "phase": "subspace_discovery"})
    subspace_fit = make_split("subspace_fit")
    subspace_selection = make_split("subspace_selection")
    subspace_generator = torch.Generator(device=device)
    subspace_generator.manual_seed(args.seed + 1009)
    subspace_actions = fixed_fermat_actions_torch(
        args.subspace_group_samples,
        generator=subspace_generator,
        complex_dtype=complex_dtype,
        device=device,
    )
    subspace_fit_objective = orbit_augment_dataset(
        subspace_fit,
        subspace_actions,
    )
    subspace_selection_objective = orbit_augment_dataset(
        subspace_selection,
        subspace_actions,
    )
    left_active = int(canonical.coefficient_cores[args.bond_index].shape[0])
    right_active = int(
        canonical.coefficient_cores[args.bond_index + 1].shape[1]
    )
    fit_analysis = normal_gradient_schmidt_analysis(
        canonical,
        subspace_fit_objective,
        bond_index=args.bond_index,
        source_bond_dimension=source_bond_dimension,
        left_active=left_active,
        right_active=right_active,
        chunk_size=args.operator_chunk_size,
    )
    selection_analysis = normal_gradient_schmidt_analysis(
        canonical,
        subspace_selection_objective,
        bond_index=args.bond_index,
        source_bond_dimension=source_bond_dimension,
        left_active=left_active,
        right_active=right_active,
        chunk_size=args.operator_chunk_size,
    )
    fit_left = fit_analysis["left_singular_vectors"][:, : args.subspace_rank]
    selection_left = selection_analysis["left_singular_vectors"][
        :, : args.subspace_rank
    ]
    fit_right = torch.conj(
        torch.transpose(
            fit_analysis["right_adjoint"][: args.subspace_rank],
            0,
            1,
        )
    )
    selection_right = torch.conj(
        torch.transpose(
            selection_analysis["right_adjoint"][: args.subspace_rank],
            0,
            1,
        )
    )
    consensus_left, left_spectrum = consensus_principal_basis(
        fit_left,
        selection_left,
        args.subspace_rank,
    )
    consensus_right, right_spectrum = consensus_principal_basis(
        fit_right,
        selection_right,
        args.subspace_rank,
    )
    left_null = random_subspace_ordered_null(
        int(consensus_left.shape[0]),
        args.subspace_rank,
        draws=args.null_draws,
        quantile=args.null_quantile,
        seed=args.seed + 4001,
    )
    right_null = random_subspace_ordered_null(
        int(consensus_right.shape[0]),
        args.subspace_rank,
        draws=args.null_draws,
        quantile=args.null_quantile,
        seed=args.seed + 4002,
    )
    significant_left = significant_prefix(
        left_spectrum,
        left_null["ordered_quantiles"],
    )
    significant_right = significant_prefix(
        right_spectrum,
        right_null["ordered_quantiles"],
    )
    coefficient_rank = min(
        args.maximum_coefficient_rank,
        significant_left,
        significant_right,
    )
    if coefficient_rank <= 0:
        raise RuntimeError("no stable Schmidt pair exceeds the random null")
    left_basis = consensus_left[:, :coefficient_rank].detach()
    right_basis = consensus_right[:, :coefficient_rank].detach()
    del (
        fit_analysis,
        selection_analysis,
        subspace_fit,
        subspace_selection,
        subspace_fit_objective,
        subspace_selection_objective,
    )
    release_cuda()

    optimization_model = copy.deepcopy(canonical)
    baseline_core = (
        optimization_model.activate_two_site_coefficient_core_(args.bond_index)
        .detach()
        .clone()
    )
    optimization_model.two_site_core.requires_grad_(False)
    optimization_model.requires_grad_(False).eval()
    coefficient_count = coefficient_rank**2
    theta = torch.zeros(
        2 * coefficient_count,
        dtype=real_dtype,
        device=device,
    )

    coefficient_generator = torch.Generator(device=device)
    coefficient_generator.manual_seed(args.seed + 2009)
    coefficient_actions = fixed_fermat_actions_torch(
        args.optimization_group_samples,
        generator=coefficient_generator,
        complex_dtype=complex_dtype,
        device=device,
    )
    coefficient_fit = make_split("coefficient_fit")
    coefficient_selection = make_split("coefficient_selection")
    coefficient_fit_objective = orbit_augment_dataset(
        coefficient_fit,
        coefficient_actions,
    )
    coefficient_selection_objective = orbit_augment_dataset(
        coefficient_selection,
        coefficient_actions,
    )

    def jacobian_progress(label: str) -> Callable[[int, int], None]:
        def progress(index: int, count: int) -> None:
            if index == 1 or index == count or index % max(1, count // 20) == 0:
                write_json(
                    status_path,
                    {
                        "state": "running",
                        "phase": f"coefficient_{label}_jacobian",
                        "iteration": index,
                        "count": count,
                    },
                )
                if not args.quiet:
                    print(
                        f"{label} jacobian={index}/{count}",
                        flush=True,
                    )

        return progress

    fit_factory = reduced_raw_function_factory(
        optimization_model,
        baseline_core,
        left_basis,
        right_basis,
        coefficient_fit_objective,
    )
    selection_factory = reduced_raw_function_factory(
        optimization_model,
        baseline_core,
        left_basis,
        right_basis,
        coefficient_selection_objective,
    )
    fit_residual, fit_jacobian, fit_linearization = (
        explicit_native_residual_jacobian(
            theta,
            coefficient_fit_objective,
            fit_factory,
            chunk_size=args.operator_chunk_size,
            progress=jacobian_progress("fit"),
        )
    )
    selection_residual, selection_jacobian, selection_linearization = (
        explicit_native_residual_jacobian(
            theta,
            coefficient_selection_objective,
            selection_factory,
            chunk_size=args.operator_chunk_size,
            progress=jacobian_progress("selection"),
        )
    )
    curvature = torch.transpose(fit_jacobian, 0, 1) @ fit_jacobian
    gradient = torch.transpose(fit_jacobian, 0, 1) @ fit_residual
    curvature_scale = float(
        torch.trace(curvature) / max(1, curvature.shape[0])
    )
    if not np.isfinite(curvature_scale) or curvature_scale <= 0:
        raise FloatingPointError("reduced GN curvature has no finite scale")
    identity = torch.eye(
        curvature.shape[0],
        dtype=curvature.dtype,
        device=curvature.device,
    )
    rows: list[dict[str, Any]] = []
    deltas: list[torch.Tensor] = []
    fit_e2 = float(torch.dot(fit_residual, fit_residual))
    selection_e2 = float(torch.dot(selection_residual, selection_residual))
    for ridge_factor in args.ridge_factors:
        ridge = float(ridge_factor * curvature_scale)
        delta_base = torch.linalg.solve(
            curvature + ridge * identity,
            -gradient,
        )
        for step_scale in args.step_scales:
            delta = float(step_scale) * delta_base
            predicted_fit = float(
                torch.dot(
                    fit_residual + fit_jacobian @ delta,
                    fit_residual + fit_jacobian @ delta,
                )
            )
            predicted_selection = float(
                torch.dot(
                    selection_residual + selection_jacobian @ delta,
                    selection_residual + selection_jacobian @ delta,
                )
            )
            rows.append(
                {
                    "ridge_factor": float(ridge_factor),
                    "ridge": ridge,
                    "step_scale": float(step_scale),
                    "parameter_rms": float(torch.sqrt(torch.mean(delta**2))),
                    "predicted_fit_e2": predicted_fit,
                    "predicted_selection_e2": predicted_selection,
                    "predicted_fit_capture": 1.0 - predicted_fit / fit_e2,
                    "predicted_selection_capture": (
                        1.0 - predicted_selection / selection_e2
                    ),
                    "shortlisted": False,
                    "actual": None,
                    "eligible": False,
                }
            )
            deltas.append(delta.detach().clone())
    predicted_eligible = [
        index
        for index, row in enumerate(rows)
        if row["predicted_fit_capture"] > 0
        and row["predicted_selection_capture"] > 0
    ]
    shortlisted = sorted(
        predicted_eligible,
        key=lambda index: (
            rows[index]["predicted_selection_e2"],
            rows[index]["predicted_fit_e2"],
            index,
        ),
    )[: args.nonlinear_shortlist]
    for index in shortlisted:
        rows[index]["shortlisted"] = True

    coefficient_baseline = {
        "base": model_summary(
            optimization_model,
            coefficient_selection,
            batch_size=args.eval_batch_size,
        ),
        "objective": model_summary(
            optimization_model,
            coefficient_selection_objective,
            batch_size=args.eval_batch_size,
        ),
    }
    for index in shortlisted:
        set_reduced_parameter_(
            optimization_model,
            deltas[index],
            baseline_core,
            left_basis,
            right_basis,
        )
        actual = {
            "base": model_summary(
                optimization_model,
                coefficient_selection,
                batch_size=args.eval_batch_size,
            ),
            "objective": model_summary(
                optimization_model,
                coefficient_selection_objective,
                batch_size=args.eval_batch_size,
            ),
        }
        rows[index]["actual"] = actual
        rows[index]["eligible"] = bool(
            actual["base"]["e2"] < coefficient_baseline["base"]["e2"]
            and actual["objective"]["e2"]
            < coefficient_baseline["objective"]["e2"]
            and actual["base"]["sigma"] <= coefficient_baseline["base"]["sigma"]
            and actual["objective"]["sigma"]
            <= coefficient_baseline["objective"]["sigma"]
            and actual["base"]["minimum_metric_eigenvalue"] > 0
            and actual["objective"]["minimum_metric_eigenvalue"] > 0
        )
        set_reduced_parameter_(
            optimization_model,
            theta,
            baseline_core,
            left_basis,
            right_basis,
        )
    eligible = [index for index, row in enumerate(rows) if row["eligible"]]
    selected_index = (
        min(
            eligible,
            key=lambda index: (
                rows[index]["actual"]["objective"]["e2"],
                rows[index]["actual"]["base"]["sigma"],
                index,
            ),
        )
        if eligible
        else None
    )
    selected_parameter = (
        deltas[selected_index] if selected_index is not None else None
    )
    del (
        fit_residual,
        fit_jacobian,
        selection_residual,
        selection_jacobian,
        curvature,
        gradient,
        coefficient_fit,
        coefficient_selection,
        coefficient_fit_objective,
        coefficient_selection_objective,
    )
    release_cuda()

    gate_report = None
    accepted_gate = False
    tail_report = None
    accepted_tail = False
    if selected_parameter is not None:
        write_json(status_path, {"state": "running", "phase": "independent_gate"})
        gate = make_split("gate")
        gate_generator = torch.Generator(device=device)
        gate_generator.manual_seed(args.seed + 3009)
        gate_actions = fixed_fermat_actions_torch(
            args.optimization_group_samples,
            generator=gate_generator,
            complex_dtype=complex_dtype,
            device=device,
        )
        gate_objective = orbit_augment_dataset(gate, gate_actions)
        gate_baseline = {
            "base": model_summary(
                optimization_model,
                gate,
                batch_size=args.eval_batch_size,
            ),
            "objective": model_summary(
                optimization_model,
                gate_objective,
                batch_size=args.eval_batch_size,
            ),
        }
        baseline_ratios = {
            "base": normalized_ratio(
                optimization_model,
                gate,
                batch_size=args.eval_batch_size,
            ),
            "objective": normalized_ratio(
                optimization_model,
                gate_objective,
                batch_size=args.eval_batch_size,
            ),
        }
        set_reduced_parameter_(
            optimization_model,
            selected_parameter,
            baseline_core,
            left_basis,
            right_basis,
        )
        gate_candidate = {
            "base": model_summary(
                optimization_model,
                gate,
                batch_size=args.eval_batch_size,
            ),
            "objective": model_summary(
                optimization_model,
                gate_objective,
                batch_size=args.eval_batch_size,
            ),
        }
        candidate_ratios = {
            "base": normalized_ratio(
                optimization_model,
                gate,
                batch_size=args.eval_batch_size,
            ),
            "objective": normalized_ratio(
                optimization_model,
                gate_objective,
                batch_size=args.eval_batch_size,
            ),
        }
        paired = {}
        for name, dataset in (("base", gate), ("objective", gate_objective)):
            weights = np.asarray(dataset["weights_numpy"], dtype=np.float64)
            paired[name] = {
                "sigma": paired_improvement(
                    baseline_ratios[name],
                    candidate_ratios[name],
                    weights,
                    power=1,
                ),
                "e2": paired_improvement(
                    baseline_ratios[name],
                    candidate_ratios[name],
                    weights,
                    power=2,
                ),
            }
        gate_report = {
            "baseline": gate_baseline,
            "candidate": gate_candidate,
            "paired_improvement": paired,
        }
        accepted_gate = strict_gate(gate_candidate, gate_baseline, paired)
        del gate, gate_objective, baseline_ratios, candidate_ratios
        release_cuda()

    if accepted_gate:
        write_json(status_path, {"state": "running", "phase": "tail_gate"})
        tail = make_split("tail")
        tail_generator = torch.Generator(device=device)
        tail_generator.manual_seed(args.seed + 3009)
        tail_actions = fixed_fermat_actions_torch(
            args.optimization_group_samples,
            generator=tail_generator,
            complex_dtype=complex_dtype,
            device=device,
        )
        tail_objective = orbit_augment_dataset(tail, tail_actions)
        set_reduced_parameter_(
            optimization_model,
            theta,
            baseline_core,
            left_basis,
            right_basis,
        )
        tail_baseline = {
            "base": model_summary(
                optimization_model,
                tail,
                batch_size=args.eval_batch_size,
            ),
            "objective": model_summary(
                optimization_model,
                tail_objective,
                batch_size=args.eval_batch_size,
            ),
        }
        set_reduced_parameter_(
            optimization_model,
            selected_parameter,
            baseline_core,
            left_basis,
            right_basis,
        )
        tail_candidate = {
            "base": model_summary(
                optimization_model,
                tail,
                batch_size=args.eval_batch_size,
            ),
            "objective": model_summary(
                optimization_model,
                tail_objective,
                batch_size=args.eval_batch_size,
            ),
        }
        accepted_tail = bool(
            tail_candidate["base"]["e2"] < tail_baseline["base"]["e2"]
            and tail_candidate["objective"]["e2"]
            < tail_baseline["objective"]["e2"]
            and tail_candidate["base"]["sigma"] <= tail_baseline["base"]["sigma"]
            and tail_candidate["objective"]["sigma"]
            <= tail_baseline["objective"]["sigma"]
            and tail_candidate["base"]["minimum_metric_eigenvalue"] > 0
            and tail_candidate["objective"]["minimum_metric_eigenvalue"] > 0
            and tail_nonworse(tail_candidate["base"], tail_baseline["base"])
            and tail_nonworse(
                tail_candidate["objective"],
                tail_baseline["objective"],
            )
        )
        tail_report = {
            "baseline": tail_baseline,
            "candidate": tail_candidate,
        }
        del tail, tail_objective
        release_cuda()

    saved_model = None
    commit_report = None
    if accepted_tail and selected_parameter is not None:
        target_bond_dimension = source_bond_dimension + coefficient_rank
        expanded = expanded_model(
            canonical,
            payload,
            target_bond_dimension=target_bond_dimension,
            dtype=complex_dtype,
            device=device,
        )
        expanded_core = expanded.activate_two_site_coefficient_core_(
            args.bond_index
        )
        candidate_core = real_coefficients_to_supercore(
            selected_parameter,
            baseline_core,
            left_basis,
            right_basis,
        )
        with torch.no_grad():
            expanded_core.zero_()
            expanded_core[
                : candidate_core.shape[0],
                : candidate_core.shape[1],
                :,
                :,
            ].copy_(candidate_core)
        left_factor, right_factor, split = (
            expanded.split_two_site_coefficient_core(
                maximum_bond_dimension=target_bond_dimension,
                direction="symmetric",
            )
        )
        active_copy = copy.deepcopy(expanded)
        expanded.commit_two_site_coefficient_split_(left_factor, right_factor)
        verification_indices = indices["gate"][: min(32, args.gate_size)]
        verification = tensor_split(
            np.asarray(data["X_val"][verification_indices], dtype=np.float32),
            np.asarray(validation_pullbacks[verification_indices]),
            np.asarray(data["y_val"][verification_indices], dtype=np.float64),
            source_degree=source_degree,
            complex_dtype=complex_dtype,
            real_dtype=real_dtype,
            device=device,
        )
        relative_change = maximum_relative_metric_difference(
            active_copy,
            expanded,
            verification,
            batch_size=args.eval_batch_size,
        )
        if relative_change > 2.0e-10:
            raise RuntimeError("factorized reduced update changed the metric")
        output_payload = copy.deepcopy(payload)
        output_payload["state_dict"] = {
            key: value.detach().cpu().clone()
            for key, value in expanded.state_dict().items()
        }
        output_payload["precision"] = "complex128"
        output_payload["bond_dimension"] = target_bond_dimension
        output_payload["positive_floor"] = float(expanded.positive_floor)
        output_payload["stable_schmidt_update"] = {
            "schema": "quintic-tn-stable-schmidt-coefficients-v1",
            "source_model": str(model_path),
            "source_model_sha256": sha256_file(model_path),
            "bond_index": args.bond_index,
            "coefficient_rank": coefficient_rank,
            "real_coefficient_parameters": int(theta.numel()),
            "target_bond_dimension": target_bond_dimension,
        }
        temporary = output_model.with_suffix(output_model.suffix + ".tmp")
        torch.save(output_payload, temporary)
        temporary.replace(output_model)
        saved_model = str(output_model)
        commit_report = {
            "split": split,
            "verification_maximum_relative_metric_change": relative_change,
            "verification_tolerance": 2.0e-10,
        }

    report = {
        "schema": "quintic-tn-stable-schmidt-coefficients-v1",
        "scientific_scope": {
            "purpose": (
                "Separate stable Schmidt-subspace discovery from coefficient "
                "and sign fitting in a small native-E2 Gauss-Newton problem."
            ),
            "data_isolation": (
                "A and B determine the common subspace; C-fit determines GN "
                "directions; C-selection selects one candidate; D gates it. "
                "The 20k tail reservoir opens only after D passes."
            ),
            "claim_limit": (
                "A saved model is still a development checkpoint. The final "
                "200k blind set remains unopened."
            ),
        },
        "configuration": {
            **{
                key: value
                for key, value in vars(args).items()
                if not isinstance(value, Path)
            },
            "model": str(model_path),
            "source_run_dir": str(source_dir),
            "pullbacks_dir": str(pullbacks_dir),
            "output_model": str(output_model),
            "output_report": str(output_report),
            "device": str(device),
        },
        "source": {
            "model": str(model_path),
            "model_sha256": sha256_file(model_path),
            "dataset": str(dataset_path),
            "dataset_sha256": sha256_file(dataset_path),
            "train_pullbacks": str(train_pullbacks_path),
            "train_pullbacks_sha256": sha256_file(train_pullbacks_path),
            "validation_pullbacks": str(validation_pullbacks_path),
            "validation_pullbacks_sha256": sha256_file(
                validation_pullbacks_path
            ),
            "indices": str(indices_path),
            "indices_sha256": sha256_file(indices_path),
            "saved_model": saved_model,
            "saved_model_sha256": (
                sha256_file(output_model) if saved_model is not None else None
            ),
        },
        "pair_scale": pair_scale,
        "subspace": {
            "left_principal_cosine_squares": left_spectrum,
            "right_principal_cosine_squares": right_spectrum,
            "left_null": left_null,
            "right_null": right_null,
            "significant_left_prefix": significant_left,
            "significant_right_prefix": significant_right,
            "coefficient_rank": coefficient_rank,
            "real_coefficient_parameters": int(theta.numel()),
        },
        "linearization": {
            "fit": fit_linearization,
            "selection": selection_linearization,
            "curvature_scale": curvature_scale,
        },
        "coefficient_selection": {
            "baseline": coefficient_baseline,
            "rows": rows,
            "selected_index": selected_index,
        },
        "gate": gate_report,
        "accepted_gate": accepted_gate,
        "tail": tail_report,
        "accepted_tail": accepted_tail,
        "commit": commit_report,
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(output_report, report)
    write_json(
        status_path,
        {
            "state": "complete",
            "phase": "complete",
            "accepted_gate": accepted_gate,
            "accepted_tail": accepted_tail,
            "saved_model": saved_model,
            "wall_seconds": report["wall_seconds"],
        },
    )
    print(
        json.dumps(
            {
                "coefficient_rank": coefficient_rank,
                "selected_index": selected_index,
                "accepted_gate": accepted_gate,
                "accepted_tail": accepted_tail,
                "saved_model": saved_model,
                "output_report": str(output_report),
                "wall_seconds": report["wall_seconds"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
