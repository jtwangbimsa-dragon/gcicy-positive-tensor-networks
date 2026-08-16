#!/usr/bin/env python3
"""Grow one bond of a trained quintic TN with a nested native-MA rank branch."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import (  # noqa: E402
    PositiveTensorNetworkMetric,
    positive_tensor_network_from_artifact_payload,
)
from scripts.audit_quintic_tn_metric_tangent_reachability import (  # noqa: E402
    MaskedComplexParameterVectorizer,
)
from scripts.audit_quintic_tn_tangent_reachability import (  # noqa: E402
    ComplexParameterVectorizer,
    lanczos_tridiagonal,
    real_inner,
    ridge_coefficients_from_lanczos,
    vector_norm,
)
from scripts.refine_quintic_hard_symmetry_channel_native_ma import (  # noqa: E402
    MatrixFreeNativeMARatioJacobian,
    model_summary,
)
from scripts.refine_quintic_hard_symmetry_supercore_native_ma import (  # noqa: E402
    capture,
    renormalized_dataset_prefix,
    tail_nonworse,
)
from scripts.train_quintic_full_h_same_points import (  # noqa: E402
    sha256_file,
    write_json,
)
from scripts.train_quintic_positive_tensor_network_same_points import (  # noqa: E402
    fixed_fermat_actions_torch,
    tensor_split,
)
from scripts.train_type11_positive_tensor_network import (  # noqa: E402
    expand_tensor_network_state_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--pullbacks-dir", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--fit-train-start", type=int, default=400_000)
    parser.add_argument("--fit-size", type=int, default=512)
    parser.add_argument("--selection-validation-start", type=int, default=100_000)
    parser.add_argument("--selection-size", type=int, default=512)
    parser.add_argument("--bond-index", type=int, default=14)
    parser.add_argument("--rank-increase", type=int, default=4)
    parser.add_argument("--target-bond-dimension", type=int, default=0)
    parser.add_argument("--group-samples", type=int, default=4)
    parser.add_argument("--operator-chunk-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--screen-fit-size", type=int, default=128)
    parser.add_argument("--screen-selection-size", type=int, default=128)
    parser.add_argument("--nonlinear-shortlist", type=int, default=6)
    parser.add_argument("--lanczos-steps", type=int, default=4)
    parser.add_argument(
        "--ridge-factors",
        type=float,
        nargs="+",
        default=(30.0, 10.0, 3.0, 1.0),
    )
    parser.add_argument(
        "--step-scales",
        type=float,
        nargs="+",
        default=(0.0625, 0.125, 0.25, 0.5, 1.0),
    )
    parser.add_argument(
        "--search-precision",
        choices=("complex64", "complex128"),
        default="complex64",
    )
    parser.add_argument("--seed", type=int, default=202607268)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.fit_size,
        args.selection_size,
        args.rank_increase,
        args.group_samples,
        args.operator_chunk_size,
        args.eval_batch_size,
        args.screen_fit_size,
        args.screen_selection_size,
        args.nonlinear_shortlist,
        args.lanczos_steps,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("all sample, rank, batch, and iteration sizes must be positive")
    if (
        args.fit_train_start < 0
        or args.selection_validation_start < 0
        or args.bond_index < 0
        or args.target_bond_dimension < 0
    ):
        raise ValueError("starts, bond index, and target bond dimension cannot be negative")
    if args.screen_fit_size > args.fit_size:
        raise ValueError("fit screen exceeds the fit set")
    if args.screen_selection_size > args.selection_size:
        raise ValueError("selection screen exceeds the selection set")
    if any(not np.isfinite(value) or value <= 0 for value in args.ridge_factors):
        raise ValueError("ridge factors must be finite and positive")
    if any(
        not np.isfinite(value) or not 0 < value <= 1
        for value in args.step_scales
    ):
        raise ValueError("step scales must lie in (0, 1]")


def expanded_model(
    source: torch.nn.Module,
    payload: dict[str, Any],
    *,
    target_bond_dimension: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.nn.Module:
    """Embed a canonicalized uniform-bond model into a larger zero-padded model."""

    if source.architecture != "shared_local_dictionary":
        raise ValueError("rank growth requires a shared local dictionary")
    dictionary = np.asarray(
        source.physical_dictionary.detach().cpu(),
        dtype=np.complex128,
    )
    reference_h = np.asarray(
        source.reference_h.detach().cpu(),
        dtype=np.complex128,
    )
    result = PositiveTensorNetworkMetric(
        reference_h,
        site_count=int(source.site_count),
        bond_dimension=int(target_bond_dimension),
        target_normalization=float(source.target_normalization),
        positive_floor=float(source.positive_floor),
        initialization_noise=0.0,
        physical_dictionary=dictionary,
        trainable_physical_dictionary=False,
        transfer_implementation=str(
            payload.get("transfer_implementation", "vectorized")
        ),
        seed=0,
        dtype=dtype,
        device=device,
    )
    state = expand_tensor_network_state_dict(
        result.state_dict(),
        source.state_dict(),
    )
    result.load_state_dict(state)
    result.requires_grad_(False).eval()
    return result


def bond_growth_masks(
    model: torch.nn.Module,
    *,
    bond_index: int,
    source_bond_dimension: int,
    rank_increase: int,
) -> tuple[str, torch.Tensor, str, torch.Tensor, int, int]:
    """Return the two nonzero blocks that form a new rank-r path."""

    if bond_index < 0 or bond_index + 1 >= model.site_count:
        raise ValueError("rank-growth bond lies outside the chain")
    left_name = f"coefficient_cores.{bond_index}"
    right_name = f"coefficient_cores.{bond_index + 1}"
    parameters = dict(model.named_parameters())
    left = parameters[left_name]
    right = parameters[right_name]
    first_new = int(source_bond_dimension)
    stop_new = first_new + int(rank_increase)
    if stop_new > left.shape[1] or stop_new > right.shape[0]:
        raise ValueError("preallocated bond has too few new channels")
    left_active = 1 if bond_index == 0 else int(source_bond_dimension)
    right_active = (
        1
        if bond_index + 1 == model.site_count - 1
        else int(source_bond_dimension)
    )
    if left_active > left.shape[0] or right_active > right.shape[1]:
        raise ValueError("source support exceeds an adjacent bond")
    left_mask = torch.zeros_like(left, dtype=torch.bool)
    right_mask = torch.zeros_like(right, dtype=torch.bool)
    left_mask[:left_active, first_new:stop_new, :] = True
    right_mask[first_new:stop_new, :right_active, :] = True
    return (
        left_name,
        left_mask,
        right_name,
        right_mask,
        left_active,
        right_active,
    )


def orbit_augment_dataset(
    dataset: dict[str, Any],
    actions: tuple[tuple[torch.Tensor, torch.Tensor], ...],
) -> dict[str, Any]:
    """Repeat a dataset over fixed Fermat group images with normalized weights."""

    if not actions:
        raise ValueError("orbit augmentation requires at least one action")
    values = []
    derivatives = []
    for permutation, phases in actions:
        permutation = permutation.to(
            dtype=torch.long,
            device=dataset["values"].device,
        )
        phases = phases.to(
            dtype=dataset["values"].dtype,
            device=dataset["values"].device,
        )
        values.append(dataset["values"][:, permutation] * phases[None, :])
        derivatives.append(
            dataset["derivatives"][:, permutation, :]
            * phases[None, :, None]
        )
    action_count = len(actions)
    weights = torch.cat([dataset["weights"]] * action_count) / action_count
    weights_numpy = (
        np.tile(dataset["weights_numpy"], action_count) / action_count
    )
    return {
        **dataset,
        "count": int(dataset["count"] * action_count),
        "values": torch.cat(values),
        "derivatives": torch.cat(derivatives),
        "log_omega": torch.cat([dataset["log_omega"]] * action_count),
        "weights": weights,
        "weights_numpy": weights_numpy,
    }


def maximum_relative_metric_difference(
    left: torch.nn.Module,
    right: torch.nn.Module,
    dataset: dict[str, Any],
    *,
    batch_size: int,
) -> float:
    maximum = 0.0
    left.requires_grad_(False).eval()
    right.requires_grad_(False).eval()
    with torch.no_grad():
        for start in range(0, dataset["count"], batch_size):
            stop = min(start + batch_size, dataset["count"])
            values = dataset["values"][start:stop]
            derivatives = dataset["derivatives"][start:stop]
            left_metric = left(values, derivatives)
            right_metric = right(values, derivatives)
            denominator = torch.clamp(
                torch.linalg.matrix_norm(left_metric, ord="fro"),
                min=torch.finfo(left_metric.real.dtype).tiny,
            )
            relative = (
                torch.linalg.matrix_norm(
                    right_metric - left_metric,
                    ord="fro",
                )
                / denominator
            )
            maximum = max(maximum, float(torch.max(relative)))
    return maximum


def normal_gradient_schmidt_analysis(
    model: torch.nn.Module,
    fit_objective: dict[str, Any],
    *,
    bond_index: int,
    source_bond_dimension: int,
    left_active: int,
    right_active: int,
    chunk_size: int,
) -> dict[str, Any]:
    """Resolve the normal component of dE2/dTheta into Schmidt directions."""

    temporary = copy.deepcopy(model)
    temporary.requires_grad_(False).eval()
    active = temporary.activate_two_site_coefficient_core_(bond_index)
    active.requires_grad_(True)
    vectorizer = ComplexParameterVectorizer.from_module(
        temporary,
        parameter_names=("two_site_core",),
    )
    theta = vectorizer.pack(temporary)
    operator = MatrixFreeNativeMARatioJacobian(
        temporary,
        vectorizer,
        theta,
        fit_objective,
        chunk_size=chunk_size,
    )
    residual = operator.residual()
    gradient = operator.vjp(residual).reshape_as(active)
    physical_left = int(active.shape[2])
    physical_right = int(active.shape[3])
    baseline_block = active[
        :left_active,
        :right_active,
        :,
        :,
    ]
    gradient_block = gradient[
        :left_active,
        :right_active,
        :,
        :,
    ]
    baseline_matrix = baseline_block.permute(0, 2, 3, 1).reshape(
        left_active * physical_left,
        physical_right * right_active,
    )
    gradient_matrix = gradient_block.permute(0, 2, 3, 1).reshape(
        left_active * physical_left,
        physical_right * right_active,
    )
    baseline_left, baseline_singular, baseline_right_adjoint = torch.linalg.svd(
        baseline_matrix,
        full_matrices=False,
    )
    numerical_tolerance = (
        torch.finfo(baseline_singular.dtype).eps
        * max(baseline_matrix.shape)
        * baseline_singular[0]
    )
    numerical_rank = int(
        torch.count_nonzero(baseline_singular > numerical_tolerance)
    )
    inherited_rank = min(
        int(source_bond_dimension),
        numerical_rank,
        baseline_singular.numel(),
    )
    inherited_left = baseline_left[:, :inherited_rank]
    inherited_right = torch.conj(
        torch.transpose(
            baseline_right_adjoint[:inherited_rank],
            0,
            1,
        )
    )
    projected = gradient_matrix
    if inherited_rank:
        projected = projected - inherited_left @ (
            torch.conj(torch.transpose(inherited_left, 0, 1))
            @ projected
        )
        projected = projected - (
            projected @ inherited_right
        ) @ torch.conj(torch.transpose(inherited_right, 0, 1))
    left_singular_vectors, singular, right_adjoint = torch.linalg.svd(
        projected,
        full_matrices=False,
    )
    if not bool(torch.all(torch.isfinite(singular))):
        raise FloatingPointError("normal-gradient SVD contains nonfinite directions")
    squared = torch.square(singular)
    total = torch.sum(squared)
    full_gradient_energy = torch.sum(torch.abs(gradient_matrix) ** 2)

    def energy_fraction(stop: int) -> float:
        return float(
            torch.sum(squared[: min(stop, squared.numel())])
            / torch.clamp(total, min=torch.finfo(total.dtype).tiny)
        )

    diagnostics = {
        "fit_native_e2": float(real_inner(residual, residual)),
        "active_supercore_complex_parameters": int(theta.numel()),
        "baseline_numerical_rank": numerical_rank,
        "inherited_rank_projected_out": inherited_rank,
        "baseline_singular_values": [
            float(value) for value in baseline_singular[:16].detach().cpu()
        ],
        "normal_gradient_singular_values": [
            float(value) for value in singular[:16].detach().cpu()
        ],
        "full_supercore_gradient_energy": float(full_gradient_energy),
        "normal_gradient_energy": float(total),
        "normal_gradient_fraction_of_full": float(
            total
            / torch.clamp(
                full_gradient_energy,
                min=torch.finfo(full_gradient_energy.dtype).tiny,
            )
        ),
        "normal_gradient_score_rank4": float(torch.sum(squared[:4])),
        "normal_gradient_score_rank8": float(torch.sum(squared[:8])),
        "normal_gradient_energy_fraction_rank4": energy_fraction(4),
        "normal_gradient_energy_fraction_rank8": energy_fraction(8),
        "rank5_to_rank8_energy_fraction": max(
            0.0,
            energy_fraction(8) - energy_fraction(4),
        ),
    }
    temporary.use_two_site_core = False
    del operator, temporary
    return {
        "left_singular_vectors": left_singular_vectors.detach(),
        "singular_values": singular.detach(),
        "right_adjoint": right_adjoint.detach(),
        "physical_left": physical_left,
        "physical_right": physical_right,
        "diagnostics": diagnostics,
    }


def left_singular_system(
    matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return all left singular vectors and values through a small Gram matrix."""

    if matrix.ndim != 2 or min(matrix.shape) <= 0:
        raise ValueError("left singular system requires a nonempty matrix")
    gram = matrix @ torch.conj(torch.transpose(matrix, 0, 1))
    gram = 0.5 * (gram + torch.conj(torch.transpose(gram, 0, 1)))
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.arange(
        eigenvalues.numel() - 1,
        -1,
        -1,
        device=eigenvalues.device,
    )
    eigenvalues = torch.index_select(eigenvalues, 0, order)
    eigenvectors = torch.index_select(eigenvectors, 1, order)
    singular_values = torch.sqrt(torch.clamp(eigenvalues.real, min=0.0))
    return eigenvectors, singular_values


def project_three_site_double_normal(
    baseline: torch.Tensor,
    gradient: torch.Tensor,
    *,
    inherited_rank: int,
) -> dict[str, Any]:
    """Project a three-site gradient outside both inherited outer supports.

    Inputs use the grouped shape ``(left*physical, middle_physical,
    right_physical*right)``.  The surviving component requires new Schmidt
    channels across both internal cuts and is therefore invisible to a
    first-order update that grows only one of the two bonds.
    """

    if baseline.ndim != 3 or gradient.shape != baseline.shape:
        raise ValueError("three-site baseline and gradient must be aligned tensors")
    if inherited_rank <= 0:
        raise ValueError("inherited rank must be positive")
    left_dimension, middle_dimension, right_dimension = baseline.shape
    baseline_left, baseline_left_singular = left_singular_system(
        baseline.reshape(left_dimension, middle_dimension * right_dimension)
    )
    baseline_right, baseline_right_singular = left_singular_system(
        baseline.permute(2, 0, 1).reshape(
            right_dimension,
            left_dimension * middle_dimension,
        )
    )

    def numerical_rank(values: torch.Tensor, matrix_shape: tuple[int, int]) -> int:
        tolerance = (
            torch.finfo(values.dtype).eps
            * max(matrix_shape)
            * values[0]
        )
        return int(torch.count_nonzero(values > tolerance))

    left_numerical_rank = numerical_rank(
        baseline_left_singular,
        (left_dimension, middle_dimension * right_dimension),
    )
    right_numerical_rank = numerical_rank(
        baseline_right_singular,
        (right_dimension, left_dimension * middle_dimension),
    )
    left_rank = min(
        int(inherited_rank),
        left_numerical_rank,
        baseline_left.shape[1],
    )
    right_rank = min(
        int(inherited_rank),
        right_numerical_rank,
        baseline_right.shape[1],
    )
    inherited_left = baseline_left[:, :left_rank]
    inherited_right = baseline_right[:, :right_rank]

    projected = gradient
    if left_rank:
        coefficients = torch.einsum(
            "ai,abc->ibc",
            torch.conj(inherited_left),
            projected,
        )
        projected = projected - torch.einsum(
            "ai,ibc->abc",
            inherited_left,
            coefficients,
        )
    if right_rank:
        matrix = projected.reshape(
            left_dimension * middle_dimension,
            right_dimension,
        )
        matrix = matrix - (
            matrix @ inherited_right
        ) @ torch.conj(torch.transpose(inherited_right, 0, 1))
        projected = matrix.reshape(
            left_dimension,
            middle_dimension,
            right_dimension,
        )
    return {
        "projected": projected,
        "inherited_left": inherited_left,
        "inherited_right": inherited_right,
        "left_numerical_rank": left_numerical_rank,
        "right_numerical_rank": right_numerical_rank,
        "baseline_left_singular_values": baseline_left_singular,
        "baseline_right_singular_values": baseline_right_singular,
    }


def three_site_normal_gradient_analysis(
    model: torch.nn.Module,
    fit_objective: dict[str, Any],
    *,
    start: int,
    source_bond_dimension: int,
    chunk_size: int,
    score_ranks: tuple[int, ...] = (2, 4, 8),
) -> dict[str, Any]:
    """Resolve the double-normal dE2/dTheta into three Tucker mode spaces."""

    if not score_ranks or any(rank <= 0 for rank in score_ranks):
        raise ValueError("three-site score ranks must be positive")
    temporary = copy.deepcopy(model)
    temporary.requires_grad_(False).eval()
    active = temporary.activate_three_site_coefficient_core_(start)
    active.requires_grad_(True)
    vectorizer = ComplexParameterVectorizer.from_module(
        temporary,
        parameter_names=("three_site_core",),
    )
    theta = vectorizer.pack(temporary)
    operator = MatrixFreeNativeMARatioJacobian(
        temporary,
        vectorizer,
        theta,
        fit_objective,
        chunk_size=chunk_size,
    )
    residual = operator.residual()
    gradient = operator.vjp(residual).reshape_as(active)
    left_bond, right_bond, first_physical, middle_physical, right_physical = (
        active.shape
    )
    grouped_shape = (
        left_bond * first_physical,
        middle_physical,
        right_physical * right_bond,
    )
    baseline = active.permute(0, 2, 3, 4, 1).reshape(grouped_shape)
    grouped_gradient = gradient.permute(0, 2, 3, 4, 1).reshape(grouped_shape)
    projection = project_three_site_double_normal(
        baseline,
        grouped_gradient,
        inherited_rank=source_bond_dimension,
    )
    projected = projection["projected"]

    left_basis, left_singular = left_singular_system(
        projected.reshape(
            grouped_shape[0],
            grouped_shape[1] * grouped_shape[2],
        )
    )
    middle_basis, middle_singular = left_singular_system(
        projected.permute(1, 0, 2).reshape(
            grouped_shape[1],
            grouped_shape[0] * grouped_shape[2],
        )
    )
    right_basis, right_singular = left_singular_system(
        projected.permute(2, 0, 1).reshape(
            grouped_shape[2],
            grouped_shape[0] * grouped_shape[1],
        )
    )
    normal_energy = torch.sum(torch.abs(projected) ** 2)
    full_gradient_energy = torch.sum(torch.abs(grouped_gradient) ** 2)

    scores: dict[str, float] = {}
    fractions: dict[str, float] = {}
    for rank in score_ranks:
        left_stop = min(rank, left_basis.shape[1])
        middle_stop = min(rank, middle_basis.shape[1])
        right_stop = min(rank, right_basis.shape[1])
        core = torch.einsum(
            "ai,bj,abc,ck->ijk",
            torch.conj(left_basis[:, :left_stop]),
            torch.conj(middle_basis[:, :middle_stop]),
            projected,
            torch.conj(right_basis[:, :right_stop]),
        )
        score = torch.sum(torch.abs(core) ** 2)
        scores[f"rank{rank}"] = float(score)
        fractions[f"rank{rank}"] = float(
            score
            / torch.clamp(
                normal_energy,
                min=torch.finfo(normal_energy.dtype).tiny,
            )
        )

    diagnostics = {
        "fit_native_e2": float(real_inner(residual, residual)),
        "active_supercore_complex_parameters": int(theta.numel()),
        "grouped_shape": list(grouped_shape),
        "left_inherited_rank_projected_out": int(
            projection["inherited_left"].shape[1]
        ),
        "right_inherited_rank_projected_out": int(
            projection["inherited_right"].shape[1]
        ),
        "baseline_left_numerical_rank": projection["left_numerical_rank"],
        "baseline_right_numerical_rank": projection["right_numerical_rank"],
        "baseline_left_singular_values": [
            float(value)
            for value in projection["baseline_left_singular_values"][
                :16
            ].detach().cpu()
        ],
        "baseline_right_singular_values": [
            float(value)
            for value in projection["baseline_right_singular_values"][
                :16
            ].detach().cpu()
        ],
        "double_normal_gradient_energy": float(normal_energy),
        "full_supercore_gradient_energy": float(full_gradient_energy),
        "double_normal_fraction_of_full": float(
            normal_energy
            / torch.clamp(
                full_gradient_energy,
                min=torch.finfo(full_gradient_energy.dtype).tiny,
            )
        ),
        "left_mode_singular_values": [
            float(value) for value in left_singular[:16].detach().cpu()
        ],
        "middle_mode_singular_values": [
            float(value) for value in middle_singular[:16].detach().cpu()
        ],
        "right_mode_singular_values": [
            float(value) for value in right_singular[:16].detach().cpu()
        ],
        "tucker_scores": scores,
        "tucker_normal_energy_fractions": fractions,
    }
    temporary.clear_three_site_coefficient_core_()
    del operator, temporary
    return {
        "left_basis": left_basis.detach(),
        "middle_basis": middle_basis.detach(),
        "right_basis": right_basis.detach(),
        "diagnostics": diagnostics,
    }


def residual_aligned_right_seed(
    model: torch.nn.Module,
    fit_objective: dict[str, Any],
    *,
    bond_index: int,
    source_bond_dimension: int,
    rank_increase: int,
    left_active: int,
    right_active: int,
    chunk_size: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Find new right Schmidt vectors from the normal component of dE2/dTheta."""

    analysis = normal_gradient_schmidt_analysis(
        model,
        fit_objective,
        bond_index=bond_index,
        source_bond_dimension=source_bond_dimension,
        left_active=left_active,
        right_active=right_active,
        chunk_size=chunk_size,
    )
    singular = analysis["singular_values"]
    right_adjoint = analysis["right_adjoint"]
    if singular.numel() < rank_increase or not bool(
        torch.all(torch.isfinite(singular[:rank_increase]))
    ):
        raise FloatingPointError("normal-gradient SVD has too few finite directions")
    if float(singular[rank_increase - 1]) <= 0:
        raise FloatingPointError("normal-gradient rank is below the requested increase")
    right_matrix = right_adjoint[:rank_increase]
    right_seed = right_matrix.reshape(
        rank_increase,
        analysis["physical_right"],
        right_active,
    ).permute(0, 2, 1)
    return right_seed, analysis["diagnostics"]


def set_seed_(
    model: torch.nn.Module,
    *,
    source_bond_dimension: int,
    rank_increase: int,
    right_active: int,
    right_name: str,
    right_seed: torch.Tensor,
) -> None:
    right = dict(model.named_parameters())[right_name]
    expected = (rank_increase, right_active, right.shape[2])
    if tuple(right_seed.shape) != expected:
        raise ValueError(f"right seed has shape {tuple(right_seed.shape)}, expected {expected}")
    with torch.no_grad():
        right[
            source_bond_dimension : source_bond_dimension + rank_increase,
            :right_active,
            :,
        ].copy_(right_seed)


def copy_masked_values_(
    target: torch.Tensor,
    source: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    """Copy selected source entries into a parameter without advanced-index copies."""

    if target.shape != source.shape or target.shape != mask.shape:
        raise ValueError("masked source, target, and mask shapes must agree")
    resolved_mask = mask.to(device=target.device, dtype=torch.bool)
    selected = source.to(device=target.device, dtype=target.dtype)[resolved_mask]
    with torch.no_grad():
        target.masked_scatter_(resolved_mask, selected)


def balance_branch_(
    model: torch.nn.Module,
    *,
    source_bond_dimension: int,
    rank_increase: int,
    left_active: int,
    right_active: int,
    left_name: str,
    right_name: str,
) -> dict[str, Any]:
    """Balance the learned branch without changing its two-site update."""

    left = dict(model.named_parameters())[left_name]
    right = dict(model.named_parameters())[right_name]
    channel_slice = slice(
        source_bond_dimension,
        source_bond_dimension + rank_increase,
    )
    left_block = left[:left_active, channel_slice, :]
    right_block = right[channel_slice, :right_active, :]
    left_matrix = left_block.permute(0, 2, 1).reshape(-1, rank_increase)
    right_matrix = right_block.permute(0, 2, 1).reshape(rank_increase, -1)
    update = left_matrix @ right_matrix
    left_singular, singular, right_adjoint = torch.linalg.svd(
        update,
        full_matrices=False,
    )
    square_root = torch.sqrt(singular[:rank_increase])
    balanced_left = left_singular[:, :rank_increase] * square_root[None, :]
    balanced_right = square_root[:, None] * right_adjoint[:rank_increase]
    reconstructed = balanced_left @ balanced_right
    relative_error = float(
        torch.linalg.vector_norm(reconstructed - update)
        / torch.clamp(
            torch.linalg.vector_norm(update),
            min=torch.finfo(update.real.dtype).tiny,
        )
    )
    tolerance = 2.0e-5 if update.dtype == torch.complex64 else 1.0e-11
    if relative_error > tolerance:
        raise RuntimeError("balancing changed the rank-growth branch")
    with torch.no_grad():
        left[:left_active, channel_slice, :].copy_(
            balanced_left.reshape(
                left_active,
                left.shape[2],
                rank_increase,
            ).permute(0, 2, 1)
        )
        right[channel_slice, :right_active, :].copy_(
            balanced_right.reshape(
                rank_increase,
                right.shape[2],
                right_active,
            ).permute(0, 2, 1)
        )
    return {
        "singular_values": [
            float(value) for value in singular[:rank_increase].detach().cpu()
        ],
        "relative_reconstruction_error": relative_error,
        "tolerance": tolerance,
    }


def candidate_shortlist(rows: list[dict[str, Any]], limit: int) -> list[int]:
    positive = [
        index
        for index, row in enumerate(rows)
        if row["screen"]["positive_metric"]
    ]
    ordered = sorted(
        positive,
        key=lambda index: (
            rows[index]["screen"]["selection_objective"]["e2"],
            rows[index]["screen"]["selection_base"]["sigma"],
            rows[index]["screen"]["fit_objective"]["e2"],
            index,
        ),
    )
    return ordered[:limit]


def _half_step_impl(
    *,
    model: torch.nn.Module,
    side: str,
    parameter_name: str,
    mask: torch.Tensor,
    fit_base: dict[str, Any],
    selection_base: dict[str, Any],
    fit_objective: dict[str, Any],
    selection_objective: dict[str, Any],
    screen_fit_base: dict[str, Any],
    screen_selection_base: dict[str, Any],
    screen_fit_objective: dict[str, Any],
    screen_selection_objective: dict[str, Any],
    args: argparse.Namespace,
    status: Path,
) -> dict[str, Any]:
    dict(model.named_parameters())[parameter_name].requires_grad_(True)
    vectorizer = MaskedComplexParameterVectorizer.from_module(
        model,
        parameter_name,
        mask,
    )
    theta = vectorizer.pack(model)
    operator = MatrixFreeNativeMARatioJacobian(
        model,
        vectorizer,
        theta,
        fit_objective,
        chunk_size=args.operator_chunk_size,
    )
    residual = operator.residual()
    baseline_energy = float(real_inner(residual, residual))
    gradient = operator.vjp(residual)
    gradient_norm = float(vector_norm(gradient))
    rayleigh_scale = gradient_norm**2 / max(
        baseline_energy,
        np.finfo(float).tiny,
    )
    baselines = {
        "fit_base": model_summary(
            model,
            fit_base,
            batch_size=args.eval_batch_size,
        ),
        "selection_base": model_summary(
            model,
            selection_base,
            batch_size=args.eval_batch_size,
        ),
        "fit_objective": model_summary(
            model,
            fit_objective,
            batch_size=args.eval_batch_size,
        ),
        "selection_objective": model_summary(
            model,
            selection_objective,
            batch_size=args.eval_batch_size,
        ),
        "screen_fit_base": model_summary(
            model,
            screen_fit_base,
            batch_size=args.eval_batch_size,
        ),
        "screen_selection_base": model_summary(
            model,
            screen_selection_base,
            batch_size=args.eval_batch_size,
        ),
        "screen_fit_objective": model_summary(
            model,
            screen_fit_objective,
            batch_size=args.eval_batch_size,
        ),
        "screen_selection_objective": model_summary(
            model,
            screen_selection_objective,
            batch_size=args.eval_batch_size,
        ),
    }
    report: dict[str, Any] = {
        "side": side,
        "parameter_name": parameter_name,
        "active_complex_parameters": int(theta.numel()),
        "gradient_norm": gradient_norm,
        "rayleigh_scale": rayleigh_scale,
        "baselines": baselines,
        "rows": [],
        "selected": None,
        "accepted": False,
    }
    if not np.isfinite(rayleigh_scale) or rayleigh_scale <= 0:
        report["reason"] = "no finite residual-aligned tangent"
        return report

    def progress(iteration: int, alpha: float, beta: float) -> None:
        write_json(
            status,
            {
                "state": "running",
                "phase": "rank_growth_lanczos",
                "side": side,
                "iteration": iteration,
                "count": args.lanczos_steps,
            },
        )
        if not args.quiet:
            print(
                f"side={side} lanczos={iteration}/{args.lanczos_steps} "
                f"alpha={alpha:.6e} beta={beta:.6e}",
                flush=True,
            )

    lanczos = lanczos_tridiagonal(
        operator.normal,
        residual,
        steps=args.lanczos_steps,
        callback=progress,
    )
    ridge_entries = []
    for factor in args.ridge_factors:
        ridge = float(factor * rayleigh_scale)
        coefficients = ridge_coefficients_from_lanczos(lanczos, ridge)
        ridge_entries.append(
            {
                "ridge_factor": float(factor),
                "ridge": ridge,
                "dual": lanczos.basis @ coefficients,
            }
        )
    ridge_deltas = -operator.vjp_many(
        torch.stack([entry["dual"] for entry in ridge_entries])
    )
    rows: list[dict[str, Any]] = []
    deltas: list[torch.Tensor] = []
    for ridge_index, entry in enumerate(ridge_entries):
        for step_scale in args.step_scales:
            delta = float(step_scale) * ridge_deltas[ridge_index]
            vectorizer.commit_(model, theta + delta)
            screen = {
                "fit_base": model_summary(
                    model,
                    screen_fit_base,
                    batch_size=args.eval_batch_size,
                ),
                "selection_base": model_summary(
                    model,
                    screen_selection_base,
                    batch_size=args.eval_batch_size,
                ),
                "fit_objective": model_summary(
                    model,
                    screen_fit_objective,
                    batch_size=args.eval_batch_size,
                ),
                "selection_objective": model_summary(
                    model,
                    screen_selection_objective,
                    batch_size=args.eval_batch_size,
                ),
            }
            vectorizer.commit_(model, theta)
            screen["positive_metric"] = bool(
                all(
                    screen[key]["minimum_metric_eigenvalue"] > 0
                    for key in screen
                    if isinstance(screen[key], dict)
                )
            )
            rows.append(
                {
                    "ridge_factor": entry["ridge_factor"],
                    "ridge": entry["ridge"],
                    "step_scale": float(step_scale),
                    "step_rms": float(
                        torch.sqrt(torch.mean(torch.abs(delta) ** 2))
                    ),
                    "screen": screen,
                    "screen_shortlisted": False,
                    "fit_base": None,
                    "selection_base": None,
                    "fit_objective": None,
                    "selection_objective": None,
                    "fit_objective_capture": None,
                    "selection_objective_capture": None,
                    "intermediate_eligible": False,
                }
            )
            deltas.append(delta.detach().clone())

    shortlisted = candidate_shortlist(rows, args.nonlinear_shortlist)
    shortlisted_set = set(shortlisted)
    for index, row in enumerate(rows):
        row["screen_shortlisted"] = index in shortlisted_set
    for index in shortlisted:
        vectorizer.commit_(model, theta + deltas[index])
        fit_base_trial = model_summary(
            model,
            fit_base,
            batch_size=args.eval_batch_size,
        )
        selection_base_trial = model_summary(
            model,
            selection_base,
            batch_size=args.eval_batch_size,
        )
        fit_objective_trial = model_summary(
            model,
            fit_objective,
            batch_size=args.eval_batch_size,
        )
        selection_objective_trial = model_summary(
            model,
            selection_objective,
            batch_size=args.eval_batch_size,
        )
        vectorizer.commit_(model, theta)
        fit_capture = capture(
            baselines["fit_objective"]["e2"],
            fit_objective_trial["e2"],
        )
        selection_capture = capture(
            baselines["selection_objective"]["e2"],
            selection_objective_trial["e2"],
        )
        eligible = bool(
            fit_capture > 0
            and selection_capture > 0
            and selection_base_trial["sigma"]
            <= baselines["selection_base"]["sigma"]
            and selection_base_trial["minimum_metric_eigenvalue"] > 0
            and selection_objective_trial["minimum_metric_eigenvalue"] > 0
        )
        rows[index].update(
            {
                "fit_base": fit_base_trial,
                "selection_base": selection_base_trial,
                "fit_objective": fit_objective_trial,
                "selection_objective": selection_objective_trial,
                "fit_objective_capture": fit_capture,
                "selection_objective_capture": selection_capture,
                "intermediate_eligible": eligible,
            }
        )
        if not args.quiet:
            print(
                f"side={side} ridge={rows[index]['ridge_factor']:.6g} "
                f"step={rows[index]['step_scale']:.6g} "
                f"fit_orbit_E2_capture={fit_capture:.6f} "
                f"selection_orbit_E2_capture={selection_capture:.6f} "
                f"selection_sigma={selection_base_trial['sigma']:.8e} "
                f"eligible={eligible}",
                flush=True,
            )
    eligible_indices = [
        index for index, row in enumerate(rows) if row["intermediate_eligible"]
    ]
    selected_index = (
        min(
            eligible_indices,
            key=lambda index: (
                rows[index]["selection_objective"]["e2"],
                rows[index]["selection_base"]["sigma"],
                index,
            ),
        )
        if eligible_indices
        else None
    )
    if selected_index is not None:
        vectorizer.commit_(model, theta + deltas[selected_index])
    report.update(
        {
            "lanczos": {
                "steps_completed": int(lanczos.tridiagonal.shape[0]),
                "breakdown": bool(lanczos.breakdown),
                "basis_orthogonality_error": lanczos.orthogonality_error,
            },
            "shortlisted_candidate_indices": shortlisted,
            "rows": rows,
            "selected": None if selected_index is None else rows[selected_index],
            "accepted": selected_index is not None,
        }
    )
    return report


def half_step(
    *,
    model: torch.nn.Module,
    side: str,
    parameter_name: str,
    mask: torch.Tensor,
    fit_base: dict[str, Any],
    selection_base: dict[str, Any],
    fit_objective: dict[str, Any],
    selection_objective: dict[str, Any],
    screen_fit_base: dict[str, Any],
    screen_selection_base: dict[str, Any],
    screen_fit_objective: dict[str, Any],
    screen_selection_objective: dict[str, Any],
    args: argparse.Namespace,
    status: Path,
) -> dict[str, Any]:
    """Run one ALS half-step and release its autodiff workspace before the next."""

    active_parameter = dict(model.named_parameters())[parameter_name]
    try:
        return _half_step_impl(
            model=model,
            side=side,
            parameter_name=parameter_name,
            mask=mask,
            fit_base=fit_base,
            selection_base=selection_base,
            fit_objective=fit_objective,
            selection_objective=selection_objective,
            screen_fit_base=screen_fit_base,
            screen_selection_base=screen_selection_base,
            screen_fit_objective=screen_fit_objective,
            screen_selection_objective=screen_selection_objective,
            args=args,
            status=status,
        )
    finally:
        active_parameter.requires_grad_(False)
        if active_parameter.device.type == "cuda":
            torch.cuda.synchronize(active_parameter.device)
        gc.collect()
        if active_parameter.device.type == "cuda":
            torch.cuda.empty_cache()


def global_gate(
    candidate: dict[str, dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
) -> bool:
    return bool(
        candidate["fit_objective"]["e2"] < baseline["fit_objective"]["e2"]
        and candidate["selection_objective"]["e2"]
        < baseline["selection_objective"]["e2"]
        and candidate["selection_base"]["e2"] < baseline["selection_base"]["e2"]
        and candidate["selection_base"]["sigma"]
        <= baseline["selection_base"]["sigma"]
        and candidate["selection_objective"]["sigma"]
        <= baseline["selection_objective"]["sigma"]
        and candidate["selection_base"]["minimum_metric_eigenvalue"] > 0
        and candidate["selection_objective"]["minimum_metric_eigenvalue"] > 0
        and tail_nonworse(
            candidate["selection_base"],
            baseline["selection_base"],
        )
        and tail_nonworse(
            candidate["selection_objective"],
            baseline["selection_objective"],
        )
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    started = time.perf_counter()
    output_model = args.output_model.expanduser().resolve()
    output_report = args.output_report.expanduser().resolve()
    status = output_report.with_suffix(output_report.suffix + ".status.json")
    indices_path = output_report.with_suffix(output_report.suffix + ".indices.npz")
    for path in (output_model, output_report, status, indices_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite artifact: {path}")
    output_model.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    write_json(status, {"state": "running", "phase": "loading"})

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
        raise ValueError("rank growth requires a quintic TN artifact")
    if payload.get("architecture") != "shared_local_dictionary":
        raise ValueError("rank growth requires a shared local dictionary")
    source_bond_dimension = int(payload["bond_dimension"])
    target_bond_dimension = (
        int(args.target_bond_dimension)
        if args.target_bond_dimension
        else source_bond_dimension + int(args.rank_increase)
    )
    if target_bond_dimension < source_bond_dimension + args.rank_increase:
        raise ValueError("target bond does not contain the requested new channels")
    source_degree = int(payload.get("source_degree", 1))
    if source_degree != 1:
        raise ValueError("Fermat orbit augmentation currently requires O(1) source jets")

    data = np.load(dataset_path, allow_pickle=False)
    fit_stop = args.fit_train_start + args.fit_size
    selection_stop = args.selection_validation_start + args.selection_size
    if fit_stop > len(data["X_train"]):
        raise ValueError("fit window exceeds X_train")
    if selection_stop > len(data["X_val"]):
        raise ValueError("selection window exceeds X_val")
    fit_indices = np.arange(args.fit_train_start, fit_stop, dtype=np.int64)
    selection_indices = np.arange(
        args.selection_validation_start,
        selection_stop,
        dtype=np.int64,
    )
    np.savez_compressed(
        indices_path,
        fit_train_indices=fit_indices,
        selection_validation_indices=selection_indices,
    )
    train_pullbacks = np.load(train_pullbacks_path, mmap_mode="r")
    validation_pullbacks = np.load(validation_pullbacks_path, mmap_mode="r")

    search_complex_dtype = (
        torch.complex64
        if args.search_precision == "complex64"
        else torch.complex128
    )
    search_real_dtype = (
        torch.float32
        if search_complex_dtype == torch.complex64
        else torch.float64
    )

    def make_split(
        *,
        domain: str,
        indices: np.ndarray,
        complex_dtype: torch.dtype,
        real_dtype: torch.dtype,
    ) -> dict[str, Any]:
        if domain == "train":
            x = data["X_train"]
            y = data["y_train"]
            pullbacks = train_pullbacks
        elif domain == "validation":
            x = data["X_val"]
            y = data["y_val"]
            pullbacks = validation_pullbacks
        else:
            raise ValueError("unknown data domain")
        return tensor_split(
            np.asarray(x[indices], dtype=np.float32),
            np.asarray(pullbacks[indices]),
            np.asarray(y[indices], dtype=np.float64),
            source_degree=source_degree,
            complex_dtype=complex_dtype,
            real_dtype=real_dtype,
            device=device,
        )

    fit = make_split(
        domain="train",
        indices=fit_indices,
        complex_dtype=search_complex_dtype,
        real_dtype=search_real_dtype,
    )
    selection = make_split(
        domain="validation",
        indices=selection_indices,
        complex_dtype=search_complex_dtype,
        real_dtype=search_real_dtype,
    )
    preservation_indices = selection_indices[: min(64, args.selection_size)]
    preservation = make_split(
        domain="validation",
        indices=preservation_indices,
        complex_dtype=torch.complex128,
        real_dtype=torch.float64,
    )

    reference_h = np.asarray(
        payload["state_dict"]["reference_h"].detach().cpu(),
        dtype=np.complex128,
    )
    original_exact = positive_tensor_network_from_artifact_payload(
        reference_h,
        payload,
        device=device,
        trainable_physical_dictionary=False,
    ).to(device=device, dtype=torch.complex128)
    original_exact.requires_grad_(False).eval()
    canonical_exact = copy.deepcopy(original_exact)
    canonical_exact.mixed_canonicalize_coefficient_pair_(args.bond_index)
    global_scale = canonical_exact.normalize_coefficient_chain_scale_(
        site_index=args.bond_index
    )
    canonical_relative_change = maximum_relative_metric_difference(
        original_exact,
        canonical_exact,
        preservation,
        batch_size=args.eval_batch_size,
    )
    expanded_exact = expanded_model(
        canonical_exact,
        payload,
        target_bond_dimension=target_bond_dimension,
        dtype=torch.complex128,
        device=device,
    )
    expansion_relative_change = maximum_relative_metric_difference(
        original_exact,
        expanded_exact,
        preservation,
        batch_size=args.eval_batch_size,
    )
    exact_tolerance = 2.0e-10
    if max(canonical_relative_change, expansion_relative_change) > exact_tolerance:
        raise RuntimeError("canonicalization or zero-padding changed the baseline metric")

    search_model = copy.deepcopy(expanded_exact).to(
        device=device,
        dtype=search_complex_dtype,
    )
    search_model.requires_grad_(False).eval()
    (
        left_name,
        left_mask,
        right_name,
        right_mask,
        left_active,
        right_active,
    ) = bond_growth_masks(
        search_model,
        bond_index=args.bond_index,
        source_bond_dimension=source_bond_dimension,
        rank_increase=args.rank_increase,
    )
    if bool(torch.any(torch.abs(dict(search_model.named_parameters())[left_name][left_mask]) > 0)):
        raise RuntimeError("new left channels are not zero after exact expansion")
    if bool(torch.any(torch.abs(dict(search_model.named_parameters())[right_name][right_mask]) > 0)):
        raise RuntimeError("new right channels are not zero after exact expansion")

    action_generator = torch.Generator(device=device)
    action_generator.manual_seed(args.seed + 1009)
    actions = fixed_fermat_actions_torch(
        args.group_samples,
        generator=action_generator,
        complex_dtype=search_complex_dtype,
        device=device,
    )
    fit_objective = orbit_augment_dataset(fit, actions)
    selection_objective = orbit_augment_dataset(selection, actions)
    screen_fit = renormalized_dataset_prefix(fit, args.screen_fit_size)
    screen_selection = renormalized_dataset_prefix(
        selection,
        args.screen_selection_size,
    )
    screen_fit_objective = orbit_augment_dataset(screen_fit, actions)
    screen_selection_objective = orbit_augment_dataset(screen_selection, actions)

    write_json(status, {"state": "running", "phase": "gradient_seed"})
    right_seed, seed_diagnostics = residual_aligned_right_seed(
        search_model,
        fit_objective,
        bond_index=args.bond_index,
        source_bond_dimension=source_bond_dimension,
        rank_increase=args.rank_increase,
        left_active=left_active,
        right_active=right_active,
        chunk_size=args.operator_chunk_size,
    )
    baselines = {
        "fit_base": model_summary(
            search_model,
            fit,
            batch_size=args.eval_batch_size,
        ),
        "selection_base": model_summary(
            search_model,
            selection,
            batch_size=args.eval_batch_size,
        ),
        "fit_objective": model_summary(
            search_model,
            fit_objective,
            batch_size=args.eval_batch_size,
        ),
        "selection_objective": model_summary(
            search_model,
            selection_objective,
            batch_size=args.eval_batch_size,
        ),
    }
    set_seed_(
        search_model,
        source_bond_dimension=source_bond_dimension,
        rank_increase=args.rank_increase,
        right_active=right_active,
        right_name=right_name,
        right_seed=right_seed,
    )
    seeded = {
        "fit_base": model_summary(
            search_model,
            fit,
            batch_size=args.eval_batch_size,
        ),
        "selection_base": model_summary(
            search_model,
            selection,
            batch_size=args.eval_batch_size,
        ),
        "fit_objective": model_summary(
            search_model,
            fit_objective,
            batch_size=args.eval_batch_size,
        ),
        "selection_objective": model_summary(
            search_model,
            selection_objective,
            batch_size=args.eval_batch_size,
        ),
    }
    epoch_zero_change = {
        domain: {
            key: seeded[domain][key] - baselines[domain][key]
            for key in ("sigma", "e2")
        }
        for domain in baselines
    }
    epoch_zero_tolerance = (
        2.0e-6 if search_complex_dtype == torch.complex64 else 1.0e-11
    )
    if max(
        abs(value)
        for domain in epoch_zero_change.values()
        for value in domain.values()
    ) > epoch_zero_tolerance:
        raise RuntimeError("right-only seed changed the epoch-zero metric")

    half_steps = []
    for side, parameter_name, mask in (
        ("left", left_name, left_mask),
        ("right", right_name, right_mask),
    ):
        write_json(
            status,
            {
                "state": "running",
                "phase": "rank_growth_als",
                "side": side,
            },
        )
        row = half_step(
            model=search_model,
            side=side,
            parameter_name=parameter_name,
            mask=mask,
            fit_base=fit,
            selection_base=selection,
            fit_objective=fit_objective,
            selection_objective=selection_objective,
            screen_fit_base=screen_fit,
            screen_selection_base=screen_selection,
            screen_fit_objective=screen_fit_objective,
            screen_selection_objective=screen_selection_objective,
            args=args,
            status=status,
        )
        half_steps.append(row)
        if not row["accepted"]:
            break

    balancing = None
    search_candidate = None
    accepted_on_search_selection = False
    if len(half_steps) == 2 and all(row["accepted"] for row in half_steps):
        balancing = balance_branch_(
            search_model,
            source_bond_dimension=source_bond_dimension,
            rank_increase=args.rank_increase,
            left_active=left_active,
            right_active=right_active,
            left_name=left_name,
            right_name=right_name,
        )
        search_candidate = {
            "fit_base": model_summary(
                search_model,
                fit,
                batch_size=args.eval_batch_size,
            ),
            "selection_base": model_summary(
                search_model,
                selection,
                batch_size=args.eval_batch_size,
            ),
            "fit_objective": model_summary(
                search_model,
                fit_objective,
                batch_size=args.eval_batch_size,
            ),
            "selection_objective": model_summary(
                search_model,
                selection_objective,
                batch_size=args.eval_batch_size,
            ),
        }
        accepted_on_search_selection = global_gate(search_candidate, baselines)

    exact_candidate = None
    accepted_on_exact_selection = False
    if accepted_on_search_selection:
        exact_left = dict(expanded_exact.named_parameters())[left_name]
        exact_right = dict(expanded_exact.named_parameters())[right_name]
        search_left = dict(search_model.named_parameters())[left_name]
        search_right = dict(search_model.named_parameters())[right_name]
        copy_masked_values_(exact_left, search_left, left_mask)
        copy_masked_values_(exact_right, search_right, right_mask)
        fit_exact = make_split(
            domain="train",
            indices=fit_indices,
            complex_dtype=torch.complex128,
            real_dtype=torch.float64,
        )
        selection_exact = make_split(
            domain="validation",
            indices=selection_indices,
            complex_dtype=torch.complex128,
            real_dtype=torch.float64,
        )
        exact_action_generator = torch.Generator(device=device)
        exact_action_generator.manual_seed(args.seed + 1009)
        exact_actions = fixed_fermat_actions_torch(
            args.group_samples,
            generator=exact_action_generator,
            complex_dtype=torch.complex128,
            device=device,
        )
        fit_objective_exact = orbit_augment_dataset(fit_exact, exact_actions)
        selection_objective_exact = orbit_augment_dataset(
            selection_exact,
            exact_actions,
        )
        exact_baseline_model = expanded_model(
            canonical_exact,
            payload,
            target_bond_dimension=target_bond_dimension,
            dtype=torch.complex128,
            device=device,
        )
        exact_baselines = {
            "fit_base": model_summary(
                exact_baseline_model,
                fit_exact,
                batch_size=args.eval_batch_size,
            ),
            "selection_base": model_summary(
                exact_baseline_model,
                selection_exact,
                batch_size=args.eval_batch_size,
            ),
            "fit_objective": model_summary(
                exact_baseline_model,
                fit_objective_exact,
                batch_size=args.eval_batch_size,
            ),
            "selection_objective": model_summary(
                exact_baseline_model,
                selection_objective_exact,
                batch_size=args.eval_batch_size,
            ),
        }
        exact_candidate = {
            "fit_base": model_summary(
                expanded_exact,
                fit_exact,
                batch_size=args.eval_batch_size,
            ),
            "selection_base": model_summary(
                expanded_exact,
                selection_exact,
                batch_size=args.eval_batch_size,
            ),
            "fit_objective": model_summary(
                expanded_exact,
                fit_objective_exact,
                batch_size=args.eval_batch_size,
            ),
            "selection_objective": model_summary(
                expanded_exact,
                selection_objective_exact,
                batch_size=args.eval_batch_size,
            ),
        }
        accepted_on_exact_selection = global_gate(
            exact_candidate,
            exact_baselines,
        )
        if accepted_on_exact_selection:
            output_payload = copy.deepcopy(payload)
            output_payload["state_dict"] = {
                key: value.detach().cpu().clone()
                for key, value in expanded_exact.state_dict().items()
            }
            output_payload["precision"] = "complex128"
            output_payload["bond_dimension"] = target_bond_dimension
            output_payload["positive_floor"] = float(expanded_exact.positive_floor)
            output_payload["rank_growth"] = {
                "schema": "quintic-tn-native-ma-rank-growth-v1",
                "source_model": str(model_path),
                "source_model_sha256": sha256_file(model_path),
                "bond_index": args.bond_index,
                "source_bond_dimension": source_bond_dimension,
                "target_bond_dimension": target_bond_dimension,
                "rank_increase": args.rank_increase,
                "effective_added_real_parameters": int(
                    2
                    * args.rank_increase
                    * (left_active + right_active)
                    * search_left.shape[2]
                ),
                "selection_is_not_confirmation": True,
            }
            temporary = output_model.with_suffix(output_model.suffix + ".tmp")
            torch.save(output_payload, temporary)
            temporary.replace(output_model)

    report = {
        "schema": "quintic-tn-native-ma-rank-growth-v1",
        "scientific_scope": {
            "purpose": (
                "Test whether four genuinely new Schmidt channels lower the "
                "native normalized MA objective from the best k=30 checkpoint."
            ),
            "nestedness": (
                "The D12 source is canonicalized and zero-padded to D16. The "
                "new right factor is seeded while the new left factor remains "
                "zero, so the epoch-zero metric is the source metric."
            ),
            "objective": (
                "Fit and selection directions use native E2 with the derivative "
                "of sample volume normalization included. Fixed Fermat group "
                "images are included in the optimization objective."
            ),
            "selection_rule": (
                "Both base and orbit-averaged selection E2 must improve; base "
                "and orbit sigma/tails must be nonworse; positivity must hold."
            ),
            "claim_limit": (
                "A saved model is a development candidate and requires one new "
                "frozen confirmation plus a separate 20k tail gate."
            ),
        },
        "configuration": {
            **vars(args),
            "model": str(model_path),
            "source_run_dir": str(source_dir),
            "pullbacks_dir": str(pullbacks_dir),
            "output_model": str(output_model),
            "output_report": str(output_report),
            "device": str(device),
            "source_bond_dimension": source_bond_dimension,
            "target_bond_dimension": target_bond_dimension,
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
            "saved_model": str(output_model) if accepted_on_exact_selection else None,
            "saved_model_sha256": (
                sha256_file(output_model)
                if accepted_on_exact_selection
                else None
            ),
        },
        "exact_embedding": {
            "global_scale": global_scale,
            "canonical_maximum_relative_metric_change": canonical_relative_change,
            "expanded_maximum_relative_metric_change": expansion_relative_change,
            "tolerance": exact_tolerance,
        },
        "parameterization": {
            "left_parameter": left_name,
            "right_parameter": right_name,
            "left_active_dimension": left_active,
            "right_active_dimension": right_active,
            "active_left_complex_parameters": int(torch.count_nonzero(left_mask)),
            "active_right_complex_parameters": int(torch.count_nonzero(right_mask)),
            "effective_added_real_parameters": int(
                2
                * (torch.count_nonzero(left_mask) + torch.count_nonzero(right_mask))
            ),
        },
        "gradient_seed": seed_diagnostics,
        "epoch_zero_change": epoch_zero_change,
        "epoch_zero_tolerance": epoch_zero_tolerance,
        "baseline": baselines,
        "half_steps": half_steps,
        "balancing": balancing,
        "search_candidate": search_candidate,
        "accepted_on_search_selection": accepted_on_search_selection,
        "exact_candidate": exact_candidate,
        "accepted_on_exact_selection": accepted_on_exact_selection,
        "next_gate": (
            "Run exactly one fresh frozen 5k confirmation and 20k tail audit."
            if accepted_on_exact_selection
            else "Do not open confirmation; this central rank-4 candidate failed selection."
        ),
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(output_report, report)
    write_json(status, {"state": "complete", "phase": "complete"})
    print(
        json.dumps(
            {
                "accepted_on_search_selection": accepted_on_search_selection,
                "accepted_on_exact_selection": accepted_on_exact_selection,
                "output_model": (
                    str(output_model) if accepted_on_exact_selection else None
                ),
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
