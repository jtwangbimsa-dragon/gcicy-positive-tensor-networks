#!/usr/bin/env python3
"""Optimize a matrix-free symmetry-allowed bond supercore with native MA E2."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
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

from gcicy_metric.pipeline import positive_tensor_network_from_artifact_payload  # noqa: E402
from scripts.audit_quintic_tn_tangent_reachability import (  # noqa: E402
    lanczos_tridiagonal,
    real_inner,
    ridge_coefficients_from_lanczos,
    select_disjoint_indices,
    vector_norm,
)
from scripts.refine_quintic_hard_symmetry_channel_native_ma import (  # noqa: E402
    MatrixFreeNativeMARatioJacobian,
    model_summary,
    weighted_log_mean_exp,
)
from scripts.train_quintic_full_h_same_points import (  # noqa: E402
    sha256_file,
    write_json,
)
from scripts.train_quintic_positive_tensor_network_same_points import (  # noqa: E402
    tensor_split,
    training_log_volume,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expanded-model", type=Path, required=True)
    parser.add_argument(
        "--right-seed-model",
        type=Path,
        default=None,
        help=(
            "Optional previously calibrated model whose retained right Schmidt "
            "subspace replaces the randomized full-supercore SVD."
        ),
    )
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--pullbacks-dir", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--data-start", type=int, default=200_000)
    parser.add_argument("--data-limit", type=int, default=10_000)
    parser.add_argument("--fit-size", type=int, default=4096)
    parser.add_argument("--selection-size", type=int, default=4096)
    parser.add_argument("--bond-index", type=int, default=5)
    parser.add_argument("--retained-rank", type=int, default=4)
    parser.add_argument("--oversample", type=int, default=2)
    parser.add_argument("--operator-chunk-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument(
        "--cache-fit-prefix",
        action="store_true",
        help=(
            "Cache fit-set value/first/mixed jets immediately before the "
            "optimized bond pair in host memory."
        ),
    )
    parser.add_argument("--prefix-cache-batch-size", type=int, default=8)
    parser.add_argument("--screen-fit-size", type=int, default=0)
    parser.add_argument("--screen-selection-size", type=int, default=0)
    parser.add_argument(
        "--nonlinear-shortlist",
        type=int,
        default=0,
        help=(
            "Number of nonlinear candidates promoted from the small screen to "
            "the complete fit/selection gate. Zero disables screening."
        ),
    )
    parser.add_argument("--lanczos-steps", type=int, default=4)
    parser.add_argument(
        "--record-fit-linear-capture",
        action="store_true",
        help=(
            "Run one extra fit JVP per ridge for diagnostics. This value does "
            "not participate in nonlinear candidate selection."
        ),
    )
    parser.add_argument(
        "--record-selection-linear-capture",
        action="store_true",
        help=(
            "Run one extra selection JVP per ridge for diagnostics. This value "
            "does not participate in nonlinear candidate selection."
        ),
    )
    parser.add_argument(
        "--reuse-krylov-lifts",
        action="store_true",
        help=(
            "Approximate ridge VJPs by combining VJPs cached during Lanczos. "
            "This is mathematically equivalent but can change complex64 "
            "roundoff, so it is disabled for production by default."
        ),
    )
    parser.add_argument(
        "--verify-krylov-cache",
        action="store_true",
        help=(
            "Recompute each lifted ridge direction explicitly and report the "
            "relative numerical discrepancy. Intended for smoke tests only."
        ),
    )
    parser.add_argument(
        "--separate-ridge-vjps",
        action="store_true",
        help=(
            "Apply each explicit ridge VJP in a separate forward pass. This "
            "debug-only reference path is used to verify the default "
            "shared-forward multi-ridge VJP."
        ),
    )
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
        default=(0.25, 0.5, 1.0),
    )
    parser.add_argument("--svd-relative-cutoff", type=float, default=1.0e-5)
    parser.add_argument("--expected-parameter-count", type=int, default=0)
    parser.add_argument("--seed", type=int, default=202607256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.data_limit,
        args.fit_size,
        args.selection_size,
        args.retained_rank,
        args.operator_chunk_size,
        args.eval_batch_size,
        args.prefix_cache_batch_size,
        args.lanczos_steps,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("sample, rank, chunk, batch, and Lanczos sizes are positive")
    if args.data_start < 0 or args.bond_index < 0 or args.oversample < 0:
        raise ValueError("data start, bond index, and oversampling cannot be negative")
    screening = (
        args.screen_fit_size,
        args.screen_selection_size,
        args.nonlinear_shortlist,
    )
    if any(value < 0 for value in screening):
        raise ValueError("screening sizes cannot be negative")
    if any(value > 0 for value in screening) and not all(
        value > 0 for value in screening
    ):
        raise ValueError("all screening controls must be positive or all zero")
    if args.expected_parameter_count < 0:
        raise ValueError("expected parameter count cannot be negative")
    if args.svd_relative_cutoff < 0:
        raise ValueError("SVD cutoff cannot be negative")
    if any(
        not np.isfinite(value) or value <= 0 for value in args.ridge_factors
    ):
        raise ValueError("ridge factors must be finite and positive")
    if any(
        not np.isfinite(value) or not 0 < value <= 1
        for value in args.step_scales
    ):
        raise ValueError("step scales must lie in (0, 1]")


def channel_mask_for_copy(
    model: torch.nn.Module,
    *,
    bond_index: int,
    source_multiplicity: int,
    target_multiplicity: int,
    copy_index: int,
    side: str,
) -> torch.Tensor:
    """Select one old-to-new or new-to-old compact multiplicity channel."""

    starts = tuple(int(value) for value in model.blocked_two_site_starts)
    if not 0 <= bond_index < len(starts) - 1:
        raise ValueError("bond index must lie between permanent degree-two blocks")
    if not source_multiplicity <= copy_index < target_multiplicity:
        raise ValueError("copy index must select a newly introduced multiplicity")
    if side not in {"left", "right"}:
        raise ValueError("channel side must be left or right")
    block_index = bond_index if side == "left" else bond_index + 1
    start = starts[block_index]
    parameter = model.blocked_two_site_orbit_parameters[block_index]
    left_boundary = start == 0
    right_boundary = start + 1 == model.site_count - 1
    if left_boundary or right_boundary:
        if parameter.numel() % target_multiplicity:
            raise ValueError("boundary compact block has an incompatible shape")
        view = parameter.reshape(-1, target_multiplicity)
        mask = torch.zeros_like(view, dtype=torch.bool)
        mask[:, copy_index] = True
        return mask.reshape(-1)

    stride = target_multiplicity**2
    if parameter.numel() % stride:
        raise ValueError("interior compact block has an incompatible shape")
    view = parameter.reshape(-1, target_multiplicity, target_multiplicity)
    mask = torch.zeros_like(view, dtype=torch.bool)
    if side == "left":
        mask[:, :source_multiplicity, copy_index] = True
    else:
        mask[:, copy_index, :source_multiplicity] = True
    return mask.reshape(-1)


def compact_orbit_scales(
    model: torch.nn.Module,
    block_index: int,
) -> torch.Tensor:
    """Return sqrt(dense orbit multiplicity) for every compact coefficient."""

    _, edge_labels_name, _ = model.blocked_two_site_sparse_metadata[block_index]
    labels = getattr(model, edge_labels_name)
    parameter = model.blocked_two_site_orbit_parameters[block_index]
    counts = torch.bincount(labels, minlength=parameter.numel()).to(
        dtype=parameter.real.dtype,
        device=parameter.device,
    )
    if not bool(torch.all(counts > 0)):
        raise ValueError("every compact orbit parameter must occur in the dense core")
    return torch.sqrt(counts)


@dataclass(frozen=True)
class ScaledMaskedComplexParameterVectorizer:
    """Expose dense-Frobenius-whitened entries of one compact parameter."""

    name: str
    shape: torch.Size
    flat_indices: torch.Tensor
    baseline: torch.Tensor
    selected_scales: torch.Tensor

    @classmethod
    def from_module(
        cls,
        module: torch.nn.Module,
        parameter_name: str,
        mask: torch.Tensor,
        scales: torch.Tensor,
    ) -> "ScaledMaskedComplexParameterVectorizer":
        parameters = dict(module.named_parameters())
        if parameter_name not in parameters:
            raise ValueError(f"unknown parameter: {parameter_name}")
        parameter = parameters[parameter_name]
        resolved_mask = mask.to(device=parameter.device, dtype=torch.bool)
        resolved_scales = scales.to(
            device=parameter.device,
            dtype=parameter.real.dtype,
        )
        if resolved_mask.shape != parameter.shape:
            raise ValueError("parameter mask has the wrong shape")
        if resolved_scales.shape != parameter.shape:
            raise ValueError("parameter scales have the wrong shape")
        indices = torch.nonzero(resolved_mask.reshape(-1), as_tuple=False).reshape(-1)
        selected_scales = torch.index_select(
            resolved_scales.reshape(-1),
            0,
            indices,
        )
        if indices.numel() == 0 or not bool(torch.all(selected_scales > 0)):
            raise ValueError("scaled parameter selection must be nonempty and positive")
        return cls(
            name=parameter_name,
            shape=parameter.shape,
            flat_indices=indices,
            baseline=parameter.detach().clone(),
            selected_scales=selected_scales,
        )

    def pack(self, module: torch.nn.Module) -> torch.Tensor:
        parameter = dict(module.named_parameters())[self.name]
        selected = torch.index_select(
            parameter.detach().reshape(-1),
            0,
            self.flat_indices,
        )
        return selected * self.selected_scales

    def unpack(self, vector: torch.Tensor) -> dict[str, torch.Tensor]:
        if vector.ndim != 1 or vector.numel() != self.flat_indices.numel():
            raise ValueError("scaled masked vector has the wrong shape")
        raw = vector / self.selected_scales
        value = self.baseline.reshape(-1).scatter(0, self.flat_indices, raw)
        return {self.name: value.reshape(self.shape)}

    def commit_(self, module: torch.nn.Module, vector: torch.Tensor) -> None:
        replacement = self.unpack(vector)[self.name]
        parameter = dict(module.named_parameters())[self.name]
        with torch.no_grad():
            parameter.copy_(replacement)


def normalized_transformed_inputs(
    model: torch.nn.Module,
    values: torch.Tensor,
    derivatives: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match the model's homogeneous normalization and dictionary transform."""

    h_values = model.reference_h @ values.unsqueeze(-1)
    reference_norm = torch.real(
        torch.sum(torch.conj(values) * h_values.squeeze(-1), dim=1)
    )
    if not bool(torch.all(reference_norm > 0)):
        raise FloatingPointError("reference section norm is not positive")
    inverse_scale = torch.rsqrt(reference_norm)
    normalized_values = values * inverse_scale[:, None]
    normalized_derivatives = derivatives * inverse_scale[:, None, None]
    source_inputs = torch.cat(
        (normalized_values.unsqueeze(-1), normalized_derivatives),
        dim=2,
    ).transpose(1, 2)
    transformed_inputs = torch.einsum(
        "qpi,nci->ncqp",
        model.physical_dictionary,
        source_inputs,
    )
    return normalized_values, normalized_derivatives, transformed_inputs


def sparse_block_inputs_with_parameter(
    model: torch.nn.Module,
    block_index: int,
    transformed_inputs: torch.Tensor,
    parameter: torch.Tensor,
) -> torch.Tensor:
    """Contract one compact blocked core without reading its module parameter."""

    edge_indices_name, edge_labels_name, shape = (
        model.blocked_two_site_sparse_metadata[block_index]
    )
    edge_indices = getattr(model, edge_indices_name)
    edge_labels = getattr(model, edge_labels_name)
    left_bond, right_bond, first_rank, second_rank = shape
    if transformed_inputs.ndim != 4:
        raise ValueError(
            "transformed two-site inputs must have shape "
            "(batch, channels, dictionary rank, outputs)"
        )
    if transformed_inputs.shape[2] != first_rank or first_rank != second_rank:
        raise ValueError("blocked core and transformed inputs disagree")

    values = transformed_inputs[:, 0]
    pair_features = torch.einsum(
        "ncxp,nyq->ncxypq",
        transformed_inputs,
        values,
    )
    pair_features = pair_features + torch.einsum(
        "nxp,ncyq->ncxypq",
        values,
        transformed_inputs,
    )
    channel_weights = torch.ones(
        transformed_inputs.shape[1],
        dtype=transformed_inputs.real.dtype,
        device=transformed_inputs.device,
    )
    channel_weights[0] = 0.5
    pair_features = pair_features * channel_weights[
        None, :, None, None, None, None
    ]
    batch_size, channel_count, _, _, first_output, second_output = (
        pair_features.shape
    )
    feature_matrix = pair_features.permute(2, 3, 0, 1, 4, 5).reshape(
        first_rank * second_rank,
        batch_size * channel_count * first_output * second_output,
    )
    row_indices = edge_indices[:, 0] * right_bond + edge_indices[:, 1]
    column_indices = edge_indices[:, 2] * second_rank + edge_indices[:, 3]
    weighted_features = parameter[edge_labels, None] * torch.index_select(
        feature_matrix,
        0,
        column_indices,
    )
    contracted = torch.zeros(
        (left_bond * right_bond, feature_matrix.shape[1]),
        dtype=parameter.dtype,
        device=parameter.device,
    ).index_add(0, row_indices, weighted_features)
    return (
        contracted.reshape(
            left_bond,
            right_bond,
            batch_size,
            channel_count,
            first_output,
            second_output,
        )
        .permute(2, 3, 0, 1, 4, 5)
        .flatten(start_dim=-2)
    )


def transport_feature_jet(
    jet: torch.Tensor,
    site_inputs: torch.Tensor,
) -> torch.Tensor:
    """Apply one value/first/mixed transfer exactly as the production model."""

    values = site_inputs[:, 0]
    derivative_sites = site_inputs[:, 1:]
    transported = torch.einsum(
        "nuvab,narp,nbsp->nuvrs",
        jet,
        torch.conj(values),
        values,
    )
    bra_updates = torch.einsum(
        "nvab,niarp,nbsp->nivrs",
        jet[:, 0],
        torch.conj(derivative_sites),
        values,
    )
    ket_updates = torch.einsum(
        "nuab,narp,njbsp->nujrs",
        jet[:, :, 0],
        torch.conj(values),
        derivative_sites,
    )
    mixed_updates = torch.einsum(
        "nab,niarp,njbsp->nijrs",
        jet[:, 0, 0],
        torch.conj(derivative_sites),
        derivative_sites,
    )
    batch_size = len(jet)
    channel_count = jet.shape[1]
    coordinate_count = channel_count - 1
    right_bond = values.shape[2]
    bra_updates = torch.cat(
        (
            bra_updates.new_zeros(
                batch_size,
                1,
                channel_count,
                right_bond,
                right_bond,
            ),
            bra_updates,
        ),
        dim=1,
    )
    ket_updates = torch.cat(
        (
            ket_updates.new_zeros(
                batch_size,
                channel_count,
                1,
                right_bond,
                right_bond,
            ),
            ket_updates,
        ),
        dim=2,
    )
    mixed_updates = torch.cat(
        (
            mixed_updates.new_zeros(
                batch_size,
                1,
                coordinate_count,
                right_bond,
                right_bond,
            ),
            mixed_updates,
        ),
        dim=1,
    )
    mixed_updates = torch.cat(
        (
            mixed_updates.new_zeros(
                batch_size,
                channel_count,
                1,
                right_bond,
                right_bond,
            ),
            mixed_updates,
        ),
        dim=2,
    )
    return transported + bra_updates + ket_updates + mixed_updates


def metric_from_feature_jet(
    model: torch.nn.Module,
    jet: torch.Tensor,
    normalized_values: torch.Tensor,
    normalized_derivatives: torch.Tensor,
) -> torch.Tensor:
    """Finish the reference floor and Kähler metric from a propagated jet."""

    learned_norm = torch.real(jet[:, 0, 0, 0, 0])
    learned_gradient = jet[:, 0, 1:, 0, 0]
    learned_mixed = jet[:, 1:, 1:, 0, 0]
    h_values = torch.einsum(
        "ab,nb->na",
        model.reference_h,
        normalized_values,
    )
    reference_norm = torch.real(
        torch.einsum(
            "na,na->n",
            torch.conj(normalized_values),
            h_values,
        )
    )
    h_derivatives = torch.einsum(
        "ab,nbj->naj",
        model.reference_h,
        normalized_derivatives,
    )
    reference_gradient = torch.einsum(
        "na,naj->nj",
        torch.conj(normalized_values),
        h_derivatives,
    )
    reference_mixed = torch.einsum(
        "nai,naj->nij",
        torch.conj(normalized_derivatives),
        h_derivatives,
    )
    power = model.site_count
    reference_power = reference_norm**power
    reference_power_gradient = (
        power
        * reference_norm[:, None] ** (power - 1)
        * reference_gradient
    )
    reference_power_mixed = (
        power
        * reference_norm[:, None, None] ** (power - 1)
        * reference_mixed
    )
    if power > 1:
        reference_power_mixed = reference_power_mixed + (
            power
            * (power - 1)
            * reference_norm[:, None, None] ** (power - 2)
            * torch.conj(reference_gradient)[:, :, None]
            * reference_gradient[:, None, :]
        )
    norm = learned_norm + model.positive_floor * reference_power
    gradient = (
        learned_gradient + model.positive_floor * reference_power_gradient
    )
    mixed = learned_mixed + model.positive_floor * reference_power_mixed
    metric = mixed / norm[:, None, None]
    metric = metric - (
        torch.conj(gradient)[:, :, None]
        * gradient[:, None, :]
        / norm[:, None, None] ** 2
    )
    metric = metric * model.target_normalization
    return 0.5 * (metric + torch.conj(torch.transpose(metric, 1, 2)))


class BondPairPrefixCache:
    """Host-memory prefix jets for repeated local bond-pair derivatives."""

    def __init__(
        self,
        model: torch.nn.Module,
        dataset: dict[str, Any],
        *,
        bond_index: int,
        batch_size: int,
        callback: Callable[[int, int], None] | None = None,
    ) -> None:
        starts = tuple(int(value) for value in model.blocked_two_site_starts)
        if bond_index < 0 or bond_index + 1 >= len(starts):
            raise ValueError("prefix cache bond index is outside the blocked chain")
        if batch_size <= 0:
            raise ValueError("prefix cache batch size must be positive")
        self.model = model
        self.dataset = dataset
        self.bond_index = int(bond_index)
        self.starts = starts
        self.device = dataset["values"].device
        self.normalized_values = torch.empty_like(dataset["values"])
        self.normalized_derivatives = torch.empty_like(dataset["derivatives"])
        transformed = None
        prefix = None

        with torch.no_grad():
            for start in range(0, dataset["count"], batch_size):
                stop = min(start + batch_size, dataset["count"])
                (
                    normalized_values,
                    normalized_derivatives,
                    transformed_inputs,
                ) = normalized_transformed_inputs(
                    model,
                    dataset["values"][start:stop],
                    dataset["derivatives"][start:stop],
                )
                if transformed is None:
                    transformed = torch.empty(
                        (dataset["count"],) + transformed_inputs.shape[1:],
                        dtype=transformed_inputs.dtype,
                        device=transformed_inputs.device,
                    )
                self.normalized_values[start:stop].copy_(normalized_values)
                self.normalized_derivatives[start:stop].copy_(
                    normalized_derivatives
                )
                transformed[start:stop].copy_(transformed_inputs)
                channel_count = transformed_inputs.shape[1]
                jet = torch.zeros(
                    (
                        stop - start,
                        channel_count,
                        channel_count,
                        1,
                        1,
                    ),
                    dtype=transformed_inputs.dtype,
                    device=transformed_inputs.device,
                )
                jet[:, 0, 0, 0, 0] = 1.0
                for block_index in range(self.bond_index):
                    parameter = model.blocked_two_site_orbit_parameters[
                        block_index
                    ]
                    site_inputs = sparse_block_inputs_with_parameter(
                        model,
                        block_index,
                        transformed_inputs,
                        parameter,
                    )
                    jet = transport_feature_jet(jet, site_inputs)
                if prefix is None:
                    prefix = torch.empty(
                        (dataset["count"],) + jet.shape[1:],
                        dtype=jet.dtype,
                        device="cpu",
                    )
                prefix[start:stop].copy_(jet.detach().cpu())
                if callback is not None:
                    callback(stop, dataset["count"])
        if transformed is None or prefix is None:
            raise ValueError("cannot cache an empty fit dataset")
        self.transformed_inputs = transformed
        self.prefix = prefix

    @property
    def host_bytes(self) -> int:
        return self.prefix.numel() * self.prefix.element_size()

    def raw_function(
        self,
        selection: slice,
        vectorizer: ScaledMaskedComplexParameterVectorizer,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        try:
            variable_block = int(vectorizer.name.rsplit(".", 1)[1])
        except (IndexError, ValueError) as error:
            raise ValueError("cannot identify the variable compact block") from error
        if variable_block not in (self.bond_index, self.bond_index + 1):
            raise ValueError("variable block is outside the cached bond pair")
        normalized_values = self.normalized_values[selection]
        normalized_derivatives = self.normalized_derivatives[selection]
        transformed_inputs = self.transformed_inputs[selection]
        prefix = self.prefix[selection].to(device=self.device)
        log_omega = self.dataset["log_omega"][selection]

        def raw(parameter_vector: torch.Tensor) -> torch.Tensor:
            parameter_override = vectorizer.unpack(parameter_vector)[
                vectorizer.name
            ]
            jet = prefix
            for block_index in range(self.bond_index, len(self.starts)):
                parameter = (
                    parameter_override
                    if block_index == variable_block
                    else self.model.blocked_two_site_orbit_parameters[block_index]
                )
                site_inputs = sparse_block_inputs_with_parameter(
                    self.model,
                    block_index,
                    transformed_inputs,
                    parameter,
                )
                jet = transport_feature_jet(jet, site_inputs)
            metric = metric_from_feature_jet(
                self.model,
                jet,
                normalized_values,
                normalized_derivatives,
            )
            return training_log_volume(metric, "cholesky") - log_omega

        return raw


def evaluate_raw_torch(
    model: torch.nn.Module,
    dataset: dict[str, Any],
    *,
    batch_size: int,
) -> torch.Tensor:
    rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, dataset["count"], batch_size):
            stop = min(start + batch_size, dataset["count"])
            metric = model(
                dataset["values"][start:stop],
                dataset["derivatives"][start:stop],
            )
            rows.append(
                training_log_volume(metric, "cholesky")
                - dataset["log_omega"][start:stop]
            )
    return torch.cat(rows)


def native_e2_raw_cotangent(
    model: torch.nn.Module,
    dataset: dict[str, Any],
    *,
    batch_size: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return d(E2/2)/d(raw log volume), including d(log Z)."""

    raw = evaluate_raw_torch(model, dataset, batch_size=batch_size)
    weights = dataset["weights"]
    log_kappa = weighted_log_mean_exp(raw, weights)
    ratio = torch.exp(raw - log_kappa)
    probability = weights * ratio
    residual = torch.sqrt(weights) * (ratio - 1.0)
    scaled = torch.sqrt(weights) * ratio * residual
    raw_cotangent = scaled - probability * torch.sum(scaled)
    return raw_cotangent.detach(), {
        "log_kappa": float(log_kappa),
        "e2": float(real_inner(residual, residual)),
        "chi": float(torch.sqrt(real_inner(residual, residual))),
        "normalization_mass": float(torch.sum(probability)),
    }


def raw_log_volume_vjp(
    model: torch.nn.Module,
    vectorizer: ScaledMaskedComplexParameterVectorizer,
    theta: torch.Tensor,
    dataset: dict[str, Any],
    raw_cotangent: torch.Tensor,
    *,
    chunk_size: int,
    raw_function_factory: (
        Callable[
            [slice, ScaledMaskedComplexParameterVectorizer],
            Callable[[torch.Tensor], torch.Tensor],
        ]
        | None
    ) = None,
) -> torch.Tensor:
    """Apply J_raw^* to a fixed real cotangent in whitened coordinates."""

    result = torch.zeros_like(theta)
    offset = 0
    for start in range(0, dataset["count"], chunk_size):
        stop = min(start + chunk_size, dataset["count"])
        selection = slice(start, stop)
        if raw_function_factory is None:
            values = dataset["values"][selection]
            derivatives = dataset["derivatives"][selection]
            log_omega = dataset["log_omega"][selection]

            def raw(parameter_vector: torch.Tensor) -> torch.Tensor:
                metric = functional_call(
                    model,
                    vectorizer.unpack(parameter_vector),
                    (values, derivatives),
                    strict=False,
                )
                return training_log_volume(metric, "cholesky") - log_omega

        else:
            raw = raw_function_factory(selection, vectorizer)

        _, pullback = vjp(raw, theta)
        result = result + pullback(raw_cotangent[offset:stop])[0]
        offset = stop
    return result


def randomized_supercore_svd(
    apply: Callable[[torch.Tensor], torch.Tensor],
    adjoint: Callable[[torch.Tensor], torch.Tensor],
    *,
    right_dimension: int,
    retained_rank: int,
    oversample: int,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
    callback: Callable[[str, int, int], None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Approximate the leading SVD using the full matrix-free operator."""

    sketch_rank = min(right_dimension, retained_rank + oversample)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    omega = torch.randn(
        right_dimension,
        sketch_rank,
        generator=generator,
        dtype=real_dtype,
        device=device,
    )
    if dtype.is_complex:
        omega = omega + 1j * torch.randn(
            omega.shape,
            generator=generator,
            dtype=real_dtype,
            device=device,
        )
        omega = omega / math.sqrt(2.0)
    omega = torch.linalg.qr(omega.to(dtype=dtype), mode="reduced").Q

    range_rows = []
    for index in range(sketch_rank):
        if callback is not None:
            callback("range", index + 1, sketch_rank)
        range_rows.append(apply(omega[:, index]))
    sample = torch.stack(range_rows, dim=1)
    basis = torch.linalg.qr(sample, mode="reduced").Q

    adjoint_rows = []
    for index in range(basis.shape[1]):
        if callback is not None:
            callback("adjoint", index + 1, basis.shape[1])
        adjoint_rows.append(adjoint(basis[:, index]))
    projected_adjoint = torch.stack(adjoint_rows, dim=1)
    projected = torch.conj(projected_adjoint).transpose(0, 1)
    small_left, singular_values, right_adjoint = torch.linalg.svd(
        projected,
        full_matrices=False,
    )
    left = basis @ small_left
    right = torch.conj(right_adjoint).transpose(0, 1)
    retained = min(retained_rank, singular_values.numel())
    diagnostics = {
        "sketch_rank": int(sketch_rank),
        "retained_rank": int(retained),
        "range_orthogonality_error": float(
            torch.linalg.vector_norm(
                torch.conj(basis).transpose(0, 1) @ basis
                - torch.eye(
                    basis.shape[1],
                    dtype=basis.dtype,
                    device=basis.device,
                )
            )
        ),
        "projected_singular_values": [
            float(value) for value in singular_values.detach().cpu()
        ],
    }
    return (
        left[:, :retained],
        singular_values[:retained],
        right[:, :retained],
        diagnostics,
    )


def masks_and_vectorizer(
    model: torch.nn.Module,
    *,
    bond_index: int,
    source_multiplicity: int,
    target_multiplicity: int,
    copy_indices: tuple[int, ...],
    side: str,
) -> tuple[list[torch.Tensor], ScaledMaskedComplexParameterVectorizer, torch.Tensor]:
    block_index = bond_index if side == "left" else bond_index + 1
    parameter_name = f"blocked_two_site_orbit_parameters.{block_index}"
    parameter = dict(model.named_parameters())[parameter_name]
    masks = [
        channel_mask_for_copy(
            model,
            bond_index=bond_index,
            source_multiplicity=source_multiplicity,
            target_multiplicity=target_multiplicity,
            copy_index=copy_index,
            side=side,
        ).reshape(parameter.shape)
        for copy_index in copy_indices
    ]
    union = torch.stack(masks).any(dim=0)
    scales = compact_orbit_scales(model, block_index).reshape(parameter.shape)
    vectorizer = ScaledMaskedComplexParameterVectorizer.from_module(
        model,
        parameter_name,
        union,
        scales,
    )
    return masks, vectorizer, scales


def set_whitened_channel_(
    parameter: torch.Tensor,
    mask: torch.Tensor,
    scales: torch.Tensor,
    values: torch.Tensor,
) -> None:
    selected_scales = scales[mask]
    if values.shape != selected_scales.shape:
        raise ValueError("whitened channel vector has the wrong shape")
    with torch.no_grad():
        parameter[mask] = values / selected_scales


def orthonormal_right_seed_subspace(
    parameter: torch.Tensor,
    masks: list[torch.Tensor],
    scales: torch.Tensor,
) -> tuple[torch.Tensor, list[float], float]:
    """Extract stored whitened channels and return their conjugate QR span."""

    seed_columns = []
    seed_norms = []
    for mask in masks:
        stored_white = parameter[mask] * scales[mask]
        norm = torch.linalg.vector_norm(stored_white)
        if not bool(torch.isfinite(norm)) or float(norm) <= 0:
            raise ValueError("right seed contains an empty retained channel")
        seed_columns.append(torch.conj(stored_white))
        seed_norms.append(float(norm))
    seed_matrix = torch.stack(seed_columns, dim=1)
    subspace = torch.linalg.qr(seed_matrix, mode="reduced").Q
    orthogonality_error = float(
        torch.linalg.vector_norm(
            torch.conj(subspace).transpose(0, 1) @ subspace
            - torch.eye(
                subspace.shape[1],
                dtype=subspace.dtype,
                device=subspace.device,
            )
        )
    )
    return subspace, seed_norms, orthogonality_error


def tail_nonworse(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    keys = ("q999", "cvar99", "cvar999", "maximum", "ratio_max")
    return bool(all(candidate[key] <= baseline[key] for key in keys))


def capture(baseline: float, candidate: float) -> float:
    return 1.0 - candidate / max(baseline, np.finfo(float).tiny)


def renormalized_dataset_prefix(
    dataset: dict[str, Any],
    count: int,
) -> dict[str, Any]:
    """Take a deterministic development prefix with separately normalized weights."""

    if count <= 0 or count > dataset["count"]:
        raise ValueError("screen prefix size is outside its dataset")
    weights = dataset["weights"][:count]
    weights = weights / torch.sum(weights)
    weights_numpy = dataset["weights_numpy"][:count]
    weights_numpy = weights_numpy / np.sum(weights_numpy)
    return {
        **dataset,
        "count": count,
        "values": dataset["values"][:count],
        "derivatives": dataset["derivatives"][:count],
        "log_omega": dataset["log_omega"][:count],
        "weights": weights,
        "weights_numpy": weights_numpy,
    }


def nonlinear_screen_shortlist(
    rows: list[dict[str, Any]],
    limit: int,
) -> list[int]:
    """Rank positive screen candidates without treating the screen as a gate."""

    if limit <= 0:
        raise ValueError("nonlinear shortlist size must be positive")
    positive = [
        index
        for index, row in enumerate(rows)
        if bool(row["screen"]["positive_metric"])
    ]
    ordered = sorted(
        positive,
        key=lambda index: (
            rows[index]["screen"]["selection"]["e2"],
            rows[index]["screen"]["fit"]["e2"],
            rows[index]["screen"]["selection"]["sigma"],
            index,
        ),
    )
    return ordered[:limit]


def main() -> None:
    args = parse_args()
    validate_args(args)
    started = time.perf_counter()
    output_model = args.output_model.expanduser().resolve()
    output_report = args.output_report.expanduser().resolve()
    status = output_report.with_suffix(output_report.suffix + ".status.json")
    indices_path = output_report.with_suffix(output_report.suffix + ".indices.npz")
    if (
        output_model.exists()
        or output_report.exists()
        or status.exists()
        or indices_path.exists()
    ):
        raise FileExistsError("refusing to overwrite a supercore artifact")
    output_model.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    write_json(status, {"state": "running", "phase": "loading"})

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model_path = args.expanded_model.expanduser().resolve()
    right_seed_path = (
        args.right_seed_model.expanduser().resolve()
        if args.right_seed_model is not None
        else None
    )
    source_dir = args.source_run_dir.expanduser().resolve()
    pullbacks_dir = args.pullbacks_dir.expanduser().resolve()
    dataset_path = source_dir / "training_data" / "dataset.npz"
    pullbacks_path = pullbacks_dir / "train_pullbacks.npy"
    for path in (model_path, dataset_path, pullbacks_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if right_seed_path is not None and not right_seed_path.exists():
        raise FileNotFoundError(right_seed_path)
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    expansion = payload.get("phase_multiplicity_expansion")
    if not isinstance(expansion, dict):
        raise ValueError("input model is not an exact multiplicity expansion")
    source_multiplicity = int(expansion["source_multiplicity"])
    target_multiplicity = int(expansion["target_multiplicity"])
    available_rank = target_multiplicity - source_multiplicity
    if args.retained_rank > available_rank:
        raise ValueError("expanded multiplicity has too few new supercore channels")
    copy_indices = tuple(
        range(
            source_multiplicity,
            source_multiplicity + args.retained_rank,
        )
    )
    source_degree = int(payload.get("source_degree", 1))
    reference_h = np.asarray(
        payload["state_dict"]["reference_h"].detach().cpu(),
        dtype=np.complex128,
    )
    complex_dtype = (
        torch.complex64
        if str(payload["precision"]) == "complex64"
        else torch.complex128
    )
    real_dtype = (
        torch.float32 if complex_dtype == torch.complex64 else torch.float64
    )

    data = np.load(dataset_path, allow_pickle=False)
    data_stop = min(args.data_start + args.data_limit, len(data["X_train"]))
    data_count = data_stop - args.data_start
    if data_count <= 0:
        raise ValueError("training-domain candidate window is empty")
    local_fit, local_selection = select_disjoint_indices(
        data_count,
        args.fit_size,
        args.selection_size,
        seed=args.seed,
    )
    fit_indices = local_fit + args.data_start
    selection_indices = local_selection + args.data_start
    np.savez_compressed(
        indices_path,
        fit_train_indices=fit_indices,
        selection_train_indices=selection_indices,
    )
    pullbacks = np.load(pullbacks_path, mmap_mode="r")

    def make_split(indices: np.ndarray) -> dict[str, Any]:
        return tensor_split(
            np.asarray(data["X_train"][indices], dtype=np.float32),
            np.asarray(pullbacks[indices]),
            np.asarray(data["y_train"][indices], dtype=np.float64),
            source_degree=source_degree,
            complex_dtype=complex_dtype,
            real_dtype=real_dtype,
            device=device,
        )

    fit = make_split(fit_indices)
    selection = make_split(selection_indices)
    screening_enabled = args.nonlinear_shortlist > 0
    if screening_enabled and (
        args.screen_fit_size > fit["count"]
        or args.screen_selection_size > selection["count"]
    ):
        raise ValueError("screening prefixes exceed fit or selection data")
    model = positive_tensor_network_from_artifact_payload(
        reference_h,
        payload,
        device=device,
        trainable_physical_dictionary=False,
    ).to(device=device, dtype=complex_dtype)
    model.force_indexed_block_contraction = True
    parameter_count = int(model.trainable_real_parameter_count)
    if args.expected_parameter_count and parameter_count != args.expected_parameter_count:
        raise RuntimeError(
            f"parameter-count gate failed: {parameter_count} != "
            f"{args.expected_parameter_count}"
        )
    model.requires_grad_(False).eval()

    left_masks, left_single, left_scales = masks_and_vectorizer(
        model,
        bond_index=args.bond_index,
        source_multiplicity=source_multiplicity,
        target_multiplicity=target_multiplicity,
        copy_indices=(copy_indices[0],),
        side="left",
    )
    right_masks, right_single, right_scales = masks_and_vectorizer(
        model,
        bond_index=args.bond_index,
        source_multiplicity=source_multiplicity,
        target_multiplicity=target_multiplicity,
        copy_indices=(copy_indices[0],),
        side="right",
    )
    left_parameter = dict(model.named_parameters())[left_single.name]
    right_parameter = dict(model.named_parameters())[right_single.name]
    left_zero = left_single.pack(model)
    right_zero = right_single.pack(model)
    if bool(torch.any(torch.abs(left_zero) > 0)) or bool(
        torch.any(torch.abs(right_zero) > 0)
    ):
        raise ValueError("supercore work channel must be zero in the exact embedding")

    fit_prefix_cache = None
    if args.cache_fit_prefix:
        write_json(
            status,
            {
                "state": "running",
                "phase": "fit_prefix_cache",
                "index": 0,
                "count": fit["count"],
            },
        )

        def prefix_progress(index: int, count: int) -> None:
            write_json(
                status,
                {
                    "state": "running",
                    "phase": "fit_prefix_cache",
                    "index": index,
                    "count": count,
                },
            )
            if not args.quiet:
                print(f"fit_prefix_cache={index}/{count}", flush=True)

        fit_prefix_cache = BondPairPrefixCache(
            model,
            fit,
            bond_index=args.bond_index,
            batch_size=args.prefix_cache_batch_size,
            callback=prefix_progress,
        )
    fit_raw_function_factory = (
        fit_prefix_cache.raw_function
        if fit_prefix_cache is not None
        else None
    )

    raw_cotangent, fit_native = native_e2_raw_cotangent(
        model,
        fit,
        batch_size=args.eval_batch_size,
    )

    def clear_work_channel() -> None:
        left_single.commit_(model, left_zero)
        right_single.commit_(model, right_zero)

    def apply_supercore(right_vector: torch.Tensor) -> torch.Tensor:
        clear_work_channel()
        right_single.commit_(model, torch.conj(right_vector))
        result = raw_log_volume_vjp(
            model,
            left_single,
            left_zero,
            fit,
            raw_cotangent,
            chunk_size=args.operator_chunk_size,
            raw_function_factory=fit_raw_function_factory,
        )
        clear_work_channel()
        return result

    def apply_supercore_adjoint(left_vector: torch.Tensor) -> torch.Tensor:
        clear_work_channel()
        left_single.commit_(model, left_vector)
        result = raw_log_volume_vjp(
            model,
            right_single,
            right_zero,
            fit,
            raw_cotangent,
            chunk_size=args.operator_chunk_size,
            raw_function_factory=fit_raw_function_factory,
        )
        clear_work_channel()
        return torch.conj(result)

    def svd_progress(phase: str, index: int, count: int) -> None:
        write_json(
            status,
            {
                "state": "running",
                "phase": f"supercore_svd_{phase}",
                "index": index,
                "count": count,
            },
        )
        if not args.quiet:
            print(f"supercore_svd_{phase}={index}/{count}", flush=True)

    all_left_masks, left_vectorizer, _ = masks_and_vectorizer(
        model,
        bond_index=args.bond_index,
        source_multiplicity=source_multiplicity,
        target_multiplicity=target_multiplicity,
        copy_indices=copy_indices,
        side="left",
    )
    all_right_masks, _, _ = masks_and_vectorizer(
        model,
        bond_index=args.bond_index,
        source_multiplicity=source_multiplicity,
        target_multiplicity=target_multiplicity,
        copy_indices=copy_indices,
        side="right",
    )
    if right_seed_path is None:
        (
            supercore_left,
            supercore_singular,
            supercore_right,
            svd_diagnostics,
        ) = randomized_supercore_svd(
            apply_supercore,
            apply_supercore_adjoint,
            right_dimension=right_zero.numel(),
            retained_rank=args.retained_rank,
            oversample=args.oversample,
            seed=args.seed + 1009,
            dtype=complex_dtype,
            device=device,
            callback=svd_progress,
        )
    else:
        write_json(
            status,
            {"state": "running", "phase": "right_seed_subspace"},
        )
        seed_payload = torch.load(
            right_seed_path,
            map_location="cpu",
            weights_only=False,
        )
        seed_expansion = seed_payload.get("phase_multiplicity_expansion")
        if not isinstance(seed_expansion, dict) or int(
            seed_expansion["target_multiplicity"]
        ) != target_multiplicity:
            raise ValueError("right seed has incompatible target multiplicity")
        seed_state = seed_payload.get("state_dict")
        if not isinstance(seed_state, dict) or right_single.name not in seed_state:
            raise ValueError("right seed is missing the compact block parameter")
        seed_parameter = seed_state[right_single.name].to(
            device=device,
            dtype=right_parameter.dtype,
        )
        if seed_parameter.shape != right_parameter.shape:
            raise ValueError("right seed compact block has the wrong shape")
        right_scale_tensor = right_scales.reshape(right_parameter.shape)
        (
            supercore_right,
            seed_norms,
            seed_orthogonality_error,
        ) = orthonormal_right_seed_subspace(
            seed_parameter,
            all_right_masks,
            right_scale_tensor,
        )
        supercore_left = torch.zeros(
            (
                left_zero.numel(),
                supercore_right.shape[1],
            ),
            dtype=complex_dtype,
            device=device,
        )
        supercore_singular = torch.as_tensor(
            seed_norms,
            dtype=real_dtype,
            device=device,
        )
        svd_diagnostics = {
            "method": "orthonormalized_right_subspace_from_seed_model",
            "seed_model": str(right_seed_path),
            "seed_model_sha256": sha256_file(right_seed_path),
            "seed_channel_norms": seed_norms,
            "retained_rank": int(supercore_right.shape[1]),
            "range_orthogonality_error": seed_orthogonality_error,
            "projected_singular_values": seed_norms,
        }
    clear_work_channel()

    for column, mask in enumerate(all_right_masks):
        set_whitened_channel_(
            right_parameter,
            mask,
            right_scales.reshape(right_parameter.shape),
            torch.conj(supercore_right[:, column]),
        )
    theta = left_vectorizer.pack(model)
    if bool(torch.any(torch.abs(theta) > 0)):
        raise RuntimeError("left supercore coordinates must start at exact zero")

    write_json(status, {"state": "running", "phase": "native_gn_operators"})
    fit_operator = MatrixFreeNativeMARatioJacobian(
        model,
        left_vectorizer,
        theta,
        fit,
        chunk_size=args.operator_chunk_size,
        raw_function_factory=fit_raw_function_factory,
    )
    selection_operator = MatrixFreeNativeMARatioJacobian(
        model,
        left_vectorizer,
        theta,
        selection,
        chunk_size=args.operator_chunk_size,
    )
    fit_residual = fit_operator.residual()
    selection_residual = selection_operator.residual()
    fit_e2 = float(real_inner(fit_residual, fit_residual))
    selection_e2 = float(real_inner(selection_residual, selection_residual))
    gradient = fit_operator.vjp(fit_residual)
    gradient_norm = float(vector_norm(gradient))
    rayleigh_scale = gradient_norm**2 / max(fit_e2, np.finfo(float).tiny)
    if not np.isfinite(rayleigh_scale) or rayleigh_scale <= 0:
        raise FloatingPointError("supercore right subspace has no finite GN scale")

    def gn_progress(iteration: int, alpha: float, beta: float) -> None:
        write_json(
            status,
            {
                "state": "running",
                "phase": "native_gn_lanczos",
                "iteration": iteration,
                "count": args.lanczos_steps,
            },
        )
        if not args.quiet:
            print(
                f"native_gn_lanczos={iteration}/{args.lanczos_steps} "
                f"alpha={alpha:.6e} beta={beta:.6e}",
                flush=True,
            )

    cache_krylov_lifts = bool(
        args.reuse_krylov_lifts or args.verify_krylov_cache
    )
    parameter_basis_images: list[torch.Tensor] = []
    normal_basis_images: list[torch.Tensor] = []

    def cached_fit_normal(cotangent: torch.Tensor) -> torch.Tensor:
        parameter_image = fit_operator.vjp(cotangent)
        normal_image = fit_operator.jvp(parameter_image)
        if cache_krylov_lifts:
            parameter_basis_images.append(parameter_image.detach())
            normal_basis_images.append(normal_image.detach())
        return normal_image

    lanczos = lanczos_tridiagonal(
        cached_fit_normal,
        fit_residual,
        steps=args.lanczos_steps,
        callback=gn_progress,
    )
    parameter_basis_matrix = None
    normal_basis_matrix = None
    if cache_krylov_lifts:
        if len(parameter_basis_images) != lanczos.tridiagonal.shape[0]:
            raise RuntimeError("cached Krylov lifts do not match the Lanczos basis")
        parameter_basis_matrix = torch.stack(parameter_basis_images, dim=1)
        normal_basis_matrix = torch.stack(normal_basis_images, dim=1)
    directions = []
    krylov_cache_verification = []
    ridge_entries = []
    for ridge_factor in args.ridge_factors:
        ridge = float(ridge_factor * rayleigh_scale)
        coefficients = ridge_coefficients_from_lanczos(lanczos, ridge)
        dual = lanczos.basis @ coefficients
        ridge_entries.append(
            {
                "ridge_factor": float(ridge_factor),
                "ridge": ridge,
                "coefficients": coefficients,
                "dual": dual,
            }
        )
    explicit_deltas = None
    if not args.reuse_krylov_lifts or args.verify_krylov_cache:
        if args.separate_ridge_vjps:
            explicit_deltas = torch.stack(
                [-fit_operator.vjp(entry["dual"]) for entry in ridge_entries],
                dim=0,
            )
        else:
            explicit_deltas = -fit_operator.vjp_many(
                torch.stack(
                    [entry["dual"] for entry in ridge_entries],
                    dim=0,
                )
            )

    for ridge_index, entry in enumerate(ridge_entries):
        ridge_factor = entry["ridge_factor"]
        ridge = entry["ridge"]
        coefficients = entry["coefficients"]
        cached_delta = None
        cached_fit_tangent = None
        if cache_krylov_lifts:
            assert parameter_basis_matrix is not None
            assert normal_basis_matrix is not None
            cached_delta = -(
                parameter_basis_matrix
                @ coefficients.to(dtype=parameter_basis_matrix.dtype)
            )
            cached_fit_tangent = -(normal_basis_matrix @ coefficients)
        explicit_delta = (
            explicit_deltas[ridge_index]
            if explicit_deltas is not None
            else None
        )
        delta = cached_delta if args.reuse_krylov_lifts else explicit_delta
        if delta is None:
            raise RuntimeError("no ridge parameter direction was constructed")
        fit_tangent = (
            fit_operator.jvp(delta)
            if args.record_fit_linear_capture
            else None
        )
        selection_tangent = (
            selection_operator.jvp(delta)
            if args.record_selection_linear_capture
            else None
        )
        if args.verify_krylov_cache:
            assert cached_delta is not None
            assert cached_fit_tangent is not None
            assert explicit_delta is not None
            explicit_fit_tangent = fit_operator.jvp(explicit_delta)
            delta_relative_error = float(
                vector_norm(cached_delta - explicit_delta)
                / torch.clamp(vector_norm(explicit_delta), min=1.0e-30)
            )
            tangent_relative_error = float(
                vector_norm(cached_fit_tangent - explicit_fit_tangent)
                / torch.clamp(
                    vector_norm(explicit_fit_tangent),
                    min=1.0e-30,
                )
            )
            krylov_cache_verification.append(
                {
                    "ridge_factor": float(ridge_factor),
                    "delta_relative_error": delta_relative_error,
                    "fit_tangent_relative_error": tangent_relative_error,
                }
            )
        directions.append(
            {
                "ridge_factor": float(ridge_factor),
                "ridge": ridge,
                "delta": delta,
                "fit_tangent": fit_tangent,
                "selection_tangent": selection_tangent,
            }
        )

    fit_baseline = model_summary(
        model,
        fit,
        batch_size=args.eval_batch_size,
    )
    selection_baseline = model_summary(
        model,
        selection,
        batch_size=args.eval_batch_size,
    )
    screen_fit = (
        renormalized_dataset_prefix(fit, args.screen_fit_size)
        if screening_enabled
        else None
    )
    screen_selection = (
        renormalized_dataset_prefix(
            selection,
            args.screen_selection_size,
        )
        if screening_enabled
        else None
    )
    screen_fit_baseline = (
        model_summary(
            model,
            screen_fit,
            batch_size=args.eval_batch_size,
        )
        if screen_fit is not None
        else None
    )
    screen_selection_baseline = (
        model_summary(
            model,
            screen_selection,
            batch_size=args.eval_batch_size,
        )
        if screen_selection is not None
        else None
    )
    rows = []
    deltas = []
    write_json(status, {"state": "running", "phase": "nonlinear_candidate_screen"})
    for direction in directions:
        for step_scale in args.step_scales:
            delta = float(step_scale) * direction["delta"]
            linear_fit = (
                fit_residual + float(step_scale) * direction["fit_tangent"]
                if direction["fit_tangent"] is not None
                else None
            )
            linear_selection = (
                selection_residual
                + float(step_scale) * direction["selection_tangent"]
                if direction["selection_tangent"] is not None
                else None
            )
            screen = None
            if screening_enabled:
                assert screen_fit is not None
                assert screen_selection is not None
                assert screen_fit_baseline is not None
                assert screen_selection_baseline is not None
                left_vectorizer.commit_(model, theta + delta)
                screen_fit_trial = model_summary(
                    model,
                    screen_fit,
                    batch_size=args.eval_batch_size,
                )
                screen_selection_trial = model_summary(
                    model,
                    screen_selection,
                    batch_size=args.eval_batch_size,
                )
                screen = {
                    "fit": screen_fit_trial,
                    "selection": screen_selection_trial,
                    "fit_actual_capture": capture(
                        screen_fit_baseline["e2"],
                        screen_fit_trial["e2"],
                    ),
                    "selection_actual_capture": capture(
                        screen_selection_baseline["e2"],
                        screen_selection_trial["e2"],
                    ),
                    "positive_metric": bool(
                        screen_fit_trial["minimum_metric_eigenvalue"] > 0
                        and screen_selection_trial["minimum_metric_eigenvalue"] > 0
                    ),
                }
                left_vectorizer.commit_(model, theta)
            row = {
                "ridge_factor": direction["ridge_factor"],
                "ridge": direction["ridge"],
                "step_scale": float(step_scale),
                "step_rms": float(torch.sqrt(torch.mean(torch.abs(delta) ** 2))),
                "fit_linear_capture": (
                    capture(
                        fit_e2,
                        float(real_inner(linear_fit, linear_fit)),
                    )
                    if linear_fit is not None
                    else None
                ),
                "selection_linear_capture": (
                    capture(
                        selection_e2,
                        float(real_inner(linear_selection, linear_selection)),
                    )
                    if linear_selection is not None
                    else None
                ),
                "screen": screen,
                "screen_shortlisted": not screening_enabled,
                "fit_actual_capture": None,
                "selection_actual_capture": None,
                "fit": None,
                "selection": None,
                "positive_metric": None,
                "selection_tail_safe": None,
                "eligible": False,
            }
            rows.append(row)
            deltas.append(delta.detach().clone())
            if screen is not None:
                print(
                    f"screen ridge_factor={direction['ridge_factor']:.6g} "
                    f"step={step_scale:.6g} "
                    f"fit_E2_capture={screen['fit_actual_capture']:.6f} "
                    f"selection_E2_capture="
                    f"{screen['selection_actual_capture']:.6f} "
                    f"selection_sigma={screen['selection']['sigma']:.8e} "
                    f"positive={screen['positive_metric']}",
                    flush=True,
                )

    if screening_enabled:
        shortlisted_indices = nonlinear_screen_shortlist(
            rows,
            args.nonlinear_shortlist,
        )
    else:
        shortlisted_indices = list(range(len(rows)))
    shortlisted_set = set(shortlisted_indices)
    for index, row in enumerate(rows):
        row["screen_shortlisted"] = index in shortlisted_set

    write_json(
        status,
        {
            "state": "running",
            "phase": "tail_safe_line_search",
            "shortlisted": shortlisted_indices,
        },
    )
    for index in shortlisted_indices:
        row = rows[index]
        delta = deltas[index]
        left_vectorizer.commit_(model, theta + delta)
        fit_trial = model_summary(
            model,
            fit,
            batch_size=args.eval_batch_size,
        )
        selection_trial = model_summary(
            model,
            selection,
            batch_size=args.eval_batch_size,
        )
        fit_actual_capture = capture(fit_e2, fit_trial["e2"])
        selection_actual_capture = capture(
            selection_e2,
            selection_trial["e2"],
        )
        positive = bool(
            fit_trial["minimum_metric_eigenvalue"] > 0
            and selection_trial["minimum_metric_eigenvalue"] > 0
        )
        selection_tail_safe = tail_nonworse(
            selection_trial,
            selection_baseline,
        )
        eligible = bool(
            positive
            and fit_actual_capture > 0
            and selection_actual_capture > 0
            and selection_trial["sigma"] <= selection_baseline["sigma"]
            and selection_tail_safe
        )
        row.update(
            {
                "fit_actual_capture": fit_actual_capture,
                "selection_actual_capture": selection_actual_capture,
                "fit": fit_trial,
                "selection": selection_trial,
                "positive_metric": positive,
                "selection_tail_safe": selection_tail_safe,
                "eligible": eligible,
            }
        )
        print(
            f"ridge_factor={row['ridge_factor']:.6g} "
            f"step={row['step_scale']:.6g} "
            f"fit_E2_capture={fit_actual_capture:.6f} "
            f"selection_E2_capture={selection_actual_capture:.6f} "
            f"selection_sigma={selection_trial['sigma']:.8e} "
            f"tail_safe={selection_tail_safe} eligible={eligible}",
            flush=True,
        )
        left_vectorizer.commit_(model, theta)

    eligible_indices = [
        index for index, row in enumerate(rows) if bool(row["eligible"])
    ]
    selected_index = (
        min(
            eligible_indices,
            key=lambda index: (
                rows[index]["selection"]["e2"],
                rows[index]["selection"]["sigma"],
            ),
        )
        if eligible_indices
        else None
    )
    accepted = selected_index is not None
    selected = None if selected_index is None else rows[selected_index]
    compression = None
    epoch_zero_svd_change = None
    if accepted:
        left_vectorizer.commit_(model, theta + deltas[selected_index])
        left_columns = []
        right_columns = []
        left_scale_tensor = left_scales.reshape(left_parameter.shape)
        right_scale_tensor = right_scales.reshape(right_parameter.shape)
        for left_mask, right_mask in zip(
            all_left_masks,
            all_right_masks,
            strict=True,
        ):
            left_columns.append(
                left_parameter[left_mask] * left_scale_tensor[left_mask]
            )
            right_columns.append(
                right_parameter[right_mask] * right_scale_tensor[right_mask]
            )
        left_matrix = torch.stack(left_columns, dim=1)
        right_matrix = torch.stack(right_columns, dim=1)
        supercore_update = left_matrix @ torch.transpose(right_matrix, 0, 1)
        compressed_left, singular_values, compressed_right_adjoint = (
            torch.linalg.svd(supercore_update, full_matrices=False)
        )
        threshold = args.svd_relative_cutoff * float(singular_values[0])
        retained = max(
            1,
            min(
                args.retained_rank,
                int(torch.count_nonzero(singular_values > threshold)),
            ),
        )
        compressed_update = (
            compressed_left[:, :retained]
            @ torch.diag(singular_values[:retained]).to(complex_dtype)
            @ compressed_right_adjoint[:retained]
        )
        update_norm = torch.linalg.vector_norm(supercore_update)
        relative_reconstruction_error = float(
            torch.linalg.vector_norm(compressed_update - supercore_update)
            / torch.clamp(update_norm, min=torch.finfo(real_dtype).tiny)
        )
        reconstruction_tolerance = max(
            1.0e-11 if complex_dtype == torch.complex128 else 2.0e-5,
            2.0 * math.sqrt(args.retained_rank) * args.svd_relative_cutoff,
        )
        if relative_reconstruction_error > reconstruction_tolerance:
            raise RuntimeError(
                "sector SVD changed the selected supercore beyond tolerance: "
                f"{relative_reconstruction_error:.6e} > "
                f"{reconstruction_tolerance:.6e}"
            )
        before = model_summary(
            model,
            {
                **fit,
                "count": min(64, fit["count"]),
                "values": fit["values"][:64],
                "derivatives": fit["derivatives"][:64],
                "weights": fit["weights"][:64]
                / torch.sum(fit["weights"][:64]),
                "weights_numpy": (
                    fit["weights_numpy"][:64]
                    / np.sum(fit["weights_numpy"][:64])
                ),
                "log_omega": fit["log_omega"][:64],
            },
            batch_size=args.eval_batch_size,
        )
        with torch.no_grad():
            for mask in all_left_masks:
                left_parameter[mask] = 0
            for mask in all_right_masks:
                right_parameter[mask] = 0
        square_root = torch.sqrt(singular_values[:retained])
        for index in range(retained):
            left_white = compressed_left[:, index] * square_root[index]
            right_white = compressed_right_adjoint[index] * square_root[index]
            set_whitened_channel_(
                left_parameter,
                all_left_masks[index],
                left_scale_tensor,
                left_white,
            )
            set_whitened_channel_(
                right_parameter,
                all_right_masks[index],
                right_scale_tensor,
                right_white,
            )
        after = model_summary(
            model,
            {
                **fit,
                "count": min(64, fit["count"]),
                "values": fit["values"][:64],
                "derivatives": fit["derivatives"][:64],
                "weights": fit["weights"][:64]
                / torch.sum(fit["weights"][:64]),
                "weights_numpy": (
                    fit["weights_numpy"][:64]
                    / np.sum(fit["weights_numpy"][:64])
                ),
                "log_omega": fit["log_omega"][:64],
            },
            batch_size=args.eval_batch_size,
        )
        epoch_zero_svd_change = {
            key: float(after[key] - before[key])
            for key in ("sigma", "chi", "minimum_metric_eigenvalue")
        }
        discarded_weight = float(
            torch.sum(torch.square(singular_values[retained:]))
            / torch.sum(torch.square(singular_values))
        )
        compression = {
            "retained_rank": retained,
            "relative_cutoff": args.svd_relative_cutoff,
            "singular_values": [
                float(value)
                for value in singular_values[: args.retained_rank].detach().cpu()
            ],
            "discarded_relative_weight": discarded_weight,
            "relative_reconstruction_error": relative_reconstruction_error,
            "reconstruction_tolerance": reconstruction_tolerance,
            "probe_metric_summary_change": epoch_zero_svd_change,
        }
        output_payload = copy.deepcopy(payload)
        output_payload["state_dict"] = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        output_payload["native_ma_supercore_refinement"] = {
            "bond_index": args.bond_index,
            "retained_rank": retained,
            "source_model": str(model_path),
            "source_model_sha256": sha256_file(model_path),
            "selection_is_not_confirmation": True,
        }
        temporary = output_model.with_suffix(output_model.suffix + ".tmp")
        torch.save(output_payload, temporary)
        temporary.replace(output_model)

    report = {
        "schema": "quintic-hard-symmetry-supercore-native-ma-v1",
        "scientific_scope": {
            "purpose": (
                "Expose the complete 552-by-552 symmetry-allowed compact "
                "bond-supercore through matrix-free actions, retain its leading "
                "residual-aligned Schmidt subspace, and optimize native E2."
            ),
            "parameter_metric": (
                "Compact orbit coordinates are whitened by square roots of their "
                "dense-core occurrence counts."
            ),
            "selection_rule": (
                "Fit and selection E2 improve, sigma is nonworse, all registered "
                "selection tail metrics are nonworse, and positivity is retained."
            ),
        },
        "configuration": {
            "data_domain": "X_train",
            "data_start": args.data_start,
            "data_limit": args.data_limit,
            "fit_size": args.fit_size,
            "selection_size": args.selection_size,
            "bond_index": args.bond_index,
            "retained_rank": args.retained_rank,
            "oversample": args.oversample,
            "right_subspace_source": (
                "randomized_full_supercore_svd"
                if right_seed_path is None
                else "calibrated_seed_model"
            ),
            "operator_chunk_size": args.operator_chunk_size,
            "eval_batch_size": args.eval_batch_size,
            "cache_fit_prefix": bool(args.cache_fit_prefix),
            "prefix_cache_batch_size": args.prefix_cache_batch_size,
            "fit_prefix_host_bytes": (
                fit_prefix_cache.host_bytes
                if fit_prefix_cache is not None
                else 0
            ),
            "screen_fit_size": args.screen_fit_size,
            "screen_selection_size": args.screen_selection_size,
            "nonlinear_shortlist": args.nonlinear_shortlist,
            "lanczos_steps": args.lanczos_steps,
            "ridge_factors": list(args.ridge_factors),
            "step_scales": list(args.step_scales),
            "record_fit_linear_capture": bool(args.record_fit_linear_capture),
            "record_selection_linear_capture": bool(
                args.record_selection_linear_capture
            ),
            "reuse_krylov_lifts": bool(args.reuse_krylov_lifts),
            "separate_ridge_vjps": bool(args.separate_ridge_vjps),
            "verify_krylov_cache": bool(args.verify_krylov_cache),
            "device": str(device),
            "precision": str(payload["precision"]),
            "seed": args.seed,
        },
        "source": {
            "expanded_model": str(model_path),
            "expanded_model_sha256": sha256_file(model_path),
            "right_seed_model": (
                str(right_seed_path) if right_seed_path is not None else None
            ),
            "right_seed_model_sha256": (
                sha256_file(right_seed_path)
                if right_seed_path is not None
                else None
            ),
            "dataset": str(dataset_path),
            "dataset_sha256": sha256_file(dataset_path),
            "train_pullbacks": str(pullbacks_path),
            "train_pullbacks_sha256": sha256_file(pullbacks_path),
            "indices": str(indices_path),
            "indices_sha256": sha256_file(indices_path),
            "output_model": str(output_model) if accepted else None,
            "output_model_sha256": sha256_file(output_model) if accepted else None,
        },
        "parameter_count": parameter_count,
        "source_multiplicity": source_multiplicity,
        "target_multiplicity": target_multiplicity,
        "supercore_space": {
            "left_complex_dimension": int(left_zero.numel()),
            "right_complex_dimension": int(right_zero.numel()),
            "full_complex_dimension": int(left_zero.numel() * right_zero.numel()),
            "full_real_dimension": int(2 * left_zero.numel() * right_zero.numel()),
            "randomized_svd": svd_diagnostics,
            "retained_projected_singular_values": [
                float(value) for value in supercore_singular.detach().cpu()
            ],
        },
        "fit_native_baseline": fit_native,
        "fit_baseline": fit_baseline,
        "selection_baseline": selection_baseline,
        "screen_fit_baseline": screen_fit_baseline,
        "screen_selection_baseline": screen_selection_baseline,
        "shortlisted_candidate_indices": shortlisted_indices,
        "gradient_norm_in_retained_right_subspace": gradient_norm,
        "rayleigh_scale": rayleigh_scale,
        "lanczos": {
            "steps_completed": int(lanczos.tridiagonal.shape[0]),
            "breakdown": bool(lanczos.breakdown),
            "basis_orthogonality_error": lanczos.orthogonality_error,
            "cache_verification": krylov_cache_verification,
        },
        "rows": rows,
        "accepted_on_selection": accepted,
        "selected": selected,
        "sector_svd_compression": compression,
        "next_gate": (
            "Use a fresh generated paired confirmation and tail audit."
            if accepted
            else "Do not open confirmation; inspect longer-block or higher-rank "
            "supercore capture."
        ),
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(output_report, report)
    write_json(status, {"state": "complete", "phase": "complete"})
    print(
        json.dumps(
            {
                "accepted_on_selection": accepted,
                "selected": selected,
                "sector_svd_compression": compression,
                "output_model": str(output_model) if accepted else None,
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
