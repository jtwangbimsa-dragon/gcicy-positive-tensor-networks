#!/usr/bin/env python3
"""Add one residual-selected sibling-channel pair to a compiled tree.

Single-edge growth is redundant when both upper sibling edges have rank one.
This script therefore expands both children of one internal parent together,
seeds orthogonal child subspaces without changing the represented metric, and
uses native-E2 GN/LM to fit only the newly exposed parent couplings.

The full-H teacher and compiler artifact are absent.  Random seeds only propose
orthogonal tangent subspaces; disjoint native selection chooses at most one,
and an untouched confirmation split decides whether the rank is committed.
"""

from __future__ import annotations

import argparse
import copy
import gc
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Callable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline.native_e2_gauss_newton import (  # noqa: E402
    MatrixFreeNormalizedE2Jacobian,
)
from gcicy_metric.pipeline.positive_multiplication_tree import (  # noqa: E402
    PositiveMultiplicationTreeMetric,
    expand_multiplication_tree_bonds,
)
from scripts.audit_quintic_tn_tangent_reachability import (  # noqa: E402
    lanczos_tridiagonal,
    real_inner,
    ridge_coefficients_from_lanczos,
    vector_norm,
)
from scripts.evaluate_generic_quintic_h4_architecture_arms import (  # noqa: E402
    build_checkpoint_model,
    infer_architecture,
)
from scripts.refine_generic_quintic_compiled_tree_native_gn import (  # noqa: E402
    load_disjoint_splits,
    load_excluded_indices,
)
from scripts.train_generic_quintic_compiled_tree import (  # noqa: E402
    metric_row,
    sha256_file,
    whiten_dataset,
)
from scripts.train_quintic_native_power_lift_tree import (  # noqa: E402
    dataset_subset,
    make_dataset,
    paired_improvement,
    tail_guard,
)
from scripts.train_quintic_full_h_same_points import write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--fit-points", type=Path, required=True)
    parser.add_argument("--fit-pullbacks", type=Path, required=True)
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
    parser.add_argument("--fit-size", type=int, default=512)
    parser.add_argument("--selection-size", type=int, default=1024)
    parser.add_argument("--confirmation-size", type=int, default=5000)
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help="save the best selection proposal without opening confirmation",
    )
    parser.add_argument("--proposal-seeds", type=int, default=6)
    parser.add_argument(
        "--rank-increments",
        type=int,
        nargs="+",
        default=(1, 2, 4),
        help="candidate sibling-rank increments compared under one selection split",
    )
    parser.add_argument("--activation-scale", type=float, default=1.0)
    parser.add_argument("--lanczos-steps", type=int, default=6)
    parser.add_argument(
        "--ridge-factors",
        type=float,
        nargs="+",
        default=(100.0, 10.0, 1.0, 0.1),
    )
    parser.add_argument(
        "--line-search-alphas",
        type=float,
        nargs="+",
        default=(0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0),
    )
    parser.add_argument("--operator-chunk-size", type=int, default=128)
    parser.add_argument(
        "--candidate-operator-chunk-size",
        type=int,
        default=128,
        help="larger forward-only chunk used for nonlinear candidate scoring",
    )
    parser.add_argument("--feature-batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument(
        "--maximum-relative-coupling-step",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--maximum-selection-tail-relative-degradation",
        type=float,
        default=0.005,
    )
    parser.add_argument("--seed", type=int, default=202607438)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--search-precision",
        choices=("checkpoint", "complex64"),
        default="checkpoint",
        help="use complex64 only for selection-space discovery",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.fit_size,
        args.selection_size,
        args.confirmation_size,
        args.proposal_seeds,
        args.lanczos_steps,
        args.operator_chunk_size,
        args.candidate_operator_chunk_size,
        args.feature_batch_size,
        args.eval_batch_size,
        args.threads,
    )
    if (
        any(value <= 0 for value in positive)
        or not args.rank_increments
        or any(value <= 0 for value in args.rank_increments)
    ):
        raise ValueError("all counts, ranks, batches, and Lanczos steps must be positive")
    if (
        not np.isfinite(args.activation_scale)
        or args.activation_scale <= 0
        or not np.isfinite(args.maximum_relative_coupling_step)
        or args.maximum_relative_coupling_step <= 0
    ):
        raise ValueError("activation and trust-region scales must be positive")
    if (
        not args.ridge_factors
        or any(value <= 0 or not np.isfinite(value) for value in args.ridge_factors)
    ):
        raise ValueError("ridge factors must be finite and positive")
    if (
        not args.line_search_alphas
        or any(
            value <= 0 or value > 1 or not np.isfinite(value)
            for value in args.line_search_alphas
        )
    ):
        raise ValueError("line-search alphas must lie in (0, 1]")


def sibling_growth_target(
    model: PositiveMultiplicationTreeMetric,
    *,
    increment: int,
) -> tuple[int, tuple[int, int], tuple[int, ...]]:
    """Choose the lowest parent whose two children are internal nodes."""

    candidates = []
    for internal_index, (left, right) in enumerate(model.topology.children):
        node = model.leaf_count + internal_index
        if (
            node != model.topology.root
            and left >= model.leaf_count
            and right >= model.leaf_count
        ):
            candidates.append((node, (left, right)))
    if not candidates:
        raise ValueError("compiled tree has no internal sibling pair to enrich")
    parent, children = min(candidates)
    target = list(model.edge_dimensions)
    for child in children:
        target[child] += increment
    return parent, children, tuple(target)


def new_coupling_parameterization(
    model: PositiveMultiplicationTreeMetric,
    *,
    parent: int,
    source_dimensions: tuple[int, ...],
) -> tuple[
    str,
    torch.Tensor,
    torch.Tensor,
    Callable[[torch.Tensor], dict[str, torch.Tensor]],
]:
    """Return coordinates for only the newly exposed parent-core entries."""

    internal_index = parent - model.leaf_count
    parameter_name = f"internal_tensors.{internal_index}"
    tensor = model.internal_tensors[internal_index]
    left, right = model.topology.children[internal_index]
    old_left = source_dimensions[left]
    old_right = source_dimensions[right]
    mask = torch.ones(
        tensor.shape,
        dtype=torch.bool,
        device=tensor.device,
    )
    mask[:, :old_left, :old_right] = False
    flat_indices = torch.nonzero(mask.reshape(-1), as_tuple=False).reshape(-1)
    if flat_indices.numel() == 0:
        raise ValueError("sibling expansion exposed no new parent coupling")
    baseline = tensor.detach().clone()
    theta = baseline.reshape(-1).index_select(0, flat_indices).clone()

    def unpack(coordinates: torch.Tensor) -> dict[str, torch.Tensor]:
        if coordinates.shape != theta.shape:
            raise ValueError("new parent-coupling vector has the wrong shape")
        flattened = baseline.reshape(-1).clone()
        flattened = flattened.scatter(0, flat_indices, coordinates)
        return {parameter_name: flattened.reshape(baseline.shape)}

    return parameter_name, theta, flat_indices, unpack


def set_new_couplings_(
    model: PositiveMultiplicationTreeMetric,
    *,
    parent: int,
    flat_indices: torch.Tensor,
    coordinates: torch.Tensor,
) -> None:
    tensor = model.internal_tensors[parent - model.leaf_count]
    with torch.no_grad():
        flattened = tensor.reshape(-1)
        flattened.index_copy_(0, flat_indices, coordinates)


def solve_seed(
    model: PositiveMultiplicationTreeMetric,
    *,
    parent: int,
    source_dimensions: tuple[int, ...],
    fit: dict[str, Any],
    selection: dict[str, Any],
    args: argparse.Namespace,
    seed_index: int,
    proposal_count: int,
    status_path: Path,
) -> dict[str, Any]:
    parameter_name, theta, flat_indices, unpack = new_coupling_parameterization(
        model,
        parent=parent,
        source_dimensions=source_dimensions,
    )
    fit_operator = MatrixFreeNormalizedE2Jacobian(
        model,
        unpack,
        theta,
        fit,
        chunk_size=args.operator_chunk_size,
    )
    selection_operator = MatrixFreeNormalizedE2Jacobian(
        model,
        unpack,
        theta,
        selection,
        chunk_size=args.operator_chunk_size,
    )
    residual = fit_operator.residual()
    fit_e2 = float(real_inner(residual, residual))
    selection_e2 = selection_operator.native_e2
    gradient = fit_operator.vjp(residual)
    gradient_norm = float(vector_norm(gradient))
    rayleigh_scale = gradient_norm**2 / max(
        fit_e2,
        np.finfo(np.float64).tiny,
    )
    if not np.isfinite(rayleigh_scale) or rayleigh_scale <= 0:
        return {
            "parameter_name": parameter_name,
            "coordinate_count_complex": int(theta.numel()),
            "failure": "no_finite_native_gradient",
            "candidates": [],
            "selected": None,
        }

    def progress(iteration: int, alpha: float, beta: float) -> None:
        write_json(
            status_path,
            {
                "state": "running",
                "phase": "proposal_lanczos",
                "seed_index": seed_index,
                "proposal_count": proposal_count,
                "iteration": iteration,
                "iterations": args.lanczos_steps,
                "alpha": alpha,
                "beta": beta,
            },
        )

    parameter_vjps: list[torch.Tensor] = []

    def normal_and_cache(cotangent: torch.Tensor) -> torch.Tensor:
        parameter_vjp = fit_operator.vjp(cotangent)
        parameter_vjps.append(parameter_vjp)
        return fit_operator.jvp(parameter_vjp)

    lanczos = lanczos_tridiagonal(
        normal_and_cache,
        residual,
        steps=min(args.lanczos_steps, fit["count"]),
        callback=progress,
    )
    if len(parameter_vjps) != lanczos.tridiagonal.shape[0]:
        raise RuntimeError("Lanczos/VJP cache dimension mismatch")
    parameter_vjp_basis = torch.stack(parameter_vjps, dim=1)
    del parameter_vjps
    parent_scale = max(
        float(
            vector_norm(
                model.internal_tensors[
                    parent - model.leaf_count
                ].detach().reshape(-1)
            )
        ),
        np.finfo(np.float64).tiny,
    )
    candidate_fit_operator = MatrixFreeNormalizedE2Jacobian(
        model,
        unpack,
        theta,
        fit,
        chunk_size=args.candidate_operator_chunk_size,
    )
    candidate_selection_operator = MatrixFreeNormalizedE2Jacobian(
        model,
        unpack,
        theta,
        selection,
        chunk_size=args.candidate_operator_chunk_size,
    )
    rows: list[dict[str, Any]] = []
    for ridge_factor in args.ridge_factors:
        ridge = float(ridge_factor * rayleigh_scale)
        coefficients = ridge_coefficients_from_lanczos(lanczos, ridge)
        base_delta = -(
            parameter_vjp_basis
            @ coefficients.to(dtype=parameter_vjp_basis.dtype)
        )
        for alpha in args.line_search_alphas:
            coordinates = (theta + float(alpha) * base_delta).detach()
            relative_step = float(vector_norm(coordinates - theta)) / parent_scale
            row: dict[str, Any] = {
                "ridge_factor": float(ridge_factor),
                "ridge": ridge,
                "alpha": float(alpha),
                "relative_coupling_step": relative_step,
                "eligible": False,
                "failure": None,
                "_coordinates": coordinates,
            }
            if relative_step > args.maximum_relative_coupling_step:
                row["failure"] = "relative_coupling_step"
                rows.append(row)
                continue
            try:
                fit_candidate = candidate_fit_operator.residual(coordinates)
                selection_candidate = candidate_selection_operator.residual(
                    coordinates
                )
                actual_fit_e2 = float(real_inner(fit_candidate, fit_candidate))
                actual_selection_e2 = float(
                    real_inner(selection_candidate, selection_candidate)
                )
                row.update(
                    {
                        "actual_fit_e2": actual_fit_e2,
                        "actual_fit_capture": 1.0 - actual_fit_e2 / fit_e2,
                        "actual_selection_e2": actual_selection_e2,
                        "actual_selection_capture": (
                            1.0 - actual_selection_e2 / selection_e2
                        ),
                    }
                )
            except (RuntimeError, FloatingPointError) as error:
                row["failure"] = f"{type(error).__name__}: {error}"
            rows.append(row)

    viable = [
        row
        for row in rows
        if row.get("actual_fit_capture", 0.0) > 0
        and row.get("actual_selection_capture", 0.0) > 0
    ]
    shortlisted = sorted(
        viable,
        key=lambda row: (
            row["actual_selection_e2"],
            row["actual_fit_e2"],
        ),
    )[:3]
    baseline_state = copy.deepcopy(model.state_dict())
    baseline_statistics, _, baseline_tail = metric_row(
        model,
        selection,
        chunk_size=args.eval_batch_size,
    )
    for row in shortlisted:
        try:
            set_new_couplings_(
                model,
                parent=parent,
                flat_indices=flat_indices,
                coordinates=row["_coordinates"],
            )
            statistics, _, tail = metric_row(
                model,
                selection,
                chunk_size=args.eval_batch_size,
            )
            row["selection_statistics"] = statistics
            row["selection_tail"] = tail
            row["eligible"] = bool(
                statistics["sigma_official_formula"]
                < baseline_statistics["sigma_official_formula"]
                and statistics["weighted_rms_abs_residual"]
                < baseline_statistics["weighted_rms_abs_residual"]
                and tail_guard(
                    tail,
                    baseline_tail,
                    relative_degradation=(
                        args.maximum_selection_tail_relative_degradation
                    ),
                )
            )
        except (RuntimeError, FloatingPointError) as error:
            row["failure"] = f"{type(error).__name__}: {error}"
        finally:
            model.load_state_dict(baseline_state)
    eligible = [row for row in shortlisted if row["eligible"]]
    selected = (
        min(
            eligible,
            key=lambda row: (
                row["actual_selection_e2"],
                row["selection_statistics"]["sigma_official_formula"],
            ),
        )
        if eligible
        else None
    )
    return {
        "parameter_name": parameter_name,
        "coordinate_count_complex": int(theta.numel()),
        "flat_indices": flat_indices.detach().cpu(),
        "fit_native_e2": fit_e2,
        "selection_native_e2": selection_e2,
        "gradient_norm": gradient_norm,
        "rayleigh_scale": rayleigh_scale,
        "lanczos_dimension": int(lanczos.tridiagonal.shape[0]),
        "candidates": rows,
        "selected": selected,
    }


def serializable_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if not key.startswith("_") and key != "flat_indices"
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.search_precision == "complex64" and not args.selection_only:
        raise ValueError(
            "complex64 search is selection-only; confirmation must use checkpoint precision"
        )
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite a sibling-rank run")
    output_dir.mkdir(parents=True)
    status_path = output_dir / "status.json"
    write_json(status_path, {"state": "running", "phase": "loading"})
    started = time.perf_counter()

    try:
        checkpoint_path = args.initial_checkpoint.expanduser().resolve()
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        if infer_architecture(payload) != "compiled-tree":
            raise ValueError("sibling rank enrichment requires a compiled tree")
        if bool(payload.get("teacher_runtime_dependency", False)):
            raise ValueError("input checkpoint still declares a teacher dependency")
        configuration = payload["configuration"]
        checkpoint_dtype = (
            torch.complex64
            if str(configuration["precision"]) == "complex64"
            else torch.complex128
        )
        dtype = (
            torch.complex64
            if args.search_precision == "complex64"
            else checkpoint_dtype
        )
        exponents = np.asarray(configuration["exponents"], dtype=np.int64)
        whitening = np.asarray(configuration["whitening"], dtype=np.complex128)
        source_model = build_checkpoint_model(payload, device=device).to(
            dtype=dtype
        )
        if not isinstance(source_model, PositiveMultiplicationTreeMetric):
            raise TypeError("checkpoint did not reconstruct a multiplication tree")
        source_dimensions = tuple(source_model.edge_dimensions)
        parent, sibling_edges, _ = sibling_growth_target(
            source_model,
            increment=1,
        )

        excluded_indices = load_excluded_indices(args.exclude_indices_file)
        split_specifications = {
                "fit": (
                    args.fit_points,
                    args.fit_pullbacks,
                    args.fit_size,
                ),
                "selection": (
                    args.selection_points,
                    args.selection_pullbacks,
                    args.selection_size,
                ),
        }
        if not args.selection_only:
            split_specifications["confirmation"] = (
                    args.confirmation_points,
                    args.confirmation_pullbacks,
                    args.confirmation_size,
                )
        split_arrays, split_indices = load_disjoint_splits(
            split_specifications,
            seed=args.seed,
            exclusions=excluded_indices,
        )
        datasets = {
            name: whiten_dataset(
                make_dataset(
                    arrays,
                    exponents,
                    feature_batch_size=args.feature_batch_size,
                    complex_dtype=dtype,
                    device=device,
                ),
                whitening,
            )
            for name, arrays in split_arrays.items()
        }
        indices_path = output_dir / "data_indices.npz"
        np.savez_compressed(indices_path, **split_indices)
        baseline_selection, baseline_selection_ratio, baseline_selection_tail = (
            metric_row(
                source_model,
                datasets["selection"],
                chunk_size=args.eval_batch_size,
            )
        )
        audit_indices = np.arange(min(128, datasets["selection"]["count"]))
        audit = dataset_subset(datasets["selection"], audit_indices)
        audit["weights_numpy"] = np.asarray(
            datasets["selection"]["weights_numpy"][audit_indices]
        )
        _, source_audit_ratio, _ = metric_row(
            source_model,
            audit,
            chunk_size=args.eval_batch_size,
        )

        seed_results: list[dict[str, Any]] = []
        total_proposals = len(args.rank_increments) * args.proposal_seeds
        proposal_index = 0
        for rank_increment in args.rank_increments:
            _, _, target_dimensions = sibling_growth_target(
                source_model,
                increment=rank_increment,
            )
            for seed_index in range(args.proposal_seeds):
                proposal_index += 1
                proposal_seed = (
                    args.seed
                    + 1000
                    + 101 * int(rank_increment)
                    + seed_index
                )
                expanded = expand_multiplication_tree_bonds(
                    source_model,
                    target_dimensions,
                    relative_activation_scale=args.activation_scale,
                    seed=proposal_seed,
                    orthogonalize_new_outputs=True,
                )
                _, expanded_audit_ratio, _ = metric_row(
                    expanded,
                    audit,
                    chunk_size=args.eval_batch_size,
                )
                embedding_ratio_error = float(
                    np.max(np.abs(expanded_audit_ratio - source_audit_ratio))
                )
                tolerance = 2.0e-5 if dtype == torch.complex64 else 2.0e-11
                if embedding_ratio_error > tolerance:
                    raise FloatingPointError(
                        "orthogonal sibling expansion changed the initial metric"
                    )
                result = solve_seed(
                    expanded,
                    parent=parent,
                    source_dimensions=source_dimensions,
                    fit=datasets["fit"],
                    selection=datasets["selection"],
                    args=args,
                    seed_index=proposal_index,
                    proposal_count=total_proposals,
                    status_path=status_path,
                )
                result.update(
                    {
                        "seed": proposal_seed,
                        "rank_increment": int(rank_increment),
                        "target_edge_dimensions": target_dimensions,
                        "proposal_index": proposal_index,
                        "proposal_count": total_proposals,
                        "embedding_ratio_max_absolute": embedding_ratio_error,
                    }
                )
                for candidate in result["candidates"]:
                    if "_coordinates" in candidate:
                        candidate["_coordinates"] = (
                            candidate["_coordinates"].detach().cpu()
                        )
                seed_results.append(result)
                write_json(
                    output_dir / "proposal_progress.json",
                    {
                        "rows": [
                            {
                                **{
                                    key: value
                                    for key, value in row.items()
                                    if key
                                    not in {
                                        "candidates",
                                        "selected",
                                        "flat_indices",
                                    }
                                },
                                "selected": (
                                    None
                                    if row["selected"] is None
                                    else serializable_row(row["selected"])
                                ),
                            }
                            for row in seed_results
                        ]
                    },
                )
                del expanded
                gc.collect()
                torch.cuda.empty_cache()

        eligible_seeds = [
            row for row in seed_results if row["selected"] is not None
        ]
        chosen = (
            min(
                eligible_seeds,
                key=lambda row: (
                    row["selected"]["actual_selection_e2"],
                    row["selected"]["selection_statistics"][
                        "sigma_official_formula"
                    ],
                ),
            )
            if eligible_seeds
            else None
        )

        if args.selection_only:
            proposal_model = source_model
            proposal_path = None
            chosen_dimensions = source_dimensions
            if chosen is not None:
                chosen_dimensions = tuple(
                    int(value) for value in chosen["target_edge_dimensions"]
                )
                checkpoint_source_model = build_checkpoint_model(
                    payload,
                    device=device,
                )
                proposal_model = expand_multiplication_tree_bonds(
                    checkpoint_source_model,
                    chosen_dimensions,
                    relative_activation_scale=args.activation_scale,
                    seed=int(chosen["seed"]),
                    orthogonalize_new_outputs=True,
                )
                set_new_couplings_(
                    proposal_model,
                    parent=parent,
                    flat_indices=chosen["flat_indices"].to(device=device),
                    coordinates=chosen["selected"]["_coordinates"].to(
                        device=device,
                        dtype=checkpoint_dtype,
                    ),
                )
                proposal_payload = copy.deepcopy(payload)
                proposal_payload["state_dict"] = {
                    key: value.detach().cpu().clone()
                    for key, value in proposal_model.state_dict().items()
                }
                proposal_configuration = dict(configuration)
                proposal_configuration["edge_dimensions"] = chosen_dimensions
                proposal_configuration["bond_dimension"] = max(chosen_dimensions)
                proposal_payload["configuration"] = proposal_configuration
                proposal_payload["parent_checkpoint_sha256"] = sha256_file(
                    checkpoint_path
                )
                proposal_payload["teacher_runtime_dependency"] = False
                proposal_payload["confirmation_passes"] = False
                proposal_payload["selection_only"] = True
                proposal_payload["native_sibling_rank_enrichment"] = {
                    "parent": parent,
                    "sibling_edges": sibling_edges,
                    "source_edge_dimensions": source_dimensions,
                    "target_edge_dimensions": chosen_dimensions,
                    "rank_increment": int(chosen["rank_increment"]),
                    "chosen_seed": int(chosen["seed"]),
                }
                proposal_path = output_dir / "selection_proposal.pt"
                torch.save(proposal_payload, proposal_path)
            report = {
                "schema": "generic-quintic-compiled-tree-sibling-rank-selection-v1",
                "teacher_role": "absent_after_round_zero",
                "teacher_runtime_dependency": False,
                "configuration": {
                    **vars(args),
                    "initial_checkpoint": str(checkpoint_path),
                    "output_dir": str(output_dir),
                },
                "rank_growth": {
                    "parent": parent,
                    "sibling_edges": sibling_edges,
                    "source_edge_dimensions": source_dimensions,
                    "candidate_rank_increments": list(args.rank_increments),
                    "chosen_rank_increment": (
                        None
                        if chosen is None
                        else int(chosen["rank_increment"])
                    ),
                    "target_edge_dimensions": (
                        None
                        if chosen is None
                        else tuple(chosen["target_edge_dimensions"])
                    ),
                    "search_precision": str(dtype),
                    "checkpoint_precision": str(checkpoint_dtype),
                },
                "selection": {
                    "baseline": baseline_selection,
                    "baseline_tail": baseline_selection_tail,
                    "proposal_seeds": [
                        {
                            **{
                                key: value
                                for key, value in row.items()
                                if key
                                not in {
                                    "candidates",
                                    "selected",
                                    "flat_indices",
                                }
                            },
                            "candidates": [
                                serializable_row(candidate)
                                for candidate in row["candidates"]
                            ],
                            "selected": (
                                None
                                if row["selected"] is None
                                else serializable_row(row["selected"])
                            ),
                        }
                        for row in seed_results
                    ],
                    "chosen_seed": (
                        None if chosen is None else int(chosen["seed"])
                    ),
                    "chosen": (
                        None
                        if chosen is None
                        else serializable_row(chosen["selected"])
                    ),
                },
                "confirmation": {
                    "opened": False,
                    "passes": False,
                },
                "model": {
                    "source_real_parameter_count": (
                        source_model.trainable_real_parameter_count
                    ),
                    "proposal_real_parameter_count": (
                        None
                        if chosen is None
                        else proposal_model.trainable_real_parameter_count
                    ),
                },
                "data": {
                    "indices": str(indices_path),
                    "indices_sha256": sha256_file(indices_path),
                },
                "proposal_checkpoint": (
                    None if proposal_path is None else str(proposal_path)
                ),
                "proposal_checkpoint_sha256": (
                    None
                    if proposal_path is None
                    else sha256_file(proposal_path)
                ),
                "wall_seconds": time.perf_counter() - started,
            }
            write_json(output_dir / "report.json", report)
            write_json(
                status_path,
                {
                    "state": "complete",
                    "phase": "selection_complete",
                    "selection_found_candidate": chosen is not None,
                    "confirmation_opened": False,
                    "proposal_checkpoint": (
                        None if proposal_path is None else str(proposal_path)
                    ),
                },
            )
            print(status_path.read_text(encoding="utf-8"), flush=True)
            return

        confirmation_passes = False
        baseline_confirmation = None
        baseline_confirmation_tail = None
        candidate_confirmation = None
        candidate_confirmation_tail = None
        paired = None
        final_model = source_model
        if chosen is not None:
            chosen_target_dimensions = tuple(
                int(value) for value in chosen["target_edge_dimensions"]
            )
            baseline_confirmation, baseline_confirmation_ratio, (
                baseline_confirmation_tail
            ) = metric_row(
                source_model,
                datasets["confirmation"],
                chunk_size=args.eval_batch_size,
            )
            final_model = expand_multiplication_tree_bonds(
                source_model,
                chosen_target_dimensions,
                relative_activation_scale=args.activation_scale,
                seed=int(chosen["seed"]),
                orthogonalize_new_outputs=True,
            )
            flat_indices = chosen["flat_indices"].to(device=device)
            set_new_couplings_(
                final_model,
                parent=parent,
                flat_indices=flat_indices,
                coordinates=chosen["selected"]["_coordinates"].to(
                    device=device
                ),
            )
            candidate_confirmation, candidate_confirmation_ratio, (
                candidate_confirmation_tail
            ) = metric_row(
                final_model,
                datasets["confirmation"],
                chunk_size=args.eval_batch_size,
            )
            paired = paired_improvement(
                baseline_confirmation_ratio,
                candidate_confirmation_ratio,
                datasets["confirmation"]["weights_numpy"],
            )
            confirmation_passes = bool(
                candidate_confirmation["sigma_official_formula"]
                < baseline_confirmation["sigma_official_formula"]
                and candidate_confirmation["weighted_rms_abs_residual"]
                < baseline_confirmation["weighted_rms_abs_residual"]
                and paired["e2"]["ci95_low"] > 0
                and paired["sigma"]["ci95_low"] > 0
                and tail_guard(
                    candidate_confirmation_tail,
                    baseline_confirmation_tail,
                    relative_degradation=0.0,
                )
            )
        if not confirmation_passes:
            final_model = source_model
            final_dimensions = source_dimensions
        else:
            final_dimensions = chosen_target_dimensions

        output_payload = copy.deepcopy(payload)
        output_payload["state_dict"] = {
            key: value.detach().cpu().clone()
            for key, value in final_model.state_dict().items()
        }
        final_configuration = dict(configuration)
        final_configuration["edge_dimensions"] = tuple(final_dimensions)
        final_configuration["bond_dimension"] = max(final_dimensions)
        final_configuration["adaptive_rank_round"] = int(
            configuration.get("adaptive_rank_round", 0)
        ) + (1 if confirmation_passes else 0)
        output_payload["configuration"] = final_configuration
        output_payload["parent_checkpoint_sha256"] = sha256_file(checkpoint_path)
        output_payload["teacher_runtime_dependency"] = False
        output_payload["confirmation_passes"] = confirmation_passes
        output_payload["native_sibling_rank_enrichment"] = {
            "parent": parent,
            "sibling_edges": sibling_edges,
            "source_edge_dimensions": source_dimensions,
            "target_edge_dimensions": (
                None
                if chosen is None
                else tuple(chosen["target_edge_dimensions"])
            ),
            "rank_increment": (
                None if chosen is None else int(chosen["rank_increment"])
            ),
            "chosen_seed": None if chosen is None else int(chosen["seed"]),
            "confirmation_passes": confirmation_passes,
        }
        checkpoint_output = output_dir / (
            "accepted.pt" if confirmation_passes else "baseline_retained.pt"
        )
        torch.save(output_payload, checkpoint_output)
        report = {
            "schema": "generic-quintic-compiled-tree-sibling-rank-v1",
            "teacher_role": "absent_after_round_zero",
            "teacher_runtime_dependency": False,
            "objective": {
                "name": "model-normalized native MA E2",
                "normalization_derivative_included": True,
            },
            "configuration": {
                **vars(args),
                "initial_checkpoint": str(checkpoint_path),
                "output_dir": str(output_dir),
            },
            "rank_growth": {
                "parent": parent,
                "sibling_edges": sibling_edges,
                "source_edge_dimensions": source_dimensions,
                "candidate_rank_increments": list(args.rank_increments),
                "chosen_rank_increment": (
                    None
                    if chosen is None
                    else int(chosen["rank_increment"])
                ),
                "target_edge_dimensions": (
                    None
                    if chosen is None
                    else tuple(chosen["target_edge_dimensions"])
                ),
                "orthogonal_child_activation": True,
                "new_parent_couplings_only": True,
            },
            "selection": {
                "baseline": baseline_selection,
                "baseline_tail": baseline_selection_tail,
                "proposal_seeds": [
                    {
                        **{
                            key: value
                            for key, value in row.items()
                            if key
                            not in {
                                "candidates",
                                "selected",
                                "flat_indices",
                            }
                        },
                        "candidates": [
                            serializable_row(candidate)
                            for candidate in row["candidates"]
                        ],
                        "selected": (
                            None
                            if row["selected"] is None
                            else serializable_row(row["selected"])
                        ),
                    }
                    for row in seed_results
                ],
                "chosen_seed": None if chosen is None else int(chosen["seed"]),
                "chosen": (
                    None
                    if chosen is None
                    else serializable_row(chosen["selected"])
                ),
            },
            "confirmation": {
                "opened": chosen is not None,
                "baseline": baseline_confirmation,
                "baseline_tail": baseline_confirmation_tail,
                "candidate": candidate_confirmation,
                "candidate_tail": candidate_confirmation_tail,
                "paired_improvement": paired,
                "passes": confirmation_passes,
            },
            "model": {
                "source_real_parameter_count": (
                    source_model.trainable_real_parameter_count
                ),
                "candidate_real_parameter_count": (
                    (
                        expand_multiplication_tree_bonds(
                            source_model,
                            tuple(chosen["target_edge_dimensions"]),
                            relative_activation_scale=args.activation_scale,
                            seed=args.seed + 9999,
                            orthogonalize_new_outputs=True,
                        ).trainable_real_parameter_count
                    )
                    if chosen is not None
                    else None
                ),
                "committed_real_parameter_count": (
                    final_model.trainable_real_parameter_count
                ),
            },
            "data": {
                "indices": str(indices_path),
                "indices_sha256": sha256_file(indices_path),
                "excluded_indices_files": [
                    {
                        "path": str(path.expanduser().resolve()),
                        "sha256": sha256_file(path.expanduser().resolve()),
                    }
                    for path in args.exclude_indices_file
                ],
            },
            "checkpoint": str(checkpoint_output),
            "checkpoint_sha256": sha256_file(checkpoint_output),
            "wall_seconds": time.perf_counter() - started,
        }
        write_json(output_dir / "report.json", report)
        write_json(
            status_path,
            {
                "state": "complete",
                "phase": "complete",
                "selection_found_candidate": chosen is not None,
                "confirmation_passes": confirmation_passes,
                "checkpoint": str(checkpoint_output),
            },
        )
        print(status_path.read_text(encoding="utf-8"), flush=True)
    except Exception as error:
        write_json(
            status_path,
            {
                "state": "failed",
                "phase": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "wall_seconds": time.perf_counter() - started,
            },
        )
        raise


if __name__ == "__main__":
    main()
