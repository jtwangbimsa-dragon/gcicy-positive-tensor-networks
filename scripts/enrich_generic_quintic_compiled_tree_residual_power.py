#!/usr/bin/env python3
"""Grow compiled-tree sibling ranks from native-E2 block-power directions.

The full-H teacher is absent.  New sibling rows are discovered by alternating
native-residual VJPs:

```
right subspace -> gradient in the left complement
left subspace  -> gradient in the right complement
```

Each probe preserves the parent checkpoint exactly because the variable child
rows start at zero.  Once a residual-aligned pair of subspaces is found, only
the newly exposed parent couplings are fitted by native-E2 GN/LM.  Independent
selection chooses at most one rank proposal; confirmation is deliberately left
to a separate process.
"""

from __future__ import annotations

import argparse
import copy
import gc
from pathlib import Path
import sys
import time
import traceback
from typing import Any

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
    real_inner,
    vector_norm,
)
from scripts.enrich_generic_quintic_compiled_tree_sibling_rank import (  # noqa: E402
    serializable_row,
    set_new_couplings_,
    sibling_growth_target,
    solve_seed,
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
)
from scripts.train_quintic_full_h_same_points import write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--fit-points", type=Path, required=True)
    parser.add_argument("--fit-pullbacks", type=Path, required=True)
    parser.add_argument("--selection-points", type=Path, required=True)
    parser.add_argument("--selection-pullbacks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--exclude-indices-file",
        type=Path,
        nargs="*",
        default=(),
    )
    parser.add_argument("--fit-size", type=int, default=512)
    parser.add_argument("--selection-size", type=int, default=1024)
    parser.add_argument(
        "--rank-increments",
        type=int,
        nargs="+",
        default=(1, 2, 4),
    )
    parser.add_argument("--power-iterations", type=int, default=2)
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
    parser.add_argument("--operator-chunk-size", type=int, default=16)
    parser.add_argument(
        "--candidate-operator-chunk-size",
        type=int,
        default=128,
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
    parser.add_argument("--seed", type=int, default=202607447)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    counts = (
        args.fit_size,
        args.selection_size,
        args.power_iterations,
        args.lanczos_steps,
        args.operator_chunk_size,
        args.candidate_operator_chunk_size,
        args.feature_batch_size,
        args.eval_batch_size,
        args.threads,
    )
    if any(value <= 0 for value in counts):
        raise ValueError("all sizes, iterations, and batches must be positive")
    if (
        not args.rank_increments
        or any(value <= 0 for value in args.rank_increments)
        or len(set(args.rank_increments)) != len(args.rank_increments)
    ):
        raise ValueError("rank increments must be distinct positive integers")
    if (
        not args.ridge_factors
        or any(value <= 0 for value in args.ridge_factors)
        or not args.line_search_alphas
        or any(value <= 0 or value > 1 for value in args.line_search_alphas)
        or args.maximum_relative_coupling_step <= 0
    ):
        raise ValueError("ridge and trust-region settings are invalid")


def orthonormal_complement_rows(
    candidates: torch.Tensor,
    fixed_rows: torch.Tensor,
) -> torch.Tensor:
    """Orthonormalize candidate coefficient rows outside a fixed row space."""

    if candidates.ndim != 2 or fixed_rows.ndim != 2:
        raise ValueError("candidate and fixed rows must be matrices")
    if candidates.shape[1] != fixed_rows.shape[1]:
        raise ValueError("candidate and fixed rows use different ambient spaces")
    requested = candidates.shape[0]
    fixed_columns, _ = torch.linalg.qr(
        torch.transpose(fixed_rows, 0, 1),
        mode="reduced",
    )
    candidate_columns = torch.transpose(candidates, 0, 1)
    candidate_columns = candidate_columns - fixed_columns @ (
        torch.conj(torch.transpose(fixed_columns, 0, 1))
        @ candidate_columns
    )
    new_columns, triangular = torch.linalg.qr(
        candidate_columns,
        mode="reduced",
    )
    diagonal = torch.abs(torch.diagonal(triangular))
    scale = float(torch.max(diagonal))
    if not np.isfinite(scale) or scale <= torch.finfo(
        torch.real(candidates).dtype
    ).tiny:
        raise FloatingPointError("native block-power action is numerically zero")
    tolerance = (
        max(candidate_columns.shape)
        * torch.finfo(torch.real(candidates).dtype).eps
        * scale
    )
    if int(torch.count_nonzero(diagonal > tolerance)) < requested:
        raise FloatingPointError("native block-power subspace lost numerical rank")
    rows = torch.transpose(new_columns[:, :requested], 0, 1)
    return rows.detach()


def random_complement_rows(
    fixed_rows: torch.Tensor,
    *,
    count: int,
    seed: int,
) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(count, fixed_rows.shape[1])) + 1j * rng.normal(
        size=(count, fixed_rows.shape[1])
    )
    candidates = torch.tensor(
        values,
        dtype=fixed_rows.dtype,
        device=fixed_rows.device,
    )
    return orthonormal_complement_rows(candidates, fixed_rows)


def child_parameter_name(
    model: PositiveMultiplicationTreeMetric,
    edge: int,
) -> str:
    if edge < model.leaf_count:
        raise ValueError("residual block power currently targets internal children")
    return f"internal_tensors.{edge - model.leaf_count}"


def child_rows_parameterization(
    model: PositiveMultiplicationTreeMetric,
    *,
    edge: int,
    old_count: int,
) -> tuple[torch.Tensor, Any]:
    name = child_parameter_name(model, edge)
    tensor = dict(model.named_parameters())[name]
    baseline = tensor.detach().clone()
    new_shape = (tensor.shape[0] - old_count,) + tensor.shape[1:]
    theta = baseline[old_count:].reshape(-1).clone()

    def unpack(coordinates: torch.Tensor) -> dict[str, torch.Tensor]:
        if coordinates.shape != theta.shape:
            raise ValueError("child-row coordinate vector has the wrong shape")
        return {
            name: torch.cat(
                (
                    baseline[:old_count],
                    coordinates.reshape(new_shape),
                ),
                dim=0,
            )
        }

    return theta, unpack


def build_zero_probe(
    source: PositiveMultiplicationTreeMetric,
    *,
    parent: int,
    sibling_edges: tuple[int, int],
    increment: int,
    variable_side: int,
    fixed_rows: torch.Tensor,
    seed: int,
) -> tuple[PositiveMultiplicationTreeMetric, torch.Tensor, Any]:
    """Build an exact-embedding probe with one zero child and one fixed child."""

    if variable_side not in (0, 1):
        raise ValueError("variable side must be zero or one")
    source_dimensions = tuple(source.edge_dimensions)
    target = list(source_dimensions)
    for edge in sibling_edges:
        target[edge] += increment
    probe = expand_multiplication_tree_bonds(
        source,
        tuple(target),
        relative_activation_scale=1.0,
        seed=seed,
        orthogonalize_new_outputs=True,
    )
    child_names = tuple(
        child_parameter_name(probe, edge) for edge in sibling_edges
    )
    parameters = dict(probe.named_parameters())
    old_counts = tuple(source_dimensions[edge] for edge in sibling_edges)
    with torch.no_grad():
        for side, (name, old_count) in enumerate(
            zip(child_names, old_counts, strict=True)
        ):
            tensor = parameters[name]
            tensor[old_count:].zero_()
            if side != variable_side:
                tensor[old_count:].copy_(
                    fixed_rows.reshape(tensor[old_count:].shape)
                )
        parent_tensor = probe.internal_tensors[parent - probe.leaf_count]
        left_old, right_old = old_counts
        parent_tensor[
            :,
            left_old:,
            right_old:,
        ].zero_()
        for channel in range(increment):
            parent_tensor[
                0,
                left_old + channel,
                right_old + channel,
            ] = 1.0
    variable_edge = sibling_edges[variable_side]
    theta, unpack = child_rows_parameterization(
        probe,
        edge=variable_edge,
        old_count=old_counts[variable_side],
    )
    if float(vector_norm(theta)) != 0.0:
        raise RuntimeError("block-power probe did not start at the exact embedding")
    return probe, theta, unpack


def native_child_action(
    source: PositiveMultiplicationTreeMetric,
    *,
    parent: int,
    sibling_edges: tuple[int, int],
    increment: int,
    variable_side: int,
    fixed_rows: torch.Tensor,
    dataset: dict[str, Any],
    chunk_size: int,
    seed: int,
) -> tuple[torch.Tensor, float]:
    """Apply the native residual supercore gradient to one child block."""

    probe, theta, unpack = build_zero_probe(
        source,
        parent=parent,
        sibling_edges=sibling_edges,
        increment=increment,
        variable_side=variable_side,
        fixed_rows=fixed_rows,
        seed=seed,
    )
    operator = MatrixFreeNormalizedE2Jacobian(
        probe,
        unpack,
        theta,
        dataset,
        chunk_size=chunk_size,
    )
    residual = operator.residual()
    gradient = operator.vjp(residual)
    gradient_norm = float(vector_norm(gradient))
    variable_edge = sibling_edges[variable_side]
    name = child_parameter_name(probe, variable_edge)
    tensor = dict(probe.named_parameters())[name]
    rows = gradient.reshape(increment, -1).detach()
    del operator, probe, residual, gradient, tensor
    gc.collect()
    return rows, gradient_norm


def native_block_power_subspaces(
    source: PositiveMultiplicationTreeMetric,
    *,
    parent: int,
    sibling_edges: tuple[int, int],
    increment: int,
    dataset: dict[str, Any],
    iterations: int,
    chunk_size: int,
    seed: int,
    status_path: Path,
    rank_index: int,
    rank_count: int,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, float]]]:
    """Find residual-aligned sibling row spaces without changing the model."""

    names = tuple(child_parameter_name(source, edge) for edge in sibling_edges)
    parameters = dict(source.named_parameters())
    old_rows = tuple(
        parameters[name].detach().reshape(parameters[name].shape[0], -1)
        for name in names
    )
    right_rows = random_complement_rows(
        old_rows[1],
        count=increment,
        seed=seed,
    )
    diagnostics: list[dict[str, float]] = []
    left_rows = random_complement_rows(
        old_rows[0],
        count=increment,
        seed=seed + 1,
    )
    for iteration in range(iterations):
        write_json(
            status_path,
            {
                "state": "running",
                "phase": "native_block_power",
                "rank_index": rank_index,
                "rank_count": rank_count,
                "rank_increment": increment,
                "iteration": iteration + 1,
                "iterations": iterations,
                "side": "left",
            },
        )
        left_action, left_norm = native_child_action(
            source,
            parent=parent,
            sibling_edges=sibling_edges,
            increment=increment,
            variable_side=0,
            fixed_rows=right_rows,
            dataset=dataset,
            chunk_size=chunk_size,
            seed=seed + 10 * iteration,
        )
        left_rows = orthonormal_complement_rows(
            -left_action,
            old_rows[0],
        )
        write_json(
            status_path,
            {
                "state": "running",
                "phase": "native_block_power",
                "rank_index": rank_index,
                "rank_count": rank_count,
                "rank_increment": increment,
                "iteration": iteration + 1,
                "iterations": iterations,
                "side": "right",
            },
        )
        previous_right = right_rows
        right_action, right_norm = native_child_action(
            source,
            parent=parent,
            sibling_edges=sibling_edges,
            increment=increment,
            variable_side=1,
            fixed_rows=left_rows,
            dataset=dataset,
            chunk_size=chunk_size,
            seed=seed + 10 * iteration + 1,
        )
        right_rows = orthonormal_complement_rows(
            -right_action,
            old_rows[1],
        )
        overlap = float(
            torch.square(
                torch.linalg.matrix_norm(
                    previous_right
                    @ torch.conj(torch.transpose(right_rows, 0, 1)),
                    ord="fro",
                )
            )
            / increment
        )
        diagnostics.append(
            {
                "iteration": float(iteration + 1),
                "left_action_norm": left_norm,
                "right_action_norm": right_norm,
                "right_subspace_overlap": overlap,
            }
        )
    return left_rows, right_rows, diagnostics


def build_aligned_expansion(
    source: PositiveMultiplicationTreeMetric,
    *,
    parent: int,
    sibling_edges: tuple[int, int],
    left_rows: torch.Tensor,
    right_rows: torch.Tensor,
    seed: int,
) -> PositiveMultiplicationTreeMetric:
    increment = int(left_rows.shape[0])
    if right_rows.shape[0] != increment:
        raise ValueError("left and right residual subspaces have unequal ranks")
    source_dimensions = tuple(source.edge_dimensions)
    target = list(source_dimensions)
    for edge in sibling_edges:
        target[edge] += increment
    expanded = expand_multiplication_tree_bonds(
        source,
        tuple(target),
        relative_activation_scale=1.0,
        seed=seed,
        orthogonalize_new_outputs=True,
    )
    names = tuple(child_parameter_name(expanded, edge) for edge in sibling_edges)
    parameters = dict(expanded.named_parameters())
    directions = (left_rows, right_rows)
    with torch.no_grad():
        for edge, name, rows in zip(
            sibling_edges,
            names,
            directions,
            strict=True,
        ):
            old_count = source_dimensions[edge]
            tensor = parameters[name]
            old_rows = tensor[:old_count].reshape(old_count, -1)
            target_norm = torch.mean(
                torch.linalg.vector_norm(old_rows, dim=1)
            )
            tensor[old_count:].copy_(
                (
                    target_norm
                    * rows.to(dtype=tensor.dtype, device=tensor.device)
                ).reshape(tensor[old_count:].shape)
            )
        parent_tensor = expanded.internal_tensors[parent - expanded.leaf_count]
        left_old = source_dimensions[sibling_edges[0]]
        right_old = source_dimensions[sibling_edges[1]]
        parent_tensor[:, left_old:, :].zero_()
        parent_tensor[:, :, right_old:].zero_()
    return expanded


def private_to_cpu(result: dict[str, Any]) -> None:
    if "flat_indices" in result:
        result["flat_indices"] = result["flat_indices"].detach().cpu()
    for candidate in result.get("candidates", []):
        if "_coordinates" in candidate:
            candidate["_coordinates"] = candidate["_coordinates"].detach().cpu()


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite a residual-power run")
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
            raise ValueError("residual rank enrichment requires a compiled tree")
        if bool(payload.get("teacher_runtime_dependency", False)):
            raise ValueError("input checkpoint still depends on the teacher")
        configuration = payload["configuration"]
        exponents = np.asarray(configuration["exponents"], dtype=np.int64)
        whitening = np.asarray(configuration["whitening"], dtype=np.complex128)
        source = build_checkpoint_model(payload, device=device).to(
            dtype=torch.complex64
        )
        if not isinstance(source, PositiveMultiplicationTreeMetric):
            raise TypeError("checkpoint did not reconstruct a multiplication tree")
        source_dimensions = tuple(source.edge_dimensions)
        parent, sibling_edges, _ = sibling_growth_target(source, increment=1)

        excluded = load_excluded_indices(args.exclude_indices_file)
        arrays, indices = load_disjoint_splits(
            {
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
            },
            seed=args.seed,
            exclusions=excluded,
        )
        indices_path = output_dir / "data_indices.npz"
        np.savez_compressed(indices_path, **indices)
        datasets = {
            name: whiten_dataset(
                make_dataset(
                    split,
                    exponents,
                    feature_batch_size=args.feature_batch_size,
                    complex_dtype=torch.complex64,
                    device=device,
                ),
                whitening,
            )
            for name, split in arrays.items()
        }
        baseline_selection, _, baseline_tail = metric_row(
            source,
            datasets["selection"],
            chunk_size=args.eval_batch_size,
        )
        audit_count = min(64, datasets["selection"]["count"])
        audit = dataset_subset(
            datasets["selection"],
            np.arange(audit_count),
        )
        audit["weights_numpy"] = np.asarray(
            datasets["selection"]["weights_numpy"][:audit_count]
        )
        _, baseline_audit_ratio, _ = metric_row(
            source,
            audit,
            chunk_size=args.eval_batch_size,
        )

        rank_results: list[dict[str, Any]] = []
        rank_values = sorted(args.rank_increments)
        for rank_index, increment in enumerate(rank_values, start=1):
            subspace_seed = args.seed + 1000 * increment
            left_rows, right_rows, power_diagnostics = (
                native_block_power_subspaces(
                    source,
                    parent=parent,
                    sibling_edges=sibling_edges,
                    increment=increment,
                    dataset=datasets["fit"],
                    iterations=args.power_iterations,
                    chunk_size=args.operator_chunk_size,
                    seed=subspace_seed,
                    status_path=status_path,
                    rank_index=rank_index,
                    rank_count=len(rank_values),
                )
            )
            expanded = build_aligned_expansion(
                source,
                parent=parent,
                sibling_edges=sibling_edges,
                left_rows=left_rows,
                right_rows=right_rows,
                seed=subspace_seed + 500,
            )
            _, expanded_audit_ratio, _ = metric_row(
                expanded,
                audit,
                chunk_size=args.eval_batch_size,
            )
            embedding_error = float(
                np.max(
                    np.abs(expanded_audit_ratio - baseline_audit_ratio)
                )
            )
            if embedding_error > 2.0e-5:
                raise FloatingPointError(
                    "residual-aligned rank expansion changed the baseline metric"
                )
            result = solve_seed(
                expanded,
                parent=parent,
                source_dimensions=source_dimensions,
                fit=datasets["fit"],
                selection=datasets["selection"],
                args=args,
                seed_index=rank_index,
                proposal_count=len(rank_values),
                status_path=status_path,
            )
            result.update(
                {
                    "rank_increment": increment,
                    "subspace_seed": subspace_seed,
                    "power_diagnostics": power_diagnostics,
                    "embedding_ratio_max_absolute": embedding_error,
                    "target_edge_dimensions": tuple(expanded.edge_dimensions),
                    "_left_rows": left_rows.detach().cpu(),
                    "_right_rows": right_rows.detach().cpu(),
                }
            )
            private_to_cpu(result)
            rank_results.append(result)
            del expanded, left_rows, right_rows
            gc.collect()

        eligible = [
            result for result in rank_results if result["selected"] is not None
        ]
        chosen = (
            min(
                eligible,
                key=lambda result: (
                    result["selected"]["actual_selection_e2"],
                    result["selected"]["selection_statistics"][
                        "sigma_official_formula"
                    ],
                    result["rank_increment"],
                ),
            )
            if eligible
            else None
        )

        proposal_path = None
        proposal_model = None
        if chosen is not None:
            high_source = build_checkpoint_model(payload, device=device)
            proposal_model = build_aligned_expansion(
                high_source,
                parent=parent,
                sibling_edges=sibling_edges,
                left_rows=chosen["_left_rows"],
                right_rows=chosen["_right_rows"],
                seed=int(chosen["subspace_seed"]) + 500,
            )
            set_new_couplings_(
                proposal_model,
                parent=parent,
                flat_indices=chosen["flat_indices"].to(device=device),
                coordinates=chosen["selected"]["_coordinates"].to(
                    device=device,
                    dtype=proposal_model.reference_h.dtype,
                ),
            )
            proposal_payload = copy.deepcopy(payload)
            proposal_payload["state_dict"] = {
                key: value.detach().cpu().clone()
                for key, value in proposal_model.state_dict().items()
            }
            proposal_configuration = dict(configuration)
            proposal_configuration["edge_dimensions"] = tuple(
                proposal_model.edge_dimensions
            )
            proposal_configuration["bond_dimension"] = max(
                proposal_model.edge_dimensions
            )
            proposal_payload["configuration"] = proposal_configuration
            proposal_payload["parent_checkpoint_sha256"] = sha256_file(
                checkpoint_path
            )
            proposal_payload["teacher_runtime_dependency"] = False
            proposal_payload["selection_only"] = True
            proposal_payload["confirmation_passes"] = False
            proposal_payload["native_residual_power_enrichment"] = {
                "parent": parent,
                "sibling_edges": sibling_edges,
                "rank_increment": int(chosen["rank_increment"]),
                "power_iterations": args.power_iterations,
                "teacher_used": False,
            }
            proposal_path = output_dir / "selection_proposal.pt"
            torch.save(proposal_payload, proposal_path)

        report = {
            "schema": "generic-quintic-residual-block-power-rank-v1",
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
                "candidate_rank_increments": rank_values,
                "chosen_rank_increment": (
                    None if chosen is None else int(chosen["rank_increment"])
                ),
                "proposal_method": "native_e2_block_power",
                "exact_embedding_during_subspace_discovery": True,
            },
            "selection": {
                "baseline": baseline_selection,
                "baseline_tail": baseline_tail,
                "rank_results": [
                    {
                        **{
                            key: value
                            for key, value in result.items()
                            if key
                            not in {
                                "candidates",
                                "selected",
                                "flat_indices",
                                "_left_rows",
                                "_right_rows",
                            }
                        },
                        "candidates": [
                            serializable_row(candidate)
                            for candidate in result["candidates"]
                        ],
                        "selected": (
                            None
                            if result["selected"] is None
                            else serializable_row(result["selected"])
                        ),
                    }
                    for result in rank_results
                ],
                "chosen": (
                    None
                    if chosen is None
                    else {
                        "rank_increment": int(chosen["rank_increment"]),
                        **serializable_row(chosen["selected"]),
                    }
                ),
            },
            "confirmation": {
                "opened": False,
                "passes": False,
            },
            "model": {
                "source_real_parameter_count": (
                    source.trainable_real_parameter_count
                ),
                "proposal_real_parameter_count": (
                    None
                    if proposal_model is None
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
