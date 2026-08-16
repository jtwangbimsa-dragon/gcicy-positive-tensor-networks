#!/usr/bin/env python3
"""Native-E2 full-root proposal and adaptive SVD commit for a generic tree."""

from __future__ import annotations

import argparse
import copy
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
    five_leaf_two_three_topology,
    reassociate_five_leaf_tree_to_two_three,
)
from gcicy_metric.pipeline.root_residual_tree import (  # noqa: E402
    RootResidualSupercoreMetric,
    commit_root_residual_svd,
    root_residual_svd,
    root_supercore_matrix,
)
from scripts.audit_quintic_tn_tangent_reachability import (  # noqa: E402
    ComplexParameterVectorizer,
    lanczos_tridiagonal,
    real_inner,
    ridge_dual_from_lanczos,
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
from scripts.train_generic_quintic_adaptive_tree_rank import (  # noqa: E402
    exact_embedding_audit,
    state_to_cpu,
    zero_nonpositive,
)
from scripts.train_generic_quintic_compiled_tree import (  # noqa: E402
    metric_row,
    sha256_file,
    whiten_dataset,
)
from scripts.train_quintic_full_h_same_points import write_json  # noqa: E402
from scripts.train_quintic_native_power_lift_tree import (  # noqa: E402
    dataset_subset,
    make_dataset,
    paired_improvement,
    tail_guard,
)


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
    parser.add_argument("--fit-size", type=int, default=2048)
    parser.add_argument("--selection-size", type=int, default=4096)
    parser.add_argument("--confirmation-size", type=int, default=5000)
    parser.add_argument("--lanczos-steps", type=int, default=16)
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
    parser.add_argument(
        "--rank-candidates",
        type=int,
        nargs="+",
        default=(2, 4, 8, 16, 25),
    )
    parser.add_argument("--minimum-selection-e2-gain", type=float, default=0.005)
    parser.add_argument("--minimum-retained-gain", type=float, default=0.95)
    parser.add_argument("--maximum-relative-core-norm", type=float, default=0.5)
    parser.add_argument(
        "--maximum-selection-tail-relative-degradation",
        type=float,
        default=0.005,
    )
    parser.add_argument("--operator-chunk-size", type=int, default=128)
    parser.add_argument("--feature-batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--audit-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=202607461)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--discovery-only",
        action="store_true",
        help="Save U,V without opening confirmation or committing a one-step model.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.fit_size,
        args.selection_size,
        args.confirmation_size,
        args.lanczos_steps,
        args.operator_chunk_size,
        args.feature_batch_size,
        args.eval_batch_size,
        args.audit_size,
        args.threads,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("all data, batch, iteration, and thread sizes must be positive")
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
    if (
        not args.rank_candidates
        or any(rank <= 0 for rank in args.rank_candidates)
    ):
        raise ValueError("root residual ranks must be positive")
    if not 0 < args.minimum_retained_gain <= 1:
        raise ValueError("retained-gain threshold must lie in (0, 1]")
    if args.minimum_selection_e2_gain <= 0:
        raise ValueError("minimum selection gain must be positive")
    if args.maximum_relative_core_norm <= 0:
        raise ValueError("maximum relative core norm must be positive")


def set_core_(
    model: RootResidualSupercoreMetric,
    matrix: torch.Tensor,
) -> None:
    with torch.no_grad():
        model.residual_core.copy_(matrix)


def candidate_row(
    model: RootResidualSupercoreMetric,
    operator: MatrixFreeNormalizedE2Jacobian,
    vector: torch.Tensor,
    dataset: dict[str, Any],
    *,
    base_e2: float,
    baseline_statistics: dict[str, Any],
    baseline_tail: dict[str, float],
    eval_batch_size: int,
    maximum_tail_degradation: float,
) -> dict[str, Any]:
    residual = operator.residual(vector)
    e2 = float(real_inner(residual, residual))
    gain = 1.0 - e2 / base_e2
    if gain <= 0:
        return {
            "e2": e2,
            "e2_gain": gain,
            "eligible": False,
            "failure": "selection_e2",
        }
    set_core_(model, vector.reshape(model.residual_core.shape))
    statistics, _, tail = metric_row(
        model,
        dataset,
        chunk_size=eval_batch_size,
    )
    eligible = bool(
        statistics["sigma_official_formula"]
        < baseline_statistics["sigma_official_formula"]
        and zero_nonpositive(statistics)
        and tail_guard(
            tail,
            baseline_tail,
            relative_degradation=maximum_tail_degradation,
        )
    )
    return {
        "e2": e2,
        "e2_gain": gain,
        "statistics": statistics,
        "tail": tail,
        "eligible": eligible,
    }


def json_candidate(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result.pop("_vector", None)
    result.pop("_matrix", None)
    return result


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite a root-residual run")
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
            raise ValueError("root residual requires a compiled-tree checkpoint")
        if bool(payload.get("teacher_runtime_dependency", False)):
            raise ValueError("input checkpoint still declares a teacher dependency")
        configuration = payload["configuration"]
        dtype = (
            torch.complex64
            if str(configuration["precision"]) == "complex64"
            else torch.complex128
        )
        exponents = np.asarray(configuration["exponents"], dtype=np.int64)
        whitening = np.asarray(configuration["whitening"], dtype=np.complex128)

        source_model = build_checkpoint_model(payload, device=device)
        if not isinstance(source_model, PositiveMultiplicationTreeMetric):
            raise TypeError("checkpoint did not reconstruct a multiplication tree")
        if source_model.leaf_count != 5:
            raise ValueError("current root proposer requires five degree-four leaves")
        source_model.eval()
        base_model = (
            source_model
            if source_model.topology == five_leaf_two_three_topology()
            else reassociate_five_leaf_tree_to_two_three(source_model)
        )
        base_model.eval()

        excluded = load_excluded_indices(args.exclude_indices_file)
        split_arrays, split_indices = load_disjoint_splits(
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
                "confirmation": (
                    args.confirmation_points,
                    args.confirmation_pullbacks,
                    args.confirmation_size,
                ),
            },
            seed=args.seed,
            exclusions=excluded,
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

        audit_count = min(args.audit_size, datasets["selection"]["count"])
        audit = dataset_subset(
            datasets["selection"],
            np.arange(audit_count),
        )
        rotation_audit = exact_embedding_audit(
            source_model,
            base_model,
            audit,
            chunk_size=args.eval_batch_size,
        )
        write_json(
            status_path,
            {
                "state": "running",
                "phase": "linearization",
                "rotation_audit": rotation_audit,
            },
        )

        baseline_selection, _, baseline_selection_tail = metric_row(
            base_model,
            datasets["selection"],
            chunk_size=args.eval_batch_size,
        )
        residual_model = RootResidualSupercoreMetric.complete_basis(base_model)
        vectorizer = ComplexParameterVectorizer.from_module(
            residual_model,
            ("residual_core",),
        )
        theta = vectorizer.pack(residual_model).detach()
        fit_operator = MatrixFreeNormalizedE2Jacobian(
            residual_model,
            vectorizer.unpack,
            theta,
            datasets["fit"],
            chunk_size=args.operator_chunk_size,
        )
        selection_operator = MatrixFreeNormalizedE2Jacobian(
            residual_model,
            vectorizer.unpack,
            theta,
            datasets["selection"],
            chunk_size=args.operator_chunk_size,
        )
        fit_residual = fit_operator.residual()
        fit_e2 = float(real_inner(fit_residual, fit_residual))
        selection_e2 = selection_operator.native_e2
        gradient = fit_operator.vjp(fit_residual)
        gradient_norm = float(vector_norm(gradient))
        rayleigh_scale = gradient_norm**2 / max(
            fit_e2,
            np.finfo(np.float64).tiny,
        )
        if not np.isfinite(rayleigh_scale) or rayleigh_scale <= 0:
            raise FloatingPointError("root residual has no finite native direction")

        def progress(iteration: int, alpha: float, beta: float) -> None:
            write_json(
                status_path,
                {
                    "state": "running",
                    "phase": "lanczos",
                    "iteration": iteration,
                    "iterations": args.lanczos_steps,
                    "alpha": alpha,
                    "beta": beta,
                },
            )
            if not args.quiet:
                print(
                    f"lanczos={iteration}/{args.lanczos_steps} "
                    f"alpha={alpha:.6e} beta={beta:.6e}",
                    flush=True,
                )

        lanczos = lanczos_tridiagonal(
            fit_operator.normal,
            fit_residual,
            steps=args.lanczos_steps,
            callback=progress,
        )
        base_core_norm = max(
            float(torch.linalg.vector_norm(root_supercore_matrix(base_model))),
            np.finfo(np.float64).tiny,
        )
        full_rows: list[dict[str, Any]] = []
        write_json(status_path, {"state": "running", "phase": "full_selection"})
        for ridge_factor in args.ridge_factors:
            ridge = float(ridge_factor * rayleigh_scale)
            dual = ridge_dual_from_lanczos(lanczos, ridge)
            direction = -fit_operator.vjp(dual)
            for alpha in args.line_search_alphas:
                vector = float(alpha) * direction
                relative_core_norm = float(vector_norm(vector)) / base_core_norm
                row: dict[str, Any] = {
                    "ridge_factor": float(ridge_factor),
                    "ridge": ridge,
                    "alpha": float(alpha),
                    "relative_core_norm": relative_core_norm,
                    "eligible": False,
                    "failure": None,
                    "_vector": vector.detach().clone(),
                }
                if relative_core_norm > args.maximum_relative_core_norm:
                    row["failure"] = "relative_core_norm"
                    full_rows.append(row)
                    continue
                try:
                    fit_candidate = fit_operator.residual(vector)
                    row["fit_e2"] = float(
                        real_inner(fit_candidate, fit_candidate)
                    )
                    row["fit_e2_gain"] = 1.0 - row["fit_e2"] / fit_e2
                    if row["fit_e2_gain"] <= 0:
                        row["failure"] = "fit_e2"
                        full_rows.append(row)
                        continue
                    row.update(
                        candidate_row(
                            residual_model,
                            selection_operator,
                            vector,
                            datasets["selection"],
                            base_e2=selection_e2,
                            baseline_statistics=baseline_selection,
                            baseline_tail=baseline_selection_tail,
                            eval_batch_size=args.eval_batch_size,
                            maximum_tail_degradation=(
                                args.maximum_selection_tail_relative_degradation
                            ),
                        )
                    )
                    row["eligible"] = bool(
                        row["eligible"] and row["fit_e2_gain"] > 0
                    )
                except (RuntimeError, FloatingPointError) as error:
                    row["failure"] = f"{type(error).__name__}: {error}"
                finally:
                    set_core_(residual_model, torch.zeros_like(residual_model.residual_core))
                full_rows.append(row)

        eligible_full = [row for row in full_rows if row["eligible"]]
        best_full = (
            min(
                eligible_full,
                key=lambda row: (
                    row["e2"],
                    row["statistics"]["sigma_official_formula"],
                    row["relative_core_norm"],
                ),
            )
            if eligible_full
            else None
        )

        rank_rows: list[dict[str, Any]] = []
        selected = None
        active_subspace_path = None
        if best_full is not None:
            full_matrix = best_full["_vector"].reshape(
                residual_model.residual_core.shape
            )
            decomposition = root_residual_svd(full_matrix)
            left_shape = residual_model.left_basis.shape[1:]
            right_shape = residual_model.right_basis.shape[1:]
            active_subspace_path = output_dir / "active_subspace.pt"
            torch.save(
                {
                    "schema": "generic-quintic-root-active-subspace-v1",
                    "parent_checkpoint_sha256": sha256_file(checkpoint_path),
                    "topology_children": tuple(base_model.topology.children),
                    "base_edge_dimensions": tuple(base_model.edge_dimensions),
                    "full_matrix": full_matrix.detach().cpu(),
                    "left_basis": decomposition.left.T.reshape(
                        decomposition.rank,
                        *left_shape,
                    ).detach().cpu(),
                    "right_basis": decomposition.right_h.reshape(
                        decomposition.rank,
                        *right_shape,
                    ).detach().cpu(),
                    "singular_values": decomposition.singular_values.detach().cpu(),
                    "initial_core": torch.diag(
                        decomposition.singular_values
                    ).detach().cpu(),
                    "fit_e2": fit_e2,
                    "selection_e2": selection_e2,
                    "best_full": json_candidate(best_full),
                },
                active_subspace_path,
            )
            full_gain = float(best_full["e2_gain"])
            for requested_rank in sorted(set(args.rank_candidates)):
                rank = min(requested_rank, decomposition.rank)
                truncated = (
                    decomposition.left[:, :rank]
                    * decomposition.singular_values[:rank][None, :]
                ) @ decomposition.right_h[:rank]
                vector = truncated.reshape(-1)
                row = {
                    "requested_rank": requested_rank,
                    "rank": rank,
                    "coefficient_energy_retained": (
                        decomposition.retained_fraction(rank)
                    ),
                    "_vector": vector.detach().clone(),
                    "_matrix": truncated.detach().clone(),
                }
                row.update(
                    candidate_row(
                        residual_model,
                        selection_operator,
                        vector,
                        datasets["selection"],
                        base_e2=selection_e2,
                        baseline_statistics=baseline_selection,
                        baseline_tail=baseline_selection_tail,
                        eval_batch_size=args.eval_batch_size,
                        maximum_tail_degradation=(
                            args.maximum_selection_tail_relative_degradation
                        ),
                    )
                )
                row["incremental_gain_retained"] = (
                    row["e2_gain"] / full_gain
                    if full_gain > 0
                    else float("-inf")
                )
                row["eligible"] = bool(
                    row["eligible"]
                    and row["e2_gain"] >= args.minimum_selection_e2_gain
                    and row["incremental_gain_retained"]
                    >= args.minimum_retained_gain
                )
                rank_rows.append(row)
                set_core_(
                    residual_model,
                    torch.zeros_like(residual_model.residual_core),
                )
            eligible_ranks = [row for row in rank_rows if row["eligible"]]
            selected = (
                min(
                    eligible_ranks,
                    key=lambda row: (
                        row["rank"],
                        row["e2"],
                        row["statistics"]["sigma_official_formula"],
                    ),
                )
                if eligible_ranks
                else None
            )

        committed = None
        commit_audit = None
        if selected is not None and not args.discovery_only:
            committed = commit_root_residual_svd(
                base_model,
                selected["_matrix"],
                rank=int(selected["rank"]),
            )
            set_core_(residual_model, selected["_matrix"])
            commit_audit = exact_embedding_audit(
                residual_model,
                committed,
                audit,
                chunk_size=args.eval_batch_size,
            )
            set_core_(
                residual_model,
                torch.zeros_like(residual_model.residual_core),
            )

        confirmation_passes = False
        baseline_confirmation = None
        baseline_confirmation_tail = None
        candidate_confirmation = None
        candidate_confirmation_tail = None
        paired = None
        if committed is not None:
            write_json(
                status_path,
                {
                    "state": "running",
                    "phase": "confirmation",
                    "selected_rank": int(selected["rank"]),
                },
            )
            baseline_confirmation, baseline_ratio, baseline_confirmation_tail = (
                metric_row(
                    base_model,
                    datasets["confirmation"],
                    chunk_size=args.eval_batch_size,
                )
            )
            candidate_confirmation, candidate_ratio, candidate_confirmation_tail = (
                metric_row(
                    committed,
                    datasets["confirmation"],
                    chunk_size=args.eval_batch_size,
                )
            )
            paired = paired_improvement(
                baseline_ratio,
                candidate_ratio,
                datasets["confirmation"]["weights_numpy"],
            )
            confirmation_passes = bool(
                candidate_confirmation["sigma_official_formula"]
                < baseline_confirmation["sigma_official_formula"]
                and candidate_confirmation["weighted_rms_abs_residual"]
                < baseline_confirmation["weighted_rms_abs_residual"]
                and paired["e2"]["ci95_low"] > 0
                and paired["sigma"]["ci95_low"] > 0
                and zero_nonpositive(candidate_confirmation)
                and tail_guard(
                    candidate_confirmation_tail,
                    baseline_confirmation_tail,
                    relative_degradation=0.0,
                )
            )

        final_model = committed if confirmation_passes else base_model
        final_configuration = copy.deepcopy(configuration)
        final_configuration.update(
            {
                "architecture": "compiled-tree",
                "bond_dimension": max(final_model.edge_dimensions),
                "edge_dimensions": tuple(final_model.edge_dimensions),
                "topology_children": tuple(final_model.topology.children),
                "root_residual_round": int(
                    configuration.get("root_residual_round", 0)
                )
                + (1 if confirmation_passes else 0),
            }
        )
        output_checkpoint = output_dir / (
            "discovery_baseline.pt"
            if args.discovery_only
            else (
                "accepted.pt"
                if confirmation_passes
                else "diagnostic_rejected.pt"
            )
        )
        torch.save(
            {
                "schema": "generic-quintic-root-residual-tree-v1",
                "state_dict": state_to_cpu(final_model),
                "configuration": final_configuration,
                "parent_checkpoint_sha256": sha256_file(checkpoint_path),
                "teacher_runtime_dependency": False,
                "confirmation_passes": confirmation_passes,
            },
            output_checkpoint,
        )
        indices_path = output_dir / "data_indices.npz"
        np.savez_compressed(indices_path, **split_indices)
        report = {
            "schema": "generic-quintic-root-residual-proposal-v1",
            "teacher_role": "absent",
            "objective": {
                "name": "model-normalized native MA E2",
                "normalization_derivative_included": True,
            },
            "configuration": {
                **vars(args),
                "initial_checkpoint": str(checkpoint_path),
                "output_dir": str(output_dir),
            },
            "parent": {
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "source_topology": list(source_model.topology.children),
                "reassociated_topology": list(base_model.topology.children),
                "source_edge_dimensions": list(source_model.edge_dimensions),
                "reassociated_edge_dimensions": list(base_model.edge_dimensions),
                "rotation_audit": rotation_audit,
                "baseline_selection": baseline_selection,
                "baseline_selection_tail": baseline_selection_tail,
            },
            "linearization": {
                "complete_core_shape": list(residual_model.residual_core.shape),
                "complete_complex_parameters": int(
                    residual_model.residual_core.numel()
                ),
                "fit_e2": fit_e2,
                "selection_e2": selection_e2,
                "gradient_norm": gradient_norm,
                "rayleigh_scale": rayleigh_scale,
                "base_core_norm": base_core_norm,
                "lanczos": {
                    "dimension": int(lanczos.tridiagonal.shape[0]),
                    "operator_applications": lanczos.operator_applications,
                    "breakdown": lanczos.breakdown,
                    "orthogonality_error": lanczos.orthogonality_error,
                },
            },
            "full_proposals": [json_candidate(row) for row in full_rows],
            "best_full": (
                json_candidate(best_full) if best_full is not None else None
            ),
            "rank_candidates": [json_candidate(row) for row in rank_rows],
            "selected": (
                json_candidate(selected) if selected is not None else None
            ),
            "active_subspace": (
                {
                    "path": str(active_subspace_path),
                    "sha256": sha256_file(active_subspace_path),
                }
                if active_subspace_path is not None
                else None
            ),
            "commit_audit": commit_audit,
            "confirmation": {
                "opened": committed is not None,
                "baseline": baseline_confirmation,
                "baseline_tail": baseline_confirmation_tail,
                "candidate": candidate_confirmation,
                "candidate_tail": candidate_confirmation_tail,
                "paired_improvement": paired,
                "passes": confirmation_passes,
            },
            "data": {
                "indices": str(indices_path),
                "indices_sha256": sha256_file(indices_path),
                "counts": {
                    name: int(len(rows))
                    for name, rows in split_indices.items()
                },
                "excluded_indices_files": [
                    str(path.expanduser().resolve())
                    for path in args.exclude_indices_file
                ],
            },
            "checkpoint": str(output_checkpoint),
            "checkpoint_sha256": sha256_file(output_checkpoint),
            "wall_seconds": time.perf_counter() - started,
        }
        write_json(output_dir / "report.json", report)
        write_json(
            status_path,
            {
                "state": "complete",
                "phase": "complete",
                "full_proposal_found": best_full is not None,
                "rank_candidate_found": selected is not None,
                "confirmation_passes": confirmation_passes,
                "checkpoint": str(output_checkpoint),
                "wall_seconds": time.perf_counter() - started,
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
