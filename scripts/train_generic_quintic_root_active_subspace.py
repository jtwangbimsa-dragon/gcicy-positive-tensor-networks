#!/usr/bin/env python3
"""Exploit one discovered root active subspace with multi-step native-E2 GN."""

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
    parser.add_argument("--active-subspace", type=Path, required=True)
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
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument(
        "--initial-core-scale",
        type=float,
        default=1.0,
        help="Scale the discovery proposal inside the fixed U,V subspace.",
    )
    parser.add_argument("--fit-size", type=int, default=4096)
    parser.add_argument("--selection-size", type=int, default=4096)
    parser.add_argument("--confirmation-size", type=int, default=5000)
    parser.add_argument("--maximum-inner-steps", type=int, default=20)
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
        default=(0.0625, 0.125, 0.25, 0.5, 1.0),
    )
    parser.add_argument("--maximum-nonlinear-candidates", type=int, default=6)
    parser.add_argument("--minimum-trust-ratio", type=float, default=0.05)
    parser.add_argument(
        "--minimum-fit-relative-improvement",
        type=float,
        default=1.0e-5,
    )
    parser.add_argument(
        "--minimum-relative-gradient-norm",
        type=float,
        default=1.0e-3,
    )
    parser.add_argument("--maximum-relative-core-norm", type=float, default=0.5)
    parser.add_argument("--selection-interval", type=int, default=2)
    parser.add_argument("--selection-patience", type=int, default=3)
    parser.add_argument(
        "--minimum-selection-relative-improvement",
        type=float,
        default=1.0e-4,
    )
    parser.add_argument(
        "--maximum-selection-tail-relative-degradation",
        type=float,
        default=0.005,
    )
    parser.add_argument("--operator-chunk-size", type=int, default=128)
    parser.add_argument("--feature-batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--audit-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=202607468)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.rank,
        args.fit_size,
        args.selection_size,
        args.confirmation_size,
        args.maximum_inner_steps,
        args.lanczos_steps,
        args.maximum_nonlinear_candidates,
        args.selection_interval,
        args.selection_patience,
        args.operator_chunk_size,
        args.feature_batch_size,
        args.eval_batch_size,
        args.audit_size,
        args.threads,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("all ranks, sizes, iterations, and threads must be positive")
    if not np.isfinite(args.initial_core_scale) or args.initial_core_scale < 0:
        raise ValueError("initial core scale must be finite and nonnegative")
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
    if args.minimum_trust_ratio < 0:
        raise ValueError("minimum trust ratio must be nonnegative")
    if args.minimum_fit_relative_improvement < 0:
        raise ValueError("minimum fit improvement must be nonnegative")
    if args.minimum_relative_gradient_norm < 0:
        raise ValueError("minimum relative gradient norm must be nonnegative")
    if args.maximum_relative_core_norm <= 0:
        raise ValueError("maximum relative core norm must be positive")
    if args.minimum_selection_relative_improvement < 0:
        raise ValueError("minimum selection improvement must be nonnegative")


def set_core_(
    model: RootResidualSupercoreMetric,
    vectorizer: ComplexParameterVectorizer,
    vector: torch.Tensor,
) -> None:
    with torch.no_grad():
        model.residual_core.copy_(
            vectorizer.unpack(vector)["residual_core"]
        )


def selection_is_eligible(
    statistics: dict[str, Any],
    tail: dict[str, float],
    baseline_statistics: dict[str, Any],
    baseline_tail: dict[str, float],
    *,
    maximum_tail_degradation: float,
) -> bool:
    return bool(
        statistics["sigma_official_formula"]
        < baseline_statistics["sigma_official_formula"]
        and statistics["weighted_rms_abs_residual"]
        < baseline_statistics["weighted_rms_abs_residual"]
        and zero_nonpositive(statistics)
        and tail_guard(
            tail,
            baseline_tail,
            relative_degradation=maximum_tail_degradation,
        )
    )


def json_candidate(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result.pop("_vector", None)
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
        raise FileExistsError("refusing to overwrite an active-subspace run")
    output_dir.mkdir(parents=True)
    status_path = output_dir / "status.json"
    write_json(status_path, {"state": "running", "phase": "loading"})
    started = time.perf_counter()

    try:
        checkpoint_path = args.initial_checkpoint.expanduser().resolve()
        checkpoint_sha256 = sha256_file(checkpoint_path)
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        if infer_architecture(payload) != "compiled-tree":
            raise ValueError("active-subspace training requires a tree checkpoint")
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
        source_model.eval()
        base_model = (
            source_model
            if source_model.topology == five_leaf_two_three_topology()
            else reassociate_five_leaf_tree_to_two_three(source_model)
        )
        base_model.eval()

        artifact_path = args.active_subspace.expanduser().resolve()
        artifact = torch.load(
            artifact_path,
            map_location="cpu",
            weights_only=False,
        )
        if artifact.get("schema") != "generic-quintic-root-active-subspace-v1":
            raise ValueError("unsupported active-subspace artifact")
        if artifact.get("parent_checkpoint_sha256") != checkpoint_sha256:
            raise ValueError("active subspace was discovered from another checkpoint")
        artifact_topology = tuple(
            tuple(int(value) for value in pair)
            for pair in artifact["topology_children"]
        )
        if artifact_topology != base_model.topology.children:
            raise ValueError("active subspace uses another root topology")
        available_rank = int(len(artifact["singular_values"]))
        if args.rank > available_rank:
            raise ValueError("requested rank exceeds the discovered subspace")

        left_basis = artifact["left_basis"][: args.rank].to(
            device=device,
            dtype=dtype,
        )
        right_basis = artifact["right_basis"][: args.rank].to(
            device=device,
            dtype=dtype,
        )
        singular_values = artifact["singular_values"][: args.rank].to(
            device=device,
            dtype=torch.float32 if dtype == torch.complex64 else torch.float64,
        )
        initial_core = torch.diag(singular_values.to(dtype=dtype))
        initial_core = args.initial_core_scale * initial_core
        residual_model = RootResidualSupercoreMetric(
            base_model,
            left_basis=left_basis,
            right_basis=right_basis,
            initial_core=initial_core,
        )
        residual_model.eval()
        vectorizer = ComplexParameterVectorizer.from_module(
            residual_model,
            ("residual_core",),
        )
        theta = vectorizer.pack(residual_model).detach()
        base_core_norm = max(
            float(torch.linalg.vector_norm(root_supercore_matrix(base_model))),
            np.finfo(np.float64).tiny,
        )

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
        zero_model = RootResidualSupercoreMetric(
            base_model,
            left_basis=left_basis,
            right_basis=right_basis,
        )
        zero_audit = exact_embedding_audit(
            base_model,
            zero_model,
            audit,
            chunk_size=args.eval_batch_size,
        )

        baseline_selection, _, baseline_selection_tail = metric_row(
            base_model,
            datasets["selection"],
            chunk_size=args.eval_batch_size,
        )
        initial_selection, _, initial_selection_tail = metric_row(
            residual_model,
            datasets["selection"],
            chunk_size=args.eval_batch_size,
        )
        initial_eligible = selection_is_eligible(
            initial_selection,
            initial_selection_tail,
            baseline_selection,
            baseline_selection_tail,
            maximum_tail_degradation=(
                args.maximum_selection_tail_relative_degradation
            ),
        )
        best_core = theta.detach().clone() if initial_eligible else None
        best_selection = initial_selection if initial_eligible else None
        best_selection_tail = initial_selection_tail if initial_eligible else None
        best_selection_e2 = (
            initial_selection["weighted_rms_abs_residual"] ** 2
            if initial_eligible
            else float("inf")
        )
        best_step = 0 if initial_eligible else None
        initial_gradient_norm = None
        selection_stalls = 0
        stop_reason = "maximum_inner_steps"
        trajectory: list[dict[str, Any]] = []

        for step in range(1, args.maximum_inner_steps + 1):
            write_json(
                status_path,
                {
                    "state": "running",
                    "phase": "inner_linearization",
                    "step": step,
                    "maximum_steps": args.maximum_inner_steps,
                    "best_step": best_step,
                },
            )
            set_core_(residual_model, vectorizer, theta)
            fit_operator = MatrixFreeNormalizedE2Jacobian(
                residual_model,
                vectorizer.unpack,
                theta,
                datasets["fit"],
                chunk_size=args.operator_chunk_size,
            )
            fit_residual = fit_operator.residual()
            fit_e2 = float(real_inner(fit_residual, fit_residual))
            gradient = fit_operator.vjp(fit_residual)
            gradient_norm = float(vector_norm(gradient))
            if initial_gradient_norm is None:
                initial_gradient_norm = max(
                    gradient_norm,
                    np.finfo(np.float64).tiny,
                )
            relative_gradient_norm = gradient_norm / initial_gradient_norm
            if relative_gradient_norm <= args.minimum_relative_gradient_norm:
                stop_reason = "projected_gradient"
                break
            rayleigh_scale = gradient_norm**2 / max(
                fit_e2,
                np.finfo(np.float64).tiny,
            )
            if not np.isfinite(rayleigh_scale) or rayleigh_scale <= 0:
                stop_reason = "nonfinite_rayleigh_scale"
                break

            def progress(iteration: int, alpha: float, beta: float) -> None:
                write_json(
                    status_path,
                    {
                        "state": "running",
                        "phase": "inner_lanczos",
                        "step": step,
                        "maximum_steps": args.maximum_inner_steps,
                        "iteration": iteration,
                        "iterations": args.lanczos_steps,
                        "alpha": alpha,
                        "beta": beta,
                    },
                )

            lanczos = lanczos_tridiagonal(
                fit_operator.normal,
                fit_residual,
                steps=args.lanczos_steps,
                callback=progress,
            )
            predicted_rows: list[dict[str, Any]] = []
            for ridge_factor in args.ridge_factors:
                ridge = float(ridge_factor * rayleigh_scale)
                dual = ridge_dual_from_lanczos(lanczos, ridge)
                direction = -fit_operator.vjp(dual)
                tangent = fit_operator.jvp(direction)
                for alpha in args.line_search_alphas:
                    candidate = theta + float(alpha) * direction
                    relative_core_norm = float(vector_norm(candidate)) / base_core_norm
                    predicted_residual = fit_residual + float(alpha) * tangent
                    predicted_e2 = float(
                        real_inner(predicted_residual, predicted_residual)
                    )
                    predicted_reduction = fit_e2 - predicted_e2
                    row: dict[str, Any] = {
                        "ridge_factor": float(ridge_factor),
                        "ridge": ridge,
                        "alpha": float(alpha),
                        "relative_core_norm": relative_core_norm,
                        "predicted_e2": predicted_e2,
                        "predicted_reduction": predicted_reduction,
                        "_vector": candidate.detach().clone(),
                    }
                    row["eligible_prediction"] = bool(
                        relative_core_norm <= args.maximum_relative_core_norm
                        and predicted_reduction > 0
                    )
                    predicted_rows.append(row)

            shortlisted = sorted(
                (
                    row
                    for row in predicted_rows
                    if row["eligible_prediction"]
                ),
                key=lambda row: (
                    row["predicted_e2"],
                    row["relative_core_norm"],
                ),
            )[: args.maximum_nonlinear_candidates]
            nonlinear_rows: list[dict[str, Any]] = []
            for row in shortlisted:
                candidate_residual = fit_operator.residual(row["_vector"])
                candidate_e2 = float(
                    real_inner(candidate_residual, candidate_residual)
                )
                actual_reduction = fit_e2 - candidate_e2
                trust_ratio = actual_reduction / max(
                    row["predicted_reduction"],
                    np.finfo(np.float64).tiny,
                )
                row.update(
                    {
                        "actual_e2": candidate_e2,
                        "actual_reduction": actual_reduction,
                        "actual_relative_improvement": (
                            actual_reduction / fit_e2
                        ),
                        "trust_ratio": trust_ratio,
                        "eligible": bool(
                            actual_reduction
                            >= args.minimum_fit_relative_improvement * fit_e2
                            and trust_ratio >= args.minimum_trust_ratio
                        ),
                    }
                )
                nonlinear_rows.append(row)
            eligible = [row for row in nonlinear_rows if row["eligible"]]
            if not eligible:
                trajectory.append(
                    {
                        "step": step,
                        "fit_e2": fit_e2,
                        "gradient_norm": gradient_norm,
                        "relative_gradient_norm": relative_gradient_norm,
                        "rayleigh_scale": rayleigh_scale,
                        "lanczos_dimension": int(
                            lanczos.tridiagonal.shape[0]
                        ),
                        "predicted_candidates": [
                            json_candidate(row) for row in predicted_rows
                        ],
                        "nonlinear_candidates": [
                            json_candidate(row) for row in nonlinear_rows
                        ],
                        "accepted": False,
                    }
                )
                stop_reason = "no_trustworthy_fit_step"
                break

            accepted = min(
                eligible,
                key=lambda row: (
                    row["actual_e2"],
                    row["relative_core_norm"],
                ),
            )
            theta = accepted["_vector"].detach().clone()
            set_core_(residual_model, vectorizer, theta)
            row_report: dict[str, Any] = {
                "step": step,
                "fit_e2": fit_e2,
                "gradient_norm": gradient_norm,
                "relative_gradient_norm": relative_gradient_norm,
                "rayleigh_scale": rayleigh_scale,
                "lanczos_dimension": int(lanczos.tridiagonal.shape[0]),
                "lanczos_operator_applications": lanczos.operator_applications,
                "accepted": True,
                "accepted_candidate": json_candidate(accepted),
                "nonlinear_candidates": [
                    json_candidate(row) for row in nonlinear_rows
                ],
            }

            evaluate_selection = bool(
                step == 1
                or step % args.selection_interval == 0
                or step == args.maximum_inner_steps
            )
            if evaluate_selection:
                write_json(
                    status_path,
                    {
                        "state": "running",
                        "phase": "inner_selection",
                        "step": step,
                        "maximum_steps": args.maximum_inner_steps,
                    },
                )
                selection_statistics, _, selection_tail = metric_row(
                    residual_model,
                    datasets["selection"],
                    chunk_size=args.eval_batch_size,
                )
                selection_e2 = (
                    selection_statistics["weighted_rms_abs_residual"] ** 2
                )
                selection_eligible = selection_is_eligible(
                    selection_statistics,
                    selection_tail,
                    baseline_selection,
                    baseline_selection_tail,
                    maximum_tail_degradation=(
                        args.maximum_selection_tail_relative_degradation
                    ),
                )
                improved = bool(
                    selection_eligible
                    and selection_e2
                    < best_selection_e2
                    * (1.0 - args.minimum_selection_relative_improvement)
                )
                row_report.update(
                    {
                        "selection": selection_statistics,
                        "selection_tail": selection_tail,
                        "selection_e2": selection_e2,
                        "selection_eligible": selection_eligible,
                        "selection_improved": improved,
                    }
                )
                if improved:
                    best_core = theta.detach().clone()
                    best_selection = selection_statistics
                    best_selection_tail = selection_tail
                    best_selection_e2 = selection_e2
                    best_step = step
                    selection_stalls = 0
                else:
                    selection_stalls += 1
                if selection_stalls >= args.selection_patience:
                    stop_reason = "selection_patience"
                    trajectory.append(row_report)
                    break
            trajectory.append(row_report)
            if not args.quiet:
                print(
                    f"step={step} fit_gain="
                    f"{accepted['actual_relative_improvement']:.6e} "
                    f"trust={accepted['trust_ratio']:.4f} "
                    f"best_step={best_step}",
                    flush=True,
                )

        if (
            trajectory
            and trajectory[-1]["step"] % args.selection_interval != 0
            and trajectory[-1]["step"] != 1
        ):
            set_core_(residual_model, vectorizer, theta)
            final_selection, _, final_selection_tail = metric_row(
                residual_model,
                datasets["selection"],
                chunk_size=args.eval_batch_size,
            )
            final_selection_e2 = (
                final_selection["weighted_rms_abs_residual"] ** 2
            )
            final_eligible = selection_is_eligible(
                final_selection,
                final_selection_tail,
                baseline_selection,
                baseline_selection_tail,
                maximum_tail_degradation=(
                    args.maximum_selection_tail_relative_degradation
                ),
            )
            if final_eligible and final_selection_e2 < best_selection_e2:
                best_core = theta.detach().clone()
                best_selection = final_selection
                best_selection_tail = final_selection_tail
                best_selection_e2 = final_selection_e2
                best_step = trajectory[-1]["step"]

        committed = None
        commit_rank = 0
        commit_audit = None
        best_complete_matrix = None
        if best_core is not None:
            set_core_(residual_model, vectorizer, best_core)
            best_complete_matrix = (
                residual_model.complete_residual_matrix().detach()
            )
            decomposition = root_residual_svd(best_complete_matrix)
            if decomposition.rank:
                threshold = (
                    float(decomposition.singular_values[0])
                    * max(best_complete_matrix.shape)
                    * torch.finfo(decomposition.singular_values.dtype).eps
                )
                commit_rank = int(
                    torch.sum(decomposition.singular_values > threshold)
                )
            if commit_rank:
                committed = commit_root_residual_svd(
                    base_model,
                    best_complete_matrix,
                    rank=commit_rank,
                )
                commit_audit = exact_embedding_audit(
                    residual_model,
                    committed,
                    audit,
                    chunk_size=args.eval_batch_size,
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
                    "best_step": best_step,
                    "commit_rank": commit_rank,
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
                "root_active_subspace_round": int(
                    configuration.get("root_active_subspace_round", 0)
                )
                + (1 if confirmation_passes else 0),
                "root_active_subspace_rank": args.rank,
                "root_active_subspace_commit_rank": commit_rank,
                "root_active_subspace_best_step": best_step,
            }
        )
        output_checkpoint = output_dir / (
            "accepted.pt"
            if confirmation_passes
            else "baseline_retained.pt"
        )
        torch.save(
            {
                "schema": "generic-quintic-root-active-subspace-tree-v1",
                "state_dict": state_to_cpu(final_model),
                "configuration": final_configuration,
                "parent_checkpoint_sha256": checkpoint_sha256,
                "active_subspace_sha256": sha256_file(artifact_path),
                "teacher_runtime_dependency": False,
                "confirmation_passes": confirmation_passes,
            },
            output_checkpoint,
        )
        active_state_path = output_dir / "active_subspace_state.pt"
        torch.save(
            {
                "schema": "generic-quintic-root-active-subspace-state-v1",
                "parent_checkpoint_sha256": checkpoint_sha256,
                "active_subspace_sha256": sha256_file(artifact_path),
                "rank": args.rank,
                "left_basis": left_basis.detach().cpu(),
                "right_basis": right_basis.detach().cpu(),
                "best_core": (
                    best_core.detach().cpu()
                    if best_core is not None
                    else None
                ),
                "best_complete_matrix": (
                    best_complete_matrix.detach().cpu()
                    if best_complete_matrix is not None
                    else None
                ),
                "best_step": best_step,
                "stop_reason": stop_reason,
            },
            active_state_path,
        )
        indices_path = output_dir / "data_indices.npz"
        np.savez_compressed(indices_path, **split_indices)
        report = {
            "schema": "generic-quintic-root-active-subspace-training-v1",
            "teacher_role": "absent",
            "algorithm": {
                "outer": "frozen residual-aligned U,V from discovery artifact",
                "inner": "multi-step normalized-native-E2 GN/LM in C",
                "confirmation_uses": 1 if committed is not None else 0,
            },
            "configuration": {
                **vars(args),
                "initial_checkpoint": str(checkpoint_path),
                "active_subspace": str(artifact_path),
                "output_dir": str(output_dir),
            },
            "parent": {
                "checkpoint_sha256": checkpoint_sha256,
                "source_topology": list(source_model.topology.children),
                "reassociated_topology": list(base_model.topology.children),
                "source_edge_dimensions": list(source_model.edge_dimensions),
                "reassociated_edge_dimensions": list(base_model.edge_dimensions),
                "rotation_audit": rotation_audit,
                "zero_core_audit": zero_audit,
            },
            "active_subspace": {
                "artifact_sha256": sha256_file(artifact_path),
                "available_rank": available_rank,
                "rank": args.rank,
                "trainable_complex_parameters": int(
                    residual_model.residual_core.numel()
                ),
                "trainable_real_parameters": (
                    residual_model.trainable_real_parameter_count
                ),
                "initial_core_scale": args.initial_core_scale,
                "base_core_norm": base_core_norm,
            },
            "selection": {
                "baseline": baseline_selection,
                "baseline_tail": baseline_selection_tail,
                "initial": initial_selection,
                "initial_tail": initial_selection_tail,
                "initial_eligible": initial_eligible,
                "best": best_selection,
                "best_tail": best_selection_tail,
                "best_e2": (
                    best_selection_e2
                    if np.isfinite(best_selection_e2)
                    else None
                ),
                "best_step": best_step,
            },
            "trajectory": trajectory,
            "stop_reason": stop_reason,
            "commit": {
                "rank": commit_rank,
                "audit": commit_audit,
            },
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
            "active_state": str(active_state_path),
            "active_state_sha256": sha256_file(active_state_path),
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
                "best_step": best_step,
                "stop_reason": stop_reason,
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
