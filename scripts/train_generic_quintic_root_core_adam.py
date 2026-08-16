#!/usr/bin/env python3
"""Directly train a full or fixed-subspace root residual, then compress it."""

from __future__ import annotations

import argparse
import copy
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
    native_e2_streaming_backward,
    paired_improvement,
    tail_guard,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--active-subspace",
        type=Path,
        help="Optional saved U,V basis. Omit for the complete 25x25 root.",
    )
    parser.add_argument("--subspace-rank", type=int)
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
    parser.add_argument("--train-size", type=int, default=20000)
    parser.add_argument("--selection-size", type=int, default=4096)
    parser.add_argument("--confirmation-size", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--stochastic-batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--maximum-relative-core-norm", type=float, default=0.75)
    parser.add_argument("--selection-eval-every", type=int, default=10)
    parser.add_argument("--maximum-training-seconds", type=float, default=300.0)
    parser.add_argument(
        "--rank-candidates",
        type=int,
        nargs="+",
        default=(2, 4, 8, 16, 25),
    )
    parser.add_argument("--minimum-retained-gain", type=float, default=0.95)
    parser.add_argument("--train-chunk-size", type=int, default=256)
    parser.add_argument("--feature-batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--audit-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=202607470)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.train_size,
        args.selection_size,
        args.confirmation_size,
        args.epochs,
        args.stochastic_batch_size,
        args.learning_rate,
        args.gradient_clip_norm,
        args.maximum_relative_core_norm,
        args.selection_eval_every,
        args.maximum_training_seconds,
        args.train_chunk_size,
        args.feature_batch_size,
        args.eval_batch_size,
        args.audit_size,
        args.threads,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("all sizes, budgets, and optimization values must be positive")
    if (args.active_subspace is None) != (args.subspace_rank is None):
        raise ValueError("active-subspace path and rank must be supplied together")
    if args.subspace_rank is not None and args.subspace_rank <= 0:
        raise ValueError("subspace rank must be positive")
    if (
        not args.rank_candidates
        or any(rank <= 0 for rank in args.rank_candidates)
    ):
        raise ValueError("compression ranks must be positive")
    if not 0 < args.minimum_retained_gain <= 1:
        raise ValueError("retained-gain threshold must lie in (0, 1]")


def load_root_model(
    base_model: PositiveMultiplicationTreeMetric,
    args: argparse.Namespace,
    *,
    checkpoint_sha256: str,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[RootResidualSupercoreMetric, dict[str, Any]]:
    if args.active_subspace is None:
        return RootResidualSupercoreMetric.complete_basis(base_model), {
            "kind": "complete",
            "complex_parameters": 625,
        }
    path = args.active_subspace.expanduser().resolve()
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if artifact.get("schema") != "generic-quintic-root-active-subspace-v1":
        raise ValueError("unsupported active-subspace artifact")
    if artifact.get("parent_checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("active subspace belongs to another checkpoint")
    rank = int(args.subspace_rank)
    if rank > len(artifact["singular_values"]):
        raise ValueError("subspace rank exceeds the discovered basis")
    model = RootResidualSupercoreMetric(
        base_model,
        left_basis=artifact["left_basis"][:rank].to(device=device, dtype=dtype),
        right_basis=artifact["right_basis"][:rank].to(device=device, dtype=dtype),
    )
    return model, {
        "kind": "fixed-subspace",
        "artifact": str(path),
        "artifact_sha256": sha256_file(path),
        "rank": rank,
        "complex_parameters": rank * rank,
    }


def gain_retained(
    baseline: float,
    full: float,
    candidate: float,
) -> float:
    full_gain = baseline - full
    if full_gain <= 0:
        return float("-inf")
    return (baseline - candidate) / full_gain


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
        raise FileExistsError("refusing to overwrite a direct root-core run")
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
            raise ValueError("direct root training requires a tree checkpoint")
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
        model, arm = load_root_model(
            base_model,
            args,
            checkpoint_sha256=checkpoint_sha256,
            dtype=dtype,
            device=device,
        )
        model.train()
        base_core_norm = max(
            float(torch.linalg.vector_norm(root_supercore_matrix(base_model))),
            np.finfo(np.float64).tiny,
        )

        excluded = load_excluded_indices(args.exclude_indices_file)
        split_arrays, split_indices = load_disjoint_splits(
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
        audit = dataset_subset(
            datasets["selection"],
            np.arange(min(args.audit_size, datasets["selection"]["count"])),
        )
        rotation_audit = exact_embedding_audit(
            source_model,
            base_model,
            audit,
            chunk_size=args.eval_batch_size,
        )
        zero_audit = exact_embedding_audit(
            base_model,
            model,
            audit,
            chunk_size=args.eval_batch_size,
        )
        baseline_selection, _, baseline_selection_tail = metric_row(
            base_model,
            datasets["selection"],
            chunk_size=args.eval_batch_size,
        )
        best_core = torch.zeros_like(model.residual_core, device="cpu")
        best_epoch = 0
        best_selection = baseline_selection
        best_selection_tail = baseline_selection_tail
        history: list[dict[str, Any]] = []
        optimizer = torch.optim.Adam(
            (model.residual_core,),
            lr=args.learning_rate,
        )
        rng = np.random.default_rng(args.seed + 17)
        training_started = time.perf_counter()
        stop_reason = "epochs"

        for epoch in range(1, args.epochs + 1):
            optimizer.zero_grad(set_to_none=True)
            count = min(args.stochastic_batch_size, datasets["fit"]["count"])
            indices = rng.choice(
                datasets["fit"]["count"],
                size=count,
                replace=False,
            )
            batch = dataset_subset(datasets["fit"], indices)
            train_e2, _ = native_e2_streaming_backward(
                model,
                batch,
                chunk_size=args.train_chunk_size,
            )
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    (model.residual_core,),
                    args.gradient_clip_norm,
                )
            )
            previous = model.residual_core.detach().clone()
            optimizer.step()
            with torch.no_grad():
                relative_core_norm = (
                    float(torch.linalg.vector_norm(model.residual_core))
                    / base_core_norm
                )
                if relative_core_norm > args.maximum_relative_core_norm:
                    model.residual_core.mul_(
                        args.maximum_relative_core_norm / relative_core_norm
                    )
                    relative_core_norm = args.maximum_relative_core_norm
                relative_step = (
                    float(
                        torch.linalg.vector_norm(
                            model.residual_core - previous
                        )
                    )
                    / base_core_norm
                )

            evaluate = bool(
                epoch % args.selection_eval_every == 0
                or epoch == args.epochs
                or time.perf_counter() - training_started
                >= args.maximum_training_seconds
            )
            selection = None
            selection_tail = None
            accepted = False
            if evaluate:
                selection, _, selection_tail = metric_row(
                    model,
                    datasets["selection"],
                    chunk_size=args.eval_batch_size,
                )
                accepted = bool(
                    selection["sigma_official_formula"]
                    < baseline_selection["sigma_official_formula"]
                    and selection["weighted_rms_abs_residual"]
                    < best_selection["weighted_rms_abs_residual"]
                    and zero_nonpositive(selection)
                )
                if accepted:
                    best_epoch = epoch
                    best_core = model.residual_core.detach().cpu().clone()
                    best_selection = selection
                    best_selection_tail = selection_tail
            row = {
                "epoch": epoch,
                "train_e2": float(train_e2),
                "gradient_norm": gradient_norm,
                "relative_step": relative_step,
                "relative_core_norm": relative_core_norm,
                "selection": selection,
                "selection_tail": selection_tail,
                "accepted": accepted,
                "wall_seconds": time.perf_counter() - training_started,
            }
            history.append(row)
            write_json(
                status_path,
                {
                    "state": "running",
                    "phase": "training",
                    "arm": arm,
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "latest": row,
                },
            )
            if not args.quiet and evaluate:
                print(
                    f"epoch={epoch} train_e2={float(train_e2):.6e} "
                    f"sigma={selection['sigma_official_formula']:.6e} "
                    f"chi={selection['weighted_rms_abs_residual']:.6e} "
                    f"best={best_epoch}",
                    flush=True,
                )
            if time.perf_counter() - training_started >= args.maximum_training_seconds:
                stop_reason = "wall_time"
                break

        with torch.no_grad():
            model.residual_core.copy_(
                best_core.to(device=device, dtype=dtype)
            )
        full_matrix = model.complete_residual_matrix().detach()
        decomposition = root_residual_svd(full_matrix)
        rank_rows: list[dict[str, Any]] = []
        baseline_sigma = baseline_selection["sigma_official_formula"]
        baseline_e2 = baseline_selection["weighted_rms_abs_residual"] ** 2
        full_sigma = best_selection["sigma_official_formula"]
        full_e2 = best_selection["weighted_rms_abs_residual"] ** 2
        selected_rank = None
        selected_matrix = None
        if best_epoch > 0:
            for requested_rank in sorted(set(args.rank_candidates)):
                rank = min(requested_rank, decomposition.rank)
                truncated = (
                    decomposition.left[:, :rank]
                    * decomposition.singular_values[:rank][None, :]
                ) @ decomposition.right_h[:rank]
                complete = RootResidualSupercoreMetric.complete_basis(base_model)
                with torch.no_grad():
                    complete.residual_core.copy_(truncated)
                statistics, _, tail = metric_row(
                    complete,
                    datasets["selection"],
                    chunk_size=args.eval_batch_size,
                )
                e2 = statistics["weighted_rms_abs_residual"] ** 2
                e2_retained = gain_retained(baseline_e2, full_e2, e2)
                sigma_retained = gain_retained(
                    baseline_sigma,
                    full_sigma,
                    statistics["sigma_official_formula"],
                )
                eligible = bool(
                    min(e2_retained, sigma_retained)
                    >= args.minimum_retained_gain
                    and zero_nonpositive(statistics)
                )
                rank_rows.append(
                    {
                        "requested_rank": requested_rank,
                        "rank": rank,
                        "coefficient_energy_retained": (
                            decomposition.retained_fraction(rank)
                        ),
                        "e2_gain_retained": e2_retained,
                        "sigma_gain_retained": sigma_retained,
                        "statistics": statistics,
                        "tail": tail,
                        "eligible": eligible,
                    }
                )
                if selected_rank is None and eligible:
                    selected_rank = rank
                    selected_matrix = truncated.detach().clone()
            if selected_rank is None:
                selected_rank = decomposition.rank
                selected_matrix = full_matrix

        committed = (
            commit_root_residual_svd(
                base_model,
                selected_matrix,
                rank=int(selected_rank),
            )
            if selected_matrix is not None
            else None
        )
        commit_audit = (
            exact_embedding_audit(
                RootResidualSupercoreMetric.complete_basis(base_model),
                base_model,
                audit,
                chunk_size=args.eval_batch_size,
            )
            if committed is None
            else None
        )
        if committed is not None:
            complete_best = RootResidualSupercoreMetric.complete_basis(base_model)
            with torch.no_grad():
                complete_best.residual_core.copy_(selected_matrix)
            commit_audit = exact_embedding_audit(
                complete_best,
                committed,
                audit,
                chunk_size=args.eval_batch_size,
            )

        confirmation_passes = False
        confirmation: dict[str, Any] = {"opened": False, "passes": False}
        if committed is not None:
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
            confirmation = {
                "opened": True,
                "baseline": baseline_confirmation,
                "baseline_tail": baseline_confirmation_tail,
                "candidate": candidate_confirmation,
                "candidate_tail": candidate_confirmation_tail,
                "paired_improvement": paired,
                "passes": confirmation_passes,
            }

        final_model = committed if confirmation_passes else base_model
        final_configuration = copy.deepcopy(configuration)
        final_configuration.update(
            {
                "architecture": "compiled-tree",
                "bond_dimension": max(final_model.edge_dimensions),
                "edge_dimensions": tuple(final_model.edge_dimensions),
                "topology_children": tuple(final_model.topology.children),
                "root_core_direct_arm": arm["kind"],
                "root_core_direct_best_epoch": best_epoch,
                "root_core_direct_selected_rank": selected_rank,
            }
        )
        output_checkpoint = output_dir / (
            "accepted.pt" if confirmation_passes else "baseline_retained.pt"
        )
        torch.save(
            {
                "schema": "generic-quintic-direct-root-core-v1",
                "state_dict": state_to_cpu(final_model),
                "configuration": final_configuration,
                "parent_checkpoint_sha256": checkpoint_sha256,
                "teacher_runtime_dependency": False,
                "confirmation_passes": confirmation_passes,
            },
            output_checkpoint,
        )
        learned_core_path = output_dir / "learned_root_core.pt"
        torch.save(
            {
                "schema": "generic-quintic-learned-root-core-v1",
                "parent_checkpoint_sha256": checkpoint_sha256,
                "arm": arm,
                "best_epoch": best_epoch,
                "basis_left": model.left_basis.detach().cpu(),
                "basis_right": model.right_basis.detach().cpu(),
                "reduced_core": best_core,
                "complete_matrix": full_matrix.detach().cpu(),
                "singular_values": decomposition.singular_values.detach().cpu(),
                "selected_rank": selected_rank,
            },
            learned_core_path,
        )
        indices_path = output_dir / "data_indices.npz"
        np.savez_compressed(indices_path, **split_indices)
        report = {
            "schema": "generic-quintic-direct-root-core-training-v1",
            "teacher_role": "absent",
            "arm": arm,
            "configuration": {
                **vars(args),
                "initial_checkpoint": str(checkpoint_path),
                "output_dir": str(output_dir),
            },
            "parent": {
                "checkpoint_sha256": checkpoint_sha256,
                "rotation_audit": rotation_audit,
                "zero_core_audit": zero_audit,
                "base_core_norm": base_core_norm,
            },
            "training": {
                "best_epoch": best_epoch,
                "stop_reason": stop_reason,
                "baseline_selection": baseline_selection,
                "baseline_selection_tail": baseline_selection_tail,
                "best_selection": best_selection,
                "best_selection_tail": best_selection_tail,
                "history": history,
            },
            "compression": {
                "singular_values": decomposition.singular_values,
                "rank_candidates": rank_rows,
                "selected_rank": selected_rank,
                "commit_audit": commit_audit,
            },
            "confirmation": confirmation,
            "data": {
                "indices": str(indices_path),
                "indices_sha256": sha256_file(indices_path),
                "counts": {
                    name: int(len(rows))
                    for name, rows in split_indices.items()
                },
            },
            "learned_core": str(learned_core_path),
            "learned_core_sha256": sha256_file(learned_core_path),
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
                "arm": arm,
                "best_epoch": best_epoch,
                "selected_rank": selected_rank,
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
