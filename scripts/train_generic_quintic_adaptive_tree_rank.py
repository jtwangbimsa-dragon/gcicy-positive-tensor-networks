#!/usr/bin/env python3
"""Run one controlled adaptive-rank round for the generic-quintic tree."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline.positive_multiplication_tree import (  # noqa: E402
    PositiveMultiplicationTreeMetric,
    expand_multiplication_tree_bonds,
)
from scripts.evaluate_generic_quintic_h4_architecture_arms import (  # noqa: E402
    build_checkpoint_model,
    checkpoint_contract,
    sha256_file,
)
from scripts.train_generic_quintic_compiled_tree import (  # noqa: E402
    metric_row,
    whiten_dataset,
)
from scripts.train_quintic_full_h_same_points import write_json  # noqa: E402
from scripts.train_quintic_native_power_lift_tree import (  # noqa: E402
    dataset_subset,
    load_pool_arrays,
    make_dataset,
    normalized_native_e2,
    paired_improvement,
    tail_guard,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--compiler-artifact", type=Path, required=True)
    parser.add_argument("--native-points", type=Path, required=True)
    parser.add_argument("--native-pullbacks", type=Path, required=True)
    parser.add_argument("--selection-points", type=Path, required=True)
    parser.add_argument("--selection-pullbacks", type=Path, required=True)
    parser.add_argument("--confirmation-points", type=Path, required=True)
    parser.add_argument("--confirmation-pullbacks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=100_000)
    parser.add_argument("--selection-size", type=int, default=20_000)
    parser.add_argument("--confirmation-size", type=int, default=50_000)
    parser.add_argument("--rank-increment", type=int, default=1)
    parser.add_argument(
        "--activation-scale",
        type=float,
        default=1.0,
        help=(
            "Norm of the seeded nonzero factor in the exactly nested rank "
            "channel. The opposite factor remains zero, so unit scale does "
            "not change the represented metric and avoids a frozen bilinear "
            "gauge."
        ),
    )
    parser.add_argument("--proposal-seeds-per-group", type=int, default=1)
    parser.add_argument("--proposal-fit-size", type=int, default=1024)
    parser.add_argument("--proposal-validation-size", type=int, default=1024)
    parser.add_argument("--proposal-screen-only", action="store_true")
    parser.add_argument(
        "--candidate-group",
        action="append",
        default=None,
        help="Only screen/train the named adaptive group; may be repeated.",
    )
    parser.add_argument("--scan-steps", type=int, default=100)
    parser.add_argument("--scan-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--scan-eval-every", type=int, default=25)
    parser.add_argument("--refine-steps", type=int, default=300)
    parser.add_argument("--refine-learning-rate", type=float, default=3.0e-6)
    parser.add_argument("--refine-eval-every", type=int, default=25)
    parser.add_argument("--stochastic-batch-size", type=int, default=8192)
    parser.add_argument("--minimum-relative-control-gain", type=float, default=1.0e-3)
    parser.add_argument(
        "--maximum-selection-tail-relative-degradation",
        type=float,
        default=0.005,
    )
    parser.add_argument("--feature-batch-size", type=int, default=2048)
    parser.add_argument("--train-chunk-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--maximum-embedding-potential-error",
        type=float,
        default=1.0e-5,
    )
    parser.add_argument(
        "--maximum-embedding-metric-relative-error",
        type=float,
        default=2.0e-4,
    )
    parser.add_argument("--seed", type=int, default=202607423)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def adaptive_edge_groups(
    model: PositiveMultiplicationTreeMetric,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    groups: list[tuple[str, tuple[int, ...]]] = []
    if model.shared_leaf:
        groups.append(("shared_leaf_edges", tuple(range(model.leaf_count))))
    else:
        groups.extend(
            (f"leaf_edge_{leaf}", (leaf,))
            for leaf in range(model.leaf_count)
        )
    groups.extend(
        (f"internal_edge_{edge}", (edge,))
        for edge in range(model.leaf_count, model.topology.root)
    )
    return tuple(groups)


def select_named_groups(
    groups: Sequence[tuple[str, tuple[int, ...]]],
    requested: Sequence[str] | None,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    groups = tuple(groups)
    if not requested:
        return groups
    requested_names = set(requested)
    available_names = {label for label, _ in groups}
    unknown = sorted(requested_names - available_names)
    if unknown:
        raise ValueError(
            "unknown adaptive group(s): "
            + ", ".join(unknown)
            + "; available groups: "
            + ", ".join(sorted(available_names))
        )
    return tuple(
        (label, group)
        for label, group in groups
        if label in requested_names
    )


def target_dimensions(
    source: Sequence[int],
    group: Sequence[int],
    increment: int,
) -> tuple[int, ...]:
    target = [int(value) for value in source]
    for edge in group:
        target[int(edge)] += increment
    return tuple(target)


def state_to_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def zero_nonpositive(statistics: dict[str, Any]) -> bool:
    return statistics["nonpositive_min_eigenvalue"]["count"] == 0


@torch.no_grad()
def exact_embedding_audit(
    source: PositiveMultiplicationTreeMetric,
    expanded: PositiveMultiplicationTreeMetric,
    dataset: dict[str, Any],
    *,
    chunk_size: int,
) -> dict[str, float]:
    potential_error = 0.0
    metric_absolute_error = 0.0
    metric_relative_error = 0.0
    for start in range(0, dataset["count"], chunk_size):
        stop = min(start + chunk_size, dataset["count"])
        values = dataset["values"][start:stop]
        derivatives = dataset["derivatives"][start:stop]
        source_potential, source_metric = source.potential_and_metric(
            values,
            derivatives,
        )
        expanded_potential, expanded_metric = expanded.potential_and_metric(
            values,
            derivatives,
        )
        metric_delta = expanded_metric - source_metric
        source_scale = torch.linalg.matrix_norm(
            source_metric,
            ord="fro",
            dim=(-2, -1),
        )
        relative_rows = torch.linalg.matrix_norm(
            metric_delta,
            ord="fro",
            dim=(-2, -1),
        ) / torch.clamp(source_scale, min=torch.finfo(source_scale.dtype).eps)
        potential_error = max(
            potential_error,
            float(torch.max(torch.abs(expanded_potential - source_potential))),
        )
        metric_absolute_error = max(
            metric_absolute_error,
            float(torch.max(torch.abs(metric_delta))),
        )
        metric_relative_error = max(
            metric_relative_error,
            float(torch.max(relative_rows)),
        )
    return {
        "potential_max_absolute": potential_error,
        "metric_max_absolute": metric_absolute_error,
        "metric_max_relative_frobenius": metric_relative_error,
    }


def tree_new_parent_gradient_vector(
    model: PositiveMultiplicationTreeMetric,
    *,
    source_dimensions: Sequence[int],
    expanded_edges: Sequence[int],
) -> torch.Tensor:
    parents = model.topology.parent_map()
    rows = []
    for edge in expanded_edges:
        parent, side = parents[int(edge)]
        parent_tensor = model.internal_tensors[
            parent - model.leaf_count
        ]
        gradient = parent_tensor.grad
        if gradient is None:
            raise RuntimeError("native loss did not populate a parent gradient")
        start = int(source_dimensions[int(edge)])
        selected = (
            gradient[:, start:, :]
            if side == 0
            else gradient[:, :, start:]
        )
        rows.append(selected.reshape(-1))
    if not rows:
        raise ValueError("at least one expanded edge is required")
    return torch.cat(rows)


def cross_sample_gradient_statistics(
    fit_gradient: torch.Tensor,
    validation_gradient: torch.Tensor,
) -> dict[str, float]:
    fit = fit_gradient.detach().reshape(-1).to(torch.complex128)
    validation = validation_gradient.detach().reshape(-1).to(torch.complex128)
    if fit.shape != validation.shape or fit.numel() == 0:
        raise ValueError("cross-sample gradients are misaligned")
    fit_norm = float(torch.linalg.vector_norm(fit))
    validation_norm = float(torch.linalg.vector_norm(validation))
    denominator = fit_norm * validation_norm
    cosine = (
        0.0
        if denominator == 0.0
        else float(torch.real(torch.vdot(fit, validation))) / denominator
    )
    stable_score = max(cosine, 0.0) * np.sqrt(
        fit_norm * validation_norm
    )
    return {
        "fit_gradient_norm": fit_norm,
        "validation_gradient_norm": validation_norm,
        "cross_sample_cosine": cosine,
        "stable_alignment_score": float(stable_score),
    }


def native_tree_rank_gradient(
    model: PositiveMultiplicationTreeMetric,
    dataset: dict[str, Any],
    *,
    source_dimensions: Sequence[int],
    expanded_edges: Sequence[int],
    chunk_size: int,
) -> tuple[torch.Tensor, float]:
    model.zero_grad(set_to_none=True)
    loss, _ = normalized_native_e2(
        model,
        dataset,
        chunk_size=chunk_size,
    )
    loss.backward()
    gradient = tree_new_parent_gradient_vector(
        model,
        source_dimensions=source_dimensions,
        expanded_edges=expanded_edges,
    ).detach()
    model.zero_grad(set_to_none=True)
    return gradient, float(loss.detach().cpu())


def make_batch_plan(
    *,
    count: int,
    batch_size: int,
    steps: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    active_size = min(count, batch_size)
    return np.stack(
        [
            rng.choice(count, size=active_size, replace=False)
            for _ in range(steps)
        ]
    )


def train_arm(
    model: PositiveMultiplicationTreeMetric,
    *,
    training: dict[str, Any],
    selection: dict[str, Any],
    batch_plan: np.ndarray,
    learning_rate: float,
    eval_every: int,
    train_chunk_size: int,
    eval_batch_size: int,
    gradient_clip_norm: float,
    maximum_tail_degradation: float,
) -> dict[str, Any]:
    initial_statistics, _, initial_tail = metric_row(
        model,
        selection,
        chunk_size=eval_batch_size,
    )
    best_statistics = initial_statistics
    best_tail = initial_tail
    best_state = state_to_cpu(model)
    history: list[dict[str, Any]] = [
        {
            "step": 0,
            "sigma": initial_statistics["sigma_official_formula"],
            "chi": initial_statistics["weighted_rms_abs_residual"],
            **initial_tail,
        }
    ]
    parameters = list(model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len(batch_plan),
        eta_min=0.05 * learning_rate,
    )
    for step, indices in enumerate(batch_plan, start=1):
        active = dataset_subset(training, indices)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = normalized_native_e2(
            model,
            active,
            chunk_size=train_chunk_size,
        )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            gradient_clip_norm,
        )
        optimizer.step()
        scheduler.step()
        if step % eval_every != 0 and step != len(batch_plan):
            continue
        statistics, _, candidate_tail = metric_row(
            model,
            selection,
            chunk_size=eval_batch_size,
        )
        score = float(statistics["weighted_rms_abs_residual"])
        passes = bool(
            score < best_statistics["weighted_rms_abs_residual"]
            and statistics["sigma_official_formula"]
            <= initial_statistics["sigma_official_formula"]
            and zero_nonpositive(statistics)
            and tail_guard(
                candidate_tail,
                initial_tail,
                relative_degradation=maximum_tail_degradation,
            )
        )
        row = {
            "step": step,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_e2": float(loss.detach().cpu()),
            "gradient_norm": float(gradient_norm),
            "sigma": statistics["sigma_official_formula"],
            "chi": score,
            "passes_gate": passes,
            **candidate_tail,
        }
        history.append(row)
        if passes:
            best_statistics = statistics
            best_tail = candidate_tail
            best_state = state_to_cpu(model)
    model.load_state_dict(best_state)
    return {
        "model": model,
        "initial_statistics": initial_statistics,
        "initial_tail": initial_tail,
        "best_statistics": best_statistics,
        "best_tail": best_tail,
        "best_state": best_state,
        "history": history,
    }


def build_tree(
    payload: dict[str, Any],
    edge_dimensions: Sequence[int],
    *,
    device: torch.device,
) -> PositiveMultiplicationTreeMetric:
    reference_h, configuration, architecture = checkpoint_contract(payload)
    if architecture != "compiled-tree":
        raise ValueError("adaptive rank requires a compiled-tree checkpoint")
    dtype = (
        torch.complex64
        if configuration["precision"] == "complex64"
        else torch.complex128
    )
    return PositiveMultiplicationTreeMetric(
        reference_h,
        leaf_count=int(configuration["leaf_count"]),
        bond_dimension=tuple(int(value) for value in edge_dimensions),
        source_normalization=float(configuration["source_normalization"]),
        positive_floor=float(configuration["positive_floor"]),
        shared_leaf=bool(configuration["shared_leaf"]),
        dtype=dtype,
        device=device,
    )


def main() -> None:
    args = parse_args()
    if (
        min(
            args.train_size,
            args.selection_size,
            args.confirmation_size,
            args.rank_increment,
            args.proposal_seeds_per_group,
            args.scan_steps,
            args.scan_eval_every,
            args.refine_steps,
            args.refine_eval_every,
            args.stochastic_batch_size,
            args.feature_batch_size,
            args.train_chunk_size,
            args.eval_batch_size,
            args.threads,
        )
        <= 0
        or min(
            args.activation_scale,
            args.scan_learning_rate,
            args.refine_learning_rate,
            args.gradient_clip_norm,
            args.maximum_embedding_potential_error,
            args.maximum_embedding_metric_relative_error,
        )
        <= 0
        or not 0 <= args.minimum_relative_control_gain < 1
        or (
            args.proposal_screen_only
            and args.proposal_seeds_per_group <= 1
        )
        or (
            args.proposal_seeds_per_group > 1
            and (
                min(
                    args.proposal_fit_size,
                    args.proposal_validation_size,
                )
                <= 0
                or (
                    args.proposal_fit_size
                    + args.proposal_validation_size
                    > args.train_size
                )
            )
        )
    ):
        raise ValueError("invalid adaptive-rank configuration")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite an adaptive-rank run")
    output_dir.mkdir(parents=True)
    status_path = output_dir / "status.json"
    write_json(status_path, {"state": "running", "phase": "loading"})
    started = time.perf_counter()

    try:
        initial_checkpoint_path = (
            args.initial_checkpoint.expanduser().resolve()
        )
        payload = torch.load(
            initial_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        reference_h, configuration, architecture = checkpoint_contract(payload)
        if architecture != "compiled-tree":
            raise ValueError("initial checkpoint is not a compiled tree")
        precision = str(configuration["precision"])
        dtype = (
            torch.complex64 if precision == "complex64" else torch.complex128
        )
        compiler_path = args.compiler_artifact.expanduser().resolve()
        compiler = np.load(compiler_path, allow_pickle=False)
        exponents = np.asarray(
            compiler["degree4_exponents"],
            dtype=np.int64,
        )
        whitening = np.asarray(
            compiler["degree4_whitening"],
            dtype=np.complex128,
        )
        native_arrays, native_indices = load_pool_arrays(
            args.native_points,
            args.native_pullbacks,
            size=args.train_size,
            seed=args.seed + 1,
        )
        selection_arrays, selection_indices = load_pool_arrays(
            args.selection_points,
            args.selection_pullbacks,
            size=args.selection_size,
            seed=args.seed + 2,
        )
        confirmation_arrays, confirmation_indices = load_pool_arrays(
            args.confirmation_points,
            args.confirmation_pullbacks,
            size=args.confirmation_size,
            seed=args.seed + 3,
        )
        training = whiten_dataset(
            make_dataset(
                native_arrays,
                exponents,
                feature_batch_size=args.feature_batch_size,
                complex_dtype=dtype,
                device=device,
            ),
            whitening,
        )
        selection = whiten_dataset(
            make_dataset(
                selection_arrays,
                exponents,
                feature_batch_size=args.feature_batch_size,
                complex_dtype=dtype,
                device=device,
            ),
            whitening,
        )
        confirmation = whiten_dataset(
            make_dataset(
                confirmation_arrays,
                exponents,
                feature_batch_size=args.feature_batch_size,
                complex_dtype=dtype,
                device=device,
            ),
            whitening,
        )
        base_model = build_checkpoint_model(payload, device=device)
        if not isinstance(base_model, PositiveMultiplicationTreeMetric):
            raise TypeError("checkpoint did not reconstruct a multiplication tree")
        source_dimensions = tuple(base_model.edge_dimensions)
        baseline_statistics, baseline_ratio, baseline_tail = metric_row(
            base_model,
            selection,
            chunk_size=args.eval_batch_size,
        )
        audit_indices = np.arange(min(256, selection["count"]))
        audit = dataset_subset(selection, audit_indices)
        audit["weights_numpy"] = np.asarray(
            selection["weights_numpy"][audit_indices]
        )
        _, audit_ratio, _ = metric_row(
            base_model,
            audit,
            chunk_size=args.eval_batch_size,
        )
        scan_plan = make_batch_plan(
            count=training["count"],
            batch_size=args.stochastic_batch_size,
            steps=args.scan_steps,
            seed=args.seed + 10,
        )
        groups = select_named_groups(
            adaptive_edge_groups(base_model),
            args.candidate_group,
        )
        arm_specs: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = [
            ("control", (), source_dimensions)
        ]
        arm_specs.extend(
            (
                label,
                group,
                target_dimensions(
                    source_dimensions,
                    group,
                    args.rank_increment,
                ),
            )
            for label, group in groups
        )
        activation_seeds = {
            label: args.seed + 100 + arm_index
            for arm_index, (label, _, _) in enumerate(arm_specs)
            if label != "control"
        }
        proposal_screen: dict[str, Any] = {}
        if args.proposal_seeds_per_group > 1:
            proposal_rng = np.random.default_rng(args.seed + 30)
            proposal_indices = proposal_rng.choice(
                training["count"],
                size=(
                    args.proposal_fit_size
                    + args.proposal_validation_size
                ),
                replace=False,
            )
            proposal_fit = dataset_subset(
                training,
                proposal_indices[: args.proposal_fit_size],
            )
            proposal_validation = dataset_subset(
                training,
                proposal_indices[args.proposal_fit_size :],
            )
            for group_index, (label, group, dimensions) in enumerate(
                arm_specs[1:]
            ):
                write_json(
                    status_path,
                    {
                        "state": "running",
                        "phase": "proposal_screen",
                        "arm": label,
                        "arm_index": group_index,
                        "arm_count": len(arm_specs) - 1,
                    },
                )
                seed_rows = []
                for seed_offset in range(args.proposal_seeds_per_group):
                    candidate_seed = (
                        args.seed
                        + 1000
                        + group_index * args.proposal_seeds_per_group
                        + seed_offset
                    )
                    source_model = build_checkpoint_model(
                        payload,
                        device=device,
                    )
                    proposal_model = expand_multiplication_tree_bonds(
                        source_model,
                        dimensions,
                        relative_activation_scale=args.activation_scale,
                        seed=candidate_seed,
                    )
                    fit_gradient, fit_loss = native_tree_rank_gradient(
                        proposal_model,
                        proposal_fit,
                        source_dimensions=source_dimensions,
                        expanded_edges=group,
                        chunk_size=args.train_chunk_size,
                    )
                    validation_gradient, validation_loss = (
                        native_tree_rank_gradient(
                            proposal_model,
                            proposal_validation,
                            source_dimensions=source_dimensions,
                            expanded_edges=group,
                            chunk_size=args.train_chunk_size,
                        )
                    )
                    row = {
                        "seed": candidate_seed,
                        "fit_e2": fit_loss,
                        "validation_e2": validation_loss,
                        **cross_sample_gradient_statistics(
                            fit_gradient,
                            validation_gradient,
                        ),
                    }
                    seed_rows.append(row)
                    del proposal_model
                    del source_model
                    torch.cuda.empty_cache()
                chosen_seed_row = max(
                    seed_rows,
                    key=lambda row: (
                        row["stable_alignment_score"],
                        row["cross_sample_cosine"],
                    ),
                )
                activation_seeds[label] = int(chosen_seed_row["seed"])
                proposal_screen[label] = {
                    "chosen_seed": activation_seeds[label],
                    "candidates": seed_rows,
                }
                write_json(
                    output_dir / "proposal_screen.json",
                    proposal_screen,
                )
        if args.proposal_screen_only:
            report = {
                "schema": "generic-quintic-adaptive-tree-proposal-screen-v1",
                "configuration": {
                    **vars(args),
                    "initial_checkpoint": str(initial_checkpoint_path),
                    "compiler_artifact": str(compiler_path),
                    "output_dir": str(output_dir),
                },
                "source_edge_dimensions": source_dimensions,
                "baseline_selection": baseline_statistics,
                "baseline_tail": baseline_tail,
                "proposal_screen": proposal_screen,
                "wall_seconds": time.perf_counter() - started,
            }
            write_json(output_dir / "report.json", report)
            write_json(
                status_path,
                {
                    "state": "complete",
                    "phase": "proposal_screen_complete",
                    "wall_seconds": time.perf_counter() - started,
                },
            )
            print(json.dumps(report, indent=2, sort_keys=True, default=str))
            return
        arms: dict[str, Any] = {}
        embedding_audits: dict[str, dict[str, float]] = {}
        for arm_index, (label, group, dimensions) in enumerate(arm_specs):
            write_json(
                status_path,
                {
                    "state": "running",
                    "phase": "scan",
                    "arm": label,
                    "arm_index": arm_index,
                    "arm_count": len(arm_specs),
                },
            )
            source_model = build_checkpoint_model(payload, device=device)
            if label == "control":
                model = source_model
                exact_metric_error = 0.0
                embedding_audit = {
                    "potential_max_absolute": 0.0,
                    "metric_max_absolute": 0.0,
                    "metric_max_relative_frobenius": 0.0,
                    "normalized_ratio_max_absolute": 0.0,
                }
            else:
                model = expand_multiplication_tree_bonds(
                    source_model,
                    dimensions,
                    relative_activation_scale=args.activation_scale,
                    seed=activation_seeds[label],
                )
                _, candidate_audit_ratio, _ = metric_row(
                    model,
                    audit,
                    chunk_size=args.eval_batch_size,
                )
                exact_metric_error = float(
                    np.max(np.abs(candidate_audit_ratio - audit_ratio))
                )
                embedding_audit = exact_embedding_audit(
                    source_model,
                    model,
                    audit,
                    chunk_size=args.eval_batch_size,
                )
                embedding_audit["normalized_ratio_max_absolute"] = (
                    exact_metric_error
                )
                embedding_audits[label] = embedding_audit
                write_json(
                    output_dir / "embedding_audit.json",
                    embedding_audits,
                )
                if (
                    embedding_audit["potential_max_absolute"]
                    > args.maximum_embedding_potential_error
                    or embedding_audit["metric_max_relative_frobenius"]
                    > args.maximum_embedding_metric_relative_error
                ):
                    raise FloatingPointError(
                        f"rank expansion changed the initial metric for {label}"
                    )
            embedding_audits[label] = embedding_audit
            result = train_arm(
                model,
                training=training,
                selection=selection,
                batch_plan=scan_plan,
                learning_rate=args.scan_learning_rate,
                eval_every=args.scan_eval_every,
                train_chunk_size=args.train_chunk_size,
                eval_batch_size=args.eval_batch_size,
                gradient_clip_norm=args.gradient_clip_norm,
                maximum_tail_degradation=(
                    args.maximum_selection_tail_relative_degradation
                ),
            )
            arms[label] = {
                "group": group,
                "edge_dimensions": dimensions,
                "exact_metric_error": exact_metric_error,
                "embedding_audit": embedding_audit,
                "activation_seed": (
                    None
                    if label == "control"
                    else activation_seeds[label]
                ),
                "trainable_real_parameter_count": (
                    result["model"].trainable_real_parameter_count
                ),
                "initial_statistics": result["initial_statistics"],
                "best_statistics": result["best_statistics"],
                "best_tail": result["best_tail"],
                "history": result["history"],
                "best_state": result["best_state"],
            }
            write_json(
                output_dir / "scan_progress.json",
                {
                    completed_label: {
                        key: value
                        for key, value in completed.items()
                        if key not in {"best_state", "history"}
                    }
                    for completed_label, completed in arms.items()
                },
            )
            del result["model"]
            del source_model
            torch.cuda.empty_cache()

        control = arms["control"]
        eligible: list[tuple[float, str]] = []
        for label, candidate in arms.items():
            if label == "control":
                continue
            control_chi = float(
                control["best_statistics"]["weighted_rms_abs_residual"]
            )
            candidate_chi = float(
                candidate["best_statistics"]["weighted_rms_abs_residual"]
            )
            relative_gain = (control_chi - candidate_chi) / control_chi
            candidate["relative_chi_gain_over_control"] = relative_gain
            passes = bool(
                relative_gain >= args.minimum_relative_control_gain
                and candidate["best_statistics"]["sigma_official_formula"]
                <= control["best_statistics"]["sigma_official_formula"]
                and zero_nonpositive(candidate["best_statistics"])
                and tail_guard(
                    candidate["best_tail"],
                    control["best_tail"],
                    relative_degradation=(
                        args.maximum_selection_tail_relative_degradation
                    ),
                )
            )
            candidate["passes_control_gate"] = passes
            if passes:
                eligible.append((relative_gain, label))
        chosen_label = max(eligible)[1] if eligible else None

        final_model = build_checkpoint_model(payload, device=device)
        matched_control_model = build_checkpoint_model(payload, device=device)
        final_dimensions = source_dimensions
        refinement: dict[str, Any] | None = None
        control_refinement: dict[str, Any] | None = None
        if chosen_label is not None:
            write_json(
                status_path,
                {
                    "state": "running",
                    "phase": "matched_refinement",
                    "chosen_arm": chosen_label,
                    "refine_steps_per_arm": args.refine_steps,
                },
            )
            chosen = arms[chosen_label]
            final_dimensions = tuple(chosen["edge_dimensions"])
            final_model = build_tree(
                payload,
                final_dimensions,
                device=device,
            )
            final_model.load_state_dict(chosen["best_state"])
            matched_control_model.load_state_dict(control["best_state"])
            refine_plan = make_batch_plan(
                count=training["count"],
                batch_size=args.stochastic_batch_size,
                steps=args.refine_steps,
                seed=args.seed + 20,
            )
            control_refinement_result = train_arm(
                matched_control_model,
                training=training,
                selection=selection,
                batch_plan=refine_plan,
                learning_rate=args.refine_learning_rate,
                eval_every=args.refine_eval_every,
                train_chunk_size=args.train_chunk_size,
                eval_batch_size=args.eval_batch_size,
                gradient_clip_norm=args.gradient_clip_norm,
                maximum_tail_degradation=(
                    args.maximum_selection_tail_relative_degradation
                ),
            )
            matched_control_model = control_refinement_result["model"]
            control_refinement = {
                key: value
                for key, value in control_refinement_result.items()
                if key not in {"model", "best_state"}
            }
            refinement_result = train_arm(
                final_model,
                training=training,
                selection=selection,
                batch_plan=refine_plan,
                learning_rate=args.refine_learning_rate,
                eval_every=args.refine_eval_every,
                train_chunk_size=args.train_chunk_size,
                eval_batch_size=args.eval_batch_size,
                gradient_clip_norm=args.gradient_clip_norm,
                maximum_tail_degradation=(
                    args.maximum_selection_tail_relative_degradation
                ),
            )
            final_model = refinement_result["model"]
            refinement = {
                key: value
                for key, value in refinement_result.items()
                if key not in {"model", "best_state"}
            }

        write_json(
            status_path,
            {
                "state": "running",
                "phase": "confirmation",
                "chosen_arm": chosen_label,
                "confirmation_size": args.confirmation_size,
            },
        )
        parent_confirmation_model = build_checkpoint_model(payload, device=device)
        parent_confirmation, confirmation_parent_ratio, confirmation_parent_tail = (
            metric_row(
                parent_confirmation_model,
                confirmation,
                chunk_size=args.eval_batch_size,
            )
        )
        control_confirmation, confirmation_control_ratio, confirmation_control_tail = (
            metric_row(
                matched_control_model,
                confirmation,
                chunk_size=args.eval_batch_size,
            )
        )
        candidate_confirmation, candidate_ratio, candidate_tail = metric_row(
            final_model,
            confirmation,
            chunk_size=args.eval_batch_size,
        )
        paired_vs_control = paired_improvement(
            confirmation_control_ratio,
            candidate_ratio,
            confirmation["weights_numpy"],
        )
        paired_vs_parent = paired_improvement(
            confirmation_parent_ratio,
            candidate_ratio,
            confirmation["weights_numpy"],
        )
        confirmation_passes = bool(
            chosen_label is not None
            and candidate_confirmation["sigma_official_formula"]
            < control_confirmation["sigma_official_formula"]
            and candidate_confirmation["weighted_rms_abs_residual"]
            < control_confirmation["weighted_rms_abs_residual"]
            and candidate_confirmation["sigma_official_formula"]
            < parent_confirmation["sigma_official_formula"]
            and candidate_confirmation["weighted_rms_abs_residual"]
            < parent_confirmation["weighted_rms_abs_residual"]
            and zero_nonpositive(candidate_confirmation)
            and tail_guard(
                candidate_tail,
                confirmation_control_tail,
                relative_degradation=(
                    args.maximum_selection_tail_relative_degradation
                ),
            )
            and paired_vs_control["e2"]["ci95_low"] > 0
            and paired_vs_control["sigma"]["ci95_low"] > 0
        )

        output_checkpoint_path = output_dir / (
            "accepted.pt"
            if confirmation_passes
            else "diagnostic_rejected.pt"
        )
        final_configuration = dict(configuration)
        final_configuration.update(
            {
                "architecture": "compiled-tree",
                "bond_dimension": max(final_dimensions),
                "edge_dimensions": final_dimensions,
                "adaptive_rank_round": int(
                    configuration.get("adaptive_rank_round", 0)
                )
                + (1 if chosen_label is not None else 0),
            }
        )
        torch.save(
            {
                "schema": "generic-quintic-adaptive-compiled-positive-tree-v1",
                "state_dict": state_to_cpu(final_model),
                "configuration": final_configuration,
                "parent_checkpoint_sha256": sha256_file(
                    initial_checkpoint_path
                ),
                "teacher_runtime_dependency": False,
                "confirmation_passes": confirmation_passes,
            },
            output_checkpoint_path,
        )
        serializable_arms = {
            label: {
                key: value
                for key, value in result.items()
                if key != "best_state"
            }
            for label, result in arms.items()
        }
        report = {
            "schema": "generic-quintic-adaptive-tree-rank-v1",
            "configuration": {
                **vars(args),
                "initial_checkpoint": str(initial_checkpoint_path),
                "compiler_artifact": str(compiler_path),
                "output_dir": str(output_dir),
            },
            "source_edge_dimensions": source_dimensions,
            "chosen_arm": chosen_label,
            "final_edge_dimensions": final_dimensions,
            "baseline_selection": baseline_statistics,
            "baseline_tail": baseline_tail,
            "arms": serializable_arms,
            "proposal_screen": proposal_screen,
            "control_refinement": control_refinement,
            "refinement": refinement,
            "confirmation": {
                "parent": parent_confirmation,
                "parent_tail": confirmation_parent_tail,
                "baseline": control_confirmation,
                "matched_control": control_confirmation,
                "candidate": candidate_confirmation,
                "baseline_tail": confirmation_control_tail,
                "matched_control_tail": confirmation_control_tail,
                "candidate_tail": candidate_tail,
                "paired_improvement": paired_vs_control,
                "paired_improvement_vs_control": paired_vs_control,
                "paired_improvement_vs_parent": paired_vs_parent,
                "passes": confirmation_passes,
            },
            "checkpoint": str(output_checkpoint_path),
            "checkpoint_sha256": sha256_file(output_checkpoint_path),
            "trainable_real_parameter_count": (
                final_model.trainable_real_parameter_count
            ),
            "data_indices": {
                "native": native_indices,
                "selection": selection_indices,
                "confirmation": confirmation_indices,
            },
            "wall_seconds": time.perf_counter() - started,
        }
        write_json(output_dir / "report.json", report)
        write_json(
            status_path,
            {
                "state": "complete",
                "phase": "complete",
                "chosen_arm": chosen_label,
                "confirmation_passes": confirmation_passes,
                "wall_seconds": time.perf_counter() - started,
            },
        )
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    except Exception as error:
        write_json(
            status_path,
            {
                "state": "failed",
                "phase": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "wall_seconds": time.perf_counter() - started,
            },
        )
        raise


if __name__ == "__main__":
    main()
