#!/usr/bin/env python3
"""Adaptive direct block-coordinate training for a compiled quintic tree."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
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
    expand_multiplication_tree_bonds,
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
    normalized_native_e2,
    paired_improvement,
    tail_guard,
)


DEFAULT_DIRECT_BLOCK_REAL_PARAMETER_LIMIT = 10_000


@dataclass(frozen=True)
class DirectBlock:
    name: str
    parameter_names: tuple[str, ...]
    real_parameter_count: int
    learning_rate: float
    mask_name: str | None = None
    mask_selection: tuple[int, int] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
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
    parser.add_argument(
        "--real-parameter-limit",
        type=int,
        default=DEFAULT_DIRECT_BLOCK_REAL_PARAMETER_LIMIT,
    )
    parser.add_argument(
        "--target-internal-edge-dimension",
        type=int,
        default=None,
        help=(
            "Exactly embed every nonroot internal edge in this dimension "
            "before direct block training."
        ),
    )
    parser.add_argument(
        "--target-edge",
        action="append",
        default=[],
        metavar="EDGE:DIMENSION",
        help="Exactly expand one nonroot internal edge; may be repeated.",
    )
    parser.add_argument("--rank-activation-scale", type=float, default=1.0)
    parser.add_argument(
        "--orthogonalize-new-outputs",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--only-internal",
        action="store_true",
        help="Train only the complete internal block after optional expansion.",
    )
    parser.add_argument(
        "--expansion-only",
        action="store_true",
        help=(
            "Train only parent couplings and newly added child outputs for "
            "the requested rank expansion."
        ),
    )
    parser.add_argument("--sweeps", type=int, default=1)
    parser.add_argument("--epochs-per-block", type=int, default=120)
    parser.add_argument("--selection-eval-every", type=int, default=10)
    parser.add_argument("--early-stopping-evaluations", type=int, default=4)
    parser.add_argument("--internal-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--leaf-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--train-size", type=int, default=30_000)
    parser.add_argument("--selection-size", type=int, default=10_000)
    parser.add_argument("--confirmation-size", type=int, default=20_000)
    parser.add_argument("--stochastic-batch-size", type=int, default=16_384)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--backward-mode",
        choices=("graph", "streaming"),
        default="graph",
    )
    parser.add_argument(
        "--maximum-selection-tail-relative-degradation",
        type=float,
        default=0.02,
    )
    parser.add_argument("--train-chunk-size", type=int, default=2048)
    parser.add_argument("--feature-batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=202607471)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--development-only",
        action="store_true",
        help="Skip confirmation so repeated adaptive rounds use development data only.",
    )
    parser.add_argument(
        "--defer-block-acceptance",
        action="store_true",
        help=(
            "Keep provisional parent/child states through a complete expansion "
            "cycle and apply one selection gate to the combined update."
        ),
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.real_parameter_limit,
        args.rank_activation_scale,
        args.sweeps,
        args.epochs_per_block,
        args.selection_eval_every,
        args.early_stopping_evaluations,
        args.internal_learning_rate,
        args.leaf_learning_rate,
        args.train_size,
        args.selection_size,
        args.confirmation_size,
        args.stochastic_batch_size,
        args.gradient_clip_norm,
        args.train_chunk_size,
        args.feature_batch_size,
        args.eval_batch_size,
        args.threads,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("all block, optimization, and data values must be positive")
    if args.maximum_selection_tail_relative_degradation < 0:
        raise ValueError("selection tail allowance must be nonnegative")
    if (
        args.target_internal_edge_dimension is not None
        and args.target_internal_edge_dimension <= 0
    ):
        raise ValueError("target internal edge dimension must be positive")
    if args.target_internal_edge_dimension is not None and args.target_edge:
        raise ValueError(
            "use either a common internal target or explicit edge targets"
        )
    if args.expansion_only and not (
        args.target_internal_edge_dimension is not None or args.target_edge
    ):
        raise ValueError("expansion-only training requires a rank expansion")


def parse_target_edges(specifications: list[str]) -> dict[int, int]:
    targets: dict[int, int] = {}
    for specification in specifications:
        try:
            edge_text, dimension_text = specification.split(":", maxsplit=1)
            edge = int(edge_text)
            dimension = int(dimension_text)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "target edges must use EDGE:DIMENSION syntax"
            ) from error
        if edge < 0 or dimension <= 0:
            raise ValueError("target edge and dimension must be nonnegative")
        if edge in targets:
            raise ValueError(f"edge {edge} has more than one target")
        targets[edge] = dimension
    return targets


def split_parameter_blocks(
    name: str,
    parameter: torch.Tensor,
    *,
    real_parameter_limit: int,
    learning_rate: float,
) -> list[DirectBlock]:
    max_complex = real_parameter_limit // 2
    if max_complex <= 0:
        raise ValueError("real parameter limit cannot hold one complex parameter")
    blocks = []
    for start in range(0, parameter.numel(), max_complex):
        stop = min(start + max_complex, parameter.numel())
        blocks.append(
            DirectBlock(
                name=f"{name}[{start}:{stop}]",
                parameter_names=(name,),
                real_parameter_count=2 * (stop - start),
                learning_rate=learning_rate,
                mask_name=name,
                mask_selection=(start, stop),
            )
        )
    return blocks


def enumerate_direct_blocks(
    model: PositiveMultiplicationTreeMetric,
    *,
    real_parameter_limit: int,
    internal_learning_rate: float,
    leaf_learning_rate: float,
) -> list[DirectBlock]:
    named = dict(model.named_parameters())
    internal_names = tuple(
        name for name in named if name.startswith("internal_tensors.")
    )
    internal_real = 2 * sum(named[name].numel() for name in internal_names)
    blocks: list[DirectBlock] = []
    if internal_real <= real_parameter_limit:
        blocks.append(
            DirectBlock(
                name="internal_all",
                parameter_names=internal_names,
                real_parameter_count=internal_real,
                learning_rate=internal_learning_rate,
            )
        )
    else:
        for name in internal_names:
            blocks.extend(
                split_parameter_blocks(
                    name,
                    named[name],
                    real_parameter_limit=real_parameter_limit,
                    learning_rate=internal_learning_rate,
                )
            )

    if model.shared_leaf:
        name = "shared_leaf_tensor"
        parameter = named[name]
        row_size = parameter[0].numel()
        row_real = 2 * row_size
        if row_real <= real_parameter_limit:
            for row in range(parameter.shape[0]):
                start = row * row_size
                stop = start + row_size
                blocks.append(
                    DirectBlock(
                        name=f"shared_leaf_output_{row}",
                        parameter_names=(name,),
                        real_parameter_count=row_real,
                        learning_rate=leaf_learning_rate,
                        mask_name=name,
                        mask_selection=(start, stop),
                    )
                )
        else:
            blocks.extend(
                split_parameter_blocks(
                    name,
                    parameter,
                    real_parameter_limit=real_parameter_limit,
                    learning_rate=leaf_learning_rate,
                )
            )
    else:
        for name in named:
            if name.startswith("leaf_tensors."):
                blocks.extend(
                    split_parameter_blocks(
                        name,
                        named[name],
                        real_parameter_limit=real_parameter_limit,
                        learning_rate=leaf_learning_rate,
                    )
                )
    if not blocks:
        raise ValueError("tree exposes no direct training blocks")
    if any(
        block.real_parameter_count > real_parameter_limit
        for block in blocks
    ):
        raise AssertionError("direct block exceeds the registered limit")
    return blocks


def enumerate_expansion_blocks(
    model: PositiveMultiplicationTreeMetric,
    *,
    source_edge_dimensions: tuple[int, ...],
    real_parameter_limit: int,
    learning_rate: float,
    leaf_learning_rate: float | None = None,
) -> list[DirectBlock]:
    named = dict(model.named_parameters())
    parents = model.topology.parent_map()
    expanded_edges = [
        edge
        for edge, (source, target) in enumerate(
            zip(
                source_edge_dimensions,
                model.edge_dimensions,
                strict=True,
            )
        )
        if target > source
    ]
    if not expanded_edges:
        raise ValueError("expansion-only training found no enlarged edge")
    blocks: list[DirectBlock] = []
    parent_names: list[str] = []
    for edge in expanded_edges:
        parent, _ = parents[edge]
        name = f"internal_tensors.{parent - model.leaf_count}"
        if name not in parent_names:
            parent_names.append(name)
    for name in parent_names:
        parameter = named[name]
        if 2 * parameter.numel() <= real_parameter_limit:
            blocks.append(
                DirectBlock(
                    name=f"parent_coupling_{name}",
                    parameter_names=(name,),
                    real_parameter_count=2 * parameter.numel(),
                    learning_rate=learning_rate,
                )
            )
        else:
            blocks.extend(
                split_parameter_blocks(
                    name,
                    parameter,
                    real_parameter_limit=real_parameter_limit,
                    learning_rate=learning_rate,
                )
            )

    maximum_complex = real_parameter_limit // 2
    internal_edges = [
        edge for edge in expanded_edges if edge >= model.leaf_count
    ]
    for edge in internal_edges:
        name = f"internal_tensors.{edge - model.leaf_count}"
        parameter = named[name]
        row_size = parameter[0].numel()
        start = source_edge_dimensions[edge] * row_size
        stop = model.edge_dimensions[edge] * row_size
        for chunk_start in range(start, stop, maximum_complex):
            chunk_stop = min(chunk_start + maximum_complex, stop)
            blocks.append(
                DirectBlock(
                    name=(
                        f"new_output_edge_{edge}"
                        f"[{chunk_start}:{chunk_stop}]"
                    ),
                    parameter_names=(name,),
                    real_parameter_count=2 * (chunk_stop - chunk_start),
                    learning_rate=learning_rate,
                    mask_name=name,
                    mask_selection=(chunk_start, chunk_stop),
                )
            )
    leaf_edges = [
        edge for edge in expanded_edges if edge < model.leaf_count
    ]
    if leaf_edges:
        if not model.shared_leaf:
            raise ValueError(
                "shared-leaf expansion blocks require shared leaf tensors"
            )
        if leaf_edges != list(range(model.leaf_count)):
            raise ValueError("every shared leaf edge must expand together")
        source_leaf_dimensions = {
            source_edge_dimensions[edge] for edge in leaf_edges
        }
        target_leaf_dimensions = {
            model.edge_dimensions[edge] for edge in leaf_edges
        }
        if (
            len(source_leaf_dimensions) != 1
            or len(target_leaf_dimensions) != 1
        ):
            raise ValueError("shared leaf edge dimensions are inconsistent")
        name = "shared_leaf_tensor"
        parameter = named[name]
        row_size = parameter[0].numel()
        start = next(iter(source_leaf_dimensions)) * row_size
        stop = next(iter(target_leaf_dimensions)) * row_size
        for chunk_start in range(start, stop, maximum_complex):
            chunk_stop = min(chunk_start + maximum_complex, stop)
            blocks.append(
                DirectBlock(
                    name=(
                        "new_shared_leaf_output"
                        f"[{chunk_start}:{chunk_stop}]"
                    ),
                    parameter_names=(name,),
                    real_parameter_count=2 * (chunk_stop - chunk_start),
                    learning_rate=(
                        learning_rate
                        if leaf_learning_rate is None
                        else leaf_learning_rate
                    ),
                    mask_name=name,
                    mask_selection=(chunk_start, chunk_stop),
                )
            )
    if any(
        block.real_parameter_count > real_parameter_limit
        for block in blocks
    ):
        raise AssertionError("rank-growth block exceeds the registered limit")
    return blocks


def configure_block(
    model: PositiveMultiplicationTreeMetric,
    block: DirectBlock,
) -> tuple[list[torch.nn.Parameter], Any | None]:
    model.freeze_all_()
    named = dict(model.named_parameters())
    parameters = []
    for name in block.parameter_names:
        parameter = named[name]
        parameter.requires_grad_(True)
        parameters.append(parameter)
    hook = None
    if block.mask_name is not None:
        parameter = named[block.mask_name]
        start, stop = block.mask_selection
        mask = torch.zeros_like(parameter).reshape(-1)
        mask[start:stop] = 1
        mask = mask.reshape(parameter.shape)
        hook = parameter.register_hook(lambda gradient: gradient * mask)
    return parameters, hook


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
        raise FileExistsError("refusing to overwrite an adaptive block run")
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
            raise ValueError("adaptive direct training requires a tree checkpoint")
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
        expansion: dict[str, Any] | None = None
        model = source_model
        explicit_targets = parse_target_edges(args.target_edge)
        if (
            args.target_internal_edge_dimension is not None
            or explicit_targets
        ):
            source_dimensions = tuple(source_model.edge_dimensions)
            target_dimensions = list(source_dimensions)
            if args.target_internal_edge_dimension is not None:
                for edge in range(
                    source_model.leaf_count,
                    source_model.topology.root,
                ):
                    target_dimensions[edge] = (
                        args.target_internal_edge_dimension
                    )
            else:
                leaf_targets = {
                    edge: dimension
                    for edge, dimension in explicit_targets.items()
                    if edge < source_model.leaf_count
                }
                if leaf_targets:
                    if not source_model.shared_leaf:
                        raise ValueError(
                            "explicit leaf growth requires shared leaves"
                        )
                    if set(leaf_targets) != set(
                        range(source_model.leaf_count)
                    ):
                        raise ValueError(
                            "every shared leaf edge must receive a target"
                        )
                    if len(set(leaf_targets.values())) != 1:
                        raise ValueError(
                            "shared leaf targets must have one dimension"
                        )
                for edge, dimension in explicit_targets.items():
                    if not 0 <= edge < source_model.topology.root:
                        raise ValueError(
                            f"edge {edge} is not a nonroot tree edge"
                        )
                    target_dimensions[edge] = dimension
            target_dimensions = tuple(target_dimensions)
            if any(
                target < source
                for source, target in zip(
                    source_dimensions,
                    target_dimensions,
                    strict=True,
                )
            ):
                raise ValueError(
                    "target internal dimension cannot shrink an existing edge"
                )
            if target_dimensions == source_dimensions:
                raise ValueError("requested internal expansion changes no edge")
            model = expand_multiplication_tree_bonds(
                source_model,
                target_dimensions,
                relative_activation_scale=args.rank_activation_scale,
                seed=args.seed + 7,
                orthogonalize_new_outputs=args.orthogonalize_new_outputs,
            )
            expansion = {
                "source_edge_dimensions": source_dimensions,
                "target_edge_dimensions": target_dimensions,
                "activation_scale": args.rank_activation_scale,
                "orthogonalized": args.orthogonalize_new_outputs,
            }
        model.eval()
        initial_state = copy.deepcopy(model.state_dict())
        blocks = enumerate_direct_blocks(
            model,
            real_parameter_limit=args.real_parameter_limit,
            internal_learning_rate=args.internal_learning_rate,
            leaf_learning_rate=args.leaf_learning_rate,
        )
        if args.expansion_only:
            if expansion is None:
                raise AssertionError("rank expansion metadata is unavailable")
            blocks = enumerate_expansion_blocks(
                model,
                source_edge_dimensions=tuple(
                    expansion["source_edge_dimensions"]
                ),
                real_parameter_limit=args.real_parameter_limit,
                learning_rate=args.internal_learning_rate,
                leaf_learning_rate=args.leaf_learning_rate,
            )
        elif args.only_internal:
            blocks = [
                block
                for block in blocks
                if all(
                    name.startswith("internal_tensors.")
                    for name in block.parameter_names
                )
            ]
            if not blocks:
                raise ValueError("tree exposes no admissible internal block")

        excluded = load_excluded_indices(args.exclude_indices_file)
        split_specifications = {
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
        }
        if not args.development_only:
            split_specifications["confirmation"] = (
                args.confirmation_points,
                args.confirmation_pullbacks,
                args.confirmation_size,
            )
        split_arrays, split_indices = load_disjoint_splits(
            split_specifications,
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
        embedding_audit = None
        if expansion is not None:
            audit_count = min(256, datasets["selection"]["count"])
            audit = dataset_subset(
                datasets["selection"],
                np.arange(audit_count),
            )
            embedding_audit = exact_embedding_audit(
                source_model,
                model,
                audit,
                chunk_size=args.eval_batch_size,
            )
            if (
                embedding_audit["potential_max_absolute"] > 1.0e-5
                or embedding_audit["metric_max_relative_frobenius"] > 2.0e-4
            ):
                raise RuntimeError(
                    "nested rank expansion did not preserve the parent metric"
                )
            del source_model
        initial_selection, _, initial_selection_tail = metric_row(
            model,
            datasets["selection"],
            chunk_size=args.eval_batch_size,
        )
        current_selection = initial_selection
        current_selection_tail = initial_selection_tail
        accepted_blocks = 0
        history: list[dict[str, Any]] = []
        rng = np.random.default_rng(args.seed + 31)
        deferred_best_state = initial_state
        deferred_best_selection = initial_selection
        deferred_best_tail = initial_selection_tail
        deferred_updates = 0

        for sweep in range(1, args.sweeps + 1):
            sweep_accepts = 0
            for block_index, block in enumerate(blocks):
                block_start_state = copy.deepcopy(model.state_dict())
                block_start_selection = current_selection
                block_start_tail = current_selection_tail
                best_state = block_start_state
                best_selection = block_start_selection
                best_tail = block_start_tail
                best_epoch = 0
                latest_selection = block_start_selection
                latest_tail = block_start_tail
                stale_evaluations = 0
                parameters, hook = configure_block(model, block)
                optimizer = torch.optim.Adam(
                    parameters,
                    lr=block.learning_rate,
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=args.epochs_per_block,
                    eta_min=0.05 * block.learning_rate,
                )
                rows = []
                try:
                    for epoch in range(1, args.epochs_per_block + 1):
                        model.train()
                        optimizer.zero_grad(set_to_none=True)
                        count = min(
                            args.stochastic_batch_size,
                            datasets["fit"]["count"],
                        )
                        indices = rng.choice(
                            datasets["fit"]["count"],
                            size=count,
                            replace=False,
                        )
                        batch = dataset_subset(datasets["fit"], indices)
                        if args.backward_mode == "graph":
                            train_e2, _ = normalized_native_e2(
                                model,
                                batch,
                                chunk_size=args.train_chunk_size,
                            )
                            train_e2.backward()
                        else:
                            train_e2, _ = native_e2_streaming_backward(
                                model,
                                batch,
                                chunk_size=args.train_chunk_size,
                            )
                        gradient_norm = float(
                            torch.nn.utils.clip_grad_norm_(
                                parameters,
                                args.gradient_clip_norm,
                            )
                        )
                        optimizer.step()
                        scheduler.step()
                        evaluate = bool(
                            epoch % args.selection_eval_every == 0
                            or epoch == args.epochs_per_block
                        )
                        row = {
                            "epoch": epoch,
                            "train_e2": float(train_e2),
                            "gradient_norm": gradient_norm,
                            "learning_rate": optimizer.param_groups[0]["lr"],
                        }
                        if evaluate:
                            selection, _, tail = metric_row(
                                model,
                                datasets["selection"],
                                chunk_size=args.eval_batch_size,
                            )
                            improves = bool(
                                selection["sigma_official_formula"]
                                < best_selection["sigma_official_formula"]
                                and selection["weighted_rms_abs_residual"]
                                < best_selection["weighted_rms_abs_residual"]
                                and zero_nonpositive(selection)
                                and tail_guard(
                                    tail,
                                    block_start_tail,
                                    relative_degradation=(
                                        args.maximum_selection_tail_relative_degradation
                                    ),
                                )
                            )
                            row.update(
                                {
                                    "selection": selection,
                                    "selection_tail": tail,
                                    "accepted": improves,
                                }
                            )
                            latest_selection = selection
                            latest_tail = tail
                            if improves:
                                best_state = copy.deepcopy(model.state_dict())
                                best_selection = selection
                                best_tail = tail
                                best_epoch = epoch
                                stale_evaluations = 0
                            else:
                                stale_evaluations += 1
                            deferred_improves = bool(
                                args.defer_block_acceptance
                                and selection["sigma_official_formula"]
                                < deferred_best_selection[
                                    "sigma_official_formula"
                                ]
                                and selection[
                                    "weighted_rms_abs_residual"
                                ]
                                < deferred_best_selection[
                                    "weighted_rms_abs_residual"
                                ]
                                and zero_nonpositive(selection)
                                and tail_guard(
                                    tail,
                                    initial_selection_tail,
                                    relative_degradation=(
                                        args.maximum_selection_tail_relative_degradation
                                    ),
                                )
                            )
                            if deferred_improves:
                                deferred_best_state = copy.deepcopy(
                                    model.state_dict()
                                )
                                deferred_best_selection = selection
                                deferred_best_tail = tail
                                deferred_updates += 1
                            if not args.quiet:
                                print(
                                    f"sweep={sweep} block={block.name} "
                                    f"epoch={epoch} "
                                    f"sigma={selection['sigma_official_formula']:.6e} "
                                    f"chi={selection['weighted_rms_abs_residual']:.6e} "
                                    f"best={best_epoch}",
                                    flush=True,
                                )
                        rows.append(row)
                        write_json(
                            status_path,
                            {
                                "state": "running",
                                "phase": "block_training",
                                "sweep": sweep,
                                "block_index": block_index,
                                "block": asdict(block),
                                "epoch": epoch,
                                "best_epoch": best_epoch,
                                "accepted_blocks": accepted_blocks,
                            },
                        )
                        if (
                            evaluate
                            and stale_evaluations
                            >= args.early_stopping_evaluations
                        ):
                            break
                finally:
                    if hook is not None:
                        hook.remove()

                accepted = best_epoch > 0
                if args.defer_block_acceptance:
                    current_selection = latest_selection
                    current_selection_tail = latest_tail
                    if accepted:
                        sweep_accepts += 1
                else:
                    model.load_state_dict(best_state)
                    if accepted:
                        current_selection = best_selection
                        current_selection_tail = best_tail
                        accepted_blocks += 1
                        sweep_accepts += 1
                    else:
                        current_selection = block_start_selection
                        current_selection_tail = block_start_tail
                history.append(
                    {
                        "sweep": sweep,
                        "block": asdict(block),
                        "accepted": accepted,
                        "best_epoch": best_epoch,
                        "start_selection": block_start_selection,
                        "start_selection_tail": block_start_tail,
                        "best_selection": best_selection,
                        "best_selection_tail": best_tail,
                        "rows": rows,
                    }
                )
            if sweep_accepts == 0:
                break

        if args.defer_block_acceptance:
            model.load_state_dict(deferred_best_state)
            current_selection = deferred_best_selection
            current_selection_tail = deferred_best_tail
            accepted_blocks = int(deferred_updates > 0)
        candidate_state = copy.deepcopy(model.state_dict())
        confirmation = None
        if args.development_only:
            confirmation_passes = None
            final_state = (
                candidate_state if accepted_blocks > 0 else initial_state
            )
        else:
            model.load_state_dict(initial_state)
            (
                baseline_confirmation,
                baseline_ratio,
                baseline_confirmation_tail,
            ) = metric_row(
                model,
                datasets["confirmation"],
                chunk_size=args.eval_batch_size,
            )
            model.load_state_dict(candidate_state)
            (
                candidate_confirmation,
                candidate_ratio,
                candidate_confirmation_tail,
            ) = metric_row(
                model,
                datasets["confirmation"],
                chunk_size=args.eval_batch_size,
            )
            paired = paired_improvement(
                baseline_ratio,
                candidate_ratio,
                datasets["confirmation"]["weights_numpy"],
            )
            confirmation_passes = bool(
                accepted_blocks > 0
                and candidate_confirmation["sigma_official_formula"]
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
            final_state = (
                candidate_state if confirmation_passes else initial_state
            )
            confirmation = {
                "baseline": baseline_confirmation,
                "baseline_tail": baseline_confirmation_tail,
                "candidate": candidate_confirmation,
                "candidate_tail": candidate_confirmation_tail,
                "paired_improvement": paired,
                "passes": confirmation_passes,
            }
        model.load_state_dict(final_state)
        final_configuration = copy.deepcopy(configuration)
        final_configuration.update(
            {
                "adaptive_direct_block_limit_real": args.real_parameter_limit,
                "adaptive_direct_accepted_blocks": accepted_blocks,
                "adaptive_direct_sweeps": args.sweeps,
                "adaptive_direct_development_only": args.development_only,
                "edge_dimensions": tuple(model.edge_dimensions),
            }
        )
        if expansion is not None:
            final_configuration["adaptive_direct_expansion"] = expansion
        if args.development_only:
            output_checkpoint = output_dir / (
                "development_accepted.pt"
                if accepted_blocks > 0
                else "baseline_retained.pt"
            )
        else:
            output_checkpoint = output_dir / (
                "accepted.pt"
                if confirmation_passes
                else "baseline_retained.pt"
            )
        torch.save(
            {
                "schema": "generic-quintic-adaptive-direct-blocks-v1",
                "state_dict": state_to_cpu(model),
                "configuration": final_configuration,
                "parent_checkpoint_sha256": checkpoint_sha256,
                "teacher_runtime_dependency": False,
                "confirmation_passes": confirmation_passes,
            },
            output_checkpoint,
        )
        candidate_path = output_dir / "development_candidate.pt"
        model.load_state_dict(candidate_state)
        torch.save(
            {
                "schema": "generic-quintic-adaptive-direct-development-v1",
                "state_dict": state_to_cpu(model),
                "configuration": final_configuration,
                "parent_checkpoint_sha256": checkpoint_sha256,
                "teacher_runtime_dependency": False,
                "confirmation_passes": confirmation_passes,
            },
            candidate_path,
        )
        indices_path = output_dir / "data_indices.npz"
        np.savez_compressed(indices_path, **split_indices)
        report = {
            "schema": "generic-quintic-adaptive-direct-blocks-report-v1",
            "teacher_role": "absent",
            "configuration": {
                **vars(args),
                "initial_checkpoint": str(checkpoint_path),
                "output_dir": str(output_dir),
            },
            "blocks": [asdict(block) for block in blocks],
            "expansion": expansion,
            "embedding_audit": embedding_audit,
            "initial_selection": initial_selection,
            "initial_selection_tail": initial_selection_tail,
            "final_selection": current_selection,
            "final_selection_tail": current_selection_tail,
            "accepted_blocks": accepted_blocks,
            "history": history,
            "confirmation": confirmation,
            "data": {
                "indices": str(indices_path),
                "indices_sha256": sha256_file(indices_path),
                "counts": {
                    name: int(len(rows))
                    for name, rows in split_indices.items()
                },
            },
            "development_candidate": str(candidate_path),
            "development_candidate_sha256": sha256_file(candidate_path),
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
                "accepted_blocks": accepted_blocks,
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
