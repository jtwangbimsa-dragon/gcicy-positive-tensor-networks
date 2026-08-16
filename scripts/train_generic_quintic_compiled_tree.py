#!/usr/bin/env python3
"""Matched native-E2 training arms from one compiled generic-quintic H4 teacher."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
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
    BinaryTreeTopology,
    PositiveMultiplicationTreeMetric,
    build_teacher_compiled_multiplication_tree,
)
from gcicy_metric.pipeline.exact_lift_tree import (  # noqa: E402
    build_exact_power_lift_tree,
)
from gcicy_metric.pipeline.projective_residual_phi import (  # noqa: E402
    ProjectiveResidualPhiMetric,
)
from scripts.train_quintic_native_power_lift_tree import (  # noqa: E402
    dataset_subset,
    load_pool_arrays,
    make_dataset,
    native_e2_streaming_backward,
    normalized_native_e2,
    paired_improvement,
    statistics,
    tail_guard,
    tail_summary,
)
from scripts.train_quintic_full_h_same_points import write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architecture",
        choices=("compiled-tree", "direct-tn", "residual-phi"),
        default="compiled-tree",
        help=(
            "Use the multiplication tree, a direct dense-local-core power-lift "
            "TN, or a residual-phi model under one matched protocol."
        ),
    )
    parser.add_argument("--compiler-artifact", type=Path, required=True)
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        help="Continue from a checkpoint of the selected architecture.",
    )
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
    parser.add_argument("--bond-dimension", type=int, default=5)
    parser.add_argument("--activation-scale", type=float, default=2.0e-2)
    parser.add_argument("--positive-floor", type=float, default=1.0e-8)
    parser.add_argument("--phi-hidden-width", type=int, default=144)
    parser.add_argument("--phi-hidden-layers", type=int, default=3)
    parser.add_argument("--phi-potential-scale", type=float, default=1.0)
    parser.add_argument("--root-epochs", type=int, default=80)
    parser.add_argument("--internal-epochs", type=int, default=160)
    parser.add_argument("--all-epochs", type=int, default=240)
    parser.add_argument("--root-learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--internal-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--all-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--stochastic-batch-size", type=int, default=4096)
    parser.add_argument("--selection-eval-every", type=int, default=5)
    parser.add_argument(
        "--early-stopping-patience-evaluations",
        type=int,
        default=0,
        help="Stop a stage after this many evaluations without a significant gain.",
    )
    parser.add_argument(
        "--minimum-relative-chi-improvement",
        type=float,
        default=0.0,
        help="Relative chi gain required to reset early-stopping patience.",
    )
    parser.add_argument("--feature-batch-size", type=int, default=1024)
    parser.add_argument("--train-chunk-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--maximum-selection-tail-relative-degradation",
        type=float,
        default=0.005,
    )
    parser.add_argument("--seed", type=int, default=202607404)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--precision",
        choices=("complex64", "complex128"),
        default="complex64",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plateau_progress(
    *,
    score: float,
    passes_gate: bool,
    reference_score: float,
    evaluations_without_gain: int,
    minimum_relative_improvement: float,
) -> tuple[float, int, bool]:
    threshold = reference_score * (1.0 - minimum_relative_improvement)
    significant = bool(passes_gate and score <= threshold)
    if significant:
        return score, 0, True
    return reference_score, evaluations_without_gain + 1, False


def should_activate_dormant_channels(
    architecture: str,
    initial_checkpoint: Path | None,
    *,
    teacher_compiled_structure: bool = False,
) -> bool:
    return bool(
        architecture == "compiled-tree"
        and initial_checkpoint is None
        and not teacher_compiled_structure
    )


def whiten_dataset(
    dataset: dict[str, Any],
    whitening: np.ndarray,
) -> dict[str, Any]:
    transform = torch.tensor(
        whitening,
        dtype=dataset["values"].dtype,
        device=dataset["values"].device,
    )
    result = dict(dataset)
    result["values"] = torch.einsum(
        "ab,nb->na",
        transform,
        dataset["values"],
    )
    result["derivatives"] = torch.einsum(
        "ab,nbj->naj",
        transform,
        dataset["derivatives"],
    )
    return result


def add_projective_inputs(
    dataset: dict[str, Any],
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, Any]:
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    result = dict(dataset)
    result["real_coordinates"] = torch.tensor(
        arrays[0],
        dtype=real_dtype,
        device=device,
    )
    result["pullbacks"] = torch.tensor(
        arrays[2],
        dtype=dtype,
        device=device,
    )
    return result


def metric_row(
    model: torch.nn.Module,
    dataset: dict[str, Any],
    *,
    chunk_size: int,
) -> tuple[dict[str, Any], np.ndarray, dict[str, float]]:
    result, ratio = statistics(model, dataset, chunk_size=chunk_size)
    return result, ratio, tail_summary(result)


def build_model(
    args: argparse.Namespace,
    reference_h: np.ndarray,
    *,
    power: int,
    source_normalization: float,
    dtype: torch.dtype,
    device: torch.device,
    compiler: Any | None = None,
) -> torch.nn.Module:
    if args.architecture == "compiled-tree":
        compiled_keys = {
            "compiled_leaf_tensor",
            "compiled_leaf_combiner",
            "compiled_topology_children",
        }
        if compiler is not None and compiled_keys.issubset(set(compiler.files)):
            topology = BinaryTreeTopology(
                leaf_count=power,
                children=tuple(
                    tuple(int(value) for value in pair)
                    for pair in np.asarray(
                        compiler["compiled_topology_children"],
                        dtype=np.int64,
                    )
                ),
            )
            model = build_teacher_compiled_multiplication_tree(
                reference_h,
                leaf_tensor=np.asarray(
                    compiler["compiled_leaf_tensor"],
                    dtype=np.complex128,
                ),
                leaf_combiner=np.asarray(
                    compiler["compiled_leaf_combiner"],
                    dtype=np.complex128,
                ),
                leaf_count=power,
                source_normalization=source_normalization,
                positive_floor=args.positive_floor,
                topology=topology,
                dtype=dtype,
                device=device,
            )
            if "compiled_edge_dimensions" in compiler.files:
                expected = tuple(
                    int(value)
                    for value in np.asarray(
                        compiler["compiled_edge_dimensions"],
                        dtype=np.int64,
                    )
                )
                if model.edge_dimensions != expected:
                    raise ValueError(
                        "compiled model dimensions disagree with the artifact"
                    )
            return model
        return PositiveMultiplicationTreeMetric(
            reference_h,
            leaf_count=power,
            bond_dimension=args.bond_dimension,
            source_normalization=source_normalization,
            positive_floor=args.positive_floor,
            shared_leaf=True,
            dtype=dtype,
            device=device,
        )
    if args.architecture == "residual-phi":
        return ProjectiveResidualPhiMetric(
            reference_h,
            normalization=source_normalization,
            ambient_dimension=5,
            hidden_width=args.phi_hidden_width,
            hidden_layers=args.phi_hidden_layers,
            potential_scale=args.phi_potential_scale,
            complex_dtype=dtype,
            device=device,
        )
    return build_exact_power_lift_tree(
        reference_h,
        source_normalization=source_normalization,
        power=power,
        bond_dimension=args.bond_dimension,
        positive_floor=args.positive_floor,
        precision=args.precision,
        device=device,
    )


def main() -> None:
    args = parse_args()
    if (
        min(
            args.train_size,
            args.selection_size,
            args.confirmation_size,
            args.bond_dimension,
            args.phi_hidden_width,
            args.phi_hidden_layers,
            args.stochastic_batch_size,
            args.selection_eval_every,
            args.feature_batch_size,
            args.train_chunk_size,
            args.eval_batch_size,
            args.threads,
        )
        <= 0
        or min(args.root_epochs, args.internal_epochs, args.all_epochs) < 0
        or args.early_stopping_patience_evaluations < 0
        or not 0.0 <= args.minimum_relative_chi_improvement < 1.0
        or min(
            args.root_learning_rate,
            args.internal_learning_rate,
            args.all_learning_rate,
            args.gradient_clip_norm,
            args.activation_scale,
            args.phi_potential_scale,
        )
        <= 0
    ):
        raise ValueError("matched-arm training configuration is invalid")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    dtype = (
        torch.complex64
        if args.precision == "complex64"
        else torch.complex128
    )
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite a matched-arm run")
    output_dir.mkdir(parents=True)
    status_path = output_dir / "status.json"
    write_json(status_path, {"state": "running", "phase": "loading"})
    started = time.perf_counter()

    try:
        compiler_path = args.compiler_artifact.expanduser().resolve()
        compiler = np.load(compiler_path, allow_pickle=False)
        compiler_schema = str(compiler["schema"])
        if compiler_schema not in {
            "generic-quintic-full-h-tree-compiler-v1",
            "generic-quintic-full-h-tree-compiler-v2",
        }:
            raise ValueError("unsupported compiler artifact")
        teacher_compiled_structure = {
            "compiled_leaf_tensor",
            "compiled_leaf_combiner",
            "compiled_topology_children",
        }.issubset(set(compiler.files))
        degree = int(compiler["degree"])
        power = int(compiler["power"])
        if degree != 4 or power != 5:
            raise ValueError("compiled-tree experiment expects H4 lifted to k20")
        exponents = np.asarray(
            compiler["degree4_exponents"],
            dtype=np.int64,
        )
        whitening = np.asarray(
            compiler["degree4_whitening"],
            dtype=np.complex128,
        )
        reference_h = np.asarray(
            compiler["whitened_teacher_h"],
            dtype=np.complex128,
        )
        reference_h = reference_h * (
            len(reference_h) / np.trace(reference_h).real
        )
        source_normalization = float(compiler["source_normalization"])

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
        write_json(status_path, {"state": "running", "phase": "features"})
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
        if args.architecture == "residual-phi":
            training = add_projective_inputs(
                training,
                native_arrays,
                dtype=dtype,
                device=device,
            )
            selection = add_projective_inputs(
                selection,
                selection_arrays,
                dtype=dtype,
                device=device,
            )
            confirmation = add_projective_inputs(
                confirmation,
                confirmation_arrays,
                dtype=dtype,
                device=device,
            )

        model = build_model(
            args,
            reference_h,
            power=power,
            source_normalization=source_normalization,
            dtype=dtype,
            device=device,
            compiler=compiler,
        )
        parent_checkpoint_sha256 = None
        if args.initial_checkpoint is not None:
            checkpoint_path = args.initial_checkpoint.expanduser().resolve()
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            checkpoint_configuration = checkpoint.get("configuration", {})
            checkpoint_architecture = checkpoint_configuration.get(
                "architecture"
            )
            if checkpoint_architecture is None:
                checkpoint_architecture = (
                    "compiled-tree"
                    if "compiled-positive-tree"
                    in str(checkpoint.get("schema", ""))
                    else None
                )
            if checkpoint_architecture != args.architecture:
                raise ValueError(
                    "initial checkpoint architecture does not match the arm"
                )
            model.load_state_dict(checkpoint["state_dict"])
            parent_checkpoint_sha256 = sha256_file(checkpoint_path)
        baseline_statistics, baseline_ratio, baseline_tail = metric_row(
            model,
            selection,
            chunk_size=args.eval_batch_size,
        )
        if should_activate_dormant_channels(
            args.architecture,
            args.initial_checkpoint,
            teacher_compiled_structure=teacher_compiled_structure,
        ):
            model.activate_dormant_output_channels_(
                relative_scale=args.activation_scale,
                seed=args.seed + 4,
            )
        activated_statistics, activated_ratio, _ = metric_row(
            model,
            selection,
            chunk_size=args.eval_batch_size,
        )
        activation_metric_error = float(
            np.max(np.abs(activated_ratio - baseline_ratio))
        )
        if activation_metric_error > (
            2.0e-5 if dtype == torch.complex64 else 2.0e-11
        ):
            raise FloatingPointError(
                "dormant-channel activation changed the initial metric"
            )

        history: list[dict[str, Any]] = [
            {
                "stage": "initial",
                "epoch": 0,
                "sigma": baseline_statistics["sigma_official_formula"],
                "chi": baseline_statistics["weighted_rms_abs_residual"],
                **baseline_tail,
            }
        ]
        best_state = copy.deepcopy(model.state_dict())
        best_score = float(
            baseline_statistics["weighted_rms_abs_residual"]
        )
        best_row = history[0]
        global_epoch = 0
        write_json(output_dir / "history.json", {"rows": history})
        print(
            f"parameters={model.trainable_real_parameter_count} "
            f"baseline_sigma={history[0]['sigma']:.6e} "
            f"baseline_chi={history[0]['chi']:.6e}",
            flush=True,
        )

        stages = (
            ("root", args.root_epochs, args.root_learning_rate),
            ("internal", args.internal_epochs, args.internal_learning_rate),
            ("all", args.all_epochs, args.all_learning_rate),
        )
        early_stopping: dict[str, Any] = {}
        write_json(status_path, {"state": "running", "phase": "native_e2"})
        for stage, epochs, learning_rate in stages:
            if epochs == 0:
                continue
            model.load_state_dict(best_state)
            if args.architecture == "compiled-tree":
                model.set_trainable_stage_(stage)
            else:
                for parameter in model.parameters():
                    parameter.requires_grad_(True)
            parameters = [
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad
            ]
            optimizer = torch.optim.Adam(parameters, lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs,
                eta_min=0.05 * learning_rate,
            )
            significant_score = best_score
            evaluations_without_gain = 0
            for stage_epoch in range(1, epochs + 1):
                global_epoch += 1
                if args.stochastic_batch_size < training["count"]:
                    indices = rng.choice(
                        training["count"],
                        size=args.stochastic_batch_size,
                        replace=False,
                    )
                    active = dataset_subset(training, indices)
                else:
                    active = training
                model.train()
                optimizer.zero_grad(set_to_none=True)
                if args.architecture == "residual-phi":
                    loss, _ = native_e2_streaming_backward(
                        model,
                        active,
                        chunk_size=args.train_chunk_size,
                    )
                else:
                    loss, _ = normalized_native_e2(
                        model,
                        active,
                        chunk_size=args.train_chunk_size,
                    )
                    loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    parameters,
                    args.gradient_clip_norm,
                )
                optimizer.step()
                scheduler.step()

                if (
                    stage_epoch % args.selection_eval_every == 0
                    or stage_epoch == epochs
                ):
                    candidate_statistics, _, candidate_tail = metric_row(
                        model,
                        selection,
                        chunk_size=args.eval_batch_size,
                    )
                    score = float(
                        candidate_statistics["weighted_rms_abs_residual"]
                    )
                    sigma = float(
                        candidate_statistics["sigma_official_formula"]
                    )
                    passes = bool(
                        score < best_score
                        and sigma
                        <= float(
                            baseline_statistics["sigma_official_formula"]
                        )
                        and tail_guard(
                            candidate_tail,
                            baseline_tail,
                            relative_degradation=(
                                args.maximum_selection_tail_relative_degradation
                            ),
                        )
                    )
                    row = {
                        "stage": stage,
                        "stage_epoch": stage_epoch,
                        "epoch": global_epoch,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "train_e2": float(loss.detach().cpu()),
                        "gradient_norm": float(gradient_norm),
                        "sigma": sigma,
                        "chi": score,
                        "passes_selection_gate": passes,
                        **candidate_tail,
                    }
                    history.append(row)
                    if passes:
                        best_score = score
                        best_state = copy.deepcopy(model.state_dict())
                        best_row = row
                    (
                        significant_score,
                        evaluations_without_gain,
                        significant,
                    ) = plateau_progress(
                        score=score,
                        passes_gate=passes,
                        reference_score=significant_score,
                        evaluations_without_gain=evaluations_without_gain,
                        minimum_relative_improvement=(
                            args.minimum_relative_chi_improvement
                        ),
                    )
                    row["significant_improvement"] = significant
                    row["evaluations_without_significant_gain"] = (
                        evaluations_without_gain
                    )
                    write_json(output_dir / "history.json", {"rows": history})
                    write_json(
                        status_path,
                        {
                            "state": "running",
                            "phase": "native_e2",
                            "stage": stage,
                            "stage_epoch": stage_epoch,
                            "global_epoch": global_epoch,
                            "best_chi": best_score,
                        },
                    )
                    print(
                        f"stage={stage} epoch={stage_epoch}/{epochs} "
                        f"sigma={sigma:.6e} chi={score:.6e} "
                        f"accepted={passes}",
                        flush=True,
                    )
                    if (
                        args.early_stopping_patience_evaluations > 0
                        and evaluations_without_gain
                        >= args.early_stopping_patience_evaluations
                    ):
                        early_stopping[stage] = {
                            "stage_epoch": stage_epoch,
                            "global_epoch": global_epoch,
                            "evaluations_without_significant_gain": (
                                evaluations_without_gain
                            ),
                            "minimum_relative_chi_improvement": (
                                args.minimum_relative_chi_improvement
                            ),
                            "significant_reference_chi": significant_score,
                        }
                        write_json(
                            status_path,
                            {
                                "state": "running",
                                "phase": "native_e2",
                                "stage": stage,
                                "stage_epoch": stage_epoch,
                                "global_epoch": global_epoch,
                                "best_chi": best_score,
                                "early_stopped_stage": early_stopping[stage],
                            },
                        )
                        print(
                            f"stage={stage} early_stop epoch={stage_epoch} "
                            f"best_chi={best_score:.6e}",
                            flush=True,
                        )
                        break

        model.load_state_dict(best_state)
        candidate_confirmation, candidate_ratio, candidate_tail = metric_row(
            model,
            confirmation,
            chunk_size=args.eval_batch_size,
        )
        baseline_model = build_model(
            args,
            reference_h,
            power=power,
            source_normalization=source_normalization,
            dtype=dtype,
            device=device,
            compiler=compiler,
        )
        baseline_confirmation, confirmation_base_ratio, confirmation_base_tail = (
            metric_row(
                baseline_model,
                confirmation,
                chunk_size=args.eval_batch_size,
            )
        )
        paired = paired_improvement(
            confirmation_base_ratio,
            candidate_ratio,
            confirmation["weights_numpy"],
        )
        confirmation_passes = bool(
            candidate_confirmation["sigma_official_formula"]
            < baseline_confirmation["sigma_official_formula"]
            and candidate_confirmation["weighted_rms_abs_residual"]
            < baseline_confirmation["weighted_rms_abs_residual"]
            and paired["e2"]["ci95_low"] > 0
            and paired["sigma"]["ci95_low"] > 0
            and tail_guard(
                candidate_tail,
                confirmation_base_tail,
                relative_degradation=0.0,
            )
        )
        checkpoint_path = output_dir / (
            "accepted.pt" if confirmation_passes else "diagnostic_rejected.pt"
        )
        torch.save(
            {
                "schema": (
                    "generic-quintic-compiled-positive-tree-v1"
                    if args.architecture == "compiled-tree"
                    else (
                        "generic-quintic-direct-positive-tn-v1"
                        if args.architecture == "direct-tn"
                        else "generic-quintic-h4-residual-phi-v1"
                    )
                ),
                "state_dict": {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
                "configuration": {
                    "leaf_count": power,
                    "bond_dimension": max(
                        getattr(model, "edge_dimensions", (args.bond_dimension,))
                    ),
                    "edge_dimensions": tuple(
                        int(value)
                        for value in getattr(
                            model,
                            "edge_dimensions",
                            (args.bond_dimension,) * (2 * power - 2),
                        )
                    ),
                    "source_normalization": source_normalization,
                    "positive_floor": args.positive_floor,
                    "shared_leaf": args.architecture == "compiled-tree",
                    "architecture": args.architecture,
                    "initialization_mode": (
                        "teacher-schmidt-compiled"
                        if teacher_compiled_structure
                        and args.architecture == "compiled-tree"
                        else "legacy-fixed-rank"
                    ),
                    "precision": args.precision,
                    "phi_hidden_width": args.phi_hidden_width,
                    "phi_hidden_layers": args.phi_hidden_layers,
                    "phi_potential_scale": args.phi_potential_scale,
                    "early_stopping_patience_evaluations": (
                        args.early_stopping_patience_evaluations
                    ),
                    "minimum_relative_chi_improvement": (
                        args.minimum_relative_chi_improvement
                    ),
                    "degree": degree,
                    "exponents": exponents,
                    "whitening": whitening,
                },
                "compiler_artifact_sha256": sha256_file(compiler_path),
                "parent_checkpoint_sha256": parent_checkpoint_sha256,
                "teacher_runtime_dependency": False,
                "confirmation_passes": confirmation_passes,
            },
            checkpoint_path,
        )
        indices_path = output_dir / "data_indices.npz"
        np.savez_compressed(
            indices_path,
            native=native_indices,
            selection=selection_indices,
            confirmation=confirmation_indices,
        )
        report = {
            "schema": (
                "generic-quintic-compiled-tree-native-e2-v1"
                if args.architecture == "compiled-tree"
                else (
                    "generic-quintic-direct-tn-native-e2-v1"
                    if args.architecture == "direct-tn"
                    else "generic-quintic-h4-residual-phi-native-e2-v1"
                )
            ),
            "teacher_role": "round_zero_coordinate_and_tree_initialization_only",
            "teacher_runtime_dependency_after_initialization": False,
            "parent_checkpoint_sha256": parent_checkpoint_sha256,
            "configuration": {
                **vars(args),
                "compiler_artifact": str(compiler_path),
                "output_dir": str(output_dir),
            },
            "model": {
                "target_degree": degree * power,
                "leaf_count": power,
                "bond_dimension": max(
                    getattr(model, "edge_dimensions", (args.bond_dimension,))
                ),
                "edge_dimensions": list(
                    getattr(model, "edge_dimensions", ())
                ),
                "initialization_mode": (
                    "teacher-schmidt-compiled"
                    if teacher_compiled_structure
                    and args.architecture == "compiled-tree"
                    else "legacy-fixed-rank"
                ),
                "architecture": args.architecture,
                "trainable_real_parameter_count": (
                    model.trainable_real_parameter_count
                ),
                "best_selection_row": best_row,
            },
            "activation": {
                "maximum_ratio_change": activation_metric_error,
            },
            "early_stopping": early_stopping,
            "selection_baseline": baseline_statistics,
            "confirmation": {
                "baseline": baseline_confirmation,
                "candidate": candidate_confirmation,
                "baseline_tail": confirmation_base_tail,
                "candidate_tail": candidate_tail,
                "paired_improvement": paired,
                "passes": confirmation_passes,
            },
            "data_indices": {
                "artifact": str(indices_path),
                "artifact_sha256": sha256_file(indices_path),
                "counts": {
                    "native": len(native_indices),
                    "selection": len(selection_indices),
                    "confirmation": len(confirmation_indices),
                },
            },
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "wall_seconds": time.perf_counter() - started,
        }
        write_json(output_dir / "report.json", report)
        write_json(
            status_path,
            {
                "state": "complete",
                "phase": "complete",
                "confirmation_passes": confirmation_passes,
                "best_chi": best_score,
                "wall_seconds": time.perf_counter() - started,
            },
        )
        print(
            json.dumps(
                {
                    "confirmation_passes": confirmation_passes,
                    "trainable_real_parameter_count": (
                        model.trainable_real_parameter_count
                    ),
                    "baseline_sigma": baseline_confirmation[
                        "sigma_official_formula"
                    ],
                    "candidate_sigma": candidate_confirmation[
                        "sigma_official_formula"
                    ],
                    "baseline_chi": baseline_confirmation[
                        "weighted_rms_abs_residual"
                    ],
                    "candidate_chi": candidate_confirmation[
                        "weighted_rms_abs_residual"
                    ],
                    "checkpoint": str(checkpoint_path),
                    "wall_seconds": report["wall_seconds"],
                },
                indent=2,
                sort_keys=True,
            )
        )
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
