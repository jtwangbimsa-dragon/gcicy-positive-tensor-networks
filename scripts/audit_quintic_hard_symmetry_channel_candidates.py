#!/usr/bin/env python3
"""Rank exact-nested hard-symmetry bond channels by residual capture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gcicy_metric.pipeline import positive_tensor_network_from_artifact_payload  # noqa: E402
from scripts.audit_quintic_tn_metric_tangent_reachability import (  # noqa: E402
    MaskedComplexParameterVectorizer,
    MatrixFreeTeacherMetricJacobian,
    capture_fraction,
    residual_metrics,
)
from scripts.audit_quintic_tn_tangent_reachability import (  # noqa: E402
    real_inner,
    select_disjoint_indices,
    vector_norm,
)
from scripts.initialize_quintic_hard_symmetry_channel import (  # noqa: E402
    one_sided_channel_masks,
    seed_one_sided_channel_,
)
from scripts.refine_quintic_tn_reynolds_metric import teacher_targets  # noqa: E402
from scripts.train_quintic_full_h_same_points import (  # noqa: E402
    sha256_file,
    write_json,
)
from scripts.train_quintic_positive_tensor_network_same_points import (  # noqa: E402
    fixed_fermat_actions_torch,
    tensor_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expanded-model", type=Path, required=True)
    parser.add_argument("--teacher-model", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--pullbacks-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-start", type=int, default=0)
    parser.add_argument("--candidate-limit", type=int, default=5000)
    parser.add_argument("--probe-size", type=int, default=64)
    parser.add_argument(
        "--selection-size",
        type=int,
        default=0,
        help=(
            "independent points used to rank probe-fitted candidates; zero "
            "retains the legacy two-way audit"
        ),
    )
    parser.add_argument("--holdout-size", type=int, default=64)
    parser.add_argument("--group-samples", type=int, default=64)
    parser.add_argument("--teacher-batch-size", type=int, default=16)
    parser.add_argument("--teacher-action-batch-size", type=int, default=64)
    parser.add_argument("--operator-chunk-size", type=int, default=1)
    parser.add_argument("--mode", choices=("metric", "logdet"), default="logdet")
    parser.add_argument("--bond-indices", type=int, nargs="*")
    parser.add_argument("--seeds-per-bond", type=int, default=2)
    parser.add_argument(
        "--step-scales",
        type=float,
        nargs="+",
        default=(0.125, 0.25, 0.5, 1.0),
    )
    parser.add_argument("--relative-seed-scale", type=float, default=1.0e-3)
    parser.add_argument(
        "--seed-reference-scale",
        choices=("inherited_rms", "unit_rms"),
        default="inherited_rms",
    )
    parser.add_argument("--expected-parameter-count", type=int, default=0)
    parser.add_argument("--seed", type=int, default=202607244)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.candidate_limit,
        args.probe_size,
        args.holdout_size,
        args.group_samples,
        args.teacher_batch_size,
        args.teacher_action_batch_size,
        args.operator_chunk_size,
        args.seeds_per_bond,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("all sample, batch, chunk, and seed counts must be positive")
    if args.candidate_start < 0 or args.selection_size < 0:
        raise ValueError("candidate start and selection size cannot be negative")
    if not np.isfinite(args.relative_seed_scale) or args.relative_seed_scale <= 0:
        raise ValueError("relative seed scale must be finite and positive")
    if args.expected_parameter_count < 0:
        raise ValueError("expected parameter count cannot be negative")
    if any(
        not np.isfinite(value) or not 0 < value <= 1
        for value in args.step_scales
    ):
        raise ValueError("step scales must lie in (0, 1]")


def steepest_probe_candidate(
    probe: MatrixFreeTeacherMetricJacobian,
    baseline_residual: torch.Tensor | None = None,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    """Return the probe-optimal linear step along the negative gradient."""

    probe_residual = (
        probe.residual() if baseline_residual is None else baseline_residual
    )
    gradient = probe.vjp(probe_residual)
    gradient_norm = vector_norm(gradient)
    if not bool(torch.isfinite(gradient_norm)) or float(gradient_norm) <= 0:
        raise FloatingPointError("candidate channel has no finite residual gradient")
    direction = -gradient / gradient_norm
    probe_tangent = probe.jvp(direction)
    tangent_squared = float(real_inner(probe_tangent, probe_tangent))
    if not np.isfinite(tangent_squared) or tangent_squared <= 0:
        raise FloatingPointError("candidate channel has no finite tangent response")
    alpha = -float(real_inner(probe_residual, probe_tangent)) / tangent_squared
    predicted_probe = probe_residual + alpha * probe_tangent
    result = {
        "probe_baseline": residual_metrics(probe_residual),
        "gradient_norm": float(gradient_norm),
        "unit_direction_tangent_squared": tangent_squared,
        "optimal_linear_alpha": alpha,
        "probe_linear_capture": capture_fraction(probe_residual, predicted_probe),
        "probe_linear_residual": residual_metrics(predicted_probe),
    }
    return result, alpha * direction, probe_residual.detach()


def confirm_linear_holdout(
    holdout: MatrixFreeTeacherMetricJacobian,
    proposed_step: torch.Tensor,
    baseline_residual: torch.Tensor | None = None,
) -> dict[str, Any]:
    baseline = (
        holdout.residual() if baseline_residual is None else baseline_residual
    )
    predicted = baseline + holdout.jvp(proposed_step)
    return {
        "baseline": residual_metrics(baseline),
        "linear_capture": capture_fraction(baseline, predicted),
        "linear_residual": residual_metrics(predicted),
    }


def linear_step_scale_curve(
    operator: MatrixFreeTeacherMetricJacobian,
    proposed_step: torch.Tensor,
    step_scales: tuple[float, ...],
    baseline_residual: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Evaluate one frozen direction at preregistered linear step scales."""

    baseline = (
        operator.residual() if baseline_residual is None else baseline_residual
    )
    tangent = operator.jvp(proposed_step)
    rows = []
    for scale in step_scales:
        predicted = baseline + float(scale) * tangent
        rows.append(
            {
                "step_scale": float(scale),
                "linear_capture": capture_fraction(baseline, predicted),
                "linear_residual": residual_metrics(predicted),
            }
        )
    return baseline, rows


def passes_confirmation_gate(rank_score: float) -> bool:
    """Only a finite, genuinely improving selection result may be confirmed."""

    return bool(np.isfinite(rank_score) and rank_score > 0.0)


def split_candidate_indices(
    candidate_count: int,
    fit_size: int,
    selection_size: int,
    confirmation_size: int,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return disjoint fit, model-selection, and final-confirmation indices."""

    fit_and_selection, confirmation = select_disjoint_indices(
        candidate_count,
        fit_size + selection_size,
        confirmation_size,
        seed=seed,
    )
    fit = fit_and_selection[:fit_size]
    selection = fit_and_selection[fit_size:]
    return fit, selection, confirmation


def main() -> None:
    args = parse_args()
    validate_args(args)
    started = time.perf_counter()
    output = args.output.expanduser().resolve()
    status = output.with_suffix(output.suffix + ".status.json")
    if output.exists() or status.exists():
        raise FileExistsError("refusing to overwrite a channel-candidate audit")
    output.parent.mkdir(parents=True, exist_ok=True)
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
    teacher_path = args.teacher_model.expanduser().resolve()
    source_dir = args.source_run_dir.expanduser().resolve()
    pullbacks_dir = args.pullbacks_dir.expanduser().resolve()
    dataset_path = source_dir / "training_data" / "dataset.npz"
    pullbacks_path = pullbacks_dir / "validation_pullbacks.npy"
    for path in (model_path, teacher_path, dataset_path, pullbacks_path):
        if not path.exists():
            raise FileNotFoundError(path)

    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    teacher_payload = torch.load(teacher_path, map_location="cpu", weights_only=False)
    expansion = payload.get("phase_multiplicity_expansion")
    if not isinstance(expansion, dict):
        raise ValueError("candidate model is not an exact multiplicity expansion")
    if not bool(expansion.get("function_preserving_by_construction", False)):
        raise ValueError("candidate model must be expanded with zero activation noise")
    source_multiplicity = int(expansion["source_multiplicity"])
    target_multiplicity = int(expansion["target_multiplicity"])
    source_degree = int(payload.get("source_degree", 1))
    reference_h = np.asarray(
        payload["state_dict"]["reference_h"].detach().cpu(),
        dtype=np.complex128,
    )
    block_count = len(payload["fermat_two_site_block_starts"])
    available_bonds = tuple(range(block_count - 1))
    bonds = (
        available_bonds
        if not args.bond_indices
        else tuple(int(value) for value in args.bond_indices)
    )
    if not bonds or any(value not in available_bonds for value in bonds):
        raise ValueError("requested bond index is outside the blocked chain")

    data = np.load(dataset_path, allow_pickle=False)
    candidate_stop = min(
        args.candidate_start + args.candidate_limit,
        len(data["X_val"]),
    )
    candidate_count = candidate_stop - args.candidate_start
    if candidate_count <= 0:
        raise ValueError("candidate index window is empty")
    probe_indices, selection_indices, holdout_indices = split_candidate_indices(
        candidate_count,
        args.probe_size,
        args.selection_size,
        args.holdout_size,
        seed=args.seed,
    )
    probe_indices = probe_indices + args.candidate_start
    selection_indices = selection_indices + args.candidate_start
    holdout_indices = holdout_indices + args.candidate_start
    validation_pullbacks = np.load(pullbacks_path, mmap_mode="r")

    def arrays(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.asarray(data["X_val"][indices], dtype=np.float32),
            np.asarray(validation_pullbacks[indices]),
            np.asarray(data["y_val"][indices], dtype=np.float64),
        )

    probe_arrays = arrays(probe_indices)
    selection_arrays = arrays(selection_indices) if args.selection_size else None
    holdout_arrays = arrays(holdout_indices)
    target_probe = tensor_split(
        *probe_arrays,
        source_degree=source_degree,
        complex_dtype=torch.complex128,
        real_dtype=torch.float64,
        device=device,
    )
    target_holdout = tensor_split(
        *holdout_arrays,
        source_degree=source_degree,
        complex_dtype=torch.complex128,
        real_dtype=torch.float64,
        device=device,
    )
    target_selection = (
        tensor_split(
            *selection_arrays,
            source_degree=source_degree,
            complex_dtype=torch.complex128,
            real_dtype=torch.float64,
            device=device,
        )
        if selection_arrays is not None
        else None
    )
    teacher = positive_tensor_network_from_artifact_payload(
        reference_h,
        teacher_payload,
        device=device,
        trainable_physical_dictionary=False,
    ).to(device=device, dtype=torch.complex128)
    teacher.requires_grad_(False).eval()
    action_generator = torch.Generator(device=device)
    action_generator.manual_seed(args.seed + 1009)
    actions = fixed_fermat_actions_torch(
        args.group_samples,
        generator=action_generator,
        complex_dtype=torch.complex128,
        device=device,
    )
    write_json(status, {"state": "running", "phase": "teacher_targets"})
    _, probe_target_metric = teacher_targets(
        teacher,
        target_probe,
        actions,
        batch_size=args.teacher_batch_size,
        action_batch_size=args.teacher_action_batch_size,
    )
    selection_target_metric = None
    if target_selection is not None:
        _, selection_target_metric = teacher_targets(
            teacher,
            target_selection,
            actions,
            batch_size=args.teacher_batch_size,
            action_batch_size=args.teacher_action_batch_size,
        )
    _, holdout_target_metric = teacher_targets(
        teacher,
        target_holdout,
        actions,
        batch_size=args.teacher_batch_size,
        action_batch_size=args.teacher_action_batch_size,
    )
    del teacher, actions, target_probe, target_holdout, target_selection

    student_dtype = (
        torch.complex64 if str(payload["precision"]) == "complex64" else torch.complex128
    )
    real_dtype = torch.float32 if student_dtype == torch.complex64 else torch.float64
    probe = tensor_split(
        *probe_arrays,
        source_degree=source_degree,
        complex_dtype=student_dtype,
        real_dtype=real_dtype,
        device=device,
    )
    holdout = tensor_split(
        *holdout_arrays,
        source_degree=source_degree,
        complex_dtype=student_dtype,
        real_dtype=real_dtype,
        device=device,
    )
    selection = (
        tensor_split(
            *selection_arrays,
            source_degree=source_degree,
            complex_dtype=student_dtype,
            real_dtype=real_dtype,
            device=device,
        )
        if selection_arrays is not None
        else None
    )
    probe_target_metric = probe_target_metric.to(dtype=student_dtype)
    holdout_target_metric = holdout_target_metric.to(dtype=student_dtype)
    if selection_target_metric is not None:
        selection_target_metric = selection_target_metric.to(dtype=student_dtype)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    rows = []
    reference_probe_residual = None
    reference_selection_residual = None
    selected_step = None
    for bond_index in bonds:
        for trial in range(args.seeds_per_bond):
            candidate_seed = args.seed + 100_003 * bond_index + trial
            write_json(
                status,
                {
                    "state": "running",
                    "phase": "candidates",
                    "bond_index": bond_index,
                    "trial": trial,
                },
            )
            model = positive_tensor_network_from_artifact_payload(
                reference_h,
                payload,
                device=device,
                trainable_physical_dictionary=False,
            ).to(device=device, dtype=student_dtype)
            model.force_indexed_block_contraction = True
            parameter_count = int(model.trainable_real_parameter_count)
            if (
                args.expected_parameter_count
                and parameter_count != args.expected_parameter_count
            ):
                raise RuntimeError(
                    f"parameter-count gate failed: {parameter_count} != "
                    f"{args.expected_parameter_count}"
                )
            generator = torch.Generator(device=device)
            generator.manual_seed(candidate_seed)
            activation = seed_one_sided_channel_(
                model,
                bond_index=bond_index,
                source_multiplicity=source_multiplicity,
                target_multiplicity=target_multiplicity,
                relative_scale=args.relative_seed_scale,
                generator=generator,
                reference_scale=args.seed_reference_scale,
            )
            left_mask, _ = one_sided_channel_masks(
                model,
                bond_index=bond_index,
                source_multiplicity=source_multiplicity,
                target_multiplicity=target_multiplicity,
            )
            parameter_name = f"blocked_two_site_orbit_parameters.{bond_index}"
            parameter = dict(model.named_parameters())[parameter_name]
            vectorizer = MaskedComplexParameterVectorizer.from_module(
                model,
                parameter_name,
                left_mask.reshape(parameter.shape),
            )
            theta = vectorizer.pack(model)
            # Only the replacement vector is differentiated by torch.func. If
            # the inherited model parameters stay trainable, every one-point
            # residual retains a full-model autograd graph and exhausts a 4090.
            model.requires_grad_(False).eval()
            probe_operator = MatrixFreeTeacherMetricJacobian(
                model,
                vectorizer,
                theta,
                probe,
                probe_target_metric,
                mode=args.mode,
                chunk_size=args.operator_chunk_size,
            )
            result, proposed_step, current_probe = steepest_probe_candidate(
                probe_operator,
                reference_probe_residual,
            )
            selection_evaluation = None
            ranked_step = proposed_step
            if selection is not None and selection_target_metric is not None:
                selection_operator = MatrixFreeTeacherMetricJacobian(
                    model,
                    vectorizer,
                    theta,
                    selection,
                    selection_target_metric,
                    mode=args.mode,
                    chunk_size=args.operator_chunk_size,
                )
                if reference_selection_residual is None:
                    reference_selection_residual = (
                        selection_operator.residual().detach().clone()
                    )
                _, selection_curve = linear_step_scale_curve(
                    selection_operator,
                    proposed_step,
                    tuple(args.step_scales),
                    reference_selection_residual,
                )
                selected_scale = max(
                    selection_curve,
                    key=lambda item: item["linear_capture"],
                )
                ranked_step = float(selected_scale["step_scale"]) * proposed_step
                selection_evaluation = {
                    "baseline": residual_metrics(reference_selection_residual),
                    "step_scale_curve": selection_curve,
                    "selected_step_scale": selected_scale["step_scale"],
                    "linear_capture": selected_scale["linear_capture"],
                    "linear_residual": selected_scale["linear_residual"],
                }
            if reference_probe_residual is None:
                reference_probe_residual = current_probe.detach().clone()
            preservation_error = float(
                torch.max(torch.abs(current_probe - reference_probe_residual))
            )
            row = {
                "bond_index": int(bond_index),
                "trial": int(trial),
                "candidate_seed": int(candidate_seed),
                "parameter_name": parameter_name,
                "active_complex_parameter_count": int(theta.numel()),
                "proposed_step_rms": float(
                    torch.sqrt(torch.mean(torch.abs(proposed_step) ** 2))
                ),
                "ranked_step_rms": float(
                    torch.sqrt(torch.mean(torch.abs(ranked_step) ** 2))
                ),
                "epoch_zero_probe_residual_max_abs_change": preservation_error,
                "activation": activation,
                **result,
            }
            if selection_evaluation is not None:
                row["selection_evaluation"] = selection_evaluation
                row["candidate_rank_score"] = selection_evaluation["linear_capture"]
            else:
                row["candidate_rank_score"] = row["probe_linear_capture"]
            rows.append(row)
            if selected_step is None or row["candidate_rank_score"] > max(
                previous["candidate_rank_score"] for previous in rows[:-1]
            ):
                selected_step = ranked_step.detach().cpu().clone()
            if not args.quiet:
                print(
                    f"bond={bond_index} trial={trial} "
                    f"fit_capture={row['probe_linear_capture']:.4f} "
                    f"rank_score={row['candidate_rank_score']:.4f} "
                    f"step={selection_evaluation['selected_step_scale'] if selection_evaluation else 1.0:g}",
                    flush=True,
                )
            del model, probe_operator
            if selection_evaluation is not None:
                del selection_operator
            if device.type == "cuda":
                torch.cuda.empty_cache()

    selected = max(rows, key=lambda row: row["candidate_rank_score"])
    if selected_step is None:
        raise RuntimeError("candidate audit selected no channel step")
    confirmation_opened = passes_confirmation_gate(
        float(selected["candidate_rank_score"])
    )
    if confirmation_opened:
        selected_model = positive_tensor_network_from_artifact_payload(
            reference_h,
            payload,
            device=device,
            trainable_physical_dictionary=False,
        ).to(device=device, dtype=student_dtype)
        selected_model.force_indexed_block_contraction = True
        selected_generator = torch.Generator(device=device)
        selected_generator.manual_seed(int(selected["candidate_seed"]))
        seed_one_sided_channel_(
            selected_model,
            bond_index=int(selected["bond_index"]),
            source_multiplicity=source_multiplicity,
            target_multiplicity=target_multiplicity,
            relative_scale=args.relative_seed_scale,
            generator=selected_generator,
            reference_scale=args.seed_reference_scale,
        )
        selected_left_mask, _ = one_sided_channel_masks(
            selected_model,
            bond_index=int(selected["bond_index"]),
            source_multiplicity=source_multiplicity,
            target_multiplicity=target_multiplicity,
        )
        selected_parameter_name = str(selected["parameter_name"])
        selected_parameter = dict(selected_model.named_parameters())[
            selected_parameter_name
        ]
        selected_vectorizer = MaskedComplexParameterVectorizer.from_module(
            selected_model,
            selected_parameter_name,
            selected_left_mask.reshape(selected_parameter.shape),
        )
        selected_theta = selected_vectorizer.pack(selected_model)
        selected_model.requires_grad_(False).eval()
        holdout_operator = MatrixFreeTeacherMetricJacobian(
            selected_model,
            selected_vectorizer,
            selected_theta,
            holdout,
            holdout_target_metric,
            mode=args.mode,
            chunk_size=args.operator_chunk_size,
        )
        selected["holdout_confirmation"] = confirm_linear_holdout(
            holdout_operator,
            selected_step.to(device=device, dtype=selected_theta.dtype),
        )
    else:
        selected["holdout_confirmation"] = {
            "state": "skipped",
            "reason": "best independent selection capture was nonpositive",
        }
    indices_path = output.with_suffix(output.suffix + ".indices.npz")
    np.savez_compressed(
        indices_path,
        probe_indices=probe_indices,
        selection_indices=selection_indices,
        holdout_indices=holdout_indices,
    )
    report = {
        "schema": "quintic-hard-symmetry-channel-candidate-audit-v2",
        "scientific_scope": {
            "selection_rule": (
                "Fit each direction on probe points, rank candidates on an "
                "independent selection set when configured, and construct the "
                "confirmation operator only after one improving candidate is frozen."
            ),
            "candidate": (
                "One exact-nested virtual copy with a fixed one-sided seed and an "
                "optimized opposite factor tangent."
            ),
            "limitation": (
                "This is a one-direction linear preflight. Accepted nonlinear "
                "checkpoints require separate bulk and tail gates."
            ),
        },
        "configuration": {
            "device": str(device),
            "mode": args.mode,
            "candidate_count": candidate_count,
            "candidate_start": args.candidate_start,
            "probe_size": args.probe_size,
            "selection_size": args.selection_size,
            "holdout_size": args.holdout_size,
            "group_samples": args.group_samples,
            "operator_chunk_size": args.operator_chunk_size,
            "bond_indices": list(bonds),
            "seeds_per_bond": args.seeds_per_bond,
            "step_scales": list(args.step_scales),
            "relative_seed_scale": args.relative_seed_scale,
            "seed_reference_scale": args.seed_reference_scale,
            "source_multiplicity": source_multiplicity,
            "target_multiplicity": target_multiplicity,
            "seed": args.seed,
        },
        "source": {
            "expanded_model": str(model_path),
            "expanded_model_sha256": sha256_file(model_path),
            "teacher_model": str(teacher_path),
            "teacher_model_sha256": sha256_file(teacher_path),
            "dataset": str(dataset_path),
            "dataset_sha256": sha256_file(dataset_path),
            "validation_pullbacks": str(pullbacks_path),
            "validation_pullbacks_sha256": sha256_file(pullbacks_path),
            "indices": str(indices_path),
            "indices_sha256": sha256_file(indices_path),
        },
        "selected_candidate": selected,
        "confirmation_opened": confirmation_opened,
        "rows": rows,
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(output, report)
    write_json(status, {"state": "complete", "phase": "complete"})
    print(
        json.dumps(
            {
                "output": str(output),
                "selected_bond": selected["bond_index"],
                "selected_trial": selected["trial"],
                "selected_probe_capture": selected["probe_linear_capture"],
                "selected_rank_score": selected["candidate_rank_score"],
                "confirmation_opened": confirmation_opened,
                "selected_holdout_capture": selected["holdout_confirmation"].get(
                    "linear_capture"
                ),
                "wall_seconds": report["wall_seconds"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
